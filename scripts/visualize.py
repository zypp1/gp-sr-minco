#!/usr/bin/env python3
"""Plot z0 through z3 and the expert for one raw Episode window."""

import argparse
import os

import numpy as np
import torch

from neural_minco.data import EpisodeWindowDataset, validate_lidar_compatibility
from neural_minco.model import load_model
from neural_minco.runtime import decode_prediction


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Matplotlib 3.1 needs this import to register the 3D projection.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.checkpoint, device)
    dataset = EpisodeWindowDataset((args.episode,))
    validate_lidar_compatibility(dataset, model.lidar_config)
    sample = dataset[args.index]
    observation = sample["obs"][None].to(device)
    state_body = sample["state_body"][None].to(device)
    with torch.inference_mode():
        _, auxiliary = model(observation, state_body, return_stages=True)
        stage_points = [
            decode_prediction(model, raw, state_body, samples=80).position[0].cpu().numpy()
            for raw in auxiliary["raw_stages"]
        ]

    valid = sample["dense_traj_valid"].numpy()
    expert = sample["dense_traj"].numpy()[valid]
    figure = plt.figure(figsize=(12, 4))
    image_axis = figure.add_subplot(1, 2, 1)
    image_axis.imshow(sample["obs"][0].numpy(), cmap="viridis", aspect="auto")
    image_axis.set(title="Range image", xlabel="azimuth column", ylabel="elevation row")
    path_axis = figure.add_subplot(1, 2, 2, projection="3d")
    path_axis.plot(*expert.T, color="black", linewidth=2.5, label="expert")
    colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(stage_points)))
    for index, (points, color) in enumerate(zip(stage_points, colors)):
        path_axis.plot(*points.T, color=color, label=f"z{index}")
    path_axis.scatter([0.0], [0.0], [0.0], color="black", marker="o")
    path_axis.set(xlabel="body x [m]", ylabel="body y [m]", zlabel="body z [m]")
    path_axis.legend()
    figure.tight_layout()
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
