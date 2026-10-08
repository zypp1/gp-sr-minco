#!/usr/bin/env python3
"""Predict one trajectory from one raw Episode window."""

import argparse
import json
import os

import numpy as np
import torch

from neural_minco.data import EpisodeWindowDataset, validate_lidar_compatibility
from neural_minco.model import load_model
from neural_minco.runtime import predict


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", help="Optional output NPZ path")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.checkpoint, device)
    dataset = EpisodeWindowDataset((args.episode,))
    validate_lidar_compatibility(dataset, model.lidar_config)
    sample = dataset[args.index]
    prediction = predict(model, sample["obs"][None], sample["state_body"][None], samples=args.samples)
    arrays = {
        name: value[0].detach().cpu().numpy()
        for name, value in {
            "Q": prediction.waypoints,
            "T": prediction.durations,
            "p_end": prediction.endpoint,
            "v_end": prediction.terminal_velocity,
            "a_end": prediction.terminal_acceleration,
            "coefficients": prediction.coefficients,
            "position": prediction.position,
            "velocity": prediction.velocity,
            "acceleration": prediction.acceleration,
            "jerk": prediction.jerk,
        }.items()
    }
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(args.output, **arrays)
    print(json.dumps({"T": arrays["T"].tolist(), "p_end": arrays["p_end"].tolist()}, indent=2))


if __name__ == "__main__":
    main()
