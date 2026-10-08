#!/usr/bin/env python3
"""Evaluate all four planner stages on a manifest split."""

import argparse
import json
import os

import torch

from neural_minco.data import EpisodeWindowDataset, load_manifest, split_episode_files
from neural_minco.evaluate import evaluate_model
from neural_minco.model import load_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--map", help="Optional fixed-map NPZ for global clearance")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", help="Evaluation JSON path")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.checkpoint, device)
    manifest = load_manifest(args.manifest)
    dataset = EpisodeWindowDataset(split_episode_files(manifest, args.data, args.split))
    report = evaluate_model(
        model, dataset, device=device, batch_size=args.batch_size, workers=args.workers, map_path=args.map
    )
    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint)), f"evaluation_{args.split}.json"
    )
    output_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(output_dir, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": output, "z3": report["layers"][-1]}, indent=2))


if __name__ == "__main__":
    main()
