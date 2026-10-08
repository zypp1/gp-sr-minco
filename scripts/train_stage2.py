#!/usr/bin/env python3
"""Run the isolated Refiner-only Stage-2 comparison from a frozen Stage-1 checkpoint."""

import argparse

from neural_minco.config import stage2_config
from neural_minco.data import load_datasets
from neural_minco.model import load_model
from neural_minco.stage2 import train_stage2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--maps", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dynamics-weight", type=float, default=0.0)
    parser.add_argument("--collision-weight", type=float, default=0.0)
    args = parser.parse_args()
    model, checkpoint = load_model(args.checkpoint, "cpu")
    config = stage2_config(
        checkpoint["config"], dynamics_weight=args.dynamics_weight, collision_weight=args.collision_weight
    )
    train_data, val_data = load_datasets(args.data, args.manifest)
    print(f"Loaded train/val windows: {len(train_data)}/{len(val_data)}; updating shared_refiner only", flush=True)
    print(
        train_stage2(
            model, config, train_data, val_data, args.output, args.manifest, args.checkpoint, args.maps, args.device
        )
    )


if __name__ == "__main__":
    main()
