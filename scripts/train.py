#!/usr/bin/env python3
"""Train the current planner or verify one real Episode batch."""

import argparse
import json
import os

import torch

from neural_minco.config import default_config
from neural_minco.data import (
    EpisodeWindowDataset,
    load_datasets,
    load_manifest,
    split_episode_files,
    validate_lidar_compatibility,
)
from neural_minco.model import build_model
from neural_minco.train import Stage1Loss, Stage1Trainer, fit_output_scales


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Directory of Episode NPZ files")
    parser.add_argument("--manifest", required=True, help="Flight-level split manifest")
    parser.add_argument("--output", help="New run directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--resume", help="Run-local ckpt_last.pth")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run one real forward/loss/backward without writing artifacts"
    )
    return parser.parse_args(argv)


def dry_run(args):
    """Fit scales on one training Episode and execute a real Stage-1 batch."""
    manifest = load_manifest(args.manifest)
    train_files = split_episode_files(manifest, args.data, "train")
    dataset = EpisodeWindowDataset((train_files[0],))
    config = fit_output_scales(default_config(), dataset)
    validate_lidar_compatibility(dataset, config["lidar"])
    device = torch.device(args.device)
    model = build_model(config).to(device).train()
    criterion = Stage1Loss(config["loss"])
    sample = dataset[0]
    batch = {name: value[None].to(device) for name, value in sample.items() if isinstance(value, torch.Tensor)}
    _, auxiliary = model(batch["obs"], batch["state_body"], return_stages=True)
    loss, terms = criterion(model, auxiliary, batch, batch["state_body"])
    loss.backward()
    result = {
        "episode": train_files[0],
        "device": str(device),
        "loss": float(loss.detach()),
        "stages": len(auxiliary["raw_stages"]),
        "waypoint_scale": config["model"]["waypoint_scale"],
        "terminal_velocity_scale": config["model"]["terminal_velocity_scale"],
        "z3_position_loss": float(terms["position"]),
    }
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dry_run:
        dry_run(args)
        return

    if args.resume:
        run_dir = os.path.dirname(os.path.abspath(args.resume))
    elif args.output:
        run_dir = args.output
    else:
        raise ValueError("--output is required unless --resume is used")
    train_dataset, val_dataset = load_datasets(args.data, args.manifest)
    trainer = Stage1Trainer(
        train_dataset, val_dataset, run_dir, manifest_path=args.manifest, device=args.device, resume=args.resume
    )
    print(trainer.train())


if __name__ == "__main__":
    main()
