"""Experimental Refiner-only fine-tuning; the network and Stage-1 loss are unchanged."""

import csv
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import ACCELERATION_LIMIT_MPS2, JERK_LIMIT_MPS3, VELOCITY_LIMIT_MPS
from .data import EpisodeUniformSampler, validate_lidar_compatibility
from .evaluate import load_fixed_esdf, minco_derivative_maxima
from .geometry import quaternion_matrix_numpy, query_esdf
from .minco import sample_future_trajectory_batch
from .train import (
    TRAIN_LOSS_TERMS,
    Stage1Loss,
    _assert_finite_loss,
    _set_seed,
    _tensor_batch,
    decode_stages,
    evaluate_imitation,
)


def freeze_proposal(model):
    """Freeze all z0 owners, including the query encoder reused by the Refiner."""
    model.requires_grad_(False)
    model.shared_refiner.requires_grad_(True)
    return list(model.shared_refiner.parameters())


class MapDataset(Dataset):
    """Add a map index and precomputed full-quaternion rotation, without changing Episodes."""

    def __init__(self, dataset, map_ids):
        self.dataset = dataset
        self.map_index = np.asarray([map_ids.index(str(value)) for value in dataset.map_id], dtype=np.int64)
        self.rotation = quaternion_matrix_numpy(dataset.state[:, 9:13]).astype(np.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {
            **self.dataset[index],
            "map_index": int(self.map_index[index]),
            "body_to_world_rotation": torch.from_numpy(self.rotation[index]),
        }


class MapClearance:
    """Matched ESDFs with a solid ground half-space and conservative finite-map bounds."""

    def __init__(self, esdfs, ground_height):
        self.esdfs = esdfs
        self.ground_height = float(ground_height)

    @classmethod
    def load(cls, paths, config):
        esdfs = [load_fixed_esdf(path, voxel=config["map_voxel_m"])[0] for path in paths]
        return cls(esdfs, config["ground_height_m"])

    def __call__(self, world, map_index):
        distances = []
        for esdf in self.esdfs:
            distance = query_esdf(esdf, world)
            lower = world.new_tensor(esdf["origin"])
            upper = lower + (world.new_tensor(esdf["shape"]) - 1.0) * esdf["voxel"]
            # Unlike distance extrapolation, signed faces do not reward leaving the map.
            boundary = torch.minimum(world - lower, upper - world).amin(dim=-1)
            distances.append(torch.minimum(distance, boundary))
        stacked = torch.stack(distances, dim=0)
        selected = stacked[map_index, torch.arange(len(world), device=world.device)]
        return torch.minimum(selected, world[..., 2] - self.ground_height)


def physical_losses(decoded, batch, config, clearance=None):
    """Equal losses on z1/z2/z3, sampled on actual MINCO curves, never Hermite proxies."""
    coefficients = torch.cat([stage["coefficients"] for stage in decoded[1:]])
    durations = torch.cat([stage["T"] for stage in decoded[1:]])
    dynamics = collision = coefficients.new_zeros(())
    if config["dynamics_weight"]:
        maxima = minco_derivative_maxima(coefficients, durations, config["dynamics_samples_per_segment"])
        limits = maxima.new_tensor([VELOCITY_LIMIT_MPS, ACCELERATION_LIMIT_MPS2, JERK_LIMIT_MPS3])
        dynamics = F.relu(maxima / limits - 1.0).sum(dim=-1).mean()
    if config["collision_weight"]:
        points = sample_future_trajectory_batch(coefficients, durations, config["collision_samples"])
        rotation = batch["body_to_world_rotation"].repeat(3, 1, 1)
        world = torch.bmm(points, rotation.transpose(1, 2)) + batch["state"][:, None, :3].repeat(3, 1, 1)
        distance = clearance(world, batch["map_index"].repeat(3)).amin(dim=-1)
        collision = F.relu(1.0 - distance / config["clearance_m"]).mean()
    return dynamics, collision


class Stage2Loss(Stage1Loss):
    """Keep the complete imitation anchor and add absolute physical hinges."""

    def __init__(self, config, clearance=None):
        super().__init__(config["loss"])
        self.physical_config = config["stage2"]
        self.clearance = clearance

    def forward(self, planner, auxiliary, batch, state_body):
        decoded = decode_stages(auxiliary["raw_stages"], state_body, planner, self.decoder)
        imitation, report = self.from_decoded(planner, auxiliary, batch, decoded)
        dynamics, collision = physical_losses(decoded, batch, self.physical_config, self.clearance)
        total = imitation
        if self.physical_config["dynamics_weight"]:
            total = total + self.physical_config["dynamics_weight"] * dynamics
        if self.physical_config["collision_weight"]:
            total = total + self.physical_config["collision_weight"] * collision
        report.update(
            loss=total.detach(), stage1=imitation.detach(), dynamics=dynamics.detach(), collision=collision.detach()
        )
        return total, report


def train_stage2(model, config, train_data, val_data, run_dir, manifest_path, initial_checkpoint, map_paths, device):
    """Run a fresh matched schedule and retain both final-epoch and ADE-best checkpoints."""
    device = torch.device(device)
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"new experiment requires a new output directory: {run_dir}")
    if min(config["stage2"][key] for key in ("dynamics_weight", "collision_weight")) < 0:
        raise ValueError("physical loss weights must be nonnegative")
    map_ids = [Path(path).stem for path in map_paths]
    if len(map_ids) != len(set(map_ids)):
        raise ValueError("each training map must have one unique map_id")
    for dataset in (train_data, val_data):
        validate_lidar_compatibility(dataset, config["lidar"])
    training = config["training"]
    _set_seed(training["seed"])
    model.to(device)
    parameters = freeze_proposal(model)
    sampler = EpisodeUniformSampler(train_data.episode_index, seed=training["seed"])
    options = dict(
        batch_size=training["batch_size"], num_workers=training["num_workers"], pin_memory=device.type == "cuda"
    )
    if training["num_workers"]:
        options.update(persistent_workers=True, prefetch_factor=config["stage2"]["prefetch_factor"])
    train_loader = DataLoader(MapDataset(train_data, map_ids), sampler=sampler, drop_last=True, **options)
    val_loader = DataLoader(val_data, shuffle=False, **options)
    clearance = MapClearance.load(map_paths, config["stage2"]) if config["stage2"]["collision_weight"] else None
    criterion = Stage2Loss(config, clearance)
    val_criterion = Stage1Loss(config["loss"])
    optimizer = torch.optim.AdamW(parameters, lr=training["learning_rate"], weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training["epochs"] * len(train_loader), eta_min=training["minimum_learning_rate"]
    )
    config["stage2"].update(
        initial_checkpoint=os.path.abspath(initial_checkpoint), map_paths=[os.path.abspath(path) for path in map_paths]
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest_path, run_dir / "split_manifest.json")
    (run_dir / "params.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    names = (*TRAIN_LOSS_TERMS, "stage1", "dynamics", "collision", "grad_norm")
    columns = [
        "epoch",
        "step",
        "lr",
        "seconds",
        *(f"train_{name}" for name in names),
        "val_loss",
        "val_ade_m",
        "val_fde_m",
    ]
    step, best_ade = 0, float("inf")
    with (run_dir / "history.csv").open("w", newline="") as stream, SummaryWriter(
        str(run_dir / "tensorboard")
    ) as writer:
        history = csv.DictWriter(stream, fieldnames=columns)
        history.writeheader()
        for epoch in range(training["epochs"]):
            model.train()
            sampler.set_epoch(epoch)
            start = time.perf_counter()
            running = torch.zeros(len(names), device=device)
            progress = tqdm(
                train_loader, desc=f"epoch {epoch + 1}/{training['epochs']}", unit="batch", dynamic_ncols=True
            )
            for count, batch in enumerate(progress, 1):
                batch = _tensor_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                _, auxiliary = model(batch["obs"], batch["state_body"], return_stages=True)
                loss, terms = criterion(model, auxiliary, batch, batch["state_body"])
                _assert_finite_loss(loss)
                loss.backward()
                terms["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                    parameters, training["gradient_clip"], error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                step += 1
                running += torch.stack([terms[name].detach() for name in names])
                if step % training["log_interval"] == 0 or count == len(train_loader):
                    means = dict(zip(names, (running / count).cpu().tolist()))
                    for name, value in means.items():
                        writer.add_scalar(f"train/{name}", value, step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
                    progress.set_postfix(loss=f"{means['loss']:.5f}", refresh=False)
            val = evaluate_imitation(model, val_criterion, val_loader, device)
            seconds = time.perf_counter() - start
            history.writerow(
                dict(
                    epoch=epoch,
                    step=step,
                    lr=optimizer.param_groups[0]["lr"],
                    seconds=seconds,
                    **{f"train_{name}": value for name, value in means.items()},
                    **{f"val_{name}": val[name] for name in ("loss", "ade_m", "fde_m")},
                )
            )
            stream.flush()
            improved = val["ade_m"] < best_ade
            best_ade = min(best_ade, val["ade_m"])
            checkpoint = dict(
                state_dict=model.state_dict(),
                config=config,
                epoch=epoch,
                step=step,
                steps_per_epoch=len(train_loader),
                best_ade=best_ade,
            )
            if improved:
                torch.save(checkpoint, run_dir / "ckpt_best.pth")
            checkpoint.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
            torch.save(checkpoint, run_dir / "ckpt_last.pth")
            for name in ("loss", "ade_m", "fde_m"):
                writer.add_scalar(f"val/{name}", val[name], step)
            writer.flush()
            print(
                f"epoch {epoch + 1}: train={means['loss']:.5f}, "
                f"val ADE/FDE={val['ade_m']:.5f}/{val['fde_m']:.5f}, {seconds:.1f}s",
                flush=True,
            )
    return str(run_dir / "ckpt_last.pth")
