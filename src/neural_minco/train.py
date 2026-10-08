"""Stage-1 imitation objective and training loop for the current planner."""

import copy
import csv
import json
import os
import random
import shutil
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import default_config
from .data import EpisodeUniformSampler, EpisodeWindowDataset, validate_lidar_compatibility
from .evaluate import episode_balanced_mean, trajectory_displacement_errors
from .minco import MinimumJerkDecoder, sample_trajectory_batch_with_derivatives
from .model import TrajectoryAttentionPlanner, build_model, decode_action


def fit_output_scales(config: Mapping[str, object], train_dataset: EpisodeWindowDataset) -> dict:
    """Fit physical output scales from selected training labels only."""
    resolved = copy.deepcopy(config)
    valid_points = train_dataset.dense_traj[train_dataset.dense_traj_valid]
    if not len(valid_points) or not len(train_dataset.v_end):
        raise ValueError("training data must contain trajectory and terminal-velocity labels")
    headroom = float(resolved["training"]["output_scale_headroom"])
    if not np.isfinite(headroom) or headroom < 1.0:
        raise ValueError("training.output_scale_headroom must be finite and at least one")
    waypoint_scale = float(np.linalg.norm(valid_points, axis=-1).max()) * headroom
    velocity_scale = float(np.linalg.norm(train_dataset.v_end, axis=-1).max()) * headroom
    resolved["model"]["waypoint_scale"] = max(1.0, waypoint_scale)
    resolved["model"]["terminal_velocity_scale"] = max(1.0, velocity_scale)
    return resolved


def decode_stages(
    raw_stages: Sequence[Mapping[str, torch.Tensor]],
    state_body: torch.Tensor,
    planner: TrajectoryAttentionPlanner,
    decoder: MinimumJerkDecoder,
) -> Tuple[Dict[str, torch.Tensor], ...]:
    """Decode z0..z3 together and sample each at ``0.1..2.0`` seconds."""
    if len(raw_stages) != 4:
        raise ValueError("Stage 1 requires exactly four outputs: z0, z1, z2, z3")
    batch_size = len(state_body)
    fields = ("Q", "p_end", "v_end", "a_end", "ratio_logits")
    packed_raw = {name: torch.cat([stage[name] for stage in raw_stages], dim=0) for name in fields}
    packed_state = state_body.repeat(4, 1)
    waypoints, durations, endpoint, velocity_end, acceleration_end = decode_action(
        packed_raw,
        waypoint_scale=planner.waypoint_scale,
        terminal_velocity_scale=planner.v_end_scale,
        terminal_acceleration_scale=planner.acceleration_scale,
        horizon=planner.horizon,
    )
    coefficients = decoder(waypoints, durations, endpoint, velocity_end, packed_state, acceleration_end)
    sampled = sample_trajectory_batch_with_derivatives(coefficients, durations, 21, derivs=(0, 1, 2, 3))
    decoded = []
    for stage_index in range(4):
        start = stage_index * batch_size
        stop = start + batch_size
        decoded.append(
            {
                "Q": waypoints[start:stop],
                "T": durations[start:stop],
                "p_end": endpoint[start:stop],
                "v_end": velocity_end[start:stop],
                "a_end": acceleration_end[start:stop],
                "coefficients": coefficients[start:stop],
                "position": sampled[0][start:stop, 1:],
                "velocity": sampled[1][start:stop, 1:],
                "acceleration": sampled[2][start:stop, 1:],
                "jerk": sampled[3][start:stop, 1:],
                "jerk_with_start": sampled[3][start:stop],
            }
        )
    return tuple(decoded)


def _imitation_loss(
    decoded: Mapping[str, torch.Tensor],
    target_position: torch.Tensor,
    target_velocity: torch.Tensor,
    target_valid: torch.Tensor,
    target_terminal_acceleration: torch.Tensor,
    loss_config: Mapping[str, object],
    velocity_scale: float,
    acceleration_scale: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    valid = target_valid.to(decoded["position"].dtype)
    denominator = valid.sum().clamp_min(1.0)
    position_error = torch.linalg.vector_norm(decoded["position"] - target_position, dim=-1)
    position = (
        F.smooth_l1_loss(
            position_error, torch.zeros_like(position_error), beta=float(loss_config["position_beta"]), reduction="none"
        )
        * valid
    ).sum() / denominator
    velocity = (
        torch.linalg.vector_norm(decoded["velocity"] - target_velocity, dim=-1) / velocity_scale * valid
    ).sum() / denominator
    terminal_acceleration = F.smooth_l1_loss(
        decoded["a_end"] / acceleration_scale,
        target_terminal_acceleration / acceleration_scale,
        beta=float(loss_config["terminal_acceleration_beta"]),
    )
    ratios = decoded["T"] / decoded["T"].sum(dim=-1, keepdim=True)
    ratio_barrier = F.relu(float(loss_config["minimum_ratio"]) - ratios).square().mean()
    total = (
        position
        + float(loss_config["velocity_weight"]) * velocity
        + float(loss_config["terminal_acceleration_weight"]) * terminal_acceleration
        + float(loss_config["ratio_barrier_weight"]) * ratio_barrier
    )
    return total, {
        "position": position,
        "velocity": velocity,
        "terminal_acceleration": terminal_acceleration,
        "ratio_barrier": ratio_barrier,
    }


def _jerk_nonregression(
    decoded_stages: Sequence[Mapping[str, torch.Tensor]], jerk_scale: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    peak_ratio = torch.stack(
        [
            torch.linalg.vector_norm(stage["jerk_with_start"], dim=-1).amax(dim=1) / jerk_scale
            for stage in decoded_stages
        ],
        dim=1,
    )
    violation = F.relu(peak_ratio - 1.0)
    transitions = F.relu(violation[:, 1:] - violation[:, :-1].detach())
    return transitions.mean(), transitions.mean(dim=0), peak_ratio


class Stage1Loss(nn.Module):
    """Supervise all four actions on the same physical 20-point trajectory."""

    def __init__(self, loss_config: Optional[Mapping[str, object]] = None):
        super().__init__()
        source = default_config()["loss"] if loss_config is None else loss_config
        self.loss_config = dict(source)
        self.decoder = MinimumJerkDecoder()

    def forward(
        self,
        planner: TrajectoryAttentionPlanner,
        auxiliary: Mapping[str, object],
        batch: Mapping[str, torch.Tensor],
        state_body: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return the scalar Stage-1 loss and detached reporting tensors."""
        decoded_stages = decode_stages(auxiliary["raw_stages"], state_body, planner, self.decoder)
        return self.from_decoded(planner, auxiliary, batch, decoded_stages)

    def from_decoded(self, planner, auxiliary, batch, decoded_stages):
        """Use one shared MINCO decode for imitation and optional physical supervision."""
        layer_losses = []
        layer_terms = []
        for decoded in decoded_stages:
            loss, terms = _imitation_loss(
                decoded,
                batch["dense_traj"],
                batch["dense_velocity"],
                batch["dense_traj_valid"],
                batch["a_end"],
                self.loss_config,
                planner.velocity_scale,
                planner.acceleration_scale,
            )
            layer_losses.append(loss)
            layer_terms.append(terms)
        imitation = torch.stack(layer_losses).mean()

        correction_norms = auxiliary["correction_norms"]
        correction_regularization = correction_norms[:, 1:].square().mean()
        z0_logits = auxiliary["raw_stages"][0]["ratio_logits"]
        z0_logits = z0_logits - z0_logits.mean(dim=-1, keepdim=True)
        ratio_prior = z0_logits.square().mean()
        jerk_nonregression, jerk_transitions, peak_jerk_ratio = _jerk_nonregression(decoded_stages, planner.jerk_scale)
        total = (
            imitation
            + float(self.loss_config["delta_weight"]) * correction_regularization
            + float(self.loss_config["z0_ratio_weight"]) * ratio_prior
            + float(self.loss_config["jerk_nonregression_weight"]) * jerk_nonregression
        )
        final = decoded_stages[-1]
        final_terms = layer_terms[-1]
        report = {
            "loss": total,
            "imitation": imitation,
            "position": final_terms["position"],
            "velocity": final_terms["velocity"],
            "terminal_acceleration": final_terms["terminal_acceleration"],
            "ratio_barrier": final_terms["ratio_barrier"],
            "correction_regularization": correction_regularization,
            "ratio_prior": ratio_prior,
            "jerk_nonregression": jerk_nonregression,
            "jerk_nonregression_transitions": jerk_transitions,
            "peak_jerk_ratio": peak_jerk_ratio,
            "layer_losses": torch.stack(layer_losses),
            "points": final["position"],
            "velocity_samples": final["velocity"],
            "acceleration_samples": final["acceleration"],
            "jerk_samples": final["jerk"],
            "durations": final["T"],
            "coefficients": final["coefficients"],
        }
        return total, {name: value.detach() for name, value in report.items()}


TRAIN_LOSS_TERMS = ("loss", "position", "velocity", "terminal_acceleration", "ratio_prior", "jerk_nonregression")
HISTORY_COLUMNS = (
    "epoch",
    "step",
    "lr",
    *(f"train_{name}" for name in TRAIN_LOSS_TERMS),
    "train_grad_norm",
    "val_loss",
    "val_ade_m",
    "val_fde_m",
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def _assert_finite_loss(loss: torch.Tensor) -> None:
    """Fail before backward without synchronizing a CUDA scalar to Python."""
    finite = torch.isfinite(loss.detach())
    if loss.device.type == "cuda":
        torch._assert_async(finite, "training loss is not finite")
    elif not bool(finite):
        raise FloatingPointError("training loss is not finite")


def _loader(
    dataset: EpisodeWindowDataset,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
    sampler: Optional[EpisodeUniformSampler] = None,
) -> DataLoader:
    options = {"batch_size": batch_size, "num_workers": workers, "pin_memory": device.type == "cuda"}
    if workers:
        options.update({"persistent_workers": True, "prefetch_factor": 4})
    if sampler is None:
        options["shuffle"] = False
    else:
        options["sampler"] = sampler
        options["drop_last"] = len(dataset) >= batch_size
    return DataLoader(dataset, **options)


def _tensor_batch(batch: Mapping[str, object], device: torch.device) -> dict:
    return {
        name: value.to(device, non_blocking=True) for name, value in batch.items() if isinstance(value, torch.Tensor)
    }


@torch.no_grad()
def evaluate_imitation(
    model: TrajectoryAttentionPlanner, criterion: Stage1Loss, loader: DataLoader, device: torch.device
) -> dict:
    """Evaluate loss and final-stage ADE/FDE on one fixed split."""
    model.eval()
    weighted_loss = torch.zeros((), device=device)
    window_count = 0
    ade_chunks = []
    fde_chunks = []
    episode_chunks = []
    for batch in tqdm(loader, desc="validation", unit="batch", dynamic_ncols=True, leave=False):
        tensors = _tensor_batch(batch, device)
        state_body = tensors["state_body"]
        _, auxiliary = model(tensors["obs"], state_body, return_stages=True)
        loss, terms = criterion(model, auxiliary, tensors, state_body)
        window_ade, window_fde = trajectory_displacement_errors(
            terms["points"], tensors["dense_traj"], tensors["dense_traj_valid"]
        )
        batch_size = len(state_body)
        weighted_loss = weighted_loss + loss.detach() * batch_size
        window_count += batch_size
        ade_chunks.append(window_ade)
        fde_chunks.append(window_fde)
        episode_chunks.append(tensors["episode_index"])
    if not window_count:
        raise ValueError("validation loader is empty")
    ade = torch.cat(ade_chunks).cpu().numpy()
    fde = torch.cat(fde_chunks).cpu().numpy()
    episodes = torch.cat(episode_chunks).cpu().numpy()
    metrics = {
        "loss": float(weighted_loss.cpu()) / window_count,
        "ade_m": episode_balanced_mean(ade, episodes),
        "fde_m": episode_balanced_mean(fde, episodes),
        "window_ade_m": float(ade.mean()),
        "window_fde_m": float(fde.mean()),
        "windows": int(len(ade)),
        "episodes": int(len(np.unique(episodes))),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise RuntimeError("validation produced a non-finite metric")
    return metrics


def _write_json(path: str, payload: Mapping[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


class Stage1Trainer:
    """Run the fixed ten-epoch Stage-1 protocol with exact epoch resume."""

    def __init__(
        self,
        train_dataset: EpisodeWindowDataset,
        val_dataset: EpisodeWindowDataset,
        run_dir: str,
        *,
        manifest_path: str,
        device: str = "cuda",
        resume: Optional[str] = None,
        config: Optional[Mapping[str, object]] = None,
    ):
        if resume is not None and config is not None:
            raise ValueError("resume uses its checkpoint config; start a new run to change it")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training requested but CUDA is unavailable")
        self.run_dir = os.path.abspath(run_dir)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        checkpoint = None
        if resume is None:
            source = default_config() if config is None else copy.deepcopy(config)
            self.config = fit_output_scales(source, train_dataset)
        else:
            resume = os.path.abspath(resume)
            if os.path.dirname(resume) != self.run_dir:
                raise ValueError("resume checkpoint must live inside run_dir")
            if not os.path.isfile(resume):
                raise FileNotFoundError(resume)
            checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, Mapping) or "config" not in checkpoint:
                raise ValueError("resume checkpoint must contain the complete config")
            self.config = copy.deepcopy(checkpoint["config"])

        self.training = self.config["training"]
        if int(self.training.get("log_interval", 20)) < 1:
            raise ValueError("training.log_interval must be positive")
        if float(self.config["model"]["dropout"]) != 0.0:
            raise ValueError("current training and resume require model.dropout=0")
        validate_lidar_compatibility(train_dataset, self.config["lidar"])
        validate_lidar_compatibility(val_dataset, self.config["lidar"])
        _set_seed(int(self.training["seed"]))
        self.model = build_model(self.config).to(self.device)
        self.criterion = Stage1Loss(self.config["loss"])
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.training["learning_rate"]),
            weight_decay=float(self.training["weight_decay"]),
        )
        self.sampler = EpisodeUniformSampler(train_dataset.episode_index, seed=int(self.training["seed"]))
        batch_size = int(self.training["batch_size"])
        workers = int(self.training["num_workers"])
        self.train_loader = _loader(
            train_dataset, batch_size=batch_size, workers=workers, device=self.device, sampler=self.sampler
        )
        self.val_loader = _loader(val_dataset, batch_size=batch_size, workers=workers, device=self.device)
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = int(self.training["epochs"]) * self.steps_per_epoch
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, self.total_steps), eta_min=float(self.training["minimum_learning_rate"])
        )
        self.epoch = 0
        self.step = 0
        self.best_ade = float("inf")
        if checkpoint is None:
            os.makedirs(self.run_dir, exist_ok=False)
            self._create_run_files(manifest_path)
        else:
            self._resume(checkpoint, manifest_path)

    def _create_run_files(self, manifest_path: str) -> None:
        shutil.copy2(manifest_path, os.path.join(self.run_dir, "split_manifest.json"))
        _write_json(os.path.join(self.run_dir, "params.json"), self.config)
        with open(os.path.join(self.run_dir, "history.csv"), "w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=HISTORY_COLUMNS).writeheader()

    def _resume(self, checkpoint: Mapping[str, object], manifest_path: str) -> None:
        installed_manifest = os.path.join(self.run_dir, "split_manifest.json")
        if _read_json(installed_manifest) != _read_json(manifest_path):
            raise ValueError("resume manifest differs from the run manifest")
        if _read_json(os.path.join(self.run_dir, "params.json")) != self.config:
            raise ValueError("resume checkpoint config differs from params.json")

        required = ("state_dict", "optimizer", "scheduler", "epoch", "step", "steps_per_epoch", "best_ade")
        missing = [name for name in required if name not in checkpoint]
        if missing:
            raise ValueError("resume checkpoint is missing %s" % ", ".join(missing))

        best_ade = float(checkpoint["best_ade"])
        if not np.isfinite(best_ade):
            raise ValueError("resume checkpoint best_ade must be finite")

        checkpoint_epoch = int(checkpoint["epoch"])
        checkpoint_step = int(checkpoint["step"])
        checkpoint_steps_per_epoch = int(checkpoint["steps_per_epoch"])
        if not 0 <= checkpoint_epoch < int(self.training["epochs"]):
            raise ValueError("resume checkpoint epoch is outside the training schedule")
        if checkpoint_steps_per_epoch != self.steps_per_epoch:
            raise ValueError("resume loader steps differ from the checkpoint")
        expected_step = (checkpoint_epoch + 1) * self.steps_per_epoch
        if checkpoint_step != expected_step:
            raise ValueError("resume step is inconsistent with its completed epoch")

        scheduler_state = checkpoint["scheduler"]
        if not isinstance(scheduler_state, Mapping):
            raise ValueError("resume scheduler state must be a mapping")
        if int(scheduler_state.get("T_max", -1)) != self.total_steps:
            raise ValueError("resume scheduler duration differs from the current protocol")
        if int(scheduler_state.get("last_epoch", -1)) != checkpoint_step:
            raise ValueError("resume scheduler position differs from checkpoint step")

        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(scheduler_state)
        self.epoch = checkpoint_epoch + 1
        self.step = checkpoint_step
        self.best_ade = best_ade

    def _checkpoint(self, include_optimizer: bool) -> dict:
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "config": self.config,
            "epoch": self.epoch,
            "step": self.step,
            "steps_per_epoch": self.steps_per_epoch,
            "best_ade": self.best_ade,
        }
        if include_optimizer:
            checkpoint["optimizer"] = self.optimizer.state_dict()
            checkpoint["scheduler"] = self.scheduler.state_dict()
        return checkpoint

    def _train_epoch(self, writer=None) -> dict:
        self.model.train()
        self.sampler.set_epoch(self.epoch)
        log_interval = int(self.training.get("log_interval", 20))
        running = {name: torch.zeros((), device=self.device) for name in TRAIN_LOSS_TERMS}
        running["grad_norm"] = torch.zeros((), device=self.device)
        progress = tqdm(
            self.train_loader,
            desc="epoch %d/%d" % (self.epoch + 1, int(self.training["epochs"])),
            unit="batch",
            dynamic_ncols=True,
        )
        batches = 0
        for batch in progress:
            self.optimizer.zero_grad(set_to_none=True)
            tensors = _tensor_batch(batch, self.device)
            state_body = tensors["state_body"]
            _, auxiliary = self.model(tensors["obs"], state_body, return_stages=True)
            loss, terms = self.criterion(self.model, auxiliary, tensors, state_body)
            _assert_finite_loss(loss)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float(self.training["gradient_clip"]), error_if_nonfinite=True
            )
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1
            batches += 1
            for name in TRAIN_LOSS_TERMS:
                running[name] = running[name] + terms[name]
            running["grad_norm"] = running["grad_norm"] + grad_norm.detach()
            if self.step % log_interval == 0 or batches == len(self.train_loader):
                values = torch.stack([value / batches for value in running.values()]).cpu().tolist()
                metrics = dict(zip(running, values))
                learning_rate = self.optimizer.param_groups[0]["lr"]
                progress.set_postfix(loss="%.5f" % metrics["loss"], lr="%.2e" % learning_rate, refresh=False)
                if writer is not None:
                    for name, value in metrics.items():
                        writer.add_scalar("train/%s" % name, value, self.step)
                    writer.add_scalar("train/lr", learning_rate, self.step)
        if not batches:
            raise ValueError("training loader is empty")
        return metrics

    def _append_history(self, train_metrics: Mapping[str, float], val_metrics: Mapping[str, float]) -> None:
        row = {
            "epoch": self.epoch,
            "step": self.step,
            "lr": self.optimizer.param_groups[0]["lr"],
            **{f"train_{name}": train_metrics[name] for name in TRAIN_LOSS_TERMS},
            "train_grad_norm": train_metrics["grad_norm"],
            "val_loss": val_metrics["loss"],
            "val_ade_m": val_metrics["ade_m"],
            "val_fde_m": val_metrics["fde_m"],
        }
        with open(os.path.join(self.run_dir, "history.csv"), "a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=HISTORY_COLUMNS).writerow(row)

    def train(self) -> str:
        """Run every configured epoch and return the best-checkpoint path."""
        from torch.utils.tensorboard import SummaryWriter

        epochs = int(self.training["epochs"])
        with SummaryWriter(os.path.join(self.run_dir, "tensorboard"), purge_step=self.step + 1) as writer:
            for epoch in range(self.epoch, epochs):
                self.epoch = epoch
                train_metrics = self._train_epoch(writer)
                val_metrics = evaluate_imitation(self.model, self.criterion, self.val_loader, self.device)
                self._append_history(train_metrics, val_metrics)
                if val_metrics["ade_m"] < self.best_ade:
                    self.best_ade = val_metrics["ade_m"]
                    torch.save(self._checkpoint(include_optimizer=False), os.path.join(self.run_dir, "ckpt_best.pth"))
                torch.save(self._checkpoint(include_optimizer=True), os.path.join(self.run_dir, "ckpt_last.pth"))
                for name in ("loss", "ade_m", "fde_m"):
                    writer.add_scalar("val/%s" % name, val_metrics[name], self.step)
                writer.flush()
                print(
                    "epoch %d/%d: train=%.5f, val ADE/FDE=%.5f/%.5f m"
                    % (epoch + 1, epochs, train_metrics["loss"], val_metrics["ade_m"], val_metrics["fde_m"])
                )
        return os.path.join(self.run_dir, "ckpt_best.pth")


__all__ = ["Stage1Loss", "Stage1Trainer", "decode_stages", "evaluate_imitation", "fit_output_scales"]
