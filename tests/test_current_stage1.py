import csv
import random
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import neural_minco.train as current_train
from neural_minco.config import default_config
from neural_minco.data import CURRENT_LIDAR_GEOMETRY
from neural_minco.model import build_model
from neural_minco.train import Stage1Loss, Stage1Trainer, _assert_finite_loss, _jerk_nonregression


def test_stage1_supervises_z0_through_z3_and_backpropagates(monkeypatch):
    config = default_config()
    model = build_model(config)
    observation = torch.rand(1, 1, 30, 180)
    state_body = torch.tensor([[1.0, 0.1, 0.0, 0.1, 0.0, 0.0, 7.0, 1.0, 0.4]])
    _, auxiliary = model(observation, state_body, return_stages=True)
    assert len(auxiliary["raw_stages"]) == 4
    assert auxiliary["correction_norms"].shape == (1, 4)

    target = torch.stack((torch.linspace(0.1, 2.0, 20), torch.zeros(20), torch.zeros(20)), dim=-1)[None]
    batch = {
        "dense_traj": target,
        "dense_velocity": torch.tensor([1.0, 0.0, 0.0]).expand(1, 20, 3),
        "dense_traj_valid": torch.ones(1, 20, dtype=torch.bool),
        "a_end": torch.zeros(1, 3),
    }
    criterion = Stage1Loss()
    decoder_calls = []
    original_forward = criterion.decoder.forward

    def record_decoder(*args, **kwargs):
        decoder_calls.append(len(args[0]))
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(criterion.decoder, "forward", record_decoder)
    loss, terms = criterion(model, auxiliary, batch, state_body)
    loss.backward()

    assert decoder_calls == [4]
    assert terms["layer_losses"].shape == (4,)
    assert terms["jerk_nonregression_transitions"].shape == (3,)
    assert torch.isfinite(loss)
    for head in (model.global_proposal.output_head, model.shared_refiner.output_head):
        assert head[-1].weight.grad is not None
        assert torch.isfinite(head[-1].weight.grad).all()


def test_current_training_contract_requires_zero_dropout(tmp_path, monkeypatch):
    config = default_config()
    assert config["model"]["dropout"] == 0.0
    config["model"]["dropout"] = 0.1
    monkeypatch.setattr(current_train, "fit_output_scales", lambda source, _dataset: source)
    with pytest.raises(ValueError, match="dropout=0"):
        Stage1Trainer(None, None, str(tmp_path / "run"), manifest_path="unused.json", device="cpu", config=config)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("log_interval", [0, -1])
def test_training_rejects_nonpositive_log_interval_before_creating_run(tmp_path, monkeypatch, log_interval):
    config = default_config()
    config["training"]["log_interval"] = log_interval
    monkeypatch.setattr(current_train, "fit_output_scales", lambda source, _dataset: source)
    with pytest.raises(ValueError, match="training.log_interval must be positive"):
        Stage1Trainer(None, None, str(tmp_path / "run"), manifest_path="unused.json", device="cpu", config=config)
    assert not (tmp_path / "run").exists()


def test_jerk_nonregression_compares_each_stage_to_detached_previous_stage():
    peaks = torch.tensor([[18.0, 24.0, 12.0, 30.0]], requires_grad=True)
    stages = [
        {"jerk_with_start": torch.stack((peak, peak * 0, peak * 0), dim=-1)[:, None]} for peak in peaks.unbind(dim=1)
    ]

    loss, transitions, ratios = _jerk_nonregression(stages, jerk_scale=12.0)
    torch.testing.assert_close(ratios, peaks / 12.0)
    torch.testing.assert_close(transitions, torch.tensor([0.5, 0.0, 1.5]))
    torch.testing.assert_close(loss, torch.tensor(2.0 / 3.0))
    loss.backward()
    torch.testing.assert_close(peaks.grad, torch.tensor([[0.0, 1 / 36, 0.0, 1 / 36]]))


def test_fresh_training_rejects_lidar_mismatch_without_creating_run(tmp_path, monkeypatch):
    class Dataset:
        lidar_geometry = CURRENT_LIDAR_GEOMETRY.copy()

    config = default_config()
    config["lidar"]["r_max"] = 30.0
    monkeypatch.setattr(current_train, "fit_output_scales", lambda _config, _dataset: config)
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="LiDAR geometry disagrees"):
        Stage1Trainer(Dataset(), Dataset(), str(run_dir), manifest_path=str(tmp_path / "manifest.json"), device="cpu")

    assert not run_dir.exists()


def test_nonfinite_loss_fails_before_backward_on_cpu():
    with pytest.raises(FloatingPointError, match="not finite"):
        _assert_finite_loss(torch.tensor(float("nan")))


def _resume_shell(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    (run_dir / "split_manifest.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "params.json").write_text("{}\n", encoding="utf-8")
    trainer = Stage1Trainer.__new__(Stage1Trainer)
    trainer.run_dir = str(run_dir)
    trainer.config = {}
    trainer.training = {"epochs": 10}
    trainer.steps_per_epoch = 2
    trainer.total_steps = 20
    return trainer, str(manifest)


def _resume_checkpoint(best_ade=0.5):
    return {
        "state_dict": {},
        "optimizer": {},
        "scheduler": {"T_max": 20, "last_epoch": 2},
        "epoch": 0,
        "step": 2,
        "steps_per_epoch": 2,
        "best_ade": best_ade,
    }


def test_exact_resume_requires_finite_best_ade(tmp_path):
    trainer, manifest = _resume_shell(tmp_path)
    checkpoint = _resume_checkpoint()
    del checkpoint["best_ade"]
    with pytest.raises(ValueError, match="missing best_ade"):
        trainer._resume(checkpoint, manifest)

    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="best_ade must be finite"):
            trainer._resume(_resume_checkpoint(value), manifest)


def test_resume_rejects_configuration_override_before_loading(tmp_path):
    with pytest.raises(ValueError, match="resume uses its checkpoint config"):
        Stage1Trainer(
            None,
            None,
            str(tmp_path / "run"),
            manifest_path="unused.json",
            device="cpu",
            resume="unused.pth",
            config=default_config(),
        )


def test_explicit_training_config_is_copied_before_fitting(tmp_path, monkeypatch):
    config = default_config()
    config["training"]["seed"] = 43

    def stop_after_fitting(source, _dataset):
        assert source["training"]["seed"] == 43
        source["training"]["seed"] = 42
        raise RuntimeError("checked config isolation")

    monkeypatch.setattr(current_train, "fit_output_scales", stop_after_fitting)
    with pytest.raises(RuntimeError, match="checked config isolation"):
        Stage1Trainer(None, None, str(tmp_path / "run"), manifest_path="unused.json", device="cpu", config=config)
    assert config["training"]["seed"] == 43
    assert not (tmp_path / "run").exists()


def _monitoring_trainer(tmp_path, log_interval=2):
    class Planner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, observation, _state_body, return_stages=False):
            return None, {"prediction": observation * self.weight}

    def criterion(_model, auxiliary, _batch, _state_body):
        loss = auxiliary["prediction"].square().mean()
        return loss, {name: loss.detach() + index for index, name in enumerate(current_train.TRAIN_LOSS_TERMS)}

    tmp_path.mkdir(parents=True, exist_ok=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    trainer = Stage1Trainer.__new__(Stage1Trainer)
    trainer.device = torch.device("cpu")
    trainer.run_dir = str(run_dir)
    trainer.training = {"epochs": 1, "gradient_clip": 10.0}
    if log_interval is not None:
        trainer.training["log_interval"] = log_interval
    trainer.config = {"training": trainer.training}
    trainer.model = Planner()
    trainer.criterion = criterion
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1, gamma=0.5)
    trainer.sampler = SimpleNamespace(set_epoch=lambda _epoch: None)
    trainer.train_loader = [{"obs": torch.ones(1, 1), "state_body": torch.zeros(1, 9)} for _ in range(3)]
    trainer.val_loader = []
    trainer.steps_per_epoch = len(trainer.train_loader)
    trainer.epoch = trainer.step = 0
    trainer.best_ade = float("inf")
    trainer._create_run_files(str(manifest))
    return trainer


@pytest.mark.parametrize("log_interval, expected_steps", [(2, [2, 3]), (None, [3])])
def test_tensorboard_records_running_metrics_without_changing_training(
    tmp_path, monkeypatch, log_interval, expected_steps
):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    trainer = _monitoring_trainer(tmp_path / "logged", log_interval)
    reference = _monitoring_trainer(tmp_path / "reference", log_interval)
    torch_rng, numpy_rng, python_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
    reference_metrics = reference._train_epoch()
    reference_torch_rng, reference_numpy_rng = torch.get_rng_state(), np.random.get_state()
    reference_python_rng = random.getstate()
    torch.set_rng_state(torch_rng)
    np.random.set_state(numpy_rng)
    random.setstate(python_rng)
    val_metrics = {"loss": 0.5, "ade_m": 0.25, "fde_m": 0.75}
    monkeypatch.setattr(current_train, "evaluate_imitation", lambda *_args: val_metrics)
    trainer.train()

    assert torch.equal(torch.get_rng_state(), reference_torch_rng)
    numpy_rng = np.random.get_state()
    assert numpy_rng[0] == reference_numpy_rng[0] and numpy_rng[2:] == reference_numpy_rng[2:]
    np.testing.assert_array_equal(numpy_rng[1], reference_numpy_rng[1])
    assert random.getstate() == reference_python_rng
    torch.testing.assert_close(trainer.model.weight, reference.model.weight, rtol=0, atol=0)
    assert trainer.optimizer.state_dict() == reference.optimizer.state_dict()
    assert trainer.scheduler.state_dict() == reference.scheduler.state_dict()
    assert trainer.step == reference.step == 3
    events = EventAccumulator(str(tmp_path / "logged" / "run" / "tensorboard")).Reload()
    expected_tags = {"train/" + name for name in (*current_train.TRAIN_LOSS_TERMS, "grad_norm", "lr")}
    expected_tags.update("val/" + name for name in val_metrics)
    assert set(events.Tags()["scalars"]) == expected_tags
    expected_losses = {2: 3.28, 3: (4.0 + 2.56 + 2.0736) / 3.0}
    for index, name in enumerate(current_train.TRAIN_LOSS_TERMS):
        scalars = events.Scalars("train/" + name)
        assert [event.step for event in scalars] == expected_steps
        assert [event.value for event in scalars] == pytest.approx(
            [expected_losses[step] + index for step in expected_steps]
        )
    for name, expected in {"grad_norm": {2: 3.6, 3: 3.36}, "lr": {2: 0.025, 3: 0.0125}}.items():
        assert [event.value for event in events.Scalars("train/" + name)] == pytest.approx(
            [expected[step] for step in expected_steps]
        )
    for name, expected in val_metrics.items():
        (scalar,) = events.Scalars("val/" + name)
        assert scalar.step == 3 and scalar.value == pytest.approx(expected)
    with open(tmp_path / "logged" / "run" / "history.csv", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        (row,) = reader
        assert tuple(reader.fieldnames) == current_train.HISTORY_COLUMNS
    for name, value in reference_metrics.items():
        assert float(row["train_" + name]) == value
    checkpoint = torch.load(tmp_path / "logged" / "run" / "ckpt_last.pth", weights_only=False)
    assert checkpoint["config"] == trainer.config
    assert ("log_interval" in checkpoint["config"]["training"]) == (log_interval is not None)


def test_tensorboard_resume_preserves_completed_step_and_purges_later_events(tmp_path, monkeypatch):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    trainer = _monitoring_trainer(tmp_path)
    log_dir = tmp_path / "run" / "tensorboard"
    with SummaryWriter(str(log_dir)) as writer:
        for step in (0, 3, 4, 6):
            writer.add_scalar("train/loss", 100.0 + step, step)
    trainer.epoch, trainer.step = 1, 3
    trainer.training["epochs"] = 2
    monkeypatch.setattr(current_train, "evaluate_imitation", lambda *_args: {"loss": 0.5, "ade_m": 0.25, "fde_m": 0.75})
    trainer.train()

    scalars = EventAccumulator(str(log_dir)).Reload().Scalars("train/loss")
    assert [event.step for event in scalars] == [0, 3, 4, 6]
    assert [event.value for event in scalars] == pytest.approx([100.0, 103.0, 4.0, (4.0 + 2.56 + 2.0736) / 3.0])


def test_tensorboard_writer_closes_when_training_raises(tmp_path, monkeypatch):
    import torch.utils.tensorboard

    trainer = _monitoring_trainer(tmp_path)
    closed = []

    @contextmanager
    def writer(_log_dir, purge_step):
        assert purge_step == 1
        try:
            yield object()
        finally:
            closed.append(True)

    def fail(_writer=None):
        raise RuntimeError("interrupted training")

    monkeypatch.setattr(torch.utils.tensorboard, "SummaryWriter", writer)
    monkeypatch.setattr(trainer, "_train_epoch", fail)
    with pytest.raises(RuntimeError, match="interrupted training"):
        trainer.train()
    assert closed == [True]
