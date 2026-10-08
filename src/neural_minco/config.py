"""Configuration defaults for the current GP-SR planner."""

import copy

COLLISION_RADIUS_M = 0.3
VELOCITY_LIMIT_MPS = 6.0
ACCELERATION_LIMIT_MPS2 = 6.0
JERK_LIMIT_MPS3 = 12.0
SENSOR_SYNC_TOLERANCE_S = 0.005

_CONFIG = {
    "model": {
        "input_channels": 1,
        "input_height": 30,
        "input_width": 180,
        "fine_channels": 64,
        "coarse_channels": 128,
        "local_channels": 64,
        "fine_blocks": 2,
        "coarse_blocks": 3,
        "token_dim": 256,
        "attention_heads": 8,
        "ffn_dim": 1024,
        "proposal_layers": 3,
        "state_dim": 128,
        "action_feature_dim": 128,
        "action_readout": "joint",
        "relation_dim": 64,
        "refiner_attention_normalization": "joint",
        "refiner_position_scheme": "hermite10",
        "refiner_query_mode": "rebuild",
        "refiner_windows": [[9, 17], [5, 9], [3, 5]],
        "trajectory_samples": 20,
        "dropout": 0.0,
        "horizon_s": 2.0,
        "waypoint_scale": 12.0,
        "terminal_velocity_scale": VELOCITY_LIMIT_MPS,
        "velocity_scale": VELOCITY_LIMIT_MPS,
        "acceleration_scale": ACCELERATION_LIMIT_MPS2,
        "goal_scale": 30.0,
        "jerk_scale": JERK_LIMIT_MPS3,
        "ratio_correction_step": 0.5,
        "clearance_radius": COLLISION_RADIUS_M,
    },
    "lidar": {
        "r_max": 20.0,
        "n_horizon": 180,
        "n_vertical": 90,
        "horizon_fov_deg": 360.0,
        "vertical_fov_min_deg": -7.0,
        "vertical_fov_max_deg": 52.0,
        "n_frames": 1,
    },
    "data": {"horizon": 2.0, "trajectory_dt": 0.1, "trajectory_samples": 20, "anchor_stride": 5},
    "loss": {
        "position_beta": 0.3,
        "velocity_weight": 0.25,
        "terminal_acceleration_weight": 0.1,
        "terminal_acceleration_beta": 0.02,
        "delta_weight": 0.01,
        "minimum_ratio": 0.05,
        "ratio_barrier_weight": 0.01,
        "z0_ratio_weight": 0.1,
        "jerk_nonregression_weight": 0.03,
    },
    "training": {
        "seed": 42,
        "epochs": 10,
        "batch_size": 32,
        "num_workers": 8,
        "log_interval": 20,
        "learning_rate": 3.0e-4,
        "minimum_learning_rate": 1.0e-5,
        "weight_decay": 1.0e-4,
        "gradient_clip": 1.0,
        "output_scale_headroom": 1.05,
    },
}


def default_config():
    """Return an independent copy of the complete configuration."""
    return copy.deepcopy(_CONFIG)


def stage2_config(source, *, dynamics_weight=0.0, collision_weight=0.0):
    """Preserve fitted Stage-1 metadata and start a new Refiner-only schedule."""
    config = copy.deepcopy(source)
    config["training"].update(learning_rate=3e-5, minimum_learning_rate=1e-5, num_workers=2)
    config["stage2"] = {
        "trainable_module": "shared_refiner",
        "dynamics_weight": float(dynamics_weight),
        "collision_weight": float(collision_weight),
        "dynamics_samples_per_segment": 1001,
        "collision_samples": 128,
        "clearance_m": COLLISION_RADIUS_M,
        "ground_height_m": 0.0,
        "map_voxel_m": 0.2,
        "prefetch_factor": 2,
        "primary_checkpoint": "ckpt_last.pth",
    }
    return config
