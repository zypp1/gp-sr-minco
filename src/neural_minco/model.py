"""Current Global Proposal + Shared Refiner planner in plain PyTorch.

The public forward path accepts a normalized range image and physical true-body
state. It returns one fixed-duration, three-segment minimum-jerk action.
"""

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ACCELERATION_LIMIT_MPS2
from .lidar import get_ray_geometry

RANGE_HEIGHT = 30
RANGE_WIDTH = 180
GLOBAL_HEIGHT = 15
GLOBAL_WIDTH = 90
ACTION_DIM = 18
HERMITE_SAMPLES = 10


def _positive(config: Mapping[str, object], key: str) -> float:
    value = float(config[key])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("model.%s must be finite and positive" % key)
    return value


def _positive_int(config: Mapping[str, object], key: str) -> int:
    value = config[key]
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError("model.%s must be a positive integer" % key)
    return int(value)


def validate_config(model_config: Mapping[str, object], lidar_config: Mapping[str, object]) -> None:
    """Validate fixed contracts and internally coupled dimensions."""
    for key, expected in {
        "input_channels": 1,
        "input_height": RANGE_HEIGHT,
        "input_width": RANGE_WIDTH,
        "trajectory_samples": 20,
        "horizon_s": 2.0,
    }.items():
        if model_config.get(key) != expected:
            raise ValueError("model.%s must be %r" % (key, expected))

    integer_keys = (
        "fine_channels",
        "coarse_channels",
        "local_channels",
        "fine_blocks",
        "coarse_blocks",
        "token_dim",
        "attention_heads",
        "ffn_dim",
        "proposal_layers",
        "state_dim",
        "action_feature_dim",
        "relation_dim",
    )
    values = {key: _positive_int(model_config, key) for key in integer_keys}
    if values["token_dim"] % values["attention_heads"]:
        raise ValueError("model.token_dim must be divisible by attention_heads")
    if values["state_dim"] + values["action_feature_dim"] != values["token_dim"]:
        raise ValueError("state_dim + action_feature_dim must equal token_dim")
    # Frozen H10 checkpoints predate the readout/query metadata keys. Their
    # only semantics were joint/rebuild; support geometry must remain explicit.
    for key, expected in {
        "action_readout": "joint",
        "refiner_attention_normalization": "joint",
        "refiner_query_mode": "rebuild",
    }.items():
        if model_config.get(key, expected) != expected:
            raise ValueError("model.%s must be %r for the selected planner" % (key, expected))
    if model_config.get("refiner_position_scheme") != "hermite10":
        raise ValueError("model.refiner_position_scheme must explicitly be 'hermite10'")

    windows = model_config.get("refiner_windows")
    if not isinstance(windows, Sequence) or len(windows) != 3:
        raise ValueError("model.refiner_windows must contain exactly three windows")
    previous = None
    for window in windows:
        if not isinstance(window, Sequence) or len(window) != 2:
            raise ValueError("each Refiner window must be [rows, columns]")
        rows, columns = int(window[0]), int(window[1])
        if (
            rows <= 0
            or columns <= 0
            or rows % 2 == 0
            or columns % 2 == 0
            or rows > RANGE_HEIGHT
            or columns > RANGE_WIDTH
        ):
            raise ValueError("Refiner windows must be positive odd native-grid sizes")
        if previous is not None and (rows > previous[0] or columns > previous[1]):
            raise ValueError("Refiner windows must be non-increasing")
        previous = (rows, columns)

    for key in (
        "waypoint_scale",
        "terminal_velocity_scale",
        "velocity_scale",
        "acceleration_scale",
        "goal_scale",
        "jerk_scale",
        "ratio_correction_step",
        "clearance_radius",
    ):
        _positive(model_config, key)
    dropout = float(model_config["dropout"])
    if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
        raise ValueError("model.dropout must lie in [0,1)")

    expected_lidar = {
        "n_horizon": 180,
        "n_vertical": 90,
        "horizon_fov_deg": 360.0,
        "vertical_fov_min_deg": -7.0,
        "vertical_fov_max_deg": 52.0,
        "n_frames": 1,
    }
    for key, expected in expected_lidar.items():
        if lidar_config.get(key) != expected:
            raise ValueError("lidar.%s must be %r" % (key, expected))
    r_max = float(lidar_config["r_max"])
    if not math.isfinite(r_max) or r_max <= 0.0:
        raise ValueError("lidar.r_max must be finite and positive")


class CircularConv2d(nn.Module):
    """Convolve with circular azimuth and zero-padded elevation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.pad(features, (self.padding, self.padding, 0, 0), mode="circular")
        features = F.pad(features, (0, 0, self.padding, self.padding))
        return self.conv(features)


def _normalization_groups(channels: int) -> int:
    return math.gcd(channels, 8)


class ResidualBlock(nn.Module):
    """A same-resolution residual block for range-image features."""

    def __init__(self, channels: int, dropout: float):
        super().__init__()
        groups = _normalization_groups(channels)
        self.conv1 = CircularConv2d(channels, channels, 3)
        self.norm1 = nn.GroupNorm(groups, channels, affine=False)
        self.conv2 = CircularConv2d(channels, channels, 3)
        self.norm2 = nn.GroupNorm(groups, channels, affine=False)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        features = F.silu(self.norm1(self.conv1(features)))
        features = self.dropout(self.norm2(self.conv2(features)))
        return F.silu(residual + features)


class DualScaleLiDARMemory(nn.Module):
    """Encode one range image into fine local and coarse full-ray memories."""

    _GEOMETRY_BUFFERS = ("ray_direction", "ray_theta", "ray_phi")

    def __init__(self, model_config: Mapping[str, object], lidar_config: Mapping[str, object]):
        super().__init__()
        direction, theta, phi = get_ray_geometry(
            n_horizon=int(lidar_config["n_horizon"]),
            n_vertical=int(lidar_config["n_vertical"]),
            horizon_fov_deg=float(lidar_config["horizon_fov_deg"]),
            vertical_fov_min_deg=float(lidar_config["vertical_fov_min_deg"]),
            vertical_fov_max_deg=float(lidar_config["vertical_fov_max_deg"]),
        )
        if direction.shape != (RANGE_HEIGHT * RANGE_WIDTH, 3):
            raise ValueError("LiDAR geometry must produce a 30x180 range image")
        self.register_buffer("ray_direction", torch.from_numpy(direction).float())
        self.register_buffer("ray_theta", torch.from_numpy(theta).float())
        self.register_buffer("ray_phi", torch.from_numpy(phi).float())

        fine_channels = int(model_config["fine_channels"])
        coarse_channels = int(model_config["coarse_channels"])
        local_channels = int(model_config["local_channels"])
        token_dim = int(model_config["token_dim"])
        dropout = float(model_config["dropout"])
        relation_dim = int(model_config["relation_dim"])
        self.r_max = float(lidar_config["r_max"])
        self.token_dim = token_dim

        self.stem = nn.Sequential(
            CircularConv2d(5, fine_channels, 3),
            nn.GroupNorm(_normalization_groups(fine_channels), fine_channels, affine=False),
            nn.SiLU(),
        )
        self.fine_stage = nn.Sequential(
            *[ResidualBlock(fine_channels, dropout) for _ in range(int(model_config["fine_blocks"]))]
        )
        self.coarse_projection = nn.Sequential(
            CircularConv2d(fine_channels, coarse_channels, 1),
            nn.GroupNorm(_normalization_groups(coarse_channels), coarse_channels, affine=False),
            nn.SiLU(),
        )
        self.coarse_stage = nn.Sequential(
            *[ResidualBlock(coarse_channels, dropout) for _ in range(int(model_config["coarse_blocks"]))]
        )
        self.global_content_projection = CircularConv2d(coarse_channels, token_dim, 1)
        self.local_fine_projection = CircularConv2d(fine_channels, local_channels, 1)
        self.local_coarse_projection = CircularConv2d(coarse_channels, local_channels, 1)
        self.local_refine = nn.Sequential(
            CircularConv2d(local_channels, local_channels, 3),
            nn.GroupNorm(_normalization_groups(local_channels), local_channels, affine=False),
            nn.SiLU(),
        )
        self.local_content_projection = CircularConv2d(local_channels, token_dim, 1)
        self.direction_embedding = nn.Linear(3, token_dim, bias=False)
        self.observation_embedding = nn.Sequential(
            nn.Linear(5, relation_dim, bias=False), nn.SiLU(), nn.Linear(relation_dim, token_dim, bias=False)
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        for name in self._GEOMETRY_BUFFERS:
            key = prefix + name
            expected = self._buffers[name]
            if key in state_dict:
                actual = state_dict[key]
                if actual.shape != expected.shape or not torch.equal(
                    actual.to(device=expected.device, dtype=expected.dtype), expected
                ):
                    error_msgs.append("%s disagrees with deterministic LiDAR geometry" % key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _sensory_input(self, observation: torch.Tensor) -> torch.Tensor:
        normalized_range = observation[:, 0]
        hit_value = (normalized_range < 1.0).to(observation.dtype)
        direction = self.ray_direction.to(observation).reshape(RANGE_HEIGHT, RANGE_WIDTH, 3)
        endpoint = hit_value[..., None] * normalized_range[..., None] * direction
        return torch.cat(
            [(hit_value * (1.0 - normalized_range))[:, None], endpoint.permute(0, 3, 1, 2), hit_value[:, None]], dim=1
        )

    def _semantic_tokens(
        self, content: torch.Tensor, normalized_range: torch.Tensor, direction: torch.Tensor, hit: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hit_value = hit[..., None].to(content.dtype)
        observation_feature = torch.cat(
            [
                hit_value * (1.0 - normalized_range[..., None]),
                hit_value * normalized_range[..., None] * direction,
                hit_value,
            ],
            dim=-1,
        )
        tokens = content + self.direction_embedding(direction) + self.observation_embedding(observation_feature)
        points = self.r_max * normalized_range[..., None] * direction
        return tokens, points

    def _coarse_rays(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = len(observation)
        native_range = observation[:, 0]
        patches = (
            native_range.reshape(batch, GLOBAL_HEIGHT, 2, GLOBAL_WIDTH, 2)
            .permute(0, 1, 3, 2, 4)
            .reshape(batch, GLOBAL_HEIGHT, GLOBAL_WIDTH, 4)
        )
        selected_range, selected_index = patches.min(dim=-1)
        native_direction = (
            self.ray_direction.to(observation)
            .reshape(GLOBAL_HEIGHT, 2, GLOBAL_WIDTH, 2, 3)
            .permute(0, 2, 1, 3, 4)
            .reshape(GLOBAL_HEIGHT, GLOBAL_WIDTH, 4, 3)
        )
        expanded = native_direction[None].expand(batch, -1, -1, -1, -1)
        gather_index = selected_index[..., None, None].expand(-1, -1, -1, 1, 3)
        nearest_direction = torch.gather(expanded, 3, gather_index).squeeze(3)
        mean_direction = F.normalize(native_direction.sum(dim=2), dim=-1)
        hit = selected_range < 1.0
        direction = torch.where(hit[..., None], nearest_direction, mean_direction[None])
        return (selected_range.reshape(batch, -1), direction.reshape(batch, -1, 3), hit.reshape(batch, -1))

    def forward(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch = len(observation)
        fine = self.fine_stage(self.stem(self._sensory_input(observation)))
        coarse = F.avg_pool2d(fine, kernel_size=2, stride=2)
        coarse = self.coarse_stage(self.coarse_projection(coarse))

        global_content = self.global_content_projection(coarse)
        global_content = global_content.permute(0, 2, 3, 1).reshape(batch, GLOBAL_HEIGHT * GLOBAL_WIDTH, self.token_dim)
        local_feature = self.local_fine_projection(fine) + F.interpolate(
            self.local_coarse_projection(coarse), size=(RANGE_HEIGHT, RANGE_WIDTH), mode="nearest"
        )
        local_feature = self.local_refine(local_feature)
        local_content = self.local_content_projection(local_feature)
        local_content = local_content.permute(0, 2, 3, 1).reshape(batch, RANGE_HEIGHT * RANGE_WIDTH, self.token_dim)

        local_range = observation[:, 0].reshape(batch, -1)
        local_hit = local_range < 1.0
        local_direction = self.ray_direction.to(observation)[None].expand(batch, -1, -1)
        local_tokens, local_points = self._semantic_tokens(local_content, local_range, local_direction, local_hit)
        global_range, global_direction, global_hit = self._coarse_rays(observation)
        global_tokens, global_points = self._semantic_tokens(global_content, global_range, global_direction, global_hit)
        return {
            "local_tokens": local_tokens,
            "local_points": local_points,
            "local_hits": local_hit,
            "ray_elevation": self.ray_phi,
            "global_tokens": global_tokens,
            "global_points": global_points,
            "global_hits": global_hit,
        }


class FeedForward(nn.Module):
    """A pre-normalized residual feed-forward block."""

    def __init__(self, token_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.layers = nn.Sequential(
            nn.Linear(token_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim, bias=False),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.layers(self.norm(tokens))


class StateContextEncoder(nn.Module):
    """Map the jointly normalized state to one context vector."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(9, state_dim), nn.SiLU(), nn.Linear(state_dim, state_dim), nn.LayerNorm(state_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


class JointQueryEncoder(nn.Module):
    """Fuse the complete reference action and state into one query token."""

    def __init__(self, action_feature_dim: int, token_dim: int):
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Linear(ACTION_DIM, action_feature_dim),
            nn.SiLU(),
            nn.Linear(action_feature_dim, action_feature_dim),
            nn.LayerNorm(action_feature_dim),
        )
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, action: torch.Tensor, state_context: torch.Tensor) -> torch.Tensor:
        ratios = torch.softmax(action[:, 15:18], dim=-1)
        action_value = torch.cat([action[:, :15], ratios], dim=-1)
        query = torch.cat([self.action_encoder(action_value), state_context], dim=-1)
        return self.norm(query)[:, None]


class ProposalDecoderLayer(nn.Module):
    """One cross-attention and feed-forward layer for the joint query."""

    def __init__(self, token_dim: int, heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(token_dim)
        self.key_norm = nn.LayerNorm(token_dim)
        self.value_norm = nn.LayerNorm(token_dim)
        self.cross_attention = nn.MultiheadAttention(token_dim, heads, dropout=dropout, bias=False, batch_first=True)
        self.ffn = FeedForward(token_dim, ffn_dim, dropout)

    def forward(self, query: torch.Tensor, memory_key: torch.Tensor, memory_value: torch.Tensor) -> torch.Tensor:
        update, _ = self.cross_attention(
            self.query_norm(query), self.key_norm(memory_key), self.value_norm(memory_value), need_weights=False
        )
        return self.ffn(query + update)


class RouteRelativeEncoding(nn.Module):
    """Describe each coarse hit relative to the decoded reference route."""

    def __init__(self, token_dim: int, relation_dim: int, r_max: float, clearance_radius: float):
        super().__init__()
        self.r_max = float(r_max)
        self.clearance_radius = float(clearance_radius)
        self.hidden = nn.Linear(6, relation_dim, bias=False)
        self.key_output = nn.Linear(relation_dim, token_dim, bias=False)
        self.value_output = nn.Linear(relation_dim, token_dim, bias=False)
        nn.init.zeros_(self.key_output.weight)
        nn.init.zeros_(self.value_output.weight)

    def forward(
        self, points: torch.Tensor, hits: torch.Tensor, reference_position: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            distance = torch.cdist(points.detach(), reference_position.detach())
            nearest_distance, nearest_index = distance.min(dim=-1)
            nearest_position = torch.gather(reference_position.detach(), 1, nearest_index[..., None].expand(-1, -1, 3))
            displacement = points.detach() - nearest_position
            feature = torch.cat(
                [
                    (displacement / self.r_max).clamp(-2.0, 2.0),
                    (nearest_distance / self.r_max).clamp(0.0, 2.0)[..., None],
                    torch.tanh((nearest_distance - self.clearance_radius) / self.clearance_radius)[..., None],
                    (nearest_index.to(points.dtype) / float(reference_position.shape[1] - 1))[..., None],
                ],
                dim=-1,
            )
        hidden = F.silu(self.hidden(feature))
        gate = hits[..., None].to(hidden.dtype)
        return gate * self.key_output(hidden), gate * self.value_output(hidden)


def _center_ratio(correction: torch.Tensor) -> torch.Tensor:
    ratio = correction[:, 15:18]
    return torch.cat([correction[:, :15], ratio - ratio.mean(dim=-1, keepdim=True)], dim=-1)


def _bounded_residual(reference: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
    bound = 1.0 - 1e-5
    return torch.tanh(torch.atanh(reference.clamp(-bound, bound)) + correction)


class GlobalProposal(nn.Module):
    """Apply three independent decoder layers over all 1,350 coarse rays."""

    def __init__(self, model_config: Mapping[str, object], r_max: float):
        super().__init__()
        token_dim = int(model_config["token_dim"])
        self.token_dim = token_dim
        self.route_encoding = RouteRelativeEncoding(
            token_dim, int(model_config["relation_dim"]), r_max, float(model_config["clearance_radius"])
        )
        self.decoder_layers = nn.ModuleList(
            [
                ProposalDecoderLayer(
                    token_dim,
                    int(model_config["attention_heads"]),
                    int(model_config["ffn_dim"]),
                    float(model_config["dropout"]),
                )
                for _ in range(int(model_config["proposal_layers"]))
            ]
        )
        self.output_head = nn.Sequential(
            nn.Linear(token_dim, token_dim, bias=False), nn.SiLU(), nn.Linear(token_dim, ACTION_DIM, bias=False)
        )
        nn.init.normal_(self.output_head[-1].weight, mean=0.0, std=1e-3)

    def forward(
        self, query: torch.Tensor, memory: Mapping[str, torch.Tensor], reference_position: torch.Tensor
    ) -> torch.Tensor:
        """Return one joint action correction after all three decoder layers."""
        global_tokens = memory["global_tokens"]
        key_update, value_update = self.route_encoding(
            memory["global_points"], memory["global_hits"], reference_position
        )
        memory_key = global_tokens + key_update
        memory_value = global_tokens + value_update
        for layer in self.decoder_layers:
            query = layer(query, memory_key, memory_value)
        return _center_ratio(self.output_head(query[:, 0]))


class JointPositionWindowAttention(nn.Module):
    """Let one query read the union of trajectory-centered windows."""

    def __init__(self, model_config: Mapping[str, object], r_max: float):
        super().__init__()
        token_dim = int(model_config["token_dim"])
        heads = int(model_config["attention_heads"])
        relation_dim = int(model_config["relation_dim"])
        windows = [tuple(int(value) for value in item) for item in model_config["refiner_windows"]]
        self.token_dim = token_dim
        self.heads = heads
        self.head_dim = token_dim // heads
        self.scale = self.head_dim**-0.5
        self.r_max = float(r_max)
        self.clearance_radius = float(model_config["clearance_radius"])
        self.position_count = HERMITE_SAMPLES
        self.maximum_rows = max(window[0] for window in windows)
        self.maximum_columns = max(window[1] for window in windows)
        self.query = nn.Linear(token_dim, token_dim, bias=False)
        self.key = nn.Linear(token_dim, token_dim, bias=False)
        self.value = nn.Linear(token_dim, token_dim, bias=False)
        self.output = nn.Linear(token_dim, token_dim, bias=False)
        self.position_encoding = nn.Sequential(
            nn.Linear(7, relation_dim, bias=False), nn.SiLU(), nn.Linear(relation_dim, token_dim, bias=False)
        )
        self.position_role = nn.Parameter(torch.zeros(self.position_count, token_dim))
        self.relative_position = nn.Sequential(
            nn.Linear(8, relation_dim, bias=False), nn.SiLU(), nn.Linear(relation_dim, token_dim, bias=False)
        )
        self.relative_position_score = nn.Linear(token_dim, heads, bias=False)
        self.direction_relation = nn.Linear(4, relation_dim, bias=False)
        nn.init.zeros_(self.direction_relation.weight)
        self.dropout = nn.Dropout(float(model_config["dropout"]))

        row_radius = self.maximum_rows // 2
        column_radius = self.maximum_columns // 2
        row, column = torch.meshgrid(
            torch.arange(-row_radius, row_radius + 1, dtype=torch.long),
            torch.arange(-column_radius, column_radius + 1, dtype=torch.long),
            indexing="ij",
        )
        self.register_buffer("row_offset", row.reshape(-1))
        self.register_buffer("column_offset", column.reshape(-1))

    def _window_indices(
        self, position: torch.Tensor, tangent: torch.Tensor, ray_elevation: torch.Tensor, window: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        position_norm = torch.linalg.vector_norm(position, dim=-1, keepdim=True)
        tangent_norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
        fallback = torch.zeros_like(position)
        fallback[..., 0] = 1.0
        direction = torch.where(
            position_norm > 1e-4,
            position / position_norm.clamp_min(1e-6),
            torch.where(tangent_norm > 1e-4, tangent / tangent_norm.clamp_min(1e-6), fallback),
        )
        azimuth = torch.atan2(direction[..., 1], direction[..., 0])
        elevation = torch.asin(direction[..., 2].clamp(-1.0, 1.0))
        center_column = torch.floor((azimuth + math.pi) / (2.0 * math.pi) * RANGE_WIDTH).long() % RANGE_WIDTH
        row_elevation = ray_elevation.reshape(RANGE_HEIGHT, RANGE_WIDTH)[:, 0].to(position)
        center_row = (elevation[..., None] - row_elevation).abs().argmin(dim=-1)

        active = (self.row_offset.abs() <= window[0] // 2) & (self.column_offset.abs() <= window[1] // 2)
        row_offset = self.row_offset[active]
        column_offset = self.column_offset[active]
        row = center_row[..., None] + row_offset
        column = center_column[..., None] + column_offset
        valid = (row >= 0) & (row < RANGE_HEIGHT)
        index = row.clamp(0, RANGE_HEIGHT - 1) * RANGE_WIDTH
        index = index + column.remainder(RANGE_WIDTH)
        return index, valid, row_offset, column_offset

    def forward(
        self,
        query: torch.Tensor,
        position: torch.Tensor,
        tangent: torch.Tensor,
        descriptor: torch.Tensor,
        memory: Mapping[str, torch.Tensor],
        window: Tuple[int, int],
    ) -> torch.Tensor:
        """Read full local windows with one joint Q/K/V attention operation."""
        batch, _, channels = query.shape
        index, valid, row_offset, column_offset = self._window_indices(
            position, tangent, memory["ray_elevation"], window
        )
        support_size = index.shape[-1]
        batch_index = torch.arange(batch, device=query.device)[:, None, None]
        local_tokens = memory["local_tokens"][batch_index, index]
        local_points = memory["local_points"][batch_index, index]
        local_hits = memory["local_hits"][batch_index, index]

        position_context = self.position_encoding(descriptor) + self.position_role[None]
        local_tokens = local_tokens + position_context[:, :, None]
        key_count = self.position_count * support_size
        local_tokens = local_tokens.reshape(batch, key_count, channels)
        q = self.query(query).reshape(batch, 1, self.heads, self.head_dim)
        k = self.key(local_tokens).reshape(batch, key_count, self.heads, self.head_dim)
        v = self.value(local_tokens).reshape(batch, key_count, self.heads, self.head_dim)

        relative_message = self._relative_message(
            position, tangent, local_points, local_hits, valid, row_offset, column_offset, query.dtype
        ).reshape(batch, key_count, channels)
        relative_by_head = relative_message.reshape(batch, key_count, self.heads, self.head_dim)

        score = torch.einsum("bqhd,bkhd->bhqk", q, k) * self.scale
        score = score + self.relative_position_score(relative_message).permute(0, 2, 1)[:, :, None]
        valid = valid.reshape(batch, key_count)
        score = score.masked_fill(~valid[:, None, None], -torch.inf)
        probability = self.dropout(torch.softmax(score, dim=-1))
        content = torch.einsum("bhqk,bkhd->bqhd", probability, v)
        relative_content = torch.einsum("bhqk,bkhd->bqhd", probability, relative_by_head)
        return self.output((content + relative_content).reshape(batch, 1, channels))

    def _relative_message(
        self,
        position: torch.Tensor,
        tangent: torch.Tensor,
        local_points: torch.Tensor,
        local_hits: torch.Tensor,
        valid: torch.Tensor,
        row_offset: torch.Tensor,
        column_offset: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode eight clearance/window and four direction features per ray."""
        batch = len(position)
        displacement = local_points - position[:, :, None]
        hit_displacement = torch.where(local_hits[..., None], displacement / self.r_max, torch.zeros_like(displacement))
        radial_margin = (
            torch.linalg.vector_norm(local_points, dim=-1) - torch.linalg.vector_norm(position, dim=-1)[..., None]
        ) / self.r_max
        row_scale = float(max(self.maximum_rows // 2, 1))
        column_scale = float(max(self.maximum_columns // 2, 1))
        rows = row_offset.view(1, 1, -1).expand(batch, self.position_count, -1)
        columns = column_offset.view(1, 1, -1).expand(batch, self.position_count, -1)
        relative = torch.cat(
            [
                hit_displacement,
                radial_margin[..., None],
                (rows.to(dtype) / row_scale)[..., None],
                (columns.to(dtype) / column_scale)[..., None],
                local_hits[..., None].to(dtype),
            ],
            dim=-1,
        )
        signed_clearance = (
            torch.linalg.vector_norm(displacement, dim=-1) - self.clearance_radius
        ) / self.clearance_radius
        visible = local_hits & valid
        clearance = torch.where(
            visible[..., None], torch.tanh(signed_clearance)[..., None], torch.zeros_like(signed_clearance)[..., None]
        )

        obstacle_distance = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
        obstacle_direction = displacement / obstacle_distance.clamp_min(1e-6)
        unit_tangent = tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1e-6)
        alignment = (obstacle_direction * unit_tangent[:, :, None]).sum(dim=-1, keepdim=True)
        direction = torch.cat([obstacle_direction, alignment], dim=-1)
        direction = torch.where(visible[..., None], direction, torch.zeros_like(direction))

        relative_hidden = self.relative_position[0](torch.cat([relative, clearance], dim=-1))
        relative_hidden = relative_hidden + self.direction_relation(direction)
        return self.relative_position[2](self.relative_position[1](relative_hidden))


class SharedRefiner(nn.Module):
    """Predict one joint residual from ten Hermite-centered LiDAR windows."""

    def __init__(self, model_config: Mapping[str, object], r_max: float):
        super().__init__()
        token_dim = int(model_config["token_dim"])
        dropout = float(model_config["dropout"])
        self.waypoint_scale = float(model_config["waypoint_scale"])
        self.v_end_scale = float(model_config["terminal_velocity_scale"])
        self.horizon = float(model_config["horizon_s"])
        self.proxy_samples = HERMITE_SAMPLES
        self.stage_embedding = nn.Parameter(torch.zeros(3, token_dim))
        self.query_norm = nn.LayerNorm(token_dim)
        self.local_attention = JointPositionWindowAttention(model_config, r_max)
        self.ffn = FeedForward(token_dim, int(model_config["ffn_dim"]), dropout)
        self.output_head = nn.Sequential(
            nn.Linear(token_dim, token_dim, bias=False), nn.SiLU(), nn.Linear(token_dim, ACTION_DIM, bias=False)
        )
        nn.init.zeros_(self.output_head[-1].weight)

    def position_context(
        self, action: torch.Tensor, physical_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build ten physical lookup centers, tangents and 7D descriptors."""
        batch = len(action)
        normalized_knots = action[:, :9].reshape(batch, 3, 3)
        knots = torch.cat([action.new_zeros(batch, 1, 3), self.waypoint_scale * normalized_knots], dim=1)
        ratio = torch.softmax(action[:, 15:18], dim=-1)
        duration = self.horizon * ratio
        segment_end = duration.cumsum(dim=1)
        progress = torch.arange(1, self.proxy_samples + 1, device=action.device, dtype=action.dtype)
        progress = (progress / float(self.proxy_samples))[None].expand(batch, -1)
        time = self.horizon * progress
        segment = torch.searchsorted(segment_end.contiguous(), time.contiguous()).clamp(max=2)
        segment_start = torch.cat([duration.new_zeros(batch, 1), segment_end[:, :-1]], dim=1)
        local_time = time - torch.gather(segment_start, 1, segment)
        local_duration = torch.gather(duration, 1, segment)
        u = (local_time / local_duration)[..., None]

        internal_velocity_1 = (knots[:, 2] - knots[:, 0]) / (duration[:, 0:1] + duration[:, 1:2])
        internal_velocity_2 = (knots[:, 3] - knots[:, 1]) / (duration[:, 1:2] + duration[:, 2:3])
        velocity = torch.stack(
            [physical_state[:, :3], internal_velocity_1, internal_velocity_2, self.v_end_scale * action[:, 9:12]], dim=1
        )

        def select(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
            return torch.gather(value, 1, index[..., None].expand(-1, -1, 3))

        p0 = select(knots[:, :-1], segment)
        p1 = select(knots[:, 1:], segment)
        v0 = select(velocity[:, :-1], segment)
        v1 = select(velocity[:, 1:], segment)
        dt = local_duration[..., None]
        u2 = u.square()
        u3 = u2 * u
        position = (
            (2.0 * u3 - 3.0 * u2 + 1.0) * p0
            + (u3 - 2.0 * u2 + u) * dt * v0
            + (-2.0 * u3 + 3.0 * u2) * p1
            + (u3 - u2) * dt * v1
        )
        tangent = (
            (6.0 * u2 - 6.0 * u) * p0 / dt
            + (3.0 * u2 - 4.0 * u + 1.0) * v0
            + (-6.0 * u2 + 6.0 * u) * p1 / dt
            + (3.0 * u2 - 2.0 * u) * v1
        )
        unit_tangent = tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1e-6)
        descriptor = torch.cat([position / self.waypoint_scale, unit_tangent, progress[..., None]], dim=-1)
        return position, tangent, descriptor

    def forward(
        self,
        parameter_query: torch.Tensor,
        action: torch.Tensor,
        physical_state: torch.Tensor,
        memory: Mapping[str, torch.Tensor],
        window: Tuple[int, int],
        stage: int,
    ) -> torch.Tensor:
        """Re-query fresh support and return one joint action correction."""
        position, tangent, descriptor = self.position_context(action, physical_state)
        query = parameter_query + self.stage_embedding[stage].view(1, 1, -1)
        update = self.local_attention(self.query_norm(query), position, tangent, descriptor, memory, window)
        query = self.ffn(query + update)
        return _center_ratio(self.output_head(query[:, 0]))


def yaw_frame(state_body: torch.Tensor) -> torch.Tensor:
    """Return goal-, velocity-, or acceleration-aligned horizontal axes."""
    candidates = (state_body[:, 6:8], state_body[:, 0:2], state_body[:, 3:5])
    selected = torch.zeros_like(candidates[0])
    unresolved = torch.ones(len(state_body), dtype=torch.bool, device=state_body.device)
    for candidate in candidates:
        usable = torch.linalg.vector_norm(candidate, dim=-1) > 1e-6
        take = unresolved & usable
        selected = torch.where(take[:, None], candidate, selected)
        unresolved = unresolved & ~usable
    norm = torch.linalg.vector_norm(selected, dim=-1, keepdim=True)
    fallback = torch.zeros_like(selected)
    fallback[:, 0] = 1.0
    return torch.where(norm > 1e-6, selected / norm.clamp_min(1e-6), fallback)


def _rotate_to_frame(value: torch.Tensor, forward: torch.Tensor) -> torch.Tensor:
    shape = (value.shape[0],) + (1,) * (value.ndim - 2)
    forward_x = forward[:, 0].reshape(shape)
    forward_y = forward[:, 1].reshape(shape)
    x = value[..., 0] * forward_x + value[..., 1] * forward_y
    y = -value[..., 0] * forward_y + value[..., 1] * forward_x
    return torch.stack([x, y, value[..., 2]], dim=-1)


def _rotate_from_frame(value: torch.Tensor, forward: torch.Tensor) -> torch.Tensor:
    shape = (value.shape[0],) + (1,) * (value.ndim - 2)
    forward_x = forward[:, 0].reshape(shape)
    forward_y = forward[:, 1].reshape(shape)
    x = value[..., 0] * forward_x - value[..., 1] * forward_y
    y = value[..., 0] * forward_y + value[..., 1] * forward_x
    return torch.stack([x, y, value[..., 2]], dim=-1)


def _canonical_roll(observation: torch.Tensor, forward: torch.Tensor) -> torch.Tensor:
    width = observation.shape[-1]
    angle = torch.atan2(forward[:, 1], forward[:, 0])
    shift = torch.round(angle * float(width) / (2.0 * math.pi)).long()
    columns = (torch.arange(width, device=observation.device)[None] + shift[:, None]).remainder(width)
    return torch.gather(
        observation, 3, columns[:, None, None].expand(-1, observation.shape[1], observation.shape[2], -1)
    )


def _radial_clip(value: torch.Tensor, limit: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    return value * (float(limit) / norm.clamp_min(float(limit))).clamp(max=1.0)


def _radial_tanh(value: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    scale = torch.where(norm > 1e-7, torch.tanh(norm) / norm.clamp_min(1e-7), torch.ones_like(norm))
    return value * scale


def _radial_atanh(value: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    limit = 1.0 - torch.finfo(value.dtype).eps
    scale = torch.where(norm > 1e-7, torch.atanh(norm.clamp_max(limit)) / norm.clamp_min(1e-7), torch.ones_like(norm))
    return value * scale


def _update_action(reference: torch.Tensor, correction: torch.Tensor, ratio_step: float) -> torch.Tensor:
    geometry = _bounded_residual(reference[:, :12], correction[:, :12])
    acceleration = _radial_tanh(_radial_atanh(reference[:, 12:15]) + correction[:, 12:15])
    ratio = reference[:, 15:18] + ratio_step * torch.tanh(correction[:, 15:18])
    ratio = ratio - ratio.mean(dim=-1, keepdim=True)
    return torch.cat([geometry, acceleration, ratio], dim=-1)


def _raw_from_action(action: torch.Tensor, canonical_forward: torch.Tensor) -> Dict[str, torch.Tensor]:
    parameters = action[:, :15].reshape(len(action), 5, 3)
    body = _rotate_from_frame(parameters, canonical_forward)
    return {
        "Q": body[:, 0:2],
        "p_end": body[:, 2],
        "v_end": body[:, 3],
        "a_end": body[:, 4],
        "ratio_logits": action[:, 15:18],
    }


class ConstantVelocityReference(nn.Module):
    """Construct a uniform-time constant-velocity initial action."""

    def __init__(self, horizon_s: float, waypoint_scale: float, terminal_velocity_scale: float):
        super().__init__()
        self.horizon = float(horizon_s)
        self.waypoint_scale = float(waypoint_scale)
        self.velocity_scale = float(terminal_velocity_scale)

    @torch.no_grad()
    def forward(self, physical_state: torch.Tensor) -> torch.Tensor:
        velocity = physical_state[:, :3]
        times = physical_state.new_tensor([self.horizon / 3.0, 2.0 * self.horizon / 3.0, self.horizon])
        position = velocity[:, None] * times[None, :, None]
        vectors = torch.cat(
            [(position / self.waypoint_scale).flatten(1), velocity / self.velocity_scale, torch.zeros_like(velocity)],
            dim=1,
        )
        bound = 1.0 - 1e-5
        return torch.cat(
            [
                vectors[:, :12].clamp(-bound, bound),
                _radial_clip(vectors[:, 12:15], 0.995),
                physical_state.new_zeros(len(physical_state), 3),
            ],
            dim=1,
        )


class TrajectoryAttentionPlanner(nn.Module):
    """One Global Proposal followed by three calls to one shared Refiner."""

    def __init__(self, model_config: Mapping[str, object], lidar_config: Mapping[str, object]):
        super().__init__()
        validate_config(model_config, lidar_config)
        self.model_config = dict(model_config)
        self.lidar_config = dict(lidar_config)
        self.horizon = float(model_config["horizon_s"])
        self.waypoint_scale = float(model_config["waypoint_scale"])
        self.v_end_scale = float(model_config["terminal_velocity_scale"])
        self.velocity_scale = float(model_config["velocity_scale"])
        self.acceleration_scale = float(model_config["acceleration_scale"])
        self.goal_scale = float(model_config["goal_scale"])
        self.jerk_scale = float(model_config["jerk_scale"])
        self.ratio_correction_step = float(model_config["ratio_correction_step"])
        self.refiner_windows = tuple(
            tuple(int(value) for value in window) for window in model_config["refiner_windows"]
        )

        self.encoder = DualScaleLiDARMemory(model_config, lidar_config)
        self.reference = ConstantVelocityReference(
            horizon_s=self.horizon, waypoint_scale=self.waypoint_scale, terminal_velocity_scale=self.v_end_scale
        )
        self.state_encoder = StateContextEncoder(int(model_config["state_dim"]))
        self.query_encoder = JointQueryEncoder(int(model_config["action_feature_dim"]), int(model_config["token_dim"]))
        self.global_proposal = GlobalProposal(model_config, float(lidar_config["r_max"]))
        self.shared_refiner = SharedRefiner(model_config, float(lidar_config["r_max"]))
        self.register_buffer(
            "reference_times",
            torch.arange(1, int(model_config["trajectory_samples"]) + 1).float()
            * (self.horizon / float(model_config["trajectory_samples"])),
        )

    def _normalize_state(self, physical_state: torch.Tensor) -> torch.Tensor:
        scale = physical_state.new_tensor(
            [self.velocity_scale] * 3 + [self.acceleration_scale] * 3 + [self.goal_scale] * 3
        )
        return physical_state / scale[None]

    @staticmethod
    def _validate_inputs(observation: torch.Tensor, state_body: torch.Tensor) -> None:
        if not isinstance(observation, torch.Tensor) or not observation.is_floating_point():
            raise ValueError("observation must be a floating tensor")
        if not isinstance(state_body, torch.Tensor) or not state_body.is_floating_point():
            raise ValueError("physical state must be a floating tensor")
        if observation.ndim != 4 or tuple(observation.shape[1:]) != (1, RANGE_HEIGHT, RANGE_WIDTH):
            raise ValueError("observation must have shape (B,1,30,180)")
        if state_body.ndim != 2 or state_body.shape != (len(observation), 9):
            raise ValueError("physical state must have shape (B,9)")
        if observation.device != state_body.device:
            raise ValueError("observation and physical state must share a device")
        observation_is_valid = (
            torch.isfinite(observation).all() & (observation >= 0.0).all() & (observation <= 1.0).all()
        )
        state_is_valid = torch.isfinite(state_body).all()
        if observation.is_cuda:
            torch._assert_async(observation_is_valid, "observation must be finite and normalized to [0,1]")
            torch._assert_async(state_is_valid, "physical state must be finite")
        else:
            if not bool(observation_is_valid):
                raise ValueError("observation must be finite and normalized to [0,1]")
            if not bool(state_is_valid):
                raise ValueError("physical state must be finite")

    @torch.no_grad()
    def _reference_positions(self, physical_state: torch.Tensor) -> torch.Tensor:
        """Sample the analytic constant-velocity path without invoking MINCO."""
        velocity = physical_state[:, :3]
        return velocity[:, None] * self.reference_times.to(physical_state)[None, :, None]

    def forward(self, observation: torch.Tensor, state_body: torch.Tensor, return_stages: bool = False):
        """Predict from normalized LiDAR and physical body-frame state.

        The default result is the final z3 raw action. With ``return_stages``,
        the second result contains all four raw actions and their correction
        norms for the Stage-1 objective.
        """
        self._validate_inputs(observation, state_body)
        canonical_forward = yaw_frame(state_body)
        canonical_observation = _canonical_roll(observation, canonical_forward)
        canonical_physical = _rotate_to_frame(state_body.reshape(-1, 3, 3), canonical_forward).reshape(-1, 9)
        normalized_state = self._normalize_state(canonical_physical)

        memory = self.encoder(canonical_observation)
        state_context = self.state_encoder(normalized_state)
        action = self.reference(canonical_physical)
        reference_position = self._reference_positions(canonical_physical)
        query = self.query_encoder(action.detach(), state_context)
        correction = self.global_proposal(query, memory, reference_position)
        action = _update_action(action, correction, self.ratio_correction_step)
        if return_stages:
            stages = [_raw_from_action(action, canonical_forward)]
            correction_norms = [torch.linalg.vector_norm(correction, dim=-1)]

        for stage, window in enumerate(self.refiner_windows):
            previous_action = action.detach()
            query = self.query_encoder(previous_action, state_context)
            correction = self.shared_refiner(query, previous_action, canonical_physical, memory, window, stage)
            action = _update_action(previous_action, correction, self.ratio_correction_step)
            if return_stages:
                stages.append(_raw_from_action(action, canonical_forward))
                correction_norms.append(torch.linalg.vector_norm(correction, dim=-1))

        if not return_stages:
            return _raw_from_action(action, canonical_forward)
        final = stages[-1]
        return final, {"raw_stages": tuple(stages), "correction_norms": torch.stack(correction_norms, dim=-1)}


def decode_action(
    raw: Mapping[str, torch.Tensor],
    *,
    waypoint_scale: float,
    terminal_velocity_scale: float,
    terminal_acceleration_scale: float = ACCELERATION_LIMIT_MPS2,
    horizon: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode a raw network action into physical M3 boundary values."""
    logits = raw["ratio_logits"]
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("ratio_logits must have shape (B,3)")
    return (
        float(waypoint_scale) * raw["Q"],
        torch.softmax(logits, dim=-1) * float(horizon),
        float(waypoint_scale) * raw["p_end"],
        float(terminal_velocity_scale) * raw["v_end"],
        float(terminal_acceleration_scale) * raw["a_end"],
    )


def build_model(config: Optional[Mapping[str, object]] = None) -> TrajectoryAttentionPlanner:
    """Build the current planner from a complete configuration."""
    if config is None:
        from .config import default_config

        config = default_config()
    return TrajectoryAttentionPlanner(config["model"], config["lidar"])


def load_model(path: str, device: str = "cpu"):
    """Strictly load a checkpoint produced by this clean implementation."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError("checkpoint must contain config and state_dict")
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint
