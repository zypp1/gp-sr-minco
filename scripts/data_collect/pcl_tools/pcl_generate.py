"""Generate deterministic static XYZ maps for point-cloud simulation.

The output pair contains identical ordered float32 points:
PCD is binary for the simulator, while NPZ stores ``points[N,3]`` for map-aware tools.
Use ``--scene {forest,buildings,mixed}`` for complete presets and ``--print-config`` to inspect overrides.

    python3 pcl_generate.py --scene forest --output output/maps/forest_seed1 --preview
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

if __package__:
    from .pcl_convert import write_pcd_binary
else:
    from pcl_convert import write_pcd_binary

PathLike = Union[str, Path]

SCENE_TYPES = ("forest", "buildings", "mixed")
TREE_STYLES = ("mixed", "broadleaf", "conifer")
TREE_FORMS = ("upright", "spreading", "leaning", "orchard", "willow", "shrub", "conifer")
TREE_FORM_OPTIONS = ("mixed", *TREE_FORMS)
LOW_TREE_FORMS = ("shrub", "orchard", "spreading")

# Preferred height, crown radius, crown height, trunk radius, and lean ranges.
_TREE_FORM_PROFILES = {
    "upright": ((7.0, 18.0), (1.8, 5.0), (1.4, 4.2), (0.20, 0.55), (0.00, 0.07)),
    "spreading": ((4.0, 9.0), (3.0, 6.0), (0.9, 2.0), (0.18, 0.45), (0.03, 0.14)),
    "leaning": ((5.0, 13.0), (2.0, 5.0), (1.2, 3.2), (0.16, 0.45), (0.12, 0.27)),
    "orchard": ((2.0, 4.2), (1.8, 3.8), (0.8, 1.7), (0.11, 0.30), (0.01, 0.10)),
    "willow": ((8.0, 16.0), (3.8, 6.0), (2.5, 5.5), (0.35, 0.70), (0.00, 0.08)),
    "shrub": ((1.8, 3.5), (0.8, 2.1), (0.45, 0.95), (0.06, 0.16), (0.02, 0.16)),
    "conifer": ((6.0, 18.0), (1.1, 3.2), (0.0, 0.0), (0.16, 0.48), (0.00, 0.08)),
}


@dataclass(frozen=True)
class SceneConfig:
    """Define one reproducible static scene in metres."""

    scene: str = "forest"
    tree_style: str = "mixed"
    tree_form: str = "mixed"
    size_x: float = 150.0
    size_y: float = 150.0
    seed: int = 1
    tree_count: int = 400
    building_count: int = 0
    clear_radius: float = 6.0
    ground_spacing: float = 0.20
    obstacle_spacing: float = 0.12
    crown_spacing: float = 0.18
    crown_fill: float = 0.14
    voxel_size: float = 0.10
    tree_min_spacing: float = 2.0
    low_tree_min_spacing: float = 4.0
    trunk_radius_min: float = 0.08
    trunk_radius_max: float = 0.70
    tree_height_min: float = 2.0
    tree_height_max: float = 18.0
    crown_radius_min: float = 0.8
    crown_radius_max: float = 6.0
    crown_height_min: float = 0.6
    crown_height_max: float = 6.5
    building_size_min: float = 24.0
    building_size_max: float = 24.0
    building_height_min: float = 10.0
    building_height_max: float = 10.0
    building_gap: float = 2.0
    tree_building_gap: float = 0.3
    wall_thickness: float = 0.24
    max_placement_attempts: int = 5000


@dataclass(frozen=True)
class ScenePreset:
    """Pair one complete scene configuration with its user-facing purpose."""

    description: str
    config: SceneConfig


SCENE_PRESETS: Dict[str, ScenePreset] = {
    "forest": ScenePreset(
        description="150 x 150 m varied forest with 400 trees and no buildings", config=SceneConfig()
    ),
    "buildings": ScenePreset(
        description="150 x 150 m low-rise block with eight fixed 24 x 24 x 10 m buildings",
        config=SceneConfig(scene="buildings", tree_count=0, building_count=8),
    ),
    "mixed": ScenePreset(
        description="200 x 200 m dense forest with 1,800 trees and four fixed 24 x 24 x 10 m buildings",
        config=SceneConfig(scene="mixed", size_x=200.0, size_y=200.0, tree_count=1800, building_count=4),
    ),
}


@dataclass(frozen=True)
class SceneResult:
    """Store generated points and their self-contained provenance metadata."""

    points: np.ndarray
    metadata: Dict[str, object]


@dataclass(frozen=True)
class _Tree:
    x: float
    y: float
    trunk_radius: float
    trunk_height: float
    crown_radius_x: float
    crown_radius_y: float
    crown_radius_z: float
    crown_center_z: float
    yaw: float
    style: str
    form: str
    lean_x: float
    lean_y: float
    stem_wander: float
    trunk_wave_phase: float

    @property
    def footprint_radius(self) -> float:
        lean = math.hypot(self.lean_x, self.lean_y)
        return lean + 1.40 * max(self.crown_radius_x, self.crown_radius_y) + 0.35


@dataclass(frozen=True)
class _Building:
    x: float
    y: float
    width: float
    depth: float
    height: float
    yaw: float

    @property
    def footprint_radius(self) -> float:
        return 0.5 * math.hypot(self.width, self.depth)


def _validate_config(config: SceneConfig) -> None:
    """Reject configurations that cannot define a usable bounded scene."""
    if config.scene not in SCENE_TYPES:
        raise ValueError(f"scene must be one of {SCENE_TYPES}, got {config.scene!r}")
    if config.tree_style not in TREE_STYLES:
        raise ValueError(f"tree_style must be one of {TREE_STYLES}, got {config.tree_style!r}")
    if config.tree_form not in TREE_FORM_OPTIONS:
        raise ValueError(f"tree_form must be one of {TREE_FORM_OPTIONS}, got {config.tree_form!r}")
    if config.tree_form != "mixed" and config.tree_style != "mixed":
        form_style = "conifer" if config.tree_form == "conifer" else "broadleaf"
        if form_style != config.tree_style:
            raise ValueError("tree_form conflicts with tree_style")
    finite_values = [
        config.size_x,
        config.size_y,
        config.clear_radius,
        config.ground_spacing,
        config.obstacle_spacing,
        config.crown_spacing,
        config.crown_fill,
        config.voxel_size,
        config.tree_min_spacing,
        config.low_tree_min_spacing,
        config.trunk_radius_min,
        config.trunk_radius_max,
        config.tree_height_min,
        config.tree_height_max,
        config.crown_radius_min,
        config.crown_radius_max,
        config.crown_height_min,
        config.crown_height_max,
        config.building_size_min,
        config.building_size_max,
        config.building_height_min,
        config.building_height_max,
        config.building_gap,
        config.tree_building_gap,
        config.wall_thickness,
    ]
    if not np.isfinite(finite_values).all():
        raise ValueError("scene parameters must be finite")
    if config.size_x <= 0.0 or config.size_y <= 0.0:
        raise ValueError("map dimensions must be positive")
    if config.clear_radius < 0.0 or 2.0 * config.clear_radius >= min(config.size_x, config.size_y):
        raise ValueError("clear radius must fit strictly inside the map")
    if min(config.ground_spacing, config.obstacle_spacing, config.crown_spacing, config.voxel_size) <= 0.0:
        raise ValueError("point spacings and voxel size must be positive")
    if not 0.0 <= config.crown_fill <= 1.0:
        raise ValueError("crown fill must be in [0,1]")
    if config.tree_count < 0 or config.building_count < 0:
        raise ValueError("object counts must be nonnegative")
    if config.scene == "forest" and config.tree_count == 0:
        raise ValueError("forest scene requires at least one tree")
    if config.scene == "forest" and config.building_count != 0:
        raise ValueError("forest scene requires building_count=0; use scene='mixed' for trees and buildings")
    if config.scene == "buildings" and config.building_count == 0:
        raise ValueError("buildings scene requires at least one building")
    if config.scene == "buildings" and config.tree_count != 0:
        raise ValueError("buildings scene requires tree_count=0; use scene='mixed' for trees and buildings")
    if config.scene == "mixed" and (config.tree_count == 0 or config.building_count == 0):
        raise ValueError("mixed scene requires at least one tree and one building")
    positive_ranges = [
        ("trunk radius", config.trunk_radius_min, config.trunk_radius_max),
        ("tree height", config.tree_height_min, config.tree_height_max),
        ("crown radius", config.crown_radius_min, config.crown_radius_max),
        ("crown height", config.crown_height_min, config.crown_height_max),
        ("building size", config.building_size_min, config.building_size_max),
        ("building height", config.building_height_min, config.building_height_max),
    ]
    for name, lower, upper in positive_ranges:
        if lower <= 0.0 or upper < lower:
            raise ValueError(f"{name} range must be positive and ordered")
    if config.tree_min_spacing <= 0.0 or config.building_gap < 0.0 or config.wall_thickness <= 0.0:
        raise ValueError("object spacing and wall thickness must be positive")
    if config.tree_building_gap < 0.0:
        raise ValueError("tree-building gap must be nonnegative")
    if config.low_tree_min_spacing < config.tree_min_spacing:
        raise ValueError("low-tree spacing must be at least the general tree spacing")
    if config.max_placement_attempts <= 0:
        raise ValueError("max placement attempts must be positive")


def _axis_samples(lower: float, upper: float, spacing: float) -> np.ndarray:
    """Sample a closed interval without extending past either endpoint."""
    count = int(math.floor((upper - lower) / spacing))
    values = lower + np.arange(count + 1, dtype=np.float64) * spacing
    if values[-1] < upper - 1e-9:
        values = np.append(values, upper)
    else:
        values[-1] = upper
    return values.astype(np.float32)


def _ground_points(config: SceneConfig) -> np.ndarray:
    """Sample a flat bounded ground plane whose midpoint is the world origin."""
    x = _axis_samples(-0.5 * config.size_x, 0.5 * config.size_x, config.ground_spacing)
    y = _axis_samples(-0.5 * config.size_y, 0.5 * config.size_y, config.ground_spacing)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.column_stack([xx.reshape(-1), yy.reshape(-1), np.zeros(xx.size, dtype=np.float32)])


def _valid_center(x: float, y: float, radius: float, config: SceneConfig) -> bool:
    """Return whether a circular footprint fits inside the map and outside the takeoff clearing."""
    boundary_pad = radius + config.voxel_size
    if abs(x) > 0.5 * config.size_x - boundary_pad:
        return False
    if abs(y) > 0.5 * config.size_y - boundary_pad:
        return False
    clear_pad = config.clear_radius + radius + math.sqrt(2.0) * config.voxel_size
    return math.hypot(x, y) >= clear_pad


def _building_grid_centers(config: SceneConfig, footprint_radius: float) -> List[Tuple[float, float]]:
    """Return cell-centred building positions on the smallest usable rectangular grid."""
    grid_capacity = max(2, config.building_count)
    for _ in range(config.building_count + 2):
        columns = int(math.ceil(math.sqrt(grid_capacity)))
        rows = int(math.ceil(grid_capacity / columns))
        x_step = config.size_x / columns
        y_step = config.size_y / rows
        x_centers = -0.5 * config.size_x + (np.arange(columns) + 0.5) * x_step
        y_centers = -0.5 * config.size_y + (np.arange(rows) + 0.5) * y_step
        candidates = [
            (float(x), float(y))
            for y in y_centers
            for x in x_centers
            if _valid_center(float(x), float(y), footprint_radius, config)
        ]
        if len(candidates) >= config.building_count:
            return candidates[: config.building_count]
        grid_capacity += 1
    raise RuntimeError("map is too small for an axis-aligned building grid outside the origin clearing")


def _axis_aligned_buildings_overlap(first: _Building, second: _Building, gap: float) -> bool:
    """Return whether two axis-aligned footprints violate their requested edge gap."""
    overlap_x = abs(first.x - second.x) < 0.5 * (first.width + second.width) + gap
    overlap_y = abs(first.y - second.y) < 0.5 * (first.depth + second.depth) + gap
    return overlap_x and overlap_y


def _distance_to_building_footprint(x: float, y: float, building: _Building) -> float:
    """Measure planar distance from one point to an axis-aligned building footprint."""
    outside_x = max(abs(x - building.x) - 0.5 * building.width, 0.0)
    outside_y = max(abs(y - building.y) - 0.5 * building.depth, 0.0)
    return math.hypot(outside_x, outside_y)


def _required_tree_spacing(first_form: str, second_form: str, config: SceneConfig) -> float:
    """Use the larger low-tree spacing whenever either member of a tree pair is low."""
    if first_form in LOW_TREE_FORMS or second_form in LOW_TREE_FORMS:
        return config.low_tree_min_spacing
    return config.tree_min_spacing


def _place_buildings(config: SceneConfig, rng: np.random.Generator) -> List[_Building]:
    """Place axis-aligned low-rise buildings at regular grid-cell centres."""
    if config.scene == "forest":
        return []
    dimensions = [
        (
            float(rng.uniform(config.building_size_min, config.building_size_max)),
            float(rng.uniform(config.building_size_min, config.building_size_max)),
            float(rng.uniform(config.building_height_min, config.building_height_max)),
        )
        for _ in range(config.building_count)
    ]
    max_radius = max(0.5 * math.hypot(width, depth) for width, depth, _height in dimensions)
    centers = _building_grid_centers(config, max_radius)
    buildings = [
        _Building(x, y, width, depth, height, 0.0) for (x, y), (width, depth, height) in zip(centers, dimensions)
    ]
    if any(
        _axis_aligned_buildings_overlap(first, second, config.building_gap)
        for index, first in enumerate(buildings)
        for second in buildings[index + 1 :]
    ):
        raise RuntimeError("building grid cells are too small for the configured sizes and gap")
    return buildings


def _sample_profile_range(
    rng: np.random.Generator, configured: Tuple[float, float], preferred: Tuple[float, float]
) -> float:
    """Sample a morphology range clipped to explicit global configuration bounds."""
    lower = max(configured[0], preferred[0])
    upper = min(configured[1], preferred[1])
    if lower > upper:
        lower, upper = configured
    return float(rng.uniform(lower, upper))


def _place_trees(config: SceneConfig, buildings: List[_Building], rng: np.random.Generator) -> List[_Tree]:
    """Place tree trunks with overlapping crowns but separated stem centers."""
    if config.scene == "buildings":
        return []
    trees: List[_Tree] = []
    for tree_index in range(config.tree_count):
        for _attempt in range(config.max_placement_attempts):
            if config.tree_form != "mixed":
                form = config.tree_form
                style = "conifer" if form == "conifer" else "broadleaf"
            else:
                style = config.tree_style
                if style == "mixed":
                    style = "conifer" if rng.random() < 0.12 else "broadleaf"
                if style == "conifer":
                    form = "conifer"
                else:
                    form = str(
                        rng.choice(
                            ["upright", "spreading", "leaning", "orchard", "willow", "shrub"],
                            p=[0.18, 0.17, 0.08, 0.24, 0.08, 0.25],
                        )
                    )

            height_range, crown_range, crown_height_range, trunk_range, lean_range = _TREE_FORM_PROFILES[form]
            total_height = _sample_profile_range(rng, (config.tree_height_min, config.tree_height_max), height_range)
            trunk_radius = _sample_profile_range(rng, (config.trunk_radius_min, config.trunk_radius_max), trunk_range)
            crown_radius_x = _sample_profile_range(rng, (config.crown_radius_min, config.crown_radius_max), crown_range)
            aspect_range = (0.86, 1.16) if form == "orchard" else (0.75, 1.32)
            crown_radius_y = float(
                np.clip(crown_radius_x * rng.uniform(*aspect_range), config.crown_radius_min, config.crown_radius_max)
            )
            if form == "conifer":
                preferred_crown_height = (0.30 * total_height, 0.42 * total_height)
            else:
                preferred_crown_height = crown_height_range
            crown_radius_z = _sample_profile_range(
                rng, (config.crown_height_min, config.crown_height_max), preferred_crown_height
            )
            crown_center_z = total_height - crown_radius_z
            trunk_height = total_height
            lean_angle = float(rng.uniform(-math.pi, math.pi))
            lean_distance = float(rng.uniform(*lean_range)) * total_height
            lean_x = lean_distance * math.cos(lean_angle)
            lean_y = lean_distance * math.sin(lean_angle)
            stem_wander = min(0.22, float(rng.uniform(0.025, 0.085)) * max(crown_radius_x, crown_radius_y))
            radius = lean_distance + 1.40 * max(crown_radius_x, crown_radius_y) + 0.35
            x_limit = 0.5 * config.size_x - radius - config.voxel_size
            y_limit = 0.5 * config.size_y - radius - config.voxel_size
            if x_limit <= 0.0 or y_limit <= 0.0:
                continue
            x = float(rng.uniform(-x_limit, x_limit))
            y = float(rng.uniform(-y_limit, y_limit))
            if not _valid_center(x, y, radius, config):
                continue
            if any(
                _distance_to_building_footprint(x, y, building) < radius + config.tree_building_gap
                for building in buildings
            ):
                continue
            if any(
                math.hypot(x - other.x, y - other.y) < _required_tree_spacing(form, other.form, config)
                for other in trees
            ):
                continue
            trees.append(
                _Tree(
                    x=x,
                    y=y,
                    trunk_radius=trunk_radius,
                    trunk_height=trunk_height,
                    crown_radius_x=crown_radius_x,
                    crown_radius_y=crown_radius_y,
                    crown_radius_z=crown_radius_z,
                    crown_center_z=crown_center_z,
                    yaw=float(rng.uniform(-math.pi, math.pi)),
                    style=style,
                    form=form,
                    lean_x=lean_x,
                    lean_y=lean_y,
                    stem_wander=stem_wander,
                    trunk_wave_phase=float(rng.uniform(-math.pi, math.pi)),
                )
            )
            break
        else:
            raise RuntimeError(
                f"could place only {len(trees)} of {config.tree_count} trees; "
                "reduce tree count/minimum spacing or clear radius"
            )
        if len(trees) <= tree_index:
            raise RuntimeError("map is too small for the configured crown-radius range")
    return trees


def _tapered_segment_points(
    start: np.ndarray, end: np.ndarray, start_radius: float, end_radius: float, spacing: float
) -> np.ndarray:
    """Sample a tapered branch shell and centerline between two 3D endpoints."""
    direction = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length <= 1e-8:
        return np.asarray(start, dtype=np.float32).reshape(1, 3)
    axis = direction / length
    reference = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    basis_u = np.cross(axis, reference)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(axis, basis_u)
    progress = _axis_samples(0.0, 1.0, spacing / length)
    centers = np.asarray(start, dtype=np.float64) + progress[:, None] * direction
    radii = start_radius + progress * (end_radius - start_radius)
    ring_count = max(5, int(math.ceil(2.0 * math.pi * start_radius / spacing)))
    angle = np.linspace(0.0, 2.0 * math.pi, ring_count, endpoint=False)
    radial = np.cos(angle)[:, None] * basis_u + np.sin(angle)[:, None] * basis_v
    shell = centers[:, None, :] + radii[:, None, None] * radial[None, :, :]
    return np.concatenate([shell.reshape(-1, 3), centers]).astype(np.float32)


def _normalize(vector: np.ndarray) -> np.ndarray:
    """Return a unit float64 vector and reject degenerate geometry."""
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-10:
        raise ValueError("cannot normalize a zero-length direction")
    return vector / norm


def _stem_centerline(tree: _Tree, rng: np.random.Generator) -> np.ndarray:
    """Build a leaning quadratic main stem with smooth low-amplitude lateral wander."""
    section_count = max(8, int(math.ceil(tree.trunk_height / 0.9)))
    progress = np.linspace(0.0, 1.0, section_count + 1)
    crown_scale = min(tree.crown_radius_x, tree.crown_radius_y)
    mid_angle = tree.yaw + float(rng.uniform(-0.45, 0.45))
    top_angle = mid_angle + float(rng.uniform(-0.65, 0.65))
    mid_offset = float(rng.uniform(0.02, 0.07)) * crown_scale
    top_offset = float(rng.uniform(0.04, 0.12)) * crown_scale
    bottom = np.array([tree.x, tree.y, 0.0], dtype=np.float64)
    control = np.array(
        [
            tree.x + 0.42 * tree.lean_x + mid_offset * math.cos(mid_angle),
            tree.y + 0.42 * tree.lean_y + mid_offset * math.sin(mid_angle),
            0.52 * tree.trunk_height,
        ]
    )
    top = np.array(
        [
            tree.x + tree.lean_x + top_offset * math.cos(top_angle),
            tree.y + tree.lean_y + top_offset * math.sin(top_angle),
            tree.trunk_height,
        ]
    )
    one_minus = 1.0 - progress
    centerline = (
        one_minus[:, None] ** 2 * bottom
        + 2.0 * one_minus[:, None] * progress[:, None] * control
        + progress[:, None] ** 2 * top
    )
    wander_angle = tree.yaw + 0.5 * math.pi
    wander = tree.stem_wander * np.sin(math.pi * progress) * np.sin(2.0 * math.pi * progress + tree.trunk_wave_phase)
    centerline[:, 0] += wander * math.cos(wander_angle)
    centerline[:, 1] += wander * math.sin(wander_angle)
    return centerline.astype(np.float32)


def _curved_centerline(
    start: np.ndarray,
    direction: np.ndarray,
    length: float,
    section_count: int,
    gnarliness: float,
    upward_pull: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Grow a piecewise-linear branch with small random bends and upward tropism."""
    current = np.asarray(start, dtype=np.float64)
    heading = _normalize(direction)
    points = [current.copy()]
    step_length = length / section_count
    up = np.array([0.0, 0.0, 1.0])
    for _section in range(section_count):
        noise = rng.normal(size=3)
        noise -= float(noise @ heading) * heading
        noise_norm = float(np.linalg.norm(noise))
        if noise_norm > 1e-10:
            noise /= noise_norm
        upward = up - float(up @ heading) * heading
        heading = _normalize(heading + gnarliness * noise + upward_pull * upward)
        current = current + step_length * heading
        points.append(current.copy())
    return np.asarray(points, dtype=np.float32)


def _sample_centerline(centerline: np.ndarray, fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate an origin and tangent at one fractional centerline position."""
    fraction = float(np.clip(fraction, 0.0, 1.0))
    scaled = fraction * (centerline.shape[0] - 1)
    lower = min(int(math.floor(scaled)), centerline.shape[0] - 2)
    alpha = scaled - lower
    origin = (1.0 - alpha) * centerline[lower] + alpha * centerline[lower + 1]
    tangent = _normalize(centerline[lower + 1] - centerline[lower])
    return origin.astype(np.float32), tangent


def _polyline_points(centerline: np.ndarray, start_radius: float, end_radius: float, spacing: float) -> np.ndarray:
    """Sample tapered occupied surfaces along every centerline section."""
    progress = np.linspace(0.0, 1.0, centerline.shape[0])
    radii = start_radius + progress * (end_radius - start_radius)
    return _polyline_points_with_radii(centerline, radii, spacing)


def _polyline_points_with_radii(centerline: np.ndarray, radii: np.ndarray, spacing: float) -> np.ndarray:
    """Sample occupied surfaces using one explicitly controlled radius per section node."""
    if radii.shape != (centerline.shape[0],):
        raise ValueError("radii must provide one value per centerline point")
    return np.concatenate(
        [
            _tapered_segment_points(
                centerline[index], centerline[index + 1], float(radii[index]), float(radii[index + 1]), spacing
            )
            for index in range(centerline.shape[0] - 1)
        ]
    )


def _branch_direction(
    parent_direction: np.ndarray, angle: float, radial_angle: float, minimum_vertical: float
) -> np.ndarray:
    """Rotate away from a parent direction using one azimuth around its axis."""
    parent = _normalize(parent_direction)
    reference = np.array([0.0, 0.0, 1.0]) if abs(parent[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    basis_u = _normalize(np.cross(parent, reference))
    basis_v = np.cross(parent, basis_u)
    radial = math.cos(radial_angle) * basis_u + math.sin(radial_angle) * basis_v
    direction = math.cos(angle) * parent + math.sin(angle) * radial
    direction[2] = max(direction[2], minimum_vertical)
    return _normalize(direction)


def _trunk_points(tree: _Tree, centerline: np.ndarray, spacing: float) -> np.ndarray:
    """Sample a curved main stem with butt flare, taper, and smooth radius undulation."""
    progress = np.linspace(0.0, 1.0, centerline.shape[0])
    taper = 1.0 - 0.84 * progress**0.82
    base_flare = 1.0 + 0.22 * np.exp(-progress / 0.075)
    undulation = (
        1.0
        + 0.075 * np.sin(3.0 * math.pi * progress + tree.trunk_wave_phase)
        + 0.030 * np.sin(7.0 * math.pi * progress - 0.7 * tree.trunk_wave_phase)
    )
    radii = np.maximum(0.035, tree.trunk_radius * taper * base_flare * undulation)
    return _polyline_points_with_radii(centerline, radii, spacing)


def _foliage_cluster(
    center: np.ndarray, radii: Tuple[float, float, float], spacing: float, fill: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample one porous ellipsoidal foliage cluster."""
    radius_x, radius_y, radius_z = radii
    x = _axis_samples(-radius_x, radius_x, spacing)
    y = _axis_samples(-radius_y, radius_y, spacing)
    z = _axis_samples(-radius_z, radius_z, spacing)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    radius_sq = (xx / radius_x) ** 2 + (yy / radius_y) ** 2 + (zz / radius_z) ** 2
    inside = radius_sq <= 1.0
    local = np.column_stack([xx[inside], yy[inside], zz[inside]]).astype(np.float32)
    normalized_radius = np.sqrt(radius_sq[inside])
    shell_width = min(0.45, 2.0 * spacing / min(radii))
    probability = np.full(local.shape[0], fill, dtype=np.float32)
    probability[normalized_radius >= 1.0 - shell_width] = max(0.35, fill)
    local = local[rng.random(local.shape[0]) < probability]
    if local.shape[0] == 0:
        local = np.zeros((1, 3), dtype=np.float32)
    jitter = rng.uniform(-0.15 * spacing, 0.15 * spacing, size=local.shape).astype(np.float32)
    local += jitter
    local += np.asarray(center, dtype=np.float32)
    return local


def _leaf_radii(tree: _Tree, scale: float) -> Tuple[float, float, float]:
    """Return compact terminal foliage radii instead of one solid crown blob."""
    return (
        float(np.clip(scale * tree.crown_radius_x, 0.20, 0.48)),
        float(np.clip(scale * tree.crown_radius_y, 0.20, 0.48)),
        float(np.clip(0.85 * scale * tree.crown_radius_z, 0.18, 0.40)),
    )


def _broadleaf_crown_points(
    tree: _Tree, trunk: np.ndarray, branch_spacing: float, foliage_spacing: float, fill: float, rng: np.random.Generator
) -> np.ndarray:
    """Grow stratified primary limbs, recursive side branches, and terminal leaf clusters."""
    crown_bottom = tree.crown_center_z - tree.crown_radius_z
    crown_height = 2.0 * tree.crown_radius_z
    foliage_fill = max(fill, 0.18) if tree.form == "orchard" else fill
    cluster_spacing = min(foliage_spacing, 0.16) if tree.form == "orchard" else foliage_spacing
    if tree.form == "spreading":
        primary_count = int(rng.integers(10, 14))
        crown_start, crown_span = 0.04, 0.58
        branch_angle_range = (64.0, 80.0)
        target_reach_range = (0.78, 0.96)
        minimum_vertical = 0.05
        secondary_count = 4
        child_length_range = (0.30, 0.43)
        leaf_scale_range = (0.13, 0.19)
    elif tree.form == "leaning":
        primary_count = int(rng.integers(8, 11))
        crown_start, crown_span = 0.06, 0.65
        branch_angle_range = (55.0, 70.0)
        target_reach_range = (0.74, 0.92)
        minimum_vertical = 0.09
        secondary_count = 3
        child_length_range = (0.28, 0.41)
        leaf_scale_range = (0.12, 0.17)
    elif tree.form == "orchard":
        primary_count = int(rng.integers(9, 13))
        crown_start, crown_span = 0.02, 0.67
        branch_angle_range = (58.0, 76.0)
        target_reach_range = (0.76, 0.95)
        minimum_vertical = 0.07
        secondary_count = 4
        child_length_range = (0.30, 0.44)
        leaf_scale_range = (0.14, 0.21)
    else:
        primary_count = int(rng.integers(7, 10))
        crown_start, crown_span = 0.08, 0.68
        branch_angle_range = (48.0, 62.0)
        target_reach_range = (0.70, 0.88)
        minimum_vertical = 0.16
        secondary_count = 3
        child_length_range = (0.27, 0.39)
        leaf_scale_range = (0.11, 0.16)
    radial_slots = rng.permutation(primary_count)
    radial_offset = tree.yaw + float(rng.uniform(-math.pi, math.pi))
    parts = []
    for primary_index in range(primary_count):
        height_slot = (primary_index + float(rng.uniform(0.18, 0.82))) / primary_count
        crown_progress = crown_start + crown_span * height_slot
        origin_z = crown_bottom + crown_progress * crown_height
        origin, trunk_direction = _sample_centerline(trunk, origin_z / tree.trunk_height)
        radial_angle = (
            radial_offset
            + 2.0 * math.pi * (radial_slots[primary_index] + float(rng.uniform(-0.32, 0.32))) / primary_count
        )
        envelope = math.sqrt(max(0.04, 1.0 - ((origin_z - tree.crown_center_z) / tree.crown_radius_z) ** 2))
        ellipse_radius = 1.0 / math.sqrt(
            (math.cos(radial_angle) / tree.crown_radius_x) ** 2 + (math.sin(radial_angle) / tree.crown_radius_y) ** 2
        )
        branch_angle = math.radians(float(rng.uniform(*branch_angle_range)))
        direction = _branch_direction(trunk_direction, branch_angle, radial_angle, minimum_vertical)
        horizontal = max(0.25, float(np.linalg.norm(direction[:2])))
        target_reach = ellipse_radius * envelope * float(rng.uniform(*target_reach_range))
        length = target_reach / horizontal
        primary = _curved_centerline(
            origin, direction, length, section_count=4, gnarliness=0.055, upward_pull=0.025, rng=rng
        )
        trunk_radius = tree.trunk_radius * (1.0 - 0.80 * origin_z / tree.trunk_height)
        base_radius = max(0.045, trunk_radius * float(rng.uniform(0.30, 0.40)))
        parts.append(_polyline_points(primary, base_radius, max(0.024, 0.38 * base_radius), branch_spacing))

        secondary_slots = rng.permutation(secondary_count)
        for secondary_index in range(secondary_count):
            fraction = 0.44 + (secondary_index + float(rng.uniform(0.20, 0.80))) * (0.48 / secondary_count)
            child_origin, parent_direction = _sample_centerline(primary, fraction)
            child_angle = math.radians(float(rng.uniform(38.0, 56.0)))
            child_azimuth = (
                2.0 * math.pi * (secondary_slots[secondary_index] + float(rng.uniform(-0.35, 0.35))) / secondary_count
            )
            child_direction = _branch_direction(parent_direction, child_angle, child_azimuth, minimum_vertical=0.10)
            child_length = length * float(rng.uniform(*child_length_range))
            secondary = _curved_centerline(
                child_origin,
                child_direction,
                child_length,
                section_count=3,
                gnarliness=0.075,
                upward_pull=0.045,
                rng=rng,
            )
            child_radius = max(0.028, base_radius * float(rng.uniform(0.48, 0.58)))
            parts.append(_polyline_points(secondary, child_radius, max(0.018, 0.35 * child_radius), branch_spacing))
            parts.append(
                _foliage_cluster(
                    secondary[-1],
                    _leaf_radii(tree, float(rng.uniform(*leaf_scale_range))),
                    cluster_spacing,
                    foliage_fill,
                    rng,
                )
            )
        parts.append(
            _foliage_cluster(
                primary[-1],
                _leaf_radii(tree, float(rng.uniform(leaf_scale_range[0] + 0.02, leaf_scale_range[1] + 0.02))),
                cluster_spacing,
                foliage_fill,
                rng,
            )
        )

    top_origin, top_direction = _sample_centerline(trunk, 0.82)
    top = _curved_centerline(
        top_origin,
        top_direction,
        0.18 * tree.trunk_height,
        section_count=3,
        gnarliness=0.035,
        upward_pull=0.06,
        rng=rng,
    )
    parts.append(_foliage_cluster(top[-1], _leaf_radii(tree, 0.16), cluster_spacing, foliage_fill, rng))
    return np.concatenate(parts).astype(np.float32, copy=False)


def _shrub_crown_points(
    tree: _Tree, branch_spacing: float, foliage_spacing: float, fill: float, rng: np.random.Generator
) -> np.ndarray:
    """Grow several near-ground stems and compact forked foliage for a shrub silhouette."""
    stem_count = int(rng.integers(5, 9))
    radial_slots = rng.permutation(stem_count)
    parts = []
    for stem_index in range(stem_count):
        radial_angle = (
            tree.yaw + 2.0 * math.pi * (radial_slots[stem_index] + float(rng.uniform(-0.30, 0.30))) / stem_count
        )
        start_radius = float(rng.uniform(0.00, 0.12))
        start = np.array(
            [tree.x + start_radius * math.cos(radial_angle), tree.y + start_radius * math.sin(radial_angle), 0.0]
        )
        tilt = math.radians(float(rng.uniform(24.0, 48.0)))
        direction = _normalize(
            [math.sin(tilt) * math.cos(radial_angle), math.sin(tilt) * math.sin(radial_angle), math.cos(tilt)]
        )
        stem_length = tree.trunk_height * float(rng.uniform(0.72, 0.98))
        stem = _curved_centerline(
            start, direction, stem_length, section_count=4, gnarliness=0.075, upward_pull=0.035, rng=rng
        )
        stem_radius = max(0.025, tree.trunk_radius * float(rng.uniform(0.45, 0.78)))
        parts.append(_polyline_points(stem, stem_radius, max(0.014, 0.30 * stem_radius), branch_spacing))

        for side in (-1.0, 1.0):
            fraction = float(rng.uniform(0.45, 0.76))
            child_origin, parent_direction = _sample_centerline(stem, fraction)
            child_direction = _branch_direction(
                parent_direction,
                math.radians(float(rng.uniform(38.0, 60.0))),
                radial_angle + side * float(rng.uniform(0.55, 1.00)),
                minimum_vertical=0.15,
            )
            twig = _curved_centerline(
                child_origin,
                child_direction,
                tree.trunk_height * float(rng.uniform(0.20, 0.34)),
                section_count=2,
                gnarliness=0.085,
                upward_pull=0.045,
                rng=rng,
            )
            twig_radius = max(0.016, 0.42 * stem_radius)
            parts.append(_polyline_points(twig, twig_radius, max(0.010, 0.30 * twig_radius), branch_spacing))
            parts.append(
                _foliage_cluster(
                    twig[-1],
                    _leaf_radii(tree, float(rng.uniform(0.16, 0.24))),
                    min(foliage_spacing, 0.14),
                    max(fill, 0.28),
                    rng,
                )
            )
        parts.append(
            _foliage_cluster(
                stem[-1],
                _leaf_radii(tree, float(rng.uniform(0.18, 0.27))),
                min(foliage_spacing, 0.14),
                max(fill, 0.28),
                rng,
            )
        )
    return np.concatenate(parts).astype(np.float32, copy=False)


def _willow_crown_points(
    tree: _Tree, trunk: np.ndarray, branch_spacing: float, foliage_spacing: float, fill: float, rng: np.random.Generator
) -> np.ndarray:
    """Grow a wide upper crown followed by long gravity-biased hanging branchlets."""
    crown_bottom = max(0.28 * tree.trunk_height, tree.crown_center_z - tree.crown_radius_z)
    primary_count = int(rng.integers(12, 17))
    radial_slots = rng.permutation(primary_count)
    parts = []
    for primary_index in range(primary_count):
        height_slot = (primary_index + float(rng.uniform(0.15, 0.85))) / primary_count
        origin_z = crown_bottom + 0.88 * height_slot * (tree.trunk_height - crown_bottom)
        origin, trunk_direction = _sample_centerline(trunk, origin_z / tree.trunk_height)
        radial_angle = (
            tree.yaw + 2.0 * math.pi * (radial_slots[primary_index] + float(rng.uniform(-0.28, 0.28))) / primary_count
        )
        ellipse_radius = 1.0 / math.sqrt(
            (math.cos(radial_angle) / tree.crown_radius_x) ** 2 + (math.sin(radial_angle) / tree.crown_radius_y) ** 2
        )
        direction = _branch_direction(
            trunk_direction, math.radians(float(rng.uniform(58.0, 76.0))), radial_angle, minimum_vertical=0.08
        )
        horizontal = max(0.25, float(np.linalg.norm(direction[:2])))
        primary_length = ellipse_radius * float(rng.uniform(0.68, 0.90)) / horizontal
        primary = _curved_centerline(
            origin, direction, primary_length, section_count=4, gnarliness=0.045, upward_pull=0.015, rng=rng
        )
        local_trunk_radius = tree.trunk_radius * (1.0 - 0.78 * origin_z / tree.trunk_height)
        base_radius = max(0.055, local_trunk_radius * float(rng.uniform(0.28, 0.38)))
        parts.append(_polyline_points(primary, base_radius, max(0.025, 0.34 * base_radius), branch_spacing))

        droop_count = 3
        for droop_index in range(droop_count):
            fraction = 0.45 + (droop_index + float(rng.uniform(0.15, 0.85))) * (0.48 / droop_count)
            child_origin, _parent_direction = _sample_centerline(primary, fraction)
            child_angle = radial_angle + float(rng.uniform(-0.38, 0.38))
            downward = float(rng.uniform(0.38, 0.82))
            child_direction = _normalize([math.cos(child_angle), math.sin(child_angle), -downward])
            requested_length = tree.crown_radius_z * float(rng.uniform(0.90, 1.55))
            floor_limited = max(0.35, 0.90 * (float(child_origin[2]) - 0.35) / abs(child_direction[2]))
            child_length = min(requested_length, floor_limited)
            droop = _curved_centerline(
                child_origin,
                child_direction,
                child_length,
                section_count=4,
                gnarliness=0.055,
                upward_pull=-0.035,
                rng=rng,
            )
            droop_radius = max(0.020, base_radius * float(rng.uniform(0.32, 0.45)))
            parts.append(_polyline_points(droop, droop_radius, max(0.012, 0.28 * droop_radius), branch_spacing))
            for leaf_fraction in (0.58, 0.82, 1.0):
                leaf_origin, _leaf_direction = _sample_centerline(droop, leaf_fraction)
                parts.append(
                    _foliage_cluster(
                        leaf_origin,
                        _leaf_radii(tree, float(rng.uniform(0.065, 0.105))),
                        min(foliage_spacing, 0.18),
                        max(fill, 0.16),
                        rng,
                    )
                )
    return np.concatenate(parts).astype(np.float32, copy=False)


def _conifer_crown_points(
    tree: _Tree, trunk: np.ndarray, branch_spacing: float, foliage_spacing: float, fill: float, rng: np.random.Generator
) -> np.ndarray:
    """Grow a monopodial leader with height-stratified whorls and short side twigs."""
    crown_bottom = tree.crown_center_z - tree.crown_radius_z
    crown_height = tree.trunk_height - crown_bottom
    level_count = max(8, int(math.ceil(crown_height / 0.72)))
    parts = []
    for level in range(level_count):
        progress = (level + float(rng.uniform(0.18, 0.62))) / level_count
        origin_z = crown_bottom + progress * crown_height
        origin, _trunk_direction = _sample_centerline(trunk, origin_z / tree.trunk_height)
        branch_count = 5 if progress < 0.72 else 4
        phase = tree.yaw + level * math.radians(137.5)
        radial_slots = rng.permutation(branch_count)
        for branch_index in range(branch_count):
            radial_angle = (
                phase + 2.0 * math.pi * (radial_slots[branch_index] + float(rng.uniform(-0.22, 0.22))) / branch_count
            )
            ellipse_radius = 1.0 / math.sqrt(
                (math.cos(radial_angle) / tree.crown_radius_x) ** 2
                + (math.sin(radial_angle) / tree.crown_radius_y) ** 2
            )
            horizontal_reach = max(0.18, ellipse_radius * (1.0 - progress) ** 0.62 * float(rng.uniform(0.72, 0.90)))
            vertical = float(rng.uniform(-0.18, -0.04) + 0.28 * progress)
            direction = _normalize([math.cos(radial_angle), math.sin(radial_angle), vertical])
            length = horizontal_reach / max(0.25, float(np.linalg.norm(direction[:2])))
            primary = _curved_centerline(
                origin, direction, length, section_count=3, gnarliness=0.035, upward_pull=0.065, rng=rng
            )
            trunk_radius = tree.trunk_radius * (1.0 - 0.80 * origin_z / tree.trunk_height)
            base_radius = max(0.038, trunk_radius * float(rng.uniform(0.22, 0.30)))
            parts.append(_polyline_points(primary, base_radius, max(0.018, 0.32 * base_radius), branch_spacing))

            for side in (-1.0, 1.0):
                fraction = float(rng.uniform(0.50, 0.82))
                child_origin, _parent_direction = _sample_centerline(primary, fraction)
                child_angle = radial_angle + side * float(rng.uniform(0.45, 0.75))
                child_direction = _normalize(
                    [math.cos(child_angle), math.sin(child_angle), float(rng.uniform(0.18, 0.42))]
                )
                child_length = length * float(rng.uniform(0.20, 0.31))
                twig = _curved_centerline(
                    child_origin,
                    child_direction,
                    child_length,
                    section_count=2,
                    gnarliness=0.045,
                    upward_pull=0.04,
                    rng=rng,
                )
                twig_radius = max(0.022, 0.48 * base_radius)
                parts.append(_polyline_points(twig, twig_radius, max(0.014, 0.35 * twig_radius), branch_spacing))
                parts.append(
                    _foliage_cluster(
                        twig[-1], _leaf_radii(tree, float(rng.uniform(0.08, 0.12))), foliage_spacing, fill, rng
                    )
                )
            parts.append(
                _foliage_cluster(
                    primary[-1], _leaf_radii(tree, float(rng.uniform(0.09, 0.13))), foliage_spacing, fill, rng
                )
            )
    return np.concatenate(parts).astype(np.float32, copy=False)


def _tree_points(
    tree: _Tree, branch_spacing: float, foliage_spacing: float, fill: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate one bounded tree from a curved stem and a style-specific branch hierarchy."""
    trunk = _stem_centerline(tree, rng)
    parts = [_trunk_points(tree, trunk, branch_spacing)]
    if tree.form == "shrub":
        parts.append(_shrub_crown_points(tree, branch_spacing, foliage_spacing, fill, rng))
    elif tree.form == "willow":
        parts.append(_willow_crown_points(tree, trunk, branch_spacing, foliage_spacing, fill, rng))
    elif tree.style == "broadleaf":
        parts.append(_broadleaf_crown_points(tree, trunk, branch_spacing, foliage_spacing, fill, rng))
    else:
        parts.append(_conifer_crown_points(tree, trunk, branch_spacing, foliage_spacing, fill, rng))
    points = np.concatenate(parts)
    horizontal_radius = np.linalg.norm(points[:, :2] - np.array([tree.x, tree.y]), axis=1)
    upper_z = tree.trunk_height + max(0.45, 0.18 * tree.crown_radius_z)
    bounded = (horizontal_radius <= tree.footprint_radius) & (points[:, 2] >= 0.0) & (points[:, 2] <= upper_z)
    return points[bounded].astype(np.float32, copy=False)


def _wall_face(
    constant_axis: int, constant_value: float, first_values: np.ndarray, second_values: np.ndarray
) -> np.ndarray:
    """Create one axis-aligned rectangular face in local building coordinates."""
    first, second = np.meshgrid(first_values, second_values, indexing="ij")
    points = np.empty((first.size, 3), dtype=np.float32)
    free_axes = [axis for axis in range(3) if axis != constant_axis]
    points[:, constant_axis] = constant_value
    points[:, free_axes[0]] = first.reshape(-1)
    points[:, free_axes[1]] = second.reshape(-1)
    return points


def _building_points(building: _Building, spacing: float, thickness: float) -> np.ndarray:
    """Sample thick walls and a roof before rotating them into world coordinates."""
    half_width = 0.5 * building.width
    half_depth = 0.5 * building.depth
    x = _axis_samples(-half_width, half_width, spacing)
    y = _axis_samples(-half_depth, half_depth, spacing)
    z = _axis_samples(0.0, building.height, spacing)
    layers = _axis_samples(0.0, min(thickness, half_width, half_depth), spacing)
    parts = []
    for inset in layers:
        parts.append(_wall_face(0, -half_width + inset, y, z))
        parts.append(_wall_face(0, half_width - inset, y, z))
        parts.append(_wall_face(1, -half_depth + inset, x, z))
        parts.append(_wall_face(1, half_depth - inset, x, z))
        parts.append(_wall_face(2, building.height - inset, x, y))
    local = np.concatenate(parts)
    cosine = math.cos(building.yaw)
    sine = math.sin(building.yaw)
    world = np.empty_like(local)
    world[:, 0] = cosine * local[:, 0] - sine * local[:, 1] + building.x
    world[:, 1] = sine * local[:, 0] + cosine * local[:, 1] + building.y
    world[:, 2] = local[:, 2]
    return world


def _voxelize(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Quantize, deduplicate, and lexicographically order occupied XYZ voxels."""
    indices = np.rint(np.asarray(points, dtype=np.float64) / voxel_size).astype(np.int64)
    unique = np.unique(indices, axis=0)
    return np.ascontiguousarray(unique.astype(np.float64) * voxel_size, dtype=np.float32)


def _minimum_tree_distance(trees: List[_Tree], forms: Tuple[str, ...] = ()) -> Union[float, None]:
    """Measure the minimum stem-centre distance for all trees or selected forms."""
    selected = [tree for tree in trees if not forms or tree.form in forms]
    minimum = math.inf
    for index, first in enumerate(selected):
        for second in selected[index + 1 :]:
            minimum = min(minimum, math.hypot(first.x - second.x, first.y - second.y))
    return None if math.isinf(minimum) else minimum


def _minimum_high_low_tree_distance(trees: List[_Tree]) -> Union[float, None]:
    """Measure the minimum centre distance between one low and one taller tree form."""
    minimum = math.inf
    for index, first in enumerate(trees):
        first_is_low = first.form in LOW_TREE_FORMS
        for second in trees[index + 1 :]:
            if first_is_low == (second.form in LOW_TREE_FORMS):
                continue
            minimum = min(minimum, math.hypot(first.x - second.x, first.y - second.y))
    return None if math.isinf(minimum) else minimum


def _minimum_tree_building_clearance(trees: List[_Tree], buildings: List[_Building]) -> Union[float, None]:
    """Measure the minimum conservative crown-envelope clearance from any building wall."""
    if not trees or not buildings:
        return None
    return min(
        _distance_to_building_footprint(tree.x, tree.y, building) - tree.footprint_radius
        for tree in trees
        for building in buildings
    )


def generate_scene(config: SceneConfig) -> SceneResult:
    """Generate one deterministic point cloud without writing files."""
    _validate_config(config)
    rng = np.random.default_rng(config.seed)
    buildings = _place_buildings(config, rng)
    trees = _place_trees(config, buildings, rng)

    ground = _ground_points(config)
    tree_parts = [
        _tree_points(tree, config.obstacle_spacing, config.crown_spacing, config.crown_fill, rng) for tree in trees
    ]
    building_parts = [
        _building_points(building, config.obstacle_spacing, config.wall_thickness) for building in buildings
    ]
    trees_raw = sum(part.shape[0] for part in tree_parts)
    buildings_raw = sum(part.shape[0] for part in building_parts)
    points = _voxelize(np.concatenate([ground, *tree_parts, *building_parts]), config.voxel_size)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    tree_statistics: Dict[str, object] = {}
    if trees:
        tree_statistics = {
            "height_m": [min(tree.trunk_height for tree in trees), max(tree.trunk_height for tree in trees)],
            "lean_m": [
                min(math.hypot(tree.lean_x, tree.lean_y) for tree in trees),
                max(math.hypot(tree.lean_x, tree.lean_y) for tree in trees),
            ],
            "crown_diameter_m": [
                min(2.0 * max(tree.crown_radius_x, tree.crown_radius_y) for tree in trees),
                max(2.0 * max(tree.crown_radius_x, tree.crown_radius_y) for tree in trees),
            ],
            "crown_height_m": [
                min(2.0 * tree.crown_radius_z for tree in trees),
                max(2.0 * tree.crown_radius_z for tree in trees),
            ],
            "minimum_stem_spacing_m": _minimum_tree_distance(trees),
            "minimum_low_tree_spacing_m": _minimum_tree_distance(trees, LOW_TREE_FORMS),
            "minimum_high_low_tree_spacing_m": _minimum_high_low_tree_distance(trees),
            "minimum_building_clearance_m": _minimum_tree_building_clearance(trees, buildings),
        }
    metadata: Dict[str, object] = {
        "format": "static_xyz_occupancy",
        "units": "metres",
        "frame": "world",
        "scene": config.scene,
        "seed": config.seed,
        "point_count": int(points.shape[0]),
        "bounds": {"min": [float(value) for value in lower], "max": [float(value) for value in upper]},
        "objects": {"trees": len(trees), "buildings": len(buildings)},
        "tree_styles": {style: sum(tree.style == style for tree in trees) for style in TREE_STYLES if style != "mixed"},
        "tree_forms": {form: sum(tree.form == form for tree in trees) for form in TREE_FORMS},
        "tree_statistics": tree_statistics,
        "building_placements": [
            {
                "center_m": [building.x, building.y],
                "size_m": [building.width, building.depth, building.height],
                "yaw_rad": building.yaw,
            }
            for building in buildings
        ],
        "components": {
            "ground_raw": int(ground.shape[0]),
            "trees_raw": int(trees_raw),
            "buildings_raw": int(buildings_raw),
        },
        "config": asdict(config),
    }
    return SceneResult(points=points, metadata=metadata)


def _preview_sample(points: np.ndarray, seed: int, limit: int = 100_000) -> np.ndarray:
    """Retain obstacles preferentially while bounding preview rendering cost."""
    rng = np.random.default_rng(seed)
    obstacle = points[points[:, 2] > 0.05]
    ground = points[points[:, 2] <= 0.05]
    obstacle_limit = min(obstacle.shape[0], int(0.8 * limit))
    ground_limit = min(ground.shape[0], limit - obstacle_limit)
    obstacle_index = rng.choice(obstacle.shape[0], obstacle_limit, replace=False)
    ground_index = rng.choice(ground.shape[0], ground_limit, replace=False)
    return np.concatenate([obstacle[obstacle_index], ground[ground_index]])


def write_preview(path: PathLike, points: np.ndarray, seed: int) -> None:
    """Write bounded-cost map views plus a local side view for checking object shape."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d

    preview = _preview_sample(points, seed)
    figure = plt.figure(figsize=(18, 6), constrained_layout=True)
    top = figure.add_subplot(1, 3, 1)
    top.scatter(preview[:, 0], preview[:, 1], c=preview[:, 2], cmap="terrain", s=0.25, linewidths=0)
    top.set_aspect("equal")
    top.set_title("Top view (colour = height)")
    top.set_xlabel("x [m]")
    top.set_ylabel("y [m]")

    oblique = figure.add_subplot(1, 3, 2, projection=mplot3d.Axes3D.name)
    oblique.scatter(
        preview[:, 0],
        preview[:, 1],
        preview[:, 2],
        c=preview[:, 2],
        cmap="terrain",
        s=0.20,
        linewidths=0,
        depthshade=False,
    )
    oblique.view_init(elev=26, azim=-55)
    oblique.set_title("Oblique point-cloud view")
    oblique.set_xlabel("x [m]")
    oblique.set_ylabel("y [m]")
    oblique.set_zlabel("z [m]")

    local = figure.add_subplot(1, 3, 3)
    low_obstacles = points[(points[:, 2] >= 0.25) & (points[:, 2] <= 1.25)]
    if low_obstacles.shape[0] > 0:
        anchor_index = np.argmin(np.linalg.norm(low_obstacles[:, :2], axis=1))
        anchor = low_obstacles[anchor_index, :2]
        local_radius = 6.5
        local_mask = np.linalg.norm(points[:, :2] - anchor, axis=1) <= local_radius
        local_points = points[local_mask & (points[:, 2] > 0.05)]
        if local_points.shape[0] > 30_000:
            local_rng = np.random.default_rng(seed + 1)
            local_points = local_points[local_rng.choice(local_points.shape[0], 30_000, replace=False)]
        local.scatter(
            local_points[:, 0] - anchor[0],
            local_points[:, 2],
            c=local_points[:, 1] - anchor[1],
            cmap="coolwarm",
            s=1.0,
            linewidths=0,
        )
        local.set_xlim(-local_radius, local_radius)
        local.set_ylim(bottom=0.0)
    local.set_title("Nearest object side view")
    local.set_xlabel("local x [m]")
    local.set_ylabel("z [m]")
    figure.savefig(Path(path).expanduser(), dpi=180)
    plt.close(figure)


def save_scene(
    result: SceneResult, output_base: PathLike, preview: bool = False, force: bool = False
) -> Dict[str, Path]:
    """Write an exact PCD/NPZ pair, JSON provenance, and optional PNG preview."""
    base = Path(output_base).expanduser()
    if base.suffix:
        raise ValueError("output must be a base path without an extension")
    paths = {"pcd": base.with_suffix(".pcd"), "npz": base.with_suffix(".npz"), "json": base.with_suffix(".json")}
    if preview:
        paths["png"] = base.with_suffix(".png")
    existing = [path for path in paths.values() if path.exists()]
    if existing and not force:
        raise FileExistsError(f"{existing[0]} already exists; pass --force to overwrite generated files")
    base.parent.mkdir(parents=True, exist_ok=True)
    write_pcd_binary(paths["pcd"], result.points)
    np.savez_compressed(paths["npz"], points=result.points)
    paths["json"].write_text(json.dumps(result.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if preview:
        write_preview(paths["png"], result.points, int(result.metadata["seed"]))
    return paths


def format_scene_presets() -> str:
    """Render the complete built-in scene defaults for command-line discovery."""
    lines = ["Built-in scene presets:"]
    for name in SCENE_TYPES:
        preset = SCENE_PRESETS[name]
        lines.append(f"  {name:<9} {preset.description}")
    lines.append("Explicit command-line values override the selected preset.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the stable command-line interface without parsing process arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=(
            f"{format_scene_presets()}\n\n"
            "Examples:\n"
            "  python3 pcl_generate.py --list-presets\n"
            "  python3 pcl_generate.py --scene mixed --print-config\n"
            "  python3 pcl_generate.py --scene forest --output output/maps/forest_seed1 --preview"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    preset_group = parser.add_argument_group("scene preset and reproducibility")
    preset_group.add_argument(
        "--scene",
        choices=SCENE_TYPES,
        default="forest",
        help="select a complete forest, building, or mixed preset (default: forest)",
    )
    preset_group.add_argument("--seed", type=int, help="random seed (preset default: 1)")
    preset_group.add_argument(
        "--size",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="map width and depth in metres (preset defaults: forest/buildings 150 150; mixed 200 200)",
    )
    preset_group.add_argument(
        "--clear-radius", type=float, help="obstacle-free radius around the world origin in metres (preset default: 6)"
    )
    preset_group.add_argument(
        "--list-presets", action="store_true", help="print the built-in scene presets and exit without generating files"
    )
    preset_group.add_argument(
        "--print-config",
        action="store_true",
        help="print the fully resolved and validated configuration, then exit without generating files",
    )

    tree_group = parser.add_argument_group("tree geometry and placement")
    tree_group.add_argument(
        "--tree-style",
        choices=TREE_STYLES,
        help="tree family; mixed samples broadleaf and conifer trees (preset default: mixed)",
    )
    tree_group.add_argument(
        "--tree-form",
        choices=TREE_FORM_OPTIONS,
        help="force one form, or mixed to sample all forms allowed by --tree-style (preset default: mixed)",
    )
    tree_group.add_argument(
        "--tree-count", type=int, help="number of trees (preset defaults: forest 400; buildings 0; mixed 1800)"
    )
    tree_group.add_argument(
        "--tree-min-spacing",
        type=float,
        help="minimum spacing between any two stem centres in metres (preset default: 2)",
    )
    tree_group.add_argument(
        "--low-tree-min-spacing",
        type=float,
        help="minimum spacing for any pair containing a shrub, orchard, or spreading tree (preset default: 4)",
    )
    tree_group.add_argument(
        "--tree-height",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="allowed total tree-height range in metres (preset default: 2 18)",
    )
    tree_group.add_argument(
        "--trunk-radius",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="allowed trunk-radius range in metres (preset default: 0.08 0.70)",
    )
    tree_group.add_argument(
        "--crown-radius",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="allowed horizontal crown-radius range in metres (preset default: 0.8 6.0)",
    )
    tree_group.add_argument(
        "--crown-height",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="allowed vertical crown-radius range in metres (preset default: 0.6 6.5)",
    )

    building_group = parser.add_argument_group("building geometry and placement")
    building_group.add_argument(
        "--building-count", type=int, help="number of buildings (preset defaults: forest 0; buildings 8; mixed 4)"
    )
    building_group.add_argument(
        "--building-size",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="square building width/depth range in metres (preset default: 24 24)",
    )
    building_group.add_argument(
        "--building-height",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="building-height range in metres (preset default: 10 10)",
    )
    building_group.add_argument(
        "--building-gap", type=float, help="minimum edge-to-edge gap between buildings in metres (preset default: 2)"
    )
    building_group.add_argument(
        "--tree-building-gap",
        type=float,
        help="minimum tree-envelope clearance from building walls in metres (preset default: 0.3)",
    )
    building_group.add_argument(
        "--wall-thickness", type=float, help="sampled wall and roof thickness in metres (preset default: 0.24)"
    )

    sampling_group = parser.add_argument_group("point sampling and placement")
    sampling_group.add_argument(
        "--ground-spacing", type=float, help="ground-plane point spacing in metres (preset default: 0.20)"
    )
    sampling_group.add_argument(
        "--obstacle-spacing",
        type=float,
        help="trunk, branch, and building point spacing in metres (preset default: 0.12)",
    )
    sampling_group.add_argument(
        "--crown-spacing", type=float, help="tree-crown surface point spacing in metres (preset default: 0.18)"
    )
    sampling_group.add_argument(
        "--crown-fill", type=float, help="interior crown sampling ratio in [0,1] (preset default: 0.14)"
    )
    sampling_group.add_argument(
        "--voxel-size", type=float, help="final deduplication voxel size in metres (preset default: 0.10)"
    )
    sampling_group.add_argument(
        "--max-placement-attempts",
        type=int,
        help="maximum placement attempts per tree before failing (preset default: 5000)",
    )

    output_group = parser.add_argument_group("outputs")
    output_group.add_argument(
        "--output", help="output base path without an extension; required unless listing presets or printing config"
    )
    output_group.add_argument("--preview", action="store_true", help="also write a height-coloured PNG preview")
    output_group.add_argument("--force", action="store_true", help="overwrite every selected output file")
    return parser


def resolve_config(args: argparse.Namespace) -> SceneConfig:
    """Apply explicit command-line overrides to one complete scene preset."""
    config = SCENE_PRESETS[args.scene].config
    scalar_fields = {
        "seed": "seed",
        "tree_style": "tree_style",
        "tree_form": "tree_form",
        "tree_count": "tree_count",
        "tree_min_spacing": "tree_min_spacing",
        "low_tree_min_spacing": "low_tree_min_spacing",
        "building_count": "building_count",
        "building_gap": "building_gap",
        "tree_building_gap": "tree_building_gap",
        "wall_thickness": "wall_thickness",
        "clear_radius": "clear_radius",
        "ground_spacing": "ground_spacing",
        "obstacle_spacing": "obstacle_spacing",
        "crown_spacing": "crown_spacing",
        "crown_fill": "crown_fill",
        "voxel_size": "voxel_size",
        "max_placement_attempts": "max_placement_attempts",
    }
    overrides = {
        config_name: getattr(args, argument_name)
        for argument_name, config_name in scalar_fields.items()
        if getattr(args, argument_name) is not None
    }
    if args.size is not None:
        overrides.update(size_x=args.size[0], size_y=args.size[1])
    range_fields = {
        "tree_height": ("tree_height_min", "tree_height_max"),
        "trunk_radius": ("trunk_radius_min", "trunk_radius_max"),
        "crown_radius": ("crown_radius_min", "crown_radius_max"),
        "crown_height": ("crown_height_min", "crown_height_max"),
        "building_size": ("building_size_min", "building_size_max"),
        "building_height": ("building_height_min", "building_height_max"),
    }
    for argument_name, (minimum_name, maximum_name) in range_fields.items():
        values = getattr(args, argument_name)
        if values is not None:
            overrides[minimum_name], overrides[maximum_name] = values
    return replace(config, **overrides)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse generation settings and write one reproducible static map."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_presets:
        print(format_scene_presets())
        return 0

    try:
        config = resolve_config(args)
        _validate_config(config)
    except ValueError as exc:
        parser.exit(2, f"[pcl-generate] ERROR: {exc}\n")

    if args.print_config:
        print(json.dumps(asdict(config), indent=2, sort_keys=True))
        return 0
    if args.output is None:
        parser.error("--output is required unless --list-presets or --print-config is used")

    try:
        result = generate_scene(config)
        paths = save_scene(result, args.output, preview=args.preview, force=args.force)
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"[pcl-generate] ERROR: {exc}\n")

    bounds = result.metadata["bounds"]
    print(
        f"[pcl-generate] {result.metadata['scene']} seed={result.metadata['seed']}: "
        f"{result.points.shape[0]} points, bounds={bounds['min']}..{bounds['max']}"
    )
    for name, path in paths.items():
        print(f"[pcl-generate] wrote {name.upper()} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
