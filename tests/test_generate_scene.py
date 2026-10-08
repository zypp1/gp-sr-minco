"""Focused tests for deterministic static point-cloud scene generation."""

import json

import numpy as np
import pytest

from scripts.data_collect.pcl_tools.pcl_convert import load_points
from scripts.data_collect.pcl_tools.pcl_generate import (
    SceneConfig,
    build_parser,
    generate_scene,
    main,
    resolve_config,
    save_scene,
)


def _small_config(**overrides):
    values = {
        "scene": "forest",
        "size_x": 20.0,
        "size_y": 18.0,
        "seed": 7,
        "tree_count": 5,
        "building_count": 0,
        "clear_radius": 3.0,
        "ground_spacing": 1.0,
        "obstacle_spacing": 0.4,
        "crown_spacing": 0.5,
        "crown_fill": 0.12,
        "voxel_size": 0.2,
        "building_size_min": 3.0,
        "building_size_max": 5.0,
        "building_height_min": 2.0,
        "building_height_max": 4.0,
    }
    values.update(overrides)
    return SceneConfig(**values)


@pytest.mark.parametrize(
    ("scene", "size", "tree_count", "building_count"),
    [("forest", (150.0, 150.0), 400, 0), ("buildings", (150.0, 150.0), 0, 8), ("mixed", (200.0, 200.0), 1800, 4)],
)
def test_scene_cli_resolves_complete_presets(scene, size, tree_count, building_count):
    args = build_parser().parse_args(["--scene", scene, "--print-config"])

    config = resolve_config(args)

    assert (config.size_x, config.size_y) == size
    assert config.tree_count == tree_count
    assert config.building_count == building_count
    assert config.building_size_min == config.building_size_max == 24.0
    assert config.building_height_min == config.building_height_max == 10.0


def test_cli_overrides_tree_building_and_sampling_preset_values():
    args = build_parser().parse_args(
        [
            "--scene",
            "mixed",
            "--size",
            "80",
            "70",
            "--seed",
            "9",
            "--tree-count",
            "12",
            "--tree-height",
            "3",
            "11",
            "--trunk-radius",
            "0.1",
            "0.4",
            "--crown-radius",
            "1",
            "4",
            "--crown-height",
            "0.8",
            "3",
            "--building-count",
            "2",
            "--building-size",
            "8",
            "12",
            "--building-height",
            "5",
            "9",
            "--building-gap",
            "3",
            "--wall-thickness",
            "0.3",
            "--max-placement-attempts",
            "8000",
            "--voxel-size",
            "0.2",
            "--print-config",
        ]
    )

    config = resolve_config(args)

    assert (config.size_x, config.size_y, config.seed) == (80.0, 70.0, 9)
    assert (config.tree_count, config.tree_height_min, config.tree_height_max) == (12, 3.0, 11.0)
    assert (config.trunk_radius_min, config.trunk_radius_max) == (0.1, 0.4)
    assert (config.crown_radius_min, config.crown_radius_max) == (1.0, 4.0)
    assert (config.crown_height_min, config.crown_height_max) == (0.8, 3.0)
    assert (config.building_count, config.building_size_min, config.building_size_max) == (2, 8.0, 12.0)
    assert (config.building_height_min, config.building_height_max) == (5.0, 9.0)
    assert (config.building_gap, config.wall_thickness) == (3.0, 0.3)
    assert (config.max_placement_attempts, config.voxel_size) == (8000, 0.2)


def test_cli_discovery_and_config_inspection_do_not_write_files(capsys):
    assert main(["--list-presets"]) == 0
    preset_output = capsys.readouterr().out
    assert all(scene in preset_output for scene in ("forest", "buildings", "mixed"))

    assert main(["--scene", "mixed", "--tree-count", "20", "--print-config"]) == 0
    config_output = json.loads(capsys.readouterr().out)
    assert config_output["scene"] == "mixed"
    assert config_output["size_x"] == 200.0
    assert config_output["tree_count"] == 20


def test_cli_help_exposes_every_scene_control_group():
    help_text = build_parser().format_help()

    for option in (
        "--scene",
        "--seed",
        "--size",
        "--clear-radius",
        "--list-presets",
        "--print-config",
        "--tree-style",
        "--tree-form",
        "--tree-count",
        "--tree-min-spacing",
        "--low-tree-min-spacing",
        "--tree-height",
        "--trunk-radius",
        "--crown-radius",
        "--crown-height",
        "--building-count",
        "--building-size",
        "--building-height",
        "--building-gap",
        "--tree-building-gap",
        "--wall-thickness",
        "--ground-spacing",
        "--obstacle-spacing",
        "--crown-spacing",
        "--crown-fill",
        "--voxel-size",
        "--max-placement-attempts",
        "--output",
        "--preview",
        "--force",
    ):
        assert option in help_text


def test_forest_generation_is_seeded_unique_and_center_clear():
    config = _small_config()

    first = generate_scene(config)
    second = generate_scene(config)
    changed = generate_scene(_small_config(seed=8))

    np.testing.assert_array_equal(first.points, second.points)
    assert not np.array_equal(first.points, changed.points)
    assert np.unique(first.points, axis=0).shape == first.points.shape
    assert first.metadata["objects"]["trees"] == 5
    assert first.metadata["objects"]["buildings"] == 0
    assert sum(first.metadata["tree_styles"].values()) == 5
    assert sum(first.metadata["tree_forms"].values()) == 5
    assert first.metadata["tree_statistics"]["height_m"][0] >= config.tree_height_min
    assert first.metadata["tree_statistics"]["height_m"][1] <= config.tree_height_max
    assert first.metadata["tree_statistics"]["lean_m"][1] > 0.0

    lower = first.points.min(axis=0)
    upper = first.points.max(axis=0)
    np.testing.assert_allclose(lower[:2], [-10.0, -9.0], atol=1e-6)
    np.testing.assert_allclose(upper[:2], [10.0, 9.0], atol=1e-6)
    assert lower[2] == 0.0
    assert upper[2] > 3.0

    obstacle = first.points[:, 2] > 0.05
    radius = np.linalg.norm(first.points[obstacle, :2], axis=1)
    assert radius.min() >= config.clear_radius


def test_mixed_scene_contains_requested_objects_and_static_ground():
    result = generate_scene(_small_config(scene="mixed", size_x=36.0, size_y=32.0, tree_count=4, building_count=2))

    assert result.metadata["objects"] == {"trees": 4, "buildings": 2}
    assert result.metadata["components"]["ground_raw"] > 0
    assert result.metadata["components"]["trees_raw"] > 0
    assert result.metadata["components"]["buildings_raw"] > 0
    assert np.any(result.points[:, 2] == 0.0)


def test_buildings_scene_contains_only_requested_axis_aligned_buildings():
    result = generate_scene(
        _small_config(
            scene="buildings",
            size_x=40.0,
            size_y=40.0,
            tree_count=0,
            building_count=1,
            building_size_min=6.0,
            building_size_max=6.0,
            building_height_min=4.0,
            building_height_max=4.0,
        )
    )

    assert result.metadata["objects"] == {"trees": 0, "buildings": 1}
    assert result.metadata["components"]["trees_raw"] == 0
    assert result.metadata["components"]["buildings_raw"] > 0
    assert result.metadata["building_placements"][0]["yaw_rad"] == 0.0


def test_four_buildings_use_axis_aligned_two_by_two_grid():
    result = generate_scene(
        _small_config(
            scene="mixed",
            size_x=60.0,
            size_y=52.0,
            tree_count=4,
            building_count=4,
            building_size_min=4.0,
            building_size_max=4.0,
            building_height_min=3.0,
            building_height_max=3.0,
        )
    )

    placements = result.metadata["building_placements"]
    assert len(placements) == 4
    assert all(building["yaw_rad"] == 0.0 for building in placements)
    assert all(building["size_m"] == [4.0, 4.0, 3.0] for building in placements)
    assert len({building["center_m"][0] for building in placements}) == 2
    assert len({building["center_m"][1] for building in placements}) == 2
    assert sum(building["center_m"][0] for building in placements) == pytest.approx(0.0)
    assert sum(building["center_m"][1] for building in placements) == pytest.approx(0.0)


@pytest.mark.parametrize("tree_style", ["broadleaf", "conifer"])
def test_explicit_tree_style_controls_every_generated_tree(tree_style):
    result = generate_scene(_small_config(tree_style=tree_style))

    expected = {"broadleaf": 0, "conifer": 0}
    expected[tree_style] = 5
    assert result.metadata["tree_styles"] == expected
    if tree_style == "conifer":
        assert result.metadata["tree_forms"]["conifer"] == 5
    else:
        assert result.metadata["tree_forms"]["conifer"] == 0


def test_default_mixed_forest_covers_short_wide_and_large_crown_forms():
    result = generate_scene(
        SceneConfig(
            size_x=100.0,
            size_y=100.0,
            seed=19,
            tree_count=60,
            clear_radius=3.0,
            ground_spacing=2.0,
            obstacle_spacing=0.5,
            crown_spacing=0.8,
            voxel_size=0.25,
        )
    )

    assert all(count > 0 for count in result.metadata["tree_forms"].values())
    statistics = result.metadata["tree_statistics"]
    assert statistics["height_m"][0] < 3.0
    assert statistics["crown_diameter_m"][1] > 10.0
    assert statistics["minimum_low_tree_spacing_m"] >= 4.0
    assert statistics["minimum_high_low_tree_spacing_m"] >= 4.0


def test_low_tree_forms_keep_their_larger_minimum_spacing():
    config = _small_config(
        size_x=44.0, size_y=44.0, tree_count=6, clear_radius=0.0, tree_form="shrub", low_tree_min_spacing=4.0
    )

    result = generate_scene(config)

    assert result.metadata["tree_statistics"]["minimum_low_tree_spacing_m"] >= 4.0


def test_mixed_scene_records_tree_to_building_clearance():
    config = _small_config(
        scene="mixed", size_x=52.0, size_y=48.0, tree_count=8, building_count=2, tree_building_gap=0.3
    )

    result = generate_scene(config)

    assert result.metadata["tree_statistics"]["minimum_building_clearance_m"] >= 0.3


@pytest.mark.parametrize("tree_form", ["shrub", "orchard", "willow"])
def test_exact_tree_form_can_be_generated_for_visual_checks(tree_form):
    result = generate_scene(
        _small_config(size_x=44.0, size_y=44.0, tree_count=2, clear_radius=0.0, tree_form=tree_form)
    )

    assert result.metadata["tree_forms"][tree_form] == 2
    assert sum(result.metadata["tree_forms"].values()) == 2


def test_save_scene_writes_exact_binary_pcd_npz_pair_and_metadata(tmp_path):
    result = generate_scene(_small_config(tree_count=2))
    output_base = tmp_path / "forest_demo"

    paths = save_scene(result, output_base)

    assert b"DATA binary\n" in paths["pcd"].read_bytes()[:256]
    np.testing.assert_array_equal(load_points(paths["pcd"]), result.points)
    with np.load(paths["npz"], allow_pickle=False) as data:
        assert data.files == ["points"]
        np.testing.assert_array_equal(data["points"], result.points)
    metadata = json.loads(paths["json"].read_text())
    assert metadata["point_count"] == result.points.shape[0]

    with pytest.raises(FileExistsError, match="--force"):
        save_scene(result, output_base)


@pytest.mark.parametrize(
    "config",
    [
        _small_config(scene="unknown"),
        _small_config(tree_style="unknown"),
        _small_config(tree_form="unknown"),
        _small_config(tree_style="conifer", tree_form="willow"),
        _small_config(size_x=5.0, clear_radius=3.0),
        _small_config(ground_spacing=0.0),
        _small_config(crown_fill=1.1),
        _small_config(low_tree_min_spacing=1.0),
        _small_config(tree_building_gap=-0.1),
        _small_config(scene="forest", tree_count=0),
        _small_config(scene="forest", building_count=1),
        _small_config(scene="buildings", tree_count=0, building_count=0),
        _small_config(scene="buildings", tree_count=1, building_count=1),
    ],
)
def test_generate_scene_rejects_invalid_configuration(config):
    with pytest.raises(ValueError):
        generate_scene(config)
