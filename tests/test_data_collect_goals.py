"""Focused tests for random-goal sampling and replacement policy."""

import json
import math

import numpy as np
import pytest

from scripts.data_collect import goals


def test_sample_rejects_out_of_bounds_candidate_instead_of_clipping(monkeypatch):
    draws = iter([0.0, 5.0, math.pi, 5.0])
    clearance_queries = []
    monkeypatch.setattr(goals.random, "uniform", lambda _low, _high: next(draws))
    monkeypatch.setattr(goals, "_clearance", lambda _esdf, xyz: clearance_queries.append(np.asarray(xyz)) or 2.0)

    sampled = goals._sample(
        object(),
        origin=np.asarray([9.0, 0.0, 1.5]),
        z=1.5,
        dist_min=5.0,
        dist_max=5.0,
        xr=(-10.0, 10.0),
        yr=(-10.0, 10.0),
        max_tries=2,
    )

    np.testing.assert_allclose(sampled, [4.0, 0.0, 1.5], atol=1e-12)
    assert len(clearance_queries) == 1
    np.testing.assert_allclose(clearance_queries[0], sampled)


def test_sample_does_not_query_clearance_for_out_of_bounds_candidate(monkeypatch):
    monkeypatch.setattr(goals.random, "uniform", lambda low, high: low if high > 6.0 else high)
    monkeypatch.setattr(
        goals, "_clearance", lambda _esdf, _xyz: (_ for _ in ()).throw(AssertionError("out-of-bounds clearance query"))
    )

    sampled = goals._sample(
        object(),
        origin=np.asarray([9.0, 0.0, 1.5]),
        z=1.5,
        dist_min=5.0,
        dist_max=5.0,
        xr=(-10.0, 10.0),
        yr=(-10.0, 10.0),
        max_tries=1,
    )

    assert sampled is None


def test_sample_rejects_building_interior_before_querying_esdf(monkeypatch):
    draws = iter([0.0, 5.0, math.pi, 5.0])
    clearance_queries = []
    monkeypatch.setattr(goals.random, "uniform", lambda _low, _high: next(draws))
    monkeypatch.setattr(goals, "_clearance", lambda _esdf, xyz: clearance_queries.append(np.asarray(xyz)) or 2.0)
    building = goals.BuildingVolume(center_x=5.0, center_y=0.0, width=4.0, depth=4.0, height=6.0, yaw=0.0)

    sampled = goals._sample(
        object(),
        origin=np.asarray([0.0, 0.0, 1.5]),
        z=1.5,
        dist_min=5.0,
        dist_max=5.0,
        xr=(-10.0, 10.0),
        yr=(-10.0, 10.0),
        max_tries=2,
        buildings=(building,),
    )

    np.testing.assert_allclose(sampled, [-5.0, 0.0, 1.5], atol=1e-12)
    assert len(clearance_queries) == 1
    np.testing.assert_allclose(clearance_queries[0], sampled)


def test_building_volume_respects_yaw_height_and_safety_margin():
    building = goals.BuildingVolume(center_x=0.0, center_y=0.0, width=4.0, depth=2.0, height=3.0, yaw=0.5 * math.pi)

    assert building.contains([0.0, 1.5, 1.0])
    assert not building.contains([1.5, 0.0, 1.0])
    assert not building.contains([0.0, 0.0, 3.2])
    assert building.contains([1.5, 0.0, 1.0], margin=0.6)


def test_building_volume_detects_crossing_segment_but_not_clear_or_overhead_path():
    building = goals.BuildingVolume(center_x=-50.0, center_y=-50.0, width=24.0, depth=24.0, height=10.0, yaw=0.0)

    assert building.intersects_segment([-47.35, -27.15, 1.55], [-36.69, -53.73, 1.5], margin=0.3)
    assert not building.intersects_segment([-47.35, -27.15, 1.55], [-30.0, -30.0, 1.5], margin=0.3)
    assert not building.intersects_segment([-47.35, -27.15, 12.0], [-36.69, -53.73, 12.0], margin=0.3)


def test_sample_rejects_goal_whose_route_segment_crosses_building(monkeypatch):
    draws = iter([-0.5 * math.pi, 20.0, 0.0, 20.0])
    clearance_queries = []
    monkeypatch.setattr(goals.random, "uniform", lambda _low, _high: next(draws))
    monkeypatch.setattr(goals, "_clearance", lambda _esdf, xyz: clearance_queries.append(np.asarray(xyz)) or 2.0)
    building = goals.BuildingVolume(center_x=0.0, center_y=0.0, width=10.0, depth=10.0, height=6.0, yaw=0.0)

    sampled = goals._sample(
        object(),
        origin=np.asarray([0.0, 10.0, 1.5]),
        z=1.5,
        dist_min=20.0,
        dist_max=20.0,
        xr=(-25.0, 25.0),
        yr=(-25.0, 25.0),
        max_tries=2,
        buildings=(building,),
    )

    np.testing.assert_allclose(sampled, [20.0, 10.0, 1.5], atol=1e-12)
    assert len(clearance_queries) == 1


def test_building_metadata_is_required_and_validated(tmp_path):
    map_path = tmp_path / "scene.npz"
    metadata_path = map_path.with_suffix(".json")
    lower = np.asarray([-10.0, -8.0, 0.0])
    upper = np.asarray([10.0, 8.0, 6.0])
    metadata = {
        "format": "static_xyz_occupancy",
        "frame": "world",
        "point_count": 123,
        "bounds": {"min": lower.tolist(), "max": upper.tolist()},
        "objects": {"trees": 0, "buildings": 1},
        "building_placements": [{"center_m": [2.0, -1.0], "size_m": [8.0, 6.0, 5.0], "yaw_rad": 0.25}],
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    buildings, source = goals._load_building_volumes(map_path, point_count=123, lower=lower, upper=upper)

    assert source == metadata_path
    assert len(buildings) == 1
    assert buildings[0].contains([2.0, -1.0, 1.5])
    with pytest.raises(ValueError, match="point_count"):
        goals._load_building_volumes(map_path, point_count=124, lower=lower, upper=upper)
    metadata_path.unlink()
    with pytest.raises(FileNotFoundError, match="building-safe goal sampling"):
        goals._load_building_volumes(map_path, point_count=123, lower=lower, upper=upper)


def test_replacement_reason_waits_for_collector_result_without_own_timeout():
    current = np.asarray([10.0, 0.0, 1.5])

    assert goals._replacement_reason(current, None, False, 10_000.0, 0.5) is None
    assert goals._replacement_reason(current, None, True, 10_000.0, 0.5) == "failure"


def test_replacement_reason_waits_after_saved_success():
    current = np.asarray([0.5, 0.0, 1.5])

    assert goals._replacement_reason(None, None, False, 10.0, 0.5) == "initial"
    assert goals._replacement_reason(current, 100.0, False, 100.499, 0.5) is None
    assert goals._replacement_reason(current, 100.0, False, 100.5, 0.5) == "success"


def test_success_event_must_match_complete_goal_position():
    goal = np.asarray([1.0, 2.0, 3.0])

    assert goals._same_goal(goal, [1.0005, 2.0, 3.0])
    assert not goals._same_goal(goal, [1.0, 2.0, 3.01])
    assert not goals._same_goal(None, goal)


def test_odom_goal_reach_requires_complete_3d_distance_and_low_speed():
    goal = np.asarray([1.0, 2.0, 3.0])

    assert goals._has_reached_state([1.3, 2.4, 3.0], [0.1, 0.0, 0.0], goal, distance_tolerance=0.5, speed_tolerance=0.2)
    assert not goals._has_reached_state(
        [1.0, 2.0, 3.6], [0.0, 0.0, 0.0], goal, distance_tolerance=0.5, speed_tolerance=0.2
    )
    assert not goals._has_reached_state(goal, [0.3, 0.0, 0.0], goal, distance_tolerance=0.5, speed_tolerance=0.2)
    assert not goals._has_reached_state(None, None, goal, distance_tolerance=0.5, speed_tolerance=0.2)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        goals._has_reached_state(goal, [0.0, 0.0, 0.0], goal, distance_tolerance=-0.1, speed_tolerance=0.2)
