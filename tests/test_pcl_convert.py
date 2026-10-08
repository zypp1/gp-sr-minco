"""Focused tests for dependency-free point-cloud map conversion."""

import struct

import numpy as np
import pytest

from scripts.data_collect.pcl_tools.pcl_convert import convert_point_cloud, load_points


def _write_binary_ply(path, points):
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    path.write_bytes(header.encode("ascii") + np.asarray(points, dtype="<f4").tobytes())


def _write_ascii_pcd(path, points):
    points = np.asarray(points, dtype=np.float32)
    header = (
        "# test cloud\n"
        "VERSION 0.7\n"
        "FIELDS intensity z x y\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {len(points)}\n"
        "HEIGHT 1\n"
        f"POINTS {len(points)}\n"
        "DATA ascii\n"
    )
    rows = np.column_stack([np.arange(len(points)), points[:, 2], points[:, 0], points[:, 1]])
    with path.open("w") as handle:
        handle.write(header)
        np.savetxt(handle, rows, fmt="%.9g")


def _write_binary_pcd(path, points):
    points = np.asarray(points, dtype=np.float32)
    header = (
        "VERSION 0.7\n"
        "FIELDS y x z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {len(points)}\n"
        "HEIGHT 1\n"
        f"POINTS {len(points)}\n"
        "DATA binary\n"
    )
    records = np.empty(len(points), dtype=[("y", "<f4"), ("x", "<f4"), ("z", "<f4")])
    for axis in ("x", "y", "z"):
        records[axis] = points[:, "xyz".index(axis)]
    path.write_bytes(header.encode("ascii") + records.tobytes())


def _literal_lzf(data):
    """Encode small test data as valid LZF literal runs without back-references."""
    encoded = bytearray()
    for start in range(0, len(data), 32):
        chunk = data[start : start + 32]
        encoded.append(len(chunk) - 1)
        encoded.extend(chunk)
    return bytes(encoded)


def _write_binary_compressed_pcd(path, points):
    points = np.asarray(points, dtype="<f4")
    header = (
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {len(points)}\n"
        "HEIGHT 1\n"
        f"POINTS {len(points)}\n"
        "DATA binary_compressed\n"
    )
    field_major = b"".join(points[:, axis].tobytes() for axis in range(3))
    compressed = _literal_lzf(field_major)
    sizes = struct.pack("<II", len(compressed), len(field_major))
    path.write_bytes(header.encode("ascii") + sizes + compressed)


def test_convert_binary_ply_to_npz_and_pcd(tmp_path):
    points = np.array([[1.25, -2.5, 3.75], [4.0, 5.5, 6.25]], dtype=np.float32)
    source = tmp_path / "map.ply"
    npz_out = tmp_path / "map.npz"
    pcd_out = tmp_path / "map.pcd"
    _write_binary_ply(source, points)

    converted = convert_point_cloud(source, npz_out=npz_out, pcd_out=pcd_out, pcd_encoding="ascii")

    np.testing.assert_array_equal(converted, points)
    with np.load(npz_out) as data:
        assert data.files == ["points"]
        np.testing.assert_array_equal(data["points"], points)
    pcd_points = np.loadtxt(pcd_out, comments="#", skiprows=11, ndmin=2)
    np.testing.assert_allclose(pcd_points, points, atol=1e-7)
    assert "POINTS 2\nDATA ascii\n" in pcd_out.read_text()


def test_load_ascii_ply_selects_xyz_from_extra_scalar_properties(tmp_path):
    source = tmp_path / "colored.ply"
    source.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 2\n"
        "property uchar red\n"
        "property float z\n"
        "property float x\n"
        "property float y\n"
        "end_header\n"
        "255 3.0 1.0 2.0\n"
        "0 6.0 4.0 5.0\n"
    )

    points = load_points(source)

    np.testing.assert_array_equal(points, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))


def test_load_binary_ply_rejects_truncated_vertex_data(tmp_path):
    source = tmp_path / "truncated.ply"
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 2\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    source.write_bytes(header.encode("ascii") + struct.pack("<fff", 1.0, 2.0, 3.0))

    with pytest.raises(ValueError, match="expected 2 vertices"):
        load_points(source)


def test_convert_refuses_to_overwrite_without_force(tmp_path):
    source = tmp_path / "map.ply"
    npz_out = tmp_path / "map.npz"
    _write_binary_ply(source, np.zeros((1, 3), dtype=np.float32))
    npz_out.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="--force"):
        convert_point_cloud(source, npz_out=npz_out)

    assert npz_out.read_bytes() == b"keep"


def test_npz_input_preserves_existing_npz_to_pcd_workflow(tmp_path):
    points = np.array([[-1.0, 0.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    source = tmp_path / "map.npz"
    pcd_out = tmp_path / "map.pcd"
    np.savez_compressed(source, points=points)

    converted = convert_point_cloud(source, pcd_out=pcd_out, pcd_encoding="ascii")

    np.testing.assert_array_equal(converted, points)
    np.testing.assert_allclose(np.loadtxt(pcd_out, comments="#", skiprows=11), points, atol=1e-7)


def test_convert_ascii_pcd_to_npz_and_binary_ply(tmp_path):
    points = np.array([[1.25, -2.5, 3.75], [4.0, 5.5, 6.25]], dtype=np.float32)
    source = tmp_path / "map.pcd"
    npz_out = tmp_path / "map.npz"
    ply_out = tmp_path / "map.ply"
    _write_ascii_pcd(source, points)

    converted = convert_point_cloud(source, npz_out=npz_out, ply_out=ply_out)

    np.testing.assert_array_equal(converted, points)
    np.testing.assert_array_equal(load_points(ply_out), points)
    with np.load(npz_out) as data:
        np.testing.assert_array_equal(data["points"], points)
    assert b"format binary_little_endian 1.0" in ply_out.read_bytes()[:128]


def test_default_pcd_output_is_binary_and_round_trips(tmp_path):
    points = np.array([[1.25, -2.5, 3.75], [4.0, 5.5, 6.25]], dtype=np.float32)
    source = tmp_path / "map.ply"
    pcd_out = tmp_path / "map.pcd"
    _write_binary_ply(source, points)

    convert_point_cloud(source, pcd_out=pcd_out)

    assert b"DATA binary\n" in pcd_out.read_bytes()[:256]
    np.testing.assert_array_equal(load_points(pcd_out), points)


def test_convert_can_center_xy_ground_z_then_apply_explicit_offset(tmp_path):
    points = np.array([[10.0, 20.0, 1.0], [14.0, 28.0, 5.0]], dtype=np.float32)
    expected = np.array([[-1.0, -6.0, 0.5], [3.0, 2.0, 4.5]], dtype=np.float32)
    source = tmp_path / "map.ply"
    npz_out = tmp_path / "map.npz"
    pcd_out = tmp_path / "map.pcd"
    _write_binary_ply(source, points)

    converted = convert_point_cloud(
        source, npz_out=npz_out, pcd_out=pcd_out, center_xy=True, ground_z=True, offset=(1.0, -2.0, 0.5)
    )

    np.testing.assert_array_equal(converted, expected)
    np.testing.assert_array_equal(load_points(pcd_out), expected)
    with np.load(npz_out) as data:
        np.testing.assert_array_equal(data["points"], expected)


@pytest.mark.parametrize("writer", [_write_binary_pcd, _write_binary_compressed_pcd])
def test_load_binary_pcd_encodings(tmp_path, writer):
    points = np.array([[-1.0, 0.25, 2.5], [3.75, 4.0, -5.5]], dtype=np.float32)
    source = tmp_path / "map.pcd"
    writer(source, points)

    converted = load_points(source)

    np.testing.assert_array_equal(converted, points)


def test_load_pcd_rejects_missing_xyz_field(tmp_path):
    source = tmp_path / "bad.pcd"
    source.write_text(
        "VERSION 0.7\nFIELDS x y\nSIZE 4 4\nTYPE F F\nCOUNT 1 1\nWIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA ascii\n0 0\n"
    )

    with pytest.raises(ValueError, match="include x, y, and z"):
        load_points(source)
