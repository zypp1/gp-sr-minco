"""Convert XYZ point clouds between PLY, PCD, and NPZ formats.

The converter has no Open3D or PCL dependency.  It reads ASCII or binary PLY, ASCII/binary/
binary-compressed PCD, and NPZ containing ``points`` with shape ``[N,3]``.  Outputs contain XYZ
only: PLY and PCD default to binary little-endian, and NPZ stores compressed float32 ``points``.

Example:

    python3 pcl_convert.py --in cloud.pcd --npz-out cloud.npz --ply-out cloud.ply
"""

import argparse
import struct
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

_PLY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}
_PLY_FORMAT_ENDIAN = {"binary_little_endian": "<", "binary_big_endian": ">"}
_PCD_DTYPES = {
    ("F", 4): "<f4",
    ("F", 8): "<f8",
    ("I", 1): "i1",
    ("I", 2): "<i2",
    ("I", 4): "<i4",
    ("I", 8): "<i8",
    ("U", 1): "u1",
    ("U", 2): "<u2",
    ("U", 4): "<u4",
    ("U", 8): "<u8",
}


def _read_ply_header(handle: BinaryIO) -> Tuple[str, int, List[Tuple[str, str]]]:
    """Read a PLY header and return format, vertex count, and scalar vertex properties."""
    first = handle.readline()
    if first.rstrip(b"\r\n") != b"ply":
        raise ValueError("PLY file must start with the 'ply' magic line")

    ply_format = None
    elements: List[Dict[str, object]] = []
    current = None
    header_bytes = len(first)
    while True:
        raw = handle.readline()
        if not raw:
            raise ValueError("PLY header is missing end_header")
        header_bytes += len(raw)
        if header_bytes > 1024 * 1024:
            raise ValueError("PLY header exceeds 1 MiB")
        try:
            line = raw.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("PLY header must be ASCII") from exc
        if not line or line.startswith("comment ") or line.startswith("obj_info "):
            continue

        fields = line.split()
        if fields[0] == "format":
            if len(fields) != 3 or fields[2] != "1.0":
                raise ValueError(f"unsupported PLY format declaration: {line}")
            ply_format = fields[1]
        elif fields[0] == "element":
            if len(fields) != 3:
                raise ValueError(f"malformed PLY element declaration: {line}")
            try:
                count = int(fields[2])
            except ValueError as exc:
                raise ValueError(f"invalid PLY element count: {line}") from exc
            if count < 0:
                raise ValueError(f"negative PLY element count: {line}")
            current = {"name": fields[1], "count": count, "properties": []}
            elements.append(current)
        elif fields[0] == "property":
            if current is None:
                raise ValueError("PLY property appeared before an element")
            if len(fields) >= 2 and fields[1] == "list":
                current["properties"].append(("__list__", line))
            elif len(fields) == 3 and fields[1] in _PLY_DTYPES:
                current["properties"].append((fields[2], fields[1]))
            else:
                raise ValueError(f"unsupported PLY property declaration: {line}")
        elif fields[0] == "end_header":
            break
        else:
            raise ValueError(f"unsupported PLY header line: {line}")

    if ply_format not in ("ascii", *_PLY_FORMAT_ENDIAN):
        raise ValueError(f"unsupported PLY format: {ply_format}")
    vertex_indices = [i for i, element in enumerate(elements) if element["name"] == "vertex"]
    if len(vertex_indices) != 1:
        raise ValueError("PLY must contain exactly one vertex element")
    vertex_index = vertex_indices[0]
    if any(int(element["count"]) for element in elements[:vertex_index]):
        raise ValueError("PLY vertex data must precede other non-empty elements")

    vertex = elements[vertex_index]
    properties = list(vertex["properties"])
    if any(name == "__list__" for name, _ in properties):
        raise ValueError("PLY vertex list properties are unsupported")
    names = [name for name, _ in properties]
    if len(names) != len(set(names)):
        raise ValueError("PLY vertex property names must be unique")
    if not all(axis in names for axis in ("x", "y", "z")):
        raise ValueError("PLY vertex properties must include scalar x, y, and z")
    return str(ply_format), int(vertex["count"]), properties


def _read_ply_ascii(handle: BinaryIO, count: int, properties: List[Tuple[str, str]]) -> np.ndarray:
    """Read ASCII PLY vertex rows and select their XYZ columns."""
    indices = [next(i for i, prop in enumerate(properties) if prop[0] == axis) for axis in ("x", "y", "z")]
    points = np.empty((count, 3), dtype=np.float32)
    for row in range(count):
        raw = handle.readline()
        if not raw:
            raise ValueError(f"PLY expected {count} vertices but ended after {row}")
        fields = raw.split()
        if len(fields) != len(properties):
            raise ValueError(f"PLY vertex {row} has {len(fields)} values; expected {len(properties)}")
        try:
            points[row] = [float(fields[index]) for index in indices]
        except ValueError as exc:
            raise ValueError(f"PLY vertex {row} contains a non-numeric XYZ value") from exc
    return points


def _read_ply_binary(handle: BinaryIO, ply_format: str, count: int, properties: List[Tuple[str, str]]) -> np.ndarray:
    """Read binary PLY vertex records and select their XYZ fields."""
    endian = _PLY_FORMAT_ENDIAN[ply_format]
    dtype = np.dtype([(name, endian + _PLY_DTYPES[data_type]) for name, data_type in properties])
    vertices = np.fromfile(handle, dtype=dtype, count=count)
    if vertices.shape[0] != count:
        raise ValueError(f"PLY expected {count} vertices but found {vertices.shape[0]}")
    return np.column_stack([vertices[axis] for axis in ("x", "y", "z")]).astype(np.float32)


def _read_pcd_header(handle: BinaryIO) -> Tuple[List[Tuple[str, str, int]], int, str]:
    """Read a PCD header and return field specifications, point count, and data encoding."""
    entries: Dict[str, List[str]] = {}
    header_bytes = 0
    while True:
        raw = handle.readline()
        if not raw:
            raise ValueError("PCD header is missing DATA")
        header_bytes += len(raw)
        if header_bytes > 1024 * 1024:
            raise ValueError("PCD header exceeds 1 MiB")
        try:
            line = raw.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("PCD header must be ASCII") from exc
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        key = fields[0].upper()
        if key in entries:
            raise ValueError(f"duplicate PCD header field: {key}")
        entries[key] = fields[1:]
        if key == "DATA":
            break

    names = entries.get("FIELDS")
    sizes_raw = entries.get("SIZE")
    types = entries.get("TYPE")
    if not names or not sizes_raw or not types:
        raise ValueError("PCD header requires FIELDS, SIZE, and TYPE")
    counts_raw = entries.get("COUNT", ["1"] * len(names))
    if not (len(names) == len(sizes_raw) == len(types) == len(counts_raw)):
        raise ValueError("PCD FIELDS, SIZE, TYPE, and COUNT lengths must match")
    if len(names) != len(set(names)):
        raise ValueError("PCD field names must be unique")
    if not all(axis in names for axis in ("x", "y", "z")):
        raise ValueError("PCD fields must include x, y, and z")

    try:
        sizes = [int(value) for value in sizes_raw]
        counts = [int(value) for value in counts_raw]
    except ValueError as exc:
        raise ValueError("PCD SIZE and COUNT values must be integers") from exc
    if any(count <= 0 for count in counts):
        raise ValueError("PCD COUNT values must be positive")
    field_specs = []
    for name, data_type, size, count in zip(names, types, sizes, counts):
        key = (data_type.upper(), size)
        if key not in _PCD_DTYPES:
            raise ValueError(f"unsupported PCD field type/size: {name} {data_type} {size}")
        field_specs.append((name, _PCD_DTYPES[key], count))
    if any(counts[names.index(axis)] != 1 for axis in ("x", "y", "z")):
        raise ValueError("PCD x, y, and z fields must each have COUNT 1")

    try:
        if "POINTS" in entries:
            point_count = int(entries["POINTS"][0])
        else:
            point_count = int(entries["WIDTH"][0]) * int(entries.get("HEIGHT", ["1"])[0])
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError("PCD header requires a valid POINTS or WIDTH/HEIGHT count") from exc
    if point_count < 0:
        raise ValueError("PCD point count must be non-negative")
    data_fields = entries["DATA"]
    if len(data_fields) != 1 or data_fields[0].lower() not in ("ascii", "binary", "binary_compressed"):
        raise ValueError(f"unsupported PCD DATA encoding: {' '.join(data_fields)}")
    return field_specs, point_count, data_fields[0].lower()


def _read_pcd_ascii(handle: BinaryIO, field_specs: List[Tuple[str, str, int]], count: int) -> np.ndarray:
    """Read ASCII PCD records and select their XYZ scalar fields."""
    offsets = {}
    scalar_count = 0
    for name, _, field_count in field_specs:
        offsets[name] = scalar_count
        scalar_count += field_count
    points = np.empty((count, 3), dtype=np.float32)
    for row in range(count):
        raw = handle.readline()
        if not raw:
            raise ValueError(f"PCD expected {count} points but ended after {row}")
        fields = raw.split()
        if len(fields) != scalar_count:
            raise ValueError(f"PCD point {row} has {len(fields)} values; expected {scalar_count}")
        try:
            points[row] = [float(fields[offsets[axis]]) for axis in ("x", "y", "z")]
        except ValueError as exc:
            raise ValueError(f"PCD point {row} contains a non-numeric XYZ value") from exc
    return points


def _read_pcd_binary(handle: BinaryIO, field_specs: List[Tuple[str, str, int]], count: int) -> np.ndarray:
    """Read point-major binary PCD records and select their XYZ fields."""
    dtype_fields = []
    for name, data_type, field_count in field_specs:
        if field_count == 1:
            dtype_fields.append((name, data_type))
        else:
            dtype_fields.append((name, data_type, (field_count,)))
    vertices = np.fromfile(handle, dtype=np.dtype(dtype_fields), count=count)
    if vertices.shape[0] != count:
        raise ValueError(f"PCD expected {count} points but found {vertices.shape[0]}")
    return np.column_stack([vertices[axis] for axis in ("x", "y", "z")]).astype(np.float32)


def _lzf_decompress(compressed: bytes, expected_size: int) -> bytes:
    """Decompress the LZF byte stream used by PCD binary_compressed data."""
    output = bytearray()
    cursor = 0
    while cursor < len(compressed):
        control = compressed[cursor]
        cursor += 1
        if control < 32:
            length = control + 1
            if cursor + length > len(compressed):
                raise ValueError("truncated LZF literal run in PCD data")
            output.extend(compressed[cursor : cursor + length])
            cursor += length
        else:
            length = control >> 5
            reference_offset = (control & 0x1F) << 8
            if length == 7:
                if cursor >= len(compressed):
                    raise ValueError("truncated LZF match length in PCD data")
                length += compressed[cursor]
                cursor += 1
            if cursor >= len(compressed):
                raise ValueError("truncated LZF match offset in PCD data")
            reference_offset += compressed[cursor]
            cursor += 1
            reference = len(output) - reference_offset - 1
            length += 2
            if reference < 0:
                raise ValueError("invalid LZF back-reference in PCD data")
            for _ in range(length):
                output.append(output[reference])
                reference += 1
        if len(output) > expected_size:
            raise ValueError("LZF PCD data exceeds its declared uncompressed size")
    if len(output) != expected_size:
        raise ValueError(f"LZF PCD data decoded to {len(output)} bytes; expected {expected_size}")
    return bytes(output)


def _read_pcd_binary_compressed(handle: BinaryIO, field_specs: List[Tuple[str, str, int]], count: int) -> np.ndarray:
    """Read field-major LZF-compressed PCD records and select their XYZ fields."""
    sizes_raw = handle.read(8)
    if len(sizes_raw) != 8:
        raise ValueError("PCD binary_compressed data is missing its size prefix")
    compressed_size, uncompressed_size = struct.unpack("<II", sizes_raw)
    compressed = handle.read(compressed_size)
    if len(compressed) != compressed_size:
        raise ValueError(f"PCD expected {compressed_size} compressed bytes but found {len(compressed)}")
    expected_size = sum(np.dtype(data_type).itemsize * field_count * count for _, data_type, field_count in field_specs)
    if uncompressed_size != expected_size:
        raise ValueError(f"PCD declares {uncompressed_size} uncompressed bytes; expected {expected_size}")
    unpacked = _lzf_decompress(compressed, uncompressed_size)

    columns = {}
    offset = 0
    for name, data_type, field_count in field_specs:
        item_count = count * field_count
        field = np.frombuffer(unpacked, dtype=data_type, count=item_count, offset=offset)
        if name in ("x", "y", "z"):
            columns[name] = field.reshape(count, field_count)[:, 0]
        offset += field.nbytes
    return np.column_stack([columns[axis] for axis in ("x", "y", "z")]).astype(np.float32)


def _read_pcd(handle: BinaryIO) -> np.ndarray:
    """Read XYZ values from any supported PCD DATA encoding."""
    field_specs, count, encoding = _read_pcd_header(handle)
    if encoding == "ascii":
        return _read_pcd_ascii(handle, field_specs, count)
    if encoding == "binary":
        return _read_pcd_binary(handle, field_specs, count)
    return _read_pcd_binary_compressed(handle, field_specs, count)


def _validate_points(points: np.ndarray, source: Path) -> np.ndarray:
    """Return contiguous float32 XYZ points after validating the map contract."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{source} points must have shape [N,3], got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError(f"{source} contains no points")
    if not np.isfinite(points).all():
        raise ValueError(f"{source} contains non-finite XYZ coordinates")
    return np.ascontiguousarray(points)


def load_points(path: PathLike) -> np.ndarray:
    """Load contiguous float32 XYZ points from PLY, PCD, or NPZ."""
    source = Path(path).expanduser()
    suffix = source.suffix.lower()
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as data:
            if "points" not in data:
                raise ValueError(f"{source} is missing the required 'points' array")
            points = data["points"]
    elif suffix == ".ply":
        with source.open("rb") as handle:
            ply_format, count, properties = _read_ply_header(handle)
            if ply_format == "ascii":
                points = _read_ply_ascii(handle, count, properties)
            else:
                points = _read_ply_binary(handle, ply_format, count, properties)
    elif suffix == ".pcd":
        with source.open("rb") as handle:
            points = _read_pcd(handle)
    else:
        raise ValueError(f"unsupported input extension '{source.suffix}'; expected .ply, .pcd, or .npz")
    return _validate_points(points, source)


def _pcd_header(point_count: int, encoding: str) -> str:
    """Return a PCD v0.7 XYZ header for the requested data encoding."""
    return (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {point_count}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {point_count}\n"
        f"DATA {encoding}\n"
    )


def write_pcd_ascii(path: PathLike, points: np.ndarray) -> None:
    """Write a dependency-free ASCII PCD containing float32 XYZ fields."""
    destination = Path(path).expanduser()
    with destination.open("w", encoding="ascii") as handle:
        handle.write(_pcd_header(points.shape[0], "ascii"))
        np.savetxt(handle, points, fmt="%.9g")


def write_pcd_binary(path: PathLike, points: np.ndarray) -> None:
    """Write a binary PCD containing point-major little-endian float32 XYZ fields."""
    destination = Path(path).expanduser()
    records = np.ascontiguousarray(points, dtype="<f4")
    with destination.open("wb") as handle:
        handle.write(_pcd_header(records.shape[0], "binary").encode("ascii"))
        records.tofile(handle)


def write_ply_binary(path: PathLike, points: np.ndarray) -> None:
    """Write a binary little-endian PLY containing float32 XYZ vertex properties."""
    destination = Path(path).expanduser()
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with destination.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(np.asarray(points, dtype="<f4").tobytes())


def _prepare_output(source: Path, destination: Path, force: bool) -> None:
    """Validate and create an output location before conversion starts."""
    if destination.resolve() == source.resolve():
        raise ValueError(f"output path must differ from input path: {destination}")
    if destination.exists() and not force:
        raise FileExistsError(f"{destination} already exists; pass --force to overwrite it")
    destination.parent.mkdir(parents=True, exist_ok=True)


def convert_point_cloud(
    input_path: PathLike,
    npz_out: Optional[PathLike] = None,
    pcd_out: Optional[PathLike] = None,
    ply_out: Optional[PathLike] = None,
    center_xy: bool = False,
    ground_z: bool = False,
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    pcd_encoding: str = "binary",
    force: bool = False,
) -> np.ndarray:
    """Convert and optionally translate one point cloud to requested formats."""
    source = Path(input_path).expanduser()
    if npz_out is None and pcd_out is None and ply_out is None:
        raise ValueError("at least one output path is required")
    if pcd_encoding not in ("ascii", "binary"):
        raise ValueError(f"unsupported PCD output encoding: {pcd_encoding}")
    translation = np.asarray(offset, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("offset must contain three finite XYZ values")
    npz_path = Path(npz_out).expanduser() if npz_out is not None else None
    pcd_path = Path(pcd_out).expanduser() if pcd_out is not None else None
    ply_path = Path(ply_out).expanduser() if ply_out is not None else None
    outputs = [(npz_path, ".npz"), (pcd_path, ".pcd"), (ply_path, ".ply")]
    selected = []
    for destination, expected_suffix in outputs:
        if destination is None:
            continue
        if destination.suffix.lower() != expected_suffix:
            raise ValueError(f"{destination} must use the {expected_suffix} extension")
        _prepare_output(source, destination, force)
        selected.append(destination.resolve())
    if len(selected) != len(set(selected)):
        raise ValueError("output paths must be distinct")

    points = load_points(source)
    if center_xy:
        lower = points[:, :2].min(axis=0).astype(np.float64)
        upper = points[:, :2].max(axis=0).astype(np.float64)
        translation[:2] -= (lower + upper) * 0.5
    if ground_z:
        translation[2] -= float(points[:, 2].min())
    points += translation.astype(np.float32)
    if npz_path is not None:
        np.savez_compressed(npz_path, points=points)
    if pcd_path is not None:
        if pcd_encoding == "ascii":
            write_pcd_ascii(pcd_path, points)
        else:
            write_pcd_binary(pcd_path, points)
    if ply_path is not None:
        write_ply_binary(ply_path, points)
    return points


def main() -> None:
    """Parse conversion arguments and print the validated point-cloud summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="input_path", required=True, help="input .ply, .pcd, or .npz point cloud")
    parser.add_argument("--npz-out", help="output compressed .npz with key 'points'")
    parser.add_argument("--pcd-out", help="output .pcd")
    parser.add_argument("--ply-out", help="output binary little-endian .ply")
    parser.add_argument(
        "--pcd-encoding",
        choices=("binary", "ascii"),
        default="binary",
        help="PCD DATA encoding (binary; use ascii only for text interoperability)",
    )
    parser.add_argument(
        "--center-xy",
        action="store_true",
        help="translate the XY bounding-box midpoint to the origin; leave Z unchanged",
    )
    parser.add_argument("--ground-z", action="store_true", help="translate the minimum Z coordinate to zero")
    parser.add_argument(
        "--offset",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        default=(0.0, 0.0, 0.0),
        help="XYZ translation in metres, applied after automatic centering and grounding",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = parser.parse_args()
    if args.npz_out is None and args.pcd_out is None and args.ply_out is None:
        parser.error("at least one of --npz-out, --pcd-out, or --ply-out is required")

    points = convert_point_cloud(
        args.input_path,
        npz_out=args.npz_out,
        pcd_out=args.pcd_out,
        ply_out=args.ply_out,
        center_xy=args.center_xy,
        ground_z=args.ground_z,
        offset=tuple(args.offset),
        pcd_encoding=args.pcd_encoding,
        force=args.force,
    )
    print(
        f"[pcl_convert] {args.input_path}: {points.shape[0]} points "
        f"x[{points[:, 0].min():.3f},{points[:, 0].max():.3f}] "
        f"y[{points[:, 1].min():.3f},{points[:, 1].max():.3f}] "
        f"z[{points[:, 2].min():.3f},{points[:, 2].max():.3f}]"
    )
    if args.npz_out is not None:
        print(f"[pcl_convert] wrote NPZ -> {Path(args.npz_out).expanduser()}")
    if args.pcd_out is not None:
        print(f"[pcl_convert] wrote PCD ({args.pcd_encoding}) -> {Path(args.pcd_out).expanduser()}")
    if args.ply_out is not None:
        print(f"[pcl_convert] wrote PLY -> {Path(args.ply_out).expanduser()}")


if __name__ == "__main__":
    main()
