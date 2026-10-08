"""Protect the boundary between local collection code and SUPER integration files."""

import xml.etree.ElementTree as ET
from pathlib import Path


def test_data_collect_subdirectories_have_single_responsibilities():
    data_collect = Path(__file__).resolve().parents[1] / "scripts" / "data_collect"
    copied_files = {path.name for path in (data_collect / "super_config").iterdir() if path.is_file()}

    assert copied_files == {
        "nmp.yaml",
        "nmp_collect.launch",
        "nmp_infer.launch",
        "nmp_mid360.yaml",
        "super_mid360.patch",
        # Tall-forest scenes used for the reported dataset.
        "nmp_mid360_tall_forest_dense2000_seed42.yaml",
        "nmp_mid360_tall_forest_dense2000_spacing3p2_seed42.yaml",
        "nmp_mid360_tall_forest_dense3000_seed42.yaml",
        "nmp_tall_forest_teacher_v1.yaml",
    }
    assert not any((data_collect / name).exists() for name in ("config", "maps", "patches"))
    for local_name in ("clean.py", "collect.py", "goals.py", "validate.py"):
        assert (data_collect / local_name).is_file()
    assert {path.name for path in (data_collect / "collect_utils").iterdir() if path.is_file()} == {"sync.py"}
    assert {path.name for path in (data_collect / "pcl_tools").iterdir() if path.is_file()} == {
        "pcl_generate.py",
        "pcl_convert.py",
    }


def test_collection_defaults_use_matching_mixed_map_representations():
    root = Path(__file__).resolve().parents[1]
    scene_stem = "mixed_forest_200_dense1800_seed1"
    simulator_config = (root / "scripts" / "data_collect" / "super_config" / "nmp_mid360.yaml").read_text(
        encoding="utf-8"
    )
    simulator_launch = (root / "scripts" / "data_collect" / "super_config" / "nmp_collect.launch").read_text(
        encoding="utf-8"
    )
    launcher = (root / "scripts" / "run" / "collect.sh").read_text(encoding="utf-8")

    assert f'pcd_name: "{scene_stem}.pcd"' in simulator_config
    assert '<env name="__GL_SYNC_TO_VBLANK" value="0"/>' in simulator_launch
    assert '<env name="vblank_mode" value="0"/>' in simulator_launch
    assert f"output/maps/{scene_stem}.npz" in launcher
    assert "scripts/data_collect/super_config/nmp_mid360.yaml" in launcher
    assert "scripts/data_collect/super_config/nmp.yaml" in launcher
    assert "$SUPER_WS/src/SUPER/super_planner/config/nmp.yaml" not in launcher


def test_inference_global_cloud_height_filter_is_visualization_only():
    launch_path = Path(__file__).resolve().parents[1] / "scripts" / "data_collect" / "super_config" / "nmp_infer.launch"
    root = ET.parse(launch_path).getroot()

    args = {node.attrib["name"]: node.attrib.get("default") for node in root.findall("arg")}
    assert args["global_viz_z_max"] == "4.0"

    rviz_group = root.find("group[@if='$(arg rviz)']")
    assert rviz_group is not None
    filter_node = rviz_group.find("node[@name='global_cloud_visual_filter']")
    assert filter_node is not None
    assert filter_node.attrib["pkg"] == "nodelet"
    assert filter_node.attrib["args"] == "standalone pcl/PassThrough"

    filter_remaps = {node.attrib["from"]: node.attrib["to"] for node in filter_node.findall("remap")}
    assert filter_remaps == {"~input": "/global_pc", "~output": "/nmp/global_pc_visual"}
    filter_params = " ".join((filter_node.findtext("rosparam") or "").split())
    assert "filter_field_name: z" in filter_params
    assert "filter_limit_max: $(arg global_viz_z_max)" in filter_params

    rviz_nodes = rviz_group.findall("node[@pkg='rviz']")
    assert {node.attrib["name"] for node in rviz_nodes} == {"top_down", "fpv"}
    for rviz_node in rviz_nodes:
        remaps = {node.attrib["from"]: node.attrib["to"] for node in rviz_node.findall("remap")}
        assert remaps["/global_pc"] == "/nmp/global_pc_visual"
