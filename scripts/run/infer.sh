#!/usr/bin/env bash
set -euo pipefail

# Launch PerfectDrone, random goals, and the current GP-SR ROS wrapper.
# SUPER's fsm_node must remain stopped because infer_node.py owns /planning/pos_cmd.

export PLANNER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PLANNER_MAP="${PLANNER_MAP:-$PLANNER_ROOT/output/maps/mixed_forest_200_dense1800_seed1.npz}"
export SUPER_WS="${SUPER_WS:-$HOME/code/planner_ws/super_ws}"
export PLANNER_SIM_CONFIG="${PLANNER_SIM_CONFIG:-$PLANNER_ROOT/scripts/data_collect/super_config/nmp_mid360.yaml}"
export PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-}"
export PLANNER_DEVICE="${PLANNER_DEVICE:-cuda}"
export PLANNER_COMMAND_RATE="${PLANNER_COMMAND_RATE:-100}"
cd "$PLANNER_ROOT"

if [[ -z "$PLANNER_CHECKPOINT" ]]; then
    echo "Set PLANNER_CHECKPOINT=/path/to/current-gp-sr-checkpoint.pth" >&2
    exit 1
fi
if [[ ! -f "$PLANNER_CHECKPOINT" ]]; then
    echo "Current GP-SR checkpoint not found: $PLANNER_CHECKPOINT" >&2
    exit 1
fi
if [[ ! -f "$PLANNER_MAP" ]]; then
    echo "Goal-sampling map not found: $PLANNER_MAP" >&2
    echo "Set PLANNER_MAP=/path/to/map.npz" >&2
    exit 1
fi
if [[ ! -f "${PLANNER_MAP%.npz}.json" ]]; then
    echo "Goal-sampling metadata not found: ${PLANNER_MAP%.npz}.json" >&2
    echo "Keep the matching pcl_generate.py JSON sidecar." >&2
    exit 1
fi
if [[ ! -f "$SUPER_WS/devel/setup.bash" ]]; then
    echo "SUPER workspace setup not found: $SUPER_WS/devel/setup.bash" >&2
    exit 1
fi

sim_cmd='
    source /opt/ros/noetic/setup.bash
    source "$SUPER_WS/devel/setup.bash"
    roslaunch "$PLANNER_ROOT/scripts/data_collect/super_config/nmp_infer.launch" \
        sim_config_path:="$PLANNER_SIM_CONFIG"
    exec bash'

goals_cmd='
    source /opt/ros/noetic/setup.bash
    conda activate planner
    source "$SUPER_WS/devel/setup.bash"
    cd "$PLANNER_ROOT"
    export PYTHONPATH="$PLANNER_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    python3 scripts/data_collect/goals.py \
        --map "$PLANNER_MAP" \
        --completion-source odom
    exec bash'

infer_cmd='
    source /opt/ros/noetic/setup.bash
    conda activate planner
    source "$SUPER_WS/devel/setup.bash"
    cd "$PLANNER_ROOT"
    export PYTHONPATH="$PLANNER_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    python3 scripts/infer_node.py \
        --checkpoint "$PLANNER_CHECKPOINT" \
        --device "$PLANNER_DEVICE" \
        --execute \
        --command-rate "$PLANNER_COMMAND_RATE" \
        --no-lidar-viz
    exec bash'

gnome-terminal \
    --tab --title=sim -e "bash -ilc '$sim_cmd'" \
    --tab --title=goals -e "bash -ilc '$goals_cmd'" \
    --tab --title=infer -e "bash -ilc '$infer_cmd'" 2>/dev/null
