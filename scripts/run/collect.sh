#!/usr/bin/env bash
set -euo pipefail

# Collect complete goal-execution episodes: simulator + goals + episode collector.

export PLANNER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PLANNER_MAP="${PLANNER_MAP:-$PLANNER_ROOT/output/maps/mixed_forest_200_dense1800_seed1.npz}"
export SUPER_WS="${SUPER_WS:-$HOME/code/planner_ws/super_ws}"
# Read the versioned runtime parameters directly so the configured 6/6/12 command envelope
# is reproducible and cannot be silently replaced by a stale copy in the SUPER workspace.
export PLANNER_SIM_CONFIG="${PLANNER_SIM_CONFIG:-$PLANNER_ROOT/scripts/data_collect/super_config/nmp_mid360.yaml}"
export PLANNER_SUPER_CONFIG="${PLANNER_SUPER_CONFIG:-$PLANNER_ROOT/scripts/data_collect/super_config/nmp.yaml}"
export EPISODE_MAX="${EPISODE_MAX:-5000}"
export EPISODE_OUT="${EPISODE_OUT:-$PLANNER_ROOT/data}"
export EPISODE_TIMEOUT="${EPISODE_TIMEOUT:-30}"
export GOAL_TOLERANCE="${GOAL_TOLERANCE:-0.1}"
export GOAL_SPEED_TOLERANCE="${GOAL_SPEED_TOLERANCE:-0.2}"
export NEXT_GOAL_DELAY="${NEXT_GOAL_DELAY:-0.5}"
export MIN_LIDAR_RATE="${MIN_LIDAR_RATE:-9.5}"
export MAX_LIDAR_GAP="${MAX_LIDAR_GAP:-0.15}"
cd "$PLANNER_ROOT"

mkdir -p "$EPISODE_OUT"

sim_cmd='
    source /opt/ros/noetic/setup.bash
    source "$SUPER_WS/devel/setup.bash"
    roslaunch "$PLANNER_ROOT/scripts/data_collect/super_config/nmp_collect.launch" \
        sim_config_path:="$PLANNER_SIM_CONFIG" \
        planner_config_path:="$PLANNER_SUPER_CONFIG"
    exec bash'

goals_cmd='
    conda activate planner
    source /opt/ros/noetic/setup.bash
    source "$SUPER_WS/devel/setup.bash"
    cd "$PLANNER_ROOT"
    python3 scripts/data_collect/goals.py \
        --map "$PLANNER_MAP" \
        --max-successes "$EPISODE_MAX" \
        --next-goal-delay "$NEXT_GOAL_DELAY"
    exec bash'

collect_cmd='
    conda activate planner
    source /opt/ros/noetic/setup.bash
    source "$SUPER_WS/devel/setup.bash"
    cd "$PLANNER_ROOT"
    python3 scripts/data_collect/collect.py \
        --map "$PLANNER_MAP" \
        --out "$EPISODE_OUT" \
        --max-episodes "$EPISODE_MAX" \
        --timeout "$EPISODE_TIMEOUT" \
        --goal-tolerance "$GOAL_TOLERANCE" \
        --goal-speed-tolerance "$GOAL_SPEED_TOLERANCE" \
        --min-lidar-rate "$MIN_LIDAR_RATE" \
        --max-lidar-gap "$MAX_LIDAR_GAP"
    exec bash'

gnome-terminal \
    --tab --title=sim     -e "bash -ilc '$sim_cmd'" \
    --tab --title=goals   -e "bash -ilc '$goals_cmd'" \
    --tab --title=episode -e "bash -ilc '$collect_cmd'" 2>/dev/null
