#!/bin/bash
# Bench2Drive smoke test with the built-in NPC agent.
# Manages CARLA server lifecycle (start + cleanup on exit).
#
# Usage:
#   scripts/run_b2d_npc.sh [routes_xml]
#
# Defaults to dev10_single.xml; pass bench2drive220.xml etc. to override.
set -eux

clear

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${REPO_ROOT}"

B2D_ROOT=${REPO_ROOT}/Bench2Drive
ROUTES=$(readlink -f "${1:-${B2D_ROOT}/leaderboard/data/dev10_single.xml}")

VENV_PYTHON=${REPO_ROOT}/.venv/bin/python
export CARLA_ROOT=${CARLA_ROOT:-$HOME/CARLA_0.9.16}

if pgrep -f CarlaUE4 > /dev/null; then
    echo "ERROR: CARLA is already running. Kill it first: pkill -f CarlaUE4 && sleep 5" >&2
    exit 1
fi

export PORT=${PORT:-30000}
export TM_PORT=${TM_PORT:-50000}

export PYTHONPATH=${PYTHONPATH:-}
export PYTHONPATH=${CARLA_ROOT}/PythonAPI:${PYTHONPATH}
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}
export PYTHONPATH=${B2D_ROOT}/leaderboard:${PYTHONPATH}
export PYTHONPATH=${B2D_ROOT}/leaderboard/team_code:${PYTHONPATH}
export PYTHONPATH=${B2D_ROOT}/scenario_runner:${PYTHONPATH}
export SCENARIO_RUNNER_ROOT=${B2D_ROOT}/scenario_runner
export LEADERBOARD_ROOT=${B2D_ROOT}/leaderboard

TEAM_AGENT=${B2D_ROOT}/leaderboard/leaderboard/autoagents/npc_agent.py
TEAM_CONFIG=/dev/null

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${REPO_ROOT}/results/b2d_npc_${TIMESTAMP}
mkdir -p "${OUTPUT_DIR}"

CHECKPOINT_ENDPOINT=${OUTPUT_DIR}/eval.json
SAVE_PATH=${OUTPUT_DIR}/
export SAVE_PATH

export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0
export REPETITIONS=1
export RESUME=True
export IS_BENCH2DRIVE=True
export PLANNER_TYPE=only_traj
export GPU_RANK=${GPU_RANK:-0}

cd "${B2D_ROOT}"
CUDA_VISIBLE_DEVICES=${GPU_RANK} "${VENV_PYTHON}" \
  ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
  --routes=${ROUTES} \
  --repetitions=${REPETITIONS} \
  --track=${CHALLENGE_TRACK_CODENAME} \
  --checkpoint=${CHECKPOINT_ENDPOINT} \
  --agent=${TEAM_AGENT} \
  --agent-config=${TEAM_CONFIG} \
  --debug=${DEBUG_CHALLENGE} \
  --resume=${RESUME} \
  --port=${PORT} \
  --traffic-manager-port=${TM_PORT} \
  --gpu-rank=${GPU_RANK}
