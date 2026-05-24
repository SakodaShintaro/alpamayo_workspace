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

export CARLA_ROOT=${CARLA_ROOT:-$HOME/CARLA_0.9.16}

if pgrep -f CarlaUE4 > /dev/null; then
    echo "ERROR: CARLA is already running. Kill it first: pkill -f CarlaUE4 && sleep 5" >&2
    exit 1
fi

export PYTHONPATH=${PYTHONPATH:-}
export PYTHONPATH=${CARLA_ROOT}/PythonAPI:${PYTHONPATH}
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}
export PYTHONPATH=${B2D_ROOT}/leaderboard:${PYTHONPATH}
export PYTHONPATH=${B2D_ROOT}/leaderboard/team_code:${PYTHONPATH}
export PYTHONPATH=${B2D_ROOT}/scenario_runner:${PYTHONPATH}
export SCENARIO_RUNNER_ROOT=${B2D_ROOT}/scenario_runner
export IS_BENCH2DRIVE=True

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${REPO_ROOT}/results/b2d_npc_${TIMESTAMP}
mkdir -p "${OUTPUT_DIR}"
export SAVE_PATH=${OUTPUT_DIR}/

cd "${B2D_ROOT}"
CUDA_VISIBLE_DEVICES=0 "${REPO_ROOT}/.venv/bin/python" \
  ${B2D_ROOT}/leaderboard/leaderboard/leaderboard_evaluator.py \
  --routes=${ROUTES} \
  --repetitions=1 \
  --track=SENSORS \
  --checkpoint=${OUTPUT_DIR}/eval.json \
  --agent=${B2D_ROOT}/leaderboard/leaderboard/autoagents/npc_agent.py \
  --agent-config=/dev/null \
  --debug=0 \
  --resume=True \
  --port=30000 \
  --traffic-manager-port=50000 \
  --gpu-rank=0
