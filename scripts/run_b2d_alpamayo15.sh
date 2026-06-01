#!/bin/bash
# Bench2Drive eval with the Alpamayo 1.5 team_code agent.
# Manages CARLA server lifecycle (start + cleanup on exit) via leaderboard_evaluator.
#
# Usage:
#   scripts/run_b2d_alpamayo15.sh [routes_xml]
#
# Defaults to route_25865_overtake.xml; pass bench2drive220.xml etc. to override.
set -eux

clear

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${REPO_ROOT}"

B2D_ROOT=${REPO_ROOT}/Bench2Drive
ROUTES=$(readlink -f "${1:-${B2D_ROOT}/leaderboard/data/route_25865_overtake.xml}")

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
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH}
export SCENARIO_RUNNER_ROOT=${B2D_ROOT}/scenario_runner
export IS_BENCH2DRIVE=True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ALPAMAYO_FRONT_ONLY=0

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${REPO_ROOT}/results/b2d_alpamayo15_${TIMESTAMP}
mkdir -p "${OUTPUT_DIR}"
export SAVE_PATH=${OUTPUT_DIR}/

cd "${B2D_ROOT}"
CUDA_VISIBLE_DEVICES=0 "${REPO_ROOT}/.venv/bin/python" \
  ${B2D_ROOT}/leaderboard/leaderboard/leaderboard_evaluator.py \
  --routes=${ROUTES} \
  --repetitions=1 \
  --track=SENSORS \
  --checkpoint=${OUTPUT_DIR}/eval.json \
  --agent=${REPO_ROOT}/team_code/alpamayo15_agent.py \
  --agent-config=/dev/null \
  --debug=0 \
  --resume=True \
  --port=30000 \
  --traffic-manager-port=50000 \
  --gpu-rank=0 2>&1 | tee "${OUTPUT_DIR}/run.log"

for spec_dir in "${OUTPUT_DIR}"/*/spectator; do
    [ -d "${spec_dir}" ] || continue
    scenario_dir=$(dirname "${spec_dir}")
    file_list=$(find "${spec_dir}" -name "*.png" | sort -V)
    [ -z "${file_list}" ] && continue
    ffmpeg -y -r 10 \
           -f concat -safe 0 -i <(printf "file '%s'\n" ${file_list}) \
           -vcodec libx264 \
           -pix_fmt yuv420p \
           -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
           -r 10 \
           "${scenario_dir}/spectator.mp4"
done

cat ${OUTPUT_DIR}/eval.json
