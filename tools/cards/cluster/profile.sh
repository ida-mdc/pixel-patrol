#!/usr/bin/env bash
# Profile one manifest dataset with PixelPatrol.
#   local test:  ./profile.sh <id> sample    fetches 3 member files, runs pixel-patrol locally
#   cluster:     sbatch profile.sh <id> full  fetches everything, runs pixel-patrol via pixel-patrol-slurm
# Env: PP_CARDS_HOME (only if you sbatch from somewhere other than the repo root), PP_CARDS_DATA (data root), PP_CARDS_OUT (parquet dir),
#      PP_SLURM_ARGS (default "--jobs 8 --cores 4 --memory 16GB --walltime 12:00:00", add --partition=... here)
#SBATCH --job-name=pp-card
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=pp-card-%j.log
set -euo pipefail
ID=${1:?dataset id}; MODE=${2:?sample|full}
# sbatch runs a copy of this script from the Slurm spool dir, so locate our siblings explicitly there.
if [[ -n ${SLURM_JOB_ID:-} ]]; then
  HERE=${PP_CARDS_HOME:-${SLURM_SUBMIT_DIR:?}/tools/cards/cluster}   # submit from the repo root, or set PP_CARDS_HOME
else
  HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi
[[ -f "$HERE/prepare.py" ]] || { echo "prepare.py not found in $HERE - set PP_CARDS_HOME to tools/cards/cluster" >&2; exit 2; }
export PP_CARDS_DATA=${PP_CARDS_DATA:-$HERE/data}
OUT=${PP_CARDS_OUT:-$HERE/out}; mkdir -p "$OUT"
SRC="$PP_CARDS_DATA/$ID/$MODE"

if [[ ! -d "$SRC" ]]; then
  if [[ $MODE == sample ]]; then python3 "$HERE/prepare.py" "$ID" --sample 3; else python3 "$HERE/prepare.py" "$ID"; fi
fi
eval "PP_ARGS=($(python3 "$HERE/prepare.py" "$ID" --pp-args))"
PP_ARGS+=(--name "$ID" -o "$OUT/$ID-$MODE.parquet" --log-file)

if [[ $MODE == sample ]]; then
  pixel-patrol process "$SRC" "${PP_ARGS[@]}" --max-workers 2
else
  # shellcheck disable=SC2086
  pixel-patrol-slurm ${PP_SLURM_ARGS:---jobs 8 --cores 4 --memory 16GB --walltime 12:00:00} -- "$SRC" "${PP_ARGS[@]}"
fi
