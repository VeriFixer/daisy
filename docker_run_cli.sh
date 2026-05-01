#!/usr/bin/env bash
set -euo pipefail
# docker_run_cli.sh - Run a local Dafny `.dfy` file through the Daisy CLI inside Docker
# Usage:
#   ./docker_run_cli.sh /path/to/file.dfy [options] [-- additional src.cli args]
# Options:
#   -m MODEL         model name (default: openrouter-free)
#   -l LOCALIZATION  localization strategy (default: LLM)
#   -r RESULTS_DIR   host path to store results (default: ./results)
#   -i IMAGE         docker image name (default: daisy:latest)
#   -b               build image if missing
#   -o OUTFILE       write stdout to this host file (optional)

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGE="daisy:latest"
BUILD_IF_MISSING=0
MODEL="openrouter-free"
LOCALIZATION="LLM"
RESULTS_DIR="$REPO_ROOT/results"
OUTFILE=""

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/file.dfy [options] [-- extra args to src.cli]"
  exit 2
fi

FILE="$1"; shift

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m) MODEL="$2"; shift 2;;
    -l) LOCALIZATION="$2"; shift 2;;
    -r) RESULTS_DIR="$2"; shift 2;;
    -i) IMAGE="$2"; shift 2;;
    -b) BUILD_IF_MISSING=1; shift;;
    -o) OUTFILE="$2"; shift 2;;
    --) shift; EXTRA_ARGS=("$@"); break;;
    *) echo "Unknown option: $1"; exit 2;;
  esac
done

# Ensure image exists (build if requested)
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  if [[ $BUILD_IF_MISSING -eq 1 ]]; then
    echo "Docker image $IMAGE not found — building..."
    docker build -t "$IMAGE" "$REPO_ROOT"
  else
    echo "Docker image $IMAGE not found. Re-run with -b to build it." >&2
    exit 3
  fi
fi

# Normalize paths
FILE_ABS="$(realpath -e "$FILE")" || { echo "File not found: $FILE" >&2; exit 2; }
RESULTS_DIR_ABS="$(mkdir -p "$RESULTS_DIR" && realpath "$RESULTS_DIR")"

# Determine container path for the input file
if [[ "$FILE_ABS" == "$REPO_ROOT"* ]]; then
  # file is inside repo mount → use /app/<relpath>
  RELPATH="${FILE_ABS#$REPO_ROOT/}"
  CONTAINER_FILE="/app/$RELPATH"
  EXTRA_VOLUME_ARGS=("-v" "$REPO_ROOT:/app:ro")
else
  # file outside repo → mount as /data/input.dfy
  CONTAINER_FILE="/data/input.dfy"
  EXTRA_VOLUME_ARGS=("-v" "$REPO_ROOT:/app:ro" "-v" "$FILE_ABS:/data/input.dfy:ro")
fi

# Always mount results dir (writable)
EXTRA_VOLUME_ARGS+=("-v" "$RESULTS_DIR_ABS:/app/results")

DOCKER_CMD=(docker run --rm -i -w /app)
for v in "${EXTRA_VOLUME_ARGS[@]}"; do DOCKER_CMD+=("$v"); done
DOCKER_CMD+=("$IMAGE" python -m src.cli "$CONTAINER_FILE" --model "$MODEL" --localization "$LOCALIZATION")
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  DOCKER_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Running: ${DOCKER_CMD[*]}"
if [[ -n "$OUTFILE" ]]; then
  mkdir -p "$(dirname "$OUTFILE")"
  "${DOCKER_CMD[@]}" | tee "$OUTFILE"
else
  "${DOCKER_CMD[@]}"
fi
