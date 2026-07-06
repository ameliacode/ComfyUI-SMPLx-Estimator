#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# dev_refresh.sh — develop → refresh test loop for the comfyui-smplx-estimator node.
#
# Run this every time you finish a development change. It:
#   1. (optional) runs pytest if a test suite + pytest are available
#   2. restarts the tester ComfyUI at ~/github/ComfyUI
#   3. waits for the server to come up
#   4. verifies the comfyui-smplx-estimator node imported cleanly (no IMPORT FAILED)
#   5. confirms ClickPose / MotionAGFormer / 3D Pose Editor are registered
#      via the /object_info API (authoritative)
#
# Exit 0 = node reloaded cleanly and is queryable. Non-zero = something broke
# (the relevant log excerpt is printed).
#
# NOTE: Python changes need this restart. JS-only changes (js/, web/) only need
#       a browser hard-refresh (Ctrl+Shift+R) — but a restart re-serves them too,
#       so running this is always safe.
#
# Usage:
#   scripts/dev_refresh.sh            # restart + verify
#   scripts/dev_refresh.sh --tests    # also run pytest first (skips if absent)
#   scripts/dev_refresh.sh --no-start # only verify a server that's already up
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

COMFY_DIR="/home/wswg3/github/ComfyUI"
PY="$COMFY_DIR/venv/bin/python3.10"
LOG="/tmp/comfyui.log"
REPO="/home/wswg3/project/comfyui-smplx-estimator"
HOST="127.0.0.1"
PORT="8188"
NODE_FOLDER="comfyui-smplx-estimator"
EXPECTED_NODES=("LoadSMPLX" "LoadNLF" "LoadMultiHMR" "LoadWiLoR" "NLFSMPLXEstimator" "MultiHMREstimator" "WiLoRHandEstimator" "SMPLXEditor" "ExportMesh")
READY_TIMEOUT=90   # seconds to wait for the server to answer

RUN_TESTS=0
DO_START=1
for arg in "$@"; do
  case "$arg" in
    --tests)    RUN_TESTS=1 ;;
    --no-start) DO_START=0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

c_red()  { printf '\033[31m%s\033[0m\n' "$*"; }
c_grn()  { printf '\033[32m%s\033[0m\n' "$*"; }
c_ylw()  { printf '\033[33m%s\033[0m\n' "$*"; }
step()   { printf '\n\033[1m▶ %s\033[0m\n' "$*"; }

fail() { c_red "✘ FAIL: $*"; echo "── last 30 log lines ─────────────────────────"; tail -30 "$LOG" 2>/dev/null; exit 1; }

# ── 1. optional pytest ───────────────────────────────────────────────────────
if [ "$RUN_TESTS" -eq 1 ]; then
  step "Running pytest"
  if "$PY" -c "import pytest" 2>/dev/null && \
     { ls "$REPO"/tests/test_*.py "$REPO"/test_*.py >/dev/null 2>&1; }; then
    ( cd "$REPO" && "$PY" -m pytest -q ) || fail "pytest failed — not restarting ComfyUI"
    c_grn "✓ tests passed"
  else
    c_ylw "• no pytest and/or no tests found — skipping (nothing to run yet)"
  fi
fi

# ── 2. restart ComfyUI ───────────────────────────────────────────────────────
if [ "$DO_START" -eq 1 ]; then
  step "Restarting tester ComfyUI"
  if pgrep -u "$USER" -f 'venv/bin/python3.10 main.py' >/dev/null 2>&1; then
    pkill -u "$USER" -f 'venv/bin/python3.10 main.py'
    echo "• stopped previous instance"
    for _ in $(seq 1 10); do
      pgrep -u "$USER" -f 'venv/bin/python3.10 main.py' >/dev/null 2>&1 || break
      sleep 0.5
    done
  else
    echo "• no previous instance running"
  fi

  cd "$COMFY_DIR" || fail "ComfyUI dir not found: $COMFY_DIR"
  : > "$LOG"   # truncate so we only read this run's output
  # Default to GPU 0 (standard). To run on another GPU (e.g. if 0 is busy):
  #   CUDA_VISIBLE_DEVICES=1 ./scripts/dev_refresh.sh
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  echo "• CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  nohup "$PY" main.py --listen 0.0.0.0 --port "$PORT" > "$LOG" 2>&1 &
  echo "• launched (pid $!) → $LOG"
fi

# ── 3. wait for the server to answer ─────────────────────────────────────────
step "Waiting for server on $HOST:$PORT (≤${READY_TIMEOUT}s)"
ready=0
for _ in $(seq 1 "$READY_TIMEOUT"); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://$HOST:$PORT/system_stats" 2>/dev/null || true)
  if [ "$code" = "200" ]; then ready=1; break; fi
  # bail early if the process died during import
  if [ "$DO_START" -eq 1 ] && ! pgrep -u "$USER" -f 'venv/bin/python3.10 main.py' >/dev/null 2>&1; then
    fail "ComfyUI process exited during startup"
  fi
  sleep 1
done
[ "$ready" -eq 1 ] || fail "server did not become ready within ${READY_TIMEOUT}s"
c_grn "✓ server is up"

# ── 4. check the node imported without failure ───────────────────────────────
# Only flag problems inside OUR node — other custom nodes' tracebacks (xatlas,
# comfy_dynamic_widgets, …) are not our concern and must not trip the loop.
step "Checking custom-node import"
if grep -iq "$NODE_FOLDER.*IMPORT FAILED\|IMPORT FAILED.*$NODE_FOLDER" "$LOG"; then
  fail "$NODE_FOLDER reported IMPORT FAILED"
fi
# A real failure in our code leaves a traceback frame pointing at a .py under
# custom_nodes/comfyui-smplx-estimator/ (or our package modules).
if grep -qE 'File ".*custom_nodes/'"$NODE_FOLDER"'/.*\.py"' "$LOG"; then
  c_ylw "• a traceback frame points inside the node — inspect:"
  grep -nE 'File ".*custom_nodes/'"$NODE_FOLDER"'/.*\.py"|Error|Traceback' "$LOG" | tail -15
  fail "traceback originating in $NODE_FOLDER"
fi
c_grn "✓ no import failure inside the node"

# ── 5. confirm nodes are registered via /object_info ─────────────────────────
step "Verifying node registration via /object_info"
info=$(curl -s "http://$HOST:$PORT/object_info" 2>/dev/null)
[ -n "$info" ] || fail "/object_info returned nothing"

missing=()
for n in "${EXPECTED_NODES[@]}"; do
  if echo "$info" | jq -e --arg k "$n" 'has($k)' >/dev/null 2>&1; then
    echo "  ✓ $n"
  else
    echo "  ✗ $n  (NOT registered)"
    missing+=("$n")
  fi
done
[ "${#missing[@]}" -eq 0 ] || fail "nodes not registered: ${missing[*]}"

step "Import timing (for reference)"
grep -i "$NODE_FOLDER\|comfyui-smplx-estimator" "$LOG" | grep -i "second\|import" | tail -3 || true

c_grn "✔ ALL GOOD — node reloaded cleanly and all ${#EXPECTED_NODES[@]} nodes are registered."
echo "  UI: http://$HOST:$PORT/   ·   JS changes: browser hard-refresh (Ctrl+Shift+R)"
exit 0
