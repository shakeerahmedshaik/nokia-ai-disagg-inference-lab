#!/usr/bin/env bash
# Usage: bash run.sh infer.py [args...] [--wait]            disaggregated: prefill on `prefill`, decode on `decode`
#        bash run.sh infer.py --role colocated [args...]    reference: one process on `prefill`, no network
# Logs go to /tmp/<script>.log in each worker. The store (TCP bootstrap) is hosted by the prefill worker.
# With --wait the run is supervised: if one side exits while the other keeps waiting (a crashed or wedged peer),
# or the run exceeds RUN_TIMEOUT seconds, both are killed, the logs are printed to stderr, and the run is retried
# once on a different store port. python runs with -X faulthandler, so a native crash leaves a traceback.
set -euo pipefail

LAB=clab-ai-inference
script=$1; shift
wait_done=0
colocated=0
args=()
skip=0
for a in "$@"; do
  if [ "$skip" = 1 ]; then skip=0; [ "$a" = colocated ] && colocated=1; args+=("$a"); continue; fi
  case "$a" in
    --wait) wait_done=1 ;;
    --role) skip=1; args+=("$a") ;;
    *) args+=("$a") ;;
  esac
done
name=$(basename "$script" .py)
RUN_TIMEOUT=${RUN_TIMEOUT:-900}

if [ "$colocated" = 1 ]; then nodes=(prefill); else nodes=(decode prefill); fi

alive() { docker exec "$LAB-$1" pgrep -f "/lab/app/$script" >/dev/null 2>&1; }
killall_nodes() { for w in prefill decode; do docker exec "$LAB-$w" pkill -f "/lab/app/$script" >/dev/null 2>&1 || true; done; }

start() {
  local port=$1 w dev role_arg
  local common=(-e MASTER_ADDR=10.10.1.2 -e MASTER_PORT="$port" -e TORCH_THREADS="${TORCH_THREADS:-4}")
  for w in "${nodes[@]}"; do
    dev=rxe_p; [ "$w" = decode ] && dev=rxe_d
    role_arg=(--role "$w"); [ "$colocated" = 1 ] && role_arg=()
    docker exec -d "${common[@]}" -e ROCE_DEV=$dev "$LAB-$w" \
      sh -c "exec /usr/bin/python3 -X faulthandler -u /lab/app/$script ${role_arg[*]:-} ${args[*]:-} > /tmp/$name.log 2>&1"
  done
}

# Returns 0 when every node finished, 1 on a hang or timeout.
supervise() {
  local t0=$SECONDS lone=0 w up
  sleep 2
  while :; do
    up=0
    for w in "${nodes[@]}"; do alive "$w" && up=$((up + 1)) || true; done
    [ "$up" -eq 0 ] && return 0
    if [ "$up" -lt "${#nodes[@]}" ]; then lone=$((lone + 1)); else lone=0; fi
    [ "$lone" -ge 30 ] && { echo "run.sh: one side exited and the other has been waiting for 30 s" >&2; return 1; }
    [ $((SECONDS - t0)) -ge "$RUN_TIMEOUT" ] && { echo "run.sh: timeout after ${RUN_TIMEOUT}s" >&2; return 1; }
    sleep 1
  done
}

show() { local w; for w in "${nodes[@]}"; do echo "===== $w /tmp/$name.log"; docker exec "$LAB-$w" cat "/tmp/$name.log"; done; }

if [ "$wait_done" = 1 ]; then
  for attempt in 1 2; do
    start $((29500 + attempt - 1))
    if supervise; then break; fi
    echo "run.sh: attempt $attempt failed; logs follow" >&2
    show >&2
    killall_nodes
    [ "$attempt" = 2 ] && { echo "run.sh: giving up" >&2; exit 1; }
    sleep 3
  done
else
  start 29500
  sleep 3
fi
show
