#!/usr/bin/env bash
# Usage: bash run.sh infer.py [args...] [--wait]            disaggregated: prefill on `prefill`, decode on `decode`
#        bash run.sh infer.py --role colocated [args...]    reference: one process on `prefill`, no network
# Logs go to /tmp/<script>.log in each worker. The store (TCP bootstrap) is hosted by the prefill worker.
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

common=(-e MASTER_ADDR=10.10.1.2 -e MASTER_PORT=29500 -e TORCH_THREADS="${TORCH_THREADS:-4}")
if [ "$colocated" = 1 ]; then
  nodes=(prefill)
else
  nodes=(decode prefill)   # decode joins the store; start it first so its log is complete when prefill ends
fi
for w in "${nodes[@]}"; do
  dev=rxe_p; [ "$w" = decode ] && dev=rxe_d
  role_arg=(--role "$w"); [ "$colocated" = 1 ] && role_arg=()
  docker exec -d "${common[@]}" -e ROCE_DEV=$dev "$LAB-$w" \
    sh -c "exec /usr/bin/python3 -u /lab/app/$script ${role_arg[*]:-} ${args[*]:-} > /tmp/$name.log 2>&1"
done

if [ "$wait_done" = 1 ]; then
  sleep 2
  for w in "${nodes[@]}"; do
    while docker exec "$LAB-$w" pgrep -f "/lab/app/$script" >/dev/null; do sleep 1; done
  done
else
  sleep 3
fi

for w in "${nodes[@]}"; do
  echo "===== $w /tmp/$name.log"
  docker exec "$LAB-$w" cat "/tmp/$name.log"
done
