#!/usr/bin/env bash
# Usage: scripts/rxe-stats.sh   prints Soft-RoCE transport counters and leaf ingress discards.
# duplicate_request / out_of_seq_request / rcvd_seq_err / completer_retry_err rise when frames are
# lost and RC go-back-N retransmission kicks in (no PFC in this fabric).
set -euo pipefail

LAB=clab-ai-inference
for w in prefill:rxe_p decode:rxe_d; do
  node=${w%%:*} dev=${w##*:}
  printf '%s %s ' "$node" "$dev"
  docker exec "$LAB-$node" rdma statistic show link "$dev/1" |
    grep -oE '(sent_pkts|rcvd_pkts|duplicate_request|out_of_seq_request|rcvd_seq_err|completer_retry_err|retry_exceeded_err) [0-9]+' |
    tr '\n' ' '
  echo
done
for n in leaf1 leaf2; do
  printf '%s ethernet-1/10 ' "$n"
  docker exec "$LAB-$n" sr_cli -d "info from state interface ethernet-1/10 statistics" |
    awk '/in-discarded-packets|in-error-packets/ {printf "%s=%s ", $1, $2} END {print ""}'
done
