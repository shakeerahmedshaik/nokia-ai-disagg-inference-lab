#!/usr/bin/env bash
# Spine-link failure in the middle of a KV transfer.
# Both senders are paced at PACE (default 100 Mbit/s, the nearly loss-free regime, see sweeps). REQUESTS
# back-to-back requests run. The leaf1 uplink that carries the QPs is learned from transfers 1 and 2 (the QPs, and
# so their ECMP path, stay the same for the whole run). When transfer START_AT begins (detected from the prefill
# eth1 TX rate) plus FAIL_DELAY seconds, that uplink is set admin-state disable through an already-open sr_cli session (a fresh sr_cli
# takes about 2.8 s, longer than the transfer). It is enabled again DOWN_FOR seconds later.
# The RC QPs lose in-flight frames, time out, and go-back-N retransmit once BGP has withdrawn the path and ECMP
# has re-hashed the flows onto the other spine. Every request must still give the same tokens and KV crc32.
set -euo pipefail

LAB=clab-ai-inference
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
REQUESTS=${REQUESTS:-5}
TOKENS=${TOKENS:-1024}
START_AT=${START_AT:-3}
DOWN_FOR=${DOWN_FOR:-12}
PACE=${PACE:-100mbit}
FAIL_DELAY=${FAIL_DELAY:-1.5}   # seconds into the transfer before the link is disabled

txb() { docker exec "$LAB-prefill" cat /sys/class/net/eth1/statistics/tx_bytes; }
uplinks() {   # both counters in parallel (each sr_cli start-up costs about 2.8 s)
  local pa pb
  docker exec "$LAB-leaf1" sr_cli -d "info from state interface ethernet-1/1 statistics out-octets" | awk '{print $2}' > /tmp/up1 & pa=$!
  docker exec "$LAB-leaf1" sr_cli -d "info from state interface ethernet-1/2 statistics out-octets" | awk '{print $2}' > /tmp/up2 & pb=$!
  wait $pa $pb
  echo "$(cat /tmp/up1) $(cat /tmp/up2)"
}
route() { docker exec "$LAB-leaf1" sr_cli -d "show network-instance default ipv4 route" | grep -A1 '^10.10.2.0/24'; }
stamp() { echo "$(date +%T.%3N) $*"; }

# One persistent sr_cli session to leaf1, so the commit lands within a fraction of a second.
exec 3> >(docker exec -i "$LAB-leaf1" sr_cli -e > /tmp/failover-sess.out 2>&1)
link() { printf 'enter candidate\nset / interface %s admin-state %s\ncommit now\n' "$1" "$2" >&3; stamp "leaf1 $1 admin-state $2 (sent)"; }
sleep 3

bash scripts/pace.sh "$PACE" >/dev/null
bash scripts/rxe-stats.sh > outputs/failover-rxe-before.txt
bash run.sh infer.py --prompt-tokens "$TOKENS" --max-new-tokens 8 --requests "$REQUESTS" --wait > /tmp/failover-run.txt &
run=$!

bursts=0 active=0 prev=$(txb) victim="" snap=""
while kill -0 $run 2>/dev/null; do
  sleep 0.2
  cur=$(txb)
  if [ $((cur - prev)) -gt 600000 ]; then   # about 3 MB/s or more
    if [ "$active" = 0 ]; then
      bursts=$((bursts + 1)); stamp "KV transfer #$bursts started"
      if [ "$bursts" -eq "$START_AT" ]; then
        sleep "$FAIL_DELAY"
        link "$victim" disable
        break
      fi
    fi
    active=1
  else
    if [ "$active" = 1 ]; then               # a transfer just ended: read the uplink counters
      stamp "KV transfer #$bursts ended"
      now=$(uplinks)
      if [ -n "$snap" ]; then
        read -r a1 a2 <<<"$snap"; read -r b1 a2b <<<"$now"
        if [ $((b1 - a1)) -ge $((a2b - a2)) ]; then victim=ethernet-1/1; else victim=ethernet-1/2; fi
        stamp "uplink bytes during transfer #$bursts: ethernet-1/1=$((b1 - a1)) ethernet-1/2=$((a2b - a2)) -> victim $victim"
      fi
      snap=$now
    fi
    active=0
    # snapshot taken at the end of transfer 1 serves as baseline for transfer 2
  fi
  prev=$cur
done
[ -n "$victim" ] || { echo "no victim learned"; exit 1; }
sleep 3
route || true
sleep $((DOWN_FOR - 3))
link "$victim" enable
wait $run
echo "== waiting for ECMP to return to 2 next-hops"
t0=$(date +%s)
for i in $(seq 1 60); do
  n=$(route | grep -cE '192\.168\.1[12]\.1' || true)
  [ "$n" -ge 2 ] && { stamp "2 next-hops again after $(( $(date +%s) - t0 )) s of polling"; break; }
  sleep 2
done
route
exec 3>&-
bash scripts/rxe-stats.sh > outputs/failover-rxe-after.txt
bash scripts/pace.sh off >/dev/null

echo "== per request (decode: kv_ready_ms, ttft_ms | prefill: prefill_ms, exposed_transfer_ms) | kv_crc32"
for r in $(seq 0 $((REQUESTS - 1))); do
  d=$(grep -E "role=decode req=$r prompt" /tmp/failover-run.txt)
  p=$(grep -E "role=prefill req=$r prompt" /tmp/failover-run.txt)
  printf 'req=%s kv_ready_ms=%s ttft_ms=%s prefill_ms=%s exposed_transfer_ms=%s kv_crc32=%s\n' "$r" \
    "$(grep -oP 'kv_ready_ms=\K\S+' <<<"$d")" "$(grep -oP ' ttft_ms=\K\S+' <<<"$d")" \
    "$(grep -oP 'prefill_ms=\K\S+' <<<"$p")" "$(grep -oP 'exposed_transfer_ms=\K\S+' <<<"$p")" \
    "$(grep -oP 'kv_crc32=\K\S+' <<<"$d")"
done
echo "== distinct generated_ids across requests (1 = every request produced identical tokens)"
grep -oP 'role=decode req=[0-9]+ generated_ids=\K.*' /tmp/failover-run.txt | sort -u | wc -l
echo "== distinct kv_crc32 across requests (1 = KV bytes identical every time)"
grep -oP 'role=decode req=[0-9]+ prompt.* kv_crc32=\K\S+' /tmp/failover-run.txt | sort -u | wc -l
