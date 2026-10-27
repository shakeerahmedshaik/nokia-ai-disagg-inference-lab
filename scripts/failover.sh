#!/usr/bin/env bash
# Spine-link failure in the middle of a KV transfer.
# Both senders are paced at 100 Mbit/s (the lossless-ish regime, see sweeps). A run of REQUESTS back-to-back
# requests starts; at the START_AT-th KV transfer (detected from prefill eth1 TX rate) the leaf1 uplink that is
# carrying the most bytes is set admin-state disable, and it is enabled again DOWN_FOR seconds later.
# The RC QPs lose the in-flight frames, time out, and go-back-N retransmit once BGP has withdrawn the path and
# ECMP re-hashes the flows onto the other spine. Every request must still produce the same tokens and KV crc32.
set -euo pipefail

LAB=clab-ai-inference
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
REQUESTS=${REQUESTS:-6}
TOKENS=${TOKENS:-512}
START_AT=${START_AT:-3}
DOWN_FOR=${DOWN_FOR:-12}

txb() { docker exec "$LAB-prefill" cat /sys/class/net/eth1/statistics/tx_bytes; }
outb() { docker exec "$LAB-leaf1" sr_cli -d "info from state interface $1 statistics out-octets" | awk '{print $2}'; }
link() {
  printf 'set / interface %s admin-state %s\ncommit now\n' "$1" "$2" | docker exec -i "$LAB-leaf1" sr_cli -e >/dev/null
  echo "$(date +%T.%3N) leaf1 $1 admin-state $2"
}

bash scripts/pace.sh 100mbit >/dev/null
bash scripts/rxe-stats.sh > outputs/failover-rxe-before.txt
bash run.sh infer.py --prompt-tokens "$TOKENS" --max-new-tokens 8 --requests "$REQUESTS" --wait > /tmp/failover-run.txt &
run=$!

# Wait for the START_AT-th transfer burst: prefill eth1 TX above 1.5 MB per 0.25 s.
bursts=0 active=0 prev=$(txb)
while [ "$bursts" -lt "$START_AT" ]; do
  sleep 0.25
  cur=$(txb)
  if [ $((cur - prev)) -gt 1500000 ]; then
    [ "$active" = 0 ] && bursts=$((bursts + 1)) && echo "$(date +%T.%3N) KV transfer #$bursts started"
    active=1
  else
    active=0
  fi
  prev=$cur
  kill -0 $run 2>/dev/null || { echo "run ended before transfer #$START_AT"; exit 1; }
done
sleep 0.5
# Pick the leaf1 uplink carrying the flow(s) now.
a1=$(outb ethernet-1/1); a2=$(outb ethernet-1/2); sleep 0.5
b1=$(outb ethernet-1/1); b2=$(outb ethernet-1/2)
if [ $((b1 - a1)) -ge $((b2 - a2)) ]; then victim=ethernet-1/1; other=ethernet-1/2; else victim=ethernet-1/2; other=ethernet-1/1; fi
echo "uplink bytes in 0.5 s: ethernet-1/1=$((b1 - a1)) ethernet-1/2=$((b2 - a2)) -> failing $victim"
link "$victim" disable
sleep 3
docker exec "$LAB-leaf1" sr_cli -d "show network-instance default ipv4 route" | grep -A1 '^10.10.2.0/24'
sleep $((DOWN_FOR - 3))
link "$victim" enable
wait $run
sleep 5
echo "== route after recovery"
docker exec "$LAB-leaf1" sr_cli -d "show network-instance default ipv4 route" | grep -A1 '^10.10.2.0/24'
bash scripts/rxe-stats.sh > outputs/failover-rxe-after.txt
bash scripts/pace.sh off >/dev/null

echo "== per request (decode view)"
grep -E 'role=decode req=[0-9]+ prompt' /tmp/failover-run.txt |
  grep -oP 'req=\K[0-9]+| ttft_ms=\K[0-9.]+| kv_ready_ms=\K[0-9.]+| kv_crc32=\K\S+' | paste - - - - | column -t
echo "== distinct generated_ids across requests (1 means every request gave identical tokens)"
grep -oP 'role=decode req=[0-9]+ generated_ids=\K.*' /tmp/failover-run.txt | sort -u | wc -l
echo "== distinct kv_crc32 across requests"
grep -oP 'role=decode req=[0-9]+ prompt.* kv_crc32=\K\S+' /tmp/failover-run.txt | sort -u | wc -l
