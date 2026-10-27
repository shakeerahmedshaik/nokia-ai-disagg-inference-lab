#!/usr/bin/env bash
# One-variable-at-a-time sweeps. Each row is one disaggregated request (16 new tokens) with a header capture on
# prefill eth1, checked against a colocated reference of the same prompt length and KV dtype.
# Usage: scripts/sweep.sh [group...]   groups: tokens dtype channels mode pacing delay transport (default: all)
# Output: one summary row per run. Redirect to outputs/sweeps.txt.
set -euo pipefail

LAB=clab-ai-inference
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
BASE_TOKENS=${BASE_TOKENS:-512}
NEW=16
REPS=${REPS:-1}                       # repeat each row of the mode group (the host CPU is shared, so timings are noisy)
PACES=${PACES:-"off 100mbit 60mbit"}   # tbf rates for the pacing group

discards() {
  for n in leaf1 leaf2; do
    docker exec "$LAB-$n" sr_cli -d "info from state interface ethernet-1/10 statistics in-discarded-packets" | awk '{print $2}'
  done | awk '{s+=$1} END {print s}'
}

field() { grep -oP "$2=\K\S+" <<<"$1" | head -1; }

# Colocated reference ids for (tokens, dtype), cached for the sweep.
ref_ids() {
  local f=/tmp/sweep-ref-$1-$2.txt
  [ -s "$f" ] || bash run.sh infer.py --role colocated --prompt-tokens "$1" --max-new-tokens $NEW --kv-dtype "$2" --wait |
    grep -oP 'generated_ids=\K.*' | head -1 > "$f"
  cat "$f"
}

row() {
  local label=$1 tokens=$2 dtype=$3 transport=$4; shift 4
  local ref d0 d1 flt
  ref=$(ref_ids "$tokens" "$dtype")
  bash scripts/fabric.sh counters > /tmp/sweep-c0.txt
  d0=$(discards)
  bash scripts/capture.sh start sweep --snap 96 >/dev/null
  bash run.sh infer.py --prompt-tokens "$tokens" --max-new-tokens $NEW --kv-dtype "$dtype" --transport "$transport" "$@" --wait > /tmp/sweep-run.txt
  bash scripts/capture.sh stop sweep >/dev/null
  d1=$(discards)
  bash scripts/fabric.sh counters > /tmp/sweep-c1.txt
  local pf dc gen kvb frames bytes split
  pf=$(grep 'role=prefill req=0 prompt' /tmp/sweep-run.txt)
  dc=$(grep 'role=decode req=0 prompt' /tmp/sweep-run.txt)
  gen=$(grep -oP 'role=decode req=0 generated_ids=\K.*' /tmp/sweep-run.txt | head -1)
  kvb=$(field "$pf" kv_bytes)
  flt="udp.dstport==4791"; [ "$transport" = tcp ] && flt="tcp.port==29600"
  docker cp captures/sweep.pcap "$LAB-prefill:/tmp/sweep-a.pcap" >/dev/null
  read -r frames bytes < <(docker exec "$LAB-prefill" tshark -r /tmp/sweep-a.pcap -Y "$flt" -T fields -e frame.len 2>/dev/null |
    awk '{n++; s+=$1} END {print n+0, s+0}')
  split=$(python3 scripts/ecmp-delta.py /tmp/sweep-c0.txt /tmp/sweep-c1.txt |
    awk '$1=="spine1"&&$2=="ethernet-1/1"{a=$4} $1=="spine2"&&$2=="ethernet-1/1"{b=$4} END {printf "%s/%s", a, b}')
  local match=NO
  [ "$gen" = "$ref" ] && [ "$(field "$pf" kv_crc32)" = "$(field "$dc" kv_crc32)" ] && match=yes
  printf '%-16s kv_MB=%-6.1f wire_MB=%-7.1f wire/kv=%-5.2f prefill_ms=%-7.0f exposed_xfer_ms=%-7.0f ttft_ms=%-7.0f spine1/spine2_MB=%-11s leaf_discards=%-6s tokens_and_kv_match=%s\n' \
    "$label" "$(echo "$kvb/1000000" | bc -l)" "$(echo "$bytes/1000000" | bc -l)" \
    "$(echo "$bytes/$kvb" | bc -l)" "$(field "$pf" prefill_ms)" "$(field "$pf" exposed_transfer_ms)" \
    "$(field "$dc" ttft_ms)" "$split" "$((d1 - d0))" "$match"
}

groups=("$@"); [ ${#groups[@]} -gt 0 ] || groups=(tokens dtype channels mode pacing delay transport)
has() { local g; for g in "${groups[@]}"; do [ "$g" = "$1" ] && return 0; done; return 1; }

bash scripts/pace.sh off >/dev/null
bash scripts/netem.sh del >/dev/null

has tokens && { echo "## prompt length (fp32, 2 channels, bulk)"
  for t in 32 128 512 1024; do row "tokens=$t" "$t" fp32 roce; done; }
has dtype && { echo "## KV dtype (tokens=$BASE_TOKENS)"
  for d in fp32 fp16; do row "dtype=$d" "$BASE_TOKENS" "$d" roce; done; }
has channels && { echo "## RC QPs (channels) (tokens=$BASE_TOKENS)"
  for c in 1 2 4; do row "channels=$c" "$BASE_TOKENS" fp32 roce --channels "$c"; done; }
has mode && { echo "## bulk vs layerwise push (tokens=$BASE_TOKENS)"
  for i in $(seq "$REPS"); do for m in bulk layerwise; do row "mode=$m#$i" "$BASE_TOKENS" fp32 roce --mode "$m"; done; done; }
has pacing && { echo "## sender pacing on both workers (tokens=$BASE_TOKENS)"
  for r in $PACES; do bash scripts/pace.sh "$r" >/dev/null; row "pace=$r" "$BASE_TOKENS" fp32 roce; done
  bash scripts/pace.sh off >/dev/null; }
has delay && { echo "## netem delay on decode eth1 (tokens=$BASE_TOKENS)"
  for d in 0 5 20; do
    if [ "$d" = 0 ]; then bash scripts/netem.sh del >/dev/null; else bash scripts/netem.sh add "$d" >/dev/null; fi
    row "delay=${d}ms" "$BASE_TOKENS" fp32 roce
  done
  bash scripts/netem.sh del >/dev/null; }
has transport && { echo "## RoCEv2 RDMA WRITE vs TCP (tokens=$BASE_TOKENS)"
  for t in roce tcp; do row "transport=$t" "$BASE_TOKENS" fp32 "$t"; done; }
true
