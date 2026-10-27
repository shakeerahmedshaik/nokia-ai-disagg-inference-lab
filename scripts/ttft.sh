#!/usr/bin/env bash
# Experiment C: time to first token, colocated vs disaggregated, repeated REPS times (the host CPU is shared by all
# containers, so single runs are noisy). Prints every run and the medians.
# TTFT definitions: colocated = prefill_ms. Disaggregated = max(KV fully in decode memory, first token published)
# minus the request start, i.e. decode emits token 0 once it holds the whole cache.
# Usage: scripts/ttft.sh [prompt_tokens=512] [reps=3]
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
T=${1:-512}; REPS=${2:-3}
NEW=16
field() { grep -oP "$2=\K[0-9.]+" <<<"$1" | head -1; }
median() { sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }

declare -A TTFT PRE XFER ITL
add() { TTFT[$1]+="$2 "; PRE[$1]+="$3 "; XFER[$1]+="$4 "; ITL[$1]+="$5 "; }

for i in $(seq "$REPS"); do
  c=$(bash run.sh infer.py --role colocated --prompt-tokens "$T" --max-new-tokens $NEW --wait | grep 'role=colocated req=0 prompt')
  add colocated "$(field "$c" ttft_ms)" "$(field "$c" prefill_ms)" 0 "$(field "$c" itl_mean_ms)"
  for cfg in "disagg-bulk::--mode bulk" "disagg-layerwise::--mode layerwise" "disagg-layerwise-paced300::--mode layerwise"; do
    name=${cfg%%::*}; args=${cfg##*::}
    case $name in *paced300) bash scripts/pace.sh 300mbit >/dev/null ;; *) bash scripts/pace.sh off >/dev/null ;; esac
    out=$(bash run.sh infer.py --prompt-tokens "$T" --max-new-tokens $NEW $args --wait)
    d=$(grep 'role=decode req=0 prompt' <<<"$out"); p=$(grep 'role=prefill req=0 prompt' <<<"$out")
    add "$name" "$(field "$d" ttft_ms)" "$(field "$p" prefill_ms)" "$(field "$p" exposed_transfer_ms)" "$(field "$d" itl_mean_ms)"
    echo "run=$i $name ttft_ms=$(field "$d" ttft_ms) prefill_ms=$(field "$p" prefill_ms) exposed_transfer_ms=$(field "$p" exposed_transfer_ms) itl_mean_ms=$(field "$d" itl_mean_ms)"
  done
  echo "run=$i colocated ttft_ms=$(field "$c" ttft_ms) prefill_ms=$(field "$c" prefill_ms) itl_mean_ms=$(field "$c" itl_mean_ms)"
done
bash scripts/pace.sh off >/dev/null

echo "## medians over $REPS runs, prompt_tokens=$T"
printf '%-28s %10s %12s %20s %12s\n' config ttft_ms prefill_ms exposed_transfer_ms itl_mean_ms
for k in colocated disagg-bulk disagg-layerwise disagg-layerwise-paced300; do
  printf '%-28s %10s %12s %20s %12s\n' "$k" "$(tr ' ' '\n' <<<"${TTFT[$k]}" | sed '/^$/d' | median)" \
    "$(tr ' ' '\n' <<<"${PRE[$k]}" | sed '/^$/d' | median)" "$(tr ' ' '\n' <<<"${XFER[$k]}" | sed '/^$/d' | median)" \
    "$(tr ' ' '\n' <<<"${ITL[$k]}" | sed '/^$/d' | median)"
done
