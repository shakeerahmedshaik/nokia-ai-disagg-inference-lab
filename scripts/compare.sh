#!/usr/bin/env bash
# Experiment B: the same request colocated, then disaggregated over RoCEv2 (with a capture and counters),
# then disaggregated with the received KV thrown away (negative control).
# Usage: scripts/compare.sh [prompt_tokens=128] [new_tokens=32] [extra infer.py args...]
# Pass criteria: generated ids and KV crc32 identical in colocated and disaggregated runs; control diverges.
set -euo pipefail

LAB=clab-ai-inference
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
P=${1:-128}; N=${2:-32}; shift 2 || true
EXTRA=("$@")
OUT=outputs

field() { grep -oP "$2=\K\S+" <<<"$1" | head -1; }
ids()   { grep -oP 'generated_ids=\K.*' "$1" | head -1; }

echo "## colocated reference ($P prompt tokens, $N new tokens)"
bash run.sh infer.py --role colocated --prompt-tokens "$P" --max-new-tokens "$N" "${EXTRA[@]}" --wait > $OUT/expB-colocated.txt
grep 'role=colocated req=0 prompt' $OUT/expB-colocated.txt | cut -c1-300

echo "## disaggregated over RoCEv2 (capture on prefill eth1)"
bash scripts/rxe-stats.sh > $OUT/expB-rxe-before.txt
bash scripts/fabric.sh counters > $OUT/expB-counters-before.txt
bash scripts/capture.sh start expB >/dev/null
bash run.sh infer.py --prompt-tokens "$P" --max-new-tokens "$N" "${EXTRA[@]}" --wait > $OUT/expB-disagg.txt
bash scripts/capture.sh stop expB >/dev/null
bash scripts/rxe-stats.sh > $OUT/expB-rxe-after.txt
bash scripts/fabric.sh counters > $OUT/expB-counters-after.txt
grep -E 'role=(prefill|decode) req=0 (prompt|decode)' $OUT/expB-disagg.txt | cut -c1-420

echo "## negative control: decode ignores the received KV"
bash run.sh infer.py --prompt-tokens "$P" --max-new-tokens "$N" "${EXTRA[@]}" --control zero-kv --wait > $OUT/expB-control.txt
grep 'role=decode req=0 generated_ids' $OUT/expB-control.txt | cut -c1-200

echo "## verdict"
c=$(ids $OUT/expB-colocated.txt); d=$(ids $OUT/expB-disagg.txt); z=$(ids $OUT/expB-control.txt)
ck=$(field "$(grep 'role=colocated req=0 prompt' $OUT/expB-colocated.txt)" kv_crc32)
pk=$(field "$(grep 'role=prefill req=0 prompt' $OUT/expB-disagg.txt)" kv_crc32)
dk=$(field "$(grep 'role=decode req=0 prompt' $OUT/expB-disagg.txt)" kv_crc32)
echo "kv_crc32 colocated=$ck prefill=$pk decode=$dk"
[ "$ck" = "$pk" ] && [ "$pk" = "$dk" ] && echo "PASS: KV bytes identical end to end" || echo "FAIL: KV crc mismatch"
[ "$c" = "$d" ] && echo "PASS: disaggregated tokens identical to colocated" || echo "FAIL: tokens differ"
[ "$c" != "$z" ] && echo "PASS: control (KV discarded) diverges, so the KV carries the state" || echo "FAIL: control did not diverge"
