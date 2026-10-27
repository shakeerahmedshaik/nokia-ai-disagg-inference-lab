#!/usr/bin/env bash
# Reproduce every experiment in order and write the raw outputs to outputs/ (about 45 minutes).
# Prerequisite: containerlab deploy -t topology.clab.yml && bash setup.sh
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
mkdir -p outputs captures

echo "### Experiment A: one small request, full capture"
bash scripts/fabric.sh counters > outputs/expA-counters-before.txt
bash scripts/capture.sh start expA
bash run.sh infer.py --prompt-tokens 32 --max-new-tokens 8 --wait > outputs/expA-infer.txt 2>&1
bash scripts/capture.sh stop expA
bash scripts/fabric.sh counters > outputs/expA-counters-after.txt
bash scripts/analyze.sh captures/expA.pcap > outputs/expA-analysis.txt 2>&1
python3 scripts/ecmp-delta.py outputs/expA-counters-before.txt outputs/expA-counters-after.txt > outputs/expA-ecmp.txt

echo "### Experiment B: colocated vs disaggregated, plus the negative control"
bash scripts/compare.sh 128 32 | tee outputs/expB-summary.txt
bash scripts/analyze.sh captures/expB.pcap > outputs/expB-analysis.txt 2>&1
python3 scripts/ecmp-delta.py outputs/expB-counters-before.txt outputs/expB-counters-after.txt > outputs/expB-ecmp.txt

echo "### Experiment C: time to first token, medians of 3"
bash scripts/ttft.sh 128 3 > outputs/expC-ttft-128.txt 2>&1
bash scripts/ttft.sh 512 3 > outputs/expC-ttft-512.txt 2>&1

echo "### Experiment D/E/F: sweeps (prompt length, dtype, channels, bulk vs layerwise x3, pacing, delay, transport)"
REPS=3 PACES="off 60mbit 100mbit 150mbit 200mbit 300mbit 400mbit" bash scripts/sweep.sh > outputs/sweeps.txt 2>&1

echo "### Experiment G: spine link failure in the middle of a KV transfer"
bash scripts/failover.sh > outputs/failover.txt 2>&1
echo "all done"
