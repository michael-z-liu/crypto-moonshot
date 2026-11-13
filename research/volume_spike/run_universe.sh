#!/usr/bin/env bash
# M13: re-run the main tests on a wider coin universe, unchanged (no re-tuning).
#   research/volume_spike/run_universe.sh poolA   # + delisted coins
#   research/volume_spike/run_universe.sh all     # + delisted + every other archived USDT pair
# Outputs go to cache_<universe>/ and results/universe_<universe>/.
set -euo pipefail
cd "$(dirname "$0")"
export MOONSHOT_UNIVERSE="$1"
step() { echo "== $(date -u +%H:%M:%SZ) $MOONSHOT_UNIVERSE: $*"; }

step events;        uv run python build_events.py 2>/dev/null
step finalists;     uv run python m6_test.py > "results/universe_$1/m6_test_output.txt"
step walkforward;   uv run python m7_walkforward.py t 30
step sequences;     uv run python build_seq2.py 2>/dev/null && uv run python build_seq2.py fwd 2>/dev/null
if [ "$1" = "all" ]; then
  variants="base coin_only win2h win24h eth market all_ch more_scalars t30m t4h huber bce xs tcn gru mlp wide path path_all"
else
  variants="base market path_all"
fi
step cnn $variants; uv run python m12_cnn_wf.py run $variants
step eval;          uv run python m12_cnn_wf.py eval
if [ "$1" = "all" ]; then step ensemble; uv run python m12_cnn_wf.py ens; fi
step done
