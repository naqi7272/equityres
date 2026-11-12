#!/usr/bin/env bash
# Run the research stages in order. Usage:
#   bash scripts/run_all.sh                          # real data, stages 1 to 5
#   bash scripts/run_all.sh config/synthetic.yaml    # offline smoke test
# Stage 6 (the holdout) is deliberately NOT run here. Run it by hand, once.
set -euo pipefail
CFG=${1:+--config $1}
PY=${PYTHON:-python}
$PY -m pytest -q
for stage in 01_download 02_build_data 03_signals 04_evaluate_signals 05_combine_backtest; do
  echo "=== $stage ==="
  $PY scripts/$stage.py $CFG
done
echo "Done. When the configuration is final: $PY scripts/06_holdout.py $CFG"
