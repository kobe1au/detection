#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-all}"

case "${STAGE}" in
  temporal|t1|i1|i1_dbta|adaptation|dbta)
    exec bash scripts/run_train_2026.sh i1
    ;;
  replay|memory)
    exec bash scripts/run_train_2026.sh replay
    ;;
  align|alignment|t2|i2)
    exec bash scripts/run_train_2026.sh i2
    ;;
  gate|fusion|t3|i3)
    exec bash scripts/run_train_2026.sh i3
    ;;
  baseline|base|baselines|cmp)
    exec bash scripts/run_train_2026.sh baselines
    ;;
  sweep|ratio|ratios)
    exec bash scripts/run_train_2026.sh ratio
    ;;
  final)
    exec bash scripts/run_train_2026.sh final
    ;;
  full)
    exec bash scripts/run_train_2026.sh full
    ;;
  main|chain|story)
    exec bash scripts/run_train_2026.sh main
    ;;
  all)
    exec bash scripts/run_train_2026.sh all
    ;;
  *)
    echo "Usage: $0 [all|baselines|i1|i1_dbta|replay|i2|i3|ratio|final|full|main]"
    exit 1
    ;;
esac
