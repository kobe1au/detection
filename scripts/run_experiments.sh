#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-all}"

case "${STAGE}" in
  temporal|t1)
    exec bash scripts/run_train_2026.sh i1
    ;;
  align|t2)
    exec bash scripts/run_train_2026.sh i2
    ;;
  gate|t3)
    exec bash scripts/run_train_2026.sh i3
    ;;
  baseline|base|baselines|cmp)
    exec bash scripts/run_train_2026.sh baselines
    ;;
  final)
    exec bash scripts/run_train_2026.sh final
    ;;
  all)
    exec bash scripts/run_train_2026.sh all
    ;;
  *)
    echo "Usage: $0 [temporal|align|gate|baseline|baselines|cmp|final|all]"
    exit 1
    ;;
esac
