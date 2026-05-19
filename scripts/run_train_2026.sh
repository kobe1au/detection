#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

BASE_CONFIG="${BASE_CONFIG:-config/base.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE="${1:-all}"

BASELINES=(
  "config/train_2026/baselines/00_api_only.yaml"
  "config/train_2026/baselines/01_graph_only.yaml"
  "config/train_2026/baselines/02_concat_erm.yaml"
  "config/train_2026/baselines/03_cross_attention.yaml"
  "config/train_2026/baselines/04_temporal_supcon.yaml"
  "config/train_2026/baselines/05_groupdro.yaml"
  "config/train_2026/baselines/06_vrex.yaml"
  "config/train_2026/baselines/07_irm.yaml"
  "config/train_2026/baselines/08_coral.yaml"
)

I1_TEMPORAL=(
  "config/train_2026/i1_temporal/00_erm_concat.yaml"
  "config/train_2026/i1_temporal/01_proto_current.yaml"
  "config/train_2026/i1_temporal/02_proto_current_010.yaml"
  "config/train_2026/i1_temporal/03_proto_future_weak.yaml"
  "config/train_2026/i1_temporal/04_ours_fixed_scaffold_erm.yaml"
  "config/train_2026/i1_temporal/05_ours_fixed_proto_trajectory.yaml"
)

I2_ALIGNMENT=(
  "config/train_2026/i2_alignment/00_temporal_concat.yaml"
  "config/train_2026/i2_alignment/01_cross_attention.yaml"
  "config/train_2026/i2_alignment/02_ours_fixed_no_alignment.yaml"
  "config/train_2026/i2_alignment/03_semantic_alignment.yaml"
  "config/train_2026/i2_alignment/04_method_aware_context.yaml"
  "config/train_2026/i2_alignment/05_temporal_guided_alignment.yaml"
)

I3_FUSION=(
  "config/train_2026/i3_fusion/00_fixed_no_gate.yaml"
  "config/train_2026/i3_fusion/01_learned_gate_no_reliability.yaml"
  "config/train_2026/i3_fusion/02_quality_gate.yaml"
  "config/train_2026/i3_fusion/03_drift_gate.yaml"
  "config/train_2026/i3_fusion/04_quality_drift_gate.yaml"
)

FINAL=(
  "config/train_2026/final/ours_2026_no_future.yaml"
  "config/train_2026/final/ours_2026_no_semantic_align.yaml"
  "config/train_2026/final/ours_2026_no_reliability_gate.yaml"
  "config/train_2026/final/ours_2026.yaml"
)

run_group() {
  local group_name="$1"
  shift
  local overrides=("$@")

  echo ""
  echo "============================================================"
  echo "Running group: ${group_name}"
  echo "Base config: ${BASE_CONFIG}"
  echo "============================================================"

  for override in "${overrides[@]}"; do
    echo "==> Running ${override}"
    "${PYTHON_BIN}" -m fusion.train --base "${BASE_CONFIG}" --override "${override}"
  done
}

case "${STAGE}" in
  baseline|base|baselines|cmp)
    run_group "baselines-from-scratch" "${BASELINES[@]}"
    ;;
  i1|temporal|t1)
    run_group "innovation-1-temporal-prototype" "${I1_TEMPORAL[@]}"
    ;;
  i2|alignment|align|t2)
    run_group "innovation-2-method-aware-alignment" "${I2_ALIGNMENT[@]}"
    ;;
  i3|fusion|gate|t3)
    run_group "innovation-3-quality-drift-gate" "${I3_FUSION[@]}"
    ;;
  final)
    run_group "final-ablation" "${FINAL[@]}"
    ;;
  all)
    run_group "baselines-from-scratch" "${BASELINES[@]}"
    run_group "innovation-1-temporal-prototype" "${I1_TEMPORAL[@]}"
    run_group "innovation-2-method-aware-alignment" "${I2_ALIGNMENT[@]}"
    run_group "innovation-3-quality-drift-gate" "${I3_FUSION[@]}"
    run_group "final-ablation" "${FINAL[@]}"
    ;;
  *)
    echo "Usage: $0 [all|baselines|i1|i2|i3|final]"
    exit 1
    ;;
esac
