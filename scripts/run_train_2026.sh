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
)

CONTINUAL=(
  "config/train_2026/continual/00_zero_adapt_concat.yaml"
  "config/train_2026/continual/01_i1_adapt_010.yaml"
  "config/train_2026/continual/02_i1_i2_adapt_010.yaml"
  "config/train_2026/continual/03_i1_i2_i3_adapt_010.yaml"
  "config/train_2026/continual/04_i1_i2_i3_dynamic_replay_adapt_010.yaml"
)

RATIO_SWEEP=(
  "config/train_2026/continual/01_i1_adapt_005.yaml"
  "config/train_2026/continual/01_i1_adapt_010.yaml"
  "config/train_2026/continual/01_i1_adapt_020.yaml"
  "config/train_2026/continual/01_i1_adapt_100.yaml"
  "config/train_2026/continual/02_i1_i2_adapt_005.yaml"
  "config/train_2026/continual/02_i1_i2_adapt_010.yaml"
  "config/train_2026/continual/02_i1_i2_adapt_020.yaml"
  "config/train_2026/continual/02_i1_i2_adapt_100.yaml"
  "config/train_2026/continual/03_i1_i2_i3_adapt_005.yaml"
  "config/train_2026/continual/03_i1_i2_i3_adapt_010.yaml"
  "config/train_2026/continual/03_i1_i2_i3_adapt_020.yaml"
  "config/train_2026/continual/03_i1_i2_i3_adapt_100.yaml"
)

I2_ALIGNMENT=(
  "config/train_2026/i2_alignment/00_no_alignment.yaml"
  "config/train_2026/i2_alignment/01_paired_alignment_only.yaml"
  "config/train_2026/i2_alignment/02_class_aware_alignment_loss.yaml"
  "config/train_2026/i2_alignment/03_method_mask_only.yaml"
  "config/train_2026/i2_alignment/04_full_class_aware_alignment.yaml"
)

I3_FUSION=(
  "config/train_2026/i3_fusion/00_fixed_gate.yaml"
  "config/train_2026/i3_fusion/01_learned_gate_no_quality_uncertainty.yaml"
  "config/train_2026/i3_fusion/02_quality_gate.yaml"
  "config/train_2026/i3_fusion/03_uncertainty_gate.yaml"
  "config/train_2026/i3_fusion/04_quality_uncertainty_gate.yaml"
  "config/train_2026/i3_fusion/05_pseudo_oracle_gate.yaml"
)

FINAL=(
  "config/train_2026/final/continual_ours_2026_adapt_005.yaml"
  "config/train_2026/final/continual_ours_2026_adapt_010.yaml"
  "config/train_2026/final/continual_ours_2026_adapt_020.yaml"
  "config/train_2026/final/continual_ours_2026_adapt_100.yaml"
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
    run_group "baselines" "${BASELINES[@]}"
    ;;
  continual|i1|main)
    run_group "continual-main-chain" "${CONTINUAL[@]}"
    ;;
  sweep|ratio|ratios)
    run_group "continual-ratio-sweep" "${RATIO_SWEEP[@]}"
    ;;
  i2|alignment|align)
    run_group "class-aware-alignment-ablation" "${I2_ALIGNMENT[@]}"
    ;;
  i3|fusion|gate)
    run_group "quality-aware-fusion-ablation" "${I3_FUSION[@]}"
    ;;
  final)
    run_group "final-ratio-sweep" "${FINAL[@]}"
    ;;
  all)
    run_group "baselines" "${BASELINES[@]}"
    run_group "continual-main-chain" "${CONTINUAL[@]}"
    run_group "class-aware-alignment-ablation" "${I2_ALIGNMENT[@]}"
    run_group "quality-aware-fusion-ablation" "${I3_FUSION[@]}"
    run_group "final-ratio-sweep" "${FINAL[@]}"
    ;;
  *)
    echo "Usage: $0 [all|baselines|continual|sweep|i2|i3|final]"
    exit 1
    ;;
esac
