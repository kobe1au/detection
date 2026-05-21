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

MAIN_CHAIN=(
  # = i1_adaptation/00_zero_adapt_concat.yaml
  "config/train_2026/main_chain/00_zero_adapt_concat.yaml"

  # = i1_adaptation/03_adapt_020_dynamic_replay.yaml
  "config/train_2026/main_chain/01_i1_adapt_020.yaml"

  # = i2_alignment/04_full_class_aware_alignment.yaml
  "config/train_2026/main_chain/02_i1_i2_adapt_020.yaml"

  # = i3_fusion/04_quality_uncertainty_gate.yaml
  "config/train_2026/main_chain/03_i1_i2_i3_adapt_020.yaml"
)

I1_ADAPTATION=(
  "config/train_2026/i1_adaptation/00_zero_adapt_concat.yaml"
  "config/train_2026/i1_adaptation/01_adapt_020_no_replay.yaml"
  "config/train_2026/i1_adaptation/02_adapt_020_static_replay.yaml"
  "config/train_2026/i1_adaptation/03_adapt_020_dynamic_replay.yaml"
  "config/train_2026/i1_adaptation/10_ratio_005_dynamic_replay.yaml"
  "config/train_2026/i1_adaptation/11_ratio_010_dynamic_replay.yaml"
  "config/train_2026/i1_adaptation/12_ratio_020_dynamic_replay.yaml"
  "config/train_2026/i1_adaptation/13_ratio_100_dynamic_replay.yaml"
)

RATIO_SWEEP=(
  "config/train_2026/ratio_sweep/00_full_adapt_000.yaml"
  "config/train_2026/ratio_sweep/00_full_adapt_005.yaml"
  "config/train_2026/ratio_sweep/01_full_adapt_010.yaml"
  "config/train_2026/ratio_sweep/02_full_adapt_020.yaml"
  "config/train_2026/ratio_sweep/03_full_adapt_100.yaml"
)

REPLAY_ABLATION=(
  "config/train_2026/replay_ablation/00_no_replay_adapt_020.yaml"
  "config/train_2026/replay_ablation/01_static_replay_adapt_020.yaml"
  "config/train_2026/replay_ablation/02_dynamic_year_class_replay_adapt_020.yaml"
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
  main|chain|continual)
    run_group "main-chain-20pct" "${MAIN_CHAIN[@]}"
    ;;
  i1|adaptation)
    run_group "i1-adaptation-ablation" "${I1_ADAPTATION[@]}"
    ;;
  sweep|ratio|ratios)
    run_group "full-model-ratio-sweep" "${RATIO_SWEEP[@]}"
    ;;
  replay|memory)
    run_group "replay-ablation-20pct" "${REPLAY_ABLATION[@]}"
    ;;
  i2|alignment|align)
    run_group "class-aware-alignment-ablation" "${I2_ALIGNMENT[@]}"
    ;;
  i3|fusion|gate)
    run_group "quality-aware-fusion-ablation" "${I3_FUSION[@]}"
    ;;
  final)
    run_group "full-model-ratio-sweep" "${RATIO_SWEEP[@]}"
    ;;
  all)
    run_group "baselines" "${BASELINES[@]}"
    run_group "main-chain-20pct" "${MAIN_CHAIN[@]}"
    run_group "i1-adaptation-ablation" "${I1_ADAPTATION[@]}"
    run_group "replay-ablation-20pct" "${REPLAY_ABLATION[@]}"
    run_group "class-aware-alignment-ablation" "${I2_ALIGNMENT[@]}"
    run_group "quality-aware-fusion-ablation" "${I3_FUSION[@]}"
    run_group "full-model-ratio-sweep" "${RATIO_SWEEP[@]}"
    ;;
  *)
    echo "Usage: $0 [all|baselines|main|i1|replay|sweep|i2|i3|final]"
    exit 1
    ;;
esac
