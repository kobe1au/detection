# Robust Experiment Summary

The active experiment set is `config/experiments/tri_modal_robust`.

## Run Commands

```bash
python run.py final
python run.py all
python run.py api
python run.py graph
python run.py manifest
python run.py reliability
```

Training entry:

```bash
python -m fusion.train --config config/experiments/tri_modal_robust/T7_tri_modal_full_soft_consistency.yaml
```

## Baselines

- `T0_api_only.yaml`
- `T1_graph_only.yaml`
- `T2_manifest_only.yaml`
- `T3_api_graph_concat.yaml`
- `T4_api_graph_manifest_concat.yaml`
- `T4b_api_graph_manifest_concat_robust_aug.yaml`
- `T5_tri_modal_fixed_gate.yaml`
- `T6_tri_modal_reliability_gate.yaml`
- `T7_tri_modal_full_soft_consistency.yaml`
- `T7_tri_modal_full_no_aug.yaml`
- `T7_tri_modal_full_no_gate_prior.yaml`
- `T7_tri_modal_full_no_manifest_aux.yaml`

## Robust Tests

- clean
- api_degraded
- graph_degraded
- api_graph_degraded
- manifest_degraded
- all_degraded
- api_missing
- graph_missing
- manifest_missing

`gate_diagnostics.csv` records `w_api`, `w_graph`, `w_manifest`, `w_joint`, `q_api/q_graph/q_manifest`, `pert_api/pert_graph/pert_manifest`, `r_api/r_graph/r_manifest`, modality alive flags, pairwise consistency, branch confidence, and concrete perturbation types.
