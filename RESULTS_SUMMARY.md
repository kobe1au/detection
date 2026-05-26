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
python -m fusion.robust.train --config config/experiments/tri_modal_robust/T7_tri_modal_full_soft_consistency.yaml
```

## Baselines

- `T0_api_only.yaml`
- `T1_graph_only.yaml`
- `T2_manifest_only.yaml`
- `T3_api_graph_concat.yaml`
- `T4_api_graph_manifest_concat.yaml`
- `T5_tri_modal_fixed_gate.yaml`
- `T6_tri_modal_reliability_gate.yaml`
- `T7_tri_modal_full_soft_consistency.yaml`

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

`gate_diagnostics.csv` records `w_api`, `w_graph`, `w_manifest`, `w_joint`, `q_manifest`, `pert_manifest`, `api_manifest_consistency`, and `graph_manifest_consistency`.
