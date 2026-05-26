# Robust Fusion Comparison Plan

The codebase now targets API + Graph + Manifest robust fusion only.

## Main Comparisons

1. API-only
2. Graph-only
3. Manifest-only
4. API + Graph concat
5. API + Graph + Manifest concat
6. Tri-modal fixed gate
7. Tri-modal reliability gate
8. Tri-modal learned conflict-aware gate

## Robustness Protocol

Evaluate every main method on:

- clean
- api_degraded
- graph_degraded
- api_graph_degraded
- manifest_degraded
- all_degraded
- api_missing
- graph_missing
- manifest_missing

## Diagnostics

Plot gate weights against perturbation strength:

- API degradation should reduce `w_api`.
- Graph degradation should reduce `w_graph`.
- Manifest degradation should reduce `w_manifest`.
- API + Graph degradation should increase relative Manifest or Joint reliance when Manifest remains reliable.
- All-degraded performance should degrade more gracefully than simple concatenation.
