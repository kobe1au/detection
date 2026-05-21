# 2026 Experiment Configs

Official experiments use the hand-written YAML files in this directory. The helper script
`scripts/make_ablation_configs.py` is for exploratory config generation only.

Protocol:

- Historical training: 2018-2021
- Validation: 2022
- Recent-year adaptation pool: 2023
- Final test: 2024
- Paper default adaptation ratio: 20%
- Default replay strategy for continual experiments: `dynamic_year_class`

Run order:

1. `scripts/run_train_2026.sh baselines`
   - API-only, Graph-only, Concat ERM, Cross-attention baselines.
2. `scripts/run_train_2026.sh main`
   - B0 zero-adapt, B1 I1, B2 I1+I2, B3 I1+I2+I3 at 20% adaptation.
3. `scripts/run_train_2026.sh i1`
   - I1-only adaptation ablations: no replay, static replay, dynamic replay, and I1-only ratio sweep.
4. `scripts/run_train_2026.sh replay`
   - No replay, static replay, dynamic year-class replay at 20% adaptation.
5. `scripts/run_train_2026.sh i2`
   - Class-aware API-Graph alignment ablations at 20% adaptation.
6. `scripts/run_train_2026.sh i3`
   - Quality-aware fusion gate ablations at 20% adaptation.
7. `scripts/run_train_2026.sh sweep`
   - Full model adaptation-ratio sweep: 5%, 10%, 20%, 100%.

Directory roles:

- `baselines/`: modality and fusion baselines without recent-year adaptation.
- `main_chain/`: main incremental innovation table at the 20% setting.
- `i1_adaptation/`: I1-only adaptation/replay/ratio ablations using concat fusion.
- `ratio_sweep/`: full model under 5%, 10%, 20%, and 100% adaptation.
- `replay_ablation/`: no replay vs static replay vs dynamic year-class replay.
- `i2_alignment/`: API-Graph alignment ablations.
- `i3_fusion/`: fusion gate ablations.
