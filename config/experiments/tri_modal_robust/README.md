# Tri-modal Robust Fusion Experiments

This directory contains the clean experiment plan for the robust API + Graph + Manifest framework.

## Groups

- `i1/`: innovation 1, reliability modeling for code common-mode degradation.
- `i2/`: innovation 2, cross-source soft consistency evidence and loss.
- `i3/`: innovation 3, four-branch adaptive fusion and gate strategy.
- `full/`: final method and major ablations.
- `seed/`: multi-seed runs for the final method only.
- `tune/`: sensitivity checks for innovation-related parameters. Do not mix these with the main ablation tables.

## Recommended Order

Run the core method first:

```bash
python -m fusion.train --config config/experiments/tri_modal_robust/full/ours.yaml
```

The helper runner can also select grouped experiments:

```bash
python run.py main --dry-run
python run.py i1 --dry-run
python run.py i2 --dry-run
python run.py i3 --dry-run
python run.py i2,i3 --dry-run
python run.py seed --dry-run
```

`run.py` deliberately excludes `tune/` configs. Use the Optuna driver for tuning so test
evaluation cannot be mixed into parameter selection.

Then run innovation-specific ablations:

```bash
python -m fusion.train --config config/experiments/tri_modal_robust/i1/api_graph_concat.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i1/tri_modal_concat.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i1/reliability_gate.yaml

python -m fusion.train --config config/experiments/tri_modal_robust/i2/no_consistency.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i2/consistency_evidence_only.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i2/conflict_evidence_only.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i2/evidence_only.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i2/loss_only.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i2/semantic_reconstruction_only.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i2/evidence_plus_loss.yaml

python -m fusion.train --config config/experiments/tri_modal_robust/i3/fixed_gate.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i3/confidence_gate.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i3/reliability_gate.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i3/learned_gate_no_alive_mask.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i3/learned_gate_no_prior.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i3/learned_gate_with_prior.yaml
```

After the best setting is confirmed, run:

```bash
python -m fusion.train --config config/experiments/tri_modal_robust/seed/seed_42.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/seed/seed_2024.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/seed/seed_3407.yaml
```

## Stage-wise Optuna Tuning

Optuna tuning uses representative robust validation for checkpoint selection and does not
load or evaluate the test split. Run the stages in order:

The fixed robust-validation checkpoint score is a weighted macro-F1 average:

- clean validation: `0.40`
- API+Graph degraded at strength `0.5`: `0.25`
- Manifest degraded at strength `0.5`: `0.15`
- all modalities degraded at strength `0.5`: `0.10`
- API, Graph, and Manifest missing: `0.0333` each

These scenarios and weights must be frozen before examining final test results. Changing
them after observing test performance would make the final test no longer independent.

Use a new output directory and study name for every protocol version. The study stores a
configuration fingerprint and rejects incompatible resumed trials.

```bash
python scripts/tune_robust_optuna.py --stage i2 --trials 25 \
  --study-name robust_v2_i2 --output-dir results/optuna/robust_v2

python scripts/tune_robust_optuna.py --stage i3 --trials 25 \
  --study-name robust_v2_i3 --output-dir results/optuna/robust_v2 \
  --config config/experiments/tri_modal_robust/tune/optuna_base.yaml \
  results/optuna/robust_v2/best_i2_override.yaml

python scripts/tune_robust_optuna.py --stage aug --trials 9 \
  --study-name robust_v2_aug --output-dir results/optuna/robust_v2 \
  --config config/experiments/tri_modal_robust/tune/optuna_base.yaml \
  results/optuna/robust_v2/best_i2_override.yaml \
  results/optuna/robust_v2/best_i3_override.yaml
```

The augmentation stage is an exact 3-by-3 grid over perturbation probability and strength
profile, so it has nine unique trials. The i2 search includes
`cross_source_consistency_weight=0`; this allows the study to reject the cross-source
loss if it does not improve representative robust validation. Semantic reconstruction
and cross-source consistency are separate loss terms and must be reported separately.

Tuning and final training use the same 60-epoch budget, early-stopping rule, deterministic
mode, and robust-composite checkpoint metric. The only intended difference is that tuning
does not load or evaluate test data.

Use one seed during broad search. Do not use another Optuna search as a substitute for
multi-seed confirmation. After all three stages are fixed, train the exact selected
configuration with the three seed overrides:

```bash
python -m fusion.train --config config/experiments/tri_modal_robust/full/ours.yaml config/experiments/tri_modal_robust/seed/seed_42.yaml results/optuna/robust_v2/best_i2_override.yaml results/optuna/robust_v2/best_i3_override.yaml results/optuna/robust_v2/best_aug_override.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/full/ours.yaml config/experiments/tri_modal_robust/seed/seed_2024.yaml results/optuna/robust_v2/best_i2_override.yaml results/optuna/robust_v2/best_i3_override.yaml results/optuna/robust_v2/best_aug_override.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/full/ours.yaml config/experiments/tri_modal_robust/seed/seed_3407.yaml results/optuna/robust_v2/best_i2_override.yaml results/optuna/robust_v2/best_i3_override.yaml results/optuna/robust_v2/best_aug_override.yaml
```

The seed override must appear before the generated best-parameter overrides because each
seed config inherits `full/ours.yaml`.

After selecting the final parameters, run the complete test protocol from `full/ours.yaml`
instead of `optuna_base.yaml`:

```bash
python scripts/make_post_optuna_configs.py \
  --tag robust_v2 \
  --best-i2 results/optuna/robust_v2/best_i2_override.yaml \
  --best-i3 results/optuna/robust_v2/best_i3_override.yaml \
  --best-aug results/optuna/robust_v2/best_aug_override.yaml

python run.py post_optuna/robust_v2/full --dry-run
python run.py post_optuna/robust_v2/full
```

The generated `post_optuna/<tag>` configs lock the safe override order:
base method, best selected parameters, then the final ablation override. This prevents
best gate parameters from overwriting fixed/reliability gate ablations.
Use the generated configs for the final paper tables:

```bash
python run.py post_optuna/robust_v2/i1 --dry-run
python run.py post_optuna/robust_v2/i2 --dry-run
python run.py post_optuna/robust_v2/i3 --dry-run
python run.py post_optuna/robust_v2/full --dry-run
python run.py post_optuna/robust_v2/seed --dry-run

python run.py post_optuna/robust_v2/i1
python run.py post_optuna/robust_v2/i2
python run.py post_optuna/robust_v2/i3
python run.py post_optuna/robust_v2/full
python run.py post_optuna/robust_v2/seed
```

The original `i1/`, `i2/`, `i3/`, and `full/` configs remain useful for development
and sanity checks.  They are not the frozen post-tuning protocol once Optuna has been
used.

## External-Style Reference Baselines

Use these as reference implementations, not claims of exact paper reproduction.
They provide classical/static comparison points for the final method:

```bash
python scripts/train_static_baselines.py \
  --config config/experiments/tri_modal_robust/base_tri_modal_robust.yaml \
  --out-dir results/static_reference_baselines \
  --run-test \
  --robust-test
```

The script includes Drebin-style sparse static features, MaMaDroid-style API-type
transition features, API bag-of-words, Manifest-only, and tri-modal static linear
baselines. Report them separately from the internal neural ablations.

## Real Failure Slices

Build quality/failure slice CSVs before final robustness evaluation:

```bash
python scripts/build_real_failure_slices.py \
  --config config/experiments/tri_modal_robust/base_tri_modal_robust.yaml \
  --splits val test \
  --out-dir results/robust_slices \
  --extra-eval-yaml results/robust_slices/extra_eval_slices.yaml
```

Then evaluate the selected checkpoint/method on those slices by appending the generated
override:

```bash
python -m fusion.train --config \
  config/experiments/tri_modal_robust/post_optuna/robust_v2/full/full_ours.yaml \
  results/robust_slices/extra_eval_slices.yaml
```

These slices are the main evidence for real extractor failures: low API quality, low
graph quality, low API-Graph alignment, Manifest parse failures, and partial multi-DEX
failures.

## Calibration Diagnostics

Every neural evaluation now reports `brier`, `ece_10`, `mean_confidence`, and
`confidence_accuracy_gap` in `summary.yaml`. Per-sample `gate_diagnostics.csv` also
contains final confidence and correctness, which supports calibration and gate-weight
correlation plots.

## Notes

The default data paths target the AutoDL layout:

- train pt: `/root/autodl-tmp/pts/train`
- val pt: `/root/autodl-tmp/pts/val`
- test pt: `/pts/test`
- labels: `results/labels/{train,val,test}.csv`

If these paths change, update only `base_tri_modal_robust.yaml`.

The main gate uses observable post-extraction quality, confidence, consistency, conflict,
and alive signals. Synthetic `pert_*` values are diagnostics only. The explicit
`full/ours_oracle_perturbation_evidence.yaml` ablation exposes perturbation strength to the
gate and must never be reported as the main method.

Because the current test split has already been inspected during development, publication
claims require a newly locked final test or an external real-obfuscation/failure set.
