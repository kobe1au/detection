# Source-Aware APK Evidence Graph Malware Detection

This repository implements a robust Android malware detection pipeline based on **source-aware APK evidence graphs** and **reliability-aware multi-view KL consistency learning**.

The active mainline is no longer contrastive-learning based. The current training objective is:

```text
Cross-entropy classification
+ optional clean/degraded-view KL consistency
+ reliability-aware consistency weighting
```

The supported loss modes are:

```text
ce_only      CE only
plain_kl     CE + unweighted clean/degraded KL consistency
compact_kl   CE + reliability-aware clean/degraded KL consistency
```

## Main Ideas

This project is organized around three thesis-level ideas.

1. **Source-aware APK heterogeneous evidence graph modeling**

   APK evidence is represented as a typed heterogeneous graph. The graph integrates code evidence, Manifest declaration evidence, derived risk semantics, string/static hints, and cross-source alignment evidence.

2. **Reliability-aware multi-view KL consistency learning**

   The model is trained on clean and degraded graph views. Instead of using InfoNCE or batch-wise contrastive loss, the current mainline uses KL consistency between clean and perturbed predictions. The consistency strength is adjusted according to view type, counterfactual weight, and observable code/Manifest reliability.

3. **Counterfactual reliability-aware latent fusion with Manifest shortcut suppression**

   The model fuses method, API-family, permission, component, risk, string-hint, and global graph tokens. Fusion is guided by reliability, source information, and code-Manifest conflict so that Manifest evidence is not treated as an unconditional shortcut under corrupted or shuffled Manifest scenarios.

## Repository Layout

```text
extract/
  extract_graph_api.py              Reusable DEX/API/call-graph extractor

fusion/
  aeg_builder.py                    Builds versioned APK evidence graph payloads
  constants.py                      Node, edge, source, and view definitions
  dataset.py                        AEG Dataset, augmentation, and PyG collation
  io_utils.py                       Safe AEG payload and checkpoint loading
  losses.py                         CE + plain/reliability-weighted KL consistency
  manifest_features.py              Manifest parsing, vocabulary, and vectorization
  model.py                          Typed graph encoder + reliability-aware latent fusion
  perturbations.py                  Clean/degraded graph-view perturbations
  train.py                          Training, validation, robust evaluation, diagnostics

scripts/
  build_aeg_pts_direct.py           APK -> AEG .pt builder
  validate_aeg_pts.py               PT/schema/contract validator
  summarize_aeg_diagnostics.py      Diagnostics summarization

config/
  extract_aeg.yaml                  APK -> AEG extraction config
  experiments/aeg_robust/
    base.yaml                       Shared experiment template
    main/                           Full method
    loss/                           CE-only / Plain-KL / Compact-KL experiments
    r1_graph/                       Graph/source/evidence ablations
    r3_fusion/                      Reliability/fusion ablations

tests/
  test_aeg_smoke.py                 Smoke tests for payloads, model, losses, configs
```

## Build AEG PT Files

Build or rebuild the Manifest vocabulary from the train split only:

```bash
python scripts/build_aeg_pts_direct.py \
  --config config/extract_aeg.yaml \
  --rebuild-vocab \
  --no-resume \
  --workers 8
```

Resume later without rebuilding the vocabulary:

```bash
python scripts/build_aeg_pts_direct.py \
  --config config/extract_aeg.yaml \
  --no-rebuild-vocab \
  --resume \
  --workers 8
```

If disk space cannot hold train PT files yet, build only the train-derived Manifest vocabulary first:

```bash
python scripts/build_aeg_pts_direct.py \
  --config config/extract_aeg_vocab_train_only.yaml \
  --rebuild-vocab \
  --vocab-only \
  --workers 8
```

Then generate validation and test PT files with the frozen train vocabulary:

```bash
python scripts/build_aeg_pts_direct.py \
  --config config/extract_aeg_val_test.yaml \
  --no-rebuild-vocab \
  --resume \
  --workers 8
```

Generate train PT files later without rebuilding the frozen Manifest vocabulary:

```bash
python scripts/build_aeg_pts_direct.py \
  --config config/extract_aeg_train_only.yaml \
  --no-rebuild-vocab \
  --resume \
  --workers 8
```

Validate generated PT files before training:

```bash
python scripts/validate_aeg_pts.py \
  --config config/extract_aeg.yaml \
  --sample-per-split 100
```

Use `--all` for a full validation pass. The validator checks CSV/PT id consistency, node feature dimensions, schema versions, payload contract versions, and required tensor fields.

## Data Integrity and Trust Boundaries

Training uses strict CSV/PT integrity by default.

The training pipeline rejects:

```text
CSV ids without corresponding AEG .pt files
extra PT samples not listed in the split CSV
sample id overlap across train/val/test
package_name overlap across train/val/test when enabled
schema version mismatch
payload contract mismatch
invalid tensor shapes or missing required fields
```

AEG payload loading uses `load_aeg_payload()` in fail-closed mode. It first tries safe tensor loading and validates the payload contract after deserialization. This validation is not a substitute for artifact provenance. For untrusted PT files, verify external checksums or signatures before loading.

## Experiment Configuration

The shared experiment template is:

```text
config/experiments/aeg_robust/base.yaml
```

Before training, edit the data paths in `base.yaml` or use your own copied configuration so that these paths point to generated AEG PT files and label CSV files:

```yaml
data:
  train:
    pt_dir: D:/pts_aeg/train
    csv: results/labels/train.csv

  val:
    pt_dir: D:/pts_aeg/val
    csv: results/labels/val.csv

  test:
    pt_dir: D:/pts_aeg/test
    csv: results/labels/test.csv
```

The default training loss in `base.yaml` is the full method:

```yaml
loss:
  mode: compact_kl
  ce_weight: 1.0
  consistency_weight: 0.05
  aug_ce_weight: 0.0
```

For `plain_kl` and `compact_kl`, training augmentation must be enabled:

```yaml
robust:
  train_aug: true
```

For `ce_only`, augmentation can be disabled to avoid an unnecessary second forward pass:

```yaml
robust:
  train_aug: false

loss:
  mode: ce_only
```

## Train the Full Method

Run the full reliability-aware KL method:

```bash
python -m fusion.train --config config/experiments/aeg_robust/main/full_compact_kl.yaml
```

Or use the compact experiment runner:

```bash
python run.py final
```

Equivalent aliases include:

```bash
python run.py ours
python run.py full
python run.py compact
```

## Experiment Runner

`run.py` resolves experiment aliases and groups under:

```text
config/experiments/aeg_robust/
```

Useful single targets:

```bash
python run.py final
python run.py ce_only
python run.py plain_kl
python run.py compact_kl
```

Useful groups:

```bash
python run.py main
python run.py loss
python run.py r1_graph
python run.py r3_fusion
python run.py all
```

Preview resolved configs without training:

```bash
python run.py all --dry-run
```

Expected active groups:

```text
main        Full compact-KL method
loss        CE-only, Plain-KL, Compact-KL, and consistency-weight variants
r1_graph    Graph/source/evidence ablations
r3_fusion   Reliability and fusion ablations
all         Main + loss + graph + fusion experiment set
```

The old `i1/i2/i3` and contrastive-loss experiment names are no longer part of the active mainline.

## Core Experiments

### 1. Main Method

```bash
python run.py final
```

Full method:

```text
Source-aware AEG encoder
Reliability-aware latent fusion
Reliability-weighted clean/degraded KL consistency
Robust validation checkpoint selection
```

Primary config:

```text
config/experiments/aeg_robust/main/full_compact_kl.yaml
```

### 2. Loss Ablation

```bash
python run.py loss
```

This group compares:

```text
CE-only
Plain KL consistency
Reliability-weighted KL consistency
Compact-KL with weaker consistency weight
Compact-KL with stronger consistency weight
```

Expected interpretation:

```text
CE-only      Tests the classifier without multi-view consistency
Plain-KL     Tests whether clean/degraded consistency helps
Compact-KL   Tests whether reliability-aware weighting improves robustness
```

The key thesis claim should be supported by robust-view metrics, especially under API/graph degradation, Manifest corruption, missing evidence, and all-degraded views.

### 3. Source-Aware Graph Ablation

```bash
python run.py r1_graph
```

This group tests the contribution of typed/source/quality-aware graph modeling.

Typical ablations:

```text
code_only
manifest_only
no_edge_source
no_node_quality
no_edge_quality
no_alignment
no_risk_nodes
```

The strict single-source baselines use `masked_node_types` rather than an inactive high-level field.

For example, `code_only` masks Manifest-side and derived risk nodes:

```yaml
model:
  masked_node_types:
    - PERMISSION
    - COMPONENT
    - INTENT
    - RISK_SEMANTIC
```

`manifest_only` masks code-side and derived risk nodes:

```yaml
model:
  masked_node_types:
    - METHOD
    - API_FAMILY
    - STRING_HINT
    - RISK_SEMANTIC
```

`no_risk_nodes` separately tests the contribution of derived risk semantic nodes.

### 4. Fusion and Shortcut-Suppression Ablation

```bash
python run.py r3_fusion
```

This group tests the contribution of reliability-aware and conflict-aware latent fusion.

Typical ablations:

```text
no_reliability_bias
no_conflict_bias
mean_fusion
```

The most important robust views for this group are:

```text
manifest_noisy
manifest_shuffled
manifest_noisy_blind
manifest_shuffled_blind
manifest_missing
```

The goal is to show that the full fusion mechanism is more stable when Manifest evidence is corrupted, shuffled, or missing.

## Robust Evaluation

Robust evaluation is enabled by default in the experiment template.

Default robust views include:

```text
api_degraded
graph_degraded
api_graph_degraded
manifest_noisy
manifest_shuffled
manifest_noisy_blind
manifest_shuffled_blind
api_missing
graph_missing
manifest_missing
all_degraded
```

Recommended reported metrics:

```text
Clean Macro-F1
Robust Macro-F1 per view
Robust average Macro-F1
AUC
Average precision
ECE
Brier score
Robustness drop / degradation rate
```

The default checkpoint metric is `robust_composite`, which combines clean validation performance and representative robust validation views. Test and robust-test metrics are computed only after the best checkpoint is selected.

## Outputs

Each training run writes:

```text
best.pt
experiment_metadata.json
history.csv
summary.json
diagnostics_val.csv
diagnostics_test_clean.csv
diagnostics_test_<view>_<strength>.csv
```

`summary.json` contains the compact final metrics.

`history.csv` contains per-epoch training and validation metrics.

`experiment_metadata.json` is written at run start and updated on success or failure. It records:

```text
resolved config path
output directory
seed
runtime environment
git commit/branch/dirty status
dataset statistics
schema and payload contract versions
best epoch
best score
result files
final validation/test/robust-test metrics
```

## Diagnostics

Diagnostics include:

```text
prediction
malware probability
clean/perturbed view ids
requested/effective robust view names
reliability scalars
code-Manifest similarity
code-Manifest conflict
latent attention mass over evidence tokens
Manifest shuffle fallback status
Manifest donor sid when available
```

The latent attention columns include:

```text
attn_method
attn_api_family
attn_permission
attn_component
attn_risk
attn_string_hint
attn_global
```

These diagnostics are intended to support analysis of Manifest shortcut suppression and source-aware evidence usage.

Summarize diagnostics after a run:

```bash
python scripts/summarize_aeg_diagnostics.py \
  --input-dir results/aeg_robust/main/full_compact_kl \
  --min-count 20
```

## External Obfuscation Evaluation

For a real Obfuscapk-style benchmark, generate scenario PT files with the external extraction config, map obfuscated APK hashes back to clean test labels, and evaluate a trained checkpoint without retraining.

Example workflow:

```bash
python scripts/build_obfuscapk_label_csvs.py \
  --config config/extract_obfuscapk.yaml \
  --clean-labels results/labels/test.csv \
  --output-dir results/labels_obfuscapk

python scripts/evaluate_aeg_checkpoint.py \
  --checkpoint results/aeg_robust/main/full_compact_kl/best.pt \
  --config config/eval_obfuscapk.yaml \
  --output-dir results/aeg_robust/main/full_compact_kl/external
```

Report external robustness as:

```text
Clean test Macro-F1
Obfuscated test Macro-F1
Absolute drop
Relative drop
ECE change
```

## Tests

Run syntax checks:

```bash
python -m py_compile \
  fusion/losses.py \
  fusion/train.py \
  fusion/model.py \
  fusion/dataset.py \
  fusion/perturbations.py \
  run.py
```

Run smoke tests:

```bash
pytest tests/test_aeg_smoke.py
```

Check experiment paths:

```bash
python run.py final --dry-run
python run.py loss --dry-run
python run.py r1_graph --dry-run
python run.py r3_fusion --dry-run
python run.py all --dry-run
```

The dry-run output should reference only the current experiment structure:

```text
main/full_compact_kl.yaml
loss/ce_only.yaml
loss/plain_kl.yaml
loss/compact_kl.yaml
r1_graph/...
r3_fusion/...
```

It should not reference removed contrastive configs such as:

```text
full/ours.yaml
i1
i2
i3
multiview_contrast.yaml
no_clean_degraded_contrast.yaml
no_source_degraded_contrast.yaml
no_cross_source_contrast.yaml
```

## Thesis Experiment Mapping

Recommended result sections:

### RQ1: Does source-aware APK evidence graph modeling help?

Use:

```bash
python run.py r1_graph
```

Compare:

```text
Full AEG
Code-only
Manifest-only
No edge source
No node quality
No edge quality
No alignment
No risk nodes
```

### RQ2: Does reliability-aware KL consistency improve robustness?

Use:

```bash
python run.py loss
```

Compare:

```text
CE-only
Plain KL
Reliability-weighted KL
Compact-KL weight variants
```

### RQ3: Does reliability/conflict-aware fusion suppress Manifest shortcuts?

Use:

```bash
python run.py r3_fusion
```

Compare:

```text
Full fusion
No reliability bias
No conflict bias
Mean fusion
```

Focus on:

```text
manifest_noisy
manifest_shuffled
manifest_noisy_blind
manifest_shuffled_blind
manifest_missing
```

### RQ4: Is the full method robust under synthetic degradation and missing evidence?

Use the robust-test metrics emitted by the full method.

Report:

```text
clean
api_degraded
graph_degraded
api_graph_degraded
manifest_noisy
manifest_shuffled
missing evidence
all_degraded
```

### RQ5: Does the method generalize to real obfuscation or temporal drift?

Use external Obfuscapk evaluation or year-based test slices if available.

## Notes on Reproducibility

Use fixed seeds for comparable experiments:

```text
train.seed = 42
eval.seed = 2026
```

For three-seed reporting, create separate configs such as:

```text
main/full_compact_kl_seed43.yaml
main/full_compact_kl_seed44.yaml
```

Keep `eval.seed` fixed across seeds so robust-view sampling is comparable.

Recommended minimum report:

```text
mean ± std over 3 train seeds for the full method
single-seed screening for large ablation groups
three-seed confirmation for the strongest baselines
```

## Active Mainline

The active mainline is:

```text
Source-aware AEG
+ reliability-aware latent fusion
+ CE / Plain-KL / Compact-KL objectives
+ robust validation and diagnostics
```

The old contrastive-learning objective is no longer active. Avoid using or reintroducing experiment names based on:

```text
InfoNCE
multi-view contrastive learning
clean-degraded contrast
source-level contrast
cross-source contrast
counterfactual contrastive loss
```

Use the current terminology instead:

```text
multi-view KL consistency
reliability-aware consistency weighting
Manifest shortcut suppression
source-aware evidence graph modeling
```
