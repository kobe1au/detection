# Source-Aware APK Evidence Graph Malware Detection

This repository contains the clean mainline for robust Android malware detection based on three ideas:

1. **Source-aware APK heterogeneous evidence graph modeling**: code evidence, Manifest declaration evidence, derived risk semantics, and alignment evidence are represented as typed nodes and typed edges in one APK evidence graph.
2. **Obfuscation-invariant reliability-weighted multi-view contrastive learning**: clean and degraded graph views are trained to stay close when perturbations preserve the APK label, while code/Manifest contrast is weighted per sample by observable reliability and conflict.
3. **Counterfactual reliability-aware latent fusion with Manifest shortcut suppression**: latent fusion attends to method, API-family, permission, component, risk, static-hint, and global tokens using reliability, source, and conflict-aware bias so declaration evidence is not treated as an unconditional anchor.

Previous experimental branches are no longer part of the active mainline.

## Layout

```text
extract/
  extract_graph_api.py          Reusable DEX/API/call-graph extractor

fusion/
  aeg_builder.py                Builds schema-v6 APK evidence graph payloads
  constants.py                  AEG node/edge/source/view definitions
  dataset.py                    AEG Dataset and PyG batch collation
  manifest_features.py          Manifest parsing, vocab, vectorization
  model.py                      Typed graph encoder + reliability-aware latent fusion
  losses.py                     CE + multi-view contrast + counterfactual KL
  perturbations.py              AEG graph-view perturbations
  train.py                      Clean AEG training/evaluation entry

scripts/
  build_aeg_pts_direct.py       APK -> AEG .pt direct builder

config/
  extract_aeg.yaml              APK -> AEG extraction config
  experiments/aeg_robust/       Full method and ablation configs
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

If disk space cannot hold train PT files yet, build only the train-derived
Manifest vocabulary first:

```bash
python scripts/build_aeg_pts_direct.py \
  --config config/extract_aeg_vocab_train_only.yaml \
  --rebuild-vocab \
  --vocab-only \
  --workers 8
```

Then generate only validation and test PT files with the train vocabulary:

```bash
python scripts/build_aeg_pts_direct.py \
  --config config/extract_aeg_val_test.yaml \
  --no-rebuild-vocab \
  --resume \
  --workers 8
```

Generate train PT files later without rebuilding the frozen Manifest
vocabulary:

```bash
python scripts/build_aeg_pts_direct.py \
  --config config/extract_aeg_train_only.yaml \
  --no-rebuild-vocab \
  --resume \
  --workers 8
```

The script writes `aeg_pt_index.csv` under `data.out_root`. Failed APKs are recorded in the same index with `status=failed`; generation does not stop unless `execution.fail_on_error=true`. APKs not present in the configured label CSV are skipped and written to `aeg_ignored_apks.csv`. Resume validates the payload contract and node feature dimension before reusing existing PT files. Build fingerprints are kept as diagnostic metadata, not as a hard training-time blocker.

Extraction first scans invoke targets for all methods, selects the methods kept
by the configured graph budget, and only then builds expensive local CFG
features. Manifest extraction is performed inside the same APK build worker
when the train-only vocabulary is already frozen. `execution.hash_workers`
parallelizes exact SHA256 verification without trusting filenames; benchmark
`1/2/4` on the actual disk because excessive parallel reads can hurt HDDs.

The canonical config stores the final AEG plus compact quality/audit metadata,
but omits graph/API/Manifest intermediates that can be reconstructed from the
final typed graph (`aeg.retain_intermediate_features=false`). It also stores
large graph tensors as float16/int32/uint8 on disk
(`aeg.storage_dtype=float16`); the Dataset converts them back to float32/long
before training. Enable intermediate retention or float32 storage only for
extractor debugging because either substantially increases PT storage.

Every generated PT is checked against a versioned payload contract before it
is saved. Dataset loading and training preflight repeat that validation and
reject missing fields, invalid tensor shapes, schema mismatches, or mixed
contract versions instead of silently padding malformed samples. The
compact contract retains the typed graph, reliability values, semantic
aggregates, package/year metadata, Manifest parse status, multi-DEX extraction
status, and behavior-risk slice flags needed by the three proposed methods and
their robustness diagnostics.

The canonical extraction config disables API-derived graph behavior hints
(`graph.use_behavior_hints=false`) so API evidence is not leaked into graph
method features during `api_missing` or `api_degraded` evaluations. Use
`config/extract_aeg_behavior_hints.yaml` only as an explicit ablation. That
ablation sets `aeg.node_feature_dim=519` because hint channels are appended
after the base `2 * graph.vocab_size + 3` method feature; smaller dimensions
would truncate the hint channels and make the ablation ineffective.

Training uses strict CSV/PT integrity by default. If `results/labels/{train,val,test}.csv` contains ids without corresponding AEG `.pt` files, or a PT folder contains extra samples not in its CSV, training fails instead of silently changing the split size.
Training also rejects repeated `package_name` values across train/val/test by
default (`data.enforce_package_isolation=true`) to reduce package-level split
leakage.

## Train

Edit `config/experiments/aeg_robust/base.yaml` so `data.{train,val,test}.pt_dir` and label CSV paths point to your generated AEG PT files.

Run the full method:

```bash
python -m fusion.train --config config/experiments/aeg_robust/full/ours.yaml
```

Or use the compact runner:

```bash
python run.py final
python run.py i1
python run.py i2
python run.py i3
python run.py full_seeds
```

Checkpoint selection uses a validation-only composite score by default: clean validation macro-F1 plus representative robust validation views. Test and robust-test metrics are computed only after the best checkpoint is selected.

## Innovation Experiments

```bash
python run.py i1
python run.py i2
python run.py i3
python run.py full_seeds
```

`i1` isolates typed/source/quality-aware graph encoding under clean training.
`i2` isolates clean-degraded, source-level, and cross-source contrastive
objectives with counterfactual fusion disabled. `i3` reuses Full as its complete
endpoint and removes individual latent-fusion mechanisms. Run single-seed
screening first, then report three-seed mean and standard deviation for Full
and the strongest competing configurations.

## Outputs

Each training run writes:

```text
best.pt
history.csv
summary.json
diagnostics_val.csv
diagnostics_test_clean.csv
diagnostics_test_<view>_<strength>.csv
```

Diagnostics include reliability scalars, code-Manifest similarity/conflict, and latent attention mass over method, API-family, permission, component, risk, static-hint, and global tokens.

Aggregate synthetic robustness and real-failure slices after a run:

```bash
python scripts/summarize_aeg_diagnostics.py \
  --input-dir results/aeg_robust/full/ours \
  --min-count 20
```

For a real Obfuscapk benchmark, generate scenario PT files with
`config/extract_obfuscapk.yaml`, map the changed APK hashes back to the clean
test labels, then evaluate without retraining:

```bash
python scripts/build_obfuscapk_label_csvs.py \
  --config config/extract_obfuscapk.yaml \
  --clean-labels results/labels/test.csv \
  --output-dir results/labels_obfuscapk

python scripts/evaluate_aeg_checkpoint.py \
  --checkpoint results/aeg_robust/full/ours/best.pt \
  --config config/eval_obfuscapk.yaml \
  --output-dir results/aeg_robust/full/ours/external
```
