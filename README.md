# Source-Aware APK Evidence Graph Malware Detection

This repository contains the clean mainline for robust Android malware detection based on three ideas:

1. **Source-aware APK heterogeneous evidence graph modeling**: code evidence, Manifest declaration evidence, derived risk semantics, and alignment evidence are represented as typed nodes and typed edges in one APK evidence graph.
2. **Obfuscation-invariant reliability-weighted multi-view contrastive learning**: clean and degraded graph views are trained to stay close when perturbations preserve the APK label, while code/Manifest contrast is weighted by observable reliability.
3. **Counterfactual reliability-aware latent fusion with Manifest shortcut suppression**: latent fusion attends to code, Manifest, risk, and global graph tokens using reliability and conflict-aware bias so noisy declaration evidence is not treated as an unconditional anchor.

Previous experimental branches are no longer part of the active mainline.

## Layout

```text
extract/
  extract_graph_api.py          Reusable DEX/API/call-graph extractor

fusion/
  aeg_builder.py                Builds schema-v4 APK evidence graph payloads
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

The script writes `aeg_pt_index.csv` under `data.out_root`. Failed APKs are recorded in the same index with `status=failed`; generation does not stop unless `execution.fail_on_error=true`.

## Train

Edit `config/experiments/aeg_robust/base.yaml` so `data.{train,val,test}.pt_dir` and label CSV paths point to your generated AEG PT files.

Run the full method:

```bash
python -m fusion.train --config config/experiments/aeg_robust/full/ours.yaml
```

Or use the compact runner:

```bash
python run.py final
python run.py ablation
```

## Main Ablations

```bash
python -m fusion.train --config config/experiments/aeg_robust/ablation/no_clean_degraded_contrast.yaml
python -m fusion.train --config config/experiments/aeg_robust/ablation/no_cross_source_contrast.yaml
python -m fusion.train --config config/experiments/aeg_robust/ablation/no_counterfactual.yaml
python -m fusion.train --config config/experiments/aeg_robust/ablation/no_reliability_bias.yaml
python -m fusion.train --config config/experiments/aeg_robust/ablation/no_conflict_bias.yaml
python -m fusion.train --config config/experiments/aeg_robust/ablation/no_aug.yaml
```

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

Diagnostics include reliability scalars, code-Manifest similarity/conflict, and latent attention mass over code, Manifest, risk, and global tokens.
