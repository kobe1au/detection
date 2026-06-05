# Conflict-aware Soft-prior Tri-modal Fusion for Android Malware Detection

This repository implements a robust tri-modal static Android malware detector based on:

- **API**: explicit code behavior evidence.
- **Graph**: structured call-context evidence.
- **Manifest**: heterogeneous soft declaration prior.

The current mainline is **not** a time-drift / DBTA / continual-adaptation project. The code is organized around robust degradation, soft-prior reliability modeling, semantic consistency, and conflict-aware adaptive fusion.

## Core Idea

API sequences and call graphs are different representations, but both are extracted from DEX/static code analysis. Under reflection, dynamic loading, packing, string encryption, call indirection, dead-code injection, and control-flow obfuscation, API and Graph can degrade together. This is the code-source common-mode failure problem.

Manifest provides a heterogeneous declaration view, but it is noisy: permissions may be redundant, merged from SDKs, declared but unused, or incompletely parsed. Therefore Manifest is used as a **soft declaration prior**, not as a hard anchor.

The model uses sample-level reliability evidence and conflict-aware gate weights to decide how much to trust API, Graph, Manifest, and Joint branches.

## Innovations

1. **Heterogeneous tri-modal reliability modeling for code common-mode failure**
   - Models `q_api`, `q_graph`, `q_manifest` (sample-level quality scores).
   - Models `pert_api`, `pert_graph`, `pert_manifest` (synthetic perturbation strength; oracle metadata unavailable for real-world APKs).
   - Default reliability: `r = q` (perturbation strength is unobservable in practice).
   - When `use_perturbation_evidence=True` (ablations only): `r = q * (1 - pert)`.

2. **Manifest-guided cross-source soft consistency learning**
   - API, Graph, and Manifest are mapped into a shared 12-D semantic category space.
   - Consistency is used as gate evidence and optional soft category supervision.
   - No Manifest hard-anchor embedding alignment is used.

3. **Conflict-aware asymmetric four-branch adaptive fusion**
   - API branch -> `api_logits`
   - Graph branch -> `graph_logits`
   - Manifest branch -> `manifest_logits`
   - Joint branch -> `joint_logits`
   - Final prediction is reliability-gated logit fusion.

## Fusion Type

This project is best described as:

```text
reliability-aware adaptive late fusion
```

More precisely:

```text
asymmetric hybrid late fusion with a joint representation branch
```

The final fusion is logit-level:

```text
logits =
  w_api      * api_logits
+ w_graph    * graph_logits
+ w_manifest * manifest_logits
+ w_joint    * joint_logits
```

There is no cross-modal cross-attention in the current main model. API may use Transformer self-attention internally, and Graph can use graph/self-attention encoders, but API/Graph/Manifest do not attend to each other token-by-token.

## Repository Layout

```text
fusion/
  dataset.py                  Robust tri-modal dataset and collate logic
  model.py                    API/Graph/Manifest/Joint model and gate
  losses.py                   CE, branch auxiliary, soft consistency, gate prior
  perturbations.py            API/Graph/Manifest degradation protocols
  semantic_categories.py      Shared 12-D semantic taxonomy and mappings
  manifest_features.py        Manifest vocabulary and vectorization
  train.py                    Training and clean/robust evaluation entrypoint

extract/
  extract_graph_api.py        API and call-graph extractor

scripts/
  build_tri_modal_pts_direct.py       APK -> tri-modal .pt direct builder
  summarize_robust_dataset_quality.py Pre-training data quality statistics
  summarize_tri_modal_pts.py          Semantic coverage summary
  filter_csv_by_existing_pts.py       Filter label CSVs by generated .pt files
  build_real_failure_slices.py        Real low-quality/failure slice CSVs
  make_manifest_shortcut_controls.py  Manifest shuffled/zeroed/noisy controls
  run_obfuscapk_benchmark.py          Real obfuscation benchmark generator

config/
  extract_tri_model.yaml
  experiments/tri_modal_robust/
```

## Design Notes

**Graph branch input separation.** The canonical setup uses 515-D structural node features. API-derived 4-D node hints are disabled during extraction, dropped from legacy 519-D PT files, and disabled in graph readout. The extracted subgraph is still selected around sensitive API seeds, and Graph semantic counts are alignment-derived, so Graph must not be described as independent of API evidence. A behavior-hint ablation requires a separately generated 519-D PT dataset.

**Reliability evidence.** The default main method uses `r = q` — only observable quality scores. Synthetic perturbation strength `pert` is oracle metadata that is unavailable for real-world APKs. The `use_perturbation_evidence` flag gates `r = q * (1 - pert)` exclusively for controlled ablations.

**Semantic category perturbations.** API category counts are recomputed after API edits. New PT schema files also store train-vocabulary term-to-category maps, allowing Manifest permission/intent edits to update semantic counts coherently. Graph structural edits do not randomly modify semantic counts. Synthetic `pert_*` values remain diagnostics and are not exposed to the main gate.

## Environment

Use Python 3.10. A typical setup is:

```bash
conda create -n malware python=3.10
conda activate malware
pip install -r requirements.txt
```

If installing PyTorch / PyG manually, match the CUDA version on the training machine. Verify the environment before running experiments:

```bash
python -c "import torch; import torch_geometric; print(torch.__version__, torch.cuda.is_available())"
python -m fusion.train --help
```

## Data Format

Training uses `.pt` files plus split CSVs.

Expected split CSV columns:

```text
sha256,label
```

The dataset matches rows by `.pt` filename stem, usually `{sha256}.pt`.

The tri-modal `.pt` files should contain API, Graph, Manifest, reliability, and metadata fields. Important fields include:

```text
api_ids
api_type_ids
api_semantic_category_counts

call_x
call_edge_index
method_api_edge_index
graph_semantic_category_counts

manifest_x
manifest_category_counts
manifest_stats

q_api
q_graph
q_manifest
q_align
pert_api
pert_graph
pert_manifest

label
year
sid / sha256
```

`year` is metadata only. It must not enter the model, gate, loss, or checkpoint selection.

## Build APK -> Tri-modal PT

Edit:

```text
config/extract_tri_model.yaml
```

Important fields:

```yaml
data:
  split_dirs:
    train: 
    val: 
    test: 
  out_root: 
  out_dirs:
    train: 
    val: 
    test: 

manifest:
  vocab_path: 
  rebuild_vocab: true

execution:
  workers: 4
  resume: false
  fail_on_error: false
```

First full generation:

```bash
python scripts/build_tri_modal_pts_direct.py \
  --config config/extract_tri_model.yaml \
  --rebuild-vocab \
  --no-resume \
  --allow-failures \
  --workers 8
```

Resume after interruption:

```bash
python scripts/build_tri_modal_pts_direct.py \
  --config config/extract_tri_model.yaml \
  --no-rebuild-vocab \
  --resume \
  --allow-failures \
  --workers 8
```

The direct builder records failures instead of exiting when `fail_on_error=false` or `--allow-failures` is used. Failed APKs are written to the index/failed output and should be removed from the corresponding label CSV before training.

Each new PT stores a schema/config/vocabulary fingerprint. Resume skips only matching PTs. Legacy PTs without a fingerprint are rejected unless `execution.allow_legacy_resume=true` is explicitly set. Do not enable that option for formal experiments.

Current-schema PTs also contain Manifest term-to-category maps, component-derived category counts, and continuous Manifest extraction coverage. The formal experiment base config locks `data.min_pt_schema_version: 3`, so all I1/I2/I3/Full results use one consistent PT schema. Regenerate PTs before formal experiments; yesterday's legacy-PT results remain pilot results only.

The builder writes:

```text
tri_modal_pt_index.csv
failed_tri_modal_direct.json
manifest_vocab.yaml
_manifest_jsonl/
```

Do not rebuild the Manifest vocabulary while resuming existing `.pt` files. A new vocabulary with old `.pt` files can silently make Manifest vector dimensions/semantics inconsistent.

## Filter CSVs After PT Generation

If your `.pt` files are under a single root such as:

```text
D:/pts/train
D:/pts/val
D:/pts/test
```

filter label CSVs by existing PT files:

```bash
python scripts/filter_csv_by_existing_pts.py \
  --csv-dir results/labels \
  --pt-root D:/pts \
  --splits train val test
```

If split outputs are on different disks, use `tri_modal_pt_index.csv` to audit successful PTs and update `results/labels/{train,val,test}.csv` accordingly.

## Dataset Quality Check

Before training, run:

```bash
python scripts/summarize_robust_dataset_quality.py \
  --config config/experiments/tri_modal_robust/base_tri_modal_robust.yaml \
  --splits train val test \
  --write-rows
```

Check at least:

```text
num_samples per split
label ratio per split
q_api mean/std/p10/p50/p90
q_graph mean/std/p10/p50/p90
q_align mean/std/p10/p50/p90
q_manifest mean/std/p10/p50/p90
api_semantic_nonzero_ratio
graph_semantic_nonzero_ratio
manifest_semantic_nonzero_ratio
manifest_parse_error_ratio
multi_dex_partial_failed_ratio
```

Also check semantic coverage:

```bash
python scripts/summarize_tri_modal_pts.py \
  --pt-root D:/pts \
  --splits train val test
```

Do not start formal training until CSV/PT matching and semantic coverage are clean.

## Training

Main entrypoint:

```bash
python -m fusion.train --config config/experiments/tri_modal_robust/full/ours.yaml
```

Multiple configs can be overlaid left to right:

```bash
python -m fusion.train \
  --config config/experiments/tri_modal_robust/base_tri_modal_robust.yaml \
           path/to/override.yaml
```

Outputs are written under:

```text
experiments/tri_modal_robust/{exp_name}/{seed}/
```

Important output files:

```text
best_tri_modal_robust.pt
summary.yaml
gate_diagnostics.csv
gate_diagnostics_extra_eval.csv
metrics_extra_eval.json
```

The primary checkpoint metric is macro-F1. The training code reports:

```text
f1          = macro_f1
macro_f1
f1_pos
macro_recall
recall_pos
acc
auc
ap
```

Do not use positive-class F1 as the main robustness metric. In missing-modality cases, a model can predict only one class and still get a misleading positive-class F1.

## Main Experiment Matrix

Recommended order:

```bash
python -m fusion.train --config config/experiments/tri_modal_robust/full/ours.yaml

python -m fusion.train --config config/experiments/tri_modal_robust/i1/api_only.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i1/graph_only.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i1/manifest_only.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i1/api_graph_concat.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i1/tri_modal_concat.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i1/reliability_gate.yaml

python -m fusion.train --config config/experiments/tri_modal_robust/i2/no_consistency.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i2/evidence_only.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i2/loss_only.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i2/evidence_plus_loss.yaml

python -m fusion.train --config config/experiments/tri_modal_robust/i3/fixed_gate.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i3/confidence_gate.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i3/reliability_gate.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i3/learned_gate_no_prior.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i3/learned_gate_with_prior.yaml
```

Baselines:

```text
i1/api_only API-only
i1/graph_only Graph-only
i1/manifest_only Manifest-only
i1/api_graph_concat API+Graph concat
i1/tri_modal_concat API+Graph+Manifest concat
i3/fixed_gate Tri-modal fixed gate
T6 Tri-modal reliability gate
T7 Full method
```

Ablations:

```text
T4b API+Graph+Manifest concat + robust augmentation
T7 full no augmentation
T7 full no gate prior
T7 full no manifest auxiliary branch
graph_full_api semantic-source ablation
graph_zero semantic-source ablation
```

## Robust Evaluation

The base config evaluates:

```text
clean
api_degraded
graph_degraded
api_graph_degraded
manifest_degraded
all_degraded
api_missing
graph_missing
manifest_missing
```

with strengths:

```text
0.1, 0.3, 0.5, 0.7, 0.9
```

Use `summary.yaml` for metrics and `gate_diagnostics.csv` for per-sample gate/evidence analysis.

Key claims to verify:

```text
API degraded strength increases      -> w_api decreases
Graph degraded strength increases    -> w_graph decreases
Manifest degraded strength increases -> w_manifest decreases
API+Graph degraded                   -> Manifest/Joint relative support increases
All degraded                         -> Ours drops less than concat
```

## Manifest Shortcut Control

Manifest-only can be strong. That is not automatically a flaw, but it creates a shortcut risk. Generate controls:

```bash
python scripts/make_manifest_shortcut_controls.py \
  --config config/experiments/tri_modal_robust/full/ours.yaml \
  --splits test \
  --controls shuffled zeroed noisy \
  --resume
```

Then evaluate the already-trained checkpoint with a small override YAML that sets `eval.eval_only`, `eval.checkpoint_path`, and the generated control sets under `eval.extra_sets`:

```bash
python -m fusion.train \
  --config config/experiments/tri_modal_robust/full/ours.yaml path/to/manifest_control_eval.yaml
```

Expected behavior:

```text
Concat should be more sensitive to Manifest corruption.
Ours should avoid blindly relying on Manifest when q/pert/alive/consistency indicate conflict or degradation.
```

## Real Failure Slices

Synthetic perturbation is not enough for a strong robustness paper. Build real low-quality/failure slices from generated PTs:

```bash
python scripts/build_real_failure_slices.py \
  --config config/experiments/tri_modal_robust/full/ours.yaml \
  --splits test \
  --write-empty
```

Evaluate the locked checkpoint with an override YAML that sets `eval.eval_only: true`, `eval.checkpoint_path`, and the slice CSVs under `eval.extra_sets`. Extra-set CSVs may select a subset of a larger PT directory; missing CSV samples are still rejected.

```bash
python -m fusion.train \
  --config config/experiments/tri_modal_robust/full/ours.yaml path/to/failure_slice_eval.yaml
```

Useful slices include:

```text
api_low_quality
graph_low_quality
align_low_quality
manifest_low_quality
code_common_failure
graph_semantic_missing
manifest_parse_failed
multi_dex_partial_failed
```

## Obfuscation Benchmark

If Obfuscapk is available, generate real obfuscated APKs:

```bash
python scripts/run_obfuscapk_benchmark.py \
  --csv results/labels/test.csv \
  --apk-dir D:/resource/test \
  --out-root D:/obfuscapk_benchmark \
  --limit 200 \
  --techniques reflection call_indirection string_encrypt manifest_noise \
  --resume
```

Build PTs for obfuscated APKs:

```bash
python scripts/build_tri_modal_pts_direct.py \
  --config config/extract_obfuscapk.yaml \
  --no-rebuild-vocab \
  --resume \
  --allow-failures \
  --workers 8
```

Evaluate:

```bash
python -m fusion.train \
  --config config/experiments/tri_modal_robust/full/ours.yaml path/to/obfuscapk_eval.yaml
```

## Interpreting Current Baselines

A healthy pattern is:

```text
Graph-only    >= API-only >= Manifest-only
```

Manifest-only should be useful but not dominant. If Manifest-only exceeds API/Graph by a large margin, audit shortcut leakage:

```text
family leakage
duplicate APKs
SDK/signature artifacts
train/test split contamination
Manifest-only label artifacts
```

For single-modality baselines:

```text
API-only should collapse on api_missing.
Graph-only should collapse on graph_missing.
Manifest-only should collapse on manifest_missing.
Non-target modality perturbations should not change predictions.
```

These checks are stronger than clean accuracy alone.

## Common Pitfalls

- Do not build Manifest vocab from train+val+test. Build it from train only.
- Do not report positive-class F1 as the main robustness metric. Use macro-F1.
- Do not treat Manifest as ground truth. It is a noisy declaration prior.
- Do not interpret synthetic perturbation as sufficient real-world robustness. Add failure slices and, if possible, Obfuscapk.
- Do not rebuild vocab while resuming old PT generation.
- Do not let `year` enter model/gate/loss/checkpoint logic.
- Do not compare raw API type histograms with Manifest categories. Use the shared 12-D semantic taxonomy.

## Quick Sanity Commands

```bash
python -m pytest tests/test_robust_smoke.py -q

python scripts/summarize_robust_dataset_quality.py \
  --config config/experiments/tri_modal_robust/base_tri_modal_robust.yaml \
  --splits train val test

python -m fusion.train \
  --config config/experiments/tri_modal_robust/i1/api_only.yaml
```

If these fail, fix the data/config/runtime issue before running the full experiment matrix.
