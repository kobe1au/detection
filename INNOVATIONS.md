# Robust Tri-modal Fusion Mainline

Current mainline:

**Conflict-aware Soft-prior Tri-modal Fusion for Robust Android Malware Detection**

The repository now contains only the robust tri-modal mainline.

## Modalities

- API: explicit code behavior evidence.
- Graph: structured call-context evidence.
- Manifest: heterogeneous soft declaration prior.

Manifest is not a hard anchor and does not replace API or Graph. It only contributes reliability evidence, soft category-level consistency, and branch-level decision support.

## Innovations

1. Heterogeneous tri-modal reliability modeling for code common-mode failure.
2. Manifest-guided cross-source soft consistency in a shared, runtime-validated 12-D functional category space, with structural-context graph category alignment via `method_api_edge_index`.
3. Conflict-aware four-branch adaptive fusion with information-source asymmetry (behavior vs. declaration).

## Semantic Category Space

API, Graph, and Manifest category counts must all use the same 12-D taxonomy:

```text
network, sms, location, contacts, storage, telephony,
camera_media, receiver, component_exposure, dynamic_loading,
crypto, system_settings
```

The robust model no longer compares raw API type-id histograms with Manifest categories. API type ids are mapped into this taxonomy, Graph counts are accepted only when already provided in this taxonomy, and Manifest counts are produced from the same category list.

`T7_tri_modal_full_soft_consistency.yaml` enables the trainable soft consistency loss. This loss uses lightweight semantic projection heads over API, Graph, and Manifest embeddings, then applies reliability-weighted cosine direction losses against category-count soft targets. It is category-level soft supervision, not Manifest hard-anchor embedding alignment.

### Trust chain (runtime-validated)

The 12-D space is connected to the extractor through three hand-written tables that must stay consistent:

- `extract.extract_graph_api.API_CATEGORY_NAMES` — extractor's per-API-event type id space (id 0 reserved for `other` / unknown).
- `fusion.semantic_categories.DEFAULT_API_TYPE_ID_TO_CATEGORY` — mapping from extractor type ids into the 12-D taxonomy.
- `fusion.manifest_features.DEFAULT_CATEGORIES` — the canonical 12-D category list.

`validate_api_type_mapping()` is called at every model `__init__` and at the start of the direct tri-modal builder. It raises with a precise diff when a mapping key falls outside the extractor range or a mapping value is not in the 12-D taxonomy, preventing the three tables from silently drifting apart.

### Structural-context graph category alignment

The Graph branch is structural; reusing the full API histogram as its semantic distribution would conflate "events present anywhere in the sample" with "events that the call graph actually carries". `graph_semantic_counts_from_method_api_edges` in `fusion/semantic_categories.py` instead aggregates only the API events anchored to graph methods through `method_api_edge_index`. This is the default Graph semantic source.

Three sources are exposed for ablation through `data.graph_semantic_source` in the dataset config:

- `alignment` (default) — only `method_api_edge_index`-anchored events contribute to graph counts.
- `full_api` — graph counts are a verbatim copy of the API histogram (ablation baseline showing what the alignment-based design adds).
- `zero` — graph counts forced to all zeros (sanity baseline isolating cross-modal leakage from soft consistency / gate evidence).

Reference configs: `config/experiments/tri_modal_robust/base_tri_modal_robust_graph_full_api.yaml` and `..._graph_zero.yaml`.

## Manifest Vocabulary

Build Manifest vocabularies from train Manifest JSONL only:

```bash
python scripts/build_manifest_vocab_from_train.py --train-manifest-jsonl path/to/train_manifest.jsonl --vocab config/manifest_vocab.yaml
```

`scripts/augment_pts_with_manifest.py --build-vocab` is guarded so it only runs with `--split train` and an explicit `--train-jsonl-for-vocab`; val/test augmentation must load an existing train-built vocab.

For an end-to-end APK to tri-modal `.pt` build, use the direct builder. The old staged `scripts/build_tri_modal_pts.py` entry is deprecated and exits with an error.

```bash
python scripts/build_tri_modal_pts_direct.py --config config/extract_tri_model.yaml --apk-root path/to/apks --out-root path/to/pts_tri --rebuild-vocab
```

The direct builder writes `tri_modal_pt_index.csv` under `out-root` with `split, sha256, apk_name, apk_path, pt_path, status, reason`, which should be used when merging labels into train/val/test CSVs.

After building `.pt` files, check semantic coverage before training:

```bash
python scripts/summarize_tri_modal_pts.py --pt-root path/to/pts_tri
```

## Robust Fusion

The model uses four decision branches:

- API branch
- Graph branch
- Manifest branch
- Joint branch

Final logits:

```text
logits =
  w_api * api_logits
+ w_graph * graph_logits
+ w_manifest * manifest_logits
+ w_joint * joint_logits
```

Gate evidence includes only:

- modality quality
- perturbation strength
- reliability
- branch confidence
- API/Graph disagreement
- API/Manifest and Graph/Manifest consistency
- modality alive signals

### Information-source asymmetry

"Asymmetric" here refers to the *information source* type, not architectural connectivity. The four branches are wired symmetrically (each produces logits, each receives a gate weight), but the three modalities play different evidential roles:

- API and Graph carry *behavior* evidence — what the code actually does. They participate in conflict detection (`api_graph_disagreement`) and drive the joint branch as primary signal sources.
- Manifest carries *declarative* evidence — what the app claims it can do. It contributes (a) an independent branch logit, (b) reliability-weighted cosine consistency against API/Graph in the shared 12-D space, and (c) `q_manifest` / `pert_manifest` / consistency terms inside the gate evidence vector. The gate's initialization bias favors the joint branch (`bias[3] = 0.5`, others `0`), which softly de-emphasizes any single-modality decision — including a Manifest-only one — before learned evidence accumulates.

This asymmetry is expressed through the evidence dimensions and loss terms, not by removing the Manifest branch from the prediction path. Keeping the Manifest branch alive preserves `manifest_only` baselines and direct observation of `w_manifest` under Manifest perturbation.

`year` can remain in datasets as metadata for statistics or split auditing, but it must not enter model forward, gate evidence, loss, or checkpoint selection.
