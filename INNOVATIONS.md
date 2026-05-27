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
2. Manifest-guided cross-source soft consistency in a shared 12-D functional category space.
3. Conflict-aware asymmetric four-branch adaptive fusion.

## Semantic Category Space

API, Graph, and Manifest category counts must all use the same 12-D taxonomy:

```text
network, sms, location, contacts, storage, telephony,
camera_media, receiver, component_exposure, dynamic_loading,
crypto, system_settings
```

The robust model no longer compares raw API type-id histograms with Manifest categories. API type ids are mapped into this taxonomy, Graph counts are accepted only when already provided in this taxonomy, and Manifest counts are produced from the same category list.

`T7_tri_modal_full_soft_consistency.yaml` enables the trainable soft consistency loss. This loss uses lightweight semantic projection heads over API, Graph, and Manifest embeddings, then applies reliability-weighted cosine direction losses against category-count soft targets. It is category-level soft supervision, not Manifest hard-anchor embedding alignment.

## Manifest Vocabulary

Build Manifest vocabularies from train Manifest JSONL only:

```bash
python scripts/build_manifest_vocab_from_train.py --train-manifest-jsonl path/to/train_manifest.jsonl --vocab config/manifest_vocab.yaml
```

`scripts/augment_pts_with_manifest.py --build-vocab` is guarded so it only runs with `--split train` and an explicit `--train-jsonl-for-vocab`; val/test augmentation must load an existing train-built vocab.

For an end-to-end APK to tri-modal `.pt` build, use:

```bash
python scripts/build_tri_modal_pts.py --graph-config config/extract_graph_api.yaml --apk-root path/to/apks --graph-out-root path/to/pts_api_graph --tri-out-root path/to/pts_tri --rebuild-vocab
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

`year` can remain in datasets as metadata for statistics or split auditing, but it must not enter model forward, gate evidence, loss, or checkpoint selection.
