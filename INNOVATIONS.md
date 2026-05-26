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
2. Manifest-guided cross-source soft consistency in functional category space.
3. Conflict-aware asymmetric four-branch adaptive fusion.

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
