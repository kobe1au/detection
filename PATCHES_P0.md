# Complete Code Patches for Remaining P0 Fixes

## ⚠️ CRITICAL: Review Before Applying

This document contains all code modifications needed. Please review carefully before applying to prevent bugs.

---

## P0-2: Small Batch Contrast Protection

### File: `fusion/losses.py` (Line 10, add constant)

```python
# Add after imports, before _info_nce function
MIN_TEMPERATURE = 1e-3  # Minimum temperature to prevent gradient explosion
```

### File: `fusion/losses.py` (Line 16-17, update)

```python
# BEFORE:
logits_ab = a @ b.t() / max(float(temperature), 1e-4)
logits_ba = b @ a.t() / max(float(temperature), 1e-4)

# AFTER:
logits_ab = a @ b.t() / max(float(temperature), MIN_TEMPERATURE)
logits_ba = b @ a.t() / max(float(temperature), MIN_TEMPERATURE)
```

### File: `fusion/train.py` (add to load_config function, after line 49)

```python
def load_config(path: str | Path) -> dict[str, Any]:
    """Load config with base inheritance.
    
    Raises FileNotFoundError with helpful message if config not found but .example exists.
    """
    path = Path(path)
    if not path.exists():
        example = path.parent / f"{path.stem}.example{path.suffix}"
        if example.exists():
            raise FileNotFoundError(
                f"Config not found: {path}\n"
                f"Copy {example.name} to {path.name} and update paths for your environment."
            )
        raise FileNotFoundError(f"Config not found: {path}")
    
    # ... rest of existing code
```

### File: `fusion/train.py` (update _loader function, around line 175)

```python
def _loader(cfg: dict[str, Any], dataset: AEGDataset, *, train: bool) -> DataLoader:
    train_cfg = cfg.get("train", {}) or {}
    batch_size = int(train_cfg.get("batch_size" if train else "eval_batch_size", train_cfg.get("batch_size", 24)))
    workers = int(train_cfg.get("num_workers", 0))
    
    # P0-2: Check if contrastive learning is enabled
    loss_cfg = cfg.get("loss", {}) or {}
    contrast_enabled = any(
        float(loss_cfg.get(k, 0.0)) > 0.0
        for k in [
            "clean_degraded_contrast_weight",
            "source_degraded_contrast_weight", 
            "cross_source_contrast_weight",
        ]
    )
    
    # P0-2: Drop last batch when contrast is enabled to ensure batch_size >= 2
    drop_last = False
    if train and contrast_enabled:
        drop_last = True
        LOGGER.info(
            "Contrastive learning enabled: setting drop_last=True "
            "to ensure all batches have size >= 2"
        )
    
    generator = torch.Generator()
    train_seed = int(train_cfg.get("seed", 42))
    eval_seed = int((cfg.get("eval", {}) or {}).get("seed", 2026))
    generator.manual_seed(train_seed if train else eval_seed)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=drop_last,  # ← ADD THIS
        num_workers=workers,
        pin_memory=bool(train_cfg.get("pin_memory", False)),
        persistent_workers=workers > 0 and (
            train or bool(train_cfg.get("persistent_eval_workers", False))
        ),
        generator=generator,
        worker_init_fn=_seed_worker if workers > 0 else None,
        collate_fn=aeg_collate_fn,
    )
```

### File: `fusion/train.py` (add validation in run() function, after line 446)

```python
def run(cfg: dict[str, Any]) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO)
    train_cfg = cfg.get("train", {}) or {}
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    device = _device(cfg)
    out_dir = Path(train_cfg.get("output_dir", "results/aeg_robust/run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # P0-2: Validate batch size for contrastive learning
    loss_cfg = cfg.get("loss", {}) or {}
    contrast_weights = [
        float(loss_cfg.get("clean_degraded_contrast_weight", 0.1)),
        float(loss_cfg.get("source_degraded_contrast_weight", 0.05)),
        float(loss_cfg.get("cross_source_contrast_weight", 0.03)),
    ]
    batch_size = int(train_cfg.get("batch_size", 24))
    
    if any(w > 0 for w in contrast_weights) and batch_size < 2:
        raise ValueError(
            f"Contrastive learning requires batch_size >= 2, got {batch_size}. "
            f"InfoNCE will return 0 when batch_size=1. "
            f"Either increase batch_size or disable contrast (set weights to 0)."
        )
    
    # ... rest of existing code
```

### File: `fusion/train.py` (add runtime check in train_one_epoch, after line 328)

```python
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict[str, Any],
    epoch: int,
) -> dict[str, float]:
    model.train()
    use_aug = bool((cfg.get("robust", {}) or {}).get("train_aug", True))
    grad_clip = float((cfg.get("train", {}) or {}).get("grad_clip", 1.0))
    
    # P0-2: Check contrast enabled for runtime validation
    loss_cfg = cfg.get("loss", {}) or {}
    contrast_enabled = any(
        float(loss_cfg.get(k, 0.0)) > 0.0
        for k in [
            "clean_degraded_contrast_weight",
            "source_degraded_contrast_weight",
            "cross_source_contrast_weight",
        ]
    )
    
    totals: dict[str, float] = {}
    steps = 0
    for batch in tqdm(loader, desc=f"train {epoch}", leave=False):
        batch = _move_batch(batch, device)
        
        # P0-2: Runtime check for batch size
        local_bsz = int(batch["clean"].y.numel())
        if contrast_enabled and local_bsz < 2:
            raise ValueError(
                f"Contrastive learning requires local batch size >= 2, got {local_bsz}. "
                f"Increase batch_size, enable drop_last, or set contrast weights to 0."
            )
        
        # ... rest of existing code
```

---

## P0-3: Safe PT Loading

### File: `fusion/dataset.py` (add method to AEGDataset class, after __init__)

```python
def _safe_load(self, path: Path) -> dict:
    """Load AEG PT file with defense-in-depth.
    
    Defense layers:
    - Primary: weights_only=True restricts pickle deserialization (PyTorch 2.0+)
    - Secondary: validate_payload_on_load checks schema/structure post-load
    
    Note: Neither layer prevents content-level semantic tampering.
    For untrusted sources, verify external checksums before loading.
    """
    try:
        # PyTorch 2.0+: safe deserialization + memory-mapped loading
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        try:
            # Fallback 1: weights_only without mmap
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            # Fallback 2: legacy mode (no pickle safety)
            import logging
            logging.getLogger(__name__).warning(
                "This PyTorch version does not support weights_only/mmap. "
                "Loading %s with legacy torch.load; only use trusted PT files.",
                path.name,
            )
            payload = torch.load(path, map_location="cpu")
    
    # Secondary validation (post-load)
    if self.validate_payload_on_load:
        validate_aeg_payload(payload)
    
    return payload
```

### File: `fusion/dataset.py` (update __getitem__ method, around line 258)

```python
def __getitem__(self, idx: int) -> dict[str, Data]:
    path, label = self.samples[idx]
    payload = self._safe_load(path)  # ← CHANGE from torch.load()
    clean = payload_to_data(payload, label=label, validate_payload=self.validate_payload_on_load)
    # ... rest of existing code
```

### File: `fusion/dataset.py` (update donor loading, around line 274)

```python
if donor_idx is not None:
    donor_path, donor_label = self.samples[donor_idx]
    donor_payload = self._safe_load(donor_path)  # ← CHANGE from torch.load()
    out["manifest_donor"] = payload_to_data(
        donor_payload,
        label=donor_label,
        validate_payload=self.validate_payload_on_load,
    )
```

---

## P0-5: Experiment Metadata

### File: `fusion/train.py` (add imports at top)

```python
import socket
import subprocess
import sys
from datetime import datetime
```

### File: `fusion/train.py` (add in run() function, before training loop)

```python
def run(cfg: dict[str, Any]) -> dict[str, Any]:
    # ... existing setup code ...
    
    # P0-5: Record complete experiment metadata
    try:
        import torch_geometric
        pyg_version = torch_geometric.__version__
    except (ImportError, AttributeError):
        pyg_version = None
    
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent.parent,
        ).decode().strip()
    except Exception:
        git_commit = None
    
    try:
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent.parent,
        ).decode().strip()
    except Exception:
        git_branch = None
    
    try:
        git_dirty = len(subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent.parent,
        )) > 0
    except Exception:
        git_dirty = None
    
    experiment_metadata = {
        "config": cfg,
        "resolved_paths": {
            "train_pt_dir": str(Path(train_ds.pt_dir).resolve()),
            "val_pt_dir": str(Path(val_ds.pt_dir).resolve()),
            "output_dir": str(out_dir.resolve()),
        },
        "data": {
            "num_train": len(train_ds),
            "num_val": len(val_ds),
            "train_stats": split_label_stats(train_ds),
            "val_stats": split_label_stats(val_ds),
            "aeg_build_fingerprint": aeg_build_fingerprint,
        },
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "torch_geometric_version": pyg_version,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "git": {
            "commit": git_commit,
            "branch": git_branch,
            "dirty": git_dirty,
        },
        "reproducibility": {
            "seed": seed,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "command": " ".join(sys.argv),
    }
    
    (out_dir / "experiment_metadata.json").write_text(
        json.dumps(experiment_metadata, indent=2, default=str),
        encoding="utf-8",
    )
    LOGGER.info("Experiment metadata saved to %s", out_dir / "experiment_metadata.json")
    
    # ... continue with training loop ...
```

---

## Verification Checklist

After applying patches:

- [ ] Run `python scripts/check_pts.py` - should work with directory path
- [ ] Load base.example.yaml without base.yaml - should show helpful error
- [ ] Train with batch_size=1 and contrast enabled - should fail fast
- [ ] Train with batch_size=24 - should work normally
- [ ] Check experiment_metadata.json is created with full info
- [ ] Verify PT files load with weights_only on PyTorch 2.0+
- [ ] Test on older PyTorch - should fallback gracefully

---

## Next Steps

1. Review all patches carefully
2. Apply patches one at a time
3. Test after each patch
4. Commit after successful tests
5. Update README with security note

