# Code Modification Plan - P0 Bug Fixes and Improvements

## Status: Implementation Plan

### ✅ Completed
1. Fixed `scripts/check_pts.py` - corrected path from file to directory

### 🔄 In Progress - P0 Fixes

#### P0-1: Config Portability (base.example.yaml)
**Location**: `config/experiments/aeg_robust/base.yaml`
**Action**: 
- Rename to `base.example.yaml`
- Add `base.yaml` to `.gitignore`
- Update `fusion/train.py:load_config()` to provide helpful error

#### P0-2: Small Batch Contrast Protection
**Location**: `fusion/train.py` and `fusion/losses.py`
**Action**:
- Add batch size validation in `run()` function
- Add `drop_last=True` when contrast is enabled
- Add runtime check in training loop

#### P0-3: Safe PT Loading
**Location**: `fusion/dataset.py:AEGDataset.__getitem__`
**Action**:
- Implement `_safe_load()` method with weights_only=True
- Add mmap support
- Add fallback for older PyTorch versions

#### P0-4: Manifest Shuffle Diagnostics
**Location**: `fusion/dataset.py` and `fusion/train.py`
**Action**:
- Add `manifest_shuffle_fallback` field to Data
- Add `effective_view` field
- Update diagnostics to record fallback events
- Add summary reporting after evaluation

#### P0-5: Experiment Metadata
**Location**: `fusion/train.py:run()`
**Action**:
- Create `experiment_metadata.json` with full environment info
- Record git commit, versions, resolved paths
- Add determinism flags

### 📋 Implementation Order

1. P0-1: Config (low risk, high value)
2. P0-3: Safe loading (security critical)
3. P0-2: Batch validation (prevents silent failure)
4. P0-5: Metadata (easy, no breaking changes)
5. P0-4: Shuffle diagnostics (complex, needs careful testing)

## Bug Verification Checklist

After each modification:
- [ ] Code runs without syntax errors
- [ ] Existing functionality unchanged
- [ ] New behavior documented in comments
- [ ] Edge cases handled
- [ ] No new bugs introduced

## Files to Modify

1. `config/experiments/aeg_robust/base.yaml` → rename to `base.example.yaml`
2. `.gitignore` - add base.yaml and *.local.yaml
3. `fusion/train.py` - load_config, batch validation, metadata
4. `fusion/dataset.py` - safe_load, shuffle diagnostics
5. `fusion/losses.py` - temperature minimum constant
6. `README.md` - security note, shuffle behavior

