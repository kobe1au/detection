#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run only the adaptation phase from a shared historical checkpoint.

This entrypoint is for I1 replay/ratio ablations where all variants share the
same historical 2018-2021 checkpoint. It leaves `python -m fusion.train`
unchanged and supports either a CLI checkpoint or warm-start keys in the
override YAML:

train:
  warm_start_historical_ckpt: experiments/i1_zero_adapt_concat/42/best_i1_zero_adapt_concat.pt
  start_phase: adaptation
"""

from __future__ import annotations

import argparse
import copy
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fusion.constants import TrainingConstants
from fusion.mm_dataset import MultiModalMalwareDataset, hierarchical_collate_fn
from fusion.model import MalwareModelWithXAttn
from fusion.selective_metrics import aurc, eaurc
from fusion.train import (
    _binary_detection_metrics,
    _stage_loss_cfg,
    append_metrics_csv,
    build_adaptation_loader,
    build_global_domain_years,
    build_grad_scaler,
    build_year_subset_loaders,
    collect_calibrated_outputs,
    compute_temporal_selection_score,
    deep_update,
    eval_one_epoch,
    evaluate_temporal_windows,
    init_metrics_csv,
    load_yaml_file,
    preflight_validate_protocol,
    resolve_path,
    save_checkpoint,
    save_config_snapshot,
    save_run_metadata,
    select_device,
    set_seed,
    setup_logger,
    train_one_epoch,
    validate_full_config,
    WarmupCosineScheduler,
)
from fusion.calibration import TemperatureScaling


def _strip_warm_start_keys(cfg: dict) -> tuple[dict, dict]:
    cfg_for_validation = copy.deepcopy(cfg)
    train_cfg = cfg_for_validation.setdefault("train", {})
    warm_keys = {}
    for key in ("warm_start_historical_ckpt", "start_phase"):
        if key in train_cfg:
            warm_keys[key] = train_cfg.pop(key)
    return cfg_for_validation, warm_keys


def _build_optimizer_scheduler(model, train_cfg: dict, remaining_epochs: int):
    base_lr = float(train_cfg["lr"])
    wd = float(train_cfg["weight_decay"])
    warmup = int(train_cfg["warmup_epochs"])
    eta_min = float(train_cfg["eta_min"])
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd)
    after = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(int(remaining_epochs) - warmup, 1),
        eta_min=eta_min,
    )
    return opt, WarmupCosineScheduler(opt, warmup, after)


def _load_ckpt(path: str, model, device, logger):
    if not path:
        raise ValueError("historical checkpoint is required")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Historical checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state.get("state_dict", state))
    logger.info(f"Loaded shared historical checkpoint: {path}")


def _make_dataset(c_data, fusion_mode, need_alignment_mask, domain_years, drop_hints, csv_path, pt_dir, is_train, data_root):
    return MultiModalMalwareDataset(
        pt_dir=resolve_path(data_root, pt_dir) if not os.path.isabs(str(pt_dir)) else str(pt_dir),
        csv_path=csv_path,
        is_train=is_train,
        robust_aug=False,
        max_api_events_per_sample=c_data["max_api_events_per_sample"],
        fusion_mode=fusion_mode,
        need_alignment_mask=need_alignment_mask,
        domain_years=domain_years,
        drop_graph_behavior_hints=drop_hints,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True)
    ap.add_argument("--override", type=str, required=True)
    ap.add_argument("--historical-ckpt", type=str, default="")
    args = ap.parse_args()

    raw_cfg = deep_update(load_yaml_file(args.base), load_yaml_file(args.override))
    cfg_for_validation, warm_keys = _strip_warm_start_keys(raw_cfg)
    cfg = validate_full_config(cfg_for_validation)

    c_data, c_model, c_train, c_loss = cfg["data"], cfg["model"], cfg["train"], cfg["loss"]
    warm_ckpt = args.historical_ckpt or str(warm_keys.get("warm_start_historical_ckpt") or "")
    start_phase = str(warm_keys.get("start_phase") or "adaptation").lower()
    if start_phase != "adaptation":
        raise ValueError("train_adapt_from_ckpt only supports start_phase=adaptation")
    if not c_data.get("adapt_csv"):
        raise ValueError("adaptation-only training requires data.adapt_csv")

    data_root = os.getenv("DATA_ROOT", "")
    set_seed(int(c_train["seed"]))
    device = select_device(c_train["device"])
    use_amp = bool(c_train["use_amp"]) and device.type == "cuda"

    out_dir = resolve_path(data_root, c_data["out_dir"])
    exp_name = str(c_train["exp_name"])
    ckpt_dir = os.path.join(out_dir, exp_name, str(c_train["seed"]))
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = setup_logger(os.path.join(ckpt_dir, "adapt_from_ckpt.log"))
    csv_path = os.path.join(ckpt_dir, "metrics_adapt_from_ckpt.csv")
    init_metrics_csv(csv_path)
    save_config_snapshot(raw_cfg, os.path.join(ckpt_dir, "config_adapt_from_ckpt.yaml"))

    fusion_mode = str(c_model["fusion_mode"])
    c_api = c_model["api_encoder"]
    c_graph = c_model["graph_encoder"]
    c_alignment = c_model["alignment"]
    c_gate = c_model["gate"]
    need_alignment_mask = fusion_mode == "ours" and bool(c_alignment["enabled"])
    drop_hints = bool(c_graph["drop_extracted_behavior_hints"])

    train_csv = resolve_path(data_root, c_data["train_csv"])
    adapt_csv = resolve_path(data_root, c_data["adapt_csv"])
    val_csv = resolve_path(data_root, c_data["val_csv"])
    test_csv = resolve_path(data_root, c_data["test_csv"])
    extra_test_csvs = [resolve_path(data_root, spec["test_csv"]) for spec in (c_data.get("extra_tests", []) or [])]

    preflight_validate_protocol(cfg, train_csv, adapt_csv, val_csv, test_csv, extra_test_csvs)
    domain_years = build_global_domain_years(train_csv, adapt_csv, val_csv, test_csv, *extra_test_csvs)

    ds_tr = _make_dataset(c_data, fusion_mode, need_alignment_mask, domain_years, drop_hints, train_csv, c_data["train_pt_dir"], True, data_root)
    ds_adapt = _make_dataset(c_data, fusion_mode, need_alignment_mask, domain_years, drop_hints, adapt_csv, c_data.get("adapt_pt_dir") or c_data["train_pt_dir"], True, "")
    ds_val = _make_dataset(c_data, fusion_mode, need_alignment_mask, domain_years, drop_hints, val_csv, c_data["val_pt_dir"], False, data_root)

    graph_dim = int(getattr(ds_tr, "feature_dim", TrainingConstants.IN_FEAT_DIM))
    for name, ds in (("adapt", ds_adapt), ("val", ds_val)):
        if int(getattr(ds, "feature_dim", graph_dim)) != graph_dim:
            raise ValueError(f"Historical/{name} graph feature dimensions do not match")

    model = MalwareModelWithXAttn(
        num_classes=int(c_model["num_classes"]),
        api_emb_dim=TrainingConstants.API_EMB_DIM,
        graph_emb_dim=TrainingConstants.GRAPH_EMB_DIM,
        align_dim=TrainingConstants.ALIGN_DIM,
        max_nodes_gnn=int(c_model["max_nodes_gnn"]),
        max_xattn_nodes=int(c_model["max_xattn_nodes"]),
        in_feat_dim=graph_dim,
        xattn_heads=TrainingConstants.XATTN_HEADS,
        fusion_mode=fusion_mode,
        graph_encoder_type=str(c_graph["type"]),
        graph_hidden=int(c_graph["hidden"]),
        graph_heads=int(c_graph["heads"]),
        graph_layers=int(c_graph["layers"]),
        graph_use_behavior_hint=bool(c_graph["use_behavior_hint"]),
        api_encoder_type=str(c_api["type"]),
        api_num_hash_buckets=int(c_api["num_hash_buckets"]),
        api_type_vocab_size=int(c_api["type_vocab_size"]),
        api_max_seq_len=int(c_api["max_seq_len"]),
        api_heads=int(c_api["heads"]),
        api_layers=int(c_api["layers"]),
        alignment_penalty_scale=float(c_alignment["penalty_scale"]),
        alignment_bonus_scale=float(c_alignment["bonus_scale"]),
        alignment_context_scale=float(c_alignment["context_scale"]),
        use_alignment_bias=bool(c_alignment["enabled"]),
        use_adaptive_alignment_bias=bool(c_alignment["adaptive_bias"]),
        use_quality_gate_inputs=bool(c_gate["quality_inputs"]),
        use_uncertainty_gate=bool(c_gate["uncertainty_inputs"]),
        gate_mode=str(c_gate["mode"]),
        gate_detach=bool(c_gate["detach"]),
        late_fusion_api_weight=0.5,
    ).to(device)
    save_run_metadata(os.path.join(ckpt_dir, "run_metadata_adapt_from_ckpt.yaml"), cfg, model)
    _load_ckpt(warm_ckpt, model, device, logger)

    pin_memory = bool(c_train["pin_memory"]) and not need_alignment_mask
    loader_base = dict(
        num_workers=int(c_train["num_workers"]),
        collate_fn=hierarchical_collate_fn,
        pin_memory=pin_memory,
        persistent_workers=bool(c_train["persistent_workers"]),
    )
    if int(c_train["num_workers"]) > 0:
        loader_base["prefetch_factor"] = int(c_train["prefetch_factor"])

    dl_adapt, n_adapt, n_replay = build_adaptation_loader(
        ds_tr,
        ds_adapt,
        cfg,
        int(c_train["batch_size"]),
        dict(loader_base),
        model=model,
        device=device,
        use_amp=use_amp,
        logger=logger,
        selection_dump_path=os.path.join(ckpt_dir, "dbta_selection.csv"),
    )
    dl_val = DataLoader(ds_val, batch_size=int(c_train["eval_batch_size"]), shuffle=False, **loader_base)
    val_year_loaders = build_year_subset_loaders(ds_val, int(c_train["eval_batch_size"]), dict(loader_base))

    adaptation_epochs = int(c_train["adaptation_epochs"])
    historical_epochs = int(c_train.get("historical_epochs", 0))
    optimizer, scheduler = _build_optimizer_scheduler(model, c_train, adaptation_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(c_train["label_smoothing"]))
    scaler = build_grad_scaler(device=device, enabled=use_amp)
    best_score = float("-inf")
    no_improve = 0
    patience = int(c_train["patience"])
    min_delta = float(c_train["min_delta"])
    best_p = os.path.join(ckpt_dir, f"best_{exp_name}.pt")

    logger.info(
        "Adaptation-only run | "
        f"ckpt={warm_ckpt} adapt_samples_per_epoch={n_adapt} replay_samples_per_epoch={n_replay} "
        f"adapt_ratio={float(c_train.get('adaptation_ratio', 1.0)):.3f} "
        f"replay_ratio={float(c_train.get('replay_ratio', 0.25)):.3f}"
    )

    for local_epoch in range(1, adaptation_epochs + 1):
        epoch = historical_epochs + local_epoch
        sampler = getattr(dl_adapt, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        loss_cfg = _stage_loss_cfg(c_loss, "main")
        loss_cfg["_continual_phase"] = "adaptation"
        tr = train_one_epoch(
            model, dl_adapt, optimizer, scaler, criterion, device, epoch,
            historical_epochs + adaptation_epochs, loss_cfg=loss_cfg, logger=logger,
            use_amp=use_amp, grad_accum_steps=int(c_train["grad_accum_steps"])
        )
        val = eval_one_epoch(
            model, dl_val, criterion, device, epoch,
            num_epochs=historical_epochs + adaptation_epochs,
            use_amp=use_amp, logger=logger, strict=True, phase="val"
        )
        selection_score, latest_f1, worst_f1, aut_f1 = val[2], val[2], val[2], val[2]
        if val_year_loaders:
            ym = evaluate_temporal_windows(
                model, val_year_loaders, criterion, device, epoch,
                historical_epochs + adaptation_epochs, use_amp=use_amp,
                logger=logger, tag="val", strict=True,
            )
            selection_score, latest_f1, worst_f1, aut_f1 = compute_temporal_selection_score(ym)

        improved = selection_score > best_score + min_delta
        if improved:
            best_score = selection_score
            no_improve = 0
            save_checkpoint(best_p, epoch, model, optimizer, scheduler, scaler, best_score, no_improve, cfg)
        else:
            no_improve += 1
        save_checkpoint(os.path.join(ckpt_dir, f"last_{exp_name}.pt"), epoch, model, optimizer, scheduler, scaler, best_score, no_improve, cfg)

        append_metrics_csv(csv_path, dict(
            epoch=epoch, stage="main", continual_phase="adaptation",
            train_loss=tr[0], train_f1=tr[1], train_acc=tr[2], train_cls=tr[3],
            train_alignment=tr[4], train_branch_aux=tr[5], train_gate_oracle=0.0,
            train_uncertainty=tr[7], train_gate_disagreement=tr[8], train_gate_entropy=tr[9],
            train_gate_api=tr[10], train_gate_graph=tr[11], train_gate_joint=tr[12],
            val_loss=val[0], val_cls_loss=val[1], val_f1=val[2], val_acc=val[3],
            val_softmax_aurc=val[4], val_softmax_eaurc=val[5],
            val_gate_api=val[6], val_gate_graph=val[7], val_gate_joint=val[8],
            val_uncertainty=val[9], val_gate_disagreement=val[10], val_gate_entropy=val[11],
            lr=optimizer.param_groups[0]["lr"], best_score=best_score, is_best=int(improved),
            no_improve=no_improve, selection_score=selection_score, latest_f1=latest_f1,
            worst_f1=worst_f1, aut_f1=aut_f1,
        ))
        scheduler.step()
        if no_improve >= patience:
            logger.info(f"Early stop during adaptation: no_improve={no_improve}")
            break

    if os.path.exists(best_p):
        model.load_state_dict(torch.load(best_p, map_location=device, weights_only=False)["state_dict"])

    ds_test = _make_dataset(c_data, fusion_mode, need_alignment_mask, domain_years, drop_hints, test_csv, c_data["test_pt_dir"], False, "")
    test_lk = dict(loader_base)
    test_lk["pin_memory"] = False
    test_lk["persistent_workers"] = False
    dl_test = DataLoader(ds_test, batch_size=int(c_train["eval_batch_size"]), shuffle=False, **test_lk)
    test_dump_path = os.path.join(ckpt_dir, "eval_dumps", f"eval_dump_{exp_name}_test.csv")
    test = eval_one_epoch(
        model, dl_test, criterion, device, epoch=0, num_epochs=1,
        use_amp=use_amp, logger=logger, dump_path=test_dump_path, strict=True, phase="test"
    )

    calibrator = TemperatureScaling().to(device)
    calibrator.fit(model, dl_val, device, max_iter=50, lr=0.01, use_amp=use_amp, strict=True)
    test_probs, test_labels = collect_calibrated_outputs(model, calibrator, dl_test, device, use_amp, strict=True, phase="test_calibrated")
    test_preds = test_probs.argmax(axis=1)
    det_metrics = _binary_detection_metrics(test_labels, test_probs, test_preds)
    test_conf = test_probs.max(axis=1)
    test_correct = (test_preds == test_labels).astype(np.float64)

    final_metrics_path = os.path.join(ckpt_dir, f"final_metrics_{exp_name}.csv")
    with open(final_metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "exp_name", "adaptation_ratio", "phase", "macro_f1", "malware_recall", "fnr",
            "auroc", "auprc", "raw_softmax_aurc", "raw_softmax_eaurc",
            "cal_softmax_aurc", "cal_softmax_eaurc", "gate_api", "gate_graph", "gate_joint",
            "uncertainty", "gate_disagreement", "gate_entropy", "warm_start_historical_ckpt",
        ])
        writer.writeheader()
        writer.writerow({
            "exp_name": exp_name,
            "adaptation_ratio": float(c_train.get("adaptation_ratio", 0.0)),
            "phase": "final_test",
            "macro_f1": f"{test[2]:.6f}",
            "malware_recall": f"{det_metrics['malware_recall']:.6f}",
            "fnr": f"{det_metrics['fnr']:.6f}",
            "auroc": f"{det_metrics['auroc']:.6f}",
            "auprc": f"{det_metrics['auprc']:.6f}",
            "raw_softmax_aurc": f"{test[4]:.6f}",
            "raw_softmax_eaurc": f"{test[5]:.6f}",
            "cal_softmax_aurc": f"{float(aurc(test_conf, test_correct)):.6f}",
            "cal_softmax_eaurc": f"{float(eaurc(test_conf, test_correct)):.6f}",
            "gate_api": f"{test[6]:.6f}",
            "gate_graph": f"{test[7]:.6f}",
            "gate_joint": f"{test[8]:.6f}",
            "uncertainty": f"{test[9]:.6f}",
            "gate_disagreement": f"{test[10]:.6f}",
            "gate_entropy": f"{test[11]:.6f}",
            "warm_start_historical_ckpt": warm_ckpt,
        })
    logger.info(f"Final metrics CSV: {final_metrics_path}")


if __name__ == "__main__":
    main()
