import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm import tqdm

from fusion.utils import get_amp_context, prepare_batch

logger = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    """
    温度缩放校准器（Guo et al., ICML 2017）
    """

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.ones(1) * 1.5))

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp_min(1e-6)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits / self.temperature, dim=-1)

    def fit(self, model, val_loader, device, max_iter=50, lr=0.01, use_amp=False, strict=True):
        model.eval()
        self.to(device)
        self.train()
        all_logits = []
        all_labels = []
        logger.info("Collecting validation logits for calibration...")
        failed_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Calibration data collection"):
                if batch is None:
                    if strict:
                        raise RuntimeError("[calibration] got None batch")
                    failed_batches += 1
                    continue

                if strict and int(batch.get("num_failed", 0) or 0) > 0:
                    raise RuntimeError(
                        "[calibration] failed samples found: "
                        + "; ".join(
                            f"sid={x.get('sid')} path={x.get('path')} reason={x.get('reason')}"
                            for x in (batch.get("failed_items", []) or [])[:5]
                        )
                    )
                try:
                    result = prepare_batch(batch, device,
                                        skip_graph=False,
                                        skip_masks=(model.fusion_mode != "ours" or not getattr(model, "use_alignment_bias", False)))
                    if result[2] is None:
                        failed_batches += 1
                        continue
                    graph, masks, y, _, explicit_info, _ = result
                    q_apis, q_graphs, q_aligns, pert_apis, pert_graphs, time_ids = explicit_info
                    with get_amp_context(device, enabled=use_amp):
                        logits, _ = model(
                            graph_data=graph,
                            explicit_qs=(q_apis, q_graphs, q_aligns, pert_apis, pert_graphs),
                            time_ids=time_ids,
                            masks=masks,
                        )
                    if logits.shape[0] != y.shape[0]:
                        logger.warning(f"Batch size mismatch: {logits.shape[0]} vs {y.shape[0]}")
                        failed_batches += 1
                        continue
                    all_logits.append(logits.detach().cpu())
                    all_labels.append(y.detach().cpu()) 
                except Exception as e:
                    logger.warning(f"Calibration batch failed: {e}")
                    failed_batches += 1
                    continue

        if not all_logits or not all_labels:
            logger.error("No valid batches for calibration!")
            raise RuntimeError("Calibration failed: no valid data collected")

        if len(all_logits) != len(all_labels):
            logger.error(f"Logits/labels count mismatch: {len(all_logits)} vs {len(all_labels)}")
            raise RuntimeError("Calibration data inconsistency")

        if failed_batches > 0:
            logger.warning(f"Calibration: {failed_batches} batches failed")

        all_logits = torch.cat(all_logits, dim=0).to(device).float()
        all_labels = torch.cat(all_labels, dim=0).to(device)

        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(all_logits / self.temperature, all_labels)
            if not torch.isfinite(loss):
                loss = all_logits.new_tensor(0.0, requires_grad=True)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        with torch.no_grad():
            before_probs = torch.softmax(all_logits, dim=-1)
            after_probs = self(all_logits)
            ece_b = self._compute_ece(before_probs, all_labels)
            ece_a = self._compute_ece(after_probs, all_labels)
            nll_b = self._compute_nll(before_probs, all_labels)
            nll_a = self._compute_nll(after_probs, all_labels)
            br_b  = self._compute_brier(before_probs, all_labels)
            br_a  = self._compute_brier(after_probs, all_labels)

        logger.info(
            f"Temperature calibration done: T={self.temperature.item():.4f} | "
            f"ECE: {ece_b:.4f}→{ece_a:.4f} | "
            f"NLL: {nll_b:.4f}→{nll_a:.4f} | "
            f"Brier: {br_b:.4f}→{br_a:.4f}"
        )

    @staticmethod
    def _compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
        confidences, predictions = probs.max(dim=-1)
        accuracies = predictions.eq(labels)

        ece = torch.tensor(0.0, device=probs.device)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                bin_acc = accuracies[in_bin].float().mean()
                bin_conf = confidences[in_bin].mean()
                ece += (in_bin.sum().float() / len(probs)) * torch.abs(bin_acc - bin_conf)

        return ece.item()
    
    @staticmethod
    def _compute_nll(probs: torch.Tensor, labels: torch.Tensor) -> float:
        """Negative log-likelihood (average)."""
        idx = torch.arange(len(labels), device=labels.device)
        p_true = probs[idx, labels].clamp_min(1e-12)
        return float(-p_true.log().mean().item())
    @staticmethod
    def _compute_brier(probs: torch.Tensor, labels: torch.Tensor) -> float:
        """Multi-class Brier score (lower is better)."""
        B, C = probs.shape
        onehot = F.one_hot(labels, num_classes=C).float()
        return float(((probs - onehot) ** 2).sum(dim=-1).mean().item())
