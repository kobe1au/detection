"""
Split Conformal Prediction for malware classification with coverage guarantees.

Reference:
  - Vovk et al., "Algorithmic Learning in a Random World", 2005.
  - Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction", 2022.

Guarantee:
    For alpha=0.1, prediction set will cover true label ≥ 90% of the time
    (under exchangeability). Sets of size > 1 are "rejected" (uncertain).
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class ConformalPredictor:
    """
    APS (Adaptive Prediction Sets) conformal predictor.
    """
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: target miscoverage level (e.g., 0.1 = 90% coverage)
        """
        self.alpha = float(alpha)
        self.q_hat: Optional[float] = None
        self.n_cal: int = 0
    
    def calibrate(self, cal_probs: torch.Tensor, cal_labels: torch.Tensor):
        probs = cal_probs.cpu().numpy()
        labels = cal_labels.cpu().numpy()
        n = len(labels)
        self.n_cal = n
        
        # ★ 修复：确保 contiguous
        sorted_desc = np.sort(probs, axis=1)[:, ::-1].copy()
        sorted_idx = np.argsort(-probs, axis=1)
        cum_probs_sorted = np.cumsum(sorted_desc, axis=1)
        
        # 映射回原始类别位置
        unsort_idx = np.argsort(sorted_idx, axis=1)
        cum_probs = np.take_along_axis(cum_probs_sorted, unsort_idx, axis=1)
        scores = cum_probs[np.arange(n), labels]
        
        # Finite-sample correction
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self.q_hat = float(np.quantile(scores, q_level, method="higher"))
        logger.info(
            f"Conformal calibrated: n={n} alpha={self.alpha:.2f} q_hat={self.q_hat:.4f}"
        )
    
    def predict(self, probs: torch.Tensor) -> torch.Tensor:
        if self.q_hat is None:
            raise RuntimeError("Must call calibrate() first")
        p = probs.cpu().numpy()
        sorted_idx = np.argsort(-p, axis=1)
        sorted_p = np.take_along_axis(p, sorted_idx, axis=1).copy()  # ★ 确保 contiguous
        cum_p = np.cumsum(sorted_p, axis=1)
        in_set_sorted = cum_p <= self.q_hat
        in_set_sorted[:, 0] = True
        # ★ 向量化 unsort，消除 for 循环
        unsort_idx = np.argsort(sorted_idx, axis=1)
        sets = np.take_along_axis(in_set_sorted, unsort_idx, axis=1)
        return torch.from_numpy(sets).bool()
    
    def predict_with_reject(self, probs: torch.Tensor):
        sets = self.predict(probs)                    # CPU tensor
        set_sizes = sets.sum(dim=-1)                  # CPU
        preds = probs.cpu().argmax(dim=-1)          # ★ 强制 CPU
        rejected = set_sizes > 1
        return preds, rejected, set_sizes
    
    def state_dict(self):
        return {"alpha": self.alpha, "q_hat": self.q_hat, "n_cal": self.n_cal}
    
    def load_state_dict(self, d):
        self.alpha = d["alpha"]
        self.q_hat = d["q_hat"]
        self.n_cal = d["n_cal"]

class OnlineConformalPredictor(ConformalPredictor):
    """
    Sliding-window adaptive conformal predictor.
    
    Reference:
      - Gibbs & Candes, "Adaptive Conformal Inference Under Distribution Shift", 
        NeurIPS 2021.
    
    在部署过程中，随着新标注样本到达，滑动更新 q_hat，
    使 coverage 在时间漂移下保持稳定。
    """
    
    def __init__(self, alpha: float = 0.1, window_size: int = 500,
                 gamma: float = 0.005):
        """
        Args:
            alpha: target miscoverage level
            window_size: maximum number of scores to retain
            gamma: step size for online alpha adjustment (Gibbs & Candes)
                   larger gamma → faster adaptation, less stable
        """
        super().__init__(alpha)
        self.window_size = int(window_size)
        self.gamma = float(gamma)
        self.score_buffer: list[float] = []
        self.alpha_t = float(alpha)  # adaptive alpha, updated online
    
    def _compute_scores(self, probs: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
        """Compute APS nonconformity scores for a batch."""
        p = probs.cpu().numpy()
        y = labels.cpu().numpy()
        n = len(y)
        sorted_desc = np.sort(p, axis=1)[:, ::-1].copy()
        sorted_idx = np.argsort(-p, axis=1)
        cum_probs_sorted = np.cumsum(sorted_desc, axis=1)
        unsort_idx = np.argsort(sorted_idx, axis=1)
        cum_probs = np.take_along_axis(cum_probs_sorted, unsort_idx, axis=1)
        scores = cum_probs[np.arange(n), y]
        return scores
    
    def calibrate(self, cal_probs: torch.Tensor, cal_labels: torch.Tensor):
        """Initial calibration — also seeds the score buffer."""
        super().calibrate(cal_probs, cal_labels)
        # Seed buffer with calibration scores
        scores = self._compute_scores(cal_probs, cal_labels)
        self.score_buffer = scores.tolist()
        if len(self.score_buffer) > self.window_size:
            self.score_buffer = self.score_buffer[-self.window_size:]
        self.alpha_t = self.alpha
    
    def update(self, new_probs: torch.Tensor, new_labels: torch.Tensor):
        """
        Online update with newly labeled samples.
        
        Uses the Gibbs & Candes update rule:
            alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
        where err_t = fraction of samples NOT covered by current prediction sets.
        """
        if new_probs.numel() == 0:
            return
        
        # 1. Check coverage under current q_hat
        sets = self.predict(new_probs)  # [N, C] bool
        labels_np = new_labels.cpu().numpy()
        n = len(labels_np)
        labels_cpu = new_labels.cpu()
        idx = torch.arange(n)
        covered = sets[idx, labels_cpu].numpy()
        err_t = 1.0 - covered.mean()
        
        # 2. Gibbs & Candes adaptive alpha update
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - err_t)
        self.alpha_t = float(np.clip(self.alpha_t, 0.001, 0.5))
        
        # 3. Add new scores to buffer
        new_scores = self._compute_scores(new_probs, new_labels)
        self.score_buffer.extend(new_scores.tolist())
        if len(self.score_buffer) > self.window_size:
            self.score_buffer = self.score_buffer[-self.window_size:]
        
        # 4. Recompute q_hat with adaptive alpha
        buf = np.array(self.score_buffer)
        n_buf = len(buf)
        q_level = min(np.ceil((n_buf + 1) * (1 - self.alpha_t)) / n_buf, 1.0)
        self.q_hat = float(np.quantile(buf, q_level, method="higher"))
        self.n_cal = n_buf
    
    def state_dict(self):
        d = super().state_dict()
        d.update({
            "window_size": self.window_size,
            "gamma": self.gamma,
            "alpha_t": self.alpha_t,
            "score_buffer": self.score_buffer,
        })
        return d
    
    def load_state_dict(self, d):
        super().load_state_dict(d)
        self.window_size = int(d.get("window_size", 500))
        self.gamma = float(d.get("gamma", 0.005))
        self.alpha_t = float(d.get("alpha_t", self.alpha))
        self.score_buffer = list(d.get("score_buffer", []))