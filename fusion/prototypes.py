from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.constants import ArchitectureConstants


class PrototypeMemory(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int, momentum: float = 0.99):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.register_buffer(
            "prototypes",
            F.normalize(torch.randn(self.num_classes, self.feature_dim), dim=-1),
        )
        self.register_buffer("seen", torch.zeros(self.num_classes, dtype=torch.long))

    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels: torch.Tensor):
        if feats is None or labels is None or feats.numel() == 0:
            return
        feats = F.normalize(feats.detach().float(), dim=-1)
        labels = labels.detach().long().view(-1)
        for c in labels.unique():
            c = int(c.item())
            mask = labels == c
            if mask.sum() == 0:
                continue
            center = F.normalize(feats[mask].mean(dim=0), dim=-1)
            if self.seen[c] == 0:
                self.prototypes[c] = center.to(
                    device=self.prototypes.device,
                    dtype=self.prototypes.dtype,
                )
            else:
                old = self.prototypes[c]
                new = center.to(device=old.device, dtype=old.dtype)
                self.prototypes[c] = F.normalize(
                    self.momentum * old + (1.0 - self.momentum) * new,
                    dim=-1,
                ).to(device=self.prototypes.device, dtype=self.prototypes.dtype)
            self.seen[c] += int(mask.sum().item())

    @torch.no_grad()
    def update_weighted(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ):
        if feats is None or labels is None or feats.numel() == 0:
            return
        feats = F.normalize(feats.detach().float(), dim=-1)
        labels = labels.detach().long().view(-1)
        weights = weights.detach().float().view(-1).to(feats.device).clamp_min(0.0)

        for c in labels.unique():
            c = int(c.item())
            mask = labels == c
            if mask.sum() == 0:
                continue
            w = weights[mask].unsqueeze(-1)
            if w.sum() <= 1e-8:
                continue
            w_norm = w / w.sum().clamp_min(1e-8)
            center = F.normalize((w_norm * feats[mask]).sum(dim=0), dim=-1)
            if self.seen[c] == 0:
                self.prototypes[c] = center.to(
                    device=self.prototypes.device,
                    dtype=self.prototypes.dtype,
                )
            else:
                old = self.prototypes[c]
                new = center.to(device=old.device, dtype=old.dtype)
                self.prototypes[c] = F.normalize(
                    self.momentum * old + (1.0 - self.momentum) * new,
                    dim=-1,
                ).to(device=self.prototypes.device, dtype=self.prototypes.dtype)
            self.seen[c] += int(mask.sum().item())

    def get_loss_quality_gated(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        quality_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if feats is None or labels is None or feats.numel() == 0:
            dev = feats.device if feats is not None else (
                labels.device if labels is not None else torch.device("cpu")
            )
            dt = feats.dtype if feats is not None else torch.float32
            return torch.tensor(0.0, device=dev, dtype=dt)

        feats = F.normalize(feats.float(), dim=-1)
        labels = labels.long().view(-1)
        seen_mask = self.seen[labels] > 0
        if not seen_mask.any():
            return feats.new_tensor(0.0)

        feats_seen = feats[seen_mask]
        labels_seen = labels[seen_mask]
        proto = self.prototypes[labels_seen].to(feats.device, feats.dtype)
        base_loss = 1.0 - F.cosine_similarity(feats_seen, proto, dim=-1)
        if quality_weights is not None:
            q = quality_weights.float().view(-1).to(feats.device)[seen_mask]
            q_norm = q / q.mean().clamp_min(1e-8)
            return (base_loss * q_norm).mean()
        return base_loss.mean()


class TemporalPrototypeMemory(nn.Module):
    """Cross-batch year-label-cluster prototype memory.

    The memory keeps multiple sub-prototypes for each year-label cell so drift
    is measured against the nearest local mode instead of one coarse class
    average. This is important for APKs because benign and malware samples are
    both multi-modal across app categories, SDK patterns, packing styles, and
    behavior families.
    """

    def __init__(
        self,
        num_domains: int,
        num_classes: int,
        feature_dim: int,
        momentum: float = 0.99,
        num_clusters: int = 4,
    ):
        super().__init__()
        self.num_domains = max(int(num_domains), 1)
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.num_clusters = max(int(num_clusters), 1)
        self.current_margin = 0.20
        self.current_margin_weight = 0.50
        self.future_rank_margin = 0.15
        self.future_rank_weight = 0.50
        proto = F.normalize(
            torch.randn(
                self.num_domains,
                self.num_classes,
                self.num_clusters,
                self.feature_dim,
            ),
            dim=-1,
        )
        self.register_buffer("prototypes", proto)
        self.register_buffer(
            "seen",
            torch.zeros(
                self.num_domains,
                self.num_classes,
                self.num_clusters,
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "spread",
            torch.zeros(
                self.num_domains,
                self.num_classes,
                self.num_clusters,
                dtype=torch.float32,
            ),
        )

    @torch.no_grad()
    def reset(self):
        self.prototypes.copy_(F.normalize(torch.randn_like(self.prototypes), dim=-1))
        self.seen.zero_()
        self.spread.zero_()

    @torch.no_grad()
    def occupancy_stats(self) -> dict[str, float]:
        occupied = self.seen > 0
        occupied_clusters = int(occupied.sum().item())
        occupied_cells_mask = occupied.any(dim=2)
        occupied_cells = int(occupied_cells_mask.sum().item())
        total_cells = int(self.num_domains * self.num_classes)
        mean_clusters_per_seen_cell = 0.0
        if occupied_cells > 0:
            mean_clusters_per_seen_cell = float(
                occupied.sum(dim=2)[occupied_cells_mask].float().mean().item()
            )
        return {
            "occupied_cells": float(occupied_cells),
            "total_cells": float(total_cells),
            "occupied_clusters": float(occupied_clusters),
            "mean_clusters_per_seen_cell": mean_clusters_per_seen_cell,
        }

    def _valid_label_time_mask(self, labels: torch.Tensor, time_ids: torch.Tensor) -> torch.Tensor:
        return (
            (labels >= 0)
            & (labels < self.num_classes)
            & (time_ids >= 0)
            & (time_ids < self.num_domains)
        )

    @staticmethod
    def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
        weights = weights.float().clamp_min(0.0)
        return weights / weights.sum().clamp_min(1e-8)

    def _target_cluster_count(self, observed_count: int) -> int:
        """Grow clusters gradually as a year-label cell accumulates samples."""
        observed_count = max(int(observed_count), 1)
        return min(self.num_clusters, max(1, int(observed_count ** 0.5)))

    def _latest_previous_seen_domain(self, dom_i: int, class_i: int, cluster_i: int) -> Optional[int]:
        if dom_i <= 0:
            return None
        seen_domains = torch.where(self.seen[:dom_i, class_i, cluster_i] > 0)[0]
        if seen_domains.numel() == 0:
            return None
        return int(seen_domains[-1].item())

    def _select_diverse_centers(
        self,
        feats: torch.Tensor,
        weights: torch.Tensor,
        existing: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        """Greedy farthest-first initialization inside one year-label cell."""
        if feats.numel() == 0 or count <= 0:
            return feats[:0]

        count = min(int(count), feats.size(0))
        weights = weights.to(device=feats.device, dtype=feats.dtype).clamp_min(1e-8)
        selected = []
        refs = existing.to(device=feats.device, dtype=feats.dtype)
        used = torch.zeros(feats.size(0), device=feats.device, dtype=torch.bool)

        for _ in range(count):
            if refs.numel() == 0:
                score = weights.clone()
            else:
                sim = torch.matmul(feats, refs.t()).max(dim=1).values
                score = (1.0 - sim.clamp(-1.0, 1.0)).clamp_min(0.0) * weights.sqrt()
            score = score.masked_fill(used, -1.0)
            idx = int(torch.argmax(score).item())
            if used[idx]:
                break
            used[idx] = True
            selected.append(feats[idx])
            refs = torch.cat([refs, feats[idx].view(1, -1)], dim=0)

        if not selected:
            return feats[:0]
        return torch.stack(selected, dim=0)

    @torch.no_grad()
    def update_weighted(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        time_ids: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ):
        if feats is None or labels is None or time_ids is None or feats.numel() == 0:
            return

        feats = F.normalize(feats.detach().float(), dim=-1)
        labels = labels.detach().long().view(-1)
        time_ids = time_ids.detach().long().view(-1)
        if labels.numel() != feats.size(0) or time_ids.numel() != feats.size(0):
            return
        if weights is None:
            weights = torch.ones(feats.size(0), device=feats.device, dtype=feats.dtype)
        else:
            weights = weights.detach().float().view(-1).to(feats.device).clamp_min(0.0)
            if weights.numel() != feats.size(0):
                weights = torch.ones(feats.size(0), device=feats.device, dtype=feats.dtype)

        valid = self._valid_label_time_mask(labels, time_ids)
        if not valid.any():
            return
        feats = feats[valid]
        labels = labels[valid]
        time_ids = time_ids[valid].clamp(0, self.num_domains - 1)
        weights = weights[valid]

        compound_key = time_ids * self.num_classes + labels
        for ck in compound_key.unique():
            dom_i = int(ck.item()) // self.num_classes
            c_i = int(ck.item()) % self.num_classes
            mask = compound_key == ck
            if mask.sum() == 0:
                continue

            w = weights[mask].unsqueeze(-1)
            if w.sum() <= 1e-8:
                continue
            group_feats = feats[mask]
            group_weights = weights[mask]

            seen_clusters = self.seen[dom_i, c_i] > 0
            unseen_idx = torch.where(~seen_clusters)[0]
            current_seen_clusters = int(seen_clusters.sum().item())
            observed_count = int(self.seen[dom_i, c_i].sum().item()) + int(group_feats.size(0))
            target_clusters = self._target_cluster_count(observed_count)
            num_new = max(0, min(int(unseen_idx.numel()), target_clusters - current_seen_clusters))
            if num_new > 0:
                carried = 0
                # Keep cluster identities comparable across years. If cluster k
                # existed in an earlier year, initialize the new year-label cell
                # from the latest historical prototype of the same k before EMA
                # updates with current samples.
                for k in unseen_idx:
                    if carried >= num_new:
                        break
                    k_i = int(k.item())
                    prev_dom = self._latest_previous_seen_domain(dom_i, c_i, k_i)
                    if prev_dom is None:
                        continue
                    self.prototypes[dom_i, c_i, k_i] = self.prototypes[prev_dom, c_i, k_i].to(
                                                                device=self.prototypes.device,
                                                                dtype=self.prototypes.dtype,
                                                            )
                    self.spread[dom_i, c_i, k_i] = self.spread[prev_dom, c_i, k_i].to(
                        device=self.spread.device,
                        dtype=self.spread.dtype,
                    )
                    self.seen[dom_i, c_i, k_i] = 1
                    carried += 1

                seen_clusters = self.seen[dom_i, c_i] > 0
                remaining_new = num_new - carried
                unseen_idx = torch.where(~seen_clusters)[0]
                existing = self.prototypes[dom_i, c_i, seen_clusters]
                if remaining_new <= 0:
                    new_centers = group_feats[:0]
                else:
                    new_centers = self._select_diverse_centers(
                        group_feats,
                        group_weights,
                        existing,
                        remaining_new,
                    )
                for j, center in enumerate(new_centers):
                    k_i = int(unseen_idx[j].item())
                    self.prototypes[dom_i, c_i, k_i] = F.normalize(center, dim=-1).to(
                        device=self.prototypes.device,
                        dtype=self.prototypes.dtype,
                    )
                    self.seen[dom_i, c_i, k_i] = 1

            seen_clusters = self.seen[dom_i, c_i] > 0
            if not seen_clusters.any():
                continue

            cluster_ids = torch.where(seen_clusters)[0]
            cluster_proto = self.prototypes[dom_i, c_i, cluster_ids].to(
                device=group_feats.device,
                dtype=group_feats.dtype,
            )
            nearest = torch.matmul(group_feats, cluster_proto.t()).argmax(dim=1)
            assigned = cluster_ids.to(group_feats.device)[nearest]

            for k in assigned.unique():
                k_i = int(k.item())
                cmask = assigned == k
                if not cmask.any():
                    continue
                cw = self._normalize_weights(group_weights[cmask]).unsqueeze(-1)
                center = F.normalize((cw * group_feats[cmask]).sum(dim=0), dim=-1)
                old = self.prototypes[dom_i, c_i, k_i]
                new = center.to(device=old.device, dtype=old.dtype)
                old_count = int(self.seen[dom_i, c_i, k_i].item())
                sample_sim = torch.matmul(group_feats[cmask], center.view(-1, 1)).view(-1).clamp(-1.0, 1.0)
                sample_spread = ((1.0 - sample_sim) * 0.5).clamp(0.0, 1.0)
                cw_flat = self._normalize_weights(group_weights[cmask]).to(sample_spread.device)
                batch_spread = (cw_flat * sample_spread).sum().to(device=self.spread.device, dtype=self.spread.dtype)
                new_count = int(cmask.sum().item())
                total_count = max(old_count + new_count, 1)
                old_spread = self.spread[dom_i, c_i, k_i]
                self.spread[dom_i, c_i, k_i] = (
                    old_spread * float(old_count) + batch_spread * float(new_count)
                ) / float(total_count)
                self.prototypes[dom_i, c_i, k_i] = F.normalize(
                    self.momentum * old + (1.0 - self.momentum) * new,
                    dim=-1,
                ).to(device=old.device, dtype=old.dtype)
                self.seen[dom_i, c_i, k_i] += int(cmask.sum().item())

    def get_loss_quality_gated(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        time_ids: torch.Tensor,
        quality_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if feats is None or labels is None or time_ids is None or feats.numel() == 0:
            dev = feats.device if feats is not None else (
                labels.device if labels is not None else torch.device("cpu")
            )
            dt = feats.dtype if feats is not None else torch.float32
            return torch.tensor(0.0, device=dev, dtype=dt)

        feats = F.normalize(feats.float(), dim=-1)
        labels = labels.long().view(-1)
        time_ids = time_ids.long().view(-1).clamp(0, self.num_domains - 1)
        if labels.numel() != feats.size(0) or time_ids.numel() != feats.size(0):
            return feats.new_tensor(0.0)
        valid_label = (labels >= 0) & (labels < self.num_classes)
        safe_labels = labels.clamp(0, self.num_classes - 1)
        cluster_valid = (self.seen[time_ids, safe_labels] > 0).to(device=feats.device)
        seen_mask = valid_label.to(feats.device) & cluster_valid.any(dim=1)
        if not seen_mask.any():
            return feats.new_tensor(0.0)

        feats_seen = feats[seen_mask]
        seen_time_ids = time_ids[seen_mask]
        seen_labels = safe_labels[seen_mask]
        all_proto = self.prototypes[seen_time_ids].to(
            feats.device,
            feats.dtype,
        )
        all_valid = (self.seen[seen_time_ids] > 0).to(device=feats.device)
        cos = torch.einsum("bd,bckd->bck", feats_seen, all_proto).clamp(-1.0, 1.0)
        dist = (1.0 - cos).masked_fill(~all_valid, 2.0)
        class_dist = dist.min(dim=2).values

        row = torch.arange(feats_seen.size(0), device=feats.device)
        same_dist = class_dist[row, seen_labels]
        other_valid = all_valid.any(dim=2)
        other_valid[row, seen_labels] = False
        if other_valid.any():
            other_dist = class_dist.masked_fill(~other_valid, 2.0).min(dim=1).values
            has_other = other_valid.any(dim=1)
            margin_loss = F.relu(same_dist - other_dist + self.current_margin)
            base_loss = same_dist + self.current_margin_weight * margin_loss * has_other.to(feats.dtype)
        else:
            base_loss = same_dist
        if quality_weights is not None:
            q_all = quality_weights.float().view(-1).to(feats.device)
            if q_all.numel() == feats.size(0):
                q = q_all[seen_mask]
                q_norm = q / q.mean().clamp_min(1e-8)
                return (base_loss * q_norm).mean()
        return base_loss.mean()

    def forecast_next_prototypes(
        self,
        time_ids: torch.Tensor,
        velocity_scale: float = 1.0,
        min_history: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict virtual next-year class-cluster prototypes for each sample domain."""
        if time_ids is None or time_ids.numel() == 0:
            return self.prototypes[:0], self.seen[:0].bool()

        domain_ids = time_ids.long().view(-1).clamp(0, self.num_domains - 1)
        current = self.prototypes[domain_ids]
        current_seen = self.seen[domain_ids] > 0

        prev_ids = (domain_ids - 1).clamp_min(0)
        previous = self.prototypes[prev_ids]
        previous_seen = (domain_ids[:, None, None] > 0) & (self.seen[prev_ids] > 0)

        has_trajectory = current_seen & previous_seen
        velocity = current - previous
        future = F.normalize(current + float(velocity_scale) * velocity, dim=-1)

        if int(min_history) <= 1:
            future = torch.where(has_trajectory.unsqueeze(-1), future, current)
            valid = current_seen
        else:
            valid = has_trajectory

        # For validation/test years beyond the training horizon, their own
        # year-label-cluster prototypes are unseen. Extrapolate each cluster from the
        # last two observed historical sub-prototypes so temporal drift remains
        # meaningful for future years without using future labels.
        min_hist = max(int(min_history), 2)
        for c in range(self.num_classes):
            for k in range(self.num_clusters):
                seen_domains = torch.where(self.seen[:, c, k] > 0)[0]
                if seen_domains.numel() < min_hist:
                    continue
                latest = seen_domains[-1]
                prev = seen_domains[-2]
                future_rows = domain_ids > latest
                if not future_rows.any():
                    continue
                gap = (domain_ids[future_rows] - latest).to(
                    device=self.prototypes.device,
                    dtype=self.prototypes.dtype,
                ).view(-1, 1)
                cls_velocity = self.prototypes[latest, c, k] - self.prototypes[prev, c, k]
                extrapolated = F.normalize(
                    self.prototypes[latest, c, k].view(1, -1)
                    + float(velocity_scale) * gap * cls_velocity.view(1, -1),
                    dim=-1,
                )
                future[future_rows, c, k] = extrapolated
                valid[future_rows, c, k] = True

        return future, valid

    def get_future_forecast_loss(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        time_ids: torch.Tensor,
        quality_weights: Optional[torch.Tensor] = None,
        temperature: float = 0.2,
        velocity_scale: float = 1.0,
        min_history: int = 2,
    ) -> torch.Tensor:
        """Constrain current features with predicted next-year class-cluster prototypes."""
        if feats is None or labels is None or time_ids is None or feats.numel() == 0:
            dev = feats.device if feats is not None else (
                labels.device if labels is not None else torch.device("cpu")
            )
            dt = feats.dtype if feats is not None else torch.float32
            return torch.tensor(0.0, device=dev, dtype=dt)

        z = F.normalize(feats.float(), dim=-1)
        labels = labels.long().view(-1)
        time_ids = time_ids.long().view(-1)
        if labels.numel() != z.size(0) or time_ids.numel() != z.size(0):
            return z.new_tensor(0.0)

        future_proto, valid_cls = self.forecast_next_prototypes(
            time_ids,
            velocity_scale=velocity_scale,
            min_history=min_history,
        )
        future_proto = future_proto.to(device=z.device, dtype=z.dtype)
        valid_cls = valid_cls.to(device=z.device)
        if (
            future_proto.size(0) != z.size(0)
            or future_proto.size(1) != self.num_classes
            or future_proto.size(2) != self.num_clusters
        ):
            return z.new_tensor(0.0)

        safe_labels = labels.clamp(0, self.num_classes - 1)
        row = torch.arange(z.size(0), device=z.device)
        valid_label = (labels >= 0) & (labels < self.num_classes)
        valid_class = valid_cls.any(dim=2)
        valid_row = valid_label & valid_class[row, safe_labels] & (valid_class.sum(dim=1) > 1)
        if not valid_row.any():
            return z.new_tensor(0.0)

        temp = max(float(temperature), 1e-4)
        logits = torch.einsum("bd,bckd->bck", z, future_proto) / temp
        logits = logits.masked_fill(~valid_cls, -1e4).max(dim=2).values
        logits = logits.masked_fill(~valid_class, -1e4)
        per_sample = F.cross_entropy(
            logits[valid_row],
            safe_labels[valid_row],
            reduction="none",
        )

        class_sim = torch.einsum("bd,bckd->bck", z, future_proto).clamp(-1.0, 1.0)
        class_sim = class_sim.masked_fill(~valid_cls, -1e4).max(dim=2).values
        sim_valid = class_sim[valid_row]
        cls_valid = valid_class[valid_row].clone()
        labels_valid = safe_labels[valid_row]
        local_row = torch.arange(sim_valid.size(0), device=z.device)
        pos_sim = sim_valid[local_row, labels_valid]
        cls_valid[local_row, labels_valid] = False
        has_negative = cls_valid.any(dim=1)
        if has_negative.any():
            neg_sim = sim_valid.masked_fill(~cls_valid, -1e4).max(dim=1).values
            rank_loss = F.relu(neg_sim - pos_sim + self.future_rank_margin)
            per_sample = per_sample + self.future_rank_weight * rank_loss * has_negative.to(z.dtype)

        if quality_weights is not None:
            q = quality_weights.float().view(-1).to(z.device)
            if q.numel() == z.size(0):
                q = q[valid_row].clamp(0.0, 1.0)
                q = q / q.mean().clamp_min(1e-8)
                per_sample = per_sample * q.detach()

        return per_sample.mean()

    def _nearest_class_distances(
        self,
        z: torch.Tensor,
        time_ids: torch.Tensor,
        include_future: bool = True,
        velocity_scale: float = 0.5,
        min_history: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        proto = self.prototypes[time_ids].to(device=z.device, dtype=z.dtype)
        valid = (self.seen[time_ids] > 0).to(device=z.device)

        cos = torch.einsum("bd,bckd->bck", z, proto).clamp(-1.0, 1.0)
        cluster_dist = ((1.0 - cos) * 0.5).clamp(0.0, 1.0).masked_fill(~valid, 1.0)
        class_dist = cluster_dist.min(dim=2).values
        valid_class = valid.any(dim=2)

        if include_future:
            future_proto, future_valid = self.forecast_next_prototypes(
                time_ids,
                velocity_scale=velocity_scale,
                min_history=min_history,
            )
            if (
                future_proto.size(0) == z.size(0)
                and future_proto.size(1) == self.num_classes
                and future_proto.size(2) == self.num_clusters
            ):
                future_proto = future_proto.to(device=z.device, dtype=z.dtype)
                future_valid = future_valid.to(device=z.device)
                future_cos = torch.einsum("bd,bckd->bck", z, future_proto).clamp(-1.0, 1.0)
                future_cluster_dist = ((1.0 - future_cos) * 0.5).clamp(0.0, 1.0)
                future_cluster_dist = future_cluster_dist.masked_fill(~future_valid, 1.0)
                future_dist = future_cluster_dist.min(dim=2).values
                future_valid_class = future_valid.any(dim=2)

                both = valid_class & future_valid_class
                future_only = (~valid_class) & future_valid_class
                class_dist = torch.where(both, 0.7 * class_dist + 0.3 * future_dist, class_dist)
                class_dist = torch.where(future_only, future_dist, class_dist)
                valid_class = valid_class | future_valid_class

        class_dist = class_dist.masked_fill(~valid_class, 1.0)
        return class_dist.clamp(0.0, 1.0), valid_class

    def _cluster_reliability(
        self,
        time_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        domain_ids = time_ids.long().view(-1).clamp(0, self.num_domains - 1)
        if domain_ids.numel() == 0:
            empty = torch.zeros(
                (0, self.num_classes, self.num_clusters),
                device=device,
                dtype=dtype,
            )
            return empty, empty.bool()

        counts = self.seen[domain_ids].to(device=device, dtype=dtype).clone()
        spreads = self.spread[domain_ids].to(device=device, dtype=dtype).clone()
        ages = torch.zeros_like(counts)

        # For validation/test years with no own prototypes, borrow the latest
        # historical cluster statistics and penalize them by temporal age.
        for b, dom in enumerate(domain_ids.detach().cpu().tolist()):
            for c in range(self.num_classes):
                for k in range(self.num_clusters):
                    if float(counts[b, c, k].item()) > 0.0:
                        continue
                    seen_domains = torch.where(self.seen[: dom + 1, c, k] > 0)[0]
                    if seen_domains.numel() == 0:
                        continue
                    hist_dom = int(seen_domains[-1].item())
                    counts[b, c, k] = self.seen[hist_dom, c, k].to(device=device, dtype=dtype)
                    spreads[b, c, k] = self.spread[hist_dom, c, k].to(device=device, dtype=dtype)
                    ages[b, c, k] = float(max(dom - hist_dom, 0))

        valid = counts > 0
        count_scale = max(float(ArchitectureConstants.PROTO_RELIABILITY_COUNT_SCALE), 1.0)
        spread_scale = max(float(ArchitectureConstants.PROTO_RELIABILITY_SPREAD_SCALE), 1e-6)
        age_decay = max(float(ArchitectureConstants.PROTO_RELIABILITY_AGE_DECAY), 0.0)
        count_rel = (torch.log1p(counts.clamp_min(0.0)) / math.log1p(count_scale)).clamp(0.0, 1.0)
        spread_rel = torch.exp(-spreads.clamp_min(0.0) / spread_scale).clamp(0.0, 1.0)
        age_rel = torch.exp(-ages.clamp_min(0.0) * age_decay).clamp(0.0, 1.0)
        reliability = (count_rel * spread_rel * age_rel).masked_fill(~valid, 0.0)
        return reliability.clamp(0.0, 1.0), valid

    def estimate_drift(
        self,
        feats: torch.Tensor,
        time_ids: torch.Tensor,
        class_probs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        include_future: bool = True,
        velocity_scale: float = 0.5,
        min_history: int = 2,
    ) -> torch.Tensor:
        """Estimate sample-level temporal drift without using test labels.

        The score is a cosine distance to the nearest current year-label-cluster prototype,
        optionally blended with the predicted next-year prototype distance.
        During inference, class_probs should come from detached model
        probabilities so no ground-truth label is required.
        """
        if feats is None or time_ids is None or feats.numel() == 0:
            dev = feats.device if feats is not None else (
                time_ids.device if time_ids is not None else torch.device("cpu")
            )
            return torch.zeros((0, 1), device=dev, dtype=torch.float32)

        z = F.normalize(feats.float(), dim=-1)
        time_ids = time_ids.long().view(-1).clamp(0, self.num_domains - 1)
        if time_ids.numel() != z.size(0):
            return z.new_zeros((z.size(0), 1))

        class_dist, valid_class = self._nearest_class_distances(
            z,
            time_ids,
            include_future=include_future,
            velocity_scale=velocity_scale,
            min_history=min_history,
        )

        if labels is not None:
            labels = labels.long().view(-1).to(z.device)
            if labels.numel() != z.size(0):
                return z.new_zeros((z.size(0), 1))
            valid_label = (labels >= 0) & (labels < self.num_classes)
            safe_labels = labels.clamp(0, self.num_classes - 1)
            row = torch.arange(z.size(0), device=z.device)
            label_valid = valid_label & valid_class[row, safe_labels]
            current_score = z.new_zeros((z.size(0),))
            if label_valid.any():
                current_score[label_valid] = class_dist[row[label_valid], safe_labels[label_valid]]
                has_current = label_valid
            else:
                has_current = label_valid
        else:
            if class_probs is None:
                class_probs = torch.full(
                    (z.size(0), self.num_classes),
                    1.0 / float(self.num_classes),
                    device=z.device,
                    dtype=z.dtype,
                )
            else:
                class_probs = class_probs.float().to(z.device)
                if class_probs.shape != class_dist.shape:
                    return z.new_zeros((z.size(0), 1))
            weights = class_probs.clamp_min(0.0) * valid_class.to(z.dtype)
            denom = weights.sum(dim=1).clamp_min(1e-8)
            current_score = (class_dist * weights).sum(dim=1) / denom
            has_current = weights.sum(dim=1) > 1e-8

        score = torch.where(has_current, current_score, torch.zeros_like(current_score))
        return score.view(-1, 1).clamp(0.0, 1.0)

    def estimate_risk_components(
        self,
        feats: torch.Tensor,
        time_ids: torch.Tensor,
        class_probs: Optional[torch.Tensor] = None,
        include_future: bool = True,
        velocity_scale: float = 0.5,
        min_history: int = 2,
    ) -> dict[str, torch.Tensor]:
        """Prototype diagnostics for calibrated temporal risk estimation.

        Unlike the legacy drift score, these components expose whether the
        predicted class agrees with the nearest temporal prototype class.
        """
        if feats is None or time_ids is None or feats.numel() == 0:
            dev = feats.device if feats is not None else (
                time_ids.device if time_ids is not None else torch.device("cpu")
            )
            empty = torch.zeros((0, 1), device=dev, dtype=torch.float32)
            return {
                "temporal_drift": empty,
                "prototype_pred_distance": empty,
                "prototype_margin_risk": empty,
                "prototype_label_mismatch": empty,
                "prototype_reliability_risk": empty,
            }

        z = F.normalize(feats.float(), dim=-1)
        time_ids = time_ids.long().view(-1).clamp(0, self.num_domains - 1)
        if time_ids.numel() != z.size(0):
            zeros = z.new_zeros((z.size(0), 1))
            return {
                "temporal_drift": zeros,
                "prototype_pred_distance": zeros,
                "prototype_margin_risk": zeros,
                "prototype_label_mismatch": zeros,
                "prototype_reliability_risk": zeros,
            }

        class_dist, valid_class = self._nearest_class_distances(
            z,
            time_ids,
            include_future=include_future,
            velocity_scale=velocity_scale,
            min_history=min_history,
        )
        cluster_reliability, reliability_valid = self._cluster_reliability(
            time_ids,
            device=z.device,
            dtype=z.dtype,
        )

        if class_probs is None:
            class_probs = torch.full(
                (z.size(0), self.num_classes),
                1.0 / float(self.num_classes),
                device=z.device,
                dtype=z.dtype,
            )
        else:
            class_probs = class_probs.float().to(z.device)
            if class_probs.shape != class_dist.shape:
                class_probs = torch.full_like(class_dist, 1.0 / float(self.num_classes))

        row = torch.arange(z.size(0), device=z.device)
        pred_label = class_probs.argmax(dim=1).clamp(0, self.num_classes - 1)
        pred_valid = valid_class[row, pred_label]
        pred_dist = z.new_zeros((z.size(0),))
        pred_dist[pred_valid] = class_dist[row[pred_valid], pred_label[pred_valid]].to(
                                    dtype=pred_dist.dtype,
                                    device=pred_dist.device,
                                )

        weights = class_probs.clamp_min(0.0) * valid_class.to(z.dtype)
        denom = weights.sum(dim=1).clamp_min(1e-8)
        weighted_drift = (class_dist * weights).sum(dim=1) / denom
        has_weighted = weights.sum(dim=1) > 1e-8

        masked_dist = class_dist.masked_fill(~valid_class, 1.0)
        sorted_dist, sorted_idx = masked_dist.sort(dim=1)
        best_dist = sorted_dist[:, 0]
        best_label = sorted_idx[:, 0]
        if self.num_classes > 1:
            second_dist = sorted_dist[:, 1]
        else:
            second_dist = torch.ones_like(best_dist)
        has_any = valid_class.any(dim=1)

        margin = (second_dist - best_dist).clamp(0.0, 1.0)
        margin_scale = float(ArchitectureConstants.PROTO_MARGIN_RISK_SCALE)
        margin_risk = (1.0 - margin / max(margin_scale, 1e-6)).clamp(0.0, 1.0)
        margin_risk = torch.where(has_any, margin_risk, torch.zeros_like(margin_risk))

        mismatch = (pred_label != best_label) & pred_valid & has_any
        label_mismatch = mismatch.to(z.dtype)

        class_reliability = cluster_reliability.masked_fill(~reliability_valid, 0.0).max(dim=2).values
        class_reliability = class_reliability.masked_fill(~valid_class, 0.0)
        pred_reliability = z.new_zeros((z.size(0),))
        pred_reliability[pred_valid] = class_reliability[row[pred_valid], pred_label[pred_valid]].to(
            dtype=pred_reliability.dtype,
            device=pred_reliability.device,
        )
        reliability_risk = torch.where(
            has_any,
            1.0 - pred_reliability,
            torch.zeros_like(pred_reliability),
        ).clamp(0.0, 1.0)

        weighted_drift = torch.where(has_weighted, weighted_drift, pred_dist)
        return {
            "temporal_drift": weighted_drift.view(-1, 1).clamp(0.0, 1.0),
            "prototype_pred_distance": pred_dist.view(-1, 1).clamp(0.0, 1.0),
            "prototype_margin_risk": margin_risk.view(-1, 1).clamp(0.0, 1.0),
            "prototype_label_mismatch": label_mismatch.view(-1, 1).clamp(0.0, 1.0),
            "prototype_reliability_risk": reliability_risk.view(-1, 1).clamp(0.0, 1.0),
        }
