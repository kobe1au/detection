import torch
from torch_geometric.data import Batch, Data

from fusion.model import MalwareModelWithXAttn


def make_graph_batch(batch_size: int = 2, in_feat_dim: int = 8):
    data_list = []
    for _ in range(batch_size):
        x = torch.randn(3, in_feat_dim)
        edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]],
            dtype=torch.long,
        )
        data = Data(x=x, edge_index=edge_index)
        data.sensitive_mask = torch.tensor([1, 0, 1], dtype=torch.uint8)
        data_list.append(data)

    batch = Batch.from_data_list(data_list)
    per_sample_api = 4
    total_api = batch_size * per_sample_api
    batch.api_ids = torch.arange(total_api, dtype=torch.long) + 1
    batch.api_type_ids = torch.tensor([1, 2, 0, 3] * batch_size, dtype=torch.long)
    batch.api_sensitive_mask = torch.tensor([1.0, 0.0, 0.0, 1.0] * batch_size)
    batch.api_in_graph_mask = torch.tensor([1.0, 1.0, 0.0, 1.0] * batch_size)
    batch.api_method_index = torch.tensor([0, 1, -1, 2] * batch_size, dtype=torch.long)
    batch.api_batch = torch.arange(batch_size).repeat_interleave(per_sample_api)
    batch.method_api_edge_index = torch.empty((2, 0), dtype=torch.long)
    batch.api_category_counts = torch.zeros((batch_size, 16), dtype=torch.float32)
    return batch


def make_masks(batch_size: int = 2):
    masks = []
    for _ in range(batch_size):
        m = torch.zeros((3, 4), dtype=torch.float32)
        m[0, 0] = 1.0
        m[1, 1] = 0.5
        m[2, 3] = 1.0
        masks.append(m)
    return masks


def make_explicit_qs(batch_size: int = 2):
    values = []
    for v in (0.8, 0.7, 0.6, 0.0, 0.0):
        values.append(torch.full((batch_size, 1), float(v)))
    return tuple(values)


def make_model(
    fusion_mode: str,
    *,
    gate_mode: str = "learned",
    use_time_gate_inputs: bool = False,
    use_temporal_reliability: bool = False,
    use_drift_reliability: bool = False,
    num_time_domains: int = 2,
    historical_time_id_max: int = 0,
):
    return MalwareModelWithXAttn(
        num_classes=2,
        api_emb_dim=16,
        graph_emb_dim=16,
        align_dim=16,
        max_nodes_gnn=16,
        max_xattn_nodes=8,
        in_feat_dim=8,
        fusion_mode=fusion_mode,
        graph_encoder_type="gcn",
        graph_hidden=16,
        graph_heads=2,
        graph_layers=1,
        api_encoder_type="bigru",
        api_max_seq_len=16,
        api_heads=2,
        api_layers=1,
        xattn_heads=2,
        gate_mode=gate_mode,
        num_time_domains=num_time_domains,
        historical_time_id_max=historical_time_id_max,
        use_time_gate_inputs=use_time_gate_inputs,
        use_temporal_reliability=use_temporal_reliability,
        use_drift_reliability=use_drift_reliability,
    )


def make_loss_cfg():
    return {
        "semantic_alignment_weight": 0.03,
        "class_aware_alignment_same_class_weight": 0.25,
        "class_aware_alignment_temperature": 0.2,
        "max_local_align_nodes": 8,
        "max_local_align_tokens": 8,
        "stage1_branch_aux_weight": 0.0,
        "branch_aux_weight": 0.10,
    }
