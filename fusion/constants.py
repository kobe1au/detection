class ArchitectureConstants:
    # ── API sequence encoder ──
    API_NUM_HASH_BUCKETS = 8192
    API_TYPE_VOCAB_SIZE = 16
    API_TOKEN_EMB_DIM = 128
    API_PROJ_HIDDEN = 256
    API_DROPOUT = 0.15

    # ── Classification head ──
    HEAD_HIDDEN_DIMS = [256, 128]
    HEAD_DROPOUT_RATES = [0.3, 0.2]
    HEAD_DROPOUT = 0.2
    FEATURE_DROPOUT = 0.0

    # ── Quality gate ──
    GATE_HIDDEN_DIM = 128
    GATE_INIT_BIAS = 0.0

    # ── Cross-attention ──
    XATTN_DROPOUT = 0.1

    # ── Alignment-aware cross-attention ──
    MASK_HARD_NEG = -1e4
    ALIGN_BIAS_PENALTY = -2.0
    ALIGN_BIAS_BONUS = 0.5
    ALIGN_CONTEXT_SCALE = 0.35

    # ── Modality alive detection ──
    MODALITY_ALIVE_THRESHOLD = 0.01

    # ── Prototype soft weighting ──
    PROTO_SOFT_WEIGHT_POWER = 2
    PROTO_SOFT_WEIGHT_BASE = 0.1

    # ── Drift score weights ──
    DRIFT_W_TEMPORAL = 0.4
    DRIFT_W_DISAGREE = 0.3
    DRIFT_W_ENTROPY = 0.3
    PROTO_MARGIN_RISK_SCALE = 0.25
    PROTO_RELIABILITY_COUNT_SCALE = 32.0
    PROTO_RELIABILITY_SPREAD_SCALE = 0.25
    PROTO_RELIABILITY_AGE_DECAY = 0.35
    PROTO_MAX_VELOCITY_NORM = 0.5
    PROTO_FUTURE_BLEND_MAX = 0.3
    TEMPORAL_RISK_POS_WEIGHT_MAX = 5.0
    TEMPORAL_RISK_RANK_MARGIN = 0.20
    TEMPORAL_RISK_RANK_WEIGHT = 0.50


class TrainingConstants:
    WARMUP_INIT_SCALE = 0.01
    GRAD_CLIP_MAX_NORM = 1.0

    API_EMB_DIM = 128
    GRAPH_EMB_DIM = 128
    ALIGN_DIM = 128
    IN_FEAT_DIM = 515
    XATTN_HEADS = 4


class AugmentationConstants:
    SEMANTIC_QUALITY_ALPHA = 0.3
    ALIGN_SENSITIVE_COVER_WEIGHT = 0.30
    ALIGN_NODE_COVER_WEIGHT = 0.20
    ALIGN_QAPI_WEIGHT = 0.25
    ALIGN_QGRAPH_WEIGHT = 0.25

    GRAPH_LOCAL_BREAK_WEAK_STRENGTH = 0.15
    GRAPH_REWIRE_RATIO = 0.30

    TEMPORAL_AUG_DELTA_MIN = 0.02
    TEMPORAL_AUG_DELTA_MAX = 0.05

    EVAL_PERTURB_TYPES = {
        None,
        "api_event_dropout",
        "api_sensitive_event_dropout",
        "modality_dropout_api",
        "graph_sparsify",
        "graph_local_break",
        "graph_target_redirection",
        "graph_control_flow_flattening",
        "graph_dead_code_injection",
        "graph_feature_obfuscation",
        "modality_dropout_graph",
    }

    API_AUG_TYPES = [
        "api_event_dropout",
        "api_sensitive_event_dropout",
    ]
    API_AUG_WEIGHTS = [0.65, 0.35]

    GRAPH_AUG_TYPES = [
        "graph_sparsify",
        "graph_local_break",
        "graph_target_redirection",
        "graph_control_flow_flattening",
        "graph_dead_code_injection",
        "graph_feature_obfuscation",
    ]


class BucketSamplerConstants:
    NODE_WEIGHT = 1.0
    API_EVENT_WEIGHT = 0.5
    COMPLEXITY_WEIGHT = 0.0

    DROP_LAST_THRESHOLD = 0.5


class GraphPretrainConstants:
    MASK_RATIO = 0.25
    SENSITIVE_MASK_BOOST = 2.0
    EDGE_PRED_HIDDEN = 128
    FEAT_RECON_WEIGHT = 0.5
    EDGE_PRED_WEIGHT = 1.0
    PRETRAIN_LR = 1e-3
    PRETRAIN_EPOCHS = 20
    PRETRAIN_BATCH_SIZE = 64


class QualityLearnerConstants:
    QUALITY_EMB_DIM = 32
    QUALITY_HIDDEN_DIM = 64
    QUALITY_TEMPERATURE = 0.1

    LIGHT_AUG_THRESHOLD = 0.2
    HEAVY_AUG_THRESHOLD = 0.5
