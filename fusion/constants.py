from __future__ import annotations

import hashlib
import json

AEG_SCHEMA_VERSION = 6
AEG_EXTRACTION_PIPELINE_VERSION = 1
AEG_PAYLOAD_CONTRACT_VERSION = 2


NODE_TYPES = {
    "APK": 0,
    "METHOD": 1,
    "API_FAMILY": 2,
    "PERMISSION": 3,
    "INTENT": 4,
    "COMPONENT": 5,
    "RISK_SEMANTIC": 6,
    "STRING_HINT": 7,
}

NODE_TYPE_NAMES = tuple(sorted(NODE_TYPES, key=NODE_TYPES.get))
NUM_NODE_TYPES = len(NODE_TYPES)


EDGE_TYPES = {
    "APK_HAS_METHOD": 0,
    "METHOD_IN_APK": 1,
    "METHOD_CALLS_METHOD": 2,
    "METHOD_CALLED_BY_METHOD": 3,
    "METHOD_INVOKES_API_FAMILY": 4,
    "API_FAMILY_INVOKED_BY_METHOD": 5,
    "APK_REQUESTS_PERMISSION": 6,
    "PERMISSION_REQUESTED_BY_APK": 7,
    "APK_HAS_COMPONENT": 8,
    "COMPONENT_IN_APK": 9,
    "COMPONENT_DECLARES_INTENT": 10,
    "INTENT_DECLARED_BY_COMPONENT": 11,
    "COMPONENT_MATCHES_METHOD": 12,
    "METHOD_MATCHES_COMPONENT": 13,
    "PERMISSION_RELATED_TO_API_FAMILY": 14,
    "API_FAMILY_RELATED_TO_PERMISSION": 15,
    "METHOD_HAS_RISK": 16,
    "RISK_OBSERVED_IN_METHOD": 17,
    "MANIFEST_HAS_RISK": 18,
    "RISK_DECLARED_BY_MANIFEST": 19,
    "METHOD_HAS_STRING_HINT": 20,
    "STRING_HINT_IN_METHOD": 21,
}

EDGE_TYPE_NAMES = tuple(sorted(EDGE_TYPES, key=EDGE_TYPES.get))
NUM_EDGE_TYPES = len(EDGE_TYPES)

REVERSE_EDGE_TYPES = {
    "APK_HAS_METHOD": "METHOD_IN_APK",
    "METHOD_CALLS_METHOD": "METHOD_CALLED_BY_METHOD",
    "METHOD_INVOKES_API_FAMILY": "API_FAMILY_INVOKED_BY_METHOD",
    "APK_REQUESTS_PERMISSION": "PERMISSION_REQUESTED_BY_APK",
    "APK_HAS_COMPONENT": "COMPONENT_IN_APK",
    "COMPONENT_DECLARES_INTENT": "INTENT_DECLARED_BY_COMPONENT",
    "COMPONENT_MATCHES_METHOD": "METHOD_MATCHES_COMPONENT",
    "PERMISSION_RELATED_TO_API_FAMILY": "API_FAMILY_RELATED_TO_PERMISSION",
    "METHOD_HAS_RISK": "RISK_OBSERVED_IN_METHOD",
    "MANIFEST_HAS_RISK": "RISK_DECLARED_BY_MANIFEST",
    "METHOD_HAS_STRING_HINT": "STRING_HINT_IN_METHOD",
}


SOURCE_TYPES = {
    "code": 0,
    "manifest": 1,
    "derived": 2,
    "alignment": 3,
}

SOURCE_TYPE_NAMES = tuple(sorted(SOURCE_TYPES, key=SOURCE_TYPES.get))
NUM_SOURCE_TYPES = len(SOURCE_TYPES)


VIEW_TYPES = {
    "clean": 0,
    "api_degraded": 1,
    "graph_degraded": 2,
    "api_graph_degraded": 3,
    "manifest_degraded": 4,
    "manifest_zeroed": 5,
    "manifest_noisy": 6,
    "manifest_shuffled": 7,
    "all_degraded": 8,
    "api_missing": 9,
    "graph_missing": 10,
    "manifest_missing": 11,
    "manifest_noisy_blind": 12,
    "manifest_shuffled_blind": 13,
}

VIEW_TYPE_NAMES = tuple(sorted(VIEW_TYPES, key=VIEW_TYPES.get))


STRING_HINT_KEYWORDS = {
    "reflection": ("reflect", "class#forname", "getmethod", "invoke"),
    "dynamic_loading": ("dexclassloader", "pathclassloader", "loadclass", ".dex", ".jar"),
    "native": ("loadlibrary", ".so", "native"),
    "network": ("http", "https", "socket", "urlconnection", "okhttp"),
    "crypto": ("cipher", "message_digest", "crypto", "keystore"),
}


class QualityConstants:
    """Heuristic quality normalizers used when building AEG records.

    These values describe extraction integrity, not maliciousness. They are
    intentionally conservative because the downstream model should learn from
    typed evidence, not from inflated quality shortcuts.
    """

    API_COUNT_NORM = 200.0
    API_DIVERSITY_SCALE = 4.0
    API_COUNT_WEIGHT = 0.35
    API_DIVERSITY_WEIGHT = 0.20
    API_COVERAGE_WEIGHT = 0.25
    API_TYPE_WEIGHT = 0.20

    GRAPH_NODE_NORM = 120.0
    GRAPH_NODE_WEIGHT = 0.35
    GRAPH_EDGE_WEIGHT = 0.35
    GRAPH_FEATURE_WEIGHT = 0.30

    ALIGN_NODE_COVER_WEIGHT = 0.50
    ALIGN_API_COVER_WEIGHT = 0.50


def stable_table_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


AEG_SCHEMA_TABLES = {
    "node_types": NODE_TYPES,
    "edge_types": EDGE_TYPES,
    "source_types": SOURCE_TYPES,
    "view_types": VIEW_TYPES,
}

AEG_SCHEMA_TABLE_FINGERPRINT = stable_table_hash(AEG_SCHEMA_TABLES)


AEG_REQUIRED_PAYLOAD_FIELDS = (
    "schema_version",
    "aeg_schema_fingerprint",
    "aeg_payload_contract_version",
    "aeg_payload_contract_fingerprint",
    "aeg_meta",
    "sid",
    "sha256",
    "apk_name",
    "split",
    "package_name",
    "year",
    "sample_meta",
    "storage_dtype",
    "node_x",
    "node_type",
    "node_source",
    "node_quality",
    "node_semantic",
    "edge_index",
    "edge_type",
    "edge_quality",
    "edge_source",
    "api_semantic_category_counts",
    "graph_semantic_category_counts",
    "manifest_category_counts",
    "manifest_component_category_counts",
    "manifest_stats",
    "q_api",
    "q_graph",
    "q_manifest",
    "q_align",
    "pert_api",
    "pert_graph",
    "pert_manifest",
    "manifest_parse_ok",
    "manifest_parse_error",
    "dex_success_ratio",
    "multi_dex_total",
    "multi_dex_success",
    "graph_behavior_hints",
    "graph_behavior_hint_start",
    "graph_behavior_hint_dim",
    "has_reflection",
    "has_dynamic_loading",
    "has_native",
    "has_string_encryption_hint",
)

AEG_PAYLOAD_CONTRACT_FINGERPRINT = stable_table_hash(
    {
        "version": AEG_PAYLOAD_CONTRACT_VERSION,
        "required_fields": AEG_REQUIRED_PAYLOAD_FIELDS,
    }
)
