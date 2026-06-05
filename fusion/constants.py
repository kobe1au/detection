from __future__ import annotations

AEG_SCHEMA_VERSION = 4


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
    "METHOD_CALLS_METHOD": 1,
    "METHOD_INVOKES_API_FAMILY": 2,
    "APK_REQUESTS_PERMISSION": 3,
    "APK_HAS_COMPONENT": 4,
    "COMPONENT_DECLARES_INTENT": 5,
    "COMPONENT_MATCHES_METHOD": 6,
    "PERMISSION_RELATED_TO_API_FAMILY": 7,
    "METHOD_HAS_RISK": 8,
    "MANIFEST_HAS_RISK": 9,
    "METHOD_HAS_STRING_HINT": 10,
}

EDGE_TYPE_NAMES = tuple(sorted(EDGE_TYPES, key=EDGE_TYPES.get))
NUM_EDGE_TYPES = len(EDGE_TYPES)


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
    "all_degraded": 7,
    "api_missing": 8,
    "graph_missing": 9,
    "manifest_missing": 10,
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
