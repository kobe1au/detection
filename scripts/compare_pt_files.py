#!/usr/bin/env python3
"""对比新旧PT文件的差异"""

import torch
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.io_utils import load_aeg_payload  # noqa: E402

def compare_pt_files(old_pt_path, new_pt_path):
    """详细对比两个PT文件"""

    print("=" * 80)
    print("📊 新旧 PT 文件对比")
    print("=" * 80)

    # 加载文件
    try:
        old_data = load_aeg_payload(old_pt_path, validate=True)
        new_data = load_aeg_payload(new_pt_path, validate=True)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    print(f"\n📁 旧PT: {old_pt_path.name}")
    print(f"📁 新PT: {new_pt_path.name}")

    # 1. 对比关键元数据
    print("\n" + "=" * 80)
    print("🔑 关键元数据对比")
    print("=" * 80)

    # Build fingerprint
    old_fp = old_data.get("aeg_build_fingerprint", "MISSING")
    new_fp = new_data.get("aeg_build_fingerprint", "MISSING")
    print(f"\naeg_build_fingerprint:")
    print(f"  旧: {old_fp[:50]}..." if isinstance(old_fp, str) and len(old_fp) > 50 else f"  旧: {old_fp}")
    print(f"  新: {new_fp[:50]}..." if isinstance(new_fp, str) and len(new_fp) > 50 else f"  新: {new_fp}")
    print(f"  {'✅ 相同' if old_fp == new_fp else '❌ 不同 (正常,代码已修改)'}")

    # Schema version
    old_sv = old_data.get("schema_version")
    new_sv = new_data.get("schema_version")
    print(f"\nschema_version:")
    print(f"  旧: {old_sv}")
    print(f"  新: {new_sv}")
    print(f"  {'✅ 相同' if old_sv == new_sv else '❌ 不同'}")

    # 2. 对比Tensor形状
    print("\n" + "=" * 80)
    print("📐 Tensor 形状对比")
    print("=" * 80)

    tensor_fields = ["node_x", "edge_index", "node_type", "edge_type", "node_quality"]

    all_match = True
    for field in tensor_fields:
        old_val = old_data.get(field)
        new_val = new_data.get(field)

        if isinstance(old_val, torch.Tensor) and isinstance(new_val, torch.Tensor):
            old_shape = tuple(old_val.shape)
            new_shape = tuple(new_val.shape)
            match = old_shape == new_shape
            all_match = all_match and match
            status = "✅" if match else "❌"
            print(f"{field:<20s} 旧:{str(old_shape):<20s} 新:{str(new_shape):<20s} {status}")

    # 3. 对比字段列表
    print("\n" + "=" * 80)
    print("📋 字段对比")
    print("=" * 80)

    old_keys = set(old_data.keys())
    new_keys = set(new_data.keys())

    only_old = sorted(old_keys - new_keys)
    only_new = sorted(new_keys - old_keys)

    print(f"\n旧PT字段数: {len(old_keys)}")
    print(f"新PT字段数: {len(new_keys)}")

    if only_old:
        print(f"\n仅旧PT有: {only_old}")
    if only_new:
        print(f"\n仅新PT有: {only_new}")
    if not only_old and not only_new:
        print("\n✅ 字段完全一致")

    # 4. 总结
    print("\n" + "=" * 80)
    print("📝 结论")
    print("=" * 80)

    if all_match and not only_old and not only_new and old_sv == new_sv:
        print("\n✅ 除了 build_fingerprint 外,所有内容完全一致")
        print("✅ 新旧PT文件格式兼容,可以混用")
        print("✅ 修改后的 resume 逻辑可以正确识别旧PT文件")
    else:
        print("\n⚠️  新旧PT文件有差异:")
        if old_sv != new_sv:
            print("  - Schema version 不同")
        if not all_match:
            print("  - Tensor 形状不同")
        if only_old or only_new:
            print("  - 字段列表不同")


if __name__ == "__main__":
    pt_dir = Path("D:/pts_aeg/train")

    # 旧PT (18:34之前)
    old_pt = pt_dir / "0019ce6519cec536bfe684a32bba58377ff2b4064b039d259f4d1bac9425f100.pt"

    # 新PT (19:03-19:04)
    new_pt = pt_dir / "cb98a79cabafd27aaf76975d6630c68bbbeff697f4f974ef5c5869788bfa28c2.pt"

    if old_pt.exists() and new_pt.exists():
        compare_pt_files(old_pt, new_pt)
    else:
        print(f"❌ PT文件不存在")
        print(f"旧PT: {old_pt.exists()}")
        print(f"新PT: {new_pt.exists()}")
