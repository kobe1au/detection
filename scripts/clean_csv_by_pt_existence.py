#!/usr/bin/env python3
"""
清理 CSV 文件：删除在 PT 目录中不存在的样本行

比对 results/labels/{train,val,test}.csv 和 D:/pts_aeg/{train,val,test} 中的 PT 文件，
删除 CSV 中那些对应 PT 文件不存在的数据行。
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def get_pt_files(pt_dir: Path) -> Set[str]:
    """获取 PT 目录中所有文件的 stem（不含扩展名的文件名）"""
    if not pt_dir.exists():
        print(f"警告: PT 目录不存在: {pt_dir}")
        return set()
    
    pt_files = set()
    for pt_file in pt_dir.glob("*.pt"):
        # 使用 stem 去掉 .pt 扩展名
        stem = pt_file.stem.lower()
        pt_files.add(stem)
    
    print(f"  找到 {len(pt_files)} 个 PT 文件")
    return pt_files


def clean_csv_file(
    csv_path: Path,
    pt_files: Set[str],
    output_path: Path,
    id_column: str = "sha256"
) -> Tuple[int, int, int]:
    """
    清理单个 CSV 文件
    
    返回: (总行数, 保留行数, 删除行数)
    """
    if not csv_path.exists():
        print(f"错误: CSV 文件不存在: {csv_path}")
        return 0, 0, 0
    
    kept_rows = []
    deleted_rows = []
    
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        if not fieldnames or id_column not in fieldnames:
            print(f"错误: CSV 文件缺少 '{id_column}' 列")
            return 0, 0, 0
        
        for row in reader:
            sample_id = row[id_column].strip().lower()
            
            # 检查对应的 PT 文件是否存在
            if sample_id in pt_files:
                kept_rows.append(row)
            else:
                deleted_rows.append(row)
    
    # 写入清理后的 CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)
    
    total = len(kept_rows) + len(deleted_rows)
    return total, len(kept_rows), len(deleted_rows)


def main():
    # 项目根目录
    project_root = Path(__file__).parent.parent
    
    # CSV 文件路径
    csv_base_dir = project_root / "results" / "labels"
    
    # PT 文件根目录
    pt_base_dir = Path("D:/pts_aeg")
    
    # 需要处理的 split
    splits = ["train", "val", "test"]
    
    print("=" * 60)
    print("开始清理 CSV 文件（删除不存在 PT 文件的样本）")
    print("=" * 60)
    print(f"CSV 目录: {csv_base_dir}")
    print(f"PT 目录: {pt_base_dir}")
    print()
    
    # 统计信息
    total_stats = {
        "total": 0,
        "kept": 0,
        "deleted": 0
    }
    
    for split in splits:
        print(f"处理 {split} split...")
        
        # CSV 文件路径
        csv_path = csv_base_dir / f"{split}.csv"
        
        # PT 目录路径
        pt_dir = pt_base_dir / split
        
        if not csv_path.exists():
            print(f"  跳过: CSV 文件不存在 - {csv_path}")
            continue
        
        # 获取 PT 文件列表
        print(f"  扫描 PT 目录: {pt_dir}")
        pt_files = get_pt_files(pt_dir)
        
        if not pt_files:
            print(f"  警告: PT 目录为空或不存在，跳过")
            continue
        
        # 创建备份
        backup_path = csv_base_dir / f"{split}.csv.backup"
        print(f"  创建备份: {backup_path.name}")
        import shutil
        shutil.copy2(csv_path, backup_path)
        
        # 清理 CSV
        print(f"  清理 CSV 文件...")
        total, kept, deleted = clean_csv_file(
            csv_path=csv_path,
            pt_files=pt_files,
            output_path=csv_path,  # 直接覆盖原文件
            id_column="sha256"
        )
        
        # 更新统计
        total_stats["total"] += total
        total_stats["kept"] += kept
        total_stats["deleted"] += deleted
        
        # 打印结果
        print(f"  结果:")
        print(f"    原始行数: {total}")
        print(f"    保留行数: {kept} ({kept/total*100:.1f}%)" if total > 0 else "    保留行数: 0")
        print(f"    删除行数: {deleted} ({deleted/total*100:.1f}%)" if total > 0 else "    删除行数: 0")
        
        # 保存删除的样本列表
        if deleted > 0:
            deleted_log_path = csv_base_dir / f"{split}_deleted_samples.txt"
            print(f"  保存删除的样本列表: {deleted_log_path.name}")
            # 重新读取找出被删除的样本
            deleted_ids = []
            with open(backup_path, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sample_id = row["sha256"].strip().lower()
                    if sample_id not in pt_files:
                        deleted_ids.append(sample_id)
            
            with open(deleted_log_path, 'w', encoding='utf-8') as f:
                f.write(f"# {split} split - 删除的样本 (PT 文件不存在)\n")
                f.write(f"# 总计: {len(deleted_ids)} 个样本\n\n")
                for sample_id in deleted_ids:
                    f.write(f"{sample_id}\n")
        
        print()
    
    # 打印总体统计
    print("=" * 60)
    print("清理完成！总体统计:")
    print("=" * 60)
    print(f"总样本数: {total_stats['total']}")
    print(f"保留样本: {total_stats['kept']} ({total_stats['kept']/total_stats['total']*100:.1f}%)" if total_stats['total'] > 0 else "保留样本: 0")
    print(f"删除样本: {total_stats['deleted']} ({total_stats['deleted']/total_stats['total']*100:.1f}%)" if total_stats['total'] > 0 else "删除样本: 0")
    print()
    print("备份文件保存在: results/labels/*.csv.backup")
    print("删除的样本列表: results/labels/*_deleted_samples.txt")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
