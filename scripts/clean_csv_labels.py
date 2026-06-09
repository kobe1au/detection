#!/usr/bin/env python3
"""
清理 CSV 标签文件，删除没有对应 PT 文件的样本行
"""
import csv
import sys
from pathlib import Path
from typing import Dict, Set, Tuple


def get_pt_files(pt_dir: Path) -> Set[str]:
    """获取 PT 目录中所有文件的 SHA256（不带 .pt 扩展名）"""
    if not pt_dir.exists():
        print(f"警告: PT 目录不存在: {pt_dir}")
        return set()
    
    pt_files = set()
    for pt_file in pt_dir.glob("*.pt"):
        # 文件名就是 SHA256
        sha256 = pt_file.stem.lower()
        pt_files.add(sha256)
    
    return pt_files


def clean_csv(
    csv_path: Path,
    pt_files: Set[str],
    output_path: Path,
    backup: bool = True
) -> Tuple[int, int]:
    """
    清理 CSV 文件，只保留有对应 PT 文件的行
    
    返回: (原始行数, 保留行数)
    """
    if not csv_path.exists():
        print(f"错误: CSV 文件不存在: {csv_path}")
        return 0, 0
    
    # 备份原文件
    if backup:
        backup_path = csv_path.with_suffix('.csv.bak')
        if not backup_path.exists():
            import shutil
            shutil.copy2(csv_path, backup_path)
            print(f"    [+] Backed up: {backup_path}")
    
    # 读取 CSV
    rows_to_keep = []
    total_rows = 0
    deleted_samples = []
    
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        if not fieldnames:
            print(f"错误: CSV 文件为空: {csv_path}")
            return 0, 0
        
        # 确定 SHA256 列名
        sha256_field = None
        for field in ['sha256', 'id', 'SHA256', 'ID']:
            if field in fieldnames:
                sha256_field = field
                break
        
        if not sha256_field:
            print(f"错误: 找不到 SHA256 或 ID 列: {csv_path}")
            print(f"可用列: {fieldnames}")
            return 0, 0
        
        for row in reader:
            total_rows += 1
            sha256 = row[sha256_field].strip().lower()
            
            if not sha256:
                print(f"警告: 第 {total_rows} 行缺少 SHA256")
                continue
            
            if sha256 in pt_files:
                rows_to_keep.append(row)
            else:
                deleted_samples.append({
                    'sha256': sha256,
                    'label': row.get('label', ''),
                    'split': row.get('split', ''),
                    'pkg_name': row.get('pkg_name', '')
                })
    
    # 写入清理后的 CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        if rows_to_keep:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_to_keep)
    
    kept_rows = len(rows_to_keep)
    deleted_rows = total_rows - kept_rows
    
    print(f"    Stats for {csv_path.name}:")
    print(f"      Original rows: {total_rows}")
    print(f"      Kept rows:     {kept_rows}")
    print(f"      Deleted rows:  {deleted_rows}")
    
    if deleted_samples:
        deleted_log = output_path.parent / f"deleted_{csv_path.stem}.csv"
        with open(deleted_log, 'w', encoding='utf-8', newline='') as f:
            if deleted_samples:
                writer = csv.DictWriter(f, fieldnames=['sha256', 'label', 'split', 'pkg_name'])
                writer.writeheader()
                writer.writerows(deleted_samples)
        print(f"      Deleted log:   {deleted_log}")
    
    return total_rows, kept_rows


def main():
    # 配置路径
    project_root = Path(__file__).resolve().parent.parent
    
    # PT 文件目录
    pt_dirs = {
        'train': Path('D:/pts_aeg/train'),
        'val': Path('D:/pts_aeg/val'),
        'test': Path('D:/pts_aeg/test')
    }
    
    # CSV 文件目录
    csv_dir = project_root / 'results' / 'labels'
    csv_files = {
        'train': csv_dir / 'train.csv',
        'val': csv_dir / 'val.csv',
        'test': csv_dir / 'test.csv'
    }
    
    print("=" * 60)
    print("Clean CSV labels - Remove samples without PT files")
    print("=" * 60)
    
    # 统计信息
    total_stats = {'total': 0, 'kept': 0, 'deleted': 0}
    
    for split in ['train', 'val', 'test']:
        print(f"\n[*] Processing {split} split...")
        
        # 获取 PT 文件列表
        pt_dir = pt_dirs[split]
        pt_files = get_pt_files(pt_dir)
        print(f"    PT files count: {len(pt_files)}")
        
        if not pt_files:
            print(f"    [!] Warning: No PT files in {pt_dir}")
            continue
        
        # 清理 CSV
        csv_path = csv_files[split]
        if not csv_path.exists():
            print(f"    [!] Warning: CSV not found: {csv_path}")
            continue
        
        total, kept = clean_csv(
            csv_path=csv_path,
            pt_files=pt_files,
            output_path=csv_path,
            backup=True
        )
        
        total_stats['total'] += total
        total_stats['kept'] += kept
        total_stats['deleted'] += (total - kept)
    
    # 总结
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"   Total original rows: {total_stats['total']}")
    print(f"   Total kept rows:     {total_stats['kept']}")
    print(f"   Total deleted rows:  {total_stats['deleted']}")
    if total_stats['total'] > 0:
        print(f"   Keep ratio:          {total_stats['kept']/total_stats['total']*100:.2f}%")
    print("=" * 60)
    print("\n[+] Cleaning completed!")
    print("\nNotes:")
    print("   - Original files backed up as .csv.bak")
    print("   - Deleted records saved in deleted_*.csv")
    print("   - To restore, rename .bak files manually")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] User interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
