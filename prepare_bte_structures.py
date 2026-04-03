#!/usr/bin/env python3
"""
准备 BTE 验证材料的初始结构
================================
从 bte4mp_iso 的 monotonic_decreasing 结果中：
1. 过滤掉金属
2. 按 Δκ 降序排列
3. 将原胞 POSCAR 复制到 qflow 的 structures 目录

用法：
    python prepare_bte_structures.py                    # 准备所有材料
    python prepare_bte_structures.py --top 20           # 只准备前 20 个
    python prepare_bte_structures.py --dry-run           # 只打印列表
    python prepare_bte_structures.py --target-dir /path  # 指定输出目录
"""

import csv
import shutil
import argparse
from pathlib import Path

# 金属 mp_id（排除）
METAL_IDS = {
    'mp-6988', 'mp-24728', 'mp-1018157', 'mp-1009770', 'mp-24154',
    'mp-1215913', 'mp-1217106', 'mp-117', 'mp-974920', 'mp-974558',
    'mp-866134', 'mp-1211', 'mp-580234', 'mp-1232560', 'mp-13915',
    'mp-1185260', 'mp-10675', 'mp-972285', 'mp-1187090', 'mp-865562',
    'mp-1222280', 'mp-10198', 'mp-1208897', 'mp-1183423', 'mp-1226009',
    'mp-1017629', 'mp-1749', 'mp-864758', 'mp-1183468', 'mp-1187711',
    'mp-983566', 'mp-2132',
}

BTE_BASE = Path('/data/bte4mp_iso')
SUMMARY_CSV = BTE_BASE / 'organized_results' / 'monotonic_decreasing' / 'summary.csv'
STRUCTURES_DIR = BTE_BASE / 'structures'


def load_materials():
    """加载并筛选非金属材料，按 Δκ 降序"""
    materials = []
    with open(SUMMARY_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['mp_id'] not in METAL_IDS:
                row['max_drop_abs'] = float(row['max_drop_abs'])
                row['kappa_0GPa'] = float(row['kappa_0GPa'])
                row['max_drop_pct'] = float(row['max_drop_pct'])
                materials.append(row)
    materials.sort(key=lambda x: x['max_drop_abs'], reverse=True)
    return materials


def main():
    parser = argparse.ArgumentParser(description='Prepare BTE structures for qflow')
    parser.add_argument('--top', type=int, default=None, help='Only top N materials')
    parser.add_argument('--dry-run', action='store_true', help='Only print list')
    parser.add_argument('--target-dir', type=str, default=None,
                        help='Target structures directory (default: ./bte_structures)')
    args = parser.parse_args()

    materials = load_materials()
    if args.top:
        materials = materials[:args.top]

    target_dir = Path(args.target_dir) if args.target_dir else Path('./bte_structures')

    print(f"Non-metal monotonic_decreasing materials: {len(materials)}")
    print(f"Target directory: {target_dir}")
    print()
    print(f"{'Rank':>4s}  {'mp_id':>15s}  {'Formula':>12s}  {'κ₀':>8s}  {'Δκ':>8s}  {'Drop%':>6s}")
    print('-' * 62)

    copied = 0
    for i, m in enumerate(materials):
        rank = i + 1
        mp_id = m['mp_id']
        formula = m['formula']
        print(f"{rank:4d}  {mp_id:>15s}  {formula:>12s}  "
              f"{m['kappa_0GPa']:8.2f}  {m['max_drop_abs']:8.2f}  {m['max_drop_pct']:5.1f}%", end='')

        if args.dry_run:
            print()
            continue

        # 查找 POSCAR 源文件
        src_poscar = STRUCTURES_DIR / mp_id / 'POSCAR'
        if not src_poscar.exists():
            print(f"  [SKIP] no POSCAR")
            continue

        # 创建目标目录
        dest_dir = target_dir / mp_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_poscar = dest_dir / 'POSCAR'

        if dest_poscar.exists():
            print(f"  [EXISTS]")
        else:
            shutil.copy2(str(src_poscar), str(dest_poscar))
            print(f"  [COPIED]")
            copied += 1

    if not args.dry_run:
        print(f"\nCopied {copied} new structures to {target_dir}")
        print(f"Total structures in target: {sum(1 for d in target_dir.iterdir() if d.is_dir())}")


if __name__ == '__main__':
    main()
