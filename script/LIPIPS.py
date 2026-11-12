# -*- coding: utf-8 -*-
"""
LPIPS evaluation for Base SCMix and LoRA-adapted SCMix.

Per-class pairing: each generated image is compared against all 5 references
of its category and averaged (mean, not min).
  - batch level: mean of per-image means over 30 images.
  - overall: mean±std over 21 batch means.
  - per-class: mean±std over 3 batch means.

Outputs (S:\ComfyUI\LIPIPS\):
  lpips_details.csv
  lpips_batch.csv
  lpips_summary.csv
  lpips_stats.txt
"""

import os
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import lpips
from PIL import Image
from tqdm import tqdm
import pandas as pd


# ==================== 配置 ====================
ROOT = Path(r"S:\ComfyUI\LIPIPS")
REF_ROOT = ROOT / "held-out"

MODELS = {
    "Base SCMix": ROOT / "pattern_base",
    "LoRA-adapted SCMix": ROOT / "pattern",
}

CATEGORIES = ["cloud", "floral", "grass", "hui", "lotus", "peony", "scroll"]
BATCHES = ["batch_01", "batch_02", "batch_03"]

HELDOUT_MAP = {
    "cloud": "Cloud Pattern",
    "floral": "Floral Pattern",
    "grass": "Grass Pattern",
    "hui": "Hui Pattern",
    "lotus": "Lotus Pattern",
    "peony": "Peony Pattern",
    "scroll": "Repeating-Scroll Pattern",
}

OUTPUT_DETAIL_CSV = ROOT / "lpips_details.csv"
OUTPUT_BATCH_CSV = ROOT / "lpips_batch.csv"
OUTPUT_SUMMARY_CSV = ROOT / "lpips_summary.csv"
OUTPUT_STATS_TXT = ROOT / "lpips_stats.txt"

IMG_SIZE = 224


# ==================== 工具函数 ====================
def list_pngs(folder: Path):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    )


def load_tensor(path: Path, device):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    return t.to(device)


# ==================== 主程序 ====================
def main():
    print("=" * 70)
    print("LPIPS 计算（Base SCMix / LoRA-adapted SCMix）")
    print("协议：按类配对，每张生成图 vs 类内全部 5 张参考取 mean")
    print("=" * 70)

    if not REF_ROOT.exists():
        raise FileNotFoundError(f"参考集目录不存在: {REF_ROOT}")
    for name, d in MODELS.items():
        if not d.exists():
            raise FileNotFoundError(f"模型目录不存在: {name} -> {d}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("加载 LPIPS 模型 (alex) ...")
    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    print("\n预加载 held-out 参考图 ...")
    ref_tensors_by_cat = {}
    ref_files_by_cat = {}
    for cat in CATEGORIES:
        ref_dir = REF_ROOT / HELDOUT_MAP[cat]
        ref_files = list_pngs(ref_dir)
        if len(ref_files) == 0:
            print(f"  [警告] {cat}: 参考目录无 PNG -> {ref_dir}")
            continue
        ref_tensors = [load_tensor(f, device) for f in ref_files]
        ref_tensors_by_cat[cat] = ref_tensors
        ref_files_by_cat[cat] = ref_files
        print(f"  {cat:8} <- {HELDOUT_MAP[cat]:24} ({len(ref_files)} 张)")

    print("\n" + "=" * 70)
    print("逐图像 LPIPS ...")
    print("=" * 70)

    detail_rows = []

    for model_name, model_dir in MODELS.items():
        for category in CATEGORIES:
            if category not in ref_tensors_by_cat:
                print(f"[{model_name}] {category}: 无参考图，跳过")
                continue

            ref_tensors = ref_tensors_by_cat[category]
            ref_names = [f.name for f in ref_files_by_cat[category]]

            for batch in BATCHES:
                gen_dir = model_dir / category / batch
                gen_files = list_pngs(gen_dir)
                if len(gen_files) == 0:
                    print(f"[{model_name}] {category}/{batch}: 无 PNG，跳过")
                    continue
                if len(gen_files) != 30:
                    print(f"[{model_name}] {category}/{batch}: 警告 {len(gen_files)} 张（预期 30）")

                desc = f"{model_name} | {category}/{batch}"
                for gen_path in tqdm(gen_files, desc=desc, unit="张", leave=False):
                    try:
                        gen_t = load_tensor(gen_path, device)
                        dists = []
                        with torch.no_grad():
                            for ref_t in ref_tensors:
                                d = lpips_model(gen_t, ref_t).item()
                                dists.append(d)

                        arr = np.array(dists, dtype=np.float64)
                        detail_rows.append({
                            "model": model_name,
                            "motif": category,
                            "batch": batch,
                            "filename": gen_path.name,
                            "n_refs": len(dists),
                            "lpips_mean": float(arr.mean()),
                            "lpips_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                            "lpips_min": float(arr.min()),
                            "lpips_max": float(arr.max()),
                            "best_ref": ref_names[int(arr.argmin())],
                        })
                    except Exception as e:
                        print(f"\n[错误] {gen_path}: {e}")
                        continue

    if len(detail_rows) == 0:
        print("没有计算到任何图像，退出")
        return

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(OUTPUT_DETAIL_CSV, index=False, encoding="utf-8-sig")
    print(f"\n逐图像明细已保存: {OUTPUT_DETAIL_CSV}（{len(detail_df)} 行）")

    # ==================== batch 级 ====================
    batch_rows = []
    for model_name in MODELS:
        for category in CATEGORIES:
            for batch in BATCHES:
                cell = detail_df[
                    (detail_df["model"] == model_name)
                    & (detail_df["motif"] == category)
                    & (detail_df["batch"] == batch)
                ]
                if len(cell) == 0:
                    continue
                vals = cell["lpips_mean"].to_numpy(dtype=np.float64)
                batch_rows.append({
                    "model": model_name,
                    "motif": category,
                    "batch": batch,
                    "n_images": int(len(vals)),
                    "n_refs": int(cell["n_refs"].iloc[0]),
                    "lpips_mean": float(vals.mean()),
                    "lpips_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "lpips_min": float(vals.min()),
                    "lpips_max": float(vals.max()),
                    "lpips_median": float(np.median(vals)),
                })

    batch_df = pd.DataFrame(batch_rows)
    batch_df.to_csv(OUTPUT_BATCH_CSV, index=False, encoding="utf-8-sig")
    print(f"batch 级已保存: {OUTPUT_BATCH_CSV}（{len(batch_df)} 行）")

    # ==================== 汇总 ====================
    summary_rows = []

    for model_name in MODELS:
        mb = batch_df[batch_df["model"] == model_name]
        if len(mb) == 0:
            continue

        for category in CATEGORIES:
            cb = mb[mb["motif"] == category]
            if len(cb) == 0:
                continue
            means = cb["lpips_mean"].to_numpy(dtype=np.float64)
            summary_rows.append({
                "model": model_name,
                "level": "class",
                "motif": category,
                "batch": "ALL",
                "n_batches": int(len(means)),
                "n_images": int(cb["n_images"].sum()),
                "n_refs": int(cb["n_refs"].iloc[0]),
                "lpips_mean": float(means.mean()),
                "lpips_std": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
            })

        means = mb["lpips_mean"].to_numpy(dtype=np.float64)
        summary_rows.append({
            "model": model_name,
            "level": "overall",
            "motif": "OVERALL",
            "batch": "ALL",
            "n_batches": int(len(means)),
            "n_images": int(mb["n_images"].sum()),
            "n_refs": int(mb["n_refs"].iloc[0]),
            "lpips_mean": float(means.mean()),
            "lpips_std": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"汇总 CSV 已保存: {OUTPUT_SUMMARY_CSV}（{len(summary_df)} 行）")

    # ==================== 控制台报告 ====================
    print("\n" + "=" * 70)
    print("overall = 21 个 batch mean 的 mean±std")
    print("=" * 70)
    for model_name in MODELS:
        row = summary_df[
            (summary_df["model"] == model_name) & (summary_df["level"] == "overall")
        ]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        print(f"\n  [{model_name}]")
        print(f"    图像数: {r['n_images']}，参考数/类: {r['n_refs']}")
        print(f"    LPIPS = {r['lpips_mean']:.4f} ± {r['lpips_std']:.4f}")

    print("\n" + "=" * 70)
    print("per-class = 每类 3 个 batch mean 的 mean±std")
    print("=" * 70)
    for model_name in MODELS:
        print(f"\n  [{model_name}]")
        for category in CATEGORIES:
            row = summary_df[
                (summary_df["model"] == model_name)
                & (summary_df["level"] == "class")
                & (summary_df["motif"] == category)
            ]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            print(
                f"    {category:8}: n={r['n_images']:3d}, refs={r['n_refs']}, "
                f"LPIPS={r['lpips_mean']:.4f} ± {r['lpips_std']:.4f}"
            )

    # ==================== 文本报告 ====================
    with open(OUTPUT_STATS_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("LPIPS 计算报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("【口径说明】\n")
        f.write("  - 按类配对：cloud -> Cloud Pattern, floral -> Floral Pattern,\n")
        f.write("    grass -> Grass Pattern, hui -> Hui Pattern, lotus -> Lotus Pattern,\n")
        f.write("    peony -> Peony Pattern, scroll -> Repeating-Scroll Pattern。\n")
        f.write("  - 每张生成图 vs 该类全部 5 张参考图，取 mean（不取 min）。\n")
        f.write("  - batch 级：每 batch 30 张图的 per-image mean 再取 mean。\n")
        f.write("  - overall = 21 个 batch mean 的 mean ± std。\n")
        f.write("  - per-class = 每类 3 个 batch mean 的 mean ± std。\n")
        f.write("  - 与旧脚本（min over 100 random refs，报告 0.5317）不可比，\n")
        f.write("    正文须说明协议已变更。\n\n")

        f.write("【配置信息】\n")
        f.write(f"  根目录: {ROOT}\n")
        f.write(f"  参考集: {REF_ROOT}\n")
        f.write(f"  LPIPS: alex, resize {IMG_SIZE}x{IMG_SIZE}, 归一化 x/127.5-1\n")
        for name, d in MODELS.items():
            f.write(f"  {name}: {d}\n")
        f.write(f"  总图像数: {len(detail_df)}\n\n")

        f.write("【整体统计】\n")
        for model_name in MODELS:
            row = summary_df[
                (summary_df["model"] == model_name) & (summary_df["level"] == "overall")
            ]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            f.write(f"\n  [{model_name}]\n")
            f.write(f"    图像数: {r['n_images']}, batch 数: {r['n_batches']}, 每类参考数: {r['n_refs']}\n")
            f.write(f"    LPIPS: {r['lpips_mean']:.4f} ± {r['lpips_std']:.4f}\n")

        f.write("\n【各纹样统计】\n")
        for model_name in MODELS:
            f.write(f"\n  [{model_name}]\n")
            for category in CATEGORIES:
                row = summary_df[
                    (summary_df["model"] == model_name)
                    & (summary_df["level"] == "class")
                    & (summary_df["motif"] == category)
                ]
                if len(row) == 0:
                    continue
                r = row.iloc[0]
                f.write(
                    f"    {category}: n={r['n_images']}, refs={r['n_refs']}, "
                    f"LPIPS={r['lpips_mean']:.4f} ± {r['lpips_std']:.4f}\n"
                )

        f.write("\n" + "=" * 70 + "\n")
        f.write("报告结束\n")
        f.write("=" * 70 + "\n")

    print(f"\n文本报告已保存: {OUTPUT_STATS_TXT}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()