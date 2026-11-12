# -*- coding: utf-8 -*-
"""
CLIP Score evaluation for Base SCMix and LoRA-adapted SCMix.

Directory layout: {model_dir}/{category}/{batch_01|02|03}/*.png
prompts.csv columns: motif, prompt
"""

import os
import torch
import clip
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from pathlib import Path


# ==================== 配置 ====================
ROOT = Path(r"S:\ComfyUI\CLIP")
PROMPTS_CSV = ROOT / "prompts.csv"

MODELS = {
    "Base SCMix": ROOT / "pattern_base",
    "LoRA-adapted SCMix": ROOT / "pattern",
}

CATEGORIES = ["cloud", "floral", "grass", "hui", "lotus", "peony", "scroll"]
BATCHES = ["batch_01", "batch_02", "batch_03"]

OUTPUT_DETAIL_CSV = ROOT / "clip_scores_detailed.csv"
OUTPUT_FILTERED_CSV = ROOT / "clip_scores_filtered_026.csv"
OUTPUT_SUMMARY_CSV = ROOT / "clip_scores_summary.csv"
OUTPUT_STATS_TXT = ROOT / "clip_scores_stats_026.txt"

THRESHOLD = 0.26
CLIP_MAX_TOKENS = 77


# ==================== 工具函数 ====================
def list_pngs(folder: Path):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    )


def encode_prompts(model, motif_to_prompt, device):
    cache = {}
    print("预编码 prompts ...")
    for motif, prompt in motif_to_prompt.items():
        text = clip.tokenize([prompt], truncate=True).to(device)
        with torch.no_grad():
            tf = model.encode_text(text)
            tf = tf / tf.norm(dim=-1, keepdim=True)
        cache[motif] = tf
        print(f"  {motif:8}: {len(prompt)} chars")
    return cache


# ==================== 主程序 ====================
def main():
    print("=" * 70)
    print("CLIP Score 计算（Base SCMix / LoRA-adapted SCMix）")
    print(f"阈值: {THRESHOLD}")
    print("=" * 70)

    if not PROMPTS_CSV.exists():
        raise FileNotFoundError(f"prompts.csv 不存在: {PROMPTS_CSV}")
    for name, d in MODELS.items():
        if not d.exists():
            raise FileNotFoundError(f"模型目录不存在: {name} -> {d}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    print("加载 CLIP 模型 (ViT-B/32) ...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    prompts_df = pd.read_csv(PROMPTS_CSV)
    if "motif" not in prompts_df.columns or "prompt" not in prompts_df.columns:
        raise ValueError("prompts.csv 必须包含 'motif' 和 'prompt' 两列")
    motif_to_prompt = dict(zip(prompts_df["motif"], prompts_df["prompt"]))

    print(f"\n加载了 {len(motif_to_prompt)} 个纹样提示词:")
    for motif in motif_to_prompt:
        print(f"  - {motif}")

    text_feat_cache = encode_prompts(model, motif_to_prompt, device)

    print("\n" + "=" * 70)
    print("扫描并计算 CLIP Score ...")
    print("=" * 70)

    results = []

    for model_name, model_dir in MODELS.items():
        for category in CATEGORIES:
            if category not in text_feat_cache:
                print(f"[警告] prompts.csv 缺少类别 {category}，跳过")
                continue

            text_feat = text_feat_cache[category]
            prompt = motif_to_prompt[category]

            for batch in BATCHES:
                batch_dir = model_dir / category / batch
                image_files = list_pngs(batch_dir)
                if len(image_files) == 0:
                    print(f"[{model_name}] {category}/{batch}: 无 PNG，跳过")
                    continue
                if len(image_files) != 30:
                    print(f"[{model_name}] {category}/{batch}: 警告 {len(image_files)} 张（预期 30）")

                desc = f"{model_name} | {category}/{batch}"
                for img_path in tqdm(image_files, desc=desc, unit="张", leave=False):
                    try:
                        image = preprocess(
                            Image.open(img_path).convert("RGB")
                        ).unsqueeze(0).to(device)

                        with torch.no_grad():
                            img_feat = model.encode_image(image)
                            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                            similarity = (img_feat @ text_feat.T).item()

                        results.append({
                            "model": model_name,
                            "motif": category,
                            "batch": batch,
                            "filename": img_path.name,
                            "image_path": str(img_path),
                            "clip_score": float(similarity),
                            "passed_threshold": bool(similarity >= THRESHOLD),
                            "prompt_length": len(prompt),
                            "prompt_preview": (
                                prompt[:100] + "..." if len(prompt) > 100 else prompt
                            ),
                        })
                    except Exception as e:
                        print(f"\n[错误] {img_path}: {e}")
                        continue

    if len(results) == 0:
        print("没有计算到任何图像，退出")
        return

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DETAIL_CSV, index=False, encoding="utf-8-sig")
    print(f"\n逐图像明细已保存: {OUTPUT_DETAIL_CSV}（{len(df)} 行）")

    filtered_df = df[df["passed_threshold"]].copy()
    filtered_df.to_csv(OUTPUT_FILTERED_CSV, index=False, encoding="utf-8-sig")
    print(f"过滤后明细已保存: {OUTPUT_FILTERED_CSV}（{len(filtered_df)} 行）")

    # ==================== 汇总 ====================
    print("\n" + "=" * 70)
    print("汇总统计（batch-level）")
    print("=" * 70)

    summary_rows = []

    for model_name in MODELS:
        model_df = df[df["model"] == model_name]
        model_f = filtered_df[filtered_df["model"] == model_name]
        if len(model_df) == 0:
            continue

        batch_rows = []
        for category in CATEGORIES:
            for batch in BATCHES:
                cell = model_df[
                    (model_df["motif"] == category) & (model_df["batch"] == batch)
                ]
                cell_f = model_f[
                    (model_f["motif"] == category) & (model_f["batch"] == batch)
                ]
                if len(cell) == 0:
                    continue
                row = {
                    "model": model_name,
                    "level": "batch",
                    "motif": category,
                    "batch": batch,
                    "n_images": int(len(cell)),
                    "passed_images": int(len(cell_f)),
                    "pass_rate": float(len(cell_f) / len(cell)),
                    "mean_before": float(cell["clip_score"].mean()),
                    "std_before": float(cell["clip_score"].std(ddof=1)) if len(cell) > 1 else 0.0,
                    "mean_after": float(cell_f["clip_score"].mean()) if len(cell_f) > 0 else np.nan,
                    "std_after": float(cell_f["clip_score"].std(ddof=1)) if len(cell_f) > 1 else (0.0 if len(cell_f) == 1 else np.nan),
                }
                batch_rows.append(row)
                summary_rows.append(row)

        batch_df = pd.DataFrame(batch_rows)

        for category in CATEGORIES:
            cb = batch_df[batch_df["motif"] == category]
            if len(cb) == 0:
                continue
            n = int(cb["n_images"].sum())
            p = int(cb["passed_images"].sum())
            row = {
                "model": model_name,
                "level": "class",
                "motif": category,
                "batch": "ALL",
                "n_images": n,
                "passed_images": p,
                "pass_rate": float(p / n) if n > 0 else np.nan,
                "mean_before": float(cb["mean_before"].mean()),
                "std_before": float(cb["mean_before"].std(ddof=1)) if len(cb) > 1 else 0.0,
                "mean_after": float(cb["mean_after"].mean()) if cb["mean_after"].notna().any() else np.nan,
                "std_after": float(cb["mean_after"].std(ddof=1)) if cb["mean_after"].notna().sum() > 1 else np.nan,
            }
            summary_rows.append(row)

        if len(batch_df) > 0:
            n = int(batch_df["n_images"].sum())
            p = int(batch_df["passed_images"].sum())
            row = {
                "model": model_name,
                "level": "overall",
                "motif": "OVERALL",
                "batch": "ALL",
                "n_images": n,
                "passed_images": p,
                "pass_rate": float(p / n) if n > 0 else np.nan,
                "mean_before": float(batch_df["mean_before"].mean()),
                "std_before": float(batch_df["mean_before"].std(ddof=1)) if len(batch_df) > 1 else 0.0,
                "mean_after": float(batch_df["mean_after"].mean()) if batch_df["mean_after"].notna().any() else np.nan,
                "std_after": float(batch_df["mean_after"].std(ddof=1)) if batch_df["mean_after"].notna().sum() > 1 else np.nan,
            }
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"\n汇总 CSV 已保存: {OUTPUT_SUMMARY_CSV}（{len(summary_df)} 行）")

    # ==================== 控制台报告 ====================
    print("\n" + "=" * 70)
    print("整体汇总（overall = 21 个 batch mean 的 mean±std）")
    print("=" * 70)
    for model_name in MODELS:
        row = summary_df[
            (summary_df["model"] == model_name) & (summary_df["level"] == "overall")
        ]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        print(f"\n  [{model_name}]")
        print(f"    图像总数: {r['n_images']}，通过阈值(≥{THRESHOLD}): {r['passed_images']} ({r['pass_rate']*100:.1f}%)")
        print(f"    过滤前 CLIP: {r['mean_before']:.4f} ± {r['std_before']:.4f}")
        if not np.isnan(r["mean_after"]):
            print(f"    过滤后 CLIP: {r['mean_after']:.4f} ± {r['std_after']:.4f}")
        else:
            print(f"    过滤后 CLIP: 无数据")

    print("\n" + "=" * 70)
    print("各纹样汇总（per-class = 每类 3 个 batch mean 的 mean±std）")
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
            after = (
                f"{r['mean_after']:.4f} ± {r['std_after']:.4f}"
                if not np.isnan(r["mean_after"]) else "无数据"
            )
            print(
                f"    {category:8}: n={r['n_images']:3d}, "
                f"pass={r['passed_images']:3d} ({r['pass_rate']*100:5.1f}%), "
                f"before={r['mean_before']:.4f} ± {r['std_before']:.4f}, "
                f"after={after}"
            )

    # ==================== 文本报告 ====================
    with open(OUTPUT_STATS_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("CLIP Score 计算报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"阈值: {THRESHOLD}\n")
        f.write("=" * 70 + "\n\n")

        f.write("【口径说明】\n")
        f.write("  - 每个 batch (30 张) 取 mean，7 类 × 3 batch = 21 个 batch mean。\n")
        f.write("  - overall : 21 个 batch mean 的 mean ± std。\n")
        f.write("  - per-class: 每类 3 个 batch mean 的 mean ± std。\n")
        f.write("  - 主表报过滤前 mean 与 pass rate；过滤后同 batch 级口径。\n")
        f.write("  - held-out 未参与 CLIP score。\n\n")

        f.write("【配置信息】\n")
        f.write(f"  根目录: {ROOT}\n")
        f.write(f"  提示词文件: {PROMPTS_CSV}\n")
        f.write(f"  模型目录:\n")
        for name, d in MODELS.items():
            f.write(f"    {name}: {d}\n")
        f.write(f"  CLIP 模型: ViT-B/32\n")
        f.write(f"  纹样数量: {len(motif_to_prompt)}\n")
        f.write(f"  总图像数: {len(df)}\n\n")

        f.write("【整体统计】\n")
        for model_name in MODELS:
            row = summary_df[
                (summary_df["model"] == model_name) & (summary_df["level"] == "overall")
            ]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            f.write(f"\n  [{model_name}]\n")
            f.write(f"    图像总数: {r['n_images']}\n")
            f.write(f"    通过阈值(≥{THRESHOLD}): {r['passed_images']} ({r['pass_rate']*100:.1f}%)\n")
            f.write(f"    过滤前: {r['mean_before']:.4f} ± {r['std_before']:.4f}\n")
            if not np.isnan(r["mean_after"]):
                f.write(f"    过滤后: {r['mean_after']:.4f} ± {r['std_after']:.4f}\n")
            else:
                f.write(f"    过滤后: 无数据\n")

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
                f.write(f"\n    {category}:\n")
                f.write(f"      图像数: {r['n_images']}\n")
                f.write(f"      通过阈值: {r['passed_images']}/{r['n_images']} ({r['pass_rate']*100:.1f}%)\n")
                f.write(f"      过滤前: {r['mean_before']:.4f} ± {r['std_before']:.4f}\n")
                if not np.isnan(r["mean_after"]):
                    f.write(f"      过滤后: {r['mean_after']:.4f} ± {r['std_after']:.4f}\n")
                else:
                    f.write(f"      过滤后: 无数据\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("报告结束\n")
        f.write("=" * 70 + "\n")

    print(f"\n文本报告已保存: {OUTPUT_STATS_TXT}")


if __name__ == "__main__":
    main()