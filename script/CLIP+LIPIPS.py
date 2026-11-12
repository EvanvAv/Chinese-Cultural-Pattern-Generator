# -*- coding: utf-8 -*-
"""
CLIP filtering + LPIPS intra-class diversity, multi-threshold.

Outputs (S:\ComfyUI\CLIP+LIPIPS\results\):
  clip_scores_details.csv
  diversity_class_{thr}.csv
  diversity_summary_{thr}.csv
  summary_all_thresholds.csv
  final_report.txt
"""

import os
from pathlib import Path
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import clip
import lpips
from PIL import Image
from tqdm import tqdm


# ==================== 配置 ====================
ROOT = Path(r"S:\ComfyUI\CLIP+LIPIPS")
PROMPTS_CSV = ROOT / "prompts.csv"
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "Base SCMix": ROOT / "pattern_base",
    "LoRA-adapted SCMix": ROOT / "pattern",
}

CATEGORIES = ["cloud", "floral", "grass", "hui", "lotus", "peony", "scroll"]
BATCHES = ["batch_01", "batch_02", "batch_03"]

THRESHOLDS = [0.25, 0.255, 0.26, 0.265, 0.27]

DIVERSITY_MODEL = "LoRA-adapted SCMix"

CLIP_MAX_TOKENS = 77
LPIPS_SIZE = 224

OUTPUT_CLIP_DETAILS = OUTPUT_DIR / "clip_scores_details.csv"
OUTPUT_ALL_SUMMARY = OUTPUT_DIR / "summary_all_thresholds.csv"
OUTPUT_REPORT = OUTPUT_DIR / "final_report.txt"


# ==================== 工具函数 ====================
def list_pngs(folder: Path):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    )


def load_clip_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print("加载 CLIP 模型 (ViT-B/32) ...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, device


def load_lpips_model(device):
    print("加载 LPIPS 模型 (alex) ...")
    model = lpips.LPIPS(net="alex").to(device)
    model.eval()
    return model


def encode_prompts(clip_model, motif_to_prompt, device):
    cache = {}
    print("\n预编码 prompts ...")
    for motif, prompt in motif_to_prompt.items():
        text = clip.tokenize([prompt], truncate=True).to(device)
        with torch.no_grad():
            tf = clip_model.encode_text(text)
            tf = tf / tf.norm(dim=-1, keepdim=True)
        cache[motif] = tf
        print(f"  {motif:8}: {len(prompt)} chars")
    return cache


def load_lpips_tensor(path: Path, device):
    img = Image.open(path).convert("RGB").resize((LPIPS_SIZE, LPIPS_SIZE))
    arr = np.array(img).astype(np.float32)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    return t.to(device)


# ==================== CLIP 逐图像 ====================
def compute_clip_details(text_feat_cache, clip_model, preprocess, device):
    print("\n" + "=" * 60)
    print("计算 CLIP Score（逐图像，一次）")
    print("=" * 60)

    rows = []

    for model_name, model_dir in MODELS.items():
        for category in CATEGORIES:
            if category not in text_feat_cache:
                print(f"[警告] prompts.csv 缺少类别 {category}，跳过")
                continue
            text_feat = text_feat_cache[category]

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
                        image = preprocess(
                            Image.open(gen_path).convert("RGB")
                        ).unsqueeze(0).to(device)
                        with torch.no_grad():
                            img_feat = clip_model.encode_image(image)
                            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                            score = (img_feat @ text_feat.T).item()
                        rows.append({
                            "model": model_name,
                            "motif": category,
                            "batch": batch,
                            "filename": gen_path.name,
                            "image_path": str(gen_path),
                            "clip_score": float(score),
                        })
                    except Exception as e:
                        print(f"\n[错误] {gen_path}: {e}")
                        continue

    return pd.DataFrame(rows)


# ==================== 预计算类内配对 LPIPS ====================
def precompute_pair_lpips(clip_df, lpips_model, device):
    print("\n" + "=" * 60)
    print(f"预计算类内配对 LPIPS（仅 {DIVERSITY_MODEL}，一次）")
    print("=" * 60)

    pair_cache = {}

    for category in CATEGORIES:
        cat_df = clip_df[
            (clip_df["model"] == DIVERSITY_MODEL) & (clip_df["motif"] == category)
        ]
        paths = sorted(cat_df["image_path"].tolist())
        n = len(paths)
        if n < 2:
            print(f"  {category:8}: 仅 {n} 张，跳过配对")
            pair_cache[category] = {}
            continue

        print(f"  {category:8}: {n} 张，预计算 {n*(n-1)//2} 对 ...")
        tensors = []
        valid_paths = []
        for p in paths:
            try:
                tensors.append(load_lpips_tensor(Path(p), device))
                valid_paths.append(p)
            except Exception as e:
                print(f"    [警告] 加载失败 {p}: {e}")

        cat_pairs = {}
        m = len(valid_paths)
        if m >= 2:
            with torch.no_grad():
                for i, j in tqdm(
                    combinations(range(m), 2),
                    desc=f"    {category}",
                    total=m * (m - 1) // 2,
                    leave=False,
                ):
                    try:
                        d = lpips_model(tensors[i], tensors[j]).item()
                        cat_pairs[(valid_paths[i], valid_paths[j])] = float(d)
                    except Exception as e:
                        print(f"    [警告] 配对失败 {i},{j}: {e}")
        pair_cache[category] = cat_pairs

        del tensors
        torch.cuda.empty_cache()

    return pair_cache


# ==================== 某阈值下 diversity ====================
def diversity_at_threshold(clip_df, pair_cache, threshold):
    class_rows = []
    image_scores = []

    for category in CATEGORIES:
        cat_df = clip_df[
            (clip_df["model"] == DIVERSITY_MODEL) & (clip_df["motif"] == category)
        ]
        passed_paths = sorted(
            cat_df[cat_df["clip_score"] >= threshold]["image_path"].tolist()
        )
        n = len(passed_paths)

        cat_pairs = pair_cache.get(category, {})

        if n < 2:
            class_rows.append({
                "model": DIVERSITY_MODEL,
                "threshold": threshold,
                "motif": category,
                "n_images": n,
                "n_pairs": 0,
                "avg_lpips": np.nan,
                "std_lpips": np.nan,
                "min_lpips": np.nan,
                "max_lpips": np.nan,
            })
            for p in passed_paths:
                image_scores.append({
                    "model": DIVERSITY_MODEL,
                    "threshold": threshold,
                    "motif": category,
                    "image_path": p,
                    "filename": os.path.basename(p),
                    "avg_lpips": np.nan,
                    "num_pairs": 0,
                })
            continue

        passed_set = set(passed_paths)
        vals = []
        sum_dist = {p: 0.0 for p in passed_paths}
        cnt = {p: 0 for p in passed_paths}

        for (p1, p2), d in cat_pairs.items():
            if p1 in passed_set and p2 in passed_set:
                vals.append(d)
                sum_dist[p1] += d
                sum_dist[p2] += d
                cnt[p1] += 1
                cnt[p2] += 1

        if vals:
            arr = np.array(vals, dtype=np.float64)
            class_rows.append({
                "model": DIVERSITY_MODEL,
                "threshold": threshold,
                "motif": category,
                "n_images": n,
                "n_pairs": len(arr),
                "avg_lpips": float(arr.mean()),
                "std_lpips": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "min_lpips": float(arr.min()),
                "max_lpips": float(arr.max()),
            })
            for p in passed_paths:
                image_scores.append({
                    "model": DIVERSITY_MODEL,
                    "threshold": threshold,
                    "motif": category,
                    "image_path": p,
                    "filename": os.path.basename(p),
                    "avg_lpips": sum_dist[p] / cnt[p] if cnt[p] > 0 else np.nan,
                    "num_pairs": cnt[p],
                })
        else:
            class_rows.append({
                "model": DIVERSITY_MODEL,
                "threshold": threshold,
                "motif": category,
                "n_images": n,
                "n_pairs": 0,
                "avg_lpips": np.nan,
                "std_lpips": np.nan,
                "min_lpips": np.nan,
                "max_lpips": np.nan,
            })

    return class_rows, image_scores


# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("CLIP 过滤 + LPIPS 类内多样性（多阈值）")
    print(f"阈值: {THRESHOLDS}")
    print(f"Diversity 模型: {DIVERSITY_MODEL}；Base diversity: N/A")
    print("=" * 60)

    if not PROMPTS_CSV.exists():
        raise FileNotFoundError(f"prompts.csv 不存在: {PROMPTS_CSV}")
    for name, d in MODELS.items():
        if not d.exists():
            raise FileNotFoundError(f"模型目录不存在: {name} -> {d}")

    prompts_df = pd.read_csv(PROMPTS_CSV)
    if "motif" not in prompts_df.columns or "prompt" not in prompts_df.columns:
        raise ValueError("prompts.csv 必须包含 'motif' 和 'prompt' 两列")
    motif_to_prompt = dict(zip(prompts_df["motif"], prompts_df["prompt"]))

    print(f"\n加载了 {len(motif_to_prompt)} 个纹样提示词:")
    for m in motif_to_prompt:
        print(f"  - {m}")

    clip_model, preprocess, device = load_clip_models()
    lpips_model = load_lpips_model(device)
    text_feat_cache = encode_prompts(clip_model, motif_to_prompt, device)

    clip_df = compute_clip_details(text_feat_cache, clip_model, preprocess, device)
    if len(clip_df) == 0:
        print("没有计算到任何图像，退出")
        return
    clip_df.to_csv(OUTPUT_CLIP_DETAILS, index=False, encoding="utf-8-sig")
    print(f"\nCLIP 逐图像明细已保存: {OUTPUT_CLIP_DETAILS}（{len(clip_df)} 行）")

    pair_cache = precompute_pair_lpips(clip_df, lpips_model, device)

    all_summary_rows = []
    all_class_rows_by_thr = {}
    all_summary_rows_by_thr = {}

    for thr in THRESHOLDS:
        thr_str = f"{thr:.3f}"
        print("\n" + "=" * 60)
        print(f"阈值 {thr_str}")
        print("=" * 60)

        for model_name in MODELS:
            m = clip_df[clip_df["model"] == model_name]
            if len(m) == 0:
                continue
            passed = m[m["clip_score"] >= thr]
            pass_rate = len(passed) / len(m) * 100.0
            print(f"  [{model_name}] pass {len(passed)}/{len(m)} ({pass_rate:.1f}%)")

            if model_name == DIVERSITY_MODEL:
                class_rows, _ = diversity_at_threshold(clip_df, pair_cache, thr)
                class_df = pd.DataFrame(class_rows)
                valid = class_df.dropna(subset=["avg_lpips"])
                if len(valid) == 0:
                    div_mean, div_std, n_valid = np.nan, np.nan, 0
                else:
                    vals = valid["avg_lpips"].to_numpy(dtype=np.float64)
                    div_mean = float(vals.mean())
                    div_std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                    n_valid = int(len(vals))

                summary_df = pd.DataFrame([{
                    "model": model_name,
                    "threshold": thr,
                    "n_valid_classes": n_valid,
                    "diversity_mean": div_mean,
                    "diversity_std": div_std,
                }])

                class_path = OUTPUT_DIR / f"diversity_class_{thr_str}.csv"
                summary_path = OUTPUT_DIR / f"diversity_summary_{thr_str}.csv"
                class_df.to_csv(class_path, index=False, encoding="utf-8-sig")
                summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
                print(f"    已保存: {class_path.name}, {summary_path.name}")
                print(f"    Diversity overall = {div_mean:.4f} ± {div_std:.4f} (有效类 {n_valid}/7)")

                all_class_rows_by_thr[thr_str] = class_df
                all_summary_rows_by_thr[thr_str] = summary_df

                all_summary_rows.append({
                    "threshold": thr,
                    "model": model_name,
                    "total_images": int(len(m)),
                    "passed_images": int(len(passed)),
                    "pass_rate": pass_rate,
                    "n_valid_classes": n_valid,
                    "diversity_mean": div_mean,
                    "diversity_std": div_std,
                })
            else:
                all_summary_rows.append({
                    "threshold": thr,
                    "model": model_name,
                    "total_images": int(len(m)),
                    "passed_images": int(len(passed)),
                    "pass_rate": pass_rate,
                    "n_valid_classes": np.nan,
                    "diversity_mean": np.nan,
                    "diversity_std": np.nan,
                })

    summary_all = pd.DataFrame(all_summary_rows)
    summary_all.to_csv(OUTPUT_ALL_SUMMARY, index=False, encoding="utf-8-sig")
    print(f"\n合并汇总已保存: {OUTPUT_ALL_SUMMARY}（{len(summary_all)} 行）")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("CLIP 过滤 + LPIPS 类内多样性（多阈值）\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"阈值: {THRESHOLDS}\n")
        f.write("=" * 60 + "\n\n")

        f.write("【口径说明】\n")
        f.write("  - CLIP：两模型都计算，逐图像输出一次；本脚本不做 batch/class 汇总。\n")
        f.write("  - Diversity：仅 LoRA-adapted SCMix；Base 报 N/A。\n")
        f.write("  - Diversity 层级：类内全量配对（该类所有过滤后图像两两组合）。\n")
        f.write("  - overall = 7 个类均值的 mean±std（只对有效类）。\n")
        f.write("  - 表注：diversity 是类内全量配对，不是 batch-level。\n")
        f.write("  - held-out 不参与 diversity。\n\n")

        f.write("【配置信息】\n")
        f.write(f"  根目录: {ROOT}\n")
        f.write(f"  提示词文件: {PROMPTS_CSV}\n")
        f.write(f"  CLIP: ViT-B/32\n")
        f.write(f"  LPIPS: alex, resize {LPIPS_SIZE}x{LPIPS_SIZE}\n")
        for name, d in MODELS.items():
            f.write(f"  {name}: {d}\n")
        f.write(f"  Diversity 模型: {DIVERSITY_MODEL}\n")
        f.write(f"  总图像数: {len(clip_df)}\n\n")

        for thr in THRESHOLDS:
            thr_str = f"{thr:.3f}"
            f.write("=" * 60 + "\n")
            f.write(f"阈值 {thr_str}\n")
            f.write("=" * 60 + "\n")

            f.write("【CLIP pass rate】\n")
            for model_name in MODELS:
                m = clip_df[clip_df["model"] == model_name]
                if len(m) == 0:
                    continue
                passed = m[m["clip_score"] >= thr]
                f.write(
                    f"  [{model_name}] {len(passed)}/{len(m)} "
                    f"({len(passed)/len(m)*100:.1f}%)\n"
                )

            f.write("\n【Diversity】\n")
            for model_name in MODELS:
                if model_name != DIVERSITY_MODEL:
                    f.write(f"  [{model_name}] N/A（不报告）\n")
                    continue

            class_df = all_class_rows_by_thr.get(thr_str)
            summary_df = all_summary_rows_by_thr.get(thr_str)
            if class_df is not None and len(class_df) > 0:
                row = summary_df.iloc[0]
                if np.isnan(row["diversity_mean"]):
                    f.write(f"  [{DIVERSITY_MODEL}] overall: N/A（无有效类）\n")
                else:
                    f.write(
                        f"  [{DIVERSITY_MODEL}] overall: "
                        f"{row['diversity_mean']:.4f} ± {row['diversity_std']:.4f} "
                        f"(有效类 {row['n_valid_classes']}/7)\n"
                    )
                for _, c in class_df.iterrows():
                    if np.isnan(c["avg_lpips"]):
                        f.write(f"    {c['motif']}: N/A (n={c['n_images']})\n")
                    else:
                        f.write(
                            f"    {c['motif']}: {c['avg_lpips']:.4f} ± {c['std_lpips']:.4f} "
                            f"(n={c['n_images']}, pairs={c['n_pairs']})\n"
                        )
            f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("报告结束\n")
        f.write("=" * 60 + "\n")

    print(f"\n文本报告已保存: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()