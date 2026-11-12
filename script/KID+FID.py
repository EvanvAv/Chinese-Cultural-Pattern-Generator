# -*- coding: utf-8 -*-
"""
FID + KID evaluation for Base SCMix and LoRA-adapted SCMix.

Per batch: 7 categories x 30 images = 210 generated images vs full held-out (n=35).
Metrics computed per batch (n_batches=3); mean±std across batches.
95% CI via image-level bootstrap on both gen and ref sets.
KID is primary; FID is complementary due to small reference set.

Outputs (S:\ComfyUI\KID+FID\):
  fid_kid_details.csv   3 rows per model (batch_01/02/03)
  fid_kid_summary.csv   1 row per model (mean±std + CI)
"""

import os
import csv
import shutil
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cleanfid.fid import build_feature_extractor


# ==================== 配置 ====================
ROOT = Path(r"S:\ComfyUI\KID+FID")
REF_ROOT = ROOT / "held-out"

MODELS = {
    "Base SCMix": ROOT / "pattern_base",
    "LoRA-adapted SCMix": ROOT / "pattern",
}

CATEGORIES = ["cloud", "floral", "grass", "hui", "lotus", "peony", "scroll"]
BATCHES = ["batch_01", "batch_02", "batch_03"]

OUT_DIR = ROOT
CACHE_DIR = ROOT / "_cache"
CACHE_REF_DIR = CACHE_DIR / "ref_35"
CACHE_GEN_ROOT = CACHE_DIR / "gen_flat"

N_BOOTSTRAP = 1000
SEED = 42
DEVICE = "cuda"
DEVICE_T = torch.device(DEVICE)
MODE = "clean"
BATCH_SIZE = 8

DETAIL_CSV = OUT_DIR / "fid_kid_details.csv"
SUMMARY_CSV = OUT_DIR / "fid_kid_summary.csv"


# ==================== 文件列表工具 ====================
def list_pngs(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    )


def link_or_copy(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def make_flat_dir(src_files, dst_dir):
    dst_dir = Path(dst_dir)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    flat = []
    for i, f in enumerate(src_files):
        f = Path(f)
        dst = dst_dir / f"{i:04d}_{f.name}"
        link_or_copy(f, dst)
        flat.append(dst)
    return flat


def prepare_flat_reference():
    ref_files = sorted(
        [p for p in REF_ROOT.rglob("*") if p.is_file() and p.suffix.lower() == ".png"]
    )
    if not ref_files:
        raise RuntimeError(f"参考集中没有找到 PNG: {REF_ROOT}")
    flat = make_flat_dir(ref_files, CACHE_REF_DIR)
    print(f"参考集平铺完成: {len(flat)} 张 -> {CACHE_REF_DIR}")
    return flat, len(flat)


def collect_batch_files(model_dir, batch):
    files = []
    per_category = {}
    for category in CATEGORIES:
        cat_batch_dir = model_dir / category / batch
        cat_files = list_pngs(cat_batch_dir)
        per_category[category] = len(cat_files)
        files.extend(cat_files)
    return sorted(files), per_category


def prepare_flat_gen(model_name, batch, gen_files):
    if not gen_files:
        return None, 0
    key = f"{model_name}__{batch}".replace(" ", "_")
    dst = CACHE_GEN_ROOT / key
    flat = make_flat_dir(gen_files, dst)
    return flat, len(flat)


# ==================== 手动特征提取 ====================
def _imread_clean(filename):
    img = Image.open(filename).convert("RGB")
    img = img.resize((299, 299), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    arr = (arr - 127.5) / 127.5
    arr = arr.transpose(2, 0, 1)
    return arr


def extract_features_manual(files, feature_fn):
    feats = []
    for i in range(0, len(files), BATCH_SIZE):
        batch = files[i:i + BATCH_SIZE]
        arrs = [_imread_clean(str(f)) for f in batch]
        imgs = torch.from_numpy(np.stack(arrs)).to(DEVICE_T)
        with torch.no_grad():
            out = feature_fn(imgs)
            if isinstance(out, tuple):
                out = out[0]
            feats.append(out.detach().cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float64)


def get_features(files, feature_fn):
    if isinstance(files, (str, Path)):
        files = list_pngs(files)
    files = [str(p) for p in files]
    feat = extract_features_manual(files, feature_fn)
    if feat.shape[0] != len(files):
        raise RuntimeError(f"特征数 {feat.shape[0]} != 文件数 {len(files)}")
    return feat


# ==================== GPU 版 FID / KID ====================
def fid_gpu(x, y):
    n1 = x.shape[0]
    n2 = y.shape[0]

    mu1 = x.mean(dim=0)
    mu2 = y.mean(dim=0)
    xc = x - mu1
    yc = y - mu2

    _, S1, Vh1 = torch.linalg.svd(xc / (n1 - 1) ** 0.5, full_matrices=False)
    _, S2, Vh2 = torch.linalg.svd(yc / (n2 - 1) ** 0.5, full_matrices=False)

    B = (S1.unsqueeze(1) * (Vh1 @ Vh2.T)) * S2.unsqueeze(0)
    tr_covmean = torch.linalg.svdvals(B).sum()

    diff = mu1 - mu2
    fid = (
        diff.dot(diff)
        + (S1 ** 2).sum()
        + (S2 ** 2).sum()
        - 2.0 * tr_covmean
    )
    return float(fid.item())


def kid_gpu(x, y, degree=3):
    d = x.shape[1]
    m = x.shape[0]
    n = y.shape[0]

    k_xx = ((x @ x.T) / d + 1.0) ** degree
    k_yy = ((y @ y.T) / d + 1.0) ** degree
    k_xy = ((x @ y.T) / d + 1.0) ** degree

    sum_xx = (k_xx.sum() - k_xx.diagonal().sum()) / (m * (m - 1))
    sum_yy = (k_yy.sum() - k_yy.diagonal().sum()) / (n * (n - 1))
    sum_xy = k_xy.mean()

    return float((sum_xx + sum_yy - 2.0 * sum_xy).item())


def bootstrap_batch_gpu(gen_feats_gpu, ref_feats_gpu, n_boot, seed):
    rng = np.random.default_rng(seed)
    n_gen = gen_feats_gpu.shape[0]
    n_ref = ref_feats_gpu.shape[0]

    fid_boot = np.empty(n_boot, dtype=np.float64)
    kid_boot = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        gi = torch.from_numpy(rng.integers(0, n_gen, size=n_gen)).to(DEVICE_T)
        ri = torch.from_numpy(rng.integers(0, n_ref, size=n_ref)).to(DEVICE_T)
        x = gen_feats_gpu[gi]
        y = ref_feats_gpu[ri]

        fid_boot[i] = fid_gpu(x, y)
        kid_boot[i] = kid_gpu(x, y)

        if (i + 1) % 100 == 0 or (i + 1) == n_boot:
            print(f"    bootstrap {i + 1}/{n_boot}")

    return fid_boot, kid_boot


def stable_seed(*parts):
    s = "|".join(str(p) for p in parts)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return SEED + int(h[:8], 16)


def quantile_ci(values):
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


# ==================== 主程序 ====================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FID / KID evaluation（全量参考 35 张 × 3 批次）")
    print(f"参考集根目录: {REF_ROOT}")
    print(f"模型目录: {MODELS}")
    print(f"bootstrap 次数: {N_BOOTSTRAP}, seed: {SEED}")
    print(f"device: {DEVICE}")
    print("=" * 70)

    if not REF_ROOT.exists():
        raise FileNotFoundError(f"参考集目录不存在: {REF_ROOT}")
    for name, d in MODELS.items():
        if not d.exists():
            raise FileNotFoundError(f"模型目录不存在: {name} -> {d}")
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，但脚本配置了 device='cuda'")

    ref_files, n_ref = prepare_flat_reference()

    print("构建 cleanfid 特征提取器 ...")
    feature_fn = build_feature_extractor(mode=MODE, device=DEVICE_T)
    if hasattr(feature_fn, "eval"):
        feature_fn.eval()

    print("提取参考集特征 ...")
    ref_feats_np = get_features(ref_files, feature_fn)
    ref_feats_gpu = torch.as_tensor(ref_feats_np, dtype=torch.float64, device=DEVICE_T)
    print(f"参考集特征 shape: {tuple(ref_feats_gpu.shape)}")

    all_details = []
    all_summaries = []

    for model_name, model_dir in MODELS.items():
        print("\n" + "=" * 70)
        print(f"模型: {model_name}")
        print(f"目录: {model_dir}")
        print("=" * 70)

        model_details = []
        model_fid_boot_stack = []
        model_kid_boot_stack = []

        for batch in BATCHES:
            gen_files, per_category = collect_batch_files(model_dir, batch)
            print(f"\n[{model_name}] {batch}: 收集到 {len(gen_files)} 张")
            print(f"  各类张数: {per_category}")

            if len(gen_files) == 0:
                print("  跳过：没有 PNG")
                continue
            if len(gen_files) != 210:
                print(f"  警告：预期 210 张（7×30），实际 {len(gen_files)} 张")

            gen_flat_dir, n_gen = prepare_flat_gen(model_name, batch, gen_files)

            gen_feats_np = get_features(gen_flat_dir, feature_fn)
            gen_feats_gpu = torch.as_tensor(
                gen_feats_np, dtype=torch.float64, device=DEVICE_T
            )
            print(f"  生成集特征 shape: {tuple(gen_feats_gpu.shape)}")

            fid_val = fid_gpu(gen_feats_gpu, ref_feats_gpu)
            kid_val = kid_gpu(gen_feats_gpu, ref_feats_gpu)
            print(f"  FID={fid_val:.6f}, KID={kid_val:.6f}")

            print(f"  bootstrap FID/KID ({N_BOOTSTRAP} 次) ...")
            seed = stable_seed(model_name, batch)
            fid_boot, kid_boot = bootstrap_batch_gpu(
                gen_feats_gpu, ref_feats_gpu, N_BOOTSTRAP, seed
            )

            fid_ci_low, fid_ci_high = quantile_ci(fid_boot)
            kid_ci_low, kid_ci_high = quantile_ci(kid_boot)

            row = {
                "model": model_name,
                "batch": batch,
                "n_gen": n_gen,
                "n_ref": n_ref,
                "fid": fid_val,
                "fid_ci_low": fid_ci_low,
                "fid_ci_high": fid_ci_high,
                "kid": kid_val,
                "kid_ci_low": kid_ci_low,
                "kid_ci_high": kid_ci_high,
            }
            model_details.append(row)
            all_details.append(row)
            model_fid_boot_stack.append(fid_boot)
            model_kid_boot_stack.append(kid_boot)

            print(f"  FID 95% CI = [{fid_ci_low:.6f}, {fid_ci_high:.6f}]")
            print(f"  KID 95% CI = [{kid_ci_low:.6f}, {kid_ci_high:.6f}]")

        if not model_details:
            print(f"模型 {model_name} 没有有效批次，跳过汇总")
            continue

        fid_vals = np.array([r["fid"] for r in model_details], dtype=np.float64)
        kid_vals = np.array([r["kid"] for r in model_details], dtype=np.float64)

        fid_mean = float(np.mean(fid_vals))
        fid_std = float(np.std(fid_vals, ddof=1)) if len(fid_vals) > 1 else 0.0
        kid_mean = float(np.mean(kid_vals))
        kid_std = float(np.std(kid_vals, ddof=1)) if len(kid_vals) > 1 else 0.0

        fid_boot_mean = np.vstack(model_fid_boot_stack).mean(axis=0)
        kid_boot_mean = np.vstack(model_kid_boot_stack).mean(axis=0)

        fid_ci_low, fid_ci_high = quantile_ci(fid_boot_mean)
        kid_ci_low, kid_ci_high = quantile_ci(kid_boot_mean)

        summary = {
            "model": model_name,
            "fid_mean": fid_mean,
            "fid_std": fid_std,
            "fid_ci_low": fid_ci_low,
            "fid_ci_high": fid_ci_high,
            "kid_mean": kid_mean,
            "kid_std": kid_std,
            "kid_ci_low": kid_ci_low,
            "kid_ci_high": kid_ci_high,
            "n_ref": n_ref,
            "n_batches": len(model_details),
        }
        all_summaries.append(summary)

        print("\n" + "-" * 70)
        print(f"模型汇总: {model_name}（n_batches={len(model_details)}）")
        print(
            f"  FID = {fid_mean:.6f} ± {fid_std:.6f}, "
            f"CI=[{fid_ci_low:.6f}, {fid_ci_high:.6f}]"
        )
        print(
            f"  KID = {kid_mean:.6f} ± {kid_std:.6f}, "
            f"CI=[{kid_ci_low:.6f}, {kid_ci_high:.6f}]"
        )
        print("-" * 70)

    # ==================== 写明细 CSV ====================
    detail_fields = [
        "model", "batch", "n_gen", "n_ref",
        "fid", "fid_ci_low", "fid_ci_high",
        "kid", "kid_ci_low", "kid_ci_high",
    ]
    with open(DETAIL_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        for row in all_details:
            writer.writerow(row)

    # ==================== 写汇总 CSV ====================
    summary_fields = [
        "model", "fid_mean", "fid_std", "fid_ci_low", "fid_ci_high",
        "kid_mean", "kid_std", "kid_ci_low", "kid_ci_high",
        "n_ref", "n_batches",
    ]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in all_summaries:
            writer.writerow(row)

    print("\n" + "=" * 70)
    print("完成")
    print(f"明细 CSV: {DETAIL_CSV}（每模型 3 行）")
    print(f"汇总 CSV: {SUMMARY_CSV}（每模型 1 行）")
    print("=" * 70)
    print("\n【表注建议】")
    print("  KID is the primary metric; FID is reported as a complementary metric")
    print("  due to the small held-out set (n=35).")
    print("  FID/KID are computed between each batch of 210 generated images")
    print("  (7 categories x 30 images) and the full held-out set (n=35);")
    print("  mean±std over 3 batches; 95% CI via 1000 image-level bootstrap.")


if __name__ == "__main__":
    main()