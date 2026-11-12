import os
import csv
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.fftpack import dct
import torch
import clip

# ==================== 配置 ====================
DATA_DIR = r"S:/ComfyUI/pHash + CLIP cosine/30_culpart"
OUT_DIR = r"S:/ComfyUI/pHash + CLIP cosine/dedup_out"

HASH_SIZE = 8
HAMMING_THRESHOLD = 5
CLIP_THRESHOLD = 0.98


def phash_bits(img, hash_size=HASH_SIZE, highfreq_factor=4):
    img_size = hash_size * highfreq_factor
    img = img.convert("L").resize((img_size, img_size), Image.LANCZOS)
    pixels = np.asarray(img, dtype=np.float32)
    d = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = d[:hash_size, :hash_size]
    med = np.median(low)
    bits = 0
    for b in (low > med).flatten():
        bits = (bits << 1) | int(b)
    return np.uint64(bits)


def hamming_matrix(bits):
    n = len(bits)
    x = bits[:, None] ^ bits[None, :]
    bytes_view = x.view(np.uint8).reshape(n, n, 8)
    return np.unpackbits(bytes_view, axis=2).sum(axis=2)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith(".png"))
    paths = [os.path.join(DATA_DIR, f) for f in files]
    print(f"找到 {len(paths)} 张 png")

    print("\n计算 pHash...")
    bits = np.zeros(len(paths), dtype=np.uint64)
    resolutions = []
    for i, p in enumerate(tqdm(paths)):
        img = Image.open(p).convert("RGB")
        resolutions.append(img.size[0] * img.size[1])
        bits[i] = phash_bits(img)

    print("\n计算 CLIP embedding...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    embs = np.zeros((len(paths), 512), dtype=np.float32)
    with torch.no_grad():
        for i, p in enumerate(tqdm(paths)):
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            e = model.encode_image(img)
            e = e / e.norm(dim=-1, keepdim=True)
            embs[i] = e.cpu().numpy()[0]

    print("\n计算近重复矩阵...")
    H = hamming_matrix(bits)
    C = embs @ embs.T
    near_phash = (H <= HAMMING_THRESHOLD)
    near_clip  = (C >= CLIP_THRESHOLD)
    np.fill_diagonal(near_phash, False)
    np.fill_diagonal(near_clip, False)
    near = near_phash | near_clip

    iu = np.triu_indices(len(paths), k=1)
    print(f"pHash 命中对: {near_phash[iu].sum()}")
    print(f"CLIP  命中对: {near_clip[iu].sum()}")
    print(f"任一命中对: {near[iu].sum()}")

    uf = UnionFind(len(paths))
    rows, cols = np.where(np.triu(near, k=1))
    for a, b in zip(rows, cols):
        uf.union(int(a), int(b))

    groups = {}
    for i in range(len(paths)):
        groups.setdefault(uf.find(i), []).append(i)

    keep_idx, removed_of = [], {}
    for members in groups.values():
        best = max(members, key=lambda i: resolutions[i])
        keep_idx.append(best)
        for m in members:
            if m != best:
                removed_of[m] = best
    keep_idx = sorted(keep_idx)
    kept_set = set(keep_idx)

    print(f"\n去重前: {len(paths)}  去重后: {len(keep_idx)}  剔除: {len(removed_of)}")

    with open(os.path.join(OUT_DIR, "dedup_report.csv"),
              "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "status", "kept_as", "resolution"])
        for i, p in enumerate(paths):
            w.writerow([
                os.path.basename(p),
                "kept" if i in kept_set else "removed",
                os.path.basename(paths[removed_of[i]]) if i in removed_of else "",
                resolutions[i],
            ])

    for split_name, idxs in [("kept", keep_idx),
                             ("removed", sorted(removed_of.keys()))]:
        dst_dir = os.path.join(OUT_DIR, split_name)
        os.makedirs(dst_dir, exist_ok=True)
        for i in idxs:
            src_png = paths[i]
            shutil.copy2(src_png, dst_dir)
            src_txt = os.path.splitext(src_png)[0] + ".txt"
            if os.path.exists(src_txt):
                shutil.copy2(src_txt, dst_dir)

    print(f"\n输出: {OUT_DIR}")
    print("  dedup_report.csv")
    print("  kept/     保留的图+txt")
    print("  removed/  剔除的图+txt")


if __name__ == "__main__":
    main()