import os
import torch
import clip
import lpips
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from itertools import combinations
from datetime import datetime

# ==================== 配置 ====================
GEN_IMAGES_DIR = "S:/ComfyUI/CLIP+LIPIPS/gen img"
PROMPTS_CSV = "S:/ComfyUI/CLIP+LIPIPS/prompts.csv"
OUTPUT_DIR = "S:/ComfyUI/CLIP+LIPIPS/results"

# 阈值设置
CLIP_THRESHOLD = 0.26

# 输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_clip_model():
    """CLIP模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print("加载CLIP模型...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def load_lpips_model(device):
    """LPIPS模型"""
    print("加载LPIPS模型...")
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    return lpips_model

def scan_images(motif_to_prompt):
    """生成图像"""
    print("\n" + "="*60)
    print("扫描生成图像...")
    print("="*60)

    image_data = []
    motif_image_count = {}

    for motif in motif_to_prompt.keys():
        motif_dir = os.path.join(GEN_IMAGES_DIR, motif)
        if not os.path.exists(motif_dir):
            print(f"目录不存在: {motif_dir}")
            motif_image_count[motif] = 0
            continue

        image_files = [f for f in os.listdir(motif_dir)
                      if f.lower().endswith(('.png'))]

        motif_image_count[motif] = len(image_files)
        print(f"  {motif}: {len(image_files)} 张图片")

        for img_file in image_files:
            image_data.append({
                'motif': motif,
                'image_path': os.path.join(motif_dir, img_file),
                'filename': img_file
            })

    total_images = len(image_data)
    print(f"\n总计: {total_images} 张生成图像")
    return image_data, motif_image_count, total_images

def compute_clip_scores(image_data, motif_to_prompt, clip_model, preprocess, device):
    """计算CLIP Score"""
    print("\n" + "="*60)
    print("计算CLIP Score...")
    print("="*60)

    results = []

    for item in tqdm(image_data, desc="处理图像", unit="张"):
        try:
            # 加载图像
            image = preprocess(Image.open(item['image_path']).convert('RGB')).unsqueeze(0).to(device)

            # 提示词
            prompt = motif_to_prompt[item['motif']]

            # 进行编码
            text = clip.tokenize([prompt], truncate=True).to(device)

            with torch.no_grad():
                image_features = clip_model.encode_image(image)
                text_features = clip_model.encode_text(text)

                # 归一化
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # 计算相似度
                similarity = (image_features @ text_features.T).item()

            results.append({
                'motif': item['motif'],
                'filename': item['filename'],
                'image_path': item['image_path'],
                'clip_score': similarity,
                'passed': similarity >= CLIP_THRESHOLD
            })

        except Exception as e:
            print(f"\n处理 {item['image_path']} 时出错: {e}")
            continue

    df = pd.DataFrame(results)
    return df

def compute_lpips_diversity(filtered_df, lpips_model, device):
    """计算过滤后LPIPS多样性和每张图像的平均LPIPS得分"""
    print("\n" + "="*60)
    print("计算LPIPS多样性...")
    print("="*60)

    motifs = filtered_df['motif'].unique()
    diversity_results = []
    image_scores = []  # 存储图像的平均LPIPS得分

    for motif in motifs:
        motif_df = filtered_df[filtered_df['motif'] == motif]
        image_paths = motif_df['image_path'].tolist()

        if len(image_paths) < 2:
            print(f"\n{motif}: 只有 {len(image_paths)} 张图像，无法计算多样性")
            diversity_results.append({
                'motif': motif,
                'num_images': len(image_paths),
                'num_pairs': 0,
                'avg_lpips': None,
                'std_lpips': None,
                'min_lpips': None,
                'max_lpips': None
            })
            for img_path in image_paths:
                image_scores.append({
                    'motif': motif,
                    'image_path': img_path,
                    'filename': os.path.basename(img_path),
                    'avg_lpips': None,
                    'num_pairs': 0
                })
            continue

        num_pairs = len(image_paths) * (len(image_paths) - 1) // 2
        print(f"\n{motif}: {len(image_paths)} 张图像，计算 {num_pairs} 个图像对...")

        lpips_values = []
        sum_dist = {path: 0.0 for path in image_paths}
        count = {path: 0 for path in image_paths}

        for img1_path, img2_path in tqdm(combinations(image_paths, 2),
                                         desc=motif,
                                         total=num_pairs):
            try:
                # 加载图像
                img1 = Image.open(img1_path).convert('RGB')
                img2 = Image.open(img2_path).convert('RGB')

                img1 = img1.resize((224, 224))
                img2 = img2.resize((224, 224))

                img1_tensor = (torch.tensor(np.array(img1)).float().permute(2,0,1).unsqueeze(0) / 127.5 - 1).to(device)
                img2_tensor = (torch.tensor(np.array(img2)).float().permute(2,0,1).unsqueeze(0) / 127.5 - 1).to(device)

                with torch.no_grad():
                    dist = lpips_model(img1_tensor, img2_tensor).item()

                lpips_values.append(dist)

                sum_dist[img1_path] += dist
                sum_dist[img2_path] += dist
                count[img1_path] += 1
                count[img2_path] += 1

            except Exception as e:
                print(f"处理图像对时出错: {e}")
                continue

        if lpips_values:
            avg_lpips = np.mean(lpips_values)
            std_lpips = np.std(lpips_values)
            min_lpips = np.min(lpips_values)
            max_lpips = np.max(lpips_values)

            print(f"平均LPIPS: {avg_lpips:.4f} ± {std_lpips:.4f}")

            diversity_results.append({
                'motif': motif,
                'num_images': len(image_paths),
                'num_pairs': len(lpips_values),
                'avg_lpips': avg_lpips,
                'std_lpips': std_lpips,
                'min_lpips': min_lpips,
                'max_lpips': max_lpips
            })

            # 计算图像的平均LPIPS得分
            for img_path in image_paths:
                cnt = count[img_path]
                if cnt > 0:
                    avg = sum_dist[img_path] / cnt
                else:
                    avg = None
                image_scores.append({
                    'motif': motif,
                    'image_path': img_path,
                    'filename': os.path.basename(img_path),
                    'avg_lpips': avg,
                    'num_pairs': cnt
                })
        else:
            diversity_results.append({
                'motif': motif,
                'num_images': len(image_paths),
                'num_pairs': 0,
                'avg_lpips': None,
                'std_lpips': None,
                'min_lpips': None,
                'max_lpips': None
            })
            for img_path in image_paths:
                image_scores.append({
                    'motif': motif,
                    'image_path': img_path,
                    'filename': os.path.basename(img_path),
                    'avg_lpips': None,
                    'num_pairs': 0
                })

    image_scores_df = pd.DataFrame(image_scores)
    return diversity_results, image_scores_df

def main():
    print("="*60)
    print("CLIP过滤 + LPIPS多样性计算")
    print(f"CLIP阈值: {CLIP_THRESHOLD}")
    print("="*60)

    # 提示词
    prompts_df = pd.read_csv(PROMPTS_CSV)
    motif_to_prompt = dict(zip(prompts_df['motif'], prompts_df['prompt']))
    motifs = list(motif_to_prompt.keys())
    print(f"加载了 {len(motifs)} 个纹样的提示词:")
    for motif in motifs:
        print(f"  - {motif}")

    # 加载模型
    clip_model, preprocess, device = load_clip_model()
    lpips_model = load_lpips_model(device)

    # 扫描图像
    image_data, motif_image_count, total_images = scan_images(motif_to_prompt)

    if total_images == 0:
        print("没有找到任何图像")
        return

    # CLIP Score
    clip_df = compute_clip_scores(image_data, motif_to_prompt, clip_model, preprocess, device)

    # 保存CLIP结果
    clip_output = os.path.join(OUTPUT_DIR, "clip_scores_all.csv")
    clip_df.to_csv(clip_output, index=False, encoding='utf-8-sig')
    print(f"\nCLIP结果已保存到: {clip_output}")

    # CLIP统计
    passed_df = clip_df[clip_df['passed']]
    print("\n" + "="*60)
    print("CLIP Score统计")
    print("="*60)
    print(f"总图像数: {len(clip_df)}")
    print(f"通过阈值(≥{CLIP_THRESHOLD}): {len(passed_df)}/{len(clip_df)} ({len(passed_df)/len(clip_df)*100:.1f}%)")

    # CLIP通过率
    print("\nCLIP通过率:")
    for motif in motifs:
        motif_df = clip_df[clip_df['motif'] == motif]
        motif_passed = passed_df[passed_df['motif'] == motif]
        print(f"  {motif:12}: {len(motif_passed)}/{len(motif_df)} ({len(motif_passed)/len(motif_df)*100:.1f}%)")

    # 计算LPIPS多样性
    if len(passed_df) == 0:
        print("\n没有图像通过CLIP阈值，无法计算多样性")
        return

    diversity_results, image_scores_df = compute_lpips_diversity(passed_df, lpips_model, device)

    # 保存多样性结果
    diversity_df = pd.DataFrame(diversity_results)
    diversity_output = os.path.join(OUTPUT_DIR, "diversity_results.csv")
    diversity_df.to_csv(diversity_output, index=False, encoding='utf-8-sig')
    print(f"\n多样性结果已保存到: {diversity_output}")

    # 保存图像的LPIPS得分
    if not image_scores_df.empty:
        image_scores_output = os.path.join(OUTPUT_DIR, "image_lpips_scores.csv")
        image_scores_df.to_csv(image_scores_output, index=False, encoding='utf-8-sig')
        print(f"图像的LPIPS得分已保存到: {image_scores_output}")

    # 计算整体多样性
    valid_results = [r for r in diversity_results if r['avg_lpips'] is not None]
    if valid_results:
        overall_avg = np.mean([r['avg_lpips'] for r in valid_results])
        overall_std = np.std([r['avg_lpips'] for r in valid_results])

        print(f"有效纹样数: {len(valid_results)}/{len(motifs)}")
        print(f"整体平均LPIPS: {overall_avg:.4f} ± {overall_std:.4f}")

        print("\n各纹样多样性:")
        for r in valid_results:
            print(f"  {r['motif']:12}: {r['avg_lpips']:.4f} ± {r['std_lpips']:.4f} (n={r['num_images']})")

    # 生成报告
    report_path = os.path.join(OUTPUT_DIR, "final_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("CLIP过滤 + LPIPS多样性计算报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"CLIP阈值: {CLIP_THRESHOLD}\n")
        f.write("="*60 + "\n\n")

        f.write("【CLIP统计】\n")
        f.write(f"总图像数: {len(clip_df)}\n")
        f.write(f"通过阈值: {len(passed_df)}/{len(clip_df)} ({len(passed_df)/len(clip_df)*100:.1f}%)\n\n")

        f.write("各纹样通过率:\n")
        for motif in motifs:
            motif_df = clip_df[clip_df['motif'] == motif]
            motif_passed = passed_df[passed_df['motif'] == motif]
            f.write(f"  {motif}: {len(motif_passed)}/{len(motif_df)} ({len(motif_passed)/len(motif_df)*100:.1f}%)\n")

        f.write("\n【多样性统计】\n")
        for r in valid_results:
            f.write(f"  {r['motif']}: {r['avg_lpips']:.4f} ± {r['std_lpips']:.4f} (n={r['num_images']})\n")

        if valid_results:
            f.write(f"\n整体平均LPIPS: {overall_avg:.4f} ± {overall_std:.4f}\n")

        if not image_scores_df.empty:
            f.write("\n【每张图像LPIPS得分统计】\n")
            avg_scores = image_scores_df.groupby('motif')['avg_lpips'].mean()
            std_scores = image_scores_df.groupby('motif')['avg_lpips'].std()
            for motif in motifs:
                if motif in avg_scores.index and not pd.isna(avg_scores[motif]):
                    f.write(f"  {motif}: 平均独特性得分 = {avg_scores[motif]:.4f} ± {std_scores[motif]:.4f}\n")

    print(f"\n报告已保存到: {report_path}")
    print("="*60)

if __name__ == "__main__":
    main()