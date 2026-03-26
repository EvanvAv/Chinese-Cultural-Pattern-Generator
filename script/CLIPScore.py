
import os
import torch
import clip
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime

# ==================== 配置 ====================
GEN_IMAGES_DIR = "S:/ComfyUI/CLIP/gen img"
PROMPTS_CSV = "S:/ComfyUI/CLIP/prompts.csv"
OUTPUT_DETAIL_CSV = "clip_scores_detailed.csv"
OUTPUT_SUMMARY_CSV = "clip_scores_summary.csv"
OUTPUT_FILTERED_CSV = "clip_scores_filtered_026.csv"
OUTPUT_STATS_TXT = "clip_scores_stats_026.txt"

# 阈值设置
THRESHOLD = 0.26

# CLIP最大token长度
CLIP_MAX_TOKENS = 77

# ==================== 主程序 ====================
def main():
    print("="*70)
    print("CLIP Score计算")
    print(f"阈值: {THRESHOLD}")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print("加载CLIP模型...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 读取提示词
    prompts_df = pd.read_csv(PROMPTS_CSV)
    motif_to_prompt = dict(zip(prompts_df['motif'], prompts_df['prompt']))
    motifs = list(motif_to_prompt.keys())
    print(f"加载了 {len(motifs)} 个纹样的提示词:")
    for motif in motifs:
        print(f"  - {motif}")

    print("\n" + "="*70)
    print("扫描生成图像...")
    print("="*70)

    image_data = []
    motif_image_count = {}

    for motif in motifs:
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

    if total_images == 0:
        print("没有找到任何图像")
        return

    # 计算CLIP Score
    print("\n" + "="*70)
    print("计算CLIP Score...")
    print("="*70)

    results = []

    for item in tqdm(image_data, desc="处理图像", unit="张"):
        try:
            # 加载图像
            image = preprocess(Image.open(item['image_path']).convert('RGB')).unsqueeze(0).to(device)

            # 获取对应提示词
            prompt = motif_to_prompt[item['motif']]

            # 对提示词进行编码
            text = clip.tokenize([prompt], truncate=True).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)

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
                'passed_threshold': similarity >= THRESHOLD,
                'prompt_length': len(prompt),
                'prompt_preview': prompt[:100] + '...' if len(prompt) > 100 else prompt
            })

        except Exception as e:
            print(f"\n处理 {item['image_path']} 时出错: {e}")
            continue

    df = pd.DataFrame(results)

    # ==================== 保存结果 ====================
    df.to_csv(OUTPUT_DETAIL_CSV, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {OUTPUT_DETAIL_CSV}")

    # ==================== 保存过滤后的结果 ====================
    filtered_df = df[df['passed_threshold']]
    filtered_df.to_csv(OUTPUT_FILTERED_CSV, index=False, encoding='utf-8-sig')
    print(f"过滤后结果（≥{THRESHOLD}）已保存到: {OUTPUT_FILTERED_CSV}")

    # ==================== 生成汇总统计 ====================
    print("\n" + "="*70)
    print("CLIP Score汇总统计）")
    print("="*70)

    # 过滤前
    overall_mean_before = df['clip_score'].mean()
    overall_std_before = df['clip_score'].std()
    overall_min_before = df['clip_score'].min()
    overall_max_before = df['clip_score'].max()
    overall_median_before = df['clip_score'].median()

    # 过滤后
    overall_mean_after = filtered_df['clip_score'].mean() if len(filtered_df) > 0 else 0
    overall_std_after = filtered_df['clip_score'].std() if len(filtered_df) > 0 else 0
    overall_min_after = filtered_df['clip_score'].min() if len(filtered_df) > 0 else 0
    overall_max_after = filtered_df['clip_score'].max() if len(filtered_df) > 0 else 0
    overall_median_after = filtered_df['clip_score'].median() if len(filtered_df) > 0 else 0

    print(f"\n【整体统计】")
    print(f"  总图像数: {len(df)} 张")
    print(f"  通过阈值图像: {len(filtered_df)} 张 ({len(filtered_df)/len(df)*100:.1f}%)")
    print(f"\n  过滤前 - 平均: {overall_mean_before:.4f} ± {overall_std_before:.4f}")
    print(f"          中位数: {overall_median_before:.4f}")
    print(f"          范围: {overall_min_before:.4f} ~ {overall_max_before:.4f}")
    print(f"\n  过滤后 - 平均: {overall_mean_after:.4f} ± {overall_std_after:.4f}")
    print(f"          中位数: {overall_median_after:.4f}")
    print(f"          范围: {overall_min_after:.4f} ~ {overall_max_after:.4f}")

    # 按纹样统计
    print(f"\n【各纹样统计】")

    summary_data = []

    for motif in motifs:
        motif_df = df[df['motif'] == motif]
        motif_filtered = filtered_df[filtered_df['motif'] == motif]

        if len(motif_df) == 0:
            print(f"  {motif:12}: 无数据")
            continue

        passed_count = len(motif_filtered)
        total_count = len(motif_df)
        pass_rate = passed_count / total_count * 100

        motif_mean_before = motif_df['clip_score'].mean()
        motif_std_before = motif_df['clip_score'].std()
        motif_mean_after = motif_filtered['clip_score'].mean() if passed_count > 0 else 0
        motif_std_after = motif_filtered['clip_score'].std() if passed_count > 0 else 0

        print(f"\n  {motif:12}:")
        print(f"    图像数: {total_count} 张")
        print(f"    通过阈值: {passed_count}/{total_count} ({pass_rate:.1f}%)")
        print(f"    过滤前: {motif_mean_before:.4f} ± {motif_std_before:.4f}")
        if passed_count > 0:
            print(f"    过滤后: {motif_mean_after:.4f} ± {motif_std_after:.4f}")
        else:
            print(f"    过滤后: 无数据")

        summary_data.append({
            'motif': motif,
            'total_images': total_count,
            'passed_images': passed_count,
            'pass_rate': pass_rate,
            'mean_before': motif_mean_before,
            'std_before': motif_std_before,
            'mean_after': motif_mean_after if passed_count > 0 else None,
            'std_after': motif_std_after if passed_count > 0 else None
        })

    summary_df = pd.DataFrame(summary_data)

    overall_row = pd.DataFrame([{
        'motif': 'OVERALL',
        'total_images': len(df),
        'passed_images': len(filtered_df),
        'pass_rate': len(filtered_df)/len(df)*100,
        'mean_before': overall_mean_before,
        'std_before': overall_std_before,
        'mean_after': overall_mean_after,
        'std_after': overall_std_after
    }])

    summary_df = pd.concat([summary_df, overall_row], ignore_index=True)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding='utf-8-sig')
    print(f"\n汇总统计已保存到: {OUTPUT_SUMMARY_CSV}")

    # ==================== 生成文本报告 ====================
    with open(OUTPUT_STATS_TXT, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("CLIP Score 计算报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"阈值: {THRESHOLD}\n")
        f.write("="*70 + "\n\n")

        f.write(f"【配置信息】\n")
        f.write(f"  生成图像目录: {GEN_IMAGES_DIR}\n")
        f.write(f"  提示词文件: {PROMPTS_CSV}\n")
        f.write(f"  纹样数量: {len(motifs)}\n")
        f.write(f"  总图像数: {total_images}\n\n")

        f.write(f"【整体统计】\n")
        f.write(f"  总图像数: {len(df)} 张\n")
        f.write(f"  通过阈值(≥{THRESHOLD})图像: {len(filtered_df)} 张 ({len(filtered_df)/len(df)*100:.1f}%)\n\n")
        f.write(f"  过滤前:\n")
        f.write(f"    平均: {overall_mean_before:.4f} ± {overall_std_before:.4f}\n")
        f.write(f"    中位数: {overall_median_before:.4f}\n")
        f.write(f"    范围: {overall_min_before:.4f} ~ {overall_max_before:.4f}\n\n")
        f.write(f"  过滤后:\n")
        f.write(f"    平均: {overall_mean_after:.4f} ± {overall_std_after:.4f}\n")
        f.write(f"    中位数: {overall_median_after:.4f}\n")
        f.write(f"    范围: {overall_min_after:.4f} ~ {overall_max_after:.4f}\n\n")

        f.write(f"【各纹样统计】\n")
        for item in summary_data:
            f.write(f"\n  {item['motif']}:\n")
            f.write(f"    图像数: {item['total_images']} 张\n")
            f.write(f"    通过阈值: {item['passed_images']}/{item['total_images']} ({item['pass_rate']:.1f}%)\n")
            f.write(f"    过滤前: {item['mean_before']:.4f} ± {item['std_before']:.4f}\n")
            if item['mean_after'] is not None:
                f.write(f"    过滤后: {item['mean_after']:.4f} ± {item['std_after']:.4f}\n")
            else:
                f.write(f"    过滤后: 无数据\n")

        f.write("\n" + "="*70 + "\n")
        f.write("报告结束\n")
        f.write("="*70 + "\n")

    print(f"\n文本报告已保存到: {OUTPUT_STATS_TXT}")

if __name__ == "__main__":
    main()