import os
import torch
import lpips
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

# ==================== 配置 ====================
# 生成图像目录
GEN_IMAGES_DIR = "S:/ComfyUI/LIPIPS/gen img"

# 真实图像目录
REAL_IMAGES_DIR = "S:/ComfyUI/LIPIPS/rel img"

# 计算次数
N_RUNS = 10


# ==================== 主程序 ====================
def compute_one_run(gen_images, all_real_images, lpips_model, device, run_idx):
    """LPIPS计算"""
    print(f"\n--- 第 {run_idx + 1}/{N_RUNS} 次计算 ---")

    max_real = min(len(all_real_images), 100)
    real_samples = random.sample(all_real_images, max_real)
    print(f"本次使用 {len(real_samples)} 张真实图像")

    all_lpips = []

    for gen_path in tqdm(gen_images, desc=f"Run {run_idx + 1}", leave=False):
        try:
            # 加载生成图像
            gen_img = Image.open(gen_path).convert('RGB')
            gen_img = gen_img.resize((224, 224))
            gen_tensor = (torch.tensor(np.array(gen_img)).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1).to(device)

            # 与多张真实图像比较，取最相似的真实图像
            min_lpips = float('inf')

            for real_path in real_samples:
                try:
                    real_img = Image.open(real_path).convert('RGB')
                    real_img = real_img.resize((224, 224))
                    real_tensor = (
                                torch.tensor(np.array(real_img)).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1).to(
                        device)

                    with torch.no_grad():
                        dist = lpips_model(gen_tensor, real_tensor).item()

                    if dist < min_lpips:
                        min_lpips = dist

                except Exception as e:
                    continue

            if min_lpips != float('inf'):
                all_lpips.append(min_lpips)

        except Exception as e:
            print(f"处理 {gen_path} 时出错: {e}")
            continue

    if len(all_lpips) == 0:
        return None

    return {
        "mean": np.mean(all_lpips),
        "std": np.std(all_lpips),
        "min": np.min(all_lpips),
        "max": np.max(all_lpips),
        "median": np.median(all_lpips),
        "count": len(all_lpips)
    }


def main():
    print(f"生成图像目录: {GEN_IMAGES_DIR}")
    print(f"真实图像目录: {REAL_IMAGES_DIR}")
    print(f"计算次数: {N_RUNS}")

    if not os.path.exists(GEN_IMAGES_DIR):
        print(f"❌ 错误: 生成图像目录不存在: {GEN_IMAGES_DIR}")
        return
    if not os.path.exists(REAL_IMAGES_DIR):
        print(f"❌ 错误: 真实图像目录不存在: {REAL_IMAGES_DIR}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 加载LPIPS模型
    print("加载LPIPS模型...")
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    # ==================== 图像路径 ====================
    print("\n扫描图像文件...")

    gen_images = []
    for root, dirs, files in os.walk(GEN_IMAGES_DIR):
        for file in files:
            if file.lower().endswith(('.png')):
                gen_images.append(os.path.join(root, file))

    real_images = []
    for root, dirs, files in os.walk(REAL_IMAGES_DIR):
        for file in files:
            if file.lower().endswith(('.png')):
                real_images.append(os.path.join(root, file))

    print(f"找到生成图像: {len(gen_images)} 张")
    print(f"找到真实图像: {len(real_images)} 张")

    if len(gen_images) == 0 or len(real_images) == 0:
        print("图像不足，无法计算")
        return

    # ==================== 循环计算 ====================
    all_results = []

    for run_idx in range(N_RUNS):
        result = compute_one_run(gen_images, real_images, lpips_model, device, run_idx)
        if result:
            all_results.append(result)
            print(f"  第 {run_idx + 1} 次平均LPIPS: {result['mean']:.4f}")

    if len(all_results) == 0:
        print("所有计算都失败")
        return

    # ==================== 统计结果 ====================
    means = [r['mean'] for r in all_results]
    stds = [r['std'] for r in all_results]
    mins = [r['min'] for r in all_results]
    maxs = [r['max'] for r in all_results]
    medians = [r['median'] for r in all_results]

    print("\n" + "=" * 50)
    print("LPIPS计算结果汇总")
    print("=" * 50)
    print(f"有效计算次数: {len(all_results)}/{N_RUNS}")
    print(f"\n平均值的统计:")
    print(f"  最终平均LPIPS: {np.mean(means):.4f} ± {np.std(means):.4f}")
    print(f"  最小值: {np.min(means):.4f}")
    print(f"  最大值: {np.max(means):.4f}")

    print(f"\n标准差统计:")
    print(f"  平均标准差: {np.mean(stds):.4f}")

    print(f"\n所有结果明细:")
    for i, r in enumerate(all_results):
        print(f"  Run {i + 1}: {r['mean']:.4f} ± {r['std']:.4f} (min={r['min']:.4f}, max={r['max']:.4f})")

    with open("lpips_10runs_results.txt", "w") as f:
        f.write("LPIPS 计算结果\n")
        f.write(f"生成图像目录: {GEN_IMAGES_DIR}\n")
        f.write(f"真实图像目录: {REAL_IMAGES_DIR}\n")
        f.write(f"生成图像总数: {len(gen_images)}\n")
        f.write(f"真实图像总数: {len(real_images)}\n")
        f.write(f"计算次数: {N_RUNS}\n")
        f.write(f"有效计算次数: {len(all_results)}/{N_RUNS}\n\n")

        f.write("平均值统计:\n")
        f.write(f"  最终平均LPIPS: {np.mean(means):.4f} ± {np.std(means):.4f}\n")
        f.write(f"  最小值: {np.min(means):.4f}\n")
        f.write(f"  最大值: {np.max(means):.4f}\n\n")

        f.write("各次结果:\n")
        for i, r in enumerate(all_results):
            f.write(
                f"  Run {i + 1}: {r['mean']:.4f} ± {r['std']:.4f} (min={r['min']:.4f}, max={r['max']:.4f}, n={r['count']})\n")

    print("\n结果已保存到 lpips_10runs_results.txt")
    print("=" * 50)



if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    main()