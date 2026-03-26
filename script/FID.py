
import os
from cleanfid import fid

# ==================== 配置 ====================
# 生成图像目录
GEN_IMAGES_DIR = "S:/ComfyUI/FID/gen img"

# 真实图像目录
REAL_IMAGES_DIR = "S:/ComfyUI/FID/rel img"

# ==================== 主程序 ====================
print(f"生成图像目录: {GEN_IMAGES_DIR}")
print(f"真实图像目录: {REAL_IMAGES_DIR}")

if not os.path.exists(GEN_IMAGES_DIR):
    print(f"错误: 生成图像目录不存在: {GEN_IMAGES_DIR}")
    exit(1)

if not os.path.exists(REAL_IMAGES_DIR):
    print(f"错误: 真实图像目录不存在: {REAL_IMAGES_DIR}")
    exit(1)

total_images = 0
motif_folders = [f for f in os.listdir(GEN_IMAGES_DIR)
                 if os.path.isdir(os.path.join(GEN_IMAGES_DIR, f))]

print(f"\n找到 {len(motif_folders)} 个纹样子文件夹")

for folder in motif_folders:
    motif_dir = os.path.join(GEN_IMAGES_DIR, folder)
    image_files = [f for f in os.listdir(motif_dir)
                   if f.lower().endswith(('.png'))]
    total_images += len(image_files)
    short_name = folder[:20] + "..." if len(folder) > 20 else folder
    print(f"  {short_name}: {len(image_files)} 张")

print(f"\n生成图像总数: {total_images} 张")

real_images = [f for f in os.listdir(REAL_IMAGES_DIR)
               if f.lower().endswith(('.png'))]
print(f"真实图像总数: {len(real_images)} 张")

print("\n" + "=" * 50)

try:
    overall_fid = fid.compute_fid(
        GEN_IMAGES_DIR,
        REAL_IMAGES_DIR,
        mode="clean",
        device="cuda",
        num_workers=0
    )

    print("\n" + "=" * 50)
    print(f"FID = {overall_fid:.4f}")
    print("=" * 50)

    with open("fid_result.txt", "w") as f:
        f.write(f"整体 FID: {overall_fid:.4f}\n")
        f.write(f"生成图像总数: {total_images}\n")
        f.write(f"真实图像总数: {len(real_images)}\n")
    print("结果已保存到 fid_result.txt")

except Exception as e:
    print(f"❌ 计算FID时出错: {e}")

print("\n计算完成")