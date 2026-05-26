"""
3D Gaussian Splatting 本地训练管线
输入：照片文件夹 → COLMAP → 3DGS 训练 → 导出 .ply/.splat
"""

import argparse
import os
import subprocess
import struct
import json
import sys
from pathlib import Path

# ─── CUDA/nvcc 环境配置 ──────────────────────────────────────────
def setup_cuda_env():
    """查找并设置 CUDA 环境变量，让 gsplat 能找到 nvcc。"""
    # 检查 pip 安装的 nvidia 包中的 CUDA
    pip_cuda = os.path.join(
        os.path.dirname(sys.executable), "..", "Lib", "site-packages", "nvidia", "cu13"
    )
    pip_cuda = os.path.abspath(pip_cuda)
    if os.path.isdir(os.path.join(pip_cuda, "bin")):
        os.environ.setdefault("CUDA_HOME", pip_cuda)
        os.environ.setdefault("CUDA_PATH", pip_cuda)
        bin_dir = os.path.join(pip_cuda, "bin")
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

setup_cuda_env()

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# ─── 配置 ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COLMAP_EXE = os.environ.get("COLMAP_EXE", "")
if not COLMAP_EXE:
    # 自动查找 colmap（优先脚本目录下的便携版）
    colmap_candidates = [
        os.path.join(SCRIPT_DIR, "colmap", "bin", "colmap.exe"),
        os.path.join(SCRIPT_DIR, "..", "colmap", "bin", "colmap.exe"),
    ]
    for candidate in colmap_candidates:
        if os.path.exists(candidate):
            COLMAP_EXE = candidate
            break
    if not COLMAP_EXE:
        COLMAP_EXE = "colmap"  # 回退到 PATH
TRAINING_STEPS = 30_000
WARMUP_STEPS = 500


# ═══════════════════════════════════════════════════════════════════════
# COLMAP 解析
# ═══════════════════════════════════════════════════════════════════════

def parse_colmap_text(sparse_dir: str):
    """解析 COLMAP 文本格式输出。"""
    cameras = {}
    with open(os.path.join(sparse_dir, "cameras.txt"), "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split()
            cam_id = int(parts[0])
            model = parts[1]
            w, h = int(parts[2]), int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[cam_id] = {"model": model, "width": w, "height": h, "params": params}

    images = {}
    with open(os.path.join(sparse_dir, "images.txt"), "r") as f:
        raw_lines = f.readlines()
    lines = [l.strip() for l in raw_lines if not l.startswith("#") and l.strip() != ""]
    # images.txt 每两行一组：第一行是元数据，第二行是特征点
    assert len(lines) % 2 == 0, f"images.txt 行数({len(lines)})应为偶数"
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        img_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name = parts[9]
        # 构建 world-to-camera 矩阵 (camtoworld → viewmat)
        q = np.array([qw, qx, qy, qz])
        R = quat_to_rotmat(q)
        t = np.array([tx, ty, tz])
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = t
        images[img_id] = {
            "camera_id": cam_id,
            "name": name,
            "viewmat": viewmat,
            "qvec": q,
            "tvec": t,
        }

    points3d = {}
    pt_path = os.path.join(sparse_dir, "points3D.txt")
    if os.path.exists(pt_path):
        with open(pt_path, "r") as f:
            for line in f:
                if line.startswith("#") or line.strip() == "":
                    continue
                parts = line.strip().split()
                pt_id = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
                points3d[pt_id] = {"position": np.array([x, y, z]), "color": np.array([r, g, b])}

    return cameras, images, points3d


def quat_to_rotmat(q):
    """四元数到旋转矩阵 (w, x, y, z)"""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def get_camera_intrinsics(cam, device="cuda"):
    """从 COLMAP 相机参数构建内参矩阵 K."""
    model = cam["model"]
    params = cam["params"]
    w, h = cam["width"], cam["height"]

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[0], params[1], params[2]
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        f = (fx + fy) / 2
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, _ = params[0], params[1], params[2], params[3]
    elif model == "RADIAL":
        f, cx, cy, _, _ = params[0], params[1], params[2], params[3], params[4]
    else:
        f, cx, cy = params[0], params[1], params[2]

    K = torch.tensor([
        [f, 0, cx + 0.5],
        [0, f, cy + 0.5],
        [0, 0, 1],
    ], dtype=torch.float32, device=device)
    return K, cam["width"], cam["height"]


# ═══════════════════════════════════════════════════════════════════════
# 3DGS 模型
# ═══════════════════════════════════════════════════════════════════════

class GaussianModel:
    """可训练的 3D Gaussian 参数。"""

    def __init__(self, points3d=None, device="cuda"):
        self.device = device
        if points3d is not None and len(points3d) > 0:
            positions = np.array([p["position"] for p in points3d.values()])
            colors = np.array([p["color"] for p in points3d.values()])
            n = len(positions)
            print(f"[初始化] 从 {n} 个点云初始化高斯")
            self.means = torch.tensor(positions, dtype=torch.float32, device=device, requires_grad=True)
            # 颜色存为 logit 空间值，通过 sigmoid 转换为 [0,1]
            color_init = (colors / 255.0) * 2.0 - 1.0  # [0,1] → [-1,1] logit 初始值
            self.colors_logit = torch.tensor(color_init, dtype=torch.float32, device=device, requires_grad=True)
        else:
            n = 100_000
            self.means = torch.zeros((n, 3), device=device, requires_grad=True)
            self.colors_logit = torch.zeros((n, 3), device=device, requires_grad=True)

        # 四元数 (归一化在 forward 中做)
        self.quats = torch.randn((n, 4), device=device, requires_grad=True)
        # 尺度 (exp 空间，保证正数)
        dist = torch.cdist(self.means[:min(n, 1000)], self.means[:min(n, 1000)])
        avg_dist = dist[dist > 0].mean().item() if dist.numel() > 1 else 0.1
        init_scale = np.log(max(avg_dist * 0.5, 0.01))
        self.scales_log = torch.full((n, 3), init_scale, device=device, requires_grad=True)
        # 不透明度 (logit space)
        self.opacities_logit = torch.zeros((n,), device=device, requires_grad=True)

    @property
    def opacities(self):
        return torch.sigmoid(self.opacities_logit)

    @property
    def colors(self):
        return torch.sigmoid(self.colors_logit)

    @property
    def scales(self):
        return torch.exp(self.scales_log)

    def get_params(self):
        return {
            "means": self.means,
            "quats": self.quats,
            "scales_log": self.scales_log,
            "opacities_logit": self.opacities_logit,
            "colors_logit": self.colors_logit,
        }

    def save_ply(self, path):
        """导出为 PLY 格式 (标准 3DGS PLY 格式)。"""
        means = self.means.detach().cpu().numpy()
        scales = self.scales.detach().cpu().numpy()
        quats = self.quats.detach().cpu().numpy()
        opacities = torch.sigmoid(self.opacities_logit).detach().cpu().numpy()
        colors = self.colors.detach().cpu().numpy()
        n = len(means)

        # SH DC coefficients: color → SH0
        sh0 = ((colors - 0.5) / 0.2820947917738781).astype(np.float32)  # inverse of SH scaling

        with open(path, "wb") as f:
            f.write(b"ply\nformat binary_little_endian 1.0\n")
            f.write(f"element vertex {n}\n".encode())
            f.write(b"property float x\nproperty float y\nproperty float z\n")
            f.write(b"property float nx\nproperty float ny\nproperty float nz\n")
            f.write(b"property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n")
            f.write(b"property float opacity\n")
            f.write(b"property float scale_0\nproperty float scale_1\nproperty float scale_2\n")
            f.write(b"property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n")
            f.write(b"end_header\n")
            for i in range(n):
                f.write(struct.pack("<fff", means[i, 0], means[i, 1], means[i, 2]))
                f.write(struct.pack("<fff", 0.0, 0.0, 0.0))  # normals
                f.write(struct.pack("<fff", sh0[i, 0], sh0[i, 1], sh0[i, 2]))
                f.write(struct.pack("<f", opacities[i]))
                f.write(struct.pack("<fff", scales[i, 0], scales[i, 1], scales[i, 2]))
                q = quats[i] / (np.linalg.norm(quats[i]) + 1e-10)
                f.write(struct.pack("<ffff", q[0], q[1], q[2], q[3]))
        print(f"[导出] PLY 保存到: {path}")

    def save_splat(self, path):
        """导出为 .splat 格式 (标准 32 字节/点, 兼容 antimatter15 等浏览器查看器)。
        格式: pos(12B) + rgba(4B) + scale(12B) + quat(4B packed) = 32B
        """
        means = self.means.detach().cpu().numpy()
        scales = self.scales.detach().cpu().numpy()
        quats = self.quats.detach().cpu().numpy()
        opacities = torch.sigmoid(self.opacities_logit).detach().cpu().numpy()
        colors = self.colors.detach().cpu().numpy()

        n = len(means)
        buf = bytearray(n * 32)
        for i in range(n):
            off = i * 32
            # 位置 (3 float32)
            struct.pack_into("fff", buf, off, means[i, 0], means[i, 1], means[i, 2])
            # 颜色 RGB + alpha (4 uint8)
            c = np.clip(colors[i] * 255, 0, 255).astype(np.uint8)
            a = np.clip(opacities[i] * 255, 0, 255).astype(np.uint8)
            struct.pack_into("BBBB", buf, off + 12, c[0], c[1], c[2], a)
            # scale (3 float32)
            struct.pack_into("fff", buf, off + 16, scales[i, 0], scales[i, 1], scales[i, 2])
            # 旋转 (4 uint8 packed, 每个分量 f32 → int8 映射到 [0,255])
            q = quats[i] / (np.linalg.norm(quats[i]) + 1e-10)
            q_packed = np.clip((q * 128 + 128).astype(np.int32), 0, 255).astype(np.uint8)
            struct.pack_into("BBBB", buf, off + 28, q_packed[0], q_packed[1], q_packed[2], q_packed[3])
        with open(path, "wb") as f:
            f.write(buf)
        print(f"[导出] SPLAT 保存到: {path} ({n} 点)")


# ═══════════════════════════════════════════════════════════════════════
# 步骤 1: COLMAP
# ═══════════════════════════════════════════════════════════════════════

def run_colmap(images_dir: str, output_dir: str):
    db_path = os.path.join(output_dir, "database.db")
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)

    print("[COLMAP] 提取图像特征...")
    print("[COLMAP] 如果照片分辨率很高，这一步可能需要 5-30 分钟")
    subprocess.run([
        COLMAP_EXE, "feature_extractor",
        "--database_path", db_path,
        "--image_path", images_dir,
        "--FeatureExtraction.use_gpu", "1",
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.max_num_features", "8192",
        "--SiftExtraction.max_image_size", "1600",
    ], check=True)

    print("[COLMAP] 特征匹配...")
    subprocess.run([
        COLMAP_EXE, "exhaustive_matcher",
        "--database_path", db_path,
        "--FeatureMatching.use_gpu", "1",
    ], check=True)

    print("[COLMAP] 稀疏重建...")
    result = subprocess.run([
        COLMAP_EXE, "mapper",
        "--database_path", db_path,
        "--image_path", images_dir,
        "--output_path", os.path.join(output_dir, "sparse"),
    ])

    # 查找 mapper 输出目录（子目录 0, 1, 等）
    sparse_parent = os.path.join(output_dir, "sparse")
    if not os.path.isdir(sparse_dir) or not os.listdir(sparse_dir):
        dirs = sorted([d for d in os.listdir(sparse_parent)
                        if os.path.isdir(os.path.join(sparse_parent, d))], key=int)
        if dirs:
            sparse_dir = os.path.join(sparse_parent, dirs[-1])
        else:
            raise RuntimeError("COLMAP mapper 未能生成重建结果")

    # 转文本格式
    print("[COLMAP] 导出文本格式...")
    text_dir = os.path.join(output_dir, "sparse_text")
    os.makedirs(text_dir, exist_ok=True)
    subprocess.run([
        COLMAP_EXE, "model_converter",
        "--input_path", sparse_dir,
        "--output_path", text_dir,
        "--output_type", "TXT",
    ], check=True)

    print(f"[COLMAP] 完成！")
    return text_dir


# ═══════════════════════════════════════════════════════════════════════
# 步骤 2: 训练 3DGS
# ═══════════════════════════════════════════════════════════════════════

def train_3dgs(images_dir: str, colmap_text_dir: str, output_dir: str,
               steps: int = TRAINING_STEPS, port: int = 8080):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[3DGS] 使用设备: {device}")

    # 解析 COLMAP 结果
    cameras, images, points3d = parse_colmap_text(colmap_text_dir)
    print(f"[数据] 相机: {len(cameras)}, 图像: {len(images)}, 点云: {len(points3d)}")

    # 加载图像
    train_data = []
    for img_id, img_info in images.items():
        img_name = img_info["name"]
        # 在原始图片目录中查找
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            # 尝试递归搜索
            for root, dirs, files in os.walk(images_dir):
                if img_name in files:
                    img_path = os.path.join(root, img_name)
                    break
        if not os.path.exists(img_path):
            print(f"[警告] 找不到图像: {img_name}，跳过")
            continue

        cam = cameras[img_info["camera_id"]]
        K, W, H = get_camera_intrinsics(cam, device=device)
        viewmat = torch.tensor(img_info["viewmat"], dtype=torch.float32, device=device)

        img = Image.open(img_path).convert("RGB")
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32, device=device) / 255.0

        train_data.append({
            "image": img_tensor,
            "viewmat": viewmat,
            "K": K,
            "width": W,
            "height": H,
        })

    if len(train_data) < 3:
        raise RuntimeError(f"成功读取的图像只有 {len(train_data)} 张，需要至少 3 张")

    print(f"[数据] 成功加载 {len(train_data)} 张训练图像")

    # 初始化高斯模型
    gaussians = GaussianModel(points3d, device=device)
    params = gaussians.get_params()

    # 优化器
    optimizer = torch.optim.Adam([
        {"params": params["means"], "lr": 1.6e-4},
        {"params": params["quats"], "lr": 1e-3},
        {"params": params["scales_log"], "lr": 5e-3},
        {"params": params["opacities_logit"], "lr": 5e-2},
        {"params": params["colors_logit"], "lr": 2.5e-2},
    ], eps=1e-15)

    # 训练循环
    from gsplat import rasterization

    h, w = train_data[0]["height"], train_data[0]["width"]
    print(f"[训练] 分辨率: {w}x{h}, 步数: {steps}")

    progress = tqdm(range(steps), desc="训练 3DGS")

    for step in progress:
        # 随机选择一张训练图像
        idx = np.random.randint(len(train_data))
        data = train_data[idx]

        # 转为 (C, 4, 4) / (C, 3, 3) 格式 (C=1 cameras)
        viewmat = data["viewmat"].unsqueeze(0)  # [1, 4, 4]
        K = data["K"].unsqueeze(0)  # [1, 3, 3]

        # 前向：光栅化
        render_colors, render_alphas, meta = rasterization(
            means=gaussians.means,
            quats=F.normalize(gaussians.quats, dim=-1),
            scales=gaussians.scales.clamp(min=1e-7),
            opacities=gaussians.opacities,
            colors=gaussians.colors,
            viewmats=viewmat,
            Ks=K,
            width=data["width"],
            height=data["height"],
            tile_size=16,
            backgrounds=torch.zeros((3,), device=device),
        )

        # 损失：L1 + SSIM
        gt = data["image"]
        rendered = render_colors[0]  # [H, W, 3]

        l1_loss = F.l1_loss(rendered, gt)
        ssim_loss = 0.2 * (1 - compute_ssim(rendered.permute(2, 0, 1).unsqueeze(0),
                                             gt.permute(2, 0, 1).unsqueeze(0)))
        loss = l1_loss + ssim_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        for p in [params["means"], params["scales_log"], params["quats"]]:
            if p.grad is not None:
                p.grad.data.clamp_(-1.0, 1.0)

        optimizer.step()

        # 进度
        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "pts": gaussians.means.shape[0],
        })

        # 每 1000 步保存检查点
        if step > 0 and step % 5000 == 0:
            ckpt_dir = os.path.join(output_dir, f"checkpoint_{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            # 保存当前状态
            torch.save({
                "step": step,
                "gaussians": gaussians.get_params(),
                "optimizer": optimizer.state_dict(),
                "loss": loss.item(),
            }, os.path.join(ckpt_dir, "model.pt"))

    # 训练完成，导出
    gaussians.save_ply(os.path.join(output_dir, "model.ply"))
    gaussians.save_splat(os.path.join(output_dir, "model.splat"))

    # 场景信息
    scene_info = {
        "num_points": gaussians.means.shape[0],
        "num_cameras": len(train_data),
        "training_steps": steps,
        "loss": float(loss.item()),
    }
    with open(os.path.join(output_dir, "scene.json"), "w") as f:
        json.dump(scene_info, f, indent=2)

    return gaussians


def compute_ssim(img1, img2):
    """简化的 SSIM 计算。"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, 3, stride=1, padding=1)
    mu2 = F.avg_pool2d(img2, 3, stride=1, padding=1)

    sigma1_sq = F.avg_pool2d(img1 ** 2, 3, stride=1, padding=1) - mu1 ** 2
    sigma2_sq = F.avg_pool2d(img2 ** 2, 3, stride=1, padding=1) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2, 3, stride=1, padding=1) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# ═══════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="3DGS 本地训练管线")
    parser.add_argument("images", type=str, help="照片文件夹路径")
    parser.add_argument("--output", "-o", type=str, default="./output",
                        help="输出目录")
    parser.add_argument("--steps", "-s", type=int, default=TRAINING_STEPS,
                        help=f"训练步数 ({TRAINING_STEPS})")
    parser.add_argument("--skip-colmap", action="store_true",
                        help="跳过 COLMAP，使用现有的 sparse_text 目录")
    parser.add_argument("--sparse-text", type=str, default=None,
                        help="已有 COLMAP 文本输出目录")
    args = parser.parse_args()

    images_dir = os.path.abspath(args.images)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(images_dir):
        print(f"错误: 照片文件夹不存在: {images_dir}")

    # Step 1: COLMAP / 跳过
    if args.skip_colmap:
        colmap_text_dir = args.sparse_text
        if colmap_text_dir is None:
            # 在输出目录中查找
            colmap_text_dir = os.path.join(output_dir, "colmap_output", "sparse_text")
        if not os.path.exists(colmap_text_dir):
            print(f"错误: 找不到 COLMAP 结果: {colmap_text_dir}")
            return
        print(f"跳过 COLMAP，使用: {colmap_text_dir}")
    else:
        colmap_out = os.path.join(output_dir, "colmap_output")
        os.makedirs(colmap_out, exist_ok=True)
        colmap_text_dir = run_colmap(images_dir, colmap_out)

    # Step 2: 训练
    train_3dgs(images_dir, colmap_text_dir, output_dir, steps=args.steps)

    print("\n" + "=" * 50)
    print("全部完成！")
    print(f"  模型: {os.path.join(output_dir, 'model.splat')}")
    print(f"  查看器: 打开 viewer.html，拖拽 .splat 文件")
    print("=" * 50)


if __name__ == "__main__":
    main()
