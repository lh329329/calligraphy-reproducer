#!/usr/bin/env python3
"""
Calligraphy Writing Animation — 书法书写过程动画生成

从一幅完整书法作品图片，还原人类从头到尾一笔一划的书写过程。

技术路线:
  骨架化(skeletonize) → 分叉点拆分笔画 → 沿着轨迹逐步"生长"墨迹 → MP4 视频

用法:
  python animate_v2.py --input calligraphy.png
  python animate_v2.py --input calligraphy.png --fps 60 --output writing.mp4
  python animate_v2.py --input calligraphy.png --bg dark --paper cream
"""

import cv2
import numpy as np
import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional
from skimage.morphology import skeletonize


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Stroke:
    """单个笔画"""
    index: int                          # 笔画序号（书写顺序）
    mask: np.ndarray                    # 笔画区域 mask (H,W) bool
    path: List[Tuple[int, int]]         # 骨架轨迹坐标 [(x,y), ...]
    centroid: Tuple[int, int]           # 中心点 (cx, cy)


# ============================================================
# 1. 预处理
# ============================================================

def load_and_binarize(
    path: str,
    bg_mode: str = "auto",
    denoise: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载图像并二值化

    Args:
        path:     图像路径
        bg_mode:  背景模式 "auto"|"dark"|"light"
        denoise:  是否去噪

    Returns:
        (binary, gray, bgr): 二值图(ink=255), 灰度图, 原图
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"无法读取图像: {path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"[预处理] 图像尺寸: {w}x{h}")

    # --- CLAHE 增强 ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # --- 去噪 ---
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, h=8,
                                         templateWindowSize=7,
                                         searchWindowSize=21)
        print("[预处理] NLM 去噪完成")

    # --- 二值化 ---
    # Otsu 自动阈值
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 判断是否需要反转: 若 ink 占多数则反转
    ink_ratio = (binary > 0).sum() / binary.size
    if bg_mode == "auto":
        if ink_ratio > 0.5:
            binary = 255 - binary
    elif bg_mode == "dark":
        # 黑底白字: 文字是亮的，反转使 ink=255
        binary = 255 - binary
    # light: 白底黑字，binary 已正确 (ink=255)

    ink_pct = (binary > 0).sum() / binary.size * 100
    print(f"[预处理] 墨迹像素: {ink_pct:.1f}%")

    # --- 形态学清理 ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary, gray, bgr


# ============================================================
# 2. 骨架化
# ============================================================

def compute_skeleton(binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    骨架化 + 检测端点和分叉点

    Returns:
        (skeleton, endpoints, junctions): 三个 bool 数组
    """
    skel = skeletonize(binary > 0)

    # 统计每个骨架像素的 8 邻域数
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbors = cv2.filter2D(skel.astype(np.uint8), -1, kernel)

    endpoints = skel & (neighbors == 1)
    junctions = skel & (neighbors >= 3)

    print(f"[骨架化] 骨架像素: {skel.sum()}, "
          f"端点: {endpoints.sum()}, 分叉点: {junctions.sum()}")

    return skel, endpoints, junctions


# ============================================================
# 3. 笔画拆分
# ============================================================

def decompose_strokes(
    skeleton: np.ndarray,
    junctions: np.ndarray,
    min_area: int = 8
) -> List[np.ndarray]:
    """
    在分叉点处打断骨架，提取独立笔画段

    Args:
        skeleton:  骨架 bool 数组
        junctions: 分叉点 bool 数组
        min_area:  最小笔画面积（像素），过滤杂讯

    Returns:
        [mask, ...] 按位置排序的笔画 mask 列表
    """
    # 移除分叉点 → 骨架断裂成独立段
    split = skeleton.copy()
    split[junctions] = False

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(split.astype(np.uint8), connectivity=8)
    )

    masks = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            masks.append({
                'mask': (labels == i),
                'cy': int(centroids[i][1]),
                'cx': int(centroids[i][0]),
                'area': area,
            })

    # 按书写顺序排序: 从上到下，从左到右
    masks.sort(key=lambda m: (m['cy'] // 20, m['cx']))

    print(f"[笔画拆分] {len(masks)} 段 (min_area={min_area})")

    return [m['mask'] for m in masks]


# ============================================================
# 4. 轨迹追踪
# ============================================================

def trace_path(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    从骨架 mask 追踪得到有序像素路径

    从端点出发，贪心遍历所有骨架像素，返回 (x, y) 列表
    """
    pts = np.argwhere(mask)
    if len(pts) < 2:
        return [(int(p[1]), int(p[0])) for p in pts]

    pts_set = set((int(p[1]), int(p[0])) for p in pts)

    # 找端点作为起点
    start = None
    for x, y in pts_set:
        n = sum(1 for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if (dx or dy) and (x + dx, y + dy) in pts_set)
        if n == 1:
            start = (x, y)
            break
    if start is None:
        start = (int(pts[0][1]), int(pts[0][0]))

    # 贪心遍历
    path = [start]
    visited = {start}
    cur = start
    dirs = [(1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

    while len(visited) < len(pts_set):
        x, y = cur
        found = False
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if (nx, ny) in pts_set and (nx, ny) not in visited:
                path.append((nx, ny))
                visited.add((nx, ny))
                cur = (nx, ny)
                found = True
                break
        if not found:
            break  # 孤立点或环

    return path


def build_strokes(masks: List[np.ndarray]) -> List[Stroke]:
    """为每个 mask 追踪路径，构建 Stroke 对象列表"""
    strokes = []
    for i, mask in enumerate(masks):
        path = trace_path(mask)
        ys, xs = np.where(mask)
        cx, cy = int(xs.mean()), int(ys.mean())
        strokes.append(Stroke(
            index=i,
            mask=mask,
            path=path,
            centroid=(cx, cy),
        ))
    # 过滤路径过短的笔画
    strokes = [s for s in strokes if len(s.path) >= 6]
    for i, s in enumerate(strokes):
        s.index = i
    print(f"[轨迹追踪] {len(strokes)} 条有效轨迹")
    return strokes


# ============================================================
# 5. 书写动画渲染
# ============================================================

def compute_stroke_width_map(binary: np.ndarray) -> np.ndarray:
    """距离变换: 每个墨迹像素到边缘的距离 → 局部笔画宽度"""
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    return dist


def render_animation(
    strokes: List[Stroke],
    original_bgr: np.ndarray,
    binary: np.ndarray,
    gray: np.ndarray,
    canvas_size: Tuple[int, int],
    fps: int = 60,
    steps_per_stroke: int = 50,
    paper_color: Tuple[int, int, int] = (250, 245, 235),
    ink_color: Tuple[int, int, int] = (28, 22, 18),
) -> List[np.ndarray]:
    """
    渲染逐笔书写动画帧

    每笔: 沿着骨架轨迹逐步"生长"，墨迹从起笔处向终点延展

    Args:
        strokes:          排序后的笔画列表
        original_bgr:     原图 (用于提取墨色纹理)
        binary:           二值图
        gray:             灰度图
        canvas_size:      (w, h)
        fps:              帧率
        steps_per_stroke: 每笔的步数（越多越平滑）
        paper_color:      纸张底色 (B,G,R)
        ink_color:        墨色基色 (B,G,R)

    Returns:
        [(H,W,3) uint8] 帧列表
    """
    w, h = canvas_size
    width_map = compute_stroke_width_map(binary)

    # 墨色密度: 从原图灰度提取，亮的像素 = 浓墨
    # 对于白纸黑字: dark pixel = ink. 对于黑底白字: 已做反转处理
    if (binary > 0).sum() / binary.size > 0.5:
        # 黑底图: 亮像素=墨
        ink_density = gray.astype(np.float32) / 255.0
    else:
        # 白纸图: 暗像素=墨
        ink_density = 1.0 - gray.astype(np.float32) / 255.0

    ink_density = np.clip(ink_density, 0, 1)

    # 纸张
    paper = np.full((h, w, 3), paper_color, dtype=np.float32)

    # 墨色层 (B,G,R)
    ink_layer = np.zeros((h, w, 3), dtype=np.float32)
    ink_layer[:, :, 0] = ink_color[0]
    ink_layer[:, :, 1] = ink_color[1]
    ink_layer[:, :, 2] = ink_color[2]

    frames = []
    global_mask = np.zeros((h, w), dtype=np.float32)  # 已揭示的墨迹区域

    # --- 空纸 intro ---
    for _ in range(fps // 2):  # 0.5 秒
        frames.append(paper.copy().astype(np.uint8))

    total = len(strokes)
    for si, s in enumerate(strokes):
        path = s.path
        if len(path) < 2:
            continue

        # 对路径做等距采样
        n = min(steps_per_stroke, len(path))
        indices = np.linspace(0, len(path) - 1, n, dtype=int)

        # 当前笔画的累积 mask
        stroke_mask = np.zeros((h, w), dtype=np.float32)

        prev_frame = None
        for step_i, idx in enumerate(indices):
            px, py = path[idx]
            radius = max(2.0, width_map[py, px] * 0.75)
            cv2.circle(stroke_mask, (px, py), int(radius), 1.0, -1)

            # 控制帧密度: 每 2 步出一帧 → 每笔约 25 帧
            if step_i % 2 == 0 and step_i > 0:
                frame = paper.copy()
                combined = np.maximum(global_mask, stroke_mask)
                alpha = (combined * ink_density).clip(0, 1)
                alpha_3 = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
                frame = frame * (1 - alpha_3) + ink_layer * alpha_3
                frame = np.clip(frame, 0, 255).astype(np.uint8)

                # 去重: 仅在与上一帧不同时加入
                if prev_frame is None or not np.array_equal(frame, prev_frame):
                    frames.append(frame)
                    prev_frame = frame

        # 收笔: 完整揭示该笔画
        global_mask = np.maximum(global_mask, stroke_mask)
        final_frame = paper.copy()
        alpha = (global_mask * ink_density).clip(0, 1)
        alpha_3 = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
        final_frame = final_frame * (1 - alpha_3) + ink_layer * alpha_3
        final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)

        # 收笔停留 (4 帧)
        for _ in range(4):
            frames.append(final_frame)

        if (si + 1) % 60 == 0:
            print(f"  [{si+1}/{total}] {len(frames)} 帧")

    # 成品展示
    for _ in range(fps * 2):  # 2 秒
        frames.append(final_frame if 'final_frame' in dir() else frames[-1])

    return frames


# ============================================================
# 6. 视频编码
# ============================================================

def write_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 60,
    bitrate: str = "8M"
):
    """
    写入 MP4 视频文件

    优先用 avc1 (H.264)，回退到 mp4v
    """
    if not frames:
        print("[错误] 无帧可写")
        return

    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # 尝试 H.264
    codecs = [
        ('avc1', '.mp4'),
        ('H264', '.mp4'),
        ('mp4v', '.mp4'),
    ]

    for codec, ext in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out_path = output_path if output_path.endswith(ext) else output_path + ext
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if writer.isOpened():
            for f in frames:
                writer.write(f)
            writer.release()
            sz = os.path.getsize(out_path) / 1024
            print(f"[视频] {out_path}")
            print(f"[视频] {len(frames)} 帧, "
                  f"{len(frames)/fps:.1f}s @ {fps}fps, "
                  f"{sz:.0f} KB, 编码: {codec}")
            return
        writer.release()

    # 都失败了，写 PNG 序列
    print("[警告] 无法创建视频编码器，保存为 PNG 序列...")
    out_dir = output_path.replace('.mp4', '_frames')
    os.makedirs(out_dir, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(f"{out_dir}/frame_{i:05d}.png", f)
    print(f"[视频] PNG 序列保存至 {out_dir}")


# ============================================================
# 7. 流水线
# ============================================================

def run_pipeline(
    input_path: str,
    output_path: str = "output/writing.mp4",
    fps: int = 60,
    bg_mode: str = "auto",
    paper: str = "cream",
    steps_per_stroke: int = 50,
):
    """执行完整流水线"""
    print("=" * 60)
    print("  书法书写过程动画生成")
    print("=" * 60)
    print(f"  输入: {input_path}")
    print(f"  输出: {output_path}")
    print(f"  帧率: {fps} fps")
    print(f"  背景: {bg_mode}")
    print("=" * 60)

    # 1. 预处理
    binary, gray, bgr = load_and_binarize(input_path, bg_mode=bg_mode)
    h, w = binary.shape[:2]

    # 2. 骨架化
    skeleton, endpoints, junctions = compute_skeleton(binary)

    # 3. 笔画拆分
    masks = decompose_strokes(skeleton, junctions)

    # 4. 轨迹追踪
    strokes = build_strokes(masks)
    if not strokes:
        print("[错误] 未检测到笔画，请检查输入图像")
        return

    # 5. 渲染
    paper_colors = {
        "cream": (250, 245, 235),
        "white": (255, 255, 255),
        "rice": (248, 240, 225),
    }
    ink_colors = {
        "cream": (28, 22, 18),
        "white": (20, 18, 15),
        "rice": (30, 25, 20),
    }

    print(f"[渲染] {len(strokes)} 笔画, {fps}fps ...")
    frames = render_animation(
        strokes, bgr, binary, gray, (w, h),
        fps=fps,
        steps_per_stroke=steps_per_stroke,
        paper_color=paper_colors.get(paper, paper_colors["cream"]),
        ink_color=ink_colors.get(paper, ink_colors["cream"]),
    )
    print(f"[渲染] 共 {len(frames)} 帧, {len(frames)/fps:.1f}s")

    # 6. 编码
    write_video(frames, output_path, fps=fps)

    print()
    print(f"  笔画数: {len(strokes)}")
    print(f"  帧数:   {len(frames)}")
    print(f"  时长:   {len(frames)/fps:.1f}s @ {fps}fps")
    print(f"  输出:   {os.path.abspath(output_path)}")
    print("=" * 60)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="书法书写过程动画生成 — 从成品图还原逐笔书写过程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python animate_v2.py --input calligraphy.png
  python animate_v2.py --input calligraphy.png --fps 30 --output my_writing.mp4
  python animate_v2.py --input photo.jpg --bg light --paper white
        """
    )
    parser.add_argument("--input", "-i", required=True,
                        help="输入图像路径")
    parser.add_argument("--output", "-o", default="output/writing.mp4",
                        help="输出视频路径 (默认: output/writing.mp4)")
    parser.add_argument("--fps", type=int, default=60,
                        help="帧率 (默认: 60)")
    parser.add_argument("--bg", choices=["auto", "dark", "light"],
                        default="auto",
                        help="背景模式 (默认: auto)")
    parser.add_argument("--paper", choices=["cream", "white", "rice"],
                        default="cream",
                        help="纸张色调 (默认: cream)")
    parser.add_argument("--steps", type=int, default=50,
                        help="每笔步数，越大越平滑 (默认: 50)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[错误] 文件不存在: {args.input}")
        sys.exit(1)

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        fps=args.fps,
        bg_mode=args.bg,
        paper=args.paper,
        steps_per_stroke=args.steps,
    )


if __name__ == "__main__":
    main()
