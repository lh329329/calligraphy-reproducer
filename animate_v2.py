#!/usr/bin/env python3
"""
书法书写过程动画生成 — 骨架化 + 轨迹生长

从一幅完整书法作品图片，还原人类从头到尾一笔一划的书写过程。

技术路线:
  二值化 → 骨架化 → 分叉点拆分笔画 → 轨迹追踪 → 逐笔生长动画 → MP4

用法:
  python animate_v2.py -i calligraphy.png
  python animate_v2.py -i calligraphy.png --fps 60 -o writing.mp4
  python animate_v2.py -i photo.jpg --bg light --paper white
"""

import cv2
import numpy as np
import argparse
import os
import sys
import subprocess
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Optional
from skimage.morphology import skeletonize


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Stroke:
    index: int
    mask: np.ndarray
    path: List[Tuple[int, int]]
    centroid: Tuple[int, int]


# ============================================================
# 1. 预处理
# ============================================================

def detect_background(gray: np.ndarray) -> str:
    """
    采样图像判断背景色: dark(黑底白字) / light(白纸黑字)

    策略: 同时采样四角+中心区域。若四角和中心不一致(如截图有UI框),
    以中心区域为准(中心才是真正的书法内容区)。
    """
    h, w = gray.shape
    sz = min(30, h // 8, w // 8)
    corners = [gray[0:sz, 0:sz], gray[0:sz, -sz:],
               gray[-sz:, 0:sz], gray[-sz:, -sz:]]
    corner_mean = float(np.mean(corners))

    # 中心区域 (书法内容所在)
    ch, cw = h // 3, w // 3
    center = gray[ch:2*ch, cw:2*cw]
    center_mean = float(center.mean())

    # 以中心为准 (截图类图像的边角可能有UI)
    result = "dark" if center_mean < 100 else "light"
    print(f"[背景检测] 四角={corner_mean:.0f} 中心={center_mean:.0f} → {result}")
    return result


def load_and_binarize(path: str, bg_mode: str = "auto",
                      denoise: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载 + CLAHE + 去噪 + Otsu 二值化 → (binary: ink=255, gray, bgr)"""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"无法读取图像: {path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"[预处理] 尺寸: {w}x{h}")

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 去噪
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, h=8,
                                         templateWindowSize=7, searchWindowSize=21)

    # 背景检测
    bg_type = detect_background(gray) if bg_mode == "auto" else bg_mode

    # Otsu: 亮→255, 暗→0
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 确保 ink=255, bg=0
    if bg_type == "light":
        binary = 255 - binary  # 白纸黑字: 墨迹=暗, 需反转
    # dark: 黑底白字, 文字=亮, Otsu 已正确

    ink_pct = (binary > 0).sum() / binary.size * 100
    print(f"[预处理] 墨迹: {ink_pct:.1f}% (背景: {bg_type})")

    # 形态学清理
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    return binary, gray, bgr


# ============================================================
# 2. 骨架化
# ============================================================

def compute_skeleton(binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """骨架化 + 端点/分叉点检测"""
    skel = skeletonize(binary > 0)

    kn = np.ones((3, 3), dtype=np.uint8); kn[1, 1] = 0
    nb = cv2.filter2D(skel.astype(np.uint8), -1, kn)
    endpoints = skel & (nb == 1)
    junctions = skel & (nb >= 3)

    print(f"[骨架化] 像素={skel.sum()}, 端点={endpoints.sum()}, 分叉点={junctions.sum()}")
    return skel, endpoints, junctions


# ============================================================
# 3. 笔画拆分
# ============================================================

def decompose_strokes(skeleton: np.ndarray, junctions: np.ndarray,
                      min_area: int = 6) -> List[np.ndarray]:
    """在分叉点打断骨架，提取独立笔画段"""
    split = skeleton.copy()
    split[junctions] = False

    nl, labels, stats, cents = cv2.connectedComponentsWithStats(
        split.astype(np.uint8), connectivity=8)

    masks = []
    for i in range(1, nl):
        a = stats[i, cv2.CC_STAT_AREA]
        if a >= min_area:
            masks.append({
                'mask': labels == i,
                'cy': int(cents[i][1]), 'cx': int(cents[i][0]),
                'area': a,
            })

    # 按书写顺序: 从上到下，从左到右
    masks.sort(key=lambda m: (m['cy'] // 20, m['cx']))
    print(f"[笔画拆分] {len(masks)} 段 (min_area={min_area})")
    return [m['mask'] for m in masks]


# ============================================================
# 4. 轨迹追踪
# ============================================================

def trace_path(mask: np.ndarray) -> List[Tuple[int, int]]:
    """从骨架 mask 贪心追踪有序像素路径 [(x,y), ...]"""
    pts = np.argwhere(mask)
    if len(pts) < 2:
        return [(int(p[1]), int(p[0])) for p in pts]

    pts_set = set((int(p[1]), int(p[0])) for p in pts)

    # 找端点作起点
    start = None
    for x, y in pts_set:
        n = sum(1 for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if (dx or dy) and (x + dx, y + dy) in pts_set)
        if n == 1:
            start = (x, y); break
    if start is None:
        start = (int(pts[0][1]), int(pts[0][0]))

    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    path = [start]; visited = {start}; cur = start

    while len(visited) < len(pts_set):
        x, y = cur; ok = False
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if (nx, ny) in pts_set and (nx, ny) not in visited:
                path.append((nx, ny)); visited.add((nx, ny))
                cur = (nx, ny); ok = True; break
        if not ok:
            break
    return path


def build_strokes(masks: List[np.ndarray], min_path_len: int = 8) -> List[Stroke]:
    """追踪每条路径，构建 Stroke 列表（过滤过短路径）"""
    strokes = []
    for i, mask in enumerate(masks):
        path = trace_path(mask)
        if len(path) < min_path_len:
            continue
        ys, xs = np.where(mask)
        strokes.append(Stroke(
            index=i, mask=mask, path=path,
            centroid=(int(xs.mean()), int(ys.mean())),
        ))
    for i, s in enumerate(strokes):
        s.index = i
    print(f"[轨迹追踪] {len(strokes)} 条有效轨迹 (min_path={min_path_len})")
    return strokes


# ============================================================
# 5. 书写动画渲染
# ============================================================

def render_animation(
    strokes: List[Stroke],
    binary: np.ndarray,
    gray: np.ndarray,
    canvas_size: Tuple[int, int],
    fps: int = 60,
    steps_per_stroke: int = 60,
    paper_color: Tuple[int, int, int] = (248, 244, 235),
    ink_color: Tuple[int, int, int] = (28, 22, 18),
) -> List[np.ndarray]:
    """
    渲染逐笔书写动画

    每笔: 沿骨架轨迹逐步展开墨迹，模拟"生长"效果
    """
    w, h = canvas_size
    width_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # 墨色密度: 从灰度图提取
    # 黑底图: 亮像素=墨；白纸图: 暗像素=墨
    if binary.mean() < gray.mean():
        ink_density = gray.astype(np.float32) / 255.0
    else:
        ink_density = 1.0 - gray.astype(np.float32) / 255.0
    ink_density = np.clip(ink_density, 0, 1)

    paper = np.full((h, w, 3), paper_color, dtype=np.float32)
    ink_layer = np.full((h, w, 3), ink_color, dtype=np.float32)

    frames = []
    global_mask = np.zeros((h, w), dtype=np.float32)

    # 白纸 intro
    for _ in range(fps // 2):
        frames.append(paper.copy().astype(np.uint8))

    total = len(strokes)
    for si, s in enumerate(strokes):
        path = s.path
        if len(path) < 2:
            continue

        n = min(steps_per_stroke, len(path))
        indices = np.linspace(0, len(path) - 1, n, dtype=int)
        stroke_mask = np.zeros((h, w), dtype=np.float32)
        prev = None

        for step_i, idx in enumerate(indices):
            px, py = path[idx]
            radius = max(2.0, width_map[py, px] * 0.8)
            cv2.circle(stroke_mask, (px, py), int(radius), 1.0, -1)

            # 每 2 步出一帧
            if step_i % 2 == 0 and step_i > 0:
                combined = np.maximum(global_mask, stroke_mask)
                alpha = (combined * ink_density).clip(0, 1)
                alpha_3 = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
                frame = paper * (1 - alpha_3) + ink_layer * alpha_3
                frame = np.clip(frame, 0, 255).astype(np.uint8)

                if prev is None or not np.array_equal(frame, prev):
                    frames.append(frame)
                    prev = frame

        # 收笔
        global_mask = np.maximum(global_mask, stroke_mask)
        alpha = (global_mask * ink_density).clip(0, 1)
        alpha_3 = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
        final_frame = paper * (1 - alpha_3) + ink_layer * alpha_3
        final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)

        for _ in range(4):
            frames.append(final_frame)

        if (si + 1) % 80 == 0:
            print(f"  [{si+1}/{total}] {len(frames)} 帧")

    # 成品展示 2 秒
    for _ in range(fps * 2):
        frames.append(final_frame if final_frame is not None else frames[-1])

    return frames


# ============================================================
# 6. 视频编码 (ffmpeg, 高画质)
# ============================================================

def _find_ffmpeg() -> Optional[str]:
    """查找 ffmpeg 路径"""
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    for p in ["C:/ffmpeg/bin/ffmpeg.exe",
              "C:/Program Files/ffmpeg/bin/ffmpeg.exe",
              "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if os.path.exists(p):
            return p
    return None


def write_video_ffmpeg(frames: List[np.ndarray], output_path: str,
                       fps: int = 60, crf: int = 12) -> bool:
    """
    通过 ffmpeg 管道写入 H.264 MP4

    crf: 质量参数，越小越好 (0=无损, 12=极高, 18=高, 23=默认)
    """
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return False

    h, w = frames[0].shape[:2]
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "medium",
        "-crf", str(crf), "-pix_fmt", "yuv420p",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    ret = proc.wait()
    return ret == 0


def write_video_opencv(frames: List[np.ndarray], output_path: str,
                       fps: int = 60) -> bool:
    """OpenCV 回退方案"""
    h, w = frames[0].shape[:2]
    # 尝试多个编码器
    for codec in ["avc1", "H264", "mp4v", "X264"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if writer.isOpened():
            for f in frames:
                writer.write(f)
            writer.release()
            print(f"[编码] codec={codec}, 大小={os.path.getsize(output_path)/1024:.0f}KB")
            return True
        writer.release()
    return False


def write_video(frames: List[np.ndarray], output_path: str, fps: int = 60):
    """写入 MP4 视频 (优先 ffmpeg 高画质，回退 OpenCV)"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # 确保输出为 .mp4
    if not output_path.endswith(".mp4"):
        output_path += ".mp4"

    if write_video_ffmpeg(frames, output_path, fps):
        sz = os.path.getsize(output_path)
        dur = len(frames) / fps
        print(f"[视频] {output_path}")
        print(f"[视频] {len(frames)}帧 {dur:.1f}s @{fps}fps, "
              f"{sz/1024:.0f}KB, 码率={sz*8/dur/1000:.0f}kbps, ffmpeg H.264")
        return

    if write_video_opencv(frames, output_path, fps):
        return

    # 最终回退: 保存 PNG 帧序列
    out_dir = output_path.replace(".mp4", "_frames")
    os.makedirs(out_dir, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(f"{out_dir}/frame_{i:05d}.png", f)
    print(f"[视频] PNG 序列 → {out_dir}")


# ============================================================
# 7. 流水线
# ============================================================

PAPER_PRESETS = {
    "cream": (248, 244, 235),
    "white": (255, 255, 255),
    "rice":  (250, 245, 230),
}
INK_PRESETS = {
    "cream": (28, 22, 18),
    "white": (20, 18, 15),
    "rice":  (30, 25, 20),
}


def run_pipeline(input_path: str, output_path: str = "output/writing.mp4",
                 fps: int = 60, bg_mode: str = "auto",
                 paper: str = "cream", steps: int = 60):
    """执行完整流水线"""
    print("=" * 55)
    print("  书法书写过程动画 v2")
    print(f"  输入: {input_path}")
    print(f"  输出: {output_path}")
    print(f"  帧率: {fps}fps  背景: {bg_mode}  纸: {paper}")
    print("=" * 55)

    # 1. 预处理
    binary, gray, _bgr = load_and_binarize(input_path, bg_mode=bg_mode)
    h, w = binary.shape[:2]

    # 2. 骨架化
    skeleton, _ep, junctions = compute_skeleton(binary)

    # 3. 笔画拆分
    masks = decompose_strokes(skeleton, junctions)

    # 4. 轨迹追踪
    strokes = build_strokes(masks)
    if not strokes:
        print("[错误] 未检测到笔画")
        return

    # 5. 渲染
    print(f"[渲染] {len(strokes)} 笔, {fps}fps ...")
    frames = render_animation(
        strokes, binary, gray, (w, h), fps=fps,
        steps_per_stroke=steps,
        paper_color=PAPER_PRESETS.get(paper, PAPER_PRESETS["cream"]),
        ink_color=INK_PRESETS.get(paper, INK_PRESETS["cream"]),
    )
    dur = len(frames) / fps
    print(f"[渲染] {len(frames)} 帧, {dur:.1f}s")

    # 6. 编码
    write_video(frames, output_path, fps)

    print(f"\n  笔画: {len(strokes)}  帧: {len(frames)}  时长: {dur:.1f}s")
    print(f"  输出: {os.path.abspath(output_path)}")
    print("=" * 55)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="书法书写过程动画 — 成品图还原逐笔书写过程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python animate_v2.py -i calligraphy.png
  python animate_v2.py -i photo.jpg --bg light --paper white
  python animate_v2.py -i noche.png --bg dark --fps 30
        """)
    parser.add_argument("-i", "--input", required=True, help="输入图像路径")
    parser.add_argument("-o", "--output", default="output/writing.mp4", help="输出视频路径")
    parser.add_argument("--fps", type=int, default=60, help="帧率 (默认 60)")
    parser.add_argument("--bg", choices=["auto", "dark", "light"], default="auto",
                        help="背景模式 (默认 auto: 四角采样自动判断)")
    parser.add_argument("--paper", choices=["cream", "white", "rice"], default="cream",
                        help="纸张色调 (默认 cream)")
    parser.add_argument("--steps", type=int, default=60, help="每笔步数 (默认 60)")

    args = parser.parse_args()
    if not os.path.exists(args.input):
        print(f"[错误] 文件不存在: {args.input}")
        sys.exit(1)

    run_pipeline(args.input, args.output, args.fps, args.bg, args.paper, args.steps)


if __name__ == "__main__":
    main()
