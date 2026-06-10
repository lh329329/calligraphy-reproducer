"""
  I/O
"""

import os
import json
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Optional


# ============================================================
# 
# ============================================================

def ensure_cache_dir(cache_root: str, *subdirs: str) -> str:
    """"""
    path = os.path.join(cache_root, *subdirs)
    os.makedirs(path, exist_ok=True)
    return path


def cache_path(cache_root: str, filename: str) -> str:
    """"""
    return os.path.join(cache_root, filename)


def save_cache_json(obj, filepath: str):
    """ dataclass  JSON """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_cache_json(filepath: str):
    """ JSON """
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache_image(image: np.ndarray, filepath: str):
    """ (16-bit PNG )"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath, image)


def load_cache_image(filepath: str) -> Optional[np.ndarray]:
    """"""
    if not os.path.exists(filepath):
        return None
    return cv2.imread(filepath, cv2.IMREAD_UNCHANGED)


# ============================================================
# 
# ============================================================

class Timer:
    """  """

    def __init__(self, name: str = ""):
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        print(f"[] {self.name} ...")
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"[] {self.name}  {elapsed:.1f}s")


# ============================================================
# 
# ============================================================

def draw_char_bboxes(image: np.ndarray,
                     regions: list,
                     color: tuple = (0, 255, 0)) -> np.ndarray:
    """"""
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for r in regions:
        x, y, w, h = r.bbox if hasattr(r, 'bbox') else r
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        char_id = getattr(r, 'char_id', 0)
        cv2.putText(vis, str(char_id), (x + 2, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return vis


def draw_stroke_masks(image: np.ndarray,
                      strokes: list) -> np.ndarray:
    """"""
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
    ]

    for i, s in enumerate(strokes):
        mask = s.mask if hasattr(s, 'mask') else s
        color = colors[i % len(colors)]
        overlay = np.zeros_like(vis)
        overlay[mask > 0] = color
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

    return vis


# ============================================================
# 
# ============================================================

def auto_adjust_to_odd(n: int) -> int:
    """ ()"""
    return n if n % 2 == 1 else n + 1


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """ IoU ()"""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)
