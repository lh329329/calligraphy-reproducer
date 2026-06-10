"""
   

 ():
   0 :   Perlin
   1 :    +  + 
   2 :    ( 85-95%)
   3 : /  
   4 : /   ()

:  (alpha)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

from config import ReproducerConfig
from stroke_seg import StrokeMask


# ============================================================
# 
# ============================================================

@dataclass
class RenderLayers:
    """"""
    paper:          np.ndarray   #  (H, W) uint8
    ink_bleed:      np.ndarray   #  (H, W) uint8
    ink_dense:      np.ndarray   #  (H, W) uint8
    ink_light:      np.ndarray   #  (H, W) uint8
    flying_white:   np.ndarray   #  (H, W) uint8
    alpha_masks:    Dict[str, np.ndarray]  #  alpha 
    canvas_size:    tuple        #  (w, h)


# ============================================================
# 
# ============================================================

def generate_paper_texture(canvas_size: tuple,
                           seed: int = 42) -> np.ndarray:
    """
    

    :  Perlin-like 
      - :  ()
      - : 
      - : 

    Args:
        canvas_size:  (w, h)
        seed:        

    Returns:
        uint8  (H, W)
    """
    np.random.seed(seed)
    w, h = canvas_size

    # 
    def _smooth_noise(size, scale):
        """"""
        sw, sh = max(1, size[0] // scale), max(1, size[1] // scale)
        noise = np.random.rand(sh, sw).astype(np.float32)
        noise = cv2.resize(noise, (size[0], size[1]),
                           interpolation=cv2.INTER_LINEAR)
        return noise

    # 
    layer1 = _smooth_noise((w, h), 1)   * 0.05   # 
    layer2 = _smooth_noise((w, h), 8)   * 0.03   # 
    layer3 = _smooth_noise((w, h), 32)  * 0.02   # 
    layer4 = _smooth_noise((w, h), 128) * 0.01   # 

    texture = layer1 + layer2 + layer3 + layer4
    texture = (texture * 255).astype(np.uint8)

    return texture


# ============================================================
# 
# ============================================================

def simulate_ink_bleed(mask: np.ndarray,
                       radius: int = 3,
                       intensity: float = 0.6) -> np.ndarray:
    """
    

    :
      1.   
      2.   
      3.   

    Args:
        mask:       (H,W) uint8, =255
        radius:     (px)
        intensity:  (0-1)

    Returns:
         (H,W) uint8
    """
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    # : 
    dist = cv2.distanceTransform(
        255 - mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )

    # 
    dist = np.clip(dist, 0, radius)

    # 
    np.random.seed(42)
    noise = np.random.normal(0, 0.12, mask.shape).astype(np.float32)

    #  + 
    bleed = intensity * np.exp(-dist * (1.0 + noise) / (radius * 0.5))
    bleed = np.clip(bleed, 0, 1)

    #  uint8
    bleed = (bleed * 255).astype(np.uint8)

    return bleed


def simulate_ink_bleed_batch(strokes: List[StrokeMask],
                             canvas_size: tuple,
                             cfg: ReproducerConfig) -> np.ndarray:
    """
    

    Args:
        strokes:     
        canvas_size: 
        cfg:         

    Returns:
         (H, W) uint8
    """
    w, h = canvas_size
    canvas = np.zeros((h, w), dtype=np.float32)

    for s in strokes:
        if s.mask.sum() == 0:
            continue
        bleed = simulate_ink_bleed(
            s.mask, cfg.bleed_radius, cfg.bleed_intensity
        )
        bleed_f = bleed.astype(np.float32) / 255.0
        #  ()
        canvas = np.maximum(canvas, bleed_f)

    canvas = np.clip(canvas, 0, 1)
    return (canvas * 255).astype(np.uint8)


# ============================================================
#  (K-means)
# ============================================================

def extract_ink_levels(gray: np.ndarray,
                       mask: Optional[np.ndarray] = None,
                       n_levels: int = 5) -> List[np.ndarray]:
    """
     K-means  N 

    Args:
        gray:     
        mask:      ()
        n_levels: 

    Returns:
        [mask_0, mask_1, ...] , 
    """
    if mask is None:
        mask = np.ones_like(gray, dtype=np.uint8) * 255

    # 
    pixels = gray[mask > 0].reshape(-1, 1).astype(np.float32)

    if len(pixels) < n_levels:
        return [mask]

    # K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, n_levels, None, criteria, 10,
        cv2.KMEANS_PP_CENTERS
    )

    #  ()
    order = np.argsort(centers.ravel())

    # 
    level_masks = []
    full_labels = np.zeros_like(gray, dtype=np.int32)
    full_labels[mask > 0] = labels.ravel()

    for i in range(n_levels):
        level_idx = order[i]
        level_mask = np.zeros_like(gray, dtype=np.uint8)
        level_mask[full_labels == level_idx] = 255
        level_masks.append(level_mask)

    return level_masks


# ============================================================
# 
# ============================================================

def extract_flying_white(mask: np.ndarray,
                         gray: np.ndarray) -> np.ndarray:
    """
    

    :
      - /
      -    = 
      - 

    Args:
        mask: 
        gray: 

    Returns:
         (H,W) uint8, 255=
    """
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    # 1. 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, -1)

    # 2.  =  - 
    holes = cv2.subtract(filled, mask)

    # 3.   
    gray_f = gray.astype(np.float32)
    local_mean = cv2.GaussianBlur(gray_f, (7, 7), 0)
    local_sq = cv2.GaussianBlur(gray_f * gray_f, (7, 7), 0)
    local_var = local_sq - local_mean * local_mean
    local_var = np.clip(local_var, 0, 255)

    #  + 
    var_mask = (local_var > np.percentile(local_var[mask > 0], 85)
                if mask.sum() > 0 else np.zeros_like(local_var))
    var_mask = var_mask.astype(np.uint8) * 255
    var_mask[mask == 0] = 0

    # 4.  + 
    fw = cv2.bitwise_or(holes, var_mask)

    return fw


# ============================================================
# /
# ============================================================

def extract_seals(bgr: np.ndarray) -> np.ndarray:
    """
    

     HSV 

    Args:
        bgr: BGR 

    Returns:
         (H,W) uint8 RGB
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    #  HSV  (0180)
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([156, 40, 40])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 
    seal = cv2.bitwise_and(bgr, bgr, mask=mask)

    return seal


# ============================================================
# 
# ============================================================

def render_layers(
    strokes: List[StrokeMask],
    canvas_size: tuple,
    cfg: ReproducerConfig,
    bgr_image: Optional[np.ndarray] = None,
) -> RenderLayers:
    """
    

    Args:
        strokes:     
        canvas_size: 
        cfg:         
        bgr_image:    ()

    Returns:
        RenderLayers 
    """
    w, h = canvas_size

    print(f"[] : {len(strokes)} ")

    #   0 :  
    paper_texture = generate_paper_texture(canvas_size,
                                           cfg.paper_texture_seed)
    paper = np.full((h, w), cfg.paper_base_color, dtype=np.uint8)
    paper = cv2.add(paper, paper_texture.reshape(h, w))
    paper = np.clip(paper, 235, 255).astype(np.uint8)

    print("  [] ")

    #   
    all_mask = np.zeros((h, w), dtype=np.uint8)
    dense_mask = np.zeros((h, w), dtype=np.uint8)
    fw_mask = np.zeros((h, w), dtype=np.uint8)

    for s in strokes:
        if s.mask.sum() == 0:
            continue
        bx, by, bw, bh = s.bbox
        #  mask 
        # : stroke.mask 
        # bbox 
        # 
        if bw > 0 and bh > 0:
            try:
                all_mask[by:by + bh, bx:bx + bw] = np.maximum(
                    all_mask[by:by + bh, bx:bx + bw],
                    s.mask
                )
                if s.is_flying_white:
                    fw_mask[by:by + bh, bx:bx + bw] = np.maximum(
                        fw_mask[by:by + bh, bx:bx + bw],
                        s.mask
                    )
                else:
                    dense_mask[by:by + bh, bx:bx + bw] = np.maximum(
                        dense_mask[by:by + bh, bx:bx + bw],
                        s.mask
                    )
            except ValueError:
                continue

    print("  [] ")

    #   1 :  
    ink_bleed = simulate_ink_bleed_batch(strokes, canvas_size, cfg)
    print(f"  [] : radius={cfg.bleed_radius}")

    #   2 :  
    ink_dense = dense_mask.copy()
    print("  [] ")

    #   3 :  
    flying_white = fw_mask.copy()
    print("  [] ")

    #   4 :  
    seals = np.zeros((h, w, 3), dtype=np.uint8)
    if bgr_image is not None:
        seals = extract_seals(bgr_image)
        print("  [] ")

    # alpha 
    alpha_masks = {
        "paper": np.ones((h, w), dtype=np.float32),
        "ink_bleed": (ink_bleed.astype(np.float32) / 255.0) * 0.3,
        "ink_dense": (ink_dense.astype(np.float32) / 255.0) * 0.95,
        "flying_white": (flying_white.astype(np.float32) / 255.0) * 0.7,
        "seals": (np.any(seals > 0, axis=-1).astype(np.float32)),
    }

    print("[] ")

    return RenderLayers(
        paper=paper,
        ink_bleed=ink_bleed,
        ink_dense=ink_dense,
        ink_light=np.zeros_like(ink_dense),
        flying_white=flying_white,
        alpha_masks=alpha_masks,
        canvas_size=canvas_size,
    )


# ============================================================
# 
# ============================================================

def composite_layers(layers: RenderLayers) -> np.ndarray:
    """
    :         

     alpha :
      result = result * (1 - alpha) + layer * alpha

    Args:
        layers: RenderLayers 

    Returns:
         (H, W) uint8
    """
    # canvas_size stored as (w, h) — use directly from layers
    h, w = layers.paper.shape[:2]

    # Start with paper base
    result = layers.paper.astype(np.float32)

    # Layer stack: paper -> bleed -> light -> dense -> flying_white
    stack = [
        ("ink_bleed", layers.ink_bleed),
        ("ink_light", layers.ink_light),
        ("ink_dense", layers.ink_dense),
        ("flying_white", layers.flying_white),
    ]

    for name, layer_data in stack:
        alpha = layers.alpha_masks.get(name,
                                        np.zeros((h, w), dtype=np.float32))

        # Ensure matching shapes
        if layer_data.shape != result.shape:
            layer_data = cv2.resize(layer_data, (w, h))
        if alpha.shape != result.shape:
            alpha = cv2.resize(alpha, (w, h))

        layer_f = layer_data.astype(np.float32)
        # Alpha blend: result = result * (1 - alpha) + layer * alpha
        result = result * (1.0 - alpha) + layer_f * alpha

    result = np.clip(result, 0, 255).astype(np.uint8)

    print("[] ")
    return result
