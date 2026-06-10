"""
   SVG

:
  1.  CharRegion 
  2.  PNG (16-bit ) + TIFF ()
  3.  SVG ( + )
"""

import os
import cv2
import numpy as np
from typing import List, Tuple

from PIL import Image

from config import ReproducerConfig
from vectorize import VectorGlyph
from layout import CharRegion


# ============================================================
# 
# ============================================================

def reconstruct_layout(
    char_images: List[np.ndarray],
    char_regions: List[CharRegion],
    canvas_size: tuple,
    background_color: int = 255
) -> np.ndarray:
    """
    

    Args:
        char_images:       ()
        char_regions:     
        canvas_size:       (w, h)
        background_color: 

    Returns:
         (H, W) uint8
    """
    w, h = canvas_size
    canvas = np.full((h, w), background_color, dtype=np.uint8)

    for char_img, region in zip(char_images, char_regions):
        x, y, bw, bh = region.bbox
        if char_img is None or char_img.size == 0:
            continue

        #  ()
        if char_img.shape[:2] != (bh, bw):
            char_img = cv2.resize(char_img, (bw, bh))

        # 
        try:
            canvas[y:y + bh, x:x + bw] = char_img
        except ValueError:
            # 
            h_avail = min(bh, h - y)
            w_avail = min(bw, w - x)
            canvas[y:y + h_avail, x:x + w_avail] = \
                char_img[:h_avail, :w_avail]

    return canvas


# ============================================================
# 
# ============================================================

def export_bitmap(image: np.ndarray,
                  output_path: str,
                  dpi: int = 600):
    """
     (16-bit PNG / TIFF)

    PNG:  
    TIFF: 16-bit 

    Args:
        image:        (H,W) uint8  uint16
        output_path:  (.png  .tiff)
        dpi:         
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_path.endswith(".tiff") or output_path.endswith(".tif"):
        # 16-bit TIFF 
        img_16 = (image.astype(np.uint16) * 257)  # 8-bit  16-bit
        pil_img = Image.fromarray(img_16)
        pil_img.save(output_path, format="TIFF", dpi=(dpi, dpi),
                     compression="tiff_lzw")
        print(f"[] TIFF : {output_path} ({dpi} DPI, 16-bit)")

    elif output_path.endswith(".png"):
        # PNG 
        cv2.imwrite(output_path, image)
        print(f"[] PNG : {output_path} ({dpi} DPI)")

    else:
        cv2.imwrite(output_path, image)
        print(f"[] : {output_path}")


# ============================================================
# SVG 
# ============================================================

def export_svg(svg_content: str, output_path: str):
    """
     SVG 

    Args:
        svg_content:  SVG 
        output_path:  
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

    file_size = os.path.getsize(output_path)
    print(f"[] SVG : {output_path} "
          f"({file_size / 1024:.1f} KB)")


# ============================================================
# 
# ============================================================

def export_all(
    bitmap: np.ndarray,
    svg_content: str,
    cfg: ReproducerConfig
):
    """
     + SVG 

    Args:
        bitmap:      
        svg_content: SVG 
        cfg:         
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    base_name = os.path.splitext(
        os.path.basename(cfg.input_path)
    )[0]

    for fmt in cfg.output_formats:
        if fmt == "png":
            path = os.path.join(cfg.output_dir,
                                f"{base_name}_{cfg.output_dpi}dpi.png")
            export_bitmap(bitmap, path, cfg.output_dpi)

        elif fmt == "tiff":
            path = os.path.join(cfg.output_dir,
                                f"{base_name}_{cfg.output_dpi}dpi.tiff")
            export_bitmap(bitmap, path, cfg.output_dpi)

        elif fmt == "svg":
            path = os.path.join(cfg.output_dir,
                                f"{base_name}_full.svg")
            export_svg(svg_content, path)

    print(f"\n{'='*50}")
    print(f"[]   {cfg.output_dir}")
    print(f"{'='*50}")
