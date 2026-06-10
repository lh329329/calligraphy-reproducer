"""
   

:
  1. 
  2. 
  3.  (, /)
  4. 

: list[CharRegion]
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from config import ReproducerConfig


# ============================================================
# 
# ============================================================

@dataclass
class CharRegion:
    """"""
    char_id:    int            #  (0..N-1)
    bbox:       tuple          # (x, y, w, h) 
    image:      np.ndarray     # 
    centroid:   tuple          # (cx, cy)  ()


# ============================================================
# 
# ============================================================

def segment_layout(binary: np.ndarray,
                   gray: np.ndarray,
                   cfg: ReproducerConfig) -> List[CharRegion]:
    """
    

    1.   
    2.   
    3.   
    4.  +  CharRegion 

    Args:
        binary:  (=255, =0)
        gray:    ()
        cfg:    

    Returns:
        list[CharRegion]: 
    """
    #  (255)
    work = 255 - binary if binary.mean() > 128 else binary.copy()

    #  1. :  
    h_proj = np.sum(work, axis=1) / 255  # 
    row_bounds = _find_projection_ranges(h_proj, min_width=cfg.min_char_area // 4)

    if not row_bounds:
        print("[] ")
        return _segment_by_connected_components(binary, gray, cfg)

    print(f"[]  {len(row_bounds)} ")

    #  2. :  
    char_id = 0
    regions = []

    for row_idx, (r_start, r_end) in enumerate(row_bounds):
        row_slice = work[r_start:r_end, :]
        v_proj = np.sum(row_slice, axis=0) / 255
        col_bounds = _find_projection_ranges(v_proj, min_width=cfg.min_char_area // 8)

        for col_idx, (c_start, c_end) in enumerate(col_bounds):
            # 
            w = c_end - c_start
            h = r_end - r_start
            if w * h < cfg.min_char_area:
                continue

            #  margin
            x1 = max(0, c_start - cfg.char_margin)
            y1 = max(0, r_start - cfg.char_margin)
            x2 = min(gray.shape[1], c_end + cfg.char_margin)
            y2 = min(gray.shape[0], r_end + cfg.char_margin)

            # 
            char_img = gray[y1:y2, x1:x2]

            # 
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            region = CharRegion(
                char_id=char_id,
                bbox=(x1, y1, x2 - x1, y2 - y1),
                image=char_img,
                centroid=(cx, cy)
            )
            regions.append(region)
            char_id += 1

    print(f"[]  {len(regions)} ")
    return regions


def _find_projection_ranges(proj: np.ndarray,
                            min_width: int = 5) -> List[tuple]:
    """
    

    Args:
        proj:      
        min_width:  ()

    Returns:
        [(start, end), ...] 
    """
    mask = proj > 0
    ranges = []
    in_range = False
    start = 0

    for i, val in enumerate(mask):
        if val and not in_range:
            start = i
            in_range = True
        elif not val and in_range:
            if i - start >= min_width:
                ranges.append((start, i))
            in_range = False

    if in_range and len(mask) - start >= min_width:
        ranges.append((start, len(mask)))

    return ranges


# ============================================================
# : 
# ============================================================

def _segment_by_connected_components(binary: np.ndarray,
                                     gray: np.ndarray,
                                     cfg: ReproducerConfig) -> List[CharRegion]:
    """
      

    :
      - / ()
      - 
    """
    work = 255 - binary if binary.mean() > 128 else binary.copy()

    #   
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)

    # 
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )

    regions = []
    char_id = 0

    #  (label 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < cfg.min_char_area:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # 
        x1 = max(0, x - cfg.char_margin)
        y1 = max(0, y - cfg.char_margin)
        x2 = min(gray.shape[1], x + w + cfg.char_margin)
        y2 = min(gray.shape[0], y + h + cfg.char_margin)

        cx, cy = centroids[i]

        region = CharRegion(
            char_id=char_id,
            bbox=(x1, y1, x2 - x1, y2 - y1),
            image=gray[y1:y2, x1:x2],
            centroid=(int(cx), int(cy))
        )
        regions.append(region)
        char_id += 1

    # 
    regions.sort(key=lambda r: (r.centroid[1], r.centroid[0]))

    # 
    for i, r in enumerate(regions):
        r.char_id = i

    print(f"[] : {len(regions)} ")
    return regions


def extract_char_images(gray: np.ndarray,
                        regions: List[CharRegion]) -> List[CharRegion]:
    """
     ( region.image )

    Args:
        gray:    
        regions: CharRegion 

    Returns:
         image  CharRegion 
    """
    for r in regions:
        x, y, w, h = r.bbox
        r.image = gray[y:y + h, x:x + w]
    return regions
