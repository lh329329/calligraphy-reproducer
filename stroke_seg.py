"""
    SAM 

:
  1.  SAM vit_h 
  2. :   SAM predict  NMS   
  3. 
  4.  (ThreadPoolExecutor)

SAM  ():
  - :   
  - :   
  - :   
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from skimage.morphology import skeletonize

from config import ReproducerConfig
from layout import CharRegion


# ============================================================
# 
# ============================================================

@dataclass
class StrokeMask:
    """"""
    stroke_id:        int          #  ()
    char_id:          int          # 
    mask:             np.ndarray   #  (H, W) uint8 {0, 255}
    bbox:             tuple        #  (x, y, w, h)
    ink_density:      float        #  0.0 ~ 1.0
    is_flying_white:  bool         # 
    score:            float        # SAM 


# ============================================================
# SAM 
# ============================================================

def load_sam_model(cfg: ReproducerConfig):
    """
     SAM  GPU  SamPredictor

    Args:
        cfg:  ()

    Returns:
        SamPredictor 

    Raises:
        ImportError:   segment-anything 
        FileNotFoundError: 
    """
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        raise ImportError(
            " segment-anything: pip install segment-anything"
        )

    import os
    if not os.path.exists(cfg.sam_checkpoint):
        raise FileNotFoundError(
            f"SAM : {cfg.sam_checkpoint}\n"
            f" models/ :\n"
            f"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        )

    print(f"[]  SAM {cfg.sam_model_type} ...")

    sam = sam_model_registry[cfg.sam_model_type](
        checkpoint=cfg.sam_checkpoint
    )
    sam.to(device=cfg.sam_device)
    predictor = SamPredictor(sam)

    print(f"[] SAM {cfg.sam_model_type}  (device={cfg.sam_device})")
    return predictor


# ============================================================
# 
# ============================================================

def generate_prompts(binary: np.ndarray,
                     max_prompts: int = 30) -> tuple:
    """
     SAM  ()

    :
      1.   
      2.   
      3.     

    Args:
        binary:       (=255, =0)
        max_prompts: 

    Returns:
        (coords, labels):  (N,2),  (N,) 1=positive 0=negative
    """
    h, w = binary.shape
    coords_list = []
    labels_list = []

    #  :  
    skel = skeletonize(binary > 0)
    fork_pts = _find_branch_points(skel)

    for pt in fork_pts[:max_prompts // 3]:
        coords_list.append(pt)
        labels_list.append(1)  # positive

    #  :  
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    for i in range(1, num_labels):  # 
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 30:  # 
            continue
        cx, cy = centroids[i]
        coords_list.append([cx, cy])
        labels_list.append(1)  # positive

    #  :  
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)
    bg_mask = (dilated == 0)

    bg_coords = np.argwhere(bg_mask)
    if len(bg_coords) > 0:
        n_neg = min(max_prompts // 4, len(bg_coords))
        idxs = np.random.choice(len(bg_coords), n_neg, replace=False)
        for idx in idxs:
            y, x = bg_coords[idx]
            coords_list.append([x, y])
            labels_list.append(0)  # negative

    #   
    if len(coords_list) > max_prompts:
        coords_list = coords_list[:max_prompts]
        labels_list = labels_list[:max_prompts]

    if len(coords_list) == 0:
        # : 
        coords_list.append([w // 2, h // 2])
        labels_list.append(1)

    coords = np.array(coords_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)

    return coords, labels


def _find_branch_points(skeleton: np.ndarray) -> List[list]:
    """
     ( >= 3)

    Args:
        skeleton: 

    Returns:
        [(x, y), ...] 
    """
    # 3x3   
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)

    # :  >= 4 (+3)
    fork_mask = (neighbors >= 4) & skeleton
    fork_coords = np.argwhere(fork_mask)

    #  (x, y) 
    pts = [[int(c[1]), int(c[0])] for c in fork_coords]
    return pts


# ============================================================
# NMS 
# ============================================================

def nms_strokes(masks: List[StrokeMask],
                iou_threshold: float = 0.6) -> List[StrokeMask]:
    """
      SAM 

    Args:
        masks:        
        iou_threshold: IoU 

    Returns:
        
    """
    if len(masks) <= 1:
        return masks

    # 
    sorted_indices = np.argsort([-m.score for m in masks])
    keep = []

    for idx in sorted_indices:
        overlaps = False
        for k in keep:
            iou = _compute_mask_iou(masks[idx].mask, masks[k].mask)
            if iou > iou_threshold:
                overlaps = True
                break

        if not overlaps:
            keep.append(idx)

    result = [masks[i] for i in sorted(keep)]
    return result


def _compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """ IoU"""
    a = mask_a > 0
    b = mask_b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


# ============================================================
# 
# ============================================================

def order_strokes(masks: List[StrokeMask]) -> List[StrokeMask]:
    """
    :   

     y  1/10    x 
    """
    def sort_key(m: StrokeMask):
        x, y, _, _ = m.bbox
        return (y // 10, x)

    sorted_masks = sorted(masks, key=sort_key)

    # 
    for i, m in enumerate(sorted_masks):
        m.stroke_id = i

    return sorted_masks


# ============================================================
# 
# ============================================================

def analyze_stroke(mask: np.ndarray, gray: np.ndarray) -> tuple:
    """
    :  + 

    :  ( > 10%)

    Args:
        mask: 
        gray: 

    Returns:
        (ink_density, is_flying_white)
    """
    if mask.sum() == 0:
        return 0.0, False

    # : 
    gray_region = gray[mask > 0]
    ink_density = 1.0 - float(np.mean(gray_region)) / 255.0

    # : 
    # 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, -1)

    hole_area = filled.sum() - mask.sum()
    total_area = filled.sum()
    hole_ratio = hole_area / total_area if total_area > 0 else 0

    is_flying_white = hole_ratio > 0.08

    return ink_density, is_flying_white


# ============================================================
# 
# ============================================================

def segment_strokes_single_char(
    predictor,
    char_region: CharRegion,
    cfg: ReproducerConfig,
    gray_char: Optional[np.ndarray] = None
) -> List[StrokeMask]:
    """
     SAM 

    :
      1.  (resize + pad  SAM )
      2. 
      3. SAM predict  
      4.    StrokeMask
      5. NMS 
      6. 
      7. 

    Args:
        predictor:   SAM SamPredictor 
        char_region: 
        cfg:         
        gray_char:    (,  char_region.image)

    Returns:
        list[StrokeMask]: 
    """
    image = gray_char if gray_char is not None else char_region.image
    if image is None or image.size == 0:
        return []

    h, w = image.shape

    #  (SAM )
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image

    # SAM set_image
    predictor.set_image(image_rgb)

    # 
    _, binary = cv2.threshold(image, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = 255 - binary  # : =255

    # 
    coords, labels = generate_prompts(binary)

    if len(coords) == 0:
        return []

    # SAM 
    masks, scores, _ = predictor.predict(
        point_coords=coords,
        point_labels=labels,
        multimask_output=True,  #  3 
        mask_input=None,
    )

    # 
    stroke_masks = []
    for i in range(masks.shape[0]):
        score = float(scores[i]) if i < len(scores) else 0.0
        if score < 0.3:
            continue

        m = masks[i].astype(np.uint8) * 255
        area = m.sum() / 255

        # 
        if area < 30 or area > w * h * 0.95:
            continue

        # 
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        # 
        ink_density, is_fw = analyze_stroke(m, image)

        sm = StrokeMask(
            stroke_id=i,
            char_id=char_region.char_id,
            mask=m,
            bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
            ink_density=ink_density,
            is_flying_white=is_fw,
            score=score,
        )
        stroke_masks.append(sm)

    # NMS 
    stroke_masks = nms_strokes(stroke_masks)

    # 
    stroke_masks = order_strokes(stroke_masks)

    return stroke_masks


# ============================================================
# 
# ============================================================

def refine_stroke_boundaries(mask: np.ndarray,
                             gray: np.ndarray) -> np.ndarray:
    """
    

    Args:
        mask: 
        gray: 

    Returns:
        
    """
    #   
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #   
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)

    return refined


# ============================================================
# 
# ============================================================

def segment_all_chars(
    predictor,
    char_regions: List[CharRegion],
    cfg: ReproducerConfig
) -> List[StrokeMask]:
    """
    

     ThreadPoolExecutor  SAM
    SAM  set_image  predict  GPU Python  CPU-GPU 

    Args:
        predictor:    SAM SamPredictor ()
        char_regions: 
        cfg:          

    Returns:
        list[StrokeMask]:  (car_id )
    """
    all_strokes = []

    print(f"[] : {len(char_regions)} , "
          f"={cfg.parallel_workers}")

    if len(char_regions) == 1:
        # : 
        strokes = segment_strokes_single_char(
            predictor, char_regions[0], cfg
        )
        all_strokes.extend(strokes)
    else:
        # : 
        with ThreadPoolExecutor(max_workers=cfg.parallel_workers) as executor:
            futures = {}
            for cr in char_regions:
                future = executor.submit(
                    segment_strokes_single_char,
                    predictor, cr, cfg
                )
                futures[future] = cr.char_id

            for future in as_completed(futures):
                try:
                    strokes = future.result()
                    all_strokes.extend(strokes)
                except Exception as e:
                    char_id = futures[future]
                    print(f"[]  {char_id} : {e}")

    #  char_id  stroke_id 
    all_strokes.sort(key=lambda s: (s.char_id, s.stroke_id))

    print(f"[] : {len(all_strokes)} ")
    return all_strokes
