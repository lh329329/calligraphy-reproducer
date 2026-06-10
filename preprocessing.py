"""
   

:
    load_image  detect_source_type  (scan|photo) 
    deskew  denoise  enhance_edges  adaptive_binarize

scan:    CLAHE  
photo:       CLAHE  
"""

import cv2
import numpy as np
from config import ReproducerConfig


# ============================================================
# 
# ============================================================

def load_image(path: str) -> tuple:
    """
     (, BGR)

    Args:
        path: 

    Returns:
        (gray, bgr):  (H,W) uint8, BGR (H,W,3) uint8
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f": {path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    print(f"[] : {bgr.shape[1]}{bgr.shape[0]} px")
    return gray, bgr


# ============================================================
# 
# ============================================================

def detect_source_type(image: np.ndarray) -> str:
    """
    : 'scan'  'photo'

    :
      - scan:   (V  std/mean < 0.15) + 
      - photo: 

    Args:
        image: BGR 

    Returns:
        'scan' | 'photo'
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)

    # : V 
    uniformity = float(np.std(v) / (np.mean(v) + 1e-6))

    # :  + 
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # /
    h_count, v_count = 0, 0
    if lines is not None:
        for line in lines:
            theta = line[0][1]
            if abs(theta - np.pi / 2) < 0.15:   #  8.6
                v_count += 1
            elif theta < 0.15 or theta > np.pi - 0.15:  # 
                h_count += 1

    is_scan = uniformity < 0.15 and h_count > 2 and v_count > 2

    result = "scan" if is_scan else "photo"
    print(f"[] : {result} (uniformity={uniformity:.3f})")
    return result


# ============================================================
# 
# ============================================================

def deskew(gray: np.ndarray) -> tuple:
    """
    

    :
      1. Canny 
      2.   
      3.   

    Args:
        gray: 

    Returns:
        (deskewed, angle): , 
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        print("[] ")
        return gray, 0.0

    # 
    angles = []
    for line in lines:
        theta = line[0][1]
        # 
        if 0.05 < theta < np.pi - 0.05:
            angle = np.degrees(theta) - 90
            if abs(angle) < 45:
                angles.append(angle)

    if len(angles) < 3:
        return gray, 0.0

    # 
    angle = float(np.median(angles))
    print(f"[] : {angle:.2f}")

    if abs(angle) < 0.3:
        print("[] ")
        return gray, 0.0

    # 
    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=255)

    print(f"[] : {angle:.2f}")
    return deskewed, angle


# ============================================================
# 
# ============================================================

def denoise(gray: np.ndarray, strength: int = 10) -> np.ndarray:
    """
      

    : NLM 
    

    Args:
        gray:   (uint8)
        strength:  (h ,  5-15)

    Returns:
        
    """
    result = cv2.fastNlMeansDenoising(gray, None, h=strength,
                                       templateWindowSize=7,
                                       searchWindowSize=21)
    print(f"[] NLM: h={strength}")
    return result


# ============================================================
# CLAHE 
# ============================================================

def enhance_edges(gray: np.ndarray,
                  clip_limit: float = 2.0,
                  grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
      

    Args:
        gray:       
        clip_limit:  ()
        grid_size:  

    Returns:
        
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=grid_size)
    enhanced = clahe.apply(gray)
    print(f"[] CLAHE: clip={clip_limit}, grid={grid_size}")
    return enhanced


# ============================================================
#  ()
# ============================================================

def adaptive_binarize(gray: np.ndarray,
                      block_size: int = 11,
                      C: int = 3) -> np.ndarray:
    """
    

    :
       Gaussian :
        T(x,y) = local_mean(x,y) - C

      
       Otsu 

    Args:
        gray:       
        block_size:  ()
        C:           ()

    Returns:
         (0=/, 255=/)
    """
    if block_size % 2 == 0:
        block_size += 1

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, C
    )

    # : (255) (0)
    binary = 255 - binary

    print(f"[] : block={block_size}, C={C}")
    return binary


# ============================================================
#  ()
# ============================================================

def normalize_illumination(gray: np.ndarray,
                           kernel_size: int = 31) -> np.ndarray:
    """
      

    I_corrected = I / blur(I) * mean(blur(I))

    Args:
        gray:         
        kernel_size:   ()

    Returns:
        
    """
    blur = cv2.GaussianBlur(gray.astype(np.float32),
                             (kernel_size, kernel_size), 0)
    mean_val = np.mean(blur)
    normalized = gray.astype(np.float32) / (blur + 1e-6) * mean_val
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    print(f"[] : kernel={kernel_size}")
    return normalized


# ============================================================
#  ()
# ============================================================

def correct_perspective(gray: np.ndarray) -> np.ndarray:
    """
      

    Args:
        gray: 

    Returns:
        
    """
    #  + 
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # 
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray

    largest = max(contours, key=cv2.contourArea)

    #   
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) != 4:
        print("[] ")
        return gray

    # :       
    pts = approx.reshape(4, 2)
    rect = _order_points(pts)

    # 
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
    corrected = cv2.warpPerspective(gray, M, (max_width, max_height),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=255)

    print(f"[] : {gray.shape}  {corrected.shape}")
    return corrected


def _order_points(pts: np.ndarray) -> np.ndarray:
    """:       """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # : x+y 
    rect[2] = pts[np.argmax(s)]  # : x+y 

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # : y-x 
    rect[3] = pts[np.argmax(diff)]  # : y-x 

    return rect


# ============================================================
# 
# ============================================================

def preprocess_pipeline(input_path: str,
                        cfg: ReproducerConfig) -> tuple:
    """
    

    :
      1. 
      2. 
      3. :
           scan:    CLAHE  
           photo:       CLAHE  
      4.  (, )

    Args:
        input_path: 
        cfg:        

    Returns:
        (binary, enhanced_gray):  (), 
    """
    gray, bgr = load_image(input_path)
    source_type = detect_source_type(bgr)

    #  photo :  
    if source_type == "photo":
        gray = normalize_illumination(gray)
        gray = correct_perspective(gray)

    #   
    if cfg.deskew_enabled:
        gray, angle = deskew(gray)

    gray = denoise(gray, cfg.denoise_strength)
    gray = enhance_edges(gray, cfg.clahe_clip_limit, cfg.clahe_grid_size)

    block = cfg.adaptive_block
    if block % 2 == 0:
        block += 1
    binary = adaptive_binarize(gray, block, cfg.adaptive_C)

    return binary, gray


# ============================================================
#  ()
# ============================================================

def preprocess_scan(gray: np.ndarray, cfg: ReproducerConfig) -> tuple:
    """"""
    gray = denoise(gray, cfg.denoise_strength)
    gray = enhance_edges(gray, cfg.clahe_clip_limit, cfg.clahe_grid_size)
    block = cfg.adaptive_block if cfg.adaptive_block % 2 == 1 else cfg.adaptive_block + 1
    binary = adaptive_binarize(gray, block, cfg.adaptive_C)
    return binary, gray


def preprocess_photo(gray: np.ndarray, cfg: ReproducerConfig) -> tuple:
    """"""
    gray = normalize_illumination(gray)
    gray = correct_perspective(gray)
    if cfg.deskew_enabled:
        gray, _ = deskew(gray)
    gray = denoise(gray, cfg.denoise_strength)
    gray = enhance_edges(gray, cfg.clahe_clip_limit, cfg.clahe_grid_size)
    block = cfg.adaptive_block if cfg.adaptive_block % 2 == 1 else cfg.adaptive_block + 1
    binary = adaptive_binarize(gray, block, cfg.adaptive_C)
    return binary, gray
