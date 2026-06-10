"""
  ReproducerConfig

:
    from config import ReproducerConfig
    cfg = ReproducerConfig(input_path="./calligraphy.jpg")
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ReproducerConfig:
    """"""

    # ============================================================
    # 
    # ============================================================
    input_path:     str   = ""             # 
    output_dir:     str   = "./output"     # 
    output_dpi:     int   = 600            # 
    output_formats: List[str] = field(     # 
        default_factory=lambda: ["png", "tiff", "svg"]
    )

    # ============================================================
    # 
    # ============================================================
    deskew_enabled:   bool  = True         # 
    denoise_strength: int   = 10           #  (h, )
    clahe_clip_limit: float = 2.0          # CLAHE 
    clahe_grid_size:  Tuple[int, int] = (8, 8)  # CLAHE 
    adaptive_block:   int   = 11           #  ()
    adaptive_C:       int   = 3            # 

    # ============================================================
    # 
    # ============================================================
    min_char_area:    int   = 100          #  (px), 
    char_margin:      int   = 8            #  (px), 

    # ============================================================
    # SAM 
    # ============================================================
    sam_model_type:   str   = "vit_l"      # : vit_b | vit_l | vit_h
    sam_checkpoint:   str   = ""  #  ()
    sam_device:       str   = "cuda"       # : cuda | cpu

    # ============================================================
    #  (potrace)
    # ============================================================
    potrace_turdsize:     int   = 2        #  ()
    potrace_alphamax:     float = 1.0      #  ()
    potrace_opttolerance: float = 0.2      #  ()

    # ============================================================
    # 
    # ============================================================
    ink_levels:         int   = 5          #  (K-means )
    bleed_radius:       int   = 3          #  (px)
    bleed_intensity:    float = 0.6        #  (0~1)
    paper_texture_seed: int   = 42         # 
    paper_base_color:   int   = 245        #  (0-255)

    # ============================================================
    # 
    # ============================================================
    parallel_workers:   int   = 4          # 
    use_cache:          bool  = True       # 

    # ============================================================
    #  ()
    # ============================================================
    @property
    def cache_dir(self) -> str:
        """"""
        import os
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ".calligraphy_cache"
        )
