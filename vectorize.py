"""
    potrace 

:
  1.   potrace Bitmap  Path list
  2. Path  SVG d 
  3.  ( path )
  4.  ()

:
  potracer ( Python potrace): pip install potracer
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import ReproducerConfig
from stroke_seg import StrokeMask


# ============================================================
# 
# ============================================================

@dataclass
class VectorGlyph:
    """"""
    stroke_id:   int      # 
    char_id:     int      # 
    svg_path_d:  str      # SVG d 
    bbox:        tuple    #  (x, y, w, h)
    num_curves:  int      # 


# ============================================================
#   SVG 
# ============================================================

def mask_to_svg_path(mask: np.ndarray,
                     cfg: ReproducerConfig) -> str:
    """
     SVG path d 

     potrace 

    Args:
        mask:  (H,W) uint8, =255
        cfg:  

    Returns:
        SVG d  "M 10 20 C 15 25 ... Z"
    """
    from potrace import Bitmap, POTRACE_TURNPOLICY_MINORITY

    # potrace : True=/, 
    #  255=,  True ()
    bitmap_data = (mask > 128)

    if not bitmap_data.any():
        return "", 0

    bm = Bitmap(bitmap_data)

    # trace :
    # - turdsize:  ()
    # - alphamax: 
    # - opttolerance: 
    # - turnpolicy: 
    path_list = bm.trace(
        turdsize=cfg.potrace_turdsize,
        alphamax=cfg.potrace_alphamax,
        opttolerance=cfg.potrace_opttolerance,
        turnpolicy=POTRACE_TURNPOLICY_MINORITY,
    )

    #  potrace path list  SVG d 
    d_parts = []
    curve_count = 0

    for curve in path_list:
        # curve.start_point: 
        start = curve.start_point
        d_parts.append(f"M {start.x:.1f} {start.y:.1f}")

        # curve.segments: 
        for seg in curve.segments:
            if seg.is_corner:
                # Corner segment: straight line to 'c'
                end = seg.c
                d_parts.append(f"L {end.x:.1f} {end.y:.1f}")
            else:
                # Bezier segment: cubic curve with c1, c2, end_point
                c1 = seg.c1
                c2 = seg.c2
                end = seg.end_point
                d_parts.append(
                    f"C {c1.x:.1f} {c1.y:.1f} "
                    f"{c2.x:.1f} {c2.y:.1f} "
                    f"{end.x:.1f} {end.y:.1f}"
                )
                curve_count += 1

        # 
        d_parts.append("Z")

    d = " ".join(d_parts)
    return d, curve_count


# ============================================================
# 
# ============================================================

def vectorize_stroke(stroke: StrokeMask,
                     cfg: ReproducerConfig) -> VectorGlyph:
    """
    

    Args:
        stroke: 
        cfg:    

    Returns:
        VectorGlyph 
    """
    d, num_curves = mask_to_svg_path(stroke.mask, cfg)

    return VectorGlyph(
        stroke_id=stroke.stroke_id,
        char_id=stroke.char_id,
        svg_path_d=d,
        bbox=stroke.bbox,
        num_curves=num_curves,
    )


# ============================================================
# 
# ============================================================

def vectorize_all(strokes: List[StrokeMask],
                  cfg: ReproducerConfig) -> List[VectorGlyph]:
    """
    

    Args:
        strokes: 
        cfg:     

    Returns:
        list[VectorGlyph]: 
    """
    print(f"[] : {len(strokes)} , "
          f"={cfg.parallel_workers * 2}")

    glyphs = []

    if len(strokes) <= 4:
        # : 
        for s in strokes:
            if s.mask.sum() > 0:
                glyphs.append(vectorize_stroke(s, cfg))
    else:
        # :  ( CPU )
        with ThreadPoolExecutor(
            max_workers=cfg.parallel_workers * 2
        ) as executor:
            futures = {
                executor.submit(vectorize_stroke, s, cfg): s.stroke_id
                for s in strokes if s.mask.sum() > 0
            }

            for future in as_completed(futures):
                try:
                    glyphs.append(future.result())
                except Exception as e:
                    sid = futures[future]
                    print(f"[]  {sid} : {e}")

    # 
    glyphs.sort(key=lambda g: (g.char_id, g.stroke_id))

    total_curves = sum(g.num_curves for g in glyphs)
    print(f"[] : {len(glyphs)} , "
          f"{total_curves} ")
    return glyphs


# ============================================================
# SVG 
# ============================================================

def build_svg_document(
    glyphs: List[VectorGlyph],
    char_regions: list,
    canvas_size: tuple,
    cfg: ReproducerConfig
) -> str:
    """
     SVG 

    :
      <g id="ink-dense">   
      <g id="flying-white">
      <g id="ink-light">   

    Args:
        glyphs:       
        char_regions:  ()
        canvas_size:   (w, h)
        cfg:          

    Returns:
         SVG 
    """
    w, h = canvas_size

    # 
    dense_paths = []
    light_paths = []
    fw_paths = []

    for g in glyphs:
        if g.svg_path_d:
            #  StrokeMask 
            # 
            path_elem = (
                f'<path d="{g.svg_path_d}" '
                f'fill="#000000" stroke="none" '
                f'data-char="{g.char_id}" data-stroke="{g.stroke_id}"/>'
            )
            dense_paths.append(path_elem)

    # 
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{w}" height="{h}"
     viewBox="0 0 {w} {h}"
     version="1.1">
  <desc>   CalligraphyReproducer </desc>

  <!--  -->
  <rect width="{w}" height="{h}"
        fill="rgb({cfg.paper_base_color},{cfg.paper_base_color},{cfg.paper_base_color})"/>

  <!--  -->
  <g id="layer-ink-bleed" opacity="0.3">
  </g>

  <!--  -->
  <g id="layer-ink-dense" opacity="0.95">
    {chr(10).join('    ' + p for p in dense_paths)}
  </g>

  <!--  -->
  <g id="layer-flying-white" opacity="0.7">
    {chr(10).join('    ' + p for p in fw_paths)}
  </g>

  <!--  -->
  <g id="layer-ink-light" opacity="0.5">
    {chr(10).join('    ' + p for p in light_paths)}
  </g>
</svg>'''

    return svg
