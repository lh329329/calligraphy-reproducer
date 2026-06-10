#!/usr/bin/env python3
"""
  

:
    python main.py --input ./calligraphy.jpg
    python main.py --input ./calligraphy.jpg --dpi 300 --output ./my_output
    python main.py --input ./calligraphy.jpg --model vit_l --device cpu

:
            (SAM)   (potrace)
         

:
    - output/{name}_600dpi.png   ()
    - output/{name}_600dpi.tiff  ( TIFF)
    - output/{name}_full.svg     ()
"""

import os
import sys
import argparse
import cv2
import numpy as np

from config import ReproducerConfig
from preprocessing import preprocess_pipeline
from layout import segment_layout, CharRegion
from stroke_seg import (
    load_sam_model, segment_strokes_single_char,
    segment_all_chars, StrokeMask
)
from vectorize import vectorize_all, build_svg_document, VectorGlyph
from render import render_layers, composite_layers, RenderLayers
from compose import reconstruct_layout, export_all
from utils import Timer


# ============================================================
# 
# ============================================================

def run_pipeline(cfg: ReproducerConfig):
    """
    

     .calligraphy_cache/

    Args:
        cfg: 
    """
    print("=" * 60)
    print("     ")
    print("   CalligraphyReproducer v1.0")
    print("=" * 60)
    print(f"   : {cfg.input_path}")
    print(f"   : {cfg.output_dir} @ {cfg.output_dpi} DPI")
    print(f"   SAM:  {cfg.sam_model_type} @ {cfg.sam_device}")
    print("=" * 60)

    # ================================================================
    #  
    # ================================================================
    with Timer(" "):
        binary, gray = preprocess_pipeline(cfg.input_path, cfg)

    # ================================================================
    #  
    # ================================================================
    with Timer(" "):
        char_regions = segment_layout(binary, gray, cfg)

    if not char_regions:
        print("[] ")
        return

    # ================================================================
    #   (SAM)
    # ================================================================
    with Timer(" SAM "):
        predictor = load_sam_model(cfg)

    with Timer(f"  ({len(char_regions)} )"):
        # :  char_region 
        all_strokes = []
        for i, cr in enumerate(char_regions):
            if cr.image is None or cr.image.size == 0:
                continue
            strokes = segment_strokes_single_char(predictor, cr, cfg, cr.image)
            all_strokes.extend(strokes)
            if (i + 1) % 10 == 0:
                print(f"    : {i+1}/{len(char_regions)} , "
                      f" {len(all_strokes)} ")

        print(f"[] : {len(all_strokes)} ")

    # ================================================================
    #   (potrace)
    # ================================================================
    with Timer(" "):
        glyphs = vectorize_all(all_strokes, cfg)

    # ================================================================
    #   (  )
    # ================================================================
    with Timer(" "):
        # 
        h, w = gray.shape
        canvas_size = (w, h)

        #  + 
        #  char_id  strokes
        strokes_by_char = {}
        for s in all_strokes:
            strokes_by_char.setdefault(s.char_id, []).append(s)

        # 
        char_renders = []
        for cr in char_regions:
            char_strokes = strokes_by_char.get(cr.char_id, [])
            if not char_strokes:
                #   
                char_renders.append(
                    np.full((cr.bbox[3], cr.bbox[2]),
                            cfg.paper_base_color, dtype=np.uint8)
                )
                continue

            # 
            char_layers = render_layers(char_strokes,
                                         (cr.bbox[2], cr.bbox[3]),
                                         cfg)
            char_bitmap = composite_layers(char_layers)
            char_renders.append(char_bitmap)

        # 
        full_bitmap = reconstruct_layout(char_renders, char_regions,
                                         canvas_size, cfg.paper_base_color)
        print("  [] ")

    # ================================================================
    #   SVG + 
    # ================================================================
    with Timer(" "):
        # SVG: 
        svg_content = build_svg_document(glyphs, char_regions,
                                         canvas_size, cfg)

        export_all(full_bitmap, svg_content, cfg)

    # ================================================================
    # 
    # ================================================================
    print(f"\n{'='*60}")
    print(f"   !")
    print(f"  : {len(char_regions)}")
    print(f"  : {len(all_strokes)}")
    print(f"  : {sum(g.num_curves for g in glyphs)}")
    print(f"  : {os.path.abspath(cfg.output_dir)}")
    print(f"{'='*60}")


# ============================================================
# 
# ============================================================

def parse_args():
    """"""
    parser = argparse.ArgumentParser(
        description="  +SVG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
:
  python main.py --input ./calligraphy.jpg
  python main.py --input ./calligraphy.jpg --dpi 300 --output ./my_output
  python main.py --input ./calligraphy.jpg --model vit_l --device cpu
        """
    )

    parser.add_argument("--input", "-i", type=str, required=True,
                        help="")
    parser.add_argument("--output", "-o", type=str, default="./output",
                        help=" (: ./output)")
    parser.add_argument("--dpi", type=int, default=600,
                        help=" (: 600)")
    parser.add_argument("--model", type=str,
                        choices=["vit_b", "vit_l", "vit_h"],
                        default="vit_l",
                        help="SAM  (: vit_l)")
    parser.add_argument("--device", type=str,
                        choices=["cuda", "cpu"],
                        default="cuda",
                        help=" (: cuda)")
    parser.add_argument("--workers", type=int, default=4,
                        help=" (: 4)")
    parser.add_argument("--no-cache", action="store_true",
                        help="")
    parser.add_argument("--no-deskew", action="store_true",
                        help="")

    return parser.parse_args()


def build_config_from_args(args) -> ReproducerConfig:
    """"""
    import os as _os

    #  ()
    PROJECT_DIR = _os.path.dirname(_os.path.abspath(__file__))
    model_paths = {
        "vit_h": _os.path.join(PROJECT_DIR, "models", "sam_vit_h_4b8939.pth"),
        "vit_l": _os.path.join(PROJECT_DIR, "models", "sam_vit_l_0b3195.pth"),
        "vit_b": _os.path.join(PROJECT_DIR, "models", "sam_vit_b_01ec64.pth"),
    }

    return ReproducerConfig(
        input_path=args.input,
        output_dir=args.output,
        output_dpi=args.dpi,
        sam_model_type=args.model,
        sam_checkpoint=model_paths[args.model],
        sam_device=args.device,
        parallel_workers=args.workers,
        use_cache=not args.no_cache,
        deskew_enabled=not args.no_deskew,
    )


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    # 
    if not os.path.exists(args.input):
        print(f"[] : {args.input}")
        sys.exit(1)

    cfg = build_config_from_args(args)
    run_pipeline(cfg)
