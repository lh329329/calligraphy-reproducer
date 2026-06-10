"""
Quick stroke animation — global SAM segmentation, no char-level layout.
"""
import cv2, numpy as np, os, sys
from PIL import Image

PROJ = 'C:/Users/lh329/.claude/projects/C--Windows-System32/calligraphy_reproducer'
sys.path.insert(0, PROJ)


def run(input_img, output_gif):
    from config import ReproducerConfig
    from preprocessing import preprocess_pipeline
    from stroke_seg import load_sam_model, StrokeMask
    from render import generate_paper_texture
    from skimage.morphology import skeletonize

    cfg = ReproducerConfig(
        input_path=input_img,
        sam_checkpoint=f'{PROJ}/models/sam_vit_l_0b3195.pth',
        denoise_strength=8, min_char_area=100,
    )

    print('[1] Preprocessing...')
    binary, gray = preprocess_pipeline(input_img, cfg)
    h, w = gray.shape
    print(f'  Image: {w}x{h}')

    print('[2] Loading SAM...')
    predictor = load_sam_model(cfg)

    print('[3] Tiled stroke segmentation...')
    from stroke_seg import nms_strokes, generate_prompts

    # Tile the image into overlapping blocks for finer SAM granularity
    tile_w, tile_h = 400, 500
    overlap = 100
    all_strokes = []
    seen_global = set()

    for ty in range(0, h, tile_h - overlap):
        for tx in range(0, w, tile_w - overlap):
            t_x1 = tx
            t_y1 = ty
            t_x2 = min(w, tx + tile_w)
            t_y2 = min(h, ty + tile_h)

            tile = gray[t_y1:t_y2, t_x1:t_x2]
            if tile.size == 0 or tile.shape[0] < 30 or tile.shape[1] < 30:
                continue

            # Skip tiles with no ink
            if (tile > 200).mean() > 0.98:
                continue

            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
            predictor.set_image(tile_rgb)

            # Generate prompts within this tile
            tile_bin = 255 - tile if cv2.mean(tile)[0] > 128 else tile.copy()
            coords, labels = generate_prompts(tile_bin, max_prompts=40)

            if len(coords) < 1:
                continue

            masks, scores, _ = predictor.predict(
                point_coords=coords, point_labels=labels,
                multimask_output=True,
            )

            for i in range(masks.shape[0]):
                score = float(scores[i]) if i < len(scores) else 0
                if score < 0.4:
                    continue
                m = masks[i].astype(np.uint8) * 255
                area = (m > 0).sum()
                if area < 30 or area > tile_w * tile_h * 0.9:
                    continue

                # Map tile mask to global coordinates
                global_mask = np.zeros((h, w), dtype=np.uint8)
                global_mask[t_y1:t_y2, t_x1:t_x2] = m
                all_strokes.append((global_mask, score, t_x1, t_y1))

    # NMS across all tiles
    all_masks = []
    # Sort by score descending
    all_strokes.sort(key=lambda x: -x[1])
    for mask, score, _, _ in all_strokes:
        # Check overlap with already-kept masks
        overlap = False
        for kept in all_masks:
            iou = (mask.astype(bool) & kept.astype(bool)).sum() / max(1, (mask.astype(bool) | kept.astype(bool)).sum())
            if iou > 0.4:
                overlap = True
                break
        if not overlap:
            all_masks.append(mask)

    # Build StrokeMask objects
    strokes = []
    for i, mask in enumerate(all_masks):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        strokes.append(StrokeMask(
            stroke_id=i, char_id=0, mask=mask,
            bbox=(int(xs.min()), int(ys.min()),
                  int(xs.max()-xs.min()), int(ys.max()-ys.min())),
            ink_density=0.85, is_flying_white=False, score=0.8
        ))

    # Sort by position
    strokes.sort(key=lambda s: (s.bbox[1] // 40, s.bbox[0]))

    print(f'  {len(strokes)} strokes')

    # --- Animation ---
    print('[4] Rendering animation...')
    paper = np.full((h, w), 248, dtype=np.uint8)
    texture = generate_paper_texture((w, h), seed=42)
    paper = np.clip(paper.astype(np.float32) + texture.astype(np.float32) * 0.3, 0, 255)
    canvas = paper.copy()
    frames = []

    # Blank paper intro (8 frames)
    for _ in range(8):
        frames.append(Image.fromarray(canvas.astype(np.uint8)))

    for idx, s in enumerate(strokes):
        mask = s.mask.astype(np.float32) / 255.0

        # Tip dot frame
        tip = canvas.copy()
        ys, xs = np.where(mask > 0.5)
        if len(ys) > 0:
            sy, sx = ys[0], xs[0]
            r = 8
            y1, y2 = max(0, sy-r), min(h, sy+r)
            x1, x2 = max(0, sx-r), min(w, sx+r)
            tip[y1:y2, x1:x2] = 60
        frames.append(Image.fromarray(tip.astype(np.uint8)))

        # Ink application (2-3 frames)
        for alpha in [0.7, 0.9, 1.0]:
            canvas = canvas * (1.0 - mask * alpha) + 18.0 * mask * alpha
            frames.append(Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)))

        # Hold
        for _ in range(5):
            frames.append(Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)))

    # Final hold
    for _ in range(20):
        frames.append(Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)))

    # Save
    dur = max(80, min(200, 6000 // len(frames)))
    frames[0].save(output_gif, save_all=True, append_images=frames[1:],
                   duration=dur, loop=0)
    sz = os.path.getsize(output_gif)/1024
    print(f'Done: {output_gif}')
    print(f'Strokes:{len(strokes)} Frames:{len(frames)} Size:{sz:.0f}KB Time:{len(frames)*dur/1000:.1f}s')


if __name__ == '__main__':
    inp = 'C:/Users/lh329/Documents/xwechat_files/wxid_u87vgsq20bwv22_a9c4/temp/RWTemp/2026-06/06019852c5a12ccfaed037bb86bdc697.jpg'
    out = 'C:/Users/lh329/Desktop/calligraphy_writing.gif'
    if len(sys.argv) >= 2:
        inp = sys.argv[1]
    if len(sys.argv) >= 3:
        out = sys.argv[2]
    run(inp, out)
