import argparse
from pathlib import Path

import cv2
import numpy as np

from deeproof.utils.sam_refine import build_sam_prompts, _try_build_predictor


def parse_args():
    parser = argparse.ArgumentParser(description='Generate SAM/SAM2 teacher masks for distillation.')
    parser.add_argument('--images-dir', required=True, help='Input images dir')
    parser.add_argument('--seed-masks-dir', required=True, help='Seed binary masks dir')
    parser.add_argument('--out-dir', required=True, help='Output teacher masks dir')
    parser.add_argument('--model-type', default='vit_b', help='SAM model type or SAM2 config key')
    parser.add_argument('--checkpoint', required=True, help='SAM/SAM2 checkpoint path')
    parser.add_argument('--suffix', default='.png', help='Mask suffix')
    return parser.parse_args()


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    seed_dir = Path(args.seed_masks_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor, runtime = _try_build_predictor(args.model_type, args.checkpoint)
    if predictor is None:
        raise RuntimeError('SAM runtime is unavailable. Install sam2 or segment-anything.')

    seed_files = sorted(seed_dir.glob(f'*{args.suffix}'))
    for sp in seed_files:
        sid = sp.stem
        img_path = images_dir / f'{sid}.jpg'
        if not img_path.exists():
            img_path = images_dir / f'{sid}.png'
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seed = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE)
        if seed is None:
            continue
        seed_bin = (seed > 0).astype(np.uint8)
        bbox, pos, neg = build_sam_prompts(seed_bin)
        if pos.size == 0:
            continue

        predictor.set_image(img)
        points = np.concatenate([pos, neg], axis=0)
        labels = np.concatenate(
            [np.ones((len(pos),), dtype=np.int32), np.zeros((len(neg),), dtype=np.int32)],
            axis=0)
        masks, _, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=bbox[None, :],
            multimask_output=False,
        )
        if masks is None or len(masks) == 0:
            continue
        out = (masks[0] > 0).astype(np.uint8) * 255
        cv2.imwrite(str(out_dir / f'{sid}{args.suffix}'), out)
    print(f'SAM teacher masks generated with runtime={runtime} into {out_dir}')


if __name__ == '__main__':
    main()
