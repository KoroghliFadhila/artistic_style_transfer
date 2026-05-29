"""
CLI : style_transfer.py
Usage :
    python style_transfer.py --content data/content.jpg --style data/style.jpg
    python style_transfer.py --content img.jpg --style art.jpg --steps 500 --size 512
"""

import argparse
import time
from pathlib import Path

import torch

from src.models.style_transfer import StyleTransfer
from src.utils.image_utils import save_image


def parse_args():
    p = argparse.ArgumentParser(description='Neural Style Transfer (Gatys et al. 2015)')
    p.add_argument('--content',  required=True,  help='Chemin vers l\'image de contenu')
    p.add_argument('--style',    required=True,  help='Chemin vers l\'image de style')
    p.add_argument('--output',   default='output/result.jpg', help='Chemin de sortie')
    p.add_argument('--size',     type=int,   default=512,   help='Taille de l\'image (px)')
    p.add_argument('--steps',    type=int,   default=300,   help='Nombre d\'itérations')
    p.add_argument('--content-weight', type=float, default=1.0,  help='Poids contenu')
    p.add_argument('--style-weight',   type=float, default=1e6,  help='Poids style')
    p.add_argument('--tv-weight',      type=float, default=1e-4, help='Poids TV')
    p.add_argument('--optimizer', choices=['lbfgs', 'adam'], default='lbfgs')
    p.add_argument('--init',      choices=['content', 'style', 'noise'], default='content')
    p.add_argument('--save-every', type=int, default=0,
                   help='Sauvegarder l\'image tous les N pas (0 = désactivé)')
    p.add_argument('--device',    default=None,
                   help='Device (cpu/cuda). Auto-détecté si absent.')
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n🎨 Neural Style Transfer')
    print(f'   Device   : {device}')
    print(f'   Contenu  : {args.content}')
    print(f'   Style    : {args.style}')
    print(f'   Sortie   : {args.output}')
    print(f'   Steps    : {args.steps}')
    print(f'   Taille   : {args.size}px')
    print(f'   Optimizer: {args.optimizer}\n')

    model = StyleTransfer(
        device=device,
        image_size=args.size,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
    )

    def progress(step, losses):
        if step % 20 == 0 or step == 1:
            print(f'  [{step:4d}] total={losses["total"]:.4f}  '
                  f'content={losses["content"]:.4f}  '
                  f'style={losses["style"]:.6f}  '
                  f'tv={losses["tv"]:.6f}')

    output_prefix = str(Path(args.output).with_suffix('')) + '_step'

    t0 = time.time()
    result = model.transfer(
        content_path=args.content,
        style_path=args.style,
        num_steps=args.steps,
        optimizer_type=args.optimizer,
        init_from=args.init,
        progress_callback=progress,
        save_every=args.save_every,
        output_prefix=output_prefix,
    )

    save_image(result, args.output)
    elapsed = time.time() - t0
    print(f'\n✅ Image sauvegardée : {args.output}')
    print(f'⏱️  Durée : {elapsed:.1f}s\n')


if __name__ == '__main__':
    main()
