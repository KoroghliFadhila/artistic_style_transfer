"""
StyleTransfer : orchestre l'optimisation de l'image générée.
Utilise L-BFGS (recommandé par Gatys et al.) ou Adam.
"""

import torch
import torch.optim as optim
from PIL import Image
from typing import Callable, Optional
import copy

from .vgg import VGG19Features
from .losses import StyleTransferLoss
from ..utils.image_utils import (
    load_image, tensor_to_pil, preprocess, postprocess,
    DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
)


class StyleTransfer:
    """
    Applique le Neural Style Transfer (Gatys et al. 2015).

    Usage :
        st = StyleTransfer(device='cpu')
        result = st.transfer(
            content_path='data/content.jpg',
            style_path='data/style.jpg',
            num_steps=300,
        )
        result.save('output.jpg')
    """

    def __init__(
        self,
        device: str = 'cpu',
        image_size: int = DEFAULT_IMAGE_SIZE,
        content_weight: float = 1.0,
        style_weight: float = 1e6,
        tv_weight: float = 1e-4,
    ):
        self.device       = device
        self.image_size   = image_size

        self.vgg = VGG19Features(device=device)
        self.criterion = StyleTransferLoss(
            vgg=self.vgg,
            content_weight=content_weight,
            style_weight=style_weight,
            tv_weight=tv_weight,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transfer(
        self,
        content_path: str,
        style_path: str,
        num_steps: int = 300,
        optimizer_type: str = 'lbfgs',
        init_from: str = 'content',         # 'content' | 'style' | 'noise'
        progress_callback: Optional[Callable[[int, dict], None]] = None,
        save_every: int = 0,                # 0 = désactivé
        output_prefix: str = 'output/step',
    ) -> Image.Image:
        """
        Lance l'optimisation et retourne l'image finale (PIL).

        progress_callback(step, losses) est appelé à chaque itération si fourni.
        """
        # Chargement + pré-traitement
        content_tensor = preprocess(load_image(content_path, self.image_size), self.device)
        style_tensor   = preprocess(load_image(style_path,   self.image_size), self.device)

        # Extraction des features cibles (faite une seule fois)
        with torch.no_grad():
            content_features = self.vgg(content_tensor)
            style_features   = self.vgg(style_tensor)

        # Image à optimiser
        generated = self._init_image(content_tensor, style_tensor, init_from)
        generated.requires_grad_(True)

        # Optimiseur
        if optimizer_type == 'lbfgs':
            optimizer = optim.LBFGS([generated], lr=1.0, max_iter=20)
        else:
            optimizer = optim.Adam([generated], lr=0.01)

        step = [0]

        def closure():
            with torch.no_grad():
                generated.clamp_(0, 1)

            optimizer.zero_grad()
            losses = self.criterion(generated, content_features, style_features)
            losses['total'].backward()

            step[0] += 1
            if progress_callback:
                progress_callback(step[0], {k: v.item() for k, v in losses.items()})

            if save_every > 0 and step[0] % save_every == 0:
                img = postprocess(generated.detach())
                img.save(f'{output_prefix}_{step[0]:04d}.jpg')

            return losses['total']

        # Boucle principale
        if optimizer_type == 'lbfgs':
            # L-BFGS fait plusieurs évaluations de closure par appel à step()
            iterations = max(1, num_steps // 20)
            for _ in range(iterations):
                optimizer.step(closure)
        else:
            for _ in range(num_steps):
                optimizer.step(closure)

        with torch.no_grad():
            generated.clamp_(0, 1)

        return postprocess(generated.detach())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_image(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        if mode == 'content':
            return content.clone()
        elif mode == 'style':
            return style.clone()
        else:   # noise
            return torch.randn_like(content) * 0.01
