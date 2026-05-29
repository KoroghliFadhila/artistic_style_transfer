"""
Utilitaires pour le chargement et la manipulation d'images.
Gère la normalisation ImageNet utilisée par VGG19.
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Union

# ------------------------------------------------------------------ #
# Constantes ImageNet
# ------------------------------------------------------------------ #
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_SIZE = 512


# ------------------------------------------------------------------ #
# Transformations
# ------------------------------------------------------------------ #
def _get_transform(size: int) -> T.Compose:
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
    ])


_normalize   = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
_unnormalize = T.Normalize(
    mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
    std=[1 / s for s in IMAGENET_STD],
)


# ------------------------------------------------------------------ #
# API publique
# ------------------------------------------------------------------ #

def load_image(path: Union[str, Path], size: int = DEFAULT_IMAGE_SIZE) -> Image.Image:
    """Charge une image PIL et la redimensionne."""
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.LANCZOS)
    return img


def preprocess(img: Image.Image, device: str = 'cpu') -> torch.Tensor:
    """
    PIL → tenseur normalisé (1, 3, H, W) prêt pour VGG19.
    """
    t = TF.to_tensor(img).unsqueeze(0)          # (1, 3, H, W), valeurs [0, 1]
    t = _normalize(t.squeeze(0)).unsqueeze(0)   # normalisation ImageNet
    return t.to(device)


def postprocess(tensor: torch.Tensor) -> Image.Image:
    """
    Tenseur normalisé (1, 3, H, W) ou (3, H, W) → PIL Image.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = _unnormalize(tensor).clamp(0, 1)
    return TF.to_pil_image(tensor.cpu())


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Alias de postprocess."""
    return postprocess(tensor)


def pil_to_bytes(img: Image.Image, fmt: str = 'JPEG') -> bytes:
    """Convertit une PIL Image en bytes (pour l'API)."""
    import io
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=95)
    return buf.getvalue()


def save_image(img: Image.Image, path: Union[str, Path]) -> None:
    """Sauvegarde une PIL Image."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)
