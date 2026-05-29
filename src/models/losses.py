"""
Fonctions de perte pour le Neural Style Transfer.
- ContentLoss  : MSE entre les features de contenu
- StyleLoss    : MSE entre les matrices de Gram (style)
- TVLoss       : Total Variation Loss (régularisation, lissage)
- StyleTransferLoss : combinaison pondérée des trois
"""

import torch
import torch.nn as nn
from typing import Dict

from .vgg import gram_matrix, VGG19Features


class ContentLoss(nn.Module):
    """MSE entre la feature de contenu générée et la cible."""

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(generated, target.detach())


class StyleLoss(nn.Module):
    """MSE entre matrices de Gram générée et cible."""

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        G_gen = gram_matrix(generated)
        G_tgt = gram_matrix(target.detach())
        return nn.functional.mse_loss(G_gen, G_tgt)


class TVLoss(nn.Module):
    """
    Total Variation Loss : pénalise les variations brusques entre pixels voisins.
    Aide à produire une image lisse.
    """

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Différences horizontales et verticales
        diff_h = img[:, :, 1:, :] - img[:, :, :-1, :]
        diff_w = img[:, :, :, 1:] - img[:, :, :, :-1]
        return diff_h.abs().mean() + diff_w.abs().mean()


class StyleTransferLoss(nn.Module):
    """
    Perte totale = alpha * content_loss + beta * style_loss + gamma * tv_loss
    """

    def __init__(
        self,
        vgg: VGG19Features,
        content_weight: float = 1.0,
        style_weight: float = 1e6,
        tv_weight: float = 1e-4,
    ):
        super().__init__()
        self.vgg = vgg
        self.content_weight = content_weight
        self.style_weight   = style_weight
        self.tv_weight      = tv_weight

        self.content_loss = ContentLoss()
        self.style_loss   = StyleLoss()
        self.tv_loss      = TVLoss()

    def forward(
        self,
        generated: torch.Tensor,
        content_features: Dict[str, torch.Tensor],
        style_features:   Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        generated        : image en cours d'optimisation (1, 3, H, W)
        content_features : features extraites de l'image de contenu
        style_features   : features extraites de l'image de style
        Retourne un dict avec les pertes individuelles et la perte totale.
        """
        gen_features = self.vgg(generated)

        # --- Contenu ---
        c_loss = torch.tensor(0.0, device=generated.device)
        for layer in self.vgg.CONTENT_LAYERS:
            if layer in gen_features and layer in content_features:
                c_loss += self.content_loss(gen_features[layer], content_features[layer])

        # --- Style ---
        s_loss = torch.tensor(0.0, device=generated.device)
        for layer in self.vgg.STYLE_LAYERS:
            if layer in gen_features and layer in style_features:
                s_loss += self.style_loss(gen_features[layer], style_features[layer])

        # --- TV ---
        tv = self.tv_loss(generated)

        total = (
            self.content_weight * c_loss
            + self.style_weight  * s_loss
            + self.tv_weight     * tv
        )

        return {
            'total':   total,
            'content': c_loss,
            'style':   s_loss,
            'tv':      tv,
        }
