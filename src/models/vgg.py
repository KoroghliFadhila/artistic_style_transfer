"""
VGG19 Feature Extractor for Neural Style Transfer
Extrait les features des couches conv pour le contenu et le style.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List


class VGG19Features(nn.Module):
    """
    Extrait les activations intermédiaires de VGG19 pour le style transfer.
    
    Couches utilisées :
    - Contenu : conv4_2 (relu4_2)
    - Style    : conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
    """

    # Noms des couches VGG19 dans l'ordre
    LAYER_NAMES = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
    ]

    CONTENT_LAYERS = ['relu4_2']
    STYLE_LAYERS   = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device

        # Charge VGG19 pré-entraîné, on n'utilise que la partie features
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())

        # Construit un ModuleDict nommé par couche
        self.layers: Dict[str, nn.Module] = nn.ModuleDict()
        conv_idx = pool_idx = 0
        block = 1
        conv_in_block = 0

        for layer in features:
            if isinstance(layer, nn.Conv2d):
                conv_in_block += 1
                name = f'conv{block}_{conv_in_block}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu{block}_{conv_in_block}'
                layer = nn.ReLU(inplace=False)   # évite les modifs in-place
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool{block}'
                block += 1
                conv_in_block = 0
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn{block}_{conv_in_block}'
            else:
                name = f'layer_{len(self.layers)}'

            self.layers[name] = layer

        # Gèle tous les paramètres
        for param in self.parameters():
            param.requires_grad = False

        self.to(device)

    def forward(self, x: torch.Tensor,
                layers_to_extract: List[str] = None) -> Dict[str, torch.Tensor]:
        """
        Passe l'image dans VGG19 et retourne un dict {nom_couche: feature_map}.
        Si layers_to_extract est None, extrait contenu + style par défaut.
        """
        if layers_to_extract is None:
            layers_to_extract = self.CONTENT_LAYERS + self.STYLE_LAYERS

        features = {}
        out = x
        for name, layer in self.layers.items():
            out = layer(out)
            if name in layers_to_extract:
                features[name] = out
            # Arrête tôt si on a tout ce qu'il faut
            if len(features) == len(layers_to_extract):
                break
        return features


def gram_matrix(feature: torch.Tensor) -> torch.Tensor:
    """
    Calcule la matrice de Gram d'une feature map.
    feature : (batch, C, H, W)
    retourne : (batch, C, C)
    """
    B, C, H, W = feature.size()
    f = feature.view(B, C, H * W)
    G = torch.bmm(f, f.transpose(1, 2))
    return G / (C * H * W)
