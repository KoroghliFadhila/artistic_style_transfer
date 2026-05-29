"""
API REST FastAPI pour le Neural Style Transfer.

Endpoints :
  POST /transfer        → lance le style transfer, retourne l'image
  GET  /health          → vérification de l'état du serveur
  GET  /styles          → liste les styles prédéfinis disponibles
"""

import io
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from PIL import Image

from src.models.style_transfer import StyleTransfer
from src.utils.image_utils import pil_to_bytes, save_image

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE  = int(os.getenv('IMAGE_SIZE',  '512'))
NUM_STEPS   = int(os.getenv('NUM_STEPS',   '300'))
STYLE_DIR   = Path(os.getenv('STYLE_DIR',  'data/styles'))
OUTPUT_DIR  = Path(os.getenv('OUTPUT_DIR', 'data/outputs'))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
# App FastAPI
# ------------------------------------------------------------------ #
app = FastAPI(
    title='Artistic Style Transfer API',
    version='1.0.0',
    description='Neural Style Transfer basé sur VGG19 (Gatys et al. 2015)',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# Modèle chargé une seule fois au démarrage
_model: Optional[StyleTransfer] = None


def get_model() -> StyleTransfer:
    global _model
    if _model is None:
        _model = StyleTransfer(
            device=DEVICE,
            image_size=IMAGE_SIZE,
        )
    return _model


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@app.get('/health')
async def health():
    return {
        'status': 'ok',
        'device': DEVICE,
        'image_size': IMAGE_SIZE,
    }


@app.get('/styles')
async def list_styles():
    """Retourne la liste des images de style disponibles dans STYLE_DIR."""
    if not STYLE_DIR.exists():
        return {'styles': []}
    styles = [p.name for p in STYLE_DIR.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    return {'styles': sorted(styles)}


@app.post('/transfer')
async def transfer(
    content: UploadFile = File(..., description='Image de contenu (JPEG/PNG)'),
    style:   UploadFile = File(..., description='Image de style   (JPEG/PNG)'),
    num_steps:       int   = Form(default=300,   description='Nombre d\'itérations'),
    content_weight:  float = Form(default=1.0,   description='Poids contenu'),
    style_weight:    float = Form(default=1e6,   description='Poids style'),
    tv_weight:       float = Form(default=1e-4,  description='Poids Total Variation'),
    optimizer:       str   = Form(default='lbfgs', description='lbfgs | adam'),
    init_from:       str   = Form(default='content', description='content | style | noise'),
    output_format:   str   = Form(default='jpeg',  description='jpeg | png'),
):
    """
    Lance le style transfer et retourne l'image résultante.
    """
    # Validation des fichiers
    for upload in (content, style):
        if upload.content_type not in {'image/jpeg', 'image/png', 'image/jpg'}:
            raise HTTPException(status_code=400, detail=f'Format non supporté : {upload.content_type}')

    try:
        # Sauvegarde temporaire des fichiers uploadés
        with tempfile.TemporaryDirectory() as tmp:
            content_path = Path(tmp) / 'content.jpg'
            style_path   = Path(tmp) / 'style.jpg'

            content_path.write_bytes(await content.read())
            style_path.write_bytes(await style.read())

            model = get_model()
            # Mise à jour des poids si différents de ceux par défaut
            model.criterion.content_weight = content_weight
            model.criterion.style_weight   = style_weight
            model.criterion.tv_weight      = tv_weight

            steps_done = []
            def on_progress(step, losses):
                steps_done.append(step)
                if step % 50 == 0 or step == 1:
                    print(f'  Step {step:4d} | total={losses["total"]:.4f} '
                          f'| content={losses["content"]:.4f} '
                          f'| style={losses["style"]:.6f}')

            result: Image.Image = model.transfer(
                content_path=str(content_path),
                style_path=str(style_path),
                num_steps=num_steps,
                optimizer_type=optimizer,
                init_from=init_from,
                progress_callback=on_progress,
            )

        # Encode et retourne
        fmt      = 'PNG' if output_format.lower() == 'png' else 'JPEG'
        mime     = 'image/png' if fmt == 'PNG' else 'image/jpeg'
        img_bytes = pil_to_bytes(result, fmt=fmt)

        return Response(content=img_bytes, media_type=mime)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ------------------------------------------------------------------ #
# Point d'entrée
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('api:app', host='0.0.0.0', port=8000, reload=False)
