# 🎨 Artistic Style Transfer

Implémentation de **Gatys et al. (2015)** — *A Neural Algorithm of Artistic Style*

Transfère le style artistique d'une image (peinture, photo) sur une image de contenu via VGG19.

---

## Structure du projet

```
artistic_style_transfer/
├── src/
│   ├── models/
│   │   ├── vgg.py              ← Extracteur de features VGG19
│   │   ├── losses.py           ← Content / Style / TV Loss
│   │   └── style_transfer.py  ← Moteur d'optimisation
│   └── utils/
│       └── image_utils.py     ← Chargement & prétraitement images
├── notebooks/
│   └── style_transfer.ipynb  ← Notebook interactif
├── data/
│   ├── content/               ← Vos images de contenu
│   ├── styles/                ← Vos images de style
│   └── outputs/               ← Sorties API
├── output/                    ← Résultats CLI
├── api.py                     ← API REST FastAPI
├── style_transfer.py          ← Script CLI
├── test_installation.py       ← Test des dépendances
└── requirements.txt
```

---

## Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python test_installation.py
```

---

## Utilisation

### CLI

```bash
python style_transfer.py \
  --content data/content/photo.jpg \
  --style   data/styles/starry_night.jpg \
  --output  output/result.jpg \
  --steps   300 \
  --size    512
```

Options principales :

| Option | Défaut | Description |
|---|---|---|
| `--steps` | 300 | Nombre d'itérations |
| `--size` | 512 | Taille de l'image en pixels |
| `--optimizer` | lbfgs | `lbfgs` (recommandé) ou `adam` |
| `--content-weight` | 1.0 | Poids de la perte de contenu |
| `--style-weight` | 1e6 | Poids de la perte de style |
| `--tv-weight` | 1e-4 | Régularisation Total Variation |

### API REST

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/transfer \
  -F "content=@data/content/photo.jpg" \
  -F "style=@data/styles/monet.jpg" \
  -F "num_steps=300" \
  --output result.jpg
```

Documentation Swagger : http://localhost:8000/docs

### Notebook

```bash
jupyter notebook notebooks/style_transfer.ipynb
```

---

## Comment ça marche

1. **VGG19** extrait les features aux couches `relu4_2` (contenu) et `relu1_1…relu5_1` (style)
2. La **matrice de Gram** capture les corrélations inter-canaux qui encodent le style
3. L'optimiseur (**L-BFGS**) minimise : `α·L_content + β·L_style + γ·L_tv`
4. L'image initiale (copie du contenu) converge vers le résultat final

---

## Références

- Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). *A Neural Algorithm of Artistic Style*. arXiv:1508.06576
