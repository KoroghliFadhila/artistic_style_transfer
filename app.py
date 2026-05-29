"""
Interface Gradio — Artistic Style Transfer
Lance avec : python app.py
"""

import gradio as gr
import torch
import tempfile
from pathlib import Path
from PIL import Image

from src.models.style_transfer import StyleTransfer
from src.utils.image_utils import save_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Styles prédéfinis (optionnel — si tu mets des images dans data/styles/)
STYLES_DIR = Path("data/styles")
PRESET_STYLES = {}
if STYLES_DIR.exists():
    for p in sorted(STYLES_DIR.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            PRESET_STYLES[p.stem] = str(p)


def run_transfer(
    content_image,
    style_image,
    preset_style,
    num_steps,
    style_weight,
    content_weight,
    image_size,
):
    # Priorité à l'image uploadée, sinon style prédéfini
    if style_image is None and preset_style and preset_style in PRESET_STYLES:
        style_image = Image.open(PRESET_STYLES[preset_style]).convert("RGB")
    
    if content_image is None:
        raise gr.Error("Veuillez fournir une image de contenu.")
    if style_image is None:
        raise gr.Error("Veuillez fournir une image de style ou choisir un style prédéfini.")

    model = StyleTransfer(
        device=DEVICE,
        image_size=int(image_size),
        content_weight=content_weight,
        style_weight=style_weight,
    )

    logs = []
    def on_progress(step, losses):
        if step % 20 == 0 or step == 1:
            msg = f"Step {step} | total={losses['total']:.2f} | style={losses['style']:.4f}"
            logs.append(msg)
            print(msg)

    # Sauvegarde temporaire des images PIL
    with tempfile.TemporaryDirectory() as tmp:
        c_path = Path(tmp) / "content.jpg"
        s_path = Path(tmp) / "style.jpg"

        if isinstance(content_image, Image.Image):
            content_image.save(c_path)
        else:
            Image.fromarray(content_image).save(c_path)

        if isinstance(style_image, Image.Image):
            style_image.save(s_path)
        else:
            Image.fromarray(style_image).save(s_path)

        result: Image.Image = model.transfer(
            content_path=str(c_path),
            style_path=str(s_path),
            num_steps=int(num_steps),
            optimizer_type="lbfgs",
            init_from="content",
            progress_callback=on_progress,
        )

    output_path = "output/result.jpg"
    Path("output").mkdir(exist_ok=True)
    save_image(result, output_path)

    return result, "\n".join(logs)


# ── Interface ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Artistic Style Transfer", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🎨 Artistic Style Transfer
    Transfère le style artistique d'une peinture sur ta photo, via VGG19 (Gatys et al. 2015).
    """)

    with gr.Row():

        with gr.Column():
            gr.Markdown("### 📷 Image de contenu")
            content_input = gr.Image(type="pil", label="Ta photo")

            gr.Markdown("### 🖼️ Image de style")
            style_input = gr.Image(type="pil", label="Upload une peinture")

            if PRESET_STYLES:
                preset = gr.Dropdown(
                    choices=[""] + list(PRESET_STYLES.keys()),
                    value="",
                    label="Ou choisis un style prédéfini",
                )
            else:
                preset = gr.Textbox(visible=False, value="")

            gr.Markdown("### ⚙️ Paramètres")
            with gr.Row():
                steps = gr.Slider(50, 500, value=300, step=50, label="Nombre d'étapes")
                size  = gr.Slider(256, 512, value=512, step=128, label="Taille image (px)")
            with gr.Row():
                s_weight = gr.Slider(1e4, 1e7, value=1e6, step=1e4, label="Poids style")
                c_weight = gr.Slider(0.1, 10,  value=1.0, step=0.1, label="Poids contenu")

            btn = gr.Button("🚀 Lancer le style transfer", variant="primary")

        with gr.Column():
            gr.Markdown("### ✨ Résultat")
            output_image = gr.Image(type="pil", label="Image générée")
            output_log   = gr.Textbox(label="Progression", lines=10, interactive=False)

    btn.click(
        fn=run_transfer,
        inputs=[content_input, style_input, preset, steps, s_weight, c_weight, size],
        outputs=[output_image, output_log],
    )

    gr.Markdown("""
    ---
    💡 **Conseils** : utilise une peinture connue comme style (Van Gogh, Monet, Picasso).  
    Sur CPU, 300 étapes prennent ~5–15 min selon la taille.
    """)


if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)