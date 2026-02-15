import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def organize_style_images():
    """Organise les images par style artistique"""
    
    wikiart_path = Path("data/raw/wikiart")
    output_path = Path("data/processed/styles")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Styles à extraire
    styles = [
        'impressionism',
        'cubism',
        'abstract',
        'expressionism',
        'pop_art',
        'realism'
    ]
    
    for style in styles:
        style_folder = output_path / style
        style_folder.mkdir(exist_ok=True)
        
        # Copier ~100 images par style
        source_folder = wikiart_path / style
        if source_folder.exists():
            images = list(source_folder.glob("*.jpg"))[:100]
            
            for img_path in tqdm(images, desc=f"Processing {style}"):
                try:
                    # Vérifier et redimensionner
                    img = Image.open(img_path)
                    img = img.convert('RGB')
                    img = img.resize((512, 512), Image.LANCZOS)
                    
                    # Sauvegarder
                    output_file = style_folder / img_path.name
                    img.save(output_file, quality=95)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

def organize_content_images():
    """Organise les images de contenu"""
    
    coco_path = Path("data/raw/coco/val2017")
    output_path = Path("data/processed/content")
    output_path.mkdir(parents=True, exist_ok=True)
    
    if coco_path.exists():
        images = list(coco_path.glob("*.jpg"))[:1000]  # 1000 images
        
        for img_path in tqdm(images, desc="Processing content images"):
            try:
                img = Image.open(img_path)
                img = img.convert('RGB')
                
                # Resize si trop grande
                if max(img.size) > 1024:
                    img.thumbnail((1024, 1024), Image.LANCZOS)
                
                output_file = output_path / img_path.name
                img.save(output_file, quality=95)
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    print("Organizing datasets...")
    organize_style_images()
    organize_content_images()
    print("Done!")