"""
Télécharge des images de test pour Neural Style Transfer
"""

import requests
import os
from pathlib import Path
from PIL import Image
import io

def download_image(url, save_path):
    """Télécharge une image depuis une URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0'
        }
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        
        img = Image.open(io.BytesIO(response.content))
        img = img.convert('RGB')
        img.save(save_path)
        print(f"   Downloaded: {save_path.name}")
        return True
    except Exception as e:
        print(f"   Failed {save_path.name}: {e}")
        return False

def setup_test_data():
    # Créer dossiers
    content_dir = Path("data/processed/content")
    style_dir   = Path("data/processed/styles")
    results_dir = Path("results/visualizations")
    
    for d in [content_dir, style_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(" Downloading content images...")
    content_images = {
        "landscape.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-natural-beauty.jpg/800px-24701-nature-natural-beauty.jpg",
        "city.jpg":      "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Southwest_corner_of_Central_Park%2C_looking_east%2C_NYC.jpg/800px-Southwest_corner_of_Central_Park%2C_looking_east%2C_NYC.jpg",
    }
    
    for filename, url in content_images.items():
        download_image(url, content_dir / filename)

    print("\n Downloading style images...")
    style_images = {
        "starry_night.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/800px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
        "great_wave.jpg":   "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/The_Great_Wave_off_Kanagawa.jpg/800px-The_Great_Wave_off_Kanagawa.jpg",
        "the_scream.jpg":   "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg/800px-Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg",
    }
    
    for filename, url in style_images.items():
        download_image(url, style_dir / filename)

    print("\n All images downloaded!")
    print(f"  Content: {content_dir}")
    print(f"  Styles:  {style_dir}")

if __name__ == "__main__":
    setup_test_data()