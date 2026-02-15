import os
import urllib.request
import zipfile

print("Downloading datasets...")

# Créer dossiers
os.makedirs("data/raw/wikiart", exist_ok=True)
os.makedirs("data/raw/coco", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/raw/test_images", exist_ok=True)

print("Downloading sample image...")

url = "https://raw.githubusercontent.com/pytorch/examples/master/fast_neural_style/images/content-images/amber.jpg"
save_path = "data/raw/test_images/content1.jpg"

urllib.request.urlretrieve(url, save_path)

print("Download complete!")
