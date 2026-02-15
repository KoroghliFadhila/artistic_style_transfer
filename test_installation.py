"""Test complet de l'installation"""

def test_all_imports():
    print("🧪 Testing all packages...\n")
    
    packages = {
        "PyTorch": "import torch; print(f'✅ PyTorch {torch.__version__}')",
        "TorchVision": "import torchvision; print(f'✅ TorchVision {torchvision.__version__}')",
        "NumPy": "import numpy as np; print(f'✅ NumPy {np.__version__}')",
        "Pillow": "from PIL import Image; print(f'✅ Pillow {Image.__version__}')",
        "OpenCV": "import cv2; print(f'✅ OpenCV {cv2.__version__}')",
        "Matplotlib": "import matplotlib; print(f'✅ Matplotlib {matplotlib.__version__}')",
        "Seaborn": "import seaborn; print(f'✅ Seaborn {seaborn.__version__}')",
        "scikit-image": "import skimage; print(f'✅ scikit-image {skimage.__version__}')",
        "SciPy": "import scipy; print(f'✅ SciPy {scipy.__version__}')",
        "FastAPI": "import fastapi; print(f'✅ FastAPI {fastapi.__version__}')",
        "Uvicorn": "import uvicorn; print(f'✅ Uvicorn {uvicorn.__version__}')",
        "tqdm": "import tqdm; print(f'✅ tqdm {tqdm.__version__}')",
        "TensorBoard": "import tensorboard; print(f'✅ TensorBoard {tensorboard.__version__}')",
    }
    
    failed = []
    
    for name, code in packages.items():
        try:
            exec(code)
        except Exception as e:
            print(f"❌ {name}: {e}")
            failed.append(name)
    
    print("\n" + "="*60)
    if not failed:
        print("🎉 ALL PACKAGES INSTALLED SUCCESSFULLY!")
        print("✅ You're ready to start coding!")
    else:
        print(f"⚠️  Missing packages: {', '.join(failed)}")
        print("Run: pip install " + " ".join(failed.lower()))
    print("="*60)

def test_cuda():
    import torch
    print("\n🖥️  GPU Information:")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA Available")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print(f"✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  CUDA not available (running on CPU)")
    print("="*60)

def test_vgg19():
    import torch
    import torchvision.models as models
    
    print("\n🧠 Testing VGG19 Model:")
    print("="*60)
    
    try:
        vgg = models.vgg19(pretrained=True)
        print("✅ VGG19 model loaded successfully")
        print(f"✅ Model has {sum(p.numel() for p in vgg.parameters()):,} parameters")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = vgg(dummy_input)
        print(f"✅ Forward pass successful, output shape: {output.shape}")
    except Exception as e:
        print(f"❌ VGG19 test failed: {e}")
    
    print("="*60)

if __name__ == "__main__":
    test_all_imports()
    test_cuda()
    test_vgg19()
    
    print("\n🚀 Ready to start the Style Transfer project!")