"""
Neural Style Transfer - Gatys et al. (2016)
Optimisé pour CPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# VGG Feature Extractor
# ============================================================

class VGGFeatures(nn.Module):
    """
    Extrait les features de VGG19 pour le style transfer
    """
    
    CONTENT_LAYERS = {'21': 'relu4_2'}  # Pour content
    STYLE_LAYERS = {
        '0':  'relu1_1',
        '5':  'relu2_1',
        '10': 'relu3_1',
        '19': 'relu4_1',
        '28': 'relu5_1',
    }
    
    def __init__(self):
        super(VGGFeatures, self).__init__()
        
        print("   Loading VGG19...")
        vgg = models.vgg19(weights='DEFAULT').features
        
        self.layers = nn.ModuleDict()
        for name, layer in vgg.named_children():
            self.layers[name] = layer
        
        # Freeze
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
        print("   VGG19 loaded!")
    
    def forward(self, x):
        content_features = {}
        style_features = {}
        
        for name, layer in self.layers.items():
            x = layer(x)
            
            if name in self.CONTENT_LAYERS:
                content_features[self.CONTENT_LAYERS[name]] = x
            
            if name in self.STYLE_LAYERS:
                style_features[self.STYLE_LAYERS[name]] = x
        
        return content_features, style_features


# ============================================================
# Gram Matrix
# ============================================================

def gram_matrix(tensor):
    """
    Calcule la matrice de Gram pour capturer le style
    """
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    gram = torch.mm(features, features.t())
    return gram.div(b * c * h * w)


# ============================================================
# Neural Style Transfer
# ============================================================

class NeuralStyleTransfer:
    """
    Implémentation complète du Neural Style Transfer
    Optimisé CPU avec tqdm progress bar
    """
    
    def __init__(self):
        self.device = torch.device('cpu')
        
        print("\n  Device: CPU")
        print("  Note: Slower than GPU but fully functional\n")
        
        self.vgg = VGGFeatures().to(self.device)
        
        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def load_image(self, path, max_size=256):
        """
        Charge et prépare une image
        max_size=256 pour CPU (rapide)
        max_size=512 pour meilleure qualité (lent)
        """
        img = Image.open(path).convert('RGB')
        
        # Resize
        w, h = img.size
        if max(w, h) > max_size:
            if w > h:
                new_w, new_h = max_size, int(max_size * h / w)
            else:
                new_w, new_h = int(max_size * w / h), max_size
            img = img.resize((new_w, new_h), Image.LANCZOS)
        
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze(0).to(self.device)
        return tensor
    
    def save_image(self, tensor, path):
        """Sauvegarde un tensor comme image"""
        img = tensor.cpu().clone().squeeze(0)
        img = img.clamp(0, 1)
        img = transforms.ToPILImage()(img)
        img.save(path)
        print(f"\n   Saved: {path}")
        return img
    
    def compute_content_loss(self, content, generated):
        """MSE entre features de contenu"""
        return F.mse_loss(
            generated['relu4_2'],
            content['relu4_2']
        )
    
    def compute_style_loss(self, style, generated):
        """MSE entre matrices de Gram"""
        loss = 0
        
        for layer in style:
            if layer in generated:
                style_gram = gram_matrix(style[layer])
                gen_gram   = gram_matrix(generated[layer])
                loss += F.mse_loss(gen_gram, style_gram)
        
        return loss / len(style)
    
    def compute_tv_loss(self, image):
        """Total Variation pour régularisation"""
        tv_h = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]).mean()
        tv_w = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]).mean()
        return tv_h + tv_w
    
    def transfer(
        self,
        content_path,
        style_path,
        output_path="results/visualizations/output.jpg",
        image_size=256,      # 256 pour CPU (rapide)
        num_steps=200,       # 200 steps sur CPU
        content_weight=1.0,
        style_weight=1e6,
        tv_weight=1e-3,
        learning_rate=0.05,
        save_every=50,       # Sauvegarde intermédiaire
    ):
        """
        Lance le Style Transfer
        """
        
        print("=" * 60)
        print(" NEURAL STYLE TRANSFER")
        print("=" * 60)
        print(f"  Content: {content_path}")
        print(f"  Style:   {style_path}")
        print(f"  Size:    {image_size}px")
        print(f"  Steps:   {num_steps}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Créer dossier output
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        intermediate_dir = Path("results/visualizations/intermediate")
        intermediate_dir.mkdir(exist_ok=True)
        
        # Charger images
        print("\n Loading images...")
        content_img = self.load_image(content_path, image_size)
        style_img   = self.load_image(style_path, image_size)
        
        print(f"  Image size: {content_img.shape[2]}x{content_img.shape[3]}")
        
        # Initialiser image générée (clone du contenu)
        generated_img = content_img.clone().requires_grad_(True)
        
        # Optimizer LBFGS (meilleur pour NST)
        optimizer = torch.optim.LBFGS(
            [generated_img],
            lr=learning_rate,
            max_iter=1
        )
        
        # Extraire features fixes
        print("\n Extracting features...")
        with torch.no_grad():
            content_f, _ = self.vgg(self.normalize(content_img))
            _, style_f   = self.vgg(self.normalize(style_img))
        
        # Historique des losses
        history = {
            'content_loss': [],
            'style_loss': [],
            'tv_loss': [],
            'total_loss': []
        }
        
        print("\n⚡ Optimizing...\n")
        
        # Barre de progression
        pbar = tqdm(
            range(num_steps),
            desc="Style Transfer",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        step = [0]
        
        def closure():
            """Closure pour LBFGS"""
            optimizer.zero_grad()
            
            # Clamp values
            with torch.no_grad():
                generated_img.clamp_(0, 1)
            
            # Forward pass
            gen_content_f, gen_style_f = self.vgg(
                self.normalize(generated_img)
            )
            
            # Losses
            c_loss = content_weight * self.compute_content_loss(
                content_f, gen_content_f
            )
            s_loss = style_weight * self.compute_style_loss(
                style_f, gen_style_f
            )
            tv_loss = tv_weight * self.compute_tv_loss(generated_img)
            
            total_loss = c_loss + s_loss + tv_loss
            
            total_loss.backward()
            
            # Sauvegarder historique
            history['content_loss'].append(c_loss.item())
            history['style_loss'].append(s_loss.item())
            history['tv_loss'].append(tv_loss.item())
            history['total_loss'].append(total_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'C': f'{c_loss.item():.3f}',
                'S': f'{s_loss.item():.3f}',
                'TV': f'{tv_loss.item():.4f}'
            })
            pbar.update(1)
            
            # Sauvegarder intermédiaire
            if step[0] % save_every == 0 and step[0] > 0:
                intermediate_path = intermediate_dir / f"step_{step[0]:04d}.jpg"
                self.save_image(generated_img, str(intermediate_path))
            
            step[0] += 1
            return total_loss
        
        # Optimization loop
        for _ in range(num_steps):
            optimizer.step(closure)
        
        pbar.close()
        
        # Temps total
        elapsed = time.time() - start_time
        print(f"\n⏱  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        
        # Sauvegarder résultat final
        self.save_image(generated_img, output_path)
        
        return generated_img, history


# ============================================================
# Visualisation des Résultats
# ============================================================

def visualize_results(
    content_path,
    style_path,
    output_path,
    history,
    save_path="results/visualizations/comparison.jpg"
):
    """
    Crée une visualisation comparative des résultats
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # Images
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Courbes de loss
    ax4 = fig.add_subplot(gs[1, :])
    
    # Charger images
    content = Image.open(content_path)
    style   = Image.open(style_path)
    output  = Image.open(output_path)
    
    ax1.imshow(content)
    ax1.set_title(" Content Image", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(style)
    ax2.set_title(" Style Image", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    ax3.imshow(output)
    ax3.set_title(" Generated Image", fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Loss curves
    steps = range(len(history['total_loss']))
    ax4.plot(steps, history['total_loss'],   label='Total Loss',   color='#667eea', linewidth=2)
    ax4.plot(steps, history['content_loss'], label='Content Loss', color='#43e97b', linewidth=2)
    ax4.plot(steps, history['style_loss'],   label='Style Loss',   color='#fa709a', linewidth=2)
    ax4.plot(steps, history['tv_loss'],      label='TV Loss',      color='#fee140', linewidth=2)
    
    ax4.set_xlabel('Steps', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.suptitle('Neural Style Transfer Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Sauvegarder
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f" Comparison saved: {save_path}")