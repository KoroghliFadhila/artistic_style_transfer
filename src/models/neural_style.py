import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import copy

class VGGFeatures(nn.Module):
    """
    Extracteur de features VGG19 pour Neural Style Transfer
    """
    def __init__(self):
        super(VGGFeatures, self).__init__()
        
        # Charger VGG19 pré-entraîné
        vgg = models.vgg19(pretrained=True).features
        
        # Layers pour content et style
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(4)])   # relu1_2
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(4, 9)])  # relu2_2
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(9, 18)]) # relu3_4
        self.slice4 = nn.Sequential(*[vgg[i] for i in range(18, 27)]) # relu4_4
        self.slice5 = nn.Sequential(*[vgg[i] for i in range(27, 36)]) # relu5_4
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        
        return [h1, h2, h3, h4, h5]


def gram_matrix(features):
    """
    Calcule la matrice de Gram pour les features de style
    """
    batch, channels, height, width = features.size()
    features = features.view(batch * channels, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(batch * channels * height * width)


class NeuralStyleTransfer:
    """
    Implémentation de Neural Style Transfer (Gatys et al., 2016)
    """
    def __init__(
        self,
        content_weight=1.0,
        style_weight=1e6,
        tv_weight=1e-3,
        device='cuda'
    ):
        self.device = device
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        
        # Charger VGG
        self.vgg = VGGFeatures().to(device)
        
        # Normalisation ImageNet
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def load_image(self, image_path, max_size=512):
        """Charge et préprocess une image"""
        image = Image.open(image_path).convert('RGB')
        
        # Resize si nécessaire
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.LANCZOS)
        
        # Convertir en tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        image = transform(image).unsqueeze(0)
        return image.to(self.device)
    
    def save_image(self, tensor, path):
        """Sauvegarde un tensor en image"""
        image = tensor.cpu().clone().squeeze(0)
        image = image.clamp(0, 1)
        
        transform = transforms.ToPILImage()
        image = transform(image)
        image.save(path)
    
    def compute_content_loss(self, content_features, generated_features):
        """Loss de contenu"""
        return F.mse_loss(generated_features[3], content_features[3])
    
    def compute_style_loss(self, style_features, generated_features):
        """Loss de style (sur toutes les layers)"""
        style_loss = 0
        
        for style_feat, gen_feat in zip(style_features, generated_features):
            style_gram = gram_matrix(style_feat)
            gen_gram = gram_matrix(gen_feat)
            style_loss += F.mse_loss(gen_gram, style_gram)
        
        return style_loss
    
    def compute_tv_loss(self, image):
        """Total Variation Loss pour régularisation"""
        tv_h = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]).sum()
        tv_w = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]).sum()
        return tv_h + tv_w
    
    def transfer(
        self,
        content_path,
        style_path,
        output_path,
        num_steps=300,
        learning_rate=0.003,
        log_interval=50
    ):
        """
        Effectue le style transfer
        """
        print(f"Starting Neural Style Transfer...")
        print(f"Content: {content_path}")
        print(f"Style: {style_path}")
        
        # Charger images
        content_img = self.load_image(content_path)
        style_img = self.load_image(style_path)
        
        # Initialiser generated image (clone du content)
        generated_img = content_img.clone().requires_grad_(True)
        
        # Optimizer
        optimizer = torch.optim.Adam([generated_img], lr=learning_rate)
        
        # Extract features (pre-compute)
        with torch.no_grad():
            content_features = self.vgg(self.normalize(content_img))
            style_features = self.vgg(self.normalize(style_img))
        
        # Optimization loop
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Forward pass
            generated_features = self.vgg(self.normalize(generated_img))
            
            # Compute losses
            content_loss = self.compute_content_loss(
                content_features, generated_features
            )
            style_loss = self.compute_style_loss(
                style_features, generated_features
            )
            tv_loss = self.compute_tv_loss(generated_img)
            
            # Total loss
            total_loss = (
                self.content_weight * content_loss +
                self.style_weight * style_loss +
                self.tv_weight * tv_loss
            )
            
            # Backward
            total_loss.backward()
            optimizer.step()
            
            # Clamp values
            with torch.no_grad():
                generated_img.clamp_(0, 1)
            
            # Log
            if step % log_interval == 0:
                print(f"Step {step}/{num_steps} | "
                      f"Content: {content_loss.item():.2f} | "
                      f"Style: {style_loss.item():.2f} | "
                      f"TV: {tv_loss.item():.2f}")
        
        # Sauvegarder
        self.save_image(generated_img, output_path)
        print(f"Saved to {output_path}")
        
        return generated_img


# Test rapide
if __name__ == "__main__":
    nst = NeuralStyleTransfer()
    
    nst.transfer(
        content_path="data/processed/content/test1.jpg",
        style_path="data/processed/styles/impressionism/style1.jpg",
        output_path="results/visualizations/neural_style_output.jpg",
        num_steps=300
    )