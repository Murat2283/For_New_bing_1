
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
num_classes = 10
patch_size = 16         # Patch size 
hidden_size = 512       # Transformer hidden size
n_layers = 6            # Transformer layers
n_heads = 8             # Transformer heads

# Transformer Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.MultiheadAttention(hidden_size, n_heads)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        attn_out = self.attn(x, x, x)[0]
        x = x + self.norm1(attn_out)
        x = x + self.norm2(self.fc2(self.fc1(x)))
        return x

# Transformer Encoder
class Encoder(nn.Module):
    def __init__(self, n_layers, hidden_size, n_heads):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(hidden_size, n_heads) for _ in range(n_layers)])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ViT模型    
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, hidden_size, n_layers, n_heads, num_classes):
        super().__init__()
        assert image_size % patch_size == 0    # Image size必须是patch size的整数倍
        n_patches = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.positions = nn.Parameter(torch.randn((1, n_patches + 1, hidden_size)))
        self.encoder = Encoder(n_layers, hidden_size, n_heads)  
        self.classifier = nn.Linear(hidden_size, num_classes) 
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embedding(x)    # (B, C, H, W) -> (B, C, H/patch_size, W/patch_size)
        x = torch.flatten(x, start_dim=2)     # (B, C, n_patches)
        cls_token = self.cls_token.repeat(batch_size, 1, 1) 
        x = torch.cat((cls_token, x), dim=1)  # (B, 1 + n_patches, C)
        x += self.positions
        x = self.encoder(x)      # (B, 1 + n_patches, C)
        x = x[:, 0]      
        x = self.classifier(x)   # (B, num_classes)
        return x