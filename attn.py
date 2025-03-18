import torch
import torch.nn as nn
import torch.nn.functional as F

class STRAttention(nn.Module):
    def __init__(self, vit_embedding_dim, reduction_ratio=4):
        super(STRAttention, self).__init__()
        reduced_channels = vit_embedding_dim // reduction_ratio
        
        # Spatial Attention (1D since ViT outputs a sequence)
        self.conv_s = nn.Conv1d(vit_embedding_dim, reduced_channels, kernel_size=3, padding=1)
        self.conv1x1_s = nn.Conv1d(reduced_channels, vit_embedding_dim, kernel_size=1)
        
        # Temporal Attention (3D conv over time dimension)
        self.conv_t = nn.Conv1d(vit_embedding_dim, reduced_channels, kernel_size=3, padding=1)
        self.conv1x1_t = nn.Conv1d(reduced_channels, vit_embedding_dim, kernel_size=1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, vit_encodings):
        # Input shape: (Batch, Time, Channels) from ViT encoder
        b, t, c = vit_encodings.shape
        x = vit_encodings.permute(0, 2, 1)  # Change to (Batch, Channels, Time)
        
        # Spatial Attention
        attn_s = self.conv_s(x)
        attn_s = self.conv1x1_s(attn_s)
        attn_s = self.sigmoid(attn_s)
        
        # Temporal Attention
        attn_t = self.conv_t(x)
        attn_t = self.conv1x1_t(attn_t)
        attn_t = self.sigmoid(attn_t)
        
        # Apply attention weights
        x = x * attn_s * attn_t
        return x.permute(0, 2, 1)  # Back to (Batch, Time, Channels)


batch_size = 2
time_steps = 8
vit_embedding_dim = 768  # ViT output size

vit_encodings = torch.randn(batch_size, time_steps, vit_embedding_dim)  # Simulated ViT output
str_attention = STRAttention(vit_embedding_dim)
output = str_attention(vit_encodings)

print("Input shape:", vit_encodings.shape)  # (2, 8, 768)
print("Output shape:", output.shape)        # (2, 8, 768)  # Should match input shape
