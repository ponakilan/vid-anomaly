import torch
import torch.nn.functional as F
from torch import nn

class MultiScaleTemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, device, scales=[2, 4, 8, 10]):
        super().__init__()
        self.scales = scales
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads).to(device) for _ in scales
        ])
    
    def forward(self, x):
        B, T, D = x.shape
        outputs = []

        for i, window_size in enumerate(self.scales):
            attn_device = next(self.attention_layers[i].parameters()).device
            if window_size >= T:
                q = k = v = x.transpose(0, 1).to(attn_device)
                attn_output, _ = self.attention_layers[i](q, k, v)
                outputs.append(attn_output.transpose(0, 1))
            else:
                local_outputs = []
                for start in range(T - window_size + 1):
                    chunk = x[:, start:start + window_size, :].to(attn_device)
                    q = k = v = chunk.transpose(0, 1)
                    attn_output, _ = self.attention_layers[i](q, k, v)
                    local_outputs.append(attn_output.mean(0))
                local_output = torch.stack(local_outputs, dim=1)
                local_output = F.interpolate(local_output.transpose(1, 2), size=T, mode='linear').transpose(1, 2)
                outputs.append(local_output)

        final_output = torch.stack(outputs, dim=0).mean(0)
        return final_output  # (B, T, D)


class SpatioTemporalAttention(nn.Module):
    def __init__(self, embed_dim=768, reduction=4):
        super(SpatioTemporalAttention, self).__init__()
        reduced_dim = embed_dim // reduction

        self.avg_pool_t = nn.AdaptiveAvgPool1d(1)

        self.coord_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, reduced_dim),
            nn.BatchNorm1d(reduced_dim),
            nn.ReLU(inplace=True)
        )
        self.fc_t = nn.Linear(reduced_dim, embed_dim)
        self.sigmoid_t = nn.Sigmoid()

        self.spatial_conv = nn.Conv1d(embed_dim, 1, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        B, T, C = x.size()

        x_pool_t = self.avg_pool_t(x.transpose(1, 2)).squeeze(-1)
        x_pool_c = x.mean(dim=1)

        concat_pool = torch.cat([x_pool_t, x_pool_c], dim=-1)
        coord_attn = self.coord_mlp(concat_pool)
        coord_attn = self.fc_t(coord_attn)
        coord_attn = self.sigmoid_t(coord_attn).unsqueeze(1)

        spatial_attn = self.spatial_conv(x.transpose(1, 2))
        spatial_attn = self.bn(spatial_attn)
        spatial_attn = self.relu(spatial_attn)
        spatial_attn = torch.sigmoid(spatial_attn)

        out = x * coord_attn
        out = out * spatial_attn.transpose(1, 2)

        return out
