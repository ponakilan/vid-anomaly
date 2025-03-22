import torch
import torch.nn.functional as F
from torch import nn

class MultiScaleTemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, scales=[2, 4, 8, 10]):
        super().__init__()
        self.scales = scales
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads) for _ in scales
        ])
    
    def forward(self, x):
        B, T, D = x.shape
        outputs = []

        for i, window_size in enumerate(self.scales):
            if window_size >= T:
                q = k = v = x.transpose(0, 1)
                attn_output, _ = self.attention_layers[i](q, k, v)
                outputs.append(attn_output.transpose(0, 1))
            else:
                local_outputs = []
                for start in range(T - window_size + 1):
                    chunk = x[:, start:start + window_size, :]
                    q = k = v = chunk.transpose(0, 1)
                    attn_output, _ = self.attention_layers[i](q, k, v)
                    local_outputs.append(attn_output.mean(0))
                local_output = torch.stack(local_outputs, dim=1)
                local_output = F.interpolate(local_output.transpose(1, 2), size=T, mode='linear').transpose(1, 2)
                outputs.append(local_output)

        final_output = torch.stack(outputs, dim=0).mean(0)
        return final_output  # (B, T, D)
