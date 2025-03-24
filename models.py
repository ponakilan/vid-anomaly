from attention import MultiScaleTemporalAttention, SpatioTemporalAttention
from reconstructor import CNNFrameReconstructor

from torch import nn


class FrameReconstructionModel(nn.Module):
    def __init__(self, device):
        super(FrameReconstructionModel, self).__init__()
        self.attn = MultiScaleTemporalAttention(
            embed_dim=768,
            num_heads=4,
            device=device,
            scales=[10, 20, 40, 50]
        ).to(device)
        self.reconstructor = CNNFrameReconstructor().to(device)

    def forward(self, x):
        x = self.attn(x)
        x = self.reconstructor(x)
        return x.float()


class StaModel(nn.Module):
    def __init__(self, device):
        super(StaModel, self).__init__()
        self.attn = SpatioTemporalAttention(
            embed_dim=768,
        ).to(device)
        self.reconstructor = CNNFrameReconstructor().to(device)

    def forward(self, x):
        x = self.attn(x)
        x = self.reconstructor(x)
        return x.float()
    