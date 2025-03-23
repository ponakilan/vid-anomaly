from attention import MultiScaleTemporalAttention
from reconstructor import CNNFrameReconstructor

from torch import nn


class FrameReconstructionModel(nn.Module):
    def __init__(self):
        super(FrameReconstructionModel, self).__init__()
        self.attn = MultiScaleTemporalAttention(
            embed_dim=768,
            num_heads=4,
            scales=[10, 20, 40, 50]
        )
        self.reconstructor = CNNFrameReconstructor()

    def forward(self, x):
        x = self.attn(x)
        x = self.reconstructor(x)
        return x.float()   
    