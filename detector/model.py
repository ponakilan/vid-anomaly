import torch.nn as nn
from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig

class CNNFrameReconstructor(nn.Module):
    def __init__(self, embed_dim=1280, feature_dim=512, out_channels=3, img_size=224):
        super(CNNFrameReconstructor, self).__init__()
        self.img_size = img_size
        self.feature_dim = feature_dim
        self.out_channels = out_channels

        self.fc = nn.Linear(embed_dim, feature_dim * (img_size // 16) * (img_size // 16))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, feature_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feature_dim // 2, feature_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feature_dim // 4, feature_dim // 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feature_dim // 8, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (B, embed_dim) - Single embedding vector per batch item
        Output: (B, C, H, W) - Single frame per batch item
        """
        B, D = x.shape
        x = self.fc(x)
        x = x.view(B, self.feature_dim, self.img_size // 16, self.img_size // 16)
        x = self.decoder(x)
        return x


class FramePredictor(nn.Module):
    _reconstruct: bool

    def __init__(self, mae_backbone: str):
        super(FramePredictor, self).__init__()

        config = AutoConfig.from_pretrained(mae_backbone, trust_remote_code=True)
        self.processor = VideoMAEImageProcessor.from_pretrained(mae_backbone)
        self.video_mae = AutoModel.from_pretrained(mae_backbone, config=config, trust_remote_code=True)
        self._reconstruct = True

        self.reconstructor = CNNFrameReconstructor()

    def set_reconstruct(self, val):
        self._reconstruct = val

    def forward(self, x):
        videos = [list(sequence) for sequence in x]
        processed = self.processor(videos, return_tensors="pt")
        processed['pixel_values'] = processed['pixel_values'].permute(0, 2, 1, 3, 4).to(x.device)
        output = self.video_mae(**processed)
        if self._reconstruct:
            output = self.reconstructor(output)
        return output
