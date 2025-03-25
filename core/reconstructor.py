import torch.nn as nn


class CNNFrameReconstructor(nn.Module):
    def __init__(self, embed_dim=768, feature_dim=512, out_channels=3, img_size=224):
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
        x: (B, 50, embed_dim)
        Output: (B, 50, C, H, W)
        """
        B, T, D = x.shape
        x = x.view(B * T, D)
        x = self.fc(x)
        x = x.view(B * T, self.feature_dim, self.img_size // 16, self.img_size // 16)
        x = self.decoder(x)
        x = x.view(B, T, self.out_channels, self.img_size, self.img_size)
        return x
