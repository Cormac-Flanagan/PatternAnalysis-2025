import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Unet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        self.c1 = DoubleConv3D(in_channels, 64)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.c2 = DoubleConv3D(64, 128)
        # maxpool
        self.c3 = DoubleConv3D(128, 256)
        # maxpool
        self.c4 = DoubleConv3D(256, 512)
        # maxpool
        self.c5 = DoubleConv3D(512, 1024)  # Turning Point merges with c4
        self.up3 = nn.ConvTranspose3d(1024, 1024, kernel_size=2, stride=2)
        self.c6 = DoubleConv3D(1024 + 512, 512)  # Modern
        self.up2 = nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2)
        self.c7 = DoubleConv3D(512 + 256, 256)
        self.up1 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.c8 = DoubleConv3D(256 + 128, 128)
        self.up0 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.c9 = DoubleConv3D(128 + 64, 64)

        self.final = nn.Conv3d(64, num_classes, kernel_size=1)

    def forward(self, x):
        r0 = self.c1(x)
        r1 = self.c2(self.maxpool(r0))
        r2 = self.c3(self.maxpool(r1))
        r3 = self.c4(self.maxpool(r2))
        r4 = self.c5(self.maxpool(r3))

        r3 = torch.cat([r3, self.up3(r4)], dim=1)
        r3 = self.c6(r3)

        r2 = torch.cat([r2, self.up2(r3)], dim=1)
        r2 = self.c7(r2)

        r1 = torch.cat([r1, self.up1(r2)], dim=1)
        r1 = self.c8(r1)

        r0 = torch.cat([r0, self.up0(r1)], dim=1)
        return self.final(self.c9(r0))
