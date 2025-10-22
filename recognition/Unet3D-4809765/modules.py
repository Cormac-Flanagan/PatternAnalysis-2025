import torch
import torch.nn as nn
from torch.nn.functional import interpolate


class Diceloss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, output, target, eps=1e-6):
        """
        output: [batch, channels, D, H, W], softmax probabilities
        target: [batch, channels, D, H, W], one-hot
        """
        # sum over spatial dimensions
        dims = (2, 3, 4)
        intersection = (output * target).sum(dim=dims)
        union = output.sum(dim=dims) + target.sum(dim=dims)

        dice_per_class = (intersection + eps) / (union + eps)  # [batch, channels]

        # average over classes and batches, apply -2 factor
        loss = -2 * dice_per_class.mean()
        return loss


class ContextModule(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.Dropout3d(0.3),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_chanels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_chanels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class Localization(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, 1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Unet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1)
        self.tex1 = ContextModule(16)

        self.conv2 = nn.Conv3d(16, 32, 3, 2, 1)
        self.tex2 = ContextModule(32)

        self.conv3 = nn.Conv3d(32, 64, 3, 2, 1)
        self.tex3 = ContextModule(64)

        self.conv4 = nn.Conv3d(64, 128, 3, 2, 1)
        self.tex4 = ContextModule(128)

        self.conv5 = nn.Conv3d(128, 256, 3, 2, 1)
        self.tex5 = ContextModule(256)

        self.up5 = Upsample(256, 128)

        self.loc4 = Localization(256, 128)
        self.up4 = Upsample(128, 64)

        self.loc3 = Localization(128, 64)
        self.seg3 = nn.Conv3d(64, num_classes, 1)
        self.up3 = Upsample(64, 32)

        self.loc2 = Localization(64, 32)
        self.seg2 = nn.Conv3d(32, num_classes, 1)
        self.up2 = Upsample(32, 16)

        self.convF = nn.Conv3d(32, 32, 3, 1, 1)
        self.final = nn.Softmax(1)

    def forward(self, r0):
        r0 = self.conv1(r0)  # N
        r0 = r0 + self.tex1(r0)
        r1 = self.conv2(r0)  # N/2
        r1 = r1 + self.tex2(r1)
        r2 = self.conv3(r1)  # N/4
        r2 = r2 + self.tex3(r2)
        r3 = self.conv4(r2)  # N/8
        r3 = r3 + self.tex4(r3)
        r4 = self.conv5(r3)  # N/16
        r4 = r4 + self.tex5(r4)
        r3 = torch.cat([r3, self.up5(r4)], dim=1)

        torch.cuda.empty_cache()
        temp = self.loc4(r3)
        r3 = self.up4(temp)

        r2 = torch.cat([r3, r2], dim=1)
        torch.cuda.empty_cache()
        r2 = self.loc3(r2)
        r1 = torch.cat([r1, self.up3(r2)], dim=1)
        r1 = self.loc2(r1)
        r0 = torch.cat([r0, self.up2(r1)], dim=1)
        r1 = self.seg2(r1)
        r0 = self.seg2(self.convF(r0))

        r0 = r0 + interpolate(
            r1 + interpolate(self.seg3(r2), scale_factor=2, mode="nearest"),
            scale_factor=2,
            mode="nearest",
        )

        return self.final(r0)
