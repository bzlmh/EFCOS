import torch.nn as nn
import torch.nn.functional as F
import torch

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.Conv2d(dim // reduction, dim // reduction, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn

class DUC(nn.Module):
    def __init__(self, in_channel, out_channel, factor):
        super(DUC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(1, 1), stride=(1, 1))
        self.channel_attention = ChannelAttention(out_channel)
        self.pixshuffle = nn.PixelShuffle(upscale_factor=factor)
        self.conv2 = nn.Conv2d(in_channels=in_channel//4, out_channels=out_channel, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = x * self.channel_attention(x)
        x = self.pixshuffle(x)
        x = self.conv2(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention, self).__init__()
        self.conv1x1 = nn.Conv2d(2, 2, 1, bias=True)
        self.relu = nn.ReLU()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        x2 = self.conv1x1(x2)
        x2 = self.relu(x2)
        sattn = self.sa(x2)
        return sattn

class HDCblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HDCblock, self).__init__()
        self.dilate1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=4, padding=4)
        self.spatial = SpatialAttention(in_channels)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = F.relu(self.dilate1(x), inplace=True)
        dilate2_out = F.relu(self.dilate2(dilate1_out), inplace=True)
        dilate3_out = F.relu(self.dilate3(dilate2_out), inplace=True)

        out = x + dilate1_out + dilate2_out + dilate3_out
        out = out * self.spatial(out)
        out = self.downsample(out)
        return out
