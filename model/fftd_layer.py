import torch.nn as nn
import torch.fft
import torch
from torch.nn import TransformerEncoderLayer, TransformerEncoder

class AdaptiveFourierConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(AdaptiveFourierConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

        # Learnable parameters for frequency filtering
        self.cutoff_ratio = nn.Parameter(torch.tensor(0.7))
        self.scale_factor = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.size()

        # Fourier transform
        x_fft = torch.fft.fft2(x)
        real = torch.real(x_fft)
        imag = torch.imag(x_fft)

        # Adaptive cutoff
        cutoff_ratio = torch.sigmoid(self.cutoff_ratio)
        scale_factor = self.scale_factor

        h_cutoff = int(h * cutoff_ratio.item() * scale_factor.item())
        w_cutoff = int(w * cutoff_ratio.item() * scale_factor.item())

        # Low-pass filtering
        real[..., h_cutoff:-h_cutoff, w_cutoff:-w_cutoff] = 0
        imag[..., h_cutoff:-h_cutoff, w_cutoff:-w_cutoff] = 0

        # Inverse Fourier transform
        x_ifft = torch.fft.ifft2(real + 1j * imag)
        x_processed = torch.real(x_ifft)
        return x_processed


class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_dim, rank, num_heads=4, num_layers=6):
        super(TransformerFeatureExtractor, self).__init__()
        self.rank = rank
        self.input_dim = input_dim

        encoder_layers = TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=num_layers)

    def create_positional_encoding(self, h, w):
        # Generate 2D positional encoding
        position = torch.stack(
            torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij'), dim=-1
        ).float()

        pe = torch.zeros(h, w, self.input_dim)
        for i in range(self.input_dim // 2):
            pe[:, :, 2 * i] = torch.sin(position[:, :, 0] / (10000 ** (2 * i / self.input_dim)))
            pe[:, :, 2 * i + 1] = torch.cos(position[:, :, 1] / (10000 ** (2 * i / self.input_dim)))

        return pe.unsqueeze(0)

    def forward(self, x):
        b, c, h, w = x.size()

        # Flatten feature map
        x_flat = x.view(b, c, -1).permute(2, 0, 1)

        # Add positional encoding
        pos_enc = self.create_positional_encoding(h, w).expand(b, -1, -1, -1)
        pos_enc_flat = pos_enc.reshape(b, c, -1).permute(2, 0, 1).to(x_flat.device)
        x_flat = x_flat + pos_enc_flat

        # Transformer encoding
        x_transformed = self.transformer(x_flat)

        # Reshape back to feature map
        x_out = x_transformed.permute(1, 2, 0).contiguous().view(b, c, h, w)
        return x_out
