import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 10
NOISE_DIM = 100


class FCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, NUM_CLASSES)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # logits, for CrossEntropyLoss


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class EnhancedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 128 * 2 * 2)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class Generator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM):
        super().__init__()
        self.fc = nn.Linear(noise_dim, 7 * 7 * 128)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))


class EnergyModel(nn.Module):
    """Maps a 3x32x32 image in [-1, 1] to a scalar energy (low = realistic)."""

    def __init__(self):
        super().__init__()
        # Swish (SiLU) is smooth, which Langevin sampling needs; no BatchNorm
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.SiLU(),    # 16x16
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),   # 8x8
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),  # 4x4
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.SiLU(),  # 2x2
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class DiffusionUNet(nn.Module):
    """Small U-Net that predicts the noise added to an image at timestep t."""

    def __init__(self, base=64, t_dim=128):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.inp = nn.Conv2d(3, base, 3, padding=1)
        self.down1 = ResBlock(base, base, t_dim)            # 32x32
        self.down2 = ResBlock(base, base * 2, t_dim)        # 16x16
        self.mid = ResBlock(base * 2, base * 2, t_dim)      # 8x8
        self.up2 = ResBlock(base * 4, base, t_dim)          # 16x16
        self.up1 = ResBlock(base * 2, base, t_dim)          # 32x32
        self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 3, 3, padding=1))
        self.pool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t):
        t_emb = self.t_mlp(timestep_embedding(t, self.t_dim))
        x = self.inp(x)
        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)
        m = self.mid(self.pool(d2), t_emb)
        u2 = self.up2(torch.cat([self.upsample(m), d2], dim=1), t_emb)
        u1 = self.up1(torch.cat([self.upsample(u2), d1], dim=1), t_emb)
        return self.out(u1)


class DiffusionModel(nn.Module):
    """DDPM: linear beta schedule, noise-prediction U-Net, ancestral sampling."""

    def __init__(self, timesteps=200, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.timesteps = timesteps
        self.unet = DiffusionUNet()
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)

    def forward(self, x, t):
        return self.unet(x, t)

    def loss(self, x0):
        t = torch.randint(0, self.timesteps, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        ab = self.alphas_bar[t][:, None, None, None]
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        return F.mse_loss(self.unet(xt, t), noise)

    @torch.no_grad()
    def sample(self, num_samples, device="cpu"):
        x = torch.randn(num_samples, 3, 32, 32, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            eps = self.unet(x, t)
            alpha, ab, beta = self.alphas[i], self.alphas_bar[i], self.betas[i]
            x = (x - beta / (1 - ab).sqrt() * eps) / alpha.sqrt()
            if i > 0:
                x = x + beta.sqrt() * torch.randn_like(x)
        return x.clamp(-1, 1)


def get_model(model_name):
    if model_name == "FCNN":
        model = FCNN()
    elif model_name == "CNN":
        model = SimpleCNN()
    elif model_name == "EnhancedCNN":
        model = EnhancedCNN()
    elif model_name == "GAN":
        model = (Generator(), Discriminator())
    elif model_name == "EnergyModel":
        model = EnergyModel()
    elif model_name == "Diffusion":
        model = DiffusionModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model
