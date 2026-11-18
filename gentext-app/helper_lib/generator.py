import io
import math

import torch
from PIL import Image
from torchvision.utils import make_grid

from helper_lib.model import NOISE_DIM


def _sample(model, device, num_samples):
    model.to(device).eval()
    with torch.no_grad():
        images = model(torch.randn(num_samples, NOISE_DIM, device=device)).cpu()
    return (images + 1) / 2  # [-1, 1] -> [0, 1]


def generate_samples(model, device, num_samples=10):
    import matplotlib.pyplot as plt

    images = _sample(model, device, num_samples)
    cols = min(num_samples, 5)
    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, img in zip(axes.flat, images):
        ax.imshow(img.squeeze(), cmap='gray')
    plt.show()


def generate_png(model, num_samples=10, device='cpu'):
    # Same sampling as generate_samples, returned as PNG bytes for the API
    images = _sample(model, device, num_samples)
    grid = make_grid(images, nrow=min(num_samples, 5), padding=2)
    array = (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _to_png(images, num_samples):
    # images in [-1, 1], shape (N, 3, H, W)
    grid = make_grid((images.cpu() + 1) / 2, nrow=min(num_samples, 5), padding=2)
    array = (grid.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def langevin_sample(model, x, steps=60, step_size=10.0, noise=0.005):
    # Gradient descent on the energy, with a little noise at each step
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    x = x.clone().requires_grad_(True)
    for _ in range(steps):
        x.data.add_(noise * torch.randn_like(x))
        x.data.clamp_(-1, 1)
        grad = torch.autograd.grad(model(x).sum(), x)[0]
        x.data.add_(-step_size * grad)
        x.data.clamp_(-1, 1)
    for p in model.parameters():
        p.requires_grad_(True)
    return x.detach()


def generate_energy_png(model, num_samples=10, device='cpu', steps=200):
    model.to(device)
    start = torch.rand(num_samples, 3, 32, 32, device=device) * 2 - 1
    return _to_png(langevin_sample(model, start, steps=steps), num_samples)


def generate_diffusion_png(model, num_samples=10, device='cpu'):
    model.to(device).eval()
    return _to_png(model.sample(num_samples, device), num_samples)
