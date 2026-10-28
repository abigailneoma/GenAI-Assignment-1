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
