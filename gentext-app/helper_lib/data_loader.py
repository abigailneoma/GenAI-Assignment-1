import torch
from torchvision import datasets, transforms


def get_data_loader(data_dir, batch_size=32, train=True, dataset="CIFAR10"):
    if dataset == "MNIST":
        # Scaled to [-1, 1] to match the GAN generator's Tanh output
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        data = datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)
    elif dataset == "CIFAR10":
        data = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transforms.ToTensor())
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=train, drop_last=train)
    return loader
