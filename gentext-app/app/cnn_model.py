import io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

WEIGHTS_PATH = Path(__file__).parent / "cnn_cifar10.pt"

# Must match the preprocessing used for training in the notebook
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def load_model() -> CNN:
    model = CNN()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    return model


def predict(model: CNN, image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    batch = transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(batch), dim=1)[0]
    idx = int(probs.argmax())
    return {"class": CLASSES[idx], "confidence": round(float(probs[idx]), 4)}
