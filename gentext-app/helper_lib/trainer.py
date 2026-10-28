import torch

from helper_lib.model import NOISE_DIM


def train_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    model.to(device).train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}  Loss: {running_loss / len(data_loader):.4f}")
    return model


def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    # model is the (generator, discriminator) pair, optimizer is (opt_G, opt_D)
    G, D = model
    opt_G, opt_D = optimizer
    G.to(device).train()
    D.to(device).train()

    for epoch in range(epochs):
        for real, _ in data_loader:
            real = real.to(device)
            bs = real.size(0)
            real_labels = torch.ones(bs, 1, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)

            # Train discriminator
            fake = G(torch.randn(bs, NOISE_DIM, device=device))
            loss_D = criterion(D(real), real_labels) + criterion(D(fake.detach()), fake_labels)
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Train generator
            loss_G = criterion(D(fake), real_labels)
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"Epoch {epoch+1}/{epochs}  D loss: {loss_D.item():.4f}  G loss: {loss_G.item():.4f}")

    return model
