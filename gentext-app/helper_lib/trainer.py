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


def train_energy(model, data_loader, optimizer, device='cpu', epochs=10, alpha=0.1,
                 langevin_steps=60, buffer_size=8192):
    # Contrastive divergence: push energy down on real images, up on samples
    # drawn from the model by Langevin dynamics. A replay buffer keeps the
    # chains persistent so they can get further from noise over training.
    from helper_lib.generator import langevin_sample

    model.to(device).train()
    buffer = torch.rand(buffer_size, 3, 32, 32) * 2 - 1
    for epoch in range(epochs):
        total = 0.0
        for real, _ in data_loader:
            real = real.to(device) + 0.005 * torch.randn_like(real.to(device))
            bs = real.size(0)

            # 5% fresh noise, the rest from the buffer
            idx = torch.randint(0, buffer_size, (bs,))
            start = buffer[idx].clone()
            fresh = torch.rand(bs) < 0.05
            start[fresh] = torch.rand(int(fresh.sum()), 3, 32, 32) * 2 - 1
            fake = langevin_sample(model, start.to(device), steps=langevin_steps)
            buffer[idx] = fake.cpu()

            model.train()
            e_real, e_fake = model(real), model(fake)
            # alpha term keeps energies bounded so training stays stable
            loss = (e_real - e_fake).mean() + alpha * (e_real ** 2 + e_fake ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch+1}/{epochs}  Loss: {total / len(data_loader):.4f}  "
              f"E(real): {e_real.mean().item():.3f}  E(fake): {e_fake.mean().item():.3f}")
    return model


def train_diffusion(model, data_loader, optimizer, device='cpu', epochs=10):
    model.to(device).train()
    for epoch in range(epochs):
        running_loss = 0.0
        for real, _ in data_loader:
            loss = model.loss(real.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}  Loss: {running_loss / len(data_loader):.4f}")
    return model
