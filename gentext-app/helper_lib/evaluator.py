import torch


def evaluate_model(model, data_loader, criterion, device='cpu'):
    model.to(device).eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Test loss: {avg_loss:.4f}  Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy
