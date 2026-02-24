"""
Example: How to load and use the binarized model directly

This shows that once weights are binarized and saved, you can load them
directly without needing the BinaryConnect wrapper or calling binarization().
"""

import torch
from models.resnet_s import resnet20
from cifar10 import testloader

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model and load pre-binarized weights
model = resnet20()
model.load_state_dict(torch.load('best_model_binarized.pth'))
model = model.to(device)
model.eval()

print("✓ Loaded binarized model directly (no BC wrapper needed)")

# Verify weights are binary
print("\nVerifying weights are binary {-1, +1}:")
for name, param in model.named_parameters():
    if 'conv' in name and 'weight' in name:
        unique_vals = torch.unique(param.data.cpu())
        print(f"  {name}: unique values = {unique_vals.tolist()[:5]}...")
        break

# Run inference
print("\nRunning inference on CIFAR10...")
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

accuracy = 100. * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")
print(f"Correct: {correct}/{total}")
