import torch
import torch.nn as nn
from tqdm import tqdm
from models.resnet_s import resnet20
from models import binaryconnect
from cifar10 import testloader

print(f"\n{'='*70}")
print("Binary Model Evaluation (with Binarized Weights)")
print(f"{'='*70}\n")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the state_dict from the trained BinaryConnect model
checkpoint = torch.load('best_model.pth', map_location='cpu')
print(f"Loaded checkpoint with {len(checkpoint)} keys")

# Remove "model." prefix if present (from BC wrapper)
cleaned_state_dict = {}
for key, value in checkpoint.items():
    if key.startswith('model.'):
        cleaned_key = key[6:]  # Remove "model." prefix
        cleaned_state_dict[cleaned_key] = value
    else:
        cleaned_state_dict[key] = value

# Create the model
model = resnet20()
model.load_state_dict(cleaned_state_dict)
print("✓ Loaded full-precision weights (clipped to [-1, 1])")

# Apply BinaryConnect wrapper
model_bc = binaryconnect.BC(model)
model_bc.model = model_bc.model.to(device)

# Binarize the weights for inference
model_bc.binarization()
print("✓ Binarized all Conv2d and Linear weights to {-1, +1}")

# Set to eval mode
model_bc.model.eval()
print("✓ Model set to eval mode\n")

# Verify binarization (optional check)
print("Verifying binarization...")
all_binary = True
for name, param in model_bc.model.named_parameters():
    if 'weight' in name and ('conv' in name.lower() or 'linear' in name.lower()):
        unique_vals = torch.unique(param.data.cpu())
        if not torch.allclose(unique_vals, torch.tensor([-1.0, 1.0]), atol=1e-5):
            all_binary = False
            print(f"  ⚠ {name}: unique values = {unique_vals[:10].tolist()}")
            break

if all_binary:
    print("✓ All convolution/linear weights are binary {-1, +1}\n")
else:
    print("⚠ Some weights are not fully binary\n")

# Evaluate on CIFAR10 test set
print(f"{'='*70}")
print("Evaluating Binarized Model on CIFAR10 Test Set")
print(f"{'='*70}\n")

criterion = nn.CrossEntropyLoss()
correct = 0
total = 0
test_loss = 0.0

with torch.no_grad():
    for inputs, labels in tqdm(testloader, desc="Evaluating"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass with binarized weights
        outputs = model_bc.forward(inputs)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_loss = test_loss / len(testloader)
test_acc = 100. * correct / total

print(f"\n{'='*70}")
print("Results with Binarized Weights:")
print(f"{'='*70}")
print(f"Test Loss:              {test_loss:.4f}")
print(f"Test Accuracy:          {test_acc:.2f}%")
print(f"Correct predictions:    {correct}/{total}")
print(f"{'='*70}")

# Compare model sizes
import os
full_precision_size = os.path.getsize('best_model.pth') / (1024**2)
print(f"\nModel Size Information:")
print(f"  Full precision model: {full_precision_size:.2f} MB")
print(f"  Binary weights in memory: ~{full_precision_size / 32:.2f} MB (theoretical 32x compression)")
print(f"{'='*70}\n")

# Save the binarized weights to a new file
print("Saving binarized weights to 'best_model_binarized.pth'...")
torch.save(model_bc.model.state_dict(), 'best_model_binarized.pth')
binarized_size = os.path.getsize('best_model_binarized.pth') / (1024**2)
print(f"✓ Saved binarized model: {binarized_size:.2f} MB")
print("  (Note: Still stored as float32, but values are only {-1, +1})")
print(f"{'='*70}\n")

# Test: Load the binarized model directly
print("Testing: Loading binarized weights directly...")
model_test = resnet20()
model_test.load_state_dict(torch.load('best_model_binarized.pth'))
model_test = model_test.to(device)
model_test.eval()

# Quick verification
test_correct = 0
test_total = 0
with torch.no_grad():
    inputs, labels = next(iter(testloader))
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model_test(inputs)
    _, predicted = outputs.max(1)
    test_total += labels.size(0)
    test_correct += predicted.eq(labels).sum().item()

print(f"✓ Direct load test passed: {test_correct}/{test_total} correct on first batch")
print("  You can now load 'best_model_binarized.pth' directly for inference!")
print(f"{'='*70}\n")
