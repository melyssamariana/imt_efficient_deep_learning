import torch
import torch.nn as nn
from tqdm import tqdm
from models.resnet_s import resnet20
from cifar10 import testloader

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the state_dict from the wandb checkpoint (actual trained weights)
state_dict = torch.load('trys/best/best_model.pth')
print(f"Loaded state_dict with keys: {list(state_dict.keys())[:5]}...")

# If the checkpoint comes from the BinaryConnect wrapper, remove "model." prefix.
if any(key.startswith("model.") for key in state_dict.keys()):
    state_dict = {key.removeprefix("model."): value for key, value in state_dict.items()}

# Create the model
model = resnet20()

# Load the trained parameters into the model
model.load_state_dict(state_dict)
model = model.to(device)

# Convert the model to half precision (float16) for faster inference
# model = model.half()
# print("Converted model to half precision (float16).")

# If you use this model for inference (= no further training), you need to set it into eval mode
model.eval()
print("Model loaded and set to eval mode.")

# Evaluate on CIFAR10 test set
print("\n" + "="*50)
print("Testing model on CIFAR10...")
print("="*50)

criterion = nn.CrossEntropyLoss()
correct = 0
total = 0
test_loss = 0.0

with torch.no_grad():
    for inputs, labels in tqdm(testloader, desc="Evaluating"):
        inputs, labels = inputs.to(device), labels.to(device)
        # inputs = inputs.half()  # Convert inputs to half precision (float16)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_loss = test_loss / len(testloader)
test_acc = 100. * correct / total

print("\n" + "="*50)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Correct predictions: {correct}/{total}")
print("="*50)
