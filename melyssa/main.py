import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from cifar10 import trainloader, testloader, trainloader_subset
from models.restnet8_cifar import ResNet8

def train_epoch(model, trainloader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(trainloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, testloader, criterion, device):
    """Evaluate the model on the test set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = running_loss / len(testloader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

def main():
    # Configuration
    num_epochs = 80
    learning_rate = 0.1
    
    # Device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create the model
    #model = ResNet18() 
    model = DenseNet121()
    model = model.to(device)
    print(f"Model: {model.__class__.__name__}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Choose the dataloader (trainloader or trainloader_subset)
    train_data = trainloader  # Use trainloader for full dataset
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Training samples: {len(train_data.dataset)}")
    print(f"Test samples: {len(testloader.dataset)}\n")
    
    # Lists to store losses for plotting
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
  

  
    # Training loop
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_data, criterion, optimizer, device)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        
        # Store losses
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Print results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}\n")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ Best model saved! Accuracy: {best_acc:.2f}%\n")
    
    print(f"Training completed! Best accuracy: {best_acc:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    print(f"Final model saved to 'final_model.pth'")
    
    # Plot loss evolution
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, num_epochs + 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs_range, test_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Evolution', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_evolution.png', dpi=300, bbox_inches='tight')
    print(f"Loss evolution plot saved to 'loss_evolution.png'")
    plt.close()
    
    # Plot accuracy evolution
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs_range, test_accs, 'r-', label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training and Validation Accuracy Evolution', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('accuracy_evolution.png', dpi=300, bbox_inches='tight')
    print(f"Accuracy evolution plot saved to 'accuracy_evolution.png'")
    plt.close()
   
if __name__ == '__main__':
    main()
