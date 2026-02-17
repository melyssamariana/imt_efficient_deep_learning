import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

from cifar10 import trainloader, testloader, trainloader_subset
from models.resnet import ResNet18, ResNet34, ResNet50
from models.resnet_s import resnet20
from models import binaryconnect

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
        model.binarization()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        model.restore()

        optimizer.step()

        model.clip()
        
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
            
            model.binarization()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            model.restore()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = running_loss / len(testloader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Keep a copy of the best weights in memory
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return True
        self.counter += 1
        return False

    @property
    def should_stop(self):
        return self.counter >= self.patience

def main():
    # Configuration
    num_epochs = 150
    learning_rate = 0.01
    warmup_epochs = 5
    early_stopping_patience = 150
    early_stopping_min_delta = 0.0
    label_smoothing = 0.1
    weight_decay = 1e-3
    project_name = "imt_efficient_deep_learning"
    
    # Device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create the model
    model = binaryconnect.BC(resnet20()).to(device)
    print(f"Model: {model.__class__.__name__}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True,
    )
    
    # Learning rate scheduler
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, num_epochs - warmup_epochs),
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )

    # Initialize Weights & Biases
    run = wandb.init(
        project=project_name,
        name=f"{model.__class__.__name__}_20_binaryconnect_01",
        config={
            "model": model.__class__.__name__ + " (ResNet20)",
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "warmup_epochs": warmup_epochs,
            "label_smoothing": label_smoothing,
            "optimizer": "SGD",
            "momentum": 0.9,
            "nesterov": True,
            "weight_decay": weight_decay,
            "scheduler": "LinearLR+CosineAnnealingLR",
            "t_max": num_epochs - warmup_epochs,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "device": str(device),
        },
    )
    wandb.watch(model, log="gradients", log_freq=100)
    
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
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
    )
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

        # Log metrics to Weights & Biases
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "lr": scheduler.get_last_lr()[0],
            }
        )
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ Best model saved! Accuracy: {best_acc:.2f}%\n")
            wandb.log({"best_acc": best_acc})
            wandb.save("best_model.pth")

        # Early stopping based on validation loss
        early_stopping.step(test_loss, model)
        if early_stopping.should_stop:
            print(
                f"Early stopping at epoch {epoch+1}. "
                f"Best val loss: {early_stopping.best_loss:.4f}"
            )
            break
    
    print(f"Training completed! Best accuracy: {best_acc:.2f}%")

    # Restore best weights by validation loss before saving final model
    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)
    
    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    print(f"Final model saved to 'final_model.pth'")
    wandb.save("final_model.pth")
    
    # Plot loss evolution
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(train_losses) + 1)
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
    wandb.save("loss_evolution.png")
    
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
    wandb.save("accuracy_evolution.png")
    wandb.finish()

if __name__ == '__main__':
    main()
