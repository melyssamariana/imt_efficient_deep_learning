from pyexpat import model
from sched import scheduler
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.resnet import ResNet18
import wandb

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

class ConfigModel:
    def __init__(self,
                # basic info
                label="t1",
                model=ResNet18(),
                train_data=None,
                test_data=None,
                project_name = "imt_efficient_dl",
                path_backup = "./",
                input_dtype = torch.float32,
                wand_on = True,
                batch_size=128,
                
                # hyperparameters
                num_epochs=150,
                learning_rate=0.01,
                weight_decay=1e-3,
                optimizer_name='SGD',
                momentum=0.9,
                nesterov=True,
                label_smoothing=0.1,

                # training settings
                warmup_epochs=5,
                scheduler_name="LinearLR+CosineAnnealingLR",
                early_stopping_patience=150,
                early_stopping_min_delta=0.0,

                # pruning settings
                pruning_method="combined",  # "unstructured", "structured", or "combined"
                structured_ratios=[0.25],  # structured pruning ratios
                unstructured_ratios=[0.25],  # unstructured pruning ratios
                avoid_overlap=True,  # avoid overlapping pruning
                 
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 
                ):
        # basic info
        self.label = label
        self.model=model
        self.train_data = train_data
        self.test_data = test_data
        self.project_name = project_name
        self.path_backup = path_backup
        self.input_dtype = input_dtype
        self.wand_on = wand_on
        self.batch_size = batch_size

        # hyperparameters
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer_name
        self.momentum = momentum
        self.nesterov = nesterov
        self.scheduler = scheduler_name
        self.label_smoothing = label_smoothing
        
        # training settings
        self.warmup_epochs = warmup_epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device = device    

        # pruning settings
        self.pruning_method = pruning_method
        self.structured_ratios = structured_ratios
        self.unstructured_ratios = unstructured_ratios
        self.avoid_overlap = avoid_overlap

        # ensure model is on the correct device
        self.model = self.model.to(self.device)
    
        self.train_losses = []
        self.test_losses = []
        self.train_accs = []
        self.test_accs = []
     
        if self.wand_on:
            wandb.init(
                project=self.project_name,
                name=f"{self.model.__class__.__name__}_{self.label}",
                config={
                    "model": self.model.__class__.__name__,
                    "epochs": self.num_epochs,
                    "learning_rate": self.learning_rate,
                    "warmup_epochs": self.warmup_epochs,
                    "label_smoothing": self.label_smoothing,
                    "optimizer": self.optimizer,
                    "momentum": self.momentum,
                    "nesterov": self.nesterov,
                    "weight_decay": self.weight_decay,
                    "scheduler": self.scheduler,
                    "t_max": self.num_epochs - self.warmup_epochs,
                    "early_stopping_patience": self.early_stopping_patience,
                    "early_stopping_min_delta": self.early_stopping_min_delta,
                    "device": str(self.device),
                },
            )
            wandb.watch(self.model, log="gradients", log_freq=100)
            

    def train_epoch(self):
        """Train the model for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(self.train_data, desc="Training"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()

            if self.input_dtype == 'binary':
                self.model.binarization()
            
            # Forward + backward + optimize
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()

            if self.input_dtype == 'binary':
                self.model.restore()

            self.optimizer.step()

            if self.input_dtype == 'binary':
                self.model.clip()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(self.train_data)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def evaluate(self):
        """Evaluate the model on the test set"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_data, desc="Evaluating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if self.input_dtype == 'binary':
                    self.model.binarization()
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                if self.input_dtype == 'binary':
                    self.model.restore()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_loss = running_loss / len(self.test_data)
        test_acc = 100. * correct / total
        return test_loss, test_acc

    def train_loop(self):
        
        best_acc = 0.0
        
        early_stopping = EarlyStopping(
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
        )

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Evaluate
            test_loss, test_acc = self.evaluate()
            
            # Store losses
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accs.append(train_acc)
            self.test_accs.append(test_acc)
            
            # Update learning rate
            self.scheduler.step()

            if self.wand_on:
                # Log metrics to Weights & Biases
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                        "lr": self.scheduler.get_last_lr()[0],
                    }
                )
            
            print(f"Epoch {epoch+1}/{self.num_epochs} Train/Test Acc: {train_acc:.2f}%/{test_acc:.2f}% Train/Test Loss: {train_loss:.4f}/{test_loss:.4f} LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), f'{self.path_backup}/best_model_{self.label}.pth')
                print(f"✓ Best model saved! Train/Test Acc: {best_acc:.2f}%/{test_acc:.2f}% Train/Test Loss: {train_loss:.4f}/{test_loss:.4f} LR: {self.scheduler.get_last_lr()[0]:.6f}\n")
                if self.wand_on:
                    wandb.log({"best_acc": best_acc})
                    wandb.save(f"best_model_{self.label}.pth")
        
            # Early stopping based on validation loss
            early_stopping.step(test_loss, self.model)
            if early_stopping.should_stop:
                print(
                    f"Early stopping at epoch {epoch+1}. "
                    f"Best val loss: {early_stopping.best_loss:.4f}"
                )
                break
        
        print(f"Training completed! Best accuracy: {best_acc:.2f}%")
        
        # Restore best weights by validation loss before saving final model
        if early_stopping.best_state is not None:
            self.model.load_state_dict(early_stopping.best_state)
        
        # Save final model
        torch.save(self.model.state_dict(), f'{self.path_backup}/final_model_{self.label}.pth')
        print(f"Final model saved to 'final_model_{self.label}.pth'")
        if self.wand_on:
            wandb.save(f"final_model_{self.label}.pth")

        # Plot loss evolution
        plt.figure(figsize=(10, 6))
        epochs_range = range(1, len(self.train_losses) + 1)
        plt.plot(epochs_range, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs_range, self.test_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss Evolution', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('loss_evolution.png', dpi=300, bbox_inches='tight')
        print(f"Loss evolution plot saved to '{self.path_backup}/loss_evolution_{self.label}.png'")
        plt.close()
        
        if self.wand_on:
            wandb.save(f"./{self.path_backup}/loss_evolution_{self.label}.png")

        # Plot accuracy evolution
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, self.train_accs, 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs_range, self.test_accs, 'r-', label='Validation Accuracy', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Training and Validation Accuracy Evolution', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.path_backup}/accuracy_evolution_{self.label}.png', dpi=300, bbox_inches='tight')
        print(f"Accuracy evolution plot saved to '{self.path_backup}/accuracy_evolution_{self.label}.png'")
        plt.close()
        if self.wand_on:
            wandb.save(f"./{self.path_backup}/acc_{self.label}.png")
            wandb.finish()
        
    def to_qinput(self, type, x):
        if type == "fp16":
            return x.to(torch.float16)
        
    def to_qweight(self, type, x):
        if type == "fp16":
            return x.to(torch.float16)
        
    def to_qmodel(self, type, model):
        if self.device == "cuda":
            if type == "fp16":
                return model.half()
            
    def calculate_score(self, p_s, p_u, q_w, q_a, w, f):
        print(f"Calculating score with p_s={p_s}, p_u={p_u}, q_w={q_w}, q_a={q_a}, w={w}, f={f}")
        param_score = (1 - (p_s + p_u)) * (q_w / 32) * w / 5.6e6
        ops_score = (1 - p_s) * (max(q_w, q_a) / 32) * f / 2.8e8
        total_score = param_score + ops_score

        print(f"Param score: {param_score:.4f}, Ops score: {ops_score:.4f}, Total score: {total_score:.4f}")
        return total_score