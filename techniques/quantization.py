from utils import ConfigModel, EarlyStopping
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

class QuantizationAwareConfig(ConfigModel):
    def __init__(self, num_bits=8, symmetric=True, per_channel=False, **kwargs):
        super().__init__(**kwargs)
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
    '''
    def export_model(self):
        model = ''
        if self.input_dtype == 'bf16':
            model = self.model.to(dtype=torch.bfloat16).eval()
        elif self.input_dtype == 'bc':
            model = self.model.eval()  # BinaryConnect doesn't change dtype, but we set to eval mode
        torch.save(model.state_dict(), f"{self.path_backup}/model_{self.label}_{self.input_dtype}.pth")
    '''
    def train_epoch(self):
        """Train the model for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(self.train_data, desc="Training"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            #lsq
            self.optimizer.zero_grad(set_to_none=True)

            if self.input_dtype == 'bc':
                # BinaryConnect-style
                self.model.binarization()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.model.restore()
                self.optimizer.step()
                self.model.clip()

            elif self.input_dtype == 'bf16':
                inputs.half()  # Convert inputs to BF16
                # BF16 autocast
                with autocast(dtype=torch.bfloat16):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            else:
                # Default FP32
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(self.train_data)
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc
    
    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_data, desc="Evaluating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if self.input_dtype == "bf16":
                    with autocast(dtype=torch.bfloat16):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                elif self.input_dtype == "bc":
                    self.model.binarization()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    self.model.restore()
                    # Opção A (recomendada): avaliar SEM binarizar (pesos reais)
                    # outputs = self.model(inputs)
                    # loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(self.test_data)
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc
        
    def train_loop(self):
        best_acc = 0.0

        early_stopping = EarlyStopping(
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
        )

        if self.input_dtype == 'bf16':
            self.model.half()  # Convert model to BF16:

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")

            # === Train
            train_loss, train_acc = self.train_epoch()

            # === Evaluate
            # Se seu evaluate já usa self.test_data internamente, deixe self.evaluate()
            # Caso contrário, use: self.evaluate(self.test_data)
            test_loss, test_acc = self.evaluate()  

            # === Store history
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accs.append(train_acc)
            self.test_accs.append(test_acc)

            # === Scheduler step
            # Se for ReduceLROnPlateau, ele espera uma métrica (ex: val_loss)
            if self.scheduler is not None:
                if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    self.scheduler.step(test_loss)
                else:
                    self.scheduler.step()

            # === Current LR
            if self.scheduler is not None and hasattr(self.scheduler, "get_last_lr"):
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                # fallback: pega do optimizer
                current_lr = self.optimizer.param_groups[0]["lr"]

            # === W&B logging
            if self.wand_on:
                import wandb
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                        "lr": current_lr,
                        "input_dtype": self.input_dtype,
                    }
                )

            # === Save best model by accuracy
            if test_acc > best_acc:
                best_acc = test_acc
                best_path = os.path.join(self.path_backup, f"best_model_{self.label}.pth")
                #self.export_model()

                print(
                    f"✓ Best model saved! "
                    f"Best Acc: {best_acc:.2f}% | "
                    f"Train/Test Acc: {train_acc:.2f}%/{test_acc:.2f}% "
                    f"Train/Test Loss: {train_loss:.4f}/{test_loss:.4f} "
                    f"LR: {current_lr:.6f}\n"
                )

                if self.wand_on:
                    wandb.log({"best_acc": best_acc})
                    wandb.save(best_path)

            # === Early stopping by validation loss
            early_stopping.step(test_loss, self.model)
            if early_stopping.should_stop:
                print(
                    f"Early stopping at epoch {epoch+1}. "
                    f"Best val loss: {early_stopping.best_loss:.4f}"
                )
                break

        print(f"Training completed! Best accuracy: {best_acc:.2f}%")

        # === Restore best weights by validation loss before saving final model
        if early_stopping.best_state is not None:
            self.model.load_state_dict(early_stopping.best_state)

        # === Save final model
        final_path = os.path.join(self.path_backup, f"final_model_{self.label}.pth")
        torch.save(self.model.state_dict(), final_path)
        print(f"Final model saved to '{final_path}'")

        if self.wand_on:
            import wandb
            wandb.save(final_path)

        # === Plot loss evolution
        epochs_range = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, self.train_losses, label="Training Loss", linewidth=2)
        plt.plot(epochs_range, self.test_losses, label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training and Validation Loss Evolution", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        loss_plot_path = os.path.join(self.path_backup, f"loss_evolution_{self.label}.png")
        plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Loss evolution plot saved to '{loss_plot_path}'")

        if self.wand_on:
            wandb.save(loss_plot_path)

        # === Plot accuracy evolution
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, self.train_accs, label="Training Accuracy", linewidth=2)
        plt.plot(epochs_range, self.test_accs, label="Validation Accuracy", linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.title("Training and Validation Accuracy Evolution", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        acc_plot_path = os.path.join(self.path_backup, f"accuracy_evolution_{self.label}.png")
        plt.savefig(acc_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Accuracy evolution plot saved to '{acc_plot_path}'")

        if self.wand_on:
            wandb.save(acc_plot_path)
            wandb.finish()