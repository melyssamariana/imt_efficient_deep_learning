import torch
import matplotlib.pyplot as plt

def count_params(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Return number of parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def to_millions(n: int) -> float:
    return n / 1e6

# ---------------------------------------------------------
# 1) Defina/crie seus modelos aqui (exemplos abaixo)
#    Substitua pelos seus imports/construtores reais.
# ---------------------------------------------------------
# from models import ResNet, PreActResNet, DenseNet, VGG

# model_resnet = ResNet(...)
# model_preact = PreActResNet(...)
# model_densenet = DenseNet(...)
# model_vgg = VGG(...)
# model_yours = MyNewModel(...)

models = {
    "ResNet": None,
    "PreActResNet": None,
    "DenseNet": None,
    "VGG": None,
    "MyModel": None,  # o que você vai desenvolver
}

# ---------------------------------------------------------
# 2) Coloque aqui as accuracies (em %) obtidas na TASK 1
#    (ex: 85.3 significa 85.3%)
# ---------------------------------------------------------
accuracies = {
    "ResNet": 0.0,
    "PreActResNet": 0.0,
    "DenseNet": 0.0,
    "VGG": 0.0,
    "MyModel": 0.0,
}

# ---------------------------------------------------------
# 3) Conte parâmetros e prepare dados
# ---------------------------------------------------------
x_params_m = []
y_acc = []
labels = []

for name, model in models.items():
    if model is None:
        raise ValueError(f"Model '{name}' está None. Substitua por um modelo PyTorch real.")
    params = count_params(model, trainable_only=True)
    x_params_m.append(to_millions(params))
    y_acc.append(accuracies[name])
    labels.append(name)

# ---------------------------------------------------------
# 4) Plot (scatter) com labels, no estilo do exemplo
# ---------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.scatter(x_params_m, y_acc)

for x, y, lab in zip(x_params_m, y_acc, labels):
    plt.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 6))

plt.xlabel("Number of model parameters (M)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Number of Parameters (CIFAR-10)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
