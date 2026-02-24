import torch
from torchinfo import summary

def calculate_score(p_s, p_u, q_w, q_a, w, f):
    param_score = (1 - (p_s + p_u)) * (q_w / 32) * w / 5.6e6
    ops_score = (1 - p_s) * (max(q_w, q_a) / 32) * f / 2.8e8
    total_score = param_score + ops_score
    return total_score

# Example usage:
p_s = 0.15  # structured pruning
p_u = 0.8  # unstructured pruning
q_w = 8    # quantization for weights
q_a = 8    # quantization for activations
w = 269722 # number of weights
f = 40.44e6  # number of MACs
score = calculate_score(p_s, p_u, q_w, q_a, w, f)
print(f"Calculated score: {score:.4f}")

from models.resnet import ResNet18
model = ResNet18()

summary(model, input_size=(1, 3, 32, 32))
