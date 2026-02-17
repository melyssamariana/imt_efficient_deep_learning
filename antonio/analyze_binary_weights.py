import torch
import numpy as np
from models.resnet_s import resnet20
import matplotlib.pyplot as plt
import sys

# Get checkpoint path from command line or use default
checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else 'best_model.pth'

print(f"\n{'='*80}")
print("BinaryConnect Weight Analysis")
print(f"{'='*80}\n")
print(f"Analyzing checkpoint: {checkpoint_path}\n")

# Load the model trained with BinaryConnect
model = resnet20()
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Remove "model." prefix from keys if present
cleaned_state_dict = {}
for key, value in checkpoint.items():
    if key.startswith('model.'):
        cleaned_key = key[6:]
        cleaned_state_dict[cleaned_key] = value
    else:
        cleaned_state_dict[key] = value

model.load_state_dict(cleaned_state_dict)
print(f"✓ Loaded model from {checkpoint_path}\n")

# Analyze weights
print(f"{'='*80}")
print("WEIGHT ANALYSIS")
print(f"{'='*80}\n")

all_weights = []
layer_stats = []

for name, param in model.named_parameters():
    if 'weight' in name:  # Analyze only weight parameters (not biases)
        weights = param.detach().cpu().numpy().flatten()
        all_weights.extend(weights)
        
        # Calculate statistics
        min_val = weights.min()
        max_val = weights.max()
        mean_val = weights.mean()
        std_val = weights.std()
        
        # Count values outside [-1, 1]
        outside_range = np.sum((weights < -1.0) | (weights > 1.0))
        total = len(weights)
        pct_outside = 100.0 * outside_range / total
        
        # Count how many are exactly -1, +1, or in between
        exactly_minus_one = np.sum(np.isclose(weights, -1.0, atol=1e-6))
        exactly_plus_one = np.sum(np.isclose(weights, 1.0, atol=1e-6))
        in_between = total - exactly_minus_one - exactly_plus_one
        
        # Check how "binary" the weights are (close to -1 or +1)
        distances_to_binary = np.minimum(np.abs(weights - 1.0), np.abs(weights + 1.0))
        avg_distance = distances_to_binary.mean()
        
        layer_stats.append({
            'name': name,
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val,
            'outside_range': outside_range,
            'total': total,
            'pct_outside': pct_outside,
            'exactly_minus_one': exactly_minus_one,
            'exactly_plus_one': exactly_plus_one,
            'in_between': in_between,
            'avg_distance': avg_distance
        })

# Print layer-by-layer statistics
print("Layer-by-Layer Statistics:")
print(f"{'-'*80}")
for stat in layer_stats:
    print(f"\n{stat['name']}:")
    print(f"  Range: [{stat['min']:.6f}, {stat['max']:.6f}]")
    print(f"  Mean: {stat['mean']:.6f}, Std: {stat['std']:.6f}")
    print(f"  Total weights: {stat['total']}")
    print(f"  Exactly -1: {stat['exactly_minus_one']} ({100.0*stat['exactly_minus_one']/stat['total']:.2f}%)")
    print(f"  Exactly +1: {stat['exactly_plus_one']} ({100.0*stat['exactly_plus_one']/stat['total']:.2f}%)")
    print(f"  In between: {stat['in_between']} ({100.0*stat['in_between']/stat['total']:.2f}%)")
    print(f"  Outside [-1, 1]: {stat['outside_range']} ({stat['pct_outside']:.2f}%)")
    print(f"  Avg distance to nearest binary: {stat['avg_distance']:.6f}")

# Overall statistics
all_weights = np.array(all_weights)
print(f"\n{'='*80}")
print("OVERALL STATISTICS")
print(f"{'='*80}\n")
print(f"Total weight parameters: {len(all_weights)}")
print(f"Range: [{all_weights.min():.6f}, {all_weights.max():.6f}]")
print(f"Mean: {all_weights.mean():.6f}")
print(f"Std: {all_weights.std():.6f}")
print(f"Median: {np.median(all_weights):.6f}")

exactly_minus_one_total = np.sum(np.isclose(all_weights, -1.0, atol=1e-6))
exactly_plus_one_total = np.sum(np.isclose(all_weights, 1.0, atol=1e-6))
in_between_total = len(all_weights) - exactly_minus_one_total - exactly_plus_one_total
outside_range_total = np.sum((all_weights < -1.0) | (all_weights > 1.0))

print(f"\nValue Distribution:")
print(f"  Exactly -1: {exactly_minus_one_total} ({100.0*exactly_minus_one_total/len(all_weights):.2f}%)")
print(f"  Exactly +1: {exactly_plus_one_total} ({100.0*exactly_plus_one_total/len(all_weights):.2f}%)")
print(f"  In between [-1, +1]: {in_between_total} ({100.0*in_between_total/len(all_weights):.2f}%)")
print(f"  Outside [-1, +1]: {outside_range_total} ({100.0*outside_range_total/len(all_weights):.2f}%)")

distances_to_binary_total = np.minimum(np.abs(all_weights - 1.0), np.abs(all_weights + 1.0))
print(f"\nAverage distance to nearest binary value: {distances_to_binary_total.mean():.6f}")

# Determine if weights are binarized
print(f"\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}\n")

if outside_range_total == 0 and in_between_total < len(all_weights) * 0.01:
    print("✓ Weights appear to be BINARIZED (all values are at or very close to -1 or +1)")
elif outside_range_total == 0:
    print("⚠ Weights are CLIPPED to [-1, 1] but NOT fully binarized")
    print(f"  {100.0*in_between_total/len(all_weights):.2f}% of weights are between -1 and +1")
else:
    print("✗ Weights are NOT binarized")
    print(f"  {100.0*outside_range_total/len(all_weights):.2f}% of weights are outside [-1, 1]")
    print("  This suggests the model was trained with standard methods, not BinaryConnect")

print(f"\n{'='*80}\n")

# Create histogram
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(all_weights, bins=100, edgecolor='black', alpha=0.7)
plt.xlabel('Weight Value')
plt.ylabel('Count')
plt.title('Distribution of All Weights')
plt.axvline(-1, color='r', linestyle='--', label='Binary values (-1, +1)')
plt.axvline(1, color='r', linestyle='--')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Zoom in to see detail
plt.hist(all_weights, bins=200, range=(-1.5, 1.5), edgecolor='black', alpha=0.7)
plt.xlabel('Weight Value')
plt.ylabel('Count')
plt.title('Distribution of Weights (Zoomed to [-1.5, 1.5])')
plt.axvline(-1, color='r', linestyle='--', label='Binary values')
plt.axvline(1, color='r', linestyle='--')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Create output filename based on checkpoint name
import os
checkpoint_basename = os.path.splitext(os.path.basename(checkpoint_path))[0]
output_filename = f'weight_distribution_{checkpoint_basename}.png'

plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"✓ Histogram saved to '{output_filename}'")
print()
