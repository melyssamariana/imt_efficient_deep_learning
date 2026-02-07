import torch
import matplotlib.pyplot as plt
import numpy as np

# Import all model architectures
from models.vgg import VGG
from models.resnet import ResNet18, ResNet50, ResNet101
from models.densenet import DenseNet121
from models.preact_resnet import PreActResNet18

def count_parameters(model):
    """Count the total number of parameters in a model"""
    return sum(p.numel() for p in model.parameters())

def main():
    # Dictionary of models and their reported accuracies
    models_info = {
        'VGG16': {'accuracy': 92.64, 'model': VGG('VGG16')},
        'ResNet18': {'accuracy': 93.02, 'model': ResNet18()},
        'ResNet50': {'accuracy': 93.62, 'model': ResNet50()},
        'ResNet101': {'accuracy': 93.75, 'model': ResNet101()},
        'DenseNet121': {'accuracy': 95.04, 'model': DenseNet121()},
        'PreActResNet18': {'accuracy': 95.11, 'model': PreActResNet18()},
    }
    
    # Calculate parameters for each model
    model_names = []
    param_counts = []
    accuracies = []
    
    print("Model Parameter Counts:")
    print("-" * 50)
    
    for name, info in models_info.items():
        model = info['model']
        num_params = count_parameters(model)
        accuracy = info['accuracy']
        
        model_names.append(name)
        param_counts.append(num_params / 1e6)  # Convert to millions
        accuracies.append(accuracy)
        
        print(f"{name:20s}: {num_params:,} parameters ({num_params/1e6:.2f}M) - Acc: {accuracy:.2f}%")
    
    print("-" * 50)
    
    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    
    # Create color map for different model families
    colors = []
    for name in model_names:
        if 'VGG' in name:
            colors.append('#1f77b4')  # Blue
        elif 'DenseNet' in name:
            colors.append('#2ca02c')  # Green
        elif 'PreAct' in name:
            colors.append('#ff7f0e')  # Orange
        elif 'ResNet' in name:
            colors.append('#d62728')  # Red
        else:
            colors.append('#9467bd')  # Purple
    
    # Plot points
    scatter = plt.scatter(param_counts, accuracies, s=200, c=colors, alpha=0.6, edgecolors='black', linewidth=2)
    
    # Add labels for each point
    for i, name in enumerate(model_names):
        plt.annotate(name, 
                    (param_counts[i], accuracies[i]),
                    xytext=(10, 5), 
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))
    
    # Customize plot
    plt.xlabel('Number of Parameters (Millions)', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('Model Comparison: Parameters vs Accuracy', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend for model families
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', alpha=0.6, label='ResNet Family'),
        Patch(facecolor='#ff7f0e', alpha=0.6, label='PreActResNet Family'),
        Patch(facecolor='#2ca02c', alpha=0.6, label='DenseNet Family'),
        Patch(facecolor='#1f77b4', alpha=0.6, label='VGG Family'),
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    # Set y-axis limits for better visualization
    plt.ylim([min(accuracies) - 1, max(accuracies) + 1])
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to 'model_comparison.png'")
    plt.close()
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"Most efficient (best accuracy/param ratio): {model_names[np.argmax(np.array(accuracies) / np.array(param_counts))]}")
    print(f"Most accurate: {model_names[np.argmax(accuracies)]}")
    print(f"Smallest model: {model_names[np.argmin(param_counts)]}")

if __name__ == '__main__':
    main()
