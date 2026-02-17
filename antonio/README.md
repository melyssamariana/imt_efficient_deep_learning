# Efficient Deep Learning - CIFAR10 with BinaryConnect

This project implements ResNet20 for CIFAR10 classification with BinaryConnect, a method for training deep neural networks with binary weights.

## Table of Contents
- [Setup](#setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [BinaryConnect Verification Workflow](#binaryconnect-verification-workflow)
- [Files Description](#files-description)

---

## Setup

### 1. Create Virtual Environment

```bash
cd antonio/
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `torch`
- `torchvision`
- `tqdm`
- `matplotlib`
- `wandb` (for experiment tracking)
- `numpy`

---

## Training

### Standard Training (Full Precision)

Train ResNet20 with standard full precision weights:

```bash
python3 main.py
```

This will:
- Train for 150 epochs with SGD optimizer
- Use learning rate warmup (5 epochs) + cosine annealing
- Apply label smoothing (0.1) and weight decay (0.001)
- Save best model to `best_model.pth`
- Log metrics to Weights & Biases

### BinaryConnect Training

Train ResNet20 with BinaryConnect (binary weights during forward pass):

```bash
python3 main_binary.py
```

This will:
- Apply BinaryConnect: binarize weights → forward → restore → backward → clip
- Follow the sequence: `binarization()` → forward/backward → `restore()` → optimizer step → `clip()`
- Save model with **full precision weights** (clipped to [-1, 1])
- The saved model contains latent full precision parameters, not binary weights

---

## Evaluation

### 1. Evaluate Full Precision Model

```bash
python3 eval.py
```

Evaluates the model with full precision weights (as saved during training).

### 2. Evaluate with Binarized Weights

```bash
python3 eval_binarized.py
```

This will:
- Load the full precision model
- Apply `binarization()` to convert weights to {-1, +1}
- Evaluate on CIFAR10 test set
- **Save binarized weights** to `best_model_binarized.pth`

### 3. Load Pre-Binarized Model

```bash
python3 load_binarized_example.py
```

Shows how to load and use the binarized model directly without needing BinaryConnect wrapper.

---

## BinaryConnect Verification Workflow

To verify that your model was properly trained with BinaryConnect and to analyze the binarization:

### Step 1: Analyze Saved Weights (Full Precision)

```bash
python3 analyze_binary_weights.py
```

**What it checks:**
- Loads `best_model.pth` (the model saved during training)
- Analyzes weight distribution layer-by-layer
- Checks if weights are in range [-1, 1]
- Counts how many weights are exactly -1, +1, or in between
- Calculates average distance to nearest binary value
- Generates histogram visualization

**Expected output:**
```
⚠ Weights are CLIPPED to [-1, 1] but NOT fully binarized
  100.00% of weights are between -1 and +1
```

This is **correct**! BinaryConnect saves full precision weights (clipped to [-1, 1]).

### Step 2: Binarize and Evaluate

```bash
python3 eval_binarized.py
```

**What it does:**
- Loads full precision weights
- Applies `binarization()` to convert to {-1, +1}
- Verifies all weights are binary
- Evaluates accuracy with binary weights
- **Saves binarized model** to `best_model_binarized.pth`

**Expected output:**
```
✓ All convolution/linear weights are binary {-1, +1}
Test Accuracy: XX.XX%
✓ Saved binarized model: X.XX MB
```

### Step 3: Analyze Binarized Weights

```bash
python3 analyze_binary_weights.py best_model_binarized.pth
```

**What it checks:**
- Analyzes the saved binarized model
- Should show 100% of weights are exactly -1 or +1
- Histogram should show only two spikes at -1 and +1

**Expected output:**
```
✓ Weights appear to be BINARIZED (all values are at or very close to -1 or +1)
  Exactly -1: XXXXX (XX.XX%)
  Exactly +1: XXXXX (XX.XX%)
  In between: 0 (0.00%)
```

### Step 4: Verify Direct Loading Works

```bash
python3 load_binarized_example.py
```

**What it does:**
- Loads `best_model_binarized.pth` directly (no BC wrapper)
- Verifies weights are binary
- Runs inference to confirm same accuracy

---

## Understanding BinaryConnect

### During Training (`main_binary.py`):

1. **Binarization** (`binarization()`): 
   - Saves full precision weights to shadow copy
   - Converts weights to {-1, +1} for forward pass

2. **Forward & Backward**:
   - Forward pass uses binary weights
   - Gradients computed with respect to binary weights

3. **Restore** (`restore()`):
   - Restores full precision weights from shadow copy

4. **Optimizer Step**:
   - Updates full precision weights using gradients

5. **Clip** (`clip()`):
   - Clips weights to [-1, 1] range

6. **Save**:
   - Saves **full precision weights** (clipped to [-1, 1])

### For Inference:

- Load full precision weights
- Call `binarization()` once
- Use binary weights for all inference
- OR: Save after binarization and load directly

---

## Files Description

### Training Scripts
- `main.py` - Standard training with full precision weights
- `main_binary.py` - Training with BinaryConnect

### Evaluation Scripts
- `eval.py` - Evaluate full precision model
- `eval_binarized.py` - Binarize and evaluate model
- `load_binarized_example.py` - Example of loading pre-binarized model

### Analysis Scripts
- `analyze_binary_weights.py` - Analyze weight distribution
- `save_binary_model.py` - Save model with 1-bit packed weights (experimental)
- `inspect_saved_model.py` - Inspect checkpoint contents

### Model Files
- `models/resnet_s.py` - ResNet20/32/44/56/110 for CIFAR10
- `models/binaryconnect.py` - BinaryConnect implementation
- `cifar10.py` - CIFAR10 data loaders

### Checkpoint Files (Generated)
- `best_model.pth` - Best model during training (full precision, clipped)
- `final_model.pth` - Final model after training
- `best_model_binarized.pth` - Model with weights binarized to {-1, +1}
- `best_model_binary_1bit.pth` - Experimental 1-bit packed format

---

## Performance Notes

### Expected Results:
- **Full Precision ResNet20**: ~91-92% accuracy
- **BinaryConnect ResNet20**: ~88-90% accuracy (slight degradation expected)

### Model Sizes:
- **Full Precision**: ~8.7 MB (float32)
- **Binarized (still float32)**: ~8.7 MB (but only {-1, +1} values)
- **Theoretical Binary**: ~0.27 MB (32x compression if packed to 1-bit)

---

## Troubleshooting

### "Weights are not binarized" after training
This is expected! The saved checkpoint contains full precision weights (clipped to [-1, 1]). Run `eval_binarized.py` to convert to binary for inference.

### Accuracy drop with BinaryConnect
Binary weights have less capacity than full precision. A 2-4% accuracy drop is normal and acceptable for the compression benefits.

### Loading errors with "model." prefix
The BinaryConnect wrapper adds a "model." prefix. Use the cleaning code in `eval.py` or `eval_binarized.py` to handle this.

---

## References

- [BinaryConnect Paper](http://papers.nips.cc/paper/5647-binaryconnect-training-deep-neural-networks-with-b)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
