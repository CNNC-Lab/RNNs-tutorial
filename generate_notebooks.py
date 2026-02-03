#!/usr/bin/env python3
"""
Script to generate the tutorial notebooks.
Run this to create all notebook files.
"""

import json
import os

def create_notebook(cells, filename):
    """Create a Jupyter notebook from cells."""
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=1)
    print(f"Created: {filename}")

def markdown_cell(source):
    """Create a markdown cell."""
    if isinstance(source, str):
        source = source.split('\n')
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code_cell(source, outputs=None):
    """Create a code cell."""
    if isinstance(source, str):
        source = source.split('\n')
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": source
    }

# =============================================================================
# Notebook 01: Continuous-Time RNN
# =============================================================================

nb01_cells = [
    markdown_cell("""# 📈 Continuous-Time Recurrent Neural Networks

## Neural ODEs for Dynamical Systems

In this notebook, we implement a **Continuous-Time RNN (CT-RNN)** — a recurrent network whose dynamics are defined by an ODE.

### The CT-RNN Equation

$$\\tau \\frac{d\\mathbf{h}}{dt} = -\\mathbf{h} + \\phi(\\mathbf{W}_{rec}\\mathbf{h} + \\mathbf{W}_{in}\\mathbf{x} + \\mathbf{b})$$

### Why Continuous-Time?
1. **Biological plausibility**: Neurons evolve continuously
2. **Arbitrary time steps**: Can interpolate between observations  
3. **Smooth dynamics**: Amenable to analysis
4. **Memory efficient training**: Adjoint method"""),

    code_cell("""# Setup
import sys
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    !pip install -q torchdiffeq
    
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from torchdiffeq import odeint

np.random.seed(42)
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
plt.style.use('seaborn-v0_8-whitegrid')"""),

    markdown_cell("## Part 1: Generate Lorenz Data"),
    
    code_cell("""from scipy.integrate import solve_ivp

def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

# Generate trajectory
sol = solve_ivp(lorenz, (0, 210), [1,1,1], t_eval=np.arange(0, 210, 0.01), rtol=1e-10)
trajectory = sol.y[:, 1000:].T  # Remove transient

# Split and normalize
n = len(trajectory)
train_data = trajectory[:int(0.7*n)]
val_data = trajectory[int(0.7*n):int(0.85*n)]
test_data = trajectory[int(0.85*n):]

mean, std = train_data.mean(0), train_data.std(0)
train_norm = (train_data - mean) / std
val_norm = (val_data - mean) / std
test_norm = (test_data - mean) / std

print(f"Train: {train_norm.shape}, Val: {val_norm.shape}, Test: {test_norm.shape}")"""),

    code_cell("""# Visualize
fig = plt.figure(figsize=(14, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(trajectory[:5000,0], trajectory[:5000,1], trajectory[:5000,2], lw=0.5)
ax1.set_title('Lorenz Attractor')

ax2 = fig.add_subplot(122)
t = np.arange(2000) * 0.01
ax2.plot(t, trajectory[:2000,0], label='x')
ax2.plot(t, trajectory[:2000,1], label='y')
ax2.plot(t, trajectory[:2000,2], label='z')
ax2.legend()
ax2.set_xlabel('Time')
ax2.set_title('Time Series')
plt.tight_layout()
plt.show()"""),

    markdown_cell("## Part 2: Dataset and DataLoader"),
    
    code_cell("""class LorenzDataset(Dataset):
    def __init__(self, data, seq_length=50):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length]
        return x, y

seq_length = 50
batch_size = 64

train_dataset = LorenzDataset(train_norm, seq_length)
val_dataset = LorenzDataset(val_norm, seq_length)
test_dataset = LorenzDataset(test_norm, seq_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print(f"Samples - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")"""),

    markdown_cell("""## Part 3: CT-RNN Implementation

### The Core Dynamics
The CT-RNN cell computes $d\\mathbf{h}/dt$ given current state and input."""),

    code_cell("""class CTRNNCell(nn.Module):
    \"\"\"
    CT-RNN dynamics: τ dh/dt = -h + φ(W_rec @ h + W_in @ x + b)
    \"\"\"
    def __init__(self, input_size, hidden_size, tau=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.tau = tau
        
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_in = nn.Linear(input_size, hidden_size, bias=True)
        
        # Initialize for stability
        nn.init.orthogonal_(self.W_rec.weight)
        self.W_rec.weight.data *= 0.9
        nn.init.xavier_uniform_(self.W_in.weight)
    
    def forward(self, t, h, x=None):
        rec = self.W_rec(h)
        if x is not None:
            inp = self.W_in(x)
        else:
            inp = self.W_in.bias.unsqueeze(0).expand(h.shape[0], -1)
        
        dhdt = (-h + torch.tanh(rec + inp)) / self.tau
        return dhdt"""),

    code_cell("""class ODEFunc(nn.Module):
    \"\"\"Wrapper for odeint compatibility.\"\"\"
    def __init__(self, cell, x=None):
        super().__init__()
        self.cell = cell
        self.x = x
    
    def forward(self, t, h):
        return self.cell(t, h, self.x)


class ContinuousTimeRNN(nn.Module):
    \"\"\"
    Complete CT-RNN for sequence modeling.
    \"\"\"
    def __init__(self, input_size=3, hidden_size=64, output_size=3, tau=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.cell = CTRNNCell(input_size, hidden_size, tau)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, return_hidden=False):
        \"\"\"
        x: (batch, seq_len, input_dim)
        \"\"\"
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        t_span = torch.tensor([0.0, 1.0], device=device)
        
        hidden_states = []
        for i in range(seq_len):
            ode_func = ODEFunc(self.cell, x[:, i, :])
            h_traj = odeint(ode_func, h, t_span, method='rk4')
            h = h_traj[-1]
            hidden_states.append(h)
        
        hidden_states = torch.stack(hidden_states, dim=1)
        output = self.decoder(h)
        
        if return_hidden:
            return output, hidden_states
        return output"""),

    markdown_cell("## Part 4: Training"),
    
    code_cell("""# Create model
model = ContinuousTimeRNN(input_size=3, hidden_size=64, output_size=3, tau=1.0).to(device)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

# Test forward pass
sample_x, sample_y = next(iter(train_loader))
sample_x = sample_x.to(device)
out = model(sample_x)
print(f"Input: {sample_x.shape} -> Output: {out.shape}")"""),

    code_cell("""# Training loop
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += criterion(pred, y).item()
    return total_loss / len(loader)"""),

    code_cell("""# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
criterion = nn.MSELoss()

n_epochs = 100
history = {'train': [], 'val': []}
best_val = float('inf')

for epoch in tqdm(range(n_epochs)):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)
    
    history['train'].append(train_loss)
    history['val'].append(val_loss)
    
    scheduler.step(val_loss)
    
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), 'checkpoints/ctrnn_best.pt')
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")

print(f"\\nBest validation loss: {best_val:.6f}")"""),

    code_cell("""# Plot training
fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(history['train'], label='Train')
ax.semilogy(history['val'], label='Validation')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.legend()
ax.set_title('CT-RNN Training')
plt.show()"""),

    markdown_cell("## Part 5: Evaluation & Visualization"),
    
    code_cell("""# Load best model
model.load_state_dict(torch.load('checkpoints/ctrnn_best.pt'))

# Evaluate on test set
test_loss = evaluate(model, test_loader, criterion)
print(f"Test loss: {test_loss:.6f}")

# Get predictions
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        pred = model(x)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.numpy())

preds = np.concatenate(all_preds)
targets = np.concatenate(all_targets)

# Denormalize
preds_denorm = preds * std + mean
targets_denorm = targets * std + mean

print(f"Predictions shape: {preds_denorm.shape}")"""),

    code_cell("""# Compare predictions vs targets
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
n_show = 500

for i, (ax, name) in enumerate(zip(axes, ['x', 'y', 'z'])):
    ax.plot(targets_denorm[:n_show, i], 'b-', label='True', alpha=0.8)
    ax.plot(preds_denorm[:n_show, i], 'r--', label='Predicted', alpha=0.8)
    ax.set_ylabel(name)
    ax.legend(loc='upper right')

axes[-1].set_xlabel('Sample')
plt.suptitle('CT-RNN: One-Step Prediction')
plt.tight_layout()
plt.show()

# Compute R²
ss_res = np.sum((targets_denorm - preds_denorm)**2)
ss_tot = np.sum((targets_denorm - targets_denorm.mean(0))**2)
r2 = 1 - ss_res / ss_tot
print(f"R² score: {r2:.4f}")"""),

    markdown_cell("""## Part 6: Autonomous Generation

Let the network predict its own output recursively to generate a trajectory."""),
    
    code_cell("""def generate_trajectory(model, initial_seq, n_steps, device):
    \"\"\"Generate autonomous trajectory.\"\"\"
    model.eval()
    
    # Use last state from initial sequence
    with torch.no_grad():
        x = torch.tensor(initial_seq, dtype=torch.float32).unsqueeze(0).to(device)
        _, hidden = model(x, return_hidden=True)
        h = hidden[:, -1, :]
    
    # Get initial output
    current = model.decoder(h)
    
    trajectory = [current.cpu().numpy()[0]]
    
    # Generate
    for _ in range(n_steps - 1):
        with torch.no_grad():
            # Use current output as input
            ode_func = ODEFunc(model.cell, current)
            t_span = torch.tensor([0.0, 1.0], device=device)
            h_traj = odeint(ode_func, h, t_span, method='rk4')
            h = h_traj[-1]
            current = model.decoder(h)
            trajectory.append(current.cpu().numpy()[0])
    
    return np.array(trajectory)

# Generate
n_gen = 2000
initial_seq = test_norm[:seq_length]
generated = generate_trajectory(model, initial_seq, n_gen, device)

# Denormalize
generated_denorm = generated * std + mean
true_traj = test_data[:n_gen]

print(f"Generated trajectory shape: {generated_denorm.shape}")"""),

    code_cell("""# Compare generated vs true attractor
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(true_traj[:,0], true_traj[:,1], true_traj[:,2], 'b-', lw=0.5, alpha=0.7, label='True')
ax1.set_title('True Lorenz Attractor')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(generated_denorm[:,0], generated_denorm[:,1], generated_denorm[:,2], 
         'r-', lw=0.5, alpha=0.7, label='Generated')
ax2.set_title('CT-RNN Generated')

for ax in [ax1, ax2]:
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

plt.tight_layout()
plt.show()"""),

    code_cell("""# Time series comparison
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
t = np.arange(n_gen) * 0.01

for i, (ax, name) in enumerate(zip(axes, ['x', 'y', 'z'])):
    ax.plot(t, true_traj[:, i], 'b-', label='True', alpha=0.8)
    ax.plot(t, generated_denorm[:, i], 'r-', label='Generated', alpha=0.8)
    ax.set_ylabel(name)
    ax.legend()

axes[-1].set_xlabel('Time')
plt.suptitle('CT-RNN Autonomous Generation vs True Dynamics')
plt.tight_layout()
plt.show()

# Compute valid prediction time (before error exceeds threshold)
error = np.sqrt(np.mean((true_traj - generated_denorm)**2, axis=1))
threshold = 0.4 * true_traj.std()
valid_idx = np.where(error > threshold)[0]
valid_time = valid_idx[0] * 0.01 if len(valid_idx) > 0 else n_gen * 0.01
print(f"Valid prediction time: {valid_time:.2f} time units")"""),

    markdown_cell("""## Summary

We implemented a **Continuous-Time RNN** using Neural ODEs and trained it on Lorenz prediction.

**Key observations:**
- CT-RNN can learn chaotic dynamics
- One-step prediction is highly accurate
- Long-term autonomous prediction diverges (expected due to chaos!)
- The generated attractor has similar geometry to the true one

**Next:** Notebook 02 - Balanced E/I Rate Network"""),
]

# =============================================================================
# Notebook 02: Balanced Rate Network (placeholder)
# =============================================================================

nb02_cells = [
    markdown_cell("""# ⚖️ Balanced Excitatory-Inhibitory Rate Network

## Biologically Plausible Dynamics

In this notebook, we implement a rate network with **separate excitatory and inhibitory populations**, obeying Dale's law.

### The Balanced E/I Equations

$$\\tau_E \\frac{d\\mathbf{r}_E}{dt} = -\\mathbf{r}_E + \\phi(\\mathbf{W}_{EE}\\mathbf{r}_E - \\mathbf{W}_{EI}\\mathbf{r}_I + \\mathbf{I}_{ext})$$

$$\\tau_I \\frac{d\\mathbf{r}_I}{dt} = -\\mathbf{r}_I + \\phi(\\mathbf{W}_{IE}\\mathbf{r}_E - \\mathbf{W}_{II}\\mathbf{r}_I)$$

### Key Features
- **Dale's law**: E neurons only excite, I neurons only inhibit
- **Balance**: Large E and I currents cancel, leaving small net drive
- **Biological time constants**: Different τ for E and I"""),

    code_cell("""# Full implementation coming soon!
# See src/models/balanced_rate.py for the complete model

print("This notebook is under construction.")
print("Key concepts to implement:")
print("1. Separate E/I populations")
print("2. Weight constraints (positive E, negative I)")
print("3. Balance condition analysis")
print("4. Same Lorenz prediction task")"""),
]

# =============================================================================
# Notebook 03: Balanced Spiking Network (placeholder)
# =============================================================================

nb03_cells = [
    markdown_cell("""# ⚡ Balanced Spiking Network

## Reservoir Computing with LIF Neurons

In this notebook, we implement a spiking neural network using **norse** (PyTorch-native spiking neurons).

### LIF Neuron Dynamics

$$\\tau_m \\frac{dV}{dt} = -(V - V_{rest}) + I_{syn} + I_{ext}$$
$$\\text{if } V > V_{th}: \\text{spike}, V \\rightarrow V_{reset}$$

### Reservoir Approach
- Random, fixed recurrent weights
- Only train the readout layer
- Balance achieved through E/I architecture"""),

    code_cell("""# Full implementation coming soon!
# Requires: pip install norse

print("This notebook is under construction.")
print("Key concepts to implement:")
print("1. LIF neuron populations")
print("2. Sparse random connectivity")
print("3. Reservoir computing (train readout only)")
print("4. Spike-based readout mechanisms")"""),
]

# =============================================================================
# Notebook 04: Dynamical Analysis (placeholder)
# =============================================================================

nb04_cells = [
    markdown_cell("""# 🔬 Dynamical Systems Analysis

## Analyzing Trained Networks

In this notebook, we analyze our trained networks using tools from dynamical systems theory.

### Analysis Tools
1. **Fixed point finding**: Where $\\frac{d\\mathbf{h}}{dt} = 0$
2. **Stability analysis**: Eigenvalues of the Jacobian
3. **Lyapunov exponents**: Measure of chaos
4. **Attractor comparison**: Geometry metrics"""),

    code_cell("""# Full implementation coming soon!
# See src/analysis/__init__.py for tools

print("This notebook is under construction.")
print("Key analyses to perform:")
print("1. Find fixed points of trained CT-RNN")
print("2. Linearize and compute eigenvalues")
print("3. Estimate Lyapunov exponents")
print("4. Compare attractor dimensions")"""),
]

# =============================================================================
# Notebook 05: Synthesis (placeholder)
# =============================================================================

nb05_cells = [
    markdown_cell("""# 📊 Synthesis & Comparison

## Comparing All Architectures

In this final notebook, we compare all three network architectures side-by-side.

### Comparison Dimensions
1. **Prediction accuracy**: One-step and multi-step
2. **Attractor reconstruction**: Geometry similarity
3. **Computational cost**: Training time, inference time
4. **Biological plausibility**: E/I balance, spiking, Dale's law"""),

    code_cell("""# Full implementation coming soon!

print("This notebook is under construction.")
print("Key comparisons:")
print("1. Side-by-side trajectory plots")
print("2. Attractor visualization")
print("3. Quantitative metrics table")
print("4. Discussion of trade-offs")"""),
]

# =============================================================================
# Generate all notebooks
# =============================================================================

if __name__ == "__main__":
    os.makedirs('notebooks', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    create_notebook(nb01_cells, 'notebooks/01_continuous_time_rnn.ipynb')
    create_notebook(nb02_cells, 'notebooks/02_balanced_rate_network.ipynb')
    create_notebook(nb03_cells, 'notebooks/03_balanced_spiking_network.ipynb')
    create_notebook(nb04_cells, 'notebooks/04_dynamical_analysis.ipynb')
    create_notebook(nb05_cells, 'notebooks/05_synthesis.ipynb')
    
    print("\\nAll notebooks generated!")
