# RNNs as Computational Dynamical Systems

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)](https://jupyter.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> **🚀 Quick Start:** Click any "Open in Colab" badge in the [Tutorial Structure](#-tutorial-structure) table below to launch notebooks directly in your browser—no installation required!

A hands-on tutorial exploring recurrent neural networks through the lens of dynamical systems theory. Students implement the same temporal prediction task (Lorenz-63 attractor reconstruction) across multiple network architectures—from continuous-time RNNs to biologically plausible balanced spiking networks—enabling direct comparison of dynamics, performance, and interpretability.

---

## 🎯 At a Glance

| **What** | **Details** |
|----------|-------------|
| **📚 Format** | 6 Jupyter notebooks (5 core + 1 optional demo) |
| **⏱️ Duration** | ~3.5-4 hours total (3h core, 45min optional) |
| **🧪 Task** | Lorenz-63 chaotic attractor reconstruction |
| **🧠 Models** | 3 architectures: CT-RNN, Balanced Rate, Balanced Spiking |
| **📊 Analysis** | Lyapunov exponents, attractor dimensions, fixed points |
| **💻 Framework** | PyTorch + torchdiffeq (Neural ODEs) + norse (spiking) |
| **🎓 Level** | Graduate neuroscience / computational modeling |
| **🚀 Deployment** | Google Colab (no setup!) or local Jupyter |

**What makes this unique?**
- ✨ Same task across 3+ architectures → direct comparison
- ✨ Dynamical systems lens → analyze learned attractors, chaos, stability
- ✨ Production-ready `src/` code → focus on concepts, not boilerplate
- ✨ Biological constraints → Dale's law, E/I balance, spiking neurons
- ✨ Extensible framework → Notebook 05 shows how to adapt to new tasks

---

## 📑 Table of Contents

- [🎯 Learning Objectives](#-learning-objectives)
- [📚 Tutorial Structure](#-tutorial-structure)
- [📖 Detailed Notebook Descriptions](#-detailed-notebook-descriptions)
- [📊 Tutorial Data Flow](#-tutorial-data-flow)
- [🏗️ Code Organization](#️-code-organization)
- [🧠 The Unifying Task: Lorenz-63](#-the-unifying-task-lorenz-63-attractor-reconstruction)
- [🏗️ Network Architectures](#️-network-architectures-covered)
- [🚀 Quick Start](#-quick-start)
- [📦 Dependencies](#-dependencies)
- [📁 Repository Structure](#-repository-structure)
- [🎓 Target Audience](#-target-audience)
- [📖 References](#-references)

---

## 🎯 Learning Objectives

By the end of this tutorial, students will be able to:

1. **Understand RNNs as dynamical systems**: Formulate recurrent networks as continuous-time ODEs and analyze their state-space dynamics
2. **Implement biologically constrained networks**: Build rate and spiking networks with separate excitatory/inhibitory populations obeying Dale's law
3. **Analyze trained networks**: Compute fixed points, estimate Lyapunov exponents, and visualize learned attractors
4. **Compare architectures**: Evaluate trade-offs between biological plausibility, trainability, and computational efficiency

## 📚 Tutorial Structure

### Core Tutorial Notebooks (Notebooks 00-04)

| # | Notebook | Duration | Open in Colab |
|---|----------|----------|---------------|
| **00** | [**Introduction to Dynamical Systems**](notebooks/00_introduction.ipynb) | 30 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CNNC-Lab/RNNs-tutorial/blob/main/notebooks/00_introduction.ipynb) |
| **01** | [**Continuous-Time RNN**](notebooks/01_continuous_time_rnn.ipynb) | 45 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CNNC-Lab/RNNs-tutorial/blob/main/notebooks/01_continuous_time_rnn.ipynb) |
| **02** | [**Balanced Rate Network**](notebooks/02_balanced_rate_network.ipynb) | 45 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CNNC-Lab/RNNs-tutorial/blob/main/notebooks/02_balanced_rate_network.ipynb) |
| **03** | [**Balanced Spiking Network**](notebooks/03_balanced_spiking_network.ipynb) | 45 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CNNC-Lab/RNNs-tutorial/blob/main/notebooks/03_balanced_spiking_network.ipynb) |
| **04** | [**Synthesis & Comparison**](notebooks/04_synthesis.ipynb) | 30 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CNNC-Lab/RNNs-tutorial/blob/main/notebooks/04_synthesis.ipynb) |

### Optional Extension (Notebook 05)

| # | Notebook | Duration | Open in Colab |
|---|----------|----------|---------------|
| **05** | [**Flip-Flop Working Memory Task**](notebooks/05_flipflop_task.ipynb) | 45 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CNNC-Lab/RNNs-tutorial/blob/main/notebooks/05_flipflop_task.ipynb) |

**Total duration: ~3.5-4 hours** (core: 3h, optional: 45min)

---

## 📖 Detailed Notebook Descriptions

### Notebook 00: Introduction to Dynamical Systems
**[Open Notebook](notebooks/00_introduction.ipynb)** | **[Open in Colab](https://colab.research.google.com/github/CNNC-Lab/RNNs-tutorial/blob/main/notebooks/00_introduction.ipynb)**

**What you'll learn:**
- Fundamentals of dynamical systems: ODEs, phase space, trajectories
- The Lorenz-63 system: chaos, strange attractors, sensitive dependence
- How to formulate RNNs as continuous-time dynamical systems
- Data preparation and normalization for time series prediction

**What you'll do:**
- Visualize the Lorenz butterfly attractor in 3D
- Generate training/validation/test datasets (20,000 timesteps total)
- Save preprocessed data to `data/processed/lorenz_data.npz` for use in notebooks 01-05
- Set up the prediction task: given state at time t, predict state at t+1

**Key concepts:** Phase portraits, fixed points, limit cycles, chaotic attractors, sequence-to-sequence prediction

---

### Notebook 01: Continuous-Time RNN (CT-RNN)
**[Open Notebook](notebooks/01_continuous_time_rnn.ipynb)** | **[Open in Colab](https://colab.research.google.com/github/CNNC-Lab/RNNs-tutorial/blob/main/notebooks/01_continuous_time_rnn.ipynb)**

**What you'll learn:**
- Continuous-time RNN formulation: τ dh/dt = -h + f(Wh + Ux)
- Neural ODEs: differentiable ODE solvers for smooth dynamics
- Adjoint sensitivity method for memory-efficient backpropagation
- How time constants (τ) control network timescales

**What you'll do:**
- Load shared Lorenz dataset using `src.data.create_shared_dataloaders()`
- Instantiate `ContinuousTimeRNN` from `src.models`
- Train the network for 100 epochs (~5-10 min)
- Evaluate performance: R² > 0.99, RMSE < 0.01
- Visualize predictions vs ground truth in 3D phase space

**Key concepts:** Neural ODEs, torchdiffeq solvers, continuous backpropagation, ODE integration methods (Euler, RK4, Dopri5)

---

### Notebook 02: Balanced Excitatory-Inhibitory Rate Network
**[Open Notebook](notebooks/02_balanced_rate_network.ipynb)** | **[Open in Colab](https://colab.research.google.com/github/CNNC-Lab/RNNs-tutorial/blob/main/notebooks/02_balanced_rate_network.ipynb)**

**What you'll learn:**
- Dale's law: neurons are either excitatory (E) or inhibitory (I), not both
- Balanced networks: strong E and I currents that cancel on average
- Separate time constants for E (slow) and I (fast) populations
- How biological constraints affect network dynamics

**What you'll do:**
- Build a network with 48 excitatory + 16 inhibitory rate units
- Enforce Dale's law with `torch.abs()` on recurrent weights (W_EE, W_EI, W_IE, W_II)
- Train using Euler integration (dt=0.1) for discrete-time stepping
- Compare performance to CT-RNN: similar R² with interpretable E/I structure
- Analyze weight matrices and E/I balance

**Key concepts:** Dale's law, excitatory/inhibitory balance, structured connectivity, biological constraints, rate-based models

---

### Notebook 03: Balanced Spiking Network
**[Open Notebook](notebooks/03_balanced_spiking_network.ipynb)** | **[Open in Colab](https://colab.research.google.com/github/CNNC-Lab/RNNs-tutorial/blob/main/notebooks/03_balanced_spiking_network.ipynb)**

**What you'll learn:**
- Leaky integrate-and-fire (LIF) neurons: discrete spikes, membrane dynamics
- Surrogate gradients: making non-differentiable spikes trainable
- Reservoir computing: train only readout layer, freeze recurrent weights
- Rate-based vs spike-based readouts

**What you'll do:**
- Implement LIF neurons using the `norse` library
- Build two networks: (1) fully trained SNN, (2) reservoir with fixed E/I weights
- Train with surrogate gradient descent (straight-through estimator)
- Compare trained vs reservoir: both achieve R² ~0.77 (harder than rate networks!)
- Visualize spike rasters and population firing rates

**Key concepts:** Spiking neurons, membrane potential, surrogate gradients, reservoir computing, liquid state machines, sparse spiking activity

---

### Notebook 04: Synthesis & Comparison
**[Open Notebook](notebooks/04_synthesis.ipynb)** | **[Open in Colab](https://colab.research.google.com/github/CNNC-Lab/RNNs-tutorial/blob/main/notebooks/04_synthesis.ipynb)**

**What you'll learn:**
- How to systematically compare architectures across multiple dimensions
- Trade-offs between biological plausibility and performance
- Autonomous generation vs one-step prediction
- When to use which architecture

**What you'll do:**
- Load all 4 trained models (CT-RNN, Balanced Rate, SNN Trained, SNN Reservoir)
- Create comprehensive comparison table: R², RMSE, parameters, training time
- Test **autonomous generation**: models generate 10,000 timesteps in closed loop
- Compare attractor geometry, Lyapunov exponents, and correlation dimensions
- Visualize one-step predictions and long-term autonomous trajectories side-by-side

**Key insights:**
- **Best prediction accuracy**: CT-RNN (R² = 1.000)
- **Best balance of bio-plausibility & performance**: Balanced Rate (R² = 1.000)
- **Most biologically realistic**: SNNs (discrete spikes, but R² = 0.77)
- **Fastest inference**: Balanced Rate (discrete time stepping)
- **Most parameter efficient**: SNN Reservoir (only 867 trainable params!)

**Discussion prompts:** When would you use each architecture? What are the costs of biological realism? How does chaos affect long-term generation?

---

### Notebook 05: Flip-Flop Working Memory Task (Optional)
**[Open Notebook](notebooks/05_flipflop_task.ipynb)** | **[Open in Colab](https://colab.research.google.com/github/CNNC-Lab/RNNs-tutorial/blob/main/notebooks/05_flipflop_task.ipynb)**

**What you'll learn:**
- How to extend the framework to cognitive tasks beyond Lorenz
- Working memory: maintaining state without continuous input
- State-space analysis with PCA
- Fixed point structure for discrete state tasks

**What you'll do:**
- Implement 3-bit flip-flop task using `src.data.flipflop` module
- Train CT-RNN and Balanced Rate networks on toggle commands (+1/-1 pulses)
- Use `return_all_outputs=True` for sequence-to-sequence prediction
- Visualize 8 flip-flop states in PCA-reduced 2D space
- Analyze fixed points: do networks learn 8 stable attractors?

**Key concepts:** Working memory, discrete state machines, state-space visualization, sequence-to-sequence learning, PCA

**This serves as a template** for adapting the framework to your own tasks: delayed match-to-sample, context-dependent integration, motor timing, etc.

## 📊 Tutorial Data Flow

**Shared Dataset Approach** ensures all models train on identical data for fair comparison:

```
┌─────────────────────────────────────────────────────────────┐
│  Notebook 00: Introduction                                  │
│  • Generates Lorenz trajectories (20,000 timesteps)         │
│  • Normalizes: (x - mean) / std                             │
│  • Splits: train (70%) / val (15%) / test (15%)            │
│  • Saves: data/processed/lorenz_data.npz                    │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ├─────────────┬─────────────┬─────────────┐
                   ▼             ▼             ▼             ▼
          ┌────────────┐ ┌────────────┐ ┌──────────────┐ ┌──────────────┐
          │ Notebook 01│ │ Notebook 02│ │ Notebook 03  │ │ Notebook 05  │
          │  CT-RNN    │ │ Bal. Rate  │ │ Bal. Spiking │ │  Flip-Flop   │
          └──────┬─────┘ └──────┬─────┘ └──────┬───────┘ └──────────────┘
                 │              │              │          (different task)
                 │ (saves checkpoints)         │
                 ▼              ▼              ▼
          ┌───────────────────────────────────────┐
          │  checkpoints/                         │
          │  • ctrnn_best.pt                      │
          │  • balanced_rate_best.pt              │
          │  • snn_trained_best.pt                │
          │  • snn_reservoir_best.pt              │
          └───────────┬───────────────────────────┘
                      │
                      ▼
              ┌──────────────┐
              │  Notebook 04 │
              │  Synthesis   │
              └──────────────┘
```

**Key Pattern:**
```python
# In notebooks 01-04: Load shared Lorenz data
from src.data import create_shared_dataloaders
train_loader, val_loader, test_loader, info = create_shared_dataloaders()

# In notebook 05: Generate flip-flop data
from src.data.flipflop import create_flipflop_dataloaders
train_loader, val_loader, test_loader, info = create_flipflop_dataloaders()
```

**Benefits:**
- ✅ **Consistency**: All models see identical training examples
- ✅ **Efficiency**: No duplication of data generation (20,000 → 1 save + 4 loads)
- ✅ **Fair Comparison**: Same normalization, same splits, same random seed
- ✅ **Reproducibility**: Fixed dataset eliminates variability source
- ✅ **Extensibility**: Notebook 05 shows how to swap in new tasks

## 🏗️ Code Organization

All core functionality is in the `src/` package, enabling notebooks to focus on pedagogy while using production-ready code.

### 📦 `src/` Package Structure

#### **`src/__init__.py`** - Environment Setup
```python
from src import setup_environment, check_dependencies

device = setup_environment()  # Sets random seeds, configures matplotlib, detects GPU
check_dependencies()          # Validates package installations
```

**Functions:**
- `setup_environment()` - Unified environment setup (seeds for reproducibility, matplotlib backend, device selection)
- `check_dependencies()` - Validates all required packages are installed

---

#### **`src/data/`** - Data Generation & Loading
**File:** [`src/data/__init__.py`](src/data/__init__.py)

**Lorenz System Functions:**
```python
from src.data import (
    generate_lorenz_trajectory,  # Generate Lorenz-63 trajectories
    create_lorenz_dataloaders,   # Create PyTorch DataLoaders
    save_lorenz_dataset,         # Save preprocessed data
    create_shared_dataloaders,   # Load shared dataset (notebooks 01-05)
)

# Example usage
train_loader, val_loader, test_loader, info = create_shared_dataloaders()
mean, std = info['normalization']['mean'], info['normalization']['std']
```

**Key Functions:**
- `generate_lorenz_trajectory()` - Integrates Lorenz ODEs with scipy.solve_ivp
- `create_lorenz_dataloaders()` - Generates fresh data with train/val/test split
- `save_lorenz_dataset()` - Saves preprocessed data to .npz for sharing
- `load_lorenz_dataset()` - Loads saved dataset with validation
- `create_shared_dataloaders()` - **KEY**: Creates DataLoaders from saved data (used in notebooks 01-05)

**File:** [`src/data/flipflop.py`](src/data/flipflop.py)

**Flip-Flop Task Functions:**
```python
from src.data.flipflop import (
    generate_flipflop_trial,      # Single trial with toggle commands
    create_flipflop_dataloaders,  # Train/val/test splits for flip-flop
    compute_flipflop_accuracy,    # Bit-wise accuracy metric
    plot_flipflop_trial,          # Visualization
)
```

**Key Classes:**
- `LorenzDataset` - PyTorch Dataset for Lorenz sequences
- `FlipFlopDataset` - PyTorch Dataset for flip-flop trials

---

#### **`src/models/`** - Neural Network Architectures
**File:** [`src/models/__init__.py`](src/models/__init__.py)

```python
from src.models import (
    ContinuousTimeRNN, CTRNNCell,           # Neural ODE-based CT-RNN
    BalancedRateNetwork, EIRateCell,         # E/I rate network with Dale's law
    BalancedSpikingNetwork,                  # LIF spiking network with norse
    create_spiking_reservoir,                # Helper for reservoir initialization
)
```

**File:** [`src/models/ctrnn.py`](src/models/ctrnn.py)

**Classes:**
- `CTRNNCell` - Continuous-time RNN dynamics (ODE right-hand side)
- `ContinuousTimeRNN` - Complete model with encoder/decoder
  - `forward(x, return_hidden=False, return_all_outputs=False)` - Main forward pass
  - `generate(initial_state, n_steps, dt)` - Autonomous generation (closed-loop)
  - `integrate_continuous(h0, t, x)` - Arbitrary-time integration

**Parameters:**
- `solver` - ODE solver: 'euler', 'rk4', 'dopri5' (adaptive)
- `tau` - Time constant (default: 1.0)
- `use_adjoint` - Use adjoint method for memory-efficient backprop

**File:** [`src/models/balanced_rate.py`](src/models/balanced_rate.py)

**Classes:**
- `EIRateCell` - Balanced E/I rate dynamics with Dale's law
- `BalancedRateNetwork` - Complete E/I rate network
  - `step(r_e, r_i, x)` - Single Euler integration step
  - `forward(x, return_all_outputs=False)` - Process full sequence

**Key Features:**
- Separate excitatory (48 units) and inhibitory (16 units) populations
- Dale's law enforcement: W_EE, W_EI, W_IE, W_II all non-negative
- Different time constants: τ_E = 1.0, τ_I = 0.5
- Euler integration for discrete-time stepping

**File:** [`src/models/balanced_spiking.py`](src/models/balanced_spiking.py)

**Classes:**
- `BalancedSpikingNetwork` - LIF spiking network using norse
  - `forward(x)` - Returns spike-based or rate-based readout
  - Built-in E/I balance with fixed or trainable recurrent weights

**Parameters:**
- `readout_mode` - 'rate' (spike count average) or 'membrane' (voltage-based)
- `fixed_weights` - True for reservoir, False for trainable recurrent weights
- `tau_mem_e`, `tau_mem_i` - Membrane time constants

---

#### **`src/utils/`** - Training & Visualization
**File:** [`src/utils/__init__.py`](src/utils/__init__.py)

```python
from src.utils import (
    train_model,                        # Universal training loop
    evaluate,                           # Model evaluation
    compute_prediction_metrics,         # R², RMSE, MAE
    plot_training_history,              # Loss curves
    plot_lorenz_intro,                  # 3D + time series visualization
    plot_prediction_comparison_detailed, # 3-panel comparison
    plot_scatter_prediction,            # Scatter plots per dimension
)
```

**Key Functions:**

**Training:**
```python
history = train_model(
    model, train_loader, val_loader,
    n_epochs=100, lr=1e-3, device=device,
    checkpoint_dir='checkpoints', model_name='ctrnn'
)
```
- Automatic checkpointing (saves best model based on val loss)
- Early stopping support
- Returns training history dict with train/val losses

**Evaluation:**
```python
test_loss, predictions, targets = evaluate(model, test_loader, criterion, device)
metrics = compute_prediction_metrics(targets, predictions)
# Returns: {'r2': ..., 'rmse': ..., 'mae': ..., 'r2_per_dim': [...]}
```

**Visualization:**
- `plot_lorenz_intro()` - Educational 3D attractor + time series
- `plot_prediction_comparison_detailed()` - 3-panel layout with R² scores
- `plot_scatter_prediction()` - Scatter plots showing prediction quality per dimension

---

#### **`src/analysis/`** - Dynamical Systems Analysis
**File:** [`src/analysis/__init__.py`](src/analysis/__init__.py)

```python
from src.analysis import (
    estimate_lyapunov_spectrum_simple,  # Largest Lyapunov exponent
    compute_attractor_dimension,        # Correlation dimension
    find_fixed_points,                  # Fixed point finding
    analyze_fixed_point_stability,      # Eigenvalue analysis
)
```

**Key Functions:**

**Lyapunov Exponents:**
```python
lyap = estimate_lyapunov_spectrum_simple(trajectory, dt=0.01)
# Returns largest Lyapunov exponent (λ_max)
# λ > 0: chaos, λ = 0: neutrally stable, λ < 0: stable
```

**Attractor Dimension:**
```python
dim = compute_attractor_dimension(trajectory, n_points=2000)
# Returns correlation dimension estimate
# Lorenz: ~2.05, Low-dim chaos: 1-3, High-dim: > 10
```

**Fixed Points:**
```python
fixed_points = find_fixed_points(model, n_inits=100, lr=0.01)
# Finds equilibrium points where dh/dt = 0
```

**Analysis Tools:**
- State-space embedding
- Poincaré sections
- Recurrence plots
- Jacobian analysis at fixed points

---

### 🎯 Design Philosophy

**Why separate `src/` from `notebooks/`?**

1. **Pedagogical Focus**: Notebooks explain *concepts* without implementation details
2. **Code Reuse**: Same training loop, evaluation, plotting across all notebooks
3. **Maintainability**: Bug fixes in one place benefit all notebooks
4. **Production-Ready**: `src/` code is well-tested, documented, type-hinted
5. **Extensibility**: Easy to add new models/tasks by following existing patterns

**Pattern for notebooks:**
```python
# Setup (2-3 lines)
from src import setup_environment
from src.data import create_shared_dataloaders
from src.models import ContinuousTimeRNN
from src.utils import train_model, evaluate

# Load data (1 line)
train_loader, val_loader, test_loader, info = create_shared_dataloaders()

# Create model (1 line)
model = ContinuousTimeRNN(input_size=3, hidden_size=64, output_size=3)

# Train (1 line)
history = train_model(model, train_loader, val_loader, n_epochs=100)

# Evaluate (2 lines)
test_loss, preds, targets = evaluate(model, test_loader, criterion, device)
metrics = compute_prediction_metrics(targets, preds)
```

The notebooks focus on **interpreting results**, not implementing infrastructure.


## 🧠 The Unifying Task: Lorenz-63 Attractor Reconstruction

All networks are trained on the same task: predict the next state of the chaotic Lorenz system given its current state. This allows direct comparison across architectures.

```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y  
dz/dt = xy - βz

Parameters: σ=10, ρ=28, β=8/3
```

The Lorenz system exhibits:
- **Chaotic dynamics**: Sensitive dependence on initial conditions
- **Strange attractor**: The famous "butterfly" shape
- **Rich structure**: Fixed points, limit cycles, homoclinic orbits

## 🏗️ Network Architectures Covered

### 1. 🌊 Continuous-Time RNN (CT-RNN)

**Dynamics:**
```
τ dh/dt = -h + f(Wh + Ux + b)
```

**Properties:**
- **Smooth dynamics** amenable to ODE analysis and fixed point finding
- **Neural ODEs**: Integrated with torchdiffeq (Euler, RK4, Dopri5 solvers)
- **Adjoint sensitivity**: Memory-efficient backpropagation through time
- **Time constants**: τ controls network timescale (higher = slower dynamics)

**Use case:** General-purpose temporal modeling, interpretable dynamics, research settings

---

### 2. ⚖️ Balanced Excitatory-Inhibitory Rate Network

**Dynamics:**
```
τ_E dr_E/dt = -r_E + ReLU(W_EE·r_E - W_EI·r_I + I_ext)  [Excitatory]
τ_I dr_I/dt = -r_I + ReLU(W_IE·r_E - W_II·r_I)          [Inhibitory]
```

**Properties:**
- **Dale's law**: Separate E (48 units) and I (16 units) populations, all weights ≥ 0
- **Balanced dynamics**: Strong E and I currents cancel on average → irregular activity
- **Biologically interpretable**: Can map to cortical circuit connectivity patterns
- **Different timescales**: τ_E = 1.0 (slow), τ_I = 0.5 (fast) mimics biology

**Use case:** Neuroscience applications, interpretable E/I contributions, cortical modeling

---

### 3. ⚡ Balanced Spiking Network

**Dynamics:**
```
τ_m dV/dt = -(V - V_rest) + I_syn + I_ext
if V(t) > V_thresh: emit spike, V → V_reset
```

**Properties:**
- **Leaky Integrate-and-Fire (LIF)** neurons with discrete spike events
- **Sparse spiking activity**: ~10-20 Hz firing rates, event-driven computation
- **Surrogate gradients**: Straight-through estimator for backprop through spikes
- **Reservoir computing**: Option to freeze recurrent weights, train only readout

**Use case:** Neuromorphic hardware, energy-efficient computing, most biologically realistic

---

### 📊 Quick Architecture Comparison

| Feature | CT-RNN | Balanced Rate | Spiking |
|---------|--------|---------------|---------|
| **Biological Realism** | ⭐ Low | ⭐⭐ Medium | ⭐⭐⭐ High |
| **Training Ease** | ⭐⭐⭐ Easy | ⭐⭐⭐ Easy | ⭐ Hard |
| **Performance** | ⭐⭐⭐ Best | ⭐⭐⭐ Best | ⭐⭐ Good |
| **Interpretability** | ⭐⭐ Medium | ⭐⭐⭐ High | ⭐⭐ Medium |
| **Hardware Efficiency** | ⭐ Low | ⭐⭐ Medium | ⭐⭐⭐ High |
| **Neuromorphic Ready** | ❌ No | ❌ No | ✅ Yes |

*See Notebook 04 for detailed quantitative comparison*

## 🚀 Quick Start

### Option 1: 🌐 Google Colab (Recommended - Zero Setup!)

**Best for:** Quick exploration, no installation, free GPU access

1. Click any **"Open in Colab"** badge in the [Tutorial Structure](#-tutorial-structure) table
2. The notebook opens in your browser - start running cells immediately!
3. Dependencies install automatically in the first cell
4. All 7 notebooks work standalone in Colab

**Workflow:**
```
Click Colab Badge → Notebook Opens → Run First Cell (installs deps) → Start Learning!
                                     ↓
                          Takes ~30 seconds, happens once per session
```

**Note:** Colab sessions are temporary. Download any trained models or figures you want to keep.

---

### Option 2: 💻 Local Installation (Full Control)

**Best for:** Developing your own models, working offline, keeping results permanently

```bash
# Clone the repository
git clone https://github.com/CNNC-Lab/RNNs-tutorial.git
cd RNNs-tutorial

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode (recommended)
pip install -e .

# Launch Jupyter
jupyter notebook notebooks/
```

### ⚠️ Important: Notebook Execution Order

**For core tutorial (notebooks 00-04), run in sequence:**

```
00_introduction.ipynb          [MUST RUN FIRST]
    ↓ (generates shared dataset)

01_continuous_time_rnn.ipynb   [Run after 00]
02_balanced_rate_network.ipynb [Run after 00]
03_balanced_spiking_network.ipynb [Run after 00]
    ↓ (save trained model checkpoints)

04_synthesis.ipynb             [Run after 01-03]
```

**Notebook 05 (flip-flop) is independent** and can be run anytime after notebook 00.

**Why this order matters:**
- **Notebook 00** generates `data/processed/lorenz_data.npz` used by all subsequent notebooks
- **Notebooks 01-03** train models and save checkpoints to `checkpoints/`
- **Notebook 04** loads these checkpoints for synthesis and comparison
- All notebooks import from `src/` package (automatically available after `pip install -e .`)

**Common imports in every notebook:**
```python
from src import setup_environment, check_dependencies
from src.data import create_shared_dataloaders
from src.models import ContinuousTimeRNN, BalancedRateNetwork, BalancedSpikingNetwork
from src.utils import train_model, evaluate, compute_prediction_metrics
from src.analysis import estimate_lyapunov_spectrum_simple, compute_attractor_dimension
```

See [Detailed Notebook Descriptions](#-detailed-notebook-descriptions) for what each notebook covers.

## 📦 Dependencies

Core:
- `torch >= 2.0`
- `numpy`, `scipy`, `matplotlib`
- `torchdiffeq` (for Neural ODEs)
- `norse` (for spiking networks)

See [requirements.txt](requirements.txt) for complete list.

## 📁 Repository Structure

```
rnn-dynamical-systems-tutorial/
├── notebooks/                 # Jupyter notebooks (main tutorial content)
│   ├── 00_introduction.ipynb
│   ├── 01_continuous_time_rnn.ipynb
│   ├── 02_balanced_rate_network.ipynb
│   ├── 03_balanced_spiking_network.ipynb
│   ├── 04_synthesis.ipynb
│   └── 05_flipflop_task.ipynb  # Optional: Different task demo
├── src/                       # Reusable Python modules
│   ├── models/               # Network architectures
│   ├── data/                 # Data generation utilities
│   │   ├── __init__.py       # Lorenz system
│   │   └── flipflop.py       # 3-bit flip-flop task
│   ├── analysis/             # Dynamical systems analysis tools
│   └── utils/                # Plotting, helpers
├── figures/                   # Generated figures
├── data/                      # Datasets (generated)
├── checkpoints/              # Pre-trained models
├── docs/                      # Additional documentation
├── requirements.txt
└── README.md
```

## 🎓 Target Audience & Prerequisites

**Designed for:** Graduate students in computational/systems neuroscience, machine learning researchers interested in biological constraints, and anyone curious about RNNs as dynamical systems.

### Prerequisites

**Required:**
- ✅ **Python basics**: functions, loops, numpy arrays
- ✅ **PyTorch fundamentals**: tensors, `nn.Module`, training loops (can learn as you go)
- ✅ **Differential equations**: understand dx/dt = f(x), phase space concepts
- ✅ **Neural networks**: basic RNN concept (hidden state, recurrence)

**Helpful but not required:**
- 📚 Dynamical systems theory (fixed points, attractors, Lyapunov exponents)
- 📚 Computational neuroscience (E/I balance, Dale's law, LIF neurons)
- 📚 Experience with Jupyter notebooks

**No prior experience needed with:**
- ❌ Neural ODEs (torchdiffeq) - we introduce this
- ❌ Spiking neural networks (norse) - tutorial covers basics
- ❌ Chaos theory or nonlinear dynamics - explained from scratch

## 📖 References

### Dynamical Systems & RNNs
- Sussillo, D. (2014). Neural circuits as computational dynamical systems. *Current Opinion in Neurobiology*.
- Durstewitz, D. et al. (2023). Reconstructing computational dynamics from neural measurements with RNNs.

### Balanced Networks
- van Vreeswijk, C. & Sompolinsky, H. (1996). Chaos in neuronal networks with balanced excitation and inhibition.
- Ingrosso, A. & Abbott, L.F. (2019). Training dynamically balanced excitatory-inhibitory networks. *PLOS ONE*.

### Spiking Networks
- Maass, W. et al. (2002). Real-time computing without stable states: A new framework for neural computation.
- Cramer, B. et al. (2020). The Heidelberg Spiking Data Sets. *Zenke Lab*.

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## ✉️ Contact

**Renato Duarte**  
Center for Neuroscience and Cell Biology (CNC-UC)  
University of Coimbra, Portugal

---

*This tutorial was developed for the Integrative Neuroscience graduate program.*
