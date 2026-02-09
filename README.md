# RNNs as Computational Dynamical Systems

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Open in Colab:**
[![00 Introduction](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CNNC-Lab/RNNs-tutorial/blob/main/notebooks/00_introduction.ipynb)
[![01 CT-RNN](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CNNC-Lab/RNNs-tutorial/blob/main/notebooks/01_continuous_time_rnn.ipynb)
[![02 Balanced Rate](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CNNC-Lab/RNNs-tutorial/blob/main/notebooks/02_balanced_rate_network.ipynb)
[![03 Spiking](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CNNC-Lab/RNNs-tutorial/blob/main/notebooks/03_balanced_spiking_network.ipynb)
[![04 Analysis](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CNNC-Lab/RNNs-tutorial/blob/main/notebooks/04_dynamical_analysis.ipynb)
[![05 Synthesis](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CNNC-Lab/RNNs-tutorial/blob/main/notebooks/05_synthesis.ipynb)
[![06 Flip-Flop](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CNNC-Lab/RNNs-tutorial/blob/main/notebooks/06_flipflop_task.ipynb)

A hands-on tutorial exploring recurrent neural networks through the lens of dynamical systems theory. Students implement the same temporal prediction task (Lorenz-63 attractor reconstruction) across multiple network architectures—from continuous-time RNNs to biologically plausible balanced spiking networks—enabling direct comparison of dynamics, performance, and interpretability.

## 🎯 Learning Objectives

By the end of this tutorial, students will be able to:

1. **Understand RNNs as dynamical systems**: Formulate recurrent networks as continuous-time ODEs and analyze their state-space dynamics
2. **Implement biologically constrained networks**: Build rate and spiking networks with separate excitatory/inhibitory populations obeying Dale's law
3. **Analyze trained networks**: Compute fixed points, estimate Lyapunov exponents, and visualize learned attractors
4. **Compare architectures**: Evaluate trade-offs between biological plausibility, trainability, and computational efficiency

## 📚 Tutorial Structure

| Notebook | Duration | Description |
|----------|----------|-------------|
| [00_Introduction](notebooks/00_introduction.ipynb) | 30 min | Dynamical systems primer, Lorenz-63, task setup |
| [01_Continuous_Time_RNN](notebooks/01_continuous_time_rnn.ipynb) | 45 min | CT-RNN implementation, Neural ODEs, training |
| [02_Balanced_Rate_Network](notebooks/02_balanced_rate_network.ipynb) | 45 min | E/I populations, Dale's law, balanced dynamics |
| [03_Balanced_Spiking_Network](notebooks/03_balanced_spiking_network.ipynb) | 45 min | LIF neurons with norse, reservoir computing |
| [04_Dynamical_Analysis](notebooks/04_dynamical_analysis.ipynb) | 30 min | Fixed points, Lyapunov exponents, attractor comparison |
| [05_Synthesis](notebooks/05_synthesis.ipynb) | 30 min | Architecture comparison, discussion prompts |
| [06_Flip-Flop_Task](notebooks/06_flipflop_task.ipynb) | 45 min | **Demo**: Working memory task, PCA, fixed points |

**Total duration: ~4-4.5 hours** (core: 3.5h, optional demo: 45min)

## 📊 Tutorial Data Flow

**Shared Dataset Approach** for consistency:
1. **Notebook 00** generates Lorenz data and saves to `data/processed/lorenz_data.npz`
2. **Notebooks 01-05** load this shared dataset via `src.data.create_shared_dataloaders()`
3. **Notebook 06** demonstrates a different task (3-bit flip-flop) using the same framework

This ensures:
- ✅ Consistency across all models
- ✅ No duplication of data generation
- ✅ Fair comparison between architectures
- ✅ Reproducible results
- ✅ Easy adaptation to new tasks

## 🏗️ Code Organization

All core functionality is in the `src/` package:
- **`src.data`**: Lorenz generation, flip-flop task, datasets, data loading utilities
- **`src.models`**: CT-RNN, Balanced Rate/Spiking network implementations
- **`src.utils`**: Training loops, evaluation metrics, visualization
- **`src.analysis`**: Fixed points, Lyapunov exponents, dynamical analysis

The notebooks focus on pedagogy while leveraging production-ready code from `src/`.

### 🎯 Notebook 06: Extending to New Tasks

The optional **flip-flop notebook** demonstrates how to apply the framework to a different cognitive task:
- **Task**: 3-bit flip-flop working memory (toggle and maintain state)
- **Models**: CT-RNN and Balanced Rate Network trained on discrete states
- **Analysis**: PCA state-space visualization, fixed point structure for 8 states
- **Purpose**: Shows how to extend the framework to your own tasks

This serves as a template for students to explore other tasks (delayed match-to-sample, context-dependent integration, go/no-go, etc.).

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

### 1. Continuous-Time RNN (CT-RNN)
```
τ dh/dt = -h + f(Wh + Ux + b)
```
- Smooth dynamics amenable to ODE analysis
- Natural connection to neural mass models
- Trained with adjoint sensitivity methods

### 2. Balanced Excitatory-Inhibitory Rate Network
```
τ_E dr_E/dt = -r_E + φ(W_EE·r_E - W_EI·r_I + I_ext)
τ_I dr_I/dt = -r_I + φ(W_IE·r_E - W_II·r_I)
```
- Separate E/I populations (Dale's law)
- Dynamically balanced regime
- Biologically interpretable connectivity

### 3. Balanced Spiking Network
```
τ_m dV/dt = -(V - V_rest) + I_syn + I_ext
if V > V_thresh: spike, V → V_reset
```
- Leaky integrate-and-fire neurons
- Sparse random recurrent connectivity
- Reservoir computing: train readout only

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

Click any of the "Open in Colab" badges above to launch the notebooks directly in Google Colab. All dependencies are installed automatically.

### Option 2: Local Installation

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

**Run notebooks in sequence!**
1. **00_introduction.ipynb** - Generates and saves shared dataset
2. **01-03** - Can be run independently after notebook 00
3. **04-05** - Require trained models from notebooks 01-03

Each notebook imports from the `src/` package:
```python
from src import setup_environment
from src.data import create_shared_dataloaders
from src.models import ContinuousTimeRNN
from src.utils import train_model, compute_prediction_metrics
```

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
│   ├── 04_dynamical_analysis.ipynb
│   ├── 05_synthesis.ipynb
│   └── 06_flipflop_task.ipynb  # Optional: Different task demo
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

## 🎓 Target Audience

Graduate students in integrative/computational neuroscience with:
- Basic Python programming experience
- Familiarity with differential equations
- Exposure to neural network concepts
- Interest in biological neural circuits

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
