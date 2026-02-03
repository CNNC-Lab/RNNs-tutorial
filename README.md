# RNNs as Computational Dynamical Systems

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

**Total duration: ~3.5-4 hours**

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

Click the "Open in Colab" badge on any notebook. All dependencies are installed automatically.

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/rnn-dynamical-systems-tutorial.git
cd rnn-dynamical-systems-tutorial

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/
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
│   └── 05_synthesis.ipynb
├── src/                       # Reusable Python modules
│   ├── models/               # Network architectures
│   ├── data/                 # Data generation utilities
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
