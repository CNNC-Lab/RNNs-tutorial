"""
RNN Dynamical Systems Tutorial
==============================

A tutorial exploring RNNs through the lens of dynamical systems theory.
"""

__version__ = "0.1.0"


# =============================================================================
# Environment Setup Utilities
# =============================================================================

def setup_environment():
    """
    Setup environment for Colab or local execution.

    Sets random seeds, configures matplotlib, and detects device.

    Returns
    -------
    device : str
        'cuda' or 'cpu'

    Examples
    --------
    >>> from src import setup_environment
    >>> device = setup_environment()
    ✓ Environment ready. Using device: cpu
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Configure matplotlib
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Environment ready. Using device: {device}")

    return device


def check_dependencies():
    """
    Check that all required packages are installed.

    Returns
    -------
    bool
        True if all dependencies are available

    Examples
    --------
    >>> from src import check_dependencies
    >>> check_dependencies()
    ✓ All dependencies installed
    True
    """
    required = ['torch', 'numpy', 'scipy', 'matplotlib', 'torchdiffeq', 'norse']
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"⚠ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False

    print("✓ All dependencies installed")
    return True
