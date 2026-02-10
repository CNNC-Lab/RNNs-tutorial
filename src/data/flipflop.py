"""
3-Bit Flip-Flop Task
====================

A working memory task where networks must maintain the state of 3 independent flip-flops.

Task Description:
- 3 input channels (one per flip-flop)
- 3 output channels (current state of each flip-flop)
- Input commands: +1 = turn ON, -1 = turn OFF, 0 = no command
- Networks must remember and maintain state over time
- Tests working memory and temporal integration

This is a classic benchmark from:
- Sussillo & Barak (2013) - Opening the Black Box
- Mante et al. (2013) - Context-dependent computation
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


def generate_flipflop_trial(
    n_bits: int = 3,
    seq_length: int = 100,
    pulse_prob: float = 0.1,
    pulse_duration: int = 1,
    dt: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single flip-flop trial.

    Parameters
    ----------
    n_bits : int
        Number of flip-flops (default: 3)
    seq_length : int
        Length of sequence
    pulse_prob : float
        Probability of a pulse on each channel at each timestep
    pulse_duration : int
        Duration of input pulses (timesteps)
    dt : float
        Time step (for compatibility, not used in discrete task)
    seed : int, optional
        Random seed

    Returns
    -------
    inputs : np.ndarray
        Input sequence, shape (seq_length, n_bits)
        Commands: +1 = turn ON, -1 = turn OFF, 0 = no command
    targets : np.ndarray
        Target outputs, shape (seq_length, n_bits)
        Current state of each flip-flop (0 or 1)
    """
    if seed is not None:
        np.random.seed(seed)

    inputs = np.zeros((seq_length, n_bits))
    targets = np.zeros((seq_length, n_bits))

    # Initialize flip-flop states (all off)
    state = np.zeros(n_bits)

    for t in range(seq_length):
        # Generate random commands for each bit
        for i in range(n_bits):
            if np.random.rand() < pulse_prob:
                # Randomly choose to turn ON or OFF
                command = 1.0 if np.random.rand() < 0.5 else -1.0

                # Apply command
                if command > 0:
                    state[i] = 1.0  # Turn ON
                else:
                    state[i] = 0.0  # Turn OFF

                # Set input pulse for specified duration
                for d in range(pulse_duration):
                    if t + d < seq_length:
                        inputs[t + d, i] = command

        # Record current state as target
        targets[t] = state.copy()

    return inputs, targets


def generate_flipflop_dataset(
    n_trials: int = 1000,
    n_bits: int = 3,
    seq_length: int = 100,
    pulse_prob: float = 0.1,
    pulse_duration: int = 1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset of flip-flop trials.

    Parameters
    ----------
    n_trials : int
        Number of trials to generate
    n_bits : int
        Number of flip-flops
    seq_length : int
        Length of each sequence
    pulse_prob : float
        Probability of pulse on each channel
    pulse_duration : int
        Duration of input pulses
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    inputs : np.ndarray
        Input sequences, shape (n_trials, seq_length, n_bits)
    targets : np.ndarray
        Target outputs, shape (n_trials, seq_length, n_bits)
    """
    if seed is not None:
        np.random.seed(seed)

    inputs = []
    targets = []

    for i in range(n_trials):
        trial_seed = None if seed is None else seed + i
        inp, tgt = generate_flipflop_trial(
            n_bits=n_bits,
            seq_length=seq_length,
            pulse_prob=pulse_prob,
            pulse_duration=pulse_duration,
            seed=trial_seed
        )
        inputs.append(inp)
        targets.append(tgt)

    return np.array(inputs), np.array(targets)


class FlipFlopDataset(Dataset):
    """
    PyTorch Dataset for 3-bit flip-flop task.

    Parameters
    ----------
    n_trials : int
        Number of trials
    n_bits : int
        Number of flip-flops
    seq_length : int
        Length of each sequence
    pulse_prob : float
        Probability of pulse per timestep per channel
    pulse_duration : int
        Duration of input pulses
    seed : int, optional
        Random seed
    """

    def __init__(
        self,
        n_trials: int = 1000,
        n_bits: int = 3,
        seq_length: int = 100,
        pulse_prob: float = 0.1,
        pulse_duration: int = 1,
        seed: Optional[int] = None
    ):
        self.n_trials = n_trials
        self.n_bits = n_bits
        self.seq_length = seq_length

        # Generate dataset
        inputs, targets = generate_flipflop_dataset(
            n_trials=n_trials,
            n_bits=n_bits,
            seq_length=seq_length,
            pulse_prob=pulse_prob,
            pulse_duration=pulse_duration,
            seed=seed
        )

        # Convert to tensors
        self.inputs = torch.FloatTensor(inputs)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def create_flipflop_dataloaders(
    train_trials: int = 800,
    val_trials: int = 100,
    test_trials: int = 100,
    n_bits: int = 3,
    seq_length: int = 100,
    pulse_prob: float = 0.1,
    batch_size: int = 32,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Create train/val/test dataloaders for flip-flop task.

    Parameters
    ----------
    train_trials, val_trials, test_trials : int
        Number of trials for each split
    n_bits : int
        Number of flip-flops
    seq_length : int
        Sequence length
    pulse_prob : float
        Pulse probability
    batch_size : int
        Batch size
    seed : int
        Random seed

    Returns
    -------
    train_loader : DataLoader
        Training dataloader
    val_loader : DataLoader
        Validation dataloader
    test_loader : DataLoader
        Test dataloader
    info : dict
        Dataset information
    """
    # Create datasets
    train_dataset = FlipFlopDataset(
        n_trials=train_trials,
        n_bits=n_bits,
        seq_length=seq_length,
        pulse_prob=pulse_prob,
        seed=seed
    )

    val_dataset = FlipFlopDataset(
        n_trials=val_trials,
        n_bits=n_bits,
        seq_length=seq_length,
        pulse_prob=pulse_prob,
        seed=seed + 1000
    )

    test_dataset = FlipFlopDataset(
        n_trials=test_trials,
        n_bits=n_bits,
        seq_length=seq_length,
        pulse_prob=pulse_prob,
        seed=seed + 2000
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    info = {
        'n_bits': n_bits,
        'seq_length': seq_length,
        'pulse_prob': pulse_prob,
        'train_trials': train_trials,
        'val_trials': val_trials,
        'test_trials': test_trials,
        'task': 'flip-flop',
        'description': '3-bit flip-flop working memory task'
    }

    return train_loader, val_loader, test_loader, info


# =============================================================================
# Utility Functions
# =============================================================================

def compute_flipflop_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute accuracy for flip-flop task.

    Accuracy is the fraction of correctly predicted bits.

    Parameters
    ----------
    predictions : np.ndarray
        Network outputs, shape (..., n_bits)
    targets : np.ndarray
        Target states, shape (..., n_bits)

    Returns
    -------
    accuracy : float
        Bit-wise accuracy (0 to 1)
    """
    # Threshold predictions at 0.5
    pred_bits = (predictions > 0.5).astype(int)
    target_bits = targets.astype(int)

    # Compute accuracy
    correct = (pred_bits == target_bits).sum()
    total = np.prod(target_bits.shape)

    return correct / total


def plot_flipflop_trial(
    inputs: np.ndarray,
    targets: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    trial_idx: int = 0,
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Visualize a single flip-flop trial.

    Parameters
    ----------
    inputs : np.ndarray
        Input pulses, shape (n_trials, seq_length, n_bits) or (seq_length, n_bits)
        Values: +1 (turn ON), -1 (turn OFF), 0 (no command)
    targets : np.ndarray
        Target states, same shape as inputs
    predictions : np.ndarray, optional
        Network predictions, same shape
    trial_idx : int
        Which trial to plot (if batch)
    figsize : tuple
        Figure size
    """
    import matplotlib.pyplot as plt

    # Handle batch dimension
    if inputs.ndim == 3:
        inp = inputs[trial_idx]
        tgt = targets[trial_idx]
        pred = predictions[trial_idx] if predictions is not None else None
    else:
        inp = inputs
        tgt = targets
        pred = predictions

    seq_length, n_bits = inp.shape
    t = np.arange(seq_length)

    fig, axes = plt.subplots(n_bits, 1, figsize=figsize, sharex=True)
    if n_bits == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        # Plot input pulses with different colors for +1 and -1
        # ON commands (+1): green fill
        on_mask = inp[:, i] > 0
        ax.fill_between(t, 0, inp[:, i], where=on_mask, alpha=0.4,
                        label='ON command (+1)', color='green', step='mid')

        # OFF commands (-1): red fill (plot as positive for visualization)
        off_mask = inp[:, i] < 0
        ax.fill_between(t, 0, -inp[:, i], where=off_mask, alpha=0.4,
                        label='OFF command (-1)', color='red', step='mid')

        # Plot target state
        ax.plot(t, tgt[:, i], 'b-', linewidth=2, label='Target State', alpha=0.8)

        # Plot predictions if available
        if pred is not None:
            ax.plot(t, pred[:, i], 'orange', linestyle='--', linewidth=2,
                   label='Prediction', alpha=0.8)

        ax.set_ylabel(f'Flip-Flop {i+1}', fontsize=11, fontweight='bold')
        ax.set_ylim([-0.1, 1.1])
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(loc='upper right', fontsize=9)

    axes[-1].set_xlabel('Time Step', fontsize=12)
    plt.suptitle('3-Bit Flip-Flop Task', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    # Test data generation
    print("Testing 3-Bit Flip-Flop Data Generation...")

    # Generate a single trial
    inputs, targets = generate_flipflop_trial(n_bits=3, seq_length=100, seed=42)
    print(f"\nSingle trial:")
    print(f"  Inputs shape: {inputs.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Input range: [{inputs.min()}, {inputs.max()}]")
    print(f"  Target range: [{targets.min()}, {targets.max()}]")

    # Generate dataset
    train_loader, val_loader, test_loader, info = create_flipflop_dataloaders(
        train_trials=800,
        val_trials=100,
        test_trials=100,
        batch_size=32,
        seed=42
    )

    print(f"\nDataset created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Info: {info}")

    # Test batch
    batch_x, batch_y = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Inputs: {batch_x.shape}")
    print(f"  Targets: {batch_y.shape}")

    print("\n✓ All tests passed!")
