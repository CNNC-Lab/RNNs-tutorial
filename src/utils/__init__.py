"""
Utilities
=========

Helper functions for training, visualization, and evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, List, Dict, Any
from tqdm.auto import tqdm


# =============================================================================
# Training Utilities
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = 'cpu',
    clip_grad: Optional[float] = 1.0
) -> float:
    """
    Train model for one epoch.
    
    Returns
    -------
    avg_loss : float
        Average loss over epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str = 'cpu'
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model on dataset.
    
    Returns
    -------
    avg_loss : float
    predictions : np.ndarray
    targets : np.ndarray
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            n_batches += 1
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    return (
        total_loss / n_batches,
        np.concatenate(all_preds),
        np.concatenate(all_targets)
    )


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 20,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Full training loop with early stopping.
    
    Returns
    -------
    history : dict
        Training history with 'train_loss' and 'val_loss'
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//2
    )
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    
    iterator = tqdm(range(n_epochs), desc='Training') if verbose else range(n_epochs)
    
    for epoch in iterator:
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            tqdm.write(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        if epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


# =============================================================================
# Visualization Utilities
# =============================================================================

def plot_training_history(
    history: Dict[str, List[float]],
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot training and validation loss curves."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.semilogy(history['train_loss'], label='Train', alpha=0.8)
    ax.semilogy(history['val_loss'], label='Validation', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_lorenz_3d(
    trajectory: np.ndarray,
    ax: Optional[Axes3D] = None,
    color: Optional[np.ndarray] = None,
    cmap: str = 'viridis',
    alpha: float = 0.8,
    lw: float = 0.5,
    label: Optional[str] = None,
    **kwargs
) -> Axes3D:
    """
    Plot 3D Lorenz attractor trajectory.
    
    Parameters
    ----------
    trajectory : np.ndarray
        Shape (n_times, 3)
    ax : Axes3D, optional
    color : np.ndarray, optional
        Values for colormap (e.g., time)
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    
    if color is not None:
        # Plot with color gradient
        points = trajectory.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        norm = plt.Normalize(color.min(), color.max())
        lc = Line3DCollection(segments, cmap=cmap, norm=norm, alpha=alpha)
        lc.set_array(color)
        ax.add_collection(lc)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_zlim(z.min(), z.max())
    else:
        ax.plot(x, y, z, lw=lw, alpha=alpha, label=label, **kwargs)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    return ax


def plot_trajectory_comparison(
    true_traj: np.ndarray,
    pred_traj: np.ndarray,
    t: Optional[np.ndarray] = None,
    dim_names: List[str] = ['x', 'y', 'z'],
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Compare true and predicted trajectories.
    """
    n_dims = true_traj.shape[1]
    if t is None:
        t = np.arange(len(true_traj))
    
    fig, axes = plt.subplots(n_dims, 1, figsize=figsize, sharex=True)
    if n_dims == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.plot(t, true_traj[:, i], 'b-', lw=1, label='True', alpha=0.8)
        ax.plot(t, pred_traj[:, i], 'r--', lw=1, label='Predicted', alpha=0.8)
        ax.set_ylabel(dim_names[i] if i < len(dim_names) else f'Dim {i}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time')
    fig.suptitle('Trajectory Comparison', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_ei_dynamics(
    r_e: np.ndarray,
    r_i: np.ndarray,
    t: Optional[np.ndarray] = None,
    n_neurons: int = 10,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Visualize E/I population dynamics.
    
    Parameters
    ----------
    r_e, r_i : np.ndarray
        Shape (n_times, n_neurons) or (batch, n_times, n_neurons)
    """
    if r_e.ndim == 3:
        r_e = r_e[0]  # Take first batch
        r_i = r_i[0]
    
    if t is None:
        t = np.arange(len(r_e))
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # E population activity (subset of neurons)
    idx_e = np.linspace(0, r_e.shape[1]-1, min(n_neurons, r_e.shape[1])).astype(int)
    for i, idx in enumerate(idx_e):
        axes[0, 0].plot(t, r_e[:, idx], alpha=0.7, lw=0.8)
    axes[0, 0].set_title('Excitatory Neurons')
    axes[0, 0].set_ylabel('Rate')
    
    # I population activity
    idx_i = np.linspace(0, r_i.shape[1]-1, min(n_neurons, r_i.shape[1])).astype(int)
    for i, idx in enumerate(idx_i):
        axes[0, 1].plot(t, r_i[:, idx], alpha=0.7, lw=0.8)
    axes[0, 1].set_title('Inhibitory Neurons')
    
    # Population averages
    axes[1, 0].plot(t, r_e.mean(axis=1), 'b-', label='E mean', lw=2)
    axes[1, 0].fill_between(t, 
                             r_e.mean(axis=1) - r_e.std(axis=1),
                             r_e.mean(axis=1) + r_e.std(axis=1),
                             alpha=0.3)
    axes[1, 0].plot(t, r_i.mean(axis=1), 'r-', label='I mean', lw=2)
    axes[1, 0].fill_between(t,
                             r_i.mean(axis=1) - r_i.std(axis=1),
                             r_i.mean(axis=1) + r_i.std(axis=1),
                             alpha=0.3, color='red')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Rate')
    axes[1, 0].set_title('Population Averages')
    
    # E-I balance
    balance = r_e.mean(axis=1) - r_i.mean(axis=1)
    axes[1, 1].plot(t, balance, 'g-', lw=1)
    axes[1, 1].axhline(0, color='k', ls='--', alpha=0.5)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('E - I')
    axes[1, 1].set_title('E-I Balance')
    
    plt.tight_layout()
    return fig


def plot_spike_raster(
    spikes: np.ndarray,
    t: Optional[np.ndarray] = None,
    neuron_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    max_neurons: int = 100
) -> plt.Figure:
    """
    Plot spike raster.
    
    Parameters
    ----------
    spikes : np.ndarray
        Shape (n_times, n_neurons) or (batch, n_times, n_neurons)
    """
    if spikes.ndim == 3:
        spikes = spikes[0]
    
    if t is None:
        t = np.arange(spikes.shape[0])
    
    n_neurons = min(spikes.shape[1], max_neurons)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Find spike times for each neuron
    for i in range(n_neurons):
        spike_times = t[spikes[:, i] > 0.5]
        ax.scatter(spike_times, np.ones_like(spike_times) * i, 
                   marker='|', s=2, c='black', alpha=0.8)
    
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-0.5, n_neurons - 0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Neuron')
    ax.set_title('Spike Raster')
    
    return fig


def plot_weight_matrix(
    W: np.ndarray,
    ax: Optional[plt.Axes] = None,
    cmap: str = 'RdBu_r',
    vmax: Optional[float] = None,
    title: str = 'Weight Matrix'
) -> plt.Axes:
    """Plot weight matrix with E/I structure."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if vmax is None:
        vmax = np.abs(W).max()
    
    im = ax.imshow(W, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Pre-synaptic')
    ax.set_ylabel('Post-synaptic')
    ax.set_title(title)
    
    return ax


def plot_fixed_points_2d(
    fixed_points: np.ndarray,
    stability: List[dict],
    trajectory: Optional[np.ndarray] = None,
    dims: Tuple[int, int] = (0, 1),
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot fixed points with stability information in 2D projection.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    d1, d2 = dims
    
    # Plot trajectory if provided
    if trajectory is not None:
        ax.plot(trajectory[:, d1], trajectory[:, d2], 'b-', alpha=0.3, lw=0.5)
    
    # Plot fixed points
    for i, (fp, stab) in enumerate(zip(fixed_points, stability)):
        color = 'green' if stab['stable_continuous'] else 'red'
        marker = 'o' if stab['stable_continuous'] else 'x'
        ax.scatter(fp[d1], fp[d2], c=color, marker=marker, s=100, 
                   label=f"FP{i}: {stab['classification']}", zorder=5)
    
    ax.set_xlabel(f'Dimension {d1}')
    ax.set_ylabel(f'Dimension {d2}')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Fixed Points')
    
    return ax


# =============================================================================
# Metrics
# =============================================================================

def compute_prediction_metrics(
    true: np.ndarray,
    predicted: np.ndarray
) -> Dict[str, float]:
    """
    Compute various prediction quality metrics.
    """
    mse = np.mean((true - predicted)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true - predicted))
    
    # Normalized RMSE
    nrmse = rmse / (np.max(true) - np.min(true) + 1e-10)
    
    # R-squared
    ss_res = np.sum((true - predicted)**2)
    ss_tot = np.sum((true - np.mean(true))**2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    
    # Correlation
    corr = np.corrcoef(true.flatten(), predicted.flatten())[0, 1]
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'nrmse': nrmse,
        'r2': r2,
        'correlation': corr
    }


def compute_valid_time(
    true: np.ndarray,
    predicted: np.ndarray,
    threshold: float = 0.4,
    dt: float = 0.01
) -> float:
    """
    Compute valid prediction time (time before error exceeds threshold).
    
    This is a common metric for chaotic systems.
    
    Parameters
    ----------
    true, predicted : np.ndarray
        Trajectories, shape (n_times, n_dims)
    threshold : float
        Error threshold (as fraction of attractor size)
    dt : float
        Time step
        
    Returns
    -------
    valid_time : float
        Time in same units as dt before prediction fails
    """
    # Normalize by attractor size
    attractor_size = np.std(true)
    
    # Compute error over time
    error = np.sqrt(np.mean((true - predicted)**2, axis=1)) / attractor_size
    
    # Find first time error exceeds threshold
    exceed_idx = np.where(error > threshold)[0]
    
    if len(exceed_idx) > 0:
        valid_time = exceed_idx[0] * dt
    else:
        valid_time = len(true) * dt
    
    return valid_time


if __name__ == "__main__":
    print("Utilities module loaded successfully!")
    
    # Quick visualization test
    t = np.linspace(0, 10*np.pi, 1000)
    x = np.column_stack([np.sin(t), np.cos(t), t/10])
    
    fig = plot_trajectory_comparison(x, x + np.random.randn(*x.shape)*0.1, t)
    plt.savefig('/tmp/test_comparison.png', dpi=100, bbox_inches='tight')
    print("Test figure saved to /tmp/test_comparison.png")
