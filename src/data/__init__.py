"""
Data Generation Utilities
=========================

Functions for generating training data from dynamical systems,
particularly the Lorenz-63 system.
"""

import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, Any


# =============================================================================
# Lorenz-63 System
# =============================================================================

def lorenz_system(t: float, state: np.ndarray, sigma: float = 10.0, 
                  rho: float = 28.0, beta: float = 8/3) -> np.ndarray:
    """
    Lorenz-63 system of ODEs.
    
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz
    
    Parameters
    ----------
    t : float
        Time (not used, but required by solve_ivp)
    state : np.ndarray
        Current state [x, y, z]
    sigma : float
        Prandtl number (default: 10)
    rho : float
        Rayleigh number (default: 28)
    beta : float
        Geometric factor (default: 8/3)
        
    Returns
    -------
    np.ndarray
        Derivatives [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ])


def generate_lorenz_trajectory(
    t_span: Tuple[float, float] = (0, 100),
    dt: float = 0.01,
    initial_state: Optional[np.ndarray] = None,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8/3,
    transient: float = 10.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a trajectory from the Lorenz-63 system.
    
    Parameters
    ----------
    t_span : tuple
        (t_start, t_end) for integration
    dt : float
        Time step for output (integration uses adaptive stepping)
    initial_state : np.ndarray, optional
        Initial [x, y, z]. If None, random near attractor.
    sigma, rho, beta : float
        Lorenz system parameters
    transient : float
        Time to discard as transient (to reach attractor)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    t : np.ndarray
        Time points
    trajectory : np.ndarray
        States at each time point, shape (n_times, 3)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initial condition: if not provided, start near the attractor
    if initial_state is None:
        # Start near one of the fixed points and let it evolve
        initial_state = np.array([1.0, 1.0, 1.0]) + np.random.randn(3) * 0.1
    
    # Include transient in integration
    t_start, t_end = t_span
    t_eval = np.arange(t_start, t_end + transient, dt)
    
    # Integrate
    sol = solve_ivp(
        lorenz_system,
        (t_start, t_end + transient),
        initial_state,
        args=(sigma, rho, beta),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-10,
        atol=1e-12
    )
    
    # Remove transient
    transient_steps = int(transient / dt)
    t = sol.t[transient_steps:] - transient
    trajectory = sol.y[:, transient_steps:].T  # Shape: (n_times, 3)
    
    return t, trajectory


def generate_multiple_trajectories(
    n_trajectories: int = 10,
    t_span: Tuple[float, float] = (0, 50),
    dt: float = 0.01,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate multiple trajectories with different initial conditions.
    
    Returns
    -------
    t : np.ndarray
        Time points (shared across trajectories)
    trajectories : np.ndarray
        Shape (n_trajectories, n_times, 3)
    """
    trajectories = []
    for i in range(n_trajectories):
        t, traj = generate_lorenz_trajectory(t_span=t_span, dt=dt, seed=i, **kwargs)
        trajectories.append(traj)
    
    return t, np.stack(trajectories)


# =============================================================================
# PyTorch Dataset
# =============================================================================

class LorenzDataset(Dataset):
    """
    PyTorch Dataset for Lorenz trajectory prediction.
    
    Given a sequence of states, predict the next state(s).
    """
    
    def __init__(
        self,
        trajectory: np.ndarray,
        seq_length: int = 50,
        pred_length: int = 1,
        stride: int = 1,
        normalize: bool = True
    ):
        """
        Parameters
        ----------
        trajectory : np.ndarray
            Shape (n_times, 3) or (n_trajectories, n_times, 3)
        seq_length : int
            Input sequence length
        pred_length : int
            Number of steps to predict
        stride : int
            Stride between samples
        normalize : bool
            Whether to normalize to zero mean and unit variance
        """
        # Handle single or multiple trajectories
        if trajectory.ndim == 2:
            trajectory = trajectory[np.newaxis, ...]  # Add batch dimension
        
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.stride = stride
        
        # Compute normalization statistics
        if normalize:
            self.mean = trajectory.mean(axis=(0, 1))
            self.std = trajectory.std(axis=(0, 1))
            trajectory = (trajectory - self.mean) / self.std
        else:
            self.mean = np.zeros(3)
            self.std = np.ones(3)
        
        # Create samples
        self.inputs = []
        self.targets = []
        
        n_traj, n_times, n_dim = trajectory.shape
        
        for traj in trajectory:
            for i in range(0, n_times - seq_length - pred_length + 1, stride):
                self.inputs.append(traj[i:i+seq_length])
                self.targets.append(traj[i+seq_length:i+seq_length+pred_length])
        
        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        
        # Squeeze target if pred_length == 1
        if pred_length == 1:
            self.targets = self.targets.squeeze(1)
    
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.inputs[idx]),
            torch.from_numpy(self.targets[idx])
        )
    
    def get_normalization_params(self) -> Dict[str, np.ndarray]:
        """Return normalization parameters for denormalization."""
        return {'mean': self.mean, 'std': self.std}


class LorenzContinuousDataset(Dataset):
    """
    Dataset for continuous-time models.
    
    Returns (initial_state, time_points, full_trajectory) for ODE-based models.
    """
    
    def __init__(
        self,
        trajectory: np.ndarray,
        t: np.ndarray,
        segment_length: int = 100,
        stride: int = 50,
        normalize: bool = True
    ):
        """
        Parameters
        ----------
        trajectory : np.ndarray
            Shape (n_times, 3)
        t : np.ndarray
            Time points
        segment_length : int
            Number of time steps per segment
        stride : int
            Stride between segments
        normalize : bool
            Whether to normalize data
        """
        self.dt = t[1] - t[0]
        
        # Normalize
        if normalize:
            self.mean = trajectory.mean(axis=0)
            self.std = trajectory.std(axis=0)
            trajectory = (trajectory - self.mean) / self.std
        else:
            self.mean = np.zeros(3)
            self.std = np.ones(3)
        
        # Create segments
        self.segments = []
        self.time_segments = []
        
        n_times = len(trajectory)
        for i in range(0, n_times - segment_length + 1, stride):
            self.segments.append(trajectory[i:i+segment_length])
            self.time_segments.append(t[i:i+segment_length] - t[i])  # Relative time
        
        self.segments = np.array(self.segments, dtype=np.float32)
        self.time_segments = np.array(self.time_segments, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.segments)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        initial_state : torch.Tensor
            Shape (3,)
        time_points : torch.Tensor
            Shape (segment_length,)
        trajectory : torch.Tensor
            Shape (segment_length, 3)
        """
        segment = self.segments[idx]
        times = self.time_segments[idx]
        
        return (
            torch.from_numpy(segment[0]),      # Initial state
            torch.from_numpy(times),            # Time points
            torch.from_numpy(segment)           # Full trajectory
        )


# =============================================================================
# Data Loading Utilities
# =============================================================================

def create_lorenz_dataloaders(
    train_length: float = 80.0,
    val_length: float = 10.0,
    test_length: float = 10.0,
    dt: float = 0.01,
    seq_length: int = 50,
    batch_size: int = 64,
    seed: int = 42,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create train/val/test DataLoaders for Lorenz prediction task.
    
    Parameters
    ----------
    train_length, val_length, test_length : float
        Duration of each split in time units
    dt : float
        Time step
    seq_length : int
        Input sequence length
    batch_size : int
        Batch size for DataLoaders
    seed : int
        Random seed
        
    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    info : dict
        Contains normalization params and other metadata
    """
    total_length = train_length + val_length + test_length
    
    # Generate one long trajectory
    t, trajectory = generate_lorenz_trajectory(
        t_span=(0, total_length),
        dt=dt,
        seed=seed
    )
    
    # Split
    train_steps = int(train_length / dt)
    val_steps = int(val_length / dt)
    
    train_traj = trajectory[:train_steps]
    val_traj = trajectory[train_steps:train_steps+val_steps]
    test_traj = trajectory[train_steps+val_steps:]
    
    # Create datasets (normalize based on training data)
    train_dataset = LorenzDataset(train_traj, seq_length=seq_length, normalize=True)
    norm_params = train_dataset.get_normalization_params()
    
    # Apply same normalization to val/test
    val_traj_norm = (val_traj - norm_params['mean']) / norm_params['std']
    test_traj_norm = (test_traj - norm_params['mean']) / norm_params['std']
    
    val_dataset = LorenzDataset(val_traj_norm, seq_length=seq_length, normalize=False)
    val_dataset.mean, val_dataset.std = norm_params['mean'], norm_params['std']
    
    test_dataset = LorenzDataset(test_traj_norm, seq_length=seq_length, normalize=False)
    test_dataset.mean, test_dataset.std = norm_params['mean'], norm_params['std']
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    info = {
        'normalization': norm_params,
        'dt': dt,
        'seq_length': seq_length,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
    }
    
    return train_loader, val_loader, test_loader, info


# =============================================================================
# Visualization Helpers
# =============================================================================

def plot_lorenz_attractor(trajectory: np.ndarray, ax=None, **kwargs):
    """
    Plot Lorenz attractor in 3D.
    
    Parameters
    ----------
    trajectory : np.ndarray
        Shape (n_times, 3)
    ax : matplotlib 3D axis, optional
    **kwargs : passed to plot
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    default_kwargs = {'lw': 0.5, 'alpha': 0.8}
    default_kwargs.update(kwargs)
    
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], **default_kwargs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Lorenz Attractor')
    
    return ax


if __name__ == "__main__":
    # Quick test
    print("Generating Lorenz trajectory...")
    t, traj = generate_lorenz_trajectory(t_span=(0, 100), dt=0.01)
    print(f"Trajectory shape: {traj.shape}")
    print(f"Time range: {t[0]:.2f} to {t[-1]:.2f}")
    print(f"State ranges: x=[{traj[:,0].min():.2f}, {traj[:,0].max():.2f}], "
          f"y=[{traj[:,1].min():.2f}, {traj[:,1].max():.2f}], "
          f"z=[{traj[:,2].min():.2f}, {traj[:,2].max():.2f}]")
    
    # Test dataset
    dataset = LorenzDataset(traj, seq_length=50)
    print(f"\nDataset size: {len(dataset)} samples")
    x, y = dataset[0]
    print(f"Input shape: {x.shape}, Target shape: {y.shape}")
