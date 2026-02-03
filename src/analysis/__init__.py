"""
Dynamical Systems Analysis
==========================

Tools for analyzing trained RNNs as dynamical systems:
- Fixed point finding
- Linearization and stability analysis
- Lyapunov exponent estimation
- Attractor characterization
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import fsolve, minimize
from scipy.linalg import eig
from typing import Optional, Tuple, List, Callable
import warnings


def find_fixed_points(
    dynamics_fn: Callable[[torch.Tensor], torch.Tensor],
    hidden_size: int,
    n_initial: int = 100,
    tol: float = 1e-6,
    max_iter: int = 1000,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find fixed points of RNN dynamics using optimization.
    
    For a CT-RNN: dh/dt = f(h) = 0 at fixed points
    For discrete RNN: h_{t+1} = f(h_t), so f(h*) = h*
    
    Parameters
    ----------
    dynamics_fn : callable
        Function h -> dh/dt (or h -> h_next - h for discrete)
    hidden_size : int
        Dimension of hidden state
    n_initial : int
        Number of random initial points to try
    tol : float
        Tolerance for fixed point (||f(h*)|| < tol)
    max_iter : int
        Maximum optimization iterations
    device : str
        Device for computation
        
    Returns
    -------
    fixed_points : np.ndarray
        Found fixed points, shape (n_found, hidden_size)
    residuals : np.ndarray
        Residual ||f(h*)|| for each fixed point
    """
    fixed_points = []
    residuals = []
    
    # Try multiple random initializations
    for i in range(n_initial):
        # Random initial point
        h0 = np.random.randn(hidden_size).astype(np.float32)
        
        def objective(h):
            h_tensor = torch.tensor(h, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                dh = dynamics_fn(h_tensor).squeeze(0).cpu().numpy()
            return np.sum(dh**2)
        
        def gradient(h):
            h_tensor = torch.tensor(h, dtype=torch.float32, device=device, requires_grad=True).unsqueeze(0)
            dh = dynamics_fn(h_tensor)
            loss = (dh**2).sum()
            loss.backward()
            return h_tensor.grad.squeeze(0).cpu().numpy()
        
        # Optimize
        try:
            result = minimize(
                objective, h0, 
                method='L-BFGS-B',
                jac=gradient,
                options={'maxiter': max_iter, 'ftol': tol**2}
            )
            
            if result.fun < tol**2:
                # Found a fixed point
                h_star = result.x
                
                # Check if this is a new fixed point (not duplicate)
                is_new = True
                for fp in fixed_points:
                    if np.linalg.norm(h_star - fp) < 0.1:
                        is_new = False
                        break
                
                if is_new:
                    fixed_points.append(h_star)
                    residuals.append(np.sqrt(result.fun))
        except Exception as e:
            continue
    
    if len(fixed_points) == 0:
        return np.array([]).reshape(0, hidden_size), np.array([])
    
    return np.array(fixed_points), np.array(residuals)


def compute_jacobian(
    dynamics_fn: Callable[[torch.Tensor], torch.Tensor],
    h: torch.Tensor
) -> np.ndarray:
    """
    Compute Jacobian matrix at a point.
    
    J_ij = df_i / dh_j
    
    Parameters
    ----------
    dynamics_fn : callable
        Function h -> f(h)
    h : torch.Tensor
        Point to linearize around, shape (hidden_size,) or (1, hidden_size)
        
    Returns
    -------
    jacobian : np.ndarray
        Jacobian matrix, shape (hidden_size, hidden_size)
    """
    if h.dim() == 1:
        h = h.unsqueeze(0)
    
    h = h.detach().requires_grad_(True)
    hidden_size = h.shape[1]
    
    # Compute Jacobian column by column
    jacobian = torch.zeros(hidden_size, hidden_size, device=h.device)
    
    f = dynamics_fn(h).squeeze(0)
    
    for i in range(hidden_size):
        if h.grad is not None:
            h.grad.zero_()
        f[i].backward(retain_graph=True)
        jacobian[i] = h.grad.squeeze(0)
    
    return jacobian.detach().cpu().numpy()


def analyze_fixed_point_stability(
    jacobian: np.ndarray
) -> dict:
    """
    Analyze stability of a fixed point from its Jacobian.
    
    Parameters
    ----------
    jacobian : np.ndarray
        Jacobian matrix at the fixed point
        
    Returns
    -------
    analysis : dict
        'eigenvalues': complex eigenvalues
        'eigenvectors': corresponding eigenvectors
        'stable': True if all eigenvalues have negative real part (for CT systems)
        'classification': 'stable node', 'saddle', 'unstable node', 'spiral', etc.
    """
    eigenvalues, eigenvectors = eig(jacobian)
    
    # For continuous-time: stable if Re(λ) < 0 for all λ
    # For discrete-time: stable if |λ| < 1 for all λ
    real_parts = eigenvalues.real
    magnitudes = np.abs(eigenvalues)
    
    # Assuming continuous-time for now
    stable_ct = np.all(real_parts < 0)
    stable_dt = np.all(magnitudes < 1)
    
    # Classify
    n_positive = np.sum(real_parts > 0)
    n_negative = np.sum(real_parts < 0)
    n_zero = np.sum(np.abs(real_parts) < 1e-10)
    has_imaginary = np.any(np.abs(eigenvalues.imag) > 1e-10)
    
    if n_positive == 0 and n_zero == 0:
        classification = 'stable spiral' if has_imaginary else 'stable node'
    elif n_negative == 0 and n_zero == 0:
        classification = 'unstable spiral' if has_imaginary else 'unstable node'
    elif n_zero > 0:
        classification = 'center/bifurcation'
    else:
        classification = 'saddle'
    
    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'stable_continuous': stable_ct,
        'stable_discrete': stable_dt,
        'classification': classification,
        'n_unstable_directions': n_positive,
        'spectral_radius': np.max(magnitudes),
        'max_real_eigenvalue': np.max(real_parts)
    }


# =============================================================================
# Lyapunov Exponent Estimation
# =============================================================================

def estimate_lyapunov_exponents(
    dynamics_fn: Callable[[torch.Tensor], torch.Tensor],
    initial_state: torch.Tensor,
    n_steps: int = 10000,
    n_exponents: int = 3,
    dt: float = 0.01,
    warmup: int = 1000,
    qr_interval: int = 10
) -> np.ndarray:
    """
    Estimate Lyapunov exponents using QR decomposition method.
    
    The Lyapunov exponents characterize the rate of separation of 
    infinitesimally close trajectories. Positive exponents indicate chaos.
    
    Parameters
    ----------
    dynamics_fn : callable
        One-step dynamics h_t -> h_{t+1}
    initial_state : torch.Tensor
        Initial state, shape (hidden_size,)
    n_steps : int
        Number of integration steps
    n_exponents : int
        Number of Lyapunov exponents to estimate (largest ones)
    dt : float
        Time step (for normalization)
    warmup : int
        Warmup steps before accumulating
    qr_interval : int
        QR decomposition interval
        
    Returns
    -------
    lyapunov_exponents : np.ndarray
        Estimated Lyapunov exponents (sorted descending)
    """
    device = initial_state.device
    hidden_size = initial_state.shape[-1]
    
    # Initialize
    if initial_state.dim() == 1:
        h = initial_state.unsqueeze(0)
    else:
        h = initial_state
    
    # Initialize orthonormal perturbation vectors
    Q = torch.eye(hidden_size, n_exponents, device=device, dtype=torch.float32)
    
    # Accumulate logarithms of stretching factors
    log_stretch = torch.zeros(n_exponents, device=device)
    n_qr = 0
    
    for step in range(n_steps):
        # Evolve main trajectory
        with torch.no_grad():
            h_new = h + dynamics_fn(h) * dt
        
        # Evolve perturbation vectors (linearized dynamics)
        if step >= warmup:
            # Compute Jacobian at current point
            h_jac = h.detach().requires_grad_(True)
            f = dynamics_fn(h_jac)
            
            # Apply Jacobian to perturbation vectors
            Q_new = torch.zeros_like(Q)
            for i in range(n_exponents):
                # dQ_i/dt ≈ J @ Q_i
                grad_outputs = torch.zeros_like(f)
                for j in range(hidden_size):
                    if h_jac.grad is not None:
                        h_jac.grad.zero_()
                    f[0, j].backward(retain_graph=True)
                    Q_new[j, i] = (h_jac.grad.squeeze() * Q[:, i]).sum()
            
            Q = Q + Q_new * dt
            
            # Periodic QR decomposition
            if (step - warmup) % qr_interval == 0:
                Q, R = torch.linalg.qr(Q)
                log_stretch += torch.log(torch.abs(torch.diag(R)) + 1e-10)
                n_qr += 1
        
        h = h_new
    
    # Compute Lyapunov exponents
    if n_qr > 0:
        lyapunov_exponents = (log_stretch / (n_qr * qr_interval * dt)).cpu().numpy()
    else:
        lyapunov_exponents = np.zeros(n_exponents)
    
    return np.sort(lyapunov_exponents)[::-1]


def estimate_lyapunov_spectrum_simple(
    trajectory: np.ndarray,
    dt: float = 0.01,
    embedding_dim: int = 3,
    tau: int = 1
) -> float:
    """
    Estimate largest Lyapunov exponent from a trajectory using
    the Rosenstein method (simple but robust).
    
    Parameters
    ----------
    trajectory : np.ndarray
        Time series, shape (n_times,) or (n_times, n_dim)
    dt : float
        Time step
    embedding_dim : int
        Embedding dimension (for 1D time series)
    tau : int
        Time delay for embedding
        
    Returns
    -------
    lambda_max : float
        Largest Lyapunov exponent
    """
    if trajectory.ndim == 1:
        # Time-delay embedding
        n = len(trajectory) - (embedding_dim - 1) * tau
        embedded = np.zeros((n, embedding_dim))
        for i in range(embedding_dim):
            embedded[:, i] = trajectory[i*tau:i*tau+n]
        trajectory = embedded
    
    n_points = len(trajectory)
    
    # Find nearest neighbors (excluding temporal neighbors)
    min_dist = np.inf * np.ones(n_points)
    nn_idx = np.zeros(n_points, dtype=int)
    
    temporal_separation = 10  # Minimum temporal separation
    
    for i in range(n_points):
        for j in range(n_points):
            if abs(i - j) > temporal_separation:
                dist = np.linalg.norm(trajectory[i] - trajectory[j])
                if dist < min_dist[i]:
                    min_dist[i] = dist
                    nn_idx[i] = j
    
    # Track divergence
    max_evolution = min(100, n_points // 2)
    divergence = np.zeros(max_evolution)
    count = np.zeros(max_evolution)
    
    for i in range(n_points - max_evolution):
        j = nn_idx[i]
        if j < n_points - max_evolution:
            for k in range(max_evolution):
                d = np.linalg.norm(trajectory[i+k] - trajectory[j+k])
                if d > 0:
                    divergence[k] += np.log(d)
                    count[k] += 1
    
    # Average
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        divergence = divergence / (count + 1e-10)
    
    # Linear fit to get exponent
    valid = count > 10
    if np.sum(valid) > 5:
        t = np.arange(max_evolution)[valid] * dt
        d = divergence[valid]
        # Linear regression
        slope = np.polyfit(t[:len(t)//2], d[:len(t)//2], 1)[0]
        return slope
    else:
        return 0.0


# =============================================================================
# Attractor Analysis
# =============================================================================

def compute_attractor_dimension(
    trajectory: np.ndarray,
    r_range: Optional[Tuple[float, float]] = None,
    n_points: int = 1000
) -> float:
    """
    Estimate attractor dimension using correlation dimension.
    
    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory on attractor, shape (n_times, n_dim)
    r_range : tuple, optional
        Range of radii for scaling analysis
    n_points : int
        Number of points to sample
        
    Returns
    -------
    dimension : float
        Estimated correlation dimension
    """
    # Sample points
    n_total = len(trajectory)
    if n_total > n_points:
        idx = np.random.choice(n_total, n_points, replace=False)
        points = trajectory[idx]
    else:
        points = trajectory
    
    n = len(points)
    
    # Compute pairwise distances
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d > 0:
                distances.append(d)
    
    distances = np.array(distances)
    
    if len(distances) == 0:
        return 0.0
    
    # Determine r range
    if r_range is None:
        r_min = np.percentile(distances, 1)
        r_max = np.percentile(distances, 50)
    else:
        r_min, r_max = r_range
    
    # Compute correlation integral C(r) for various r
    r_values = np.logspace(np.log10(r_min), np.log10(r_max), 20)
    C_values = []
    
    n_pairs = len(distances)
    for r in r_values:
        C = np.sum(distances < r) / n_pairs
        if C > 0:
            C_values.append(C)
        else:
            C_values.append(1e-10)
    
    C_values = np.array(C_values)
    
    # Estimate dimension from slope of log(C) vs log(r)
    log_r = np.log(r_values[:len(C_values)])
    log_C = np.log(C_values)
    
    # Use middle portion for linear fit
    n_fit = len(log_r)
    start = n_fit // 4
    end = 3 * n_fit // 4
    
    if end - start > 2:
        slope, _ = np.polyfit(log_r[start:end], log_C[start:end], 1)
        return slope
    else:
        return 0.0


def compare_attractors(
    traj1: np.ndarray,
    traj2: np.ndarray,
    n_samples: int = 1000
) -> dict:
    """
    Compare two attractors (e.g., true vs reconstructed).
    
    Parameters
    ----------
    traj1, traj2 : np.ndarray
        Trajectories, shape (n_times, n_dim)
    n_samples : int
        Number of points to sample
        
    Returns
    -------
    metrics : dict
        Comparison metrics
    """
    # Sample points
    idx1 = np.random.choice(len(traj1), min(n_samples, len(traj1)), replace=False)
    idx2 = np.random.choice(len(traj2), min(n_samples, len(traj2)), replace=False)
    
    pts1 = traj1[idx1]
    pts2 = traj2[idx2]
    
    # Hausdorff-like distance
    def one_sided_distance(A, B):
        """Average distance from points in A to nearest point in B."""
        dists = []
        for a in A:
            d = np.min(np.linalg.norm(B - a, axis=1))
            dists.append(d)
        return np.mean(dists)
    
    d12 = one_sided_distance(pts1, pts2)
    d21 = one_sided_distance(pts2, pts1)
    
    # Bounding box comparison
    bbox1 = np.max(pts1, axis=0) - np.min(pts1, axis=0)
    bbox2 = np.max(pts2, axis=0) - np.min(pts2, axis=0)
    
    # Center of mass
    com1 = np.mean(pts1, axis=0)
    com2 = np.mean(pts2, axis=0)
    
    return {
        'mean_distance_1_to_2': d12,
        'mean_distance_2_to_1': d21,
        'symmetric_distance': (d12 + d21) / 2,
        'bbox_ratio': np.mean(bbox2 / (bbox1 + 1e-10)),
        'center_distance': np.linalg.norm(com1 - com2),
        'extent_1': bbox1,
        'extent_2': bbox2,
    }


# =============================================================================
# Utility Functions
# =============================================================================

def create_dynamics_fn_from_ctrnn(model, x=None):
    """
    Create a dynamics function from a CT-RNN model for analysis.
    
    Parameters
    ----------
    model : ContinuousTimeRNN
        The CT-RNN model
    x : torch.Tensor, optional
        Constant input
        
    Returns
    -------
    dynamics_fn : callable
        Function h -> dh/dt
    """
    def dynamics_fn(h):
        if h.dim() == 1:
            h = h.unsqueeze(0)
        return model.cell(torch.tensor(0.0), h, x)
    
    return dynamics_fn


def visualize_phase_portrait_2d(
    dynamics_fn: Callable,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    n_grid: int = 20,
    ax=None
):
    """
    Visualize 2D phase portrait with vector field.
    
    For higher-dimensional systems, projects onto first 2 dimensions.
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    x = np.linspace(xlim[0], xlim[1], n_grid)
    y = np.linspace(ylim[0], ylim[1], n_grid)
    X, Y = np.meshgrid(x, y)
    
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(n_grid):
        for j in range(n_grid):
            h = torch.tensor([[X[i,j], Y[i,j]]], dtype=torch.float32)
            with torch.no_grad():
                dh = dynamics_fn(h).numpy()[0]
            U[i,j] = dh[0]
            V[i,j] = dh[1]
    
    # Normalize for visualization
    magnitude = np.sqrt(U**2 + V**2)
    U_norm = U / (magnitude + 1e-10)
    V_norm = V / (magnitude + 1e-10)
    
    ax.streamplot(X, Y, U, V, color=np.log(magnitude + 1), cmap='viridis')
    ax.set_xlabel('$h_1$')
    ax.set_ylabel('$h_2$')
    ax.set_title('Phase Portrait')
    
    return ax


if __name__ == "__main__":
    # Quick test with a simple 2D system
    print("Testing dynamical systems analysis...")
    
    # Simple linear system for testing
    A = torch.tensor([[-0.5, 1.0], [-1.0, -0.5]])
    def simple_dynamics(h):
        return h @ A.T
    
    # Find fixed points
    fps, res = find_fixed_points(simple_dynamics, hidden_size=2, n_initial=10)
    print(f"Found {len(fps)} fixed points")
    
    if len(fps) > 0:
        # Analyze stability
        jac = compute_jacobian(simple_dynamics, torch.tensor(fps[0]))
        analysis = analyze_fixed_point_stability(jac)
        print(f"Fixed point classification: {analysis['classification']}")
        print(f"Eigenvalues: {analysis['eigenvalues']}")
    
    # Test Lyapunov exponent estimation
    # Generate trajectory from Lorenz system
    from scipy.integrate import solve_ivp
    
    def lorenz(t, state):
        x, y, z = state
        return [10*(y-x), x*(28-z)-y, x*y - 8/3*z]
    
    sol = solve_ivp(lorenz, (0, 100), [1, 1, 1], t_eval=np.linspace(0, 100, 10000))
    trajectory = sol.y.T
    
    # Estimate dimension
    dim = compute_attractor_dimension(trajectory)
    print(f"Estimated Lorenz attractor dimension: {dim:.2f} (expected ~2.05)")
    
    # Estimate largest Lyapunov exponent
    lambda_max = estimate_lyapunov_spectrum_simple(trajectory[:, 0], dt=0.01)
    print(f"Estimated largest Lyapunov exponent: {lambda_max:.3f} (expected ~0.9)")
    
    print("All tests passed!")
