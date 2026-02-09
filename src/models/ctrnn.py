"""
Continuous-Time Recurrent Neural Network (CT-RNN)
=================================================

Implements RNNs with continuous-time dynamics using Neural ODEs.

The CT-RNN is defined by:
    τ dh/dt = -h + f(W_rec @ h + W_in @ x + b)

where h is the hidden state, x is input, and f is a nonlinearity.

This module uses torchdiffeq for ODE integration, enabling:
- Arbitrary-time predictions
- Smooth interpolation between observations
- Adjoint sensitivity for memory-efficient backpropagation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Callable

try:
    from torchdiffeq import odeint, odeint_adjoint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("Warning: torchdiffeq not available. Install with: pip install torchdiffeq")


class CTRNNCell(nn.Module):
    """
    Continuous-Time RNN Cell defining the ODE dynamics.
    
    This is the 'f' in: dh/dt = f(h, x, t)
    
    Specifically: τ dh/dt = -h + activation(W_rec @ h + W_in @ x + b)
    
    Parameters
    ----------
    input_size : int
        Dimension of input
    hidden_size : int
        Dimension of hidden state
    tau : float
        Time constant (larger = slower dynamics)
    activation : str
        Nonlinearity: 'tanh', 'relu', 'sigmoid'
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        tau: float = 1.0,
        activation: str = 'tanh'
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = tau
        
        # Recurrent weights
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Input weights
        self.W_in = nn.Linear(input_size, hidden_size, bias=True)
        
        # Activation function
        activations = {
            'tanh': torch.tanh,
            'relu': torch.relu,
            'sigmoid': torch.sigmoid,
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        self.activation = activations[activation]
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable dynamics."""
        # Recurrent weights: scaled for stability
        nn.init.orthogonal_(self.W_rec.weight)
        self.W_rec.weight.data *= 0.9  # Spectral radius < 1 for stability
        
        # Input weights
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.zeros_(self.W_in.bias)
    
    def forward(self, t: torch.Tensor, h: torch.Tensor, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute dh/dt.
        
        Parameters
        ----------
        t : torch.Tensor
            Current time (scalar, may not be used)
        h : torch.Tensor
            Hidden state, shape (batch_size, hidden_size)
        x : torch.Tensor, optional
            External input, shape (batch_size, input_size)
            
        Returns
        -------
        dhdt : torch.Tensor
            Time derivative of hidden state
        """
        # Recurrent contribution
        rec = self.W_rec(h)
        
        # Input contribution (if provided)
        if x is not None:
            inp = self.W_in(x)
        else:
            inp = self.W_in.bias.unsqueeze(0).expand(h.shape[0], -1)
        
        # CT-RNN dynamics: τ dh/dt = -h + f(rec + inp)
        dhdt = (-h + self.activation(rec + inp)) / self.tau
        
        return dhdt


class CTRNNODEFunc(nn.Module):
    """
    Wrapper to make CTRNNCell compatible with odeint.
    
    Handles the external input by interpolating or using constant input.
    """
    
    def __init__(self, cell: CTRNNCell, x: Optional[torch.Tensor] = None):
        super().__init__()
        self.cell = cell
        self.x = x  # External input (constant during integration)
    
    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.cell(t, h, self.x)


class ContinuousTimeRNN(nn.Module):
    """
    Complete Continuous-Time RNN model for sequence modeling.
    
    Architecture:
    1. Input encoder (optional): maps input to hidden dimension
    2. CT-RNN dynamics: continuous-time recurrent processing
    3. Output decoder: maps hidden state to output
    
    Parameters
    ----------
    input_size : int
        Dimension of input
    hidden_size : int
        Dimension of hidden state
    output_size : int
        Dimension of output
    tau : float
        Time constant
    activation : str
        Nonlinearity for CT-RNN
    solver : str
        ODE solver: 'dopri5', 'euler', 'rk4', 'adaptive_heun'
    use_adjoint : bool
        Use adjoint method for backprop (memory efficient but slower)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        tau: float = 1.0,
        activation: str = 'tanh',
        solver: str = 'dopri5',
        use_adjoint: bool = False
    ):
        super().__init__()
        
        if not TORCHDIFFEQ_AVAILABLE:
            raise ImportError("torchdiffeq required for ContinuousTimeRNN")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.solver = solver
        self.use_adjoint = use_adjoint
        
        # CT-RNN cell
        self.cell = CTRNNCell(input_size, hidden_size, tau, activation)
        
        # Output decoder
        self.decoder = nn.Linear(hidden_size, output_size)
        
        # Choose ODE integrator
        self.odeint = odeint_adjoint if use_adjoint else odeint
    
    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        h0: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        return_all_outputs: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the CT-RNN.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, shape (batch_size, seq_length, input_size)
            For continuous-time mode, can be (batch_size, input_size) for constant input
        t : torch.Tensor, optional
            Time points for integration, shape (n_times,)
            If None, uses discrete steps [0, 1, 2, ..., seq_length]
        h0 : torch.Tensor, optional
            Initial hidden state, shape (batch_size, hidden_size)
        return_hidden : bool
            Whether to return hidden states at all time points
        return_all_outputs : bool
            Whether to return outputs at all timesteps (default: True)
            If False, returns only final output (for Lorenz forecasting)

        Returns
        -------
        output : torch.Tensor
            If return_hidden: (hidden_trajectory, output)
            Else if return_all_outputs: (batch_size, seq_length, output_size)
            Else: final output, shape (batch_size, output_size)
        """
        batch_size = x.shape[0]
        device = x.device

        # Initialize hidden state
        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=device)

        # Handle different input modes
        if x.dim() == 3:
            # Sequence input: process step by step with discrete-time approximation
            # For full continuous-time, use integrate_continuous method
            seq_length = x.shape[1]
            if t is None:
                t = torch.arange(seq_length + 1, dtype=torch.float32, device=device)

            hidden_states = [h0]
            h = h0

            for i in range(seq_length):
                # Integrate from t[i] to t[i+1] with current input
                ode_func = CTRNNODEFunc(self.cell, x[:, i, :])
                t_span = torch.tensor([t[i], t[i+1]], device=device)
                h_traj = self.odeint(ode_func, h, t_span, method=self.solver)
                h = h_traj[-1]  # Take final state
                hidden_states.append(h)

            hidden_states = torch.stack(hidden_states[1:], dim=1)  # (batch, seq, hidden)

            # Decode outputs
            if return_all_outputs:
                # Decode all timesteps (for sequence-to-sequence tasks like flip-flop)
                output = self.decoder(hidden_states)  # (batch, seq, output)
            else:
                # Decode only final timestep (for forecasting tasks like Lorenz)
                output = self.decoder(hidden_states[:, -1, :])  # (batch, output)

        else:
            # Constant input: integrate over time span
            if t is None:
                t = torch.linspace(0, 1, 10, device=device)

            ode_func = CTRNNODEFunc(self.cell, x)
            hidden_states = self.odeint(ode_func, h0, t, method=self.solver)
            hidden_states = hidden_states.permute(1, 0, 2)  # (batch, time, hidden)
            output = self.decoder(hidden_states[:, -1, :])

        if return_hidden:
            return hidden_states, output
        return output
    
    def integrate_continuous(
        self,
        h0: torch.Tensor,
        t: torch.Tensor,
        x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Integrate CT-RNN dynamics over arbitrary time points.
        
        Parameters
        ----------
        h0 : torch.Tensor
            Initial hidden state, shape (batch_size, hidden_size)
        t : torch.Tensor
            Time points, shape (n_times,)
        x : torch.Tensor, optional
            Constant external input, shape (batch_size, input_size)
            
        Returns
        -------
        trajectory : torch.Tensor
            Hidden states at all time points, shape (batch_size, n_times, hidden_size)
        """
        ode_func = CTRNNODEFunc(self.cell, x)
        trajectory = self.odeint(ode_func, h0, t, method=self.solver)
        return trajectory.permute(1, 0, 2)  # (batch, time, hidden)
    
    def generate(
        self,
        initial_state: torch.Tensor,
        n_steps: int,
        dt: float = 0.01
    ) -> torch.Tensor:
        """
        Autonomous generation: let the network evolve freely.
        
        The network predicts its own next state and uses that as input.
        
        Parameters
        ----------
        initial_state : torch.Tensor
            Initial output state, shape (batch_size, output_size)
        n_steps : int
            Number of steps to generate
        dt : float
            Time step
            
        Returns
        -------
        trajectory : torch.Tensor
            Generated trajectory, shape (batch_size, n_steps, output_size)
        """
        device = initial_state.device
        batch_size = initial_state.shape[0]
        
        # Initialize
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        current_output = initial_state
        
        trajectory = [current_output]
        
        for _ in range(n_steps - 1):
            # Use current output as input
            t_span = torch.tensor([0, dt], device=device)
            ode_func = CTRNNODEFunc(self.cell, current_output)
            h_traj = self.odeint(ode_func, h, t_span, method=self.solver)
            h = h_traj[-1]
            
            # Decode to output
            current_output = self.decoder(h)
            trajectory.append(current_output)
        
        return torch.stack(trajectory, dim=1)


# =============================================================================
# Utility Functions
# =============================================================================

def analyze_ctrnn_jacobian(
    cell: CTRNNCell,
    h: torch.Tensor,
    x: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Jacobian of CT-RNN dynamics at a given state.
    
    This is useful for fixed point stability analysis.
    
    Parameters
    ----------
    cell : CTRNNCell
        The CT-RNN cell
    h : torch.Tensor
        Hidden state, shape (hidden_size,) or (batch_size, hidden_size)
    x : torch.Tensor, optional
        External input
        
    Returns
    -------
    jacobian : torch.Tensor
        Jacobian matrix, shape (hidden_size, hidden_size) or (batch, hidden, hidden)
    """
    if h.dim() == 1:
        h = h.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size = h.shape[0]
    hidden_size = h.shape[1]
    
    # Compute Jacobian using autograd
    h.requires_grad_(True)
    t = torch.tensor(0.0)
    
    jacobians = []
    for b in range(batch_size):
        dhdt = cell(t, h[b:b+1], x[b:b+1] if x is not None else None)
        jac = torch.autograd.functional.jacobian(
            lambda h_: cell(t, h_.unsqueeze(0), x[b:b+1] if x is not None else None).squeeze(0),
            h[b]
        )
        jacobians.append(jac)
    
    jacobian = torch.stack(jacobians)
    
    if squeeze_output:
        jacobian = jacobian.squeeze(0)
    
    return jacobian


if __name__ == "__main__":
    # Quick test
    print("Testing Continuous-Time RNN...")
    
    # Create model
    model = ContinuousTimeRNN(
        input_size=3,
        hidden_size=64,
        output_size=3,
        tau=1.0
    )
    
    # Test forward pass
    batch_size = 8
    seq_length = 50
    x = torch.randn(batch_size, seq_length, 3)
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with hidden states
    hidden, output = model(x, return_hidden=True)
    print(f"Hidden trajectory shape: {hidden.shape}")
    
    # Test generation
    initial = torch.randn(batch_size, 3)
    generated = model.generate(initial, n_steps=100)
    print(f"Generated trajectory shape: {generated.shape}")
    
    print("All tests passed!")
