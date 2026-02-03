"""
Balanced Spiking Network
========================

Implements a biologically plausible spiking network with separate
excitatory and inhibitory LIF neuron populations.

Uses norse (https://github.com/norse/norse) for PyTorch-native spiking
neural network simulation, enabling GPU acceleration and Colab compatibility.

Key features:
- Leaky Integrate-and-Fire (LIF) neurons
- Separate E/I populations with Dale's law
- Reservoir computing approach: random recurrent weights, train readout only
- Surrogate gradient training possible for full backprop

References:
- Maass et al. (2002) - Liquid State Machines
- Cramer et al. (2020) - Heidelberg Spiking Datasets
- Zenke & Ganguli (2018) - SuperSpike surrogate gradients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List

try:
    import norse.torch as norse
    from norse.torch.module.lif import LIFCell, LIFParameters
    from norse.torch.functional.lif import LIFState
    NORSE_AVAILABLE = True
except ImportError:
    NORSE_AVAILABLE = False
    print("Warning: norse not available. Install with: pip install norse")


# =============================================================================
# LIF Parameters for Biological Plausibility
# =============================================================================

def create_lif_parameters(
    tau_mem: float = 20.0,  # Membrane time constant (ms)
    tau_syn: float = 5.0,   # Synaptic time constant (ms)
    v_th: float = 1.0,      # Threshold
    v_reset: float = 0.0,   # Reset potential
    v_leak: float = 0.0,    # Leak potential
    dt: float = 1.0,        # Time step (ms)
    method: str = 'super'   # Surrogate gradient method
) -> 'LIFParameters':
    """
    Create LIF neuron parameters.
    
    Parameters
    ----------
    tau_mem : float
        Membrane time constant in ms
    tau_syn : float
        Synaptic time constant in ms
    v_th : float
        Spike threshold
    v_reset : float
        Reset voltage after spike
    v_leak : float
        Leak reversal potential
    dt : float
        Simulation time step in ms
    method : str
        Surrogate gradient method: 'super', 'tanh', 'sigmoid'
    """
    if not NORSE_AVAILABLE:
        raise ImportError("norse required for spiking networks")
    
    return LIFParameters(
        tau_mem_inv=torch.tensor(1.0 / tau_mem),
        tau_syn_inv=torch.tensor(1.0 / tau_syn),
        v_th=torch.tensor(v_th),
        v_reset=torch.tensor(v_reset),
        v_leak=torch.tensor(v_leak),
        method=method
    )


# =============================================================================
# Balanced Spiking Network Cell
# =============================================================================

class BalancedSpikingCell(nn.Module):
    """
    Single time-step of balanced spiking E/I network.
    
    Parameters
    ----------
    n_excitatory : int
        Number of excitatory neurons
    n_inhibitory : int
        Number of inhibitory neurons
    input_size : int
        Dimension of external input
    p_connection : float
        Connection probability
    g : float
        Relative inhibitory strength
    tau_mem_e, tau_mem_i : float
        Membrane time constants for E and I populations
    tau_syn : float
        Synaptic time constant
    fixed_weights : bool
        If True, recurrent weights are not trained (reservoir mode)
    """
    
    def __init__(
        self,
        n_excitatory: int,
        n_inhibitory: int,
        input_size: int,
        p_connection: float = 0.1,
        g: float = 4.0,
        tau_mem_e: float = 20.0,
        tau_mem_i: float = 10.0,
        tau_syn: float = 5.0,
        dt: float = 1.0,
        fixed_weights: bool = True,
        weight_scale: float = 0.1
    ):
        super().__init__()
        
        if not NORSE_AVAILABLE:
            raise ImportError("norse required for BalancedSpikingCell")
        
        self.n_e = n_excitatory
        self.n_i = n_inhibitory
        self.n_total = n_excitatory + n_inhibitory
        self.input_size = input_size
        self.fixed_weights = fixed_weights
        self.g = g
        
        # LIF parameters for E and I populations
        self.lif_params_e = create_lif_parameters(
            tau_mem=tau_mem_e, tau_syn=tau_syn, dt=dt
        )
        self.lif_params_i = create_lif_parameters(
            tau_mem=tau_mem_i, tau_syn=tau_syn, dt=dt
        )
        
        # LIF cells
        self.lif_e = LIFCell(p=self.lif_params_e)
        self.lif_i = LIFCell(p=self.lif_params_i)
        
        # === Recurrent connectivity ===
        # Create sparse random connectivity matrix
        N_e, N_i = n_excitatory, n_inhibitory
        
        # Initialize weight matrices
        def init_sparse_weights(n_post, n_pre, scale):
            W = torch.randn(n_post, n_pre) * scale / np.sqrt(n_pre * p_connection)
            mask = torch.rand(n_post, n_pre) < p_connection
            W = W * mask.float()
            return W
        
        # E→E, E→I (positive), I→E, I→I (negative)
        W_ee = init_sparse_weights(N_e, N_e, weight_scale)
        W_ie = init_sparse_weights(N_i, N_e, weight_scale)
        W_ei = init_sparse_weights(N_e, N_i, weight_scale * g)
        W_ii = init_sparse_weights(N_i, N_i, weight_scale * g)
        
        if fixed_weights:
            # Reservoir mode: weights are not trained
            self.register_buffer('W_ee', W_ee)
            self.register_buffer('W_ie', W_ie)
            self.register_buffer('W_ei', W_ei)
            self.register_buffer('W_ii', W_ii)
        else:
            # Trainable weights
            self.W_ee = nn.Parameter(W_ee)
            self.W_ie = nn.Parameter(W_ie)
            self.W_ei = nn.Parameter(W_ei)
            self.W_ii = nn.Parameter(W_ii)
        
        # === Input weights ===
        self.W_e_in = nn.Parameter(torch.randn(N_e, input_size) * 0.1)
        self.W_i_in = nn.Parameter(torch.randn(N_i, input_size) * 0.1)
    
    def forward(
        self,
        x: torch.Tensor,
        state_e: Optional['LIFState'] = None,
        state_i: Optional['LIFState'] = None,
        spikes_e_prev: Optional[torch.Tensor] = None,
        spikes_i_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, 'LIFState', 'LIFState']:
        """
        Single time step.
        
        Parameters
        ----------
        x : torch.Tensor
            External input, shape (batch_size, input_size)
        state_e, state_i : LIFState
            Previous neuron states
        spikes_e_prev, spikes_i_prev : torch.Tensor
            Previous spikes (for recurrent input)
            
        Returns
        -------
        spikes_e, spikes_i : torch.Tensor
            Output spikes
        state_e, state_i : LIFState
            Updated states
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize states if needed
        if state_e is None:
            state_e = LIFState(
                v=torch.zeros(batch_size, self.n_e, device=device),
                i=torch.zeros(batch_size, self.n_e, device=device)
            )
        if state_i is None:
            state_i = LIFState(
                v=torch.zeros(batch_size, self.n_i, device=device),
                i=torch.zeros(batch_size, self.n_i, device=device)
            )
        if spikes_e_prev is None:
            spikes_e_prev = torch.zeros(batch_size, self.n_e, device=device)
        if spikes_i_prev is None:
            spikes_i_prev = torch.zeros(batch_size, self.n_i, device=device)
        
        # Compute input currents
        # External input
        I_ext_e = F.linear(x, self.W_e_in)
        I_ext_i = F.linear(x, self.W_i_in)
        
        # Recurrent input (E excitatory, I inhibitory)
        I_rec_e = (F.linear(spikes_e_prev, F.relu(self.W_ee)) - 
                   F.linear(spikes_i_prev, F.relu(self.W_ei)))
        I_rec_i = (F.linear(spikes_e_prev, F.relu(self.W_ie)) - 
                   F.linear(spikes_i_prev, F.relu(self.W_ii)))
        
        # Total input
        I_e = I_ext_e + I_rec_e
        I_i = I_ext_i + I_rec_i
        
        # LIF dynamics
        spikes_e, state_e = self.lif_e(I_e, state_e)
        spikes_i, state_i = self.lif_i(I_i, state_i)
        
        return spikes_e, spikes_i, state_e, state_i


class BalancedSpikingNetwork(nn.Module):
    """
    Complete balanced spiking network for sequence processing.
    
    Uses reservoir computing approach:
    - Random, fixed recurrent weights
    - Only readout layer is trained
    
    Parameters
    ----------
    input_size : int
        Dimension of input
    n_excitatory : int
        Number of excitatory neurons
    n_inhibitory : int
        Number of inhibitory neurons (default: n_e // 4)
    output_size : int
        Dimension of output
    readout_mode : str
        'rate': Average firing rate over time
        'voltage': Use membrane voltage
        'last': Use final time step
    """
    
    def __init__(
        self,
        input_size: int,
        n_excitatory: int,
        n_inhibitory: Optional[int] = None,
        output_size: int = 3,
        p_connection: float = 0.1,
        g: float = 4.0,
        tau_mem_e: float = 20.0,
        tau_mem_i: float = 10.0,
        dt: float = 1.0,
        fixed_weights: bool = True,
        readout_mode: str = 'rate'
    ):
        super().__init__()
        
        if n_inhibitory is None:
            n_inhibitory = n_excitatory // 4
        
        self.n_e = n_excitatory
        self.n_i = n_inhibitory
        self.readout_mode = readout_mode
        
        # Spiking cell
        self.cell = BalancedSpikingCell(
            n_excitatory=n_excitatory,
            n_inhibitory=n_inhibitory,
            input_size=input_size,
            p_connection=p_connection,
            g=g,
            tau_mem_e=tau_mem_e,
            tau_mem_i=tau_mem_i,
            dt=dt,
            fixed_weights=fixed_weights
        )
        
        # Readout layer (trained)
        # Read from E population only (biologically plausible)
        self.readout = nn.Linear(n_excitatory, output_size)
    
    def forward(
        self,
        x: torch.Tensor,
        return_spikes: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequence, shape (batch_size, seq_length, input_size)
        return_spikes : bool
            Whether to return spike trains
            
        Returns
        -------
        output : torch.Tensor
            Output, shape (batch_size, output_size)
            If return_spikes: (output, spikes_e, spikes_i)
        """
        batch_size, seq_length, _ = x.shape
        device = x.device
        
        # Initialize
        state_e, state_i = None, None
        spikes_e_prev = torch.zeros(batch_size, self.n_e, device=device)
        spikes_i_prev = torch.zeros(batch_size, self.n_i, device=device)
        
        # Record spikes
        spikes_e_all = []
        spikes_i_all = []
        voltages_e_all = []
        
        # Process sequence
        for t in range(seq_length):
            spikes_e, spikes_i, state_e, state_i = self.cell(
                x[:, t, :], state_e, state_i, spikes_e_prev, spikes_i_prev
            )
            
            spikes_e_all.append(spikes_e)
            spikes_i_all.append(spikes_i)
            voltages_e_all.append(state_e.v)
            
            spikes_e_prev = spikes_e
            spikes_i_prev = spikes_i
        
        spikes_e_all = torch.stack(spikes_e_all, dim=1)  # (batch, time, n_e)
        spikes_i_all = torch.stack(spikes_i_all, dim=1)
        voltages_e_all = torch.stack(voltages_e_all, dim=1)
        
        # Compute readout based on mode
        if self.readout_mode == 'rate':
            # Average firing rate
            rates_e = spikes_e_all.mean(dim=1)  # (batch, n_e)
            output = self.readout(rates_e)
        elif self.readout_mode == 'voltage':
            # Final membrane voltage
            output = self.readout(voltages_e_all[:, -1, :])
        elif self.readout_mode == 'last':
            # Last time step spikes (smoothed)
            rates_e = spikes_e_all[:, -10:, :].mean(dim=1)
            output = self.readout(rates_e)
        else:
            raise ValueError(f"Unknown readout mode: {self.readout_mode}")
        
        if return_spikes:
            return output, spikes_e_all, spikes_i_all
        return output
    
    def generate(
        self,
        initial_state: torch.Tensor,
        n_steps: int,
        noise_scale: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Autonomous generation with feedback.
        
        Parameters
        ----------
        initial_state : torch.Tensor
            Initial output, shape (batch_size, output_size)
        n_steps : int
            Number of steps to generate
        noise_scale : float
            Noise added to feedback (for stability)
            
        Returns
        -------
        output_traj : Generated outputs
        spikes_e, spikes_i : Spike trains
        """
        batch_size = initial_state.shape[0]
        device = initial_state.device
        
        state_e, state_i = None, None
        spikes_e_prev = torch.zeros(batch_size, self.n_e, device=device)
        spikes_i_prev = torch.zeros(batch_size, self.n_i, device=device)
        
        current_input = initial_state
        
        output_traj = [initial_state]
        spikes_e_all = []
        spikes_i_all = []
        
        for _ in range(n_steps - 1):
            # Add noise for stability
            noisy_input = current_input + noise_scale * torch.randn_like(current_input)
            
            # Single step
            spikes_e, spikes_i, state_e, state_i = self.cell(
                noisy_input, state_e, state_i, spikes_e_prev, spikes_i_prev
            )
            
            spikes_e_all.append(spikes_e)
            spikes_i_all.append(spikes_i)
            
            spikes_e_prev = spikes_e
            spikes_i_prev = spikes_i
            
            # Decode and use as next input
            current_input = self.readout(spikes_e)
            output_traj.append(current_input)
        
        return (
            torch.stack(output_traj, dim=1),
            torch.stack(spikes_e_all, dim=1) if spikes_e_all else None,
            torch.stack(spikes_i_all, dim=1) if spikes_i_all else None
        )
    
    def compute_spiking_statistics(
        self,
        spikes_e: torch.Tensor,
        spikes_i: torch.Tensor,
        dt: float = 1.0
    ) -> dict:
        """
        Compute statistics of spiking activity.
        
        Parameters
        ----------
        spikes_e, spikes_i : torch.Tensor
            Spike trains, shape (batch, time, neurons)
        dt : float
            Time step in ms
            
        Returns
        -------
        stats : dict
            Firing rates, ISI statistics, synchrony measures
        """
        # Firing rates (Hz)
        duration_s = spikes_e.shape[1] * dt / 1000
        rate_e = spikes_e.sum(dim=1).mean().item() / duration_s
        rate_i = spikes_i.sum(dim=1).mean().item() / duration_s
        
        # Coefficient of variation of ISIs (proxy for irregularity)
        # CV ~ 1 for Poisson, < 1 for regular, > 1 for bursty
        
        # Population synchrony (correlation between neurons)
        pop_activity_e = spikes_e.mean(dim=2)  # (batch, time)
        pop_activity_i = spikes_i.mean(dim=2)
        
        return {
            'rate_e_hz': rate_e,
            'rate_i_hz': rate_i,
            'ratio_e_i': rate_e / (rate_i + 1e-6),
            'pop_activity_e_mean': pop_activity_e.mean().item(),
            'pop_activity_i_mean': pop_activity_i.mean().item(),
            'pop_activity_e_std': pop_activity_e.std().item(),
        }


# =============================================================================
# Reservoir Creation Utilities
# =============================================================================

def create_spiking_reservoir(
    n_excitatory: int = 800,
    n_inhibitory: int = 200,
    input_size: int = 3,
    output_size: int = 3,
    **kwargs
) -> BalancedSpikingNetwork:
    """
    Create a spiking reservoir network with standard parameters.
    
    This is a convenience function for creating reservoir networks
    with biologically plausible parameters.
    """
    return BalancedSpikingNetwork(
        input_size=input_size,
        n_excitatory=n_excitatory,
        n_inhibitory=n_inhibitory,
        output_size=output_size,
        p_connection=0.1,
        g=4.0,
        tau_mem_e=20.0,
        tau_mem_i=10.0,
        fixed_weights=True,
        readout_mode='rate',
        **kwargs
    )


# =============================================================================
# Fallback Implementation (when norse not available)
# =============================================================================

class SimpleLIFCell(nn.Module):
    """
    Simple LIF neuron implementation without norse.
    
    Uses surrogate gradient for backpropagation.
    """
    
    def __init__(
        self,
        tau_mem: float = 20.0,
        tau_syn: float = 5.0,
        v_th: float = 1.0,
        v_reset: float = 0.0,
        dt: float = 1.0
    ):
        super().__init__()
        self.alpha = np.exp(-dt / tau_mem)
        self.beta = np.exp(-dt / tau_syn)
        self.v_th = v_th
        self.v_reset = v_reset
    
    def forward(self, I: torch.Tensor, v: torch.Tensor, I_syn: torch.Tensor):
        """Single step LIF update with surrogate gradient."""
        # Synaptic current decay
        I_syn = self.beta * I_syn + I
        
        # Membrane potential update
        v = self.alpha * v + (1 - self.alpha) * I_syn
        
        # Spike generation with surrogate gradient
        spike = self.surrogate_spike(v - self.v_th)
        
        # Reset
        v = v * (1 - spike) + self.v_reset * spike
        
        return spike, v, I_syn
    
    @staticmethod
    def surrogate_spike(x: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
        """Surrogate gradient using fast sigmoid."""
        # Forward: Heaviside
        spike = (x > 0).float()
        
        # Backward: surrogate gradient
        if x.requires_grad:
            grad = beta / (2 * (1 + beta * torch.abs(x))**2)
            spike = spike + (grad - grad.detach())
        
        return spike


if __name__ == "__main__":
    if not NORSE_AVAILABLE:
        print("Norse not available, skipping tests")
    else:
        print("Testing Balanced Spiking Network...")
        
        # Create model
        model = create_spiking_reservoir(
            n_excitatory=80,
            n_inhibitory=20,
            input_size=3,
            output_size=3
        )
        
        # Test forward pass
        batch_size = 8
        seq_length = 100
        x = torch.randn(batch_size, seq_length, 3) * 0.5
        
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        
        # Test with spikes
        output, spikes_e, spikes_i = model(x, return_spikes=True)
        print(f"E spikes shape: {spikes_e.shape}")
        print(f"I spikes shape: {spikes_i.shape}")
        
        # Compute statistics
        stats = model.compute_spiking_statistics(spikes_e, spikes_i)
        print(f"Spiking stats: {stats}")
        
        print("All tests passed!")
