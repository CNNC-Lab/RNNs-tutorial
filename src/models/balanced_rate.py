"""
Balanced Excitatory-Inhibitory Rate Network
============================================

Implements a biologically plausible rate network with separate
excitatory and inhibitory populations, obeying Dale's law.

The network dynamics follow:
    τ_E dr_E/dt = -r_E + φ(W_EE @ r_E - W_EI @ r_I + W_Ein @ x + b_E)
    τ_I dr_I/dt = -r_I + φ(W_IE @ r_E - W_II @ r_I + W_Iin @ x + b_I)

Key features:
- Separate E/I populations (Dale's law: neurons are either E or I)
- Positive weight constraints on E→* connections
- Dynamically balanced regime (large mean inputs cancel)
- Sparse connectivity option

References:
- van Vreeswijk & Sompolinsky (1996, 1998)
- Ingrosso & Abbott (2019) - Training balanced networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class EIRateCell(nn.Module):
    """
    Single-step dynamics for balanced E/I rate network.
    
    Parameters
    ----------
    n_excitatory : int
        Number of excitatory units
    n_inhibitory : int
        Number of inhibitory units
    input_size : int
        Dimension of external input
    tau_e : float
        Time constant for E population (ms)
    tau_i : float
        Time constant for I population (ms)
    dt : float
        Integration time step (ms)
    connection_prob : float
        Connection probability (for sparse networks)
    g : float
        Relative strength of inhibition (balance parameter)
    activation : str
        Nonlinearity: 'relu', 'softplus', 'tanh'
    """
    
    def __init__(
        self,
        n_excitatory: int,
        n_inhibitory: int,
        input_size: int,
        tau_e: float = 10.0,
        tau_i: float = 5.0,
        dt: float = 1.0,
        connection_prob: float = 1.0,
        g: float = 1.0,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.n_e = n_excitatory
        self.n_i = n_inhibitory
        self.n_total = n_excitatory + n_inhibitory
        self.input_size = input_size
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.dt = dt
        self.g = g
        
        # Activation function
        if activation == 'relu':
            self.phi = F.relu
        elif activation == 'softplus':
            self.phi = F.softplus
        elif activation == 'tanh':
            self.phi = lambda x: torch.tanh(x) * 0.5 + 0.5  # Maps to [0, 1]
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # === Recurrent weights (with Dale's law constraints) ===
        # E→E connections (positive)
        self.W_ee = nn.Parameter(torch.empty(n_excitatory, n_excitatory))
        # E→I connections (positive)  
        self.W_ie = nn.Parameter(torch.empty(n_inhibitory, n_excitatory))
        # I→E connections (negative, stored as positive, applied with minus sign)
        self.W_ei = nn.Parameter(torch.empty(n_excitatory, n_inhibitory))
        # I→I connections (negative, stored as positive)
        self.W_ii = nn.Parameter(torch.empty(n_inhibitory, n_inhibitory))
        
        # === Input weights ===
        self.W_e_in = nn.Parameter(torch.empty(n_excitatory, input_size))
        self.W_i_in = nn.Parameter(torch.empty(n_inhibitory, input_size))
        
        # === Biases (baseline drive) ===
        self.b_e = nn.Parameter(torch.zeros(n_excitatory))
        self.b_i = nn.Parameter(torch.zeros(n_inhibitory))
        
        # === Connection mask (for sparse connectivity) ===
        if connection_prob < 1.0:
            self.register_buffer('mask_ee', (torch.rand(n_excitatory, n_excitatory) < connection_prob).float())
            self.register_buffer('mask_ie', (torch.rand(n_inhibitory, n_excitatory) < connection_prob).float())
            self.register_buffer('mask_ei', (torch.rand(n_excitatory, n_inhibitory) < connection_prob).float())
            self.register_buffer('mask_ii', (torch.rand(n_inhibitory, n_inhibitory) < connection_prob).float())
            self.sparse = True
        else:
            self.sparse = False
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights for balanced dynamics.
        
        Uses scaling from van Vreeswijk & Sompolinsky:
        - Mean weights scale as 1/sqrt(N) for balance
        - Variance scales as 1/N
        """
        N = self.n_total
        
        # Scaling factors
        scale_ee = 1.0 / np.sqrt(self.n_e)
        scale_ie = 1.0 / np.sqrt(self.n_e)
        scale_ei = self.g / np.sqrt(self.n_i)  # Inhibition scaled by g
        scale_ii = self.g / np.sqrt(self.n_i)
        
        # Initialize with positive values (will be constrained in forward)
        nn.init.uniform_(self.W_ee, 0, scale_ee)
        nn.init.uniform_(self.W_ie, 0, scale_ie)
        nn.init.uniform_(self.W_ei, 0, scale_ei)
        nn.init.uniform_(self.W_ii, 0, scale_ii)
        
        # Input weights
        nn.init.xavier_uniform_(self.W_e_in)
        nn.init.xavier_uniform_(self.W_i_in)
        
        # Small positive bias for baseline activity
        nn.init.constant_(self.b_e, 0.5)
        nn.init.constant_(self.b_i, 0.5)
    
    def _get_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get weight matrices with Dale's law constraints (E weights positive, I weights negative).
        """
        # Ensure positive weights for E populations
        W_ee = F.softplus(self.W_ee)
        W_ie = F.softplus(self.W_ie)
        W_ei = F.softplus(self.W_ei)
        W_ii = F.softplus(self.W_ii)
        
        # Apply sparsity mask if applicable
        if self.sparse:
            W_ee = W_ee * self.mask_ee
            W_ie = W_ie * self.mask_ie
            W_ei = W_ei * self.mask_ei
            W_ii = W_ii * self.mask_ii
        
        return W_ee, W_ie, W_ei, W_ii
    
    def forward(
        self,
        r_e: torch.Tensor,
        r_i: torch.Tensor,
        x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single time step update.
        
        Parameters
        ----------
        r_e : torch.Tensor
            Excitatory rates, shape (batch_size, n_excitatory)
        r_i : torch.Tensor
            Inhibitory rates, shape (batch_size, n_inhibitory)
        x : torch.Tensor, optional
            External input, shape (batch_size, input_size)
            
        Returns
        -------
        r_e_new, r_i_new : torch.Tensor
            Updated rates
        """
        W_ee, W_ie, W_ei, W_ii = self._get_weights()
        
        # Compute currents to E population
        # I_e = W_ee @ r_e - W_ei @ r_i + W_e_in @ x + b_e
        I_e = F.linear(r_e, W_ee) - F.linear(r_i, W_ei) + self.b_e
        if x is not None:
            I_e = I_e + F.linear(x, self.W_e_in)
        
        # Compute currents to I population
        # I_i = W_ie @ r_e - W_ii @ r_i + W_i_in @ x + b_i
        I_i = F.linear(r_e, W_ie) - F.linear(r_i, W_ii) + self.b_i
        if x is not None:
            I_i = I_i + F.linear(x, self.W_i_in)
        
        # Euler integration
        dr_e = (-r_e + self.phi(I_e)) * (self.dt / self.tau_e)
        dr_i = (-r_i + self.phi(I_i)) * (self.dt / self.tau_i)
        
        r_e_new = r_e + dr_e
        r_i_new = r_i + dr_i
        
        return r_e_new, r_i_new
    
    def get_currents(
        self,
        r_e: torch.Tensor,
        r_i: torch.Tensor,
        x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get excitatory and inhibitory currents separately (for analysis).
        
        Returns
        -------
        I_e_exc : Excitatory current to E neurons
        I_e_inh : Inhibitory current to E neurons
        I_i_exc : Excitatory current to I neurons
        I_i_inh : Inhibitory current to I neurons
        """
        W_ee, W_ie, W_ei, W_ii = self._get_weights()
        
        I_e_exc = F.linear(r_e, W_ee)
        I_e_inh = F.linear(r_i, W_ei)
        I_i_exc = F.linear(r_e, W_ie)
        I_i_inh = F.linear(r_i, W_ii)
        
        return I_e_exc, I_e_inh, I_i_exc, I_i_inh


class BalancedRateNetwork(nn.Module):
    """
    Complete balanced E/I rate network for sequence modeling.
    
    Architecture:
    1. Input encoder: maps input to E/I populations
    2. E/I recurrent dynamics: balanced rate equations
    3. Output decoder: maps E population activity to output
    
    Parameters
    ----------
    input_size : int
        Dimension of input
    n_excitatory : int
        Number of excitatory units
    n_inhibitory : int
        Number of inhibitory units (default: n_e // 4 for 80/20 split)
    output_size : int
        Dimension of output
    tau_e, tau_i : float
        Time constants
    dt : float
        Integration time step
    g : float
        Inhibition strength
    """
    
    def __init__(
        self,
        input_size: int,
        n_excitatory: int,
        n_inhibitory: Optional[int] = None,
        output_size: int = 3,
        tau_e: float = 10.0,
        tau_i: float = 5.0,
        dt: float = 1.0,
        g: float = 1.5,
        connection_prob: float = 1.0,
        activation: str = 'relu'
    ):
        super().__init__()
        
        if n_inhibitory is None:
            n_inhibitory = n_excitatory // 4  # 80/20 E/I split
        
        self.n_e = n_excitatory
        self.n_i = n_inhibitory
        self.input_size = input_size
        self.output_size = output_size
        
        # E/I rate cell
        self.cell = EIRateCell(
            n_excitatory=n_excitatory,
            n_inhibitory=n_inhibitory,
            input_size=input_size,
            tau_e=tau_e,
            tau_i=tau_i,
            dt=dt,
            connection_prob=connection_prob,
            g=g,
            activation=activation
        )
        
        # Output decoder (read from E population only, biologically plausible)
        self.decoder = nn.Linear(n_excitatory, output_size)
    
    def forward(
        self,
        x: torch.Tensor,
        r_e0: Optional[torch.Tensor] = None,
        r_i0: Optional[torch.Tensor] = None,
        return_dynamics: bool = False,
        return_all_outputs: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, shape (batch_size, seq_length, input_size)
        r_e0, r_i0 : torch.Tensor, optional
            Initial E/I rates
        return_dynamics : bool
            Whether to return full E/I trajectories
        return_all_outputs : bool
            Whether to return outputs at all timesteps (default: False)
            If False, returns only final output (for Lorenz forecasting)
            If True, returns outputs at all timesteps (for sequence tasks)

        Returns
        -------
        output : torch.Tensor
            If return_all_outputs: (batch_size, seq_length, output_size)
            Else: final output, shape (batch_size, output_size)
            If return_dynamics: (r_e_traj, r_i_traj, output)
        """
        batch_size, seq_length, _ = x.shape
        device = x.device

        # Initialize rates
        if r_e0 is None:
            r_e = torch.zeros(batch_size, self.n_e, device=device)
        else:
            r_e = r_e0

        if r_i0 is None:
            r_i = torch.zeros(batch_size, self.n_i, device=device)
        else:
            r_i = r_i0

        # Store trajectories if requested
        if return_dynamics or return_all_outputs:
            r_e_traj = [r_e]
            r_i_traj = [r_i]

        # Process sequence
        for t in range(seq_length):
            r_e, r_i = self.cell(r_e, r_i, x[:, t, :])

            if return_dynamics or return_all_outputs:
                r_e_traj.append(r_e)
                r_i_traj.append(r_i)

        # Decode output
        if return_all_outputs:
            # Decode all timesteps (skip initial state, use states after each input)
            r_e_all = torch.stack(r_e_traj[1:], dim=1)  # (batch, seq, n_e)
            output = self.decoder(r_e_all)  # (batch, seq, output)
        else:
            # Decode only final timestep
            output = self.decoder(r_e)  # (batch, output)

        if return_dynamics:
            r_e_traj = torch.stack(r_e_traj, dim=1)
            r_i_traj = torch.stack(r_i_traj, dim=1)
            return r_e_traj, r_i_traj, output
        
        return output
    
    def generate(
        self,
        initial_state: torch.Tensor,
        n_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Autonomous generation.
        
        Parameters
        ----------
        initial_state : torch.Tensor
            Initial output, shape (batch_size, output_size)
        n_steps : int
            Number of steps to generate
            
        Returns
        -------
        output_traj : torch.Tensor
            Generated outputs, shape (batch_size, n_steps, output_size)
        r_e_traj, r_i_traj : torch.Tensor
            E/I rate trajectories
        """
        batch_size = initial_state.shape[0]
        device = initial_state.device
        
        # Initialize
        r_e = torch.zeros(batch_size, self.n_e, device=device)
        r_i = torch.zeros(batch_size, self.n_i, device=device)
        current_input = initial_state
        
        output_traj = [initial_state]
        r_e_traj = [r_e]
        r_i_traj = [r_i]
        
        for _ in range(n_steps - 1):
            # Use current output as input
            r_e, r_i = self.cell(r_e, r_i, current_input)
            current_input = self.decoder(r_e)
            
            output_traj.append(current_input)
            r_e_traj.append(r_e)
            r_i_traj.append(r_i)
        
        return (
            torch.stack(output_traj, dim=1),
            torch.stack(r_e_traj, dim=1),
            torch.stack(r_i_traj, dim=1)
        )
    
    def compute_balance_metrics(
        self,
        r_e: torch.Tensor,
        r_i: torch.Tensor,
        x: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute metrics to assess E/I balance.
        
        Returns
        -------
        metrics : dict
            'mean_E_current': Mean excitatory current
            'mean_I_current': Mean inhibitory current
            'balance_ratio': |E-I| / (|E|+|I|), close to 0 is balanced
            'cv_rates': Coefficient of variation of firing rates
        """
        I_e_exc, I_e_inh, I_i_exc, I_i_inh = self.cell.get_currents(r_e, r_i, x)
        
        mean_E = I_e_exc.mean().item() + I_i_exc.mean().item()
        mean_I = I_e_inh.mean().item() + I_i_inh.mean().item()
        
        balance_ratio = abs(mean_E - mean_I) / (abs(mean_E) + abs(mean_I) + 1e-6)
        
        all_rates = torch.cat([r_e, r_i], dim=-1)
        cv_rates = (all_rates.std(dim=-1) / (all_rates.mean(dim=-1) + 1e-6)).mean().item()
        
        return {
            'mean_E_current': mean_E,
            'mean_I_current': mean_I,
            'balance_ratio': balance_ratio,
            'cv_rates': cv_rates,
            'mean_r_e': r_e.mean().item(),
            'mean_r_i': r_i.mean().item(),
        }


# =============================================================================
# Analysis Utilities
# =============================================================================

def compute_ei_weight_matrix(model: BalancedRateNetwork) -> torch.Tensor:
    """
    Get the full effective weight matrix with E/I structure.
    
    Returns block matrix:
    [ W_EE  -W_EI ]
    [ W_IE  -W_II ]
    """
    W_ee, W_ie, W_ei, W_ii = model.cell._get_weights()
    
    # Build block matrix
    top_row = torch.cat([W_ee, -W_ei], dim=1)
    bottom_row = torch.cat([W_ie, -W_ii], dim=1)
    W_full = torch.cat([top_row, bottom_row], dim=0)
    
    return W_full.detach()


def analyze_connectivity(model: BalancedRateNetwork) -> dict:
    """
    Analyze connectivity statistics.
    """
    W_ee, W_ie, W_ei, W_ii = model.cell._get_weights()
    
    return {
        'W_EE_mean': W_ee.mean().item(),
        'W_EE_std': W_ee.std().item(),
        'W_EI_mean': W_ei.mean().item(),
        'W_EI_std': W_ei.std().item(),
        'W_IE_mean': W_ie.mean().item(),
        'W_IE_std': W_ie.std().item(),
        'W_II_mean': W_ii.mean().item(),
        'W_II_std': W_ii.std().item(),
    }


if __name__ == "__main__":
    # Quick test
    print("Testing Balanced E/I Rate Network...")
    
    # Create model
    model = BalancedRateNetwork(
        input_size=3,
        n_excitatory=80,
        n_inhibitory=20,
        output_size=3,
        tau_e=10.0,
        tau_i=5.0,
        g=1.5
    )
    
    # Test forward pass
    batch_size = 8
    seq_length = 50
    x = torch.randn(batch_size, seq_length, 3)
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with dynamics
    r_e_traj, r_i_traj, output = model(x, return_dynamics=True)
    print(f"E trajectory shape: {r_e_traj.shape}")
    print(f"I trajectory shape: {r_i_traj.shape}")
    
    # Compute balance metrics
    metrics = model.compute_balance_metrics(r_e_traj[:, -1], r_i_traj[:, -1])
    print(f"Balance metrics: {metrics}")
    
    # Test generation
    initial = torch.randn(batch_size, 3)
    out_traj, r_e_gen, r_i_gen = model.generate(initial, n_steps=100)
    print(f"Generated trajectory shape: {out_traj.shape}")
    
    print("All tests passed!")
