"""
High-Resolution Lookup Tables for Modular NeuroGraph
Supports 64 phase bins and 1024 magnitude bins (16x resolution increase)
Optimized with JIT compilation for critical operations
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple, Optional

# JIT compilation flag - can be disabled for debugging
ENABLE_JIT = True

def jit_compile_if_enabled(func):
    """Decorator to conditionally apply JIT compilation."""
    if ENABLE_JIT:
        try:
            return torch.jit.script(func)
        except Exception as e:
            print(f"Warning: JIT compilation failed for {func.__name__}: {e}")
            return func
    return func

class HighResolutionLookupTables(nn.Module):
    """
    High-resolution lookup tables for discrete phase-magnitude computation.
    
    Features:
    - 64 phase bins (8x increase from 8)
    - 1024 magnitude bins (4x increase from 256)
    - 16x total resolution increase
    - Efficient GPU/CPU computation
    - Gradient computation for discrete updates
    """
    
    def __init__(self, phase_bins: int = 64, mag_bins: int = 1024, device: str = 'cpu'):
        """
        Initialize high-resolution lookup tables.
        
        Args:
            phase_bins: Number of discrete phase bins (default: 64)
            mag_bins: Number of discrete magnitude bins (default: 1024)
            device: Computation device ('cpu' or 'cuda')
        """
        super().__init__()
        
        self.N = phase_bins
        self.M = mag_bins
        self.device = device
        
        print(f"🔧 Initializing High-Resolution Lookup Tables:")
        print(f"   📊 Phase bins: {phase_bins} (vs 8 legacy)")
        print(f"   📊 Magnitude bins: {mag_bins} (vs 256 legacy)")
        print(f"   📈 Resolution increase: {(phase_bins/8) * (mag_bins/256):.1f}x")
        print(f"   💾 Memory usage: ~{self.estimate_memory_usage():.1f} MB")
        
        # Initialize lookup tables
        self.setup_phase_tables()
        self.setup_magnitude_tables()
        self.setup_gradient_tables()
        
        # Move to device
        self.to(device)
        
        print(f"✅ High-resolution tables initialized on {device}")
    
    def setup_phase_tables(self):
        """Setup phase-related lookup tables."""
        # Phase values: [0, 2π) mapped to [0, N-1]
        phase_values = torch.linspace(0, 2 * math.pi, self.N + 1)[:-1]  # Exclude 2π
        
        # Cosine and sine tables for phase computation
        self.register_buffer('phase_cos_table', torch.cos(phase_values))
        self.register_buffer('phase_sin_table', torch.sin(phase_values))
        
        # Phase gradient table: -sin for cosine derivative
        self.register_buffer('phase_grad_table', -torch.sin(phase_values))
    
    def setup_magnitude_tables(self):
        """Setup magnitude-related lookup tables."""
        # Magnitude values: exponential mapping from [-3, 3] to [0, M-1]
        mag_range = torch.linspace(-3, 3, self.M)
        
        # Exponential table for magnitude computation
        self.register_buffer('mag_exp_table', torch.exp(mag_range))
        
        # Magnitude gradient table: exp(x) derivative is exp(x)
        self.register_buffer('mag_grad_table', torch.exp(mag_range))
        
        # Alternative: sigmoid mapping for bounded magnitudes
        self.register_buffer('mag_sigmoid_table', torch.sigmoid(mag_range))
        self.register_buffer('mag_sigmoid_grad_table', 
                           torch.sigmoid(mag_range) * (1 - torch.sigmoid(mag_range)))
    
    def setup_gradient_tables(self):
        """Setup gradient computation tables."""
        # Pre-compute common gradient patterns for efficiency
        
        # Phase gradient scaling factors
        phase_scale = 2 * math.pi / self.N
        self.register_buffer('phase_grad_scale', torch.tensor(phase_scale))
        
        # Magnitude gradient scaling factors
        mag_scale = 6.0 / self.M  # Range [-3, 3] divided by bins
        self.register_buffer('mag_grad_scale', torch.tensor(mag_scale))
    
    def lookup_phase(self, phase_indices: torch.Tensor) -> torch.Tensor:
        """
        Lookup cosine values for phase indices.
        
        Args:
            phase_indices: LongTensor of shape [...] with values in [0, N-1]
            
        Returns:
            FloatTensor of cosine values with same shape as input
        """
        # Clamp indices to valid range
        phase_indices = torch.clamp(phase_indices, 0, self.N - 1)
        return self.phase_cos_table[phase_indices]
    
    def lookup_phase_sin(self, phase_indices: torch.Tensor) -> torch.Tensor:
        """
        Lookup sine values for phase indices.
        
        Args:
            phase_indices: LongTensor of shape [...] with values in [0, N-1]
            
        Returns:
            FloatTensor of sine values with same shape as input
        """
        phase_indices = torch.clamp(phase_indices, 0, self.N - 1)
        return self.phase_sin_table[phase_indices]
    
    def lookup_magnitude(self, mag_indices: torch.Tensor, use_sigmoid: bool = False) -> torch.Tensor:
        """
        Lookup magnitude values for magnitude indices.
        
        Args:
            mag_indices: LongTensor of shape [...] with values in [0, M-1]
            use_sigmoid: If True, use sigmoid mapping instead of exponential
            
        Returns:
            FloatTensor of magnitude values with same shape as input
        """
        mag_indices = torch.clamp(mag_indices, 0, self.M - 1)
        
        if use_sigmoid:
            return self.mag_sigmoid_table[mag_indices]
        else:
            return self.mag_exp_table[mag_indices]
    
    def lookup_phase_grad(self, phase_indices: torch.Tensor) -> torch.Tensor:
        """
        Lookup phase gradients for discrete gradient computation.
        
        Args:
            phase_indices: LongTensor of shape [...] with values in [0, N-1]
            
        Returns:
            FloatTensor of phase gradients with same shape as input
        """
        phase_indices = torch.clamp(phase_indices, 0, self.N - 1)
        return self.phase_grad_table[phase_indices] * self.phase_grad_scale
    
    def lookup_magnitude_grad(self, mag_indices: torch.Tensor, use_sigmoid: bool = False) -> torch.Tensor:
        """
        Lookup magnitude gradients for discrete gradient computation.
        
        Args:
            mag_indices: LongTensor of shape [...] with values in [0, M-1]
            use_sigmoid: If True, use sigmoid gradients instead of exponential
            
        Returns:
            FloatTensor of magnitude gradients with same shape as input
        """
        mag_indices = torch.clamp(mag_indices, 0, self.M - 1)
        
        if use_sigmoid:
            return self.mag_sigmoid_grad_table[mag_indices] * self.mag_grad_scale
        else:
            return self.mag_grad_table[mag_indices] * self.mag_grad_scale
    
    @jit_compile_if_enabled
    def get_signal_vector(self, phase_indices: torch.Tensor, mag_indices: torch.Tensor, 
                         use_sigmoid: bool = False) -> torch.Tensor:
        """
        Compute signal vector from phase and magnitude indices.
        [JIT OPTIMIZED - Critical path method]
        
        Args:
            phase_indices: LongTensor of shape [D] with phase indices
            mag_indices: LongTensor of shape [D] with magnitude indices
            use_sigmoid: If True, use sigmoid magnitude mapping
            
        Returns:
            FloatTensor of shape [D] with signal values
        """
        cos_vals = self.lookup_phase(phase_indices)
        mag_vals = self.lookup_magnitude(mag_indices, use_sigmoid)
        
        return cos_vals * mag_vals
    
    def quantize_to_phase(self, continuous_phase: torch.Tensor) -> torch.Tensor:
        """
        Quantize continuous phase values to discrete indices.
        
        Args:
            continuous_phase: FloatTensor with values in [0, 2π)
            
        Returns:
            LongTensor with discrete phase indices in [0, N-1]
        """
        # Normalize to [0, 1) then scale to [0, N)
        normalized = (continuous_phase % (2 * math.pi)) / (2 * math.pi)
        indices = torch.floor(normalized * self.N).long()
        return torch.clamp(indices, 0, self.N - 1)
    
    def quantize_to_magnitude(self, continuous_mag: torch.Tensor, 
                            mag_range: Tuple[float, float] = (-3.0, 3.0)) -> torch.Tensor:
        """
        Quantize continuous magnitude values to discrete indices.
        
        Args:
            continuous_mag: FloatTensor with magnitude values
            mag_range: Tuple of (min, max) values for magnitude range
            
        Returns:
            LongTensor with discrete magnitude indices in [0, M-1]
        """
        min_mag, max_mag = mag_range
        
        # Clamp to range and normalize to [0, 1]
        clamped = torch.clamp(continuous_mag, min_mag, max_mag)
        normalized = (clamped - min_mag) / (max_mag - min_mag)
        
        # Scale to [0, M) and convert to indices
        indices = torch.floor(normalized * self.M).long()
        return torch.clamp(indices, 0, self.M - 1)
    
    def estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        # Each table entry is 4 bytes (float32)
        phase_tables = 4 * self.N * 3  # cos, sin, grad
        mag_tables = 4 * self.M * 4    # exp, sigmoid, exp_grad, sigmoid_grad
        total_bytes = phase_tables + mag_tables
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def get_resolution_info(self) -> dict:
        """Get information about resolution and capacity."""
        return {
            'phase_bins': self.N,
            'mag_bins': self.M,
            'total_combinations': self.N * self.M,
            'resolution_vs_legacy': (self.N / 8) * (self.M / 256),
            'memory_mb': self.estimate_memory_usage(),
            'phase_resolution': 2 * math.pi / self.N,
            'mag_resolution': 6.0 / self.M
        }
    
    @jit_compile_if_enabled
    def compute_signal_gradients(self, phase_indices: torch.Tensor, mag_indices: torch.Tensor, 
                               upstream_grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradients using continuous function derivatives with chain rule.
        [JIT OPTIMIZED - Critical gradient computation method]
        
        This is the core method for continuous gradient approximation:
        - Uses exact derivatives of cos(x), sin(x), exp(x)
        - Applies chain rule with upstream gradients from loss
        - Maps continuous gradients to discrete parameter updates
        
        Args:
            phase_indices: LongTensor of shape [D] with phase indices
            mag_indices: LongTensor of shape [D] with magnitude indices  
            upstream_grad: FloatTensor of shape [D] with gradients from loss
            
        Returns:
            Tuple of (phase_gradients, magnitude_gradients) as FloatTensors [D]
        """
        # Clamp indices to valid range
        phase_indices = torch.clamp(phase_indices, 0, self.N - 1)
        mag_indices = torch.clamp(mag_indices, 0, self.M - 1)
        
        # Get current function values
        cos_vals = self.lookup_phase(phase_indices)  # cos(phase)
        sin_vals = self.lookup_phase_sin(phase_indices)  # sin(phase)
        mag_vals = self.lookup_magnitude(mag_indices)  # exp(magnitude)
        
        # Compute partial derivatives using continuous functions:
        # For signal = cos(phase) * exp(magnitude):
        # ∂signal/∂phase = -sin(phase) * exp(magnitude)
        # ∂signal/∂magnitude = cos(phase) * exp(magnitude)
        
        phase_partial = -sin_vals * mag_vals  # -sin(phase) * exp(mag)
        mag_partial = cos_vals * mag_vals     # cos(phase) * exp(mag)
        
        # Apply chain rule with upstream gradients
        phase_grad = phase_partial * upstream_grad
        mag_grad = mag_partial * upstream_grad
        
        return phase_grad, mag_grad
    
    def quantize_gradients_to_discrete_updates(self, phase_grad: torch.Tensor, 
                                             mag_grad: torch.Tensor,
                                             learning_rate: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert continuous gradients to discrete parameter updates.
        
        Maps continuous gradients back to discrete index updates using:
        - Phase resolution: 2π / N_phase_bins
        - Magnitude resolution: 6.0 / N_mag_bins (range [-3, 3])
        
        Args:
            phase_grad: Continuous phase gradients [D]
            mag_grad: Continuous magnitude gradients [D]
            learning_rate: Learning rate for gradient descent
            
        Returns:
            Tuple of (discrete_phase_updates, discrete_mag_updates) as LongTensors
        """
        # Scale gradients by resolution and learning rate
        phase_resolution = 2 * math.pi / self.N
        mag_resolution = 6.0 / self.M
        
        # Convert to discrete updates (negative for gradient descent)
        discrete_phase_updates = torch.round(
            -learning_rate * phase_grad / phase_resolution
        ).long()
        
        discrete_mag_updates = torch.round(
            -learning_rate * mag_grad / mag_resolution  
        ).long()
        
        return discrete_phase_updates, discrete_mag_updates
    
    def apply_discrete_updates(self, current_phase_indices: torch.Tensor,
                             current_mag_indices: torch.Tensor,
                             phase_updates: torch.Tensor,
                             mag_updates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply discrete updates with proper modular arithmetic.
        
        Args:
            current_phase_indices: Current phase indices [D]
            current_mag_indices: Current magnitude indices [D]
            phase_updates: Phase updates to apply [D]
            mag_updates: Magnitude updates to apply [D]
            
        Returns:
            Tuple of (new_phase_indices, new_mag_indices)
        """
        # Apply updates with modular arithmetic for phase (wraps around)
        new_phase_indices = (current_phase_indices + phase_updates) % self.N
        
        # Apply updates with clamping for magnitude (bounded range)
        new_mag_indices = torch.clamp(
            current_mag_indices + mag_updates, 
            0, self.M - 1
        )
        
        return new_phase_indices, new_mag_indices
    
    def compute_loss_gradients_wrt_signals(self, predicted_signals: torch.Tensor,
                                         target_signals: torch.Tensor,
                                         loss_type: str = 'mse') -> torch.Tensor:
        """
        Compute gradients of loss function w.r.t. predicted signals.
        
        This provides the upstream gradients for the chain rule.
        
        Args:
            predicted_signals: Predicted signal vectors [D]
            target_signals: Target signal vectors [D]
            loss_type: Type of loss ('mse', 'cosine', 'l1')
            
        Returns:
            Gradients w.r.t. predicted signals [D]
        """
        if loss_type == 'mse':
            # MSE: ∂L/∂pred = 2 * (pred - target)
            return 2.0 * (predicted_signals - target_signals)
        elif loss_type == 'l1':
            # L1: ∂L/∂pred = sign(pred - target)
            return torch.sign(predicted_signals - target_signals)
        elif loss_type == 'cosine':
            # Cosine similarity loss (more complex, simplified version)
            diff = predicted_signals - target_signals
            return diff / (torch.norm(diff) + 1e-8)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, phase_indices: torch.Tensor, mag_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for compatibility with existing PhaseCell interface.
        
        Args:
            phase_indices: LongTensor of shape [D]
            mag_indices: LongTensor of shape [D]
            
        Returns:
            Tuple of (signal_vector, phase_gradients, mag_gradients)
        """
        signal = self.get_signal_vector(phase_indices, mag_indices)
        phase_grad = self.lookup_phase_grad(phase_indices)
        mag_grad = self.lookup_magnitude_grad(mag_indices)
        
        return signal, phase_grad, mag_grad

class QuantizationUtils:
    """Utility functions for high-resolution quantization."""
    
    @staticmethod
    def adaptive_quantize_phase(values: torch.Tensor, phase_bins: int = 64) -> torch.Tensor:
        """
        Adaptive phase quantization with improved distribution.
        
        Args:
            values: Continuous values to quantize
            phase_bins: Number of phase bins
            
        Returns:
            Quantized phase indices
        """
        # Normalize to [0, 1] using min-max scaling
        min_val, max_val = values.min(), values.max()
        if max_val > min_val:
            normalized = (values - min_val) / (max_val - min_val)
        else:
            normalized = torch.zeros_like(values)
        
        # Apply non-linear mapping for better distribution
        # Use sqrt for more uniform distribution in lower values
        mapped = torch.sqrt(normalized)
        
        # Quantize to bins
        indices = torch.floor(mapped * phase_bins).long()
        return torch.clamp(indices, 0, phase_bins - 1)
    
    @staticmethod
    def adaptive_quantize_magnitude(values: torch.Tensor, mag_bins: int = 1024) -> torch.Tensor:
        """
        Adaptive magnitude quantization with improved distribution.
        
        Args:
            values: Continuous values to quantize
            mag_bins: Number of magnitude bins
            
        Returns:
            Quantized magnitude indices
        """
        # Apply log scaling for magnitude values
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        log_values = torch.log(torch.abs(values) + epsilon)
        
        # Normalize log values
        min_log, max_log = log_values.min(), log_values.max()
        if max_log > min_log:
            normalized = (log_values - min_log) / (max_log - min_log)
        else:
            normalized = torch.zeros_like(log_values)
        
        # Quantize to bins
        indices = torch.floor(normalized * mag_bins).long()
        return torch.clamp(indices, 0, mag_bins - 1)

# Backward compatibility alias
ExtendedLookupTableModule = HighResolutionLookupTables
