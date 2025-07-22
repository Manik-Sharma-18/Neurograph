"""
Linear Input Adapter for Modular NeuroGraph
Replaces PCA with learnable linear projection (784 → 200 nodes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from typing import Dict, Tuple, Optional
from core.high_res_tables import QuantizationUtils

class LinearInputAdapter(nn.Module):
    """
    Learnable linear projection adapter for MNIST input processing.
    
    Features:
    - Direct learnable projection: 784 → 200 nodes × 5 dims × 2 components
    - High-resolution quantization (64 phase, 1024 magnitude bins)
    - Layer normalization and dropout for regularization
    - Efficient batch processing
    - No PCA dependency
    """
    
    def __init__(self, input_dim: int = 784, num_input_nodes: int = 200, 
                 vector_dim: int = 5, phase_bins: int = 64, mag_bins: int = 1024,
                 normalization: str = "layer_norm", dropout: float = 0.1, 
                 learnable: bool = True, device: str = 'cpu'):
        """
        Initialize linear input adapter.
        
        Args:
            input_dim: Input dimension (784 for MNIST)
            num_input_nodes: Number of input nodes (200)
            vector_dim: Vector dimension per node (5)
            phase_bins: Number of discrete phase bins (64)
            mag_bins: Number of discrete magnitude bins (1024)
            normalization: Normalization type ("layer_norm", "batch_norm", or None)
            dropout: Dropout probability for regularization
            learnable: Whether projection is learnable or fixed
            device: Computation device
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_input_nodes = num_input_nodes
        self.vector_dim = vector_dim
        self.phase_bins = phase_bins
        self.mag_bins = mag_bins
        self.device = device
        self.learnable = learnable
        
        # Calculate output dimension: nodes × vector_dim × 2 (phase + magnitude)
        self.output_dim = num_input_nodes * vector_dim * 2
        
        print(f"🔧 Initializing Linear Input Adapter:")
        print(f"   📊 Input dimension: {input_dim}")
        print(f"   🎯 Output nodes: {num_input_nodes}")
        print(f"   📐 Vector dimension: {vector_dim}")
        print(f"   📈 Total output dimension: {self.output_dim}")
        print(f"   🧠 Learnable: {learnable}")
        print(f"   📊 Resolution: {phase_bins}×{mag_bins}")
        
        # Linear projection layer
        if learnable:
            self.projection = nn.Linear(input_dim, self.output_dim)
            # Initialize with Xavier/Glorot initialization
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
        else:
            # Fixed random projection
            projection_matrix = torch.randn(input_dim, self.output_dim) * 0.1
            self.register_buffer('fixed_projection', projection_matrix)
            bias = torch.zeros(self.output_dim)
            self.register_buffer('fixed_bias', bias)
        
        # Normalization layer
        if normalization == "layer_norm":
            self.norm = nn.LayerNorm(self.output_dim)
        elif normalization == "batch_norm":
            self.norm = nn.BatchNorm1d(self.output_dim)
        else:
            self.norm = nn.Identity()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Load MNIST dataset
        self.setup_dataset()
        
        # Quantization utilities
        self.quantizer = QuantizationUtils()
        
        print(f"✅ Linear input adapter initialized")
    
    def setup_dataset(self):
        """Setup MNIST dataset with proper transforms."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalization
            transforms.Lambda(lambda x: x.view(-1))  # Flatten to [784]
        ])
        
        self.mnist = datasets.MNIST(
            root="data", train=True, download=True, transform=transform
        )
        
        print(f"📚 MNIST dataset loaded: {len(self.mnist)} samples")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through linear projection.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim] or [input_dim]
            
        Returns:
            Projected tensor of shape [batch_size, output_dim] or [output_dim]
        """
        # Handle single sample input
        single_sample = x.dim() == 1
        if single_sample:
            x = x.unsqueeze(0)
        
        # Apply projection
        if self.learnable:
            projected = self.projection(x)
        else:
            projected = F.linear(x, self.fixed_projection.t(), self.fixed_bias)
        
        # Apply normalization and dropout
        projected = self.norm(projected)
        projected = self.dropout(projected)
        
        # Return to original shape if single sample
        if single_sample:
            projected = projected.squeeze(0)
        
        return projected
    
    def quantize_to_phase_mag(self, projected: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize projected values to discrete phase-magnitude indices.
        
        Args:
            projected: Projected tensor of shape [output_dim]
            
        Returns:
            Tuple of (phase_indices, mag_indices) each of shape [num_nodes, vector_dim]
        """
        # Reshape to [num_nodes, vector_dim, 2] (last dim: phase, magnitude)
        reshaped = projected.view(self.num_input_nodes, self.vector_dim, 2)
        
        # Split into phase and magnitude components
        phase_continuous = reshaped[:, :, 0]  # [num_nodes, vector_dim]
        mag_continuous = reshaped[:, :, 1]    # [num_nodes, vector_dim]
        
        # Quantize using adaptive methods
        phase_indices = self.quantizer.adaptive_quantize_phase(
            phase_continuous.flatten(), self.phase_bins
        ).view(self.num_input_nodes, self.vector_dim)
        
        mag_indices = self.quantizer.adaptive_quantize_magnitude(
            mag_continuous.flatten(), self.mag_bins
        ).view(self.num_input_nodes, self.vector_dim)
        
        return phase_indices, mag_indices
    
    def get_input_context(self, idx: int, input_node_ids: list) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], int]:
        """
        Get input context for a specific MNIST sample.
        
        Args:
            idx: Sample index
            input_node_ids: List of input node IDs
            
        Returns:
            Tuple of (input_context, label)
            input_context: dict[node_id → (phase_indices, mag_indices)]
            label: Ground truth digit class
        """
        if idx >= len(self.mnist):
            raise IndexError(f"Index {idx} out of range for dataset size {len(self.mnist)}")
        
        # Get MNIST sample
        image, label = self.mnist[idx]
        image = image.to(self.device)
        
        # Forward pass through projection
        with torch.no_grad() if not self.training else torch.enable_grad():
            projected = self.forward(image)
        
        # Quantize to phase-magnitude indices
        phase_indices, mag_indices = self.quantize_to_phase_mag(projected)
        
        # Create input context dictionary
        input_context = {}
        for i, node_id in enumerate(input_node_ids):
            if i >= self.num_input_nodes:
                break
            
            input_context[node_id] = (
                phase_indices[i],  # [vector_dim]
                mag_indices[i]     # [vector_dim]
            )
        
        return input_context, label
    
    def get_batch_input_contexts(self, indices: list, input_node_ids: list) -> Tuple[list, list]:
        """
        Get input contexts for a batch of samples.
        
        Args:
            indices: List of sample indices
            input_node_ids: List of input node IDs
            
        Returns:
            Tuple of (input_contexts, labels)
        """
        input_contexts = []
        labels = []
        
        for idx in indices:
            context, label = self.get_input_context(idx, input_node_ids)
            input_contexts.append(context)
            labels.append(label)
        
        return input_contexts, labels
    
    def get_dataset_info(self) -> Dict[str, any]:
        """Get information about the dataset and adapter."""
        return {
            'dataset_size': len(self.mnist),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_input_nodes': self.num_input_nodes,
            'vector_dim': self.vector_dim,
            'phase_bins': self.phase_bins,
            'mag_bins': self.mag_bins,
            'learnable': self.learnable,
            'parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'capacity_vs_pca': self.output_dim / 50  # vs previous PCA 50 dims
        }
    
    def get_projection_stats(self) -> Dict[str, float]:
        """Get statistics about the learned projection."""
        if not self.learnable:
            return {'message': 'Fixed projection - no learnable parameters'}
        
        weight = self.projection.weight
        bias = self.projection.bias
        
        stats = {
            'weight_mean': weight.mean().item(),
            'weight_std': weight.std().item(),
            'weight_min': weight.min().item(),
            'weight_max': weight.max().item(),
            'bias_mean': bias.mean().item(),
            'bias_std': bias.std().item(),
            'weight_norm': torch.norm(weight).item(),
            'condition_number': torch.linalg.cond(weight).item()
        }
        
        return stats
    
    def visualize_projection_patterns(self, save_path: str = "logs/modular/projection_patterns.png"):
        """Visualize learned projection patterns."""
        if not self.learnable:
            print("⚠️  Cannot visualize fixed projection patterns")
            return
        
        import matplotlib.pyplot as plt
        import os
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Get projection weights
        weights = self.projection.weight.detach().cpu().numpy()  # [output_dim, input_dim]
        
        # Reshape input weights to image format for visualization
        # Take first few output dimensions
        num_patterns = min(16, self.num_input_nodes)
        patterns_per_node = self.vector_dim * 2  # phase + magnitude
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(num_patterns):
            # Get weights for this output node (average across vector dimensions)
            start_idx = i * patterns_per_node
            end_idx = start_idx + patterns_per_node
            node_weights = weights[start_idx:end_idx].mean(axis=0)  # [input_dim]
            
            # Reshape to 28x28 image
            pattern = node_weights.reshape(28, 28)
            
            axes[i].imshow(pattern, cmap='RdBu', vmin=-pattern.std(), vmax=pattern.std())
            axes[i].set_title(f'Node {i}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Projection patterns saved to {save_path}")
    
    def save_adapter(self, filepath: str):
        """Save adapter state."""
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'num_input_nodes': self.num_input_nodes,
                'vector_dim': self.vector_dim,
                'phase_bins': self.phase_bins,
                'mag_bins': self.mag_bins,
                'learnable': self.learnable
            }
        }, filepath)
        print(f"💾 Linear adapter saved to {filepath}")
    
    def load_adapter(self, filepath: str):
        """Load adapter state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        print(f"📂 Linear adapter loaded from {filepath}")

class FixedLinearInputAdapter(LinearInputAdapter):
    """Fixed (non-learnable) linear input adapter for comparison."""
    
    def __init__(self, **kwargs):
        """Initialize with learnable=False."""
        kwargs['learnable'] = False
        super().__init__(**kwargs)

class AdaptiveLinearInputAdapter(LinearInputAdapter):
    """Adaptive linear input adapter with dynamic quantization."""
    
    def __init__(self, **kwargs):
        """Initialize adaptive adapter."""
        super().__init__(**kwargs)
        
        # Track statistics for adaptive quantization
        self.register_buffer('running_phase_mean', torch.zeros(1))
        self.register_buffer('running_phase_std', torch.ones(1))
        self.register_buffer('running_mag_mean', torch.zeros(1))
        self.register_buffer('running_mag_std', torch.ones(1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    def update_running_stats(self, phase_vals: torch.Tensor, mag_vals: torch.Tensor):
        """Update running statistics for adaptive quantization."""
        if self.training:
            momentum = 0.1
            
            # Update phase statistics
            phase_mean = phase_vals.mean()
            phase_std = phase_vals.std()
            
            self.running_phase_mean = (1 - momentum) * self.running_phase_mean + momentum * phase_mean
            self.running_phase_std = (1 - momentum) * self.running_phase_std + momentum * phase_std
            
            # Update magnitude statistics
            mag_mean = mag_vals.mean()
            mag_std = mag_vals.std()
            
            self.running_mag_mean = (1 - momentum) * self.running_mag_mean + momentum * mag_mean
            self.running_mag_std = (1 - momentum) * self.running_mag_std + momentum * mag_std
            
            self.num_batches_tracked += 1
    
    def quantize_to_phase_mag(self, projected: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adaptive quantization using running statistics."""
        # Reshape and split as before
        reshaped = projected.view(self.num_input_nodes, self.vector_dim, 2)
        phase_continuous = reshaped[:, :, 0]
        mag_continuous = reshaped[:, :, 1]
        
        # Update running statistics
        self.update_running_stats(phase_continuous, mag_continuous)
        
        # Normalize using running statistics
        if self.num_batches_tracked > 0:
            phase_normalized = (phase_continuous - self.running_phase_mean) / (self.running_phase_std + 1e-8)
            mag_normalized = (mag_continuous - self.running_mag_mean) / (self.running_mag_std + 1e-8)
        else:
            phase_normalized = phase_continuous
            mag_normalized = mag_continuous
        
        # Quantize
        phase_indices = self.quantizer.adaptive_quantize_phase(
            phase_normalized.flatten(), self.phase_bins
        ).view(self.num_input_nodes, self.vector_dim)
        
        mag_indices = self.quantizer.adaptive_quantize_magnitude(
            mag_normalized.flatten(), self.mag_bins
        ).view(self.num_input_nodes, self.vector_dim)
        
        return phase_indices, mag_indices

def create_input_adapter(adapter_type: str = "linear", **kwargs) -> LinearInputAdapter:
    """
    Factory function to create input adapters.
    
    Args:
        adapter_type: Type of adapter ("linear", "fixed", "adaptive")
        **kwargs: Additional arguments for adapter
        
    Returns:
        Input adapter instance
    """
    if adapter_type == "linear":
        return LinearInputAdapter(**kwargs)
    elif adapter_type == "fixed":
        return FixedLinearInputAdapter(**kwargs)
    elif adapter_type == "adaptive":
        return AdaptiveLinearInputAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")

# Backward compatibility alias
MNISTPCAAdapter = LinearInputAdapter
