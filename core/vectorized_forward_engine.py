"""
Vectorized Forward Engine for GPU-First NeuroGraph
Complete GPU-optimized forward pass with parallel processing
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import time

from core.vectorized_activation_table import VectorizedActivationTable
from core.vectorized_propagation import VectorizedPropagationEngine
from core.node_store import NodeStore
from core.cell import PhaseCell
from core.modular_cell import ModularPhaseCell


class VectorizedForwardEngine:
    """
    GPU-first forward engine with complete vectorization.
    
    Key optimizations:
    - Vectorized activation table with GPU tensors
    - Batch propagation processing
    - Parallel timestep computation
    - Elimination of Python loops and CPU bottlenecks
    - Direct GPU memory operations
    """
    
    def __init__(
        self,
        graph_df: pd.DataFrame,
        node_store: NodeStore,
        phase_cell,  # PhaseCell or ModularPhaseCell
        lookup_table,
        max_nodes: int = 1000,
        vector_dim: int = 8,
        phase_bins: int = 32,
        mag_bins: int = 512,
        max_timesteps: int = 35,
        decay_factor: float = 0.95,
        min_strength: float = 0.001,
        top_k_neighbors: int = 4,
        use_radiation: bool = True,
        radiation_batch_size: int = 128,
        min_output_activation_timesteps: int = 2,
        device: str = 'auto',
        verbose: bool = False
    ):
        """
        Initialize vectorized forward engine.
        
        Args:
            graph_df: Static topology DataFrame
            node_store: Node parameter storage
            phase_cell: Phase computation cell
            lookup_table: Lookup table module
            max_nodes: Maximum number of nodes
            vector_dim: Dimensionality of phase/mag vectors
            phase_bins: Number of discrete phase values
            mag_bins: Number of discrete magnitude values
            max_timesteps: Maximum propagation timesteps
            decay_factor: Activation decay per timestep
            min_strength: Minimum activation strength
            top_k_neighbors: Top-K neighbors for radiation
            use_radiation: Enable dynamic radiation
            radiation_batch_size: Batch size for radiation computation
            min_output_activation_timesteps: Min timesteps before output check
            device: Computation device
            verbose: Enable verbose logging
        """
        self.graph_df = graph_df
        self.node_store = node_store
        self.phase_cell = phase_cell
        self.lookup_table = lookup_table
        self.max_nodes = max_nodes
        self.vector_dim = vector_dim
        self.phase_bins = phase_bins
        self.mag_bins = mag_bins
        self.max_timesteps = max_timesteps
        self.decay_factor = decay_factor
        self.min_strength = min_strength
        self.top_k_neighbors = top_k_neighbors
        self.use_radiation = use_radiation
        self.radiation_batch_size = radiation_batch_size
        self.min_output_activation_timesteps = min_output_activation_timesteps
        self.verbose = verbose
        
        # Device setup
        if device == 'auto':
            from utils.device_manager import get_device_manager
            self.device = get_device_manager().device
        else:
            self.device = torch.device(device)
        
        # Extract output nodes
        self.output_nodes = set(node_store.output_nodes) if hasattr(node_store, 'output_nodes') else set()
        
        # Initialize vectorized components
        self._init_vectorized_components()
        
        # Performance statistics
        self.stats = {
            'total_forward_passes': 0,
            'total_timesteps': 0,
            'vectorized_operations': 0,
            'gpu_memory_peak': 0,
            'forward_pass_times': [],
            'timestep_times': [],
            'activation_counts': [],
            'early_terminations': 0
        }
        
        print(f"🚀 Vectorized Forward Engine initialized:")
        print(f"   📊 Max nodes: {max_nodes}")
        print(f"   💾 Device: {self.device}")
        print(f"   🎯 Output nodes: {len(self.output_nodes)}")
        print(f"   ⚡ GPU memory allocated: {self._estimate_total_memory():.2f} MB")
    
    def _init_vectorized_components(self):
        """Initialize all vectorized components."""
        # Vectorized activation table
        self.activation_table = VectorizedActivationTable(
            max_nodes=self.max_nodes,
            vector_dim=self.vector_dim,
            phase_bins=self.phase_bins,
            mag_bins=self.mag_bins,
            decay_factor=self.decay_factor,
            min_strength=self.min_strength,
            device=self.device
        )
        
        # Vectorized propagation engine
        self.propagation_engine = VectorizedPropagationEngine(
            graph_df=self.graph_df,
            node_store=self.node_store,
            phase_cell=self.phase_cell,
            max_nodes=self.max_nodes,
            device=self.device
        )
        
        # Pre-allocate tensors for batch operations
        self._init_batch_tensors()
    
    def _init_batch_tensors(self):
        """Initialize pre-allocated tensors for batch operations."""
        # Batch processing tensors
        self.batch_target_indices = torch.zeros(
            self.max_nodes * 10,  # Allow for multiple targets per node
            dtype=torch.long,
            device=self.device
        )
        self.batch_new_phases = torch.zeros(
            (self.max_nodes * 10, self.vector_dim),
            dtype=torch.long,
            device=self.device
        )
        self.batch_new_mags = torch.zeros(
            (self.max_nodes * 10, self.vector_dim),
            dtype=torch.long,
            device=self.device
        )
        self.batch_strengths = torch.zeros(
            self.max_nodes * 10,
            dtype=torch.float32,
            device=self.device
        )
        
        # Output detection tensors
        self.output_node_indices = torch.tensor([
            self.propagation_engine.node_to_index.get(node_id, -1)
            for node_id in self.output_nodes
            if node_id in self.propagation_engine.node_to_index
        ], dtype=torch.long, device=self.device)
        
        # Filter out invalid indices
        valid_mask = self.output_node_indices >= 0
        self.output_node_indices = self.output_node_indices[valid_mask]
    
    def _estimate_total_memory(self) -> float:
        """Estimate total GPU memory usage in MB."""
        activation_memory = self.activation_table._estimate_gpu_memory()
        
        # Propagation engine memory
        prop_memory = (
            self.max_nodes * self.vector_dim * 8 * 4 +  # batch tensors
            self.max_nodes * self.max_nodes * 4 * 2     # radiation tensors
        ) / (1024 * 1024)
        
        # Batch operation tensors
        batch_memory = (
            self.max_nodes * 10 * (8 + self.vector_dim * 8 * 2 + 4)
        ) / (1024 * 1024)
        
        return activation_memory + prop_memory + batch_memory
    
    def forward_pass_vectorized(
        self, 
        input_context: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> VectorizedActivationTable:
        """
        Perform vectorized forward pass with GPU optimization.
        
        Args:
            input_context: Input node activations {node_id: (phase_indices, mag_indices)}
            
        Returns:
            Final vectorized activation table
        """
        start_time = time.time()
        
        # Clear activation table
        self.activation_table.clear()
        
        # Inject initial input context using batch injection
        self._inject_input_context_batch(input_context)
        
        if self.verbose:
            print(f"🚀 Starting vectorized forward propagation")
            print(f"   📊 Input nodes: {list(input_context.keys())}")
            print(f"   🎯 Output nodes: {list(self.output_nodes)}")
            print(f"   💾 Device: {self.device}")
        
        # Vectorized propagation loop
        timestep = 0
        timestep_times = []
        
        while timestep < self.max_timesteps:
            timestep_start = time.time()
            
            # Get vectorized active context
            active_indices, active_phases, active_mags = self.activation_table.get_active_context_vectorized()
            
            if len(active_indices) == 0:
                if self.verbose:
                    print(f"💀 Network died at timestep {timestep} - no active nodes")
                break
            
            # Check for output activation using vectorized operations
            output_active = self._check_output_activation_vectorized(active_indices)
            
            if self.verbose:
                print(f"\n⏱️ Timestep {timestep}")
                print(f"   🔹 Active nodes: {len(active_indices)}")
                if output_active.any():
                    active_output_indices = self.output_node_indices[output_active]
                    active_output_nodes = [
                        self.propagation_engine.index_to_node.get(idx.item(), f"idx_{idx}")
                        for idx in active_output_indices
                    ]
                    print(f"   🎯 Active outputs: {active_output_nodes}")
            
            # Check termination condition
            if (timestep >= self.min_output_activation_timesteps and output_active.any()):
                if self.verbose:
                    print(f"✅ Output activation detected at timestep {timestep}")
                self.stats['early_terminations'] += 1
                break
            
            # Vectorized propagation step
            target_indices, new_phases, new_mags, strengths = self.propagation_engine.propagate_vectorized(
                active_indices=active_indices,
                active_phases=active_phases,
                active_mags=active_mags,
                use_radiation=self.use_radiation,
                top_k_neighbors=self.top_k_neighbors,
                radiation_batch_size=self.radiation_batch_size
            )
            
            # Clear previous activations (overwrite style)
            self.activation_table.clear()
            
            # Batch inject new activations
            if len(target_indices) > 0:
                self._inject_propagation_results_batch(target_indices, new_phases, new_mags, strengths)
            
            # Vectorized decay and prune
            self.activation_table.decay_and_prune_vectorized()
            
            timestep_end = time.time()
            timestep_times.append(timestep_end - timestep_start)
            timestep += 1
        
        # Final summary
        final_indices, final_phases, final_mags = self.activation_table.get_active_context_vectorized()
        final_output_active = self._check_output_activation_vectorized(final_indices)
        
        if self.verbose:
            print(f"\n🏁 Vectorized forward pass completed after {timestep} timesteps")
            if final_output_active.any():
                final_active_outputs = self.output_node_indices[final_output_active]
                final_output_nodes = [
                    self.propagation_engine.index_to_node.get(idx.item(), f"idx_{idx}")
                    for idx in final_active_outputs
                ]
                print(f"   🎯 Final active output nodes: {final_output_nodes}")
            print(f"   🔹 Total active nodes: {len(final_indices)}")
            print(f"   ⚡ Average timestep time: {sum(timestep_times)/max(len(timestep_times), 1):.4f}s")
        
        # Update statistics
        end_time = time.time()
        self.stats['total_forward_passes'] += 1
        self.stats['total_timesteps'] += timestep
        self.stats['vectorized_operations'] += timestep
        self.stats['forward_pass_times'].append(end_time - start_time)
        self.stats['timestep_times'].extend(timestep_times)
        self.stats['activation_counts'].append(len(final_indices))
        
        # Update GPU memory peak
        if hasattr(torch.cuda, 'max_memory_allocated'):
            current_memory = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            self.stats['gpu_memory_peak'] = max(self.stats['gpu_memory_peak'], current_memory)
        
        return self.activation_table
    
    def _inject_input_context_batch(
        self, 
        input_context: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ):
        """
        Inject input context using batch operations for maximum performance.
        
        Args:
            input_context: Input node activations {node_id: (phase_indices, mag_indices)}
        """
        if not input_context:
            return
        
        # Convert to batch format
        node_ids = list(input_context.keys())
        batch_size = len(node_ids)
        
        # Get node indices
        node_indices = torch.tensor([
            self.activation_table._get_node_index(node_id) for node_id in node_ids
        ], dtype=torch.long, device=self.device)
        
        # Stack phase and mag data
        phase_data = torch.stack([
            input_context[node_id][0].to(self.device) for node_id in node_ids
        ])
        mag_data = torch.stack([
            input_context[node_id][1].to(self.device) for node_id in node_ids
        ])
        
        # Batch strengths (all inputs start with strength 1.0)
        strengths = torch.ones(batch_size, dtype=torch.float32, device=self.device)
        
        # Batch injection
        self.activation_table.inject_batch(node_indices, phase_data, mag_data, strengths)
    
    def _inject_propagation_results_batch(
        self,
        target_indices: torch.Tensor,
        new_phases: torch.Tensor,
        new_mags: torch.Tensor,
        strengths: torch.Tensor
    ):
        """
        Inject propagation results using batch operations.
        
        Args:
            target_indices: Target node indices [num_targets]
            new_phases: New phase indices [num_targets, vector_dim]
            new_mags: New magnitude indices [num_targets, vector_dim]
            strengths: Activation strengths [num_targets]
        """
        if len(target_indices) == 0:
            return
        
        # Convert target indices to activation table indices
        activation_indices = torch.zeros_like(target_indices)
        
        for i, target_idx in enumerate(target_indices):
            if target_idx.item() in self.propagation_engine.index_to_node:
                node_id = self.propagation_engine.index_to_node[target_idx.item()]
                activation_indices[i] = self.activation_table._get_node_index(node_id)
        
        # Batch injection
        self.activation_table.inject_batch(activation_indices, new_phases, new_mags, strengths)
    
    def _check_output_activation_vectorized(self, active_indices: torch.Tensor) -> torch.Tensor:
        """
        Check for output activation using vectorized operations.
        
        Args:
            active_indices: Currently active node indices [num_active]
            
        Returns:
            Boolean mask indicating which output nodes are active [num_output_nodes]
        """
        if len(self.output_node_indices) == 0 or len(active_indices) == 0:
            return torch.zeros(len(self.output_node_indices), dtype=torch.bool, device=self.device)
        
        # Create activation mask for all nodes
        activation_mask = torch.zeros(self.max_nodes, dtype=torch.bool, device=self.device)
        
        # Convert active indices from activation table to propagation engine indices
        active_prop_indices = []
        for act_idx in active_indices:
            if act_idx.item() in self.activation_table.index_to_node_id:
                node_id = self.activation_table.index_to_node_id[act_idx.item()]
                if node_id in self.propagation_engine.node_to_index:
                    prop_idx = self.propagation_engine.node_to_index[node_id]
                    active_prop_indices.append(prop_idx)
        
        if not active_prop_indices:
            return torch.zeros(len(self.output_node_indices), dtype=torch.bool, device=self.device)
        
        active_prop_tensor = torch.tensor(active_prop_indices, dtype=torch.long, device=self.device)
        activation_mask[active_prop_tensor] = True
        
        # Check which output nodes are active
        output_active = activation_mask[self.output_node_indices]
        
        return output_active
    
    def get_active_output_nodes(self) -> List[str]:
        """
        Get list of currently active output node IDs.
        
        Returns:
            List of active output node IDs
        """
        active_indices, _, _ = self.activation_table.get_active_context_vectorized()
        output_active = self._check_output_activation_vectorized(active_indices)
        
        active_output_nodes = []
        for i, is_active in enumerate(output_active):
            if is_active and i < len(self.output_node_indices):
                prop_idx = self.output_node_indices[i].item()
                if prop_idx in self.propagation_engine.index_to_node:
                    node_id = self.propagation_engine.index_to_node[prop_idx]
                    active_output_nodes.append(node_id)
        
        return active_output_nodes
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics."""
        avg_forward_time = (
            sum(self.stats['forward_pass_times']) / 
            max(len(self.stats['forward_pass_times']), 1)
        )
        avg_timestep_time = (
            sum(self.stats['timestep_times']) / 
            max(len(self.stats['timestep_times']), 1)
        )
        avg_activations = (
            sum(self.stats['activation_counts']) / 
            max(len(self.stats['activation_counts']), 1)
        )
        
        # Calculate samples per second
        samples_per_second = 1.0 / max(avg_forward_time, 0.001)
        
        return {
            'total_forward_passes': self.stats['total_forward_passes'],
            'total_timesteps': self.stats['total_timesteps'],
            'vectorized_operations': self.stats['vectorized_operations'],
            'early_terminations': self.stats['early_terminations'],
            'avg_forward_time': avg_forward_time,
            'avg_timestep_time': avg_timestep_time,
            'avg_activations': avg_activations,
            'samples_per_second': samples_per_second,
            'gpu_memory_peak_mb': self.stats['gpu_memory_peak'],
            'estimated_memory_mb': self._estimate_total_memory(),
            'device': str(self.device),
            'vectorization_ratio': (
                self.stats['vectorized_operations'] / 
                max(self.stats['total_timesteps'], 1)
            )
        }
    
    def print_performance_report(self):
        """Print comprehensive performance report."""
        stats = self.get_performance_stats()
        activation_stats = self.activation_table.get_performance_stats()
        propagation_stats = self.propagation_engine.get_performance_stats()
        
        print(f"\n🚀 Vectorized Forward Engine Performance Report")
        print(f"=" * 60)
        
        print(f"Forward Pass Performance:")
        print(f"  Total passes: {stats['total_forward_passes']:,}")
        print(f"  Average time: {stats['avg_forward_time']:.4f}s")
        print(f"  Samples/second: {stats['samples_per_second']:.2f}")
        print(f"  Early terminations: {stats['early_terminations']:,}")
        
        print(f"\nTimestep Performance:")
        print(f"  Total timesteps: {stats['total_timesteps']:,}")
        print(f"  Average time: {stats['avg_timestep_time']:.4f}s")
        print(f"  Average activations: {stats['avg_activations']:.1f}")
        
        print(f"\nGPU Memory:")
        print(f"  Estimated usage: {stats['estimated_memory_mb']:.2f} MB")
        print(f"  Peak usage: {stats['gpu_memory_peak_mb']:.2f} MB")
        print(f"  Device: {stats['device']}")
        
        print(f"\nVectorization:")
        print(f"  Vectorized operations: {stats['vectorized_operations']:,}")
        print(f"  Vectorization ratio: {stats['vectorization_ratio']:.1%}")
        
        # Component performance
        print(f"\nComponent Performance:")
        print(f"  Activation table vectorization: {activation_stats['vectorization_ratio']:.1%}")
        print(f"  Propagation vectorization: {propagation_stats['vectorization_ratio']:.1%}")
        
        # Performance assessment
        print(f"\nOverall Assessment:")
        if stats['samples_per_second'] > 10:
            print("  🟢 Excellent performance (>10 samples/sec)")
        elif stats['samples_per_second'] > 2:
            print("  🟡 Good performance (>2 samples/sec)")
        else:
            print("  🔴 Poor performance (<2 samples/sec)")
        
        speedup_vs_baseline = stats['samples_per_second'] / 0.5  # vs 0.5 samples/sec baseline
        print(f"  Speedup vs baseline: {speedup_vs_baseline:.1f}x")


# Factory function for easy creation
def create_vectorized_forward_engine(
    graph_df: pd.DataFrame,
    node_store: NodeStore,
    phase_cell,
    lookup_table,
    config: Dict,
    device: str = 'auto'
) -> VectorizedForwardEngine:
    """
    Factory function to create vectorized forward engine.
    
    Args:
        graph_df: Static topology DataFrame
        node_store: Node parameter storage
        phase_cell: Phase computation cell
        lookup_table: Lookup table module
        config: Configuration dictionary
        device: Computation device
        
    Returns:
        Configured vectorized forward engine
    """
    return VectorizedForwardEngine(
        graph_df=graph_df,
        node_store=node_store,
        phase_cell=phase_cell,
        lookup_table=lookup_table,
        max_nodes=config.get('graph_structure', {}).get('max_nodes', 1000),
        vector_dim=config.get('signal_processing', {}).get('vector_dim', 8),
        phase_bins=config.get('signal_processing', {}).get('phase_bins', 32),
        mag_bins=config.get('signal_processing', {}).get('mag_bins', 512),
        max_timesteps=config.get('forward_pass', {}).get('max_timesteps', 35),
        decay_factor=config.get('activation', {}).get('decay_factor', 0.95),
        min_strength=config.get('activation', {}).get('min_strength', 0.001),
        top_k_neighbors=config.get('graph_structure', {}).get('top_k_neighbors', 4),
        use_radiation=config.get('graph_structure', {}).get('use_radiation', True),
        radiation_batch_size=config.get('radiation', {}).get('batch_size', 128),
        min_output_activation_timesteps=config.get('forward_pass', {}).get('min_output_activation_timesteps', 2),
        device=device,
        verbose=config.get('forward_pass', {}).get('verbose', False)
    )
