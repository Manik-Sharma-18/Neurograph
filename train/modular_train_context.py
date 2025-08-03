"""
Modular Training Context for NeuroGraph
Optimized training system with vectorized operations and performance monitoring
"""

import torch
import numpy as np
import os
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from functools import wraps

# Import modular components
from utils.modular_config import ModularConfig
from core.high_res_tables import HighResolutionLookupTables
from core.modular_cell import create_phase_cell
from modules.linear_input_adapter import create_input_adapter
from modules.orthogonal_encodings import create_class_encoder
from modules.classification_loss import create_classification_loss
from train.gradient_accumulator import create_gradient_accumulator, BatchController

# Import core components
from core.node_store import NodeStore
from core.modular_forward_engine import create_vectorized_forward_engine
from core.activation_table import create_vectorized_activation_table
from core.graph import build_static_graph

# Performance monitoring decorator
def timing_decorator(func_name: str):
    """Decorator for timing critical methods."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, '_timing_stats'):
                self._timing_stats = {}
            
            start_time = time.perf_counter()
            result = func(self, *args, **kwargs)
            elapsed = time.perf_counter() - start_time
            
            if func_name not in self._timing_stats:
                self._timing_stats[func_name] = []
            self._timing_stats[func_name].append(elapsed)
            
            return result
        return wrapper
    return decorator

class ModularTrainContext:
    """
    Comprehensive modular training context for NeuroGraph.
    
    Features:
    - Modular architecture with configurable components
    - High-resolution discrete computation (64×1024)
    - Gradient accumulation with √N scaling
    - Categorical cross-entropy loss with orthogonal encodings
    - Linear input projection (learnable)
    """
    
    def __init__(self, config_path: str = "config/production.yaml"):
        """
        Initialize modular training context.
        
        Args:
            config_path: Path to configuration file
        """
        print("🚀 Initializing Modular NeuroGraph Training Context")
        print("=" * 60)
        
        # Load configuration
        self.config = ModularConfig(config_path)
        self.device = self.config.get('device', 'cpu')
        
        print(self.config.get_summary())
        
        # Initialize components
        self.setup_core_components()
        self.setup_input_processing()
        self.setup_output_processing()
        self.setup_training_components()
        self.setup_graph_structure()
        
        # Training state
        self.current_epoch = 0
        self.training_losses = []
        self.validation_accuracies = []
        
        print(f"\n✅ Modular training context initialized")
        print(f"   🎯 Mode: {self.config.mode.upper()}")
        print(f"   📊 Total parameters: {self.count_parameters():,}")
        print(f"   💾 Memory usage: ~{self.estimate_memory_usage():.1f} MB")
    
    def setup_core_components(self):
        """Setup core computational components."""
        print(f"\n🔧 Setting up core components...")
        
        # High-resolution lookup tables
        self.lookup_tables = HighResolutionLookupTables(
            phase_bins=self.config.get('resolution.phase_bins'),
            mag_bins=self.config.get('resolution.mag_bins'),
            device=self.device
        )
        
        # Modular phase cell
        cell_type = "modular"  # Could be configurable
        self.phase_cell = create_phase_cell(
            cell_type=cell_type,
            vector_dim=self.config.get('architecture.vector_dim'),
            lookup_tables=self.lookup_tables
        )
        
        # Node parameter storage (will be initialized after graph setup)
        self.node_store = None
        
        # Vectorized activation tracking
        self.activation_table = create_vectorized_activation_table(
            max_nodes=self.config.get('architecture.total_nodes'),
            vector_dim=self.config.get('architecture.vector_dim'),
            phase_bins=self.config.get('resolution.phase_bins'),
            mag_bins=self.config.get('resolution.mag_bins'),
            config=self.config.config,
            device=self.device
        )
        
        print(f"   ✅ Core components initialized")
    
    def setup_input_processing(self):
        """Setup input processing pipeline."""
        print(f"\n🔧 Setting up input processing...")
        
        # Create input adapter based on configuration
        adapter_type = self.config.get('input_processing.adapter_type')
        
        if adapter_type == "linear_projection":
            self.input_adapter = create_input_adapter(
                adapter_type="linear",
                input_dim=self.config.get('input_processing.input_dim'),
                num_input_nodes=self.config.get('architecture.input_nodes'),
                vector_dim=self.config.get('architecture.vector_dim'),
                phase_bins=self.config.get('resolution.phase_bins'),
                mag_bins=self.config.get('resolution.mag_bins'),
                normalization=self.config.get('input_processing.normalization'),
                dropout=self.config.get('input_processing.dropout'),
                learnable=self.config.get('input_processing.learnable'),
                device=self.device
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported input adapter type: {adapter_type}. Only 'linear_projection' is supported.")
        
        # Setup node IDs
        self.input_nodes = list(range(self.config.get('architecture.input_nodes')))
        self.output_nodes = list(range(
            self.config.get('architecture.input_nodes'),
            self.config.get('architecture.input_nodes') + self.config.get('architecture.output_nodes')
        ))
        
        print(f"   ✅ Input processing initialized")
        print(f"      📊 Adapter type: {adapter_type}")
        print(f"      🎯 Input nodes: {len(self.input_nodes)}")
    
    def setup_output_processing(self):
        """Setup output processing and loss computation."""
        print(f"\n🔧 Setting up output processing...")
        
        # Class encodings
        encoding_type = self.config.get('class_encoding.type')
        self.class_encoder = create_class_encoder(
            encoding_type=encoding_type,
            num_classes=self.config.get('class_encoding.num_classes'),
            encoding_dim=self.config.get('class_encoding.encoding_dim'),
            phase_bins=self.config.get('resolution.phase_bins'),
            mag_bins=self.config.get('resolution.mag_bins'),
            orthogonality_threshold=self.config.get('class_encoding.orthogonality_threshold'),
            device=self.device
        )
        
        # Classification loss
        loss_type = self.config.get('loss_function.type')
        if loss_type == "categorical_crossentropy":
            self.loss_function = create_classification_loss(
                loss_type="standard",
                num_classes=self.config.get('class_encoding.num_classes'),
                temperature=self.config.get('loss_function.temperature'),
                label_smoothing=self.config.get('loss_function.label_smoothing')
            )
        else:
            raise ValueError(f"Unsupported loss function type: {loss_type}. Only 'categorical_crossentropy' is supported.")
        
        print(f"   ✅ Output processing initialized")
        print(f"      🎯 Class encoding: {encoding_type}")
        print(f"      📉 Loss function: {loss_type}")
    
    def setup_training_components(self):
        """Setup training-specific components."""
        print(f"\n🔧 Setting up training components...")
        
        # Gradient accumulation
        grad_config = self.config.get('training.gradient_accumulation')
        if grad_config.get('enabled'):
            self.gradient_accumulator = create_gradient_accumulator(
                accumulator_type="standard",
                accumulation_steps=grad_config.get('accumulation_steps'),
                lr_scaling=grad_config.get('lr_scaling'),
                buffer_size=grad_config.get('buffer_size'),
                device=self.device
            )
            
            self.batch_controller = BatchController(self.gradient_accumulator)
        else:
            self.gradient_accumulator = None
            self.batch_controller = None
        
        # Training parameters
        self.base_lr = self.config.get('training.optimizer.base_learning_rate')
        self.effective_lr = self.config.get('training.optimizer.effective_learning_rate', self.base_lr)
        self.num_epochs = self.config.get('training.optimizer.num_epochs')
        self.warmup_epochs = self.config.get('training.optimizer.warmup_epochs')
        
        print(f"   ✅ Training components initialized")
        print(f"      📊 Gradient accumulation: {grad_config.get('enabled')}")
        if grad_config.get('enabled'):
            print(f"      🎯 Accumulation steps: {grad_config.get('accumulation_steps')}")
            print(f"      📈 LR scaling: {self.effective_lr:.4f} (base: {self.base_lr:.4f})")
    
    def setup_graph_structure(self):
        """Setup graph topology."""
        print(f"\n🔧 Setting up graph structure...")
        
        # Generate or load graph
        graph_path = self.config.get('paths.graph_path')
        
        try:
            # Try to load existing graph
            import pickle
            with open(graph_path, 'rb') as f:
                self.graph_df = pickle.load(f)
            print(f"   📂 Loaded graph from {graph_path}")
        except FileNotFoundError:
            # Generate new graph
            self.graph_df = build_static_graph(
                total_nodes=self.config.get('architecture.total_nodes'),
                num_input_nodes=self.config.get('architecture.input_nodes'),
                num_output_nodes=self.config.get('architecture.output_nodes'),
                cardinality=self.config.get('graph_structure.cardinality'),
                seed=self.config.get('architecture.seed')
            )
            
            # Save generated graph
            os.makedirs(os.path.dirname(graph_path), exist_ok=True)
            with open(graph_path, 'wb') as f:
                pickle.dump(self.graph_df, f)
            print(f"   💾 Generated and saved graph to {graph_path}")
        
        # Initialize node store with graph
        self.node_store = NodeStore(
            graph_df=self.graph_df,
            vector_dim=self.config.get('architecture.vector_dim'),
            phase_bins=self.config.get('resolution.phase_bins'),
            mag_bins=self.config.get('resolution.mag_bins')
        ).to(self.device)
        
        # Setup vectorized forward engine
        self.forward_engine = create_vectorized_forward_engine(
            graph_df=self.graph_df,
            node_store=self.node_store,
            phase_cell=self.phase_cell,
            lookup_table=self.lookup_tables,
            config=self.config.config,
            device=self.device
        )
        
        print(f"   ✅ Graph structure initialized")
        print(f"      📊 Total nodes: {len(self.graph_df)}")
        print(f"      🔗 Average connections: {self.graph_df['input_connections'].apply(len).mean():.1f}")
    
    def forward_pass(self, input_context: Dict[int, Tuple[torch.Tensor, torch.Tensor]]) -> Dict[int, torch.Tensor]:
        """
        Perform forward pass through the network.
        
        Args:
            input_context: Input node activations
            
        Returns:
            Output node signals
        """
        # Convert input context to string format for vectorized forward engine
        string_input_context = {}
        for node_id, (phase_idx, mag_idx) in input_context.items():
            string_node_id = f"n{node_id}"
            string_input_context[string_node_id] = (phase_idx, mag_idx)
        
        # Run vectorized forward propagation
        final_activation_table = self.forward_engine.forward_pass_vectorized(string_input_context)
        
        # Extract output signals using vectorized interface
        output_signals = {}
        final_context = final_activation_table.get_active_context_dict()
        
        for node_id in self.output_nodes:
            # Convert integer node ID to string format
            string_node_id = f"n{node_id}"
            if string_node_id in final_context:
                phase_idx, mag_idx = final_context[string_node_id]
                
                if phase_idx is not None and mag_idx is not None:
                    signal = self.lookup_tables.get_signal_vector(phase_idx, mag_idx)
                    output_signals[node_id] = signal
        
        return output_signals
    
    @timing_decorator("compute_loss")
    def compute_loss(self, output_signals: Dict[int, torch.Tensor], target_label: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute categorical cross-entropy loss from output signals.
        
        Args:
            output_signals: Output node signals
            target_label: Ground truth label
            
        Returns:
            Tuple of (loss, logits)
        """
        # Compute logits using cosine similarity
        class_encodings = self.class_encoder.get_all_encodings()
        logits = self.loss_function.compute_logits_from_signals(
            output_signals, class_encodings, self.lookup_tables
        )
        
        # Compute cross-entropy loss
        target_tensor = torch.tensor(target_label, device=self.device)
        loss = self.loss_function(logits, target_tensor)
        
        return loss, logits
    
    def backward_pass(self, loss: torch.Tensor, output_signals: Dict[int, torch.Tensor]) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Custom backward pass using continuous gradient approximation.
        
        This implements the core solution to the discrete gradient problem:
        1. Compute gradients w.r.t. output signals using loss function derivatives
        2. Use continuous function derivatives (cos, sin, exp) for gradient computation
        3. Map continuous gradients back to discrete parameter updates
        
        Args:
            loss: Loss tensor (not used directly - we compute gradients manually)
            output_signals: Output node signals {node_id: signal_vector}
            
        Returns:
            Node gradients: node_id -> (phase_grad, mag_grad)
        """
        node_gradients = {}
        
        # Step 1: Compute upstream gradients from loss function
        upstream_gradients = self.compute_upstream_gradients(output_signals)
        
        # Step 2: For each output node, compute continuous gradients
        for node_id in self.output_nodes:
            if node_id in output_signals and node_id in upstream_gradients:
                # Get current discrete parameters
                string_node_id = f"n{node_id}"  # Convert to string format for node_store
                
                if string_node_id in self.node_store.phase_table:
                    current_phase_idx = self.node_store.phase_table[string_node_id]
                    current_mag_idx = self.node_store.mag_table[string_node_id]
                    upstream_grad = upstream_gradients[node_id]
                    
                    # Step 3: Compute continuous gradients using lookup tables
                    phase_grad, mag_grad = self.lookup_tables.compute_signal_gradients(
                        current_phase_idx, current_mag_idx, upstream_grad
                    )
                    
                    node_gradients[node_id] = (phase_grad, mag_grad)
                    
                    # Debug information
                    if hasattr(self, 'debug_gradients') and self.debug_gradients:
                        print(f"   Node {node_id}: phase_grad_norm={torch.norm(phase_grad):.4f}, "
                              f"mag_grad_norm={torch.norm(mag_grad):.4f}")
        
        return node_gradients
    
    def compute_upstream_gradients(self, output_signals: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Compute gradients of categorical cross-entropy loss w.r.t. output signals.
        
        The gradient of cross-entropy w.r.t. logits is: (softmax_probs - one_hot_target)
        We backpropagate this through the cosine similarity computation.
        
        Args:
            output_signals: Output node signals
            
        Returns:
            Upstream gradients for each output node
        """
        upstream_gradients = {}
        
        # Get class encodings for cosine similarity computation
        class_encodings = self.class_encoder.get_all_encodings()
        num_classes = self.config.get('class_encoding.num_classes')
        
        # For each output node, compute gradient contribution
        for node_id, signal in output_signals.items():
            # Initialize gradient accumulator
            signal_grad = torch.zeros_like(signal)
            
            # Compute gradient contribution from each class
            for class_id in range(num_classes):
                if class_id in class_encodings:
                    # Get class encoding signal
                    phase_idx, mag_idx = class_encodings[class_id]
                    class_signal = self.lookup_tables.get_signal_vector(phase_idx, mag_idx)
                    
                    # Compute cosine similarity gradient
                    # ∂cosine_sim/∂signal = (class_signal - signal * dot_product) / ||signal||
                    dot_product = torch.dot(signal, class_signal)
                    signal_norm = torch.norm(signal) + 1e-8
                    class_norm = torch.norm(class_signal) + 1e-8
                    
                    cosine_grad = (class_signal / (signal_norm * class_norm) - 
                                 signal * dot_product / (signal_norm**3 * class_norm))
                    
                    # Weight by softmax gradient (simplified - assumes uniform weighting)
                    signal_grad += cosine_grad / num_classes
            
            upstream_gradients[node_id] = signal_grad
        
        return upstream_gradients
    
    def apply_continuous_gradients(self, node_gradients: Dict[int, Tuple[torch.Tensor, torch.Tensor]]):
        """
        Apply continuous gradients as discrete parameter updates.
        
        This converts the continuous gradients to discrete index updates and applies them
        to the node store with proper modular arithmetic.
        
        Args:
            node_gradients: Computed gradients for each node
        """
        for node_id, (phase_grad, mag_grad) in node_gradients.items():
            string_node_id = f"n{node_id}"
            
            if string_node_id in self.node_store.phase_table:
                # Get current parameters
                current_phase_idx = self.node_store.phase_table[string_node_id]
                current_mag_idx = self.node_store.mag_table[string_node_id]
                
                # Convert continuous gradients to discrete updates
                phase_updates, mag_updates = self.lookup_tables.quantize_gradients_to_discrete_updates(
                    phase_grad, mag_grad, self.effective_lr
                )
                
                # Apply updates with modular arithmetic
                new_phase_idx, new_mag_idx = self.lookup_tables.apply_discrete_updates(
                    current_phase_idx, current_mag_idx, phase_updates, mag_updates
                )
                
                # Update node store
                self.node_store.phase_table[string_node_id] = new_phase_idx
                self.node_store.mag_table[string_node_id] = new_mag_idx
    
    def train_single_sample(self, sample_idx: int) -> Tuple[float, float]:
        """
        Train on a single sample.
        
        Args:
            sample_idx: Sample index
            
        Returns:
            Tuple of (loss, accuracy)
        """
        # Get input context
        input_context, target_label = self.input_adapter.get_input_context(
            sample_idx, self.input_nodes
        )
        
        # Forward pass
        output_signals = self.forward_pass(input_context)
        
        if not output_signals:
            return 0.0, 0.0  # No output signals
        
        # Compute loss
        loss, logits = self.compute_loss(output_signals, target_label)
        
        # Compute accuracy
        accuracy = self.loss_function.compute_accuracy(logits, torch.tensor(target_label, device=self.device))
        
        # Backward pass
        node_gradients = self.backward_pass(loss, output_signals)
        
        # === [OPTIMIZED VECTORIZED BLOCK] ===
        # Vectorized intermediate node credit assignment using cosine alignment loss
        active_nodes = self.activation_table.get_active_nodes_at_last_timestep()
        
        if active_nodes:  # Only process if there are active nodes
            # Pre-compute and cache target vector (avoid repeated computation)
            if not hasattr(self, '_cached_target_vectors'):
                self._cached_target_vectors = {}
            
            if target_label not in self._cached_target_vectors:
                target_phase_idx, target_mag_idx = self.class_encoder.get_class_encoding(target_label)
                target_vector = self.lookup_tables.get_signal_vector(target_phase_idx, target_mag_idx)
                self._cached_target_vectors[target_label] = target_vector
            else:
                target_vector = self._cached_target_vectors[target_label]
            
            # Vectorized processing of all active nodes
            valid_nodes, node_signals, phase_indices, mag_indices = self._batch_get_node_signals(active_nodes)
            
            if len(valid_nodes) > 0:
                # Vectorized cosine similarity computation [N]
                cos_sims = torch.nn.functional.cosine_similarity(
                    node_signals, target_vector.unsqueeze(0).expand(len(valid_nodes), -1), dim=1
                )
                
                # Vectorized gradient computation [N, D]
                grad_signals = self._compute_vectorized_cosine_gradients(
                    node_signals, target_vector, cos_sims
                )
                
                # Vectorized discrete gradient computation [N, D] each
                phase_grads, mag_grads = self._compute_vectorized_discrete_gradients(
                    phase_indices, mag_indices, grad_signals
                )
                
                # Batch gradient accumulation
                if self.gradient_accumulator is not None:
                    self._batch_accumulate_gradients(valid_nodes, phase_grads, mag_grads)
        
        # Handle gradient accumulation
        if self.gradient_accumulator is not None:
            # Accumulate gradients
            update_applied = self.batch_controller.process_sample(node_gradients)
            
            if update_applied:
                # Apply accumulated updates
                nodes_updated = self.gradient_accumulator.apply_accumulated_updates(
                    self.node_store.phase_table,  # This needs to be adapted
                    self.base_lr,
                    self.config.get('resolution.phase_bins'),
                    self.config.get('resolution.mag_bins')
                )
        else:
            # Direct parameter updates
            self.apply_direct_updates(node_gradients)
        
        return loss.item(), accuracy
    
    def _batch_get_node_signals(self, active_nodes: List) -> Tuple[List, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized batch retrieval of node signals and indices.
        
        Args:
            active_nodes: List of active node IDs
            
        Returns:
            Tuple of (valid_nodes, node_signals, phase_indices, mag_indices)
        """
        valid_nodes = []
        phase_indices_list = []
        mag_indices_list = []
        
        for node_id in active_nodes:
            # Convert node_id to string format for node_store lookup
            string_node_id = f"n{node_id}" if isinstance(node_id, int) else node_id
            
            # Skip if node is not in node_store
            if string_node_id not in self.node_store.phase_table:
                continue
            
            valid_nodes.append(node_id)
            phase_indices_list.append(self.node_store.get_phase(string_node_id))
            mag_indices_list.append(self.node_store.get_mag(string_node_id))
        
        if not valid_nodes:
            return [], torch.empty(0), torch.empty(0), torch.empty(0)
        
        # Stack into tensors [N, D]
        phase_indices = torch.stack(phase_indices_list)
        mag_indices = torch.stack(mag_indices_list)
        
        # Vectorized signal computation [N, D]
        node_signals = self.lookup_tables.get_signal_vector(phase_indices, mag_indices)
        
        return valid_nodes, node_signals, phase_indices, mag_indices
    
    def _compute_vectorized_cosine_gradients(self, node_signals: torch.Tensor, 
                                           target_vector: torch.Tensor, 
                                           cos_sims: torch.Tensor) -> torch.Tensor:
        """
        Vectorized computation of cosine similarity gradients.
        
        Args:
            node_signals: Node signal vectors [N, D]
            target_vector: Target vector [D]
            cos_sims: Cosine similarities [N]
            
        Returns:
            Gradient signals [N, D]
        """
        # Vectorized norm computation [N] and [1]
        node_signal_norms = torch.norm(node_signals, p=2, dim=1, keepdim=True) + 1e-8  # [N, 1]
        target_norm = torch.norm(target_vector, p=2) + 1e-8  # scalar
        
        # Expand target vector for broadcasting [1, D] -> [N, D]
        target_expanded = target_vector.unsqueeze(0).expand(len(node_signals), -1)
        
        # Vectorized cosine gradient computation [N, D]
        # ∇(-cos_sim(u, v)) = -(v/||v|| - u*cos_sim/||u||) / ||u||
        cos_sims_expanded = cos_sims.unsqueeze(1)  # [N, 1]
        
        grad_signals = -(target_expanded / target_norm - 
                        node_signals * cos_sims_expanded / node_signal_norms) / node_signal_norms
        
        return grad_signals
    
    def _compute_vectorized_discrete_gradients(self, phase_indices: torch.Tensor,
                                             mag_indices: torch.Tensor,
                                             grad_signals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized computation of discrete gradients.
        
        Args:
            phase_indices: Phase indices [N, D]
            mag_indices: Magnitude indices [N, D]
            grad_signals: Gradient signals [N, D]
            
        Returns:
            Tuple of (phase_gradients, mag_gradients) [N, D] each
        """
        # Vectorized discrete gradient computation using lookup tables
        phase_grads, mag_grads = self.lookup_tables.compute_signal_gradients(
            phase_indices, mag_indices, grad_signals
        )
        
        return phase_grads, mag_grads
    
    def _batch_accumulate_gradients(self, valid_nodes: List, 
                                  phase_grads: torch.Tensor, 
                                  mag_grads: torch.Tensor):
        """
        Batch accumulation of gradients for multiple nodes.
        
        Args:
            valid_nodes: List of valid node IDs
            phase_grads: Phase gradients [N, D]
            mag_grads: Magnitude gradients [N, D]
        """
        for i, node_id in enumerate(valid_nodes):
            # Convert string node_id back to int for gradient accumulator
            if isinstance(node_id, str) and node_id.startswith('n'):
                numeric_node_id = int(node_id[1:])
            else:
                numeric_node_id = node_id
            
            # Accumulate gradients for this node
            self.gradient_accumulator.accumulate_gradients(
                node_id=numeric_node_id,
                phase_grad=phase_grads[i],
                mag_grad=mag_grads[i]
            )
    
    def apply_direct_updates(self, node_gradients: Dict[int, Tuple[torch.Tensor, torch.Tensor]]):
        """
        Apply gradients directly without accumulation using continuous gradient approximation.
        
        This method uses the new continuous gradient approach to convert gradients
        to discrete parameter updates with proper modular arithmetic.
        """
        for node_id, (phase_grad, mag_grad) in node_gradients.items():
            string_node_id = f"n{node_id}"  # Convert to string format for node_store
            
            if string_node_id in self.node_store.phase_table:
                # Get current parameters
                current_phase_idx = self.node_store.phase_table[string_node_id]
                current_mag_idx = self.node_store.mag_table[string_node_id]
                
                # Convert continuous gradients to discrete updates using lookup tables
                phase_updates, mag_updates = self.lookup_tables.quantize_gradients_to_discrete_updates(
                    phase_grad, mag_grad, self.effective_lr
                )
                
                # Apply updates with proper modular arithmetic
                new_phase_idx, new_mag_idx = self.lookup_tables.apply_discrete_updates(
                    current_phase_idx, current_mag_idx, phase_updates, mag_updates
                )
                
                # Update node store using .data to avoid Parameter type issues
                self.node_store.phase_table[string_node_id].data = new_phase_idx.detach()
                self.node_store.mag_table[string_node_id].data = new_mag_idx.detach()
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        epoch_losses = []
        epoch_accuracies = []
        
        # Get dataset size
        dataset_size = self.input_adapter.get_dataset_info()['dataset_size']
        samples_per_epoch = self.config.get('training.optimizer.batch_size', 5)
        
        # Sample indices for this epoch
        sample_indices = np.random.choice(dataset_size, samples_per_epoch, replace=False)
        
        for i, sample_idx in enumerate(sample_indices):
            loss, accuracy = self.train_single_sample(sample_idx)
            epoch_losses.append(loss)
            epoch_accuracies.append(accuracy)
            
            # Progress reporting
            if (i + 1) % max(1, samples_per_epoch // 4) == 0:
                avg_loss = np.mean(epoch_losses[-10:])
                avg_acc = np.mean(epoch_accuracies[-10:])
                print(f"   Sample {i+1}/{samples_per_epoch}: Loss={avg_loss:.4f}, Acc={avg_acc:.1%}")
        
        return np.mean(epoch_losses), np.mean(epoch_accuracies)
    
    def train(self) -> List[float]:
        """
        Full training loop.
        
        Returns:
            List of training losses
        """
        print(f"\n🎯 Starting Training ({self.num_epochs} epochs)")
        print("=" * 60)
        
        start_time = datetime.now()
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            if hasattr(self.input_adapter, 'train'):
                self.input_adapter.train()
            
            epoch_loss, epoch_accuracy = self.train_epoch()
            
            self.training_losses.append(epoch_loss)
            
            # Progress reporting
            elapsed = (datetime.now() - start_time).total_seconds()
            eta = elapsed * (self.num_epochs - epoch - 1) / (epoch + 1)
            
            print(f"Epoch {epoch+1:3d}/{self.num_epochs}: "
                  f"Loss={epoch_loss:.4f}, Acc={epoch_accuracy:.1%}, "
                  f"ETA={eta/60:.1f}min")
            
            # Validation (optional)
            if (epoch + 1) % 10 == 0:
                val_accuracy = self.evaluate_accuracy(num_samples=50)
                self.validation_accuracies.append(val_accuracy)
                print(f"   📊 Validation accuracy: {val_accuracy:.1%}")
        
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ Training completed in {total_time:.1f} seconds")
        
        return self.training_losses
    
    def evaluate_accuracy(self, num_samples: int = 100, use_batch_evaluation: bool = True) -> float:
        """
        Evaluate model accuracy with optional batch optimization.
        
        Args:
            num_samples: Number of samples to evaluate
            use_batch_evaluation: Use optimized batch evaluation engine
            
        Returns:
            Accuracy as float
        """
        # Use optimized batch evaluation if enabled
        if use_batch_evaluation and self.config.get('batch_evaluation.enabled', True):
            if not hasattr(self, '_batch_evaluator'):
                from core.batch_evaluation_engine import create_batch_evaluation_engine
                batch_size = self.config.get('batch_evaluation.batch_size', 16)
                self._batch_evaluator = create_batch_evaluation_engine(
                    self, batch_size=batch_size, device=self.device, verbose=False
                )
            
            results = self._batch_evaluator.evaluate_accuracy_batched(
                num_samples=num_samples, streaming=True
            )
            return results['accuracy']
        
        # Standard evaluation
        if hasattr(self.input_adapter, 'eval'):
            self.input_adapter.eval()
        
        correct = 0
        total = 0
        
        dataset_size = self.input_adapter.get_dataset_info()['dataset_size']
        sample_indices = np.random.choice(dataset_size, min(num_samples, dataset_size), replace=False)
        
        with torch.no_grad():
            for sample_idx in sample_indices:
                # Get input and forward pass
                input_context, target_label = self.input_adapter.get_input_context(
                    sample_idx, self.input_nodes
                )
                
                output_signals = self.forward_pass(input_context)
                
                if output_signals:
                    # Compute prediction
                    class_encodings = self.class_encoder.get_all_encodings()
                    logits = self.loss_function.compute_logits_from_signals(
                        output_signals, class_encodings, self.lookup_tables
                    )
                    
                    predicted_class = torch.argmax(logits).item()
                    
                    if predicted_class == target_label:
                        correct += 1
                
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        total = 0
        
        # Input adapter parameters
        if hasattr(self.input_adapter, 'parameters'):
            total += sum(p.numel() for p in self.input_adapter.parameters() if p.requires_grad)
        
        # Node store parameters (discrete indices, not trainable in traditional sense)
        if self.node_store is not None:
            total += sum(p.numel() for p in self.node_store.parameters())
        
        return total
    
    def estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        memory = 0.0
        
        # Lookup tables
        memory += self.lookup_tables.estimate_memory_usage()
        
        # Node store
        if self.node_store is not None:
            for p in self.node_store.parameters():
                memory += p.numel() * 4 / (1024 * 1024)  # 4 bytes per int64
        
        # Input adapter
        if hasattr(self.input_adapter, 'parameters'):
            for p in self.input_adapter.parameters():
                memory += p.numel() * 4 / (1024 * 1024)  # 4 bytes per float32
        
        return memory
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        checkpoint = {
            'config': self.config.config,
            'current_epoch': self.current_epoch,
            'training_losses': self.training_losses,
            'validation_accuracies': self.validation_accuracies,
            'node_store_state': self.node_store.get_state(),
            'input_adapter_state': self.input_adapter.state_dict() if hasattr(self.input_adapter, 'state_dict') else None,
            'class_encodings': self.class_encoder.get_all_encodings()
        }
        
        if self.gradient_accumulator is not None:
            checkpoint['gradient_accumulator_state'] = self.gradient_accumulator.get_statistics()
        
        torch.save(checkpoint, filepath)
        print(f"💾 Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.current_epoch = checkpoint['current_epoch']
        self.training_losses = checkpoint['training_losses']
        self.validation_accuracies = checkpoint['validation_accuracies']
        
        # Restore node store
        self.node_store.load_state(checkpoint['node_store_state'])
        
        # Restore input adapter
        if checkpoint['input_adapter_state'] is not None and hasattr(self.input_adapter, 'load_state_dict'):
            self.input_adapter.load_state_dict(checkpoint['input_adapter_state'])
        
        print(f"📂 Checkpoint loaded from {filepath}")
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get comprehensive performance statistics.
        
        Returns:
            Dictionary with timing statistics for each monitored method
        """
        if not hasattr(self, '_timing_stats'):
            return {}
        
        stats = {}
        for method_name, timings in self._timing_stats.items():
            if timings:
                stats[method_name] = {
                    'count': len(timings),
                    'total_time': sum(timings),
                    'avg_time': np.mean(timings),
                    'min_time': min(timings),
                    'max_time': max(timings),
                    'std_time': np.std(timings) if len(timings) > 1 else 0.0
                }
        
        return stats
    
    def print_performance_report(self):
        """Print detailed performance report."""
        stats = self.get_performance_stats()
        
        if not stats:
            print("📊 No performance data available")
            return
        
        print("\n📊 Performance Report")
        print("=" * 60)
        
        total_calls = sum(s['count'] for s in stats.values())
        total_time = sum(s['total_time'] for s in stats.values())
        
        print(f"Total method calls: {total_calls:,}")
        print(f"Total execution time: {total_time:.3f}s")
        print(f"Average time per call: {total_time/total_calls*1000:.2f}ms")
        
        print("\nMethod Breakdown:")
        print("-" * 60)
        
        # Sort by total time (descending)
        sorted_methods = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        for method_name, method_stats in sorted_methods:
            pct_time = (method_stats['total_time'] / total_time) * 100
            print(f"{method_name:25s} | "
                  f"Calls: {method_stats['count']:6,} | "
                  f"Total: {method_stats['total_time']:7.3f}s ({pct_time:5.1f}%) | "
                  f"Avg: {method_stats['avg_time']*1000:6.2f}ms | "
                  f"Min: {method_stats['min_time']*1000:6.2f}ms | "
                  f"Max: {method_stats['max_time']*1000:6.2f}ms")
    
    def reset_performance_stats(self):
        """Reset all performance statistics."""
        if hasattr(self, '_timing_stats'):
            self._timing_stats.clear()
        print("🔄 Performance statistics reset")
    
    def get_cache_performance(self) -> Dict[str, any]:
        """
        Get cache performance statistics from radiation system.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_stats = {}
        
        # Get radiation cache stats if available
        if hasattr(self.forward_engine, 'radiation_system'):
            radiation_system = self.forward_engine.radiation_system
            if hasattr(radiation_system, 'get_cache_performance'):
                cache_stats['radiation_cache'] = radiation_system.get_cache_performance()
        
        # Get lookup table cache stats if available
        if hasattr(self.lookup_tables, 'get_cache_stats'):
            cache_stats['lookup_tables'] = self.lookup_tables.get_cache_stats()
        
        return cache_stats
    
    def print_cache_report(self):
        """Print cache performance report."""
        cache_stats = self.get_cache_performance()
        
        if not cache_stats:
            print("📊 No cache performance data available")
            return
        
        print("\n🗄️  Cache Performance Report")
        print("=" * 60)
        
        for cache_name, stats in cache_stats.items():
            print(f"\n{cache_name.upper()}:")
            print("-" * 30)
            
            if isinstance(stats, dict):
                for key, value in stats.items():
                    if isinstance(value, float):
                        if 'rate' in key.lower() or 'ratio' in key.lower():
                            print(f"  {key:20s}: {value:.1%}")
                        else:
                            print(f"  {key:20s}: {value:.3f}")
                    else:
                        print(f"  {key:20s}: {value}")
            else:
                print(f"  Status: {stats}")

# Factory function
def create_modular_train_context(config_path: str = "config/production.yaml") -> ModularTrainContext:
    """Create modular training context."""
    return ModularTrainContext(config_path)
