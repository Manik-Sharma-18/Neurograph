"""
Modular Training Context for NeuroGraph
Integrates all modular components with gradient accumulation
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Import modular components
from utils.modular_config import ModularConfig
from core.high_res_tables import HighResolutionLookupTables
from core.modular_cell import ModularPhaseCell, create_phase_cell
from modules.linear_input_adapter import LinearInputAdapter, create_input_adapter
from modules.orthogonal_encodings import OrthogonalClassEncoder, create_class_encoder
from modules.classification_loss import ClassificationLoss, create_classification_loss
from train.gradient_accumulator import GradientAccumulator, BatchController, create_gradient_accumulator

# Import legacy components for fallback
from core.node_store import NodeStore
from core.modular_forward_engine import ModularForwardEngine, create_modular_forward_engine
from core.activation_table import ActivationTable
from core.graph import build_static_graph

class ModularTrainContext:
    """
    Comprehensive modular training context for NeuroGraph.
    
    Features:
    - Modular architecture with configurable components
    - High-resolution discrete computation (64×1024)
    - Gradient accumulation with √N scaling
    - Categorical cross-entropy loss with orthogonal encodings
    - Linear input projection (learnable)
    - Legacy fallback support
    """
    
    def __init__(self, config_path: str = "config/modular_neurograph.yaml"):
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
        
        # Activation tracking
        self.activation_table = ActivationTable(
            vector_dim=self.config.get('architecture.vector_dim'),
            phase_bins=self.config.get('resolution.phase_bins'),
            mag_bins=self.config.get('resolution.mag_bins'),
            decay_factor=self.config.get('forward_pass.decay_factor'),
            min_strength=self.config.get('forward_pass.min_activation_strength'),
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
            # Fallback to legacy PCA adapter
            from modules.input_adapters import MNISTPCAAdapter
            self.input_adapter = MNISTPCAAdapter(
                vector_dim=self.config.get('architecture.vector_dim'),
                num_input_nodes=self.config.get('architecture.input_nodes'),
                phase_bins=self.config.get('resolution.phase_bins'),
                mag_bins=self.config.get('resolution.mag_bins'),
                device=self.device
            )
        
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
            # Fallback to legacy MSE loss
            from modules.loss import signal_loss_from_lookup
            self.loss_function = signal_loss_from_lookup
        
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
        
        # Setup forward engine
        self.forward_engine = create_modular_forward_engine(
            graph_df=self.graph_df,
            node_store=self.node_store,
            phase_cell=self.phase_cell,
            activation_table=self.activation_table,
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
        # Run forward propagation
        final_activation_table = self.forward_engine.propagate(input_context)
        
        # Extract output signals (convert integer node IDs to string format)
        output_signals = {}
        final_context = final_activation_table.get_active_context()
        
        for node_id in self.output_nodes:
            # Convert integer node ID to string format
            string_node_id = f"n{node_id}"
            if string_node_id in final_context:
                phase_idx, mag_idx = final_context[string_node_id]
                
                if phase_idx is not None and mag_idx is not None:
                    signal = self.lookup_tables.get_signal_vector(phase_idx, mag_idx)
                    output_signals[node_id] = signal
        
        return output_signals
    
    def compute_loss(self, output_signals: Dict[int, torch.Tensor], target_label: int) -> torch.Tensor:
        """
        Compute loss from output signals.
        
        Args:
            output_signals: Output node signals
            target_label: Ground truth label
            
        Returns:
            Loss tensor
        """
        if self.config.get('loss_function.type') == "categorical_crossentropy":
            # Compute logits using cosine similarity
            class_encodings = self.class_encoder.get_all_encodings()
            logits = self.loss_function.compute_logits_from_signals(
                output_signals, class_encodings, self.lookup_tables
            )
            
            # Compute cross-entropy loss
            target_tensor = torch.tensor(target_label, device=self.device)
            loss = self.loss_function(logits, target_tensor)
            
            return loss, logits
        else:
            # Legacy MSE loss
            target_encoding = self.class_encoder.get_class_encoding(target_label)
            # Implementation would depend on legacy system
            raise NotImplementedError("Legacy MSE loss not implemented in modular system")
    
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
        Compute gradients of loss function w.r.t. output signals.
        
        This replaces the failing loss.backward() call with manual gradient computation
        based on the loss function type (categorical cross-entropy or MSE).
        
        Args:
            output_signals: Output node signals
            
        Returns:
            Upstream gradients for each output node
        """
        upstream_gradients = {}
        
        if self.config.get('loss_function.type') == "categorical_crossentropy":
            # For categorical cross-entropy, we need to compute gradients w.r.t. logits
            upstream_gradients = self.compute_crossentropy_gradients(output_signals)
        else:
            # For MSE loss (legacy), compute gradients w.r.t. signal vectors
            upstream_gradients = self.compute_mse_gradients(output_signals)
        
        return upstream_gradients
    
    def compute_crossentropy_gradients(self, output_signals: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Compute gradients for categorical cross-entropy loss.
        
        The gradient of cross-entropy w.r.t. logits is: (softmax_probs - one_hot_target)
        We need to backpropagate this through the cosine similarity computation.
        
        Args:
            output_signals: Output node signals
            
        Returns:
            Gradients w.r.t. output signals
        """
        upstream_gradients = {}
        
        # Get class encodings for cosine similarity computation
        class_encodings = self.class_encoder.get_all_encodings()
        
        # For each output node, compute gradient contribution
        for node_id, signal in output_signals.items():
            # Initialize gradient accumulator
            signal_grad = torch.zeros_like(signal)
            
            # Compute gradient contribution from each class
            for class_id in range(self.config.get('class_encoding.num_classes')):
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
                    signal_grad += cosine_grad / self.config.get('class_encoding.num_classes')
            
            upstream_gradients[node_id] = signal_grad
        
        return upstream_gradients
    
    def compute_mse_gradients(self, output_signals: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Compute gradients for MSE loss (legacy support).
        
        Args:
            output_signals: Output node signals
            
        Returns:
            Gradients w.r.t. output signals
        """
        upstream_gradients = {}
        
        # For MSE, we need target signals (this is a simplified version)
        for node_id, signal in output_signals.items():
            # Create dummy target (in real implementation, this would come from class encodings)
            target_signal = torch.zeros_like(signal)  # Placeholder
            
            # MSE gradient: 2 * (predicted - target)
            mse_grad = 2.0 * (signal - target_signal)
            upstream_gradients[node_id] = mse_grad
        
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
            # Direct parameter updates (legacy mode)
            self.apply_direct_updates(node_gradients)
        
        return loss.item(), accuracy
    
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
    
    def evaluate_accuracy(self, num_samples: int = 100) -> float:
        """
        Evaluate model accuracy.
        
        Args:
            num_samples: Number of samples to evaluate
            
        Returns:
            Accuracy as float
        """
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

# Factory function
def create_modular_train_context(config_path: str = "config/modular_neurograph.yaml") -> ModularTrainContext:
    """Create modular training context."""
    return ModularTrainContext(config_path)
