"""
Modular Configuration System for NeuroGraph
Supports both modular and legacy configurations with fallback mechanisms.
"""

import yaml
import os
from typing import Dict, Any, Optional
import torch

class ModularConfig:
    """Enhanced configuration loader with modular support and fallbacks."""
    
    def __init__(self, config_path: str = "config/modular_neurograph.yaml"):
        """
        Initialize modular configuration system.
        
        Args:
            config_path: Path to primary configuration file
        """
        self.config_path = config_path
        self.config = {}
        self.mode = "modular"
        self.fallback_enabled = True
        
        self.load_config()
        self.validate_config()
        self.setup_derived_parameters()
    
    def load_config(self):
        """Load configuration with fallback support."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            self.mode = self.config.get('system', {}).get('mode', 'modular')
            self.fallback_enabled = self.config.get('fallback', {}).get('enable_legacy_mode', True)
            
            print(f"✅ Loaded {self.mode} configuration from {self.config_path}")
            
        except FileNotFoundError:
            if self.fallback_enabled:
                print(f"⚠️  Primary config not found, falling back to legacy...")
                self.load_fallback_config()
            else:
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except Exception as e:
            if self.fallback_enabled and self.config.get('fallback', {}).get('auto_fallback_on_error', True):
                print(f"⚠️  Config error: {e}, falling back to legacy...")
                self.load_fallback_config()
            else:
                raise
    
    def load_fallback_config(self):
        """Load legacy configuration as fallback."""
        fallback_path = "config/default.yaml"
        try:
            with open(fallback_path, 'r') as f:
                legacy_config = yaml.safe_load(f)
            
            # Convert legacy config to modular format
            self.config = self.convert_legacy_to_modular(legacy_config)
            self.mode = "legacy"
            print(f"✅ Loaded legacy configuration from {fallback_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load fallback configuration: {e}")
    
    def convert_legacy_to_modular(self, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert legacy configuration format to modular format."""
        modular_config = {
            'system': {
                'mode': 'legacy',
                'version': '1.0',
                'description': 'Legacy configuration converted to modular format'
            },
            'architecture': {
                'total_nodes': legacy_config.get('total_nodes', 50),
                'input_nodes': legacy_config.get('num_input_nodes', 5),
                'output_nodes': legacy_config.get('num_output_nodes', 10),
                'vector_dim': legacy_config.get('vector_dim', 5),
                'seed': legacy_config.get('seed', 42)
            },
            'resolution': {
                'phase_bins': legacy_config.get('phase_bins', 8),
                'mag_bins': legacy_config.get('mag_bins', 256),
                'resolution_increase': 1
            },
            'graph_structure': {
                'cardinality': legacy_config.get('cardinality', 3),
                'top_k_neighbors': legacy_config.get('top_k_neighbors', 4),
                'use_radiation': legacy_config.get('use_radiation', True)
            },
            'input_processing': {
                'adapter_type': 'pca',
                'input_dim': 784,
                'learnable': False,
                'normalization': None,
                'dropout': 0.0
            },
            'class_encoding': {
                'type': 'random',
                'num_classes': 10,
                'encoding_dim': legacy_config.get('vector_dim', 5),
                'orthogonality_threshold': 0.5
            },
            'loss_function': {
                'type': 'mse',
                'temperature': 1.0,
                'label_smoothing': 0.0
            },
            'training': {
                'gradient_accumulation': {
                    'enabled': False,
                    'accumulation_steps': 1,
                    'lr_scaling': 'none',
                    'buffer_size': 100
                },
                'optimizer': {
                    'type': 'discrete_sgd',
                    'base_learning_rate': legacy_config.get('learning_rate', 0.001),
                    'warmup_epochs': legacy_config.get('warmup_epochs', 5),
                    'num_epochs': legacy_config.get('num_epochs', 50),
                    'batch_size': legacy_config.get('batch_size', 5)
                }
            },
            'forward_pass': {
                'max_timesteps': legacy_config.get('max_timesteps', 6),
                'decay_factor': legacy_config.get('decay_factor', 0.925),
                'min_activation_strength': legacy_config.get('min_activation_strength', 0.01),
                'min_output_activation_timesteps': legacy_config.get('min_output_activation_timesteps', 3)
            },
            'paths': {
                'graph_path': legacy_config.get('graph_path', 'config/static_graph.pkl'),
                'log_path': legacy_config.get('log_path', 'logs/'),
                'checkpoint_path': 'checkpoints/legacy/'
            },
            'fallback': {
                'enable_legacy_mode': True,
                'legacy_config_path': 'config/default.yaml',
                'auto_fallback_on_error': True
            }
        }
        
        return modular_config
    
    def validate_config(self):
        """Validate configuration parameters."""
        arch = self.config.get('architecture', {})
        resolution = self.config.get('resolution', {})
        
        # Validate architecture
        total_nodes = arch.get('total_nodes', 0)
        input_nodes = arch.get('input_nodes', 0)
        output_nodes = arch.get('output_nodes', 0)
        
        if total_nodes != input_nodes + output_nodes + arch.get('intermediate_nodes', 0):
            print(f"⚠️  Node count mismatch: {total_nodes} != {input_nodes} + {output_nodes} + {arch.get('intermediate_nodes', 0)}")
        
        # Validate resolution
        phase_bins = resolution.get('phase_bins', 8)
        mag_bins = resolution.get('mag_bins', 256)
        
        if phase_bins < 8 or mag_bins < 256:
            print(f"⚠️  Low resolution detected: {phase_bins} phase, {mag_bins} magnitude bins")
        
        # Validate gradient accumulation
        training = self.config.get('training', {})
        grad_accum = training.get('gradient_accumulation', {})
        
        if grad_accum.get('enabled', False):
            steps = grad_accum.get('accumulation_steps', 1)
            if steps < 2:
                print(f"⚠️  Gradient accumulation enabled but steps = {steps}")
    
    def setup_derived_parameters(self):
        """Setup derived parameters based on configuration."""
        arch = self.config['architecture']
        
        # Calculate intermediate nodes if not specified
        if 'intermediate_nodes' not in arch:
            arch['intermediate_nodes'] = (
                arch['total_nodes'] - 
                arch['input_nodes'] - 
                arch['output_nodes']
            )
        
        # Setup device
        self.config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Setup learning rate scaling
        training = self.config.get('training', {})
        grad_accum = training.get('gradient_accumulation', {})
        
        if grad_accum.get('enabled', False):
            base_lr = training['optimizer']['base_learning_rate']
            steps = grad_accum['accumulation_steps']
            scaling = grad_accum.get('lr_scaling', 'none')
            
            if scaling == 'sqrt':
                scaled_lr = base_lr * (steps ** 0.5)
            elif scaling == 'linear':
                scaled_lr = base_lr * steps
            else:
                scaled_lr = base_lr
            
            training['optimizer']['effective_learning_rate'] = scaled_lr
            
            print(f"📊 Learning rate scaling: {base_lr:.4f} → {scaled_lr:.4f} (√{steps} = {steps**0.5:.2f}x)")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., 'architecture.total_nodes')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path
            value: Value to set
        """
        keys = key_path.split('.')
        config_ref = self.config
        
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        config_ref[keys[-1]] = value
    
    def is_modular(self) -> bool:
        """Check if running in modular mode."""
        return self.mode == "modular"
    
    def is_legacy(self) -> bool:
        """Check if running in legacy mode."""
        return self.mode == "legacy"
    
    def get_summary(self) -> str:
        """Get configuration summary."""
        arch = self.config['architecture']
        resolution = self.config['resolution']
        training = self.config['training']
        
        summary = f"""
NeuroGraph Configuration Summary ({self.mode.upper()} mode)
{'='*50}
Architecture:
  • Total nodes: {arch['total_nodes']}
  • Input nodes: {arch['input_nodes']}
  • Output nodes: {arch['output_nodes']}
  • Vector dimension: {arch['vector_dim']}

Resolution:
  • Phase bins: {resolution['phase_bins']}
  • Magnitude bins: {resolution['mag_bins']}
  • Resolution increase: {resolution.get('resolution_increase', 1)}x

Training:
  • Gradient accumulation: {training.get('gradient_accumulation', {}).get('enabled', False)}
  • Accumulation steps: {training.get('gradient_accumulation', {}).get('accumulation_steps', 1)}
  • Base learning rate: {training['optimizer']['base_learning_rate']}
  • Effective learning rate: {training['optimizer'].get('effective_learning_rate', 'N/A')}

Input Processing:
  • Adapter type: {self.config['input_processing']['adapter_type']}
  • Learnable: {self.config['input_processing']['learnable']}

Loss Function:
  • Type: {self.config['loss_function']['type']}
  • Class encoding: {self.config['class_encoding']['type']}
        """
        
        return summary.strip()
    
    def save_config(self, output_path: Optional[str] = None):
        """Save current configuration to file."""
        if output_path is None:
            output_path = self.config_path.replace('.yaml', '_saved.yaml')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        print(f"💾 Configuration saved to {output_path}")

def load_modular_config(config_path: str = "config/modular_neurograph.yaml") -> ModularConfig:
    """Convenience function to load modular configuration."""
    return ModularConfig(config_path)

# Backward compatibility
def load_config(path: str = "config/modular_neurograph.yaml") -> Dict[str, Any]:
    """Legacy function for backward compatibility."""
    config_loader = ModularConfig(path)
    return config_loader.config
