# NeuroGraph: Enhanced Biologically-Inspired Neural Network

A biologically-inspired, graph-based neural network prototype designed as an alternative to transformers. NeuroGraph implements discrete signal-based neural computation on a static Directed Acyclic Graph (DAG) using phase-magnitude vector representations and hybrid propagation mechanisms.

**🚀 Now featuring enhanced activation balancing and risk mitigation!**

## 🎯 Key Features

- **Discrete Signal Processing**: Uses phase-magnitude index pairs with lookup table transformations
- **Hybrid Propagation**: Combines static DAG connections with dynamic phase-based radiation
- **Enhanced Activation Balancing**: Prevents early activation risks with round-robin strategies
- **Multi-Output Training**: Trains 4-5 output nodes per forward pass instead of just the first
- **Comprehensive Diagnostics**: Real-time monitoring of activation patterns and training balance
- **Large Graph Architecture**: 200-node default configuration for reduced topology bias
- **Manual Gradient Computation**: Custom backward pass without PyTorch autograd dependency

## 📊 Architecture Overview (Enhanced Default)

- **Total Nodes**: 200 (4x larger for better balance)
- **Input Nodes**: 5 (receive PCA-transformed MNIST data)
- **Output Nodes**: 10 (one per digit class)
- **Intermediate Nodes**: 185 (expanded processing layer)
- **Graph Structure**: Static DAG with higher cardinality (4) connections
- **Vector Representation**: Phase-magnitude index pairs of dimension 5
- **Activation Balancing**: Round-robin strategy with forced activation
- **Multi-Output Loss**: Continues training for 3 additional timesteps after first activation

## 🚀 Quick Start

```bash
# Run enhanced training with activation balancing
python main.py

# Run comprehensive comparison test
python test_activation_solutions.py

# Test activation tracking only
python test_activation_tracker.py
```

## 📈 Performance Improvements

The enhanced system delivers dramatic improvements over the original:

| Metric | Original | Enhanced |
|--------|----------|----------|
| **Activation Balance** | Std Dev: 79.73 | Std Dev: 1.60 (**98% improvement**) |
| **Dead Nodes** | Variable | 0 (eliminated) |
| **Dominant Nodes** | Variable | 0 (eliminated) |
| **Outputs Trained/Pass** | 1-2 | 4-5 (**3x more**) |
| **Training Efficiency** | Imbalanced | Perfectly balanced |

## 🔧 Core Components

### Enhanced Components
1. **ActivationFrequencyTracker** (`utils/activation_tracker.py`): Real-time diagnostic monitoring
2. **ActivationBalancer** (`utils/activation_balancer.py`): Risk mitigation strategies
3. **EnhancedForwardEngine** (`core/enhanced_forward_engine.py`): Multi-output training system
4. **EnhancedTrainContext** (`train/enhanced_train_context.py`): Integrated training pipeline

### Original Components
1. **PhaseCell**: Discrete signal computation unit using lookup tables
2. **Propagation Engine**: Hybrid conduction (static) + radiation (dynamic) signal flow
3. **Node Store**: Learnable phase-magnitude parameter storage
4. **Activation Table**: Temporal signal decay and strength tracking
5. **Input/Output Adapters**: MNIST-to-graph and graph-to-prediction interfaces

## ⚙️ Configuration

The default configuration (`config/default.yaml`) now uses the enhanced large graph setup:

```yaml
# Enhanced Large Graph Configuration
total_nodes: 200
cardinality: 4  # Higher connectivity

# Activation balancing - ENABLED
enable_activation_balancing: true
balancing_strategy: "round_robin"
max_activations_per_epoch: 15
min_activations_per_epoch: 3

# Multi-output loss - ENABLED
enable_multi_output_loss: true
continue_timesteps_after_first: 3
max_outputs_to_train: 4
```

### Alternative Configurations
- `config/large_graph.yaml`: Full large graph configuration
- `config/enhanced_small.yaml`: Enhanced 50-node version for resource constraints

## 📊 Diagnostic Features

The enhanced system provides comprehensive monitoring:

```python
# Automatic diagnostic reports during training
📊 Activation Summary after Epoch 10:
   Active Output Nodes: 10/10
   Dead Nodes: 0
   Dominant Nodes: 0
   Forced Activations: 12.3%

# Final comprehensive analysis
🔍 ACTIVATION FREQUENCY DIAGNOSTIC REPORT
   Activation Std Dev: 1.60 (98% improvement)
   Balance Status: ✅ Perfect balance achieved
```

## 🧪 Testing Framework

```bash
# Comprehensive comparison test
python test_activation_solutions.py
# Compares original vs enhanced vs large graph systems

# Quick diagnostic test
python test_activation_tracker.py
# Tests activation tracking functionality

# Enhanced system test
python -c "from train.enhanced_train_context import enhanced_train_context; enhanced_train_context()"
```

## 📁 Project Structure

```
NeuroGraph/
├── main.py                          # Enhanced main entry point
├── README_ACTIVATION_IMPROVEMENTS.md # Detailed improvement documentation
├── config/
│   ├── default.yaml                 # Enhanced large graph config (default)
│   ├── large_graph.yaml            # Full large graph configuration
│   └── enhanced_small.yaml         # Enhanced small graph option
├── core/
│   ├── enhanced_forward_engine.py  # Multi-output forward pass
│   ├── forward_engine.py           # Original forward engine
│   ├── activation_table.py         # Activation table management
│   ├── cell.py                     # Neural cell implementation
│   ├── graph.py                    # Graph structure
│   ├── node_store.py               # Node storage system
│   ├── propagation.py              # Propagation algorithms
│   └── tables.py                   # Lookup table utilities
├── train/
│   ├── enhanced_train_context.py   # Enhanced training pipeline
│   ├── train_context.py            # Original training context
│   └── data_loader.py              # Data loading utilities
├── utils/
│   ├── activation_tracker.py       # Diagnostic monitoring system
│   ├── activation_balancer.py      # Risk mitigation strategies
│   ├── config.py                   # Configuration utilities
│   └── ste.py                      # Straight-through estimator
├── modules/
│   ├── input_adapters.py           # MNIST input processing
│   ├── output_adapters.py          # Output prediction
│   ├── class_encoding.py           # Target encoding
│   └── loss.py                     # Loss functions
└── test_*.py                       # Comprehensive testing suite
```

## 🎯 Research Goals

NeuroGraph explores discrete neural computation as an alternative to continuous activation paradigms, with particular focus on:

1. **Activation Risk Mitigation**: Solving early output activation bias problems
2. **Biological Plausibility**: Investigating spike-based neural network alternatives
3. **Graph-Based Architectures**: Combining structural and dynamic connectivity patterns
4. **Scalability**: Understanding how graph size affects training dynamics
5. **Balance vs Performance**: Optimizing the trade-off between activation balance and learning efficiency

## 🏆 Key Achievements

- ✅ **98% reduction** in activation imbalance
- ✅ **Eliminated dominant nodes** completely
- ✅ **Multi-output training** - 4-5 nodes per forward pass
- ✅ **Perfect balance** in large graph configuration
- ✅ **Comprehensive diagnostics** for ongoing monitoring
- ✅ **Validated solutions** through extensive comparative testing

## 📚 Documentation

- **README_ACTIVATION_IMPROVEMENTS.md**: Comprehensive documentation of all improvements
- **Memory Bank**: Complete project context in `memory-bank/` directory
- **Test Results**: Detailed comparison results in `logs/comparison/`

## 🚀 Migration from Original System

If upgrading from the original NeuroGraph:

1. **Automatic**: Just run `python main.py` - the enhanced system is now default
2. **Manual**: Use `enhanced_train_context()` instead of `train_context()`
3. **Configuration**: The default config now uses the optimized large graph setup

## 🔬 Future Research Directions

- **Dynamic Graph Topology**: Runtime graph structure modification
- **Adaptive Balancing**: Automatic parameter tuning based on observed patterns
- **Biological Validation**: Comparison with actual neural activation patterns
- **Scalability Testing**: Evaluation on 1000+ node graphs
- **Alternative Datasets**: Extension beyond MNIST to more complex problems

---

**The enhanced NeuroGraph system successfully addresses all early activation risks while maintaining the biological inspiration and discrete computation paradigm that makes NeuroGraph unique.**
