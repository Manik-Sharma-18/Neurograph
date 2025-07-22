"""
Main Modular NeuroGraph Entry Point
Comprehensive modular system with gradient accumulation and high-resolution computation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random
import argparse

from train.modular_train_context import ModularTrainContext, create_modular_train_context
from utils.modular_config import ModularConfig

def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def plot_training_curves(losses, accuracies=None, save_path="logs/modular/training_curves.png"):
    """Plot and save training curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if accuracies:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Loss curve
    ax1.plot(losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_title('Modular NeuroGraph Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add loss improvement annotation
    if len(losses) > 1:
        improvement = losses[0] - losses[-1]
        ax1.annotate(f'Improvement: {improvement:+.4f}', 
                    xy=(len(losses)-1, losses[-1]), 
                    xytext=(len(losses)*0.7, max(losses)*0.8),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    
    # Accuracy curve (if available)
    if accuracies:
        ax2.plot(accuracies, 'g-', linewidth=2, label='Validation Accuracy')
        ax2.set_title('Modular NeuroGraph Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Validation Check', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add accuracy annotation
        if len(accuracies) > 0:
            final_acc = accuracies[-1]
            ax2.annotate(f'Final: {final_acc:.1%}', 
                        xy=(len(accuracies)-1, final_acc), 
                        xytext=(len(accuracies)*0.7, max(accuracies)*0.9),
                        arrowprops=dict(arrowstyle='->', color='green'),
                        fontsize=10, color='green')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Training curves saved to: {save_path}")

def comprehensive_evaluation(trainer, num_samples=300):
    """Perform comprehensive evaluation of the trained model."""
    print("\n🔍 Comprehensive Model Evaluation")
    print("=" * 50)
    
    # Overall accuracy
    accuracy = trainer.evaluate_accuracy(num_samples=num_samples)
    
    # Per-class analysis
    print(f"\n📊 Per-class analysis (sample size: {min(num_samples//10, 30)} per class):")
    class_correct = [0] * 10
    class_total = [0] * 10
    
    dataset_size = trainer.input_adapter.get_dataset_info()['dataset_size']
    samples_per_class = min(num_samples // 10, 30)
    
    for digit in range(10):
        # Find samples of this digit
        digit_indices = []
        for i in range(dataset_size):
            if len(digit_indices) >= samples_per_class:
                break
            try:
                _, label = trainer.input_adapter.get_input_context(i, trainer.input_nodes)
                if label == digit:
                    digit_indices.append(i)
            except:
                continue
        
        # Evaluate samples of this digit
        for idx in digit_indices:
            try:
                input_context, true_label = trainer.input_adapter.get_input_context(idx, trainer.input_nodes)
                output_signals = trainer.forward_pass(input_context)
                
                if output_signals:
                    class_encodings = trainer.class_encoder.get_all_encodings()
                    logits = trainer.loss_function.compute_logits_from_signals(
                        output_signals, class_encodings, trainer.lookup_tables
                    )
                    pred_label = torch.argmax(logits).item()
                    
                    if pred_label == true_label:
                        class_correct[digit] += 1
                    class_total[digit] += 1
            except:
                continue
    
    # Print per-class results
    for digit in range(10):
        if class_total[digit] > 0:
            class_acc = class_correct[digit] / class_total[digit]
            print(f"   Digit {digit}: {class_correct[digit]}/{class_total[digit]} = {class_acc:.1%}")
        else:
            print(f"   Digit {digit}: No samples evaluated")
    
    return accuracy

def analyze_system_performance(trainer):
    """Analyze system performance and components."""
    print("\n📈 System Performance Analysis")
    print("=" * 50)
    
    # Configuration summary
    config = trainer.config
    print(f"Architecture:")
    print(f"   • Total nodes: {config.get('architecture.total_nodes')}")
    print(f"   • Input nodes: {config.get('architecture.input_nodes')}")
    print(f"   • Resolution: {config.get('resolution.phase_bins')}×{config.get('resolution.mag_bins')}")
    print(f"   • Resolution increase: {config.get('resolution.resolution_increase')}x vs legacy")
    
    # Input adapter analysis
    if hasattr(trainer.input_adapter, 'get_projection_stats'):
        print(f"\nInput Adapter:")
        stats = trainer.input_adapter.get_projection_stats()
        if 'weight_norm' in stats:
            print(f"   • Weight norm: {stats['weight_norm']:.4f}")
            print(f"   • Condition number: {stats['condition_number']:.2f}")
    
    # Class encoding analysis
    if hasattr(trainer.class_encoder, 'get_encoding_stats'):
        print(f"\nClass Encodings:")
        stats = trainer.class_encoder.get_encoding_stats()
        print(f"   • Orthogonality score: {stats['orthogonality_score']:.3f}")
        print(f"   • Mean similarity: {stats['mean_similarity']:.3f}")
        print(f"   • Max similarity: {stats['max_similarity']:.3f}")
    
    # Gradient accumulation analysis
    if trainer.gradient_accumulator is not None:
        print(f"\nGradient Accumulation:")
        stats = trainer.gradient_accumulator.get_statistics()
        print(f"   • Total gradients: {stats['total_gradients_accumulated']}")
        print(f"   • Total updates: {stats['total_updates_applied']}")
        print(f"   • Avg gradient norm: {stats['average_gradient_norm']:.4f}")
        print(f"   • Gradient variance: {stats['gradient_variance']:.4f}")
    
    # Memory and parameter analysis
    print(f"\nSystem Resources:")
    print(f"   • Total parameters: {trainer.count_parameters():,}")
    print(f"   • Memory usage: {trainer.estimate_memory_usage():.1f} MB")
    print(f"   • Device: {trainer.device}")

def compare_with_baselines():
    """Compare results with previous baselines."""
    print("\n📈 Performance Comparison")
    print("=" * 50)
    print("Previous Results:")
    print("   🔸 Original (batch training): 10% accuracy")
    print("   🔸 Single-sample (50 nodes): 18% accuracy")
    print("   🔸 Specialized (50 nodes): 18% accuracy")
    print("   🔸 1000-node PCA system: ~20% accuracy")
    print("\nModular System Improvements:")
    print("   🔸 High-resolution (64×1024): 16x more capacity")
    print("   🔸 Linear projection: Learnable input features")
    print("   🔸 Orthogonal encodings: Reduced class confusion")
    print("   🔸 Gradient accumulation: Stable learning")
    print("   🔸 CCE loss: Proper classification objective")

def main():
    """Main execution function."""
    print("🚀 Modular NeuroGraph Training System")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Modular NeuroGraph Training')
    parser.add_argument('--config', type=str, default='config/modular_neurograph.yaml',
                       help='Configuration file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (5 epochs)')
    parser.add_argument('--eval-only', action='store_true', help='Evaluation only mode')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to load')
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    set_seeds(args.seed)
    print(f"🎲 Random seed set: {args.seed}")
    
    try:
        # Initialize training context
        print(f"\n🔧 Initializing Modular Training Context...")
        trainer = create_modular_train_context(args.config)
        
        # Load checkpoint if specified
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        
        # Quick test mode
        if args.quick:
            print(f"\n🧪 Quick Test Mode (5 epochs)")
            trainer.num_epochs = 5
            trainer.warmup_epochs = 2
        
        # Show baseline comparison
        compare_with_baselines()
        
        # Evaluation only mode
        if args.eval_only:
            print(f"\n📊 Evaluation Only Mode")
            final_accuracy = comprehensive_evaluation(trainer, num_samples=500)
            analyze_system_performance(trainer)
            
            print(f"\n🎯 Final Accuracy: {final_accuracy:.1%}")
            return final_accuracy
        
        # Training
        print(f"\n🎯 Starting Training ({trainer.num_epochs} epochs)")
        print("=" * 60)
        
        start_time = datetime.now()
        losses = trainer.train()
        end_time = datetime.now()
        
        training_duration = (end_time - start_time).total_seconds()
        print(f"\n⏱️  Training completed in {training_duration:.1f} seconds")
        
        # Plot training curves
        plot_training_curves(losses, trainer.validation_accuracies)
        
        # Comprehensive evaluation
        final_accuracy = comprehensive_evaluation(trainer, num_samples=500)
        
        # System performance analysis
        analyze_system_performance(trainer)
        
        # Results summary
        print("\n🎉 FINAL RESULTS")
        print("=" * 60)
        
        # Architecture summary
        config = trainer.config
        print(f"📊 Architecture:")
        print(f"   • Total nodes: {config.get('architecture.total_nodes')}")
        print(f"   • Input nodes: {config.get('architecture.input_nodes')} (vs 5 baseline)")
        print(f"   • Resolution: {config.get('resolution.phase_bins')}×{config.get('resolution.mag_bins')} (vs 8×256)")
        print(f"   • Resolution increase: {config.get('resolution.resolution_increase')}x")
        print(f"   • Parameters: {trainer.count_parameters():,}")
        
        # Performance summary
        print(f"\n🎯 Performance:")
        print(f"   • Final accuracy: {final_accuracy:.1%}")
        print(f"   • Training epochs: {len(losses)}")
        print(f"   • Final loss: {losses[-1]:.4f}")
        if len(losses) > 1:
            print(f"   • Loss improvement: {losses[0] - losses[-1]:+.4f}")
        
        # Comparison with baselines
        print(f"\n📈 Improvements vs Baselines:")
        baseline_10 = 0.10
        baseline_18 = 0.18
        baseline_20 = 0.20
        
        improvement_vs_10 = (final_accuracy - baseline_10) / baseline_10 * 100
        improvement_vs_18 = (final_accuracy - baseline_18) / baseline_18 * 100
        improvement_vs_20 = (final_accuracy - baseline_20) / baseline_20 * 100
        
        print(f"   • vs 10% baseline: {improvement_vs_10:+.1f}% relative improvement")
        print(f"   • vs 18% baseline: {improvement_vs_18:+.1f}% relative improvement")
        print(f"   • vs 20% baseline: {improvement_vs_20:+.1f}% relative improvement")
        
        # Success criteria
        print(f"\n✅ Success Criteria:")
        success_thresholds = [0.25, 0.30, 0.40, 0.50]
        
        for threshold in success_thresholds:
            if final_accuracy >= threshold:
                print(f"   🎉 ACHIEVED: {final_accuracy:.1%} ≥ {threshold:.1%} target!")
                if threshold >= 0.40:
                    print(f"   🚀 Excellent performance - ready for advanced applications!")
                elif threshold >= 0.30:
                    print(f"   🎯 Good performance - significant improvement over baselines!")
                else:
                    print(f"   📈 Moderate improvement - system shows promise!")
                break
        else:
            print(f"   📊 Current: {final_accuracy:.1%} (next target: {success_thresholds[0]:.1%})")
            print(f"   🔧 Consider further optimization or architectural changes")
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/modular/final_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        trainer.save_checkpoint(checkpoint_path)
        
        print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return final_accuracy
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None

def quick_test():
    """Quick test with reduced parameters for development."""
    print("🧪 Quick Test Mode (5 epochs)")
    print("=" * 40)
    
    try:
        # Create trainer with quick test config
        trainer = create_modular_train_context()
        
        # Override config for quick test
        trainer.num_epochs = 5
        trainer.warmup_epochs = 2
        
        # Quick training
        losses = trainer.train()
        accuracy = trainer.evaluate_accuracy(num_samples=50)
        
        print(f"\n📋 Quick Test Results:")
        print(f"   ✅ Training: {len(losses)} epochs completed")
        print(f"   📊 Final loss: {losses[-1]:.4f}")
        print(f"   🎯 Accuracy: {accuracy:.1%}")
        print(f"   🚀 System ready for full training!")
        
        return accuracy
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def benchmark_components():
    """Benchmark individual components."""
    print("⚡ Component Benchmarking")
    print("=" * 40)
    
    try:
        # Test configuration loading
        start_time = datetime.now()
        config = ModularConfig()
        config_time = (datetime.now() - start_time).total_seconds()
        print(f"   📋 Config loading: {config_time:.3f}s")
        
        # Test lookup table initialization
        from core.high_res_tables import HighResolutionLookupTables
        start_time = datetime.now()
        lookup = HighResolutionLookupTables(64, 1024)
        lookup_time = (datetime.now() - start_time).total_seconds()
        print(f"   📊 Lookup tables: {lookup_time:.3f}s")
        
        # Test input adapter
        from modules.linear_input_adapter import LinearInputAdapter
        start_time = datetime.now()
        adapter = LinearInputAdapter(784, 200, 5, 64, 1024)
        adapter_time = (datetime.now() - start_time).total_seconds()
        print(f"   🔧 Input adapter: {adapter_time:.3f}s")
        
        # Test class encoder
        from modules.orthogonal_encodings import OrthogonalClassEncoder
        start_time = datetime.now()
        encoder = OrthogonalClassEncoder(10, 5, 64, 1024)
        encoder_time = (datetime.now() - start_time).total_seconds()
        print(f"   🎯 Class encoder: {encoder_time:.3f}s")
        
        print(f"   ✅ All components initialized successfully!")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

if __name__ == "__main__":
    # Check for special modes
    if len(sys.argv) > 1:
        if sys.argv[1] == "--benchmark":
            benchmark_components()
            exit(0)
        elif sys.argv[1] == "--quick-test":
            accuracy = quick_test()
            exit(0 if accuracy is not None else 1)
    
    # Run main training
    accuracy = main()
    
    # Exit with appropriate code
    if accuracy is not None:
        print(f"\n🎯 Final accuracy: {accuracy:.1%}")
        exit(0)
    else:
        print(f"\n❌ Training failed")
        exit(1)
