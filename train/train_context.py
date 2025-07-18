# train/train_context.py

import torch
import random
from core.tables import ExtendedLookupTableModule
from core.graph import load_or_build_graph
from core.node_store import NodeStore
from core.cell import PhaseCell
from core.forward_engine import run_forward
from core.backward import backward_pass
from utils.config import load_config

from modules.input_adapters import MNISTPCAAdapter
from modules.class_encoding import generate_digit_class_encodings
from modules.loss import signal_loss_from_lookup


def train_context():
    # Load config
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load graph
    graph_keys = [
        "total_nodes", "num_input_nodes", "num_output_nodes",
        "vector_dim", "phase_bins", "mag_bins", "cardinality", "seed"
    ]
    graph_config = {k: cfg[k] for k in graph_keys}
    graph_df = load_or_build_graph(**graph_config, overwrite=False)

    # Initialize graph components
    node_store = NodeStore(graph_df, cfg["vector_dim"], cfg["phase_bins"], cfg["mag_bins"])
    lookup = ExtendedLookupTableModule(cfg["phase_bins"], cfg["mag_bins"], device=device)
    phase_cell = PhaseCell(cfg["vector_dim"], lookup)

    # Setup training components
    output_nodes = node_store.output_nodes
    input_nodes = list(node_store.input_nodes)
    warmup_epochs = cfg.get("warmup_epochs", 5)
    learning_rate = cfg["learning_rate"]
    num_epochs = cfg.get("num_epochs", 50)
    batch_size = cfg.get("batch_size", 5)  # Add batch size from config or default to 5

    # Setup MNIST + PCA injector
    adapter = MNISTPCAAdapter(
        vector_dim=cfg["vector_dim"],
        num_input_nodes=len(input_nodes),
        phase_bins=cfg["phase_bins"],
        mag_bins=cfg["mag_bins"],
        device=device
    )

    # Setup digit class encodings
    class_phase_encodings, class_mag_encodings = generate_digit_class_encodings(
        num_classes=10,
        vector_dim=cfg["vector_dim"],
        phase_bins=cfg["phase_bins"],
        mag_bins=cfg["mag_bins"],
        seed=cfg.get("seed", 42)
    )   

    loss_log = []

    for epoch in range(num_epochs):
        include_all_outputs = epoch < warmup_epochs

        # Batch processing: collect multiple samples
        batch_input_contexts = []
        batch_labels = []
        
        # Sample batch_size examples
        for _ in range(batch_size):
            sample_idx = random.randint(0, len(adapter.mnist) - 1)
            input_context, label = adapter.get_input_context(sample_idx, input_nodes)
            batch_input_contexts.append(input_context)
            batch_labels.append(label)

        # Merge all input contexts into one dictionary (assuming no node conflicts)
        merged_input_context = {}
        for input_context in batch_input_contexts:
            merged_input_context.update(input_context)

        # Run forward pass once with merged input context
        activation = run_forward(
            graph_df=graph_df,
            node_store=node_store,
            phase_cell=phase_cell,
            lookup_table=lookup,
            input_context=merged_input_context,
            vector_dim=cfg["vector_dim"],
            phase_bins=cfg["phase_bins"],
            mag_bins=cfg["mag_bins"],
            decay_factor=cfg["decay_factor"],
            min_strength=cfg["min_activation_strength"],
            max_timesteps=cfg["max_timesteps"],
            use_radiation=cfg["use_radiation"],
            top_k_neighbors=cfg["top_k_neighbors"],
            device=device,
            verbose=(epoch % 10 == 0)
        )

        # Compute loss: reconstruct target_context from individual labels
        total_loss = 0.0
        active_outputs = activation.get_active_context().keys()
        total_counted_nodes = 0
        
        # Average loss over all samples in the batch
        for batch_idx in range(batch_size):
            label = batch_labels[batch_idx]
            class_phase = class_phase_encodings[label]
            class_mag = class_mag_encodings[label]

            # Create target context for this sample
            sample_target_context = {
                node_id: (class_phase.clone(), class_mag.clone())
                for node_id in output_nodes
            }

            # Compute loss for this sample
            sample_loss = 0.0
            sample_counted_nodes = 0

            for node_id in output_nodes:
                if not include_all_outputs and node_id not in active_outputs:
                    continue

                pred_phase, pred_mag = activation.table.get(node_id, (None, None, None))[:2]
                if pred_phase is None:
                    continue

                tgt_phase, tgt_mag = sample_target_context[node_id]

                pred_phase = pred_phase.to(torch.float32)
                tgt_phase = tgt_phase.to(torch.float32)
                pred_mag = pred_mag.to(torch.float32)
                tgt_mag = tgt_mag.to(torch.float32)

                loss = signal_loss_from_lookup(
                        pred_phase, pred_mag,
                         tgt_phase, tgt_mag,
                         lookup
                        )
                sample_loss += loss.item()
                sample_counted_nodes += 1

            if sample_counted_nodes > 0:
                sample_loss /= sample_counted_nodes
                total_loss += sample_loss
                total_counted_nodes += 1

        # Average loss over the batch
        if total_counted_nodes > 0:
            total_loss /= total_counted_nodes
        loss_log.append(total_loss)

        print(f"[Epoch {epoch+1}] Batch Loss: {total_loss:.4f}")

        # Backward pass (unchanged) - use the last sample's target context for backward pass
        final_label = batch_labels[-1]
        final_target_context = {
            node_id: (class_phase_encodings[final_label].clone(), class_mag_encodings[final_label].clone())
            for node_id in output_nodes
        }

        backward_pass(
            activation_table=activation,
            node_store=node_store,
            phase_cell=phase_cell,
            lookup_table=lookup,
            target_context=final_target_context,
            output_nodes=output_nodes,
            learning_rate=learning_rate,
            include_all_outputs=include_all_outputs,
            verbose=(epoch % 10 == 0),
            device=device
        )

    return loss_log
