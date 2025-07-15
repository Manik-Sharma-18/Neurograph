# train/train_context.py

import torch
from core.tables import ExtendedLookupTableModule
from core.graph import load_or_build_graph
from core.node_store import NodeStore
from core.cell import PhaseCell
from core.forward_engine import run_forward
from core.backward import backward_pass
from config import load_config
import random

def train_context():
    # Load config
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load graph
    graph_df = load_or_build_graph(**cfg, overwrite=True)
    
    # Initialize modules
    node_store = NodeStore(graph_df, cfg["vector_dim"], cfg["phase_bins"], cfg["mag_bins"])
    lookup = ExtendedLookupTableModule(cfg["phase_bins"], cfg["mag_bins"], device=device)
    phase_cell = PhaseCell(cfg["vector_dim"], lookup)

    output_nodes = node_store.output_nodes
    warmup_epochs = cfg.get("warmup_epochs", 5)
    learning_rate = cfg["learning_rate"]
    num_epochs = cfg.get("num_epochs", 50)

    loss_log = []

    # Use fixed input node + target node pairs (placeholder logic)
    input_context = {
        random.choice(list(node_store.input_nodes)): (
            torch.randint(0, cfg["phase_bins"], (cfg["vector_dim"],)),
            torch.randint(0, cfg["mag_bins"], (cfg["vector_dim"],))
        )
    }

    target_context = {
        node_id: (
            torch.zeros(cfg["vector_dim"], dtype=torch.long),
            torch.zeros(cfg["vector_dim"], dtype=torch.long)
        )
        for node_id in output_nodes
    }

    for epoch in range(num_epochs):
        include_all_outputs = epoch < warmup_epochs

        # Forward pass
        activation = run_forward(
            graph_df=graph_df,
            node_store=node_store,
            phase_cell=phase_cell,
            lookup_table=lookup,
            input_context=input_context,
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

        # Compute loss (MSE between signal vector and target)
        total_loss = 0.0
        active_outputs = activation.get_active_context().keys()
        counted_nodes = 0

        for node_id in output_nodes:
            if not include_all_outputs and node_id not in active_outputs:
                continue

            pred_phase, pred_mag = activation.table.get(node_id, (None, None, None))[:2]
            tgt_phase, tgt_mag = target_context[node_id]

            if pred_phase is None:
                continue

            pred_phase = pred_phase.to(torch.float32)
            tgt_phase = tgt_phase.to(torch.float32)

            pred_mag = pred_mag.to(torch.float32)
            tgt_mag = tgt_mag.to(torch.float32)

            # Simple L2 loss in index space
            loss = torch.sum((pred_phase - tgt_phase) ** 2) + torch.sum((pred_mag - tgt_mag) ** 2)
            total_loss += loss.item()
            counted_nodes += 1

        if counted_nodes > 0:
            total_loss /= counted_nodes
        loss_log.append(total_loss)

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

        # Backward pass
        backward_pass(
            activation_table=activation,
            node_store=node_store,
            phase_cell=phase_cell,
            lookup_table=lookup,
            target_context=target_context,
            output_nodes=output_nodes,
            learning_rate=learning_rate,
            include_all_outputs=include_all_outputs,
            verbose=(epoch % 10 == 0),
            device=device
        )

    return loss_log
