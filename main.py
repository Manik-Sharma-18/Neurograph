# main.py

from train.train_context import train_context
from utils.config import load_config
import matplotlib.pyplot as plt
import os
import random
import torch

from modules.input_adapters import MNISTPCAAdapter
from modules.class_encoding import generate_fixed_class_encodings
from modules.output_adapters import predict_label_from_output
from core.tables import ExtendedLookupTableModule
from core.graph import load_or_build_graph
from core.node_store import NodeStore
from core.cell import PhaseCell
from core.forward_engine import run_forward

def main():
    cfg = load_config("config/default.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n 🚀 Starting NeuroGraph Training...\n")
    loss_log = train_context()

    # Plot and save convergence
    os.makedirs(cfg.get("log_path", "logs/"), exist_ok=True)
    plt.plot(loss_log)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Convergence")
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(cfg.get("log_path", "logs/"), "loss_curve.png")
    plt.savefig(out_path)
    print(f"\n📉 Saved convergence plot to {out_path}")

    print("\n🔍 Evaluating model on MNIST samples...")

    # Reload graph and components
    graph_df = load_or_build_graph(**cfg, overwrite=False)
    node_store = NodeStore(graph_df, cfg["vector_dim"], cfg["phase_bins"], cfg["mag_bins"])
    lookup = ExtendedLookupTableModule(cfg["phase_bins"], cfg["mag_bins"], device=device)
    phase_cell = PhaseCell(cfg["vector_dim"], lookup)

    input_nodes = list(node_store.input_nodes)
    output_nodes = node_store.output_nodes

    adapter = MNISTPCAAdapter(
        vector_dim=cfg["vector_dim"],
        num_input_nodes=len(input_nodes),
        phase_bins=cfg["phase_bins"],
        mag_bins=cfg["mag_bins"],
        device=device
    )

    class_encodings = generate_fixed_class_encodings(
        cfg["phase_bins"], cfg["mag_bins"], cfg["vector_dim"]
    )

    correct = 0
    total = 100  # adjust if needed
    for _ in range(total):
        idx = random.randint(0, len(adapter.mnist) - 1)
        input_context, label = adapter.get_input_context(idx, input_nodes)

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
            verbose=False
        )

        pred = predict_label_from_output(activation, output_nodes, class_encodings, lookup)
        if pred == label:
            correct += 1

    accuracy = correct / total
    print(f"\n✅ Evaluation Accuracy on {total} MNIST samples: {accuracy:.2%}")

if __name__ == "__main__":
    main()
