# core/forward_engine.py

from core.activation_table import ActivationTable
from core.propagation import propagate_step


def run_forward(
    graph_df,
    node_store,
    phase_cell,
    input_context: dict,  # node_id → (phase_idx, mag_idx)
    vector_dim: int,
    phase_bins: int,
    mag_bins: int,
    decay_factor: float,
    min_strength: float,
    max_timesteps: int,
    device='cpu',
    verbose=False
):
    """
    Run the forward pass over multiple timesteps.
    
    Args:
        graph_df: DataFrame with graph structure
        node_store: NodeStore with phase/mag indices
        phase_cell: PhaseCell module
        input_context: dict of node_id → (phase_idx [D], mag_idx [D])
        vector_dim: Dimensionality of each phase/mag vector
        decay_factor: Scalar decay per step
        min_strength: Threshold to keep a node active
        max_timesteps: Number of propagation steps
        device: CPU or CUDA
        verbose: Print per-timestep logs

    Returns:
        final ActivationTable after T steps
    """
    activation = ActivationTable(
        vector_dim=vector_dim,
        phase_bins=phase_bins,
        mag_bins=mag_bins,
        decay_factor=decay_factor,
        min_strength=min_strength,
        device=device
    )

    # Inject initial input
    for node_id, (p, m) in input_context.items():
        activation.inject(node_id, p.to(device), m.to(device), strength=1.0)

    for t in range(max_timesteps):
        ctx = activation.get_active_context()

        if verbose:
            print(f"\n Timestep {t}, active nodes: {list(ctx.keys())}")

        updates = propagate_step(ctx, node_store, phase_cell, graph_df, device)

        # Clear table before injecting new activations (temporal overwrite style)
        activation.clear()

        for target_node, new_phase, new_mag, strength in updates:
            activation.inject(target_node, new_phase, new_mag, strength)

        activation.decay_and_prune()

    return activation
