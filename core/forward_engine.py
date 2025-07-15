# core/forward_engine.py

from core.activation_table import ActivationTable
from core.propagation import propagate_step

def run_forward(
    graph_df,
    node_store,
    phase_cell,
    lookup_table,
    input_context: dict,  # node_id → (phase_idx, mag_idx)
    vector_dim: int,
    phase_bins: int,
    mag_bins: int,
    decay_factor: float,
    min_strength: float,
    max_timesteps: int,
    use_radiation: bool = True,
    top_k_neighbors: int = 4,
    device='cpu',
    verbose=False
):
    """
    Run the forward pass over multiple timesteps (static + radiation).

    Args:
        graph_df: Static topology DataFrame
        node_store: NodeStore object
        phase_cell: PhaseCell module (now gradient-aware)
        lookup_table: ExtendedLookupTableModule
        input_context: dict of node_id → (phase_idx [D], mag_idx [D])
        vector_dim: Dimensionality of phase/mag vectors
        decay_factor: Scalar decay per step
        min_strength: Minimum activation strength to keep a node active
        max_timesteps: Total propagation steps
        use_radiation: Enable phase-based dynamic neighbors
        top_k_neighbors: Top-K neighbors for radiation
        device: 'cpu' or 'cuda'
        verbose: Verbose logging per timestep

    Returns:
        Final ActivationTable object after all steps
    """
    activation = ActivationTable(
        vector_dim=vector_dim,
        phase_bins=phase_bins,
        mag_bins=mag_bins,
        decay_factor=decay_factor,
        min_strength=min_strength,
        device=device
    )

    # Inject initial input context
    for node_id, (p, m) in input_context.items():
        activation.inject(node_id, p.to(device), m.to(device), strength=1.0)

    for t in range(max_timesteps):
        ctx = activation.get_active_context()

        if verbose:
            print(f"\n⏱️ Timestep {t}")
            print(f"🔹 Active nodes: {list(ctx.keys())}")

        updates = propagate_step(
            active_nodes=ctx,
            node_store=node_store,
            phase_cell=phase_cell,
            graph_df=graph_df,
            lookup_table=lookup_table,
            use_radiation=use_radiation,
            top_k_neighbors=top_k_neighbors,
            device=device
        )

        # Overwrite style: clear previous table before injecting next wave
        activation.clear()

        for target_node, new_phase, new_mag, strength in updates:
            activation.inject(target_node, new_phase, new_mag, strength)

        activation.decay_and_prune()

    return activation
