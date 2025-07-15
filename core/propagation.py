import torch
from core.cell import PhaseCell
from core.node_store import NodeStore
from core.graph import load_or_build_graph


def propagate_step(
    active_nodes: dict,
    node_store: NodeStore,
    phase_cell: PhaseCell,
    graph_df,
    device='cpu'
):
    """
    One timestep of forward propagation.

    Args:
        active_nodes (dict): node_id → (context_phase_idx, context_mag_idx)
        node_store (NodeStore): Provides self phase/mag index vectors
        phase_cell (PhaseCell): Lookup + signal unit
        graph_df (pd.DataFrame): Static topology
        device: CPU or CUDA

    Returns:
        updates: List of (target_node_id, new_phase_idx, new_mag_idx, activation_strength)
    """
    updates = []

    for source_node, (ctx_phase_idx, ctx_mag_idx) in active_nodes.items():
        ctx_phase_idx = ctx_phase_idx.to(device)
        ctx_mag_idx = ctx_mag_idx.to(device)

        # For each target connected to source_node
        for target_node in graph_df.loc[graph_df["node_id"] == source_node, "input_connections"].values[0]:
            self_phase_idx = node_store.get_phase(target_node).to(device)
            self_mag_idx   = node_store.get_mag(target_node).to(device)

            phase_out, mag_out, _, strength = phase_cell(
                ctx_phase_idx, ctx_mag_idx, self_phase_idx, self_mag_idx
            )

            updates.append((target_node, phase_out, mag_out, strength.item()))

    return updates
