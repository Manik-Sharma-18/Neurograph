# core/radiation.py

import torch

def get_radiation_neighbors(
    current_node_id: str,
    ctx_phase_idx: torch.LongTensor,  # [D]
    node_store,
    graph_df,
    lookup_table,
    top_k: int = 4
):
    """
    Selects Top-K radiation neighbors based on phase alignment.
    Brute-force implementation — suitable for small graphs.

    Args:
        current_node_id (str): ID of the active source node
        ctx_phase_idx (LongTensor): Phase index vector from activation table [D]
        node_store (NodeStore): Provides phase vectors
        graph_df (pd.DataFrame): Contains graph topology
        lookup_table (LookupTableModule): Provides cosine table
        top_k (int): Number of neighbors to select

    Returns:
        List of target node IDs (List[str])
    """
    phase_table = node_store.phase_table
    all_nodes = set(phase_table.keys())

    # Get statically connected targets
    static_neighbors = set(
        graph_df.loc[graph_df["node_id"] == current_node_id, "input_connections"].values[0]
    )
    candidate_nodes = list(all_nodes - static_neighbors - {current_node_id})

    N = lookup_table.N
    ctx_phase = ctx_phase_idx.to(torch.long)

    scores = []
    for candidate_id in candidate_nodes:
        candidate_phase = node_store.get_phase(candidate_id).to(torch.long)  # [D]
        phase_sum = (ctx_phase + candidate_phase) % N  # [D]
        cos_values = lookup_table.lookup_phase(phase_sum)  # [D]
        alignment_score = cos_values.sum().item()
        scores.append((candidate_id, alignment_score))

    top_k_sorted = sorted(scores, key=lambda x: -x[1])[:top_k]
    return [nid for nid, _ in top_k_sorted]
