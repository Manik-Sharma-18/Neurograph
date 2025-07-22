# core/radiation.py

import torch
from typing import List, Set, Dict
import time

# Cache for static neighbors to avoid repeated DataFrame lookups
_static_neighbors_cache: Dict[str, Set[str]] = {}

def get_radiation_neighbors(
    current_node_id: str,
    ctx_phase_idx: torch.LongTensor,  # [D]
    node_store,
    graph_df,
    lookup_table,
    top_k: int = 4
) -> List[str]:
    """
    Selects Top-K radiation neighbors based on phase alignment.
    Optimized vectorized implementation with caching.

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
    # Cache static neighbors to avoid repeated DataFrame lookups
    if current_node_id not in _static_neighbors_cache:
        static_neighbors = set(
            graph_df.loc[graph_df["node_id"] == current_node_id, "input_connections"].iloc[0]
        )
        _static_neighbors_cache[current_node_id] = static_neighbors
    else:
        static_neighbors = _static_neighbors_cache[current_node_id]
    
    # Get candidate nodes (exclude static neighbors and self)
    phase_table = node_store.phase_table
    all_nodes = set(phase_table.keys())
    candidate_nodes = list(all_nodes - static_neighbors - {current_node_id})
    
    if not candidate_nodes:
        return []
    
    # Move context phase to device once
    N = lookup_table.N
    ctx_phase = ctx_phase_idx.long()
    device = ctx_phase.device
    
    # Vectorized computation with memory optimization
    with torch.no_grad():  # Disable gradient computation for inference
        # Pre-allocate scores tensor
        scores = torch.zeros(len(candidate_nodes), device=device)
        
        # Batch process candidates for better memory efficiency
        batch_size = min(64, len(candidate_nodes))  # Process in batches of 64
        
        for i in range(0, len(candidate_nodes), batch_size):
            batch_end = min(i + batch_size, len(candidate_nodes))
            batch_candidates = candidate_nodes[i:batch_end]
            
            # Vectorized alignment computation for batch
            if batch_candidates:
                # Stack all candidate phase vectors into a single tensor [batch_size, D]
                candidate_phases = torch.stack([
                    node_store.get_phase(candidate_id).long().to(device) 
                    for candidate_id in batch_candidates
                ])
                
                # Broadcast context phase to match batch dimensions [batch_size, D]
                ctx_phase_expanded = ctx_phase.unsqueeze(0).expand_as(candidate_phases)
                
                # Vectorized phase sum computation [batch_size, D]
                phase_sums = (ctx_phase_expanded + candidate_phases) % N
                
                # Vectorized lookup and sum across dimension D [batch_size]
                batch_scores = lookup_table.lookup_phase(phase_sums).sum(dim=1)
                
                # Insert batch scores into preallocated scores tensor
                scores[i:i + len(batch_candidates)] = batch_scores
    
    # Use torch.topk for efficient top-k selection
    if len(scores) > top_k:
        top_k_indices = torch.topk(scores, k=top_k, largest=True)[1]
        return [candidate_nodes[idx] for idx in top_k_indices.cpu().numpy()]
    else:
        # If we have fewer candidates than top_k, sort all
        indices = torch.argsort(scores, descending=True)
        return [candidate_nodes[idx] for idx in indices.cpu().numpy()]

def clear_radiation_cache():
    """Clear the static neighbors cache. Useful for testing or graph changes."""
    global _static_neighbors_cache
    _static_neighbors_cache.clear()

def get_cache_stats():
    """Get statistics about the radiation cache."""
    return {
        'cached_nodes': len(_static_neighbors_cache),
        'cache_size_bytes': sum(len(neighbors) * 8 for neighbors in _static_neighbors_cache.values())
    }
