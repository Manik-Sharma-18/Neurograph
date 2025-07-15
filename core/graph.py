import pandas as pd
import numpy as np
import os

def build_static_graph(
    total_nodes=35,
    num_input_nodes=5,
    num_output_nodes=5,
    vector_dim=5,
    phase_bins=8,
    mag_bins=256,
    cardinality=3,
    seed=42
) -> pd.DataFrame:
    """
    Builds a static DAG graph with node IDs, connection topology, and I/O flags.
    Does not store phase or magnitude vectors.
    """
    np.random.seed(seed)

    node_ids = [f"n{i}" for i in range(total_nodes)]
    input_nodes = node_ids[:num_input_nodes]
    output_nodes = node_ids[-num_output_nodes:]

    graph_data = []

    for i, node_id in enumerate(node_ids):
        is_input = node_id in input_nodes
        is_output = node_id in output_nodes

        row = {
            "node_id": node_id,
            "input_connections": [],
            "is_input": is_input,
            "is_output": is_output
        }

        graph_data.append(row)

    df = pd.DataFrame(graph_data)

    # Assign DAG connections
    for idx, row in df.iterrows():
        if row["is_input"]:
            continue

        node_index = int(row["node_id"][1:])
        candidate_sources = [f"n{i}" for i in range(node_index) if f"n{i}" not in output_nodes]

        if len(candidate_sources) >= cardinality:
            connections = np.random.choice(candidate_sources, size=cardinality, replace=False).tolist()
        else:
            connections = candidate_sources

        df.at[idx, 'input_connections'] = connections

    return df

def load_or_build_graph(path="config/static_graph.pkl", overwrite=False, **kwargs) -> pd.DataFrame:
    if not overwrite and os.path.exists(path):
        print(f" Loaded graph from {path}")
        return pd.read_pickle(path)

    df = build_static_graph(**kwargs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_pickle(path)
    print(f"🆕 Built new graph and saved to {path}")
    return df
