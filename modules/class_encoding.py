# modules/class_encoding.py

import torch

def generate_fixed_class_encodings(phase_bins, mag_bins, vector_dim, seed=42):
    """
    Returns a dictionary mapping digit labels (0–9) to fixed target phase/mag vectors.
    These will be used as shared targets for all output nodes (Option 1).

    Args:
        phase_bins (int): Number of discrete phase values
        mag_bins (int): Number of discrete magnitude values
        vector_dim (int): Dimension of each phase/mag vector
        seed (int): Random seed for reproducibility

    Returns:
        dict: {digit: (phase_idx [D], mag_idx [D])}
    """
    torch.manual_seed(seed)
    encodings = {}
    for digit in range(10):
        phase_idx = torch.randint(0, phase_bins, (vector_dim,))
        mag_idx   = torch.randint(0, mag_bins, (vector_dim,))
        encodings[digit] = (phase_idx, mag_idx)
    return encodings
