# core/tables.py

import torch
import math
import torch.nn as nn


def get_cos_table(N: int, device='cpu') -> torch.Tensor:
    """
    Creates a lookup table of cos(2πk/N) for discrete phase indices.
    Args:
        N (int): Number of phase bins (e.g. 8, 16)
    Returns:
        Tensor of shape [N]
    """
    indices = torch.arange(N, dtype=torch.float32, device=device)
    return torch.cos(2 * math.pi * indices / N)


def get_exp_sin_table(M: int, device='cpu') -> torch.Tensor:
    """
    Creates a lookup table of exp(sin(2πk/M)) for discrete magnitude indices.
    Args:
        M (int): Number of magnitude bins (e.g. 64, 256)
    Returns:
        Tensor of shape [M]
    """
    indices = torch.arange(M, dtype=torch.float32, device=device)
    scaled = 2 * math.pi * indices / M
    return torch.exp(torch.sin(scaled))

class LookupTableModule(nn.Module):
    def __init__(self, N: int, M: int, device='cpu'):
        """
        Stores lookup tables for phase and magnitude.
        Args:
            N (int): Number of phase bins
            M (int): Number of magnitude bins
        """
        super().__init__()
        self.N = N
        self.M = M
        self.device = device

        # Register buffers so they're moved with model and saved in state_dict
        self.register_buffer("cos_table", get_cos_table(N, device))
        self.register_buffer("exp_table", get_exp_sin_table(M, device))

    def lookup_phase(self, theta_indices: torch.LongTensor) -> torch.Tensor:
        """
        Lookup cosine values for phase indices.
        """
        return self.cos_table[theta_indices % self.N]

    def lookup_magnitude(self, mag_indices: torch.LongTensor) -> torch.Tensor:
        """
        Lookup exp(sin(.)) values for magnitude indices.
        """
        return self.exp_table[mag_indices % self.M]