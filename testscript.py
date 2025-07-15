import torch
from math import pi
from typing import Tuple

# Simulated versions of LookupTableModule and PhaseCell
class LookupTableModule(torch.nn.Module):
    def __init__(self, N: int, M: int, device='cpu'):
        super().__init__()
        self.N = N
        self.M = M
        self.device = device

        self.register_buffer("cos_table", torch.cos(2 * pi * torch.arange(N, dtype=torch.float32) / N))
        self.register_buffer("exp_table", torch.exp(torch.sin(2 * pi * torch.arange(M, dtype=torch.float32) / M)))

    def lookup_phase(self, theta_indices: torch.LongTensor) -> torch.Tensor:
        return self.cos_table[theta_indices % self.N]

    def lookup_magnitude(self, mag_indices: torch.LongTensor) -> torch.Tensor:
        return self.exp_table[mag_indices % self.M]

class PhaseCell(torch.nn.Module):
    def __init__(self, vector_dim: int, lookup_module: LookupTableModule):
        super().__init__()
        self.D = vector_dim
        self.lookup = lookup_module
        self.N = lookup_module.N
        self.M = lookup_module.M

    def forward(self, context_phase, context_mag, self_phase, self_mag) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        theta_new = (context_phase + self_phase) % self.N
        theta_new = theta_new.long()
        cos_vals = self.lookup.lookup_phase(theta_new)

        mag_new = (context_mag + self_mag) % self.M
        mag_new = mag_new.long()
        exp_vals = self.lookup.lookup_magnitude(mag_new)

        signal = cos_vals * exp_vals
        activation_strength = signal.sum()

        return theta_new, mag_new, activation_strength


# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N, M, D = 8, 256, 5

# Instantiate lookup and cell
lookup = LookupTableModule(N=N, M=M).to(device)
cell = PhaseCell(vector_dim=D, lookup_module=lookup).to(device)

# Create dummy inputs
ctx_phase = torch.randint(0, N, (D,), device=device)
ctx_mag = torch.randint(0, M, (D,), device=device)
self_phase = torch.randint(0, N, (D,), device=device)
self_mag = torch.randint(0, M, (D,), device=device)

# Run test
theta_new, mag_new, activation_strength = cell(ctx_phase, ctx_mag, self_phase, self_mag)

import pandas as pd
print(pd.DataFrame({
    "theta_new": theta_new.cpu().numpy(),
    "mag_new": mag_new.cpu().numpy(),
    "cos_val": lookup.lookup_phase(theta_new).cpu().numpy(),
    "exp_val": lookup.lookup_magnitude(mag_new).cpu().numpy(),
    "signal": (lookup.lookup_phase(theta_new) * lookup.lookup_magnitude(mag_new)).cpu().numpy()
}).assign(activation_strength=activation_strength.item()))

