# core/activation_table.py

import torch

class ActivationTable:
    def __init__(self, vector_dim, phase_bins, mag_bins, decay_factor=0.95, min_strength=0.001, device='cpu'):
        """
        Tracks and updates active neurons over time.

        Args:
            vector_dim (int): Dimensionality of each phase/mag vector
            phase_bins (int): Number of discrete phase values
            mag_bins (int): Number of discrete magnitude values
            decay_factor (float): Multiplier applied to activation strength each timestep
            min_strength (float): Threshold below which activation is pruned
            device (str): 'cpu' or 'cuda'
        """
        self.vector_dim = vector_dim
        self.phase_bins = phase_bins
        self.mag_bins = mag_bins
        self.decay = decay_factor
        self.min_strength = min_strength
        self.device = device
        self.table = {}  # node_id → (phase_idx [D], mag_idx [D], strength)

    def inject(self, node_id, phase_idx, mag_idx, strength):
        """
        Inject or update node activation.
        Both phase and magnitude are accumulated and wrapped modulo bin count.
        """
        if node_id in self.table:
            prev_phase, prev_mag, prev_strength = self.table[node_id]
            new_phase = (prev_phase + phase_idx) % self.phase_bins
            new_mag   = (prev_mag + mag_idx) % self.mag_bins
            new_strength = prev_strength + strength
            self.table[node_id] = (new_phase, new_mag, new_strength)
        else:
            wrapped_phase = phase_idx % self.phase_bins
            wrapped_mag   = mag_idx % self.mag_bins
            self.table[node_id] = (wrapped_phase, wrapped_mag, strength)

    def decay_and_prune(self):
        """
        Apply decay to strengths, and remove weak activations.
        """
        new_table = {}
        for node_id, (phase, mag, strength) in self.table.items():
            decayed = strength * self.decay
            if decayed >= self.min_strength:
                new_table[node_id] = (phase, mag, decayed)
        self.table = new_table

    def get_active_context(self):
        """
        Returns: dict of node_id → (phase_idx [D], mag_idx [D])
        """
        return {
            nid: (phase, mag)
            for nid, (phase, mag, strength) in self.table.items()
        }

    def is_active(self, node_id):
        return node_id in self.table

    def clear(self):
        self.table.clear()
