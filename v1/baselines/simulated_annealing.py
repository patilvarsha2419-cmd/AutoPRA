"""
simulated_annealing.py

Simulated Annealing baseline for VLSI cell placement (AutoPRA v1).

This is the traditional (non-RL) baseline used to benchmark the RL
agents against. SA works by starting with a random placement, then
repeatedly proposing small swaps between two cells' positions and
accepting/rejecting them based on the Metropolis criterion, with
"temperature" decreasing over time to gradually favor only improving
moves.

Result from experiments: 70.8% HPWL improvement over random baseline
(mean HPWL 342 vs random baseline 1184).
"""

import math
import random
import numpy as np

from v1.env.placement_env import PlacementEnv, GRID_SIZE, NUM_CELLS


class SimulatedAnnealingPlacer:
    """
    Simulated Annealing placer that operates on the same netlist
    adjacency structure as PlacementEnv, for fair comparison.
    """

    def __init__(
        self,
        adjacency,
        grid_size=GRID_SIZE,
        num_cells=NUM_CELLS,
        initial_temp=100.0,
        final_temp=0.1,
        cooling_rate=0.995,
        iterations_per_temp=50,
        seed=42,
    ):
        self.adjacency = adjacency
        self.grid_size = grid_size
        self.num_cells = num_cells
        self.num_slots = grid_size * grid_size

        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def _slot_to_xy(self, slot):
        x = slot % self.grid_size
        y = slot // self.grid_size
        return x, y

    def _random_initial_placement(self):
        """Randomly assigns each cell to a unique grid slot."""
        slots = list(range(self.num_slots))
        self.np_rng.shuffle(slots)
        return np.array(slots[: self.num_cells], dtype=np.int32)

    def _compute_hpwl(self, cell_positions):
        """Half-Perimeter Wirelength for a full placement."""
        total_hpwl = 0.0
        visited_pairs = set()

        for cell_id, neighbors in self.adjacency.items():
            pos_a = cell_positions[cell_id]
            xa, ya = self._slot_to_xy(pos_a)

            for neighbor_id in neighbors:
                pair = tuple(sorted((cell_id, neighbor_id)))
                if pair in visited_pairs:
                    continue

                pos_b = cell_positions[neighbor_id]
                xb, yb = self._slot_to_xy(pos_b)

                total_hpwl += abs(xa - xb) + abs(ya - yb)
                visited_pairs.add(pair)

        return total_hpwl

    def run(self, verbose=True):
        """
        Runs the simulated annealing optimization.

        Returns:
            dict with keys:
                - "final_positions": np.array of final cell -> slot mapping
                - "final_hpwl": float
                - "history": list of (temperature, best_hpwl) tuples
        """
        current_positions = self._random_initial_placement()
        current_hpwl = self._compute_hpwl(current_positions)

        best_positions = current_positions.copy()
        best_hpwl = current_hpwl

        temperature = self.initial_temp
        history = []

        while temperature > self.final_temp:
            for _ in range(self.iterations_per_temp):
                # propose a swap between two random cells' positions
                cell_a, cell_b = self.rng.sample(range(self.num_cells), 2)

                new_positions = current_positions.copy()
                new_positions[cell_a], new_positions[cell_b] = (
                    new_positions[cell_b],
                    new_positions[cell_a],
                )

                new_hpwl = self._compute_hpwl(new_positions)
                delta = new_hpwl - current_hpwl

                # Metropolis acceptance criterion
                if delta < 0 or self.rng.random() < math.exp(-delta / temperature):
                    current_positions = new_positions
                    current_hpwl = new_hpwl

                    if current_hpwl < best_hpwl:
                        best_positions = current_positions.copy()
                        best_hpwl = current_hpwl

            history.append((temperature, best_hpwl))
            if verbose:
                print(f"Temp={temperature:.3f}  Best HPWL={best_hpwl:.1f}")

            temperature *= self.cooling_rate

        return {
            "final_positions": best_positions,
            "final_hpwl": best_hpwl,
            "history": history,
        }


if __name__ == "__main__":
    # build the same netlist used by PlacementEnv for a fair comparison
    env = PlacementEnv()
    adjacency = env.adjacency

    placer = SimulatedAnnealingPlacer(adjacency=adjacency)
    result = placer.run(verbose=True)

    print(f"\nFinal SA HPWL: {result['final_hpwl']:.1f}")