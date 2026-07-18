"""
placement_env.py

Custom Gymnasium environment for VLSI cell placement using
Reinforcement Learning (AutoPRA v1).

The agent places cells one at a time onto a 12x12 grid (144 slots
for 100 cells) to minimize total Half-Perimeter Wirelength (HPWL).

Netlist: synthetic, generated with a Zipf distribution to mimic
realistic connectivity skew (few highly-connected cells, many
sparsely-connected ones).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


GRID_SIZE = 12          # 12x12 grid -> 144 slots
NUM_CELLS = 100          # number of cells to place
ZIPF_PARAM = 2.0         # controls netlist connectivity skew


class PlacementEnv(gym.Env):
    """
    Custom environment for VLSI cell placement.

    Observation:
        - grid occupancy (flattened, 144 values: 0 = empty, 1 = occupied)
        - current cell index (normalized, 1 value)
        - placed ratio so far (1 value)
        Total obs size = 144 + 1 + 1 = 146

    Action:
        - discrete, flattened grid slot index (0 to 143)

    Reward:
        - negative delta HPWL at each step (i.e. reward = -(new_HPWL - old_HPWL))
        - encourages the agent to place cells so total wirelength grows
          as slowly as possible
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.grid_size = GRID_SIZE
        self.num_slots = GRID_SIZE * GRID_SIZE
        self.num_cells = NUM_CELLS

        # action space: pick any of the 144 grid slots
        self.action_space = spaces.Discrete(self.num_slots)

        # observation space: grid occupancy + current cell idx + placed ratio
        obs_size = self.num_slots + 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # build a synthetic netlist once; reused across episodes
        self.adjacency = self._generate_netlist()

        self.reset()

    # ------------------------------------------------------------------
    # Netlist generation
    # ------------------------------------------------------------------
    def _generate_netlist(self):
        """
        Generates a synthetic adjacency structure using a Zipf
        distribution so a few cells have many connections and most
        have few (mirrors real netlist connectivity patterns).

        Returns:
            dict[int, list[int]] mapping cell_id -> list of connected cell_ids
        """
        rng = np.random.default_rng(seed=42)
        adjacency = {i: [] for i in range(self.num_cells)}

        # degree per cell drawn from a Zipf-like distribution
        degrees = rng.zipf(ZIPF_PARAM, size=self.num_cells)
        degrees = np.clip(degrees, 1, self.num_cells - 1)

        for cell_id in range(self.num_cells):
            num_connections = degrees[cell_id]
            possible = [c for c in range(self.num_cells) if c != cell_id]
            connections = rng.choice(
                possible, size=min(num_connections, len(possible)), replace=False
            )
            for other in connections:
                if other not in adjacency[cell_id]:
                    adjacency[cell_id].append(int(other))
                if cell_id not in adjacency[other]:
                    adjacency[other].append(cell_id)

        return adjacency

    # ------------------------------------------------------------------
    # Core Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # grid_occupancy[i] = -1 if empty, else the cell_id placed there
        self.grid_occupancy = -np.ones(self.num_slots, dtype=np.int32)

        # cell_positions[cell_id] = slot index, or -1 if not yet placed
        self.cell_positions = -np.ones(self.num_cells, dtype=np.int32)

        self.current_cell = 0
        self.current_hpwl = 0.0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        occupied = self.grid_occupancy[action] != -1

        if occupied:
            # invalid move: slot taken, small penalty, episode continues
            reward = -5.0
            terminated = False
            obs = self._get_obs()
            info = {"invalid_move": True}
            return obs, reward, terminated, False, info

        # place the current cell at the chosen slot
        self.grid_occupancy[action] = self.current_cell
        self.cell_positions[self.current_cell] = action

        # compute new HPWL and reward = -(delta)
        new_hpwl = self._compute_total_hpwl()
        reward = -(new_hpwl - self.current_hpwl)
        self.current_hpwl = new_hpwl

        self.current_cell += 1
        terminated = self.current_cell >= self.num_cells

        obs = self._get_obs()
        info = {"invalid_move": False}
        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_obs(self):
        occupancy = (self.grid_occupancy != -1).astype(np.float32)
        current_cell_norm = np.array(
            [self.current_cell / self.num_cells], dtype=np.float32
        )
        placed_ratio = np.array(
            [self.current_cell / self.num_cells], dtype=np.float32
        )
        return np.concatenate([occupancy, current_cell_norm, placed_ratio])

    def _slot_to_xy(self, slot):
        x = slot % self.grid_size
        y = slot // self.grid_size
        return x, y

    def _compute_total_hpwl(self):
        """
        Half-Perimeter Wirelength over all nets currently fully or
        partially placed. Only considers cells that have been placed.
        """
        total_hpwl = 0.0
        visited_pairs = set()

        for cell_id, neighbors in self.adjacency.items():
            pos_a = self.cell_positions[cell_id]
            if pos_a == -1:
                continue
            xa, ya = self._slot_to_xy(pos_a)

            for neighbor_id in neighbors:
                pair = tuple(sorted((cell_id, neighbor_id)))
                if pair in visited_pairs:
                    continue

                pos_b = self.cell_positions[neighbor_id]
                if pos_b == -1:
                    continue

                xb, yb = self._slot_to_xy(pos_b)
                total_hpwl += abs(xa - xb) + abs(ya - yb)
                visited_pairs.add(pair)

        return total_hpwl

    def final_hpwl(self):
        """Returns the HPWL of the fully placed design (call after episode ends)."""
        return self._compute_total_hpwl()