"""
placement_env.py (v2)

Congestion-aware Gymnasium environment for VLSI cell placement,
supporting both MCNC and CircuitNet benchmarks (AutoPRA v2).

Key differences from v1:
- Real benchmark netlists (MCNC cm138a, CircuitNet RISC-V subset)
  instead of synthetic Zipf netlists
- Larger grid (28x28) and cell count (up to 250 cells)
- Connectivity-ordered (topological) placement instead of arbitrary order
- Congestion and overlap penalties added to the reward, not just HPWL
- Incremental HPWL updates for efficiency at larger scale
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


CONGESTION_WEIGHT = 2.0
BIN_UTIL_WEIGHT = 2.0
OVERLAP_WEIGHT = 5.0
BIN_UTIL_THRESHOLD = 0.40  # max_quadrant_fraction penalty kicks in above this


class AutoPRAEnv(gym.Env):
    """
    Congestion-aware placement environment for AutoPRA v2.

    Observation (582 values):
        - 576 values: local density map (24x24 window flattened, or
          equivalent fixed-size density representation around current cell)
        - cell_w: normalized width of current cell
        - cell_h: normalized height of current cell
        - neighbor_centroid_x: normalized x of already-placed neighbors' centroid
        - neighbor_centroid_y: normalized y of already-placed neighbors' centroid
        - placed_neighbor_ratio: fraction of current cell's neighbors already placed
        - placed_ratio: fraction of all cells placed so far

    Action:
        - discrete, flattened grid slot index (0 to grid_w*grid_h - 1)

    Reward:
        reward = delta_HPWL
                 - CONGESTION_WEIGHT * congestion
                 - BIN_UTIL_WEIGHT * bin_utilization_penalty
                 - OVERLAP_WEIGHT * overlap_penalty

    Two supported construction modes:
        MCNC:       AutoPRAEnv(benchmark_dict, adjacency, grid_w, grid_h)
        CircuitNet: AutoPRAEnv(subset_nets, adjacency, cell_to_nets,
                                norm_w, norm_h, num_cells, grid_w, grid_h)
    """

    metadata = {"render_modes": []}

    def __init__(self, *args):
        super().__init__()

        if len(args) == 4:
            # MCNC mode: (benchmark_dict, adjacency, grid_w, grid_h)
            benchmark_dict, adjacency, grid_w, grid_h = args
            self.mode = "mcnc"
            self.benchmark_dict = benchmark_dict
            self.adjacency = adjacency
            self.num_cells = benchmark_dict["num_cells"]
            self.norm_w = {c: 1.0 for c in range(self.num_cells)}
            self.norm_h = {c: 1.0 for c in range(self.num_cells)}

        elif len(args) == 8:
            # CircuitNet mode
            (subset_nets, adjacency, cell_to_nets,
             norm_w, norm_h, num_cells, grid_w, grid_h) = args
            self.mode = "circuitnet"
            self.subset_nets = subset_nets
            self.adjacency = adjacency
            self.cell_to_nets = cell_to_nets
            self.norm_w = norm_w
            self.norm_h = norm_h
            self.num_cells = num_cells

        else:
            raise ValueError(
                "AutoPRAEnv expects either 4 args (MCNC) or 8 args (CircuitNet)"
            )

        self.grid_w = grid_w
        self.grid_h = grid_h
        self.num_slots = grid_w * grid_h

        # connectivity-ordered (topological) placement order:
        # place highest-degree / most-connected-to-already-placed cells first
        self.placement_order = self._compute_placement_order()

        self.action_space = spaces.Discrete(self.num_slots)

        obs_size = 576 + 6
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        self.reset()

    # ------------------------------------------------------------------
    def _compute_placement_order(self):
        """
        Orders cells so that at each step, the next cell placed has
        maximum connectivity to already-placed cells (falls back to
        highest raw degree for the first cell). This is a greedy
        connectivity-driven topological ordering, not a strict DAG sort.
        """
        degree = {c: len(neighbors) for c, neighbors in self.adjacency.items()}
        remaining = set(self.adjacency.keys())
        placed = set()
        order = []

        # seed with highest-degree cell
        first = max(remaining, key=lambda c: degree[c])
        order.append(first)
        placed.add(first)
        remaining.discard(first)

        while remaining:
            best_cell = None
            best_score = -1
            for c in remaining:
                score = sum(1 for n in self.adjacency[c] if n in placed)
                if score > best_score:
                    best_score = score
                    best_cell = c
            if best_cell is None:
                best_cell = next(iter(remaining))
            order.append(best_cell)
            placed.add(best_cell)
            remaining.discard(best_cell)

        return order

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid_occupancy = -np.ones(self.num_slots, dtype=np.int32)
        self.cell_positions = -np.ones(self.num_cells, dtype=np.int32)

        self.step_idx = 0
        self.current_hpwl = 0.0

        # bin utilization tracked per-quadrant for congestion penalty
        self.quadrant_counts = np.zeros(4, dtype=np.int32)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        current_cell = self.placement_order[self.step_idx]

        occupied = self.grid_occupancy[action] != -1
        if occupied:
            reward = -5.0
            terminated = False
            obs = self._get_obs()
            info = {"invalid_move": True}
            return obs, reward, terminated, False, info

        self.grid_occupancy[action] = current_cell
        self.cell_positions[current_cell] = action
        self._update_quadrant_counts(action, delta=1)

        new_hpwl = self._compute_incremental_hpwl_delta(current_cell)
        delta_hpwl = new_hpwl
        self.current_hpwl += new_hpwl

        congestion = self._compute_congestion(action)
        bin_util_penalty = self._compute_bin_util_penalty()
        overlap_penalty = 0.0  # slots are unique, so no true overlap given occupancy check above

        reward = (
            -delta_hpwl
            - CONGESTION_WEIGHT * congestion
            - BIN_UTIL_WEIGHT * bin_util_penalty
            - OVERLAP_WEIGHT * overlap_penalty
        )

        self.step_idx += 1
        terminated = self.step_idx >= self.num_cells

        obs = self._get_obs()
        info = {"invalid_move": False}
        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    def _slot_to_xy(self, slot):
        x = slot % self.grid_w
        y = slot // self.grid_w
        return x, y

    def _update_quadrant_counts(self, slot, delta):
        x, y = self._slot_to_xy(slot)
        quadrant = (1 if x >= self.grid_w // 2 else 0) + (2 if y >= self.grid_h // 2 else 0)
        self.quadrant_counts[quadrant] += delta

    def _compute_bin_util_penalty(self):
        placed_so_far = max(1, self.step_idx + 1)
        max_fraction = self.quadrant_counts.max() / placed_so_far
        return max(0.0, max_fraction - BIN_UTIL_THRESHOLD)

    def _compute_congestion(self, slot):
        """
        Local congestion estimate: density of already-placed cells in
        a small window around the given slot.
        """
        x, y = self._slot_to_xy(slot)
        window = 3
        count = 0
        total = 0
        for dx in range(-window, window + 1):
            for dy in range(-window, window + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                    total += 1
                    if self.grid_occupancy[ny * self.grid_w + nx] != -1:
                        count += 1
        return count / total if total > 0 else 0.0

    def _compute_incremental_hpwl_delta(self, placed_cell):
        """
        Computes the HPWL contribution added by placing `placed_cell`,
        considering only nets/edges to already-placed neighbors
        (incremental update, avoids recomputing full HPWL every step).
        """
        pos_a = self.cell_positions[placed_cell]
        xa, ya = self._slot_to_xy(pos_a)

        delta = 0.0
        for neighbor_id in self.adjacency[placed_cell]:
            pos_b = self.cell_positions[neighbor_id]
            if pos_b == -1:
                continue
            xb, yb = self._slot_to_xy(pos_b)
            delta += abs(xa - xb) + abs(ya - yb)

        return delta

    def _get_obs(self):
        # local density map around the next cell to place (or center if none placed)
        density_map = np.zeros((24, 24), dtype=np.float32)
        placed_slots = self.grid_occupancy[self.grid_occupancy != -1]
        if len(placed_slots) > 0:
            last_slot = placed_slots[-1]
            cx, cy = self._slot_to_xy(last_slot)
        else:
            cx, cy = self.grid_w // 2, self.grid_h // 2

        for i in range(24):
            for j in range(24):
                gx = cx - 12 + i
                gy = cy - 12 + j
                if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                    slot = gy * self.grid_w + gx
                    density_map[i, j] = 1.0 if self.grid_occupancy[slot] != -1 else 0.0

        density_flat = density_map.flatten()

        if self.step_idx < self.num_cells:
            next_cell = self.placement_order[self.step_idx]
            cell_w = self.norm_w.get(next_cell, 1.0)
            cell_h = self.norm_h.get(next_cell, 1.0)

            placed_neighbors = [
                n for n in self.adjacency[next_cell] if self.cell_positions[n] != -1
            ]
            total_neighbors = max(1, len(self.adjacency[next_cell]))
            placed_neighbor_ratio = len(placed_neighbors) / total_neighbors

            if placed_neighbors:
                xs, ys = [], []
                for n in placed_neighbors:
                    nx, ny = self._slot_to_xy(self.cell_positions[n])
                    xs.append(nx)
                    ys.append(ny)
                neighbor_centroid_x = np.mean(xs) / self.grid_w
                neighbor_centroid_y = np.mean(ys) / self.grid_h
            else:
                neighbor_centroid_x = 0.5
                neighbor_centroid_y = 0.5
        else:
            cell_w = cell_h = 0.0
            placed_neighbor_ratio = 0.0
            neighbor_centroid_x = neighbor_centroid_y = 0.5

        placed_ratio = self.step_idx / self.num_cells

        extras = np.array([
            cell_w, cell_h,
            neighbor_centroid_x, neighbor_centroid_y,
            placed_neighbor_ratio, placed_ratio
        ], dtype=np.float32)

        return np.concatenate([density_flat, extras])

    def final_hpwl(self):
        """Full HPWL recomputation for the completed placement (sanity check / final report)."""
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