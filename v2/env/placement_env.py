"""
AutoPRA v2 — Congestion-Aware Placement Environment
====================================================
Gymnasium environment for VLSI cell placement with:
  - Mixed cell sizes (real standard cell library footprints)
  - Congestion tracking via density map
  - Signal integrity awareness via reward shaping
  - Incremental HPWL for efficient large-scale computation
  - Bin utilization penalty to prevent corner clustering
  - Hard overlap penalty to prevent reward hacking

Observation Space (582 values):
  [0:576]  density map (24x24 or 28x28 flattened)
           — how full each grid slot is (0=empty, 1=occupied)
  [576]    current cell width (normalized by max width)
  [577]    current cell height (normalized by max height)
  [578]    neighbor centroid X (average X of placed connected cells)
  [579]    neighbor centroid Y (average Y of placed connected cells)
  [580]    placed neighbor ratio (fraction of neighbors placed)
  [581]    overall placed ratio (episode progress)

Action Space:
  Discrete(grid_w * grid_h) — which grid slot to place cell into

Reward Function:
  reward = delta_HPWL
         - 2.0 * congestion_overflow
         - 2.0 * bin_utilization_penalty
         - 5.0 * overlap_penalty

  delta_HPWL             : normalized HPWL change this step
  congestion_overflow    : fraction of slots with density > 1.0
  bin_utilization_penalty: max(0, max_quadrant_fraction - 0.40)
  overlap_penalty        : 5.0 if agent chose occupied slot, else 0
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict


class AutoPRAEnv(gym.Env):
    """
    VLSI Cell Placement Environment for AutoPRA v2.

    Supports both MCNC cm138a and CircuitNet RISC-V benchmarks.
    Cell sizes vary (1x1 to 3x2) matching real standard cell libraries.
    """

    # Reward penalty weights
    LAMBDA_CONG    = 2.0   # congestion overflow penalty weight
    LAMBDA_UTIL    = 2.0   # bin utilization penalty weight
    OVERLAP_PENALTY = 5.0  # hard penalty for placing on occupied slot

    def __init__(self, nets, adjacency, cell_to_nets,
                 cell_w, cell_h, num_cells, grid_w, grid_h):
        """
        Initialize the placement environment.

        Args:
            nets         (list[list]): netlist — each net is list of cell ids
            adjacency    (list[set]) : adj[cell] = set of connected cells
            cell_to_nets (dict)      : cell -> list of net indices (for incremental HPWL)
            cell_w       (np.ndarray): width of each cell in grid units
            cell_h       (np.ndarray): height of each cell in grid units
            num_cells    (int)       : total number of cells to place
            grid_w       (int)       : placement grid width
            grid_h       (int)       : placement grid height
        """
        super().__init__()

        self.nets         = nets
        self.adjacency    = adjacency
        self.cell_to_nets = cell_to_nets
        self.cell_w       = cell_w
        self.cell_h       = cell_h
        self.num_cells    = num_cells
        self.grid_w       = grid_w
        self.grid_h       = grid_h

        # Precompute placement order once (topological ordering)
        # — most-connected cell first, then most-connected-to-placed next
        # — ensures agent gets meaningful HPWL signal from step 1
        self.placement_order = self._compute_placement_order()

        # Observation: density map + 6 connectivity features
        obs_size = grid_w * grid_h + 6
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )

        # Action: which grid slot (flattened index)
        self.action_space = spaces.Discrete(grid_w * grid_h)

    def _compute_placement_order(self):
        """
        Compute connectivity-based placement order.

        Algorithm:
          1. Start with highest-degree cell (most nets)
          2. Next: cell most connected to already-placed cells
          3. Repeat until all cells ordered

        This is called topological ordering in EDA literature.
        It ensures the neighbor-centroid observation is meaningful
        from the very first placement step.

        Returns:
            list[int]: cell indices in placement order
        """
        degree = np.zeros(self.num_cells, dtype=int)
        for net in self.nets:
            for c in net:
                degree[c] += 1

        order    = []
        remaining = set(range(self.num_cells))
        placed   = set()

        # Start with highest degree cell
        first = int(np.argmax(degree))
        order.append(first)
        placed.add(first)
        remaining.remove(first)

        # Greedily add most-connected-to-placed cell
        while remaining:
            best_cell  = None
            best_score = -1
            for cell in remaining:
                score = sum(
                    1 for other in self.adjacency[cell]
                    if other in placed
                )
                if score > best_score:
                    best_score = score
                    best_cell  = cell
            order.append(best_cell)
            placed.add(best_cell)
            remaining.remove(best_cell)

        return order

    def reset(self, seed=None, options=None):
        """
        Reset environment to start of new placement episode.

        Returns:
            obs   (np.ndarray): initial observation
            info  (dict)      : empty info dict
        """
        super().reset(seed=seed)

        # density_map[x][y] = number of cells occupying slot (x,y)
        # In legal placement, all values should be 0 or 1
        self.density_map = np.zeros(
            (self.grid_w, self.grid_h), dtype=np.float32
        )

        self.cell_pos  = {}     # cell_id -> (cx, cy) center position
        self.order_idx = 0      # index into placement_order

        # Per-net HPWL for incremental computation
        # Updated only for nets connected to placed cell (efficient)
        self.net_hpwl = np.zeros(len(self.nets), dtype=np.float32)

        return self._get_obs(), {}

    def _get_obs(self):
        """
        Build observation vector.

        Returns:
            np.ndarray: shape (grid_w*grid_h + 6,)
        """
        # Density map — normalized by 4 (max reasonable cell stack)
        density_flat = np.clip(
            self.density_map.flatten() / 4.0, 0.0, 1.0
        )

        if self.order_idx < self.num_cells:
            cell = self.placement_order[self.order_idx]

            # Current cell size (normalized)
            cw = self.cell_w[cell] / 4.0
            ch = self.cell_h[cell] / 4.0

            # Neighbor centroid — where connected cells are placed
            # This tells the agent where to aim for this cell
            neighbors        = self.adjacency[cell]
            placed_neighbors = [
                self.cell_pos[n] for n in neighbors
                if n in self.cell_pos
            ]

            if placed_neighbors:
                cx = np.mean([p[0] for p in placed_neighbors]) / self.grid_w
                cy = np.mean([p[1] for p in placed_neighbors]) / self.grid_h
                nn = len(placed_neighbors) / max(1, len(neighbors))
            else:
                cx, cy, nn = 0.5, 0.5, 0.0
        else:
            cw, ch, cx, cy, nn = 0.0, 0.0, 0.5, 0.5, 0.0

        placed_ratio = len(self.cell_pos) / self.num_cells

        return np.concatenate(
            [density_flat, [cw, ch, cx, cy, nn, placed_ratio]],
            dtype=np.float32
        )

    def _check_overlap(self, cell, x, y):
        """
        Check if placing cell at (x,y) would overlap existing cells.

        Args:
            cell (int): cell index
            x    (int): top-left grid X
            y    (int): top-left grid Y

        Returns:
            bool: True if overlap would occur
        """
        w, h = self.cell_w[cell], self.cell_h[cell]
        for dx in range(w):
            for dy in range(h):
                nx, ny = x + dx, y + dy
                if nx < self.grid_w and ny < self.grid_h:
                    if self.density_map[nx][ny] >= 1.0:
                        return True
        return False

    def _find_nearest_free(self, cell, x, y):
        """
        Find nearest slot with no overlap for this cell.

        Used when agent picks an occupied slot — snaps to nearest
        legal position and applies overlap penalty to reward.

        Args:
            cell (int): cell index
            x    (int): preferred X
            y    (int): preferred Y

        Returns:
            tuple or None: (x, y) of nearest free slot
        """
        w, h = self.cell_w[cell], self.cell_h[cell]
        best      = None
        best_dist = float('inf')

        for tx in range(self.grid_w - w + 1):
            for ty in range(self.grid_h - h + 1):
                if not self._check_overlap(cell, tx, ty):
                    dist = abs(tx - x) + abs(ty - y)
                    if dist < best_dist:
                        best_dist = dist
                        best      = (tx, ty)

        return best

    def _place_cell(self, cell, x, y):
        """
        Place cell at top-left position (x, y) on the grid.

        Updates density_map and records cell center position.

        Args:
            cell (int): cell index
            x    (int): top-left X
            y    (int): top-left Y
        """
        w, h = self.cell_w[cell], self.cell_h[cell]
        for dx in range(w):
            for dy in range(h):
                nx, ny = x + dx, y + dy
                if nx < self.grid_w and ny < self.grid_h:
                    self.density_map[nx][ny] += 1.0

        # Store cell center (used for HPWL computation)
        self.cell_pos[cell] = (x + w / 2.0, y + h / 2.0)

    def _update_net_hpwl(self, cell):
        """
        Incrementally update HPWL for nets connected to placed cell.

        Instead of recomputing all nets every step (O(nets) per step),
        only update nets containing the just-placed cell.
        For 8,707 nets and avg 5 nets per cell, this is ~1000x faster.

        Args:
            cell (int): cell just placed

        Returns:
            float: total HPWL change across affected nets
        """
        delta = 0.0
        for net_idx in self.cell_to_nets[cell]:
            net    = self.nets[net_idx]
            placed = [self.cell_pos[c] for c in net if c in self.cell_pos]

            if len(placed) < 2:
                continue

            xs = [p[0] for p in placed]
            ys = [p[1] for p in placed]
            new_hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))

            delta += new_hpwl - self.net_hpwl[net_idx]
            self.net_hpwl[net_idx] = new_hpwl

        return delta

    def _congestion(self):
        """
        Compute congestion overflow fraction.

        Congestion = fraction of grid slots with density > 1.0
        (i.e., two or more cells overlapping — illegal in real chips).

        Returns:
            float: congestion overflow (0.0 = no congestion)
        """
        overflow = np.sum(self.density_map > 1.0)
        return float(overflow) / (self.grid_w * self.grid_h)

    def _bin_utilization_penalty(self):
        """
        Compute bin utilization penalty to prevent corner clustering.

        Divides grid into 4 quadrants. If any quadrant holds more
        than 40% of placed cells, apply a penalty proportional to
        the excess. This prevents the agent from packing all cells
        into one corner while still allowing HPWL-driven clustering.

        Returns:
            float: penalty (0.0 if all quadrants balanced)
        """
        hw = self.grid_w // 2
        hh = self.grid_h // 2

        q1 = np.sum(self.density_map[:hw, :hh])
        q2 = np.sum(self.density_map[hw:, :hh])
        q3 = np.sum(self.density_map[:hw, hh:])
        q4 = np.sum(self.density_map[hw:, hh:])

        total    = max(1.0, q1 + q2 + q3 + q4)
        max_frac = max(q1, q2, q3, q4) / total

        return max(0.0, max_frac - 0.40)

    def step(self, action):
        """
        Place one cell at the chosen grid slot.

        Args:
            action (int): flattened grid slot index

        Returns:
            obs     (np.ndarray): new observation
            reward  (float)     : step reward
            done    (bool)      : True when all cells placed
            trunc   (bool)      : always False
            info    (dict)      : empty
        """
        cell = self.placement_order[self.order_idx]
        w    = self.cell_w[cell]
        h    = self.cell_h[cell]

        # Decode action to (x, y) top-left position
        x = min(int(action) % self.grid_w,  self.grid_w - w)
        y = min(int(action) // self.grid_w, self.grid_h - h)

        # Handle overlap — snap to nearest free slot and penalize
        if self._check_overlap(cell, x, y):
            free = self._find_nearest_free(cell, x, y)
            if free:
                x, y = free
            overlap_pen = self.OVERLAP_PENALTY
        else:
            overlap_pen = 0.0

        # Place cell on grid
        self._place_cell(cell, x, y)
        self.order_idx += 1

        # Incremental HPWL update (efficient)
        hpwl_delta = self._update_net_hpwl(cell)
        norm_delta  = hpwl_delta / (self.grid_w + self.grid_h)

        # Compute penalty terms
        curr_cong = self._congestion()
        util_pen  = self._bin_utilization_penalty()

        # Combined reward
        reward = (
            -norm_delta                            # minimize HPWL increase
            - self.LAMBDA_CONG * curr_cong         # minimize congestion
            - self.LAMBDA_UTIL * util_pen          # prevent corner clustering
            - overlap_pen                          # prevent illegal placements
        )

        done = (self.order_idx >= self.num_cells)
        return self._get_obs(), reward, done, False, {}

    def final_hpwl(self):
        """
        Return total HPWL of current placement.

        Returns:
            float: sum of per-net HPWL values
        """
        return float(np.sum(self.net_hpwl))

    def final_congestion(self):
        """
        Return congestion overflow fraction of current placement.

        Returns:
            float: fraction of slots with density > 1.0
        """
        return self._congestion()

    def get_cell_pos(self):
        """
        Return cell position dictionary.

        Returns:
            dict: cell_id -> (cx, cy) center position
        """
        return dict(self.cell_pos)

    def get_density_map(self):
        """
        Return copy of density map.

        Returns:
            np.ndarray: shape (grid_w, grid_h)
        """
        return self.density_map.copy()