"""
AutoPRA v2 — Simulated Annealing Baseline
==========================================
Classical SA placement baseline for comparison with PPO.

Simulated Annealing works by:
  1. Start with random legal placement
  2. Pick random cell, try moving to random new position
  3. Accept move if it improves HPWL
  4. Accept worse moves with probability exp(-delta/T)
     where T = temperature (decreases over time)
  5. Return best placement found

SA is the standard classical baseline in all placement papers.
It achieves better HPWL than RL on single instances
but requires 21-222 seconds per design vs RL's <1 second inference.
"""

import numpy as np
import math
import time
from collections import defaultdict


def _check_overlap(density_map, cell, x, y, cell_w, cell_h, gw, gh):
    w, h = cell_w[cell], cell_h[cell]
    for dx in range(w):
        for dy in range(h):
            nx, ny = x + dx, y + dy
            if nx < gw and ny < gh:
                if density_map[nx][ny] >= 1.0:
                    return True
    return False


def _place_cell(density_map, cell, x, y, cell_w, cell_h, gw, gh):
    w, h = cell_w[cell], cell_h[cell]
    for dx in range(w):
        for dy in range(h):
            nx, ny = x + dx, y + dy
            if nx < gw and ny < gh:
                density_map[nx][ny] += 1.0
    return (x + w / 2.0, y + h / 2.0)


def _remove_cell(density_map, cell, x, y, cell_w, cell_h, gw, gh):
    w, h = cell_w[cell], cell_h[cell]
    for dx in range(w):
        for dy in range(h):
            nx, ny = x + dx, y + dy
            if nx < gw and ny < gh:
                density_map[nx][ny] -= 1.0


def _compute_hpwl(cell_pos, nets):
    total = 0.0
    for net in nets:
        placed = [cell_pos[c] for c in net if c in cell_pos]
        if len(placed) < 2:
            continue
        xs = [p[0] for p in placed]
        ys = [p[1] for p in placed]
        total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return total


def run_sa(nets, num_cells, cell_w, cell_h, grid_w, grid_h,
           T_start=50.0, T_end=0.01, cooling=0.9998,
           max_iter=150_000, seed=42, verbose=True):
    """
    Run Simulated Annealing placement.

    Args:
        nets      (list[list]) : netlist
        num_cells (int)        : number of cells
        cell_w    (np.ndarray) : cell widths
        cell_h    (np.ndarray) : cell heights
        grid_w    (int)        : grid width
        grid_h    (int)        : grid height
        T_start   (float)      : initial temperature
        T_end     (float)      : stopping temperature
        cooling   (float)      : temperature decay per iteration
        max_iter  (int)        : maximum iterations
        seed      (int)        : random seed
        verbose   (bool)       : print progress

    Returns:
        dict:
            best_hpwl (float) : best HPWL achieved
            best_pos  (dict)  : cell -> (cx, cy) best placement
            sa_time   (float) : runtime in seconds
    """
    rng = np.random.default_rng(seed)

    # Initial random legal placement
    density_map = np.zeros((grid_w, grid_h))
    cell_pos    = {}
    positions   = {}  # cell -> (x, y) top-left

    for cell in range(num_cells):
        placed, attempts = False, 0
        while not placed and attempts < 1000:
            x = int(rng.integers(0, grid_w - cell_w[cell] + 1))
            y = int(rng.integers(0, grid_h - cell_h[cell] + 1))
            if not _check_overlap(density_map, cell, x, y,
                                   cell_w, cell_h, grid_w, grid_h):
                cx, cy = _place_cell(density_map, cell, x, y,
                                      cell_w, cell_h, grid_w, grid_h)
                cell_pos[cell]  = (cx, cy)
                positions[cell] = (x, y)
                placed = True
            attempts += 1

    current_hpwl = _compute_hpwl(cell_pos, nets)
    best_hpwl    = current_hpwl
    best_pos     = dict(cell_pos)
    T = T_start

    start = time.time()

    for i in range(max_iter):
        # Pick random cell and try random new position
        cell     = int(rng.integers(0, num_cells))
        old_x, old_y = positions[cell]
        new_x = int(rng.integers(0, grid_w - cell_w[cell] + 1))
        new_y = int(rng.integers(0, grid_h - cell_h[cell] + 1))

        # Temporarily remove cell
        _remove_cell(density_map, cell, old_x, old_y,
                     cell_w, cell_h, grid_w, grid_h)

        if not _check_overlap(density_map, cell, new_x, new_y,
                               cell_w, cell_h, grid_w, grid_h):
            # Try new position
            old_cp = cell_pos[cell]
            cx, cy = _place_cell(density_map, cell, new_x, new_y,
                                  cell_w, cell_h, grid_w, grid_h)
            cell_pos[cell]  = (cx, cy)
            positions[cell] = (new_x, new_y)

            new_hpwl = _compute_hpwl(cell_pos, nets)
            delta    = new_hpwl - current_hpwl

            # Accept or reject
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_hpwl = new_hpwl
                if new_hpwl < best_hpwl:
                    best_hpwl = new_hpwl
                    best_pos  = dict(cell_pos)
            else:
                # Revert
                _remove_cell(density_map, cell, new_x, new_y,
                              cell_w, cell_h, grid_w, grid_h)
                _place_cell(density_map, cell, old_x, old_y,
                             cell_w, cell_h, grid_w, grid_h)
                cell_pos[cell]  = old_cp
                positions[cell] = (old_x, old_y)
        else:
            # Revert removal
            _place_cell(density_map, cell, old_x, old_y,
                         cell_w, cell_h, grid_w, grid_h)

        T *= cooling

        if verbose and (i + 1) % 30000 == 0:
            print(f"  Iter {i+1:>7,} | T: {T:.4f} | "
                  f"HPWL: {current_hpwl:.1f} | Best: {best_hpwl:.1f}")

        if T < T_end:
            if verbose:
                print(f"  Converged at iter {i+1:,}")
            break

    sa_time = time.time() - start

    return {
        'best_hpwl': best_hpwl,
        'best_pos' : best_pos,
        'sa_time'  : sa_time,
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from v2.benchmarks.mcnc_cm138a import (
        load_mcnc_cm138a, get_grid_size
    )

    print("Running SA on MCNC cm138a...")
    bm     = load_mcnc_cm138a(seed=42)
    grid_w, grid_h = get_grid_size(bm)

    result = run_sa(
        nets=bm['nets'],
        num_cells=bm['num_cells'],
        cell_w=bm['cell_w'],
        cell_h=bm['cell_h'],
        grid_w=grid_w,
        grid_h=grid_h,
        verbose=True
    )

    print(f"\nSA Results:")
    print(f"  Best HPWL : {result['best_hpwl']:.1f}")
    print(f"  Time      : {result['sa_time']:.1f}s")