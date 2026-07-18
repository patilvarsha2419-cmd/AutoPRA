"""
circuitnet_loader.py

Loader for the CircuitNet RISC-V benchmark used in AutoPRA v2.

Specifically targets RISCY-a-1-c2, a 53,586-cell real chip design.
Since training on the full chip is impractical, this loader extracts
the top-250-cell subset (ranked by connectivity/degree) used for
v2's actual training runs, along with its 488 nets.

Note: You'll need the CircuitNet dataset files (openly available from
the CircuitNet project) placed under v2/benchmarks/data/circuitnet/
for this loader to work. This module only handles parsing/subsetting,
not distribution of the dataset itself.
"""

import os
import json
import numpy as np


DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "data", "circuitnet"
)

CHIP_NAME = "RISCY-a-1-c2"
FULL_CELL_COUNT = 53586
SUBSET_SIZE = 250
GRID_W = 28
GRID_H = 28


def _load_raw_netlist(data_dir=DEFAULT_DATA_DIR, chip_name=CHIP_NAME):
    """
    Loads the raw CircuitNet netlist/placement JSON for the given chip.

    Expected file: {data_dir}/{chip_name}.json containing:
        {
            "cells": [{"id": int, "width": float, "height": float}, ...],
            "nets": [{"id": int, "cells": [cell_id, cell_id, ...]}, ...]
        }

    Returns:
        tuple(cells: list[dict], nets: list[dict])
    """
    filepath = os.path.join(data_dir, f"{chip_name}.json")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"CircuitNet chip data not found at {filepath}. "
            "Download the CircuitNet RISC-V dataset and place the "
            f"{chip_name}.json file at this path."
        )

    with open(filepath, "r") as f:
        data = json.load(f)

    return data["cells"], data["nets"]


def _select_top_subset(cells, nets, subset_size=SUBSET_SIZE):
    """
    Selects the top `subset_size` cells ranked by connectivity degree
    (number of nets each cell participates in), then filters nets to
    only those fully contained within the subset.

    Returns:
        tuple(subset_cell_ids: list[int], subset_nets: list[dict])
    """
    degree = {cell["id"]: 0 for cell in cells}

    for net in nets:
        for cell_id in net["cells"]:
            if cell_id in degree:
                degree[cell_id] += 1

    # rank cells by degree, descending
    ranked_cell_ids = sorted(degree.keys(), key=lambda c: degree[c], reverse=True)
    subset_cell_ids = set(ranked_cell_ids[:subset_size])

    # keep only nets where ALL connected cells are in the subset
    subset_nets = [
        net for net in nets
        if all(c in subset_cell_ids for c in net["cells"])
    ]

    return list(subset_cell_ids), subset_nets


def load_circuitnet_subset(
    data_dir=DEFAULT_DATA_DIR,
    chip_name=CHIP_NAME,
    subset_size=SUBSET_SIZE,
):
    """
    Loads and returns the top-N-cell subset of the given CircuitNet
    chip, formatted for AutoPRAEnv's CircuitNet constructor:

        AutoPRAEnv(subset_nets, adjacency, cell_to_nets,
                   norm_w, norm_h, num_cells, grid_w, grid_h)

    Returns:
        dict with keys:
            "subset_nets": list of net dicts (id + cell list, re-indexed 0..N-1)
            "adjacency": dict[int, list[int]] cell -> connected cells
            "cell_to_nets": dict[int, list[int]] cell -> net ids it belongs to
            "norm_w": dict[int, float] normalized cell widths
            "norm_h": dict[int, float] normalized cell heights
            "num_cells": int
            "grid_w": int
            "grid_h": int
    """
    cells, nets = _load_raw_netlist(data_dir, chip_name)
    subset_cell_ids, subset_nets_raw = _select_top_subset(cells, nets, subset_size)

    # re-index subset cells to 0..N-1 for clean use in the env
    old_to_new_id = {old_id: new_id for new_id, old_id in enumerate(subset_cell_ids)}
    cells_by_id = {cell["id"]: cell for cell in cells}

    adjacency = {new_id: [] for new_id in old_to_new_id.values()}
    cell_to_nets = {new_id: [] for new_id in old_to_new_id.values()}
    subset_nets = []

    for net_idx, net in enumerate(subset_nets_raw):
        remapped_cells = [old_to_new_id[c] for c in net["cells"]]
        subset_nets.append({"id": net_idx, "cells": remapped_cells})

        for cell_id in remapped_cells:
            cell_to_nets[cell_id].append(net_idx)

        for i in range(len(remapped_cells)):
            for j in range(i + 1, len(remapped_cells)):
                a, b = remapped_cells[i], remapped_cells[j]
                if b not in adjacency[a]:
                    adjacency[a].append(b)
                if a not in adjacency[b]:
                    adjacency[b].append(a)

    # normalize cell width/height to [0, 1] range for the env's observation space
    widths = np.array([cells_by_id[old_id]["width"] for old_id in subset_cell_ids])
    heights = np.array([cells_by_id[old_id]["height"] for old_id in subset_cell_ids])

    max_w, max_h = widths.max(), heights.max()
    norm_w = {new_id: float(widths[new_id] / max_w) for new_id in old_to_new_id.values()}
    norm_h = {new_id: float(heights[new_id] / max_h) for new_id in old_to_new_id.values()}

    return {
        "subset_nets": subset_nets,
        "adjacency": adjacency,
        "cell_to_nets": cell_to_nets,
        "norm_w": norm_w,
        "norm_h": norm_h,
        "num_cells": len(subset_cell_ids),
        "grid_w": GRID_W,
        "grid_h": GRID_H,
    }


if __name__ == "__main__":
    result = load_circuitnet_subset()
    print(f"Loaded {result['num_cells']} cells, {len(result['subset_nets'])} nets")