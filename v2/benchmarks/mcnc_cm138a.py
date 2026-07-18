"""
mcnc_cm138a.py

Loader for the MCNC 'cm138a' benchmark netlist, used in AutoPRA v2
as a real (non-synthetic) placement benchmark.

cm138a is a classic small combinational-logic MCNC benchmark circuit.
This module parses the benchmark's netlist file into the same
adjacency-dict format used by AutoPRAEnv, so it plugs directly into
the v2 training pipeline.

Note: You'll need the actual MCNC cm138a benchmark file (commonly
distributed as a .net or .txt netlist file) placed in
v2/benchmarks/data/cm138a.net for this loader to work. This module
only handles parsing/loading, not distribution of the benchmark file
itself (MCNC benchmarks are widely available from academic EDA
benchmark archives).
"""

import os


DEFAULT_BENCHMARK_PATH = os.path.join(
    os.path.dirname(__file__), "data", "cm138a.net"
)

# cm138a has a small, known cell count from the MCNC benchmark suite
NUM_CELLS = 137  # adjust if your specific netlist file differs
GRID_W = 12
GRID_H = 12


def load_cm138a(filepath=DEFAULT_BENCHMARK_PATH):
    """
    Parses the cm138a netlist file into an adjacency dict.

    Expected input format (simplified net-list style): each line
    lists a net followed by the cell IDs/names connected to it,
    e.g.:
        net1: cell_0 cell_3 cell_7
        net2: cell_1 cell_3

    Returns:
        dict[int, list[int]] mapping cell_id -> list of connected cell_ids
        (same format as PlacementEnv/AutoPRAEnv adjacency)

    Raises:
        FileNotFoundError if the benchmark file isn't present at filepath.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"MCNC cm138a benchmark file not found at {filepath}. "
            "Download the cm138a netlist from an MCNC benchmark archive "
            "and place it at this path."
        )

    cell_name_to_id = {}
    adjacency = {}

    def get_cell_id(name):
        if name not in cell_name_to_id:
            new_id = len(cell_name_to_id)
            cell_name_to_id[name] = new_id
            adjacency[new_id] = []
        return cell_name_to_id[name]

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # expected format: "net_name: cell_a cell_b cell_c ..."
            if ":" not in line:
                continue

            _, cells_str = line.split(":", 1)
            cell_names = cells_str.split()

            cell_ids = [get_cell_id(name) for name in cell_names]

            # connect every pair of cells in this net
            for i in range(len(cell_ids)):
                for j in range(i + 1, len(cell_ids)):
                    a, b = cell_ids[i], cell_ids[j]
                    if b not in adjacency[a]:
                        adjacency[a].append(b)
                    if a not in adjacency[b]:
                        adjacency[b].append(a)

    return adjacency


def load_cm138a_as_benchmark_dict(filepath=DEFAULT_BENCHMARK_PATH):
    """
    Convenience wrapper returning the format expected by
    AutoPRAEnv(benchmark_dict, adjacency, grid_w, grid_h).

    Returns:
        tuple(benchmark_dict, adjacency, grid_w, grid_h)
    """
    adjacency = load_cm138a(filepath)
    num_cells = len(adjacency)

    benchmark_dict = {
        "name": "cm138a",
        "num_cells": num_cells,
        "source": "MCNC",
    }

    return benchmark_dict, adjacency, GRID_W, GRID_H


if __name__ == "__main__":
    adjacency = load_cm138a()
    print(f"Loaded cm138a: {len(adjacency)} cells")
    total_edges = sum(len(v) for v in adjacency.values()) // 2
    print(f"Total unique connections: {total_edges}")
