"""
AutoPRA v2 — CircuitNet RISC-V Benchmark Loader
================================================
Loads real chip data from the CircuitNet dataset:
  github.com/circuitnet/CircuitNet

Design used: RISCY-a-1-c2
  - A real RISC-V processor designed using professional EDA tools
  - 53,586 cells, 54,734 nets
  - Real placement coordinates from Cadence/Synopsys EDA tool runs

We select the top-250 most-connected cells for RL training
(full 53K cells would be too large for Colab-scale training).
This is standard practice in academic placement research.

Data files used from CircuitNet:
  - node_attr: cell names and types
  - net_attr:  net names
  - pin_attr:  pin-to-cell and pin-to-net mappings
  - instance_placement_gcell: real EDA tool placement coordinates
"""

import numpy as np
import os
from collections import defaultdict


DESIGN        = "RISCY-a-1-c2"
N_CELLS       = 250     # number of cells to select for training
GRID_W        = 28      # placement grid width
GRID_H        = 28      # placement grid height


def clone_circuitnet(target_dir="CircuitNet"):
    """
    Clone CircuitNet repository if not already present.

    Args:
        target_dir (str): local directory name for the repo
    """
    if not os.path.exists(target_dir):
        print("Cloning CircuitNet repository...")
        os.system(
            f"git clone https://github.com/circuitnet/CircuitNet.git "
            f"{target_dir} --depth 1 -q"
        )
        print("✅ CircuitNet cloned.")
    else:
        print("✅ CircuitNet already available.")


def _parse_pin_attr_column(col):
    """
    Parse a pin_attr column safely.

    pin_attr columns have dtype=object and may contain
    ints or lists — handle both cases.

    Args:
        col (np.ndarray): raw column from pin_attr

    Returns:
        np.ndarray: integer array
    """
    return np.array(
        [x[0] if isinstance(x, (list, np.ndarray)) else int(x)
         for x in col],
        dtype=int
    )


def load_circuitnet_data(base_dir="CircuitNet"):
    """
    Load raw graph features from CircuitNet repository.

    Args:
        base_dir (str): path to cloned CircuitNet directory

    Returns:
        dict with keys:
            cell_names     (np.ndarray): cell name strings
            net_names      (np.ndarray): net name strings
            pin_cell_idx   (np.ndarray): cell index per pin
            pin_net_idx    (np.ndarray): net index per pin
            placement      (dict)      : cell_name -> [x1,y1,x2,y2]
    """
    graph_dir     = os.path.join(base_dir, "build_graph_demo",
                                  "graph_information")
    placement_dir = os.path.join(base_dir, "build_graph_demo",
                                  "instance_placement_gcell")

    # Load graph features
    node_attr = np.load(
        os.path.join(graph_dir, "node_attr", f"{DESIGN}_node_attr.npy"),
        allow_pickle=True
    )
    net_attr = np.load(
        os.path.join(graph_dir, "net_attr", f"{DESIGN}_net_attr.npy"),
        allow_pickle=True
    )
    pin_attr = np.load(
        os.path.join(graph_dir, "pin_attr", f"{DESIGN}_pin_attr.npy"),
        allow_pickle=True
    )

    # Find placement file for this design
    placement_files = [
        f for f in os.listdir(placement_dir)
        if DESIGN.replace("-1-c2", "") in f
        and "RISCY-a" in f
        and "FPU" not in f
    ]
    if not placement_files:
        raise FileNotFoundError(
            f"No placement file found for {DESIGN} in {placement_dir}"
        )

    placement = np.load(
        os.path.join(placement_dir, placement_files[0]),
        allow_pickle=True
    ).item()

    # Parse pin attributes
    pin_cell_idx = _parse_pin_attr_column(pin_attr[1])
    pin_net_idx  = _parse_pin_attr_column(pin_attr[2])

    print(f"✅ CircuitNet loaded: {DESIGN}")
    print(f"   Cells: {len(node_attr[0]):,} | "
          f"Nets: {len(net_attr[0]):,} | "
          f"Pins: {len(pin_cell_idx):,}")

    return {
        'cell_names'   : node_attr[0],
        'net_names'    : net_attr[0],
        'pin_cell_idx' : pin_cell_idx,
        'pin_net_idx'  : pin_net_idx,
        'placement'    : placement,
    }


def build_full_netlist(raw_data):
    """
    Reconstruct netlist from pin attributes.

    Args:
        raw_data (dict): output of load_circuitnet_data()

    Returns:
        list[list]: each net is list of cell indices (valid only)
    """
    cell_names   = raw_data['cell_names']
    net_names    = raw_data['net_names']
    pin_cell_idx = raw_data['pin_cell_idx']
    pin_net_idx  = raw_data['pin_net_idx']
    num_cells    = len(cell_names)

    # Build net -> set of cell indices
    net_to_cells = defaultdict(set)
    for i in range(len(pin_cell_idx)):
        net_to_cells[pin_net_idx[i]].add(pin_cell_idx[i])

    # Filter: valid cell indices only, at least 2 pins per net
    full_nets = []
    for net_idx in range(len(net_names)):
        cells = [c for c in net_to_cells[net_idx] if c < num_cells]
        if len(cells) >= 2:
            full_nets.append(cells)

    print(f"✅ Full netlist: {len(full_nets):,} valid nets")
    return full_nets


def select_top_cells(raw_data, full_nets, n_cells=N_CELLS):
    """
    Select top-N most-connected cells that have placement data.

    We pick the most-connected cells because:
    1. They represent the critical path of the design
    2. Their placement most significantly impacts wirelength
    3. Standard practice in academic placement subset selection

    Args:
        raw_data  (dict)      : output of load_circuitnet_data()
        full_nets (list[list]): full netlist
        n_cells   (int)       : number of cells to select

    Returns:
        list[int]: selected cell indices (original indexing)
    """
    cell_names    = raw_data['cell_names']
    placement     = raw_data['placement']
    num_cells     = len(cell_names)
    placed_names  = set(placement.keys())

    # Compute degree of each cell
    cell_degree = np.zeros(num_cells, dtype=int)
    for net in full_nets:
        for c in net:
            cell_degree[c] += 1

    # Select top-N with placement data
    selected = []
    for idx in np.argsort(cell_degree)[::-1]:
        if cell_names[idx] in placed_names:
            selected.append(int(idx))
        if len(selected) == n_cells:
            break

    print(f"✅ Selected {len(selected)} cells")
    print(f"   Degree range: {cell_degree[selected[0]]} "
          f"(max) → {cell_degree[selected[-1]]} (min of selected)")

    return selected


def build_subset(raw_data, full_nets, selected_cells):
    """
    Build subset netlist for selected cells.

    Keeps all nets where at least 2 selected cells appear.
    No degree cap — all connectivity is preserved.

    Args:
        raw_data       (dict)      : output of load_circuitnet_data()
        full_nets      (list[list]): full netlist
        selected_cells (list[int]) : selected cell indices

    Returns:
        dict with keys:
            subset_nets  (list[list]) : remapped netlist
            old_to_new   (dict)       : original idx -> new idx mapping
            cell_w       (np.ndarray) : cell widths (normalized)
            cell_h       (np.ndarray) : cell heights (normalized)
            real_cell_pos(dict)       : cell -> (x,y) from real EDA
            real_hpwl    (float)      : HPWL of real EDA placement
    """
    cell_names    = raw_data['cell_names']
    placement     = raw_data['placement']

    selected_set = set(selected_cells)
    old_to_new   = {old: new for new, old in enumerate(selected_cells)}

    # Build subset netlist
    subset_nets = []
    for net in full_nets:
        sub = [old_to_new[c] for c in net if c in selected_set]
        if len(sub) >= 2:
            subset_nets.append(sub)

    # Real cell sizes from EDA tool placement
    raw_w = np.array([
        placement[cell_names[idx]][2] - placement[cell_names[idx]][0]
        for idx in selected_cells
    ])
    raw_h = np.array([
        placement[cell_names[idx]][3] - placement[cell_names[idx]][1]
        for idx in selected_cells
    ])

    # Normalize: smallest cell = 1 grid unit
    min_w  = max(1, int(raw_w.min()))
    min_h  = max(1, int(raw_h.min()))
    norm_w = np.clip(np.round(raw_w / min_w).astype(int), 1, 4)
    norm_h = np.clip(np.round(raw_h / min_h).astype(int), 1, 4)

    # Real EDA placement coordinates (normalized to grid)
    real_x = np.array([
        (placement[cell_names[idx]][0] + placement[cell_names[idx]][2]) / 2
        for idx in selected_cells
    ])
    real_y = np.array([
        (placement[cell_names[idx]][1] + placement[cell_names[idx]][3]) / 2
        for idx in selected_cells
    ])

    norm_rx = np.round(
        (real_x - real_x.min()) / (real_x.max() - real_x.min()) * (GRID_W - 1)
    ).astype(int)
    norm_ry = np.round(
        (real_y - real_y.min()) / (real_y.max() - real_y.min()) * (GRID_H - 1)
    ).astype(int)

    real_cell_pos = {
        i: (int(norm_rx[i]), int(norm_ry[i]))
        for i in range(len(selected_cells))
    }

    # Compute real EDA HPWL (gold standard)
    real_hpwl = _compute_hpwl(real_cell_pos, subset_nets)

    print(f"✅ Subset ready: {len(selected_cells)} cells, "
          f"{len(subset_nets)} nets")
    print(f"   Real EDA HPWL (gold standard): {real_hpwl:.1f}")

    return {
        'subset_nets'  : subset_nets,
        'old_to_new'   : old_to_new,
        'cell_w'       : norm_w,
        'cell_h'       : norm_h,
        'real_cell_pos': real_cell_pos,
        'real_hpwl'    : real_hpwl,
    }


def build_adjacency(nets, num_cells):
    """
    Build adjacency list from netlist.

    Args:
        nets      (list[list]): netlist
        num_cells (int)       : total cells

    Returns:
        list[set]: adj[cell] = set of connected cells
    """
    adj = [set() for _ in range(num_cells)]
    for net in nets:
        for c in net:
            for other in net:
                if other != c:
                    adj[c].add(other)
    return adj


def build_cell_to_nets(nets, num_cells):
    """
    Build cell -> list of net indices mapping.
    Used for incremental HPWL computation.

    Args:
        nets      (list[list]): netlist
        num_cells (int)       : total cells

    Returns:
        dict[int, list[int]]: cell -> net indices
    """
    cell_to_nets = defaultdict(list)
    for net_idx, net in enumerate(nets):
        for c in net:
            cell_to_nets[c].append(net_idx)
    return cell_to_nets


def _compute_hpwl(cell_pos, nets):
    """
    Compute total HPWL across all nets.

    HPWL (Half Perimeter Wire Length) = sum over all nets of
    (max_x - min_x) + (max_y - min_y) of placed cells in net.

    Args:
        cell_pos (dict): cell -> (x, y) position
        nets     (list): netlist

    Returns:
        float: total HPWL
    """
    total = 0.0
    for net in nets:
        placed = [cell_pos[c] for c in net if c in cell_pos]
        if len(placed) < 2:
            continue
        xs = [p[0] for p in placed]
        ys = [p[1] for p in placed]
        total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return total


def load_circuitnet_benchmark(base_dir="CircuitNet"):
    """
    Full pipeline: clone → load → build netlist → select cells → build subset.

    This is the main entry point for the CircuitNet benchmark.

    Args:
        base_dir (str): path to CircuitNet directory

    Returns:
        dict with all benchmark data needed by AutoPRAEnv:
            num_cells    (int)
            grid_w       (int)
            grid_h       (int)
            nets         (list[list])
            cell_w       (np.ndarray)
            cell_h       (np.ndarray)
            adjacency    (list[set])
            cell_to_nets (dict)
            real_cell_pos(dict)
            real_hpwl    (float)
    """
    clone_circuitnet(base_dir)
    raw_data   = load_circuitnet_data(base_dir)
    full_nets  = build_full_netlist(raw_data)
    selected   = select_top_cells(raw_data, full_nets, n_cells=N_CELLS)
    subset     = build_subset(raw_data, full_nets, selected)

    nets         = subset['subset_nets']
    num_cells    = len(selected)
    adjacency    = build_adjacency(nets, num_cells)
    cell_to_nets = build_cell_to_nets(nets, num_cells)

    return {
        'num_cells'    : num_cells,
        'grid_w'       : GRID_W,
        'grid_h'       : GRID_H,
        'nets'         : nets,
        'cell_w'       : subset['cell_w'],
        'cell_h'       : subset['cell_h'],
        'adjacency'    : adjacency,
        'cell_to_nets' : cell_to_nets,
        'real_cell_pos': subset['real_cell_pos'],
        'real_hpwl'    : subset['real_hpwl'],
    }


if __name__ == "__main__":
    benchmark = load_circuitnet_benchmark()
    print(f"\nBenchmark ready for training!")
    print(f"  Cells     : {benchmark['num_cells']}")
    print(f"  Nets      : {len(benchmark['nets'])}")
    print(f"  Grid      : {benchmark['grid_w']}x{benchmark['grid_h']}")
    print(f"  Real HPWL : {benchmark['real_hpwl']:.1f}")