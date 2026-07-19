"""
AutoPRA v2 — Signal Integrity + Extended Metrics
=================================================
Computes 5 signal integrity metrics post-placement:

1. Total HPWL        — primary placement quality metric
2. Critical HPWL     — HPWL of high-fanout nets (degree > 3)
3. Non-critical HPWL — HPWL of regular signal nets (degree <= 3)
4. Max net length    — longest single wire (SI risk proxy)
5. SI violation %    — % nets exceeding length threshold
6. Weighted HPWL     — critical nets counted 2x in total score

Signal integrity in chip design:
  Long wires cause signal degradation, timing violations,
  and crosstalk. Minimizing max net length and SI violations
  directly improves chip signal integrity.

AutoPRA v2 results on CircuitNet RISC-V:
  - PPO achieves 57% fewer SI violations vs random
  - PPO achieves lowest max net length (24.5) — better than
    Real EDA tool (42.0) and SA (33.5)
"""

import numpy as np
from collections import defaultdict


# Nets longer than this threshold are SI violations
SI_THRESHOLD = 10  # grid units


def compute_si_metrics(cell_pos, nets, si_threshold=SI_THRESHOLD):
    """
    Compute all signal integrity and placement quality metrics.

    Args:
        cell_pos     (dict)      : cell -> (cx, cy) placement
        nets         (list[list]): netlist
        si_threshold (int)       : max acceptable net length

    Returns:
        dict:
            total_hpwl       (float): total HPWL across all nets
            critical_hpwl    (float): HPWL of high-fanout nets (degree>3)
            noncritical_hpwl (float): HPWL of regular nets (degree<=3)
            max_net_length   (float): length of longest net
            si_violation_pct (float): % nets exceeding threshold
            weighted_hpwl    (float): critical nets weighted 2x
            valid_nets       (int)  : number of nets with >= 2 placed cells
            si_violations    (int)  : count of SI-violating nets
    """
    total_hpwl       = 0.0
    critical_hpwl    = 0.0
    noncritical_hpwl = 0.0
    weighted_hpwl    = 0.0
    max_net_length   = 0.0
    si_violations    = 0
    valid_nets       = 0

    for net in nets:
        placed = [cell_pos[c] for c in net if c in cell_pos]
        if len(placed) < 2:
            continue

        xs = [p[0] for p in placed]
        ys = [p[1] for p in placed]
        net_hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
        degree   = len(net)

        total_hpwl += net_hpwl
        valid_nets  += 1

        # Track longest net (SI proxy)
        if net_hpwl > max_net_length:
            max_net_length = net_hpwl

        # SI violation check
        if net_hpwl > si_threshold:
            si_violations += 1

        # Critical vs non-critical nets
        if degree > 3:
            # High-fanout net (clock, reset, bus) — more important
            critical_hpwl += net_hpwl
            weighted_hpwl += 2.0 * net_hpwl  # 2x weight
        else:
            noncritical_hpwl += net_hpwl
            weighted_hpwl    += 1.0 * net_hpwl

    si_violation_pct = (si_violations / max(1, valid_nets)) * 100

    return {
        'total_hpwl'      : total_hpwl,
        'critical_hpwl'   : critical_hpwl,
        'noncritical_hpwl': noncritical_hpwl,
        'max_net_length'  : max_net_length,
        'si_violation_pct': si_violation_pct,
        'weighted_hpwl'   : weighted_hpwl,
        'valid_nets'      : valid_nets,
        'si_violations'   : si_violations,
    }


def run_full_evaluation(methods_pos, nets, si_threshold=SI_THRESHOLD):
    """
    Run SI evaluation for multiple placement methods.

    Args:
        methods_pos  (dict): method_name -> cell_pos dict
        nets         (list): netlist
        si_threshold (int) : SI violation threshold

    Returns:
        dict: method_name -> si_metrics dict
    """
    results = {}
    for name, cell_pos in methods_pos.items():
        results[name] = compute_si_metrics(cell_pos, nets, si_threshold)
    return results


def print_si_table(results, rand_hpwl=None):
    """
    Print formatted SI metrics comparison table.

    Args:
        results   (dict) : output of run_full_evaluation()
        rand_hpwl (float): random baseline HPWL for improvement %
    """
    print(f"\n{'='*92}")
    print(f"Signal Integrity + Extended Metrics")
    print(f"{'='*92}")
    print(f"  {'Method':<18} {'HPWL':>8} {'vsRand':>7} "
          f"{'CritHPWL':>10} {'NonCrit':>9} "
          f"{'MaxNet':>8} {'SIViol%':>8} {'WtHPWL':>9}")
    print(f"  {'─'*87}")

    for name, m in results.items():
        if rand_hpwl and name != 'Real EDA (gold)':
            vs = f"{(rand_hpwl - m['total_hpwl']) / rand_hpwl * 100:.1f}%"
        else:
            vs = "—"

        print(f"  {name:<18} {m['total_hpwl']:>8.1f} {vs:>7} "
              f"{m['critical_hpwl']:>10.1f} {m['noncritical_hpwl']:>9.1f} "
              f"{m['max_net_length']:>8.1f} {m['si_violation_pct']:>7.1f}% "
              f"{m['weighted_hpwl']:>9.1f}")

    print(f"{'='*92}")
    print(f"\n  SI Threshold : {SI_THRESHOLD} grid units")
    print(f"  Critical nets: degree > 3 (high fanout)")
    print(f"  Weighted HPWL: critical nets counted 2x")