"""Measurement driver for Task A4 (design-margin + OMS-composition SDD plan).

The plan's spec measured composition error (composed GSNR - actual/propagated
GSNR) only to 8 hops and extrapolated the +0.013 dB/hop rate to german_17's
longest-observed path (13 hops, seen in one 20-service placement run) for a
0.17 dB worst-case guess. Task A3 built the capture machinery
(`adapter._propagate_loading(..., capture_increments=True)` +
`gnpy_adapter/composition.py`); this script uses it to MEASURE the error out
to the topology's real maximum hop count instead of trusting the
extrapolation -- see CLAUDE.md's Requirements and the plan's own §7/§8.2.

Methodology, matching this task's brief:

1. Enumerate real OMS routing candidates with `model.solvers.compute_paths`
   -- the SAME k-shortest enumerator `solve_allocation`/`solve_rsa` use in
   production, not a synthetic path generator -- over every ordered node pair
   on german_17, and bucket them by hop count. This recovers the topology's
   TRUE maximum simple-path length directly (17 nodes -> up to 16 hops),
   rather than trusting the 13-hop figure that was only the longest path
   incidentally seen in one prior run.
2. Harvest each OMS's own per-slot `1/gsnr_lin` increment from a ONE-HOP
   propagation of that OMS ALONE (`capture_increments=True`). This is
   deliberately a *different* propagation from any multi-hop path under
   test, per the brief's anti-tautology requirement ("a different path that
   covers the same OMS") -- an isolated 1-hop propagation can never be the
   propagation being scored, for any path in the corpus. It is also the
   most conservative honest choice available: 1/gsnr_lin is measured from a
   common, path-independent FULL-comb launch condition, so an OMS's own
   contribution captured in isolation is its EARLIEST-position value -- and
   the design doc's own mechanism for the error (NLI/ASE-signal beating
   scaling with upstream power, so the SAME OMS contributes MORE noise later
   in a path) means composing every hop from its earliest-position value is
   exactly the scenario that exposes the largest such gap, not a scenario
   that happens to hide it.
3. For each candidate path, do one EXACT propagation (`capture_increments=
   True` again, for its own analytic endpoint term -- a pure function of
   that path's own terminal ROADMs/tx_osnr/baud_rate, so reusing it here
   cannot leak increment precision the way reusing increments would) and
   read back the penalty-applied per-slot GSNR alongside the composed value
   (isolated increments + this path's own endpoint term). Record the signed
   error `composed - actual` (positive = optimistic, the unsafe direction --
   see composition.py).

Run:
    PYTHONPATH="$(pwd)/src" conda run -n multilayer-optical-mcp python scripts/measure_composition_error.py
(conda run cannot take a newline-containing `-c` argument, hence a real file;
PYTHONPATH is required because this worktree shares its conda env's editable
install with a sibling worktree -- see this task's brief for why.)
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from multilayer_optical_network.data import reference_topology
from multilayer_optical_network.gnpy_adapter.adapter import (
    _apply_penalties, _extract_gsnr_osnr, _propagate_loading,
)
from multilayer_optical_network.gnpy_adapter.composition import compose_gsnr_db
from multilayer_optical_network.gnpy_adapter.loading import Channel, LoadingState
from multilayer_optical_network.model.assets import Direction
from multilayer_optical_network.model.modes import default_modes
from multilayer_optical_network.model.solvers import SolverStatus, compute_paths
from multilayer_optical_network.model.spectrum import SpectrumGrid
from multilayer_optical_network.model.topology_import import model_from_abstract_graph

MODE_ID = "400G@7.1dB"          # 87.5 GBaud, matches the spec's own measurement
K_PER_PAIR = 60                  # k-shortest candidates requested per ordered node pair
SAMPLES_PER_HOP = 30             # capped sample size per hop-count bucket
SEED = 0
DIRECTION = Direction.FORWARD    # composition identity is direction-agnostic; one is enough


def _full_comb_loading(grid: SpectrumGrid, mode_id: str) -> LoadingState:
    """Every grid slot lit with *mode_id* -- FillPolicy.FULL's comb, which is
    the only loading regime composition is valid under (see the plan's §2)."""
    return LoadingState(tuple(
        Channel(grid.freq(s), grid.spacing_hz, None, mode_id)
        for s in range(grid.num_slots)
    ))


def _all_node_ids(model) -> List[str]:
    ids = set()
    for oms in model.list_oms():
        ids.add(oms.src_node_id)
        ids.add(oms.dst_node_id)
    return sorted(ids)


def _enumerate_candidate_paths(model, k_per_pair: int) -> Dict[int, List[Tuple[str, ...]]]:
    """Bucket real OMS routing candidates by hop count, over every ordered
    node pair -- see module docstring point 1."""
    nodes = _all_node_ids(model)
    buckets: Dict[int, List[Tuple[str, ...]]] = defaultdict(list)
    for src in nodes:
        for dst in nodes:
            if src == dst:
                continue
            result = compute_paths(model, src, dst, k=k_per_pair)
            if result.status != SolverStatus.SOLUTION:
                continue
            for p in result.paths:
                buckets[len(p.oms_sequence)].append(p.oms_sequence)
    return buckets


def _harvest_isolated_increment(model, oms_id: str, direction: Direction, mode,
                                loading: LoadingState) -> Dict[int, float]:
    """slot -> this OMS's own 1/gsnr_lin increment, from a ONE-HOP propagation
    of oms_id alone -- see module docstring point 2."""
    pr = _propagate_loading(model, (oms_id,), direction, loading, mode,
                            probe_idx=0, capture_increments=True)
    return pr.oms_increments[oms_id]


def _exact_and_composed(
    model, oms_sequence: Tuple[str, ...], direction: Direction, mode,
    loading: LoadingState, increments_table: Dict[str, Dict[int, float]],
    grid: SpectrumGrid,
) -> Dict[int, Tuple[float, float]]:
    """One exact propagation of *oms_sequence* -> {slot: (exact_db, composed_db)}
    -- see module docstring point 3."""
    pr = _propagate_loading(model, oms_sequence, direction, loading, mode,
                            probe_idx=0, capture_increments=True)
    out: Dict[int, Tuple[float, float]] = {}
    for i, freq_hz in enumerate(pr.si.frequency):
        slot = grid.slot_of(float(freq_hz))
        gsnr_db, osnr_db = _extract_gsnr_osnr(pr.si, i)
        exact_db, _ = _apply_penalties(
            pr.si, i, pr.uids_list, pr.elements, pr.roadm_propagated,
            pr.baud_rate, gsnr_db, osnr_db)
        incs = [increments_table[o][slot] for o in oms_sequence]
        composed_db = compose_gsnr_db(incs, pr.endpoint_lin[slot], slot)
        if composed_db is None:
            continue
        out[slot] = (exact_db, composed_db)
    return out


def _same_path_sanity_check(model, oms_sequence, direction, mode, loading, grid) -> float:
    """Compose using increments harvested from THIS SAME propagation (the
    tautological case the brief calls out as a valid, cheap sanity check --
    NOT part of the measured error table): should reproduce the exact GSNR to
    within floating-point noise, confirming the bookkeeping itself is
    correct before trusting the cross-propagation numbers above."""
    pr = _propagate_loading(model, oms_sequence, direction, loading, mode,
                            probe_idx=0, capture_increments=True)
    worst = 0.0
    for i, freq_hz in enumerate(pr.si.frequency):
        slot = grid.slot_of(float(freq_hz))
        gsnr_db, osnr_db = _extract_gsnr_osnr(pr.si, i)
        exact_db, _ = _apply_penalties(
            pr.si, i, pr.uids_list, pr.elements, pr.roadm_propagated,
            pr.baud_rate, gsnr_db, osnr_db)
        incs = [pr.oms_increments[o][slot] for o in oms_sequence]
        composed_db = compose_gsnr_db(incs, pr.endpoint_lin[slot], slot)
        worst = max(worst, abs(composed_db - exact_db))
    return worst


def main() -> None:
    graph = json.loads(reference_topology("german_17").read_text(encoding="utf-8"))["graph"]
    modes = default_modes()
    model = model_from_abstract_graph(graph, modes=modes)
    mode = model.modes.get(MODE_ID)
    grid = SpectrumGrid.default()
    loading = _full_comb_loading(grid, MODE_ID)

    print(f"Enumerating candidate OMS paths (k={K_PER_PAIR} per ordered node pair)...")
    t0 = time.perf_counter()
    buckets = _enumerate_candidate_paths(model, K_PER_PAIR)
    max_hop = max(buckets)
    total_candidates = sum(len(v) for v in buckets.values())
    print(f"  {total_candidates} candidates across hops {min(buckets)}..{max_hop} "
         f"in {time.perf_counter() - t0:.1f}s")
    print(f"  real maximum hop count on german_17: {max_hop}")

    print("Harvesting per-OMS increments (one isolated 1-hop propagation per OMS)...")
    t0 = time.perf_counter()
    increments_table: Dict[str, Dict[int, float]] = {}
    for oms in model.list_oms():
        increments_table[oms.id] = _harvest_isolated_increment(
            model, oms.id, DIRECTION, mode, loading)
    print(f"  {len(increments_table)} OMS harvested in {time.perf_counter() - t0:.1f}s")

    rng = random.Random(SEED)

    print("\nSame-path sanity check (tautological, NOT part of the measured table):")
    for hops_check in (1, min(5, max_hop), max_hop):
        candidates = buckets.get(hops_check)
        if not candidates:
            continue
        seq = candidates[0]
        worst = _same_path_sanity_check(model, seq, DIRECTION, mode, loading, grid)
        print(f"  {hops_check} hops: max |composed - exact| = {worst:.6f} dB "
             f"(bookkeeping check)")

    print("\nComposition error (composed - actual; positive = optimistic):")
    print(f"{'hops':>4}  {'n':>5}  {'min(dB)':>10}  {'max(dB)':>10}  {'mean(dB)':>10}")
    rows: List[Tuple[int, int, float, float, float]] = []
    t0 = time.perf_counter()
    for hops in range(1, max_hop + 1):
        candidates = buckets.get(hops, [])
        if not candidates:
            continue
        sample = (candidates if len(candidates) <= SAMPLES_PER_HOP
                 else rng.sample(candidates, SAMPLES_PER_HOP))
        errors: List[float] = []
        for seq in sample:
            per_slot = _exact_and_composed(model, seq, DIRECTION, mode, loading,
                                           increments_table, grid)
            for slot, (exact_db, composed_db) in per_slot.items():
                errors.append(composed_db - exact_db)
        if not errors:
            continue
        n = len(errors)
        mn, mx = min(errors), max(errors)
        mean = sum(errors) / n
        rows.append((hops, n, mn, mx, mean))
        print(f"{hops:>4}  {n:>5}  {mn:>10.4f}  {mx:>10.4f}  {mean:>10.4f}")
    print(f"\n({time.perf_counter() - t0:.1f}s to measure)")

    overall_max = max(r[3] for r in rows)
    bound = math.ceil(overall_max * 100.0) / 100.0
    print(f"\nmeasured max signed error over all hops: {overall_max:.4f} dB")
    print(f"COMPOSITION_ERROR_BOUND_DB (measured max, rounded up) = {bound}")
    print(f"MAX_COMPOSED_HOPS (largest hop count actually measured) = {max_hop}")


if __name__ == "__main__":
    main()
