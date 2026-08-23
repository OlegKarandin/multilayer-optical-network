"""Ground-truth regression: the measured composition-error bound
(``COMPOSITION_ERROR_BOUND_DB``, ``MAX_COMPOSED_HOPS``) stays honest against a real
GNPy propagation on ``german_17``, pinned to the same methodology
``scripts/measure_composition_error.py`` used to set the constants (Task A4) --
see ``gnpy_adapter/composition.py``'s comment above the constants for the full
measured table.

Opt-in / slow: real GNPy on the packaged 17-node topology, gated behind
``OPTICAL_NET_RUN_GNPY_E2E`` per the established pattern (`tests/model/
test_scenario.py:318`) since the corpus below (spanning 1..MAX_COMPOSED_HOPS)
runs real propagations for every sampled path."""
import json
import os
import random

import pytest

from multilayer_optical_network.data import reference_topology
from multilayer_optical_network.gnpy_adapter.adapter import (
    _apply_penalties, _extract_gsnr_osnr, _propagate_loading,
)
from multilayer_optical_network.gnpy_adapter.composition import (
    COMPOSITION_ERROR_BOUND_DB, MAX_COMPOSED_HOPS, compose_gsnr_db,
)
from multilayer_optical_network.gnpy_adapter.loading import Channel, LoadingState
from multilayer_optical_network.model.assets import Direction
from multilayer_optical_network.model.modes import default_modes
from multilayer_optical_network.model.solvers import SolverStatus, compute_paths
from multilayer_optical_network.model.spectrum import SpectrumGrid
from multilayer_optical_network.model.topology_import import model_from_abstract_graph

MODE_ID = "400G@7.1dB"
SAMPLES_PER_HOP = 3
K_PER_PAIR = 60          # matches scripts/measure_composition_error.py -- small k
                          # misses the rare 15/16-hop candidates entirely (verified)
SEED = 0
DIRECTION = Direction.FORWARD


def _full_comb_loading(grid, mode_id):
    return LoadingState(tuple(
        Channel(grid.freq(s), grid.spacing_hz, None, mode_id)
        for s in range(grid.num_slots)
    ))


def _all_node_ids(model):
    ids = set()
    for oms in model.list_oms():
        ids.add(oms.src_node_id)
        ids.add(oms.dst_node_id)
    return sorted(ids)


def _candidate_corpus(model, max_hops, samples_per_hop, k_per_pair, seed):
    """A small corpus of real k-shortest routing candidates spanning
    1..max_hops -- the same enumerator (`model.solvers.compute_paths`) the
    measurement driver used, sampled far more lightly (a handful per hop
    count) to keep this regression fast."""
    nodes = _all_node_ids(model)
    buckets: dict[int, list] = {h: [] for h in range(1, max_hops + 1)}
    for src in nodes:
        for dst in nodes:
            if src == dst:
                continue
            result = compute_paths(model, src, dst, k=k_per_pair)
            if result.status != SolverStatus.SOLUTION:
                continue
            for p in result.paths:
                hops = len(p.oms_sequence)
                if hops in buckets and len(buckets[hops]) < samples_per_hop:
                    buckets[hops].append(p.oms_sequence)
        if all(len(v) >= samples_per_hop for v in buckets.values()):
            break
    rng = random.Random(seed)
    corpus = []
    for hops, seqs in buckets.items():
        corpus.extend(rng.sample(seqs, min(len(seqs), samples_per_hop)))
    return corpus


def _harvest_isolated_increment(model, oms_id, direction, mode, loading):
    """One OMS's own per-slot increment from a one-hop propagation of it
    ALONE -- deliberately a different propagation from any multi-hop path in
    the corpus, so composing never reuses the same propagation's own
    precision (mirrors the measurement driver's methodology exactly)."""
    pr = _propagate_loading(model, (oms_id,), direction, loading, mode,
                            probe_idx=0, capture_increments=True)
    return pr.oms_increments[oms_id]


def _signed_errors(model, oms_sequence, direction, mode, loading,
                   increments_table, grid):
    """composed - exact, one per carrier, for one real path (one exact
    propagation for the exact side + this path's own analytic endpoint
    term)."""
    pr = _propagate_loading(model, oms_sequence, direction, loading, mode,
                            probe_idx=0, capture_increments=True)
    errors = []
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
        errors.append(composed_db - exact_db)
    return errors


@pytest.mark.skipif(
    not os.environ.get("OPTICAL_NET_RUN_GNPY_E2E"),
    reason="slow real-GNPy build; set OPTICAL_NET_RUN_GNPY_E2E=1 to run")
def test_composition_error_stays_within_the_declared_bound():
    """Ground truth against a pinned GNPy (2.14.0, per CLAUDE.md's Requirements): the
    signed error (composed - actual; positive = optimistic) over a corpus of paths
    spanning 1..MAX_COMPOSED_HOPS on german_17 must stay inside
    COMPOSITION_ERROR_BOUND_DB. This is the test that keeps the bound honest as
    topologies and GNPy versions change -- if it fails, LOWER MAX_COMPOSED_HOPS or
    RAISE the design margin; do not widen the bound to make it pass."""
    graph = json.loads(reference_topology("german_17").read_text(encoding="utf-8"))["graph"]
    modes = default_modes()
    model = model_from_abstract_graph(graph, modes=modes)
    mode = model.modes.get(MODE_ID)
    grid = SpectrumGrid.default()
    loading = _full_comb_loading(grid, MODE_ID)

    corpus = _candidate_corpus(model, MAX_COMPOSED_HOPS, SAMPLES_PER_HOP, K_PER_PAIR, SEED)
    assert corpus, "no candidate paths found on german_17 -- corpus construction is broken"
    hops_covered = {len(seq) for seq in corpus}
    assert min(hops_covered) == 1 and max(hops_covered) == MAX_COMPOSED_HOPS, (
        f"corpus must span 1..MAX_COMPOSED_HOPS ({MAX_COMPOSED_HOPS}); got {sorted(hops_covered)}")

    increments_table = {
        oms.id: _harvest_isolated_increment(model, oms.id, DIRECTION, mode, loading)
        for oms in model.list_oms()
    }

    all_errors = []
    for seq in corpus:
        all_errors.extend(
            _signed_errors(model, seq, DIRECTION, mode, loading, increments_table, grid))

    assert all_errors
    worst = max(all_errors, key=abs)
    assert abs(worst) <= COMPOSITION_ERROR_BOUND_DB, (
        f"signed composition error {worst:.4f} dB exceeds the declared bound "
        f"{COMPOSITION_ERROR_BOUND_DB} dB -- re-measure with "
        f"scripts/measure_composition_error.py before trusting composition; "
        f"lower MAX_COMPOSED_HOPS or raise the bound only from a new measurement, "
        f"never to make this assertion pass")
