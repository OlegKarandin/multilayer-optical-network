"""Seeded gravity demand generator — the pure statistical core of the
operating-network builder (see docs/superpowers/specs/2026-07-14-...).

`generate_demands` is a deterministic function of (model, seed, params): no solver,
no QoT. It emits the frozen `solve_allocation` demand schema
`[{id, src, dst, demand_gbps, protected}]` (src/dst = optical node ids), which
`model/scenario.py` feeds through the packer to manufacture a loaded network.

Gravity model: demand between two nodes ∝ mass(u)·mass(v) / dist(u,v)^alpha, with
mass = node degree in the optical (OMS) graph (a hub proxy; override via
`node_mass`) and dist = shortest fiber length. `seed` enters ONLY through a
deterministic per-node mass jitter, so seed 0/1/2 yield distinct-but-reproducible
scenarios while a fixed seed is byte-stable.

This module is disaster-agnostic infrastructure: it knows nothing about events,
geography, or weather (CLAUDE.md hard rule).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from .network import NetworkModel
from .solvers import oms_length_km


def generate_demands(
    model: NetworkModel,
    *,
    seed: int,
    scale: float,
    alpha: float = 1.0,
    unit_gbps: float = 100.0,
    protected_fraction: float = 0.3,
    node_mass: Optional[Dict[str, float]] = None,
    mass_jitter: float = 0.15,
    pair_density: Optional[float] = None,
    protection_constraints: Optional[dict] = None,
    aggregate: bool = False,
    undirected: bool = False,
) -> List[dict]:
    """Emit a gravity-weighted demand list totalling ~`scale` Gbps of offered load.

    Args:
        seed: drives deterministic mass jitter (reproducible variety).
        scale: total offered volume (Gbps) spread across pairs by gravity weight.
        alpha: distance exponent in the gravity kernel.
        unit_gbps: quantum; each pair's offered volume becomes `round(offered/unit)`
            demands of `unit_gbps` each (nearest — pairs activate gradually as scale
            grows, keeping the scenario driver's search monotone).
        protected_fraction: share of demands (highest-gravity first) flagged
            `protected` — protection lands on the busiest, hub-incident corridors.
        node_mass: optional per-node mass override; default is OMS-graph degree.
        mass_jitter: fractional half-width of the per-node multiplicative jitter.
        pair_density: optional sparsity knob. If ``None`` (default), every
            reachable pair is considered (today's full-matrix behavior,
            unchanged). If set, each pair survives an independent weighted
            Bernoulli draw whose probability scales with the pair's gravity
            weight relative to the mean — above-mean pairs are likely (or, past
            a threshold, certain) to survive, below-mean pairs are the first to
            drop as pair_density shrinks. `scale`'s "total offered volume"
            contract is preserved by renormalizing over survivors only.
        protection_constraints: optional disjointness constraints dict
            (`{"basis": ..., "level": ..., "best_effort": ...}`, the same
            shape `solve_allocation`'s per-demand `constraints` reads)
            attached to every PROTECTED demand this call emits. `None`
            (default) preserves today's behavior: protected demands carry no
            `"constraints"` key at all, so `_demand_constraints` falls back
            to `("physical", "link", False)`. Without this, a synthesized
            operating network can never request srlg/risk_group-basis
            protection.
        aggregate: if ``False`` (default), each pair's offered volume is
            quantized into `unit_gbps`-sized demand records (today's
            behavior, for feeding `solve_allocation`). If ``True``, each
            surviving pair emits exactly one record carrying its raw,
            unquantized offered Gbps — an OD-matrix shape with no grooming
            unit involved.
        undirected: if ``False`` (default), both `(u, v)` and `(v, u)` are
            considered distinct pairs (today's behavior). If ``True``, only
            the `u < v` direction survives (gravity weight is symmetric, so
            this drops exact duplicates) — for a traffic matrix over
            unordered node pairs.

    Deterministic given (model, seed, all params). Disconnected pairs are
    skipped. `model` need not have an IP layer: if it exposes `list_routers`
    (a `NetworkModel`), node ids are router sites; otherwise (a bare
    `OpticalNetworkModel`) node ids are drawn from OMS endpoints.
    """
    routers = getattr(model, "list_routers", None)
    if routers is not None:
        nodes = sorted({r.site for r in routers()})
    else:
        nodes = sorted({n for oms in model.list_oms()
                        for n in (oms.src_node_id, oms.dst_node_id)})

    # Undirected optical graph: one edge per node pair, shortest length if parallel.
    g: nx.Graph = nx.Graph()
    g.add_nodes_from(nodes)
    for oms in model.list_oms():
        u, v = oms.src_node_id, oms.dst_node_id
        km = oms_length_km(model, oms.id)
        if g.has_edge(u, v):
            g[u][v]["km"] = min(g[u][v]["km"], km)
        else:
            g.add_edge(u, v, km=km)

    deg = dict(g.degree())
    rng = np.random.default_rng(seed)
    mass: Dict[str, float] = {}
    for nid in nodes:                    # sorted → stable RNG draw sequence
        base = float(node_mass[nid]) if node_mass is not None else float(deg.get(nid, 0))
        jitter = 1.0 + (rng.random() * 2.0 - 1.0) * mass_jitter
        mass[nid] = base * jitter

    dist = dict(nx.all_pairs_dijkstra_path_length(g, weight="km"))

    # Gravity weight per ordered pair (u != v) with a path and positive distance.
    pair_w: List[tuple] = []             # (u, v, w)
    total_w = 0.0
    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            d = dist.get(u, {}).get(v)
            if d is None or d <= 0.0:
                continue
            w = mass[u] * mass[v] / (d ** alpha)
            if w <= 0.0:
                continue
            pair_w.append((u, v, w))
            total_w += w

    if undirected:
        # Gravity weight is symmetric (w(u,v) == w(v,u)), so keeping only
        # u < v drops exact duplicates; total_w is recomputed over survivors
        # so `offered = scale * w / total_w` stays correctly normalized.
        pair_w = [(u, v, w) for u, v, w in pair_w if u < v]
        total_w = sum(w for _, _, w in pair_w)

    if pair_density is not None:
        mean_w = total_w / len(pair_w) if pair_w else 0.0
        kept: List[tuple] = []
        for u, v, w in pair_w:
            p_active = min(1.0, pair_density * w / mean_w) if mean_w > 0.0 else 0.0
            if rng.random() < p_active:
                kept.append((u, v, w))
        pair_w, total_w = kept, sum(w for _, _, w in kept)

    # Expand each pair's offered volume into unit-sized demand records (emission
    # order = sorted pairs, then unit index) carrying their gravity weight.
    # In aggregate mode, each pair instead emits one record at its raw
    # (unquantized) offered volume.
    records: List[dict] = []             # {src, dst, w, gbps}
    if total_w > 0.0:
        for u, v, w in pair_w:
            offered = scale * w / total_w
            if aggregate:
                if offered > 0.0:
                    records.append({"src": u, "dst": v, "w": w, "gbps": offered})
            else:
                for _ in range(int(round(offered / unit_gbps))):
                    records.append({"src": u, "dst": v, "w": w, "gbps": unit_gbps})

    # Protection: the top `protected_fraction` of records by gravity weight
    # (deterministic tie-break on src, dst, emission index).
    k = int(round(protected_fraction * len(records)))
    ranked = sorted(range(len(records)),
                    key=lambda i: (-records[i]["w"], records[i]["src"],
                                   records[i]["dst"], i))
    protected_idx = set(ranked[:k])

    demands: List[dict] = []
    for i, rec in enumerate(records):
        protected = i in protected_idx
        d = {
            "id": f"d{i:04d}",
            "src": rec["src"],
            "dst": rec["dst"],
            "demand_gbps": rec["gbps"],
            "protected": protected,
        }
        if protected and protection_constraints is not None:
            d["constraints"] = protection_constraints
        demands.append(d)
    return demands
