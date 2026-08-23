# src/multilayer_optical_network/model/multilayer_graph.py
"""Layered IP+optical auxiliary graph (Zhu/Mukherjee model, per-wavelength
layers, no wavelength conversion) + IGABAG single-demand placement.

Vertices:
  (ACCESS, node)        access/IP layer port for an optical node
  (WLIN, node, lam)     wavelength-layer arrival port (a WLE lands here)
  (WLOUT, node, lam)    wavelength-layer departure port (a WLE leaves here)

The WLin/WLout split (rather than one (WL,node,lam) vertex) lets a segmented
placement terminate at WLin and re-originate at WLout on the SAME wavelength; a
single shared vertex forbade that because a simple path can't revisit it.

Edges (directed; every edge carries a 'weight'):
  LPE  access(u) -> access(v)   one per existing lightpath u->v.
        Carries lightpath_id + residual_gbps; absent when margin<0, residual 0,
        or the lightpath crosses a forbidden asset. Low weight (reuse).
  WLE  (WLout,u,lam) -> (WLin,v,lam)  one per free slot `lam` on an OMS u->v;
        carries oms_id + lam. Low weight.
  TxE  access(u) -> (WLout,u,lam)   originate a new lightpath on slot lam. MODERATE
        weight (a few groom-hops' worth) so new segments are discouraged but still
        reachable by k-shortest within budget — letting hybrids interleave into
        the frontier instead of ranking behind every pure-groom path.
  RxE  (WLin,v,lam) -> access(v)   terminate a new lightpath. Zero weight.
  EXPRESS (WLin,n,lam) -> (WLout,n,lam)  optical pass-through at n on one lam. Zero
        weight; present only where n both receives and forwards on that slot.

A path access(src) -> access(dst) that stays on existing lightpaths uses only
LPE edges (grooming). A path that dips via TxE -> WLEs (bypassing pass-through
nodes on EXPRESS) -> RxE realizes a new lightpath on one wavelength. No CvtE:
wavelength continuity is structural (EXPRESS never crosses lam).

Stage 7 assumptions (recorded explicitly, from the inspection roadmap):
- `Router.site == optical-node id` is the src/dst -> optical-node resolution
  the restoration caller (`compute_restoration`) depends on.
- Wavelength continuity is structural (no cross-lambda edge): a new lightpath
  run is one lambda end-to-end, never converted mid-path.
- A returned candidate does NOT commit to a specific wavelength; provisioning
  must re-run spectrum assignment before actually lighting a new run.
- New-lightpath runs within ONE placement are NOT assumed OMS-disjoint (S7-10,
  fixed): the WLIN/WLOUT node-split means two grooming-separated runs in the
  same placement CAN legitimately share a physical OMS (one originates onto
  it, another re-enters it later via EXPRESS at a different wavelength) — the
  audit's "assumed OMS-disjoint, unstated precondition" turned out false, and
  was reproduced with the real GNPy adapter on a mesh fixture (~0.03 dB
  optimism on an 800 km / 10-span shared OMS under FillPolicy.ACTUAL,
  2026-07-29). `_build_loading` alone QoTs a new run against the committed
  `spectrum` snapshot only, which never includes an uncommitted sibling run
  from the same placement; `place_demands` now adds each OMS-overlapping
  sibling's own wavelength as an extra neighbor channel before either run is
  QoT'd. Skipped under FULL (the sibling's slot is already in the dense
  comb; adding it again would duplicate a frequency) — the bug was
  ACTUAL-specific.
- `build_layered_graph` and `place_demands` each independently accept a `grid`
  parameter (S7-12, fixed): every caller that invokes both for the same
  placement (`route_service.route_service`, `allocation`'s per-demand loop)
  now builds one `SpectrumGrid` and threads it through both calls, so the two
  `build_spectrum_state` calls can never desync on grid choice.
- `SlotClass` and `maximal_slot_classes` replace the per-wavelength cap heuristic
  (S7-14): layers are built only for ⊆-maximal free-slot signatures, not per slot,
  so graph size is bounded as occupancy rises (O(m) classes in the worst case, where
  m is at most `grid.num_slots`). The concrete wavelength choice is deferred to
  accept time via `first_fit_slot`, so spectral packing is orthogonal to route
  discovery — `place_demands` can choose the lowest free slot per placement without
  driving a new routing pass.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Iterator, List, Optional, Tuple

import networkx as nx

from .network import NetworkModel
from .spectrum import FillPolicy, SpectrumGrid, build_spectrum_state
from .exposure import oms_seq_asset_set

ACCESS = "access"
# Node-split wavelength ports: a WLE lands on WLin and departs from WLout, joined
# by an EXPRESS edge for optical pass-through. Splitting the old single (WL,n,lam)
# vertex lets a segmented placement terminate at (WLin,n,lam) and re-originate at
# (WLout,n,lam) on the SAME wavelength — a simple path could not revisit one shared
# vertex, which spuriously forced the two runs onto different slots.
WLIN = "wlin"
WLOUT = "wlout"

# Edge weights shape k-shortest DISCOVERY only (final ranking uses cost_vector).
# New lightpaths are discouraged but reachable, so hybrids/new candidates
# interleave into the frontier rather than ranking behind every pure-groom path
# (which a 1000x penalty would cause). Tunable.
_W_LPE = 1.0       # reuse an existing lightpath (one virtual hop)
_W_WLE = 0.1       # traverse one OMS on a new lightpath's wavelength
_W_NEW_LP = 5.0    # originate a new lightpath (TxE): a few groom-hops' worth
_W_RXE = 0.0
_W_EXPRESS = 0.0   # optical pass-through at a node (WLin->WLout, same lam)


def _lightpath_endpoints(model: NetworkModel, lp) -> Tuple[str, str]:
    """(src_node, dst_node) of a lightpath from its first/last OMS endpoints."""
    first = model.get_oms(lp.oms_sequence[0])
    last = model.get_oms(lp.oms_sequence[-1])
    return first.src_node_id, last.dst_node_id


def _lightpath_forbidden(model: NetworkModel, lp, forbidden_assets: FrozenSet[str]) -> bool:
    if not forbidden_assets:
        return False
    assets = set()
    for oms_id in lp.oms_sequence:
        assets |= oms_seq_asset_set(model, (oms_id,))
        oms = model.get_oms(oms_id)
        assets.add(oms.src_node_id)
        assets.add(oms.dst_node_id)
    return bool(assets & forbidden_assets)


def _residual_gbps(model: NetworkModel, lp, load: Dict[str, float]) -> float:
    """Derived capacity of the lightpath's bound IP link(s) minus current load.
    A lightpath with no IP link bound yields its full mode rate (margin-gated).

    `load` is the offered-load-per-IP-link map, built ONCE by the caller and
    passed in — rebuilding it per lightpath is O(L·S) (S5-8/S7-8).

    Over multiple bound IP links we take the **min** residual (the bottleneck),
    not the max. A groom onto this lightpath rides one of its bound IP links, and
    at graph-build time we don't know which, so the honest headroom is the tightest
    link's: `max` would report the healthiest link and overstate capacity a
    saturated sibling link can't actually provide (a confident wrong number).

    A lightpath with no recorded QoT yet (e.g. freshly lit by the live
    provision_lightpath tool, which does not seed or recompute) reads as zero
    residual, not a crash -- consistent with the no-IP-link branch below and
    with ip_routing._link_status's "unknown" (not up, not down) treatment."""
    ip_ids = model.ip_links_for_lightpath(lp.id)
    if not ip_ids:
        # no IP link bound: capacity is mode rate iff margin >= 0
        try:
            state = model.get_qot_state(lp.id)
        except LookupError:
            return 0.0
        return model.modes.get(lp.mode_id).bitrate_gbps if state.mode_feasible else 0.0
    residual = float("inf")
    for ip_id in ip_ids:
        try:
            cap = model.ip_link_capacity_gbps(ip_id)   # 0.0 when margin<0
        except LookupError:
            cap = 0.0   # no QoT recorded yet -- treat as no residual capacity
        residual = min(residual, cap - load.get(ip_id, 0.0))
    return residual


@dataclass(frozen=True)
class SlotClass:
    """One wavelength LAYER of the auxiliary graph, standing for a set of
    interchangeable grid slots.

    `oms_ids` is the class's SIGNATURE: the non-forbidden OMS on which every slot in
    `slots` is free. A layer's route set is determined entirely by its signature, and
    all WLE carry the same weight (`_W_WLE`), so if signature(A) is a subset of
    signature(B) then every route liftable onto A lifts onto B at identical cost and A
    contributes nothing. Only ⊆-maximal signatures get a layer."""
    class_id: int
    slots: Tuple[int, ...]
    oms_ids: FrozenSet[str]


def maximal_slot_classes(
    non_forbidden: List, spectrum: Dict[str, int], grid: SpectrumGrid,
) -> Tuple[SlotClass, ...]:
    """Group grid slots by free-set signature; keep only the ⊆-maximal ones.

    Supersedes the old `cap` heuristic, which stopped at the first globally-free slot
    and KEPT every slot below it ("a route may find them free on its own hops though
    occupied elsewhere" — true, but irrelevant: if a route's hops are free on slot 2
    they are also free on a slot that is free everywhere). The all-free signature is
    the TOP of the subset lattice, so whenever any slot is free on every OMS exactly
    one layer is built, no matter how full the network is — the old cap grew with
    occupancy, i.e. search cost peaked exactly when a disaster was in progress.

    Slots with an EMPTY signature (lit on every non-forbidden OMS) carry no route and
    are discarded before maximality is tested. Signatures are computed over
    NON-FORBIDDEN OMS only, so an avoid-set can change which ones are maximal.

    Bit-twiddling note: `sig[lam]` is a bitmask over the *index* of `non_forbidden`,
    which makes the O(m^2) maximality scan a pair of integer `&` comparisons rather
    than set operations. m is at most `grid.num_slots` (48)."""
    n_slots = grid.num_slots
    sig = [0] * n_slots                       # sig[lam]: bitmask over OMS index
    for i, oms in enumerate(non_forbidden):
        free = (~spectrum.get(oms.id, 0)) & grid.all_slots_mask
        bit = 1 << i
        while free:
            low = free & -free
            sig[low.bit_length() - 1] |= bit
            free ^= low
    by_sig: Dict[int, List[int]] = {}
    for lam in range(n_slots):
        if sig[lam]:                          # empty signature -> no routes
            by_sig.setdefault(sig[lam], []).append(lam)
    keys = list(by_sig)
    maximal = [s for s in keys if not any(t != s and s & t == s for t in keys)]
    maximal.sort(key=lambda s: by_sig[s][0])  # deterministic: lowest slot first
    return tuple(
        SlotClass(class_id=cid, slots=tuple(by_sig[s]),
                  oms_ids=frozenset(non_forbidden[i].id
                                    for i in range(len(non_forbidden))
                                    if (s >> i) & 1))
        for cid, s in enumerate(maximal)
    )


def build_layered_graph(
    model: NetworkModel,
    forbidden_assets: FrozenSet[str] = frozenset(),
    *,
    grid: SpectrumGrid | None = None,
    min_residual_gbps: float = 0.0,
) -> nx.MultiDiGraph:
    """Construct the layered auxiliary graph for the model's current loading.
    `forbidden_assets` prunes any OMS touching them (no WLE) and any lightpath
    crossing them (no LPE).

    `min_residual_gbps` additionally prunes any LPE edge whose lightpath's
    residual capacity is below this threshold -- capacity checked DURING graph
    construction (and therefore during Yen's routing), not clamped after the
    fact. The default, `0.0`, means no filtering beyond the existing
    residual>0 gate: today's behaviour, unchanged. Callers that want degraded
    grooming options in the candidate set (`route_service`, restoration) must
    keep the default; only a caller that knows the demand size up front (the
    allocation packer) should pass it.

    A MultiDiGraph (not a plain DiGraph) so parallel OMS between the same ordered
    node pair stay distinct per layer: on a DiGraph the second WLE
    ``(WLout,a,c)->(WLin,b,c)`` overwrites the first, silently collapsing
    parallel fibers to one route per class (S7-13). Mirrors the flat OMS solver's
    MultiDiGraph (S6-4). Parallel lightpaths on the same access hop stay distinct
    for the same reason."""
    from .ip_routing import offered_load_per_link
    grid = grid or SpectrumGrid.default()
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    spectrum = build_spectrum_state(model, grid)
    # Build the offered-load map once (S5-8/S7-8): _residual_gbps used to rebuild
    # it per lightpath (O(L·S)); it's loading-state-wide, so hoist it out of the loop.
    load = offered_load_per_link(model)

    # forbidden OMS: any OMS whose asset set / endpoints intersect forbidden_assets
    def _oms_forbidden(oms) -> bool:
        if not forbidden_assets:
            return False
        phys = set(oms_seq_asset_set(model, (oms.id,)))
        phys.add(oms.src_node_id)
        phys.add(oms.dst_node_id)
        return bool(phys & forbidden_assets)

    # access vertices for every optical node that appears on an OMS endpoint
    for oms in model.list_oms():
        g.add_node((ACCESS, oms.src_node_id))
        g.add_node((ACCESS, oms.dst_node_id))

    # LPE edges: existing lightpaths
    for lp in model.list_lightpaths():
        if _lightpath_forbidden(model, lp, forbidden_assets):
            continue
        residual = _residual_gbps(model, lp, load)
        if residual <= 0.0:
            continue
        if residual < min_residual_gbps:
            # Capacity checked DURING routing, not after. _W_LPE is the cheapest
            # weight in the graph, so a lightpath that cannot carry the demand
            # would otherwise be returned first by Yen's and then clamped to its
            # residual by `restored = min(demand, groom_cap, new_cap)` -- a
            # systematic degraded pick. Callers that WANT degraded options
            # (route_service, restoration) leave this at the 0.0 default.
            continue
        u, v = _lightpath_endpoints(model, lp)
        g.add_edge((ACCESS, u), (ACCESS, v), key=lp.id,
                   kind="LPE", lightpath_id=lp.id, residual_gbps=residual,
                   weight=_W_LPE)

    # WLE + TxE/RxE per DOMINANCE-MAXIMAL wavelength layer. See maximal_slot_classes:
    # a layer exists per ⊆-maximal free-slot signature, not per slot, so the graph
    # stays ~one layer as the network fills instead of growing with occupancy. Layers
    # are labelled by CLASS INDEX, not by a slot — the concrete wavelength is chosen
    # by first_fit_slot at accept time in place_demands, which packs lower than
    # whichever λ-variant Yen's happened to yield first (they all tie on weight).
    #
    # ONE layer suffices because of the node-split below: each optical node has a
    # WLin and a WLout port per layer,
    #   WLE      (WLout,u,c) -> (WLin,v,c)   traverse OMS u->v on this layer
    #   TxE      access(u)    -> (WLout,u,c)  originate a new lightpath
    #   RxE      (WLin,v,c)   -> access(v)    terminate a new lightpath
    #   EXPRESS  (WLin,n,c)   -> (WLout,n,c)  optical pass-through on one layer
    # A through-lightpath bypasses node n via EXPRESS (one wavelength, continuity
    # structural). A SEGMENTED placement terminates at (WLin,n,c) and re-originates at
    # (WLout,n,c) — DISTINCT vertices — so its two runs may share one wavelength. The
    # old single (WL,n,lam) vertex forbade that (a simple path can't revisit it).
    non_forbidden = [oms for oms in model.list_oms() if not _oms_forbidden(oms)]
    classes = maximal_slot_classes(non_forbidden, spectrum, grid)
    g.graph["slot_classes"] = classes
    wl_in: set = set()      # (node, class_id) reached by an incoming WLE
    wl_out: set = set()     # (node, class_id) left by an outgoing WLE
    for cls in classes:
        for oms in non_forbidden:
            if oms.id not in cls.oms_ids:
                continue    # slot lit on this OMS for every slot in the class
            u, v = oms.src_node_id, oms.dst_node_id
            c = cls.class_id
            # An OMS carries traffic src->dst ONLY (mirrors build_oms_graph's
            # directionality invariant): add the WLE in the OMS's own direction and
            # NOT the reverse. The reverse hop is served by the reverse OMS, which a
            # physical topology always provides (topology_import adds both). Adding
            # (v,u) here made a return-direction OMS traversable against its flow, so
            # _parse_paths could stitch a new lightpath from wrong-direction OMS
            # (e.g. oms_1_0 for a 0->1 hop) -> a non-contiguous oms_sequence that
            # add_lightpath rejects. key=oms.id keeps parallel OMS on the same ordered
            # (WLout,u,c)->(WLin,v,c) pair distinct (the S7-13 fix).
            g.add_edge((WLOUT, u, c), (WLIN, v, c), key=oms.id,
                       kind="WLE", oms_id=oms.id, class_id=c, weight=_W_WLE)
            g.add_edge((ACCESS, u), (WLOUT, u, c), key="TxE",
                       kind="TxE", class_id=c, weight=_W_NEW_LP)
            g.add_edge((WLIN, v, c), (ACCESS, v), key="RxE",
                       kind="RxE", class_id=c, weight=_W_RXE)
            wl_out.add((u, c))
            wl_in.add((v, c))
    # EXPRESS: optical pass-through where a node both receives and forwards on a layer
    # (so a through-lightpath continues on one wavelength without dropping to access).
    for n, c in wl_in & wl_out:
        g.add_edge((WLIN, n, c), (WLOUT, n, c), key="EXPRESS",
                   kind="EXPRESS", class_id=c, weight=_W_EXPRESS)
    return g


def lpe_edges(g: nx.MultiDiGraph) -> List[Tuple]:
    """All LPE edges as (u, v, data)."""
    return [(u, v, d) for u, v, d in g.edges(data=True) if d.get("kind") == "LPE"]


def wle_count_on_layer(g: nx.MultiDiGraph, oms_id: str, class_id: int) -> int:
    """Number of WLE edges for an OMS on a given wavelength LAYER (a dominance-maximal
    slot class, indexed from 0 — not a grid slot). 0 when the OMS is lit on that
    class's slots or forbidden, else 1: an OMS carries traffic in its own direction
    only, so it contributes a single WLE per layer — the reverse hop belongs to the
    reverse OMS."""
    return sum(1 for _, _, d in g.edges(data=True)
               if d.get("kind") == "WLE" and d.get("oms_id") == oms_id
               and d.get("class_id") == class_id)


# ---------------------------------------------------------------------------
# place_demands — IGABAG k-best placement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NewLightpathRun:
    oms_sequence: Tuple[str, ...]
    lam: int
    mode_id: str
    gsnr_db: float
    bitrate_gbps: float
    # Travel direction of the run (the demand's direction). `oms_sequence` is in
    # physical-OMS order, which may be traversed in reverse for a return-direction
    # demand; (src_node, dst_node) records the actual endpoints so provisioning
    # does not derive a reversed lightpath from oms_sequence.
    src_node: str = ""
    dst_node: str = ""


@dataclass(frozen=True)
class Placement:
    reused_lightpaths: Tuple[str, ...]
    new_lightpaths: Tuple[NewLightpathRun, ...]
    restored_gbps: float
    shortfall_gbps: float


# Enumeration budget: walk up to _PATH_BUDGET simple paths per policy, keeping
# up to _DEFAULT_K distinct feasible placements (the cost-ordered frontier).
_PATH_BUDGET = 64
_DEFAULT_K = 8

# Generous safety cap on RAW node paths drawn from shortest_simple_paths. The
# _PATH_BUDGET / _DEFAULT_K guards count DISTINCT routes / accepted placements, so
# on a topology with few distinct routes but a wide grid neither fires and the
# generator drains to exhaustion — thousands of lambda-mixing simple paths (new
# lightpaths regenerated across slots at an access node), Yen's algorithm churning
# on each. This bounds that work while staying far above the lambda-variant count
# that would otherwise starve a strictly-more-expensive distinct route (the S7-6
# guard): a full C-band's worth of slots is < 128, so 1024 clears ~8 cheaper
# distinct routes' variants before cutting off.
_RAW_PATH_CAP = 1024


def _policy_graph(g: nx.MultiDiGraph, policy: str) -> nx.MultiDiGraph:
    """Restrict the graph to a lever:
      groom_only   - drop TxE edges: reuse existing lightpaths only (no new).
      new_only     - drop LPE edges: force fresh lightpaths (no reuse).
      groom_or_new - full graph (grooming wins on weight when feasible)."""
    if policy == "groom_or_new":
        return g
    if policy not in ("groom_only", "new_only"):
        raise ValueError(f"unknown policy {policy!r}")
    drop_kind = "TxE" if policy == "groom_only" else "LPE"
    h = g.copy()
    h.remove_edges_from([(u, v, key) for u, v, key, d in h.edges(keys=True, data=True)
                         if d.get("kind") == drop_kind])
    return h


def _collapse_to_simple(h: nx.MultiDiGraph) -> nx.DiGraph:
    """Collapse parallel edges to a simple DiGraph for node-simple-path
    enumeration (`nx.shortest_simple_paths` is not implemented for multigraphs),
    each simple edge carrying the MIN parallel weight so the ordering matches the
    cheapest realisation of the hop. The per-hop parallel choices are re-expanded
    over the MultiDiGraph afterwards (`_parse_paths`) — mirrors the flat solver's
    collapse-then-expand (S6-4)."""
    simple = nx.DiGraph()
    simple.add_nodes_from(h.nodes)
    for u, v, d in h.edges(data=True):
        w = d.get("weight", 1.0)
        if simple.has_edge(u, v):
            simple[u][v]["weight"] = min(simple[u][v]["weight"], w)
        else:
            simple.add_edge(u, v, weight=w)
    return simple


def _parse_paths(
    g: nx.MultiDiGraph, path: List,
) -> Iterator[Tuple[List[str], List[Tuple[Tuple[str, ...], int, str, str]]]]:
    """Expand an access->access *vertex* path into every concrete
    (reused_lightpath_ids, new_runs) it realises, choosing among parallel edges
    per hop. On a MultiDiGraph a single node path may correspond to several routes
    when parallel OMS (or parallel lightpaths) share an ordered vertex pair
    (S7-13) — `nx.shortest_simple_paths` yields node paths only, so the per-hop
    choice is re-expanded here (mirrors the flat solver's `itertools.product`).

    Each new_run is (oms_sequence, lam, src_node, dst_node); the travel endpoints
    come from the WL-vertex node components ((WLout/WLin, node, lam)), so a return-
    direction run over a physically-forward OMS records its true direction rather
    than the OMS's physical orientation. EXPRESS hops (optical pass-through at a
    node on one wavelength) continue the current run without touching access."""
    hops = list(zip(path, path[1:]))
    # per hop: the list of parallel edge-data dicts (MultiDiGraph get_edge_data
    # returns {key: data}); a plain node path collapses these into one choice.
    per_hop = [list(g.get_edge_data(a, b).values()) for a, b in hops]
    for combo in itertools.product(*per_hop):
        reused: List[str] = []
        new_runs: List[Tuple[Tuple[str, ...], int, str, str]] = []
        cur_oms: List[str] = []
        cur_lam: Optional[int] = None
        cur_src: Optional[str] = None
        cur_dst: Optional[str] = None
        for (a, b), d in zip(hops, combo):
            kind = d.get("kind")
            if kind == "LPE":
                reused.append(d["lightpath_id"])
            elif kind == "WLE":
                cur_oms.append(d["oms_id"])
                cur_lam = d["class_id"]
                if cur_src is None:
                    cur_src = a[1]      # from-node of the first hop in this run
                cur_dst = b[1]          # to-node, advanced each hop
            elif kind == "RxE":
                if cur_oms:
                    new_runs.append((tuple(cur_oms), cur_lam, cur_src, cur_dst))
                    cur_oms, cur_lam, cur_src, cur_dst = [], None, None, None
            # TxE: entry into a wl layer; nothing to record.
            # EXPRESS: optical pass-through (WLin,n,lam)->(WLout,n,lam) — the run
            # continues on the same wavelength; nothing to record (the next WLE
            # extends cur_oms/cur_dst).
        yield reused, new_runs


def _bottleneck_residual(g: nx.MultiDiGraph, reused: List[str]) -> float:
    """Min residual_gbps across the reused LPE edges (inf if none reused)."""
    if not reused:
        return float("inf")
    by_lp = {d["lightpath_id"]: d["residual_gbps"]
             for _, _, d in g.edges(data=True) if d.get("kind") == "LPE"}
    return min(by_lp[lp] for lp in reused)


def place_demands(
    model: NetworkModel, g: nx.MultiDiGraph, qot, *,
    src: str, dst: str, demand_gbps: float, policy: str,
    k: int = _DEFAULT_K, grid: Optional[SpectrumGrid] = None,
    fill_policy: FillPolicy = None,
    stop_when: Optional[Callable[[Placement], bool]] = None,
) -> List[Placement]:
    """IGABAG for one demand, returning up to `k` DISTINCT feasible placements
    (the cost-ordered frontier under the policy), each possibly degraded. A
    placement may reuse existing lightpaths (LPE), light new ones (TxE->WLEs->
    RxE), or BOTH (a hybrid). Empty list when no feasible path exists.

    `fill_policy` selects the acceptance-probe reference loading passed to
    `_build_loading` (defaults to FULL — see FillPolicy).

    `stop_when`, when given, is a predicate over an accepted Placement: as soon
    as one is found for which it is true, enumeration stops and that placement
    is the last one returned (early-exit, not a hard k=1 — the frontier is
    still cost-ordered, it's just truncated at the first acceptable answer).
    `None` (the default) enumerates the full `k`-best frontier, i.e. today's
    behaviour — used by callers (route_service, restoration) that need to rank
    or search within the whole candidate set."""
    from .allocation import _build_loading, _best_feasible_mode
    from ..gnpy_adapter.loading import Channel, LoadingState
    if fill_policy is None:
        fill_policy = FillPolicy.FULL
    grid = grid or SpectrumGrid.default()
    h = _policy_graph(g, policy)
    simple = _collapse_to_simple(h)
    s, t = (ACCESS, src), (ACCESS, dst)
    if s not in simple or t not in simple or not nx.has_path(simple, s, t):
        return []
    spectrum = build_spectrum_state(model, grid)
    # S7-9: every new-lightpath run in this placement is probed at the SAME
    # ref_mode (the registry's first mode), regardless of which mode
    # _best_feasible_mode later selects for it. Correct only while GSNR is
    # mode-independent given a fixed probe — true here because _build_loading
    # doesn't populate the probe Channel's baud_rate_hz/roll_off from
    # ref_mode_id (they fall back to build_si_for_loading's scalar default,
    # the S2-4 residual), so the probe's spectral shape doesn't vary with
    # ref_mode either. Would need to re-probe at the delivered mode if
    # per-format baud is ever threaded through this probe.
    ref_mode = model.modes.list()[0].id
    out: List[Placement] = []
    seen: set = set()
    examined = 0     # DISTINCT routes examined (budget counter, not raw emissions)
    raw_paths = 0    # RAW node paths drawn (safety valve against generator drain)
    budget_hit = False
    for path in nx.shortest_simple_paths(simple, s, t, weight="weight"):
        if len(out) >= k or budget_hit or raw_paths >= _RAW_PATH_CAP:
            break
        raw_paths += 1
        # One node path may realise several routes when parallel OMS/lightpaths
        # share an ordered vertex pair (S7-13); expand the per-hop choices.
        for reused, new_runs in _parse_paths(h, path):
            if len(out) >= k:
                break
            # Deduplicate by structural route (reused LP ids + new OMS sequences),
            # ignoring wavelength slot: same OMS sequence on lam=0 and lam=1 is the
            # same route option, just a different channel assignment. Collapsing
            # them keeps the k-best frontier meaningful (diverse routes/groom
            # combos) instead of filling it with the same plan on every free slot.
            key = (tuple(reused), tuple(oms_seq for oms_seq, _, _, _ in new_runs))
            if key in seen:
                continue     # a lambda-variant of an already-seen route: does NOT
                             # advance the budget, so a route with many free slots
                             # can't starve structurally distinct routes.
            seen.add(key)
            examined += 1
            if examined > _PATH_BUDGET:
                budget_hit = True
                break
            realized: List[NewLightpathRun] = []
            feasible = True
            new_cap = float("inf")
            # S7-10 (fixed): a new run is QoT'd against the committed `spectrum`
            # snapshot, which never sees a co-located SIBLING new run in this
            # same placement — those aren't committed either. Under FULL this
            # is harmless (every non-probe slot is already lit in the probe
            # comb, sibling or not). Under ACTUAL it was a real, measured
            # optimism (see the module docstring's Stage 7 assumptions):
            # `_build_loading` only sees already-occupied slots, so a sibling
            # run sharing an OMS with this one (the WLIN/WLOUT+EXPRESS node-
            # split lets that happen — a run can re-enter a physical span an
            # earlier sibling already used, at a different wavelength) was
            # invisible to this run's probe, and vice versa. Fixed by adding
            # each overlapping sibling's own wavelength as an extra neighbor
            # channel before either is QoT'd — order-independent (every run
            # sees every co-located sibling that shares an OMS with it,
            # regardless of loop order), and skipped under FULL, where the
            # sibling's slot is already included in the dense comb and adding
            # it again would duplicate a frequency.
            for idx, (oms_seq, lam, run_src, run_dst) in enumerate(new_runs):
                loading = _build_loading(grid, spectrum, oms_seq, lam, ref_mode,
                                         fill_policy)
                if fill_policy is not FillPolicy.FULL:
                    oms_set = set(oms_seq)
                    sibling_lams = {
                        sib_lam
                        for j, (sib_oms_seq, sib_lam, _, _) in enumerate(new_runs)
                        if j != idx and oms_set.intersection(sib_oms_seq)
                    }
                    if sibling_lams:
                        # Defensive dedup: a sibling's slot could coincide with a
                        # frequency `_build_loading` already added from the
                        # committed spectrum (e.g. already occupied on a
                        # DIFFERENT hop of this run's own multi-hop oms_sequence)
                        # — avoid emitting two carriers at the same frequency.
                        have = {c.center_freq_hz for c in loading.channels}
                        extra = tuple(
                            Channel(grid.freq(sl), grid.spacing_hz, None, ref_mode)
                            for sl in sibling_lams
                            if grid.freq(sl) not in have
                        )
                        if extra:
                            loading = LoadingState(loading.channels + extra)
                mode, gsnr = _best_feasible_mode(model, qot, oms_seq, loading, ref_mode)
                if mode is None:
                    feasible = False
                    break
                realized.append(NewLightpathRun(oms_seq, lam, mode.id, gsnr,
                                                 mode.bitrate_gbps,
                                                 src_node=run_src, dst_node=run_dst))
                new_cap = min(new_cap, mode.bitrate_gbps)
            if not feasible:
                continue
            groom_cap = _bottleneck_residual(g, reused)
            restored = min(demand_gbps, groom_cap, new_cap)
            if restored <= 0.0:
                continue
            placement = Placement(
                reused_lightpaths=tuple(reused),
                new_lightpaths=tuple(realized),
                restored_gbps=restored,
                shortfall_gbps=max(0.0, demand_gbps - restored),
            )
            out.append(placement)
            if stop_when is not None and stop_when(placement):
                # A caller that needs one acceptable answer, not a ranked set.
                # Every further route costs two GNPy propagations (forward and
                # backward, via _best_feasible_mode) and would be discarded.
                return out
    return out
