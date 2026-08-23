from __future__ import annotations
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple
from .qot import QoTBreakdown, QoTState


class QoTResultStore:
    def __init__(
        self,
        max_results: Optional[int] = 512,
        ttl_seconds: Optional[float] = 600.0,
    ) -> None:
        self._items: OrderedDict[str, QoTBreakdown] = OrderedDict()
        self._created_at: Dict[str, float] = {}
        self._max = max_results
        self._ttl = ttl_seconds

    def put(self, breakdown: QoTBreakdown) -> str:
        self.reap()
        rid = uuid.uuid4().hex
        self._items[rid] = breakdown
        self._created_at[rid] = time.monotonic()
        if self._max is not None and len(self._items) > self._max:
            oldest, _ = self._items.popitem(last=False)
            self._created_at.pop(oldest, None)
        return rid

    def get(self, rid: str) -> QoTBreakdown:
        return self._items[rid]

    def reap(self) -> Tuple[str, ...]:
        if self._ttl is None:
            return ()
        now = time.monotonic()
        expired = [r for r, t in self._created_at.items() if now - t > self._ttl]
        for r in expired:
            self._items.pop(r, None)
            self._created_at.pop(r, None)
        return tuple(expired)


class QoTCache:
    """Content-addressed memo of ``compute_qot`` results, keyed by a fingerprint
    of every GSNR input (see ``adapter._cache_key``). Off-model and injected like
    ``QoTResultStore`` — it holds no clone/diff/freeze surface.

    There is deliberately NO invalidation: a mutated span (e.g. an
    ``inject_degradation`` NF delta) yields a different key, so a stale entry is
    simply never hit again and ages out via the bounded LRU. The one correctness
    invariant is fingerprint completeness — if the key omits a GSNR input, a hit
    returns a confident wrong number.
    """

    def __init__(self, maxsize: int = 4096) -> None:
        self._d: "OrderedDict[Any, Tuple[QoTState, QoTBreakdown]]" = OrderedDict()
        self._max = maxsize
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Optional[Tuple[QoTState, QoTBreakdown]]:
        hit = self._d.get(key)
        if hit is None:
            self.misses += 1
            return None
        self._d.move_to_end(key)          # LRU: mark most-recently used
        self.hits += 1
        return hit

    def put(self, key: Any, value: Tuple[QoTState, QoTBreakdown]) -> None:
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self._max:
            self._d.popitem(last=False)   # evict least-recently used


class HarvestCache:
    """Bounded LRU of full-comb harvest vectors, keyed by adapter.harvest_cache_key
    — `(mode_id, path physical fingerprint)`, and nothing else. No probe
    frequency (one harvest answers every slot at once), and neither the
    `oms_sequence` nor the `direction`: a harvest's value is a bare slot ->
    QoTState vector carrying no identity, so two requests whose resolved element
    chains carry identical physics deliberately alias onto one entry. On an
    undamaged (symmetric) span that is exactly a lightpath's forward and backward
    request. Off-model, injected like QoTCache. Content-addressed: a changed
    physical input — including an asymmetric one, which re-splits the two
    directions on its own — flips the key, so there is no invalidation logic."""

    def __init__(self, maxsize: int = 4096) -> None:
        self._store: "OrderedDict[Any, Dict[int, Any]]" = OrderedDict()
        self._maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Optional[Dict[int, Any]]:
        if key not in self._store:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return self._store[key]

    def put(self, key: Any, value: Dict[int, Any]) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self._maxsize:
            self._store.popitem(last=False)


class IncrementCache:
    """Bounded LRU of per-OMS ``1/gsnr_lin`` increment vectors, keyed by
    ``gnpy_adapter.composition.oms_fingerprint(model, oms_id)`` -- a single OMS's
    own physical fingerprint, not a path's. Populated from the SAME capture pass
    that already backs ``HarvestCache`` (``adapter._propagate_loading(...,
    capture_increments=True)``), so a K-hop harvest calibrates K entries here in
    one propagation, and any FUTURE path that reuses any of those K OMS -- in any
    order, any combination -- can compose its own GSNR from cached increments
    without a fresh propagation (see ``gnpy_adapter/composition.py``'s module
    docstring for the composition identity `AdapterEvaluator.compose_gsnr` reads
    this table through).

    Content-addressed like ``HarvestCache``/``QoTCache``: a changed physical
    input flips the fingerprint, so there is no invalidation logic -- a stale
    entry for an old fingerprint simply stops being looked up again and ages out
    via the bounded LRU, same discipline as ``HarvestCache``."""

    def __init__(self, maxsize: int = 4096) -> None:
        self._store: "OrderedDict[Any, Dict[int, float]]" = OrderedDict()
        self._maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Optional[Dict[int, float]]:
        if key not in self._store:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return self._store[key]

    def put(self, key: Any, value: Dict[int, float]) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self._maxsize:
            self._store.popitem(last=False)
