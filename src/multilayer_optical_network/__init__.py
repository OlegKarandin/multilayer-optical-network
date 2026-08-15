"""Public API for multilayer-optical-network.

This module re-exports the main types and functions that consumers need,
avoiding the need to import from internal module paths. Imports are lazy to
allow the optical layer to be used independently without loading IP modules.
"""

from typing import TYPE_CHECKING

__version__ = "0.1.0"

_LAZY_IMPORTS = {
    "NetworkModel": (".model.network", "NetworkModel"),
    "OpticalNetworkModel": (".model.optical_network", "OpticalNetworkModel"),
    "default_modes": (".model.modes", "default_modes"),
    "load_modulation_formats": (".model.modes", "load_modulation_formats"),
    "load_model_from_topology_file": (".topology_loader", "load_model_from_topology_file"),
    "load_model_from_state_file": (".state_file", "load_model_from_state_file"),
    "model_from_abstract_graph": (".model.topology_import", "model_from_abstract_graph"),
    "compute_qot": (".gnpy_adapter.adapter", "compute_qot"),
    "recompute_qot_under_loading": (".gnpy_adapter.adapter", "recompute_qot_under_loading"),
    "LoadingState": (".gnpy_adapter.loading", "LoadingState"),
    "Channel": (".gnpy_adapter.loading", "Channel"),
    "Direction": (".model.assets", "Direction"),
    "SolverStatus": (".model.solvers", "SolverStatus"),
    "compute_paths": (".model.solvers", "compute_paths"),
    "compute_disjoint_paths": (".model.solvers", "compute_disjoint_paths"),
    "check_disjointness": (".model.solvers", "check_disjointness"),
    "solve_rsa": (".model.allocation", "solve_rsa"),
    "solve_allocation": (".model.allocation", "solve_allocation"),
    "simulate_ip_routing": (".model.ip_routing", "simulate_ip_routing"),
    "validate_plan": (".model.validate", "validate_plan"),
    "SnapshotStore": (".model.snapshots", "SnapshotStore"),
}

__all__ = sorted(_LAZY_IMPORTS.keys())

# PEP 561: a TYPE_CHECKING-only eager-import block so static type checkers
# resolve the curated re-exports below to their real types instead of `Any`.
# `TYPE_CHECKING` is always False at runtime, so this never executes and never
# reintroduces the eager-import cost `__getattr__` exists to avoid (see
# test_optical_model_imports_without_ip_layer, which requires that importing
# just the optical layer not pull in IP modules at runtime). Keep this in sync
# with `_LAZY_IMPORTS` above.
if TYPE_CHECKING:
    from .model.network import NetworkModel  # noqa: F401
    from .model.optical_network import OpticalNetworkModel  # noqa: F401
    from .model.modes import default_modes, load_modulation_formats  # noqa: F401
    from .topology_loader import load_model_from_topology_file  # noqa: F401
    from .state_file import load_model_from_state_file  # noqa: F401
    from .model.topology_import import model_from_abstract_graph  # noqa: F401
    from .gnpy_adapter.adapter import compute_qot, recompute_qot_under_loading  # noqa: F401
    from .gnpy_adapter.loading import LoadingState, Channel  # noqa: F401
    from .model.assets import Direction  # noqa: F401
    from .model.solvers import (  # noqa: F401
        SolverStatus,
        compute_paths,
        compute_disjoint_paths,
        check_disjointness,
    )
    from .model.allocation import solve_rsa, solve_allocation  # noqa: F401
    from .model.ip_routing import simulate_ip_routing  # noqa: F401
    from .model.validate import validate_plan  # noqa: F401
    from .model.snapshots import SnapshotStore  # noqa: F401


def __getattr__(name: str):
    """Lazy-import public API symbols on access."""
    if name in _LAZY_IMPORTS:
        module_path, symbol = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, symbol)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__ + ["__version__"])
