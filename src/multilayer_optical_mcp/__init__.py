"""Public API for multilayer-optical-mcp.

This module re-exports the main types and functions that consumers need,
avoiding the need to import from internal module paths. Imports are lazy to
allow the optical layer to be used independently without loading IP modules.
"""

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


def __getattr__(name: str):
    """Lazy-import public API symbols on access."""
    if name in _LAZY_IMPORTS:
        module_path, symbol = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, symbol)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
