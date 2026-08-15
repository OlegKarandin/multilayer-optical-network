# multilayer-optical-network

[![CI](https://github.com/OlegKarandin/multilayer-optical-network/actions/workflows/ci.yml/badge.svg)](https://github.com/OlegKarandin/multilayer-optical-network/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A deterministic multi-layer optical network model: a typed IP-over-optical
topology, a [GNPy](https://github.com/Telecominfraproject/oopt-gnpy)
physical-layer quality-of-transmission (QoT) adapter, routing/RSA/disjoint-path
solvers, and a plan validator — usable standalone as a library, or as the
physical/IP-layer backend behind
[`multilayer-optical-mcp`](https://github.com/OlegKarandin/multilayer-optical-mcp),
which exposes this model's primitives as MCP tools for an agent to call. No
MCP dependency lives in this repo.

## What it does

- **IP-over-optical coupling** — an IP link is bound to the lightpath that
  carries it; its capacity is *derived* from that lightpath's transceiver
  mode, never stored. A modulation downshift, or a loading change that pushes
  margin negative, propagates automatically to reduced (or zero) IP capacity.
- **What-if analysis** — a physics-free margin-threshold screening sweep and
  a physically-grounded degradation path (`inject_degradation` →
  `recompute_qot_under_loading` → threshold crossings). Margin is always an
  *output*, never a dial.
- **Dynamic risk groups** — runtime-injected asset partitions, tested
  independently from static design-time SRLGs, so a working/protection pair
  certified disjoint at design time can be re-audited against a partition
  that didn't exist yet — see `examples/risk_group_exposure.py`.
- **Deterministic solvers** — k-shortest paths, disjoint-path computation,
  RSA, and a heuristic multi-layer allocator, each returning a typed
  `solution` / `partial` / `no_solution` result — never an exception for an
  infeasible request.
- **Snapshot/branch state engine** — every mutation is simulatable on a
  branch before it touches ground truth; `validate_plan` must pass before
  `commit_plan`, and `reconcile()` reads actual state back after a live
  commit to surface drift from partial control-plane failures.

## Architecture

```
                          GNPy adapter           Solvers          Validator
                          (QoT under            (k-shortest,      (typed
                           loading, spectrum     RSA, disjoint,    violation
                           feasibility)          heuristic alloc)  lists)
                                    |                  |               |
                                    +------------------+---------------+
                                                   |
                          State engine (snapshot/branch/diff)
                                                   |
                                       Network model (in-memory)
                          IP layer over optical layer; services;
                          static SRLGs + dynamic risk groups
```

## Install

Requires Python >= 3.11.

```bash
pip install multilayer-optical-network
```

For development (tests, lint):

```bash
pip install -e ".[dev]"
```

`pyproject.toml` is the single source of dependency truth — there is no
separate `requirements*.txt`.

## Use as a library

```python
from multilayer_optical_network import default_modes, load_model_from_state_file
from multilayer_optical_network.data import reference_topology, reference_state

model = load_model_from_state_file(
    reference_topology("german_17"), reference_state("german_17"), modes=default_modes())

print(len(model.list_lightpaths()), "lightpaths", len(model.list_services()), "services")
```

`reference_topology`/`reference_state` resolve packaged reference data —
`german_17`, a 17-node topology, and a prebuilt operating network on it (45
lightpaths, 88 services) so this loads in under a second instead of paying
the real-GNPy build cost. Build your own topology's operating network with
`multilayer-optical-network-build` (below), or import an arbitrary one with
`load_model_from_topology_file`.

## Demo

`examples/risk_group_exposure.py` walks a design-time-disjoint protection
pair through a runtime-injected risk group that reveals it was correlated all
along — exposure audit, disjointness re-check under two bases, a real failure
injection proving the correlation (contrasted against an unrelated failure
that protection survives normally), replan, heal, validate, and a live
commit + reconcile. MCP-free, seeded, deterministic:

```bash
python examples/risk_group_exposure.py
```

```
loaded german_17: 45 lightpaths, 88 services

protected service d0004: working=('ipl-cand-d0002-0', 'ipl-cand-d0001-0') protection=('ipl-prot-d0004-0',)
hazard zone (stand-in for a downstream geo-mapper's polygon): ('amp_0_1_0', 'amp_1_6_0')

[exposure] both legs intersect the hazard zone: True
[disjointness] basis=physical:   disjoint=True
[disjointness] basis=risk_group: disjoint=False shared=('hazard-zone-1',)
[validate_plan] DISJOINTNESS_COLLAPSE on d0004: True

[failure injected] 5 lightpath(s) down; d0004 protection-leg capacity after failure: 0.0 Gbps
[failure injected] d0004 dropped: True
[contrast] unrelated failure elsewhere -- d0035 restored onto protection: True

[replan] new risk_group-disjoint pair found: disjoint=True
[heal] 6 additional op(s) to clear every other endpoint violation the hazard revealed; converged=True

[commit] dry_run status=dry_run network-clean=True
[commit] live status=committed
[reconcile] in_sync=True drift=()

d0004's protection leg now runs ('ipl-fix-d0004-0',), disjoint from working under basis=risk_group.
```

Note the zone is a hand-built asset list, not a storm polygon or a geo
lookup — turning an event into that list is what a downstream application
does; this repo only ever takes the resulting asset list (`define_risk_group`
has no `map_geo_event_to_assets` counterpart, by design).

## Build a full operating network

`multilayer-optical-network-build` is the offline batch job behind the
prebuilt state shown above: it drives GNPy through a heuristic packer to
produce a loaded operating network (lightpaths with settled QoT, IP links,
a demand-derived service inventory with protection) and writes it to a state
file. It's a separate console script, not something re-run on every process
start — minutes on a small topology, tens of minutes on a large one.

```bash
multilayer-optical-network-build --topology topo.json --out state.json
```

`state.json` is a delta keyed to `topo.json` by a content fingerprint and the
GNPy version that produced it (`state_file.py`); loading it against a
different topology raises a structured `StateFileError`, not a confusing
downstream failure. Load the pair back with:

```python
from multilayer_optical_network import default_modes, load_model_from_state_file
model = load_model_from_state_file("topo.json", "state.json", modes=default_modes())
```

or, server-side, `multilayer-optical-mcp --topology topo.json --state
state.json` in the [server repo](https://github.com/OlegKarandin/multilayer-optical-mcp).

`--topology` alone (no `--state`) loads just the physical layer — ROADMs,
fibers, spans, SRLGs — with no lightpaths, IP links, or services.

Useful flags for a large or sparse topology: `--pair-density` (fraction of
node pairs offered demand — omit for the full matrix, but note that on a
large topology the full matrix quantizes every pair's offered load to zero),
`--unit-gbps` (demand quantization unit), `--protected-fraction`, and
`--protection-basis`/`--protection-level`/`--protection-best-effort` to
control which disjointness basis new protection legs are routed against.
Run `multilayer-optical-network-build --help` for the full list.

## Test

```bash
pytest tests/
```

The suite is deterministic and seeded — same inputs, same outputs, every
time. A handful of slow, real-GNPy ground-truth tests are skipped by default;
opt in with:

```bash
OPTICAL_NET_RUN_GNPY_E2E=1 pytest tests/
```

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
