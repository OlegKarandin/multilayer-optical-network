"""Risk-group exposure demo: a design-time-disjoint protection pair that a
runtime hazard reveals was correlated all along.

Deterministic and seeded: loads the packaged german_17 reference network
(46 routers, prebuilt to 45 lightpaths / 88 services -- see
`multilayer_optical_network.data.reference_state`) instead of rebuilding it
against GNPy, so this runs in well under a second. MCP-free: everything here
is a direct call into the library, the same one `pip install
multilayer-optical-network` gives you.

The story:

  1. A protected IP service has working and protection legs that were routed
     physically disjoint -- no shared fiber or amplifier -- which is the only
     disjointness a design-time build can check.
  2. At runtime, something -- a storm cell, a construction dig-in, a flood
     plain, anything that damages a set of physical assets together -- turns
     out to cover one span on EACH leg. This repo never sees the storm or the
     flood; it only ever receives the resulting asset list. Turning an event
     into that list is a downstream application's job (see CLAUDE.md's "no
     weather/geo logic" rule) -- here we hand-build the list to stand in for
     it, the same way `define_risk_group` expects it from any caller.
  3. Re-checking the SAME pair against this new "risk group" basis exposes a
     correlation that the physical-disjointness check, by construction, could
     never see.
  4. We prove it isn't just theoretical: actually failing the zone's assets
     takes the service's protection down too -- contrasted against an
     unrelated failure elsewhere, which protection survives normally.
  5. We replan the protection leg around the risk group, heal every other
     violation the same hazard revealed elsewhere in the network, validate,
     and commit -- through the exact gated path a live network would use.
"""
from __future__ import annotations

from multilayer_optical_network import default_modes, load_model_from_state_file
from multilayer_optical_network.data import reference_state, reference_topology
from multilayer_optical_network.model import objective
from multilayer_optical_network.model.allocation import make_adapter_evaluator
from multilayer_optical_network.model.commit import commit_plan, reconcile
from multilayer_optical_network.model.exposure import compute_exposure, service_asset_set
from multilayer_optical_network.model.ip_routing import simulate_ip_routing
from multilayer_optical_network.model.plan import (
    Plan, ProvisionLightpath, RerouteService, apply_op, service_oms_sequence,
)
from multilayer_optical_network.model.qot_results import QoTCache, QoTResultStore
from multilayer_optical_network.model.route_service import route_service
from multilayer_optical_network.model.snapshots import SnapshotStore
from multilayer_optical_network.model.solvers import SolverStatus, check_disjointness
from multilayer_optical_network.model.validate import ViolationType, validate_plan
from multilayer_optical_network.model.whatif import inject_failure


def _physical_span_ids(assets) -> list[str]:
    """fiber_/amp_ ids only -- a risk-group member must be a bare physical
    span, never a shared path endpoint."""
    return sorted(a for a in assets if a.startswith("fiber_") or a.startswith("amp_"))


def _pick_correlated_service(model):
    """First protected service with a physical span on each leg, plus a
    synthetic 2-asset hazard zone covering one from each -- the runtime
    correlation a design-time build had no way to anticipate."""
    for svc in model.list_services():
        if not svc.protection_path:
            continue
        working = _physical_span_ids(service_asset_set(model, svc.id, which="working"))
        protection = _physical_span_ids(service_asset_set(model, svc.id, which="protection"))
        if not working or not protection or working[0] == protection[0]:
            continue
        return svc, frozenset((working[0], protection[0]))
    raise AssertionError("no protected service with a disjoint physical leg pair found")


def _materialize_and_collect(work, placement, svc, *, prefix, ops):
    """Provision `placement` as `svc`'s new protection leg on `work`, and
    record the ops that reproduce it."""
    before = {lp.id for lp in work.list_lightpaths()}
    new_path, _seeded = objective.provision_new_runs(work, placement, svc, prefix=prefix)
    apply_op(work, RerouteService(svc.id, new_path, which="protection"))
    for lp_id in sorted({lp.id for lp in work.list_lightpaths()} - before):
        lp_obj = work.get_lightpath(lp_id)
        ip_link = next((ipl for ipl in work.list_ip_links() if ipl.lightpath_id == lp_id), None)
        ops.append(ProvisionLightpath(lightpath=lp_obj, ip_link=ip_link))
    ops.append(RerouteService(svc.id, new_path, which="protection"))
    return new_path


def _replan_protection(work, qot, service_id, *, prefix, avoid, ops):
    """Replan `service_id`'s protection leg away from `avoid`, and apply the
    first candidate that isn't a literal duplicate of its own working path."""
    svc = work.get_service(service_id)
    res = route_service(work, qot, service_id, protected=True, basis="risk_group",
                         level="link", avoid=avoid)
    if res.status not in (SolverStatus.SOLUTION, SolverStatus.PARTIAL) or not res.pairs:
        return False
    for idx, pair in enumerate(res.pairs):
        probe = work.clone()
        probe_path = _materialize_and_collect(
            probe, pair.protection, probe.get_service(service_id),
            prefix=f"{prefix}-probe{idx}", ops=[])
        if probe_path != svc.working_path:
            _materialize_and_collect(work, pair.protection, svc, prefix=prefix, ops=ops)
            return True
    return False


def _heal_endpoint_violations(work, qot, risk_group_id, store, *, max_rounds=10):
    """A real, heuristically-packed network can carry violations unrelated to
    the one service we set out to fix -- another service crossing the same
    zone, or pre-existing 1:1 protection oversubscription (`solve_allocation`
    makes no optimality claim, see CLAUDE.md). Round-trip: find every
    DISJOINTNESS_COLLAPSE / PROTECTION_OVERSUBSCRIBED violation under the
    risk-group basis, replan one affected service's protection leg, repeat
    until clean or out of budget. Returns (ops, converged)."""
    ops: list = []
    for round_no in range(max_rounds):
        report = validate_plan(work, Plan(ops=()), store=store, basis="risk_group", level="link")
        if report.ok:
            return tuple(ops), True

        progressed = False
        collapse = [v for v in report.violations if v.type is ViolationType.DISJOINTNESS_COLLAPSE]
        for v in collapse:
            if not work.get_service(v.asset_id).protection_path:
                continue
            if _replan_protection(work, qot, v.asset_id, prefix=f"heal{round_no}",
                                   avoid={"risk_groups": (risk_group_id,)}, ops=ops):
                progressed = True
                break

        if not progressed:
            oversub = [v for v in report.violations
                       if v.type is ViolationType.PROTECTION_OVERSUBSCRIBED]
            for v in oversub:
                for sid in sorted(v.detail["reserving_services"]):
                    if _replan_protection(work, qot, sid, prefix=f"heal{round_no}",
                                           avoid={"risk_groups": (risk_group_id,)}, ops=ops):
                        progressed = True
                        break
                if progressed:
                    break

        if not progressed:
            return tuple(ops), False
    return tuple(ops), False


def main() -> None:
    modes = default_modes()
    model = load_model_from_state_file(
        reference_topology("german_17"), reference_state("german_17"), modes=modes)
    print(f"loaded german_17: {len(model.list_lightpaths())} lightpaths, "
          f"{len(model.list_services())} services")

    svc, zone_assets = _pick_correlated_service(model)
    zone = tuple(sorted(zone_assets))
    print(f"\nprotected service {svc.id}: working={svc.working_path} "
          f"protection={svc.protection_path}")
    print(f"hazard zone (stand-in for a downstream geo-mapper's polygon): {zone}")

    # ---- 1. exposure audit -------------------------------------------------
    branch = model.clone()
    rg = branch.define_risk_group("hazard-zone-1", zone)
    exposure = compute_exposure(branch, svc.id, rg.id)
    print(f"\n[exposure] both legs intersect the hazard zone: {exposure.both_intersect}")

    # ---- 2. disjointness under two bases -----------------------------------
    working_oms = service_oms_sequence(branch, svc.working_path)
    protection_oms = service_oms_sequence(branch, svc.protection_path)
    endpoints = (branch.get_router(svc.src_router).site, branch.get_router(svc.dst_router).site)
    physical = check_disjointness(branch, working_oms, protection_oms, basis="physical",
                                   level="link", endpoints_a=endpoints, endpoints_b=endpoints)
    by_risk_group = check_disjointness(branch, working_oms, protection_oms, basis="risk_group",
                                        level="link", endpoints_a=endpoints, endpoints_b=endpoints)
    print(f"[disjointness] basis=physical:   disjoint={physical.disjoint}")
    print(f"[disjointness] basis=risk_group: disjoint={by_risk_group.disjoint} "
          f"shared={by_risk_group.shared_groups}")

    # ---- 3. validate_plan sees it too --------------------------------------
    store = QoTResultStore()
    report = validate_plan(branch, Plan(ops=()), store=store, basis="risk_group", level="link")
    collapse = [v for v in report.violations
                if v.type is ViolationType.DISJOINTNESS_COLLAPSE and v.asset_id == svc.id]
    print(f"[validate_plan] DISJOINTNESS_COLLAPSE on {svc.id}: {len(collapse) == 1}")

    # ---- 4. prove it: the hazard actually takes protection down -----------
    failed = branch.clone()
    failure = inject_failure(failed, zone)
    ipr = simulate_ip_routing(failed)
    prot_caps = [failed.ip_link_capacity_gbps(ip) for ip in svc.protection_path]
    print(f"\n[failure injected] {len(failure.downed_lightpaths)} lightpath(s) down; "
          f"{svc.id} protection-leg capacity after failure: {min(prot_caps)} Gbps")
    dropped = {d.service_id for d in ipr.dropped_services}
    print(f"[failure injected] {svc.id} dropped: {svc.id in dropped}")

    other = next((s for s in model.list_services()
                  if s.id != svc.id and s.protection_path
                  and check_disjointness(branch, service_oms_sequence(branch, s.working_path),
                                          service_oms_sequence(branch, s.protection_path),
                                          basis="risk_group", level="link").disjoint), None)
    if other is not None:
        independent = [a for a in _physical_span_ids(service_asset_set(branch, other.id, which="working"))
                       if a not in zone_assets]
        if independent:
            unaffected = branch.clone()
            inject_failure(unaffected, (independent[0],))
            ipr2 = simulate_ip_routing(unaffected)
            print(f"[contrast] unrelated failure elsewhere -- {other.id} restored onto "
                  f"protection: {other.id in ipr2.restored_services}")

    # ---- 5. replan away from the hazard, heal, validate, commit -----------
    cache = QoTCache()
    qot = make_adapter_evaluator(branch, store, cache=cache)
    result = route_service(branch, qot, svc.id, protected=True, basis="risk_group",
                            level="link", avoid={"risk_groups": (rg.id,)})
    pair = result.pairs[0]
    print(f"\n[replan] new risk_group-disjoint pair found: disjoint={pair.disjoint}")

    work = branch.clone()
    ops: list = []
    new_protection_path = _materialize_and_collect(work, pair.protection, svc,
                                                     prefix="fix", ops=ops)
    heal_ops, converged = _heal_endpoint_violations(work, qot, rg.id, store)
    ops.extend(heal_ops)
    print(f"[heal] {len(heal_ops)} additional op(s) to clear every other endpoint "
          f"violation the hazard revealed; converged={converged}")

    plan = Plan(ops=tuple(ops))
    snapshots = SnapshotStore(initial=branch)

    dry = commit_plan(snapshots, plan, store_results=store, dry_run=True,
                       basis="risk_group", level="link")
    print(f"\n[commit] dry_run status={dry.status} network-clean={dry.validation.ok}")

    live = commit_plan(snapshots, plan, store_results=store, dry_run=False,
                        confirm=True, basis="risk_group", level="link")
    print(f"[commit] live status={live.status}")

    drift = reconcile(snapshots, live.intended_snapshot_id)
    print(f"[reconcile] in_sync={drift.in_sync} drift={drift.drift}")

    print(f"\n{svc.id}'s protection leg now runs {new_protection_path}, "
          f"disjoint from working under basis=risk_group.")


if __name__ == "__main__":
    main()
