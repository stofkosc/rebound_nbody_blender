"""Reading the Blender scene into core specs."""

import bpy

from . import core


def body_objects(scene):
    objs = [o for o in scene.objects if o.rebound.enabled and o.rebound.kind == "BODY"]
    return sorted(objs, key=lambda o: (o.rebound.order, o.name))


def swarm_objects(scene):
    objs = [o for o in scene.objects if o.rebound.enabled and o.rebound.kind == "SWARM"]
    return sorted(objs, key=lambda o: o.name)


def length_scale(scene):
    ls = scene.rebound.length_scale
    return ls if ls > 0 else 1.0


def initial_location(obj):
    """World-space location that defines the initial position (Blender units)."""
    rb = obj.rebound
    if rb.has_initial:
        return tuple(rb.initial_location)
    return tuple(obj.matrix_world.translation)


def n_frames(scene):
    return max(1, scene.frame_end - scene.frame_start + 1)


def _build_raw_specs(scene, include_swarms=True):
    ls = length_scale(scene)
    bodies = body_objects(scene)
    slot_of = {o.name: i for i, o in enumerate(bodies)}

    def slot(obj):
        if obj is None:
            return None
        return slot_of.get(obj.name)

    specs = []
    for i, o in enumerate(bodies):
        rb = o.rebound
        prim = slot(rb.primary)
        if prim == i:
            raise core.SetupError(f"'{o.name}' cannot be its own primary.")
        loc = initial_location(o)
        specs.append(core.BodySpec(
            slot=i, name=o.name, mass=rb.mass, radius=rb.radius, mode=rb.ic_mode,
            pos=tuple(c / ls for c in loc), vel=tuple(rb.velocity),
            primary=prim if rb.ic_mode == "ORBIT" else None, ref=prim,
            a=rb.a, e=rb.e, inc=rb.inc, Omega=rb.Omega, omega=rb.omega,
            anomaly=rb.anomaly, anomaly_type=rb.anomaly_type, order=rb.order))

    swarms = swarm_objects(scene) if include_swarms else []
    sw_specs = []
    for o in swarms:
        rb = o.rebound
        sw_specs.append(core.SwarmSpec(
            name=o.name, count=rb.count, primary=slot(rb.primary),
            a_min=rb.a_min, a_max=rb.a_max, e_max=rb.e_max, inc_max=rb.inc_max,
            lon_center=rb.lon_center, lon_width=rb.lon_width, seed=rb.seed))
    return specs, sw_specs, bodies, swarms, slot_of


def reference_period(scene, specs=None, sw_specs=None, spec=None):
    """Characteristic orbital period (in simulation time units) of ``scene.rebound.duration_body``
    or the shortest bound orbit in the scene/spec."""
    s = scene.rebound
    ref_name = s.duration_body.name if (s.duration_body and s.duration_body.rebound.enabled) else None
    if spec is None:
        if specs is None:
            try:
                specs, sw_specs, *_ = _build_raw_specs(scene, include_swarms=True)
            except core.SetupError:
                return None
        spec = core.SimSpec(bodies=specs, swarms=sw_specs or [], units=s.units, G=s.G)
    return core.characteristic_period(spec, body_name=ref_name)


def resolve_sim_timing(scene, specs=None, sw_specs=None):
    """Compute ``(n_snapshots, time_per_snapshot, total_time, ref_period)`` from
    the scene's Duration & Sampling settings."""
    s = scene.rebound
    P_ref = reference_period(scene, specs=specs, sw_specs=sw_specs)
    tl_frames = n_frames(scene)
    base_tpf = max(1e-12, float(s.time_per_frame))

    # 1. Total physical duration (in simulation time units)
    if s.duration_mode == "ORBITS":
        if P_ref and P_ref > 0:
            total_time = max(1e-12, float(s.duration_orbits) * P_ref)
        else:
            total_time = max(1, tl_frames - 1) * base_tpf
    elif s.duration_mode == "TIME":
        ufac = core.time_unit_factor(s.units, s.duration_unit)
        total_time = max(1e-12, float(s.duration_time) * ufac)
    else:  # 'FRAMES'
        if s.sampling_mode == "SNAPSHOTS":
            total_time = max(1, int(s.sample_count) - 1) * base_tpf
        else:
            total_time = max(1, tl_frames - 1) * base_tpf

    # 2. Number of snapshots & time per snapshot
    if s.sampling_mode == "SNAPSHOTS":
        nf = max(2, int(s.sample_count))
        tpf = total_time / (nf - 1)
    elif s.sampling_mode == "PER_ORBIT" and P_ref and P_ref > 0:
        n_orbits = total_time / P_ref
        nf = max(2, int(round(n_orbits * max(0.01, float(s.samples_per_orbit)))) + 1)
        nf = min(nf, 500_000)
        tpf = total_time / (nf - 1)
    elif s.sampling_mode == "STEP":
        tpf = base_tpf
        nf = max(2, int(round(total_time / tpf)) + 1)
        nf = min(nf, 500_000)
        total_time = (nf - 1) * tpf
    else:  # 'TIMELINE'
        nf = max(2, tl_frames) if total_time > 0 else 1
        tpf = total_time / (nf - 1) if nf > 1 else base_tpf

    return int(nf), float(max(1e-12, tpf)), float(total_time), P_ref


def make_spec(scene, include_swarms=True, frames=None, archive=False):
    """Build a core.SimSpec from the scene.

    Returns (spec, body_objs, swarm_objs, slots) where slots maps
    object name -> body slot."""
    s = scene.rebound
    specs, sw_specs, bodies, swarms, slot_of = _build_raw_specs(scene, include_swarms=include_swarms)

    if frames == 1:
        nf = 1
        tpf = max(1e-12, float(s.time_per_frame))
    else:
        nf, tpf, _total_t, _pref = resolve_sim_timing(scene, specs=specs, sw_specs=sw_specs)
        if frames and frames > 0:
            nf = min(int(frames), nf)

    archive_path = None
    if archive and s.archive and s.archive_path:
        archive_path = bpy.path.abspath(s.archive_path)

    spec = core.SimSpec(
        bodies=specs, swarms=sw_specs, units=s.units, G=s.G, integrator=s.integrator,
        dt_mode=s.dt_mode, dt=s.dt, dt_fraction=s.dt_fraction,
        time_per_frame=tpf, n_frames=nf, exact=s.exact_mode,
        move_to_com=s.move_to_com, collision=s.collision,
        collision_resolve=s.collision_resolve, archive_path=archive_path)
    return spec, bodies, swarms, slot_of


def frame_slots(scene, slot_of):
    s = scene.rebound
    c = slot_of.get(s.frame_center.name) if s.frame_center else None
    sec = slot_of.get(s.frame_secondary.name) if s.frame_secondary else None
    return c, sec


def transforms_for(scene, runner, slot_of, k=None):
    """Frame transforms (Q, c, d) for the first k frames of a runner."""
    s = scene.rebound
    k = runner.k if k is None else k
    c, sec = frame_slots(scene, slot_of)
    mode = s.frame_mode
    if mode == "CENTERED" and c is None:
        mode = "INERTIAL"
    if mode == "COROTATING" and (c is None or sec is None):
        mode = "INERTIAL"
    return core.frame_transforms(runner.bpos[:k], runner.bvel[:k], runner.bmass[:k], mode,
                                 c, sec, s.anchor_frame, runner.offset0)


def unit_labels(scene):
    return core.UNIT_LABELS.get(scene.rebound.units, ("L", "T", "M"))


def most_massive(scene, exclude=None):
    best = None
    for o in body_objects(scene):
        if o is exclude:
            continue
        if best is None or o.rebound.mass > best.rebound.mass:
            best = o
    return best
