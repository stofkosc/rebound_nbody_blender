"""Pure-Python core of the add-on (no ``bpy`` imports).

It turns plain specification objects into a REBOUND simulation, integrates it
frame by frame, stores the samples in numpy arrays and converts them into
visualisation data (reference frames, osculating ellipses, quaternions).
Keeping this free of Blender makes it testable with plain Python + REBOUND.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

TWO_PI = 2.0 * math.pi

# REBOUND unit tuples, ordered (time, length, mass) as REBOUND expects them.
UNIT_SYSTEMS = {
    "CODE": None,
    "AU_YR2PI_MSUN": ("yr2pi", "AU", "Msun"),
    "AU_YR_MSUN": ("yr", "AU", "Msun"),
    "AU_DAY_MSUN": ("day", "AU", "Msun"),
    "SI": ("s", "m", "kg"),
}

UNIT_LABELS = {  # (length, time, mass)
    "CODE": ("L", "T", "M"),
    "AU_YR2PI_MSUN": ("AU", "yr/2\u03c0", "Msun"),
    "AU_YR_MSUN": ("AU", "yr", "Msun"),
    "AU_DAY_MSUN": ("AU", "d", "Msun"),
    "SI": ("m", "s", "kg"),
}

# Standard gravitational constants in each unit system (fallback when REBOUND isn't queried)
DEFAULT_G_BY_UNITS = {
    "AU_YR2PI_MSUN": 1.0,
    "AU_YR_MSUN": 4.0 * math.pi * math.pi,
    "AU_DAY_MSUN": 0.00029591220828559104,
    "SI": 6.6743e-11,
}

# Seconds in standard physical time units
_SECONDS_PER_UNIT = {
    "SEC": 1.0,
    "DAY": 86400.0,
    "YR": 365.25 * 86400.0,
    "KYR": 1e3 * 365.25 * 86400.0,
    "MYR": 1e6 * 365.25 * 86400.0,
}

# Seconds per 1 simulation time unit for each real unit system
_SIM_TIME_IN_SECONDS = {
    "AU_YR2PI_MSUN": (365.25 * 86400.0) / TWO_PI,
    "AU_YR_MSUN": 365.25 * 86400.0,
    "AU_DAY_MSUN": 86400.0,
    "SI": 1.0,
}


def time_unit_factor(sim_units: str, chosen_unit: str) -> float:
    """Return how many simulation time units equal 1 ``chosen_unit``
    ('SIM', 'YR', 'KYR', 'MYR', 'DAY', 'SEC')."""
    if not chosen_unit or chosen_unit == "SIM" or sim_units == "CODE":
        if sim_units == "CODE" and chosen_unit in {"KYR", "MYR"}:
            return 1e3 if chosen_unit == "KYR" else 1e6
        return 1.0
    sim_sec = _SIM_TIME_IN_SECONDS.get(sim_units)
    target_sec = _SECONDS_PER_UNIT.get(chosen_unit)
    if sim_sec and target_sec:
        return target_sec / sim_sec
    return 1.0


def time_unit_label(sim_units: str, chosen_unit: str) -> str:
    if not chosen_unit or chosen_unit == "SIM":
        return UNIT_LABELS.get(sim_units, ("L", "T", "M"))[1]
    labels = {"YR": "yr", "KYR": "kyr", "MYR": "Myr", "DAY": "d", "SEC": "s"}
    if sim_units == "CODE" and chosen_unit not in {"KYR", "MYR"}:
        return UNIT_LABELS["CODE"][1]
    return labels.get(chosen_unit, chosen_unit)

ADAPTIVE_INTEGRATORS = {"ias15", "bs"}

# Above this particle count the energy is only evaluated at the first and the
# last frame (it is an O(N^2) operation).
ENERGY_EVERY_FRAME_MAX_N = 1500


class SetupError(Exception):
    """A problem with the user's setup that should be reported, not raised."""


# --------------------------------------------------------------------------
# Specifications
# --------------------------------------------------------------------------

@dataclass
class BodySpec:
    slot: int
    name: str
    mass: float
    radius: float = 0.0
    mode: str = "CARTESIAN"            # 'CARTESIAN' | 'ORBIT'
    pos: tuple = (0.0, 0.0, 0.0)
    vel: tuple = (0.0, 0.0, 0.0)
    primary: Optional[int] = None      # slot of primary (ORBIT mode only)
    ref: Optional[int] = None          # slot used for orbit drawing / elements
    a: float = 1.0
    e: float = 0.0
    inc: float = 0.0
    Omega: float = 0.0
    omega: float = 0.0
    anomaly: float = 0.0
    anomaly_type: str = "f"            # 'f' true, 'M' mean anomaly, 'l' mean longitude
    order: int = 0


@dataclass
class SwarmSpec:
    name: str
    count: int
    primary: Optional[int]             # None -> centre of mass of all previous particles
    a_min: float = 0.8
    a_max: float = 1.2
    e_max: float = 0.05
    inc_max: float = 0.02
    lon_center: float = 0.0
    lon_width: float = TWO_PI
    seed: int = 0
    first: int = 0                     # filled in by build_simulation (index in swarm block)


@dataclass
class SimSpec:
    bodies: List[BodySpec]
    swarms: List[SwarmSpec] = field(default_factory=list)
    units: str = "CODE"
    G: float = 1.0
    integrator: str = "ias15"
    dt_mode: str = "AUTO"              # 'AUTO' | 'MANUAL'
    dt: float = 0.01
    dt_fraction: float = 0.02
    time_per_frame: float = 0.05
    n_frames: int = 250
    exact: str = "AUTO"                # 'AUTO' | 'EXACT' | 'STEP'
    move_to_com: bool = True
    collision: str = "none"
    collision_resolve: str = "merge"
    archive_path: Optional[str] = None
    # Parallel baking: simulate only swarm particles with global swarm index in
    # [lo, hi) (None = all), plus the ``probes`` (global swarm indices that are
    # simulated by every worker and recorded in float64 for the split check).
    swarm_range: Optional[tuple] = None
    probes: tuple = ()


def spec_to_dict(spec: SimSpec) -> dict:
    """JSON-serialisable form of a spec (floats round-trip exactly)."""
    return asdict(spec)


def spec_from_dict(d: dict) -> SimSpec:
    d = dict(d)
    d["bodies"] = [BodySpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in b.items()})
                   for b in d["bodies"]]
    d["swarms"] = [SwarmSpec(**s) for s in d.get("swarms", [])]
    if d.get("swarm_range") is not None:
        d["swarm_range"] = tuple(int(x) for x in d["swarm_range"])
    d["probes"] = tuple(int(x) for x in d.get("probes", ()))
    return SimSpec(**d)


def assign_swarm_offsets(spec: SimSpec) -> int:
    """Set ``first`` of every swarm (as build_simulation does); returns the total count."""
    first = 0
    for sw in spec.swarms:
        sw.first = first
        first += max(0, int(sw.count))
    return first


# --------------------------------------------------------------------------
# Setup helpers
# --------------------------------------------------------------------------

def build_order(bodies: List[BodySpec]) -> List[int]:
    """Order in which bodies are added to REBOUND.

    Massive bodies first (REBOUND needs test particles at the end), and every
    body after its primary. Within those constraints, sort by the user's
    ``order`` and then hierarchically by distance from the heaviest body (the
    natural order for Jacobi coordinates, which WHFast uses by default).
    """
    by_slot = {b.slot: b for b in bodies}
    massive = [b for b in bodies if b.mass > 0.0]
    massless = [b for b in bodies if b.mass <= 0.0]
    placed: set = set()
    result: List[int] = []
    heaviest = max(bodies, key=lambda b: b.mass) if bodies else None

    def distance(b):
        if b is heaviest:
            return -1.0
        if b.mode == "ORBIT":
            return abs(b.a)
        if heaviest is not None and heaviest.mode == "CARTESIAN":
            return math.dist(b.pos, heaviest.pos)
        return math.hypot(*b.pos)

    def place(group, label):
        remaining = sorted(group, key=lambda b: (b.order, distance(b), b.name))
        while remaining:
            for b in remaining:
                if b.primary is None or b.primary in placed:
                    result.append(b.slot)
                    placed.add(b.slot)
                    remaining.remove(b)
                    break
            else:
                b = remaining[0]
                p = by_slot.get(b.primary)
                if p is not None and p.mass <= 0.0 and b.mass > 0.0:
                    raise SetupError(
                        f"'{b.name}' has mass but its primary '{p.name}' is massless. "
                        "Massive bodies cannot orbit test particles.")
                names = ", ".join(x.name for x in remaining)
                raise SetupError(f"Circular primary references among: {names}")

    place(massive, "massive")
    place(massless, "massless")
    return result


def orbit_refs(spec: SimSpec, order: List[int]) -> dict:
    """For each body slot, the list of body slots whose centre of mass is the
    reference for its osculating orbit (empty list = no orbit)."""
    masses = {b.slot: b.mass for b in spec.bodies}
    pos_in_order = {s: i for i, s in enumerate(order)}
    refs = {}
    for b in spec.bodies:
        if b.ref is not None and b.ref != b.slot and b.ref in masses:
            refs[b.slot] = [b.ref]
        elif b.mode == "ORBIT" and b.primary is None:
            # REBOUND's default: centre of mass of all previously added particles
            earlier = order[:pos_in_order[b.slot]]
            refs[b.slot] = [s for s in earlier if masses[s] > 0.0]
        else:
            heavier = [s for s, m in masses.items() if s != b.slot and m > b.mass]
            refs[b.slot] = [max(heavier, key=lambda s: (masses[s], -s))] if heavier else []
    return refs


def ident_mode(rebound):
    """REBOUND >= 5 identifies particles by ``name`` (str), older versions by ``hash``."""
    try:
        import inspect
        params = inspect.signature(rebound.Particle.__init__).parameters
        return "name" if "name" in params else "hash"
    except (TypeError, ValueError):
        return "name"


def _ident(mode, slot):
    return str(slot + 1) if mode == "name" else slot + 1


def particle_slots(sim, mode):
    """Slot of every particle in index order (read from names/hashes)."""
    out = np.empty(sim.N, dtype=np.int64)
    for i, p in enumerate(sim.particles):
        try:
            out[i] = (int(p.name) if mode == "name" else int(p.hash.value)) - 1
        except (TypeError, ValueError):
            out[i] = -1
    return out


def _set_units(sim, spec: SimSpec):
    units = UNIT_SYSTEMS.get(spec.units)
    if units:
        sim.units = units
    else:
        sim.G = float(spec.G)


def build_simulation(rebound, spec: SimSpec):
    """Create a REBOUND simulation from a spec.

    Returns (sim, order, index_slot): ``order`` lists body slots in the order
    they were added and ``index_slot`` maps particle index -> slot. Every
    particle is tagged with the identifier ``slot + 1`` (a ``name`` in
    REBOUND >= 5, a ``hash`` before): bodies use slots 0..B-1, swarm particles
    B..B+S-1.
    """
    if not spec.bodies:
        raise SetupError("There are no REBOUND bodies in the scene.")

    sim = rebound.Simulation()
    _set_units(sim, spec)
    try:
        sim.integrator = spec.integrator
    except Exception as ex:  # unknown integrator in this REBOUND version
        raise SetupError(
            f"Integrator '{spec.integrator}' is not available in this REBOUND version ({ex}).")

    idm = ident_mode(rebound)
    order = build_order(spec.bodies)
    by_slot = {b.slot: b for b in spec.bodies}
    index_of = {}
    index_slot = []
    for slot in order:
        b = by_slot[slot]
        kw = dict(m=float(b.mass), r=float(b.radius))
        kw[idm] = _ident(idm, b.slot)
        if b.mode == "CARTESIAN":
            kw.update(x=float(b.pos[0]), y=float(b.pos[1]), z=float(b.pos[2]),
                      vx=float(b.vel[0]), vy=float(b.vel[1]), vz=float(b.vel[2]))
        else:
            if sim.N == 0:
                raise SetupError(
                    f"'{b.name}' uses orbital elements but nothing has been added before it. "
                    "Give the first body Cartesian coordinates (or set a lower 'Order' on its primary).")
            kw.update(a=float(b.a), e=float(b.e), inc=float(b.inc),
                      Omega=float(b.Omega), omega=float(b.omega))
            kw[b.anomaly_type] = float(b.anomaly)
            if b.primary is not None:
                # fetch fresh: particle pointers are invalidated when REBOUND reallocates
                kw["primary"] = sim.particles[index_of[b.primary]]
        try:
            sim.add(**kw)
        except Exception as ex:
            raise SetupError(f"Could not add '{b.name}' to REBOUND: {ex}")
        index_of[slot] = sim.N - 1
        index_slot.append(slot)

    n_massive = sum(1 for b in spec.bodies if b.mass > 0.0)
    if n_massive == 0:
        raise SetupError("At least one body needs a mass > 0.")

    # Swarms of test particles
    B = len(spec.bodies)
    lo, hi = spec.swarm_range if spec.swarm_range is not None else (0, None)
    probe_set = set(int(p) for p in spec.probes)
    first = 0
    for sw in spec.swarms:
        sw.first = first
        n = int(sw.count)
        if n <= 0:
            continue
        if sw.primary is not None and sw.primary not in index_of:
            raise SetupError(f"Swarm '{sw.name}': primary is not a REBOUND body.")
        # Draw the whole swarm (same RNG stream regardless of the slice) ...
        rng = np.random.default_rng(int(sw.seed))
        a = rng.uniform(sw.a_min, max(sw.a_max, sw.a_min), n)
        e = rng.uniform(0.0, max(0.0, min(sw.e_max, 0.99)), n)
        inc = rng.uniform(0.0, max(0.0, sw.inc_max), n)
        Om = rng.uniform(0.0, TWO_PI, n)
        om = rng.uniform(0.0, TWO_PI, n)
        lam = sw.lon_center + (rng.uniform(-0.5, 0.5, n) * sw.lon_width)
        # ... but only add the particles of this slice and the probes.
        for i in range(n):
            g = first + i
            if not ((g >= lo and (hi is None or g < hi)) or g in probe_set):
                continue
            kw = dict(m=0.0, a=float(a[i]), e=float(e[i]), inc=float(inc[i]),
                      Omega=float(Om[i]), omega=float(om[i]), l=float(lam[i]))
            kw[idm] = _ident(idm, B + g)
            if sw.primary is not None:
                kw["primary"] = sim.particles[index_of[sw.primary]]
            try:
                sim.add(**kw)
            except Exception as ex:
                raise SetupError(f"Swarm '{sw.name}': could not add particle {i}: {ex}")
            index_slot.append(B + g)
        first += n

    if sim.N > n_massive:
        sim.N_active = n_massive
        sim.testparticle_type = 0

    if spec.collision and spec.collision != "none":
        sim.collision = spec.collision
        sim.collision_resolve = spec.collision_resolve
        if spec.collision_resolve == "merge":
            try:
                sim.track_energy_offset = 1
            except Exception:
                pass

    return sim, order, np.array(index_slot, dtype=np.int64)


def read_particles(sim):
    """Return (xyz (N,3), vxyz (N,3), m (N,), r (N,)) in particle index order."""
    N = sim.N
    x = np.empty((N, 3), dtype=np.float64)
    v = np.empty((N, 3), dtype=np.float64)
    m = np.empty(N, dtype=np.float64)
    r = np.empty(N, dtype=np.float64)
    try:
        sim.serialize_particle_data(xyz=x, vxvyvz=v, m=m, r=r)
    except Exception:
        for i, p in enumerate(sim.particles):
            x[i] = (p.x, p.y, p.z)
            v[i] = (p.vx, p.vy, p.vz)
            m[i] = p.m
            r[i] = p.r
    return x, v, m, r


def min_period(spec: SimSpec, order, G, bpos, bvel, bmass):
    """Shortest bound two-body period at t=0 (bodies + swarm inner edges)."""
    refs = orbit_refs(spec, order)
    periods = []
    for b in spec.bodies:
        slots = refs.get(b.slot)
        if not slots:
            continue
        mr = bmass[slots]
        M = mr.sum()
        if M <= 0:
            continue
        rp = (bpos[slots] * mr[:, None]).sum(0) / M
        rv = (bvel[slots] * mr[:, None]).sum(0) / M
        a, e, _inc, _ev, _h = two_body_elements(bpos[b.slot] - rp, bvel[b.slot] - rv,
                                                G * (M + bmass[b.slot]))
        if np.isfinite(a) and a > 0 and e < 1:
            periods.append(TWO_PI * math.sqrt(a ** 3 / (G * (M + bmass[b.slot]))))
    total_massive = bmass[bmass > 0].sum()
    for sw in spec.swarms:
        if sw.count <= 0 or sw.a_min <= 0:
            continue
        M = bmass[sw.primary] if sw.primary is not None else total_massive
        if M > 0:
            periods.append(TWO_PI * math.sqrt(sw.a_min ** 3 / (G * M)))
    return min(periods) if periods else None


def characteristic_period(spec: SimSpec, body_name: Optional[str] = None,
                          G_override: Optional[float] = None) -> Optional[float]:
    """Pure-Python orbital period estimator for ``spec`` (no REBOUND C call required).

    If ``body_name`` is given and bound, returns that body's initial orbital period.
    Otherwise returns the shortest bound orbital period in ``spec`` (or ``None``).
    """
    if not spec.bodies:
        return None
    G = float(G_override if G_override is not None else (
        DEFAULT_G_BY_UNITS.get(spec.units, spec.G) if spec.units != "CODE" else spec.G
    ))
    if G <= 0:
        return None
    try:
        order = build_order(spec.bodies)
    except SetupError:
        order = [b.slot for b in spec.bodies]
    by_slot = {b.slot: b for b in spec.bodies}
    B = max(by_slot.keys()) + 1
    bpos = np.zeros((B, 3), dtype=np.float64)
    bvel = np.zeros((B, 3), dtype=np.float64)
    bmass = np.zeros(B, dtype=np.float64)
    placed_slots: List[int] = []
    period_by_name = {}

    for slot in order:
        b = by_slot[slot]
        bmass[slot] = max(0.0, float(b.mass))
        if b.mode == "CARTESIAN" or not placed_slots:
            bpos[slot] = [float(x) for x in b.pos]
            bvel[slot] = [float(x) for x in b.vel]
        else:
            ref_slots = [b.primary] if (b.primary is not None and b.primary in by_slot) else [
                s for s in placed_slots if bmass[s] > 0
            ]
            M = float(bmass[ref_slots].sum()) if ref_slots else 0.0
            if M > 0:
                rp = (bpos[ref_slots] * bmass[ref_slots, None]).sum(0) / M
                rv = (bvel[ref_slots] * bmass[ref_slots, None]).sum(0) / M
                mu = G * (M + bmass[slot])
                if b.a > 0 and 0 <= b.e < 1:
                    period_by_name[b.name] = TWO_PI * math.sqrt(b.a ** 3 / mu)
                    if b.anomaly_type == "l":
                        lam = float(b.anomaly)
                    elif b.anomaly_type == "M":
                        lam = float(b.Omega + b.omega + b.anomaly)
                    else:
                        lam = float(b.Omega + b.omega + b.anomaly)
                    rel_p, rel_v = elements_to_cartesian(
                        mu, max(b.a, 1e-12), min(max(b.e, 0.0), 0.999),
                        b.inc, b.Omega, b.omega, lam
                    )
                    bpos[slot] = rp + rel_p
                    bvel[slot] = rv + rel_v
        placed_slots.append(slot)

    refs = orbit_refs(spec, order)
    for b in spec.bodies:
        if b.name in period_by_name:
            continue
        slots = refs.get(b.slot)
        if not slots:
            continue
        mr = bmass[slots]
        M = float(mr.sum())
        if M <= 0:
            continue
        rp = (bpos[slots] * mr[:, None]).sum(0) / M
        rv = (bvel[slots] * mr[:, None]).sum(0) / M
        mu = G * (M + bmass[b.slot])
        a, e, *_ = two_body_elements(bpos[b.slot] - rp, bvel[b.slot] - rv, mu)
        if np.isfinite(a) and a > 0 and e < 1:
            period_by_name[b.name] = TWO_PI * math.sqrt(float(a) ** 3 / mu)

    if body_name and body_name in period_by_name:
        return period_by_name[body_name]
    return min_period(spec, order, G, bpos, bvel, bmass) or (
        min(period_by_name.values()) if period_by_name else None
    )


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

class Runner:
    """Integrates a simulation and samples it once per animation frame.

    Arrays (F = frames, B = bodies, S = swarm particles of this runner):
        bpos, bvel : (F, B, 3) float64, NaN once a body is gone (merged)
        bmass, brad: (F, B)
        spos       : (F, S, 3) float32 or None (global swarm index s_lo + i)
        ppos       : (F, P, 3) float64 or None (probe particles, see SimSpec)
        energy     : (F,) NaN where not computed
    """

    def __init__(self, rebound, spec: SimSpec, compute_energy=True):
        self.rebound = rebound
        self.spec = spec
        self.sim, self.order, self.index_slot = build_simulation(rebound, spec)
        self.ident = ident_mode(rebound)
        sim = self.sim
        self.G = float(sim.G)
        self.B = B = len(spec.bodies)
        self.S_total = S_total = sum(max(0, int(sw.count)) for sw in spec.swarms)
        lo, hi = spec.swarm_range if spec.swarm_range is not None else (0, S_total)
        hi = min(int(hi), S_total)
        lo = max(0, min(int(lo), hi))
        self.s_lo = lo
        self.S = S = hi - lo
        self.probes = np.array(sorted({int(p) for p in spec.probes if 0 <= int(p) < S_total}),
                               dtype=np.int64)
        self.F = F = max(1, int(spec.n_frames))
        self.times = np.arange(F, dtype=np.float64) * float(spec.time_per_frame)
        self.refs = orbit_refs(spec, self.order)

        self.bpos = np.full((F, B, 3), np.nan)
        self.bvel = np.full((F, B, 3), np.nan)
        self.bmass = np.full((F, B), np.nan)
        self.brad = np.full((F, B), np.nan)
        self.spos = np.full((F, S, 3), np.nan, dtype=np.float32) if S else None
        self.ppos = np.full((F, len(self.probes), 3), np.nan) if len(self.probes) else None
        self.energy = np.full(F, np.nan)

        # Raw initial state (before moving to the centre-of-mass frame)
        x, v, m, r = read_particles(sim)
        slots = self.index_slot
        isb = slots < B
        self.raw_bpos = np.full((B, 3), np.nan)
        self.raw_bvel = np.full((B, 3), np.nan)
        self.raw_bmass = np.zeros(B)
        self.raw_bpos[slots[isb]] = x[isb]
        self.raw_bvel[slots[isb]] = v[isb]
        self.raw_bmass[slots[isb]] = m[isb]
        self.raw_spos = None
        if S:
            self.raw_spos = np.full((S, 3), np.nan)
            g = slots[~isb] - B
            inr = (g >= lo) & (g < hi)
            self.raw_spos[g[inr] - lo] = x[~isb][inr]

        # Timestep
        dt = float(spec.dt)
        self.auto_period = None
        if spec.dt_mode == "AUTO":
            self.auto_period = min_period(spec, self.order, self.G,
                                          self.raw_bpos, self.raw_bvel, self.raw_bmass)
            if self.auto_period:
                dt = float(spec.dt_fraction) * self.auto_period
        if dt <= 0:
            raise SetupError("The timestep must be positive.")
        sim.dt = dt
        self.dt = dt

        com = sim.com()
        self.com0 = np.array([com.x, com.y, com.z])
        if spec.move_to_com:
            sim.move_to_com()
            self.offset0 = self.com0.copy()
        else:
            self.offset0 = np.zeros(3)

        if spec.exact == "EXACT":
            self.exact = 1
        elif spec.exact == "STEP":
            self.exact = 0
        else:
            self.exact = 1 if spec.integrator in ADAPTIVE_INTEGRATORS else 0

        self.archive_error = None
        if spec.archive_path:
            try:
                if hasattr(sim, "save_to_file"):
                    sim.save_to_file(spec.archive_path, interval=float(spec.time_per_frame),
                                     delete_file=True)
                else:  # REBOUND 3.x
                    sim.automateSimulationArchive(spec.archive_path,
                                                  interval=float(spec.time_per_frame),
                                                  deletefile=True)
            except Exception as ex:
                self.archive_error = str(ex)

        self.k = 0
        self.halted = False
        self.cancelled = False
        self.messages: List[str] = []
        self.wall = 0.0
        self.n_last = sim.N
        self.compute_energy = compute_energy
        self.energy_every_frame = compute_energy and sim.N <= ENERGY_EVERY_FRAME_MAX_N
        names = ("Collision", "Escape", "Encounter")
        self._stop_exc = tuple(getattr(rebound, n) for n in names if hasattr(rebound, n))

    # ------------------------------------------------------------------
    @property
    def done(self):
        return self.k >= self.F

    @property
    def progress(self):
        return self.k / self.F

    def _store(self, k):
        x, v, m, r = read_particles(self.sim)
        if self.sim.N != len(self.index_slot) or self.sim.N <= 64:
            # particles were removed (or reordered): re-read their identifiers
            self.index_slot = particle_slots(self.sim, self.ident)
        slots = self.index_slot
        ok = slots >= 0
        if not ok.all():
            slots, x, v, m, r = slots[ok], x[ok], v[ok], m[ok], r[ok]
        isb = slots < self.B
        bs = slots[isb]
        self.bpos[k, bs] = x[isb]
        self.bvel[k, bs] = v[isb]
        self.bmass[k, bs] = m[isb]
        self.brad[k, bs] = r[isb]
        if (~isb).any() and (self.S or self.ppos is not None):
            g = slots[~isb] - self.B
            xs = x[~isb]
            if self.S:
                inr = (g >= self.s_lo) & (g < self.s_lo + self.S)
                self.spos[k, g[inr] - self.s_lo] = xs[inr]
            if self.ppos is not None:
                j = np.clip(np.searchsorted(self.probes, g), 0, len(self.probes) - 1)
                hit = self.probes[j] == g
                self.ppos[k, j[hit]] = xs[hit]

    def _copy_previous(self, k):
        self.bpos[k] = self.bpos[k - 1]
        self.bvel[k] = self.bvel[k - 1]
        self.bmass[k] = self.bmass[k - 1]
        self.brad[k] = self.brad[k - 1]
        if self.S:
            self.spos[k] = self.spos[k - 1]
        if self.ppos is not None:
            self.ppos[k] = self.ppos[k - 1]
        self.energy[k] = self.energy[k - 1]

    def stop(self):
        """Ask a running ``step``/``run`` (e.g. in another thread) to stop soon."""
        self.cancelled = True
        try:
            self.sim.stop()  # interrupts a long sim.integrate() (REBOUND >= 3.x)
        except Exception:
            pass

    def step(self, budget=0.05):
        """Integrate and sample frames for roughly ``budget`` seconds."""
        t0 = time.perf_counter()
        sim = self.sim
        while self.k < self.F and not self.cancelled:
            k = self.k
            if k > 0 and self.halted:
                self._copy_previous(k)
            else:
                if k > 0:
                    try:
                        sim.integrate(float(self.times[k]), exact_finish_time=self.exact)
                    except self._stop_exc as ex:
                        self.halted = True
                        self.messages.append(
                            f"Stopped at t={sim.t:.6g} ({type(ex).__name__}): {ex}")
                    except Exception as ex:
                        if self.cancelled:
                            break
                        self.halted = True
                        self.messages.append(f"Integration stopped at t={sim.t:.6g}: {ex}")
                    if self.cancelled:
                        break
                self._store(k)
                if sim.N != self.n_last:
                    self.messages.append(
                        f"t={sim.t:.6g}: particle count {self.n_last} -> {sim.N} (collision)")
                    self.n_last = sim.N
                if self.compute_energy and (self.energy_every_frame or k == 0 or k == self.F - 1
                                            or self.halted):
                    try:
                        self.energy[k] = sim.energy()
                    except Exception:
                        self.compute_energy = False
            self.k += 1
            if time.perf_counter() - t0 > budget:
                break
        self.wall += time.perf_counter() - t0
        return self.progress

    def run(self):
        while not self.done and not self.cancelled:
            self.step(1.0)
        return self

    def relative_energy_error(self):
        return relative_energy_error(self.energy)


def relative_energy_error(energy):
    E0 = energy[0]
    if not np.isfinite(E0) or E0 == 0:
        return np.full(len(energy), np.nan)
    return np.abs((energy - E0) / E0)


class MergedRun:
    """Result of a bake split over several processes.

    Duck-types the parts of a finished ``Runner`` that the writers and the CSV
    export use (spec, F, B, S, G, times, refs, b*/spos arrays, energy, dt,
    offset0, messages, relative_energy_error()). ``split`` holds the
    SplitCheck of the run.
    """

    def __init__(self, spec: SimSpec, G, dt, offset0, bpos, bvel, bmass, brad, spos, energy,
                 messages, halted=False, wall=0.0, split=None, n_chunks=1):
        self.spec = spec
        self.order = build_order(spec.bodies)
        self.refs = orbit_refs(spec, self.order)
        self.G = float(G)
        self.dt = float(dt)
        self.offset0 = np.asarray(offset0, float)
        self.bpos, self.bvel, self.bmass, self.brad = bpos, bvel, bmass, brad
        self.spos = spos
        self.energy = energy
        self.F = F = bpos.shape[0]
        self.B = bpos.shape[1]
        self.S = 0 if spos is None else spos.shape[1]
        self.times = np.arange(F, dtype=np.float64) * float(spec.time_per_frame)
        self.messages = list(messages)
        self.halted = halted
        self.wall = wall
        self.split = split
        self.n_chunks = n_chunks
        self.k = F

    done = property(lambda self: True)
    progress = property(lambda self: 1.0)

    def relative_energy_error(self):
        return relative_energy_error(self.energy)


@dataclass
class SplitCheck:
    """Scientific consistency check across parallel swarm chunks.

    Each worker integrates all massive bodies plus its own slice of test
    particles and a shared set of float64 ``probe`` test particles. For
    fixed-step integrators (WHFast, Leapfrog, SABA, JANUS) trajectories across
    chunks are identical (~0 drift). For adaptive integrators (IAS15, BS),
    different test-particle subsets can induce slightly different adaptive step
    sequences; ``max_rel_body``, ``max_rel_probe`` and ``body_drift`` (F,)
    quantify this divergence in double precision.
    """
    n_chunks: int
    n_probes: int
    max_abs_body: float
    max_rel_body: float
    max_abs_probe: float
    max_rel_probe: float
    body_drift: np.ndarray


def split_specs(spec: SimSpec, n_chunks: int, n_probes: int = 8) -> List[SimSpec]:
    """Split ``spec`` into ``n_chunks`` worker specs over the swarm particles."""
    S_total = assign_swarm_offsets(spec)
    n_chunks = max(1, min(int(n_chunks), S_total))
    if n_chunks <= 1 or S_total == 0:
        return [spec]
    n_probes = max(0, min(int(n_probes), S_total))
    if n_probes > 0:
        probes = tuple(int(x) for x in np.linspace(0, S_total - 1, n_probes, dtype=np.int64))
    else:
        probes = ()
    edges = np.linspace(0, S_total, n_chunks + 1, dtype=np.int64)
    out: List[SimSpec] = []
    base = spec_to_dict(spec)
    for c in range(n_chunks):
        d = dict(base)
        d["swarm_range"] = (int(edges[c]), int(edges[c + 1]))
        d["probes"] = probes
        if c > 0:
            d["archive_path"] = None  # only chunk 0 writes SimulationArchive
        out.append(spec_from_dict(d))
    return out


def _trajectory_drift(ref: np.ndarray, others: List[np.ndarray]):
    """Return (max_abs, max_rel, per_frame_rel (F,)) comparing ``others`` to ``ref``."""
    F = ref.shape[0]
    per_frame = np.zeros(F, dtype=np.float64)
    if not others or ref.size == 0:
        return 0.0, 0.0, per_frame
    with np.errstate(invalid="ignore", divide="ignore"):
        r_norm = np.linalg.norm(ref, axis=-1)
        finite_r = r_norm[np.isfinite(r_norm)]
        scale_global = float(np.nanmax(finite_r)) if finite_r.size else 1.0
        if scale_global <= 0.0:
            scale_global = 1.0
        denom = np.where(r_norm > 1e-12 * scale_global, r_norm, scale_global)
        max_abs = 0.0
        max_rel = 0.0
        for arr in others:
            diff = np.linalg.norm(arr - ref, axis=-1)
            rel = diff / denom
            if np.isfinite(diff).any():
                a = float(np.nanmax(diff))
                if a > max_abs:
                    max_abs = a
            if np.isfinite(rel).any():
                r = float(np.nanmax(rel))
                if r > max_rel:
                    max_rel = r
                pf = np.nanmax(np.where(np.isfinite(rel), rel, 0.0), axis=-1)
                per_frame = np.maximum(per_frame, pf)
    return max_abs, max_rel, per_frame


def check_split_consistency(chunks: List[dict]) -> SplitCheck:
    """Compute ``SplitCheck`` from a list of worker result dicts."""
    n = len(chunks)
    F = chunks[0]["bpos"].shape[0]
    if n <= 1:
        return SplitCheck(1, 0, 0.0, 0.0, 0.0, 0.0, np.zeros(F, dtype=np.float64))
    ref_b = chunks[0]["bpos"]
    other_b = [c["bpos"] for c in chunks[1:]]
    max_abs_b, max_rel_b, body_drift = _trajectory_drift(ref_b, other_b)

    ref_p = chunks[0].get("ppos")
    other_p = [c["ppos"] for c in chunks[1:] if c.get("ppos") is not None]
    if ref_p is not None and ref_p.size and len(other_p) == n - 1:
        max_abs_p, max_rel_p, probe_drift = _trajectory_drift(ref_p, other_p)
        n_probes = int(ref_p.shape[1])
        body_drift = np.maximum(body_drift, probe_drift)
    else:
        max_abs_p, max_rel_p, n_probes = 0.0, 0.0, 0

    return SplitCheck(
        n_chunks=n,
        n_probes=n_probes,
        max_abs_body=max_abs_b,
        max_rel_body=max_rel_b,
        max_abs_probe=max_abs_p,
        max_rel_probe=max_rel_p,
        body_drift=body_drift,
    )


def merge_chunk_results(spec: SimSpec, chunks: List[dict], wall: float = 0.0) -> MergedRun:
    """Assemble ``chunks`` (ordered or unordered) into a ``MergedRun``."""
    chunks = sorted(chunks, key=lambda c: int(c["s_lo"]))
    S_total = assign_swarm_offsets(spec)
    c0 = chunks[0]
    F = c0["bpos"].shape[0]
    if S_total > 0:
        spos = np.full((F, S_total, 3), np.nan, dtype=np.float32)
        for c in chunks:
            lo = int(c["s_lo"])
            sp = c.get("spos")
            if sp is not None and sp.size:
                spos[:, lo:lo + sp.shape[1]] = sp
    else:
        spos = None
    split = check_split_consistency(chunks)
    msgs: List[str] = []
    seen = set()
    for c in chunks:
        for m in c.get("messages", ()):
            if m not in seen:
                seen.add(m)
                msgs.append(m)
    if split.n_chunks > 1 and max(split.max_rel_body, split.max_rel_probe) > 1e-9:
        msgs.append(
            f"Parallel split drift ({split.n_chunks} workers, {spec.integrator}): "
            f"max Δr/r = {max(split.max_rel_body, split.max_rel_probe):.2e}. "
            "Disable 'Parallel Swarm Bake' or use a fixed-step integrator for exact consistency."
        )
    halted = any(bool(c.get("halted", False)) for c in chunks)
    return MergedRun(
        spec=spec,
        G=float(c0["G"]),
        dt=float(c0["dt"]),
        offset0=c0["offset0"],
        bpos=c0["bpos"],
        bvel=c0["bvel"],
        bmass=c0["bmass"],
        brad=c0["brad"],
        spos=spos,
        energy=c0["energy"],
        messages=msgs,
        halted=halted,
        wall=wall,
        split=split,
        n_chunks=len(chunks),
    )


MIN_PARTICLES_PER_WORKER = 200


def default_worker_count() -> int:
    cpus = os.cpu_count() or 4
    return max(1, min(16, cpus - 2 if cpus > 4 else cpus))


def resolve_worker_count(spec: SimSpec, requested: int = 0, parallel: bool = True,
                         min_per_worker: int = MIN_PARTICLES_PER_WORKER) -> int:
    """Choose the effective number of parallel worker processes for ``spec``."""
    if not parallel:
        return 1
    S_total = sum(max(0, int(sw.count)) for sw in spec.swarms)
    if S_total < 2 * max(1, min_per_worker):
        return 1
    target = int(requested) if int(requested) > 0 else default_worker_count()
    max_by_size = max(1, S_total // max(1, min_per_worker))
    return max(1, min(target, max_by_size))


class ParallelPool:
    """Manages K worker subprocesses running ``worker.py`` (pure Python, no ``bpy``)."""

    def __init__(self, spec: SimSpec, n_workers: int, n_probes: int = 8,
                 compute_energy: bool = True, deps_dir: Optional[str] = None):
        self.spec = spec
        self.specs = split_specs(spec, n_workers, n_probes=n_probes)
        self.n_workers = len(self.specs)
        self.tmpdir = tempfile.mkdtemp(prefix="rebound_bake_")
        self.procs: List[subprocess.Popen] = []
        self.out_paths: List[str] = []
        self.prog_paths: List[str] = []
        self.err_files = []
        self._merged: Optional[MergedRun] = None
        self.archive_error: Optional[str] = None

        worker_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            for idx, sp in enumerate(self.specs):
                spec_path = os.path.join(self.tmpdir, f"spec_{idx}.json")
                out_path = os.path.join(self.tmpdir, f"out_{idx}.npz")
                prog_path = os.path.join(self.tmpdir, f"prog_{idx}.txt")
                err_path = os.path.join(self.tmpdir, f"err_{idx}.txt")
                with open(spec_path, "w", encoding="utf-8") as fh:
                    json.dump(spec_to_dict(sp), fh)
                cmd = [sys.executable, worker_py, spec_path, out_path, prog_path]
                if deps_dir:
                    cmd.append(deps_dir)
                if not compute_energy:
                    cmd.append("--no-energy")
                err_fh = open(err_path, "w", encoding="utf-8")
                proc = subprocess.Popen(
                    cmd, stdout=err_fh, stderr=subprocess.STDOUT,
                    creationflags=creationflags)
                self.procs.append(proc)
                self.out_paths.append(out_path)
                self.prog_paths.append(prog_path)
                self.err_files.append((err_fh, err_path))
        except Exception:
            self.kill()
            raise

    def poll_progress(self) -> float:
        total = 0.0
        for proc, ppath, (err_fh, err_path) in zip(self.procs, self.prog_paths, self.err_files):
            rc = proc.poll()
            if rc is not None and rc != 0:
                try:
                    err_fh.flush()
                    with open(err_path, "r", encoding="utf-8", errors="replace") as fh:
                        msg = fh.read().strip()
                except OSError:
                    msg = f"worker exited with code {rc}"
                tail = "\n".join(msg.splitlines()[-6:]) if msg else f"exit code {rc}"
                raise RuntimeError(f"Parallel worker failed:\n{tail}")
            if rc == 0:
                total += 1.0
                continue
            try:
                with open(ppath, "r", encoding="utf-8") as fh:
                    total += float(fh.read().strip() or 0.0)
            except (OSError, ValueError):
                pass
        return total / max(1, len(self.procs))

    @property
    def done(self) -> bool:
        return all(p.poll() == 0 for p in self.procs)

    def wait_and_collect(self, wall: float) -> MergedRun:
        if self._merged is not None:
            return self._merged
        for p in self.procs:
            p.wait()
        self.poll_progress()  # raises if any worker failed
        chunks = []
        for idx, opath in enumerate(self.out_paths):
            with np.load(opath, allow_pickle=True) as z:
                if "messages_json" in z:
                    msgs = json.loads(str(z["messages_json"]))
                else:
                    msgs = [str(m) for m in z["messages"].tolist()]
                c = {
                    "s_lo": int(z["s_lo"]),
                    "G": float(z["G"]),
                    "dt": float(z["dt"]),
                    "offset0": np.array(z["offset0"], dtype=np.float64),
                    "bpos": np.array(z["bpos"], dtype=np.float64),
                    "bvel": np.array(z["bvel"], dtype=np.float64),
                    "bmass": np.array(z["bmass"], dtype=np.float64),
                    "brad": np.array(z["brad"], dtype=np.float64),
                    "energy": np.array(z["energy"], dtype=np.float64),
                    "halted": bool(z["halted"]),
                    "wall": float(z["wall"]),
                    "messages": msgs,
                    "spos": np.array(z["spos"], dtype=np.float32) if "spos" in z else None,
                    "ppos": np.array(z["ppos"], dtype=np.float64) if "ppos" in z else None,
                }
                if idx == 0 and "archive_error" in z:
                    ae = str(z["archive_error"])
                    if ae:
                        self.archive_error = ae
                chunks.append(c)
        self._merged = merge_chunk_results(self.spec, chunks, wall=wall)
        self.cleanup()
        return self._merged

    def kill(self):
        for p in self.procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except OSError:
                    pass
        for p in self.procs:
            try:
                p.wait(timeout=1.0)
            except Exception:
                try:
                    p.kill()
                except OSError:
                    pass
        self.cleanup()

    def cleanup(self):
        for err_fh, _ in self.err_files:
            try:
                err_fh.close()
            except OSError:
                pass
        if self.tmpdir and os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            self.tmpdir = ""


def run_simulation(rebound, spec: SimSpec, n_workers: int = 0, parallel: bool = True,
                   n_probes: int = 8, compute_energy: bool = True,
                   progress_cb: Optional[Callable[[float], None]] = None,
                   deps_dir: Optional[str] = None,
                   min_per_worker: int = MIN_PARTICLES_PER_WORKER) -> Runner | MergedRun:
    """Run ``spec`` either in-process or across ``n_workers`` subprocesses.

    Can be called from any standalone Python script or HPC batch job without Blender.
    """
    eff_workers = resolve_worker_count(spec, requested=n_workers, parallel=parallel,
                                       min_per_worker=min_per_worker)
    t0 = time.perf_counter()
    if eff_workers > 1:
        build_simulation(rebound, split_specs(spec, eff_workers, n_probes=0)[0])
        pool = ParallelPool(spec, eff_workers, n_probes=n_probes,
                            compute_energy=compute_energy, deps_dir=deps_dir)
        try:
            while not pool.done:
                p = pool.poll_progress()
                if progress_cb is not None:
                    progress_cb(p)
                time.sleep(0.1)
            run = pool.wait_and_collect(time.perf_counter() - t0)
            if progress_cb is not None:
                progress_cb(1.0)
            return run
        except BaseException:
            pool.kill()
            raise
    else:
        runner = Runner(rebound, spec, compute_energy=compute_energy)
        while not runner.done:
            p = runner.step(0.25)
            if progress_cb is not None:
                progress_cb(p)
        return runner


# --------------------------------------------------------------------------
# Bake file persistence (.npz) & REBOUND SimulationArchive import (.bin)
# --------------------------------------------------------------------------

def save_run_npz(path: str, run: Runner | MergedRun, meta: Optional[dict] = None,
                 compressed: bool = False) -> str:
    """Save a finished ``Runner`` or ``MergedRun`` (with its ``SimSpec`` and optional
    Blender/scene metadata) to a portable ``.npz`` file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    assign_swarm_offsets(run.spec)
    payload = dict(
        format_version=np.int64(1),
        spec_json=np.array(json.dumps(spec_to_dict(run.spec))),
        meta_json=np.array(json.dumps(meta or {})),
        messages_json=np.array(json.dumps(list(run.messages))),
        G=np.float64(run.G),
        dt=np.float64(run.dt),
        offset0=np.asarray(run.offset0, dtype=np.float64),
        bpos=np.asarray(run.bpos, dtype=np.float64),
        bvel=np.asarray(run.bvel, dtype=np.float64),
        bmass=np.asarray(run.bmass, dtype=np.float64),
        brad=np.asarray(run.brad, dtype=np.float64),
        energy=np.asarray(run.energy, dtype=np.float64),
        halted=np.bool_(run.halted),
        wall=np.float64(getattr(run, "wall", 0.0)),
        n_chunks=np.int64(getattr(run, "n_chunks", 1)),
    )
    if run.spos is not None:
        payload["spos"] = np.asarray(run.spos, dtype=np.float32)
    split = getattr(run, "split", None)
    if split is not None:
        payload["split_n_chunks"] = np.int64(split.n_chunks)
        payload["split_n_probes"] = np.int64(split.n_probes)
        payload["split_max_abs_body"] = np.float64(split.max_abs_body)
        payload["split_max_rel_body"] = np.float64(split.max_rel_body)
        payload["split_max_abs_probe"] = np.float64(split.max_abs_probe)
        payload["split_max_rel_probe"] = np.float64(split.max_rel_probe)
        payload["split_body_drift"] = np.asarray(split.body_drift, dtype=np.float64)
    saver = np.savez_compressed if compressed else np.savez
    saver(path, **payload)
    return path


def load_run_npz(path: str) -> Tuple[MergedRun, dict]:
    """Load a ``.npz`` bake file written by ``save_run_npz``.

    Returns ``(MergedRun, meta_dict)``.
    """
    with np.load(path, allow_pickle=False) as z:
        if "spec_json" not in z:
            raise SetupError(f"'{os.path.basename(path)}' is not a valid REBOUND bake (.npz missing spec_json).")
        spec = spec_from_dict(json.loads(str(z["spec_json"])))
        assign_swarm_offsets(spec)
        meta = json.loads(str(z["meta_json"])) if "meta_json" in z else {}
        if "messages_json" in z:
            messages = json.loads(str(z["messages_json"]))
        else:
            messages = []
        G = float(z["G"])
        dt = float(z["dt"])
        offset0 = np.array(z["offset0"], dtype=np.float64)
        bpos = np.array(z["bpos"], dtype=np.float64)
        bvel = np.array(z["bvel"], dtype=np.float64)
        bmass = np.array(z["bmass"], dtype=np.float64)
        brad = np.array(z["brad"], dtype=np.float64)
        energy = np.array(z["energy"], dtype=np.float64)
        halted = bool(z["halted"]) if "halted" in z else False
        wall = float(z["wall"]) if "wall" in z else 0.0
        n_chunks = int(z["n_chunks"]) if "n_chunks" in z else 1
        spos = np.array(z["spos"], dtype=np.float32) if "spos" in z else None
        split = None
        if "split_n_chunks" in z:
            split = SplitCheck(
                n_chunks=int(z["split_n_chunks"]),
                n_probes=int(z["split_n_probes"]),
                max_abs_body=float(z["split_max_abs_body"]),
                max_rel_body=float(z["split_max_rel_body"]),
                max_abs_probe=float(z["split_max_abs_probe"]),
                max_rel_probe=float(z["split_max_rel_probe"]),
                body_drift=np.array(z["split_body_drift"], dtype=np.float64),
            )
    run = MergedRun(
        spec=spec, G=G, dt=dt, offset0=offset0,
        bpos=bpos, bvel=bvel, bmass=bmass, brad=brad,
        spos=spos, energy=energy, messages=messages,
        halted=halted, wall=wall, split=split, n_chunks=n_chunks,
    )
    return run, meta


def load_simulation_archive(rebound, path: str, units: str = "CODE",
                            max_frames: int = 0) -> Tuple[MergedRun, dict]:
    """Import a native REBOUND ``SimulationArchive`` (``.bin``) into a ``MergedRun``.

    Allows loading simulations produced by arbitrary external REBOUND scripts on
    university clusters into Blender.
    """
    sa_cls = getattr(rebound, "Simulationarchive", None) or getattr(rebound, "SimulationArchive", None)
    if sa_cls is None:
        raise SetupError("This REBOUND version does not provide Simulationarchive.")
    try:
        sa = sa_cls(path)
    except Exception as ex:
        raise SetupError(f"Could not open REBOUND SimulationArchive '{os.path.basename(path)}': {ex}")
    n_snaps = len(sa)
    if n_snaps == 0:
        raise SetupError(f"SimulationArchive '{os.path.basename(path)}' contains 0 snapshots.")

    if max_frames and max_frames > 1 and n_snaps > max_frames:
        indices = np.linspace(0, n_snaps - 1, int(max_frames), dtype=np.int64)
    else:
        indices = np.arange(n_snaps, dtype=np.int64)

    F = len(indices)
    sim0 = sa[int(indices[0])]
    N0 = sim0.N
    if N0 == 0:
        raise SetupError("SimulationArchive has 0 particles in the first snapshot.")

    x0, v0, m0, r0 = read_particles(sim0)
    n_active = getattr(sim0, "N_active", -1)
    if 0 < n_active < N0:
        B = int(n_active)
    else:
        massive_mask = m0 > 0.0
        n_massive = int(massive_mask.sum())
        # Treat trailing massless particles as a swarm if there are many (>= 16)
        if 0 < n_massive < N0 and (N0 - n_massive) >= 16 and np.all(m0[n_massive:] <= 0.0):
            B = n_massive
        else:
            B = N0
    S = max(0, N0 - B)

    bpos = np.full((F, B, 3), np.nan, dtype=np.float64)
    bvel = np.full((F, B, 3), np.nan, dtype=np.float64)
    bmass = np.full((F, B), np.nan, dtype=np.float64)
    brad = np.full((F, B), np.nan, dtype=np.float64)
    spos = np.full((F, S, 3), np.nan, dtype=np.float32) if S > 0 else None
    energy = np.full(F, np.nan, dtype=np.float64)
    times = np.zeros(F, dtype=np.float64)

    compute_e = N0 <= ENERGY_EVERY_FRAME_MAX_N
    for k, snap_idx in enumerate(indices):
        sim = sa[int(snap_idx)]
        times[k] = float(sim.t)
        x, v, m, r = read_particles(sim)
        nb = min(B, len(x))
        bpos[k, :nb] = x[:nb]
        bvel[k, :nb] = v[:nb]
        bmass[k, :nb] = m[:nb]
        brad[k, :nb] = r[:nb]
        if S > 0 and len(x) > B:
            ns = min(S, len(x) - B)
            spos[k, :ns] = x[B:B + ns].astype(np.float32)
        if compute_e or k == 0 or k == F - 1:
            try:
                energy[k] = float(sim.energy())
            except Exception:
                pass

    t_span = float(times[-1] - times[0]) if F > 1 else 0.0
    tpf = t_span / max(1, F - 1) if t_span > 0 else 0.05
    G = float(sim0.G)
    dt = float(getattr(sim0, "dt", 0.01))
    integrator = str(getattr(sim0, "integrator", "ias15"))
    if integrator not in {"ias15", "whfast", "mercurius", "trace", "saba", "leapfrog", "janus", "bs"}:
        integrator = "ias15"

    heaviest = int(np.argmax(m0[:B])) if B > 0 else 0
    bodies_spec = []
    for b in range(B):
        name = "Star" if (b == heaviest and B > 1) else f"Body {b + 1}"
        bodies_spec.append(BodySpec(
            slot=b, name=name, mass=float(m0[b]), radius=float(r0[b]),
            mode="CARTESIAN",
            pos=(float(x0[b, 0]), float(x0[b, 1]), float(x0[b, 2])),
            vel=(float(v0[b, 0]), float(v0[b, 1]), float(v0[b, 2])),
            ref=heaviest if b != heaviest else None,
            order=b,
        ))
    swarms_spec = []
    if S > 0:
        swarms_spec.append(SwarmSpec(name="Swarm", count=S, primary=heaviest, first=0))

    spec = SimSpec(
        bodies=bodies_spec,
        swarms=swarms_spec,
        units=units if units in UNIT_SYSTEMS else "CODE",
        G=G,
        integrator=integrator,
        dt_mode="MANUAL",
        dt=max(dt, 1e-12),
        time_per_frame=max(tpf, 1e-12),
        n_frames=F,
        move_to_com=False,
    )
    run = MergedRun(
        spec=spec, G=G, dt=dt, offset0=np.zeros(3),
        bpos=bpos, bvel=bvel, bmass=bmass, brad=brad,
        spos=spos, energy=energy,
        messages=[f"Imported {F} snapshots from REBOUND SimulationArchive '{os.path.basename(path)}'."],
    )
    run.times = times
    meta = {
        "label": os.path.splitext(os.path.basename(path))[0],
        "source_archive": os.path.basename(path),
    }
    return run, meta


# --------------------------------------------------------------------------
# Array helpers
# --------------------------------------------------------------------------

def ffill(a):
    """Forward-fill rows of ``a`` (shape (F, ...)) whose first value is NaN."""
    a = np.asarray(a)
    if a.shape[0] == 0:
        return a.copy()
    flat = a.reshape(a.shape[0], -1)
    bad = np.isnan(flat[:, 0])
    if not bad.any():
        return a.copy()
    idx = np.where(~bad, np.arange(a.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    return a[idx]


def interpolate_swarm_frames(P: np.ndarray, target_frames: int) -> np.ndarray:
    """Linearly resample swarm positions ``P`` of shape ``(K, S, 3)`` onto
    ``target_frames`` evenly spaced frames ``(F_anim, S, 3)``."""
    P = np.asarray(P)
    K = P.shape[0]
    F_anim = max(1, int(target_frames))
    if K == F_anim or K <= 1:
        if K == 1 and F_anim > 1:
            return np.broadcast_to(P, (F_anim,) + P.shape[1:]).copy()
        return P
    u = np.linspace(0.0, float(K - 1), F_anim, dtype=np.float64)
    i0 = np.clip(np.floor(u).astype(np.int64), 0, K - 1)
    i1 = np.clip(i0 + 1, 0, K - 1)
    alpha = (u - i0)[:, None, None]
    return (1.0 - alpha) * P[i0] + alpha * P[i1]


def _normalize(v, fallback):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    ok = n[..., 0] > 1e-300
    out = np.where(ok[..., None], v / np.where(n > 0, n, 1.0), fallback)
    return out


def ref_state(bpos, bvel, bmass, slots):
    """Mass-weighted position, velocity and mass of a set of body slots over
    time. Returns (pos (F,3), vel (F,3), M (F,)) or None."""
    if not slots:
        return None
    m = bmass[:, slots]
    valid = np.isfinite(m)
    mw = np.where(valid, m, 0.0)
    M = mw.sum(axis=1)
    P = np.where(valid[..., None], bpos[:, slots], 0.0)
    V = np.where(valid[..., None], bvel[:, slots], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = (P * mw[..., None]).sum(1) / M[:, None]
        vel = (V * mw[..., None]).sum(1) / M[:, None]
        # massless references: plain average of the valid ones
        cnt = valid.sum(1)
        zero = (M <= 0) & (cnt > 0)
        if zero.any():
            pos[zero] = P[zero].sum(1) / cnt[zero, None]
            vel[zero] = V[zero].sum(1) / cnt[zero, None]
    pos[cnt == 0] = np.nan
    vel[cnt == 0] = np.nan
    return pos, vel, M


# --------------------------------------------------------------------------
# Reference frames
# --------------------------------------------------------------------------

def frame_transforms(bpos, bvel, bmass, mode="INERTIAL", center=None, secondary=None,
                     anchor=True, offset0=None):
    """Per-frame rigid transforms x' = Q[f] @ (x - c[f]) + d.

    mode: 'INERTIAL' (simulation frame), 'CENTERED' (on body ``center``) or
    'COROTATING' (rotating with the pair center -> secondary).
    With ``anchor`` the transform is the identity map onto the *raw* input
    coordinates at the first frame, so the output starts where bodies were
    placed and only the motion is changed.
    """
    F = bpos.shape[0]
    off = np.zeros(3) if offset0 is None else np.asarray(offset0, float)
    Q = np.broadcast_to(np.eye(3), (F, 3, 3)).copy()
    c = np.zeros((F, 3))
    d = np.zeros(3)
    if F == 0:
        return Q, c, d

    if mode == "CENTERED" and center is not None:
        c = ffill(bpos[:, center])
        if anchor:
            d = c[0] + off
    elif mode == "COROTATING" and center is not None and secondary is not None \
            and center != secondary:
        p = ffill(bpos[:, center]); s = ffill(bpos[:, secondary])
        vp = ffill(bvel[:, center]); vs = ffill(bvel[:, secondary])
        mp = ffill(bmass[:, center][:, None])[:, 0]; ms = ffill(bmass[:, secondary][:, None])[:, 0]
        mt = mp + ms
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.where((mt > 0)[:, None], (mp[:, None] * p + ms[:, None] * s) / mt[:, None], p)
        rel = s - p
        relv = vs - vp
        x = _normalize(rel, np.array([1.0, 0.0, 0.0]))
        z = _normalize(np.cross(rel, relv), np.array([0.0, 0.0, 1.0]))
        y = np.cross(z, x)
        Bm = np.stack([x, y, z], axis=-1)            # columns = rotating axes
        Bt = np.transpose(Bm, (0, 2, 1))
        if anchor:
            Q = np.einsum("ij,fjk->fik", Bm[0], Bt)
            d = c[0] + off
        else:
            Q = Bt
    else:  # INERTIAL
        if anchor:
            d = off.copy()
    return Q, c, d


def apply_frame(X, Q, c, d):
    """Apply per-frame transforms to X of shape (F,3) or (F,N,3)."""
    X = np.asarray(X)
    F = X.shape[0]
    Qf, cf = Q[:F], c[:F]
    if X.ndim == 2:
        return np.einsum("fij,fj->fi", Qf, X - cf) + d
    return np.einsum("fij,fnj->fni", Qf, X - cf[:, None, :]) + d


# --------------------------------------------------------------------------
# Two-body mechanics
# --------------------------------------------------------------------------

def two_body_elements(r, v, mu):
    """Osculating elements from relative state vectors (vectorised).

    Returns a, e, inc, e_vec, h (a is negative for hyperbolic orbits).
    """
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    mu = np.asarray(mu, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rn = np.linalg.norm(r, axis=-1)
        h = np.cross(r, v)
        hn = np.linalg.norm(h, axis=-1)
        vv = (v * v).sum(-1)
        eps = 0.5 * vv - mu / rn
        a = -mu / (2.0 * eps)
        evec = np.cross(v, h) / mu[..., None] - r / rn[..., None]
        e = np.linalg.norm(evec, axis=-1)
        inc = np.arccos(np.clip(h[..., 2] / hn, -1.0, 1.0))
    return a, e, inc, evec, h


def ellipse_frames(r, v, mu, focus):
    """Affine frames that map the unit circle onto osculating ellipses.

    ellipse = centre + R @ diag(a, b, 1) @ (cos E, sin E, 0).
    Returns (centre (F,3), R (F,3,3), scale (F,3), bound (F,) bool).
    """
    a, e, _inc, evec, h = two_body_elements(r, v, mu)
    bound = np.isfinite(a) & (a > 0) & np.isfinite(e) & (e < 1.0)
    W = _normalize(h, np.array([0.0, 0.0, 1.0]))
    with np.errstate(invalid="ignore", divide="ignore"):
        P_from_e = evec / np.where(e > 0, e, 1.0)[..., None]
    rhat = _normalize(r, np.array([1.0, 0.0, 0.0]))
    P = np.where((e > 1e-9)[..., None], P_from_e, rhat)
    P = P - (P * W).sum(-1, keepdims=True) * W          # exact orthogonality
    P = _normalize(P, np.array([1.0, 0.0, 0.0]))
    Qv = np.cross(W, P)
    R = np.stack([P, Qv, W], axis=-1)
    ee = np.where(bound, e, 0.0)
    aa = np.where(bound, a, 0.0)
    b = aa * np.sqrt(1.0 - ee * ee)
    centre = focus - aa[..., None] * ee[..., None] * P
    scale = np.stack([aa, b, np.where(bound, 1.0, 0.0)], axis=-1)
    return centre, R, scale, bound


def solve_kepler(M, e, iterations=30):
    M = np.asarray(M, float)
    e = np.asarray(e, float)
    E = np.where(e < 0.8, M, np.pi * np.ones_like(M))
    for _ in range(iterations):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


def elements_to_cartesian(mu, a, e, inc, Omega, omega, l):
    """Positions and velocities from elements with mean longitude ``l``."""
    a, e, inc, Omega, omega, l = (np.asarray(x, float) for x in (a, e, inc, Omega, omega, l))
    M = np.mod(l - Omega - omega, TWO_PI)
    E = solve_kepler(M, e)
    cE, sE = np.cos(E), np.sin(E)
    b = a * np.sqrt(1.0 - e * e)
    n = np.sqrt(mu / a ** 3)
    xp = a * (cE - e)
    yp = b * sE
    denom = 1.0 - e * cE
    vxp = -a * n * sE / denom
    vyp = b * n * cE / denom
    cO, sO, co, so, ci, si = np.cos(Omega), np.sin(Omega), np.cos(omega), np.sin(omega), np.cos(inc), np.sin(inc)
    Px = cO * co - sO * so * ci
    Py = sO * co + cO * so * ci
    Pz = so * si
    Qx = -cO * so - sO * co * ci
    Qy = -sO * so + cO * co * ci
    Qz = co * si
    pos = np.stack([xp * Px + yp * Qx, xp * Py + yp * Qy, xp * Pz + yp * Qz], axis=-1)
    vel = np.stack([vxp * Px + vyp * Qx, vxp * Py + vyp * Qy, vxp * Pz + vyp * Qz], axis=-1)
    return pos, vel


def swarm_initial_positions(spec: SwarmSpec, G, prim_pos, prim_vel, prim_mass):
    """Approximate initial swarm positions (same RNG stream as build_simulation),
    used for the viewport preview before baking."""
    n = int(spec.count)
    rng = np.random.default_rng(int(spec.seed))
    a = rng.uniform(spec.a_min, max(spec.a_max, spec.a_min), n)
    e = rng.uniform(0.0, max(0.0, min(spec.e_max, 0.99)), n)
    inc = rng.uniform(0.0, max(0.0, spec.inc_max), n)
    Om = rng.uniform(0.0, TWO_PI, n)
    om = rng.uniform(0.0, TWO_PI, n)
    lam = spec.lon_center + rng.uniform(-0.5, 0.5, n) * spec.lon_width
    pos, _vel = elements_to_cartesian(G * prim_mass, a, e, inc, Om, om, lam)
    return pos + np.asarray(prim_pos)


# --------------------------------------------------------------------------
# Rotations
# --------------------------------------------------------------------------

def mat_to_quat(R):
    """Rotation matrices (...,3,3) -> unit quaternions (...,4) as (w, x, y, z)."""
    R = np.asarray(R, float)
    shp = R.shape[:-2]
    M = R.reshape(-1, 3, 3)
    q = np.empty((M.shape[0], 4))
    m00, m11, m22 = M[:, 0, 0], M[:, 1, 1], M[:, 2, 2]
    tr = m00 + m11 + m22
    k = np.argmax(np.stack([tr, m00, m11, m22], axis=1), axis=1)

    i = k == 0
    s = np.sqrt(np.maximum(1.0 + tr[i], 1e-300)) * 2.0
    q[i, 0] = 0.25 * s
    q[i, 1] = (M[i, 2, 1] - M[i, 1, 2]) / s
    q[i, 2] = (M[i, 0, 2] - M[i, 2, 0]) / s
    q[i, 3] = (M[i, 1, 0] - M[i, 0, 1]) / s

    i = k == 1
    s = np.sqrt(np.maximum(1.0 + m00[i] - m11[i] - m22[i], 1e-300)) * 2.0
    q[i, 0] = (M[i, 2, 1] - M[i, 1, 2]) / s
    q[i, 1] = 0.25 * s
    q[i, 2] = (M[i, 0, 1] + M[i, 1, 0]) / s
    q[i, 3] = (M[i, 0, 2] + M[i, 2, 0]) / s

    i = k == 2
    s = np.sqrt(np.maximum(1.0 + m11[i] - m00[i] - m22[i], 1e-300)) * 2.0
    q[i, 0] = (M[i, 0, 2] - M[i, 2, 0]) / s
    q[i, 1] = (M[i, 0, 1] + M[i, 1, 0]) / s
    q[i, 2] = 0.25 * s
    q[i, 3] = (M[i, 1, 2] + M[i, 2, 1]) / s

    i = k == 3
    s = np.sqrt(np.maximum(1.0 + m22[i] - m00[i] - m11[i], 1e-300)) * 2.0
    q[i, 0] = (M[i, 1, 0] - M[i, 0, 1]) / s
    q[i, 1] = (M[i, 0, 2] + M[i, 2, 0]) / s
    q[i, 2] = (M[i, 1, 2] + M[i, 2, 1]) / s
    q[i, 3] = 0.25 * s

    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.reshape(shp + (4,))


def quat_to_mat(q):
    q = np.asarray(q, float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def make_quats_continuous(q):
    """Flip signs so consecutive quaternions (axis 0) lie in the same hemisphere."""
    q = np.array(q, float)
    if len(q) < 2:
        return q
    dots = (q[1:] * q[:-1]).sum(-1)
    s = np.concatenate([[1.0], np.where(dots < 0, -1.0, 1.0)])
    return q * np.cumprod(s)[:, None]


# --------------------------------------------------------------------------
# Convenience: everything the visualisation needs from a finished runner
# --------------------------------------------------------------------------

def body_elements(runner: Runner, slot):
    """(a, e, inc) time series of a body relative to its orbit reference, or None."""
    slots = runner.refs.get(slot)
    st = ref_state(runner.bpos, runner.bvel, runner.bmass, slots) if slots else None
    if st is None:
        return None
    rp, rv, M = st
    mu = runner.G * (M + runner.bmass[:, slot])
    a, e, inc, _ev, _h = two_body_elements(runner.bpos[:, slot] - rp, runner.bvel[:, slot] - rv, mu)
    return a, e, inc


def body_ellipses(runner: Runner, slot):
    """Osculating ellipse frames of a body (simulation coordinates) or None."""
    slots = runner.refs.get(slot)
    st = ref_state(runner.bpos, runner.bvel, runner.bmass, slots) if slots else None
    if st is None:
        return None
    rp, rv, M = st
    mu = runner.G * (M + runner.bmass[:, slot])
    centre, R, scale, bound = ellipse_frames(runner.bpos[:, slot] - rp,
                                             runner.bvel[:, slot] - rv, mu, rp)
    bound &= np.isfinite(runner.bpos[:, slot, 0])
    return centre, R, scale, bound
