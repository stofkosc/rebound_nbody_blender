"""Baking: run REBOUND, write keyframes/trails/orbits/swarms, and manage Bake History."""

import dataclasses
import datetime
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time

import bpy
import numpy as np
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Matrix

from . import anim, cli, core, deps, nodes, scene_io

GEN_KEY = "rebound_generated"
TRAILS_COLL = "REBOUND Trails"
ORBITS_COLL = "REBOUND Orbits"
ELEMENT_KEYS = ("rb_a", "rb_e", "rb_inc")
MAX_SWARM_VERTICES = 60_000_000
MIN_PARTICLES_PER_WORKER = core.MIN_PARTICLES_PER_WORKER

# Last finished or loaded bake, reused by CSV export and visualization updates.
LAST = {"sig": None, "runner": None, "bodies": None, "swarms": None, "meta": None}

# In-memory cache of loaded/finished bakes keyed by RB_BakeEntry.uid
_RUN_CACHE = {}
_SUPPRESS_BAKE_UPDATE = False


def is_suppressing_bake_update() -> bool:
    return _SUPPRESS_BAKE_UPDATE


def spec_signature(spec):
    return repr(dataclasses.asdict(spec))


def _default_workers() -> int:
    return core.default_worker_count()


def choose_worker_count(scene, spec: core.SimSpec) -> int:
    s = scene.rebound
    return core.resolve_worker_count(
        spec,
        requested=int(s.parallel_workers),
        parallel=bool(s.parallel_bake),
        min_per_worker=MIN_PARTICLES_PER_WORKER,
    )


# ---------------------------------------------------------------------------
# Bake folder & metadata helpers
# ---------------------------------------------------------------------------

def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return slug.strip("_") or "bake"


def resolve_bake_dir(scene, create: bool = False) -> str:
    """Return an absolute directory path for storing ``.npz`` bake files."""
    raw = (scene.rebound.bake_dir or "//rebound_bakes").strip()
    if raw.startswith("//") and not bpy.data.is_saved:
        rel = raw[2:].strip("/\\") or "rebound_bakes"
        d = os.path.join(tempfile.gettempdir(), "rebound_nbody_unsaved", rel)
    else:
        d = os.path.abspath(bpy.path.abspath(raw))
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def _store_filepath(abs_path: str) -> str:
    if bpy.data.is_saved:
        try:
            return bpy.path.relpath(abs_path)
        except ValueError:
            return abs_path
    return abs_path


def _abspath_bake(filepath: str) -> str:
    if not filepath:
        return ""
    return os.path.abspath(bpy.path.abspath(filepath))


def collect_scene_meta(scene, bodies, swarms, info: str = "", warnings=()) -> dict:
    """Capture visual and scene settings so a bake can be faithfully restored later."""
    s = scene.rebound
    ls = scene_io.length_scale(scene)
    body_vis = {}
    for o in bodies:
        rb = o.rebound
        loc = list(rb.initial_location) if rb.has_initial else list(o.matrix_world.translation)
        sc = list(rb.initial_scale) if rb.has_initial else list(o.scale)
        body_vis[o.name] = {
            "color": [float(c) for c in rb.color],
            "scale": [float(c) for c in sc],
            "initial_location": [float(c) for c in loc],
            "show_trail": bool(rb.show_trail),
            "show_orbit": bool(rb.show_orbit),
        }
    swarm_vis = {}
    for o in swarms:
        rb = o.rebound
        swarm_vis[o.name] = {
            "color": [float(c) for c in rb.color],
            "particle_size": float(rb.particle_size),
            "tail": int(rb.tail),
        }
    return {
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "frame_start": int(scene.frame_start),
        "length_scale": float(ls),
        "frame_mode": str(s.frame_mode),
        "frame_center": s.frame_center.name if s.frame_center else "",
        "frame_secondary": s.frame_secondary.name if s.frame_secondary else "",
        "anchor_frame": bool(s.anchor_frame),
        "info": str(info),
        "warnings": [str(w) for w in warnings],
        "body_visuals": body_vis,
        "swarm_visuals": swarm_vis,
    }


def _set_active_bake_index(scene, idx: int):
    global _SUPPRESS_BAKE_UPDATE
    s = scene.rebound
    prev = _SUPPRESS_BAKE_UPDATE
    _SUPPRESS_BAKE_UPDATE = True
    try:
        for i, item in enumerate(s.bakes):
            item.is_loaded = (i == idx)
        if idx >= 0:
            s.active_bake_index = idx
    finally:
        _SUPPRESS_BAKE_UPDATE = prev


def register_bake_entry(scene, run, meta: dict, label: str = "", filepath: str = "",
                        save_to_disk: bool | None = None):
    """Record a finished or imported simulation in ``scene.rebound.bakes``."""
    s = scene.rebound
    now_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    seq = len(s.bakes) + 1
    uid = f"bake_{now_tag}_{seq:02d}"
    while any(b.uid == uid for b in s.bakes):
        seq += 1
        uid = f"bake_{now_tag}_{seq:02d}"

    S_total = sum(max(0, int(sw.count)) for sw in run.spec.swarms)
    if not label:
        label = meta.get("label") or (
            f"#{len(s.bakes) + 1} {run.spec.integrator.upper()} "
            f"({run.F}f, {run.B}b" + (f", {S_total}p)" if S_total else ")")
        )
    meta = dict(meta or {})
    meta["label"] = label
    meta.setdefault("created_at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if save_to_disk is None:
        save_to_disk = bool(s.save_bakes_to_disk)

    stored_path = filepath
    if not stored_path and save_to_disk:
        try:
            bdir = resolve_bake_dir(scene, create=True)
            fname = f"{uid}_{_safe_slug(label)}.npz"
            abs_path = os.path.join(bdir, fname)
            core.save_run_npz(abs_path, run, meta=meta)
            stored_path = _store_filepath(abs_path)
        except Exception:
            stored_path = ""

    entry = s.bakes.add()
    entry.uid = uid
    entry.name = label
    entry.filepath = stored_path
    entry.created_at = str(meta.get("created_at", ""))
    entry.integrator = str(run.spec.integrator)
    entry.n_frames = int(run.F)
    entry.n_bodies = int(run.B)
    entry.n_swarm = int(S_total)
    entry.sim_time = float(run.times[-1]) if len(run.times) else 0.0
    entry.unit_time = core.UNIT_LABELS.get(run.spec.units, ("L", "T", "M"))[1]
    rel = run.relative_energy_error()
    finite_rel = rel[np.isfinite(rel)]
    entry.energy_error = float(finite_rel[-1]) if finite_rel.size else -1.0
    split = getattr(run, "split", None)
    entry.split_error = (
        float(max(split.max_rel_body, split.max_rel_probe))
        if (split is not None and split.n_chunks > 1) else -1.0
    )
    entry.summary = str(meta.get("info") or s.bake_info or "")

    _RUN_CACHE[uid] = (run, meta)
    _set_active_bake_index(scene, len(s.bakes) - 1)
    return entry


def get_entry_run(entry):
    """Return ``(run, meta)`` for a ``RB_BakeEntry`` from cache or disk."""
    if entry.uid and entry.uid in _RUN_CACHE:
        return _RUN_CACHE[entry.uid]
    abs_path = _abspath_bake(entry.filepath)
    if abs_path and os.path.isfile(abs_path):
        run, meta = core.load_run_npz(abs_path)
        if entry.uid:
            _RUN_CACHE[entry.uid] = (run, meta)
        return run, meta
    raise core.SetupError(
        f"Bake file for '{entry.name}' not found ({entry.filepath or 'unsaved memory cache'})."
    )


# ---------------------------------------------------------------------------
# Generated-object bookkeeping
# ---------------------------------------------------------------------------

def ensure_collection(scene, name):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        coll[GEN_KEY] = "collection"
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


def remove_generated(scene, kinds=None):
    for obj in list(scene.objects):
        kind = obj.get(GEN_KEY)
        if not kind or (kinds is not None and kind not in kinds):
            continue
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and data.users == 0:
            if isinstance(data, bpy.types.Curve):
                bpy.data.curves.remove(data)
            elif isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
    for name in (TRAILS_COLL, ORBITS_COLL):
        coll = bpy.data.collections.get(name)
        if coll is not None and coll.get(GEN_KEY) and len(coll.all_objects) == 0:
            bpy.data.collections.remove(coll)


def _hide(obj, view_layer):
    try:
        obj.hide_set(True, view_layer=view_layer)
    except (RuntimeError, TypeError):
        obj.hide_viewport = False
    obj.hide_render = True


def _unhide(obj, view_layer):
    obj.hide_viewport = False
    obj.hide_render = False
    try:
        obj.hide_set(False, view_layer=view_layer)
    except (RuntimeError, TypeError):
        pass


# ---------------------------------------------------------------------------
# Job (background thread for single-process, subprocess pool for parallel swarms)
# ---------------------------------------------------------------------------

_ParallelPool = core.ParallelPool


class BakeJob:
    def __init__(self, scene):
        rb = deps.get_rebound()
        if rb is None:
            raise core.SetupError("REBOUND is not installed. Use 'Install REBOUND' in the REBOUND panel.")
        self.scene = scene
        bodies = scene_io.body_objects(scene)
        if not bodies:
            raise core.SetupError("There are no REBOUND bodies in the scene.")
        self.warnings = []
        self.stored = []
        for o in bodies:
            if o.parent is not None:
                mw = o.matrix_world.copy()
                o.parent = None
                o.matrix_world = mw
                self.warnings.append(f"'{o.name}' was un-parented (bodies are animated in world space).")
            r = o.rebound
            if not r.has_initial:
                r.initial_location = o.matrix_world.translation
                r.initial_scale = o.scale
                r.has_initial = True
                self.stored.append(o)
        self.pool = None
        self.runner = None
        self._thread = None
        self._thread_exc = None
        try:
            self.spec, self.bodies, self.swarms, self.slot_of = scene_io.make_spec(scene, archive=True)
            self.sig = spec_signature(self.spec)
            compute_energy = bool(scene.rebound.bake_energy)
            n_workers = choose_worker_count(scene, self.spec)
            if n_workers > 1:
                # Validate setup quickly before spawning subprocesses
                core.build_simulation(rb, core.split_specs(self.spec, n_workers, n_probes=0)[0])
                self.pool = _ParallelPool(
                    self.spec, n_workers, n_probes=int(scene.rebound.parallel_probes),
                    compute_energy=compute_energy, deps_dir=deps.deps_dir())
            else:
                self.runner = core.Runner(rb, self.spec, compute_energy=compute_energy)
                if self.runner.archive_error:
                    self.warnings.append(f"SimulationArchive not written: {self.runner.archive_error}")
        except Exception:
            self.abort()
            raise
        self.t0 = time.perf_counter()

    def start_async(self):
        """For single-process bakes, run the integration on a background thread
        so Blender's main thread and UI timer never block each other."""
        if self.runner is not None and self._thread is None:
            def _target():
                try:
                    self.runner.run()
                except Exception as ex:
                    self._thread_exc = ex
            self._thread = threading.Thread(target=_target, name="REBOUND-Bake", daemon=True)
            self._thread.start()

    @property
    def done(self):
        if self.pool is not None:
            return self.pool.done
        if self._thread is not None:
            return not self._thread.is_alive() and self.runner.done
        return self.runner.done

    def poll(self) -> float:
        """Check progress of async thread or worker subprocesses."""
        if self.pool is not None:
            return self.pool.poll_progress()
        if self._thread_exc is not None:
            raise self._thread_exc
        if self._thread is not None:
            return self.runner.progress
        return self.runner.step(0.1)

    def run_sync(self):
        if self.pool is not None:
            self.runner = self.pool.wait_and_collect(time.perf_counter() - self.t0)
            if self.pool.archive_error:
                self.warnings.append(f"SimulationArchive not written: {self.pool.archive_error}")
        else:
            self.runner.run()

    def abort(self):
        if self.pool is not None:
            self.pool.kill()
        if self.runner is not None and hasattr(self.runner, "stop"):
            self.runner.stop()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        for o in self.stored:
            try:
                o.rebound.has_initial = False
            except ReferenceError:
                pass

    def finish(self):
        if self.pool is not None and self.runner is None:
            self.runner = self.pool.wait_and_collect(time.perf_counter() - self.t0)
            if self.pool.archive_error:
                self.warnings.append(f"SimulationArchive not written: {self.pool.archive_error}")
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread_exc is not None:
                raise self._thread_exc
        self.runner.wall = time.perf_counter() - self.t0
        info = write_results(self)
        meta = collect_scene_meta(self.scene, self.bodies, self.swarms,
                                  info=info, warnings=self.warnings)
        LAST.update(sig=self.sig, runner=self.runner,
                    bodies=[o.name for o in self.bodies],
                    swarms=[o.name for o in self.swarms],
                    meta=meta)
        register_bake_entry(self.scene, self.runner, meta)
        return info


# ---------------------------------------------------------------------------
# Syncing scene objects & writing results from any Runner / MergedRun
# ---------------------------------------------------------------------------

def _finite(a, fill=0.0):
    return np.nan_to_num(a, nan=fill, posinf=fill, neginf=fill)


def _auto_length_scale(run) -> float:
    """Estimate a viewport-friendly length scale for an imported external bake."""
    pts = []
    if run.bpos.size:
        finite_b = run.bpos[0][np.isfinite(run.bpos[0, :, 0])]
        if finite_b.size:
            pts.append(finite_b)
    if run.spos is not None and run.spos.size:
        finite_s = run.spos[0][np.isfinite(run.spos[0, :, 0])]
        if finite_s.size:
            pts.append(finite_s)
    if not pts:
        return 1.0
    all_pts = np.concatenate(pts, axis=0)
    radii = np.linalg.norm(all_pts, axis=-1)
    far = float(np.nanmax(radii)) if radii.size else 0.0
    if far <= 1e-12:
        return 1.0
    if 0.2 <= far <= 15.0:
        return 1.0
    return 5.0 / far


def _sync_objects_for_run(scene, run, meta: dict | None = None):
    """Ensure the Blender scene has body and swarm objects matching ``run.spec``."""
    from . import presets

    meta = meta or {}
    s = scene.rebound
    view_layer = bpy.context.view_layer

    if "frame_start" in meta:
        scene.frame_start = int(meta["frame_start"])
    scene.frame_end = scene.frame_start + max(1, int(run.F)) - 1
    if run.spec.units in core.UNIT_SYSTEMS:
        s.units = run.spec.units
    if run.G > 0:
        s.G = float(run.G)
    if run.spec.time_per_frame > 0:
        s.time_per_frame = float(run.spec.time_per_frame)

    if "length_scale" in meta and float(meta["length_scale"]) > 0:
        s.length_scale = float(meta["length_scale"])
    elif not any(o.rebound.enabled for o in scene.objects):
        s.length_scale = _auto_length_scale(run)

    ls = scene_io.length_scale(scene)
    coll = ensure_collection(scene, presets.BODIES_COLL)
    body_vis = meta.get("body_visuals") or {}
    swarm_vis = meta.get("swarm_visuals") or {}

    specs_by_slot = sorted(run.spec.bodies, key=lambda b: b.slot)
    heaviest_slot = max(specs_by_slot, key=lambda b: b.mass).slot if specs_by_slot else 0

    body_objs = []
    slot_to_obj = {}
    for i, b in enumerate(specs_by_slot):
        vis = body_vis.get(b.name) or {}
        col = tuple(vis.get("color") or presets.PALETTE[i % len(presets.PALETTE)])
        scale_vec = tuple(vis.get("scale") or ((0.25, 0.25, 0.25) if b.slot == heaviest_slot else (0.08, 0.08, 0.08)))
        init_loc = tuple(vis.get("initial_location") or (c * ls for c in b.pos))
        obj = bpy.data.objects.get(b.name)
        if obj is None or obj.get(GEN_KEY):
            emission = 8.0 if (b.slot == heaviest_slot and len(specs_by_slot) > 1) else 0.0
            obj = presets.create_body(
                scene, b.name, b.mass, col, scale_vec[0],
                location=init_loc, emission=emission,
                show_trail=bool(vis.get("show_trail", b.slot != heaviest_slot or len(specs_by_slot) <= 3)),
            )
            obj.scale = scale_vec
        else:
            if obj.name not in scene.objects:
                coll.objects.link(obj)
            _unhide(obj, view_layer)
            obj.rebound.enabled = True
            obj.rebound.kind = "BODY"
            if "color" in vis:
                obj.rebound.color = col
            if "show_trail" in vis:
                obj.rebound.show_trail = bool(vis["show_trail"])
            if "show_orbit" in vis:
                obj.rebound.show_orbit = bool(vis["show_orbit"])

        rb = obj.rebound
        rb.mass = float(b.mass)
        rb.radius = float(b.radius)
        rb.ic_mode = b.mode if b.mode in {"CARTESIAN", "ORBIT"} else "CARTESIAN"
        rb.velocity = tuple(float(v) for v in b.vel)
        rb.a = float(b.a)
        rb.e = float(b.e)
        rb.inc = float(b.inc)
        rb.Omega = float(b.Omega)
        rb.omega = float(b.omega)
        rb.anomaly = float(b.anomaly)
        if b.anomaly_type in {"f", "M", "l"}:
            rb.anomaly_type = b.anomaly_type
        rb.order = int(b.order)
        rb.initial_location = init_loc
        rb.initial_scale = scale_vec
        rb.has_initial = True
        body_objs.append(obj)
        slot_to_obj[b.slot] = obj

    for b, obj in zip(specs_by_slot, body_objs):
        prim_slot = b.primary if b.mode == "ORBIT" else b.ref
        obj.rebound.primary = slot_to_obj.get(prim_slot) if prim_slot is not None else None

    core.assign_swarm_offsets(run.spec)
    swarm_objs = []
    for i, sw in enumerate(run.spec.swarms):
        vis = swarm_vis.get(sw.name) or {}
        col = tuple(vis.get("color") or (0.85, 0.85, 1.0, 1.0))
        prim_obj = slot_to_obj.get(sw.primary) if sw.primary is not None else slot_to_obj.get(heaviest_slot)
        obj = bpy.data.objects.get(sw.name)
        if obj is None or obj.type != "MESH" or obj.get(GEN_KEY):
            obj = presets.create_swarm(scene, sw.name, prim_obj, col)
        else:
            if obj.name not in scene.objects:
                coll.objects.link(obj)
            _unhide(obj, view_layer)
            obj.rebound.enabled = True
            obj.rebound.kind = "SWARM"
            obj.rebound.primary = prim_obj
            if "color" in vis:
                obj.rebound.color = col
        rb = obj.rebound
        rb.count = int(sw.count)
        rb.a_min = float(sw.a_min)
        rb.a_max = float(sw.a_max)
        rb.e_max = float(sw.e_max)
        rb.inc_max = float(sw.inc_max)
        rb.lon_center = float(sw.lon_center)
        rb.lon_width = float(sw.lon_width)
        rb.seed = int(sw.seed)
        if "particle_size" in vis:
            rb.particle_size = float(vis["particle_size"])
        if "tail" in vis:
            rb.tail = int(vis["tail"])
        swarm_objs.append(obj)

    active_names = {o.name for o in body_objs} | {o.name for o in swarm_objs}
    for o in list(scene.objects):
        if o.get(GEN_KEY) or o.name in active_names:
            continue
        if o.rebound.enabled or o.rebound.has_initial:
            anim.clear_rebound_action(o)
            o.rebound.enabled = False
            _hide(o, view_layer)

    if meta.get("frame_mode") in {"INERTIAL", "CENTERED", "COROTATING"}:
        s.frame_mode = meta["frame_mode"]
    if meta.get("frame_center"):
        fc = scene.objects.get(meta["frame_center"])
        if fc is not None:
            s.frame_center = fc
    if meta.get("frame_secondary"):
        fs = scene.objects.get(meta["frame_secondary"])
        if fs is not None:
            s.frame_secondary = fs
    if "anchor_frame" in meta:
        s.anchor_frame = bool(meta["anchor_frame"])

    slot_of = {o.name: b.slot for b, o in zip(specs_by_slot, body_objs)}
    return body_objs, swarm_objs, slot_of


def write_results(job):
    return apply_run_to_scene(
        job.scene,
        job.runner,
        bodies=job.bodies,
        swarms=job.swarms,
        slot_of=job.slot_of,
        warnings=job.warnings,
        wall=time.perf_counter() - job.t0,
    )


def compute_retime_mapping(scene, run):
    """Map ``run`` snapshots onto Blender timeline frames according to the scene's
    Retiming & Time Window settings.

    Returns ``(k_lo, k_hi, F_anim, frame_step, snap_frames)``.
    """
    s = scene.rebound
    K_total = max(1, int(run.F))
    times = np.asarray(run.times, dtype=np.float64)

    if s.retime_use_window and K_total >= 2:
        ufac = core.time_unit_factor(run.spec.units, s.retime_time_unit)
        t0_req = float(s.retime_start_time) * ufac
        t1_req = float(s.retime_end_time) * ufac
        k_lo = int(np.searchsorted(times, t0_req, side="left"))
        k_lo = max(0, min(k_lo, K_total - 2))
        if t1_req > t0_req:
            k_hi = int(np.searchsorted(times, t1_req, side="right"))
            k_hi = max(k_lo + 2, min(k_hi, K_total))
        else:
            k_hi = K_total
    else:
        k_lo, k_hi = 0, K_total

    K_win = max(1, k_hi - k_lo)
    t_win = float(times[k_hi - 1] - times[k_lo]) if K_win >= 2 else 0.0
    mode = s.retime_mode

    if mode == "FIT_FRAMES" and K_win >= 2:
        F_anim = max(2, int(s.retime_frames))
    elif mode == "FRAMES_PER_ORBIT" and K_win >= 2 and t_win > 0:
        P_ref = scene_io.reference_period(scene, spec=run.spec)
        if P_ref and P_ref > 0:
            F_anim = max(2, int(round((t_win / P_ref) * max(0.1, float(s.retime_frames_per_orbit)))) + 1)
        else:
            F_anim = K_win
    elif mode == "FRAMES_PER_TIME" and K_win >= 2 and t_win > 0:
        ufac = core.time_unit_factor(run.spec.units, s.retime_time_unit)
        F_anim = max(2, int(round((t_win / ufac) * max(1e-6, float(s.retime_frames_per_time)))) + 1)
    else:
        F_anim = K_win

    F_anim = min(F_anim, 500_000)
    f0 = int(scene.frame_start)
    frame_step = float(F_anim - 1) / float(max(1, K_win - 1)) if K_win > 1 else 1.0
    snap_frames = f0 + np.arange(K_win, dtype=np.float64) * frame_step
    return k_lo, k_hi, F_anim, frame_step, snap_frames


def apply_run_to_scene(scene, run, bodies=None, swarms=None, slot_of=None,
                       warnings=None, wall=None, meta=None, sync_objects=False):
    """Write keyframes, trails, orbits and swarm geometry from ``run`` into ``scene``."""
    warnings = list(warnings or (meta or {}).get("warnings") or [])
    if sync_objects or bodies is None or swarms is None or slot_of is None:
        bodies, swarms, slot_of = _sync_objects_for_run(scene, run, meta=meta)

    core.assign_swarm_offsets(run.spec)
    s = scene.rebound
    r = run
    F_total = r.F
    k_lo, k_hi, F_anim, frame_step, frames = compute_retime_mapping(scene, r)
    K_win = k_hi - k_lo
    f0 = int(scene.frame_start)
    scene.frame_end = f0 + max(1, F_anim) - 1
    scene["rb_sim_t0"] = float(r.times[k_lo]) if len(r.times) else 0.0
    scene["rb_sim_t1"] = float(r.times[k_hi - 1]) if len(r.times) else 0.0
    scene["rb_frame_start"] = int(f0)
    scene["rb_frame_end"] = int(scene.frame_end)

    ls = scene_io.length_scale(scene)
    Q_full, c_full, d = scene_io.transforms_for(scene, r, slot_of, k=F_total)
    Q = Q_full[k_lo:k_hi]
    c = c_full[k_lo:k_hi]
    out = core.apply_frame(r.bpos[k_lo:k_hi], Q, c, d)   # (K_win, B, 3) simulation units
    view_layer = bpy.context.view_layer

    # ---- bodies
    for b, obj in enumerate(bodies):
        if b >= r.B:
            break
        rb = obj.rebound
        act_name = f"RB {obj.name}"
        anim.clear_rebound_action(obj)
        loc = core.ffill(out[:, b]) * ls
        anim.set_vector_keys(obj, "location", frames, loc, act_name)
        alive = np.isfinite(r.bpos[k_lo:k_hi, b, 0])
        base = np.array(rb.initial_scale if rb.has_initial else obj.scale)
        if (not alive.all()) or s.scale_with_radius:
            fac = np.ones(K_win)
            r0 = r.brad[0, b]
            if s.scale_with_radius and r0 > 0:
                fac = _finite(core.ffill(r.brad[k_lo:k_hi, b][:, None])[:, 0] / r0, 1.0)
            fac = np.where(alive, fac, 0.0)
            anim.set_vector_keys(obj, "scale", frames, fac[:, None] * base[None, :], act_name)
        else:
            obj.scale = tuple(base)
        if s.bake_elements:
            el = core.body_elements(r, b)
            if el is not None:
                a, e, inc = el
                for key, vals in (("rb_a", a[k_lo:k_hi]), ("rb_e", e[k_lo:k_hi]),
                                  ("rb_inc", np.degrees(inc[k_lo:k_hi]))):
                    vals = _finite(core.ffill(vals[:, None])[:, 0])
                    obj[key] = float(vals[0])
                    anim.set_keys(obj, f'["{key}"]', 0, frames, vals, act_name)
        else:
            for key in ELEMENT_KEYS:
                if key in obj:
                    del obj[key]

    # ---- scene diagnostics (energy & parallel split consistency)
    rel_full = r.relative_energy_error()
    rel = rel_full[k_lo:k_hi]
    anim.remove_fcurves(scene, '["rb_')
    if s.bake_energy and np.isfinite(rel).any():
        ok = np.isfinite(rel)
        scene["rb_energy_error"] = float(rel[ok][0])
        anim.set_keys(scene, '["rb_energy_error"]', 0, frames[ok], rel[ok], "RB Scene")
    elif "rb_energy_error" in scene:
        del scene["rb_energy_error"]

    split = getattr(r, "split", None)
    if split is not None and split.n_chunks > 1:
        s.split_error = float(max(split.max_rel_body, split.max_rel_probe))
        drift = _finite(split.body_drift[k_lo:k_hi])
        scene["rb_split_error"] = float(drift[0]) if drift.size else 0.0
        anim.set_keys(scene, '["rb_split_error"]', 0, frames, drift, "RB Scene")
    else:
        s.split_error = -1.0
        if "rb_split_error" in scene:
            del scene["rb_split_error"]

    # ---- trails & orbits
    remove_generated(scene, ("trail", "orbit", "orbit_driver"))
    if s.trails:
        ng = nodes.trail_group()
        coll = ensure_collection(scene, TRAILS_COLL)
        for b, obj in enumerate(bodies):
            if b >= r.B or not obj.rebound.show_trail or K_win < 2:
                continue
            pts = core.ffill(out[:, b]) * ls
            name = f"{obj.name} Trail"
            cu = bpy.data.curves.new(name, "CURVE")
            cu.dimensions = "3D"
            sp = cu.splines.new("POLY")
            sp.points.add(K_win - len(sp.points))
            co4 = np.ones((K_win, 4), dtype=np.float32)
            co4[:, :3] = pts
            sp.points.foreach_set("co", co4.ravel())
            tob = bpy.data.objects.new(name, cu)
            coll.objects.link(tob)
            tob[GEN_KEY] = "trail"
            tob["rebound_source"] = obj.name
            mat = nodes.emission_material(nodes.material_name("Trail", obj.name),
                                          tuple(obj.rebound.color), s.trail_emission, fade=True)
            mod = nodes.ensure_gn_modifier(tob, ng)
            nodes.set_modifier_inputs(mod, {
                "Frame Start": f0, "Frame Step": float(frame_step),
                "Length": s.trail_length, "Radius": s.trail_thickness,
                "Fade": s.trail_fade, "Taper": s.trail_taper, "Material": mat})

    if s.orbits:
        ng = nodes.orbit_group()
        coll = ensure_collection(scene, ORBITS_COLL)
        for b, obj in enumerate(bodies):
            if b >= r.B or not obj.rebound.show_orbit:
                continue
            el = core.body_ellipses(r, b)
            if el is None:
                continue
            centre, R, scale, bound = (x[k_lo:k_hi] for x in el)
            if not bound.any():
                continue
            centre_o = core.apply_frame(_finite(centre), Q, c, d) * ls
            Rq = np.einsum("fij,fjk->fik", Q, _finite(R))
            quats = core.make_quats_continuous(core.mat_to_quat(Rq))
            sc = _finite(scale) * ls
            sc[:, 2] = 1.0
            sc[~bound] = 0.0
            drv = bpy.data.objects.new(f"{obj.name} Orbit Driver", None)
            coll.objects.link(drv)
            drv[GEN_KEY] = "orbit_driver"
            drv["rebound_source"] = obj.name
            drv.empty_display_type = "ARROWS"
            drv.empty_display_size = 0.2
            drv.rotation_mode = "QUATERNION"
            act_name = f"RB {drv.name}"
            anim.set_vector_keys(drv, "location", frames, centre_o, act_name)
            anim.set_vector_keys(drv, "rotation_quaternion", frames, quats, act_name)
            anim.set_vector_keys(drv, "scale", frames, sc, act_name)
            _hide(drv, view_layer)
            me = bpy.data.meshes.new(f"{obj.name} Orbit")
            oob = bpy.data.objects.new(f"{obj.name} Orbit", me)
            coll.objects.link(oob)
            oob[GEN_KEY] = "orbit"
            oob["rebound_source"] = obj.name
            mat = nodes.emission_material(nodes.material_name("Orbit", obj.name),
                                          tuple(obj.rebound.color), 0.5 * s.trail_emission)
            mod = nodes.ensure_gn_modifier(oob, ng)
            nodes.set_modifier_inputs(mod, {
                "Driver": drv, "Thickness": s.orbit_thickness,
                "Resolution": s.orbit_resolution, "Material": mat})

    # ---- swarms
    if r.spos is not None:
        for sw, obj in zip(r.spec.swarms, swarms):
            if obj.type != "MESH":
                warnings.append(f"Swarm '{obj.name}' must be a mesh object; skipped.")
                continue
            n = int(sw.count)
            P = r.spos[k_lo:k_hi, sw.first:sw.first + n].astype(np.float64)
            P = core.apply_frame(P, Q, c, d) * ls
            if F_anim != K_win:
                P = core.interpolate_swarm_frames(P, F_anim)
            valid = np.isfinite(P[..., 0])
            nv = int(valid.sum())
            if nv > MAX_SWARM_VERTICES:
                warnings.append(f"Swarm '{obj.name}' too large ({nv} points); skipped.")
                continue
            co = P[valid].astype(np.float32)
            fr_ids = np.broadcast_to(np.arange(F_anim, dtype=np.int32)[:, None], (F_anim, n))[valid]
            pid = np.broadcast_to(np.arange(n, dtype=np.int32)[None, :], (F_anim, n))[valid]
            me = bpy.data.meshes.new(f"{obj.name} Swarm Data")
            me.vertices.add(nv)
            me.vertices.foreach_set("co", co.ravel())
            at = me.attributes.new("rb_frame", "INT", "POINT")
            at.data.foreach_set("value", np.ascontiguousarray(fr_ids, dtype=np.int32))
            at = me.attributes.new("rb_id", "INT", "POINT")
            at.data.foreach_set("value", np.ascontiguousarray(pid, dtype=np.int32))
            me.update()
            old = obj.data
            obj.data = me
            if old is not None and old.users == 0:
                bpy.data.meshes.remove(old)
            obj.matrix_world = Matrix.Identity(4)
            mat = nodes.emission_material(nodes.material_name("Swarm", obj.name),
                                          tuple(obj.rebound.color), 2.0)
            mod = nodes.ensure_gn_modifier(obj, nodes.swarm_group())
            nodes.set_modifier_inputs(mod, {
                "Frame Start": f0, "Particle Size": obj.rebound.particle_size,
                "Tail Frames": obj.rebound.tail, "Material": mat})

    # ---- status
    final = rel_full[np.isfinite(rel_full)]
    s.energy_error = float(final[-1]) if final.size else -1.0
    s.baked = True
    lt = scene_io.unit_labels(scene)[1]
    if wall is None:
        wall = getattr(r, "wall", 0.0)
    n_chunks = getattr(r, "n_chunks", 1)
    mode_tag = f" ({n_chunks} workers)" if n_chunks > 1 else ""
    t_end = float(r.times[-1]) if len(r.times) else 0.0
    retime_tag = f" → {F_anim} frames" if (F_anim != F_total or k_lo > 0 or k_hi < F_total) else ""
    info = (
        f"Baked {F_total} snapshots{retime_tag} "
        f"(t = {t_end:.6g} {lt}) in {wall:.2f} s{mode_tag}, dt = {r.dt:.4g}"
    )
    if final.size:
        info += f", |ΔE/E₀| = {s.energy_error:.2e}"
    if s.split_error >= 0:
        info += f", split Δr/r = {s.split_error:.2e}"
    s.bake_info = info
    msgs = warnings + list(r.messages)
    s.bake_messages = "\n".join(msgs[:12])
    scene.frame_set(scene.frame_current)
    return info


def load_bake_entry(scene, index: int) -> str:
    """Activate and apply ``scene.rebound.bakes[index]`` in the viewport."""
    s = scene.rebound
    if not (0 <= index < len(s.bakes)):
        raise core.SetupError("No bake selected.")
    entry = s.bakes[index]
    run, meta = get_entry_run(entry)
    bodies, swarms, slot_of = _sync_objects_for_run(scene, run, meta=meta)
    info = apply_run_to_scene(
        scene, run, bodies=bodies, swarms=swarms, slot_of=slot_of,
        meta=meta, sync_objects=False,
    )
    LAST.update(
        sig=spec_signature(run.spec), runner=run,
        bodies=[o.name for o in bodies], swarms=[o.name for o in swarms],
        meta=meta,
    )
    _set_active_bake_index(scene, index)
    return f"Loaded '{entry.name}': {info}"


def restore_setup_from_entry(scene, index: int) -> str:
    """Restore initial conditions and simulation settings from ``scene.rebound.bakes[index]``
    and switch the viewport to unbaked edit mode so the user can iterate."""
    s = scene.rebound
    if not (0 <= index < len(s.bakes)):
        raise core.SetupError("No bake selected.")
    entry = s.bakes[index]
    run, meta = get_entry_run(entry)
    _sync_objects_for_run(scene, run, meta=meta)
    spec = run.spec
    if spec.integrator:
        try:
            s.integrator = spec.integrator
        except TypeError:
            pass
    if spec.dt_mode in {"AUTO", "MANUAL"}:
        s.dt_mode = spec.dt_mode
    if spec.dt > 0:
        s.dt = float(spec.dt)
    if spec.dt_fraction > 0:
        s.dt_fraction = float(spec.dt_fraction)
    if spec.exact in {"AUTO", "EXACT", "STEP"}:
        s.exact_mode = spec.exact
    s.move_to_com = bool(spec.move_to_com)
    if spec.collision in {"none", "direct", "line"}:
        s.collision = spec.collision
    if spec.collision_resolve in {"merge", "hardsphere", "halt"}:
        s.collision_resolve = spec.collision_resolve

    clear_bake(scene)
    from . import presets
    presets._place(scene)
    return f"Restored initial setup from '{entry.name}'. Ready to edit and bake a new iteration."


# ---------------------------------------------------------------------------
# Clearing (unbakes viewport while preserving Bake History)
# ---------------------------------------------------------------------------

def clear_bake(scene):
    """Remove active viewport keyframes, trails, orbits and swarm vertices and
    restore bodies to their initial positions (keeps saved bakes in Bake History)."""
    s = scene.rebound
    objs = {o.name: o for o in scene_io.body_objects(scene)}
    for o in scene.objects:
        if o.rebound.has_initial:
            objs[o.name] = o
    for obj in objs.values():
        rb = obj.rebound
        anim.clear_rebound_action(obj)
        for key in ELEMENT_KEYS:
            if key in obj:
                del obj[key]
        if rb.has_initial:
            obj.location = rb.initial_location
            obj.scale = rb.initial_scale
            rb.has_initial = False
    remove_generated(scene)
    for obj in scene_io.swarm_objects(scene):
        if obj.type == "MESH" and len(obj.data.vertices):
            old = obj.data
            obj.data = bpy.data.meshes.new(f"{obj.name} Swarm Data")
            if old.users == 0:
                bpy.data.meshes.remove(old)
    anim.remove_fcurves(scene, '["rb_')
    for prop_key in ("rb_energy_error", "rb_split_error", "rb_sim_t0", "rb_sim_t1",
                     "rb_frame_start", "rb_frame_end"):
        if prop_key in scene:
            del scene[prop_key]
    anim.clear_rebound_action(scene)
    s.baked = False
    s.bake_info = ""
    s.bake_messages = ""
    s.energy_error = -1.0
    s.split_error = -1.0
    for item in s.bakes:
        item.is_loaded = False
    LAST.update(sig=None, runner=None, meta=None)
    from . import preview
    preview.mark_dirty()


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def _redraw(context):
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            if area.type in {"VIEW_3D", "PROPERTIES"}:
                area.tag_redraw()


class REBOUND_OT_bake(bpy.types.Operator):
    """Run the REBOUND simulation over the frame range, bake to keyframes, and store in Bake History (Esc cancels)"""
    bl_idname = "rebound.bake"
    bl_label = "Bake Simulation"
    bl_options = {"REGISTER"}

    _job = None
    _timer = None

    def execute(self, context):  # synchronous (scripts, background mode)
        try:
            job = BakeJob(context.scene)
            job.run_sync()
            info = job.finish()
        except core.SetupError as ex:
            self.report({"ERROR"}, str(ex))
            return {"CANCELLED"}
        except Exception as ex:
            self.report({"ERROR"}, f"Simulation failed: {ex}")
            return {"CANCELLED"}
        self.report({"INFO"}, info)
        return {"FINISHED"}

    def invoke(self, context, event):
        s = context.scene.rebound
        if s.bake_running:
            self.report({"WARNING"}, "A bake is already running.")
            return {"CANCELLED"}
        try:
            self._job = BakeJob(context.scene)
            self._job.start_async()
        except core.SetupError as ex:
            self.report({"ERROR"}, str(ex))
            return {"CANCELLED"}
        except Exception as ex:
            self.report({"ERROR"}, f"Could not start bake: {ex}")
            return {"CANCELLED"}
        s.bake_running = True
        s.bake_progress = 0.0
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _stop(self, context):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        try:
            self._job.scene.rebound.bake_running = False
        except ReferenceError:
            pass
        _redraw(context)

    def modal(self, context, event):
        job = self._job
        if event.type == "ESC" and event.value == "PRESS":
            self._stop(context)
            job.abort()
            self.report({"WARNING"}, "Bake cancelled.")
            return {"CANCELLED"}
        if event.type == "TIMER":
            try:
                p = job.poll()
            except Exception as ex:
                self._stop(context)
                job.abort()
                self.report({"ERROR"}, f"Simulation failed: {ex}")
                return {"CANCELLED"}
            job.scene.rebound.bake_progress = p
            _redraw(context)
            if job.done:
                self._stop(context)
                try:
                    info = job.finish()
                except Exception as ex:
                    job.abort()
                    self.report({"ERROR"}, f"Writing results failed: {ex}")
                    return {"CANCELLED"}
                self.report({"INFO"}, info)
                return {"FINISHED"}
        return {"PASS_THROUGH"}


class REBOUND_OT_clear_bake(bpy.types.Operator):
    """Unbake the viewport and restore initial placement for editing (saved bakes remain in Bake History)"""
    bl_idname = "rebound.clear_bake"
    bl_label = "Unbake Viewport (Edit Setup)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        clear_bake(context.scene)
        return {"FINISHED"}


class REBOUND_OT_reapply_bake(bpy.types.Operator):
    """Re-apply Reference Frame, Length Scale, Trails and Orbits from the active bake without re-running REBOUND"""
    bl_idname = "rebound.reapply_bake"
    bl_label = "Update Visualization"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        s = context.scene.rebound
        return bool(LAST.get("runner") is not None or (0 <= s.active_bake_index < len(s.bakes)))

    def execute(self, context):
        scene = context.scene
        s = scene.rebound
        run = LAST.get("runner")
        meta = LAST.get("meta")
        if run is None and 0 <= s.active_bake_index < len(s.bakes):
            try:
                run, meta = get_entry_run(s.bakes[s.active_bake_index])
            except Exception as ex:
                self.report({"ERROR"}, str(ex))
                return {"CANCELLED"}
        if run is None:
            self.report({"ERROR"}, "No cached bake available.")
            return {"CANCELLED"}
        try:
            bodies = scene_io.body_objects(scene)
            swarms = scene_io.swarm_objects(scene)
            slot_of = {o.name: i for i, o in enumerate(bodies)}
            if len(bodies) != run.B:
                bodies, swarms, slot_of = _sync_objects_for_run(scene, run, meta=meta)
            info = apply_run_to_scene(
                scene, run, bodies=bodies, swarms=swarms, slot_of=slot_of, meta=meta
            )
        except Exception as ex:
            self.report({"ERROR"}, f"Could not update visualization: {ex}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Updated visualization ({info})")
        return {"FINISHED"}


class REBOUND_OT_load_bake(bpy.types.Operator):
    """Load and preview this stored simulation bake in the viewport"""
    bl_idname = "rebound.load_bake"
    bl_label = "Preview Selected Bake"
    bl_options = {"REGISTER", "UNDO"}

    index: bpy.props.IntProperty(name="Index", default=-1, options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return len(context.scene.rebound.bakes) > 0

    def execute(self, context):
        scene = context.scene
        s = scene.rebound
        idx = self.index if self.index >= 0 else s.active_bake_index
        self.index = -1
        try:
            msg = load_bake_entry(scene, idx)
        except Exception as ex:
            self.report({"ERROR"}, str(ex))
            return {"CANCELLED"}
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class REBOUND_OT_restore_bake_setup(bpy.types.Operator):
    """Restore initial conditions and simulation settings from the selected bake and enter edit mode to iterate"""
    bl_idname = "rebound.restore_bake_setup"
    bl_label = "Restore Setup to Iterate"
    bl_options = {"REGISTER", "UNDO"}

    index: bpy.props.IntProperty(name="Index", default=-1, options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        s = context.scene.rebound
        return 0 <= s.active_bake_index < len(s.bakes)

    def execute(self, context):
        scene = context.scene
        s = scene.rebound
        idx = self.index if self.index >= 0 else s.active_bake_index
        self.index = -1
        try:
            msg = restore_setup_from_entry(scene, idx)
        except Exception as ex:
            self.report({"ERROR"}, str(ex))
            return {"CANCELLED"}
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class REBOUND_OT_delete_bake(bpy.types.Operator):
    """Delete the selected bake from Bake History (and remove its .npz file from the Bake Folder)"""
    bl_idname = "rebound.delete_bake"
    bl_label = "Delete Bake"
    bl_options = {"REGISTER", "UNDO"}

    delete_file: bpy.props.BoolProperty(name="Delete .npz File from Disk", default=True)

    @classmethod
    def poll(cls, context):
        s = context.scene.rebound
        return 0 <= s.active_bake_index < len(s.bakes)

    def execute(self, context):
        scene = context.scene
        s = scene.rebound
        idx = s.active_bake_index
        if not (0 <= idx < len(s.bakes)):
            return {"CANCELLED"}
        entry = s.bakes[idx]
        uid = entry.uid
        abs_path = _abspath_bake(entry.filepath)
        _RUN_CACHE.pop(uid, None)
        if self.delete_file and abs_path and os.path.isfile(abs_path):
            try:
                os.remove(abs_path)
            except OSError:
                pass
        s.bakes.remove(idx)
        new_idx = min(idx, len(s.bakes) - 1)
        _set_active_bake_index(scene, new_idx)
        return {"FINISHED"}


class REBOUND_OT_clear_bake_history(bpy.types.Operator):
    """Remove all stored entries from the Bake History list (and delete their .npz files in the Bake Folder)"""
    bl_idname = "rebound.clear_bake_history"
    bl_label = "Clear Bake History"
    bl_options = {"REGISTER", "UNDO"}

    delete_files: bpy.props.BoolProperty(name="Delete .npz Files from Disk", default=True)

    @classmethod
    def poll(cls, context):
        return len(context.scene.rebound.bakes) > 0

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        scene = context.scene
        s = scene.rebound
        for entry in list(s.bakes):
            _RUN_CACHE.pop(entry.uid, None)
            if self.delete_files:
                abs_path = _abspath_bake(entry.filepath)
                if abs_path and os.path.isfile(abs_path):
                    try:
                        os.remove(abs_path)
                    except OSError:
                        pass
        s.bakes.clear()
        _set_active_bake_index(scene, -1)
        self.report({"INFO"}, "Cleared bake history.")
        return {"FINISHED"}


class REBOUND_OT_scan_bakes(bpy.types.Operator):
    """Scan the Bake Folder for .npz simulation files and add any new ones to Bake History"""
    bl_idname = "rebound.scan_bakes"
    bl_label = "Scan Bake Folder"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        s = scene.rebound
        bdir = resolve_bake_dir(scene, create=False)
        if not os.path.isdir(bdir):
            self.report({"INFO"}, f"Bake folder does not exist yet: {bdir}")
            return {"FINISHED"}
        known_paths = {os.path.normcase(_abspath_bake(b.filepath)) for b in s.bakes if b.filepath}
        added = 0
        for fname in sorted(os.listdir(bdir)):
            if not fname.lower().endswith(".npz"):
                continue
            fpath = os.path.join(bdir, fname)
            if os.path.normcase(os.path.abspath(fpath)) in known_paths:
                continue
            try:
                run, meta = core.load_run_npz(fpath)
            except Exception:
                continue
            label = meta.get("label") or os.path.splitext(fname)[0]
            register_bake_entry(
                scene, run, meta, label=label,
                filepath=_store_filepath(fpath), save_to_disk=False,
            )
            added += 1
        self.report({"INFO"}, f"Found {added} new bake(s) in {bdir}.")
        return {"FINISHED"}


class REBOUND_OT_import_bake(bpy.types.Operator, ImportHelper):
    """Import an external simulation (.npz from cli.py or native REBOUND .bin SimulationArchive)"""
    bl_idname = "rebound.import_bake"
    bl_label = "Import External Bake (.npz / .bin)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".npz"
    filter_glob: bpy.props.StringProperty(default="*.npz;*.bin", options={"HIDDEN"})
    copy_to_project: bpy.props.BoolProperty(
        name="Copy into Project Bake Folder", default=True,
        description="Save a copy of the imported simulation in the scene's Bake Folder"
    )
    max_frames: bpy.props.IntProperty(
        name="Max Archive Snapshots (0 = all)", default=0, min=0,
        description="When importing a REBOUND .bin SimulationArchive, optionally subsample to at most this many frames"
    )

    def execute(self, context):
        scene = context.scene
        s = scene.rebound
        path = os.path.abspath(bpy.path.abspath(self.filepath))
        if not os.path.isfile(path):
            self.report({"ERROR"}, f"File not found: {path}")
            return {"CANCELLED"}
        try:
            if path.lower().endswith(".bin"):
                rb = deps.get_rebound()
                if rb is None:
                    self.report({"ERROR"}, "REBOUND must be installed to read .bin SimulationArchive files.")
                    return {"CANCELLED"}
                run, meta = core.load_simulation_archive(
                    rb, path, units=s.units, max_frames=int(self.max_frames)
                )
            else:
                run, meta = core.load_run_npz(path)
        except Exception as ex:
            self.report({"ERROR"}, f"Could not import simulation: {ex}")
            return {"CANCELLED"}

        label = meta.get("label") or os.path.splitext(os.path.basename(path))[0]
        stored_path = _store_filepath(path) if path.lower().endswith(".npz") and not self.copy_to_project else ""
        entry = register_bake_entry(
            scene, run, meta, label=label, filepath=stored_path,
            save_to_disk=self.copy_to_project,
        )
        try:
            msg = load_bake_entry(scene, len(s.bakes) - 1)
        except Exception as ex:
            self.report({"ERROR"}, f"Imported entry '{entry.name}' but failed to apply to scene: {ex}")
            return {"CANCELLED"}
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class REBOUND_OT_export_bake(bpy.types.Operator, ExportHelper):
    """Export the selected bake to a standalone .npz file"""
    bl_idname = "rebound.export_bake"
    bl_label = "Export Bake (.npz)"

    filename_ext = ".npz"
    filter_glob: bpy.props.StringProperty(default="*.npz", options={"HIDDEN"})
    compressed: bpy.props.BoolProperty(name="Compress (.npz)", default=False)

    @classmethod
    def poll(cls, context):
        s = context.scene.rebound
        return bool(LAST.get("runner") is not None or (0 <= s.active_bake_index < len(s.bakes)))

    def execute(self, context):
        scene = context.scene
        s = scene.rebound
        run = LAST.get("runner")
        meta = LAST.get("meta") or {}
        if 0 <= s.active_bake_index < len(s.bakes):
            try:
                run, meta = get_entry_run(s.bakes[s.active_bake_index])
            except Exception:
                pass
        if run is None:
            self.report({"ERROR"}, "No bake available to export.")
            return {"CANCELLED"}
        try:
            core.save_run_npz(self.filepath, run, meta=meta, compressed=self.compressed)
        except Exception as ex:
            self.report({"ERROR"}, f"Export failed: {ex}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported bake to {self.filepath}")
        return {"FINISHED"}


class REBOUND_OT_export_spec(bpy.types.Operator, ExportHelper):
    """Export the current simulation setup to a .json spec file for running outside Blender with cli.py"""
    bl_idname = "rebound.export_spec"
    bl_label = "Export Spec for External Run (.json)"

    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        scene = context.scene
        try:
            spec, bodies, swarms, _slots = scene_io.make_spec(scene, archive=False)
            meta = collect_scene_meta(scene, bodies, swarms)
            cli.save_spec_file(self.filepath, spec, meta=meta)
        except Exception as ex:
            self.report({"ERROR"}, f"Could not export spec: {ex}")
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Exported spec to {self.filepath}. Run externally with: python cli.py \"{os.path.basename(self.filepath)}\" -o result.npz"
        )
        return {"FINISHED"}


classes = (
    REBOUND_OT_bake,
    REBOUND_OT_clear_bake,
    REBOUND_OT_reapply_bake,
    REBOUND_OT_load_bake,
    REBOUND_OT_restore_bake_setup,
    REBOUND_OT_delete_bake,
    REBOUND_OT_clear_bake_history,
    REBOUND_OT_scan_bakes,
    REBOUND_OT_import_bake,
    REBOUND_OT_export_bake,
    REBOUND_OT_export_spec,
)
