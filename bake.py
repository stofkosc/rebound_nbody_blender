"""Baking: run REBOUND and write keyframes, trails, orbits and swarm geometry."""

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import bpy
import numpy as np
from mathutils import Matrix

from . import anim, core, deps, nodes, scene_io

GEN_KEY = "rebound_generated"
TRAILS_COLL = "REBOUND Trails"
ORBITS_COLL = "REBOUND Orbits"
ELEMENT_KEYS = ("rb_a", "rb_e", "rb_inc")
MAX_SWARM_VERTICES = 60_000_000
MIN_PARTICLES_PER_WORKER = 200

# Last finished bake, reused by the CSV export when nothing changed.
LAST = {"sig": None, "runner": None, "bodies": None, "swarms": None}


def spec_signature(spec):
    return repr(dataclasses.asdict(spec))


def _default_workers() -> int:
    cpus = os.cpu_count() or 4
    return max(1, min(16, cpus - 2 if cpus > 4 else cpus))


def choose_worker_count(scene, spec: core.SimSpec) -> int:
    s = scene.rebound
    if not s.parallel_bake:
        return 1
    S_total = sum(max(0, int(sw.count)) for sw in spec.swarms)
    if S_total < 2 * MIN_PARTICLES_PER_WORKER:
        return 1
    req = int(s.parallel_workers)
    target = req if req > 0 else _default_workers()
    max_by_size = max(1, S_total // MIN_PARTICLES_PER_WORKER)
    return max(1, min(target, max_by_size))


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


# ---------------------------------------------------------------------------
# Job (background thread for single-process, subprocess pool for parallel swarms)
# ---------------------------------------------------------------------------

class _ParallelPool:
    """Manages K worker subprocesses running ``worker.py``."""

    def __init__(self, spec: core.SimSpec, n_workers: int, n_probes: int, compute_energy: bool):
        self.spec = spec
        self.specs = core.split_specs(spec, n_workers, n_probes=n_probes)
        self.n_workers = len(self.specs)
        self.tmpdir = tempfile.mkdtemp(prefix="rebound_bake_")
        self.procs = []
        self.out_paths = []
        self.prog_paths = []
        self.err_files = []
        self._merged = None
        self.archive_error = None

        worker_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")
        d_dir = deps.deps_dir()
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            for idx, sp in enumerate(self.specs):
                spec_path = os.path.join(self.tmpdir, f"spec_{idx}.json")
                out_path = os.path.join(self.tmpdir, f"out_{idx}.npz")
                prog_path = os.path.join(self.tmpdir, f"prog_{idx}.txt")
                err_path = os.path.join(self.tmpdir, f"err_{idx}.txt")
                with open(spec_path, "w", encoding="utf-8") as fh:
                    json.dump(core.spec_to_dict(sp), fh)
                cmd = [sys.executable, worker_py, spec_path, out_path, prog_path]
                if d_dir:
                    cmd.append(d_dir)
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

    def wait_and_collect(self, wall: float) -> core.MergedRun:
        if self._merged is not None:
            return self._merged
        for p in self.procs:
            p.wait()
        self.poll_progress()  # raises if any worker failed
        chunks = []
        for idx, opath in enumerate(self.out_paths):
            with np.load(opath, allow_pickle=True) as z:
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
                    "messages": [str(m) for m in z["messages"].tolist()],
                    "spos": np.array(z["spos"], dtype=np.float32) if "spos" in z else None,
                    "ppos": np.array(z["ppos"], dtype=np.float64) if "ppos" in z else None,
                }
                if idx == 0 and "archive_error" in z:
                    ae = str(z["archive_error"])
                    if ae:
                        self.archive_error = ae
                chunks.append(c)
        self._merged = core.merge_chunk_results(self.spec, chunks, wall=wall)
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
                    compute_energy=compute_energy)
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
        info = write_results(self)
        LAST.update(sig=self.sig, runner=self.runner,
                    bodies=[o.name for o in self.bodies], swarms=[o.name for o in self.swarms])
        return info


# ---------------------------------------------------------------------------
# Writing results
# ---------------------------------------------------------------------------

def _finite(a, fill=0.0):
    return np.nan_to_num(a, nan=fill, posinf=fill, neginf=fill)


def write_results(job):
    scene = job.scene
    s = scene.rebound
    r = job.runner
    F = r.F
    f0 = scene.frame_start
    frames = np.arange(f0, f0 + F, dtype=np.float64)
    ls = scene_io.length_scale(scene)
    Q, c, d = scene_io.transforms_for(scene, r, job.slot_of, k=F)
    out = core.apply_frame(r.bpos, Q, c, d)            # (F, B, 3) simulation units
    view_layer = bpy.context.view_layer

    # ---- bodies
    for b, obj in enumerate(job.bodies):
        rb = obj.rebound
        act_name = f"RB {obj.name}"
        anim.clear_rebound_action(obj)
        loc = core.ffill(out[:, b]) * ls
        anim.set_vector_keys(obj, "location", frames, loc, act_name)
        alive = np.isfinite(r.bpos[:, b, 0])
        if (not alive.all()) or s.scale_with_radius:
            fac = np.ones(F)
            r0 = r.brad[0, b]
            if s.scale_with_radius and r0 > 0:
                fac = _finite(core.ffill(r.brad[:, b][:, None])[:, 0] / r0, 1.0)
            fac = np.where(alive, fac, 0.0)
            base = np.array(rb.initial_scale)
            anim.set_vector_keys(obj, "scale", frames, fac[:, None] * base[None, :], act_name)
        if s.bake_elements:
            el = core.body_elements(r, b)
            if el is not None:
                a, e, inc = el
                for key, vals in (("rb_a", a), ("rb_e", e), ("rb_inc", np.degrees(inc))):
                    vals = _finite(core.ffill(vals[:, None])[:, 0])
                    obj[key] = float(vals[0])
                    anim.set_keys(obj, f'["{key}"]', 0, frames, vals, act_name)
        else:
            for key in ELEMENT_KEYS:
                if key in obj:
                    del obj[key]

    # ---- scene diagnostics (energy & parallel split consistency)
    rel = r.relative_energy_error()
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
        drift = _finite(split.body_drift)
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
        for b, obj in enumerate(job.bodies):
            if not obj.rebound.show_trail or F < 2:
                continue
            pts = core.ffill(out[:, b]) * ls
            name = f"{obj.name} Trail"
            cu = bpy.data.curves.new(name, "CURVE")
            cu.dimensions = "3D"
            sp = cu.splines.new("POLY")
            sp.points.add(F - len(sp.points))
            co4 = np.ones((F, 4), dtype=np.float32)
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
                "Frame Start": f0, "Length": s.trail_length, "Radius": s.trail_thickness,
                "Fade": s.trail_fade, "Taper": s.trail_taper, "Material": mat})

    if s.orbits:
        ng = nodes.orbit_group()
        coll = ensure_collection(scene, ORBITS_COLL)
        for b, obj in enumerate(job.bodies):
            if not obj.rebound.show_orbit:
                continue
            el = core.body_ellipses(r, b)
            if el is None:
                continue
            centre, R, scale, bound = el
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
    for sw, obj in zip(job.spec.swarms, job.swarms):
        if obj.type != "MESH":
            job.warnings.append(f"Swarm '{obj.name}' must be a mesh object; skipped.")
            continue
        n = int(sw.count)
        P = r.spos[:, sw.first:sw.first + n].astype(np.float64)
        P = core.apply_frame(P, Q, c, d) * ls
        valid = np.isfinite(P[..., 0])
        nv = int(valid.sum())
        if nv > MAX_SWARM_VERTICES:
            job.warnings.append(f"Swarm '{obj.name}' too large ({nv} points); skipped.")
            continue
        co = P[valid].astype(np.float32)
        fr_ids = np.broadcast_to(np.arange(F, dtype=np.int32)[:, None], (F, n))[valid]
        pid = np.broadcast_to(np.arange(n, dtype=np.int32)[None, :], (F, n))[valid]
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
    final = rel[np.isfinite(rel)]
    s.energy_error = float(final[-1]) if final.size else -1.0
    s.baked = True
    lt = scene_io.unit_labels(scene)[1]
    wall = time.perf_counter() - job.t0
    n_chunks = getattr(r, "n_chunks", 1)
    mode_tag = f" ({n_chunks} workers)" if n_chunks > 1 else ""
    info = f"Baked {F} frames (t = {r.times[-1]:.6g} {lt}) in {wall:.2f} s{mode_tag}, dt = {r.dt:.4g}"
    if final.size:
        info += f", |ΔE/E₀| = {s.energy_error:.2e}"
    if s.split_error >= 0:
        info += f", split Δr/r = {s.split_error:.2e}"
    s.bake_info = info
    msgs = job.warnings + r.messages
    s.bake_messages = "\n".join(msgs[:12])
    scene.frame_set(scene.frame_current)
    return info


# ---------------------------------------------------------------------------
# Clearing
# ---------------------------------------------------------------------------

def clear_bake(scene):
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
    for prop_key in ("rb_energy_error", "rb_split_error"):
        if prop_key in scene:
            del scene[prop_key]
    anim.clear_rebound_action(scene)
    s.baked = False
    s.bake_info = ""
    s.bake_messages = ""
    s.energy_error = -1.0
    s.split_error = -1.0
    LAST.update(sig=None, runner=None)
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
    """Run the REBOUND simulation over the frame range and bake it to keyframes (Esc cancels)"""
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
    """Remove baked keyframes, trails, orbits and swarm data and restore initial placement"""
    bl_idname = "rebound.clear_bake"
    bl_label = "Clear Bake"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        clear_bake(context.scene)
        return {"FINISHED"}


classes = (REBOUND_OT_bake, REBOUND_OT_clear_bake)
