"""Smaller operators: adding bodies/swarms, helpers, export."""

import csv
import math

import bpy
import numpy as np
from bpy_extras.io_utils import ExportHelper

from . import bake, core, deps, preview, presets, scene_io


class REBOUND_OT_add_body(bpy.types.Operator):
    """Add a new REBOUND body at the 3D cursor (first one becomes a star)"""
    bl_idname = "rebound.add_body"
    bl_label = "Add Body"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        bodies = scene_io.body_objects(scene)
        n = len(bodies)
        col = presets.PALETTE[n % len(presets.PALETTE)]
        if n == 0:
            obj = presets.create_body(scene, "Star", 1.0, (1.0, 0.8, 0.35, 1), 0.2,
                                      location=scene.cursor.location, emission=8.0, show_trail=False)
        else:
            star = scene_io.most_massive(scene)
            obj = presets.create_body(scene, f"Planet {n}", 1e-3, col, 0.06, ic_mode="ORBIT",
                                      primary=star, a=0.6 + 0.5 * n)
            presets._place(scene)
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return {"FINISHED"}


class REBOUND_OT_add_swarm(bpy.types.Operator):
    """Add a swarm of massless test particles orbiting the most massive body"""
    bl_idname = "rebound.add_swarm"
    bl_label = "Add Swarm"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        star = scene_io.most_massive(scene)
        if star is None:
            self.report({"ERROR"}, "Add a massive body first.")
            return {"CANCELLED"}
        obj = presets.create_swarm(scene, "Swarm", star, (0.85, 0.85, 1.0, 1))
        context.view_layer.objects.active = obj
        preview.mark_dirty()
        return {"FINISHED"}


class REBOUND_OT_make_bodies(bpy.types.Operator):
    """Turn the selected objects into REBOUND bodies (position = current location)"""
    bl_idname = "rebound.make_bodies"
    bl_label = "Make Selected Bodies"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.selected_objects)

    def execute(self, context):
        n = 0
        for i, o in enumerate(context.selected_objects):
            if o.get("rebound_generated"):
                continue
            if not o.rebound.enabled:
                o.rebound.enabled = True
                o.rebound.color = presets.PALETTE[i % len(presets.PALETTE)]
                n += 1
        self.report({"INFO"}, f"{n} new bodies.")
        return {"FINISHED"}


class REBOUND_OT_remove_body(bpy.types.Operator):
    """Exclude the active object from the simulation (the object is kept)"""
    bl_idname = "rebound.remove_body"
    bl_label = "Remove from Simulation"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.rebound.enabled

    def execute(self, context):
        context.object.rebound.enabled = False
        return {"FINISHED"}


def _quick_runner(scene, include_swarms=False):
    rb = deps.get_rebound()
    if rb is None:
        raise core.SetupError("REBOUND is not installed.")
    spec, bodies, swarms, slots = scene_io.make_spec(scene, include_swarms=include_swarms, frames=1)
    return core.Runner(rb, spec, compute_energy=False), bodies, slots


class REBOUND_OT_fit_timeline(bpy.types.Operator):
    """Set 'Time per Frame' so the frame range covers N orbits of the active body"""
    bl_idname = "rebound.fit_timeline"
    bl_label = "Fit Timeline to Orbits"
    bl_options = {"REGISTER", "UNDO"}

    orbits: bpy.props.FloatProperty(name="Orbits", default=5.0, min=0.01)

    @classmethod
    def poll(cls, context):
        o = context.object
        return o is not None and o.rebound.enabled and o.rebound.kind == "BODY"

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        scene = context.scene
        try:
            runner, bodies, slots = _quick_runner(scene)
        except core.SetupError as ex:
            self.report({"ERROR"}, str(ex))
            return {"CANCELLED"}
        b = slots.get(context.object.name)
        refs = runner.refs.get(b)
        if not refs:
            self.report({"ERROR"}, "This body has no primary to orbit (set one, or pick a lighter body).")
            return {"CANCELLED"}
        m = runner.raw_bmass
        M = m[refs].sum()
        rp = (runner.raw_bpos[refs] * m[refs, None]).sum(0) / M
        rv = (runner.raw_bvel[refs] * m[refs, None]).sum(0) / M
        mu = runner.G * (M + m[b])
        a, e, *_ = core.two_body_elements(runner.raw_bpos[b] - rp, runner.raw_bvel[b] - rv, mu)
        if not (a > 0 and e < 1):
            self.report({"ERROR"}, "The orbit is not bound.")
            return {"CANCELLED"}
        P = 2 * math.pi * math.sqrt(a ** 3 / mu)
        nf = scene_io.n_frames(scene)
        scene.rebound.time_per_frame = self.orbits * P / max(nf - 1, 1)
        self.report({"INFO"}, f"Period {P:.6g}; time per frame set to {scene.rebound.time_per_frame:.6g}.")
        return {"FINISHED"}


class REBOUND_OT_circular_velocity(bpy.types.Operator):
    """Set the velocity for a circular orbit around the primary (or heaviest body), in the XY plane"""
    bl_idname = "rebound.circular_velocity"
    bl_label = "Circular Velocity"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        o = context.object
        return o is not None and o.rebound.enabled and o.rebound.ic_mode == "CARTESIAN"

    def execute(self, context):
        scene = context.scene
        obj = context.object
        try:
            runner, bodies, slots = _quick_runner(scene)
        except core.SetupError as ex:
            self.report({"ERROR"}, str(ex))
            return {"CANCELLED"}
        b = slots[obj.name]
        refs = runner.refs.get(b)
        if not refs:
            self.report({"ERROR"}, "No heavier body or primary to orbit.")
            return {"CANCELLED"}
        m = runner.raw_bmass
        M = m[refs].sum()
        rp = (runner.raw_bpos[refs] * m[refs, None]).sum(0) / M
        rv = (runner.raw_bvel[refs] * m[refs, None]).sum(0) / M
        r = runner.raw_bpos[b] - rp
        d = np.linalg.norm(r)
        if d == 0:
            self.report({"ERROR"}, "The body sits on top of its primary.")
            return {"CANCELLED"}
        t = np.cross([0, 0, 1.0], r)
        if np.linalg.norm(t) < 1e-12 * d:
            t = np.cross([1.0, 0, 0], r)
        t /= np.linalg.norm(t)
        v = rv + math.sqrt(runner.G * (M + m[b]) / d) * t
        obj.rebound.velocity = v
        return {"FINISHED"}


class REBOUND_OT_add_time_label(bpy.types.Operator):
    """Add a text object that shows the simulation time (updates every frame, also in renders)"""
    bl_idname = "rebound.add_time_label"
    bl_label = "Add Time Label"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        cu = bpy.data.curves.new("REBOUND Time", "FONT")
        cu.body = "t = 0"
        cu.size = 0.3
        obj = bpy.data.objects.new("REBOUND Time", cu)
        obj["rebound_time_label"] = True
        obj["rb_format"] = "t = {t:.2f} {unit}"
        obj["rebound_helper"] = True
        context.collection.objects.link(obj)
        obj.location = context.scene.cursor.location
        cam = context.scene.camera
        if cam is not None:
            obj.rotation_euler = cam.rotation_euler
        preview._on_frame(context.scene)
        self.report({"INFO"}, "Edit the custom property 'rb_format' to change the text.")
        return {"FINISHED"}


class REBOUND_OT_export_csv(bpy.types.Operator, ExportHelper):
    """Export the trajectories (simulation units and coordinates) to a CSV file"""
    bl_idname = "rebound.export_csv"
    bl_label = "Export Trajectories (CSV)"
    filename_ext = ".csv"
    filter_glob: bpy.props.StringProperty(default="*.csv", options={"HIDDEN"})
    include_swarms: bpy.props.BoolProperty(name="Include Swarm Particles", default=False)

    def execute(self, context):
        scene = context.scene
        rb = deps.get_rebound()
        if rb is None:
            self.report({"ERROR"}, "REBOUND is not installed.")
            return {"CANCELLED"}
        try:
            spec, bodies, swarms, _slots = scene_io.make_spec(scene)
            sig = bake.spec_signature(spec)
            runner = bake.LAST["runner"] if bake.LAST["sig"] == sig else None
            if runner is None:
                runner = core.Runner(rb, spec).run()
        except core.SetupError as ex:
            self.report({"ERROR"}, str(ex))
            return {"CANCELLED"}
        rel = runner.relative_energy_error()
        with open(self.filepath, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "t", "body", "x", "y", "z", "vx", "vy", "vz", "m", "r",
                        "rel_energy_error"])
            for k in range(runner.F):
                fr = scene.frame_start + k
                for b, obj in enumerate(bodies):
                    p, v = runner.bpos[k, b], runner.bvel[k, b]
                    if not np.isfinite(p[0]):
                        continue
                    w.writerow([fr, f"{runner.times[k]:.17g}", obj.name, *(f"{x:.17g}" for x in p),
                                *(f"{x:.17g}" for x in v), f"{runner.bmass[k, b]:.17g}",
                                f"{runner.brad[k, b]:.17g}",
                                "" if not np.isfinite(rel[k]) else f"{rel[k]:.6e}"])
                if self.include_swarms and runner.S:
                    for sw in runner.spec.swarms:
                        for i in range(sw.count):
                            p = runner.spos[k, sw.first + i]
                            if np.isfinite(p[0]):
                                w.writerow([fr, f"{runner.times[k]:.17g}", f"{sw.name}#{i}",
                                            *(f"{x:.9g}" for x in p), "", "", "", 0, 0, ""])
        self.report({"INFO"}, f"Wrote {self.filepath}")
        return {"FINISHED"}


class REBOUND_OT_refresh_preview(bpy.types.Operator):
    """Recompute the live preview"""
    bl_idname = "rebound.refresh_preview"
    bl_label = "Refresh Preview"

    def execute(self, context):
        preview.mark_dirty()
        presets._place(context.scene)
        return {"FINISHED"}


classes = (REBOUND_OT_add_body, REBOUND_OT_add_swarm, REBOUND_OT_make_bodies,
           REBOUND_OT_remove_body, REBOUND_OT_fit_timeline, REBOUND_OT_circular_velocity,
           REBOUND_OT_add_time_label, REBOUND_OT_export_csv, REBOUND_OT_refresh_preview)
