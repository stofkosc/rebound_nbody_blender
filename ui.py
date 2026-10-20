"""Sidebar panels (3D View > N-panel > REBOUND) and the Object properties panel."""

import math
import textwrap

import bpy

from . import core, deps, preview, scene_io


def _wrap(layout, text, width=48, icon="NONE"):
    col = layout.column(align=True)
    for i, line in enumerate(textwrap.wrap(text, width)):
        col.label(text=line, icon=icon if i == 0 else "BLANK1" if icon != "NONE" else "NONE")


class REBOUND_UL_bodies(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        rb = item.rebound
        row = layout.row(align=True)
        row.prop(rb, "color", text="", emboss=True)
        row.label(text=item.name, icon="POINTCLOUD_DATA" if rb.kind == "SWARM" else "SPHERE")
        if rb.kind == "SWARM":
            row.label(text=f"{rb.count} test particles")
        else:
            row.label(text=f"m = {rb.mass:.4g}")

    def filter_items(self, context, data, propname):
        objs = getattr(data, propname)
        flags = [self.bitflag_filter_item if o.rebound.enabled else 0 for o in objs]
        return flags, []


class REBOUND_UL_bakes(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        op = row.operator(
            "rebound.load_bake",
            text="",
            icon="RADIOBUT_ON" if item.is_loaded else "RADIOBUT_OFF",
            emboss=False,
        )
        op.index = index
        row.prop(item, "name", text="", emboss=False)
        sub = row.row(align=True)
        sub.alignment = "RIGHT"
        tag = f"{item.n_frames}f · {item.n_bodies}b"
        if item.n_swarm > 0:
            tag += f"+{item.n_swarm}p"
        sub.label(text=tag)


class _Base:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "REBOUND"


class REBOUND_PT_main(_Base, bpy.types.Panel):
    bl_label = "REBOUND N-body"

    def draw(self, context):
        layout = self.layout
        s = context.scene.rebound
        if not deps.available():
            deps.draw_status(layout)
            return
        row = layout.row(align=True)
        row.prop(s, "preset", text="")
        row.operator("rebound.load_preset", text="Load", icon="IMPORT")
        col = layout.column(align=True)
        if s.bake_running:
            col.progress(factor=s.bake_progress, text=f"Baking {100 * s.bake_progress:.0f}% (Esc cancels)")
        else:
            row = col.row(align=True)
            row.scale_y = 1.5
            row.operator("rebound.bake", icon="PHYSICS")
            row.operator("rebound.clear_bake", text="", icon="LOOP_BACK")
        if s.bake_info:
            _wrap(layout.box(), s.bake_info, icon="INFO")
        if s.bake_messages:
            box = layout.box()
            for line in s.bake_messages.splitlines()[:6]:
                _wrap(box, line, icon="ERROR")


class REBOUND_PT_bakes(_Base, bpy.types.Panel):
    bl_label = "Bake History & External"
    bl_parent_id = "REBOUND_PT_main"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        s = scene.rebound

        layout.template_list(
            "REBOUND_UL_bakes", "", s, "bakes", s, "active_bake_index", rows=4
        )

        row = layout.row(align=True)
        op = row.operator("rebound.load_bake", text="Preview", icon="PLAY")
        op.index = -1
        op_res = row.operator("rebound.restore_bake_setup", text="Restore Setup", icon="LOOP_BACK")
        op_res.index = -1
        row.operator("rebound.delete_bake", text="", icon="X")

        if 0 <= s.active_bake_index < len(s.bakes):
            entry = s.bakes[s.active_bake_index]
            box = layout.box()
            col = box.column(align=True)
            hdr = f"{entry.integrator.upper()}  |  t = {entry.sim_time:.4g} {entry.unit_time}"
            if entry.created_at:
                hdr += f"  |  {entry.created_at}"
            col.label(text=hdr, icon="TIME")
            if entry.summary:
                _wrap(col, entry.summary, width=46)
            if entry.filepath:
                col.label(text=f"File: {entry.filepath}", icon="FILE")

        col = layout.column(align=True)
        col.prop(s, "save_bakes_to_disk")
        col.prop(s, "auto_preview_bake")
        if s.save_bakes_to_disk:
            row = col.row(align=True)
            row.prop(s, "bake_dir", text="Folder")
            row.operator("rebound.scan_bakes", text="", icon="FILE_REFRESH")

        layout.separator()
        col = layout.column(align=True)
        col.label(text="External / Cluster Simulations:", icon="NETWORK_DRIVE")
        row = col.row(align=True)
        row.operator("rebound.export_spec", text="Export Spec (.json)", icon="EXPORT")
        row.operator("rebound.import_bake", text="Import (.npz / .bin)", icon="IMPORT")
        row = col.row(align=True)
        row.operator("rebound.export_bake", text="Export Bake (.npz)", icon="FILE_TICK")
        row.operator("rebound.clear_bake_history", text="Clear All", icon="TRASH")


class REBOUND_PT_bodies(_Base, bpy.types.Panel):
    bl_label = "Bodies"
    bl_parent_id = "REBOUND_PT_main"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.template_list("REBOUND_UL_bodies", "", scene, "objects", scene.rebound,
                             "active_body_index", rows=4)
        row = layout.row(align=True)
        row.operator("rebound.add_body", icon="ADD")
        row.operator("rebound.add_swarm", icon="POINTCLOUD_DATA")
        row = layout.row(align=True)
        row.operator("rebound.make_bodies", icon="OBJECT_DATA")
        row.operator("rebound.remove_body", text="", icon="X")
        obj = context.object
        if obj is not None and obj.rebound.enabled:
            box = layout.box()
            box.label(text=obj.name, icon="OBJECT_DATA")
            draw_body(box, obj, scene)


def draw_body(layout, obj, scene):
    rb = obj.rebound
    s = scene.rebound
    lbl = scene_io.unit_labels(scene)
    baked = rb.has_initial
    col = layout.column()
    if baked:
        col.operator("rebound.clear_bake", text="Unbake Viewport to Drag in 3D", icon="LOOP_BACK")
    sub = col.column()
    sub.prop(rb, "kind", expand=True)
    if rb.kind == "SWARM":
        sub.prop(rb, "primary")
        sub.prop(rb, "count")
        r = sub.row(align=True)
        r.prop(rb, "a_min")
        r.prop(rb, "a_max")
        sub.prop(rb, "e_max")
        sub.prop(rb, "inc_max")
        r = sub.row(align=True)
        r.prop(rb, "lon_center", text="λ centre")
        r.prop(rb, "lon_width", text="width")
        sub.prop(rb, "seed")
        sub.separator()
        sub.prop(rb, "particle_size")
        sub.prop(rb, "tail")
        sub.prop(rb, "color")
        return
    sub.prop(rb, "mass", text=f"Mass [{lbl[2]}]")
    sub.prop(rb, "radius", text=f"Radius [{lbl[0]}]")
    sub.prop(rb, "ic_mode", expand=True)
    if rb.ic_mode == "CARTESIAN":
        sub.label(text=f"Position = location / {s.length_scale:g}", icon="EMPTY_AXIS")
        if baked:
            sub.prop(rb, "initial_location", text="Initial Location [Blender]")
        sub.prop(rb, "velocity", text=f"Velocity [{lbl[0]}/{lbl[1]}]")
        sub.operator("rebound.circular_velocity", icon="CURVE_BEZCIRCLE")
        sub.prop(rb, "primary", text="Reference")
    else:
        sub.prop(rb, "primary")
        sub.prop(rb, "a", text=f"a [{lbl[0]}]")
        sub.prop(rb, "e")
        sub.prop(rb, "inc")
        sub.prop(rb, "Omega")
        sub.prop(rb, "omega")
        r = sub.row(align=True)
        r.prop(rb, "anomaly_type", text="")
        r.prop(rb, "anomaly", text="")
        P = _period(scene, obj)
        if P:
            sub.label(text=f"Period ≈ {P:.5g} {lbl[1]}  (= {P / s.time_per_frame:.4g} frames)")
    sub.prop(rb, "order")
    col.prop(rb, "color")
    r = col.row(align=True)
    r.prop(rb, "show_trail", toggle=True)
    r.prop(rb, "show_orbit", toggle=True)
    col.operator("rebound.fit_timeline", icon="TIME")


def _period(scene, obj):
    rb = obj.rebound
    if rb.a <= 0 or rb.e >= 1:
        return None
    G = deps.gravitational_constant(scene.rebound.units, scene.rebound.G)
    if not G:
        return None
    if rb.primary is not None:
        M = rb.primary.rebound.mass
    else:  # Jacobi: everything added before (approximation for the label)
        M = sum(o.rebound.mass for o in scene_io.body_objects(scene)
                if (o.rebound.order, o.name) < (rb.order, obj.name))
    mu = G * (M + rb.mass)
    return 2 * math.pi * math.sqrt(rb.a ** 3 / mu) if mu > 0 else None


class REBOUND_PT_simulation(_Base, bpy.types.Panel):
    bl_label = "Simulation"
    bl_parent_id = "REBOUND_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        s = context.scene.rebound
        lbl = scene_io.unit_labels(context.scene)
        layout.prop(s, "units")
        if s.units == "CODE":
            layout.prop(s, "G")
        else:
            G = deps.gravitational_constant(s.units)
            if G:
                layout.label(text=f"G = {G:.6g}")
        layout.prop(s, "integrator")
        doc = deps.integrator_doc(s.integrator)
        if doc:
            _wrap(layout.box(), doc[:400] + ("…" if len(doc) > 400 else ""), width=52)
        layout.prop(s, "dt_mode")
        if s.dt_mode == "AUTO":
            layout.prop(s, "dt_fraction")
        else:
            layout.prop(s, "dt", text=f"dt [{lbl[1]}]")
        layout.prop(s, "move_to_com")
        layout.prop(s, "collision")
        if s.collision != "none":
            layout.prop(s, "collision_resolve")
            layout.label(text="Uses each body's physical radius", icon="INFO")
        layout.separator()
        col = layout.column(heading="Performance")
        col.prop(s, "parallel_bake")
        if s.parallel_bake:
            col.prop(s, "parallel_workers")
            col.prop(s, "parallel_probes")


class REBOUND_PT_time(_Base, bpy.types.Panel):
    bl_label = "Time & Retiming"
    bl_parent_id = "REBOUND_PT_main"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        scene = context.scene
        s = scene.rebound
        lbl = scene_io.unit_labels(scene)

        # ---- 1. Simulation Duration & Sampling (before bake)
        col = layout.column()
        col.prop(s, "duration_mode")
        if s.duration_mode == "ORBITS":
            col.prop(s, "duration_orbits")
            col.prop(s, "duration_body", text="Reference")
        elif s.duration_mode == "TIME":
            col.prop(s, "duration_time")
            col.prop(s, "duration_unit", text="Unit")

        col.prop(s, "sampling_mode")
        if s.sampling_mode == "SNAPSHOTS":
            col.prop(s, "sample_count")
        elif s.sampling_mode == "PER_ORBIT":
            col.prop(s, "samples_per_orbit")
            if s.duration_mode != "ORBITS":
                col.prop(s, "duration_body", text="Reference")
        elif s.sampling_mode == "STEP" or s.duration_mode == "FRAMES":
            col.prop(s, "time_per_frame", text=f"Time / Snapshot [{lbl[1]}]")

        if s.duration_mode == "FRAMES" or s.sampling_mode == "TIMELINE":
            row = col.row(align=True)
            row.prop(scene, "frame_start", text="Timeline")
            row.prop(scene, "frame_end", text="")

        nf, tpf, total_t, P_ref = scene_io.resolve_sim_timing(scene)
        box = layout.box()
        bcol = box.column(align=True)
        orb_tag = f"  (≈ {total_t / P_ref:.4g} orbits)" if (P_ref and P_ref > 0) else ""
        bcol.label(text=f"Duration: {total_t:.5g} {lbl[1]}{orb_tag}", icon="TIME")
        bcol.label(text=f"Sampling: {nf} snapshots  (Δt = {tpf:.4g} {lbl[1]})")
        if P_ref and P_ref > 0:
            ref_lbl = s.duration_body.name if s.duration_body else "shortest orbit"
            bcol.label(text=f"P_ref ({ref_lbl}) ≈ {P_ref:.5g} {lbl[1]}")

        layout.prop(s, "exact_mode")

        # ---- 2. Animation Retiming & Playback Mapping (post-bake)
        layout.separator()
        rcol = layout.column()
        rcol.label(text="Playback Retiming (Post-Bake):", icon="PREVIEW_RANGE")
        rcol.prop(s, "retime_mode", text="Mode")
        if s.retime_mode == "FIT_FRAMES":
            rcol.prop(s, "retime_frames")
        elif s.retime_mode == "FRAMES_PER_ORBIT":
            rcol.prop(s, "retime_frames_per_orbit")
            rcol.prop(s, "duration_body", text="Reference")
        elif s.retime_mode == "FRAMES_PER_TIME":
            rcol.prop(s, "retime_frames_per_time")
            rcol.prop(s, "retime_time_unit", text="Unit")

        rcol.prop(s, "retime_use_window")
        if s.retime_use_window:
            if s.retime_mode != "FRAMES_PER_TIME":
                rcol.prop(s, "retime_time_unit", text="Window Unit")
            u_lbl = core.time_unit_label(s.units, s.retime_time_unit)
            wrow = rcol.row(align=True)
            wrow.prop(s, "retime_start_time", text=f"Start [{u_lbl}]")
            wrow.prop(s, "retime_end_time", text=f"End [{u_lbl}]")

        rcol.prop(scene, "frame_start", text="Start Frame")
        rcol.operator("rebound.reapply_bake", text="Apply Retiming to Active Bake", icon="FILE_REFRESH")


class REBOUND_PT_view(_Base, bpy.types.Panel):
    bl_label = "Visualization"
    bl_parent_id = "REBOUND_PT_main"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        s = context.scene.rebound
        layout.prop(s, "length_scale")
        layout.prop(s, "frame_mode")
        if s.frame_mode != "INERTIAL":
            layout.prop(s, "frame_center")
        if s.frame_mode == "COROTATING":
            layout.prop(s, "frame_secondary")
        layout.prop(s, "anchor_frame")
        layout.separator()
        col = layout.column(heading="Trails")
        col.prop(s, "trails", text="Enable")
        sub = col.column()
        sub.active = s.trails
        sub.prop(s, "trail_length")
        sub.prop(s, "trail_thickness")
        sub.prop(s, "trail_fade")
        sub.prop(s, "trail_taper")
        sub.prop(s, "trail_emission")
        col = layout.column(heading="Orbits")
        col.prop(s, "orbits", text="Osculating Ellipses")
        sub = col.column()
        sub.active = s.orbits
        sub.prop(s, "orbit_thickness")
        sub.prop(s, "orbit_resolution")
        layout.prop(s, "scale_with_radius")
        layout.operator("rebound.reapply_bake", text="Update Visualization", icon="FILE_REFRESH")


class REBOUND_PT_preview(_Base, bpy.types.Panel):
    bl_label = "Live Preview"
    bl_parent_id = "REBOUND_PT_main"

    def draw_header(self, context):
        self.layout.prop(context.scene.rebound, "preview_enabled", text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        s = context.scene.rebound
        col = layout.column()
        col.active = s.preview_enabled
        if s.baked:
            col.label(text="Paused while viewing a bake (Unbake to preview)", icon="PAUSE")
        col.prop(s, "preview_frames")
        col.prop(s, "preview_velocity")
        if s.preview_velocity:
            col.prop(s, "velocity_scale")
        col.prop(s, "preview_swarms")
        col.prop(s, "preview_line_width")
        st = preview.status()
        if st and s.preview_enabled:
            _wrap(col.box(), st)
        col.operator("rebound.refresh_preview", icon="FILE_REFRESH")
        layout.separator()
        layout.prop(s, "hud")
        layout.operator("rebound.add_time_label", icon="FONT_DATA")


class REBOUND_PT_output(_Base, bpy.types.Panel):
    bl_label = "Data & Diagnostics"
    bl_parent_id = "REBOUND_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        s = context.scene.rebound
        col = layout.column(heading="Keyframe")
        col.prop(s, "bake_elements")
        col.prop(s, "bake_energy")
        layout.prop(s, "archive")
        if s.archive:
            layout.prop(s, "archive_path")
        if s.energy_error >= 0:
            layout.label(text=f"Final |ΔE/E₀| = {s.energy_error:.3e}")
        if s.split_error >= 0:
            icon = "CHECKMARK" if s.split_error <= 1e-12 else ("INFO" if s.split_error <= 1e-9 else "ERROR")
            layout.label(text=f"Max Parallel Split Drift Δr/r = {s.split_error:.3e}", icon=icon)
        layout.operator("rebound.export_csv", icon="EXPORT")
        _wrap(layout, "Tip: select a body and open the Graph Editor to plot rb_a, rb_e, "
                      "rb_inc; the scene properties rb_energy_error and rb_split_error hold "
                      "|ΔE/E₀| and parallel split drift Δr/r.", icon="GRAPH")


class REBOUND_PT_horizons(_Base, bpy.types.Panel):
    bl_label = "JPL Horizons"
    bl_parent_id = "REBOUND_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        s = context.scene.rebound
        layout.prop(s, "horizons_names")
        layout.prop(s, "horizons_date")
        layout.operator("rebound.horizons_import", icon="WORLD")
        layout.label(text="Needs internet ('Allow Online Access')", icon="INFO")


class REBOUND_PT_object(bpy.types.Panel):
    bl_label = "REBOUND Body"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        return context.object is not None and not context.object.get("rebound_generated")

    def draw_header(self, context):
        self.layout.prop(context.object.rebound, "enabled", text="")

    def draw(self, context):
        obj = context.object
        if not obj.rebound.enabled:
            self.layout.label(text="Enable to include this object in the simulation")
            return
        draw_body(self.layout, obj, context.scene)


classes = (REBOUND_UL_bodies, REBOUND_UL_bakes, REBOUND_PT_main, REBOUND_PT_bakes,
           REBOUND_PT_bodies, REBOUND_PT_simulation, REBOUND_PT_time, REBOUND_PT_view,
           REBOUND_PT_preview, REBOUND_PT_output, REBOUND_PT_horizons, REBOUND_PT_object)
