"""Live viewport preview (predicted paths, velocity arrows, swarm points), HUD and
time-label handler."""

import bpy
import numpy as np
from bpy.app.handlers import persistent

from . import core, deps, scene_io

_state = {"dirty": True, "runner": None, "slot_of": {}, "scene": None,
          "paths": [], "arrows": [], "points": [], "status": "", "batches": None}
_handles = []
TIME_BUDGET = 0.03


def _scene():
    try:
        sc = bpy.context.scene
        if sc is not None:
            return sc
    except AttributeError:
        pass
    wm = bpy.context.window_manager
    if wm and wm.windows:
        return wm.windows[0].scene
    return bpy.data.scenes[0] if bpy.data.scenes else None


def redraw():
    wm = bpy.context.window_manager
    if wm is None:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def mark_dirty(*_args):
    _state["dirty"] = True
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=0.15)


def _clear():
    _state.update(runner=None, paths=[], arrows=[], points=[], batches=None)


def _playing():
    wm = bpy.context.window_manager
    return any(w.screen.is_animation_playing for w in wm.windows) if wm else False


def place_orbit_bodies(scene, runner, bodies):
    """Move bodies defined by orbital elements to their initial positions."""
    ls = scene_io.length_scale(scene)
    for i, obj in enumerate(bodies):
        if obj.rebound.ic_mode != "ORBIT" or obj.rebound.has_initial:
            continue
        p = runner.raw_bpos[i] * ls
        if np.all(np.isfinite(p)) and (obj.matrix_world.translation - _vec(p)).length > 1e-9:
            if obj.parent is None:
                obj.location = p
            else:
                obj.matrix_world.translation = p


def _vec(p):
    from mathutils import Vector
    return Vector((float(p[0]), float(p[1]), float(p[2])))


def _start(scene):
    s = scene.rebound
    rb = deps.get_rebound()
    nf = scene_io.n_frames(scene)
    if s.preview_frames > 0:
        nf = min(nf, s.preview_frames)
    spec, bodies, swarms, slot_of = scene_io.make_spec(scene, include_swarms=False, frames=nf)
    runner = core.Runner(rb, spec, compute_energy=False)
    _state.update(runner=runner, slot_of=slot_of, scene=scene.name,
                  bodies=[o.name for o in bodies], status="")
    place_orbit_bodies(scene, runner, bodies)
    ls = scene_io.length_scale(scene)

    # velocity arrows (raw initial state = what the user edits)
    arrows = []
    if s.preview_velocity:
        for i, obj in enumerate(bodies):
            p0 = runner.raw_bpos[i] * ls
            p1 = p0 + runner.raw_bvel[i] * s.velocity_scale * ls
            arrows.append((np.array([p0, p1], np.float32), tuple(obj.rebound.color)))
    _state["arrows"] = arrows

    # swarm points from the elements (same RNG stream as the bake)
    points = []
    if s.preview_swarms:
        spec_full = None
        for o in scene_io.swarm_objects(scene):
            if spec_full is None:
                spec_full = scene_io.make_spec(scene, frames=1)[0]
            swspec = next((x for x in spec_full.swarms if x.name == o.name), None)
            if swspec is None or swspec.count <= 0:
                continue
            if swspec.primary is not None:
                pp, pv, pm = (runner.raw_bpos[swspec.primary], runner.raw_bvel[swspec.primary],
                              runner.raw_bmass[swspec.primary])
            else:
                m = runner.raw_bmass
                pm = m.sum()
                pp = (runner.raw_bpos * m[:, None]).sum(0) / pm
                pv = (runner.raw_bvel * m[:, None]).sum(0) / pm
            pts = core.swarm_initial_positions(swspec, runner.G, pp, pv, pm) * ls
            points.append((pts.astype(np.float32), tuple(o.rebound.color)))
    _state["points"] = points


def _update_paths(scene):
    runner = _state["runner"]
    k = runner.k
    if k < 2:
        return
    Q, c, d = scene_io.transforms_for(scene, runner, _state["slot_of"], k=k)
    out = core.apply_frame(runner.bpos[:k], Q, c, d) * scene_io.length_scale(scene)
    paths = []
    for i, name in enumerate(_state["bodies"]):
        obj = scene.objects.get(name)
        if obj is None:
            continue
        pts = out[:, i]
        pts = pts[np.isfinite(pts[:, 0])]
        if len(pts) > 1:
            paths.append((pts.astype(np.float32), tuple(obj.rebound.color)))
    _state["paths"] = paths


def _tick():
    scene = _scene()
    if scene is None:
        return None
    s = scene.rebound
    if not s.preview_enabled or s.baked or s.bake_running or not deps.available():
        if _state["paths"] or _state["arrows"] or _state["points"]:
            _clear()
            redraw()
        return None
    if _playing():
        return 0.5
    if _state["dirty"] or _state.get("scene") != scene.name:
        _state["dirty"] = False
        _clear()
        try:
            _start(scene)
        except core.SetupError as ex:
            _state["status"] = str(ex)
            redraw()
            return None
        except Exception as ex:
            _state["status"] = f"Preview failed: {ex}"
            redraw()
            return None
    runner = _state["runner"]
    if runner is None:
        return None
    if not runner.done:
        try:
            runner.step(TIME_BUDGET)
        except Exception as ex:
            _state["status"] = f"Preview failed: {ex}"
            return None
        _update_paths(scene)
        _state["batches"] = None
        _state["status"] = f"Preview {runner.k}/{runner.F} frames"
        if runner.messages:
            _state["status"] += " – " + runner.messages[-1]
        redraw()
        return 0.0 if not runner.done else None
    return None


def status():
    return _state["status"]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_view():
    scene = bpy.context.scene
    if scene is None:
        return
    s = scene.rebound
    if not s.preview_enabled or s.baked:
        return
    if not (_state["paths"] or _state["arrows"] or _state["points"]):
        return
    import gpu
    from gpu_extras.batch import batch_for_shader
    region = bpy.context.region
    if _state["batches"] is None:
        line = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
        point = gpu.shader.from_builtin("POINT_UNIFORM_COLOR")
        batches = []
        for pts, col in _state["paths"]:
            batches.append((True, line, batch_for_shader(line, "LINE_STRIP", {"pos": pts}), col, 1.0))
        for pts, col in _state["arrows"]:
            batches.append((True, line, batch_for_shader(line, "LINES", {"pos": pts}), col, 1.5))
            batches.append((False, point, batch_for_shader(point, "POINTS", {"pos": pts[1:]}), col, 8.0))
        for pts, col in _state["points"]:
            batches.append((False, point, batch_for_shader(point, "POINTS", {"pos": pts}), col, 3.0))
        _state["batches"] = batches
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("LESS_EQUAL")
    for is_line, shader, batch, col, width in _state["batches"]:
        shader.bind()
        if is_line:
            shader.uniform_float("viewportSize", (region.width, region.height))
            shader.uniform_float("lineWidth", s.preview_line_width * width)
        else:
            gpu.state.point_size_set(width)
        shader.uniform_float("color", (col[0], col[1], col[2], 0.9))
        batch.draw(shader)
    gpu.state.depth_test_set("NONE")
    gpu.state.blend_set("NONE")


def sim_time(scene):
    return (scene.frame_current - scene.frame_start + scene.frame_subframe) * scene.rebound.time_per_frame


def _draw_hud():
    scene = bpy.context.scene
    if scene is None:
        return
    s = scene.rebound
    if not s.hud:
        return
    if not any(o.rebound.enabled for o in scene.objects):
        return
    import blf
    lt = scene_io.unit_labels(scene)[1]
    lines = [f"REBOUND   t = {sim_time(scene):.4g} {lt}   (frame {scene.frame_current})"]
    if s.baked and "rb_energy_error" in scene:
        lines.append(f"|ΔE/E₀| = {scene['rb_energy_error']:.2e}")
    elif s.preview_enabled and _state["status"]:
        lines.append(_state["status"])
    if s.bake_running:
        lines.append(f"Baking… {100 * s.bake_progress:.0f}%  (Esc to cancel)")
    scale = bpy.context.preferences.system.ui_scale
    font = 0
    blf.size(font, 13 * scale)
    blf.enable(font, blf.SHADOW)
    blf.shadow(font, 3, 0, 0, 0, 0.8)
    y = 20 * scale
    for line in reversed(lines):
        blf.position(font, 20 * scale, y, 0)
        blf.color(font, 0.9, 0.9, 0.9, 1.0)
        blf.draw(font, line)
        y += 18 * scale
    blf.disable(font, blf.SHADOW)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

@persistent
def _on_depsgraph(scene, depsgraph):
    s = scene.rebound
    if not s.preview_enabled or s.baked or s.bake_running:
        return
    for u in depsgraph.updates:
        idd = u.id
        if isinstance(idd, bpy.types.Object):
            ob = getattr(idd, "original", idd)
            rb = ob.rebound
            if rb.enabled and rb.kind == "BODY" and rb.ic_mode == "CARTESIAN" and u.is_updated_transform:
                mark_dirty()
                return
        elif isinstance(idd, bpy.types.Collection):
            mark_dirty()
            return


@persistent
def _on_load(*_args):
    _clear()
    _state["status"] = ""
    mark_dirty()


@persistent
def _on_frame(scene, *_args):
    labels = [o for o in scene.objects if o.type == "FONT" and o.get("rebound_time_label")]
    if not labels:
        return
    t = sim_time(scene)
    lt = scene_io.unit_labels(scene)[1]
    for o in labels:
        fmt = o.get("rb_format", "t = {t:.2f} {unit}")
        try:
            o.data.body = fmt.format(t=t, unit=lt, frame=scene.frame_current)
        except (KeyError, ValueError, IndexError):
            o.data.body = f"t = {t:.2f} {lt}"


def register():
    _handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_view, (), "WINDOW", "POST_VIEW"))
    _handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_hud, (), "WINDOW", "POST_PIXEL"))
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)
    bpy.app.handlers.load_post.append(_on_load)
    bpy.app.handlers.frame_change_pre.append(_on_frame)


def unregister():
    for h in _handles:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(h, "WINDOW")
        except ValueError:
            pass
    _handles.clear()
    for lst, fn in ((bpy.app.handlers.depsgraph_update_post, _on_depsgraph),
                    (bpy.app.handlers.load_post, _on_load),
                    (bpy.app.handlers.frame_change_pre, _on_frame)):
        if fn in lst:
            lst.remove(fn)
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    _clear()
