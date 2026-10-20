"""Preset systems, JPL Horizons import and helpers to create body objects."""

import datetime
import math

import bmesh
import bpy
from mathutils import Vector

from . import bake, core, deps, nodes, preview, scene_io

BODIES_COLL = "REBOUND Bodies"

PALETTE = [(1.0, 0.55, 0.2, 1), (0.3, 0.65, 1.0, 1), (0.45, 0.95, 0.5, 1), (0.95, 0.35, 0.6, 1),
           (0.75, 0.55, 1.0, 1), (1.0, 0.9, 0.35, 1), (0.3, 0.95, 0.9, 1), (0.95, 0.95, 0.95, 1)]

# Names for REBOUND's built-in datasets, identified by particle count + masses.
_DATASET_NAMES = {
    "solarsystem": ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"],
    "outersolarsystem": ["Sun", "Jupiter", "Saturn", "Uranus", "Neptune"],
}
_PLANET_COLORS = {
    "Sun": (1.0, 0.8, 0.35, 1), "Mercury": (0.7, 0.65, 0.6, 1), "Venus": (0.95, 0.8, 0.55, 1),
    "Earth": (0.3, 0.55, 1.0, 1), "Mars": (0.95, 0.45, 0.25, 1), "Jupiter": (0.9, 0.7, 0.5, 1),
    "Saturn": (0.95, 0.85, 0.55, 1), "Uranus": (0.6, 0.9, 0.95, 1), "Neptune": (0.35, 0.5, 1.0, 1),
}

PRESETS = [
    ("STARTER", "Star + 2 Planets", "A star with two planets given by orbital elements"),
    ("FIGURE_EIGHT", "Figure-Eight Choreography", "Three equal masses chasing each other on a "
     "figure-eight (Chenciner & Montgomery 2000)"),
    ("PYTHAGOREAN", "Pythagorean Three-Body", "Burrau's problem: masses 3, 4, 5 at rest at the "
     "corners of a 3-4-5 triangle"),
    ("TROJANS", "Trojan Swarms", "Test particles around the L4/L5 points of a planet, shown in "
     "the co-rotating frame"),
    ("DISK", "Planetesimal Disk", "Two planets sculpting a disk of 2000 test particles"),
    ("SOLAR_SYSTEM", "Solar System (REBOUND data)", "Sun and 8 planets from REBOUND's built-in "
     "dataset (offline)"),
    ("OUTER_SOLAR_SYSTEM", "Outer Solar System (REBOUND data)", "Sun and the giant planets from "
     "REBOUND's built-in dataset (offline)"),
]


def enum_items():
    return [(k, n, d, i) for i, (k, n, d) in enumerate(PRESETS)]


# ---------------------------------------------------------------------------
# Object creation
# ---------------------------------------------------------------------------

def _sphere_mesh(name):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0)
    bm.to_mesh(me)
    bm.free()
    try:
        me.shade_smooth()
    except AttributeError:
        for p in me.polygons:
            p.use_smooth = True
    return me


def create_body(scene, name, mass, color, size, location=(0, 0, 0), emission=0.0, **settings):
    coll = bake.ensure_collection(scene, BODIES_COLL)
    obj = bpy.data.objects.new(name, _sphere_mesh(name))
    coll.objects.link(obj)
    obj.location = location
    obj.scale = (size, size, size)
    rb = obj.rebound
    rb.enabled = True
    rb.mass = mass
    rb.color = color
    for k, v in settings.items():
        setattr(rb, k, v)
    mat = nodes.body_material(nodes.material_name("Body", obj.name), color, emission)
    obj.data.materials.append(mat)
    return obj


def create_swarm(scene, name, primary, color, **settings):
    coll = bake.ensure_collection(scene, BODIES_COLL)
    obj = bpy.data.objects.new(name, bpy.data.meshes.new(f"{name} Swarm Data"))
    coll.objects.link(obj)
    rb = obj.rebound
    rb.enabled = True
    rb.kind = "SWARM"
    rb.mass = 0.0
    rb.primary = primary
    rb.color = color
    rb.show_trail = False
    rb.show_orbit = False
    for k, v in settings.items():
        setattr(rb, k, v)
    return obj


def remove_system(scene):
    bake.clear_bake(scene)
    for obj in list(scene.objects):
        if obj.rebound.enabled or obj.get("rebound_helper"):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data is not None and data.users == 0 and isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)


def setup_world(scene):
    world = bpy.data.worlds.get("REBOUND Space") or bpy.data.worlds.new("REBOUND Space")
    if world.node_tree is not None:
        bg = next((n for n in world.node_tree.nodes if n.bl_idname == "ShaderNodeBackground"), None)
        if bg is not None:
            bg.inputs["Color"].default_value = (0.002, 0.002, 0.004, 1)
            bg.inputs["Strength"].default_value = 1.0
    world.color = (0.002, 0.002, 0.004)
    scene.world = world


def setup_camera(scene, extent):
    cam_data = bpy.data.cameras.new("REBOUND Camera")
    cam_data.clip_end = max(100.0, 50 * extent)
    cam_data.clip_start = max(0.01, extent * 1e-3)
    cam = bpy.data.objects.new("REBOUND Camera", cam_data)
    cam["rebound_helper"] = True
    scene.collection.objects.link(cam)
    cam.location = Vector((0.0, -2.2 * extent, 1.5 * extent))
    cam.rotation_euler = (-cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam


def setup_light(scene, star, extent):
    light = bpy.data.lights.new("REBOUND Star Light", "POINT")
    light.energy = 1000.0 * max(extent / 4.0, 0.25) ** 2
    light.shadow_soft_size = max(star.scale[0], 0.01)
    lob = bpy.data.objects.new("REBOUND Star Light", light)
    lob["rebound_helper"] = True
    scene.collection.objects.link(lob)
    con = lob.constraints.new("COPY_LOCATION")
    con.target = star


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

def _settings(scene, **kw):
    s = scene.rebound
    defaults = dict(units="CODE", G=1.0, integrator="ias15", dt_mode="AUTO", dt_fraction=0.02,
                    move_to_com=True, collision="none", frame_mode="INERTIAL", anchor_frame=True,
                    length_scale=1.0, trails=True, trail_length=0, orbits=False, exact_mode="AUTO",
                    frame_center=None, frame_secondary=None, duration_mode="FRAMES",
                    sampling_mode="TIMELINE", duration_body=None, retime_mode="ONE_TO_ONE",
                    retime_use_window=False)
    defaults.update(kw)
    for k, v in defaults.items():
        setattr(s, k, v)


def _frames(scene, n):
    scene.frame_start = 1
    scene.frame_end = n


def preset_starter(scene):
    _settings(scene, integrator="whfast", orbits=True, length_scale=2.0,
              time_per_frame=10 * 2 * math.pi / 249)
    _frames(scene, 250)
    star = create_body(scene, "Star", 1.0, (1.0, 0.8, 0.35, 1), 0.25, emission=8.0, show_trail=False)
    create_body(scene, "Planet b", 1e-3, PALETTE[1], 0.08, ic_mode="ORBIT", primary=star,
                a=1.0, e=0.05)
    create_body(scene, "Planet c", 3e-4, PALETTE[0], 0.06, ic_mode="ORBIT", primary=star,
                a=1.7, e=0.12, inc=math.radians(4), omega=1.0, anomaly=2.0)
    return star, 3.6


def preset_figure_eight(scene):
    _settings(scene, length_scale=3.0, time_per_frame=0.05, trail_length=0)
    _frames(scene, 250)
    x1 = (0.97000436, -0.24308753, 0.0)
    v3 = (-0.93240737, -0.86473146, 0.0)
    v1 = (-v3[0] / 2, -v3[1] / 2, 0.0)
    ls = 3.0
    create_body(scene, "Body A", 1.0, PALETTE[0], 0.12, location=[c * ls for c in x1], velocity=v1)
    create_body(scene, "Body B", 1.0, PALETTE[1], 0.12, location=[-x1[0] * ls, -x1[1] * ls, 0],
                velocity=v1)
    create_body(scene, "Body C", 1.0, PALETTE[2], 0.12, location=(0, 0, 0), velocity=v3)
    return None, 3.5


def preset_pythagorean(scene):
    _settings(scene, length_scale=1.0, time_per_frame=0.07, trail_length=80)
    _frames(scene, 1000)
    for name, m, pos, col in (("Mass 3", 3.0, (1, 3, 0), PALETTE[0]),
                              ("Mass 4", 4.0, (-2, -1, 0), PALETTE[1]),
                              ("Mass 5", 5.0, (1, -1, 0), PALETTE[2])):
        create_body(scene, name, m, col, 0.1 * m ** (1 / 3), location=pos)
    return None, 5.0


def preset_trojans(scene):
    star = None
    _settings(scene, integrator="whfast", length_scale=3.0, time_per_frame=40 * 2 * math.pi / 599,
              frame_mode="COROTATING")
    _frames(scene, 600)
    star = create_body(scene, "Star", 1.0, (1.0, 0.8, 0.35, 1), 0.25, emission=8.0, show_trail=False)
    planet = create_body(scene, "Planet", 1e-3, PALETTE[1], 0.1, ic_mode="ORBIT", primary=star,
                         a=1.0, e=0.0, show_trail=False)
    s = scene.rebound
    s.frame_center = star
    s.frame_secondary = planet
    for name, lon, col in (("L4 Swarm", 60.0, PALETTE[0]), ("L5 Swarm", -60.0, PALETTE[3])):
        create_swarm(scene, name, star, col, count=400, a_min=0.99, a_max=1.01, e_max=0.02,
                     inc_max=math.radians(2), lon_center=math.radians(lon),
                     lon_width=math.radians(40), seed=1 if lon > 0 else 2, particle_size=0.012,
                     tail=6)
    return star, 3.6


def preset_disk(scene):
    _settings(scene, integrator="whfast", length_scale=1.5, time_per_frame=60 * 2 * math.pi / 599)
    _frames(scene, 600)
    star = create_body(scene, "Star", 1.0, (1.0, 0.8, 0.35, 1), 0.15, emission=8.0, show_trail=False)
    create_body(scene, "Planet b", 1e-3, PALETTE[1], 0.07, ic_mode="ORBIT", primary=star,
                a=1.0, e=0.03)
    create_body(scene, "Planet c", 3e-4, PALETTE[0], 0.05, ic_mode="ORBIT", primary=star,
                a=1.6, e=0.05, inc=math.radians(1), anomaly=2.5)
    for o in scene_io.body_objects(scene):
        if o.name.startswith("Planet"):
            o.rebound.show_trail = True
    scene.rebound.trail_length = 40
    create_swarm(scene, "Disk", star, (0.85, 0.85, 1.0, 1), count=2000, a_min=0.4, a_max=2.8,
                 e_max=0.05, inc_max=math.radians(1), particle_size=0.008, seed=7)
    return star, 4.2


def _dataset(scene, dataset, anchor=None, periods=1.0):
    rb = deps.get_rebound()
    if rb is None:
        raise core.SetupError("REBOUND is not installed.")
    sim = rb.Simulation()
    try:
        sim.add(dataset)
    except Exception as ex:
        raise core.SetupError(f"This REBOUND version has no '{dataset}' dataset ({ex}).")
    names = _DATASET_NAMES[dataset]
    if sim.N != len(names):
        names = [f"Body {i}" for i in range(sim.N)]
    # REBOUND sets AU, yr/2pi, Msun (G=1) for its datasets
    units = "AU_YR2PI_MSUN" if abs(sim.G - 1.0) < 1e-12 else "CODE"
    ps = list(sim.particles)
    # the "anchor" planet (default: outermost) sets the length scale and the timeline
    if anchor is None or anchor >= len(ps):
        ref = max(ps[1:], key=lambda p: p.x ** 2 + p.y ** 2 + p.z ** 2)
    else:
        ref = ps[anchor]
    far = math.sqrt(ref.x ** 2 + ref.y ** 2 + ref.z ** 2)
    ls = 6.0 / far
    P = ref.orbit(primary=ps[0]).P * periods
    _settings(scene, units=units, G=float(sim.G), integrator="whfast", length_scale=ls,
              time_per_frame=P / 599, orbits=True, trail_length=0)
    _frames(scene, 600)
    star = None
    for i, (p, name) in enumerate(zip(ps, names)):
        col = _PLANET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        size = 0.3 if i == 0 else 0.05 + 0.04 * (p.m / ps[0].m * 1e3) ** (1 / 3)
        obj = create_body(scene, name, p.m, col, size, location=(p.x * ls, p.y * ls, p.z * ls),
                          velocity=(p.vx, p.vy, p.vz), emission=8.0 if i == 0 else 0.0,
                          show_trail=i > 0)
        if i == 0:
            star = obj
    for o in scene_io.body_objects(scene):
        if o is not star:
            o.rebound.primary = star  # heliocentric orbits for drawing
    return star, 8.0


BUILDERS = {
    "STARTER": preset_starter,
    "FIGURE_EIGHT": preset_figure_eight,
    "PYTHAGOREAN": preset_pythagorean,
    "TROJANS": preset_trojans,
    "DISK": preset_disk,
    "SOLAR_SYSTEM": lambda sc: _dataset(sc, "solarsystem", anchor=4, periods=3.0),
    "OUTER_SOLAR_SYSTEM": lambda sc: _dataset(sc, "outersolarsystem"),
}


class REBOUND_OT_load_preset(bpy.types.Operator):
    """Create a ready-made system (replaces the current REBOUND bodies)"""
    bl_idname = "rebound.load_preset"
    bl_label = "Load Preset"
    bl_options = {"REGISTER", "UNDO"}

    preset: bpy.props.EnumProperty(name="Preset", items=enum_items())
    replace: bpy.props.BoolProperty(name="Replace Existing Bodies", default=True)
    world: bpy.props.BoolProperty(name="Dark World", default=True)
    camera: bpy.props.BoolProperty(name="Camera + Light", default=True)

    def invoke(self, context, event):
        self.preset = context.scene.rebound.preset
        return self.execute(context)

    def execute(self, context):
        scene = context.scene
        if self.replace:
            remove_system(scene)
        try:
            star, extent = BUILDERS[self.preset](scene)
        except core.SetupError as ex:
            self.report({"ERROR"}, str(ex))
            return {"CANCELLED"}
        if self.world:
            setup_world(scene)
        if self.camera:
            setup_camera(scene, extent)
            if star is not None:
                setup_light(scene, star, extent)
        _place(scene)
        preview.mark_dirty()
        self.report({"INFO"}, f"Loaded '{dict((k, n) for k, n, _ in PRESETS)[self.preset]}'. "
                              "Press Bake to simulate.")
        return {"FINISHED"}


def _place(scene):
    """Put orbital-element bodies where REBOUND says they start."""
    rb = deps.get_rebound()
    if rb is None:
        return
    try:
        spec, bodies, _sw, _slots = scene_io.make_spec(scene, include_swarms=False, frames=1)
        runner = core.Runner(rb, spec, compute_energy=False)
    except core.SetupError:
        return
    preview.place_orbit_bodies(scene, runner, bodies)


# ---------------------------------------------------------------------------
# JPL Horizons
# ---------------------------------------------------------------------------

class REBOUND_OT_horizons(bpy.types.Operator):
    """Query NASA JPL Horizons (internet) for the listed bodies and add them as Cartesian bodies"""
    bl_idname = "rebound.horizons_import"
    bl_label = "Import from Horizons"
    bl_options = {"REGISTER", "UNDO"}

    replace: bpy.props.BoolProperty(name="Replace Existing Bodies", default=True)

    def execute(self, context):
        scene = context.scene
        s = scene.rebound
        rb = deps.get_rebound()
        if rb is None:
            self.report({"ERROR"}, "REBOUND is not installed.")
            return {"CANCELLED"}
        if not getattr(bpy.app, "online_access", True):
            self.report({"ERROR"}, "Online access is disabled (Preferences > System > Network).")
            return {"CANCELLED"}
        names = [n.strip() for n in s.horizons_names.split(",") if n.strip()]
        if not names:
            self.report({"ERROR"}, "List at least one body.")
            return {"CANCELLED"}
        if s.units == "CODE" and abs(s.G - 1.0) > 1e-12:
            self.report({"ERROR"}, "Horizons needs real units: pick a unit system (or G = 1).")
            return {"CANCELLED"}
        date = s.horizons_date.strip() or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
        sim = rb.Simulation()
        units = core.UNIT_SYSTEMS.get(s.units)
        if units:
            sim.units = units
        else:
            sim.units = core.UNIT_SYSTEMS["AU_YR2PI_MSUN"]
        for n in names:
            try:
                sim.add(n, date=date)
            except Exception as ex:
                self.report({"ERROR"}, f"Horizons query for '{n}' failed: {ex}")
                return {"CANCELLED"}
        if self.replace:
            remove_system(scene)
        if s.units == "CODE":
            s.units = "AU_YR2PI_MSUN"
        ps = list(sim.particles)
        far = max([math.sqrt(p.x ** 2 + p.y ** 2 + p.z ** 2) for p in ps] + [1e-30])
        s.length_scale = 8.0 / far
        ls = s.length_scale
        heaviest = max(range(len(ps)), key=lambda i: ps[i].m)
        star = None
        for i, (p, n) in enumerate(zip(ps, names)):
            col = _PLANET_COLORS.get(n.title(), PALETTE[i % len(PALETTE)])
            obj = create_body(scene, n, p.m, col, 0.3 if i == heaviest else 0.08,
                              location=(p.x * ls, p.y * ls, p.z * ls), velocity=(p.vx, p.vy, p.vz),
                              emission=8.0 if i == heaviest else 0.0)
            if i == heaviest:
                star = obj
        for o in scene_io.body_objects(scene):
            if o is not star:
                o.rebound.primary = star
        preview.mark_dirty()
        self.report({"INFO"}, f"Imported {len(ps)} bodies for {date}.")
        return {"FINISHED"}


classes = (REBOUND_OT_load_preset, REBOUND_OT_horizons)
