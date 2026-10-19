"""Property groups: scene-wide simulation settings and per-object body settings."""

import math

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty,
                       IntProperty, PointerProperty, StringProperty)


def _dirty(self, context):
    from . import preview
    preview.mark_dirty()


def _color_update(self, context):
    from . import nodes
    obj = self.id_data
    nodes.sync_colors(obj)
    _dirty(self, context)


def _poll_body(self, obj):
    return obj is not None and obj.rebound.enabled and obj.rebound.kind == "BODY"


def _preview_toggle(self, context):
    from . import preview
    preview.mark_dirty()
    preview.redraw()


def _active_body_update(self, context):
    scene = self.id_data
    objs = scene.objects
    idx = self.active_body_index
    if 0 <= idx < len(objs):
        obj = objs[idx]
        if obj.rebound.enabled and context.view_layer is not None:
            try:
                for o in context.selected_objects:
                    o.select_set(False)
                obj.select_set(True)
                context.view_layer.objects.active = obj
            except RuntimeError:
                pass


UNITS = [
    ("CODE", "Code units (G = custom)", "Dimensionless units: you choose G (usually 1)"),
    ("AU_YR2PI_MSUN", "AU, yr/2π, Msun (G = 1)",
     "Astronomical units, solar masses and time in years/2π, so that G = 1. "
     "REBOUND's built-in datasets and Horizons use this by default"),
    ("AU_YR_MSUN", "AU, yr, Msun", "Astronomical units, years, solar masses"),
    ("AU_DAY_MSUN", "AU, day, Msun", "Astronomical units, days, solar masses"),
    ("SI", "SI (m, s, kg)", "Metres, seconds, kilograms"),
]

INTEGRATORS = [
    ("ias15", "IAS15", "Adaptive 15th-order integrator, accurate to machine precision in most "
     "cases. Good default, handles close encounters"),
    ("whfast", "WHFast", "Symplectic Wisdom-Holman integrator: fast, fixed timestep. Best with a "
     "dominant central mass and small perturbations"),
    ("mercurius", "MERCURIUS", "Hybrid: WHFast for long-term evolution, switches to IAS15 during "
     "close encounters (fixed timestep)"),
    ("trace", "TRACE", "Hybrid (almost) time-reversible integrator for systems with a dominant "
     "central mass and occasional close encounters"),
    ("saba", "SABA", "Family of symplectic integrators (Laskar & Robutel), fixed timestep"),
    ("leapfrog", "Leapfrog", "Standard symplectic leapfrog, fixed timestep"),
    ("janus", "JANUS", "Bit-wise time-reversible symplectic integrator, fixed timestep"),
    ("bs", "Bulirsch-Stoer", "Adaptive Gragg-Bulirsch-Stoer integrator"),
]

FRAME_MODES = [
    ("INERTIAL", "Inertial", "Simulation coordinates (barycentric if 'Move to COM' is on)"),
    ("CENTERED", "Centred on body", "Positions relative to one body (e.g. heliocentric)"),
    ("COROTATING", "Co-rotating pair", "Frame rotating with two bodies (synodic frame): the "
     "secondary stays on a fixed axis, e.g. to show Lagrange points"),
]


class RB_BodySettings(bpy.types.PropertyGroup):
    enabled: BoolProperty(name="REBOUND Body", default=False, update=_dirty,
                          description="Include this object in the REBOUND simulation")
    kind: EnumProperty(name="Kind", items=[
        ("BODY", "Body", "A single particle driven by this object"),
        ("SWARM", "Swarm", "Many massless test particles drawn as one object"),
    ], default="BODY", update=_dirty)

    # --- physical
    mass: FloatProperty(name="Mass", default=1e-3, min=0.0, precision=8, update=_dirty,
                        description="Mass in simulation units (0 = test particle)")
    radius: FloatProperty(name="Physical Radius", default=0.0, min=0.0, precision=6,
                          update=_dirty,
                          description="Physical radius used for collisions (simulation units). "
                                      "Does not change how big the object looks")
    color: FloatVectorProperty(name="Color", subtype="COLOR", size=4, min=0.0, max=1.0,
                               default=(1.0, 0.75, 0.3, 1.0), update=_color_update,
                               description="Colour of trails, orbits and preview lines")
    order: IntProperty(name="Order", default=0, update=_dirty,
                       description="Bodies are added to REBOUND sorted by this value (then by "
                                   "distance from the heaviest body). Matters for Jacobi coordinates")

    # --- initial conditions
    ic_mode: EnumProperty(name="Initial Conditions", items=[
        ("CARTESIAN", "Cartesian", "Position from the object's location, velocity below"),
        ("ORBIT", "Orbital Elements", "Keplerian elements relative to a primary"),
    ], default="CARTESIAN", update=_dirty)
    velocity: FloatVectorProperty(name="Velocity", size=3, subtype="XYZ", default=(0, 0, 0),
                                  precision=6, update=_dirty,
                                  description="Initial velocity in simulation units")
    primary: PointerProperty(name="Primary", type=bpy.types.Object, poll=_poll_body,
                             update=_dirty,
                             description="Body this one orbits. Orbital elements: empty = centre "
                                         "of mass of all bodies added before (Jacobi). Cartesian: "
                                         "only used for drawing the orbit and orbital elements")
    a: FloatProperty(name="Semi-major Axis", default=1.0, precision=6, update=_dirty,
                     description="Semi-major axis (negative for hyperbolic orbits)")
    e: FloatProperty(name="Eccentricity", default=0.0, min=0.0, soft_max=0.99, precision=5,
                     update=_dirty)
    inc: FloatProperty(name="Inclination", default=0.0, subtype="ANGLE", unit="ROTATION",
                       update=_dirty)
    Omega: FloatProperty(name="Long. of Asc. Node", default=0.0, subtype="ANGLE",
                         unit="ROTATION", update=_dirty, description="Longitude of the ascending node Ω")
    omega: FloatProperty(name="Arg. of Pericentre", default=0.0, subtype="ANGLE",
                         unit="ROTATION", update=_dirty, description="Argument of pericentre ω")
    anomaly_type: EnumProperty(name="Anomaly", items=[
        ("f", "True Anomaly", "True anomaly f"),
        ("M", "Mean Anomaly", "Mean anomaly M"),
        ("l", "Mean Longitude", "Mean longitude λ = Ω + ω + M"),
    ], default="f", update=_dirty)
    anomaly: FloatProperty(name="Angle", default=0.0, subtype="ANGLE", unit="ROTATION",
                           update=_dirty)

    # --- visualisation
    show_trail: BoolProperty(name="Trail", default=True, description="Create a trail when baking")
    show_orbit: BoolProperty(name="Orbit", default=True,
                             description="Create an animated osculating orbit when baking "
                                         "(if orbits are enabled in the scene settings)")

    # --- swarm
    count: IntProperty(name="Particles", default=500, min=1, soft_max=20000, update=_dirty)
    a_min: FloatProperty(name="a min", default=0.8, min=0.0, precision=4, update=_dirty,
                         description="Smallest semi-major axis")
    a_max: FloatProperty(name="a max", default=1.2, min=0.0, precision=4, update=_dirty,
                         description="Largest semi-major axis")
    e_max: FloatProperty(name="e max", default=0.05, min=0.0, max=0.99, precision=4,
                         update=_dirty, description="Eccentricities are uniform in [0, e max]")
    inc_max: FloatProperty(name="Inclination max", default=math.radians(2.0), min=0.0,
                           subtype="ANGLE", unit="ROTATION", update=_dirty)
    lon_center: FloatProperty(name="Longitude Centre", default=0.0, subtype="ANGLE",
                              unit="ROTATION", update=_dirty,
                              description="Centre of the mean-longitude range")
    lon_width: FloatProperty(name="Longitude Width", default=2 * math.pi, min=0.0,
                             max=2 * math.pi, subtype="ANGLE", unit="ROTATION", update=_dirty,
                             description="Width of the mean-longitude range (360° = full ring; "
                                         "smaller values make arcs, e.g. Trojan clouds)")
    seed: IntProperty(name="Seed", default=0, update=_dirty)
    particle_size: FloatProperty(name="Particle Size", default=0.01, min=0.0, precision=4,
                                 description="Display radius of each particle (Blender units)")
    tail: IntProperty(name="Tail Frames", default=0, min=0, soft_max=60,
                      description="Also show the last N frames as shrinking points (streaks)")

    # --- bake bookkeeping
    has_initial: BoolProperty(default=False, options={"HIDDEN"})
    initial_location: FloatVectorProperty(size=3, options={"HIDDEN"})
    initial_scale: FloatVectorProperty(size=3, default=(1, 1, 1), options={"HIDDEN"})


class RB_SceneSettings(bpy.types.PropertyGroup):
    # --- physics
    units: EnumProperty(name="Units", items=UNITS, default="CODE", update=_dirty)
    G: FloatProperty(name="G", default=1.0, min=0.0, precision=6, update=_dirty,
                     description="Gravitational constant (code units only)")
    integrator: EnumProperty(name="Integrator", items=INTEGRATORS, default="ias15", update=_dirty)
    dt_mode: EnumProperty(name="Timestep", items=[
        ("AUTO", "Auto", "Fraction of the shortest orbital period at the start"),
        ("MANUAL", "Manual", "Fixed timestep value"),
    ], default="AUTO", update=_dirty)
    dt: FloatProperty(name="dt", default=0.01, min=1e-12, precision=6, update=_dirty,
                      description="Timestep (initial guess for adaptive integrators)")
    dt_fraction: FloatProperty(name="Fraction of Period", default=0.02, min=1e-6, max=1.0,
                               precision=4, update=_dirty,
                               description="dt = this fraction of the shortest orbital period")
    move_to_com: BoolProperty(name="Move to Centre of Mass", default=True, update=_dirty,
                              description="Remove the drift of the centre of mass (REBOUND "
                                          "move_to_com)")
    collision: EnumProperty(name="Collisions", items=[
        ("none", "None", "No collision detection"),
        ("direct", "Direct", "Check all pairs for overlaps each step"),
        ("line", "Line", "Also detect collisions along the path of the last step"),
    ], default="none", update=_dirty)
    collision_resolve: EnumProperty(name="Resolve", items=[
        ("merge", "Merge", "Merge colliding bodies (mass, momentum and volume conserved)"),
        ("hardsphere", "Bounce", "Elastic hard-sphere bounce"),
        ("halt", "Stop", "Stop the simulation at the first collision"),
    ], default="merge", update=_dirty)

    # --- time mapping
    time_per_frame: FloatProperty(name="Time per Frame", default=0.05, min=1e-12, precision=5,
                                  update=_dirty,
                                  description="Simulation time that passes between two frames")
    exact_mode: EnumProperty(name="Output Times", items=[
        ("AUTO", "Auto", "Exact for adaptive integrators; for fixed-step integrators the nearest "
                         "step is used so the timestep stays constant (keeps them symplectic)"),
        ("EXACT", "Exact", "Hit every frame time exactly (REBOUND exact_finish_time=1)"),
        ("STEP", "Nearest Step", "Never shorten a step (exact_finish_time=0)"),
    ], default="AUTO", update=_dirty)

    # --- display
    frame_mode: EnumProperty(name="Reference Frame", items=FRAME_MODES, default="INERTIAL",
                             update=_dirty)
    frame_center: PointerProperty(name="Centre", type=bpy.types.Object, poll=_poll_body,
                                  update=_dirty, description="Body at the centre of the frame "
                                                             "(primary of the co-rotating pair)")
    frame_secondary: PointerProperty(name="Secondary", type=bpy.types.Object, poll=_poll_body,
                                     update=_dirty, description="Second body of the co-rotating pair")
    anchor_frame: BoolProperty(name="Keep Initial Placement", default=True, update=_dirty,
                               description="Shift/rotate the output so that the first frame matches "
                                           "where you placed the bodies. Off: raw frame coordinates "
                                           "(centre of mass / centre body at the origin)")
    length_scale: FloatProperty(name="Length Scale", default=1.0, min=1e-30, precision=6,
                                update=_dirty,
                                description="Blender units per simulation length unit")

    trails: BoolProperty(name="Trails", default=True)
    trail_length: IntProperty(name="Trail Length", default=0, min=0,
                              description="Length of trails in frames (0 = full history)")
    trail_thickness: FloatProperty(name="Thickness", default=0.01, min=0.0, precision=4,
                                   description="Trail radius in Blender units")
    trail_fade: FloatProperty(name="Fade", default=1.0, min=0.0, max=1.0, subtype="FACTOR",
                              description="Fade trails towards their old end")
    trail_taper: FloatProperty(name="Taper", default=0.6, min=0.0, max=1.0, subtype="FACTOR",
                               description="Make trails thinner towards their old end")
    trail_emission: FloatProperty(name="Glow", default=3.0, min=0.0, soft_max=50.0,
                                  description="Emission strength of trail and orbit materials")

    orbits: BoolProperty(name="Osculating Orbits", default=False,
                         description="Create animated osculating orbit ellipses")
    orbit_thickness: FloatProperty(name="Thickness", default=0.004, min=0.0, precision=4)
    orbit_resolution: IntProperty(name="Resolution", default=128, min=8, max=2048)

    scale_with_radius: BoolProperty(name="Scale by Physical Radius", default=False,
                                    description="Scale objects with their physical radius over time "
                                                "(e.g. growth after merging)")
    bake_elements: BoolProperty(name="Orbital Elements", default=True,
                                description="Keyframe a, e, inc as custom properties of each body "
                                            "(see them in the Graph Editor)")
    bake_energy: BoolProperty(name="Energy Error", default=True,
                              description="Keyframe the relative energy error |ΔE/E₀| as a scene "
                                          "custom property 'rb_energy_error'")
    archive: BoolProperty(name="Save SimulationArchive", default=False,
                          description="Also write a REBOUND SimulationArchive (one snapshot per frame) "
                                      "for analysis in Python/Jupyter")
    archive_path: StringProperty(name="Archive", subtype="FILE_PATH", default="//rebound_archive.bin")

    # --- parallel execution & scientific verification
    parallel_bake: BoolProperty(
        name="Parallel Swarm Bake", default=True,
        description="Split massless swarm particles across multiple CPU worker processes when "
                    "baking. Disable for strict single-process bitwise reproducibility with "
                    "adaptive integrators (IAS15, BS)")
    parallel_workers: IntProperty(
        name="Worker Processes", default=0, min=0, soft_max=32, max=64,
        description="Number of parallel worker processes for swarm baking (0 = automatic based on CPU cores)")
    parallel_probes: IntProperty(
        name="Accuracy Probe Particles", default=8, min=0, max=64,
        description="Shared test particles integrated in every worker (float64) to measure "
                    "trajectory divergence Δr/r between parallel chunks (0 = check massive bodies only)")

    # --- preview / HUD
    preview_enabled: BoolProperty(name="Live Preview", default=False, update=_preview_toggle,
                                  description="Draw predicted paths in the viewport while you edit "
                                              "initial conditions (before baking)")
    preview_frames: IntProperty(name="Preview Frames", default=0, min=0, update=_dirty,
                                description="How many frames to predict (0 = whole frame range)")
    preview_velocity: BoolProperty(name="Velocity Arrows", default=True, update=_preview_toggle)
    velocity_scale: FloatProperty(name="Arrow Time", default=1.0, min=0.0, precision=4,
                                  update=_preview_toggle,
                                  description="Arrow length = velocity × this much simulation time")
    preview_line_width: FloatProperty(name="Line Width", default=2.0, min=0.5, max=10.0,
                                      update=_preview_toggle)
    preview_swarms: BoolProperty(name="Swarm Points", default=True, update=_dirty,
                                 description="Show the initial positions of swarm particles")
    hud: BoolProperty(name="Overlay Text", default=True, update=_preview_toggle,
                      description="Show simulation time and energy error in the viewport")

    # --- presets & Horizons
    preset: EnumProperty(name="Preset", items=lambda self, ctx: _preset_items())
    horizons_names: StringProperty(name="Bodies", default="Sun, Mercury, Venus, Earth, Mars, Jupiter",
                                   description="Comma-separated names or NASA Horizons IDs")
    horizons_date: StringProperty(name="Date", default="",
                                  description="Epoch, e.g. '2026-01-01 00:00' or 'JD2461041.5'. "
                                              "Empty = now (UTC)")

    # --- status
    baked: BoolProperty(default=False)
    bake_running: BoolProperty(default=False)
    bake_progress: FloatProperty(default=0.0, subtype="FACTOR", min=0.0, max=1.0)
    bake_info: StringProperty(default="")
    bake_messages: StringProperty(default="")
    energy_error: FloatProperty(default=-1.0)
    split_error: FloatProperty(default=-1.0)
    active_body_index: IntProperty(default=0, update=_active_body_update)


_PRESET_CACHE = []


def _preset_items():
    from . import presets
    if not _PRESET_CACHE:
        _PRESET_CACHE.extend(presets.enum_items())
    return _PRESET_CACHE


classes = (RB_BodySettings, RB_SceneSettings)


def register_props():
    bpy.types.Object.rebound = PointerProperty(type=RB_BodySettings)
    bpy.types.Scene.rebound = PointerProperty(type=RB_SceneSettings)


def unregister_props():
    del bpy.types.Scene.rebound
    del bpy.types.Object.rebound
