"""Geometry Nodes groups and materials for trails, orbits and swarms."""

import bpy

NODES_VERSION = 4

TRAIL_GROUP = "REBOUND Trail"
ORBIT_GROUP = "REBOUND Orbit"
SWARM_GROUP = "REBOUND Swarm"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _inp(node, name):
    for s in node.inputs:
        if s.name == name and getattr(s, "enabled", True):
            return s
    raise KeyError(f"{node.bl_idname} has no input '{name}'")


def _out(node, *names):
    for name in names:
        for s in node.outputs:
            if s.name == name and getattr(s, "enabled", True):
                return s
    raise KeyError(f"{node.bl_idname} has no output {names}")


class _Builder:
    def __init__(self, ng):
        self.ng = ng
        self.x = 0

    def node(self, bl_idname, x=0, y=0, **props):
        n = self.ng.nodes.new(bl_idname)
        n.location = (x, y)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def link(self, a, b):
        self.ng.links.new(a, b)

    def math(self, op, a, b=None, x=0, y=0, clamp=False):
        """Math node; a/b are sockets or numbers."""
        n = self.node("ShaderNodeMath", x, y, operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = float(v)
            else:
                self.link(v, n.inputs[i])
        return n.outputs[0]


def _new_group(name, inputs, force=False):
    ng = bpy.data.node_groups.get(name)
    if ng is not None and ng.get("rb_version") == NODES_VERSION and not force:
        return ng, False
    if ng is None:
        ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    else:
        ng.nodes.clear()
        ng.interface.clear()
    ng["rb_version"] = NODES_VERSION
    try:
        ng.is_modifier = True
    except AttributeError:
        pass
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    for (sname, stype, default, smin, smax) in inputs:
        sock = ng.interface.new_socket(sname, in_out="INPUT", socket_type=stype)
        if default is not None and hasattr(sock, "default_value"):
            sock.default_value = default
        if smin is not None and hasattr(sock, "min_value"):
            sock.min_value = smin
        if smax is not None and hasattr(sock, "max_value"):
            sock.max_value = smax
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    return ng, True


def set_modifier_inputs(mod, values):
    """Set group inputs of a GN modifier by socket name."""
    ng = mod.node_group
    props = getattr(mod, "properties", None)          # Blender 5.x
    inputs = getattr(props, "inputs", None) if props is not None else None
    for item in ng.interface.items_tree:
        if getattr(item, "in_out", None) != "INPUT" or item.name not in values:
            continue
        sock = getattr(inputs, item.identifier, None) if inputs is not None else None
        if sock is not None and hasattr(sock, "value"):
            sock.value = values[item.name]
        else:                                          # Blender 4.x ID properties
            mod[item.identifier] = values[item.name]


def ensure_gn_modifier(obj, group, name="REBOUND"):
    mod = obj.modifiers.get(name)
    if mod is None or mod.type != "NODES":
        mod = obj.modifiers.new(name, "NODES")
    mod.node_group = group
    return mod


# ---------------------------------------------------------------------------
# Trail: poly curve with one point per frame, trimmed to the current frame
# ---------------------------------------------------------------------------

def trail_group(force=False):
    ng, fresh = _new_group(TRAIL_GROUP, [
        ("Frame Start", "NodeSocketInt", 1, None, None),
        ("Frame Step", "NodeSocketFloat", 1.0, 1e-6, None),
        ("Length", "NodeSocketInt", 0, 0, None),
        ("Radius", "NodeSocketFloat", 0.01, 0.0, None),
        ("Fade", "NodeSocketFloat", 1.0, 0.0, 1.0),
        ("Taper", "NodeSocketFloat", 0.6, 0.0, 1.0),
        ("Material", "NodeSocketMaterial", None, None, None),
    ], force)
    if not fresh:
        return ng
    b = _Builder(ng)
    gi = b.node("NodeGroupInput", -1400, 0)
    go = b.node("NodeGroupOutput", 900, 0)
    st = b.node("GeometryNodeInputSceneTime", -1400, 300)
    idx = _out(b.node("GeometryNodeInputIndex", -1400, 450), "Index")
    step = b.math("MAXIMUM", gi.outputs["Frame Step"], 1e-6, -1300, 200)
    L = b.math("DIVIDE", gi.outputs["Length"], step, -1150, 450)
    cur = b.math("DIVIDE", b.math("SUBTRACT", _out(st, "Frame"), gi.outputs["Frame Start"], -1250, 300),
                 step, -1100, 300)

    # fade value f in [0,1] along the visible part (computed before trimming)
    denom = b.math("ADD", L, b.math("MULTIPLY", b.math("LESS_THAN", L, 0.5, -1000, 600),
                                    b.math("MAXIMUM", cur, 1.0, -1000, 500), -850, 550), -700, 550)
    num = b.math("ADD", b.math("SUBTRACT", idx, cur, -850, 450), denom, -550, 500)
    f = b.math("DIVIDE", num, denom, -400, 500, clamp=True)
    fv = b.math("SUBTRACT", 1.0, b.math("MULTIPLY", gi.outputs["Fade"],
                                        b.math("SUBTRACT", 1.0, f, -250, 450), -100, 450), 50, 450)
    store = b.node("GeometryNodeStoreNamedAttribute", -400, 0, data_type="FLOAT", domain="POINT")
    b.link(gi.outputs["Geometry"], _inp(store, "Geometry"))
    _inp(store, "Name").default_value = "rb_fade"
    b.link(fv, _inp(store, "Value"))

    # delete the future and (optionally) the far past
    future = b.math("GREATER_THAN", idx, cur, -400, -250)
    past = b.math("MULTIPLY", b.math("LESS_THAN", idx, b.math("SUBTRACT", cur, L, -550, -350), -400, -350),
                  b.math("GREATER_THAN", L, 0.5, -400, -450), -250, -400)
    sel = b.math("MAXIMUM", future, past, -100, -300)
    dele = b.node("GeometryNodeDeleteGeometry", -100, 0, domain="POINT", mode="ALL")
    b.link(_out(store, "Geometry"), _inp(dele, "Geometry"))
    b.link(sel, _inp(dele, "Selection"))

    # tapered radius from the stored fade attribute
    na = b.node("GeometryNodeInputNamedAttribute", -100, -550, data_type="FLOAT")
    _inp(na, "Name").default_value = "rb_fade"
    taper = b.math("SUBTRACT", 1.0, b.math("MULTIPLY", gi.outputs["Taper"],
                                           b.math("SUBTRACT", 1.0, _out(na, "Attribute"), 50, -550),
                                           200, -550), 350, -550)
    rad = b.math("MULTIPLY", taper, gi.outputs["Radius"], 500, -550)

    profile = b.node("GeometryNodeCurvePrimitiveCircle", 200, -250, mode="RADIUS")
    _inp(profile, "Resolution").default_value = 8
    _inp(profile, "Radius").default_value = 1.0
    c2m = b.node("GeometryNodeCurveToMesh", 450, 0)
    setr = b.node("GeometryNodeSetCurveRadius", 150, 0)
    b.link(_out(dele, "Geometry"), _inp(setr, "Curve"))
    has_scale = any(s.name == "Scale" for s in c2m.inputs)
    if has_scale:
        _inp(setr, "Radius").default_value = 1.0
        b.link(rad, _inp(c2m, "Scale"))
    else:
        b.link(rad, _inp(setr, "Radius"))
    b.link(_out(setr, "Curve"), _inp(c2m, "Curve"))
    b.link(_out(profile, "Curve"), _inp(c2m, "Profile Curve"))
    _inp(c2m, "Fill Caps").default_value = True
    mat = b.node("GeometryNodeSetMaterial", 650, 0)
    b.link(_out(c2m, "Mesh"), mat.inputs[0])
    b.link(gi.outputs["Material"], _inp(mat, "Material"))
    smooth = b.node("GeometryNodeSetShadeSmooth", 800, 0)
    b.link(_out(mat, "Geometry"), smooth.inputs[0])
    b.link(smooth.outputs[0], go.inputs[0])
    return ng


# ---------------------------------------------------------------------------
# Orbit: unit circle transformed by a driver empty (loc/rot/scale = ellipse)
# ---------------------------------------------------------------------------

def orbit_group(force=False):
    ng, fresh = _new_group(ORBIT_GROUP, [
        ("Driver", "NodeSocketObject", None, None, None),
        ("Thickness", "NodeSocketFloat", 0.004, 0.0, None),
        ("Resolution", "NodeSocketInt", 128, 3, 4096),
        ("Material", "NodeSocketMaterial", None, None, None),
    ], force)
    if not fresh:
        return ng
    b = _Builder(ng)
    gi = b.node("NodeGroupInput", -900, 0)
    go = b.node("NodeGroupOutput", 700, 0)
    oi = b.node("GeometryNodeObjectInfo", -650, -200)
    oi.transform_space = "RELATIVE"
    b.link(gi.outputs["Driver"], _inp(oi, "Object"))
    circ = b.node("GeometryNodeCurvePrimitiveCircle", -650, 150, mode="RADIUS")
    b.link(gi.outputs["Resolution"], _inp(circ, "Resolution"))
    _inp(circ, "Radius").default_value = 1.0
    xf = b.node("GeometryNodeTransform", -400, 0)
    b.link(_out(circ, "Curve"), _inp(xf, "Geometry"))
    b.link(_out(oi, "Location"), _inp(xf, "Translation"))
    b.link(_out(oi, "Rotation"), _inp(xf, "Rotation"))
    b.link(_out(oi, "Scale"), _inp(xf, "Scale"))
    setr = b.node("GeometryNodeSetCurveRadius", -200, 0)
    b.link(_out(xf, "Geometry"), _inp(setr, "Curve"))
    _inp(setr, "Radius").default_value = 1.0
    prof = b.node("GeometryNodeCurvePrimitiveCircle", -200, -250, mode="RADIUS")
    _inp(prof, "Resolution").default_value = 6
    b.link(gi.outputs["Thickness"], _inp(prof, "Radius"))
    c2m = b.node("GeometryNodeCurveToMesh", 50, 0)
    b.link(_out(setr, "Curve"), _inp(c2m, "Curve"))
    b.link(_out(prof, "Curve"), _inp(c2m, "Profile Curve"))
    if any(s.name == "Scale" for s in c2m.inputs):
        one = b.node("ShaderNodeValue", -200, -450)
        one.outputs[0].default_value = 1.0
        b.link(one.outputs[0], _inp(c2m, "Scale"))
    mat = b.node("GeometryNodeSetMaterial", 300, 0)
    b.link(_out(c2m, "Mesh"), mat.inputs[0])
    b.link(gi.outputs["Material"], _inp(mat, "Material"))
    smooth = b.node("GeometryNodeSetShadeSmooth", 500, 0)
    b.link(_out(mat, "Geometry"), smooth.inputs[0])
    b.link(smooth.outputs[0], go.inputs[0])
    return ng


# ---------------------------------------------------------------------------
# Swarm: one vertex per particle per frame, keep the current frame (+tail)
# ---------------------------------------------------------------------------

def swarm_group(force=False):
    ng, fresh = _new_group(SWARM_GROUP, [
        ("Frame Start", "NodeSocketInt", 1, None, None),
        ("Particle Size", "NodeSocketFloat", 0.01, 0.0, None),
        ("Tail Frames", "NodeSocketInt", 0, 0, None),
        ("Material", "NodeSocketMaterial", None, None, None),
    ], force)
    if not fresh:
        return ng
    b = _Builder(ng)
    gi = b.node("NodeGroupInput", -1000, 0)
    go = b.node("NodeGroupOutput", 700, 0)
    st = b.node("GeometryNodeInputSceneTime", -1000, 300)
    cur = b.math("ROUND", b.math("SUBTRACT", _out(st, "Frame"), gi.outputs["Frame Start"], -800, 300),
                 None, -650, 300)
    na = b.node("GeometryNodeInputNamedAttribute", -800, 500, data_type="FLOAT")
    _inp(na, "Name").default_value = "rb_frame"
    fr = _out(na, "Attribute")
    tail = gi.outputs["Tail Frames"]
    future = b.math("GREATER_THAN", fr, cur, -500, 450)
    old = b.math("LESS_THAN", fr, b.math("SUBTRACT", cur, tail, -650, 150), -500, 200)
    sel = b.math("MAXIMUM", future, old, -350, 300)
    dele = b.node("GeometryNodeDeleteGeometry", -200, 0, domain="POINT", mode="ALL")
    b.link(gi.outputs["Geometry"], _inp(dele, "Geometry"))
    b.link(sel, _inp(dele, "Selection"))
    age = b.math("SUBTRACT", cur, fr, -350, -200)
    fade = b.math("SUBTRACT", 1.0, b.math("DIVIDE", age, b.math("ADD", tail, 1.0, -350, -350),
                                          -200, -250), -50, -250)
    size = b.math("MULTIPLY", fade, gi.outputs["Particle Size"], 100, -250)
    ico = b.node("GeometryNodeMeshIcoSphere", -200, -450)
    _inp(ico, "Radius").default_value = 1.0
    _inp(ico, "Subdivisions").default_value = 1
    mat = b.node("GeometryNodeSetMaterial", 0, -450)
    b.link(_out(ico, "Mesh"), mat.inputs[0])
    b.link(gi.outputs["Material"], _inp(mat, "Material"))
    inst = b.node("GeometryNodeInstanceOnPoints", 300, 0)
    b.link(_out(dele, "Geometry"), _inp(inst, "Points"))
    b.link(_out(mat, "Geometry"), _inp(inst, "Instance"))
    b.link(size, _inp(inst, "Scale"))
    b.link(_out(inst, "Instances"), go.inputs[0])
    return ng


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

def _fresh_tree(mat):
    if mat.node_tree is None:
        mat.use_nodes = True  # only needed on old versions
    nt = mat.node_tree
    nt.nodes.clear()
    return nt


def emission_material(name, color, strength=3.0, fade=False):
    """Glowing material; with ``fade`` it reads the 'rb_fade' attribute for alpha."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat["rebound"] = True
    nt = _fresh_tree(mat)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)
    em = nt.nodes.new("ShaderNodeEmission")
    em.location = (0, 0)
    em.inputs["Color"].default_value = color
    em.inputs["Strength"].default_value = strength
    if fade:
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.location = (-250, 150)
        attr.attribute_type = "GEOMETRY"
        attr.attribute_name = "rb_fade"
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        tr.location = (0, 150)
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.location = (200, 0)
        nt.links.new(_out(attr, "Factor", "Fac"), mix.inputs[0])
        nt.links.new(tr.outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
        try:
            mat.surface_render_method = "BLENDED"
        except (AttributeError, TypeError):
            pass
    else:
        nt.links.new(em.outputs[0], out.inputs["Surface"])
    mat.diffuse_color = color
    mat["rb_strength"] = strength
    return mat


def body_material(name, color, emission=0.0):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat["rebound"] = True
    nt = _fresh_tree(mat)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.6
    if emission > 0:
        key = "Emission Color" if "Emission Color" in bsdf.inputs else "Emission"
        bsdf.inputs[key].default_value = color
        bsdf.inputs["Emission Strength"].default_value = emission
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    mat.diffuse_color = color
    return mat


def material_name(kind, obj_name):
    return f"RB {kind} {obj_name}"


def sync_colors(obj):
    """Push a body's colour into the materials generated for it."""
    color = tuple(obj.rebound.color)
    for kind in ("Trail", "Orbit", "Swarm", "Body"):
        mat = bpy.data.materials.get(material_name(kind, obj.name))
        if mat is None or not mat.get("rebound") or mat.node_tree is None:
            continue
        mat.diffuse_color = color
        for n in mat.node_tree.nodes:
            if n.bl_idname == "ShaderNodeEmission":
                n.inputs["Color"].default_value = color
            elif n.bl_idname == "ShaderNodeBsdfPrincipled":
                n.inputs["Base Color"].default_value = color
                key = "Emission Color" if "Emission Color" in n.inputs else "Emission"
                if n.inputs["Emission Strength"].default_value > 0:
                    n.inputs[key].default_value = color
