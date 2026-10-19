"""Fast keyframe writing for Blender's layered (slotted) actions (4.4+)."""

import bpy
import numpy as np

_INTERP = {"CONSTANT": 0, "LINEAR": 1, "BEZIER": 2}


def _layered(action):
    return hasattr(action, "layers") and hasattr(action, "slots")


def ensure_action(id_data, name):
    ad = id_data.animation_data or id_data.animation_data_create()
    if ad.action is None:
        act = bpy.data.actions.new(name)
        act["rebound"] = True
        ad.action = act
    return ad.action


def fcurves_of(id_data, create=True):
    """F-Curve collection of the ID's assigned action slot (or legacy action)."""
    ad = id_data.animation_data
    if ad is None or ad.action is None:
        return None
    act = ad.action
    if not _layered(act):
        return act.fcurves
    slot = ad.action_slot
    if slot is None:
        if not create:
            return None
        slot = act.slots.new(id_data.id_type, id_data.name)
        ad.action_slot = slot
    if len(act.layers) == 0:
        if not create:
            return None
        act.layers.new("Layer")
    layer = act.layers[0]
    if len(layer.strips) == 0:
        if not create:
            return None
        layer.strips.new(type="KEYFRAME")
    strip = layer.strips[0]
    cb = strip.channelbag(slot, ensure=create)
    return cb.fcurves if cb is not None else None


def set_keys(id_data, data_path, index, frames, values, action_name, interpolation="LINEAR"):
    """Replace the F-Curve (data_path, index) with keys at ``frames``."""
    ensure_action(id_data, action_name)
    fcs = fcurves_of(id_data)
    fc = fcs.find(data_path, index=index)
    if fc is not None:
        fcs.remove(fc)
    fc = fcs.new(data_path, index=index)
    n = len(frames)
    if n == 0:
        return fc
    kp = fc.keyframe_points
    kp.add(n)
    co = np.empty(2 * n, dtype=np.float32)
    co[0::2] = frames
    co[1::2] = values
    kp.foreach_set("co", co)
    try:
        kp.foreach_set("interpolation", np.full(n, _INTERP[interpolation], dtype=np.int32))
    except Exception:
        for k in kp:
            k.interpolation = interpolation
    fc.update()
    return fc


def set_vector_keys(id_data, data_path, frames, values, action_name, interpolation="LINEAR"):
    values = np.asarray(values)
    for i in range(values.shape[1]):
        set_keys(id_data, data_path, i, frames, values[:, i], action_name, interpolation)


def remove_fcurves(id_data, prefix):
    """Remove F-Curves whose data path starts with ``prefix``."""
    fcs = fcurves_of(id_data, create=False)
    if not fcs:
        return
    for fc in [fc for fc in fcs if fc.data_path.startswith(prefix)]:
        fcs.remove(fc)


def clear_rebound_action(id_data):
    """Remove an action created by this add-on (keeps user actions)."""
    ad = id_data.animation_data
    if ad is None or ad.action is None:
        return False
    act = ad.action
    if not act.get("rebound"):
        return False
    ad.action = None
    if act.users == 0:
        bpy.data.actions.remove(act)
    return True
