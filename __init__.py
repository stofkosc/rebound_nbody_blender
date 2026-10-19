"""REBOUND N-body simulations inside Blender.

Set up bodies in the viewport, preview their paths live, bake REBOUND
integrations to keyframes, and visualise them with trails, osculating orbits
and swarms of test particles.
"""

bl_info = {
    "name": "REBOUND N-body",
    "author": "Peter",
    "version": (1, 0, 0),
    "blender": (4, 4, 0),
    "location": "3D Viewport > Sidebar > REBOUND",
    "description": "N-body simulations with REBOUND: bake, trails, orbits, swarms",
    "category": "Physics",
}

import bpy

from . import bake, deps, ops, presets, preview, props, ui

_classes = (*deps.classes, *props.classes, *bake.classes, *presets.classes,
            *ops.classes, *ui.classes)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    props.register_props()
    preview.register()


def unregister():
    preview.unregister()
    props.unregister_props()
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
