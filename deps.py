"""Locating / installing the REBOUND Python package inside Blender."""

import importlib
import os
import subprocess
import sys

import bpy

PACKAGE = __package__

_state = {"module": None, "error": "", "install_proc": None, "install_log": "", "G_cache": {}}


def deps_dir():
    """Writable directory for pip-installed dependencies."""
    try:  # Blender 4.2+ extension
        return bpy.utils.extension_path_user(PACKAGE, path="site-packages", create=True)
    except Exception:
        path = bpy.utils.user_resource("SCRIPTS", path="rebound_nbody_site_packages", create=True)
        return path


def _add_path():
    d = deps_dir()
    if d and d not in sys.path:
        sys.path.append(d)
    importlib.invalidate_caches()


def get_rebound():
    """Return the ``rebound`` module or None (error text in ``error()``)."""
    if _state["module"] is not None:
        return _state["module"]
    try:
        import rebound  # bundled wheel or already on sys.path
    except ImportError:
        _add_path()
        try:
            import rebound
        except ImportError as ex:
            _state["error"] = str(ex)
            return None
    except Exception as ex:  # broken binary etc.
        _state["error"] = f"{type(ex).__name__}: {ex}"
        return None
    _state["module"] = rebound
    _state["error"] = ""
    return rebound


def available():
    return get_rebound() is not None


def version():
    rb = get_rebound()
    return getattr(rb, "__version__", "?") if rb else None


def error():
    return _state["error"]


def gravitational_constant(units, G_code=1.0):
    """G in the chosen unit system (taken from REBOUND, cached)."""
    from . import core
    if core.UNIT_SYSTEMS.get(units) is None:
        return G_code
    cache = _state["G_cache"]
    if units not in cache:
        rb = get_rebound()
        if rb is None:
            return None
        sim = rb.Simulation()
        sim.units = core.UNIT_SYSTEMS[units]
        cache[units] = float(sim.G)
    return cache[units]


_doc_cache = {}


def integrator_doc(name):
    """First paragraph of REBOUND's own documentation for an integrator."""
    if name in _doc_cache:
        return _doc_cache[name]
    rb = get_rebound()
    text = ""
    if rb is not None:
        try:
            sim = rb.Simulation()
            sim.integrator = name
            doc = sim.integrator.__doc__ or ""
            parts = [p for p in doc.split("\n\n") if p.strip()]
            if len(parts) > 1:
                text = " ".join(parts[1].split())
                # strip markdown link brackets
                text = text.replace("[", "").replace("]", "")
        except Exception:
            text = ""
    _doc_cache[name] = text
    return text


# ---------------------------------------------------------------------------
# Installing with pip (runs as a subprocess so the UI stays responsive)
# ---------------------------------------------------------------------------

def _pip_available():
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "--version"],
                             capture_output=True, text=True, timeout=60)
        return out.returncode == 0
    except Exception:
        return False


def start_install(upgrade=False):
    if _state["install_proc"] is not None:
        return "An installation is already running."
    if not getattr(bpy.app, "online_access", True):
        return ("Online access is disabled. Enable it in Preferences > System > Network "
                "('Allow Online Access') to install REBOUND.")
    target = deps_dir()
    steps = []
    if not _pip_available():
        steps.append([sys.executable, "-m", "ensurepip", "--upgrade", "--user"])
    cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "--target", target]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append("rebound")
    steps.append(cmd)
    _state["install_steps"] = steps
    _state["install_output"] = ""
    err = _next_step()
    if err:
        return err
    _state["install_log"] = "Installing REBOUND with pip..."
    bpy.app.timers.register(_poll_install, first_interval=0.5)
    return None


def _next_step():
    steps = _state.get("install_steps") or []
    if not steps:
        return "nothing to do"
    cmd = steps.pop(0)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    try:
        _state["install_proc"] = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    except Exception as ex:
        _state["install_proc"] = None
        return f"Could not start pip: {ex}"
    return None


def _poll_install():
    proc = _state["install_proc"]
    if proc is None:
        return None
    if proc.poll() is None:
        return 0.5
    out = proc.stdout.read() if proc.stdout else ""
    _state["install_output"] += out
    _state["install_proc"] = None
    if proc.returncode == 0 and _state.get("install_steps"):
        err = _next_step()
        if err is None:
            return 0.5
        _state["install_log"] = err
    elif proc.returncode == 0:
        _state["module"] = None
        ok = available()
        _state["install_log"] = (f"Installed REBOUND {version()}." if ok else
                                 "pip finished but REBOUND could not be imported: " + error())
        if ok:
            _state["install_log"] += " (Restart Blender if an older version was already loaded.)"
    else:
        tail = "\n".join(_state["install_output"].strip().splitlines()[-6:])
        _state["install_log"] = f"pip failed (code {proc.returncode}):\n{tail}"
    _redraw_all()
    return None


def install_running():
    return _state["install_proc"] is not None


def install_log():
    return _state["install_log"]


def _redraw_all():
    wm = bpy.context.window_manager
    if wm is None:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            area.tag_redraw()


class REBOUND_OT_install(bpy.types.Operator):
    """Download and install the REBOUND Python package with pip (needs internet)"""
    bl_idname = "rebound.install_dependencies"
    bl_label = "Install REBOUND"
    bl_options = {"INTERNAL"}

    upgrade: bpy.props.BoolProperty(name="Upgrade", default=False)

    def execute(self, context):
        msg = start_install(self.upgrade)
        if msg:
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        self.report({"INFO"}, "Installing REBOUND in the background...")
        return {"FINISHED"}


class REBOUND_AP_preferences(bpy.types.AddonPreferences):
    bl_idname = PACKAGE

    def draw(self, context):
        draw_status(self.layout, full=True)


def draw_status(layout, full=False):
    col = layout.column()
    if available():
        col.label(text=f"REBOUND {version()} loaded", icon="CHECKMARK")
        if full:
            row = col.row()
            row.enabled = not install_running()
            op = row.operator(REBOUND_OT_install.bl_idname, text="Upgrade REBOUND", icon="FILE_REFRESH")
            op.upgrade = True
    else:
        col.label(text="REBOUND is not installed", icon="ERROR")
        row = col.row()
        row.enabled = not install_running()
        row.scale_y = 1.4
        row.operator(REBOUND_OT_install.bl_idname, icon="IMPORT")
        if full and error():
            col.label(text=error()[:120])
    if install_running():
        col.label(text="pip is running...", icon="TIME")
    elif install_log():
        box = col.box()
        for line in install_log().splitlines()[:8]:
            box.label(text=line[:120])
    if full:
        col.label(text=f"Install folder: {deps_dir()}")


classes = (REBOUND_OT_install, REBOUND_AP_preferences)
