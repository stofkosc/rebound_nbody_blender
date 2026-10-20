"""Subprocess worker that integrates one parallel swarm chunk.

Invoked by ``bake.py`` via ``sys.executable worker.py <spec.json> <out.npz> <prog.txt> [deps_dir]``.
Pure Python + REBOUND + numpy (never imports ``bpy``).
"""

from __future__ import annotations

import json
import os
import sys
import traceback

import numpy as np


def _write_progress(path: str, val: float):
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"{val:.6f}\n")
    except OSError:
        pass


def main():
    if len(sys.argv) < 4:
        sys.stderr.write("Usage: worker.py <spec.json> <out.npz> <prog.txt> [deps_dir] [--no-energy]\n")
        sys.exit(2)

    spec_path, out_path, prog_path = sys.argv[1], sys.argv[2], sys.argv[3]
    compute_energy = "--no-energy" not in sys.argv[4:]
    for arg in sys.argv[4:]:
        if not arg.startswith("--") and os.path.isdir(arg) and arg not in sys.path:
            sys.path.insert(0, arg)

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

    try:
        import rebound
        import core

        with open(spec_path, "r", encoding="utf-8") as fh:
            spec_dict = json.load(fh)
        spec = core.spec_from_dict(spec_dict)
        runner = core.Runner(rebound, spec, compute_energy=compute_energy)
        _write_progress(prog_path, 0.0)
        while not runner.done:
            p = runner.step(0.15)
            _write_progress(prog_path, p)

        payload = dict(
            s_lo=np.int64(runner.s_lo),
            G=np.float64(runner.G),
            dt=np.float64(runner.dt),
            offset0=runner.offset0,
            bpos=runner.bpos,
            bvel=runner.bvel,
            bmass=runner.bmass,
            brad=runner.brad,
            energy=runner.energy,
            halted=np.bool_(runner.halted),
            wall=np.float64(runner.wall),
            messages=np.array(runner.messages, dtype=object),
            messages_json=np.array(json.dumps(runner.messages)),
            archive_error=np.array(runner.archive_error or ""),
        )
        if runner.spos is not None:
            payload["spos"] = runner.spos
        if runner.ppos is not None:
            payload["ppos"] = runner.ppos
        np.savez(out_path, **payload)
        _write_progress(prog_path, 1.0)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

