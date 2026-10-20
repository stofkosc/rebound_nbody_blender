#!/usr/bin/env python3
"""Standalone command-line runner for REBOUND N-body simulations outside Blender.

Run exported Blender simulation specs (``.json``) or convert REBOUND
``SimulationArchive`` (``.bin``) files on university clusters / workstations
without Blender installed. Only requires Python + ``numpy`` + ``rebound``.

Examples:
    # Run an exported spec across 32 CPU workers and save a Blender-ready .npz:
    python cli.py my_system.json -o result.npz --workers 32

    # Prototype small in Blender, then scale up particles & frames on the cluster:
    python cli.py my_system.json -o hires.npz -w 32 --swarm-count 50000 --frames 2000

    # Convert a native REBOUND SimulationArchive (.bin) into a Blender bake (.npz):
    python cli.py --from-archive archive.bin -o archive_bake.npz
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time


def _import_core():
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    try:
        from . import core  # type: ignore
        return core
    except ImportError:
        import core  # type: ignore
        return core


def load_spec_file(path: str):
    """Load a ``(SimSpec, meta_dict)`` from a ``.json`` spec file."""
    core = _import_core()
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "spec" in data and isinstance(data["spec"], dict):
        spec = core.spec_from_dict(data["spec"])
        meta = dict(data.get("meta") or {})
    else:
        spec = core.spec_from_dict(data)
        meta = {}
    return spec, meta


def save_spec_file(path: str, spec, meta: dict | None = None):
    """Save a ``SimSpec`` and optional Blender visual metadata to a ``.json`` file."""
    core = _import_core()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "format": "rebound_nbody_spec_v1",
        "spec": core.spec_to_dict(spec),
        "meta": meta or {},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def _safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run REBOUND N-body simulations outside Blender (multicore swarm support) "
                    "and produce .npz bake files that load directly into the Blender add-on."
    )
    parser.add_argument(
        "spec", nargs="?", default=None,
        help="Path to simulation spec .json file (exported from Blender or written manually).",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output .npz bake file path (loadable in Blender via Import Bake).",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=0,
        help="Number of parallel worker processes for massless swarms (0 = auto CPU count, 1 = single process).",
    )
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Force single-process integration even if swarms are present.",
    )
    parser.add_argument(
        "--probes", type=int, default=8,
        help="Number of shared float64 accuracy probe particles across workers (default: 8).",
    )
    parser.add_argument(
        "--frames", type=int, default=None,
        help="Override number of output snapshots (n_frames).",
    )
    parser.add_argument(
        "--time-per-frame", type=float, default=None,
        help="Override simulation time per snapshot.",
    )
    parser.add_argument(
        "--orbits", type=float, default=None,
        help="Simulate for this many orbital periods of --ref-body (or shortest bound orbit).",
    )
    parser.add_argument(
        "--ref-body", type=str, default=None,
        help="Reference body name for --orbits and --samples-per-orbit (default: shortest orbit).",
    )
    parser.add_argument(
        "--duration-time", type=float, default=None,
        help="Simulate for this total duration in --duration-unit.",
    )
    parser.add_argument(
        "--duration-unit", type=str, default="SIM",
        choices=["SIM", "YR", "KYR", "MYR", "DAY", "SEC"],
        help="Unit for --duration-time (SIM, YR, KYR, MYR, DAY, SEC).",
    )
    parser.add_argument(
        "--samples-per-orbit", type=float, default=None,
        help="Number of recorded snapshots per reference orbital period.",
    )
    parser.add_argument(
        "--swarm-count", type=int, default=None,
        help="Override particle count for every swarm in the spec.",
    )
    parser.add_argument(
        "--integrator", type=str, default=None,
        help="Override REBOUND integrator (e.g. whfast, ias15, mercurius, trace, saba, leapfrog, janus, bs).",
    )
    parser.add_argument(
        "--dt", type=float, default=None,
        help="Override manual timestep dt.",
    )
    parser.add_argument(
        "--archive", type=str, default=None,
        help="Also write a native REBOUND SimulationArchive (.bin) to this path during the run.",
    )
    parser.add_argument(
        "--from-archive", type=str, default=None,
        help="Convert an existing REBOUND SimulationArchive (.bin) into a Blender .npz bake instead of running a spec.",
    )
    parser.add_argument(
        "--units", type=str, default="CODE",
        help="Unit system key when importing with --from-archive (CODE, AU_YR2PI_MSUN, AU_YR_MSUN, AU_DAY_MSUN, SI).",
    )
    parser.add_argument(
        "--no-energy", action="store_true",
        help="Skip energy error evaluation.",
    )
    parser.add_argument(
        "--compress", action="store_true",
        help="Use ZIP deflate compression when writing the output .npz file.",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Custom display name for this bake in Blender's Bake History.",
    )
    parser.add_argument(
        "--deps-dir", type=str, default=None,
        help="Optional directory containing the rebound package if not on standard PYTHONPATH.",
    )

    args = parser.parse_args(argv)
    if args.deps_dir and os.path.isdir(args.deps_dir) and args.deps_dir not in sys.path:
        sys.path.insert(0, args.deps_dir)

    core = _import_core()
    try:
        import rebound
    except ImportError as ex:
        sys.stderr.write(f"Error: Could not import 'rebound' ({ex}). Install with: pip install rebound numpy\n")
        return 1

    t0 = time.perf_counter()

    if args.from_archive:
        print(f"Loading REBOUND SimulationArchive '{args.from_archive}'...")
        max_f = args.frames or 0
        run, meta = core.load_simulation_archive(rebound, args.from_archive, units=args.units, max_frames=max_f)
    else:
        if not args.spec:
            parser.error("Provide either a spec .json path or --from-archive <archive.bin>.")
        spec, meta = load_spec_file(args.spec)
        if args.swarm_count is not None:
            for sw in spec.swarms:
                sw.count = max(0, int(args.swarm_count))
        if args.integrator is not None:
            spec.integrator = args.integrator
        if args.dt is not None:
            spec.dt_mode = "MANUAL"
            spec.dt = float(args.dt)
        if args.archive is not None:
            spec.archive_path = os.path.abspath(args.archive)

        P_ref = core.characteristic_period(spec, body_name=args.ref_body)
        total_time = None
        if args.orbits is not None:
            if not P_ref or P_ref <= 0:
                parser.error("Could not determine a bound orbital period for --orbits.")
            total_time = float(args.orbits) * P_ref
        elif args.duration_time is not None:
            ufac = core.time_unit_factor(spec.units, args.duration_unit)
            total_time = float(args.duration_time) * ufac

        if args.samples_per_orbit is not None:
            if not P_ref or P_ref <= 0:
                parser.error("Could not determine a bound orbital period for --samples-per-orbit.")
            if total_time is None:
                total_time = max(1, spec.n_frames - 1) * float(spec.time_per_frame)
            n_orb = total_time / P_ref
            spec.n_frames = max(2, int(round(n_orb * float(args.samples_per_orbit))) + 1)
            spec.time_per_frame = total_time / (spec.n_frames - 1)
        elif total_time is not None:
            if args.frames is not None:
                spec.n_frames = max(2, int(args.frames))
                spec.time_per_frame = total_time / (spec.n_frames - 1)
            elif args.time_per_frame is not None:
                spec.time_per_frame = float(args.time_per_frame)
                spec.n_frames = max(2, int(round(total_time / spec.time_per_frame)) + 1)
            else:
                spec.n_frames = max(2, int(spec.n_frames))
                spec.time_per_frame = total_time / (spec.n_frames - 1)
        else:
            if args.frames is not None:
                spec.n_frames = max(1, int(args.frames))
            if args.time_per_frame is not None:
                spec.time_per_frame = float(args.time_per_frame)

        S_total = sum(max(0, int(sw.count)) for sw in spec.swarms)
        eff_workers = core.resolve_worker_count(
            spec, requested=args.workers, parallel=not args.no_parallel
        )
        print(
            f"REBOUND {getattr(rebound, '__version__', '?')} | "
            f"integrator={spec.integrator} | frames={spec.n_frames} | "
            f"bodies={len(spec.bodies)} | swarm_particles={S_total} | workers={eff_workers}"
        )

        last_print = [0.0]

        def _progress(p: float):
            now = time.perf_counter()
            if p < 1.0 and (now - last_print[0]) < 0.25:
                return
            last_print[0] = now
            elapsed = now - t0
            eta = (elapsed / p * (1.0 - p)) if p > 1e-3 else 0.0
            sys.stdout.write(
                f"\r  Progress: {100.0 * p:5.1f}%  elapsed: {elapsed:6.1f}s  ETA: {eta:6.1f}s"
            )
            sys.stdout.flush()

        run = core.run_simulation(
            rebound,
            spec,
            n_workers=args.workers,
            parallel=not args.no_parallel,
            n_probes=args.probes,
            compute_energy=not args.no_energy,
            progress_cb=_progress,
            deps_dir=args.deps_dir,
        )
        sys.stdout.write("\n")

    wall = time.perf_counter() - t0
    run.wall = wall
    rel = run.relative_energy_error()
    final_e = float(rel[rel == rel][-1]) if (rel == rel).any() else -1.0
    split = getattr(run, "split", None)
    split_err = float(max(split.max_rel_body, split.max_rel_probe)) if (split and split.n_chunks > 1) else -1.0
    lt = core.UNIT_LABELS.get(run.spec.units, ("L", "T", "M"))[1]
    n_chunks = getattr(run, "n_chunks", 1)
    mode_tag = f" ({n_chunks} workers)" if n_chunks > 1 else ""
    info = f"Baked {run.F} frames (t = {run.times[-1]:.6g} {lt}) in {wall:.2f} s{mode_tag}, dt = {run.dt:.4g}"
    if final_e >= 0:
        info += f", |ΔE/E₀| = {final_e:.2e}"
    if split_err >= 0:
        info += f", split Δr/r = {split_err:.2e}"

    meta = dict(meta or {})
    if args.label:
        meta["label"] = args.label
    elif "label" not in meta:
        base = os.path.splitext(os.path.basename(args.spec or args.from_archive or "external"))[0]
        meta["label"] = f"{base} ({run.spec.integrator}, {run.F}f)"
    meta["created_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta["info"] = info
    meta["energy_error"] = final_e
    meta["split_error"] = split_err

    out_path = core.save_run_npz(args.output, run, meta=meta, compressed=args.compress)
    _safe_print(info)
    for m in run.messages:
        _safe_print(f"  Note: {m}")
    _safe_print(f"Saved bake to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
