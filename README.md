# REBOUND N-body for Blender

Set up N-body systems in the viewport, preview their paths live, then bake
[REBOUND](https://github.com/hannorein/rebound) integrations to keyframes with
glowing trails, animated osculating orbits and swarms of thousands of test particles.

Tested headlessly with **Blender 5.2.2 LTS + REBOUND 5.2.2** (also supports Blender 4.4+ and REBOUND 4.x).

## Install

1. Blender → *Edit › Preferences › Get Extensions* → ▾ menu → **Install from Disk…** → pick `rebound_nbody-1.0.0.zip`.
2. Enable *Preferences › System › Network › Allow Online Access* (needed once for pip).
3. Press **N** in the 3D viewport → **REBOUND** tab → **Install REBOUND** (runs `pip install rebound` into the add-on's own folder; no admin rights needed).

## Quick start

1. Pick a preset → **Load** → **Bake Simulation** → press Play.
2. Or build your own: **Add Body** (first = star, then planets with orbital elements), or select any objects → **Make Selected Bodies**.
3. Turn on **Live Preview** to see the predicted paths and velocity arrows update as you drag bodies or change values.
4. Every bake is stored in **Bake History & External** (and saved as a `.npz` file in `//rebound_bakes/`). You can tweak parameters and bake new iterations without losing previous simulations, click any saved bake in the list to preview it, or click **Restore Setup** to branch a new simulation from an older bake's initial conditions. Use **Unbake Viewport** (loop-back icon) when you want to drag Cartesian bodies in the 3D viewport again.

## Bake History & External / Cluster Simulations

### Managing multiple bakes in Blender
- **Persistent Bake History**: Each bake is automatically recorded in the **Bake History & External** panel and saved to `//rebound_bakes/*.npz` alongside your `.blend` file.
- **Instant Switching & Comparison**: Click any bake (or its radio button) in the history list to immediately load its trajectories, trails, osculating orbits, swarm point clouds, and Graph Editor diagnostics into the viewport. Double-click a bake's name to rename it.
- **Iterate Without Losing Work**: Edit body parameters (mass, orbital elements, velocity, initial location) or simulation settings directly and click **Bake Simulation** to add a new iteration, or click **Restore Setup** on any historical bake to revert the scene's initial conditions to that run.
- **Re-project Visualization Instantly**: Change Reference Frame (Inertial, Centred, Co-rotating), Length Scale, Trails, or Osculating Orbits and click **Update Visualization** to re-compute all keyframes and geometry from the active bake without re-running REBOUND.

### Running simulations outside Blender (University / HPC Clusters)
`core.py`, `worker.py`, and `cli.py` are **100% pure Python + NumPy + REBOUND** (no `bpy` dependency) and include the full multicore swarm parallelization and scientific split-drift verification.

1. **Export Setup from Blender**: Set up or prototype your system in Blender and click **Export Spec (.json)** in the *Bake History & External* panel.
2. **Run on Cluster / Server**: Copy `core.py`, `worker.py`, `cli.py`, and your `.json` spec to the remote machine (requires only `pip install rebound numpy`) and run:
   ```bash
   # Run across 32 CPU worker processes (or -w 0 for auto CPU detection):
   python cli.py my_system.json -o cluster_run.npz --workers 32

   # Scale up particle count and frame count on the cluster:
   python cli.py my_system.json -o hires_disk.npz -w 32 --swarm-count 50000 --frames 2000 --label "Disk 50k"

   # Or convert a native REBOUND SimulationArchive (.bin) from any custom script:
   python cli.py --from-archive archive.bin -o archive_bake.npz
   ```
3. **Import & Interpret in Blender**: Copy the `.npz` (or `.bin` `SimulationArchive`) file back to your computer and click **Import (.npz / .bin)** (or place `.npz` files into `//rebound_bakes/` and click **Scan Folder**). The add-on automatically creates or links the bodies and swarm meshes, writes keyframes, trails, and osculating ellipses, and lets you switch reference frames or inspect orbital elements in the Graph Editor.

## Presets

| Preset | What it shows |
|---|---|
| Star + 2 Planets | Orbital-element input, osculating orbits |
| Figure-Eight | Chenciner–Montgomery three-body choreography |
| Pythagorean Three-Body | Burrau's problem (3-4-5 masses at rest), close encounters, ejection |
| Trojan Swarms | 800 test particles around L4/L5 in the co-rotating frame |
| Planetesimal Disk | 2000 test particles sculpted by two planets |
| Solar System / Outer Solar System | REBOUND's built-in datasets (offline) |

## Main settings

- **Simulation**: unit system (code units with your own G, AU/yr/2π/Msun with G = 1, AU/yr, AU/day, SI), integrator (IAS15, WHFast, MERCURIUS, TRACE, SABA, Leapfrog, JANUS, BS) with REBOUND's own description shown in the panel, automatic timestep (a fraction of the shortest orbital period) or a manual one, collisions (merge / bounce / stop), and **Parallel Swarm Bake** (splits massless test-particle swarms across multiple CPU worker processes, with a toggle to disable for strict single-process runs).
- **Time & Retiming**:
  - **Scientific Duration Setup (Before Bake)**: Choose duration by *Timeline Frames*, *Orbital Periods* (N orbits of a reference body or shortest bound orbit), or *Physical Time* (Simulation Units, Years `yr`, Kiloyears `kyr`, Megayears `Myr`, Days `d`, Seconds `s`).
  - **Sampling Cadence**: Record one snapshot per timeline frame, a fixed total number of snapshots, *N snapshots per reference orbit* (e.g. `30` for smooth trajectories or `1` for stroboscopic resonance maps), or a fixed $\Delta t$.
  - **Post-Bake Retiming & Time Window**: After baking (or importing an external simulation), map the stored snapshots to *1 Snapshot = 1 Frame*, *Fit to Total Frames*, *Frames per Orbit* (e.g. `60` frames per reference period), or *Frames per Time Unit*, and optionally slice a `[Start, End]` time window — then click **Apply Retiming to Active Bake** (`Update Visualization`) with zero extra `.npz` disk usage.
- **Visualization**: length scale (Blender units per length unit), reference frame (inertial, centred on a body, or co-rotating with a pair), trails (length, fade, taper, glow), osculating ellipses.
- **Data & Diagnostics**:
  - a, e and inclination keyframed as custom properties `rb_a`, `rb_e`, `rb_inc` on each body (plot them in the Graph Editor)
  - |ΔE/E₀| keyframed as the scene property `rb_energy_error`
  - **Parallel Split Drift** `Δr/r` keyframed as the scene property `rb_split_error`: compares massive bodies and shared float64 probe test particles across parallel workers to verify scientific accuracy when using adaptive integrators (IAS15, BS)
  - CSV export (raw simulation coordinates and units)
  - optional REBOUND SimulationArchive, which you can open in Python with `rebound.Simulationarchive(path)`
- **Overlay / Time label**: viewport text with the simulation time and energy error. **Add Time Label** adds a text object that updates every frame, including in renders.
- **JPL Horizons**: import real bodies by name and date. This needs internet access and was not live-tested.

## How it works

- `core.py` contains no Blender code. It builds the REBOUND simulation, integrates frame by frame, and computes reference frames and osculating ellipses with numpy.
- Body motion is written as linear keyframes (Blender 5 slotted actions).
- Trails are poly curves trimmed to the current frame by the "REBOUND Trail" Geometry Nodes group, with a `rb_fade` attribute that drives transparency in the material.
- Each orbit is a unit circle mapped onto the ellipse by an animated empty (location = ellipse centre, rotation = perifocal frame, scale = a, b). This keeps the line thickness uniform and works in renders.
- Swarm particles are stored as one vertex per particle per frame. The "REBOUND Swarm" node group shows only the current frame, plus an optional tail of earlier frames.
- Bodies are added to REBOUND sorted by their *Order* value and then by distance from the heaviest body. This is the natural order for Jacobi coordinates, which WHFast uses.

## Notes

- Swarm particles are massless test particles.
- Object scale is visual only. The *Physical Radius* setting is what collisions use.
- After baking, re-bake to apply changed visualization settings.
- Large swarms × long frame ranges use memory: 2000 particles × 600 frames ≈ 1.2 M vertices.
