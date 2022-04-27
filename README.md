# FastIce.jl
Parallel (multi-)xPU iterative fast ice flow solvers

#### 🌎 For **GeoData** (input) handling, refer to [GeoData](GeoData) folder

## Steady glacier 3D M multi-xPU

> [`SteadyStateGlacier3D_xpu.jl`](scripts3D/SteadyStateGlacier3D_xpu.jl) code

Mechanical viscous steady-state flow with stress-free surface over Alpine topography in 3D

<img src="docs/images/rhone3D_1022x1022x766.png" alt="Steady-state Rhone glacier in 3D" width="800">

Multi-xPU ice flow solver and [GeoData helpers](GeoData). Here resolving steady viscous ice flow for the Rhone glacier in th Swiss Alps on 1022x1022x766 grid points on 8 Nvidia Tesla A100 server _Superzack_, hosted at VAW, ETH Zurich.

## Steady glacier 3D TM multi-xPU

> [`SteadyStateGlacier3D_TM_xpu.jl`](scripts3D/SteadyStateGlacier3D_TM_xpu.jl) code

Thermo-mechanical viscous flow with stress-free surface over synthetic topography in 3D

<img src="docs/images/synthetic_turtle3D.png" alt="Thermo-mechanical iceflow in 3D" width="800">

Multi-XPU thermo-mechanical ice flow solver.

## Steady glacier 2D

> [`SteadyStateGlacier2D.jl`](scripts/SteadyStateGlacier2D.jl) code

Mechanical viscous steady-state flow with stress-free surface over bumpy bed in 2D.

<img src="docs/images/SteadyStateGlacier2D.png" alt="Steady-state glacier in 2D" width="800">


## Refs
Reference [list](/docs/references.md) (to be updated)
