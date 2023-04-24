using FastIce
using Logging
using MPI
using ImplicitGlobalGrid
using TinyKernels
using HDF5
using LightXML
using CairoMakie

include("load_dem.jl")
include("signed_distances.jl")
include("level_sets.jl")
include("volume_fractions.jl")
include("bcs.jl")
include("stokes.jl")
include("data_io.jl")
include("hide_communication.jl")

@views av1(A) = 0.5 .* (A[1:end-1] .+ A[2:end])
@views inn_x(A) = A[2:end-1, :, :]
@views inn_y(A) = A[:, 2:end-1, :]
@views inn_z(A) = A[:, :, 2:end-1]
@views inn(A) = A[2:end-1, 2:end-1, 2:end-1]

@views function main(grid_dims,grid)
    # unpack values
    me, dims, nprocs, coords, comm_cart = grid

    # init logger
    global_logger(FastIce.Logging.MPILogger(0, comm_cart, global_logger()))

    # path to DEM data
    greenland_path = "data/BedMachine/greenland.jld2"

    # region to simulate
    global_region = (xlims=(1100.0e3, 1200.0e3), ylims=(1000.0e3, 1100.0e3))

    # load DEM
    @info "loading DEM data from the file '$greenland_path'"
    (; x, y, bed, surface) = load_dem(greenland_path, global_region)
    @info "DEM resolution: $(size(bed,1)) × $(size(bed,2))"

    @info "plot DEMs"
    if me == 0
        fig = Figure(resolution=(2000,700),fontsize=32)
        ax  = (
            bed = Axis(fig[1,1][1,1];aspect=DataAspect(),title="bedrock",xlabel="x",ylabel="y"),
            ice = Axis(fig[1,2][1,1];aspect=DataAspect(),title="ice"    ,xlabel="x",ylabel="y"),
        )
        plt = (
            bed = heatmap!(ax.bed,x,y,bed    ;colormap=:terrain),
            ice = heatmap!(ax.ice,x,y,surface;colormap=:terrain),
        )
        Colorbar(fig[1,1][1,2],plt.bed)
        Colorbar(fig[1,2][1,2],plt.ice)
        save("region.png",fig)
    end

    # compute origin and size of the domain (required for scaling and computing the grid size)
    ox, oy, oz = x[1], y[1], minimum(bed)
    lx = x[end] - ox
    ly = y[end] - oy
    lz = maximum(surface) - oz

    # shift and scale the domain before computation (center of the domain is (0,0) in x-y plane)
    δx, δy = ox + 0.5lx, oy + 0.5ly # required to avoid conversion to Vector  
    x = @. (x - δx) / lz
    y = @. (y - δy) / lz
    @. bed = (bed - oz) / lz
    @. surface = (surface - oz) / lz

    @. surface -= 0.05

    # run simulation
    dem_data = (; x, y, bed, surface)
    @info "running the simulation"
    run_simulation(dem_data, grid_dims, me, dims, coords)

    return
end

@views function run_simulation(dem_data, grid_dims, me, dims, coords)
    # physics
    # global domain origin and size
    ox_g, oy_g, oz_g = dem_data.x[1], dem_data.y[1], 0.0
    lx_g = dem_data.x[end] - ox_g
    ly_g = dem_data.y[end] - oy_g
    lz_g = 1.0

    # local domain size and origin
    lx_l, ly_l, lz_l = (lx_g, ly_g, lz_g) ./ dims
    ox_l, oy_l, oz_l = (ox_g, oy_g, oz_g) .+ coords .* (lx_l, ly_l, lz_l)

    ####################################################################
    # non-dimensional numbers
    α       = deg2rad(0)   # slope
    nglen   = 3              # Glen's law power exponent
    ρr      = 0.92           # density ratio of ice to water
    cpr     = 0.5            # heat capacity ratio of ice to water
    U_P     = 60.0           # ratio of sensible heat to gravitational potential energy
    L_P     = 37.0           # ratio of latent heat to gravitational potential energy
    Pr      = 2e-9           # Prandtl number - ratio of thermal diffusivity to momentum diffusivity
    A_L     = 5e-2           # ratio of bump amplitude to length scale
    nbump   = 10             # number of bumps
    Q_RT    = 2*26.0         # ratio of activation temperature to melting temperature
    # dimensionally independent parameters
    K       = 1.0            # consistency                   [Pa*s^(1/n)]
    ρg      = 1.0            # ice gravity pressure gradient [Pa/m      ]
    T_mlt   = 1.0            # ice melting temperature       [K         ]
    # scales
    l̄       = lz_g           # length scale                  [m         ]
    σ̄       = ρg*cos(α)*l̄    # stress scale                  [Pa        ]
    t̄       = (K/σ̄)^nglen    # time scale                    [s         ]
    T̄       = T_mlt          # temperature scale             [K         ]
    # dimensionally dependent
    lx      = lx_lz*lz       # domain length                 [m         ]
    λ_i     = Pr*σ̄*l̄^2/(T̄*t̄) # thermal conductivity          [W/m/K     ]
    ρcp     = U_P*σ̄/T̄        # ice heat capacity             [Pa/K      ]
    ρL      = L_P*σ̄          # latent heat of melting        [Pa        ]
    Q_R     = Q_RT*T_mlt     # activational temperature      [K         ]
    T_atm   = 0.9*T_mlt      # atmospheric temperature       [K         ]
    T_ini   = 0.9*T_mlt      # initial surface temperature   [K         ]
    amp     = A_L*l̄          # bump amplitude                [m         ]
    rgl     = 1.2lz          # glacier radius                [m         ]
    ηreg    = 0.5*K*(1e-6/t̄)^(1/nglen-1)
    # not important (cancels in the equations)
    ρ_w     = 1.0            # density of water              [kg/m^3    ]
    ρ_i     = ρr*ρ_w         # density of ice                [kg/m^3    ]
    cp_i    = ρcp/ρ_i        # heat capacity of ice          [J/kg/K    ]
    cp_w    = cp_i/cpr       # heat capacity of ice          [J/kg/K    ]
    L       = ρL/ρ_w         # latent heat of melting        [J/kg      ]
    # phase data
    ρ  = (ice = ρ_i , wat = ρ_w )
    cp = (ice = cp_i, wat = cp_w)
    λ  = (ice = λ_i , wat = λ_i )
    # body force
    f  = (x = ρg*sin(α), y= 0ρg, z = ρg*cos(α))
    # thermodynamics
    @inline u_ice(T)  = cp.ice*(T-T_mlt)
    @inline u_wat(T)  = L + cp.wat*(T-T_mlt)
    @inline T_lt(u_t) = (u_t < u_ice(T_mlt)) ? T_mlt + u_t/cp.ice :
                        (u_t > u_wat(T_mlt)) ? T_mlt + (u_t - L)/cp.wat : T_mlt
    @inline ω_lt(u_t) = (u_t < u_ice(T_mlt)) ? 0.0 :
                        (u_t > u_wat(T_mlt)) ? 1.0 : ρ.ice*(u_ice(T_mlt) - u_t)/(ρ.ice*(u_ice(T_mlt)-u_t) - ρ.wat*(u_wat(T_mlt)-u_t))
    ####################################################################

    # numerics
    nx, ny, nz       = grid_dims
    nx_l, ny_l, nz_l = grid_dims .+ 2 # include ghost nodes
    nx_g, ny_g, nz_g = grid_dims.*dims
    bwidth = (8, 4, 4)

    ϵtol   = (1e-4,1e-4,1e-4,1e-4)
    maxiter = 50max(nx,nz)
    ncheck  = ceil(Int,0.5max(nx,nz))
    nviz    = 1
    nsave   = 5
    nt      = 500
    χ       = 5e-3

    # preprocessing
    dx, dy, dz = lx_g / nx_g, ly_g / ny_g, lz_g / nz_g
    @info "grid spacing: dx = $dx, dy = $dy, dz = $dz"

    # take into account ghost nodes to simplify model setup
    xv_l = LinRange(ox_l - dx, ox_l + lx_l + dx, nx_l + 1)
    yv_l = LinRange(oy_l - dy, oy_l + ly_l + dy, ny_l + 1)
    zv_l = LinRange(oz_l - dz, oz_l + lz_l + dz, nz_l + 1)
    xc_l, yc_l, zc_l = av1.((xv_l, yv_l, zv_l))

    # PT params
    r = 0.7
    lτ_re_mech = 1.5min(lx_g, ly_g, lz_g) / π
    vdτ = min(dx, dy, dz) / sqrt(8.1)
    θ_dτ = lτ_re_mech * (r + 4 / 3) / vdτ
    nudτ = vdτ * lτ_re_mech
    dτ_r = 1.0 / (θ_dτ + 1.0)

    # fields allocation
    # level set
    Ψ = (
        not_solid=scalar_field(Float64, nx_l + 1, ny_l + 1, nz_l + 1),
        not_air  =scalar_field(Float64, nx_l + 1, ny_l + 1, nz_l + 1),
    )
    wt = (
        not_solid=volfrac_field(Float64, nx_l, ny_l, nz_l),
        not_air  =volfrac_field(Float64, nx_l, ny_l, nz_l),
    )
    # mechanics (stress fields include ghost nodes due to redundant computations on distributed staggered grid)
    Pr = scalar_field(Float64, nx_l, ny_l, nz_l)
    τ  = tensor_field(Float64, nx_l, ny_l, nz_l)
    ε̇  = tensor_field(Float64, nx_l, ny_l, nz_l)
    V  = vector_field(Float64, nx_l, ny_l, nz_l)
    ηs = scalar_field(Float64, nx_l, ny_l, nz_l)
    # thermal
    ρU = scalar_field(Float64,nx,ny,nz)
    T  = scalar_field(Float64,nx,ny,nz)
    qT = vector_field(Float64,nx,ny,nz)
    # hydro
    ω  = scalar_field(Float64,nx,ny,nz)
    # residuals
    Res = (
        Pr=scalar_field(Float64, nx, ny, nz),
        V =vector_field(Float64, nx, ny, nz)
    )
    # visualisation
    Vmag = scalar_field(Float64, nx, ny, nz)
    τII  = scalar_field(Float64, nx, ny, nz)
    Ψav = (
        not_air=scalar_field(Float64,nx,ny,nz),
        not_solid=scalar_field(Float64,nx,ny,nz),
    )

    # initialisation 
    # compute level sets from DEM data
    dem_grid = (dem_data.x, dem_data.y)
    Ψ_grid = (xv_l, yv_l, zv_l)

    @info "computing the level set for the ice surface"
    compute_level_set_from_dem!(Ψ.not_air, to_device(dem_data.surface), dem_grid, Ψ_grid)

    @info "computing the level set for the bedrock surface"
    compute_level_set_from_dem!(Ψ.not_solid, to_device(dem_data.bed), dem_grid, Ψ_grid)
    TinyKernels.device_synchronize(get_device())
    # invert level set to set what's below the DEM surface as inside
    @. Ψ.not_solid *= -1.0
    TinyKernels.device_synchronize(get_device())

    @info "computing volume fractions from level sets"
    for phase in eachindex(Ψ)
        compute_volume_fractions_from_level_set!(wt[phase], Ψ[phase], dx, dy, dz)
    end
    TinyKernels.device_synchronize(FastIce.get_device())

    @info "initialize mechanics"
    for comp in eachindex(V) fill!(V[comp], 0.0) end
    for comp in eachindex(τ) fill!(τ[comp], 0.0) end
    fill!(Pr, 0.0)
    fill!(ηs,0.5*K*(1e-1/t̄)^(1/nglen-1)*exp(-1/nglen*Q_R*(1/T_mlt-1/T_ini)))   
    
    @info "initialize thermo"
    for comp in eachindex(qT) fill!(qT[comp],0.0) end
    @. T  = lerp(T_atm,T_ini,wt.not_air.c) 
    @. ρU = ρ.ice*u_ice(T)
    @. ω  = ω_lt(ρU/ρ.ice)
    TinyKernels.device_synchronize(FastIce.get_device())

    # save static data
    outdir = joinpath("out_visu","egu2023/greenland")
    mkpath(outdir)
    jldsave(joinpath(outdir,"static.h5");xc,xv,yc,yv,zc,zv,Ψ,wt,dem_data)
    tcur = 0.0; isave = 1
    for it in 1:nt
        @info @sprintf("time step #%d, time = %g",it,tcur)
        empty!(iter_evo); resize!(errs_evo,(length(ϵtol),0))
        TinyKernels.device_synchronize(FastIce.get_device())
        # mechanics
        for iter in 1:maxiter
            update_σ!(Pr, τ, ε̇, V, ηs, wt, r, θ_dτ, dτ_r, dx, dy, dz)
            TinyKernels.device_synchronize(FastIce.get_device())
            update_V!(V, Pr, τ, ηs, wt, nudτ, f, dx, dy, dz; bwidth)
            TinyKernels.device_synchronize(FastIce.get_device())
            update_ηs!(ηs,ε̇,T,wt,K,nglen,Q_R,T_mlt,ηreg,χ)
            TinyKernels.device_synchronize(FastIce.get_device())
            if iter % ncheck == 0
                compute_residual!(Res,Pr,V,τ,wt,f,dx,dy,dz)
                TinyKernels.device_synchronize(FastIce.get_device())
                errs = (maximum(abs.(Res.V.x))*l̄/σ̄,
                        maximum(abs.(Res.V.y))*l̄/σ̄,
                        maximum(abs.(Res.V.z))*l̄/σ̄,
                        maximum(abs.(inn(Res.Pr)))*t̄)
                TinyKernels.device_synchronize(FastIce.get_device())
                @printf("  iter/nz # %2.1f, errs: [ Vx = %1.3e, Vy = %1.3e, Vz = %1.3e, Pr = %1.3e ]\n", iter/nz, errs...)
                push!(iter_evo, iter/nz); append!(errs_evo, errs)
                # check convergence
                if any(.!isfinite.(errs)) error("simulation failed") end
                if all(errs .< ϵtol) break end
            end
        end
        TinyKernels.device_synchronize(FastIce.get_device())
        dt = min(dx,dy,dz)^2/max(λ.ice*ρ.ice*cp.ice,λ.wat*ρ.wat*cp.wat)/6.1
        # thermal
        update_qT!(qT,T,wt,λ,T_atm,dx,dy,dz)
        TinyKernels.device_synchronize(FastIce.get_device())
        update_ρU!(ρU,qT,τ,ε̇,wt,ρ.ice*u_ice(T_atm),dt,dx,dy,dz)
        TinyKernels.device_synchronize(FastIce.get_device())
        @. T = T_lt(ρU/(ρ.ice*(1-ω) + ρ.wat*ω))
        @. ω = ω_lt(ρU/(ρ.ice*(1-ω) + ρ.wat*ω))
        TinyKernels.device_synchronize(FastIce.get_device())
        tcur += dt
        # save timestep
        if it % nsave == 0
            jldsave(joinpath(outdir,@sprintf("%04d.h5",isave));Pr,τ,ε̇,ε̇II,V,T,ω,ηs)
            isave += 1
        end
    end

    @info "saving results on disk"
    dim_g = (nx_g, ny_g, nz_g)
    update_vis_fields!(Vmag, τII, Ψav, V, τ, Ψ)
    out_h5 = "results.h5"
    ndrange = CartesianIndices(((coords[1]*nx+1):(coords[1]+1)*nx,
                                (coords[2]*ny+1):(coords[2]+1)*ny,
                                (coords[3]*nz+1):(coords[3]+1)*nz))
    fields = Dict("LS_ice" => Ψav.not_air, "LS_bed" => Ψav.not_solid, "Vmag" => Vmag, "TII" => τII, "Pr" => inn(Pr))
    @info "saving HDF5 file"
    write_h5(out_h5, fields, dim_g, ndrange)

    @info "saving XDMF file..."
    (me == 0) && write_xdmf("results.xdmf3", out_h5, fields, (xc_l[2], yc_l[2], zc_l[2]), (dx, dy, dz), dim_g)

    return
end

@tiny function _kernel_update_vis_fields!(Vmag, τII, Ψav, V, τ, Ψ)
    ix, iy, iz = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy, iz)
    @inbounds if isin(Ψ.not_air)
        pav = 0.0
        for idz = 1:2, idy = 1:2, idx = 1:2
            pav += Ψ.not_air[ix+idx, iy+idy, iz+idz]
        end
        Ψav.not_air[ix, iy, iz] = pav / 8
    end
    @inbounds if isin(Ψ.not_solid)
        pav = 0.0
        for idz = 1:2, idy = 1:2, idx = 1:2
            pav += Ψ.not_solid[ix+idx, iy+idy, iz+idz]
        end
        Ψav.not_solid[ix, iy, iz] = pav / 8
    end
    @inbounds if isin(Vmag)
        vxc = 0.5 * (V.x[ix+1, iy+1, iz+1] + V.x[ix+2, iy+1, iz+1])
        vyc = 0.5 * (V.y[ix+1, iy+1, iz+1] + V.y[ix+1, iy+2, iz+1])
        vzc = 0.5 * (V.z[ix+1, iy+1, iz+1] + V.z[ix+1, iy+1, iz+2])
        Vmag[ix, iy, iz] = sqrt(vxc^2 + vyc^2 + vzc^2)
    end
    @inbounds if isin(τII)
        τxyc = 0.25 * (τ.xy[ix, iy, iz] + τ.xy[ix+1, iy, iz] + τ.xy[ix, iy+1, iz] + τ.xy[ix+1, iy+1, iz])
        τxzc = 0.25 * (τ.xz[ix, iy, iz] + τ.xz[ix+1, iy, iz] + τ.xz[ix, iy, iz+1] + τ.xz[ix+1, iy, iz+1])
        τyzc = 0.25 * (τ.yz[ix, iy, iz] + τ.yz[ix, iy+1, iz] + τ.yz[ix, iy, iz+1] + τ.yz[ix, iy+1, iz+1])
        τII[ix, iy, iz] = sqrt(0.5 * (τ.xx[ix+1, iy+1, iz+1]^2 + τ.yy[ix+1, iy+1, iz+1]^2 + τ.zz[ix+1, iy+1, iz+1]^2) + τxyc^2 + τxzc^2 + τyzc^2)
    end
    return
end

const _update_vis_fields! = _kernel_update_vis_fields!(get_device())

function update_vis_fields!(Vmag, τII, Ψav, V, τ, Ψ)
    wait(_update_vis_fields!(Vmag, τII, Ψav, V, τ, Ψ; ndrange=axes(Vmag)))
    return
end

grid_dims = (1000, 1000, 50)

# init MPI and IGG
MPI.Init()
me, dims, nprocs, coords, comm_cart = init_global_grid(grid_dims...; init_MPI=false)
dims   = Tuple(dims)
coords = Tuple(coords)
grid   = (me,dims,nprocs,coords,comm_cart)

main(grid_dims,grid)

# finalize_global_grid(; finalize_MPI=false)
# MPI.Finalize()