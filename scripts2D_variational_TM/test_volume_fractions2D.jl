using FastIce
using TinyKernels
using HDF5
using LightXML
using UnicodePlots
using LinearAlgebra
using GeometryBasics
using CairoMakie
using ElasticArrays
using Printf
using JLD2

using CUDA
CUDA.device!(1)

include("geometry.jl")
include("signed_distances.jl")
include("level_sets.jl")
include("volume_fractions.jl")
include("bcs.jl")
include("stokes.jl")
include("thermo.jl")
include("data_io.jl")

@views inn_x(A) = A[2:end-1,:]
@views inn_y(A) = A[:,2:end-1]
@views inn(A)   = A[2:end-1,2:end-1]
@views av1(A)   = 0.5.*(A[1:end-1].+A[2:end])
@views av4(A)   = 0.25.*(A[1:end-1,1:end-1].+A[1:end-1,2:end].+A[2:end,1:end-1].+A[2:end,2:end])

# generate synthetic sinusoidal geometry with constant slope
# `ox` - domain origin in x
# `oz` - domain origin in z
# `lx` - domain extent
# `Δz` - offset of the bed from origin
# `amp` - amplitude of the bumps
# `ω` - frequency of the bumps
# `α` - slope
# `nx` - number of grid points
function generate_sinusoidal(ox,oz,lx,Δz,amp,ω,α,nx)
    xv = LinRange(0,lx,nx)
    zv = amp.*cos.((2π.*ω/lx).*xv) .+ tan(α).*xv
    zv .= zv .- minimum(zv) .+ Δz
    return Point2.(xv.+ox,zv.+oz)
end

# generate synthetic circle shape
# `ox` - domain origin in x
# `oz` - domain origin in z
# `r` - circle radius
# `nθ` - number of grid points
function generate_circle(ox,oz,r,θs,θe,nθ)
    θ  = LinRange(θs,θe*(1-1/nθ),nθ)
    zv,xv = r.*sin.(θ), r.*cos.(θ)
    return Point2.(xv.+ox,zv.+oz)
end

@views function run_simulation(nz)
    ## physics =========================================================================================================
    # non-dimensional numbers
    α       = deg2rad(-20)   # slope
    nglen   = 3              # Glen's law power exponent
    ρr      = 0.92           # density ratio of ice to water
    cpr     = 0.5            # heat capacity ratio of ice to water
    U_P     = 60.0           # ratio of sensible heat to gravitational potential energy
    L_P     = 37.0           # ratio of latent heat to gravitational potential energy
    Pr      = 2e-9           # Prandtl number - ratio of thermal diffusivity to momentum diffusivity
    A_L     = 5e-2           # ratio of bump amplitude to length scale
    lx_lz   = 3e0            # ratio of horizontal to vertical domain extents
    nbump   = 13             # number of bumps
    Q_RT    = 2*26.0         # ratio of activation temperature to melting temperature
    # dimensionally independent parameters
    lz      = 1.0            # domain size in z-direction    [m         ]
    K       = 1.0            # consistency                   [Pa*s^(1/n)]
    ρg      = 1.0            # ice gravity pressure gradient [Pa/m      ]
    T_mlt   = 1.0            # ice melting temperature       [K         ]
    # scales
    l̄       = lz             # length scale                  [m         ]
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
    ox,oz   = -0.5lx,0.0lz   # domain origin                 [m         ]
    rgl     = 1.2lz          # glacier radius                [m         ]
    ogx,ogz = 0.0lx,-0.3rgl  # glacier origin                [m         ]
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
    f  = (x = ρg*sin(α), y = ρg*cos(α))
    # thermodynamics
    @inline u_ice(T)  = cp.ice*(T-T_mlt)
    @inline u_wat(T)  = L + cp.wat*(T-T_mlt)
    @inline T_lt(u_t) = (u_t < u_ice(T_mlt)) ? T_mlt + u_t/cp.ice :
                        (u_t > u_wat(T_mlt)) ? T_mlt + (u_t - L)/cp.wat : T_mlt
    @inline ω_lt(u_t) = (u_t < u_ice(T_mlt)) ? 0.0 :
                        (u_t > u_wat(T_mlt)) ? 1.0 : ρ.ice*(u_ice(T_mlt) - u_t)/(ρ.ice*(u_ice(T_mlt)-u_t) - ρ.wat*(u_wat(T_mlt)-u_t))
    ## numerics ========================================================================================================
    nx      = ceil(Int,nz*lx/lz)
    ϵtol    = (1e-4,1e-4,1e-4)
    maxiter = 50max(nx,nz)
    ncheck  = ceil(Int,0.5max(nx,nz))
    nviz    = 5
    nsave   = 5
    nt      = 500
    # nviz    = 1
    # nsave   = 1
    # nt      = 1
    χ       = 5e-3
    ## preprocessing ===================================================================================================
    # grid spacing
    dx,dz = lx/nx,lz/nz
    @info "grid resolution: $nx × $nz"
    @info "grid spacing   : dx = $dx, dz = $dz"
    @info "generating DEM data"
    dem = (
        bed = generate_sinusoidal(ox       ,oz,lx       ,0.05lz,amp,nbump            ,0,nx),
        ice = generate_sinusoidal(ox-0.25lx,oz,lx+0.25lx,0.70lz,0  ,1    ,deg2rad(-3),nx),
        # ice = generate_circle(ogx,ogz,rgl,0,π,ceil(Int,π*rgl/dx))
    )
    TinyKernels.device_synchronize(FastIce.get_device())
    @info "computing marker chains"
    mc = (
        bed = to_device(push!(pushfirst!(copy(dem.bed),Point2(ox,0.0)),Point2(ox+lx,0.0))),
        ice = to_device(push!(pushfirst!(copy(dem.ice),Point2(ox-0.25lx,0.0)),Point2(ox+lx+0.25lx,0.0))),
        # ice = to_device(dem.ice)
    )
    # grid locations
    xv = LinRange(ox,ox+lx,nx+1)
    zv = LinRange(oz,oz+lz,nz+1)
    xc,zc = av1.((xv,zv))
    # PT params
    r          = 0.7
    lτ_re_mech = 1.5min(lx,lz)/π
    vdτ_mech   = min(dx,dz)/sqrt(5.1)
    θ_dτ       = lτ_re_mech*(r+4/3)/vdτ_mech
    nudτ       = vdτ_mech*lτ_re_mech
    dτ_r       = 1.0/(θ_dτ+1.0)
    ## fields allocation ===============================================================================================
    # level set
    Ψ = (
        not_solid = scalar_field(Float64,nx+1,nz+1),
        not_air   = scalar_field(Float64,nx+1,nz+1),
    )
    wt = (
        not_solid = volfrac_field(Float64,nx,nz),
        not_air   = volfrac_field(Float64,nx,nz),
    )
    # mechanics
    Pr = scalar_field(Float64,nx,nz)
    τ  = tensor_field(Float64,nx,nz)
    ε̇  = tensor_field(Float64,nx,nz)
    V  = vector_field(Float64,nx,nz)
    ηs = scalar_field(Float64,nx,nz)
    # thermal
    ρU = scalar_field(Float64,nx,nz)
    T  = scalar_field(Float64,nx,nz)
    qT = vector_field(Float64,nx,nz)
    # hydro
    ω  = scalar_field(Float64,nx,nz)
    # residuals
    Res = (
        Pr = scalar_field(Float64,nx  ,nz  ),
        V  = vector_field(Float64,nx-2,nz-2)
    )
    # visualisation
    Vmag = scalar_field(Float64,nx-2,nz-2)
    ε̇II  = scalar_field(Float64,nx-2,nz-2)
    Ψav  = (
        not_solid = scalar_field(Float64,nx-2,nz-2),
        not_air   = scalar_field(Float64,nx-2,nz-2),
    )
    ## initialisation ==================================================================================================
    # level set
    TinyKernels.device_synchronize(FastIce.get_device())
    for comp in eachindex(Ψ) fill!(Ψ[comp],Inf) end
    @info "computing the level set for the ice surface"
    compute_levelset!(Ψ.not_air,xv,zv,mc.ice)
    @info "computing the level set for the bedrock surface"
    compute_levelset!(Ψ.not_solid,xv,zv,mc.bed)
    TinyKernels.device_synchronize(FastIce.get_device())
    @. Ψ.not_solid *= -1.0
    TinyKernels.device_synchronize(FastIce.get_device())
    @info "computing volume fractions from level sets"
    for phase in eachindex(Ψ)
        compute_volume_fractions_from_level_set!(wt[phase],Ψ[phase],dx,dz)
    end
    TinyKernels.device_synchronize(FastIce.get_device())
    # mechanics
    for comp in eachindex(V) fill!(V[comp],0.0) end
    for comp in eachindex(τ) fill!(τ[comp],0.0) end
    for comp in eachindex(τ) fill!(ε̇[comp],0.0) end
    fill!(Pr,0.0)
    fill!(ηs,0.5*K*(1e-1/t̄)^(1/nglen-1)*exp(-1/nglen*Q_R*(1/T_mlt-1/T_ini)))    
    # fill!(ηs,1.0)
    TinyKernels.device_synchronize(FastIce.get_device())
    # thermo
    for comp in eachindex(qT) fill!(qT[comp],0.0) end
    @. T  = lerp(T_atm,T_ini,wt.not_air.c) 
    @. ρU = ρ.ice*u_ice(T)
    @. ω  = ω_lt(ρU/ρ.ice)
    TinyKernels.device_synchronize(FastIce.get_device())
    # convergence tracking
    iter_evo = Float64[]
    errs_evo = ElasticArray{Float64}(undef, length(ϵtol), 0)
    # figures
    fig = Figure(resolution=(3000,1200),fontsize=32)
    axs = (
        hmaps = (
            Pr   = Axis(fig[1,1][1,1];aspect=DataAspect(),title="p"  ),
            ε̇II  = Axis(fig[1,2][1,1];aspect=DataAspect(),title="ε̇II"),
            Vmag = Axis(fig[1,3][1,1];aspect=DataAspect(),title="|V|"),
            T    = Axis(fig[2,1][1,1];aspect=DataAspect(),title="T"  ),
            ω    = Axis(fig[2,2][1,1];aspect=DataAspect(),title="ω"  ),
            ηs   = Axis(fig[3,1][1,1];aspect=DataAspect(),title="ηs" ),
        ),
        errs = Axis(fig[2,3];yscale=log10, title="Convergence", xlabel="#iter/ny", ylabel="ϵ"),
    )
    for axname in eachindex(axs.hmaps)
        xlims!(axs.hmaps[axname],ox,ox+lx)
        ylims!(axs.hmaps[axname],oz,oz+lz)
    end
    plt = (
        hmaps = (
            Pr   = heatmap!(axs.hmaps.Pr  ,xv,zv,to_host(Pr        );colormap=:turbo),
            ε̇II  = heatmap!(axs.hmaps.ε̇II ,xv,zv,to_host(ε̇II       );colormap=:turbo),
            Vmag = heatmap!(axs.hmaps.Vmag,xv,zv,to_host(Vmag      );colormap=:turbo),
            T    = heatmap!(axs.hmaps.T   ,xc,zc,to_host(T         );colormap=:turbo),
            ω    = heatmap!(axs.hmaps.ω   ,xc,zc,to_host(ω         );colormap=:turbo),
            ηs   = heatmap!(axs.hmaps.ηs  ,xc,zc,to_host(log10.(ηs));colormap=:turbo,colorrange=(1,4)),
        ),
        errs=[scatterlines!(axs.errs, Point2.(iter_evo, errs_evo[ir, :])) for ir in eachindex(ϵtol)],
    )
    plt_bed = [
        (
            bed =  poly!(axs.hmaps[f],to_host(mc.bed);strokewidth=2,color=:black),
            ice = lines!(axs.hmaps[f],to_host(mc.ice);strokewidth=2,color=:black),
        ) for f in eachindex(axs.hmaps)
    ]
    Colorbar(fig[1,1][1,2],plt.hmaps.Pr)
    Colorbar(fig[1,2][1,2],plt.hmaps.ε̇II)
    Colorbar(fig[1,3][1,2],plt.hmaps.Vmag)
    Colorbar(fig[2,1][1,2],plt.hmaps.T)
    Colorbar(fig[2,2][1,2],plt.hmaps.ω)
    Colorbar(fig[3,1][1,2],plt.hmaps.ηs)
    TinyKernels.device_synchronize(FastIce.get_device())
    update_vis_fields!(Vmag,ε̇II,Ψav,V,ε̇,Ψ)
    plt.hmaps.Pr[3][]   = to_host(Pr)
    plt.hmaps.ε̇II[3][]  = to_host(ε̇II)
    plt.hmaps.Vmag[3][] = to_host(Vmag)
    plt.hmaps.T[3][]    = to_host(T)
    plt.hmaps.ω[3][]    = to_host(ω)
    plt.hmaps.ηs[3][]   = to_host(ηs)
    TinyKernels.device_synchronize(FastIce.get_device())
    display(fig)
    ## simulation run ==================================================================================================
    @info "time loop"
    # save static data
    outdir = joinpath("out_visu","egu2023",@sprintf("nbump_%d_slope_%.1f",nbump,rad2deg(α)))
    mkpath(outdir)
    jldsave(joinpath(outdir,"static.h5");xc,xv,zc,zv,Ψ,wt,dem,mc)
    tcur = 0.0; isave = 1
    for it in 1:nt
        @info @sprintf("time step #%d, time = %g",it,tcur)
        empty!(iter_evo); resize!(errs_evo,(length(ϵtol),0))
        TinyKernels.device_synchronize(FastIce.get_device())
        # mechanics
        for iter in 1:maxiter
            update_σ!(Pr,τ,ε̇,V,ηs,wt,r,θ_dτ,dτ_r,dx,dz)
            TinyKernels.device_synchronize(FastIce.get_device())
            update_V!(V,Pr,τ,ηs,wt,nudτ,f,dx,dz)
            TinyKernels.device_synchronize(FastIce.get_device())
            update_ηs!(ηs,ε̇,T,wt,K,nglen,Q_R,T_mlt,ηreg,χ)
            TinyKernels.device_synchronize(FastIce.get_device())
            if iter % ncheck == 0
                compute_residual!(Res,Pr,V,τ,wt,f,dx,dz)
                TinyKernels.device_synchronize(FastIce.get_device())
                errs = (maximum(abs.(Res.V.x))*l̄/σ̄,
                        maximum(abs.(Res.V.y))*l̄/σ̄,
                        maximum(abs.(inn(Res.Pr)))*t̄)
                TinyKernels.device_synchronize(FastIce.get_device())
                @printf("  iter/nz # %2.1f, errs: [ Vx = %1.3e, Vy = %1.3e, Pr = %1.3e ]\n", iter/nz, errs...)
                push!(iter_evo, iter/nz); append!(errs_evo, errs)

                # debug viz
                # for ir in eachindex(plt.errs)
                #     plt.errs[ir][1] = Point2.(iter_evo, errs_evo[ir, :])
                # end
                # autolimits!(axs.errs)
                # update_vis_fields!(Vmag,ε̇II,Ψav,V,ε̇,Ψ)
                # TinyKernels.device_synchronize(FastIce.get_device())
                # plt.hmaps.Pr[3][]   = to_host(Pr)
                # plt.hmaps.ε̇II[3][]  = to_host(ε̇II)
                # plt.hmaps.Vmag[3][] = to_host(Vmag)
                # plt.hmaps.T[3][]    = to_host(T)
                # plt.hmaps.ω[3][]    = to_host(ω)
                # plt.hmaps.ηs[3][]   = to_host(log10.(ηs))
                # yield()

                # check convergence
                if any(.!isfinite.(errs)) error("simulation failed") end
                if all(errs .< ϵtol) break end
            end
        end
        TinyKernels.device_synchronize(FastIce.get_device())
        dt = min(dx,dz)^2/max(λ.ice*ρ.ice*cp.ice,λ.wat*ρ.wat*cp.wat)/4.1
        # thermal
        update_qT!(qT,T,wt,λ,T_atm,dx,dz)
        TinyKernels.device_synchronize(FastIce.get_device())
        update_ρU!(ρU,qT,τ,ε̇,wt,ρ.ice*u_ice(T_atm),dt,dx,dz)
        TinyKernels.device_synchronize(FastIce.get_device())
        @. T = T_lt(ρU/(ρ.ice*(1-ω) + ρ.wat*ω))
        @. ω = ω_lt(ρU/(ρ.ice*(1-ω) + ρ.wat*ω))
        TinyKernels.device_synchronize(FastIce.get_device())
        tcur += dt
        # update figures
        if it % nviz == 0
            for ir in eachindex(plt.errs)
                plt.errs[ir][1] = Point2.(iter_evo, errs_evo[ir, :])
            end
            autolimits!(axs.errs)
            update_vis_fields!(Vmag,ε̇II,Ψav,V,ε̇,Ψ)
            TinyKernels.device_synchronize(FastIce.get_device())
            plt.hmaps.Pr[3][]   = to_host(Pr)
            plt.hmaps.ε̇II[3][]  = to_host(ε̇II)
            plt.hmaps.Vmag[3][] = to_host(Vmag)
            plt.hmaps.T[3][]    = to_host(T)
            plt.hmaps.ω[3][]    = to_host(ω)
            plt.hmaps.ηs[3][]   = to_host(log10.(ηs))
            display(fig)
            yield()
        end
        # save timestep
        if it % nsave == 0
            @info "saving timestep"
            update_vis_fields!(Vmag,ε̇II,Ψav,V,ε̇,Ψ)
            TinyKernels.device_synchronize(FastIce.get_device())
            jldsave(joinpath(outdir,@sprintf("%04d.jld2",isave));Pr,τ,ε̇,ε̇II,V,T,ω,ηs)
            isave += 1
        end
    end

    TinyKernels.device_synchronize(FastIce.get_device())
    @info "saving results on disk"
    out_h5 = "results.h5"
    ndrange = CartesianIndices((nx-2,nz-2))
    fields = Dict("LS_ice"=>Ψav.not_air,"LS_bed"=>Ψav.not_solid,"Vmag"=>Vmag,"TII"=>ε̇II,"Pr"=>inn(Pr))
    @info "saving HDF5 file"
    write_h5(out_h5,fields,(nx,nz),ndrange)

    @info "saving XDMF file..."
    write_xdmf("results.xdmf3",out_h5,fields,(xc[2],zc[2]),(dx,dz),(nx-2,nz,2))

    return
end

@tiny function _kernel_update_vis_fields!(Vmag, ε̇II, Ψav, V, ε̇, Ψ)
    ix,iz = @indices
    @inline isin(A) = checkbounds(Bool,A,ix,iz)
    @inbounds if isin(Ψav.not_air)
        pav = 0.0
        for idz = 0:1, idx = 0:1
            pav += Ψ.not_air[ix+idx,iz+idz]
        end
        Ψav.not_air[ix,iz] = pav/4
    end
    @inbounds if isin(Ψav.not_solid)
        pav = 0.0
        for idz = 0:1, idx = 0:1
            pav += Ψ.not_solid[ix+idx,iz+idz]
        end
        Ψav.not_solid[ix,iz] = pav/4
    end
    @inbounds if isin(Vmag)
        vxc = 0.5*(V.x[ix+1,iz+1] + V.x[ix+2,iz+1])
        vzc = 0.5*(V.y[ix+1,iz+1] + V.y[ix+1,iz+2])
        Vmag[ix,iz] = sqrt(vxc^2 + vzc^2)
    end
    @inbounds if isin(ε̇II)
        ε̇xzc = 0.25*(ε̇.xy[ix,iz]+ε̇.xy[ix+1,iz]+ε̇.xy[ix,iz+1]+ε̇.xy[ix+1,iz+1])
        ε̇II[ix,iz] = sqrt(0.5*(ε̇.xx[ix+1,iz+1]^2 + ε̇.yy[ix+1,iz+1]^2) + ε̇xzc^2)
    end
    return
end

const _update_vis_fields! = _kernel_update_vis_fields!(get_device())

function update_vis_fields!(Vmag, ε̇II, Ψav, V, ε̇, Ψ)
    wait(_update_vis_fields!(Vmag, ε̇II, Ψav, V, ε̇, Ψ; ndrange=axes(Vmag)))
    return
end

run_simulation(100)