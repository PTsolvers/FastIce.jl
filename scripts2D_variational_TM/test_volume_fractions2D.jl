using FastIce
using TinyKernels
using HDF5
using LightXML
using UnicodePlots
using LinearAlgebra
using GeometryBasics
using GLMakie
using ElasticArrays
using Printf

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
# `amp` - amplitude of the bumps
# `ω` - frequency of the bumps
# `α` - slope
# `nx` - number of grid points
function generate_sinusoidal_bed(ox,oz,lx,Δz,amp,ω,α,nx)
    xv = LinRange(0,lx,nx)
    zv = amp.*sin.((2π.*ω/lx).*xv) .+ tan(α).*xv
    zv .= zv .- minimum(zv) .+ Δz
    return Point2.(xv.+ox,zv.+oz)
end

# generate synthetic circle shape
# `ox` - domain origin in x
# `oz` - domain origin in z
# `r` - circle radius
# `nθ` - number of grid points
function generate_circle(ox,oz,r,nθ)
    θ  = LinRange(0,2π*(1-1/nθ),nθ)
    zv,xv = r.*sin.(θ), r.*cos.(θ)
    return Point2.(xv.+ox,zv.+oz)
end

@views function main(grid_dims)
    lx     = 2000.0
    ox,oz  = -lx/2, 0.0
    nbumps = 5
    amp    = 0.025lx
    α      = deg2rad(-15)
    rgl    = 0.4lx
    lr     = 2π*rgl
    nx     = grid_dims[1]
    dx     = lx/nx
    nθ     = ceil(Int,lr/dx)
    xv     = LinRange(ox,ox+lx,nx+1)
    xc     = av1(xv)

    # generate DEM
    @info "generating DEM data"
    bed = generate_sinusoidal_bed(ox,oz,lx,0.05lx,amp,nbumps,α,nx+1)
    pushfirst!(bed,Point2(ox,0.0))
    push!(bed,Point2(ox+lx,0.0))
    ice = generate_circle(-0.05lx,0.05lx,rgl,nθ)
    
    # run simulation
    dem_data = (;x=xc,bed,ice)
    @info "running the simulation"
    run_simulation(dem_data,grid_dims)

    return
end

@views function run_simulation(dem_data,grid_dims)
    # physics
    # dimensionally independent
    ox,oz = dem_data.x[1], 0.0
    lx    = dem_data.x[end] - ox
    lz    = 1.2max(
        maximum(getindex.(dem_data.bed,2)),
        maximum(getindex.(dem_data.ice,2))
    )
    ρg    = (x=0.0,y=9.81)
    ηs0   = 1e12
    λ     = (ice = 2.1   , wat = 2.1)
    ρ     = (ice = 920.0 , wat = 1000.0)
    cp    = (ice = 2100.0, wat = 4200.0)
    L     = 334e3
    T_mlt = 0.0
    T_atm = -5.0
    T_ini = -5.0

    # dimensionless numbers
    

    # scales
    # L̄ = lz
    # P̄ = ρ*g*L̄
    # T̄ = 

    # thermodynamics
    @inline u_ice(T)  = cp.ice*(T-T_mlt)
    @inline u_wat(T)  = L + cp.wat*(T-T_mlt)
    @inline T_lt(u_t) = (u_t < u_ice(T_mlt)) ? T_mlt + u_t/cp.ice :
                        (u_t > u_wat(T_mlt)) ? T_mlt + (u_t - L)/cp.wat : T_mlt
    @inline ω_lt(u_t) = (u_t < u_ice(T_mlt)) ? 0.0 :
                        (u_t > u_wat(T_mlt)) ? 1.0 : ρ.ice*(u_ice(T_mlt) - u_t)/(ρ.ice*(u_ice(T_mlt)-u_t) - ρ.wat*(u_wat(T_mlt)-u_t))

    # numerics
    nx,nz = grid_dims
    ϵtol = (1e-4,1e-4,1e-4)
    maxiter = 50max(nx,nz)
    ncheck  = ceil(Int,0.25max(nx,nz))
    nviz    = 20
    nt      = 1000
    
    # preprocessing
    dx,dz = lx/nx,lz/nz
    @info "grid spacing: dx = $dx, dz = $dz"

    xv = LinRange(ox,ox+lx,nx+1)
    zv = LinRange(oz,oz+lz,nz+1)
    xc,zc = av1.((xv,zv))
    
    # PT params
    r          = 0.7
    lτ_re_mech = 0.25min(lx,lz)/π
    vdτ_mech   = min(dx,dz)/sqrt(2.1)
    θ_dτ       = lτ_re_mech*(r+4/3)/vdτ_mech
    nudτ       = vdτ_mech*lτ_re_mech
    dτ_r       = 1.0/(θ_dτ+1.0)

    # fields allocation
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
    τII  = scalar_field(Float64,nx-2,nz-2)
    Ψav  = (
        not_solid = scalar_field(Float64,nx-2,nz-2),
        not_air   = scalar_field(Float64,nx-2,nz-2),
    )

    # initialisation
    # level set
    for comp in eachindex(Ψ) fill!(Ψ[comp],Inf) end

    @info "computing the level set for the ice surface"
    compute_levelset!(Ψ.not_air,xv,zv,to_device(dem_data.ice))

    @info "computing the level set for the bedrock surface"
    compute_levelset!(Ψ.not_solid,xv,zv,to_device(dem_data.bed))
    @. Ψ.not_solid *= -1.0

    @info "computing volume fractions from level sets"
    for phase in eachindex(Ψ)
        compute_volume_fractions_from_level_set!(wt[phase],Ψ[phase],dx,dz)
    end

    # mechanics
    for comp in eachindex(V) fill!(V[comp],0.0) end
    for comp in eachindex(τ) fill!(τ[comp],0.0) end
    fill!(Pr,0.0)
    fill!(ηs,ηs0)

    # thermo
    for comp in eachindex(qT) fill!(qT[comp],0.0) end
    @. T  = lerp(T_atm,T_ini,wt.not_air.c) 
    @. ρU = ρ.ice*u_ice(T)
    @. ω  = ω_lt(ρU/ρ.ice)

    # convergence tracking
    iter_evo = Float64[]
    errs_evo = ElasticArray{Float64}(undef, length(ϵtol), 0)

    # figures
    fig = Figure(resolution=(3000,1200),fontsize=32)
    axs = (
        hmaps = (
            Pr   = Axis(fig[1,1][1,1];aspect=DataAspect(),title="p"  ),
            τII  = Axis(fig[1,2][1,1];aspect=DataAspect(),title="τII"),
            Vmag = Axis(fig[1,3][1,1];aspect=DataAspect(),title="|V|"),
            T    = Axis(fig[2,1][1,1];aspect=DataAspect(),title="T"  ),
            ω    = Axis(fig[2,2][1,1];aspect=DataAspect(),title="ω"  ),
        ),
        errs = Axis(fig[2,3];yscale=log10, title="Convergence", xlabel="#iter/ny", ylabel="ϵ"),
    )
    for axname in eachindex(axs.hmaps)
        xlims!(axs.hmaps[axname],ox,ox+lx)
        ylims!(axs.hmaps[axname],oz,oz+lz)
    end

    plt = (
        hmaps = (
            Pr   = heatmap!(axs.hmaps.Pr  ,xv,zv,to_host(Pr  );colormap=:turbo),
            τII  = heatmap!(axs.hmaps.τII ,xv,zv,to_host(τII );colormap=:turbo),
            Vmag = heatmap!(axs.hmaps.Vmag,xv,zv,to_host(Vmag);colormap=:turbo),
            T    = heatmap!(axs.hmaps.T   ,xc,zc,to_host(T   );colormap=:turbo),
            ω    = heatmap!(axs.hmaps.ω   ,xc,zc,to_host(ω   );colormap=:turbo),
        ),
        errs=[scatterlines!(axs.errs, Point2.(iter_evo, errs_evo[ir, :])) for ir in eachindex(ϵtol)],
    )
    plt_bed = [
        (
            bed =  poly!(axs.hmaps[f],dem_data.bed;strokewidth=2,color=:black),
            ice = lines!(axs.hmaps[f],dem_data.ice;strokewidth=2,color=:black),
        ) for f in eachindex(axs.hmaps)
    ]

    Colorbar(fig[1,1][1,2],plt.hmaps.Pr)
    Colorbar(fig[1,2][1,2],plt.hmaps.τII)
    Colorbar(fig[1,3][1,2],plt.hmaps.Vmag)
    Colorbar(fig[2,1][1,2],plt.hmaps.T)
    Colorbar(fig[2,2][1,2],plt.hmaps.ω)

    update_vis_fields!(Vmag,τII,Ψav,V,τ,Ψ)
    plt.hmaps.Pr[3][]   = to_host(Pr)
    plt.hmaps.τII[3][]  = to_host(τII)
    plt.hmaps.Vmag[3][] = to_host(Vmag)
    plt.hmaps.T[3][]    = to_host(T)
    plt.hmaps.ω[3][]    = to_host(ω)
    display(fig)
    
    @info "time loop"
    for it in 1:nt
        @info "time step #$it"
        @info "iteration loop"
        empty!(iter_evo); resize!(errs_evo,(length(ϵtol),0))
        # mechanics
        for iter in 1:maxiter
            update_σ!(Pr,τ,V,ηs,wt,r,θ_dτ,dτ_r,dx,dz)
            update_V!(V,Pr,τ,ηs,wt,nudτ,ρg,dx,dz)
            if iter % ncheck == 0
                compute_residual!(Res,Pr,V,τ,wt,ρg,dx,dz)
                errs = (maximum(abs.(Res.V.x)), maximum(abs.(Res.V.y)), maximum(abs.(Res.Pr)))
                @printf("  iter/nz # %2.1f, errs: [ Vx = %1.3e, Vy = %1.3e, Pr = %1.3e ]\n", iter/nz, errs...)
                push!(iter_evo, iter/nz); append!(errs_evo, errs)
                # check convergence
                if any(.!isfinite.(errs)) error("simulation failed") end
                if all(errs .< ϵtol) break end
            end
        end
        dt = min(dx,dz)^2/max(λ.ice*cp.ice,λ.wat*cp.wat)/4.1
        # thermal
        update_qT!(qT,T,wt,λ,T_atm,dx,dz)
        update_ρU!(ρU,qT,τ,ηs,wt,ρ.ice*u_ice(T_atm),dt,dx,dz)
        @. T = T_lt(ρU/(ρ.ice*(1-ω) + ρ.wat*ω))
        @. ω = ω_lt(ρU/(ρ.ice*(1-ω) + ρ.wat*ω))
        # update figures
        if it % nviz == 0
            for ir in eachindex(plt.errs)
                plt.errs[ir][1] = Point2.(iter_evo, errs_evo[ir, :])
            end
            autolimits!(axs.errs)
            update_vis_fields!(Vmag,τII,Ψav,V,τ,Ψ)
            plt.hmaps.Pr[3][]   = to_host(Pr)
            plt.hmaps.τII[3][]  = to_host(τII)
            plt.hmaps.Vmag[3][] = to_host(Vmag)
            plt.hmaps.T[3][]    = to_host(T)
            plt.hmaps.ω[3][]    = to_host(ω)
            yield()
        end
    end

    @info "saving results on disk"
    out_h5 = "results.h5"
    ndrange = CartesianIndices((nx-2,nz-2))
    fields = Dict("LS_ice"=>Ψav.not_air,"LS_bed"=>Ψav.not_solid,"Vmag"=>Vmag,"TII"=>τII,"Pr"=>inn(Pr))
    @info "saving HDF5 file"
    write_h5(out_h5,fields,grid_dims,ndrange)

    @info "saving XDMF file..."
    write_xdmf("results.xdmf3",out_h5,fields,(xc[2],zc[2]),(dx,dz),grid_dims.-2)

    return
end

@tiny function _kernel_update_vis_fields!(Vmag, τII, Ψav, V, τ, Ψ)
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
    @inbounds if isin(τII)
        τxzc = 0.25*(τ.xy[ix,iz]+τ.xy[ix+1,iz]+τ.xy[ix,iz+1]+τ.xy[ix+1,iz+1])
        τII[ix,iz] = sqrt(0.5*(τ.xx[ix+1,iz+1]^2 + τ.yy[ix+1,iz+1]^2) + τxzc^2)
    end
    return
end

const _update_vis_fields! = _kernel_update_vis_fields!(get_device())

function update_vis_fields!(Vmag, τII, Ψav, V, τ, Ψ)
    wait(_update_vis_fields!(Vmag, τII, Ψav, V, τ, Ψ; ndrange=axes(Vmag)))
    return
end

main((500,250))