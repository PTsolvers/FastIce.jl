using FastIce
using TinyKernels
using CairoMakie
using ElasticArrays
using Printf
using SpecialFunctions

include("bcs.jl")
include("init_vis.jl")
include("level_sets.jl")
include("stokes_ve_2_bulk_tens_hype.jl")
include("volume_fractions.jl")

@views av1(A) = 0.5 .* (A[1:end-1] .+ A[2:end])
@views inn_x(A) = A[2:end-1,:]
@views inn_y(A) = A[:,2:end-1]
@views inn(A) = A[2:end-1,2:end-1]
nonan!(A) = .!isnan.(A) .* A

@views function runsim(::Type{DAT}; nx=127) where {DAT}
    # physics
    ly       = 1.0 # m
    A0       = 1.0 # Pa s ^ m
    ρg0      = 0.0 # m / s ^ 2
    ε̇bg      = 1.0 # shear
    # nondim
    ξ        = 1 / 4 # eta / G / dt
    De       = 1.0   # Deborah num
    npow     = 3.0
    mpow     = -(1 - 1 / npow)
    # scales
    l_sc     = ly
    # τ_sc     = ρg0 * l_sc                # buoyancy
    # t_sc     = (A0 / τ_sc) ^ (1 / mpow)  # buoyancy
    τ_sc     = A0 * ε̇bg ^ mpow  # shear
    t_sc     = 1 / ε̇bg          # shear
    η_sc     = τ_sc * t_sc
    # dependent
    lx       = 1.0 * ly
    ox, oy   = -0.5lx, -0.5ly
    xb1, yb1 = ox + 0.5lx, oy + 0.5ly
    rinc     = 0.1lx
    G        = τ_sc / De
    K        = 4.0 * G
    dt0      = ξ * η_sc / G
    α        = 0.0
    ϕs       = 30
    C0       = 1.8 * τ_sc
    Pd       = C0
    σd       = C0 / 2
    σt       = C0 / 1.1
    # ε̇bg      = 1.0e-10 / t_sc # buoyancy
    # numerics
    nt       = 50
    ny       = ceil(Int, (nx + 1) * ly / lx) - 1
    maxiter  = 400nx
    ncheck   = 10nx
    ϵtol     = (5e-6, 5e-6, 1e-6) .* 2
    χ        = 0.2       # viscosity relaxation
    ηmax     = 1e1       # viscosity cut-off
    η_reg    = 2e-2      # Plastic regularisation
    # preprocessing
    sinϕ     = sind(ϕs)
    cosϕ     = cosd(ϕs)
    dx, dy   = lx / nx, ly / ny
    xv, yv   = LinRange(ox, ox + lx, nx + 1), LinRange(oy, oy + ly, ny + 1)
    xc, yc   = av1(xv), av1(yv)
    mc1      = to_device(make_marker_chain_circle(Point(xb1, yb1), rinc, min(dx, dy)))
    # mc2      = to_device(make_marker_chain_circle(Point(xb2, yb2), rair, min(dx, dy)))
    ρg       = (x=ρg0 .* sin(α), y=ρg0 .* cos(α))
    # PT parameters
    r        = 0.7
    re_mech  = 8π
    lτ       = min(lx, ly)
    vdτ      = min(dx, dy) / sqrt(2.1) / 1.1
    θ_dτ     = lτ * (r + 4 / 3) / (re_mech * vdτ)
    nudτ     = vdτ * lτ / re_mech
    # level set
    Ψ  = (
        not_solid = field_array(DAT, nx + 1, ny + 1), # fluid
        not_air   = field_array(DAT, nx + 1, ny + 1), # liquid
    )
    wt = (
        not_solid = volfrac_field(DAT, nx, ny), # fluid
        not_air   = volfrac_field(DAT, nx, ny), # liquid
    )
    # mechanics
    Pr    = scalar_field(DAT, nx, ny)
    Pr_c  = scalar_field(DAT, nx, ny)
    Pr_o  = scalar_field(DAT, nx, ny)
    τ     = tensor_field(DAT, nx, ny)
    τ_o   = tensor_field(DAT, nx, ny)
    δτ    = tensor_field(DAT, nx, ny)
    ε     = tensor_field(DAT, nx, ny)
    ε_ve  = tensor_field(DAT, nx, ny)
    V     = vector_field(DAT, nx, ny)
    ηs    = scalar_field(DAT, nx, ny)
    η_ve  = scalar_field(DAT, nx, ny)
    εII   = scalar_field(DAT, nx, ny)
    τII   = scalar_field(DAT, nx, ny)
    τII_c = scalar_field(DAT, nx, ny)
    F     = scalar_field(DAT, nx, ny)
    λ     = scalar_field(DAT, nx, ny)
    dλdτ  = scalar_field(DAT, nx, ny)
    C     = scalar_field(DAT, nx, ny)
    # residuals
    Res = (
        Pr = scalar_field(DAT, nx    , ny    ),
        V  = vector_field(DAT, nx - 2, ny - 2),
    )
    # visualisation
    Vmag = field_array(DAT, nx - 2, ny - 2)
    Ψav = (
        not_solid = field_array(DAT, nx - 2, ny - 2),
        not_air   = field_array(DAT, nx - 2, ny - 2),
    )
    # initialisation
    for comp in eachindex(τ) fill!(τ[comp] , 0.0) end
    for comp in eachindex(δτ) fill!(δτ[comp], 0.0) end
    for comp in eachindex(ε) fill!(ε[comp] , 0.0) end
    for comp in eachindex(ε_ve) fill!(ε_ve[comp] , 0.0) end
    fill!(Pr   , 0.0)
    fill!(Pr_c , 0.0)
    fill!(Pr_o , 0.0)
    fill!(ηs   , η_sc)
    fill!(η_ve , (1.0 / η_sc + 1.0 / (G * dt0))^-1)
    fill!(εII  , 1e-10)
    fill!(τII  , 0.0)
    fill!(τII_c, 0.0)
    fill!(F    , 0.0)
    fill!(λ    , 0.0)
    fill!(dλdτ , 0.0)
    fill!(C    , C0)

    init!(V, ε̇bg, xv, yv)

    # compute level sets
    for comp in eachindex(Ψ) fill!(Ψ[comp], 1.0) end
    Ψ.not_air .= Inf # needs init now
    @info "computing the level set for the inclusion"
    compute_levelset!(Ψ.not_air, xv, yv, mc1)
    # compute_levelset!(Ψ.not_air, xv, yv, mc2)
    TinyKernels.device_synchronize(get_device())
    # Ψ.not_air .= min.( .-(0.0 .* xv .+ yv' .+ oy .+ 0.05), Ψ.not_air)
    @. Ψ.not_air = -Ψ.not_air

    @info "computing the level set for the bedrock"
    @. Ψ.not_solid = -(0.0 * xv + yv' - 0.05)

    @info "computing volume fractions from level sets"
    compute_volume_fractions_from_level_set!(wt.not_air, Ψ.not_air, dx, dy)
    # compute_volume_fractions_from_level_set!(wt.not_solid, Ψ.not_solid, dx, dy)
    for comp in eachindex(wt.not_solid) fill!(wt.not_solid[comp], 1.0) end

    update_vis!(Vmag, Ψav, V, Ψ)
    # convergence history
    iter_evo = Float64[]
    errs_evo = ElasticArray{Float64}(undef, length(ϵtol), 0)

    # Plot yield functions
    Pp, τIIp = collect(LinRange(-1 * C0, 5 * C0, 500)), collect(LinRange(0, 5 * C0, 500))
    P2 = Pp .+ 0.0 * τIIp'
    τ2 = 0.0 * Pp .+ τIIp'
    Fp = @. sqrt(τ2^2 + (C0 * cosϕ - σt * sinϕ)^2) - (C0 * cosϕ + P2 * sinϕ)

    # figures
    fig = Figure(resolution=(2500, 1800), fontsize=32)
    ax = (
        Pr  =Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="p"),
        τII =Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="τII"),
        wt  =Axis(fig[1, 3][1, 1]; aspect=DataAspect(), title="Volume fraction"),
        Vmag=Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="|v|"),
        εII =Axis(fig[2, 2][1, 1]; aspect=DataAspect(), title="εII"),
        ηs  =Axis(fig[2, 3][1, 1]; aspect=DataAspect(), title="log10(ηs)"),
        λ   =Axis(fig[3, 1][1, 1]; aspect=DataAspect(), title="λ"),
        F   =Axis(fig[3, 2][1, 1]; title="F", xlabel="P", ylabel="τII"),
        errs=Axis(fig[3, 3]      ; yscale=log10, title="Convergence", xlabel="# iter / nx", ylabel="error"),
    )
    plt = (
        fields=(
            Pr  =heatmap!(ax.Pr  , xc, yc, to_host(Pr  ); colormap=:turbo),
            τII =heatmap!(ax.τII , xc, yc, to_host(τII ); colormap=:turbo),
            wt  =heatmap!(ax.wt  , xc, yc, to_host(wt.not_air.c); colormap=Reverse(:grays)),
            Vmag=heatmap!(ax.Vmag, xc, yc, to_host(Vmag); colormap=:turbo),
            εII =heatmap!(ax.εII , xc, yc, to_host(εII ); colormap=:turbo),
            ηs  =heatmap!(ax.ηs  , xc, yc, to_host(log10.(ηs)); colormap=:turbo),
            λ   =heatmap!(ax.λ   , xc, yc, to_host(λ   ); colormap=:turbo),
            F   =scatter!(ax.F   , Point2f.(to_host(Pr_c)[:], to_host(τII_c)[:]), color=to_host(dλdτ[:]), colormap=:turbo),#markerspace=:data, markersize=r0
            F2  =contour!(ax.F   , Pp, τIIp, Fp, levels=-0.0:0.1:0.0, linewidth=4, label="yield"),
            F3  =  ylims!(ax.F   , -0.5, 6.0),
        ),
        errs=[scatterlines!(ax.errs, Point2.(iter_evo, errs_evo[ir, :])) for ir in eachindex(ϵtol)],
    )
    Colorbar(fig[1, 1][1, 2], plt.fields.Pr  )
    Colorbar(fig[1, 2][1, 2], plt.fields.τII )
    Colorbar(fig[1, 3][1, 2], plt.fields.wt  )
    Colorbar(fig[2, 1][1, 2], plt.fields.Vmag)
    Colorbar(fig[2, 2][1, 2], plt.fields.εII )
    Colorbar(fig[2, 3][1, 2], plt.fields.ηs  )
    Colorbar(fig[3, 1][1, 2], plt.fields.λ   )
    Colorbar(fig[3, 2][1, 2], plt.fields.F   )
    axislegend(ax.F, position=:lt)
    display(fig)
    maskA = copy(to_host(wt.not_air.c))
    maskS = copy(to_host(wt.not_solid.c))
    maskA[maskA.<1.0] .= NaN
    maskS[maskS.<1.0] .= NaN
    mask = maskA .* maskS

    # error("stop")
    @info "running simulation 🚀"
    for it in 1:nt
        dt = dt0 # (it >= 6 && it <= 10) ? dt = dt0 / 1 : dt = dt0 # if npow=3
        @printf "it # %d, dt = %1.3e \n" it dt
        update_old!(τ_o, τ, Pr_o, Pr_c, Pr, λ)
        # iteration loop
        empty!(iter_evo); resize!(errs_evo, length(ϵtol), 0)
        iter = 0; errs = 2.0 .* ϵtol
        while any(errs .>= ϵtol) && (iter += 1) <= maxiter
            increment_τ!(Pr, Pr_o, ε, ε_ve, δτ, τ, τ_o, V, η_ve, ηs, G, K, dt, wt, r, θ_dτ, dx, dy)
            compute_xyc!(ε, ε_ve, δτ, τ, τ_o, η_ve, ηs, G, dt, θ_dτ, wt)
            compute_trial_τII!(τII, δτ, τ)
            update_τ!(Pr, Pr_c, ε_ve, τ, ηs, η_ve, G, K, dt, τII, τII_c, F, λ, dλdτ, C, cosϕ, sinϕ, Pd, σd, σt, η_reg, θ_dτ, wt)
            compute_εII_η!(εII, ηs, τ, ε, wt, χ, mpow, npow, A0, ηmax)
            update_V!(V, Pr_c, τ, ηs, wt, nudτ, ρg, dx, dy)
            if iter % ncheck == 0
                compute_residual!(Res, Pr, Pr_o, Pr_c, V, τ, K, dt, wt, ρg, dx, dy)
                errs = (maximum(abs.(Res.V.x)), maximum(abs.(Res.V.y)), maximum(abs.(Res.Pr)))
                @printf "  iter/nx # %2.1f, errs: [ Vx = %1.3e, Vy = %1.3e, Pr = %1.3e ]\n" iter / nx errs...
                @printf "    max(dλdτ) = %1.3e, max(τII) = %1.3e \n" maximum(abs.(dλdτ)) maximum(nonan!(τII))
                push!(iter_evo, iter / nx); append!(errs_evo, errs)
                # visu
                for ir in eachindex(plt.errs)
                    plt.errs[ir][1] = Point2.(iter_evo, errs_evo[ir, :])
                end
                autolimits!(ax.errs)
                update_vis!(Vmag, Ψav, V, Ψ)
                plt.fields[1][3] = to_host(Pr_c) .* mask
                plt.fields[2][3] = to_host(τII) .* mask
                plt.fields[3][3] = to_host(wt.not_air.c)
                plt.fields[4][3] = to_host(Vmag) .* inn(mask)
                plt.fields[5][3] = to_host(εII) .* mask
                plt.fields[6][3] = to_host(log10.(ηs)) .* mask
                plt.fields[7][3] = to_host(λ) .* mask
                plt.fields[8][1] = Point2f.(to_host(Pr_c)[:], to_host(τII_c)[:]); plt.fields[8].color[]=to_host(dλdτ[:])
                display(fig)
            end
        end
    end
    return
end

runsim(Float64, nx=180)