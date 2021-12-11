const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : true
const gpu_id  = haskey(ENV, "GPU_ID" ) ? parse(Int , ENV["GPU_ID" ]) : 0
###
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(gpu_id)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, LinearAlgebra, MAT, Random

@parallel function compute_P!(∇V::Data.Array, Pt::Data.Array, Vx::Data.Array, Vy::Data.Array, ϕ::Data.Array, Gdτ::Data.Number, r::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy
    @all(Pt)  = @all(ϕ)*(@all(Pt) - r*Gdτ*@all(∇V))
    return
end

@parallel function compute_τ!(τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, Vx::Data.Array, Vy::Data.Array, ϕ::Data.Array, ϕv::Data.Array, μ_veτ::Data.Number, Gdτ::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(τxx) = @all(ϕ) *2.0*μ_veτ*(@d_xa(Vx)/dx + @all(τxx)/Gdτ/2.0)
    @all(τyy) = @all(ϕ) *2.0*μ_veτ*(@d_ya(Vy)/dy + @all(τyy)/Gdτ/2.0)
    @all(τxy) = @all(ϕv)*2.0*μ_veτ*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx) + @all(τxy)/Gdτ/2.0)
    return
end

@parallel function compute_dV!(Rx::Data.Array, Ry::Data.Array, dVx::Data.Array, dVy::Data.Array, Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, ϕx::Data.Array, ϕy::Data.Array, ρgx::Data.Number, ρgy::Data.Number, dτ_ρ::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(Rx)  = @d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx - @all(ϕx)*ρgx
    @all(Ry)  = @d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy - @all(ϕy)*ρgy
    @all(dVx) = dτ_ρ*@all(Rx)
    @all(dVy) = dτ_ρ*@all(Ry)
    return
end

@parallel function compute_V!(Vx::Data.Array, Vy::Data.Array, dVx::Data.Array, dVy::Data.Array, θx::Data.Array, θy::Data.Array)
    @inn(Vx) = (1.0-@all(θx))*(@inn(Vx) + @all(dVx))
    @inn(Vy) = (1.0-@all(θy))*(@inn(Vy) + @all(dVy))
    return
end

@parallel_indices (iy) function bc_x!(A::Data.Array)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

@parallel_indices (ix) function bc_y!(A::Data.Array)
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return
end

@parallel_indices (ix,iy) function init_ϕ!(ϕ,ϕv,ϕx,ϕy,gl,dx,dy,lx,ly)
    xc,yc = dx*ix-dx/2-lx/2, dy*iy-dy/2
    xv,yv = dx*ix     -lx/2, dy*iy
    if ix <= size(ϕ,1) && iy <= size(ϕ,2)
        if abs(xc) < gl/2 && yc < gl
            ϕ[ix,iy] = 1.0
        end
    end
    if ix <= size(ϕv,1) && iy <= size(ϕv,2)
        if abs(xv) < gl/2 && yv < gl
            ϕv[ix,iy] = 1.0
        end
    end
    if ix <= size(ϕx,1) && iy < size(ϕx,2)
        if abs(xv) < gl/2 && yc+dx < gl
            ϕx[ix,iy] = 1.0
        end
    end
    if ix <= size(ϕy,1) && iy < size(ϕy,2)
        if abs(xc+dx) < gl/2 && yv < gl
            ϕy[ix,iy] = 1.0
        end
    end
    return
end

@parallel_indices (ix,iy) function init_θ!(θx,θy,el,tanβ,ω,amp,dx,dy,lx,ly)
    xc,yc = dx*ix-dx/2-lx/2, dy*iy-dy/2
    xv,yv = dx*ix     -lx/2, dy*iy
    fc    = amp*sin(ω*xc/lx) + tanβ*xc + el
    fv    = amp*sin(ω*xv/lx) + tanβ*xv + el
    if ix <= size(θx,1) && iy <= size(θx,2)
        if yc - fc < 0.0
            θx[ix,iy] = 1.0
        end
    end
    if ix <= size(θy,1) && iy <= size(θy,2)
        if yv - fv < 0.0
            θy[ix,iy] = 1.0
        end
    end
    return
end


@views function Stokes2D()
    # physics
    ## dimensionally independent
    ly        = 1.0               # domain height    [m]
    μs0       = 1.0               # matrix viscosity [Pa*s]
    ρg0       = 1.0               # gravity          [Pa/m]
    ## scales
    psc       = ρg0*ly
    tsc       = μs0/psc
    vsc       = ly/tsc
    ## nondimensional parameters
    lx_ly     = 1.0
    gl_ly     = 1/2
    el_ly     = 0.15
    amp_ly    = 1/25
    α         = -0π
    tanβ      = tan(-π/12)
    ωly       = 10π
    ## dimensionally dependent
    lx        = lx_ly*ly
    gl        = gl_ly*ly
    el        = el_ly*ly
    amp       = amp_ly*ly
    ρgx       = ρg0*sin(α)
    ρgy       = ρg0*cos(α)
    ω         = ωly/ly
    # numerics
    ny        = 511
    nx        = ceil(Int,lx_ly*ny)
    maxiter   = 50ny         # maximum number of pseudo-transient iterations
    nchk      = 1ny          # error checking frequency
    nviz      = 1ny          # visualisation frequency
    ε_V       = 1e-8         # nonlinear absolute tolerence for momentum
    ε_∇V      = 1e-8         # nonlinear absolute tolerence for divergence
    CFL       = 0.95/sqrt(2) # stability condition
    Re        = 2π           # Reynolds number                     (numerical parameter #1)
    r         = 1.0          # Bulk to shear elastic modulus ratio (numerical parameter #2)
    # preprocessing
    dx, dy    = lx/nx, ly/ny # cell sizes
    max_lxy   = 0.6gl
    Vpdτ      = min(dx,dy)*CFL
    dτ_ρ      = Vpdτ*max_lxy/Re/μs0
    Gdτ       = Vpdτ^2/dτ_ρ/(r+2.0)
    μ_veτ     = 1.0/(1.0/Gdτ + 1.0/μs0)
    Xc, Yc    = LinRange(-(lx-dx)/2,(lx-dx)/2,nx  ), LinRange(dy/2,ly-dy/2,ny  )
    Xv, Yv    = LinRange(- lx/2    , lx/2    ,nx+1), LinRange(0   ,ly     ,ny+1)
    # allocation
    Pt        = @zeros(nx  ,ny  )
    ∇V        = @zeros(nx  ,ny  )
    τxx       = @zeros(nx  ,ny  )
    τyy       = @zeros(nx  ,ny  )
    τxy       = @zeros(nx-1,ny-1)
    Rx        = @zeros(nx-1,ny-2)
    Ry        = @zeros(nx-2,ny-1)
    dVx       = @zeros(nx-1,ny-2)
    dVy       = @zeros(nx-2,ny-1)
    Vx        = @zeros(nx+1,ny  )
    Vy        = @zeros(nx  ,ny+1)
    ϕ         = @zeros(nx  ,ny  )
    ϕv        = @zeros(nx-1,ny-1)
    ϕx        = @zeros(nx-1,ny-2)
    ϕy        = @zeros(nx-2,ny-1)
    θx        = @zeros(nx-1,ny-2)
    θy        = @zeros(nx-2,ny-1)
    @parallel init_ϕ!(ϕ,ϕv,ϕx,ϕy,gl,dx,dy,lx,ly)
    @parallel init_θ!(θx,θy,el,tanβ,ω,amp,dx,dy,lx,ly)
    # Iteraion loop
    err_V=2*ε_V; err_∇V=2*ε_∇V; iter=0; err_evo1=[]; err_evo2=[]
    while !((err_V <= ε_V) && (err_∇V <= ε_∇V)) && (iter <= maxiter)
        @parallel compute_P!(∇V, Pt, Vx, Vy, ϕ, Gdτ, r, dx, dy)
        @parallel compute_τ!(τxx, τyy, τxy, Vx, Vy, ϕ, ϕv, μ_veτ,Gdτ, dx, dy)
        @parallel compute_dV!(Rx, Ry, dVx, dVy, Pt, τxx, τyy, τxy, ϕx, ϕy, ρgx, ρgy, dτ_ρ, dx, dy)
        @parallel compute_V!(Vx, Vy, dVx, dVy, θx, θy)
        iter += 1
        if iter % nchk == 0
            norm_Rx = norm((1.0 .- θx).*Rx)/psc*lx/sqrt(length(Rx))
            norm_Ry = norm((1.0 .- θy).*Ry)/psc*lx/sqrt(length(Ry))
            norm_∇V = norm(ϕ.*∇V)/vsc*lx/sqrt(length(∇V))
            err_V   = maximum([norm_Rx, norm_Ry])
            err_∇V  = norm_∇V
            push!(err_evo1, maximum([norm_Rx, norm_Ry, norm_∇V])); push!(err_evo2,iter/max(nx,ny))
            @printf("# iters = %d, err_V = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e], err_∇V = %1.3e \n", iter, err_V, norm_Rx, norm_Ry, err_∇V)
        end
        if iter % nviz == 0
            p1 = heatmap(Xv,Yc,Array(Vx)',aspect_ratio=1,xlims=(Xv[1],Xv[end]),ylims=(Yc[1],Yc[end]),c=:viridis,title="Vx")
            p2 = heatmap(Xc,Yv,Array(Vy)',aspect_ratio=1,xlims=(Xc[1],Xc[end]),ylims=(Yv[1],Yv[end]),c=:viridis,title="Vy")
            p3 = heatmap(Xc,Yc,Array(Pt)',aspect_ratio=1,xlims=(Xc[1],Xc[end]),ylims=(Yc[1],Yc[end]),clims=(0,0.5psc),c=:viridis,title="Pressure")
            p4 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations/nx", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
            display(plot(p1, p2, p3, p4))
        end
    end
    return
end

Stokes2D()
