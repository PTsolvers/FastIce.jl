const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : false
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

import ParallelStencil: INDICES
ix,iy = INDICES[1], INDICES[2]
ixi,iyi = :($ix+1), :($iy+1)

const air   = 0.0
const fluid = 1.0
const solid = 2.0

macro fm(A)      esc(:( $A[$ix,$iy] == fluid )) end
macro fmxy_xi(A) esc(:( !(($A[$ix,$iy] == air && $A[$ix,$iy+1] == air) || ($A[$ix+1,$iy] == air && $A[$ix+1,$iy+1] == air)) )) end
macro fmxy_yi(A) esc(:( !(($A[$ix,$iy] == air && $A[$ix+1,$iy] == air) || ($A[$ix,$iy+1] == air && $A[$ix+1,$iy+1] == air)) )) end

macro dτ_ρ_ax()  esc(:( Vpdτ*max_lxy/Re/@av_xi(μs)         )) end
macro dτ_ρ_ay()  esc(:( Vpdτ*max_lxy/Re/@av_yi(μs)         )) end
macro Gdτ()      esc(:( Vpdτ*Re*@all(μs)/max_lxy/(r+2.0)   )) end 
macro Gdτ_av()   esc(:( Vpdτ*Re*@av(μs)/max_lxy/(r+2.0)    )) end 
macro μ_veτ()    esc(:( 1.0/(1.0/@Gdτ()    + 1.0/@all(μs)) )) end
macro μ_veτ_av() esc(:( 1.0/(1.0/@Gdτ_av() + 1.0/@av(μs))  )) end

@parallel function compute_P_τ!(∇V, Pt, τxx, τyy, τxy, Vx, Vy, μs, ϕ, r, Re, max_lxy, Vpdτ, dx, dy)
    @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy
    @all(Pt)  = 0*@fm(ϕ)*(@all(Pt) - r*@Gdτ()*@all(∇V))    
    @all(τxx) = 0*@fm(ϕ)*2.0*@μ_veτ()*(@d_xa(Vx)/dx + @all(τxx)/@Gdτ()/2.0)
    @all(τyy) = 0*@fm(ϕ)*2.0*@μ_veτ()*(@d_ya(Vy)/dy + @all(τyy)/@Gdτ()/2.0)
    @all(τxy) = @fmxy_xi(ϕ)*@fmxy_yi(ϕ)*@μ_veτ_av()*(@d_yi(Vx)/dy + 0*@d_xi(Vy)/dx + @all(τxy)/@Gdτ_av())
    return
end

@parallel_indices (ix,iy) function compute_EII_μs!(EII, μs, Vx, Vy, ϕ, μs0, npow, rel, dx, dy)
    nfluid = 0
    exy    = 0.0
    if ix <= size(EII,1)-2 && iy <= size(EII,2)-2
        if ϕ[ix,iy] == fluid && ϕ[ix+1,iy] == fluid && ϕ[ix,iy+1] == fluid && ϕ[ix+1,iy+1] == fluid
            nfluid += 1
            exy    += (Vx[ix+1,iy+1] - Vx[ix+1,iy])/dy + 0*(Vy[ix+1,iy+1] - Vy[ix,iy+1])/dx
        end
        if ϕ[ix+1,iy] == fluid && ϕ[ix+2,iy] == fluid && ϕ[ix+1,iy+1] == fluid && ϕ[ix+2,iy+1] == fluid
            nfluid += 1
            exy    += (Vx[ix+2,iy+1] - Vx[ix+2,iy])/dy + 0*(Vy[ix+2,iy+1] - Vy[ix+1,iy+1])/dx
        end
        if ϕ[ix,iy+1] == fluid && ϕ[ix+1,iy+1] == fluid && ϕ[ix,iy+2] == fluid && ϕ[ix+1,iy+2] == fluid
            nfluid += 1
            exy    += (Vx[ix+1,iy+2] - Vx[ix+1,iy+1])/dy + 0*(Vy[ix+1,iy+2] - Vy[ix,iy+2])/dx
        end
        if ϕ[ix+1,iy+1] == fluid && ϕ[ix+2,iy+1] == fluid && ϕ[ix+1,iy+2] == fluid && ϕ[ix+2,iy+2] == fluid
            nfluid += 1
            exy    += (Vx[ix+2,iy+2] - Vx[ix+2,iy+1])/dy + 0*(Vy[ix+2,iy+2] - Vy[ix+1,iy+2])/dx
        end
        if nfluid > 0
            exy /= 2*nfluid
        end
        exx = (Vx[ix+2,iy+1] - Vx[ix+1,iy+1])/dx
        eyy = (Vy[ix+1,iy+2] - Vy[ix+1,iy+1])/dy
        eii = sqrt(0*0.5*(exx*exx + eyy*eyy) + exy*exy)
        EII[ix+1,iy+1] = eii - exy
        μs_t           = clamp(μs0*eii^(npow-1.0), 1e-5*μs0, 2e4*μs0)
        μs[ix+1,iy+1]  = (μs[ix+1,iy+1]*(1-rel) + μs_t*rel)
        # μs[ix+1,iy+1] = μs0
    end
    return
end

macro sm_xi(A) esc(:( !(($A[$ix,$iyi] == solid) || ($A[$ix+1,$iyi] == solid)) )) end
macro sm_yi(A) esc(:( !(($A[$ixi,$iy] == solid) || ($A[$ixi,$iy+1] == solid)) )) end

macro fm_xi(A) esc(:( ($A[$ix,$iyi] == fluid) && ($A[$ix+1,$iyi] == fluid) )) end
macro fm_yi(A) esc(:( ($A[$ixi,$iy] == fluid) && ($A[$ixi,$iy+1] == fluid) )) end

@parallel function compute_V!(Vx, Vy, Pt, τxx, τyy, τxy, μs, ϕ, ρgx, ρgy, r, Re, max_lxy, Vpdτ, dx, dy)
    @inn(Vx) = @sm_xi(ϕ)*( @inn(Vx) + @dτ_ρ_ax()*(@d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx - @fm_xi(ϕ)*ρgx) )
    @inn(Vy) = 0*@sm_yi(ϕ)*( @inn(Vy) + @dτ_ρ_ay()*(@d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy - @fm_yi(ϕ)*ρgy) )
    return
end

@parallel function compute_τ2!(τxx2,τyy2,τxy2,Vx,Vy,μs,ϕ,dx,dy)
    @all(τxx2) = 0*@fm(ϕ)*2.0*@all(μs)*@d_xa(Vx)/dx
    @all(τyy2) = 0*@fm(ϕ)*2.0*@all(μs)*@d_ya(Vy)/dy
    @all(τxy2) = @fmxy_xi(ϕ)*@fmxy_yi(ϕ)*@av(μs)*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)
    return
end

@parallel function compute_Res!(Rx, Ry, Pt, τxx2, τyy2, τxy2, ϕ, ρgx, ρgy, dx, dy)
    @all(Rx)  = @sm_xi(ϕ)*(@d_xi(τxx2)/dx + @d_ya(τxy2)/dy - @d_xi(Pt)/dx - @fm_xi(ϕ)*ρgx)
    @all(Ry)  = 0*@sm_yi(ϕ)*(@d_yi(τyy2)/dy + @d_xa(τxy2)/dx - @d_yi(Pt)/dy - @fm_yi(ϕ)*ρgy)
    return
end

function is_in_plane(x,y,tanβ,el)
    return y < tanβ*x + el
end

@parallel_indices (ix,iy) function init_ϕ!(ϕ,gl,el,tanβ,dx,dy,lx,ly)
    xc,yc = dx*ix-dx/2-lx/2, -el+dy*iy-dy/2
    if checkbounds(Bool,ϕ,ix,iy)
        if is_in_plane(xc,yc,tanβ,gl-el)
            ϕ[ix,iy] = fluid
        end
        if is_in_plane(xc,yc,tanβ,0.0)
            ϕ[ix,iy] = solid
        end
    end
    return
end

@parallel_indices (ix,iy) function init_ϕi!(ϕ,ϕx,ϕy)
    if ix <= size(ϕx,1) && iy <= size(ϕx,2)
        ϕx[ix,iy] = air
        if ϕ[ix,iy] == fluid && ϕ[ix+1,iy] == fluid
            ϕx[ix,iy] = fluid
        end
    end
    if ix <= size(ϕy,1) && iy <= size(ϕy,2)
        ϕy[ix,iy] = air
        if ϕ[ix,iy] == fluid && ϕ[ix,iy+1] == fluid
            ϕy[ix,iy] = fluid
        end
    end
    return
end

# @parallel_indices (iy) function bc_x!(Vx,Vy,Pt,μs,EII,ϕ,del)
#     if iy <= size(Vx,2)
#         if ϕ[1,iy] == fluid
#             Vx[1,iy]   = Vx[end-1,iy-del]
#             Vy[1,iy]   = Vy[end-1,iy-del]
#             Vy[1,iy+1] = Vy[end-1,iy+1-del]
#             Pt[1,iy]   = Pt[end-1,iy-del]
#             μs[1,iy]   = μs[end-1,iy-del]
#             EII[1,iy]  = EII[end-1,iy-del]
#         end
#         if ϕ[end,iy] == fluid
#             Vx[end,iy]   = Vx[2,iy+del]
#             Vy[end,iy]   = Vy[2,iy+del]
#             Vy[end,iy+1] = Vy[2,iy+1+del]
#             Pt[end,iy]   = Pt[2,iy+del]
#             μs[end,iy]   = μs[2,iy+del]
#             EII[end,iy]  = EII[2,iy+del]
#         end
#     end
#     return
# end

@parallel_indices (iy) function bc_x!(Vx,Vy,Pt,μs,EII,ϕ,del)
    if iy <= size(Vx,2)
        Vx[1,iy]   = Vx[end-2,iy]
        Vy[1,iy]   = Vy[end-1,iy]
        # Pt[1,iy]   = Pt[end-1,iy]
        μs[1,iy]   = μs[end-1,iy]
        # EII[1,iy]  = EII[end-1,iy]
        Vx[end,iy]   = Vx[3,iy]
        Vy[end,iy]   = Vy[2,iy]
        # Pt[end,iy]   = Pt[2,iy]
        μs[end,iy]   = μs[2,iy]
        # EII[end,iy]  = EII[2,iy]
    end
    return
end

@parallel_indices (ix) function bc_y!(A)
    if ix <= size(A,1)
        A[ix,1  ] = A[ix,2    ]
        A[ix,end] = A[ix,end-1]
    end
    return
end


@views function Stokes2D()
    # physics
    ## dimensionally independent
    ly        = 1.0          # domain height    [m]
    μs0       = 1.0          # matrix viscosity [Pa*s]
    ρg0       = 1.0          # gravity          [Pa/m]
    ## scales
    psc       = ρg0*ly
    tsc       = μs0/psc
    vsc       = ly/tsc
    ## nondimensional parameters
    lx_ly     = 1.0
    gl_ly     = 0.75
    el_ly     = 0.25
    α         = -π/6
    β         = -0*π/12
    tanβ      = tan(β)
    npow      = 1.0/3.0
    ## dimensionally dependent
    lx        = lx_ly*ly
    gl        = gl_ly*ly
    el        = el_ly*ly
    ρgx       = ρg0*sin(α)
    ρgy       = ρg0*cos(α)
    # numerics
    ny        = 64
    nx        = ceil(Int,lx_ly*ny)
    nx, ny    = nx-1, ny-1
    maxiter   = 1000ny         # maximum number of pseudo-transient iterations
    nchk      = 2ny          # error checking frequency
    nviz      = 10ny          # visualisation frequency
    ε_V       = 1e-8         # nonlinear absolute tolerance for momentum
    ε_∇V      = 1e-8         # nonlinear absolute tolerance for divergence
    CFL       = 0.4/sqrt(2) # stability condition
    Re        = 2π/3           # Reynolds number                     (numerical parameter #1)
    r         = 1.0          # Bulk to shear elastic modulus ratio (numerical parameter #2)
    rel       = 1e-3
    # preprocessing
    dx, dy    = lx/nx, ly/ny # cell sizes
    max_lxy   = abs(gl-el)
    Vpdτ      = min(dx,dy)*CFL
    Xc, Yc    = LinRange(-(lx-dx)/2,(lx-dx)/2,nx  ), LinRange(-el+dy/2,ly-el-dy/2,ny  )
    Xv, Yv    = LinRange(- lx/2    , lx/2    ,nx+1), LinRange(-el     ,ly-el     ,ny+1)
    # allocation
    Pt        = @zeros(nx  ,ny  )
    ∇V        = @zeros(nx  ,ny  )
    τxx       = @zeros(nx  ,ny  )
    τyy       = @zeros(nx  ,ny  )
    τxy       = @zeros(nx-1,ny-1)
    τxx2      = @zeros(nx  ,ny  )
    τyy2      = @zeros(nx  ,ny  )
    τxy2      = @zeros(nx-1,ny-1)
    Rx        = @zeros(nx-1,ny-2)
    Ry        = @zeros(nx-2,ny-1)
    ϕx        = @zeros(nx-1,ny-2)
    ϕy        = @zeros(nx-2,ny-1)
    Vx        = @zeros(nx+1,ny  )
    Vy        = @zeros(nx  ,ny+1)
    μs        = 1e2*μs0*@ones(nx, ny)
    EII       = @zeros(nx, ny)
    ϕ         = air*@ones(nx,ny)
    Vx_v      = copy(Vx) # visu
    Vy_v      = copy(Vy) # visu
    Pt_v      = copy(Pt) # visu
    τxx_v     = copy(τxx) # visu
    τyy_v     = copy(τyy) # visu
    τxy_v     = copy(τxy) # visu
    EII_v     = copy(EII) # visu
    Rx_v      = copy(Rx) # visu
    μs_v      = copy(μs) # visu
    @parallel init_ϕ!(ϕ,gl,el,tanβ,dx,dy,lx,ly)
    @parallel init_ϕi!(ϕ,ϕx,ϕy)
    del       = findfirst(y->y==fluid,ϕ[1,:])-findfirst(y->y==fluid,ϕ[end,:])
    # iteration loop
    err_V=2*ε_V; err_∇V=2*ε_∇V; iter=0; err_evo1=[]; err_evo2=[]
    while !((err_V <= ε_V) && (err_∇V <= ε_∇V)) && (iter <= maxiter)
        if iter > 5*nx
            @parallel compute_EII_μs!(EII, μs, Vx, Vy, ϕ, μs0, npow, rel, dx, dy)
        end
        @parallel bc_x!(Vx,Vy,Pt,μs,EII,ϕ,del)
        @parallel bc_y!(μs)
        @parallel compute_P_τ!(∇V, Pt, τxx, τyy, τxy, Vx, Vy, μs, ϕ, r, Re, max_lxy, Vpdτ, dx, dy)
        @parallel compute_V!(Vx, Vy, Pt, τxx, τyy, τxy, μs, ϕ, ρgx, ρgy, r, Re, max_lxy, Vpdτ, dx, dy)
        @parallel bc_x!(Vx,Vy,Pt,μs,EII,ϕ,del)
        @parallel bc_y!(μs)
        iter += 1
        if iter % nchk == 0
            @parallel compute_τ2!(τxx2,τyy2,τxy2,Vx,Vy,μs,ϕ,dx,dy)
            @parallel compute_Res!(Rx, Ry, Pt, τxx2, τyy2, τxy2, ϕ, ρgx, ρgy, dx, dy)
            norm_Rx = norm((ϕx[2:end-1,:].==fluid).*Rx[2:end-1,:])/psc*lx/sqrt(length(Rx[2:end-1,:]))
            norm_Ry = norm((ϕy[:,2:end-1].==fluid).*Ry[:,2:end-1])/psc*lx/sqrt(length(Ry[:,2:end-1]))
            norm_∇V = norm((ϕ[2:end-1,:].==fluid).*∇V[2:end-1,:])/vsc*lx/sqrt(length(∇V[2:end-1,:]))
            err_V   = maximum([norm_Rx, norm_Ry])
            err_∇V  = norm_∇V
            push!(err_evo1, maximum([norm_Rx, norm_Ry, norm_∇V])); push!(err_evo2,iter/ny)
            @printf("# iters = %d, err_V = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e], err_∇V = %1.3e \n", iter, err_V, norm_Rx, norm_Ry, err_∇V)
            if !isfinite(err_V) || !isfinite(err_∇V) error("simulation failed") end
        end
        if iter % nviz == 0
            Vx_v .= Vx; Vx_v[Vx.==0] .= NaN
            Vy_v .= Vy; Vy_v[Vy.==0] .= NaN
            Pt_v .= Pt; Pt_v[Pt.==0] .= NaN
            τxx_v .= τxx2; τxx_v[τxx2.==0] .= NaN
            τyy_v .= τyy2; τyy_v[τyy2.==0] .= NaN
            τxy_v .= τxy2; τxy_v[τxy2.==0] .= NaN
            EII_v .= EII; EII_v[EII.==0] .= NaN
            μs_v .= log10.(μs); μs_v[ϕ.!=fluid] .= NaN
            Rx_v .= Rx
            fntsz = 7
            opts  = (aspect_ratio=1, xlims=(Xv[1],Xv[end]), ylims=(Yc[1],Yc[end]), yaxis=font(fntsz,"Courier"), xaxis=font(fntsz,"Courier"), framestyle=:box, titlefontsize=fntsz, titlefont="Courier")
            opts2 = (linewidth=2, markershape=:circle, markersize=3,yaxis = (:log10, font(fntsz,"Courier")), xaxis=font(fntsz,"Courier"), framestyle=:box, titlefontsize=fntsz, titlefont="Courier")
            p1 = heatmap(Xv,Yc,Array(Vx_v)'; c=:batlow, title="Vx", opts...)
            p2 = heatmap(Xc,Yv,Array(Vy_v)'; c=:batlow, title="Vy", opts...)
            p3 = heatmap(Xc,Yc,Array(Pt_v)'; c=:viridis, title="Pressure", opts...)
            p4 = heatmap(Xc,Yc,Array(τxx_v)'; c=:viridis, title="τxx", opts...)
            p5 = heatmap(Xc,Yc,Array(τyy_v)'; c=:viridis, title="τyy", opts...)
            p6 = heatmap(Xv[2:end-1],Yv[2:end-1],Array(τxy_v)'; c=:viridis, title="τxy", opts...)
            # p7 = heatmap(Xc,Yc,Array(EII_v)'; c=:viridis, title="EII", opts...)
            p7 = heatmap(Xc,Yc,Array(μs_v)'; c=:viridis, title="μs", opts...)
            p8 = plot(err_evo2,err_evo1; legend=false, xlabel="# iterations/nx", ylabel="log10(error)", labels="max(error)", opts2...)
            Vxc_yslice = Array(0.5 .*(Vx[round(Int,nx/2),:] .+ Vx[round(Int,nx/2)+1,:]))
            Vyc_yslice = Array(0.5 .*(Vy[round(Int,nx/2),1:end-1] .+ Vy[round(Int,nx/2),2:end]))
            # Vmc_slice  = sqrt.(Vxc_yslice.^2 + Vyc_yslice.^2)
            Vmc_slice  = Vxc_yslice
            h_exact    = (gl-el)*cos(β)
            Yc_tr      = collect(LinRange(0.0,h_exact,10))
            Yc_exact   = collect(LinRange(0.0,gl-el,10))
            # fx         = ρg0*sin(β)
            fx         = ρgx
            Vmc_exact  = ((-h_exact .+ Yc_tr).*(-fx.*(h_exact .- Yc_tr)./μs0).^(1.0/npow) .+ h_exact*(-fx*h_exact/μs0)^(1.0/npow)).*npow./(npow + 1.0)
            p9 = plot(Yv[2:end-1],Vmc_slice; label="numerical",linewidth=2,xlims=(0.0,gl-el))
            scatter!(Yc_exact, Vmc_exact; label="exact",marker=:d,xlims=(0.0,gl-el))
            display(plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, size=(1e3,600), dpi=200))
        end
    end
    return
end

Stokes2D()
