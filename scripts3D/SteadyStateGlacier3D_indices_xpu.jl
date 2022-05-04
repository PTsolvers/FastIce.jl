const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : true
const do_save = haskey(ENV, "DO_SAVE") ? parse(Bool, ENV["DO_SAVE"]) : true
###
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using ImplicitGlobalGrid,Printf,Statistics,LinearAlgebra,Random
import MPI
using HDF5,LightXML

norm_g(A) = (sum2_l = sum(A.^2); sqrt(MPI.Allreduce(sum2_l, MPI.SUM, MPI.COMM_WORLD)))
sum_g(A)  = (sum_l  = sum(A); MPI.Allreduce(sum_l, MPI.SUM, MPI.COMM_WORLD))

@views inn(A) = A[2:end-1,2:end-1,2:end-1]
@views av(A)  = convert(eltype(A),0.5)*(A[1:end-1]+A[2:end])
@views function smooth2D!(A2, A, fact)
    A2[2:end-1,2:end-1] .= A[2:end-1,2:end-1] .+ 1.0./4.1./fact.*( (A[3:end,2:end-1].-2.0.*A[2:end-1,2:end-1].+A[1:end-2,2:end-1]).+(A[2:end-1,3:end].-2.0.*A[2:end-1,2:end-1].+A[2:end-1,1:end-2]) )
    return
end

include(joinpath(@__DIR__, "helpers3D_v5.jl"))
include(joinpath(@__DIR__, "data_io.jl"     ))
include(joinpath(@__DIR__, "flagging_macros3D.jl"))

const air   = 0.0
const fluid = 1.0
const solid = 2.0

macro av_xii(A) esc(:( 0.5*($A[$ixi,$iyi,$izi] + $A[$ixi+1,$iyi  ,$izi  ]) )) end
macro av_yii(A) esc(:( 0.5*($A[$ixi,$iyi,$izi] + $A[$ixi  ,$iyi+1,$izi  ]) )) end
macro av_zii(A) esc(:( 0.5*($A[$ixi,$iyi,$izi] + $A[$ixi  ,$iyi  ,$izi+1]) )) end

@parallel_indices (ix,iy,iz) function compute_P_τ!(∇V, Pt, τxx, τyy, τzz, τxy, τxz, τyz, Vx, Vy, Vz, ϕ, r, μ_veτ, Gdτ, dx, dy, dz)
    @define_indices ix iy iz
    @in_phase ϕ fluid begin
        @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz
        @all(Pt)  = @all(Pt) - r*Gdτ*@all(∇V)
        @all(τxx) = 2.0*μ_veτ*(@d_xa(Vx)/dx + @all(τxx)/Gdτ/2.0)
        @all(τyy) = 2.0*μ_veτ*(@d_ya(Vy)/dy + @all(τyy)/Gdτ/2.0)
        @all(τzz) = 2.0*μ_veτ*(@d_za(Vz)/dz + @all(τzz)/Gdτ/2.0)
    end
    @in_phases_xy ϕ fluid fluid fluid fluid begin @all(τxy) = μ_veτ*(@d_yi(Vx)/dy + @d_xi(Vy)/dx + @all(τxy)/Gdτ) end
    @in_phases_xz ϕ fluid fluid fluid fluid begin @all(τxz) = μ_veτ*(@d_zi(Vx)/dz + @d_xi(Vz)/dx + @all(τxz)/Gdτ) end
    @in_phases_yz ϕ fluid fluid fluid fluid begin @all(τyz) = μ_veτ*(@d_zi(Vy)/dz + @d_yi(Vz)/dy + @all(τyz)/Gdτ) end
    return
end

macro fm_xi(A) esc(:( ($A[$ix,$iyi,$izi] == fluid) && ($A[$ix+1,$iyi,$izi] == fluid) )) end
macro fm_yi(A) esc(:( ($A[$ixi,$iy,$izi] == fluid) && ($A[$ixi,$iy+1,$izi] == fluid) )) end
macro fm_zi(A) esc(:( ($A[$ixi,$iyi,$iz] == fluid) && ($A[$ixi,$iyi,$iz+1] == fluid) )) end

@parallel_indices (ix,iy,iz) function compute_V!(Vx, Vy, Vz, Pt, τxx, τyy, τzz, τxy, τxz, τyz, ϕ, ρgx, ρgy, ρgz, dτ_ρ, dx, dy, dz)
    @define_indices ix iy iz
    @not_in_phases_xi ϕ solid solid begin @inn(Vx) = @inn(Vx) + dτ_ρ*(@d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @d_xi(Pt)/dx - @fm_xi(ϕ)*ρgx) end
    @not_in_phases_yi ϕ solid solid begin @inn(Vy) = @inn(Vy) + dτ_ρ*(@d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @d_yi(Pt)/dy - @fm_yi(ϕ)*ρgy) end
    @not_in_phases_zi ϕ solid solid begin @inn(Vz) = @inn(Vz) + dτ_ρ*(@d_zi(τzz)/dz + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @d_zi(Pt)/dz - @fm_zi(ϕ)*ρgz) end
    return
end

@parallel_indices (ix,iy,iz) function compute_Res!(Rx, Ry, Rz, Pt, τxx, τyy, τzz, τxy, τxz, τyz, ϕ, ρgx, ρgy, ρgz, dx, dy, dz)
    @define_indices ix iy iz
    @not_in_phases_xi ϕ solid solid begin @all(Rx) = @d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @d_xi(Pt)/dx - @fm_xi(ϕ)*ρgx end
    @not_in_phases_yi ϕ solid solid begin @all(Ry) = @d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @d_yi(Pt)/dy - @fm_yi(ϕ)*ρgy end
    @not_in_phases_zi ϕ solid solid begin @all(Rz) = @d_zi(τzz)/dz + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @d_zi(Pt)/dz - @fm_zi(ϕ)*ρgz end
    return
end

@parallel function preprocess_visu!(Vn, τII, Vx, Vy, Vz, τxx, τyy, τzz, τxy, τxz, τyz)
    # all arrays of size (nx-2,ny-2,nz-2)
    @all(Vn)  = sqrt(@av_xii(Vx)*@av_xii(Vx) + @av_yii(Vy)*@av_yii(Vy) + @av_zii(Vz)*@av_zii(Vz))
    @all(τII) = sqrt(0.5*(@inn(τxx)*@inn(τxx) + @inn(τyy)*@inn(τyy) + @inn(τzz)*@inn(τzz)) + @av_xya(τxy)*@av_xya(τxy) + @av_xza(τxz)*@av_xza(τxz) + @av_yza(τyz)*@av_yza(τyz))
    return
end

@parallel_indices (ix,iy,iz) function apply_mask!(Vn, τII, ϕ)
    if checkbounds(Bool,Vn,ix,iy,iz)
        if ϕ[ix+1,iy+1,iz+1] != fluid
             Vn[ix,iy,iz] = NaN
            τII[ix,iy,iz] = NaN
        end
    end
    return
end

"Check if index is inside phase."
function is_inside_phase(z3rot,ztopo)
    return z3rot < ztopo
end

@parallel_indices (ix,iy,iz) function set_phases!(ϕ,zsurf,zbed,R,ox,oy,oz,osx,osy,dx,dy,dz,dsx,dsy,cx,cy,cz)
    if checkbounds(Bool,ϕ,ix,iy,iz)
        ixg,iyg,izg = ix + cx*(size(ϕ,1)-2), iy + cy*(size(ϕ,2)-2), iz + cz*(size(ϕ,3)-2)
        xc,yc,zc    = ox + (ixg-1)*dx, oy + (iyg-1)*dy, oz + (izg-1)*dz
        xrot        = R[1,1]*xc + R[1,2]*yc + R[1,3]*zc
        yrot        = R[2,1]*xc + R[2,2]*yc + R[2,3]*zc
        zrot        = R[3,1]*xc + R[3,2]*yc + R[3,3]*zc
        ixr         = clamp(floor(Int, (xrot-osx)/dsx) + 1, 1, size(zsurf,1))
        iyr         = clamp(floor(Int, (yrot-osy)/dsy) + 1, 1, size(zsurf,2))
        if is_inside_phase(zrot,zsurf[ixr,iyr])
            ϕ[ix,iy,iz] = fluid
        end
        if is_inside_phase(zrot,zbed[ixr,iyr])
            ϕ[ix,iy,iz] = solid
        end
    end
    return
end

@parallel_indices (ix,iy,iz) function init_ϕi!(ϕ,ϕx,ϕy,ϕz)
    if ix <= size(ϕx,1) && iy <= size(ϕx,2) && iz <= size(ϕx,3)
        ϕx[ix,iy,iz] = air
        if ϕ[ix,iy,iz] == fluid && ϕ[ix+1,iy,iz] == fluid
            ϕx[ix,iy,iz] = fluid
        end
    end
    if ix <= size(ϕy,1) && iy <= size(ϕy,2) && iz <= size(ϕy,3)
        ϕy[ix,iy,iz] = air
        if ϕ[ix,iy,iz] == fluid && ϕ[ix,iy+1,iz] == fluid
            ϕy[ix,iy,iz] = fluid
        end
    end
    if ix <= size(ϕz,1) && iy <= size(ϕz,2) && iz <= size(ϕz,3)
        ϕz[ix,iy,iz] = air
        if ϕ[ix,iy,iz] == fluid && ϕ[ix,iy,iz+1] == fluid
            ϕz[ix,iy,iz] = fluid
        end
    end
    return
end

@views function Stokes3D(dem)
    # inputs
    # nx,ny,nz = 511,511,383      # local resolution
    # nx,ny,nz = 127,127,95       # local resolution
    nx,ny,nz = 127,127,47         # local resolution
    dim      = (2,2,1)          # MPI dims
    ns       = 2                # number of oversampling per cell
    nsm      = 5                # number of surface data smoothing steps
    out_path = "../out_visu"
    out_name = "results3D_M"
    # out_name = "results3D_M_rhone"
    # out_name = "results3D_M_greenland"
    # out_name = "results3D_M_antarctica"
    # IGG initialisation
    me,dims,nprocs,coords,comm_cart = init_global_grid(nx,ny,nz;dimx=dim[1],dimy=dim[2],dimz=dim[3])
    info     = MPI.Info()
    # define domain
    domain   = dilate(rotated_domain(dem), (0.05, 0.05, 0.05))
    lx,ly,lz = extents(domain)
    xv,yv,zv = create_grid(domain,(nx_g()+1,ny_g()+1,nz_g()+1))
    xc,yc,zc = av.((xv,yv,zv))
    dx,dy,dz = lx/nx_g(),ly/ny_g(),lz/nz_g()
    R        = rotation(dem)
    (me==0) && println("lx, ly, lz = $lx, $ly, $lz")
    (me==0) && println("dx, dy, dz = $dx, $dy, $dz")
    # physics
    ## dimensionally independent
    μs0      = 1.0               # matrix viscosity [Pa*s]
    ρg0      = 1.0               # gravity          [Pa/m]
    ## scales
    psc      = ρg0*lz
    tsc      = μs0/psc
    vsc      = lz/tsc
    ## dimensionally dependent
    ρgv      = ρg0*R'*[0,0,1]
    ρgx,ρgy,ρgz = ρgv
    # numerics
    maxiter  = 50*nx_g()    # maximum number of pseudo-transient iterations
    nchk     = 1*nx_g()     # error checking frequency
    b_width  = (8,4,4)      # boundary width
    ε_V      = 1e-8         # nonlinear absolute tolerance for momentum
    ε_∇V     = 1e-8         # nonlinear absolute tolerance for divergence
    CFL      = 0.9/sqrt(3)  # stability condition
    Re       = 2π           # Reynolds number                     (numerical parameter #1)
    r        = 1.0          # Bulk to shear elastic modulus ratio (numerical parameter #2)
    # preprocessing
    max_lxyz = 0.35*lz
    Vpdτ     = min(dx,dy,dz)*CFL
    dτ_ρ     = Vpdτ*max_lxyz/Re/μs0
    Gdτ      = Vpdτ^2/dτ_ρ/(r+2.0)
    μ_veτ    = 1.0/(1.0/Gdτ + 1.0/μs0)
    # allocation
    Pt       = @zeros(nx  ,ny  ,nz  )
    ∇V       = @zeros(nx  ,ny  ,nz  )
    τxx      = @zeros(nx  ,ny  ,nz  )
    τyy      = @zeros(nx  ,ny  ,nz  )
    τzz      = @zeros(nx  ,ny  ,nz  )
    τxy      = @zeros(nx-1,ny-1,nz-2)
    τxz      = @zeros(nx-1,ny-2,nz-1)
    τyz      = @zeros(nx-2,ny-1,nz-1)
    Rx       = @zeros(nx-1,ny-2,nz-2)
    Ry       = @zeros(nx-2,ny-1,nz-2)
    Rz       = @zeros(nx-2,ny-2,nz-1)
    ϕx       = @zeros(nx-1,ny-2,nz-2)
    ϕy       = @zeros(nx-2,ny-1,nz-2)
    ϕz       = @zeros(nx-2,ny-2,nz-1)
    Vx       = @zeros(nx+1,ny  ,nz  )
    Vy       = @zeros(nx  ,ny+1,nz  )
    Vz       = @zeros(nx  ,ny  ,nz+1)
    # set phases
    (me==0) && print("Set phases (0-air, 1-ice, 2-bedrock)...")
    Rinv     = Data.Array(R')
    # supersampled grid
    nr_box       = dem.domain
    xc_ss,yc_ss  = LinRange(nr_box.xmin,nr_box.xmax,ns*length(xc)),LinRange(nr_box.ymin,nr_box.ymax,ns*length(yc))
    dsx,dsy      = xc_ss[2] - xc_ss[1], yc_ss[2] - yc_ss[1]
    z_bed,z_surf = Data.Array.(evaluate(dem, xc_ss, yc_ss))
    ϕ            = air.*@ones(nx,ny,nz)
    z_bed2  = copy(z_bed)
    z_surf2 = copy(z_surf)
    for ism = 1:nsm
        smooth2D!(z_bed2, z_bed, 1.0)
        smooth2D!(z_surf2, z_surf, 1.0)
        z_bed, z_bed2 = z_bed2, z_bed
        z_surf, z_surf2 = z_surf2, z_surf
    end
    @parallel set_phases!(ϕ,z_surf,z_bed,Rinv,xc[1],yc[1],zc[1],xc_ss[1],yc_ss[1],dx,dy,dz,dsx,dsy,coords...)
    @parallel init_ϕi!(ϕ, ϕx, ϕy, ϕz)
    len_g = sum_g(ϕ.==fluid)
    # visu
    if do_save
        (me==0) && !ispath(out_path) && mkdir(out_path)
        Vn  = @zeros(nx-2,ny-2,nz-2)
        τII = @zeros(nx-2,ny-2,nz-2)
    end
    (me==0) && println(" done. Starting the real stuff 🚀")
    # iteration loop
    err_V=2*ε_V; err_∇V=2*ε_∇V; iter=0; err_evo1=[]; err_evo2=[]
    while !((err_V <= ε_V) && (err_∇V <= ε_∇V)) && (iter <= maxiter)
        @parallel compute_P_τ!(∇V, Pt, τxx, τyy, τzz, τxy, τxz, τyz, Vx, Vy, Vz, ϕ, r, μ_veτ, Gdτ, dx, dy, dz)
        @hide_communication b_width begin
            @parallel compute_V!(Vx, Vy, Vz, Pt, τxx, τyy, τzz, τxy, τxz, τyz, ϕ, ρgx, ρgy, ρgz, dτ_ρ, dx, dy, dz)
            update_halo!(Vx,Vy,Vz)
        end
        iter += 1
        if iter % nchk == 0
            @parallel compute_Res!(Rx, Ry, Rz, Pt, τxx, τyy, τzz, τxy, τxz, τyz, ϕ, ρgx, ρgy, ρgz, dx, dy, dz)
            norm_Rx = norm_g(Rx)/psc*lz/sqrt(len_g)
            norm_Ry = norm_g(Ry)/psc*lz/sqrt(len_g)
            norm_Rz = norm_g(Rz)/psc*lz/sqrt(len_g)
            norm_∇V = norm_g(∇V)/vsc*lz/sqrt(len_g)
            err_V   = maximum([norm_Rx, norm_Ry, norm_Rz])
            err_∇V  = norm_∇V
            # push!(err_evo1, maximum([norm_Rx, norm_Ry, norm_∇V])); push!(err_evo2,iter/nx)
            if (me==0) @printf("# iters = %d, err_V = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_Rz=%1.3e], err_∇V = %1.3e \n", iter, err_V, norm_Rx, norm_Ry, norm_Rz, err_∇V) end
            GC.gc() # force garbage collection
        end
    end
    if do_save
        dim_g = (nx_g()-2, ny_g()-2, nz_g()-2)
        @parallel preprocess_visu!(Vn, τII, Vx, Vy, Vz, τxx, τyy, τzz, τxy, τxz, τyz)
        @parallel apply_mask!(Vn, τII, ϕ)
        out_h5 = joinpath(out_path,out_name)*".h5"
        I = CartesianIndices(( (coords[1]*(nx-2) + 1):(coords[1]+1)*(nx-2),
                               (coords[2]*(ny-2) + 1):(coords[2]+1)*(ny-2),
                               (coords[3]*(nz-2) + 1):(coords[3]+1)*(nz-2) ))
        fields = Dict("ϕ"=>inn(ϕ),"Vn"=>Vn,"τII"=>τII,"Pr"=>inn(Pt))
        (me==0) && print("Saving HDF5 file...")
        write_h5(out_h5,fields,dim_g,I,comm_cart,info) # comm_cart,MPI.Info() are varargs to exclude if using non-parallel HDF5 lib
        # write_h5(out_h5,fields,dim_g,I) # comm_cart,MPI.Info() are varargs to exclude if using non-parallel HDF5 lib
        (me==0) && println(" done")
        # write XDMF
        if me == 0
            print("Saving XDMF file...")
            write_xdmf(joinpath(out_path,out_name)*".xdmf3",out_name*".h5",fields,(xc[2],yc[2],zc[2]),(dx,dy,dz),dim_g)
            println(" done")
        end
    end
    finalize_global_grid()
    return
end

# Stokes3D(load_elevation("../data/alps/data_Rhone.h5"))

# Stokes3D(load_elevation("../data/bedmachine/data_Greenland.h5"))
# Stokes3D(load_elevation("../data/bedmachine/data_Antarctica.h5"))

Stokes3D(generate_elevation(5.0,5.0,(0.0,1.0),0.0,0π,tan(-π/6),0.5,0.9))
