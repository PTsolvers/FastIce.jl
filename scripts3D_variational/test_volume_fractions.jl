using FastIce

using ImplicitGlobalGrid

function main(;dem_path,nz)
    # read data

    # init MPI
    me, dims, nprocs, coords, comm_cart = init_global_grid(grid_size...)

    # fields
    # level set
    Ψ = (
        fluid  = field_array(Float64,nx,ny,nz),
        liquid = field_array(Float64,nx,ny,nz),
    )
    wt = (
        c = (
            fluid  = field_array(Float64,nx,ny,nz),
            liquid = field_array(Float64,nx,ny,nz),
        ),
        x = (
            fluid  = field_array(Float64,nx+1,ny,nz),
            liquid = field_array(Float64,nx+1,ny,nz),
        ),
        y = (
            fluid  = field_array(Float64,nx,ny+1,nz),
            liquid = field_array(Float64,nx,ny+1,nz),
        ),
        z = (
            fluid  = field_array(Float64,nx,ny,nz+1),
            liquid = field_array(Float64,nx,ny,nz+1),
        ),
        xy = (
            fluid  = field_array(Float64,nx-1,ny-1,nz-2),
            liquid = field_array(Float64,nx-1,ny-1,nz-2),
        ),
        xz = (
            fluid  = field_array(Float64,nx-1,ny-2,nz-1),
            liquid = field_array(Float64,nx-1,ny-2,nz-1),
        ),
        yz = (
            fluid  = field_array(Float64,nx-2,ny-1,nz-1),
            liquid = field_array(Float64,nx-2,ny-1,nz-1),
        ),
    )
    # mechanics
    Pr = field_array(Float64,nx,ny,nz)
    τ  = (
        xx = field_array(Float64,nx  ,ny  ,nz  ),
        yy = field_array(Float64,nx  ,ny  ,nz  ),
        zz = field_array(Float64,nx  ,ny  ,nz  ),
        xy = field_array(Float64,nx-1,ny-1,nz-2),
        xz = field_array(Float64,nx-1,ny-2,nz-1),
        yz = field_array(Float64,nx-2,ny-1,nz-1),
    )
    V = (
        x = field_array(Float64,nx+1,ny,nz),
        y = field_array(Float64,nx,ny+1,nz),
        z = field_array(Float64,nx,ny,nz+1),
    )
    # residuals
    Res = (
        Pr = field_array(Float64,nx,ny,nz),
        V = (
            x = field_array(Float64,nx-1,ny-2,nz-2),
            y = field_array(Float64,nx-2,ny-1,nz-2),
            z = field_array(Float64,nx-2,ny-2,nz-1),
        )
    )

    finalize_global_grid()
    return
end

greenland_path = "data/BedMachineGreenland/BedMachineGreenland-v5.nc"
global_region  = (x = (), y = ())

main(;greenland_path,32)