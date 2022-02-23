using Statistics, GeoArrays, Interpolations, LinearAlgebra, HDF5

"Filter out all values of `A` based on `mask`."
function my_filter(A, mask)
    return [A[i] for i in eachindex(A) if mask[i] != 0]
end


"""
    lsq_fit(mask, zavg, xv2, yv2)

Linear least-squares fit of mean bedrock and surface data.
"""
function lsq_fit(x,y,z)
    # prepare input for least-squares regression (lsq)
    A       =  ones(length(x),3)
    B       = zeros(length(x),1)
    A[:,1] .= x; A[:,2] .= y; B .= z
    # lsq solve
    return (A'*A)\(A'*B)
end


function axis_angle_rotation_matrix(ax, θ)
    return [cos(θ)+ax[1]^2*(1-cos(θ)) ax[1]*ax[2]*(1-cos(θ))       ax[2]*sin(θ)
            ax[2]*ax[1]*(1-cos(θ))    cos(θ) + ax[2]^2*(1-cos(θ)) -ax[1]*sin(θ)
           -ax[2]*sin(θ)              ax[1]*sin(θ)                       cos(θ)]
end


"""
    extract_geodata(type::DataType, dat_name::String)

Extract geadata and return elevation maps, rotation matrix and origin.

# Arguments
- `type::DataType`: desired data type for elevation data
- `dat_name::String`: input data file
"""
@views function extract_geodata(type::DataType, dat_name::String)
    println("Starting geodata extraction ...")
    println("- load the data")
    file1     = ("../data/alps/IceThick_cr0_$(dat_name).tif")
    file2     = ("../data/alps/BedElev_cr_$(dat_name).tif"  )
    z_thick   = reverse(GeoArrays.read(file1)[:,:,1], dims=2)
    z_bed     = reverse(GeoArrays.read(file2)[:,:,1], dims=2)
    coords    = reverse(GeoArrays.coords(GeoArrays.read(file2)), dims=2)
    (x,y)     = (getindex.(coords,1), getindex.(coords,2))
    xmin,xmax = extrema(x)
    ymin,ymax = extrema(y)
    # center data in x,y plane
    x       .-= 0.5*(xmin + xmax)
    y       .-= 0.5*(ymin + ymax)
    # TODO: a step here could be rotation of the (x,y) plane using bounding box (rotating calipers)
    # define and apply masks
    mask                       = ones(type, size(z_thick))
    mask[ismissing.(z_thick)] .= 0
    z_thick[mask.==0]         .= 0
    z_thick                    = convert(Matrix{type}, z_thick)
    z_bed[ismissing.(z_bed)]  .= mean(my_filter(z_bed,mask))
    z_bed                      = convert(Matrix{type}, z_bed)
    # ground data in z axis
    z_bed                    .-= minimum(z_bed)
    # ice surface elevation and average between bed and ice
    z_surf                     = z_bed .+ z_thick
    z_avg                      = z_bed .+ convert(type,0.5).*z_thick
    println("- perform least square fit")
    αx, αy = lsq_fit(my_filter(x,mask),my_filter(y,mask),my_filter(z_avg,mask))
    # normal vector to the least-squares plane
    # rotation axis - cross product of normal vector and z-axis
    nv = [-αx  ,-αy   ,1.0]; nv ./= norm(nv)
    ax = [nv[2],-nv[1],0.0]; ax ./= norm(ax)
    # rotation matrix from rotation axis and angle
    R  = axis_angle_rotation_matrix(ax,acos(nv[3]))
    println("- save data to ../data/alps/data_$(dat_name).h5")
    h5open("../data/alps/data_$(dat_name).h5", "w") do fid
        create_group(fid, "glacier")
        fid["glacier/x",compress=3]      = x
        fid["glacier/y",compress=3]      = y
        fid["glacier/z_bed",compress=3]  = z_bed
        fid["glacier/z_surf",compress=3] = z_surf
        fid["glacier/R",compress=3]      = R
    end
    println("done.")
    return
end

@time extract_geodata(Float64, "Rhone")