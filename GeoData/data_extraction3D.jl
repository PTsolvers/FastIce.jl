using Statistics, GeoArrays, NCDatasets, Interpolations, LinearAlgebra, HDF5

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

Extract geadata and return bedrock and surface elevation maps, spatial coords and bounding-box rotation matrix.

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

"""
    extract_bm_data(type::DataType, dat_in::String; downscale::Int=20)

Extract BedMachine geadata and return bedrock and surface elevation maps and spatial coords. Bounding-box rotation matrix is I.

# Arguments
- `type::DataType`: desired data type for elevation data
- `dat_in::String`: input data file name
- `downscale::Int`: data downscale, default to 20
"""
function extract_bm_data(type::DataType, dat_in::String; downscale::Int=20)
# DEBUG: why is @views super slow?
    if dat_in == "Antarctica"
        dat_name = "BedMachineAntarctica_2020-07-15_v02"
    elseif dat_in == "Greenland"
        dat_name = "BedMachineGreenland-2021-04-20"
    else
        error("unknown input data")
    end
    
    println("Starting BedMachine $(dat_in) data extraction ...")
    println("- load the data")

    filename = if !isfile("../data/bedmachine_src/$(dat_name).nc")
        error("No $(dat_name).nc file in ../data/bedmachine_src/ folder. See https://sites.uci.edu/morlighem/dataproducts/ for download details.")
    else
        "../data/bedmachine_src/$(dat_name).nc"
    end
    
    bm = NCDataset(filename)
    z_thick   = reverse(bm[:thickness][1:downscale:end,1:downscale:end], dims=2)
    z_bed     = reverse(bm[:bed][1:downscale:end,1:downscale:end], dims=2)
    mask_     = reverse(bm[:mask][1:downscale:end,1:downscale:end], dims=2)
    (xv,yv)   = (convert(Vector{type},bm[:x][1:downscale:end]),convert(Vector{type},reverse(bm[:y][1:downscale:end])))
    (x,y)     = ([x_ for x_=xv, y_=yv],[y_ for x_=xv, y_=yv])
    xmin,xmax = extrema(x)
    ymin,ymax = extrema(y)
    # center data in x,y plane
    x       .-= 0.5*(xmin + xmax)
    y       .-= 0.5*(ymin + ymax)
    # TODO: a step here could be rotation of the (x,y) plane using bounding box (rotating calipers)
    # define and apply masks
    # flag_values   = [0, 1, 2, 3, 4]
    # flag_meanings = ocean, ice_free_land grounded_ice floating_ice lake_vostok
    if dat_in == "Antarctica"
        mask_t = (mask_.==2) .| (mask_.==4) # keep grounded_ice and lake_vostok
        mask_b = (mask_.==0) .| (mask_.==3) # set bedrock to 0 for ocean and floating_ice
    elseif dat_in == "Greenland"
        mask_t = (mask_.==2) # keep grounded_ice and lake_vostok
        mask_b = (mask_.==0) .| (mask_.==3) .| (mask_.==4) # set bedrock to 0 for ocean and floating_ice and non-Greenland land
    end
    z_thick[mask_t.==0] .= 0
    z_bed[mask_b.!=0]   .= 0 
    z_thick   = convert(Matrix{type}, z_thick)
    z_bed     = convert(Matrix{type}, z_bed)
    # ground data in z axis
    # z_bed .-= minimum(z_bed) # TODO: decide whether to centre in z or not
    # ice surface elevation and average between bed and ice
    z_surf    = z_bed .+ z_thick
    R  = [1. 0. 0.; 0. 1. 0.; 0. 0. 1.]
    # display(heatmap(x,y,z_bed',aspect_ratio=1)); error("stop")
    println("- save data to ../data/bedmachine/data_$(dat_in).h5")
    isdir("../data/bedmachine")==false && mkdir("../data/bedmachine")
    h5open("../data/bedmachine/data_$(dat_in).h5", "w") do fid
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

# @time extract_geodata(Float64, "Rhone")

@time extract_bm_data(Float64, "Antarctica"; downscale=4)
# @time extract_bm_data(Float64, "Greenland"; downscale=4)
