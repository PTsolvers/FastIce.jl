using DBFTables, DataFrames, Shapefile, Rasters, Plots

"Helper function to mask, trim and pad bedrock and ice thickness data given a glacier polygon."
@views mask_trim(rasterDat, poly, pad) = trim(mask(rasterDat; with=poly); pad=pad)

# SGI_ID  = "E22/03"
# name    = "Morteratsch"

SGI_ID  = "B73/12"
name    = "ArollaHaut"

# SGI_ID  = "B73/14"
# name    = "Arolla"

# SGI_ID  = "B43/03"
# name    = "Rhone"

# SGI_ID  = "B36/26"
# name    = "Aletsch"

# SGI_ID  = "A55f/03"
# name    = "PlaineMorte"

padding = 10

@views function extract_geom(SGI_ID::String, name::String, padding::Int; do_vis=true, do_save=true)
    # find glacier ID
    df  = DataFrame(DBFTables.Table("../data/alps_sgi/swissTLM3D_TLM_GLAMOS.dbf"))
    ID  = df[in([SGI_ID]).(df.SGI),:TLM_BODENB] # and not :UUID field!

    # read in global data
    IceThick = read(Raster("../data/alps_sgi/IceThickness.tif"))
    SurfElev = read(Raster("../data/alps_sgi/SwissALTI3D_r2019.tif"))

    count = 0
    IceThick_stack = []
    for id in ID
        count+=1
        # retrieve shape
        dftable = DataFrame(Shapefile.Table("../data/alps_sgi/swissTLM3D_TLM_BODENBEDECKUNG_ost.shp"))
        if sum(in([id]).(dftable.UUID))==0
            dftable = DataFrame(Shapefile.Table("../data/alps_sgi/swissTLM3D_TLM_BODENBEDECKUNG_west.shp"))
        end
        shape = dftable[in([id]).(dftable.UUID),:geometry]
        # find ice thickness for polygon of interest (glacier), crop and add padding, using global data
        IceThick_stack .= push!(IceThick_stack, mask_trim(IceThick, shape, padding))
    end

    IceThick_cr = mosaic(first, IceThick_stack)

    # crop surface elevation to ice thckness data
    SurfElev_cr = crop(SurfElev; to=IceThick_cr)

    # compute bedrock elevation
    IceThick_cr0 = replace_missing(IceThick_cr, 0.0)
    BedElev_cr = SurfElev_cr .- IceThick_cr0

    if do_vis || do_save
        if isdir("../data/alps")==false mkdir("../data/alps") end
    end

    # visualise
    if do_vis
        p1 = plot(IceThick_cr0, title="Ice thickness")
        p2 = plot(BedElev_cr, title="Bedrock elev.")
        display(plot(p1, p2, dpi=200))
        savefig("../data/alps/fig_$(name).png")
    end
    
    # save
    if do_save
        write("../data/alps/IceThick_cr0_$(name).tif", IceThick_cr0)
        write("../data/alps/SurfElev_cr_$(name).tif" , SurfElev_cr )
        write("../data/alps/BedElev_cr_$(name).tif"  , BedElev_cr  )
    end

    return IceThick_cr0, SurfElev_cr, BedElev_cr
end

@time extract_geom(SGI_ID, name, padding; do_vis=true)
