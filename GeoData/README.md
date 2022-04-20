# GeoData
Helper functions to select Alpine glacier geometry based on [SGI catalogue data ID](../data/SwissGlacierThickness2020.pdf) and to preprocess related elevation data.

<img src="../docs/images/fig_Rhone.png" alt="Rhone glacier data" width="600">

## Workflow
### Geometry selection
First run the [geometry_selection.jl](GeoData/geometry_selection.jl) script to extract, for a given glacier outline, the following data which will be saved as GeoTif (`.tif`):
- Ice thickness
- Surface elevation
- Bedrock elevation

The `geom_select` function expects the following data to be available in a `data/alps_sgi` folder:
- IceThickness.tif
- SwissALTI3D_r2019.tif
- swissTLM3D_TLM_BODENBEDECKUNG_ost.dbf
- swissTLM3D_TLM_BODENBEDECKUNG_ost.shp
- swissTLM3D_TLM_BODENBEDECKUNG_west.dbf
- swissTLM3D_TLM_BODENBEDECKUNG_west.shp
- swissTLM3D_TLM_GLAMOS.dbf

The `alps_sgi` folder size is about 6 GB and can be [downloaded here (dropbox)](https://www.dropbox.com/s/3htehzra9bv6j75/alps_sgi.zip?dl=0). Upon download, unzip and place it in the `data` folder. _See [Sources](#sources) for references._

The `geom_select` function takes as argument the glacier `SGI_ID` and the corresponding name `name`. As keyword args, one can modify `padding`, and switch-off viualisation `do_vis=false` or saving `do_save=false`. Type `? geom_select` in the REPL for more details.

### Data extraction
The [data_extraction.jl](GeoData/data_extraction.jl) function extracts geadata and returns bedrock and surface elevation maps, spatial coords and bounding-box rotation matrix, taking as input the ice thickness and bedrock elevation data generated in the previous step.

This step outputs an HDF5 file conataining, e.g., the following fields to be further used as input for numerical simulation:
```julia-repl
🗂️ HDF5.File: ../data/alps/data_Rhone.h5
└─ 📂 glacier
   ├─ 🔢 R
   ├─ 🔢 x
   ├─ 🔢 y
   ├─ 🔢 z_bed
   └─ 🔢 z_surf
```

## Sources
The sources of the files contained in the Dropbox folder for download are:
- [Swisstopo swissTLM3D](https://www.swisstopo.admin.ch/en/geodata/landscape/tlm3d.html#download)
  - swissTLM3D_TLM_*.dbf | [swisstlm3d_2022-03_2056_5728.shp.zip](https://data.geo.admin.ch/ch.swisstopo.swisstlm3d/swisstlm3d_2022-03/swisstlm3d_2022-03_2056_5728.shp.zip) (located in `TLM_BB/` upon unzip)

- [Swiss Glacier Thickness – Release 2020 (ETH Research Collection)](https://www.research-collection.ethz.ch/handle/20.500.11850/434697)
  - IceThickness.tif | [04_IceThickness_SwissAlps.zip (ZIP, 1.605Gb)](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/434697/04_IceThickness_SwissAlps.zip?sequence=10&isAllowed=y)
  - SwissALTI3D_r2019.tif | [08_SurfaceElevation_SwissAlps.zip (ZIP, 1.837Gb)](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/434697/08_SurfaceElevation_SwissAlps.zip?sequence=41&isAllowed=y)
