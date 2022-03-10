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

The `alps_sgi` folder size is about 6 GB and can be [downloaded here (dropbox)](https://www.dropbox.com/s/3htehzra9bv6j75/alps_sgi.zip?dl=0). Upon download, unzip and place it in the `data` folder.

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
