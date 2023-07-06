export field_array, scalar_field, vector_field, tensor_field, volfrac_field

@inline field_array(::Type{T}, args...) where {T} = TinyKernels.device_array(T, DEVICE, args...)

@inline scalar_field(::Type{T}, args...) where {T} = field_array(T, args...)

# 2D fields
@inline vector_field(::Type{T}, nx, ny) where {T} = (
    x=field_array(T, nx + 1, ny),
    y=field_array(T, nx, ny + 1)
)

@inline tensor_field(::Type{T}, nx, ny) where {T} = (
    xx=field_array(T, nx, ny),
    yy=field_array(T, nx, ny),
    xy=field_array(T, nx - 1, ny - 1)
)

@inline volfrac_field(::Type{T}, nx, ny) where {T} = (
    c=field_array(T, nx, ny),
    x=field_array(T, nx + 1, ny),
    y=field_array(T, nx, ny + 1),
    xy=field_array(T, nx - 1, ny - 1)
)

# 3D fields
@inline vector_field(::Type{T}, nx, ny, nz) where {T} = (
    x=field_array(T, nx + 1, ny, nz),
    y=field_array(T, nx, ny + 1, nz),
    z=field_array(T, nx, ny, nz + 1)
)

@inline tensor_field(::Type{T}, nx, ny, nz) where {T} = (
    xx=field_array(T, nx, ny, nz),
    yy=field_array(T, nx, ny, nz),
    zz=field_array(T, nx, ny, nz),
    xy=field_array(T, nx - 1, ny - 1, nz - 2),
    xz=field_array(T, nx - 1, ny - 2, nz - 1),
    yz=field_array(T, nx - 2, ny - 1, nz - 1)
)

@inline volfrac_field(::Type{T}, nx, ny, nz) where {T} = (
    c=field_array(T, nx, ny, nz),
    x=field_array(T, nx + 1, ny, nz),
    y=field_array(T, nx, ny + 1, nz),
    z=field_array(T, nx, ny, nz + 1),
    xy=field_array(T, nx - 1, ny - 1, nz - 2),
    xz=field_array(T, nx - 1, ny - 2, nz - 1),
    yz=field_array(T, nx - 2, ny - 1, nz - 1)
)