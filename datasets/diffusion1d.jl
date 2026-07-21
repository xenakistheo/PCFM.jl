"""
    HeatDataset

Dataset for 1D heat (diffusion) equation solutions loaded from an HDF5 file.

The HDF5 file stores `u` with Python/h5py shape (n_data, Nx, Nt).
Julia's HDF5.jl reads this with reversed dimensions: (Nt, Nx, n_data).

# Fields
  - `root`: Root directory containing the data file
  - `split`: Dataset split identifier
  - `file`: Opened HDF5 file handle
  - `u`: HDF5 dataset reference
  - `Nx`: Number of spatial points
  - `Nt`: Number of time steps
  - `n_data`: Total number of samples
"""
struct HeatDataset
    root::String
    split::String
    file::HDF5.File
    u::HDF5.Dataset
    Nx::Int
    Nt::Int
    n_data::Int
end

"""
    HeatDataset(root, split, data_file)

Open the HDF5 file at `joinpath(root, data_file)` and return a `HeatDataset`.
"""
function HeatDataset(root::String, split::String, data_file::String)
    file = h5open(joinpath(root, data_file), "r")
    u = file["u"]
    # HDF5.jl reverses dims relative to Python/h5py (C vs Fortran order).
    # Python shape: (n_data, Nx, Nt) → Julia size: (Nt, Nx, n_data)
    Nt, Nx, n_data = size(u)
    return HeatDataset(root, split, file, u, Nx, Nt, n_data)
end

Base.length(ds::HeatDataset) = ds.n_data

"""
    getindex(ds::HeatDataset, index::Int)

Return the solution array for sample `index` (1-based) as Float32 of shape (Nx, Nt).
"""
function Base.getindex(ds::HeatDataset, index::Int)
    arr = Float32.(ds.u[:, :, index])  # (Nt, Nx)
    return permutedims(arr, (2, 1))     # (Nx, Nt)
end

Base.close(ds::HeatDataset) = close(ds.file)
