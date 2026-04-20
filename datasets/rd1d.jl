"""
    RD1DDataset

Dataset for 1D Reaction-Diffusion equation solutions loaded from an HDF5 file.

The HDF5 file stores `u` with Python/h5py shape (N_ic, N_bc, Nx, nt).
Julia's HDF5.jl reads this with reversed dimensions: (nt, Nx, N_bc, N_ic).

# Fields
  - `root`: Root directory containing the data file
  - `split`: Dataset split identifier
  - `file`: Opened HDF5 file handle
  - `u`: HDF5 dataset reference
  - `N_ic`: Number of initial conditions
  - `N_bc`: Number of boundary conditions
  - `Nx`: Number of spatial points
  - `nt`: Number of time steps
  - `n_data`: Total number of samples (N_ic × N_bc)
"""
struct RD1DDataset
    root::String
    split::String
    file::HDF5.File
    u::HDF5.Dataset
    N_ic::Int
    N_bc::Int
    Nx::Int
    nt::Int
    n_data::Int
end

"""
    RD1DDataset(root, split, data_file)

Open the HDF5 file at `joinpath(root, data_file)` and return an `RD1DDataset`.
"""
function RD1DDataset(root::String, split::String, data_file::String)
    file = h5open(joinpath(root, data_file), "r")
    u = file["u"]
    # HDF5.jl reverses dims relative to Python/h5py (C vs Fortran order).
    # Python shape: (N_ic, N_bc, Nx, nt) → Julia size: (nt, Nx, N_bc, N_ic)
    nt, Nx, N_bc, N_ic = size(u)
    n_data = N_ic * N_bc
    return RD1DDataset(root, split, file, u, N_ic, N_bc, Nx, nt, n_data)
end

Base.length(ds::RD1DDataset) = ds.n_data

"""
    getindex(ds::RD1DDataset, index::Int)

Return the solution array for sample `index` (1-based) as Float32 of shape (Nx, nt).
"""
function Base.getindex(ds::RD1DDataset, index::Int)
    i_ic, i_bc = divrem(index - 1, ds.N_bc)  # 0-based
    # Julia HDF5 size is (nt, Nx, N_bc, N_ic); slice and permute to (Nx, nt)
    arr = Float32.(ds.u[:, :, i_bc + 1, i_ic + 1])  # (nt, Nx)
    return permutedims(arr, (2, 1))                   # (Nx, nt)
end

Base.close(ds::RD1DDataset) = close(ds.file)
