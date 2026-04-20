"""
    NavierStokesDataset

Dataset for 2D Navier-Stokes vorticity solutions loaded from an HDF5 file.

The HDF5 file stores `u` with Python/h5py shape (nw, nf, s, s, t).
Julia's HDF5.jl reads this with reversed dimensions: (t, s, s, nf, nw).

# Fields
  - `root`: Root directory containing the data file
  - `split`: Dataset split identifier
  - `data_file`: Name of the HDF5 data file
  - `file`: Opened HDF5 file handle
  - `data`: HDF5 dataset reference for `u`
  - `nw`: Number of initial vorticity fields
  - `nf`: Number of forcing fields
  - `s`: Spatial grid size (s × s)
  - `t`: Number of recorded time snapshots
  - `n_data`: Total number of samples (nw × nf)
"""
struct NavierStokesDataset
    root::String
    split::String
    data_file::String
    file::HDF5.File
    data::HDF5.Dataset
    nw::Int
    nf::Int
    s::Int
    t::Int
    n_data::Int
end

"""
    NavierStokesDataset(root, split, data_file)

Open the HDF5 file at `joinpath(root, data_file)` and return a `NavierStokesDataset`.
"""
function NavierStokesDataset(root::String, split::String, data_file::String)
    file = h5open(joinpath(root, data_file), "r")
    data = file["u"]
    # HDF5.jl reverses dims relative to Python/h5py (C vs Fortran order).
    # Python shape: (nw, nf, s, s, t) → Julia size: (t, s, s, nf, nw)
    t, s, _, nf, nw = size(data)
    n_data = nw * nf
    return NavierStokesDataset(root, split, data_file, file, data, nw, nf, s, t, n_data)
end

Base.length(ds::NavierStokesDataset) = ds.n_data

"""
    getindex(ds::NavierStokesDataset, index::Int)

Return the vorticity solution for sample `index` (1-based) as Float32 of shape (s, s, t).
"""
function Base.getindex(ds::NavierStokesDataset, index::Int)
    w, f = divrem(index - 1, ds.nf)  # 0-based
    # Julia HDF5 size is (t, s, s, nf, nw); slice and permute to (s, s, t)
    arr = Float32.(ds.data[:, :, :, f + 1, w + 1])  # (t, s, s)
    return permutedims(arr, (2, 3, 1))               # (s, s, t)
end

Base.close(ds::NavierStokesDataset) = close(ds.file)
