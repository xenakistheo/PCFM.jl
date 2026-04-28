"""
    FFM

Structure holding the Fourier Neural Operator model for Functional Flow Matching.

# Fields

  - `model`: The FNO model
  - `ps`: Parameters
  - `st`: States
  - `config`: Configuration dictionary

# Configuration

  - `spatial_size`: Tuple of spatial grid dimensions, e.g. `(100,)` for 1D or `(64, 64)` for 2D
  - `nt`: Temporal resolution
  - `emb_channels`: Number of time embedding channels
  - `hidden_channels`: Number of hidden channels in FNO
  - `proj_channels`: Number of projection channels
  - `n_layers`: Number of FNO layers
  - `modes`: Fourier modes tuple — must have `length(spatial_size) + 1` entries
"""
struct FFM{M, P, S}
    model::M
    ps::P
    st::S
    config::Dict{Symbol, Any}
end

"""
    FFM(; spatial_size=(100,), nt=100, emb_channels=32, hidden_channels=64,
        proj_channels=256, n_layers=4, modes=nothing,
        device=reactant_device(), rng=Random.default_rng())

Create a Functional Flow Matching model with FNO backbone.

`spatial_size` is a tuple of spatial grid dimensions. `modes` must have
`length(spatial_size) + 1` entries (one per spatial dim plus one temporal);
if omitted, sensible defaults are computed automatically.

The legacy keyword `nx` is accepted as a shorthand for `spatial_size = (nx,)`.

# Examples

```julia
# 1D diffusion / Burgers
ffm = FFM(spatial_size = (100,), nt = 100, modes = (32, 32))

# 2D Navier-Stokes
ffm = FFM(spatial_size = (64, 64), nt = 50, modes = (12, 12, 16))
```
"""
function FFM(;
        spatial_size = nothing,
        nx = nothing,    # legacy; prefer spatial_size
        nt = 100,
        emb_channels = 32,
        hidden_channels = 64,
        proj_channels = 256,
        n_layers = 4,
        modes = nothing,
        device = reactant_device(),
        rng = Random.default_rng()
)
    # Resolve spatial_size from legacy nx if needed
    if spatial_size === nothing
        nx = something(nx, 100)
        spatial_size = (nx,)
    else
        spatial_size = Tuple(spatial_size)
    end

    ndim_s = length(spatial_size)

    # Default modes: min(16, sz÷4) per spatial dim, min(16, nt÷4) temporal
    if modes === nothing
        modes = ntuple(i -> i <= ndim_s ? min(16, spatial_size[i] ÷ 4) : min(16, nt ÷ 4), ndim_s + 1)
    end
    modes = Tuple(modes)
    @assert length(modes) == ndim_s + 1 "modes must have length(spatial_size)+1=$(ndim_s+1) entries, got $(length(modes))"

    # u channel + sinusoidal time embedding + one position channel per spatial dim + temporal position
    in_channels = 1 + emb_channels + ndim_s + 1

    fno = FourierNeuralOperator(
        modes,
        in_channels,
        1,
        hidden_channels;
        num_layers = n_layers,
        lifting_channel_ratio = proj_channels ÷ hidden_channels,
        projection_channel_ratio = proj_channels,
        activation = gelu,
        fno_skip = :linear,
        channel_mlp_skip = :soft_gating,
        use_channel_mlp = true,
        channel_mlp_expansion = 1.0,
        positional_embedding = :none,
        stabilizer = tanh
    )

    ps, st = Lux.setup(rng, fno) |> device

    config = Dict{Symbol, Any}(
        :spatial_size => spatial_size,
        :nt => nt,
        :emb_channels => emb_channels,
        :hidden_channels => hidden_channels,
        :proj_channels => proj_channels,
        :n_layers => n_layers,
        :modes => modes,
        :device => device
    )
    # Keep :nx for backward compat with 1D callers
    ndim_s == 1 && (config[:nx] = spatial_size[1])

    return FFM(fno, ps, st, config)
end

"""
    compile_model(ffm::FFM)

Compile the FNO model for faster execution with Reactant.

Returns compiled model function.
"""
function compile_model(ffm::FFM)
    spatial_size = ffm.config[:spatial_size]
    nt = ffm.config[:nt]
    emb_channels = ffm.config[:emb_channels]
    device = ffm.config[:device]
    ndim_s = length(spatial_size)
    n_in = 1 + emb_channels + ndim_s + 1

    x_test = randn(Float32, spatial_size..., nt, n_in, 32) |> device
    model_compiled = Reactant.@compile ffm.model(x_test, ffm.ps, Lux.testmode(ffm.st))
    return model_compiled
end

"""
    prepare_input(x_t, t, spatial_size, nt, n_samples, emb_dim; max_positions=2000)

Prepare input tensor for FNO by concatenating state, time embedding, and position embeddings.

Works for any number of spatial dimensions: 1D `(nx, nt, 1, batch)`,
2D `(nx, ny, nt, 1, batch)`, etc.

# Arguments

  - `x_t`: Current state `(spatial_size..., nt, 1, n_samples)`
  - `t`: Time values `(n_samples,)`
  - `spatial_size`: Tuple of spatial grid sizes, e.g. `(100,)` or `(64, 64)`
  - `nt`: Number of temporal points
  - `n_samples`: Batch size
  - `emb_dim`: Sinusoidal time embedding channels
  - `max_positions`: Scale for time embedding

# Returns

  - Input tensor of shape `(spatial_size..., nt, 1+emb_dim+length(spatial_size)+1, n_samples)`
"""
function prepare_input(x_t, t, spatial_size::Tuple, nt, n_samples, emb_dim; max_positions = 2000)
    ndim_s = length(spatial_size)
    grid_dims = (spatial_size..., nt)   # all non-channel, non-batch dimensions
    total_grid = ndim_s + 1             # spatial + temporal
    chan_dim = total_grid + 1           # channel axis index in the tensor
    batch_dim = total_grid + 2

    backend = KernelAbstractions.get_backend(t)

    # Sinusoidal time embedding: (n_samples, emb_dim) → broadcast to full grid
    timesteps = t .* Float32(max_positions)
    half_dim = emb_dim ÷ 2
    emb_scale = Float32(log(max_positions)) / Float32(half_dim - 1)
    emb_base = KernelAbstractions.adapt(backend, exp.(Float32.(-collect(0:(half_dim - 1)) .* emb_scale)))
    t_emb = hcat(sin.(timesteps * emb_base'), cos.(timesteps * emb_base'))  # (n_samples, emb_dim)
    t_emb = permutedims(t_emb, (2, 1))                                       # (emb_dim, n_samples)

    t_shape = ntuple(i -> i == chan_dim ? emb_dim : (i == batch_dim ? n_samples : 1), total_grid + 2)
    t_rep   = ntuple(i -> i <= total_grid ? grid_dims[i] : 1, total_grid + 2)
    t_emb   = repeat(reshape(t_emb, t_shape), t_rep...)

    # One normalised position channel per grid dimension (spatial dims + temporal)
    pos_channels = [begin
        sz    = grid_dims[i]
        pos_i = KernelAbstractions.adapt(backend, Float32.(collect(range(0.0f0, 1.0f0, length = sz))))
        shape = ntuple(d -> d == i ? sz : 1, total_grid + 2)
        rep   = ntuple(d -> d == i ? 1 :
                            (d <= total_grid ? grid_dims[d] :
                            (d == batch_dim  ? n_samples : 1)), total_grid + 2)
        repeat(reshape(pos_i, shape), rep...)
    end for i in 1:total_grid]

    return cat(x_t, t_emb, pos_channels...; dims = chan_dim)
end

# Backward-compatible method: accept nx::Int as a 1-element spatial_size
prepare_input(x_t, t, nx::Int, nt, n_samples, emb_dim; kwargs...) =
    prepare_input(x_t, t, (nx,), nt, n_samples, emb_dim; kwargs...)

"""
    interpolate_flow(t, x_0, data, n_samples)

Linear interpolation between noise and data for flow matching.

x_t = (1-t)*x_0 + t*x_1

# Arguments

  - `t`: Time values (n_samples,)
  - `x_0`: Noise/initial state
  - `data`: Target data
  - `n_samples`: Batch size

# Returns

  - Interpolated state x_t
"""
function interpolate_flow(t, x_0, data, n_samples)
    # Reshape t to broadcast over all dims except batch (works for any tensor rank)
    t_expanded = reshape(t, ntuple(_ -> 1, ndims(x_0) - 1)..., n_samples)
    x_t = (1 .- t_expanded) .* x_0 .+ t_expanded .* data
    return x_t
end

"""
    compile_functions(ffm::FFM, batch_size::Int)

Compile all helper functions (model, interpolation, input preparation) with Reactant.

Returns a NamedTuple with compiled functions.
"""
function compile_functions(ffm::FFM, batch_size::Int)
    spatial_size = ffm.config[:spatial_size]
    nt            = ffm.config[:nt]
    emb_channels  = ffm.config[:emb_channels]
    device        = ffm.config[:device]
    ndim_s        = length(spatial_size)
    n_in          = 1 + emb_channels + ndim_s + 1

    x_test    = randn(Float32, spatial_size..., nt, n_in, batch_size) |> device
    t_test    = rand(Float32, batch_size) |> device
    x_0_test  = randn(Float32, spatial_size..., nt, 1, batch_size) |> device
    data_test = randn(Float32, spatial_size..., nt, 1, batch_size) |> device

    model_compiled      = Reactant.@compile ffm.model(x_test, ffm.ps, Lux.testmode(ffm.st))
    interpolate_compiled = Reactant.@compile interpolate_flow(t_test, x_0_test, data_test, batch_size)

    x_t_test = interpolate_flow(t_test, x_0_test, data_test, batch_size)
    prepare_input_compiled = Reactant.@compile prepare_input(
        x_t_test, t_test, spatial_size, nt, batch_size, emb_channels)

    return (
        model         = model_compiled,
        interpolate   = interpolate_compiled,
        prepare_input = prepare_input_compiled
    )
end
