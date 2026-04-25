"""
    train_ffm!(ffm::FFM, data; epochs=1000, lr=0.001f0,
               use_compiled=false, compiled_funcs=nothing, verbose=true)

Train the Functional Flow Matching model using the given data.

# Arguments

  - `ffm`: FFM model
  - `data`: Training data of shape `(spatial_size..., nt, 1, n_samples)`
  - `epochs`: Number of training epochs
  - `lr`: Learning rate
  - `use_compiled`: Whether to use compiled functions
  - `compiled_funcs`: Compiled functions from `compile_functions`
  - `verbose`: Print progress

# Returns

  - `losses`: Array of training losses
  - `tstate`: Final training state

# Example

```julia
ffm = FFM(spatial_size = (100,), nt = 100)
data = generate_diffusion_data(32, 100, 100, (1.0f0, 5.0f0), (0.0f0, Float32(π)), (0.0f0, 1.0f0))
losses, tstate = train_ffm!(ffm, data; epochs = 1000)
```
"""
function train_ffm!(ffm::FFM, data;
        epochs = 1000,
        lr = 0.001f0,
        use_compiled = true,
        compiled_funcs = nothing,
        verbose = true)
    spatial_size = ffm.config[:spatial_size]
    nt           = ffm.config[:nt]
    emb_channels = ffm.config[:emb_channels]
    device       = ffm.config[:device]

    losses = Float32[]
    tstate = Training.TrainState(ffm.model, ffm.ps, ffm.st, Adam(lr))

    n_samples = size(data)[end]
    data = data |> device

    if use_compiled && compiled_funcs !== nothing
        interpolate_fn    = compiled_funcs.interpolate
        prepare_input_fn  = compiled_funcs.prepare_input
    else
        interpolate_fn    = interpolate_flow
        prepare_input_fn  = prepare_input
    end

    for epoch in 1:epochs
        t   = rand(Float32, n_samples) |> device
        x_0 = randn(Float32, spatial_size..., nt, 1, n_samples) |> device

        x_t      = interpolate_fn(t, x_0, data, n_samples)
        v_target = data .- x_0
        x_input  = prepare_input_fn(x_t, t, spatial_size, nt, n_samples, emb_channels)

        (_, loss, _, tstate) = Training.single_train_step!(
            AutoEnzyme(), MSELoss(), (x_input, v_target), tstate;
            return_gradients = Val(false)
        )

        push!(losses, Float32(loss))
        verbose && (epoch % 100 == 0 || epoch == 1) && println("Epoch $epoch: Loss = $(Float32(loss))")
    end

    return losses, tstate
end
