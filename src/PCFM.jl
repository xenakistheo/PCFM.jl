module PCFM

using NeuralOperators
using Lux
using Random
using Optimisers
using Reactant
using MadNLP
using JuMP

# Include submodules
include("./data.jl")
include("./model.jl")
include("./training.jl")
include("./sampling.jl")
include("./constraints.jl")

# Export main functions
export FFM
export prepare_input, interpolate_flow
export train_ffm!, sample_ffm, sample_pcfm
export generate_diffusion_data
export heat_constraints!, rd_constraints!

end # module PCFM
