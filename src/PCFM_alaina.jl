module PCFM

using NeuralOperators
using Lux
using Random
using Optimisers
using FFTW  
# using Reactant
# using MadNLP
# using JuMP
# using Optimization
# using OptimizationOptimJL
using Optimization
using OptimizationOptimJL
using ForwardDiff
using ADTypes
using DifferentiationInterface


# Include submodules
include("./data.jl")
include("./model.jl")
include("./projection.jl")
include("./training.jl")
include("./sampling.jl")

# Export main functions
export FFM
export NoOpSolver
export AbstractProjectionSolver
export AnalyticICProjectionSolver
export AnalyticEnergyProjectionSolver
export AnalyticICEnergyProjectionSolver
export AnalyticMassProjectionSolver
export AnalyticICMassProjectionSolver
export PenaltyLBFGSEnergyProjectionSolver
export PenaltyLBFGSMassProjectionSolver
export IPEnergyProjectionSolver
export IPMassProjectionSolver
export RDSolver
export NSVorticityAnalyticSolver
export NSVorticityLBFGSSolver
export RDIPNewtonSolver
export NSVorticityIPNewtonSolver
export BurgersBCMassSolver, BurgersBCMassIPSolver
export BurgersICFluxSolver, BurgersICFluxIPSolver

export solve_projection, make_constraint_data
export prepare_input, interpolate_flow
export train_ffm!, sample_ffm, sample_pcfm #_final
export generate_diffusion_data
export get_array_layout, get_slice, n_spatial, set_slice!

end # module PCFM