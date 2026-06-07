module PCFM

using NeuralOperators
using Lux
using JLD2, Functors
using cuDNN
using CUDA
using Random
using Optimisers
using Reactant
using KernelAbstractions
using MadNLP, MadNLPGPU, ExaModels

using JuMP
using HDF5

using FFTW 
using Optimization
using OptimizationOptimJL

using ForwardDiff
using ADTypes
using DifferentiationInterface
# Make training API origin explicit for downstream includes.
const Training = Lux.Training

# Include submodules
include("./data.jl")
include("./model.jl")
include("./training.jl")
include("./projection.jl")
include("./sampling.jl")
include("./constraints.jl")

# Export main functions
export FFM
export prepare_input, interpolate_flow
export train_ffm!, sample_ffm, sample_pcfm, sample_pcfm_2d

# Data
export generate_diffusion_data
export load_burgers_batch, load_rd_batch, load_ns_batch

# Constraints
export heat_constraints!, rd_constraints!, rd_constraints_2!, burgers_constraints_BC_Mass!, burgers_constraints_IC_Mass_Flux!, ns_constraints!, ns_enstrophy_constraints!
export burgers_constraints_IC!, burgers_constraints_IC_Mass!



# Alaina's
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
export NSVorticityIPNewtonSolver
export NSEnstrophyLBFGSSolver
export NSEnstrophyIPNewtonSolver
export RDIPNewtonSolver
export BurgersBCMassSolver, BurgersBCMassIPSolver
export BurgersICSolver, BurgersICIPSolver
export BurgersICFluxSolver, BurgersICFluxIPSolver
export HeatICPDEEnergySolver, HeatICPDEEnergyIPSolver

export solve_projection, make_constraint_data

export prepare_input, interpolate_flow
export train_ffm!, sample_ffm, sample_pcfm #_final
export generate_diffusion_data
export get_array_layout, get_slice, n_spatial, set_slice!


end # module PCFM
