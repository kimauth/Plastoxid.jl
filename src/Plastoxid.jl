module Plastoxid

using Reexport
using Rotations
using Printf
@reexport using SparseArrays
@reexport using Ferrite

using NLsolve
using JLD2
using LinearAlgebra
using SpecialFunctions
using ForwardDiff
using NearestNeighbors
@reexport using MaterialModels
using PrettyTables
using DataStructures: CircularBuffer
using Formatting
using StaticArrays

using Logging
import DrWatson
using Tar
using CodecZstd



#adds on to Ferrite
include("cohesive_element/surf_interpolations.jl")
include("cohesive_element/surface_values.jl")
include("cohesive_element/surface_face_values.jl")
include("cohesive_element/surface_common_values.jl")
include("cohesive_element/CohesiveCell.jl")
export CohesiveCell, CohesiveQuadrilateral, CohesiveQuadraticQuadrilateral, CohesiveTetrahedron, CohesiveQuadraticTetrahedron, CohesiveHexahedron

include("Ferrite_add_on.jl")
export SurfaceInterpolation, MidPlaneInterpolation, JumpInterpolation
export SurfaceVectorValues, SurfaceScalarValues, getdetJdA, getR
export SurfaceFaceScalarValues
export shape_midplane_value, shape_midplane_gradient, function_midplane_value, function_midplane_gradient
export function_inplane_value, function_inplane_gradient

include("time_stepping/wave_functions.jl")
export triangular_wave, holdtime_wave
include("time_stepping/time_iterators.jl")
export BasicTimeIterator, CyclicHoldTimeIterator
export BasicTimeIterator_linear, BasicTimeIterator_cyclic
export cyclic_hold_loading, initial_loading_params, full_loading

include("global_solver/tolerance.jl")
export AbstractToleranceScaling
export AbsoluteTolerance, ReactionScaling
export AbstractTolerance, update_tolerance!, check_convergence
export FieldTolerance, GlobalTolerance

include("materials/material_coupled.jl")
export CoupledKolluri, CoupledKolluriState
include("materials/material_diffusion.jl")
export DiffusionCoefficient
# include("materials/material_cohesive.jl") # move functionality to MaterialModels.jl
include("materials/material_utilities.jl")


include("ProblemType.jl")
export MechanicalEquilibrium, FicksLaw, Interface, InterfaceFicksLaw
include("CellData.jl")
export CellBuffer, CoupledCellBuffer, cellbuffer


include("FESet.jl")

include("file_handling/jld2_storage.jl")
export JLD2Storage
include("file_handling/simulation_files.jl")
export SimulationFiles

include("global_solver/initial_guess.jl")
export AbstractGuess, NoGuess, FunctionBasedGuess, TimeBasedGuess
export initial_guess!

include("global_solver/residual_printout.jl")

include("global_solver/solver_newton.jl")
export Newton
export iterate!

include("global_solver/do_timestep.jl")

include("global_solver/run_simulation.jl")
export run_simulation


include("assembly/Neumann.jl")
include("assembly/assembly.jl")
include("assembly/uncoupled_assemble_cell.jl")

include("assembly/coupled_element_routine.jl")
export assembly!, assemble_cell!



include("mesh_import/mesh_reader.jl")
include("mesh_import/abaqus2Ferrite_grid.jl")
include("mesh_import/neper.jl")
export abaqus2Ferrite_grid, read_orientations


include("initial_damage.jl")

# include("cohesive_element/solid_shell_cellvalues.jl")

include("postprocessing/addon_plasticity.jl")
include("postprocessing/addon_crystalplasticity.jl")
include("postprocessing/addon_coupled_cohesive_law.jl")
include("postprocessing/postprocessing_materials.jl")
include("postprocessing/postprocessing_cohesive_materials.jl")
include("postprocessing/postprocessing_weakforms.jl")
include("postprocessing/postprocessing_fesets.jl")
include("postprocessing/entry_point_postprocessing.jl")
include("postprocessing/area_averaging.jl")
export area_averages, postprocess!, postprocess

function __init__()
    merge!(Ferrite.celltypes, cohesive_celltypes)
end

end # module
