using Test
using Plastoxid

include("test_wave_functions.jl")

include("test_tolerance.jl")

include("test_surface_interpolations.jl")
include("test_surface_values.jl")
include("test_cohesive_cell.jl")

include("test_coupled_material.jl")

include("test_cell_buffer.jl")

include("test_continuum_element.jl")
include("test_coupled_element.jl")

include("test_jld2_storage.jl")
include("test_initial_guess.jl")
include("test_newton_solver.jl")

include("test_do_timestep.jl")

include("test_grid_import.jl")

## possibly still ok
# include("element_routines/continuum.jl")
# include("element_routines/cohesive.jl")
# include("element_routines/diffusion.jl")
# include("constitutive_drivers_constitutive_driver_vandenbosch_coupled.jl")
