
function initial_material_states(material::AbstractMaterial, ncells::Int, nquadpoints::Int)
    return [[initial_material_state(material) for qp in 1:nquadpoints] for i in 1:ncells]
end

# Handle the cases of having / not having material states in FESet
abstract type AbstractMaterialStates end

"""
    NoMaterialStates <: AbstractMaterialStates

Singleton type of `MaterialStates` in case no material states are needed for a weak form.
"""
struct NoMaterialStates <: AbstractMaterialStates end


"""
    MaterialStates{MS<:AbstractMaterialState} <: AbstractMaterialStates

Holds the last converged material states and a buffer for material states during iterations.
"""
struct MaterialStates{MS<:AbstractMaterialState} <: AbstractMaterialStates
    material_states::Vector{Vector{MS}}
    material_states_temp::Vector{Vector{MS}}
end