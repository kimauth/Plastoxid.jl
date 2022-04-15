## FESet joins a group of cells that use the same celltype and integration scheme
# the cells in a FESet can be assembled in parallel via a coloring scheme

struct CellIds
    globalid::Int
    localid::Int
end

struct FESet{PT<:AbstractProblem, CD<:AbstractCellBuffer, MSS<:AbstractMaterialStates}
    problem::PT
    cellset::Vector{Vector{CellIds}}#NamedTuple{(:global_cellid, :local_cellid, :problem), Tuple{Int64, Int64, PT}}}}
    cellbuffers::Vector{CD} # one entry per thread
    states::MSS
    material_mapping::Dict{Int, Int} # map global cellids to indices of material vector in Problem
end


##################################################################################
# Handling of material states
##################################################################################

## update material states in Newton iterations
"""
    update_material_states!(feset::FESet)

Copy the material states from the buffer `material_states_temp` to `material_states` after convergence is obtained.
"""
function update_material_states!(feset::FESet)
    _update_material_states!(feset.states)
end

# specific for Problems that do not have material states (the minority)
_update_material_states!(::NoMaterialStates) = nothing

# generic for all Problems that have material states
function _update_material_states!(mss::MaterialStates)
    _unsafe_copy_material_states!(mss.material_states, mss.material_states_temp)
end

function _unsafe_copy_material_states!(
    material_states::Vector{Vector{MS}},
    material_states_temp::Vector{Vector{MS}},
) where MS <: AbstractMaterialState

    # dimension check?
    for i=1:length(material_states)
        for j=1:length(material_states[i])
            material_states[i][j] = deepcopy(material_states_temp[i][j])
        end
    end
    return nothing
end


# returns history variables in global cell order for writing to JLD2 file
"""
    get_history_variables(fe_sets::Vector{FESet})

Return the history variables of all cells (within `fe_sets`) in a vector that is ordered according to the global cell ids.
This is used for writing the material states to JLD2 files. The storage format for material states is likely to change in the future.

!!! warning

    This function returns a `Vector{Vector{AbstractMaterialState}}`.
"""
function get_history_variables(fe_sets::Vector{<:FESet})
    ncells = 0
    for fe_set in fe_sets
        for color in fe_set.cellset
            ncells += length(color)
        end
    end
    history_variables = Vector{Vector{AbstractMaterialState}}(undef, ncells)
    for feset in fe_sets
        for color in feset.cellset
            for cell in color
                history_variables[cell.globalid] = get_history_variables(feset.states, cell.localid)
            end
        end
    end
    return history_variables
end

get_history_variables(states::MaterialStates, localid::Int) = states.material_states[localid]
get_history_variables(::NoMaterialStates, ::Int) = AbstractMaterialState[]

# sorts history_variables in cell order into Problems (when reading from JLD2 file)
"""
    set_history_variables!(fe_sets::Vector{FESet}, history_variables::Vector{Vector{AbstractMaterialState}})

Take the material states from `history_variables` and sort them into the `states` field of the `fe_sets`.
`history_variables` should have one entry per cell and be sorted according to the global cell ids.
This is used for reading material states from JLD2 files. The storage format of the material states is likely to change in the future.

"""
function set_history_variables!(
    fesets::Vector{FESet},
    history_variables::Vector{Vector{AbstractMaterialState}}
)
    for feset in fesets
        for color in feset.cellset
            for cell in color
                set_cell_history_variables!(feset.states, cell.localid, history_variables[cell.globalid])
            end
        end
    end
    return nothing
end

function set_cell_history_variables!(
    states::MaterialStates{<:AbstractMaterialState},
    localid::Int,
    history_variables::Vector{<:AbstractMaterialState}
)

    # check number of integration points
    length(states.material_states[localid]) == length(history_variables) ||
        error("Cell and history variables must have equally many integration points.")

    states.material_states[localid] .= history_variables
    return nothing
end

set_cell_history_variables!(::NoMaterialStates, ::Int, ::Vector{MS}) where MS <: AbstractMaterialState = nothing

## set variables in vectors for postprocessing
function set_history_variables!(
    feset::FESet,
    material_states::AbstractVector{<:AbstractVector{MS}},
    history_variables::Vector{Vector{AbstractMaterialState}}
) where MS<:AbstractMaterialState

    for color in feset.cellset
        for cell in color
            cell_states = material_states[cell.localid]
            for qp in 1:length(cell_states)
                cell_states[qp] = history_variables[cell.globalid][qp]
            end
        end
    end
    return nothing
end

##################################################################################

## generate cellset
"""
    preprocess_cellset(grid::Grid, cellset::Set{Int64}, colors::Vector{Vector{Int}})

For a subset of cells `cellset`, distribute local cell ids and return a `Vector{Vector{CellIds}}`
that hold the combinations of global and local cellids sorted according to the cell groups given by `colors`.
Checks that all cells within `cellset` are of the same type (and thus can form a `FESet`).

!!! info
    The coloring can be obtained by:
    ```julia
    colors = Ferrite.create_coloring(g::Grid, alg::ColoringAlgorithm)
    ```
    Possible coloring algorithms are `Ferrite.WORKSTREAM` (good for unstructured grids) and `Ferrite.GREEDY` (good for structured grids).

"""
function preprocess_cellset(
    grid::Grid,
    cellset::Set{Int64}=Set(1:getncells(grid)),
    colors = [1:getncells(grid)],
)

    _check_same_celltype(grid, cellset)

    colored_cellset = Vector{CellIds}[]
    localid = 1
    for color in colors
        color_ids = CellIds[]
        for globalid in color
            if globalid ∈ cellset
                push!(color_ids, CellIds(globalid, localid))
                localid += 1
            end
        end
        push!(colored_cellset, color_ids)
    end

    return colored_cellset
end

# almost the same function exists in Ferrite, but it gives an errormessage related to the CellIterator
# checks that all cells in a given set are of the same type
function _check_same_celltype(grid::Ferrite.AbstractGrid, cellset::Set{Int64})
    celltype = typeof(grid.cells[first(cellset)])
    for cellid in cellset
        if celltype != typeof(grid.cells[cellid])
            error("The cells of this cellset are required to be of the same type, but are not.")
        end
    end
end

"""
    getncells(feset::FESet)

Return the number of cells in the `FESet`.
"""
@inline function Ferrite.getncells(feset::FESet)
    ncells = 0
    for color in feset.cellset
        ncells += length(color)
    end
    return ncells
end

## retrieve cellset from FESet
function get_global_cellset(feset::FESet)
    ncells = getncells(feset)
    global_cellids = Vector{Int}(undef, ncells)
    idx = 1
    for color in feset.cellset
        for cell in color
            global_cellids[idx] = cell.globalid
            idx += 1
        end
    end
    return Set{Int}(global_cellids)
end

"""
    get_localid(feset::FESet, globalid::Int)

Find the local cell id within `feset` that belongs to the cell with `globalid`.
Throws an error if the cell with `globalid` is not part of the `feset`.

"""
function get_localid(feset::FESet, globalid::Int)
    for color in feset.cellset
        for cell in color
            cell.globalid == globalid && return cell.localid
        end
    end
    error("Global id $globalid was not found in FESet.")
end

"""
    preprocess_material_mapping(cellsets::Vector{Set{Int}})

For cells grouped by the `cellsets`, save a dictonary that has the global cell ids as keys and the 
index within `cellsets` as values. 
"""
function preprocess_material_mapping(cellsets::Vector{Set{Int}})
    material_mapping = Dict(j=>i for i=1:length(cellsets) for j in cellsets[i])
    return material_mapping
end


###########################################################################################
# NEW API #
###########################################################################################

"""
    FESet(cellset::Set{Int}, problem::AbstractProblem, cb::AbstractCellBuffer, material_mapping::Dict{Int, Int}[, nthreads=Threads.nthreads()])

Colors the cellset and constructs the corresponding `FESet` for a given number of threads. 
Makes sure the `FESet` is ready for a multithreaded assembly if `nthreads > 1`.
"""
function FESet(
    set::Set{Int},
    grid::Ferrite.AbstractGrid,
    problem::AbstractProblem{M,D},
    cb::AbstractCellBuffer,
    material_mapping::Dict{Int, Int},
    nthreads=Threads.nthreads(),
) where {M,D}

    # TODO: assert that dimensions fit together?

    if nthreads > 1
        colors = create_coloring(grid, set; alg=Ferrite.WORKSTREAM)
        cellset = preprocess_cellset(grid, set, colors)
    else
        cellset = preprocess_cellset(grid, set)
    end

    cellbuffers = [deepcopy(cb) for i=1:nthreads]

    ncells = length(set)
    nquadpoints = getnquadpoints(cb)

    material_states = initial_material_states(first(problem.materials), ncells, nquadpoints)
    material_states_temp = deepcopy(material_states)

    states = MaterialStates(material_states, material_states_temp)

    return FESet(problem, cellset, cellbuffers, states, material_mapping)
end