
# set pre-crack via linear distribution of Δ_max along a vector
# points that are not on the vector are projected onto it via a scalar product

function precrack!(
    feset::FESet{P},
    mesh::Ferrite.AbstractGrid{dim},
    start_point::Vec{dim,T},
    end_point::Vec{dim,T},
    Δ_max_start::Vec{dim,T},
    Δ_max_end::Vec{dim,T}) where {dim,T,P<:Union{Interface, InterfaceFicksLaw}}

    isa(feset, FESet{Interface}) && error("No pre-crack implementation for uncoupled Interfaces yet.")

    # line parameter along which we run
    s_end = norm(end_point-start_point)

    # extract data
    cv = feset.celldata[1].cv_mech

    # loop over all qp
    for color in feset.cellset
        for cell in color
            getcoordinates!(feset.celldata[1].coords, mesh, cell.global_cellid)
            ms = cell.problem.material_states[cell.local_cellid]
            for qp in 1:getnquadpoints(cv)
                x = spatial_coordinate(cv, qp, feset.celldata[1].coords)
                s = project_to_line(x, start_point, end_point)
                if s >= 0.0 && s <= s_end
                    Δ_max = Δ_max_start + (Δ_max_end-Δ_max_start)*s/s_end
                    ms[qp] = CoupledKolluriState(ms[qp].T, Δ_max, ms[qp].c_max)
                end
            end
        end
    end
    return nothing
end

function project_to_line(x::Vec{dim,T},
    start_point::Vec{dim,T},
    end_point::Vec{dim,T},
    ) where {dim, T}
    
    line = end_point - start_point
    line_norm = 1/norm(line)*line

    projection = (x-start_point) ⋅ line_norm

    return projection
end

# give a smooth initial state to diffusion such that mesh discretization is not a problem in the first time stress_components
# only meant for linear isoparametric elements
function init_diffusion!(
    var,
    feset::FESet{P},
    dh::MixedDofHandler,
    start_point::Vec{dim,T},
    end_point::Vec{dim,T},
    c_boundary::T;
    bounding_box::Tuple{Vec{dim}, Vec{dim}} = (zero(Vec{dim}), ones(Vec{dim}))
    ) where {dim,T,P<:InterfaceFicksLaw}

    # line parameter along which we run
    s_end = norm(end_point-start_point)

    problem = feset.problem

    # loop over all cells
    for cell in CellIterator(dh, collect(get_global_cellset(feset)))
        coords = getcoordinates(cell)
        dofs = getcelldofs(cell)
        for i in 1:length(coords)
            x = coords[i]
            if is_in_box(x, bounding_box)
                s = project_to_line(x, start_point, end_point)
                if s >= 0.0 && s <= s_end
                    c = c_boundary*erfc(s/s_end*3)
                    set_node_value!(var, c, dofs, problem, i)
                end
            end
        end
    end
    return nothing
end

set_node_value!(var, c, dofs, problem::InterfaceFicksLaw, i) = (var[dofs[problem.dof_ranges[:c]][i]] = c)
set_node_value!(var, c, dofs, ::FicksLaw, i) = (var[dofs[i]] = c)

function is_in_box(x::Vec{dim}, box::Tuple{Vec{dim}, Vec{dim}}) where dim
    for d in 1:dim
        if x[d] < box[1][d] || x[d] > box[2][d]
            return false
        end
    end
    return true
end
