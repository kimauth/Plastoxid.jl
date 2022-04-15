mutable struct ScalarWrapper{T}
    x::T
end

@inline Base.getindex(s::ScalarWrapper) = s.x
@inline Base.setindex!(s::ScalarWrapper, v) = s.x = v
Base.copy(s::ScalarWrapper{T}) where {T} = ScalarWrapper{T}(copy(s.x))


abstract type AbstractCellBuffer end

"""
    CellBuffer{dim, T, CV, FV, C}

Holds data structures that require updating for every cell. For multi-threading, a copy of `CellBuffer` is needed for every thread.

`CellBuffer` is a generic buffer that is used by several `Problem`s.
However, depending on the weak form different fields might need to be buffered (e.g.a different number of`CellValues`).
A Problem might therefore implement its own cell buffer.
"""
struct CellBuffer{dim, T, CV<:CellValues{dim,T}, FV<:Union{Nothing, FaceValues{dim,T}}, MC} <:AbstractCellBuffer
    re::Vector{T}
    ke::Matrix{T}
    dofs::Vector{Int}
    xe::Vector{Vec{dim,T}}
    cv::CV
    # material
    material_id::ScalarWrapper{Int}
    material_cache::MC
    # Neumann BCs
    fv::FV
    nfaces::ScalarWrapper{Int} # technically only needs to be updated once per FieldHandler
    global_cellid::ScalarWrapper{Int}
end

# TODO: pretty printing for Buffer!

function cellbuffer(
    problem::AbstractProblem{M,D},
    ip_f::Interpolation{dim,shape},
    ip_geo::Interpolation{dim,shape},
    qr_cell::QuadratureRule{dim_qr,shape,T},
    qr_face::Union{Nothing,QuadratureRule{dim_face,shape,T}},
) where {M,D,dim,shape,dim_qr,T,dim_face}

    # TODO: assert that D and dim match

    cv, fv = get_values(problem, ip_f, ip_geo, qr_cell, qr_face)

    ngeobasefunctions = Ferrite.getngeobasefunctions(cv)
    nbasefunctions = Ferrite.getnbasefunctions(cv)

    dofs = Vector{Int}(undef, nbasefunctions)

    re = Vector{T}(undef, nbasefunctions)
    ke = Matrix{T}(undef, nbasefunctions, nbasefunctions)

    get_dim(::Ferrite.Values{dim}) where dim = dim
    dim_s = get_dim(cv)
    xe = Vector{Vec{dim_s,T}}(undef, ngeobasefunctions)

    cache = get_cache(first(problem.materials)) # better caching in MaterialModels?

    material_id = ScalarWrapper(0)
    nfaces = ScalarWrapper(0)
    globalid = ScalarWrapper(0)

    cb = CellBuffer(re, ke, dofs, xe, cv, material_id, cache, fv, nfaces, globalid)
    return cb
end

Ferrite.getnquadpoints(cb::CellBuffer) = getnquadpoints(cb.cv)
get_fevalues(cb::CellBuffer) = cb.cv

function Ferrite.reinit!(
    cb::AbstractCellBuffer,
    dh::Ferrite.AbstractDofHandler,
    material_mapping::Dict{Int, Int},
    global_cellid::Int
)
    # reset element stiffness matrix and force vector
    fill!(cb.ke, 0.0)
    fill!(cb.re, 0.0)

    celldofs!(cb.dofs, dh, global_cellid)
    Ferrite.cellcoords!(cb.xe, dh, global_cellid)

    # material
    cb.material_id[] = material_mapping[global_cellid]

    # Neumann BCs
    cb.nfaces[] = nfaces(dh.grid.cells[global_cellid])
    cb.global_cellid[] = global_cellid

    return cb
end

# CellBuffer for InterfaceFicksLaw
struct MixedCellBuffer{dim, T, CV_u<:CellValues{dim,T}, CV_c<:CellValues{dim,T}, FV_c<:FaceValues{dim,T}, MC} <:AbstractCellBuffer
    re::Vector{T}
    ke::Matrix{T}
    dofs::Vector{Int}
    xe::Vector{Vec{dim,T}}
    cv_u::CV_u
    cv_c::CV_c
    # material
    material_id::ScalarWrapper{Int}
    material_cache::MC
    # Neumann BCs
    fv_c::FV_c
    nfaces::ScalarWrapper{Int} # technically only needs to be updated once per FieldHandler
    global_cellid::ScalarWrapper{Int}
end

function cellbuffer(
    problem::InterfaceFicksLaw{M,D},
    ip_f_u::Interpolation{dim,shape},
    ip_f_c::Interpolation{dim,shape},
    ip_geo::Interpolation{dim,shape},
    qr_cell::QuadratureRule{dim_qr,shape,T},
    qr_face::QuadratureRule{dim_face,shape,T},
) where {M,D,dim,shape,T,dim_qr,dim_face}

    # TODO: assert that D and dim match

    cv_u, cv_c, fv_c = get_values(problem, ip_f_u, ip_f_c, ip_geo, qr_cell, qr_face)

    ngeobasefunctions = Ferrite.getngeobasefunctions(cv_u) # same for cv_u & cv_c
    nbasefunctions_u = Ferrite.getnbasefunctions(cv_u)
    nbasefunctions_c = Ferrite.getnbasefunctions(cv_c)

    ndofs = nbasefunctions_u + nbasefunctions_c # per cell
    dofs = Vector{Int}(undef, ndofs)

    re = Vector{T}(undef, ndofs)
    ke = Matrix{T}(undef, ndofs, ndofs)

    get_dim(::Ferrite.Values{dim}) where dim = dim
    dim_s = get_dim(cv_u)
    xe = Vector{Vec{dim_s,T}}(undef, ngeobasefunctions)

    cache = get_cache(first(problem.materials)) # better caching in MaterialModels?

    material_id = ScalarWrapper(0)
    nfaces = ScalarWrapper(0)
    globalid = ScalarWrapper(0)

    cb = MixedCellBuffer(re, ke, dofs, xe, cv_u, cv_c, material_id, cache, fv_c, nfaces, globalid)
    return cb
end

Ferrite.getnquadpoints(cb::MixedCellBuffer) = getnquadpoints(cb.cv_u)
get_fevalues(cb::MixedCellBuffer) = cb.cv_u # use only when it doesn't matter which cv to get!
