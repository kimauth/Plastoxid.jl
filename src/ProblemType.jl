"""
    AbstractProblem{M<:AbstractMaterial, D<:AbstractDim}

Determines the weak form that is used and contains the corresponding constitutive information.
"""
abstract type AbstractProblem{M<:AbstractMaterial, D<:AbstractDim} end

# rename as get_outofdim_measure
get_outofdim_measurement(problem::AbstractProblem) = problem.reduced_dim_measure

###########################################################################################
struct MechanicalEquilibrium{M,D} <: AbstractProblem{M,D}
    materials::Vector{M}
    dim_type::D
    reduced_dim_measure::Float64
end

function get_values(
    ::MechanicalEquilibrium,
    ip_f::Interpolation{dim,shape},
    ip_geo::Interpolation{dim,shape},
    qr_cell::QuadratureRule{dim,shape},
    qr_face::QuadratureRule{dim_face,shape},
) where {dim,shape,dim_face}
    cv = CellVectorValues(qr_cell, ip_f, ip_geo)
    fv = FaceVectorValues(qr_face, ip_f, ip_geo)
    return cv, fv
end

###########################################################################################
struct FicksLaw{M,D} <: AbstractProblem{M,D}
    materials::Vector{M}
    dim_type::D
    reduced_dim_measure::Float64
end

function get_values(
    ::FicksLaw,
    ip_f::Interpolation{dim,shape},
    ip_geo::Interpolation{dim,shape},
    qr_cell::QuadratureRule{dim,shape},
    qr_face::QuadratureRule{dim_face,shape},
) where {dim,shape,dim_face}
    cv = CellScalarValues(qr_cell, ip_f, ip_geo)
    fv = FaceScalarValues(qr_face, ip_f, ip_geo)
    return cv, fv
end

###########################################################################################
struct Interface{M,D} <: AbstractProblem{M,D}
    materials::Vector{M}
    dim_type::D
    reduced_dim_measure::Float64
end

function get_values(
    ::Interface,
    ip_f::Interpolation{dim,shape},
    ip_geo::Interpolation{dim,shape},
    qr_cell::QuadratureRule{dim_qr,shape},
    qr_face::Nothing=nothing,
) where {dim,dim_qr,shape}
    cv = SurfaceVectorValues(qr_cell, ip_f, ip_geo)
    fv = nothing
    return cv, fv
end

###########################################################################################
struct InterfaceFicksLaw{M,D} <: AbstractProblem{M,D}
    materials::Vector{M}
    shell_thickness::Float64
    dim_type::D
    reduced_dim_measure::Float64
    dof_ranges::Dict{Symbol, UnitRange{Int}}
end

function get_values(
    ::InterfaceFicksLaw,
    ip_f_u::Interpolation{dim,shape},
    ip_f_c::Interpolation{dim,shape},
    ip_geo::Interpolation{dim,shape},
    qr_cell::QuadratureRule{dim_qr,shape},
    qr_face::QuadratureRule{dim_face,shape},
) where {dim,shape,dim_qr,dim_face}

    cv_u = SurfaceVectorValues(qr_cell, ip_f_u, ip_geo)
    cv_c = SurfaceScalarValues(qr_cell, ip_f_c, ip_geo)
    fv_c = SurfaceFaceScalarValues(qr_face, ip_f_c, ip_geo)

    return cv_u, cv_c, fv_c
end

