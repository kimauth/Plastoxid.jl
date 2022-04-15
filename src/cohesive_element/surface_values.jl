# Values for cohesive zone elements
abstract type SurfaceValues{dim,dim_s,T,refshape} <: CellValues{dim_s,T,refshape} end

struct SurfaceVectorValues{dim,dim_s,T<:Real,refshape<:Ferrite.AbstractRefShape,M} <: SurfaceValues{dim,dim_s,T,refshape}
    N::Matrix{Vec{dim_s,T}}
    dNdξ::Matrix{Tensor{2,dim_s,T,M}}
    dNdx::Matrix{Tensor{2,dim_s,T,M}}
    N_mp::Matrix{Vec{dim_s,T}} # shape functions for mid-plane value
    detJdA::Vector{T}
    M::Matrix{T}
    dMdξ::Matrix{Vec{dim_s,T}}
    qr::QuadratureRule{dim,refshape,T}
    R::Vector{Tensor{2,dim_s,T,M}}
    N_qp_dict::Dict{Int, Int}
end

Ferrite.FieldTrait(::Type{<:SurfaceVectorValues}) = Ferrite.VectorValued()

function SurfaceVectorValues(
    qr::QuadratureRule,
    func_interpol::SurfaceInterpolation,
    geom_interpol::SurfaceInterpolation = MidPlaneInterpolation(func_interpol.ip_base)
)
    return SurfaceVectorValues(Float64, qr, func_interpol, geom_interpol)
end

function SurfaceVectorValues(
    ::Type{T},
    quad_rule::QuadratureRule{dim,shape},
    func_interpol::SurfaceInterpolation{dim,shape,order_f,dim_s,ip_base_f},
    geom_interpol::SurfaceInterpolation{dim,shape,order_geo,dim_s,ip_base_geo} = MidPlaneInterpolation(func_interpol.ip_base)
) where {T, dim, shape, order_f, order_geo, ip_base_f, ip_base_geo, dim_s}

    @assert Ferrite.getdim(func_interpol) == Ferrite.getdim(geom_interpol)
    @assert Ferrite.getrefshape(func_interpol) == Ferrite.getrefshape(geom_interpol) == shape
    n_qpoints = length(getweights(quad_rule))

    # mid-plane interpolation corresponding to func_interpol 
    mp_interpol = MidPlaneInterpolation(func_interpol.ip_base)

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol) * dim_s
    N    = fill(zero(Vec{dim_s,T}) * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(Tensor{2,dim_s,T}) * T(NaN), n_func_basefuncs, n_qpoints)
    dNdx = fill(zero(Tensor{2,dim_s,T}) * T(NaN), n_func_basefuncs, n_qpoints)
    N_mp = fill(zero(Vec{dim_s,T}) * T(NaN), n_func_basefuncs, n_qpoints)

    covar_base = fill(zero(Tensor{2,dim_s,T}) * T(NaN), n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M    = fill(zero(T)          * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(Vec{dim_s,T}) * T(NaN), n_geom_basefuncs, n_qpoints)

    for (qp, ξ) in enumerate(quad_rule.points)
        basefunc_count = 1
        for basefunc in 1:getnbasefunctions(func_interpol)
            dNdξ_temp, N_temp = Ferrite.gradient(ξ -> Ferrite.value(func_interpol, basefunc, ξ), ξ, :all)
            N_mp_temp = Ferrite.value(mp_interpol, basefunc, ξ)
            for comp in 1:dim_s
                N_comp = zeros(T, dim_s)
                N_comp[comp] = N_temp
                N[basefunc_count, qp] = Vec{dim_s,T}((N_comp...,))

                N_mp_comp = zeros(T, dim_s)
                N_mp_comp[comp] = N_mp_temp
                N_mp[basefunc_count, qp] = Vec{dim_s,T}((N_mp_comp...,))

                dN_comp = zeros(T, dim_s, dim_s)
                dN_comp[comp, 1:dim] = dNdξ_temp
                dNdξ[basefunc_count, qp] = Tensor{2,dim_s,T}((dN_comp...,))
                basefunc_count += 1
            end
        end
        for basefunc in 1:n_geom_basefuncs
            dMdξ_temp, M[basefunc, qp] = Ferrite.gradient(ξ -> Ferrite.value(geom_interpol, basefunc, ξ), ξ, :all)
            dM_comp = zeros(T, dim_s)
            dM_comp[1:dim] = dMdξ_temp
            dMdξ[basefunc, qp] = Vec{dim_s,T}((dM_comp...,))
        end
    end

    detJdA = fill(T(NaN), n_qpoints)
    N_qp_dict = N_qp_mapping(quad_rule, func_interpol.ip_base)

    MM = Tensors.n_components(Tensors.get_base(eltype(dNdx)))

    SurfaceVectorValues{dim,dim_s,T,shape,MM}(N, dNdξ, dNdx, N_mp, detJdA, M, dMdξ, quad_rule, covar_base, N_qp_dict)
end

@inline getdetJdA(cv::SurfaceValues, q_point::Int) = cv.detJdA[q_point]
@inline Ferrite.getdetJdV(cv::SurfaceValues, q_point::Int) = getdetJdA(cv, q_point)
@inline getR(cv::SurfaceValues, qp::Int) = cv.R[qp]

function Ferrite.reinit!(
    cv::SurfaceValues{dim,dim_s},
    x::AbstractVector{Vec{dim_s,T}},
) where {dim,dim_s,T}

    n_geom_basefuncs = Ferrite.getngeobasefunctions(cv)
    n_func_basefuncs = getnbasefunctions(cv)
    @assert length(x) == n_geom_basefuncs

    @inbounds for qp in 1:length(cv.qr.weights)
        w = cv.qr.weights[qp]
        fecv_J = zero(Tensor{2,dim_s})
        for j in 1:n_geom_basefuncs
            fecv_J += x[j] ⊗ cv.dMdξ[j, qp]
        end
        n = mid_plane_normal(fecv_J) # normal vector of the cohesive element
        e = basevec(n, dim_s)
        fecv_J = fecv_J + n/norm(n) ⊗ e 
        detJ = det(fecv_J)
        detJ > 0.0 || throw(ArgumentError("det(J) is not positive: det(J) = $(detJ)"))
        cv.detJdA[qp] = detJ * w
        # compute rotation matrix
        cv.R[qp] = rotation_matrix(fecv_J)
        # compute dNdx
        Jinv = inv(fecv_J)
        for i in 1:n_func_basefuncs
            cv.dNdx[i, qp] = cv.dNdξ[i, qp] ⋅ Jinv
            if isa(cv, SurfaceScalarValues)
                cv.dN_mpdx[i, qp] = cv.dN_mpdξ[i, qp] ⋅ Jinv
            end
        end
    end
end

mid_plane_normal(dMdx::Tensor{2,2,T}) where T = Vec{2,T}((-dMdx[2,1], dMdx[1,1]))
mid_plane_normal(dMdx::Tensor{2,3,T}) where T = dMdx[:,1] × dMdx[:,2]

function rotation_matrix(dMdx::Tensor{2,dim,T}) where {dim,T}
    R = dMdx
    for d = 1:dim
        v = R[:, d]
        N = Tensor{2,dim,T}((i,j)->i==j ? (i==d ? inv(norm(v)) : one(T) ) : zero(T))
        R = R ⋅ N
    end
    return R
end

function N_qp_mapping(qr, ip)
    mapping = Dict{Int, Int}()
    for (qp, x_qp) in enumerate(qr.points)
        for (i, x_N) in enumerate(Ferrite.reference_coordinates(ip))
            if x_N ≈ x_qp
                push!(mapping, i=>qp)
            end
        end
    end
    return mapping
end


# # meant for use with L2Projector
# # regular 2D function interpolation for unproblematic L2Projection from qp to nodes,
# # CohesiveGeometryInterpolation, because Cohesive elements are geometrically flat.
# function Ferrite.reinit!(cv::CellScalarValues{dim_s,T,refshape,FI,CohesiveGeometryInterpolation{dim, refshape, order, dim_s}},
#     x::AbstractVector{Vec{dim_s,T}}
#     ) where {dim_s,T,refshape,dim,order,FI<:Interpolation{dim_s,refshape,order}}

#     n_geom_basefuncs = Ferrite.getngeobasefunctions(cv)
#     n_func_basefuncs = Ferrite.getn_scalarbasefunctions(cv)
#     @assert length(x) == n_geom_basefuncs
#     isa(cv, CellVectorValues) && (n_func_basefuncs *= dim)


#     @inbounds for qp in 1:length(cv.qr_weights)
#         w = cv.qr_weights[qp]
#         _dXdξ = zeros(Vec{dim_s,T},dim)
#         for j in 1:n_geom_basefuncs
#             for d in 1:dim
#                 _dXdξ[d] += cv.dMdξ[j,qp][d] * x[j]
#             end
#         end
#         # set up full jacobian and jacobi determinant
#         fecv_J = zeros(T,dim_s,dim_s) # dxdξ
#         fecv_J[:,1:dim] = [_dXdξ[j].data[i] for i=1:dim_s, j=1:dim]
#         n = custom_cross(_dXdξ...) # normal vector of the cohesive element
#         fecv_J[:,end] = n ./ norm(n) # normalize normal vector
#         detJ = det(fecv_J)
#         detJ > 0.0 || throw(ArgumentError("det(J) is not positive: det(J) = $(detJ)"))
#         cv.detJdV[qp] = detJ * w
#         # compute dNdx
#         Jinv = Tensor{2, dim_s}(inv(fecv_J))
#         for i in 1:getnbasefunctions(cv)
#             cv.dNdx[i, qp] = cv.dNdξ[i, qp] ⋅ Jinv
#         end
#     end
# end

## SurfaceScalarValues
struct SurfaceScalarValues{dim,dim_s,T<:Real,refshape<:Ferrite.AbstractRefShape,M} <: SurfaceValues{dim, dim_s,T,refshape}
    N::Matrix{T} # shape functions for jump
    dNdξ::Matrix{Vec{dim_s,T}}
    dNdx::Matrix{Vec{dim_s,T}}
    N_mp::Matrix{T} # shape functions for mid-plane value
    dN_mpdξ::Matrix{Vec{dim_s,T}}
    dN_mpdx::Matrix{Vec{dim_s,T}} 
    detJdA::Vector{T}
    M::Matrix{T}
    dMdξ::Matrix{Vec{dim_s,T}}
    qr::QuadratureRule{dim,refshape,T}
    R::Vector{Tensor{2,dim_s,T,M}}
    N_qp_dict::Dict{Int, Int}
end

Ferrite.FieldTrait(::Type{<:SurfaceScalarValues}) = Ferrite.ScalarValued()

function SurfaceScalarValues(
    qr::QuadratureRule,
    func_interpol::SurfaceInterpolation,
    geom_interpol::SurfaceInterpolation=MidPlaneInterpolation(func_interpol.ip_base)
)
    return SurfaceScalarValues(Float64, qr, func_interpol, geom_interpol)
end

function SurfaceScalarValues(
    ::Type{T},
    quad_rule::QuadratureRule{dim,shape},
    func_interpol::SurfaceInterpolation{dim,shape,order_f,dim_s,ip_base_f},
    geom_interpol::SurfaceInterpolation{dim,shape,order_geo,dim_s,ip_base_geo} = MidPlaneInterpolation(func_interpol.ip_base)
) where {T, dim, shape, order_f, order_geo, ip_base_f, ip_base_geo, dim_s}

    @assert Ferrite.getdim(func_interpol) == Ferrite.getdim(geom_interpol)
    @assert Ferrite.getrefshape(func_interpol) == Ferrite.getrefshape(geom_interpol) == shape
    n_qpoints = length(getweights(quad_rule))

    # mid-plane interpolation corresponding to func_interpol 
    mp_interpol = MidPlaneInterpolation(func_interpol.ip_base)

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol)
    N    = fill(zero(T)            * T(NaN), n_func_basefuncs, n_qpoints)
    dNdx = fill(zero(Vec{dim_s,T}) * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(Vec{dim_s,T}) * T(NaN), n_func_basefuncs, n_qpoints)
    N_mp = fill(zero(T)            * T(NaN), n_func_basefuncs, n_qpoints)
    dN_mpdξ = fill(zero(Vec{dim_s,T}) * T(NaN), n_func_basefuncs, n_qpoints)
    dN_mpdx = fill(zero(Vec{dim_s,T}) * T(NaN), n_func_basefuncs, n_qpoints)

    covar_base = fill(zero(Tensor{2,dim_s,T}) * T(NaN), n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M    = fill(zero(T)            * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(Vec{dim_s,T}) * T(NaN), n_geom_basefuncs, n_qpoints)

    for (qp, ξ) in enumerate(quad_rule.points)
        for basefunc in 1:n_func_basefuncs
            dNdξ_temp, N[basefunc, qp] = Ferrite.gradient(ξ -> Ferrite.value(func_interpol, basefunc, ξ), ξ, :all)
            dN_comp = zeros(T, dim_s)
            dN_comp[1:dim] = dNdξ_temp
            dNdξ[basefunc, qp] = Vec{dim_s,T}((dN_comp...,))
            
            dN_mpdξ_temp, N_mp[basefunc, qp] = Ferrite.gradient(ξ -> Ferrite.value(mp_interpol, basefunc, ξ), ξ, :all)
            fill!(dN_comp, zero(T))
            dN_comp[1:dim] = dN_mpdξ_temp
            dN_mpdξ[basefunc, qp] = Vec{dim_s,T}((dN_comp...,))
        end
        for basefunc in 1:n_geom_basefuncs
            dMdξ_temp, M[basefunc, qp] = Ferrite.gradient(ξ -> Ferrite.value(geom_interpol, basefunc, ξ), ξ, :all)
            dM_comp = zeros(T, dim_s)
            dM_comp[1:dim] = dMdξ_temp
            dMdξ[basefunc, qp] = Vec{dim_s,T}((dM_comp...,))
        end
    end

    detJdA = fill(T(NaN), n_qpoints)
    N_qp_dict = N_qp_mapping(quad_rule, func_interpol.ip_base)

    MM = Tensors.n_components(Tensors.get_base(eltype(covar_base)))

    SurfaceScalarValues{dim,dim_s,T,shape,MM}(N, dNdξ, dNdx, N_mp, dN_mpdξ, dN_mpdx, detJdA, M, dMdξ, quad_rule, covar_base, N_qp_dict)
end
