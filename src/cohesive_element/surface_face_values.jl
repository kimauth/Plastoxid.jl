abstract type SurfaceFaceValues{dim,dim_s,T,refshape} <: FaceValues{dim_s,T,refshape} end

struct SurfaceFaceScalarValues{dim,dim_s,T<:Real,refshape<:Ferrite.AbstractRefShape,M} <: SurfaceFaceValues{dim, dim_s,T,refshape}
    N::Array{T, 3} # shape functions for jump
    dNdξ::Array{Vec{dim_s,T}, 3}
    dNdx::Array{Vec{dim_s,T}, 3}
    N_mp::Array{T, 3} # shape functions for mid-plane value
    dN_mpdξ::Array{Vec{dim_s,T}, 3}
    dN_mpdx::Array{Vec{dim_s,T}, 3} 
    detJdA::Matrix{T}
    normals::Vector{Vec{dim_s,T}}
    M::Array{T,3}
    dMdξ::Array{Vec{dim_s,T}, 3}
    qr::QuadratureRule{<:Any,refshape,T}
    R::Vector{Tensor{2,dim_s,T,M}}
    current_face::Ferrite.ScalarWrapper{Int}
end

Ferrite.FieldTrait(::Type{<:SurfaceFaceScalarValues}) = Ferrite.ScalarValued()

function SurfaceFaceScalarValues(
    qr::QuadratureRule,
    func_interpol::SurfaceInterpolation,
    geom_interpol::SurfaceInterpolation=MidPlaneInterpolation(func_interpol.ip_base)
)
    return SurfaceFaceScalarValues(Float64, qr, func_interpol, geom_interpol)
end

function SurfaceFaceScalarValues(
    ::Type{T},
    quad_rule::QuadratureRule{dim_qr,shape},
    func_interpol::SurfaceInterpolation{dim,shape,order_f,dim_s,ip_base_f},
    geom_interpol::SurfaceInterpolation{dim,shape,order_geo,dim_s,ip_base_geo} = MidPlaneInterpolation(func_interpol.ip_base)
) where {T, dim_qr, dim, shape, order_f, order_geo, ip_base_f, ip_base_geo, dim_s}

    @assert Ferrite.getdim(func_interpol) == Ferrite.getdim(geom_interpol)
    @assert Ferrite.getrefshape(func_interpol) == Ferrite.getrefshape(geom_interpol) == shape
    n_qpoints = length(getweights(quad_rule))

    face_quad_rule = Ferrite.create_face_quad_rule(quad_rule, func_interpol)
    n_faces = length(face_quad_rule)

    # Normals
    normals = zeros(Vec{dim_s,T}, n_qpoints)

    # mid-plane interpolation corresponding to func_interpol 
    mp_interpol = MidPlaneInterpolation(func_interpol.ip_base)

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol)
    N    = fill(zero(T)            * T(NaN), n_func_basefuncs, n_qpoints, n_faces)
    dNdx = fill(zero(Vec{dim_s,T}) * T(NaN), n_func_basefuncs, n_qpoints, n_faces)
    dNdξ = fill(zero(Vec{dim_s,T}) * T(NaN), n_func_basefuncs, n_qpoints, n_faces)
    N_mp = fill(zero(T)            * T(NaN), n_func_basefuncs, n_qpoints, n_faces)
    dN_mpdξ = fill(zero(Vec{dim_s,T}) * T(NaN), n_func_basefuncs, n_qpoints, n_faces)
    dN_mpdx = fill(zero(Vec{dim_s,T}) * T(NaN), n_func_basefuncs, n_qpoints, n_faces)

    covar_base = fill(zero(Tensor{2,dim_s,T}) * T(NaN), n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M    = fill(zero(T)            * T(NaN), n_geom_basefuncs, n_qpoints, n_faces)
    dMdξ = fill(zero(Vec{dim_s,T}) * T(NaN), n_geom_basefuncs, n_qpoints, n_faces)

    for face in 1:n_faces, (qp, ξ) in enumerate(face_quad_rule[face].points)
        for basefunc in 1:n_func_basefuncs
            dNdξ_temp, N[basefunc, qp, face] = Ferrite.gradient(ξ -> Ferrite.value(func_interpol, basefunc, ξ), ξ, :all)
            dN_comp = zeros(T, dim_s)
            dN_comp[1:dim] = dNdξ_temp
            dNdξ[basefunc, qp, face] = Vec{dim_s,T}((dN_comp...,))
            
            dN_mpdξ_temp, N_mp[basefunc, qp, face] = Ferrite.gradient(ξ -> Ferrite.value(mp_interpol, basefunc, ξ), ξ, :all)
            fill!(dN_comp, zero(T))
            dN_comp[1:dim] = dN_mpdξ_temp
            dN_mpdξ[basefunc, qp, face] = Vec{dim_s,T}((dN_comp...,))
        end
        for basefunc in 1:n_geom_basefuncs
            dMdξ_temp, M[basefunc, qp, face] = Ferrite.gradient(ξ -> Ferrite.value(geom_interpol, basefunc, ξ), ξ, :all)
            dM_comp = zeros(T, dim_s)
            dM_comp[1:dim] = dMdξ_temp
            dMdξ[basefunc, qp, face] = Vec{dim_s,T}((dM_comp...,))
        end
    end

    detJdA = fill(T(NaN), n_qpoints, n_faces)

    MM = Tensors.n_components(Tensors.get_base(eltype(covar_base)))

    SurfaceFaceScalarValues{dim,dim_s,T,shape,MM}(N, dNdξ, dNdx, N_mp, dN_mpdξ, dN_mpdx, detJdA, normals, M, dMdξ, quad_rule, covar_base, Ferrite.ScalarWrapper(0))
end

function Ferrite.reinit!(
    fv::SurfaceFaceValues{dim,dim_s},
    x::AbstractVector{Vec{dim_s,T}},
    face::Int,
) where {dim,dim_s,T}

    n_geom_basefuncs = Ferrite.getngeobasefunctions(fv)
    n_func_basefuncs = getnbasefunctions(fv)
    @assert length(x) == n_geom_basefuncs
    @boundscheck checkface(fv, face)
    
    fv.current_face[] = face

    @inbounds for qp in 1:length(fv.qr_weights)
        w = fv.qr_weights[qp]
        fefv_J = zero(Tensor{2,dim_s})
        for j in 1:n_geom_basefuncs
            fefv_J += x[j] ⊗ fv.dMdξ[j, qp, face]
        end
        n = mid_plane_normal(fefv_J) # normal vector of the cohesive element
        e = basevec(n, dim_s)
        fefv_J = fefv_J + n/norm(n) ⊗ e 
        detJ = det(fefv_J)
        detJ > 0.0 || throw(ArgumentError("det(J) is not positive: det(J) = $(detJ)"))
        fv.detJdA[qp, face] = detJ * w
        # local normal vectors
        weight_norm = Ferrite.weighted_normal(fefv_J, fv, face)
        local_normal = Vec{dim_s, T}(i->i<=dim ? weight_norm[i] : zero(T) )
        fv.normals[qp] = local_normal / norm(local_normal)
        # compute rotation matrix
        fv.R[qp] = rotation_matrix(fefv_J)
        # compute dNdx
        Jinv = inv(fefv_J)
        for i in 1:n_func_basefuncs
            fv.dNdx[i, qp, face] = fv.dNdξ[i, qp, face] ⋅ Jinv
        end
    end
end