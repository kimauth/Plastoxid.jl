# limited to CellScalarValues and RefCube because it hasn't been thought through for others
function _reinit!(
    cv::CellScalarValues{dim_s,T,RefCube},
    x::Vector{Vec{dim_s, T}},#Vector{Tensor{1,dim_s,T,dim_s}},
    normals::Vector{Vec{dim_s, T}},
    thickness::Float64
) where {dim_s,T}

    n_geom_basefuncs = Ferrite.getngeobasefunctions(cv) # how many in M
    n_func_basefuncs = Ferrite.getnbasefunctions(cv) # how many in N

    @inbounds for qp in 1:length(cv.qr_weights)
        n̂ = normals[qp]
        w = cv.qr_weights[qp]
        _fecv_J = zeros(dim_s, dim_s)
        for j in 1:n_geom_basefuncs
            for d in 1:(dim_s-1)
                _fecv_J[:,d] += cv.dMdξ[j,qp][d] * x[j]
            end
        end
        _fecv_J[:,end] = thickness/2 * n̂
        fecv_J = Tensor{dim_s,2}(_fecv_J)
        detJ = det(fecv_J)
        detJ > 0.0 || throw(ArgumentError("det(J) is not positive: det(J) = $(detJ)"))
        cv.detJdV[qp] = detJ * w
        Jinv = inv(fecv_J)
        for j in 1:n_func_basefuncs
            cv.dNdx[j, qp] = cv.dNdξ[j, qp] ⋅ Jinv
        end
    end
    return nothing
end

function Ferrite.reinit!(
    cv::CellScalarValues{dim_s,T,RefCube},
    scv::SurfaceVectorValues{dim_p,dim_s},
    x::Vector{Vec{dim_s, T}},
    thickness::Float64,
) where {dim_s,T,dim_p}

    Ferrite.getngeobasefunctions(cv) == length(x) || error("You are trying to reinitialize cellvalues with an unsuitable number of coordinate points.")

    nqp = getnquadpoints(cv)
    if nqp == getnquadpoints(scv)
        # extract normals for all qp, curved geometry allowed here
        normals = [Plastoxid.get_normal(scv, i) for i=1:nqp]
    else
        # assume streigt geometry (normals in all qp are the same, thus we use the normal from the 1st qp)
        normals = [Plastoxid.get_normal(scv, 1) for i=1:nqp]
    end
    _reinit!(cv, x, normals, thickness)
end
