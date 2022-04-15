function area_average(qp_vals::Vector{T}, cv::Ferrite.Values) where T
    # cv must be reinitized already
    area = 0.0
    result = zero(first(qp_vals))
    for qp in 1:getnquadpoints(cv)
        A = getdetJdV(cv, qp)
        area += A
        result += qp_vals[qp] * A
    end
    return result / area
end

function area_averages!(results::AbstractVector{T}, qp_vals::AbstractMatrix{T}, feset, dh) where T
    cb = feset.cellbuffers[Threads.threadid()]
    cv = Plastoxid.get_fevalues(cb)
    for color in feset.cellset, (;globalid, localid) in color
        Ferrite.cellcoords!(cb.xe, dh, globalid)
        reinit!(cv, cb.xe)
        results[globalid] = area_average(qp_vals[:,localid], cv)
    end
    return results
end

function area_averages(qp_vals::AbstractMatrix{T}, feset::Plastoxid.FESet, dh::Ferrite.AbstractDofHandler) where T
    results = [zero(T)*NaN for _ in 1:getncells(dh.grid)]
    area_averages!(results, qp_vals, feset, dh)
end