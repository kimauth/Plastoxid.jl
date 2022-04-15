"""
    shape_midplane_value(fe_v::Values, q_point::Int, base_function::Int)

Return the value of shape function `base_function` evaluated in
quadrature point `q_point`, where the shape function computes the average between the upper
and lower face of the cohesive element. Can only be used with `SurfaceValues`.
"""
Base.@propagate_inbounds shape_midplane_value(cv::SurfaceValues, q_point::Int, base_func::Int) = cv.N_mp[base_func, q_point]
Base.@propagate_inbounds shape_midplane_value(fv::SurfaceFaceValues, q_point::Int, base_func::Int) = fv.N_mp[base_func, q_point, fv.current_face[]]
"""
    shape_midplane_gradient(fe_v::Values, q_point::Int, base_function::Int)

Return the gradient of shape function `base_function` evaluated in
quadrature point `q_point`, where the shape function computes the average between the upper
and lower face of the cohesive element. Can only be used with `SurfaceValues`.
"""
Base.@propagate_inbounds shape_midplane_gradient(cv::SurfaceValues, q_point::Int, base_func::Int) = cv.dN_mpdx[base_func, q_point]
Base.@propagate_inbounds shape_midplane_gradient(fv::SurfaceFaceValues, q_point::Int, base_func::Int) = fv.dN_mpdx[base_func, q_point, fv.current_face[]]

function_midplane_value(fe_v::T, q_point, u, dof_range) where T<:Ferrite.Values = function_midplane_value(Ferrite.FieldTrait(T), fe_v, q_point, u, dof_range)
function_midplane_value(fe_v::T, q_point, u) where T<:Ferrite.Values = function_midplane_value(Ferrite.FieldTrait(T), fe_v, q_point, u)

"""
    function_midplane_value(fe_v::Values, q_point::Int, u::AbstractVector)

Compute the mid-plane value of the field given by `u` in a quadrature point `q_point`. `u` is a vector with values
for the degrees of freedom. For a scalar valued function, `u` contains scalars.
For a vector valued function, `u` can be a vector of scalars (for use of `VectorValues`)
or `u` can be a vector of `Vec`s (for use with ScalarValues).
"""
function function_midplane_value(::Ferrite.FieldTrait, fe_v::Ferrite.Values, q_point::Int, u::AbstractVector{T}, dof_range = eachindex(u)) where T
    n_base_funcs = getnbasefunctions(fe_v)
    @assert length(dof_range) == n_base_funcs
    @boundscheck checkbounds(u, dof_range)
    val = zero(Ferrite._valuetype(fe_v, u))
    @inbounds for (i, j) in enumerate(dof_range)
        val += shape_midplane_value(fe_v, q_point, i) * u[j]
    end
    return val
end

function function_midplane_value(::Ferrite.VectorValued, fe_v::Ferrite.Values{dim_s}, q_point::Int, u::AbstractVector{Vec{dim_s,T}}) where {dim_s,T}
    n_base_funcs = getn_scalarbasefunctions(fe_v)
    @assert length(u) == n_base_funcs
    val = zero(Vec{dim_s, T})
    basefunc = 1
    @inbounds for i in 1:n_base_funcs
        for j in 1:dim_s
            val += shape_midplane_value(fe_v, q_point, basefunc) * u[i][j]
            basefunc += 1
        end
    end
    return val
end

"""
    function_midplane_gradient(fe_v::Values{dim}, q_point::Int, u::AbstractVector)

Compute the mid-plane gradient of the field given by `u` in a quadrature point `q_point`. `u` is a vector with values
for the degrees of freedom. For a scalar valued function, `u` contains scalars.
For a vector valued function, `u` can be a vector of scalars (for use of `VectorValues`)
or `u` can be a vector of `Vec`s (for use with ScalarValues).
"""
function_midplane_gradient(fe_v::T, q_point, u) where T = function_midplane_gradient(Ferrite.FieldTrait(T), fe_v, q_point, u)
function_midplane_gradient(fe_v::T, q_point, u, dof_range) where T = function_midplane_gradient(Ferrite.FieldTrait(T), fe_v, q_point, u, dof_range)

function function_midplane_gradient(::Ferrite.FieldTrait, fe_v::Ferrite.Values{dim_s}, q_point::Int, u::AbstractVector{T}, dof_range = eachindex(u)) where {dim_s,T}
    n_base_funcs = getnbasefunctions(fe_v)
    @assert length(dof_range) == n_base_funcs
    @boundscheck checkbounds(u, dof_range)
    @boundscheck Ferrite.checkquadpoint(fe_v, q_point)
    grad = zero(Ferrite._gradienttype(fe_v, u))
    @inbounds for (i, j) in enumerate(dof_range)
        grad += shape_midplane_gradient(fe_v, q_point, i) * u[j]
    end
    return grad
end

function function_midplane_gradient(::Ferrite.ScalarValued, fe_v::Ferrite.Values{dim_s}, q_point::Int, u::AbstractVector{Vec{dim_s,T}}) where {dim_s,T}
    n_base_funcs = getn_scalarbasefunctions(fe_v)
    @assert length(u) == n_base_funcs
    @boundscheck Ferrite.checkquadpoint(fe_v, q_point)
    grad = zero(Tensor{2,dim_s,T})
    @inbounds for i in 1:n_base_funcs
        grad += u[i] ⊗ shape_midplane_gradient(fe_v, q_point, i)
    end
    return grad
end

function function_midplane_gradient(::Ferrite.VectorValued, fe_v::Ferrite.Values{dim_s}, q_point::Int, u::AbstractVector{Vec{dim_s,T}}) where {dim_s,T}
    n_base_funcs = getn_scalarbasefunctions(fe_v)
    @assert length(u) == n_base_funcs
    @boundscheck Ferrite.checkquadpoint(fe_v, q_point)
    grad = zero(Tensor{2,dim_s,T})
    basefunc_count = 1
    @inbounds for i in 1:n_base_funcs
        for j in 1:dim_s
            grad += u[i][j] * shape_midplane_gradient(fe_v, q_point, basefunc_count)
            basefunc_count += 1
        end
    end
    return grad
end

"""
    function_inplane_value(fe_v::Values, q_point::Int, u::AbstractVector)

Compute the value of the field `u` in a quadrature point `q_point`. `u` is a vector with values
for the degrees of freedom. For a scalar valued function, `u` contains scalars.
For a vector valued function, `u` can be a vector of scalars (for use of `VectorValues`)
or `u` can be a vector of `Vec`s (for use with ScalarValues).

Opposed to `function_value` and `function_midplane_value`, `function_inplane_value` takes in nodal values on the mid-plane
(and not on the actual nodes), i.e. for a 2D four-noded element, values at 2 positions must be handed in.
Values are interpolated to the gauss point positions within the mid-plane of the element.


!!! warning
    In most cases this is not what you want to use. It is meant for interpolating local variables
    within an element.
"""

function_inplane_value(fe_v::T, q_point, u, dof_range) where T = function_inplane_value(Ferrite.FieldTrait(T), fe_v, q_point, u, dof_range)
function_inplane_value(fe_v::T, q_point, u) where T = function_inplane_value(Ferrite.FieldTrait(T), fe_v, q_point, u)

function function_inplane_value(::Ferrite.FieldTrait, fe_v::SurfaceValues{dim,dim_s}, q_point::Int, u::AbstractVector{T}, dof_range = eachindex(u)) where {dim,dim_s,T}
    n_base_funcs = getnbasefunctions(fe_v) ÷ 2
    @assert length(dof_range) == n_base_funcs
    @boundscheck checkbounds(u, dof_range)
    val = zero(Ferrite._valuetype(fe_v, u))
    @inbounds for (i, j) in enumerate(dof_range)
        val += shape_value(fe_v, q_point, i+n_base_funcs) * u[j]
    end
    return val
end

function function_inplane_value(::Ferrite.VectorValued, fe_v::SurfaceValues{dim,dim_s}, q_point::Int, u::AbstractVector{Vec{dim_s,T}}) where {dim,dim_s,T}
    n_base_funcs = Ferrite.getn_scalarbasefunctions(fe_v) ÷ 2
    @assert length(u) == n_base_funcs
    val = zero(Vec{dim_s, T})
    basefunc = 2n_base_funcs + 1 # loop over 2nd half of fe_v.N
    @inbounds for i in 1:n_base_funcs
        for j in 1:dim_s
            val += shape_value(fe_v, q_point, basefunc) * u[i][j]
            basefunc += 1
        end
    end
    return val
end


function_inplane_gradient(fe_v::T, q_point, u) where T = function_inplane_gradient(Ferrite.FieldTrait(T), fe_v, q_point, u)
function_inplane_gradient(fe_v::T, q_point, u, dof_range) where T = function_inplane_gradient(Ferrite.FieldTrait(T), fe_v, q_point, u, dof_range)

function function_inplane_gradient(::Ferrite.FieldTrait, fe_v::SurfaceValues{dim, dim_s}, q_point::Int, u::AbstractVector{T}, dof_range = eachindex(u)) where {dim,dim_s,T}
    n_base_funcs = getnbasefunctions(fe_v) ÷ 2
    @assert length(dof_range) == n_base_funcs
    @boundscheck checkbounds(u, dof_range)
    grad = zero(Ferrite._gradienttype(fe_v, u))
    @inbounds for (i, j) in enumerate(dof_range)
        grad += shape_gradient(fe_v, q_point, i+n_base_funcs) * u[j]
    end
    return grad
end

function function_inplane_gradient(::Ferrite.ScalarValued, fe_v::SurfaceValues{dim,dim_s}, q_point::Int, u::AbstractVector{Vec{dim_s,T}}) where {dim,dim_s,T}
    n_base_funcs = getnbasefunctions(fe_v) ÷ 2
    @assert length(u) == n_base_funcs
    grad = zero(Tensor{2,dim_s,T})
    @inbounds for i in 1:n_base_funcs 
        grad += u[i] ⊗ shape_gradient(fe_v, q_point, i+n_base_funcs)
    end
    return grad
end

function function_inplane_gradient(::Ferrite.VectorValued, fe_v::SurfaceValues{dim,dim_s}, q_point::Int, u::AbstractVector{Vec{dim_s,T}}) where {dim,dim_s,T}
    n_base_funcs = Ferrite.getn_scalarbasefunctions(fe_v)  ÷ 2
    @assert length(u) == n_base_funcs
    grad = zero(Tensor{2,dim_s,T})
    basefunc_count = 1 + 2n_base_funcs
    @inbounds for i in 1:n_base_funcs
        for j in 1:dim_s
            grad += u[i][j] * shape_gradient(fe_v, q_point, basefunc_count)
            basefunc_count += 1
        end
    end
    return grad
end