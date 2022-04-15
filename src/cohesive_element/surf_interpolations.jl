abstract type SurfaceInterpolation{dim,shape,order,dim_s,ip<:Interpolation{dim,shape,order}} <: Interpolation{dim,shape,order} end

Ferrite.getnbasefunctions(ip::SurfaceInterpolation) = Ferrite.getnbasefunctions(ip.ip_base)*2
Ferrite.nvertexdofs(ip::SurfaceInterpolation) = Ferrite.nvertexdofs(ip.ip_base)
Ferrite.nedgedofs(ip::SurfaceInterpolation) = Ferrite.nfacedofs(ip.ip_base)
Ferrite.nfacedofs(ip::SurfaceInterpolation) = Ferrite.ncelldofs(ip.ip_base) # TODO: check this!
Ferrite.ncelldofs(ip::SurfaceInterpolation) = 0 # cohesive elements never have dofs inside the cell
Ferrite.reference_coordinates(ip::SurfaceInterpolation) = Ferrite.reference_coordinates(ip.ip_base)
Ferrite.faces(ip::SurfaceInterpolation) = Ferrite.faces(ip.ip_base) # check this

# TODO: dim inside the InterpolationInfo becomes a problem for higher order cohesive elements
# Ferrite.InterpolationInfo(ip::SurfaceInterpolation{dim,shape,order,ip_base,dim_s}) where {dim,shape,order,ip_base,dim_s} = Ferrite.InterpolationInfo(Ferrite.nvertexdofs(ip), Ferrite.nedgedofs(ip), Ferrite.nfacedofs(ip), Ferrite.ncelldofs(ip), dim_s)

struct MidPlaneInterpolation{dim,shape,order,dim_s,ip} <: SurfaceInterpolation{dim,shape,order,dim_s,ip}
    ip_base::ip
end

MidPlaneInterpolation(ip::Interpolation{dim,shape,order}) where {dim,shape,order} = MidPlaneInterpolation{dim,shape,order,dim+1,typeof(ip)}(ip)


function Ferrite.value(ip::MidPlaneInterpolation{dim,shape,order,dim_s,base_ip}, i::Int, ξ::Vec{dim}) where {dim,shape,order,dim_s,base_ip}
    n = getnbasefunctions(ip.ip_base)
    if i <= n
        return 0.5*Ferrite.value(ip.ip_base, i, ξ)
    elseif i <= 2n
        return 0.5*Ferrite.value(ip.ip_base, i-n, ξ)
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

struct JumpInterpolation{dim,shape,order,dim_s,ip} <: SurfaceInterpolation{dim,shape,order,dim_s,ip}
    ip_base::ip
end

JumpInterpolation(ip::Interpolation{dim,shape,order}) where {dim,shape,order} = JumpInterpolation{dim,shape,order,dim+1,typeof(ip)}(ip)

function Ferrite.value(ip::JumpInterpolation{dim,shape,order,dim_s,base_ip}, i::Int, ξ::Vec{dim}) where {dim,shape,order,dim_s,base_ip}
    n = getnbasefunctions(ip.ip_base)
    if i <= n
        return -Ferrite.value(ip.ip_base, i, ξ)
    elseif i <= 2n
        return Ferrite.value(ip.ip_base, i-n, ξ)
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end
