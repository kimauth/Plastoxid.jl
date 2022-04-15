
# Assume that there usually aren't many Neumann BCs to DofHandler
# therefore store them as a Tuple instead of creating a NaturalConstraintHandler
# In a NaturalConstraintHandler, nbcs could be grouped according to the fields and time would only need to be stored onces
# With many nbcs the Tuple approach probably isn't very efficient anymore
struct Neumann{F, T}
    f::F # (x, t) -> v
    faceset::Set{FaceIndex}
    field::Symbol
    t::ScalarWrapper{T}
end

Neumann(f::F, faceset::Set{FaceIndex}, field::Symbol, t::T) where {F,T} = Neumann{F,T}(f, faceset, field, ScalarWrapper(t))

get_time(nbc::Neumann) = nbc.t[]

function Ferrite.update!(nbc::Neumann{F, T}, t::T) where {F,T}
    nbc.t[] = t
    return nbc
end
