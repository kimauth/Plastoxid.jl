#display mixed grid without celltypes
function Base.show(io::IO, ::MIME"text/plain", grid::Grid{dim, Ferrite.AbstractCell, T} where {dim, T<:Real})
    print(io, "$(typeof(grid)) with $(getncells(grid)) cells and $(getnnodes(grid)) nodes")
end

#other Ferrite add-on
getcelldofs(ci::CellIterator) = ci.celldofs
Ferrite.ndofs(cell::CellIterator) = ndofs_per_cell(cell.dh, cellid(cell))
Ferrite.getdim(dh::Union{DofHandler{dim,C,T}, MixedDofHandler{dim,C,T}}) where {dim,C,T} = dim
Ferrite.getdim(qr::QuadratureRule{dim,shape,T}) where {dim,shape,T} = dim

function Ferrite.getfaceset(grid::Ferrite.AbstractGrid, set::String, fh::Ferrite.FieldHandler)
    initial_faceset = getfaceset(grid, set)

    faceset = Set{NTuple{2, Int}}()
    for (cell, face) in initial_faceset
        cell ∈ fh.cellset && push!(faceset, (cell, face))
    end
    return faceset
end

Ferrite.getncells(fh::FieldHandler) = length(fh.cellset)

# naming analogous to celldofs
function fielddofs(dh::MixedDofHandler, field::Symbol)
    fielddofs = Vector{Int}[]
    for fh in dh.fieldhandlers
        # jump to next field handler if field is not in this field handler
        findfirst(i->i == field, Ferrite.getfieldnames(fh)) == nothing && continue

        _celldofs = fill(0, ndofs_per_cell(dh, first(fh.cellset)))
        field_dof_range = dof_range(fh, field)
        nfielddofs = length(field_dof_range)
        fh_fielddofs = fill(0, nfielddofs*length(fh.cellset))
        # loop over cells in fieldhandler
        for (i, c) in enumerate(fh.cellset)
            celldofs!(_celldofs, dh, c)
            fh_fielddofs[(i-1)*nfielddofs+1:i*nfielddofs] = _celldofs[field_dof_range]
        end
        push!(fielddofs, fh_fielddofs)
    end
    unique!(sort!(vcat(fielddofs...)))
end

"""
    faceset(grid::Ferrite.AbstractGrid, nodeset::Set{Int})

Find all faces in grid whose nodes are part of `nodeset` and return them as a `Set{FaceIndex}`.

Only if all nodes in a face are in the nodeset, the face will be part of the faceset.
"""
function faceset(grid::Ferrite.AbstractGrid, nodeset::Set{Int})
    faceset = Set{FaceIndex}()
    for (cellid, cell) in enumerate(grid.cells)
        if any(map(n-> n ∈ nodeset, cell.nodes))
            # check actual faces
            for (faceid, face) in enumerate(Ferrite.faces(cell))
                if all(map(n -> n ∈ nodeset, face))
                    push!(faceset, FaceIndex(cellid, faceid))
                end
            end
        end
    end
    return faceset
end

# type piracy
Ferrite.getsparsemat(a::Vector{A}) where A<:Ferrite.AbstractSparseAssembler = Ferrite.getsparsemat(first(a))
getvector(a::Ferrite.AbstractSparseAssembler) = a.f
getvector(a::Vector{A}) where A<:Ferrite.AbstractSparseAssembler = getvector(first(a))