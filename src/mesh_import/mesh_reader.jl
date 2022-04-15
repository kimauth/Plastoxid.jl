# Courtesy to Kristoffer Carlsson
# https://github.com/KristofferC/ViscoCrystalPlast/blob/master/src/mesh_reader.jl

# TODO: replace with FerriteMeshParser.jl

# Utilities that can be useful to write a mesh parser.

# Enable to write out debug_PARSE info
const DEBUG_PARSE = false

# Checks that a line is equal to another, if not
# throws an informational error message
function check_line(line, check)
    if check != ""
        if line != check
            error("expected $check, got $line")
        end
    end
end

# Returns the next line without advancing in the buffer
function peek_line(f, check = "")
    m = mark(f)
    line = strip(readline(f))
    check_line(line, check)
    DEBUG_PARSE && println("Peeked: ", line)
    seek(f, m)
    return line
end

# Returns the next line and advance in the buffer
function eat_line(f, check = "")
    line = strip(readline(f))
    check_line(line, check)
    DEBUG_PARSE && println("Ate: ", line)
    return line
end


# Represents a set of abaqus elements of the same type
struct AbaqusElements
    numbers::Vector{Int}
    topology::Matrix{Int}
end

# Represents the nodes in the mesh
struct AbaqusNodes
    numbers::Vector{Int}
    coordinates::Matrix{Float64}
end

# Represents the mesh
struct AbaqusMesh
    nodes::AbaqusNodes
    elements::Dict{String, AbaqusElements}
    node_sets::Dict{String, Vector{Int}}
    element_sets::Dict{String, Vector{Int}}
end


iskeyword(l) = startswith(l, "*")

function get_string_block(f)
    data = split(readuntil(f, '*', keep=true), "\n")
    if data[end] == "*"
        deleteat!(data, length(data)) # Remove last *
        seek(f, position(f) - 1)
    end

    return data
end

function read_nodes!(f, node_numbers::Vector{Int}, coord_vec::Vector{Float64})
    node_data = get_string_block(f)
    for nodeline in node_data
        node = split(nodeline, ',', keepempty = false) #syntax change
        length(node) == 0 && continue
        n = parse(Int, node[1])
        x = parse(Float64, node[2])
        y = parse(Float64, node[3])
        z = length(node) == 4 ? parse(Float64, node[4]) : 0.0
        push!(node_numbers, n)
        append!(coord_vec, (x, y, z))
    end
end

function read_elements!(f, elements, topology_vectors, element_number_vectors, element_type::AbstractString, element_set="", element_sets=nothing)
    if !haskey(topology_vectors, element_type)
        topology_vectors[element_type] = Int[]
        element_number_vectors[element_type] = Int[]
    end
    topology_vec = topology_vectors[element_type]
    element_numbers = element_number_vectors[element_type]
    element_numbers_new = Int[]
    element_data = get_string_block(f)
    for elementline in element_data
        element = split(elementline, ',', keepempty = false) #syntax change
        length(element) == 0 && continue
        n = parse(Int, element[1])
        push!(element_numbers_new, n)
        vertices = [parse(Int, element[i]) for i in 2:length(element)]
        append!(topology_vec, vertices)
    end
    append!(element_numbers, element_numbers_new)
    if element_set != ""
        element_sets[element_set] = copy(element_numbers_new)
    end
end

function read_set!(f, sets, setname::AbstractString)
    if endswith(setname, "generate")
        lsplit = split(strip(eat_line(f)), ',', keepempty = false) #syntax change
        start, stop, step = [parse(Int, x) for x in lsplit]
        indices = collect(start:step:stop)
        setname = split(setname, [','])[1]
    else
        data = get_string_block(f)
        indices = Int[]
        for line in data
            indices_str = split(line, ',', keepempty = false)
            for v in indices_str
                push!(indices, parse(Int, v))
            end
        end
    end
    sets[setname] = indices
end

function load_abaqus_mesh(fn::AbstractString) 
    open(fn) do f
        node_numbers = Int[]
        coord_vec = Float64[]

        topology_vectors = Dict{String, Vector{Int}}()
        element_number_vectors = Dict{String, Vector{Int}}()

        elements = Dict{String, AbaqusElements}()
        node_sets = Dict{String, Vector{Int}}()
        element_sets = Dict{String, Vector{Int}}()
        while !eof(f)
            header = eat_line(f)
            if header == ""
                continue
            end

            if ((m = match(r"\*Part, name=(.*)", header)) != nothing)

            elseif ((m = match(r"\*Node", header)) != nothing)
                read_nodes!(f, node_numbers, coord_vec)
            elseif ((m = match(r"\*Element", header)) != nothing)
                if ((m = match(r"\*Element, type=(.*), ELSET=(.*)", header)) != nothing)
                    read_elements!(f, elements, topology_vectors, element_number_vectors,  m.captures[1], m.captures[2], element_sets)
                elseif ((m = match(r"\*Element, type=(.*)", header)) != nothing)
                    read_elements!(f, elements, topology_vectors, element_number_vectors,  m.captures[1])
                end
            elseif ((m = match(r"\*Elset, elset=(.*)", header)) != nothing)
                read_set!(f, element_sets, m.captures[1])
            elseif ((m = match(r"\*Nset, nset=(.*)", header)) != nothing)
                read_set!(f, node_sets, m.captures[1])
            elseif ((m = match(r"\*End Part", header)) != nothing)
                l = eat_line(f)
            elseif iskeyword(header)
                if eof(f)
                    break
                end
                while !iskeyword(peek_line(f))
                    eat_line(f)
                end
            else
                if eof(f)
                    break
                else
                    error("Unknown header: $header")
                end
            end
        end

        for element_type in keys(topology_vectors)
            topology_vec = topology_vectors[element_type]
            element_numbers = element_number_vectors[element_type]
            n_elements = length(element_numbers)
            elements[element_type] = AbaqusElements(element_numbers,
            reshape(topology_vec, length(topology_vec) ÷ n_elements, n_elements))
        end
        abaqus_nodes = AbaqusNodes(node_numbers, reshape(coord_vec, 3, length(coord_vec) ÷ 3))
        return AbaqusMesh(abaqus_nodes, elements, node_sets, element_sets)
    end
end
