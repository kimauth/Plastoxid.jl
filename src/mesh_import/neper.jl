
function read_orientations(fn::AbstractString)
        ngrains = Int64(0)
        rodrigues_data = ""
        open(fn) do f
            while !eof(f)
                header = readline(f)
                if header == ""
                    continue
                end

                if ((m = match(r"\**cell", header)) != nothing)
                    l = readline(f)
                    ngrains = parse(Int64, l)
                elseif ((m = match(r"\*ori", header)) != nothing)
                    l = readline(f)
                    ((m = match(r"rodrigues:active", l)) != nothing) || error("Only Rodrigues angles are supported.")
                    rodrigues_data = readuntil(f, "*")
                end
            end
        end
        # parse rodrigues angles
        temp = split.(split(rodrigues_data, "\n"))
        rodrigues_floats = [parse(Float64, temp[i][j]) for i=1:ngrains, j=1:3]
        rodrigues_angles = [Rotations.RodriguesParam(rodrigues_floats[i, :]...) for i=1:ngrains]

        return rodrigues_angles
end
