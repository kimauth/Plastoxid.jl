# only tests 2D diffusion elements, no test for 3D or 1D
# test a diffusion element with linear shape functions against its analytic ke and re
using LinearAlgebra

function quad_2D()
    a = 1.0
    b = 1.0
    d = 0.5 # depth

    dim = 2

    nodes = Node{dim, Float64}[]
    push!(nodes, Node((0.0, 0.0)))
    push!(nodes, Node((a, 0.0)))
    push!(nodes, Node((0.0, b)))
    push!(nodes, Node((a, b)))

    # generate grid with single cell
    cells = Ferrite.AbstractCell[]
    push!(cells, Quadrilateral((1,2,4,3)))

    mesh = Grid(cells, nodes)

    qr = QuadratureRule{dim,RefCube}(2) # fully integrated
    ip = Lagrange{dim,RefCube,1}() # linear shape functions

    # generate problem
    material = Diffusion()
    cross_section = TwoD(d)
    problem = FicksLaw(material, cross_section)

    # generate CellData
    cv = CellScalarValues(qr, ip)
    dofs = collect(1:4)
    coords = getcoordinates(mesh, 1)
    cell_data = Plastoxid.celldata(cv, dofs, coords)

    ndof = Ferrite.getnbasefunctions(cv)
    re = zeros(ndof)
    ke = zeros(ndof, ndof)

    ce = zeros(ndof, 2)
    ce[:, 2] = [10., 15., 20., 0.0]

    Δt = 1.0

    Plastoxid.assemble_cell!(re, ke, problem, ce, cell_data, 1, Dict{Symbol, Any}(), Δt)

    _me = a*b/36*[4 2 1 2; 2 4 2 1 ; 1 2 4 2; 2 1 2 4]
    _ke = [i==j ? 2*b/a + 2*a/b : 0.0 for i=1:4, j=1:4]
    _ke[1, 2:4] = [-2b/a+a/b -b/a-a/b b/a-2a/b]
    _ke[2, 3:4] = [b/a-2a/b -b/a-a/b]
    _ke[3, 4] = -2b/a+a/b
    _ke = 1/6*Symmetric(_ke)

    ke_analytic = (_me + Δt*material.D*_ke)*d

    # compare to analytic solution
    @test ke ≈ ke_analytic
    @test re ≈ ke_analytic*ce[:,2]
end

@testset "diffusion element routine" begin
    quad_2D()
end
