# only tests 2D cohesive elements, no test for 3D
# test a cohesive element with linear cohesive law against its analytic ke and re
using LinearAlgebra

function cohesive_2D()
    h = 0.1 # depth
    L = 2.0 # length

    dim = 2

    nodes = Node{dim, Float64}[]
    push!(nodes, Node((0.0, 0.0)))
    push!(nodes, Node((L, 0.0)))
    push!(nodes, Node((0.0, 0.0)))
    push!(nodes, Node((L, 0.0)))

    # generate grid with single cell
    cells = Ferrite.AbstractCell[]
    push!(cells, CohesiveCell{2,4,4}((1,2,4,3)))

    mesh = Grid(cells, nodes)

    qr = QuadratureRule{dim-1,RefCube}(2) # fully integrated
    ip = CohesiveZone{dim-1,RefCube,1,dim}() # linear shape functions

    # generate problem
    material = Plastoxid.LinearCohesive(2.0e6)
    ms = initial_material_states(material, Set(1:getncells(mesh)), qr)
    ms_temp = initial_material_states(material, Set(1:getncells(mesh)), qr)
    cross_section = TwoD(h)
    problem = Interface(material, ms, ms_temp, cross_section)

    # generate CellData
    cv = SurfaceVectorValues(qr, ip)
    dofs = collect(1:8)
    coords = getcoordinates(mesh, 1)
    cell_data = Plastoxid.celldata(cv, dofs, coords)

    ndof = Ferrite.getnbasefunctions(cv)
    re = zeros(ndof)
    ke = zeros(ndof, ndof)

    ue = zeros(ndof, 2)
    ue[:, 2] = [0., 0., 0.5, 0.2, 0.1, 0.4, -0.2, 0.1]*0.0001

    Plastoxid.assemble_cell!(re, ke, problem, ue, cell_data, 1)

    ke_analytic = zeros(8,8)
    ke_analytic[1:4,1:4] = [i == j ? h*1/4*L*(1+1/3)*material.K⁰ : 0.0 for i=1:4, j=1:4]
    ke_analytic[1:2, 3:4] += [i==j ? h*1/4*L*(1-1/3)*material.K⁰ : 0.0 for i=1:2, j=1:2]
    ke_analytic[5:8, 5:8] = ke_analytic[1:4, 1:4]
    ke_analytic[1:4, 5:8] = [i == j ? -h*1/4*L*(1-1/3)*material.K⁰ : 0.0 for i=1:4, j=1:4]
    ke_analytic[1:2, 7:8] += [i==j ? -h*1/4*L*(1+1/3)*material.K⁰ : 0.0 for i=1:2, j=1:2]
    ke_analytic[3:4, 5:6] += ke_analytic[1:2, 7:8]
    ke_analytic = Symmetric(ke_analytic)

    # compare to analytic solution
    @test ke ≈ ke_analytic
    @test re ≈ ke_analytic*ue[:,2]
end

function test_ke()
    L = 2.0 # length
    α = 30. # deg
    dim = 2
    h = 1.0 # depth

    nodes = Node{dim, Float64}[]
    push!(nodes, Node((0.0, 0.0)))
    push!(nodes, Node((L*cosd(α), L*sind(α))))
    push!(nodes, Node((L*cosd(α), L*sind(α))))
    push!(nodes, Node((0.0, 0.0)))

    # generate grid with single cell
    cells = Ferrite.AbstractCell[]
    push!(cells, CohesiveCell{2,4,4}((1,2,3,4)))

    mesh = Grid(cells, nodes)

    qr = QuadratureRule{dim-1,RefCube}(2) # gauss integration
    ip = CohesiveZone{dim-1,RefCube,1,dim}() # linear shape functions

    # generate problem
    material = Plastoxid.XuNeedleman()
    ms = initial_material_states(material, Set(1:getncells(mesh)), qr)
    ms_temp = initial_material_states(material, Set(1:getncells(mesh)), qr)
    cross_section = TwoD(h)
    problem = Interface(material, ms, ms_temp, cross_section)

    # generate CellData
    cv = SurfaceVectorValues(qr, ip)
    dofs = collect(1:8)
    coords = getcoordinates(mesh, 1)
    cell_data = Plastoxid.celldata(cv, dofs, coords)

    ndof = Ferrite.getnbasefunctions(cv)
    re = zeros(ndof)
    ke = zeros(ndof, ndof)

    ## test convergence behavior

    # We aim for a displacement state that is well before the critical separations,
    # thus force controlled iterations should work well.
    ue = zeros(8,2)
    dofs_p = [1,2,3,4] # prescribed dofs
    dofs_f = [5,6,7,8] # free dofs
    re_p = [-20., 90., -10, 45.] # prescribed forces
    guess_f = zeros(4)
    ue[dofs_f, 2] = guess_f

    η = Float64[]
    for iter = 1:10
        ue[dofs_f, 2] = guess_f
        fill!(re, 0.0); fill!(ke, 0.0)
        Plastoxid.assemble_cell!(re, ke, problem, ue, cell_data, 1)
        push!(η, norm(re[dofs_f]-re_p))
        print("iter: ", iter, "; residual:", η[iter])
        iter > 2 ? println("; rate: ", (log(η[iter-1])-log(η[iter]))/(log(η[iter-2])-log(η[iter-1]))) : println()
        push!(residuals, norm(re[dofs_f]-re_p))
        if norm(re[dofs_f]-re_p) < 1e-6
            break
        end
        guess_f += -ke[dofs_f, dofs_f]\(re[dofs_f]-re_p)
    end
    # test that last two convergence rates are sufficiently large
    @test (log(η[end-1])-log(η[end]))/(log(η[end-2])-log(η[end-1])) > 1.5
    @test (log(η[end-2])-log(η[end-1]))/(log(η[end-3])-log(η[end-2])) > 1.5
end

function test_rotation()
    L = 2.0 # length
    α = 30. # deg
    dim = 2
    h = 1.0 # depth

    ## generate an element that is rotated by 30deg
    nodes1 = Node{dim, Float64}[]
    push!(nodes1, Node((0.0, 0.0)))
    push!(nodes1, Node((L*cosd(α), L*sind(α))))
    push!(nodes1, Node((L*cosd(α), L*sind(α))))
    push!(nodes1, Node((0.0, 0.0)))

    # generate grid with single cell
    cells = Ferrite.AbstractCell[]
    push!(cells, CohesiveCell{2,4,4}((1,2,3,4)))

    mesh1 = Grid(cells, nodes1)

    qr = QuadratureRule{dim-1,RefCube}(2) # gauss integration
    ip = CohesiveZone{dim-1,RefCube,1,dim}() # linear shape functions

    # generate CellData
    cv1 = SurfaceVectorValues(qr, ip)
    dofs = collect(1:8)
    coords1 = getcoordinates(mesh1, 1)
    cell_data1 = Plastoxid.celldata(cv1, dofs, coords1)

    ## generate an element that is parallel to the x-axis
    nodes2 = Node{dim, Float64}[]
    push!(nodes2, Node((0.0, 0.0)))
    push!(nodes2, Node((L, 0.0)))
    push!(nodes2, Node((L, 0.0)))
    push!(nodes2, Node((0.0, 0.0)))

    # generate grid with single cell
    cells = Ferrite.AbstractCell[]
    push!(cells, CohesiveCell{2,4,4}((1,2,3,4)))

    mesh2 = Grid(cells, nodes2)

    qr = QuadratureRule{dim-1,RefCube}(2) # gauss integration
    ip = CohesiveZone{dim-1,RefCube,1,dim}() # linear shape functions

    # generate CellData
    cv2 = SurfaceVectorValues(qr, ip)
    dofs = collect(1:8)
    coords2 = getcoordinates(mesh2, 1)
    cell_data2 = Plastoxid.celldata(cv2, dofs, coords2)

    # generate problem
    material = Plastoxid.XuNeedleman()
    ms = initial_material_states(material, Set(1:getncells(mesh)), qr)
    ms_temp = initial_material_states(material, Set(1:getncells(mesh)), qr)
    cross_section = TwoD(h)
    problem = Interface(material, ms, ms_temp, cross_section)

    ndof = Ferrite.getnbasefunctions(cv1)

    re1 = zeros(ndof)
    ke1 = zeros(ndof, ndof)

    ue1 = zeros(ndof, 2)
    ue1[5:6, 2] = [-sind(α); cosd(α)]*material.δₙ

    Plastoxid.assemble_cell!(re1, ke1, problem, ue1, cell_data1, 1)

    re2 = zeros(ndof)
    ke2 = zeros(ndof, ndof)

    ue2 = zeros(ndof, 2)
    ue2[6, 2] = material.δₙ

    Plastoxid.assemble_cell!(re2, ke2, problem, ue2, cell_data2, 1)

    # test that the result vector is the same, just rotated
    R1 = cell_data1.cv.R[1]
    @test Vec((re1[1], re1[2])) ⋅ R1 ≈ Vec((re2[1], re2[2]))
end


@testset "cohesive element routine" begin
    cohesive_2D()
    test_ke()
    test_rotation()
end
