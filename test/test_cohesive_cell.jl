@testset "CohesiveCell" begin
    coords = [Vec((2.0, 0.0)), Vec((8.0, 4.0)), Vec((0.0, 3.0)),  Vec((6.0, 7.0))]
    nodes = Node.(coords)

    cells = [CohesiveQuadrilateral((1,2,3,4))]

    grid = Grid(cells, nodes)

    dh = DofHandler(grid)
    push!(dh, :u, 2)
    close!(dh)
    @test ndofs(dh) == 8

    # ip_base = Lagrange{1,RefCube,1}()
    # ip_f = JumpInterpolation(ip_base)
    # ip_geo = MidPlaneInterpolation(ip_base)
    # qr = QuadratureRule{1,RefCube}(:lobatto, 2)
    # cv = SurfaceVectorValues(qr, ip_f, ip_geo)

    cell = CellIterator(dh)
    reinit!(cell, 1)

    xe = getcoordinates(cell)
    @test xe == coords

    celldofs(cell) = collect(1:8)

    # add bc to node
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, Set((3,)), (x,t)->1.0, 1))
    close!(ch)
    @test ch.prescribed_dofs == [5,]

    # add bc to face
    # ch = ConstraintHandler(dh)
    # add!(ch, Dirichlet(:u, Set((FaceIndex((1,1)),)), (x,t)->1.0, 1))
    # close!(ch)
end