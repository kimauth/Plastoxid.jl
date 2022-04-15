using LinearAlgebra

# only tests 2D continuum elements, no test for 3D

function quad_2D()

    h = 0.1 # thickness
    a = 1.0 # height
    b = 2.0 # width

    dim = 2

    # generate grid with single cell
    mesh = generate_grid(Quadrilateral, (1,1), Vec((0.0, 0.0)), Vec(2.0, 1.0))

    qr = QuadratureRule{dim,RefCube}(2) # fully integrated
    ip = Lagrange{dim,RefCube,1}() # linear shape functions

    # generate problem
    material = LinearElastic(E=200e3, ν=0.3)
    material_states = initial_material_states(material, Set{Int}(1:getncells(mesh)), qr)
    material_states_temp = initial_material_states(material, Set{Int}(1:getncells(mesh)), qr)
    cross_section = PlaneStress(h)
    problem = MechanicalEquilibrium(material, material_states, material_states_temp, cross_section)

    # generate CellData
    cv = CellVectorValues(qr, ip)
    dofs = collect(1:8)
    coords = getcoordinates(mesh, 1)
    cell_data = Plastoxid.celldata(cv, dofs, coords)

    ndof = Ferrite.getnbasefunctions(cv)
    re = zeros(ndof)
    ke = zeros(ndof, ndof)

    ue = zeros(ndof, 2)
    ue[[6,8], 2] = ones(2)*0.1

    Plastoxid.assemble_cell!(re, ke, problem, ue, cell_data, 1)

    # generate analytical solution
    ke_analytic = zeros(ndof, ndof)
    ν = material.ν
    E = material.E
    for i in 1:ndof
        ke_analytic[i,i] = iseven(i) ? 4b^2+2a^2*(1-ν) : 4a^2+2b^2*(1-ν)
    end
    ke_analytic[1,2:end] = [3a*b*(1+ν)/2 -4a^2+b^2*(1-ν) -3a*b*(1-3ν)/2 -2a^2-b^2*(1-ν) -3a*b*(1+ν)/2 2a^2-2b^2*(1-ν) 3a*b*(1-3ν)/2]
    ke_analytic[2,3:end] = [3a*b*(1-3ν)/2 2b^2-2a^2*(1-ν) -3a*b*(1+ν)/2 -2b^2-a^2*(1-ν) -3a*b*(1-3ν)/2 -4b^2+a^2*(1-ν)]
    ke_analytic[3,4:end] = [-3a*b*(1+ν)/2 2a^2-2b^2*(1-ν) -3a*b*(1-3ν)/2 -2a^2-b^2*(1-ν) 3a*b*(1+ν)/2]
    ke_analytic[4,5:end] = [3a*b*(1-3ν)/2 -4b^2+a^2*(1-ν) 3a*b*(1+ν)/2 -2b^2-a^2*(1-ν)]
    ke_analytic[5,6:end] = [3a*b*(1+ν)/2 -4a^2+b^2*(1-ν) -3a*b*(1-3ν)/2]
    ke_analytic[6,7:end] = [3a*b*(1-3ν)/2 2b^2-2a^2*(1-ν)]
    ke_analytic[7,8:end] = [-3a*b*(1+ν)/2]
    ke_analytic = E*h/(12a*b*(1-ν^2))*LinearAlgebra.Symmetric(ke_analytic)

    # compare to analytic solution
    @test ke ≈ ke_analytic
    @test re ≈ ke_analytic*ue[:,2]
end

@testset "continuum element routine" begin
    quad_2D()
end
