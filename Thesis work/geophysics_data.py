import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from discretize import TensorMesh
from discretize.utils import mkvc

from SimPEG.utils import plot2Ddata, model_builder, surface2ind_topo
import discretize
from pymatsolver import SolverLU
from SimPEG import maps
from SimPEG.potential_fields import gravity

from SimPEG import maps, data, data_misfit, inverse_problem, regularization, optimization, directives, inversion, utils


def geophysics_data_final(points_of_interest):
    # Defining topography 
    [x_topo, y_topo] = np.meshgrid(np.linspace(0, 100, 41), np.linspace(0, 100, 41))
    z_topo = np.array([0]*1681)
    x_topo, y_topo = mkvc(x_topo), mkvc(y_topo)
    topo_xyz = np.c_[x_topo, y_topo, z_topo]

    # Defining the Survey
    # Define the observation/reciever locations
    x = np.linspace(0.0, 100.0, 10)
    y = np.linspace(0.0, 100.0, 10)
    x, y = np.meshgrid(x, y)
    z = np.array([0]*(100))
    x, y = mkvc(x.T), mkvc(y.T)

    #fun_interp = LinearNDInterpolator(np.c_[x_topo, y_topo], z_topo)
    #z = fun_interp(np.c_[x, y])
    receiver_locations = np.c_[x, y, z]



    # Define the component(s) of the field 
    components = ["gz"]
    receiver_list = gravity.receivers.Point(receiver_locations, components=components)
    receiver_list = [receiver_list]
    source_field = gravity.sources.SourceField(receiver_list=receiver_list)
    survey = gravity.survey.Survey(source_field)

    # Define Tensor Mesh
    dhx = 1
    dhy = 5
    dhz = 1
    hx = [(dhx, 0, 0), (dhx, 100), (dhx, 0, 0)]
    hy = [(dhy, 0, 0), (dhy, 20), (dhy, 0, 0)]
    hz = [(dhz, 0, 0), (dhz, 50), (dhz, 0, 0)]
    mesh = TensorMesh([hx, hy, hz], x0="00N")

    # Define density contrast values for each unit in g/cc
    background_density = 0.0
    formation_density = 0.2

    ind_active = surface2ind_topo(mesh, topo_xyz) 
    nC = int(ind_active.sum())
    model_map = maps.IdentityMap(nP=nC)
    model = background_density * np.ones(nC)

    ind_polygon = model_builder.get_indices_polygon(mesh, points_of_interest)

    ind_polygon = ind_polygon[ind_active]
    model[ind_polygon] = formation_density

    simulation = gravity.simulation.Simulation3DIntegral(survey=survey, mesh=mesh, rhoMap=model_map, 
                                                        ind_active=ind_active, store_sensitivities="forward_only")
    dpred = simulation.dpred(model)
    maximum_anomaly = np.max(np.abs(dpred))
    rec = receiver_list[0].locations

    return rec, dpred
   


def geophysics_data_(points_of_interest, resolution):
    # Defining topography 
    [x_topo, y_topo] = np.meshgrid(np.linspace(0, 100, 41), np.linspace(0, 100, 41))
    z_topo = np.array([0]*1681)
    x_topo, y_topo = mkvc(x_topo), mkvc(y_topo)
    topo_xyz = np.c_[x_topo, y_topo, z_topo]

    # Defining the Survey
    # Define the observation/reciever locations
    x = np.linspace(0.0, 100.0, 10)
    y = np.linspace(0.0, 100.0, 10)
    x, y = np.meshgrid(x, y)
    z = np.array([0]*(100))
    x, y = mkvc(x.T), mkvc(y.T)

    #fun_interp = LinearNDInterpolator(np.c_[x_topo, y_topo], z_topo)
    #z = fun_interp(np.c_[x, y])
    receiver_locations = np.c_[x, y, z]



    # Define the component(s) of the field 
    components = ["gz"]
    receiver_list = gravity.receivers.Point(receiver_locations, components=components)
    receiver_list = [receiver_list]
    source_field = gravity.sources.SourceField(receiver_list=receiver_list)
    survey = gravity.survey.Survey(source_field)

    # Define Tensor Mesh
    dhx = 1
    dhy = 50
    dhz = 1
    hx = [(dhx, 0, 0), (dhx, 100), (dhx, 0, 0)]
    hy = [(dhy, 0, 0), (dhy, 2), (dhy, 0, 0)]
    hz = [(dhz, 0, 0), (dhz, 50), (dhz, 0, 0)]
    mesh = TensorMesh([hx, hy, hz], x0="00N")

    # Define density contrast values for each unit in g/cc
    background_density = 0.0
    formation_density = 0.2

    ind_active = surface2ind_topo(mesh, topo_xyz) 
    nC = int(ind_active.sum())
    model_map = maps.IdentityMap(nP=nC)
    model = background_density * np.ones(nC)

    ind_polygon = model_builder.get_indices_polygon(mesh, points_of_interest)

    ind_polygon = ind_polygon[ind_active]
    model[ind_polygon] = formation_density

    simulation = gravity.simulation.Simulation3DIntegral(survey=survey, mesh=mesh, rhoMap=model_map, 
                                                        ind_active=ind_active, store_sensitivities="forward_only")
    dpred = simulation.dpred(model)
    maximum_anomaly = np.max(np.abs(dpred))
    np.random.seed(737)
    noise = 0.01 * maximum_anomaly * np.random.rand(len(dpred))
    topo_xyz = np.c_[topo_xyz]
    dobs = np.c_[receiver_locations, dpred + noise]
    dobs = dobs[:, -1]
    maximum_anomaly = np.max(np.abs(dobs))
    uncertainties = 0.01 * maximum_anomaly * np.ones(np.shape(dobs))
    receiver_list = gravity.receivers.Point(receiver_locations, components="gz")
    receiver_list = [receiver_list]
    source_field = gravity.sources.SourceField(receiver_list=receiver_list)
    survey = gravity.survey.Survey(source_field)
    data_object = data.Data(survey, dobs=dobs, standard_deviation=uncertainties)
    ind_active = surface2ind_topo(mesh, topo_xyz)
    nC = int(ind_active.sum())
    model_map = maps.IdentityMap(nP=nC) 
    starting_model = np.zeros(nC)
    simulation = gravity.simulation.Simulation3DIntegral(survey=survey, mesh=mesh, rhoMap=model_map, ind_active=ind_active)
    # LEAST SQUARE METHOD
    dmis = data_misfit.L2DataMisfit(data=data_object, simulation=simulation)
    dmis.W = utils.sdiag(1 / uncertainties)
    reg = regularization.Sparse(mesh, active_cells=ind_active, mapping=model_map)
    reg.norms = [0, 2, 2, 2]
    opt = optimization.ProjectedGNCG(maxIter=10, lower=-1.0, upper=1.0, maxIterLS=20, maxIterCG=10, tolCG=1e-3)
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e0)
    update_IRLS = directives.Update_IRLS(f_min_change=1e-4, max_irls_iterations=30, coolEpsFact=1.5, beta_tol=1e-2)
    beta_schedule = directives.BetaSchedule(coolingFactor=5, coolingRate=1)
    save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
    update_jacobi = directives.UpdatePreconditioner()
    # Add sensitivity weights
    sensitivity_weights = directives.UpdateSensitivityWeights(everyIter=False)
    # The directives are defined as a list.
    directives_list = [update_IRLS,sensitivity_weights,
        starting_beta, beta_schedule, save_iteration, update_jacobi,
    ]
    # Here we combine the inverse problem and the set of directives
    inv = inversion.BaseInversion(inv_prob, directives_list)
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    # Run inversion
    recovered_model = inv.run(starting_model)
    print(plotting_map)
    return recovered_model, plotting_map, mesh
























def geophysics_data(points_of_interest, resolution):

    # Direct Current (DC) operator
    grid = discretize.TensorMesh(resolution)
    direct = grid.face_divergence
    sigma = 1e-2 * np.ones(grid.nC)
    MsigI = grid.get_face_inner_product(sigma, invert_model=True, invert_matrix=True)
    A = -direct * MsigI * direct.T
    A[-1, -1] /= grid.cell_volumes[-1]
    rhs = np.zeros(grid.nC)
    txind = discretize.utils.closest_points_index(grid, points_of_interest)
    rhs[txind] = np.r_[1]

    # Solve DC problem (LU solver)
    AinvtM = SolverLU(A)
    phitM = AinvtM * rhs
    result = phitM * -100
    # plot type 1
    final = np.reshape(np.array(result), (resolution[1], resolution[0]))
    return final

