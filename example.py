from main.data import file


sub2 = [
        [[[0.0, 2, 0], [3, 2, -2], [6, 2, -5], [7, 2, -8], [9, 2, -10], [15, 2, -14]],
        [[0.0, 5, 0], [3, 5, -3], [6, 5, -5], [7, 5, -9], [9, 5, -12], [15, 5, -15]],
        [[0.0, 10, 0], [3, 10, -2], [6, 10, -5], [7, 10, -8], [9, 10, -11], [15, 10, -16]],
        [[0.0, 15, -1.0], [3, 15, -4], [6, 15, -6], [7, 15, -8], [9, 15, -11.5], [15, 15, -15]],
        [[0.0, 20, 1.0], [3, 20, -2], [6, 20, -4], [7, 20, -8], [9, 20, -11], [15, 20, -16]]],

        [[[0.0, 2, 3], [3, 2, 1], [6, 2, -2], [7, 2, -5], [9, 2, -7], [15, 2, -11]],
        [[0.0, 5, 3], [3, 5, 0], [6, 5, -2], [7, 5, -6], [9, 5, -9], [15, 5, -12]],
        [[0.0, 10, 3], [3, 10, 1], [6, 10, -2], [7, 10, -5], [9, 10, -8], [15, 10, -13]],
        [[0.0, 15, 2.0], [3, 15, -1], [6, 15, -3], [7, 15, -5], [9, 15, -8.5], [15, 15, -12]],
        [[0.0, 20, 4.0], [3, 20, 1], [6, 20, -1], [7, 20, -5], [9, 20, -8], [15, 20, -13]]]
        ]


w = 1
data = file.read_data(sub2, w)

subdivision_data = data.visualize_interactive(100, 100)
level = 5
volumetrics = data.volumetric_mesh(level, 1)
volume_voxel = data.vol_voxelization_2()
iteration = 10
lith_block_MC = data.gaussian_MC(100, 100, level, iteration)

simulation = data.simulation()
probability = data.probability()
entropy = data.entropy()

viz_gravity = data.visualize_geo_final()



# Prior distribution
iteration = 50
lith_block_MC_ = data.gaussian_MC_prior(100, 100, level, iteration)

simulation_ = data.simulation()
probability_ = data.probability()
entropy_ = data.entropy()

sim_mcmc = data.gaussian_MCMC(level)

posterior = data.plot_posterior_distribution()
prior = data.prior_top()

comparison = data.prior_posterior_top()

map_ = data.MAP_model()

error_plot = data.error()

map_sim = data.map_simulation()

#probability = data.probability()

#entropy = data.entropy()

