import numpy as np
import pymc as pm
import pyvista as pv
import calculation
np.random.seed(1234)
def gaussian_pymc(all__, a, b, density):
    
    # This is where to do the simulation but pymc wont work on pycharm this way
        
    def vol_thickness(top, bottom):
        
        n = len(top)
        sum = 0
        for i in range(n):
            sum += (top[i] - bottom[i]) ** 2
        return (sum) ** 0.5
        
    def distribution(prior, uncert):
        for i in range(len(prior)):
            prior[i][-1] = uncert[i]
        return prior

    def surface_uncertainties(all__, a, b, density, control_top, control_bottom, top_unc, bottom_unc, knots_list_u, knots_list_v, dimension):
        # control points with uncertainties
        
        
        top_unc 
        bottom_unc
        
        
        
        dim = (a, b, 1)
        surf_top_, _ = all__.subsurfmodel(a, b, control_top, dimension[0], knots_list_u[0], knots_list_v[0], degree=3)
        surf_bottom_, _ = all__.subsurfmodel(a, b, control_bottom, dimension[1], knots_list_u[1], knots_list_v[1], degree=3)
        surf_top = all__.mesh_points(surf_top_, dim)
        surf_bottom = all__.mesh_points(surf_bottom_, dim)
        
        # -------- volume --------#
        div = np.linspace(0, 1, density) 
        interp_z = div * surf_bottom[2][..., None] + (1 - div) * surf_top[2][..., None]
        interp_y = div * surf_bottom[1][..., None] + (1 - div) * surf_top[1][..., None]
        interp_x = div * surf_bottom[0][..., None] + (1 - div) * surf_top[0][..., None]
        volume = interp_z[..., None] * [0, 0, 1] + interp_y[..., None] * [0, 1, 0] + interp_x[..., None] * [1, 0, 0]
        mesh = pv.StructuredGrid(*volume.transpose())
        
        results = [mesh, surf_top, surf_bottom]
        
        return results

        
    def thickness_uncertainties(s_uncertainties):
                
        # --------- thickness ------------#
        top = list(zip(np.array(s_uncertainties[1])))
        bottom = list(zip(np.array(s_uncertainties[2])))
        thick = []
        for i in range(len(top)):
            thickness = calculation.euc_dist(top[i], bottom[i])
            thick.append(thickness)
        thickness_average = np.mean(thick)

        return thickness_average

   
    def layer_analysis(all__, control_1, control_2):
        
            #---------- voxelization / lith_block-----------#
        mask, mask_ = all__.voxel_vol(mesh)  # they are one dimensional

        return mask
        
        
        
        
    # for checks
    # single value for both thickness and gravity
    # we will extract a distribution using them as the mean and small standardard
    # then we extract their min and max to form the range is which them must fulfill
        
    dist_gravity_prior = np.random.normal(all__.data['geo_val_sum'], 1, size=100)
    dist_thickness_prior  = np.random.normal(all__.data['thickness_average'], 1, size=100)
        
    
    # a single list of all the gravity and lithology (0, 1) values
    original_gravity = all__.data['final_geo']
    original_lith = all__.data['lith_block']

    
    
    
    
    
    dimension = all__.data['dimension']
    knot_list_u = all__.data['knots_list_u']
    knot_list_v = all__.data['knots_list_v']
    control_point_top_obs = all__.data['ctr_points_plot'][0]
    control_point_bottom_obs = all__.data['ctr_points_plot'][1]
    priors_0_z = [item[-1] for item in control_point_top_obs]
    priors_1_z = [item[-1] for item in control_point_bottom_obs]

    with pm.Model as model:
        
        
        
        
        
        top_surface_uncertainty_z = pm.Normal('top_surface_uncertainties', mu=priors_0_z, sigma=1, shape=len(priors_0_z))
        bottom_surface_uncertainty_z = pm.Normal('bottom_surface_uncertainties', mu=priors_1_z, sigma=1, shape=len(priors_1_z))
        
        # 1. 
        surface_uncertainties = pm.Deterministic('surface_uncertainties', surface_uncertainties(all__ = all__, a = a, b = b, 
                                                                                                density = density, control_top = control_point_top_obs, control_bottom = control_point_bottom_obs, 
                                                                                                top_unc = top_surface_uncertainty_z, bottom_unc= bottom_surface_uncertainty_z, knot_list_u = knot_list_u,
                                                                                               knot_list_v = knot_list_v, dimension = dimension))
        
        # 2
        # thickness

        
        # last. where other go and then we decide 
        
        
        lith_block = pm.Deterministic('$lith_block$', lith_block(a, b, density, bottom_surface_uncertainty, top_surface_uncertainty))

    
    
    with model:
        step = pm.Metropolis()
        trace = pm.sample(10, tune=50, step=step, return_inferencedata=False)

        model = pm.Model([top_surface_uncertainty_z, bottom_surface_uncertainty_z, top_surface_uncertainty, bottom_surface_uncertainty, lith_block])
        mod = pm.MCMC(model)
        mod.sample(iter=1000, burn=50)
        lith_blocks = mod.trace('lith_block')[:]

        
    #lith_block_ = np.array([])
    lith_block_ = []
    for i in lith_blocks:
        lith_block_.append(i[0][0])
    
    all__.data['all_lith_block'] = lith_block_

    ############

    pm.Normal("likelihood", mu=prior_array, observed=observed_arrays)
    final_lith_block.append(mask)
    final_lith_block = np.append(final_lith_block, mask)
    self.data['all_lith_block'] = final_lith_block.reshape(size, -1)
