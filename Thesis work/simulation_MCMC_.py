import numpy as np
import pymc as pm
import pyvista as pv
import theano.tensor as T
import calculation
np.random.seed(1234)

'''
# Original everything
original_control_top = data.data['ctr_points_plot'][0]
original_control_bottom = data.data['ctr_points_plot'][1]
original_lith_block = data.data['lith_block']
gravy = data.data['mask_bool']

# 2
sim_mcmc = data.gaussian_MCMC(density)
'''



def gaussian_pymc(all__, a, b, density):
    
    # This is where to do the simulation but pymc wont work on pycharm this way
        
    def vol_thickness(top, bottom):
        
        #n = len(top)
        sum = 0
        for i in range(top.eval()):
            sum += (top[i] - bottom[i]) ** 2
        return (sum) ** 0.5
    
        
    def control_point_unc(prior, shape_0, uncert):
        #uncert = uncert.tag.test_value
        #prior = prior.eval
        #r = np.array([])
        a = T.zeros((shape_0, 3))
        for i in range(shape_0.eval()):
            f = T.set_subtensor(prior[i][-1], uncert[i])
            re = T.set_subtensor(a[i], f)
            
        return a
    
    
    def surface_uncertainties(a, b, density, shape_0, shape_1, control_top, control_bottom, top_unc, bottom_unc, knots_list_u, knots_list_v, dimension):
        
    
        # control points are to be estimated from the input here which are with uncertainties
        top_unc = control_point_unc(control_top, shape_0, top_unc)
        bottom_unc = control_point_unc(control_bottom, shape_1, bottom_unc)
        
        dim = (a, b, 1)
        surf_top_, _ = all__.subsurfmodel(a.eval(), b, top_unc, dimension[0], knots_list_u[0], knots_list_v[0], degree=3)
        surf_bottom_, _ = all__.subsurfmodel(a, b, bottom_unc, dimension[1], knots_list_u[1], knots_list_v[1], degree=3)
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

   

    def layer_analysis(mesh, thickness_uncertainties, obs_gravity, obs_lith, gravity_prior, dist_thickness_prior, thickness_min, thickness_max):
        
        #----------- Condition -------------------#
        
        try:
             
            #---------- voxelization / lith_block-----------#
            mask, mask_ = all__.voxel_vol(mesh[0])  # they are one dimensional
            index = np.where(np.array(mask) == 1)[0]
            geo_val_average = np.mean(list(obs_gravity[index]))
            
            if (geo_val_average => gravity_prior) and (thickness_min <= thickness_uncertainties <= thickness_max):
                return mask
            else:
                raise Exception()
        
        except Exception:
            
            return obs_lith

        
        
        
        
    # for checks
    # single value for both thickness and gravity
    # we will extract a distribution using them as the mean and small standardard
    # then we extract their min and max to form the range is which them must fulfill
    gravity_prior = np.random.normal(all__.data['geo_val_average'])
    dist_thickness_prior  = np.random.normal(all__.data['thickness_average'], 1,size=100)

        
    thickness_min = T.as_tensor_variable(min(dist_thickness_prior))
    thickness_max = T.as_tensor_variable(max(dist_thickness_prior))
    
    # a single list of all the gravity and lithology (0, 1) values
    obs_gravity = T.as_tensor_variable(all__.data['final_geo']) #original
    obs_lith = T.as_tensor_variable(all__.data['lith_block']) #original
    
    a = T.as_tensor_variable(a)
    b = T.as_tensor_variable(b)
    dimension = T.as_tensor_variable(all__.data['dimension'])
    knots_list_u = T.as_tensor_variable(all__.data['knots_list_u'])
    knots_list_v = T.as_tensor_variable(all__.data['knots_list_v'])

    control_point_top_obs = all__.data['ctr_points_plot'][0] #original
    control_point_bottom_obs = all__.data['ctr_points_plot'][1] #original
    
    priors_0_z = [item[-1] for item in control_point_top_obs] #original
    shape_00 = len(priors_0_z)
    priors_1_z = [item[-1] for item in control_point_bottom_obs] #original
    shape_10 = len(priors_1_z)
    
    control_point_top_obs =  T.as_tensor_variable(control_point_top_obs)
    control_point_bottom_obs =  T.as_tensor_variable(control_point_bottom_obs)
    
    

    with pm.Model as model:
        
        top_surface_uncertainty_z = pm.Normal('top_surface_uncertainties', mu=priors_0_z, sigma=1, shape=shape_0)
        bottom_surface_uncertainty_z = pm.Normal('bottom_surface_uncertainties', mu=priors_1_z, sigma=1, shape=shape_1)
        shape_0_ = T.as_tensor_variable(shape_00)
        shape_1_ = T.as_tensor_variable(shape_10)

        
        # 1. 
        surface_uncertainties = pm.Deterministic('surface_uncertainties', surface_uncertainties(a, b , density, shape_0_, shape_1_, control_point_top_obs, control_point_bottom_obs, top_surface_uncertainty_z, bottom_surface_uncertainty_z,knot_list_u, knot_list_v, dimension))
        
        # 2
        # thickness
        thickness_uncertainties = pm.Deterministic('thickness_uncertainties', thickness_uncertainties(surface_uncertainties))
        
        
        # 3. last. where other go and then we decide
        layer_analysis = pm.Deterministic('layer_analysis', layer_analysis(surface_uncertainties, thickness_uncertainties, obs_gravity, obs_lith, gravity_prior, dist_thickness_prior, thickness_min, thickness_max))

    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
    with model:
        step = pm.Metropolis()
        trace = pm.sample(10, tune=50, step=step, return_inferencedata=False)

        model = pm.Model([top_surface_uncertainty_z, bottom_surface_uncertainty_z, top_surface_uncertainty, bottom_surface_uncertainty, lith_block])
        mod = pm.MCMC(model)
        mod.sample(iter=1000, burn=50)
        lith_blocks = llomod.trace('lith_block')[:]

        
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
