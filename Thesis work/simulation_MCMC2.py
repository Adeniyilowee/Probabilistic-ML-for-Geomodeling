import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import calculation
import pyvista as pv
import numpy as np
from geophysics_data import geophysics_data_, geophysics_data_final

class Bayesian_Inference_Analysis(object):
    
    def __init__(self, Model, a , b, layer_density) -> None:
        self.model = Model
        self.a = a
        self.b = b
        self.density = layer_density
        dtype= "float64"
        
        
        if dtype == "float32":
            self.tfdtype = tf.float32
        elif dtype == "float64":
            self.tfdtype = tf.float64
        self.dtype = dtype
        self.reset_sess()

    
    
    def evaluate(self, tensors):
        """Evaluates Tensor or EagerTensor to Numpy `ndarray`s.
        Args:
        tensors: Object of `Tensor` or EagerTensor`s; can be `list`, `tuple`,
          `namedtuple` or combinations thereof.
 
        Returns:
          ndarrays: Object with same structure as `tensors` except with `Tensor` or
            `EagerTensor`s replaced by Numpy `ndarray`s.
        """
        if tf.executing_eagerly():
            return tf.nest.pack_sequence_as(
                tensors,
                [t.numpy() if tf.is_tensor(t) else t
                 for t in tf.nest.flatten(tensors)])
        return sess.run(tensors)
    
    def session_options(self, enable_gpu_ram_resizing=True, enable_xla=True):
        """
        Allowing the notebook to make use of GPUs if they're available.
    
        XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear 
        algebra that optimizes TensorFlow computations.
        """
        config = tf.compat.v1.ConfigProto()
        config.log_device_placement = True
        if enable_gpu_ram_resizing:
            config.gpu_options.allow_growth = True
        if enable_xla:
            # Enable on XLA. https://www.tensorflow.org/performance/xla/.
            config.graph_options.optimizer_options.global_jit_level = (
                tf.compat.v1.OptimizerOptions.ON_1)
        return config
    
    def reset_sess(self, config=None):
        """
        Convenience function to create the TF graph & session or reset them.
        """
        if config is None:
            config = self.session_options()
        global sess
        tf.compat.v1.reset_default_graph()
        try:
            sess.close()
        except:
            pass
        sess = tf.compat.v1.InteractiveSession(config=config)

    def loss_minimize(self):
        loss =  tf.negative(self.joint_log_prob(self.top_init, self.bottom_init))
        return loss

    def loss(self, top_init, bottom_init):
        loss =  tf.negative(self.joint_log_prob(top_init, bottom_init))
        return loss
      
    def likelihood(self,grav_data):
        self.Obs_grav = grav_data

            
    def top_prior(self, mean, shape, sigma):
        '''
           prior distribution of the upper surface control points
        '''
        self.mu_prior_top = np.array(mean)
        
        # define the model parameters based on 
        self.std_top = tf.ones(shape, self.tfdtype)
        
        # define the model parameters based on cov matrix
        mu = np.array(mean)
        self.cov_matrix_top = sigma**2*tf.eye(mu.shape[0])
 
        
        
    def bottom_prior(self, mean, shape, sigma):
        '''
           prior distribution of the lower surface control points
        '''

        self.mu_prior_bottom = np.array(mean)
        
        # define the model parameters based on 
        self.std_bottom = tf.ones(shape, self.tfdtype)
        
        # define the model parameters based on cov matrix
        mu = np.array(mean)
        self.cov_matrix_bottom = sigma**2*tf.eye(mu.shape[0])

    def control_point_unc(self, prior_t, uncert_t, prior_b, uncert_b):

        for i in range(len(prior_t)):
            prior_t[i][-1] =  uncert_t[i]

        for i in range(len(prior_b)):
            prior_b[i][-1] =  uncert_b[i]

        return prior_t, prior_b
    
    
    def uncertainty(self, top_dist, bottom_dist):
        # control points are to be estimated from the input here which are with uncertainties
        top_unc, bottom_unc = self.control_point_unc(self.model.data['ctr_points_plot'][0], top_dist, self.model.data['ctr_points_plot'][1], bottom_dist)
        #bottom_unc = self.control_point_unc()
        dim = (self.a, self.b, 1)
        surf_top_, _ = self.model.subsurfmodel(self.a, self.b, top_unc, self.model.data['dimension'][0], self.model.data['knots_list_u'][0], self.model.data['knots_list_v'][0], degree=3)
        surf_bottom_, _ = self.model.subsurfmodel(self.a, self.b, bottom_unc, self.model.data['dimension'][1], self.model.data['knots_list_u'][1], self.model.data['knots_list_v'][1], degree=3)
        surf_top = self.model.mesh_points(surf_top_, dim)
        surf_bottom = self.model.mesh_points(surf_bottom_, dim)
        
        # -------- volume --------#
        div = np.linspace(0, 1, self.density)
        interp_z = div * surf_bottom[2][..., None] + (1 - div) * surf_top[2][..., None]
        interp_y = div * surf_bottom[1][..., None] + (1 - div) * surf_top[1][..., None]
        interp_x = div * surf_bottom[0][..., None] + (1 - div) * surf_top[0][..., None]
        volume = interp_z[..., None] * [0, 0, 1] + interp_y[..., None] * [0, 1, 0] + interp_x[..., None] * [1, 0, 0]
        
        mesh = pv.StructuredGrid(*volume.transpose())
        mask, mask_ = self.model.voxel_vol(mesh)  # they are one dimensional
 
        return mask, mask_, mesh

    def uncer_geo(self, top_dist, bottom_dist):

        mask, mask_, mesh = self.uncertainty(top_dist, bottom_dist)
        #--------- gravity ------------#
        grav = self.model.geophysics_data(mask_)
        grav_ = grav.flatten()
    
        return grav_


    def uncer_geo_(self, top_dist, bottom_dist):

        mask, mask_, mesh = self.uncertainty(top_dist, bottom_dist)
        #--------- gravity ------------#
        points_of_interest = np.array(self.model.data['mesh'].points)
        recovered_grav, plotting_map, mesh = geophysics_data_(points_of_interest, self.model.data['mask_dimension'])
        
        return recovered_grav

    def uncer_geo_final(self, top_dist, bottom_dist):

        mask, mask_, mesh = self.uncertainty(top_dist, bottom_dist)
        #--------- gravity ------------#
        points_of_interest = np.array(self.model.data['mesh'].points)
        rec, dpred = geophysics_data_final(points_of_interest)
        
        return dpred
        
        
    @tf.function
    def joint_log_prob(self, top_init, bottom_init): # , grav_initial, th_init. remember their index must correspond with the initial chain state


        ## 1 define prior as a multivariate normal distribution
        self.mvn_prior_upper = tfd.MultivariateNormalDiag(loc=self.mu_prior_top, scale_diag = self.std_top) # getting 1 distributions
        self.mvn_prior_lower = tfd.MultivariateNormalDiag(loc=self.mu_prior_bottom, scale_diag = self.std_bottom) # getting 1 distributions

        sess = tf.compat.v1.InteractiveSession()
        print('... And then HERE!')
        prior_sampling_t = self.mvn_prior_upper.sample(1)
        prior_sampling_b = self.mvn_prior_lower.sample(1)

        [t] = sess.run(prior_sampling_t)
        [b] = sess.run(prior_sampling_b)


        # forward calculating gravity
        grav_ = self.uncer_geo_final(t, b)
 

        sig_grav = 0.0001**2*tf.eye(np.array(grav_).shape[0])
        grav_dist_ = tfd.MultivariateNormalTriL(loc=grav_, scale_tril=tf.cast(tf.linalg.cholesky(sig_grav), self.tfdtype))
        

        ## 2 Likelihood Check
        likelihood_log_prob_g = grav_dist_.log_prob(self.Obs_grav)
        prior_log_prob_upper = tf.reduce_sum(self.mvn_prior_upper.log_prob(tf.cast(top_init, tf.float64)))
        prior_log_prob_lower = tf.reduce_sum(self.mvn_prior_lower.log_prob(tf.cast(bottom_init, tf.float64)))
 
        joint_log = (likelihood_log_prob_g + prior_log_prob_upper + prior_log_prob_lower)
        
        return joint_log
    
    
    
    def tf_MCMC(self, method,
                 num_results,
                 number_burnin,
                 step_size,
                 num_leapfrog_steps = None,
                 initial_chain_state = None):

        
        self.unnormalized_posterior_log_prob = lambda *args: self.joint_log_prob(*args)
        
        self.initial_chain_state = initial_chain_state


        if method == 'RWM':

            states  = self.RWMetropolis(num_results,number_burnin,step_size,parallel_iterations = 1)

            [top_init, bottom_init], kernel_results = states
            [self.model.data['top_posterior_'], self.model.data['bottom_posterior_'], kernel_results_] = self.evaluate([top_init, bottom_init, kernel_results])
            return states 
        
        
        if method == 'HMC':
            if num_leapfrog_steps is None:
                ValueError('num_leapfrog_steps is required')
            states  = self.HMCarlo(num_results,number_burnin,step_size,num_leapfrog_steps)
            
            [top_init, bottom_init], kernel_results = states
            [self.model.data['top_posterior_'], self.model.data['bottom_posterior_'], kernel_results_] = self.evaluate([top_init, bottom_init, kernel_results])

            return states
            


    
    def HMCarlo(self,num_results,number_burnin,step_size,num_leapfrog_steps):
        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo( target_log_prob_fn=self.unnormalized_posterior_log_prob,
                                                    step_size = step_size,
                                                    num_leapfrog_steps = num_leapfrog_steps)
        
        
        states  = tfp.mcmc.sample_chain(num_results=num_results,
                                        current_state=self.initial_chain_state,
                                        # trace_fn=None,
                                        kernel= hmc_kernel,
                                        num_burnin_steps=number_burnin,
                                        num_steps_between_results=0,
                                        parallel_iterations=3,
                                        seed=42)
        
        return states


    def RWMetropolis(self,num_results,number_burnin,step_size,parallel_iterations = 1):
        
        def gauss_new_state_fn(scale, dtype):
            gauss = tfd.Normal(loc=dtype(0), scale=dtype(scale))
            def _fn(state_parts, seed):

                next_state_parts = []

                part_seeds = tfp.random.split_seed(
                seed, n=len(state_parts), salt='rwmcauchy')

                for sp, ps in zip(state_parts, part_seeds):
                    next_state_parts.append(sp + gauss.sample(
                    sample_shape=sp.shape, seed=ps))
                #print('Here man')
                return next_state_parts
            return _fn

        rmh_kernel = tfp.mcmc.RandomWalkMetropolis(target_log_prob_fn = self.unnormalized_posterior_log_prob,
                                                   new_state_fn = gauss_new_state_fn(scale = step_size, dtype = np.float64))

        
        print('Error start')
        states = tfp.mcmc.sample_chain(num_results = num_results,
                                       current_state = self.initial_chain_state,
                                       #trace_fn=None,
                                       kernel = rmh_kernel,
                                       num_burnin_steps = number_burnin,
                                       num_steps_between_results = 0,
                                       parallel_iterations = 1,
                                       seed = 42)
        print('Error end')

        return states

    def MAP_estimate(self, optimizer, top_init, bottom_init, iterations = 10000, learning_rate = 0.001):


        if optimizer == 'Sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate, 
                                                momentum=0.0, nesterov=False, name="SGD")

        if optimizer == 'Nadam':
            optimizer = tf.keras.optimizers.Nadam(
                                                  learning_rate = learning_rate, 
                                                  beta_1=0.9, beta_2=0.999, epsilon=1e-07)

        if optimizer == 'Adag':
            optimizer = tf.keras.optimizers.Adagrad(
                                                    learning_rate = learning_rate,
                                                    initial_accumulator_value = 0.1, epsilon=1e-07)
         
        if optimizer == 'Adam':
            optimizer = tf.keras.optimizers.Adam(
                                                learning_rate=learning_rate, 
                                                beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        cost_dec_list = []
        cost_list = []
        top_list = []
        bottom_list = []
        
        self.top_init = tf.Variable(top_init)
        self.bottom_init = tf.Variable(bottom_init)
        tolerance  = 1e-13
        
        for i in range(iterations):

            optimizer.minimize(self.loss_minimize, var_list=[self.top_init, self.bottom_init])
            loss = self.loss(self.top_init, self.bottom_init).numpy()
            
            if cost_list: 
                if (cost_list[-1]-loss) < tolerance and (cost_list[-1]-loss) >= 0:
                    break
            
                z_ = cost_list[-1]-loss
                cost_dec_list.append(z_)
                print('i:', i, 'loss:', '{0:.10f}'.format(cost_dec_list[-1]))



            cost_list.append(loss) 

            top_list.append(self.top_init.numpy())
            bottom_list.append(self.bottom_init.numpy())

        return top_list, bottom_list, cost_list, cost_dec_list







