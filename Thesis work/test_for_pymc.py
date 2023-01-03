import pymc as pm
import theano.tensor as T


import numpy as np

'''
# CODE TO RUN FROM JUPYTER LAB
import numpy as np
import test_for_pymc

# 1
data = np.loadtxt("data/mixture_data.csv", delimiter=",")
a = np.array([1])
t = test_for_pymc.gat(1.0, data)

t.tag.test_value

# 2
import numpy as np
import test_for_pymc

full_data = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
z = np.array([1, 2, 2, 61])
shape_0 = z # np.array([1, 2, 3, 4]) #

t = test_for_pymc.tat(z, shape_0, full_data)

'''




def gat(a, data):
    
    def one_f(a):
        a = a +1        
        return a
    
    
    with pm.Model() as model:
        
        
        #a = T.as_tensor_variable(a) # important deterministic always accept only tensor... 
        fixed_one = pm.Poisson("fxd_one", 1, observed=a) # ...alternatively we can use other poinsoon that takes in np.array() and readily converts vaöue into tensor
        one = pm.Deterministic("one", one_f(fixed_one))
        p1 = pm.Uniform('p', 0, 1)
        p2 = one - p1
        p = T.stack([p1, p2])
        assignment = pm.Categorical("assignment", p, shape=data.shape[0], testval=np.random.randint(0, 2, data.shape[0]))
        
    return assignment


def tat(z, shape_0, full_data):
    
    
    def control_point_unc(prior, shape_0, uncert):
        #dat = shape_0.eval()
        #prior = prior.eval()
        w = prior.type.ndim
        #print(w)
        lent = len(shape_0.eval())
        m_pr = T.zeros(lent)
        #m_pr = T.set_subtensor(m_pr[:], uncert[:])
        #z = m_pr.eval()
        #prior = prior.eval()
        #print(z)
        a = []#T.zeros((lent, 3))
        for i in range(lent):
            #th = uncert + 0.001
            #np.append(a, th)
            
            x = T.zeros((lent))
            x = T.set_subtensor(prior[i][2], uncert)
            a.append(x.eval())
            #th[-1] = m_pr[i]
            #a[i] = th
            
            #re = T.inc_subtensor(prior[i][-1], m_pr[i]) 
            #prior[i][-1] = m_pr[i]
            #x = T.inc_subtensor(prior[i][-1], uncert[i])
            #re = T.inc_subtensor(a[i], f)
        
        
        #print(x)
        #print()
        
        #r = T.ones(3)
        #t = a * 34
        zu = T.stack(a)
        zuu = T.concatenate(a, axis=0)
        it = zu[0].eval()
        print(zu[0].type.ndim)
        print(zuu.type.ndim)
        
        #e = zu.eval()

        return zu#prior #a
    
    
    with pm.Model() as model_2:
        
        top_surface_uncertainty_z = pm.Normal('top_surface_uncertainties', mu=0, sigma=1)
        
        
        full_data_ = T.as_tensor_variable(full_data) 
        #pm.Data("full_data", value=full_data, mutable=True)
        #shape_0_ = pm.Data("shape", value=shape_0)
        shape_0_ = T.as_tensor_variable(shape_0)
        #side = 
        coords = pm.Deterministic("one", control_point_unc(prior= full_data_ , shape_0 = shape_0_, uncert = top_surface_uncertainty_z))
        
    #with model:
        #step = pm.Metropolis()
        trace = pm.sample(10, tune=50, return_inferencedata=False)
        
        
        
        
        
        
        
        
    return trace


    '''
        k = pm.Normal('k', mu=0, sd=5, testval=kinit[0])
        m = pm.Normal('m', mu=0, sd=5, testval=kinit[1])
        object_gamma_logistic = T.zeros(S)
        m_pr = T.zeros(S+1)
        #object_gamma_logistic = theano.shared(gamma_logistic, name='gamma_logistic')
        #m_pr = m  # The offset in the previous segment
        m_pr = T.inc_subtensor(m_pr[0], m)
        for i in range(0, S, 1):
            object_gamma_logistic = T.inc_subtensor(object_gamma_logistic[i],(object_t_change[i] - m_pr[i]) * (1 - k_s[i] / k_s[i + 1]))
            #object_gamma_logistic[i] = (object_t_change[i] - m_pr) * (1 - k_s[i] / k_s[i + 1])
            #m_pr = m_pr + object_gamma_logistic[i]  # update for the next segment
            m_pr = T.inc_subtensor(m_pr[i+1],object_gamma_logistic[i]+m_pr[i])


    '''
#full_data_ = pm.Poisson("full_data", 1, observed=full_data)

    class dict_value(dict):
    def __getattr__(self, name):
        return self[name]
    

args = dotdict({
    'foldername': 're_kv_88_like05_nug5010',
    'hmc_step':0.001,
    'likelihoodstd': 0.5})

#sigma = args.likelihoodstd