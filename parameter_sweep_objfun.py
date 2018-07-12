from collections import OrderedDict
import numpy as np
import math
from SALib.sample import saltelli
from SALib.analyze import sobol
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, minimize,least_squares


import OpenCOR as oc


bounds_dictionary = {'FCepsilonRI/k_f1': [-1,2], 'FCepsilonRI/k_f2': [-1,2],'FCepsilonRI/k_f3': [-1,2], 'FCepsilonRI/k_f4': [-1,2],'FCepsilonRI/K_1': [-1,2], 'FCepsilonRI/K_2': [-1,2],'FCepsilonRI/K_3': [-1,2],
	'FCepsilonRI/K_4': [-1,2], 'FCepsilonRI/K_5': [-1,2],'FCepsilonRI/K_6': [-1,2], 'FCepsilonRI/K_7': [-1,2],'FCepsilonRI/K_8': [-1,2], 'FCepsilonRI/K_9': [-1,2],'FCepsilonRI/K_10': [-1,2], 'FCepsilonRI/K_11': [-1,2],
	'FCepsilonRI/K_12': [-1,2],'FCepsilonRI/V_1': [-1,2],'FCepsilonRI/V_2': [-1,2],'FCepsilonRI/V_3': [-1,2],'FCepsilonRI/V_4': [-1,2]}

# The state variable  or variables in the model that the data represents
expt_state_uri = ['FCepsilonRI/pFC','FCepsilonRI/pSyk']

#Some example output that we are maybe aiming for
times = np.array([0,  480, 960, 1920, 3840])
num_series = 2
exp_data = np.zeros([num_series,len(times)])
exp_data[0,:] = np.array([0.0, 0.0408, 0.136, 0.105, 0.136])*.474 #pFC
exp_data[1,:] = np.array([0.0,  0.05437, 0.0644, 0.0518, 0.04373])*.474 #pSyk

#Number of samples to generate for each parameter
num_samples = 10


#List of parameters you want to exclude from fit
fit_parameters_exclude = ['FCepsilonRI/pLyn']


class Simulation(object):
    def __init__(self):
        self.simulation = oc.simulation()
        self.simulation.data().setStartingPoint(0)
        self.simulation.data().setEndingPoint(3900)
        self.simulation.data().setPointInterval(1)
        self.constants = self.simulation.data().constants()
        self.constant_parameter_names = list(self.constants.keys())
											
        for i in range(0,len(fit_parameters_exclude)):
            self.constant_parameter_names.remove(fit_parameters_exclude[i])
			
        self.model_constants = OrderedDict({k: self.constants[k]
                                            for k in self.constant_parameter_names})

        print('model const:')
        
        print(self.model_constants)
        print('const:')
        print(self.constants)
        
        # default the parameter bounds to something sensible, needs to be set directly
        bounds = []
        for c in self.constant_parameter_names:
            v = self.constants[c];
            print(c,v)
            bounds.append([bounds_dictionary[c][0], bounds_dictionary[c][1]])
        # define our sensitivity analysis problem
        self.problem = {
                   'num_vars': len(self.constant_parameter_names),
                   'names': self.constant_parameter_names,
                   'bounds': bounds
                   }
        self.samples = saltelli.sample(self.problem, num_samples)
    
    def run_once(self, c, v):
        self.simulation.resetParameters()
        self.constants[c] = v
        self.simulation.run()
        return (self.simulation.results().points().values(),
                self.simulation.results().states()['FCepsilonRI/pSyk'].values())
    
    def run_sensitvity(self, c, scale=2.0):
        self.simulation.clearResults()
        v = self.model_constants[c]
        base = self.run_once(c, v)[1][times]
        divergence = 0.0
        for s in [1.0/scale, scale]:
            trial = self.run_once(c, s*v)[1][times]
            divergence += math.sqrt(np.sum((base - trial)**2))
        return divergence
    
    def evaluate_model(self, parameter_values):
        self.simulation.clearResults()
        for i, k in enumerate(self.model_constants.keys()):
            self.constants[k] = parameter_values[i]
        #print('Parameter set: ', parameter_values)
        self.simulation.run()
        return (self.simulation.results().states()['FCepsilonRI/pSyk'].values()[times])
    
    def evaluate_ssq(self, parameter_values):
        self.simulation.clearResults()
        self.simulation.resetParameters()
		#This is not actually clearing and resetting results
        for i, k in enumerate(self.model_constants.keys()):
            self.constants[k] = 10.0**parameter_values[i]
            print(k,self.constants[k])
        #print('Parameter set: ', self.constants)
        self.simulation.run()
        trial = np.zeros([num_series,len(times)])
        ssq = np.zeros(num_series+1)
		
        for i in range(0,num_series):
            trial[i,:] = self.simulation.results().states()[expt_state_uri[i]].values()[times]
            ssq[i+1] = math.sqrt(np.sum((exp_data[i,:]-trial[i,:])**2))
        ssq[0] = np.sum(ssq[1:num_series])
        return ssq 
        
    
    def run_parameter_sweep(self):
        num_cols = num_series + 1 + self.samples.shape[1]
        Y = np.zeros([self.samples.shape[0],num_cols])
        for i, X in enumerate(self.samples):
            ssq = self.evaluate_ssq(X)
            Y[i,0] = ssq[0]
            for j in range(0,num_series):
                Y[i,j+1] = ssq[j+1]
            Y[i,(j+2):num_cols]=X
        ind = np.argsort(Y[:,0])
        Z=Y[ind]
        return Z

    def plot_n_best(self,n,param_sweep_results):
        for i in range(0,n):
            self.simulation.clearResults()
            self.simulation.resetParameters()
            for j, k in enumerate(self.model_constants.keys()):
                self.constants[k] = 10.0**param_sweep_results[i,j+num_series+1]
            #print(param_sweep_results[i,j+3])
            #print('Parameter set: ', self.constants)
            self.simulation.run()
            trial = np.zeros([num_series,len(times)])
            for i in range(0,num_series):
                trial[i,:] = self.simulation.results().states()[expt_state_uri[i]].values()[times]
            ax1.plot(times,trial[0,:])
            ax2.plot(times,trial[1,:])
            ax3.plot(param_sweep_results[i,1],param_sweep_results[i,2],"*")
			
    def model_function_lsq(self,params,times,exp_data, return_type, debug=True):
        if debug:
             print('Fitting Parameters:')
             print(params)

        self.simulation.resetParameters()
        self.simulation.clearResults()
        for j, k in enumerate(self.model_constants.keys()):
            print(j,k,params[j])
            self.constants[k] = params[j]

        try:
            self.simulation.run()
        except RuntimeError:
            print("Runtime error:")
            for n, v in enumerate(params[0:len(self.constant_parameter_names)]):
                print('  {}: {}'.format(self.constant_parameter_names[n], v))
            raise

        if return_type == 'optimisation':
            f1 = self.simulation.results().states()[expt_state_uri[0]].values()[times]-exp_data[0,:]
            f2 = self.simulation.results().states()[expt_state_uri[1]].values()[times]-exp_data[1,:]
            f = np.concatenate((f1,f2))
            print('SSD:')
            print(sum(f**2))
        elif return_type == 'visualisation':
            f1 = self.simulation.results().states()[expt_state_uri[0]].values()[times]
            f2 = self.simulation.results().states()[expt_state_uri[1]].values()[times]
            f = np.vstack((f1,f2))
        return f

plt.close('all')
fig, (ax1, ax2,ax3) = plt.subplots(3, sharey=True)
s = Simulation()

v = s.run_parameter_sweep()
s.plot_n_best(5,v)


ax1.plot(times,exp_data[0,:],'*')
ax2.plot(times,exp_data[1,:],'*')

plt.show()

initial_params = 10**v[0,num_series+1:len(v[0,:])]
print(initial_params)


opt =least_squares(s.model_function_lsq, initial_params, args=(times,exp_data, 'optimisation'),
                               xtol=1e-6,verbose=1)

f =s.model_function_lsq(opt.x, times, exp_data,'visualisation', debug=False)

fig, ax = plt.subplots()
plt.plot(times, exp_data[1,:], 'o', label='Experiment pSyk', color='red')
plt.plot(times, f[1], '-', label='Model pSyk', color='blue')
pSyk_error = f[1] - exp_data[1,:]

print('pSyk error = ' + str(np.mean(pSyk_error)) + ' (SD ' + str(np.std(pSyk_error)) +')')


fig.canvas.draw()
plt.show()
fig, ax = plt.subplots()
plt.plot(times, exp_data[0,:], 'o', label='Experiment pFC', color='red')
plt.plot(times, f[0], '-', label='Model pFC', color='blue')
pFC_error = f[0] - exp_data[0,:]
print('pFC error = ' + str(np.mean(pFC_error)) + ' (SD ' + str(np.std(pFC_error)) +')')
fig.canvas.draw()
plt.show()