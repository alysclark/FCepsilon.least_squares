from collections import OrderedDict
import numpy as np
import math
from SALib.sample import saltelli
from SALib.analyze import sobol
from matplotlib import pyplot as plt


import OpenCOR as oc

#Some example output that we are maybe aiming for
times = np.array([0,  480, 960, 1920, 3840])
pFC = np.array([0.0, 0.0408, 0.136, 0.105, 0.136])*.474
pSyk = np.array([0.0,  0.05437, 0.0644, 0.0518, 0.04373])*.474

bounds_dictionary = {'FCepsilonRI/k_f1': [-5,2], 'FCepsilonRI/k_f2': [-5,2],'FCepsilonRI/k_f3': [-5,2], 'FCepsilonRI/k_f4': [-5,2],'FCepsilonRI/K_1': [-5,2], 'FCepsilonRI/K_2': [-5,2],'FCepsilonRI/K_3': [-5,2],
	'FCepsilonRI/K_4': [-5,2], 'FCepsilonRI/K_5': [-5,2],'FCepsilonRI/K_6': [-5,2], 'FCepsilonRI/K_7': [-5,2],'FCepsilonRI/K_8': [-5,2], 'FCepsilonRI/K_9': [-5,2],'FCepsilonRI/K_10': [-5,2], 'FCepsilonRI/K_11': [-5,2],
	'FCepsilonRI/K_12': [-5,2],'FCepsilonRI/V_1': [-5,2],'FCepsilonRI/V_2': [-5,2],'FCepsilonRI/V_3': [-5,2],'FCepsilonRI/V_4': [-5,2], 'FCepsilonRI/pLyn': [-1.32640456,-1.32640455]}


class Simulation(object):
    def __init__(self):
        self.simulation = oc.simulation()
        self.simulation.data().setStartingPoint(0)
        self.simulation.data().setEndingPoint(3900)
        self.simulation.data().setPointInterval(1)
        self.constants = self.simulation.data().constants()
        self.model_constants = OrderedDict({k: self.constants[k]
                                            for k in self.constants.keys()})
        
        #print(self.model_constants)
        
        # default the parameter bounds to something sensible, needs to be set directly
        bounds = []
        print(self.constants)
        for c in self.constants:
            v = self.constants[c];
            print(c)
            print(bounds_dictionary[c][1])
            print(v)
            bounds.append([bounds_dictionary[c][0], bounds_dictionary[c][1]])
        # define our sensitivity analysis problem
        self.problem = {
                   'num_vars': len(self.constants),
                   'names': self.constants.keys(),
                   'bounds': bounds
                   }
        self.samples = saltelli.sample(self.problem, 50)
    
    def run_once(self, c, v):
        self.simulation.resetParameters()
        self.constants[c] = v
        self.simulation.run()
        return (self.simulation.results().points().values(),
                self.simulation.results().states()['FCepsilonRI/pSyk'].values())
    
    def run(self, c, scale=2.0):
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
        #print('Parameter set: ', self.constants)
        self.simulation.run()
        trial_pSyk = self.simulation.results().states()['FCepsilonRI/pSyk'].values()[times]
        ssq_pSyk = math.sqrt(np.sum((pSyk-trial_pSyk)**2))
        
        trial_pFC = self.simulation.results().states()['FCepsilonRI/pFC'].values()[times] 
        ssq_pFC = math.sqrt(np.sum((pFC-trial_pFC)**2))

        #if(ssq_pFC < 0.1 and ssq_pSyk <0.1):
            #print('Parameter set: ', self.constants)
            #print(parameter_values)
            #ax1.plot(times,trial_pFC)
            #ax2.plot(times,trial_pSyk)
            #ax3.plot(ssq_pFC,ssq_pSyk,'*')
            #print(ssq_pSyk, ssq_pFC)
        
        return([ssq_pSyk+ssq_pFC,ssq_pSyk, ssq_pFC])
        
    
    def run_parameter_sweep(self):
        Y = np.zeros([self.samples.shape[0],3+self.samples.shape[1]])
        for i, X in enumerate(self.samples):
            Y[i,0:3] = self.evaluate_ssq(X)
            Y[i,3:self.samples.shape[1]+3]=X
        ind = np.argsort(Y[:,0])
        Z=Y[ind]
        return Z

    def plot_n_best(self,n,param_sweep_results):
        print(param_sweep_results[0:3,:])
        for i in range(0,n):
            self.simulation.clearResults()
            self.simulation.resetParameters()
            for j, k in enumerate(self.model_constants.keys()):
                self.constants[k] = 10.0**param_sweep_results[i,j+3]
            #print(param_sweep_results[i,j+3])
            #print('Parameter set: ', self.constants)
            self.simulation.run()
            trial_pSyk = self.simulation.results().states()['FCepsilonRI/pSyk'].values()[times]
            #print(trial_pSyk)
            trial_pFC = self.simulation.results().states()['FCepsilonRI/pFC'].values()[times]
            ax1.plot(times,trial_pFC)
            ax2.plot(times,trial_pSyk)
            ax3.plot(param_sweep_results[i,1],param_sweep_results[i,2],"*")

plt.close('all')
fig, (ax1, ax2,ax3) = plt.subplots(3, sharey=True)
s = Simulation()

v = s.run_parameter_sweep()
s.plot_n_best(5,v)


ax1.plot(times,pFC,'*')
ax2.plot(times,pSyk,'*')

plt.show()


