from collections import OrderedDict
import numpy as np
import math
from SALib.sample import saltelli
from SALib.analyze import sobol
from matplotlib import pyplot as plt


import OpenCOR as oc

#Some example output that we are maybe aiming for
times = np.array([0,  480, 960, 1920, 3840])
pFC = np.array([0.0,  0.0208, 0.0646, 0.0495, 0.0645])
pSyk = np.array([0.0,  0.0255, 0.0303, 0.0242, 0.0202])

bounds_dictionary = {'FCepsilonRI/k_f1': [-4,2], 'FCepsilonRI/k_r1': [-3,1], 'FCepsilonRI/k_f2': [-4,2], 'FCepsilonRI/k_f3': [-4,2],
        'FCepsilonRI/k_r3': [-3,1], 'FCepsilonRI/k_f4': [-4,2], 'FCepsilonRI/k_f5': [-4,2], 'FCepsilonRI/k_r5': [-3,1],'FCepsilonRI/k_f6': [-4,2]}
class Simulation(object):
    def __init__(self):
        self.simulation = oc.simulation()
        self.simulation.data().setStartingPoint(0)
        self.simulation.data().setEndingPoint(3900)
        self.simulation.data().setPointInterval(1)
        self.constants = self.simulation.data().constants()
        self.model_constants = OrderedDict({k: self.constants[k]
                                            for k in self.constants.keys()})
        
        print(self.model_constants)
        
        # default the parameter bounds to something sensible, needs to be set directly
        bounds = []
        for c in self.constants:
            v = self.constants[c];
            print(bounds_dictionary[c][1])
            #print(v)
            bounds.append([bounds_dictionary[c][0], bounds_dictionary[c][1]])
        # define our sensitivity analysis problem
        self.problem = {
                   'num_vars': len(self.constants),
                   'names': self.constants.keys(),
                   'bounds': bounds
                   }
        self.samples = saltelli.sample(self.problem, 3)
    
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
		#This is not actually clearing and resetting results
        for i, k in enumerate(self.model_constants.keys()):
            self.constants[k] = 10.0**parameter_values[i]
        print('Parameter set: ', self.constants)
        self.simulation.run()
        trial_pSyk = self.simulation.results().states()['FCepsilonRI/pSyk'].values()[times]
        ssq_pSyk = math.sqrt(np.sum((pSyk-trial_pSyk)**2))
        
        trial_pFC = self.simulation.results().states()['FCepsilonRI/pFC'].values()[times]
        ssq_pFC = math.sqrt(np.sum((pFC-trial_pFC)**2))
        
        plt.plot(times,trial_pSyk)
        #ssq=0.0
        #for i in range(0,len(differ)):
        #    ssq = ssq+differ[i]**2
        #ssq = np.sqrt(ssq)
        print(ssq_pSyk,ssq_pFC)
        return(ssq_pSyk)
        
    
    def run_parameter_sweep(self):
        Y = np.zeros([self.samples.shape[0]])
        for i, X in enumerate(self.samples):
            Y[i] = self.evaluate_ssq(X)
        
        return Y


s = Simulation()

v = s.run_parameter_sweep()
plt.plot(times,pSyk,'*')
plt.show()
