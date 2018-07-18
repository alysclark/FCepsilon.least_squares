from collections import OrderedDict
import numpy as np
import math
from SALib.sample import saltelli
from SALib.analyze import sobol
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, minimize,least_squares


import OpenCOR as oc


dict = {'FCepsilonRI/K_1': 0.0230446,
  'FCepsilonRI/K_10': 31.2545,
  'FCepsilonRI/K_11': 19.7482,
  'FCepsilonRI/K_12': 0.974592,
  'FCepsilonRI/K_2': 0.00116872,
  'FCepsilonRI/K_3': 1.8392,
  'FCepsilonRI/K_4': 9.2245,
  'FCepsilonRI/K_5': 1.95157,
  'FCepsilonRI/K_6': 15.9857,
  'FCepsilonRI/K_7': 0.0340413,
  'FCepsilonRI/K_8': 0.0472939,
  'FCepsilonRI/K_9': 0.00105783,
  'FCepsilonRI/V_1': 37.7896,
  'FCepsilonRI/V_2': 1.43171,
  'FCepsilonRI/V_3': 0.0928482,
  'FCepsilonRI/V_4': 1.26063,
  'FCepsilonRI/k_f1': 70.733,
  'FCepsilonRI/k_f2': 1.16736,
  'FCepsilonRI/k_f3': 5.08223,
  'FCepsilonRI/k_f4': 0.238742}

varyme = 'FCepsilonRI/k_f1'

# The state variable  or variables in the model that the data represents
num_series = 2
expt_state_uri = ['FCepsilonRI/pFC','FCepsilonRI/pSyk']

#Some example output that we are maybe aiming for
times = np.array([0,  480, 960, 1920, 3840])
exp_data = np.zeros([num_series,len(times)])
exp_data[0,:] = np.array([0.0, 0.0408, 0.136, 0.105, 0.136])*.474 #pFC
exp_data[1,:] = np.array([0.0,  0.05437, 0.0644, 0.0518, 0.04373])*.474 #pSyk

#Number of samples to generate for each parameter
num_samples =  100

#Number of results to retain, if we store too many in high res parameter sweeps we can have memory issues
num_retain = 10


#List of parameters you want to exclude from fit
fit_parameters_exclude = ['FCepsilonRI/pLyn']


class Simulation(object):
    def __init__(self):
        self.simulation = oc.simulation()
        self.simulation.data().setStartingPoint(0)
        self.simulation.data().setEndingPoint(3900)
        self.simulation.data().setPointInterval(1)
        self.constants = self.simulation.data().constants()
        self.simulation.resetParameters()
        self.simulation.clearResults()
        for k,v in dict.items():
             self.constants[k]=v
        self.model_constants = OrderedDict({k: self.constants[k]
                                            for k in self.constants.keys()})  
		
    
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
    
    def evaluate_ssq(self):
        self.simulation.clearResults()

        self.simulation.run()
        trial = np.zeros([num_series,len(times)])
        ssq = np.zeros(num_series+1)
		
        for i in range(0,num_series):
            trial[i,:] = self.simulation.results().states()[expt_state_uri[i]].values()[times]
            ssq[i+1] = math.sqrt(np.sum((exp_data[i,:]-trial[i,:])**2))
        ssq[0] = np.sum(ssq[1:num_series+1])
        return ssq 
        
    
    def run_parameter_sweep(self):
        num_cols = num_series + 1 + self.samples.shape[1]
        num_rows = num_retain+1
        Y = np.zeros([num_rows,num_cols])
        for i, X in enumerate(self.samples):
            ssq = self.evaluate_ssq(X)
            j = i
            if j < num_retain:
                Y[j,0] = ssq[0]
                for k in range(0,num_series):
                    Y[j,k+1] = ssq[k+1]
                Y[j,(k+2):num_cols]=X
            else:
                Y[num_retain,0] = ssq[0]
                for k in range(0,num_series):
                    Y[num_retain,k+1] = ssq[k+1]
                Y[num_retain,(k+2):num_cols]=X
                ind = np.argsort(Y[:,0])
                Y=Y[ind]
				
			#Want to retain top N here
        ind = np.argsort(Y[:,0])
        Z=Y[ind]
        return Z

			


plt.close('all')

plt.semilogx([dict[varyme],dict[varyme]],[0,1],label='fit value')
values = np.zeros([4,len(np.arange(-3,2,0.1))])
count = 0
for i in np.arange (-3,3,.1):
      dict[varyme] = 10**i
      
      s = Simulation()
      
      v = s.evaluate_ssq()
      values[0,count] = 10**i
      values[1:4,count] = v
      count = count +1
      print(count,values[0,count-1])
print(values)
plt.semilogx(values[0,:],values[1,:],label='total error', color='red')
plt.semilogx(values[0,:],values[2,:],label='pFC', color='blue')
plt.semilogx(values[0,:],values[3,:],label='pSyk', color='black')
plt.legend()
plt.xlabel(varyme)
plt.ylabel('Error metric')

plt.show()