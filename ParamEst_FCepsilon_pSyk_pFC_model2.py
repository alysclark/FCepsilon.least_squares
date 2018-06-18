import numpy as np

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, minimize,least_squares

import OpenCOR as oc

# Experiment data -- has to be regularly sampled
expt_data = np.loadtxt('Exp_pSyk_pFC.csv', delimiter=',')
expt_time = expt_data[...,0]
expt_time = np.array([0, 240, 480, 960, 1920, 3840])
expt_pSyk = expt_data[...,1]
expt_pFC = expt_data[...,2]

# The state variable in the model that the data represents
expt_state_uri_pSyk = 'FCepsilonRI/pSyk'
expt_state_uri_pFC = 'FCepsilonRI/pFC'

# Load and initialise the simulation
simulation = oc.openSimulation('FCepsilonRI_Faeder_model2.cellml')

# In case we have reloaded an open simulation
simulation.resetParameters()
simulation.clearResults()

# Reference some simulation objects
initial_data = simulation.data()
constants = initial_data.constants()
states = initial_data.states()
results = simulation.results()

# Simulation time points must match experiment data points
initial_data.setStartingPoint(0.0)
initial_data.setEndingPoint(4000)
initial_data.setPointInterval(1)

# Specify as two parallel lists:
# 1. Uri's of parameters to estimate
# 2. Initial values of parameters
constant_parameter_names = list(constants.keys())
state_parameter_names = list(states.keys())
constant_parameter_names.remove('FCepsilonRI/k_f6')
#constant_parameter_names.remove('FCepsilonRI/k_r1')
#constant_parameter_names.remove('FCepsilonRI/k_r3')
#constant_parameter_names.remove('FCepsilonRI/k_r5')

state_parameter_names.remove('FCepsilonRI/pGrb2')
state_parameter_names.remove('FCepsilonRI/FC')
state_parameter_names.remove('FCepsilonRI/pFC')
state_parameter_names.remove('FCepsilonRI/Syk')
state_parameter_names.remove('FCepsilonRI/pSyk')
state_parameter_names.remove('FCepsilonRI/pFCLyn')
state_parameter_names.remove('FCepsilonRI/pFCSyk')
state_parameter_names.remove('FCepsilonRI/pSykGrb2')
state_parameter_names.remove('FCepsilonRI/Lyn')
state_parameter_names.remove('FCepsilonRI/pLyn')

initial_constant_params = [constants[name] for name in constant_parameter_names]
initial_state_params = [states[name] for name in state_parameter_names]
initial_params = initial_constant_params + initial_state_params

# Set bounds for parameters (optional)
parameter_bounds = [len(initial_params)*[0], len(initial_params)*[6]]
parameter_bounds[0][constant_parameter_names.index('FCepsilonRI/k_f1')] = 0.0009
parameter_bounds[1][constant_parameter_names.index('FCepsilonRI/k_f1')] = 101.16
parameter_bounds[0][constant_parameter_names.index('FCepsilonRI/k_r1')] = 0.001
parameter_bounds[1][constant_parameter_names.index('FCepsilonRI/k_r1')] = 18
parameter_bounds[0][constant_parameter_names.index('FCepsilonRI/k_f2')] = 0.00023
parameter_bounds[1][constant_parameter_names.index('FCepsilonRI/k_f2')] = 70
parameter_bounds[0][constant_parameter_names.index('FCepsilonRI/k_f3')] = 0.0009
parameter_bounds[1][constant_parameter_names.index('FCepsilonRI/k_f3')] = 101.16
parameter_bounds[0][constant_parameter_names.index('FCepsilonRI/k_r3')] = 0.001
parameter_bounds[1][constant_parameter_names.index('FCepsilonRI/k_r3')] = 18
parameter_bounds[0][constant_parameter_names.index('FCepsilonRI/k_f4')] = 0.00023
parameter_bounds[1][constant_parameter_names.index('FCepsilonRI/k_f4')] = 70
parameter_bounds[0][constant_parameter_names.index('FCepsilonRI/k_f5')] = 0.0009
parameter_bounds[1][constant_parameter_names.index('FCepsilonRI/k_f5')] = 101.16
parameter_bounds[0][constant_parameter_names.index('FCepsilonRI/k_r5')] = 0.001
parameter_bounds[1][constant_parameter_names.index('FCepsilonRI/k_r5')] = 18
#parameter_bounds[0][constant_parameter_names.index('FCepsilonRI/k_f6')] = 0.00023
#parameter_bounds[1][constant_parameter_names.index('FCepsilonRI/k_f6')] = 70
parameter_bounds[0][state_parameter_names.index('FCepsilonRI/Grb2')] = 0
parameter_bounds[1][state_parameter_names.index('FCepsilonRI/Grb2')] = 100

parameter_bounds = tuple(parameter_bounds)

'''
initial_params=np.zeros((9,1))
'''
'''
#random
for i in range(0,9):
 initial_params[i,0] = (parameter_bounds[1][i]-parameter_bounds[0][i])*np.random.rand(1)+parameter_bounds[0][i]
'''
'''
#upperbound
for i in range(0,8):
 initial_params[i,0] = parameter_bounds[1][i]
 '''
'''
#lowerbound
for i in range(0,9):
 initial_params[i,0] = parameter_bounds[0][i]
'''
'''
initial_params=initial_params[:,0]
print (initial_params)
'''

# Run the simulation using given parameter values and return the
# values of the state variable for which we have experiment data
def model_function_lsq(params, expt_time, expt_pFC, expt_pSyk, return_type, debug=False):
    if debug:
        print('Parameters:')
        print(params)

    simulation.resetParameters()    
    for n, v in enumerate(params[0:len(constant_parameter_names)]):
        constants[constant_parameter_names[n]] = v
    for n, v in enumerate(params[len(constant_parameter_names):len(state_parameter_names)+len(constant_parameter_names)]):
        states[state_parameter_names[n]] = v

    try:
        simulation.run()
    except RuntimeError:
        print("Runtime error:")
        for n, v in enumerate(params[0:len(constant_parameter_names)]):
            print('  {}: {}'.format(constant_parameter_names[n], v))
        for n, v in enumerate(params[len(constant_parameter_names):len(state_parameter_names)+len(constant_parameter_names)]):
            print('  {}: {}'.format(state_parameter_names[n], v))
        raise

    if return_type == 'optimisation':
        pSyk = (results.states()[expt_state_uri_pSyk].values()[expt_time])
        pFC = (results.states()[expt_state_uri_pFC].values()[expt_time])
        f1 = (pSyk-expt_pSyk)
        f2 = (pFC-expt_pFC)
        #print (f1, f2)
        f = np.concatenate((f1,f2))
        print('SSD:')    
        print(sum(f**2))
    elif return_type == 'visualisation':
        f1 = results.states()[expt_state_uri_pSyk].values()[expt_time]
        f2 = results.states()[expt_state_uri_pFC].values()[expt_time]
        f = np.vstack((f1,f2))
    return f

opt =least_squares(model_function_lsq, initial_params, args=(expt_time, expt_pSyk, expt_pFC, 'optimisation'),
                               bounds=parameter_bounds, ftol=1e-14, xtol=1e-14, gtol=1e-14, verbose=1)

opt_constant_parameters = opt.x[0:len(constant_parameter_names)]
opt_state_parameters = opt.x[len(constant_parameter_names):len(state_parameter_names)+len(constant_parameter_names)]

# Print fitted parameters
print (opt_state_parameters)
print('Optimal constant parameters:')
for n, v in enumerate(opt_constant_parameters):
    print('  {}: {:g} ({:g})'.format(constant_parameter_names[n], v, initial_constant_params[n]))
    
print('Optimal state parameters:')

for n, v in enumerate(opt_state_parameters):
    print('  {}: {:g} ({:g})'.format(state_parameter_names[n], v, initial_state_params[n]))
    
f =model_function_lsq(opt.x, expt_time, expt_pSyk, expt_pFC,'visualisation', debug=True)

# Plot graphs
fig, ax = plt.subplots()
plt.plot(expt_time, expt_pSyk, 'o', label='Experiment pSyk', color='red')
plt.plot(expt_time, f[0], '-', label='Model pSyk', color='blue')
pSyk_error = f[0] - expt_pSyk
print(np.mean(pSyk_error))
print(np.std(pSyk_error))
print(max(expt_pSyk))
print(max(pSyk_error))
print(min(pSyk_error))
fig.canvas.draw()
plt.show()

fig, ax = plt.subplots()
plt.plot(expt_time, expt_pFC, 'o', label='Experiment pFC', color='red')
plt.plot(expt_time, f[1], '-', label='Model pFC', color='blue')
pFC_error = f[1] - expt_pFC
print(np.mean(pFC_error))
print(np.std(pFC_error))
print(max(expt_pFC))
print(max(pFC_error))
print(min(pFC_error))
fig.canvas.draw()
plt.show()
















