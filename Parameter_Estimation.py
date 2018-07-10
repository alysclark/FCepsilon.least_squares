import numpy as np

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, minimize,least_squares

import OpenCOR as oc

# Experiment data 
times = np.array([0,  480, 960, 1920, 3840])
pFC = np.array([0.0, 0.0408, 0.136, 0.105, 0.136])*.474
pSyk = np.array([0.0,  0.05437, 0.0644, 0.0518, 0.04373])*.474

plt.close('all')


# The state variable in the model that the data represents
expt_state_uri_pFC = 'FCepsilonRI/pFC'
expt_state_uri_pSyk = 'FCepsilonRI/pSyk'

#Logical - do you want to fit parameters and state variables
fit_parameters = True
fit_state_parameter = False

#List of parameters you want to exclude from fit
fit_parameters_exclude = ['FCepsilonRI/pLyn']
#List of state variables you want to exclude from fit
fit_state_parameter_exclude = []

bounds_dictionary = {'FCepsilonRI/k_f1': [-5,2], 'FCepsilonRI/k_f2': [-5,2],'FCepsilonRI/k_f3': [-5,2], 'FCepsilonRI/k_f4': [-5,2],'FCepsilonRI/K_1': [-5,2], 'FCepsilonRI/K_2': [-5,2],'FCepsilonRI/K_3': [-5,2],
	'FCepsilonRI/K_4': [-5,2], 'FCepsilonRI/K_5': [-5,2],'FCepsilonRI/K_6': [-5,2], 'FCepsilonRI/K_7': [-5,2],'FCepsilonRI/K_8': [-5,2], 'FCepsilonRI/K_9': [-5,2],'FCepsilonRI/K_10': [-5,2], 'FCepsilonRI/K_11': [-5,2],
	'FCepsilonRI/K_12': [-5,2],'FCepsilonRI/V_1': [-5,2],'FCepsilonRI/V_2': [-5,2],'FCepsilonRI/V_3': [-5,2],'FCepsilonRI/V_4': [-5,2], 'FCepsilonRI/pLyn': [-1.32640456,-1.32640455]}

# Load and initialise the simulation
simulation = oc.openSimulation('FCepsilonRI.cellml')

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
initial_data.setEndingPoint(3900)
initial_data.setPointInterval(1)

# Specify as two parallel lists:
# 1. Uri's of parameters to estimate
# 2. Initial values of parameters
constant_parameter_names = list(constants.keys())
state_parameter_names = list(states.keys())

print(len(fit_parameters_exclude))

#If there are any parameters yoy don't want to fit you want to not fit exclude them
if fit_parameters:
    for i in range(0,len(fit_parameters_exclude)):
        print(fit_parameters_exclude[i])
        constant_parameter_names.remove(fit_parameters_exclude[i])
		
#If there are any parameters yoy don't want to fit you want to not fit exclude them
if fit_state_parameter:
    for i in range(0,len(fit_state_parameter_exclude)):
        state_parameter_names.remove(fit_state_parameter_exclude[i])

# Initialise the parameters that are going to be fixed
if fit_parameters:
    initial_constant_params = [constants[name] for name in constant_parameter_names]
else:
    initial_constant_params = []
if fit_state_parameter:
    initial_state_params = [states[name] for name in state_parameter_names]
else:
    initial_state_params = []
initial_params = initial_constant_params + initial_state_params
print(initial_params)

# # Set bounds for parameters (optional)
parameter_bounds = [len(initial_params)*[0], len(initial_params)*[6]]
for i in range(0,len(initial_params)):
    parameter_bounds[0][i] = 10**bounds_dictionary[constant_parameter_names[i]][0]
    parameter_bounds[1][i] = 10**bounds_dictionary[constant_parameter_names[i]][1]

parameter_bounds = tuple(parameter_bounds)

# Run the simulation using given parameter values and return the
# values of the state variable for which we have experiment data
def model_function_lsq(params, expt_time, expt_pSyk, expt_pGRB2, return_type, debug=False):
    if debug:
        print('Parameters:')
        print(params)

    simulation.resetParameters()
    simulation.clearResults()
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
        f1 = results.states()[expt_state_uri_pSyk].values()[times]-pSyk
        f2 = results.states()[expt_state_uri_pFC].values()[times]-pFC
        f = np.concatenate((f1,f2))
        print('SSD:')
        print(sum(f**2))
    elif return_type == 'visualisation':
        f1 = results.states()[expt_state_uri_pSyk].values()[times]
        f2 = results.states()[expt_state_uri_pFC].values()[times]
        f = np.vstack((f1,f2))
    return f

#minimize(model_function_lsq, initial_params, args=(expt_time,expt_pSyk,expt_pGRB2), bounds=np.transpose(np.array(parameter_bounds)), method='SLSQP',options={'xtol': 1e-8, 'disp': True})
# least_squares(model_function_lsq, initial_params, args=(expt_time,expt_pSyk,expt_pGRB2),
#              bounds=parameter_bounds, xtol=1e-5,verbose=1)

opt =least_squares(model_function_lsq, initial_params, args=(times,pFC,pSyk, 'optimisation'),
                               bounds=parameter_bounds,xtol=1e-6,verbose=1)


opt_constant_parameters = opt.x[0:len(constant_parameter_names)]
opt_state_parameters = opt.x[len(constant_parameter_names):len(state_parameter_names)+len(constant_parameter_names)]
print('Constant parameters:')
for n, v in enumerate(opt_constant_parameters):
    print('  {}: {:g} ({:g})'.format(constant_parameter_names[n], v, initial_constant_params[n]))
if fit_state_parameter:
    print('State parameters:')
    for n, v in enumerate(opt_state_parameters):
        print('  {}: {:g} ({:g})'.format(state_parameter_names[n], v, initial_state_params[n]))

f =model_function_lsq(opt.x, times, pSyk, pFC,'visualisation', debug=True)

fig, ax = plt.subplots()
plt.plot(times, pSyk, 'o', label='Experiment pSyk', color='red')
plt.plot(times, f[0], '-', label='Model pSyk', color='blue')
pSyk_error = f[0] - pSyk

print('pSyk error = ' + str(np.mean(pSyk_error)) + ' (SD ' + str(np.std(pSyk_error)) +')')


fig.canvas.draw()
plt.show()
fig, ax = plt.subplots()
plt.plot(times, pFC, 'o', label='Experiment pFC', color='red')
plt.plot(times, f[1], '-', label='Model pFC', color='blue')
pFC_error = f[1] - pFC
print('pFC error = ' + str(np.mean(pFC_error)) + ' (SD ' + str(np.std(pFC_error)) +')')
fig.canvas.draw()
plt.show()
#














