"""
Run a sphere simulation
"""
from Simulation.satellite import *
from Control.so3_grad_flow import *
from Control.so3_half_sontag import *

#Define sphere parameters
Ic = np.diag((0.1, 0.1, 0.3))
epsilon = 0.02 # CBF constant
alpha = 5 # Class K function constant

# Choose controller
# contr = SO3GradFlow(Ic, epsilon=epsilon) # Gradient flow velocity controller
contr = SO3HalfSontag(Ic, epsilon = epsilon, alpha = alpha) # Half-Sontag velocity controller

# Initializize a simulation object
sat = Satellite(contr)

#Run the simulation
R0 = cs.calc_Ry(np.pi/3.5) #@ cs.calc_Rx(-np.pi/6)  #np.eye(3) #
Omega0 = np.zeros((3, 1))
sat.run_sim([R0, Omega0])
sat.plot()
# sat.animate()