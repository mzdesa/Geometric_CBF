"""
Run a sphere simulation
"""
from Simulation.satellite import *
from Control.so3_grad_flow import *

#Define sphere parameters
Ic = np.diag((0.1, 0.1, 0.3))
epsilon = 0.02
contr = SO3GradFlow(Ic, epsilon=epsilon)
sat = Satellite(contr)

#Run the simulation
R0 = np.eye(3)
Omega0 = np.zeros((3, 1))
sat.run_sim([R0, Omega0])
sat.plot()
sat.animate()