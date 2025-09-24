"""
Run a sphere simulation
"""
from Simulation.sphere import *
from Control.sphere_grad_flow import *

#Define sphere parameters
m = 1
g = 0
epsilon = 0.09
contr = S2GradFlowCBF(m, g, epsilon=epsilon)
sphere = Sphere(contr)

#Run the simulation
q0 = np.array([[0, 0, 1]]).T
v0 = (np.eye(3) - q0 @ q0.T) @ np.random.uniform(-1, 1, (3, 1)) * 2
sphere.run_sim([q0, v0])
sphere.plot()