"""
Implement a lightweight dynamics skeleton class
"""
import numpy as np

class Dynamics:
    def __init__(self, contr, dt = 0.001, T = 10):
        """
        Inputs:
            contr (controller)
            dt (float): time step for integration
            T (float): time for simulation
        """
        #store controller
        self.contr = contr

        #store simulation parameters
        self.dt = dt
        self.T = T
        self.N = int(self.T//self.dt) + 1  

        #store simulation parameters
        self.tHIST = np.arange(self.N) * self.dt
        self.xHIST = []
        self.uHIST = []

    def step(self, x, u):
        """
        Step the system dynamics via integration method of your choice.
        Inputs:
            x (state)
            u (input)
        Returns:
            x+ = f(x, u) (state at t + dt)
        """
        pass

    def run_sim(self, x0):
        """
        Run the simulation with an initial condition x0
        """
        # Reset xHIST with x0 and uHIST as empty
        self.xHIST = [x0]
        self.uHIST = []

        # Run the simulation
        for k in range(self.N):
            # Control
            self.uHIST.append(self.contr.eval_input(self.xHIST[-1], self.tHIST[k]))

            # Step dynamics
            self.xHIST.append(self.step(self.xHIST[-1], self.uHIST[-1]))

        return self.tHIST, self.xHIST, self.uHIST
    
    def plot(self):
        """
        Plotting function
        """
        pass