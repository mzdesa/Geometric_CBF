"""
This file contains code for running a sphere simulation on S2.
"""
import numpy as np
import CalSim as cs
from .dynamics import *
import matplotlib as mpl
import matplotlib.pyplot as plt

class Sphere(Dynamics):
    """
    Dynamics of fully actuated spherical pendulum.
    """
    def __init__(self, contr, dt = 0.001, T = 10):
        """
        Inputs:
            contr (controller)
            m (float): mass of the sphere in KG 
            dt (float): simulation timestep
            T (float): simulation time
        State x = [q, v] (List of numpy arrays)
        """
        #Call the super init function
        super().__init__(contr = contr, dt = dt, T=T)

        #Store other system parameters
        self.m = self.contr.m
        self.g = self.contr.g #gravitational acceleration
        self.I = np.eye(3)
        self.e3 = np.array([[0, 0, 1]]).T

        #Get a desired theta from the controller object (may be none)
        try:
            self.theta_d = self.contr.theta_d
        except:
            self.theta_d = None

    def step(self, x, u):
        """
        Step the system dynamics via integration method of your choice.
        Inputs:
            x (state)
            u (input)
        Returns:
            x+ = f(x, u) (state at t + dt)
        """
        #unpack configuration and velocity from x
        qk, vk = x
        qkp1 = qk + vk * self.dt
        vkp1 = vk + self.dt * (-np.linalg.norm(vk)**2 * qk - self.g * (self.I - qk @ qk.T) @ self.e3 + 1/self.m * u)

        # Project to sphere and return
        return [qkp1/np.linalg.norm(qkp1), (self.I - qk @ qk.T) @ vkp1]
    
    def plot(self):
        """
        Plotting function
        """
        try:
            #Set matplotlib params
            mpl.rcParams['text.usetex'] = True
            mpl.rcParams['font.family'] = 'serif'
            mpl.rcParams['text.latex.preamble'] =  r"""
                                                    \usepackage{amsmath}
                                                    \usepackage{amssymb}
                                                    \usepackage{bm}
                                                    """
        except:
            print("LaTeX Not Found")

        #Check if the simulation has been run
        if len(self.xHIST) == 0:
            print("No simulation found")
            return
        else:
            #Extract X, Y, Z from configuration history
            qHIST = [x[0] for x in self.xHIST] #gives a list of configurations
            x = [qk[0, 0] for qk in qHIST]
            y = [qk[1, 0] for qk in qHIST]
            z = [qk[2, 0] for qk in qHIST]

            n_th, n_ph = 25, 25
            theta = np.linspace(0, np.pi, n_th)      # polar angle from +z
            phi   = np.linspace(0, 2*np.pi, n_ph)    # azimuth
            TH, PH = np.meshgrid(theta, phi, indexing="ij")

            X = np.sin(TH) * np.cos(PH)
            Y = np.sin(TH) * np.sin(PH)
            Z = np.cos(TH)

             # ----- plot -----
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection='3d')

            # Base sphere
            ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.3, edgecolor='none')

            #If theta_d is not none, plot the safe set
            if self.theta_d is not None:
                # ----- barrier h0(q) = q·e3 - cos(theta_d) -----
                h0 = Z - np.cos(self.theta_d)
                mask_safe = h0 >= 0  # spherical cap: angle to e3 <= theta_d

                # Safe set
                Xc = np.where(mask_safe, X, np.nan)
                Yc = np.where(mask_safe, Y, np.nan)
                Zc = np.where(mask_safe, Z, np.nan)
                ax.plot_surface(Xc, Yc, Zc, color='blue', alpha=0.2, edgecolor='none')

                # Boundary circle at polar angle theta_d
                phi_b = np.linspace(0, 2*np.pi, 100)
                xb = np.sin(self.theta_d) * np.cos(phi_b)
                yb = np.sin(self.theta_d) * np.sin(phi_b)
                zb = np.full_like(phi_b, np.cos(self.theta_d))
                ax.plot(xb, yb, zb, 'k', linewidth=2)

            #If there is a desired trajectory in the controller, plot it
            if self.contr.amp is not None:
                # time grid
                t = np.linspace(0, self.T, 500)
                
                # phi(t)
                phi = self.theta_d + self.contr.amp * np.sin(self.contr.freq * t)
                
                # curve r(t)
                xd = np.cos(t) * np.sin(phi)
                yd = np.sin(t) * np.sin(phi)
                zd = np.cos(phi)
                ax.plot(xd, yd, zd, "b:")

            # Plot trajectory
            ax.plot(x, y, z, color = 'red')
            ax.scatter(x[0], y[0], z[0], color='green', s=50, label="Start")

            # Formatting
            ax.set_title(r'Trajectory on the Sphere $\mathbb{S}^2$', fontsize=18, pad = 2)
            ax.set_xlabel(r'$x$', fontsize=18); ax.set_ylabel(r'$y$', fontsize=18); ax.set_zlabel(r'$z$', fontsize=18)
            ax.set_box_aspect([1,1,1])
            ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
            ax.grid(False)

            plt.tight_layout()
            plt.show()

            # Plot the top-down view; only plot the x and y of the above.
            fig, ax = plt.subplots(figsize=(6,6))
            th = np.linspace(0, 2*np.pi, 400)
            
            #plot sphere and safe set
            ax.fill(np.cos(th), np.sin(th), color='lightgray', alpha=0.4) # S^1 boundary

            # Cap boundary and fill: radius = sin(theta_d)
            R = np.sin(self.theta_d)
            xb, yb = R*np.cos(th), R*np.sin(th)
            ax.fill(xb, yb, color = 'blue', alpha = 0.3)
            ax.plot(xb, yb, 'k')

            ax.plot(xd, yd, "b:") # desired trajectory projection
            ax.plot(x, y, color = 'red') # trajectory projection
            ax.scatter(x[0], y[0], color='green', s=50, label="Start")
            
            ax.set_aspect('equal', 'box')
            ax.set_xlim([-1.05, 1.05]); ax.set_ylim([-1.05, 1.05])
            ax.set_title(r'Top-Down View of Safe Trajectory on the Sphere', fontsize=18, pad = 3)
            ax.set_xlabel(r'$x$', fontsize=18); ax.set_ylabel(r'$y$', fontsize=18)
            ax.grid(True, alpha=0.2)
            plt.show()
