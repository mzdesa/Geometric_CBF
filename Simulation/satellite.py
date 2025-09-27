"""
This file contains code for running an underactuated satellite simulation on SO(3)
"""
import numpy as np
import CalSim as cs
import scipy as sp
from .dynamics import *
import matplotlib as mpl
import matplotlib.pyplot as plt

#Plotting and animation tools
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

class Satellite(Dynamics):
    """
    Dynamics of an underactuated satellite
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
        self.Ic = self.contr.Ic #store inertia tensor
        self.IcInv = np.linalg.inv(self.Ic) #compute its inverse
        self.I = np.eye(3) #identity matrix
        self.e1 = np.array([[1, 0, 0]]).T
        self.e2 = np.array([[0, 1, 0]]).T
        self.e3 = np.array([[0, 0, 1]]).T
        self.e3Hat = cs.hat(self.e3)
        self.Pa = self.I - self.e3 @ self.e3.T #projection in lie algebra onto actuation distribution

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
        # unpack R and Omega from x
        Rk, Omegak = x

        # Euler integrate in lie algebra
        Rkp1 = Rk @ sp.linalg.expm(cs.hat(Omegak) * self.dt)
        Omegakp1 = Omegak + self.dt * (self.IcInv @ (cs.hat(self.Ic @ Omegak) @ Omegak + u))

        # Package as a list and return
        return [Rkp1, Omegakp1]
    
    def plot(self):
        """
        Plotting function
        """
        try:
            #Set matplotlib params to use LateX
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
            #Extract rotations from configuration history
            RHIST = [x[0] for x in self.xHIST]
            OMEGAHIST = [x[1] for x in self.xHIST]

            #Extract X, Y, Z from rotation history - compute Re3 for the projection onto the sphere
            qHIST = [R @ self.e3 for R in RHIST] #gives a list of configurations on the sphere
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
            fig.patch.set_facecolor('white') 

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
                
                # phi(t) --- Let's mess with the offset a little bit
                phi = self.theta_d + self.contr.amp * np.sin(self.contr.freq * t + self.contr.theta0) #self.contr.theta0
                
                # curve r(t)
                xd = np.cos(t) * np.sin(phi)
                yd = np.sin(t) * np.sin(phi)
                zd = np.cos(phi)
                ax.plot(xd, yd, zd, "b:")

            # Plot trajectory
            ax.plot(x, y, z, color = 'red')
            ax.scatter(x[0], y[0], z[0], color='green', s=50, label="Start")

            # Formatting
            ax.set_title(r'Projected Trajectory on $\mathbb{S}^2$', fontsize=24, pad = 0, y = 1)
            ax.set_xlabel(r'$x$', fontsize=18); ax.set_ylabel(r'$y$', fontsize=18); ax.set_zlabel(r'$z$', fontsize=18)
            ax.set_box_aspect([1,1,1])
            ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
            ax.grid(False)
            # ax.grid(True, alpha=0.01)

            plt.tight_layout()
            plt.show()

            # Plot the top-down view; only plot the x and y of the above.
            stereoProj = True
            if stereoProj:
                #Plot the stereographic projection
                def stereoX(x, z):
                    """
                    Stereographic projection on the sphere (south pole deleted, See Lee p.30)
                    """
                    return x/(1+z)
                def stereoY(y, z):
                    return y/(1+z)         
                
                fig, ax = plt.subplots(figsize=(6,6))

                #Project the z = 0 slice to stereo coords
                th = np.linspace(0, 2*np.pi, 400)
                sphereXStereo, sphereYStereo = [stereoX(np.cos(theta), 0) for theta in th], [stereoY(np.sin(theta), 0) for theta in th]
                ax.fill(sphereXStereo, sphereYStereo, color='lightgray', alpha=0.4) # S^1 boundary

                #Project the boundary of the safe region to stereo coords
                R = np.sin(self.theta_d)
                xB, yB, zB = [R * np.cos(theta) for theta in th], [R * np.sin(theta) for theta in th], np.cos(self.theta_d)
                xbS, ybS = [stereoX(xbi, zB) for xbi in xB], [stereoY(ybi, zB) for ybi in yB]
                ax.fill(xbS, ybS, color = 'blue', alpha = 0.3)
                ax.plot(xbS, ybS, 'k')

                #Convert desired trajectory to stereo coords
                xdS, ydS = [stereoX(xd[i], zd[i]) for i in range(len(xd))], [stereoY(yd[i], zd[i]) for i in range(len(yd))]
                ax.plot(xdS, ydS, "b:") # desired trajectory projection

                #Convert trajectory to stereo coords
                xS, yS = [stereoX(x[i], z[i]) for i in range(len(x))], [stereoY(y[i], z[i]) for i in range(len(y))]
                ax.plot(xS, yS, color = 'red') # trajectory projection
                ax.scatter(stereoX(x[0], z[0]), stereoY(y[0], z[0]), color='green', s=50, label="Start")
                ax.set_aspect('equal', 'box')
                ax.set_xlim([-1.05, 1.05]); ax.set_ylim([-1.05, 1.05])
                ax.set_title(r'Stereographic Projection on $\mathbb S^2$', fontsize=24, pad = 5)
                ax.set_xlabel(r'$x$', fontsize=18); ax.set_ylabel(r'$y$', fontsize=18)
                ax.grid(True, alpha=0.2)
                plt.show()

            else:
                #Plot a direct top-down view
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
                ax.set_title(r'Top-Down View of Safe Trajectory on the Sphere', fontsize=18, pad = 5)
                ax.set_xlabel(r'$x$', fontsize=18); ax.set_ylabel(r'$y$', fontsize=18)
                ax.grid(True, alpha=0.2)
                plt.show()


            # Plot values of h0, h, and u
            h0HIST = [self.contr.eval_h0(R) for R in RHIST]
            hHIST = [self.contr.eval_h(RHIST[i], OMEGAHIST[i]) for i in range(len(RHIST))]
            tau1HIST, tau2HIST = [u[0, 0] for u in self.uHIST], [u[1, 0] for u in self.uHIST]

            # Plot on a 3x1 subplot
            plotInputs = True
            if plotInputs:
                fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, constrained_layout=True, figsize=(6,6)) 
                axes[0].plot(self.tHIST, h0HIST[:-1])     
                axes[0].plot(self.tHIST, hHIST[:-1])
                axes[1].plot(self.tHIST, tau1HIST)    
                axes[1].plot(self.tHIST, tau2HIST)

                #labels
                axes[0].set_title(r"State and Configuration Safety", fontsize=24, pad = 3)
                axes[0].legend([r'$h_0$', r'$h$'], fontsize=14, loc = 'upper right')
                axes[1].set_title(r"Inputs", fontsize=24, pad = 3)
                axes[1].legend([r'$\tau_1$', r'$\tau_2$'], fontsize=14, loc = 'upper right')
                axes[1].set_xlabel(r"$t$", fontsize=18)
                axes[0].grid(True, alpha=0.2)
                axes[1].grid(True, alpha=0.2)
                plt.show()
            else:
                #Only plot the barrier values
                fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, constrained_layout=True, figsize=(6,6)) 
                axes.plot(self.tHIST, h0HIST[:-1])     
                axes.plot(self.tHIST, hHIST[:-1])

                #labels
                axes.set_title(r"State and Configuration Safety", fontsize=24, pad = 3)
                axes.legend([r'$h_0$', r'$h$'], fontsize=14)
                axes.set_xlabel(r"$t$", fontsize=18)
                axes.grid(True, alpha=0.2)
                plt.show()


            #Plot frames at a set of times. 10 second simulation, plot 10 frames
             # ----- plot -----
            fig = plt.figure(figsize=(18,18), constrained_layout = False)
            ax = fig.add_subplot(111, projection='3d')
            

            #First, get the indices.
            numFr = 10
            frDeltaTime = self.tHIST[-1]/numFr #Time between frames
            RIndexList = [i * int(len(self.tHIST)/numFr) for i in range(numFr)]
            framelen = 0.5 #length of arms in the frame

            for i in range(numFr + 1):
                #Get the rotation matrix at that frame
                if i < numFr:
                    Ri = RHIST[RIndexList[i]]
                    #Get the time
                    ti = frDeltaTime * i
                else:
                    Ri = RHIST[-1]
                    ti = self.tHIST[-1]

                #Plot the frame at (ti, 0, 0) (frames move with time along the x-axis)
                basePt = np.array([[ti, 0, 0]]).T
                ax.scatter(basePt[0, 0], basePt[1, 0], basePt[2, 0], color = 'black')

                #Calculate each axis position
                axisX = basePt + framelen * Ri @ self.e1
                axisY = basePt + framelen * Ri @ self.e2
                axisZ = basePt + framelen * Ri @ self.e3

                #Plot the axes
                ax.plot([basePt[0, 0], axisX[0, 0]], [basePt[1, 0], axisX[1, 0]], [basePt[2, 0], axisX[2, 0]], color = 'red')
                ax.plot([basePt[0, 0], axisY[0, 0]], [basePt[1, 0], axisY[1, 0]], [basePt[2, 0], axisY[2, 0]], color = 'green')
                ax.plot([basePt[0, 0], axisZ[0, 0]], [basePt[1, 0], axisZ[1, 0]], [basePt[2, 0], axisZ[2, 0]], color = 'blue')


                #Plot the sphere and safe set
                ax.plot_surface(framelen * Xc + ti, framelen * Yc, framelen * Zc, color='blue', alpha=0.2, edgecolor='none')

            # Formatting
            ax.get_zaxis().set_ticklabels([]); ax.get_yaxis().set_ticklabels([])
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_aspect('equal', 'box')
            ax.grid(False)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.show()

    def animate(self):
        """
        Animate the satellite
        """
        if self.xHIST is None:
            print("No simulation available")
            return

        #Extract the state history
        RHIST = [x[0] for x in self.xHIST]

        #Set constant animtion parameters
        FREQ = int(1/self.dt) #frequency

        #initialize figure and a point
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        plt.axis('square')

        #plot the sphere and configuration barrier
        n_th, n_ph = 25, 25
        theta = np.linspace(0, np.pi, n_th)      # polar angle from +z
        phi   = np.linspace(0, 2*np.pi, n_ph)    # azimuth
        TH, PH = np.meshgrid(theta, phi, indexing="ij")

        X = np.sin(TH) * np.cos(PH)
        Y = np.sin(TH) * np.sin(PH)
        Z = np.cos(TH)

        # ----- barrier h0(q) = q·e3 - cos(theta_d) -----
        h0 = Z - np.cos(self.theta_d)
        mask_safe = h0 >= 0  # spherical cap: angle to e3 <= theta_d

        # Base sphere
        ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.3, edgecolor='none')

        # Safe set
        Xc = np.where(mask_safe, X, np.nan)
        Yc = np.where(mask_safe, Y, np.nan)
        Zc = np.where(mask_safe, Z, np.nan)
        ax.plot_surface(Xc, Yc, Zc, color='blue', alpha=0.4, edgecolor='none')

        # Boundary circle at polar angle theta_d
        phi_b = np.linspace(0, 2*np.pi, 100)
        xb = np.sin(self.theta_d) * np.cos(phi_b)
        yb = np.sin(self.theta_d) * np.sin(phi_b)
        zb = np.full_like(phi_b, np.cos(self.theta_d))
        ax.plot(xb, yb, zb, 'k', linewidth=2)

        #define points reprenting the three axes
        x, y, z = [0, 0, 0, 0],  [0, 0, 0, 0], [0, 0, 0, 0] #center, e1Tip, e2Tip, e3Tip
        points, = ax.plot(x, y, z, 'o')

        #define lines for each of the three axes
        lines = [ax.plot([], [], [], lw=2)[0] for _ in range(3)]

        # --- trail of Re3 ---
        trail_line, = ax.plot([], [], [], color='r', lw=2, alpha=0.9)  # plot a red line
        trail_x, trail_y, trail_z = [], [], []

        def update(num, data, lines):
            #get the rotation matrix and position at num
            R = data[num]

            #Get each of the lines
            e1Line = R @ self.e1
            e2Line = R @ self.e2
            e3Line = R @ self.e3
            origin = np.zeros((3, 1))

            #define a line between the points
            lines[0].set_data(np.hstack((origin, e1Line))[0:2, :])
            lines[0].set_3d_properties(np.hstack((origin, e1Line))[2, :])
            lines[1].set_data(np.hstack((origin, e2Line))[0:2, :])
            lines[1].set_3d_properties(np.hstack((origin, e2Line))[2, :])
            lines[2].set_data(np.hstack((origin, e3Line))[0:2, :])
            lines[2].set_3d_properties(np.hstack((origin, e3Line))[2, :])

            #define the x points to plot
            xPoints = [0, e1Line[0, 0],  e2Line[0, 0],  e3Line[0, 0]]
            yPoints = [0, e1Line[1, 0],  e2Line[1, 0],  e3Line[1, 0]]
            zPoints = [0, e1Line[2, 0],  e2Line[2, 0],  e3Line[2, 0]]

            #update point markers
            points.set_data(xPoints, yPoints)
            points.set_3d_properties(zPoints, 'z')

            # --- update Re3 trail ---
            trail_x.append(e3Line[0, 0])
            trail_y.append(e3Line[1, 0])
            trail_z.append(e3Line[2, 0])
            trail_line.set_data(trail_x, trail_y)
            trail_line.set_3d_properties(trail_z)

        # Setting the axes properties
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        #run animation
        num_frames = len(RHIST)-1
        anim = animation.FuncAnimation(fig, update,  num_frames, fargs=(RHIST, lines), interval=1/FREQ*1000, blit=False)

        # Formatting
        ax.set_title('Configuration Trajectory')
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_box_aspect([1,1,1])
        ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])

        #Show animation
        plt.show()
