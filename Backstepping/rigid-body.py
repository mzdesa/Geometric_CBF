import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import CalSim as cs
import scipy as sp

#Define CBF parameters
epsilon = 10 #CBF tuning parameter
theta_d = np.pi/4 #safe angle from vertical
e3 = np.array([[0, 0, 1]]).T
e3Hat = cs.hat(e3)

#Inertia Parameters
Ic = np.diag((0.1, 0.1, 0.3))
IcInv = np.linalg.inv(Ic)
Itilde = Ic[0, 0]
I = np.eye(3)

#first iteration flag
firstIter = True
prob = None
f = None

#Define simulation parameters
dt = 0.001
T = 10
N = int(T//dt) + 1

def eval_h0(R):
    """
    Configuration barrier
    """
    return (e3.T @ R @ e3)[0, 0] - np.cos(theta_d)

def eval_h(R, Omega):
    """
    Full-state barrier
    """
    e = Omega - IcInv @ e3Hat @ R.T @ e3 #vee of velocity error in the Lie algebra
    return eval_h0(R) - 1/(2 * epsilon) * (e.T @ Ic @ e)[0, 0]

def eval_alpha(r):
    """
    Class K function
    """
    return 2*abs(np.cos(theta_d))/Itilde * r

def eval_Lfh(R, Omega):
    """
    Return the drift term of the Lie derivative
    """
    e = Omega - IcInv @ e3Hat @ R.T @ e3 #vee of velocity error in the Lie algebra
    return (Omega.T @ e3Hat @ R.T @ e3 - (1/epsilon) * e.T @ e3Hat @ cs.hat(Omega) @ R.T @ e3 - (1/epsilon) * e.T @ (cs.hat(Ic @ Omega) @ Omega))[0, 0]

def eval_Lgh(R, Omega):
    """
    Return the control term of the Lie derivative
    """
    e = Omega - IcInv @ e3Hat @ R.T @ e3
    return -1/epsilon * e.T

def eval_input(R, Omega, OSQP = False, OD = True):
    """
    Evaluate the input using the closed-form CBF-QP solution
    """
    #Evaluate the CBF terms
    h0, h, Lfh, Lgh = eval_h0(R), eval_h(R, Omega), eval_Lfh(R, Omega), eval_Lgh(R, Omega)
    alpha = eval_alpha(h)

    if OSQP:
        #Solve the standard CBF-QP in OSQP - update this later to do problem parameter updates
        if OD == False:
            #Solve the standard CBF-QP
            tau = cp.Variable((3, 1))
            cons = [Lfh + Lgh @ tau >= -alpha] #add CBF constraint
            prob = cp.Problem(cp.Minimize(cp.sum_squares(tau)), cons)
            prob.solve(solver=cp.OSQP)
            return (tau.value).reshape((3, 1))
        else:
            #Solve the OD-CBF-QP
            tau = cp.Variable((3, 1))
            omega = cp.Variable(1)
            cons = [Lfh + Lgh @ tau >= -omega * alpha, omega >= 0] #add CBF and tangent space constraints
            prob = cp.Problem(cp.Minimize(cp.sum_squares(tau) + omega**2), cons)
            prob.solve(solver=cp.OSQP)
            return (tau.value).reshape((3, 1))
    else:
        #use the analytical solution
        a = Lfh + alpha
        b = np.linalg.norm(Lgh)**2
        if np.linalg.norm(Lgh) < 1e-12:
            lam = 0
        else:
            lam = max(0, -a/b)
        return Lgh.T * lam

def simulate(R0 = np.eye(3), Omega0 = np.zeros((3, 1))):
    """
    Run the simulation
    """
    t = np.arange(N) * dt
    R = [R0]
    Omega = [Omega0]
    q = [R0 @ e3] #keep track of e3 column for plotting
    u = []
    h = [eval_h(R[0], Omega[0])]

    for k in range(N):
        # Control
        u.append(eval_input(R[-1], Omega[-1]))

        # Euler integrate in Lie Algebra
        Rk, Omegak = R[-1], Omega[-1]
        Rkp1 = Rk @ sp.linalg.expm(cs.hat(Omegak) * dt)
        Omegakp1 = Omegak + dt * (IcInv @ (cs.hat(Ic @ Omegak) @ Omegak + u[-1]))

        # Project to sphere
        R.append(Rkp1)
        Omega.append(Omegakp1)
        q.append(Rkp1 @ e3)
        
        # Evaluate barrier
        h.append(eval_h(R[-1], Omega[-1]))

    return t, R, Omega, q, u, h


def plot_sphere(x, y, z):
    n_th, n_ph = 25, 25
    theta = np.linspace(0, np.pi, n_th)      # polar angle from +z
    phi   = np.linspace(0, 2*np.pi, n_ph)    # azimuth
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    X = np.sin(TH) * np.cos(PH)
    Y = np.sin(TH) * np.sin(PH)
    Z = np.cos(TH)

    # ----- barrier h0(q) = q·e3 - cos(theta_d) -----
    h0 = Z - np.cos(theta_d)
    mask_safe = h0 >= 0  # spherical cap: angle to e3 <= theta_d

    # ----- plot -----
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Base sphere
    ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.3, edgecolor='none')

    # Safe set
    Xc = np.where(mask_safe, X, np.nan)
    Yc = np.where(mask_safe, Y, np.nan)
    Zc = np.where(mask_safe, Z, np.nan)
    ax.plot_surface(Xc, Yc, Zc, color='blue', alpha=0.4, edgecolor='none')

    # Boundary circle at polar angle theta_d
    phi_b = np.linspace(0, 2*np.pi, 100)
    xb = np.sin(theta_d) * np.cos(phi_b)
    yb = np.sin(theta_d) * np.sin(phi_b)
    zb = np.full_like(phi_b, np.cos(theta_d))
    ax.plot(xb, yb, zb, 'k', linewidth=2)

    # Plot trajectory
    ax.plot(x, y, z, color = 'red')
    ax.scatter(x[0], y[0], z[0], color='green', s=50, label="Start")
    ax.scatter(x[-1], y[-1], z[-1], color='black', s=50, label="End")

    # Formatting
    ax.set_title('Trajectory on the Sphere')
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect([1,1,1])
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
    ax.grid(False)

    plt.tight_layout()
    plt.show()


#Run Simulation
R0 = cs.calc_Rx(np.pi * np.random.rand()) @ cs.calc_Ry(np.pi * np.random.rand()) #cs.calc_Rx(np.pi/4)
Omega0 = np.random.uniform(-1, 1, (3, 1)) * 2

t, R, Omega, q, u, h = simulate(R0, Omega0)
x = [qk[0, 0] for qk in q]
y = [qk[1, 0] for qk in q]
z = [qk[2, 0] for qk in q]
plot_sphere(x, y, z)

#plot barrier
plt.plot(t, h[:-1])
plt.xlabel("Time")
plt.ylabel("h")
plt.show()