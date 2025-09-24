import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import scipy as sp
import CalSim as cs

#Define system parameters
m = 1 #mass of pendulum
g = 0 #gravity
epsilon = 50 #CBF tuning parameter
theta_d = np.pi/4 #safe angle from vertical
e3 = np.array([[0, 0, 1]]).T
I = np.eye(3)

#first iteration flag
firstIter = True
prob = None
f = None

#Define simulation parameters
dt = 0.001
T = 10
N = int(T//dt) + 1

#Set desired trajectory parameters
amp = 0.2
freq = 5

def eval_qd(t):
    """
    Sinusoidal trajectory that oscillates about the safe set.
    """
    # phi(t)
    phi = theta_d + amp * np.sin(freq * t)
    
    # curve r(t)
    x = np.cos(t) * np.sin(phi)
    y = np.sin(t) * np.sin(phi)
    z = np.cos(phi)
    return np.array([[x, y, z]]).T

def eval_qd_dot(t):
    """
    Return the time derivative of qd(t)
    """
    phi = theta_d + amp * np.sin(freq * t)
    phiDot = amp * freq * np.cos(freq * t)
    xDot = -np.sin(t) * np.sin(phi) + np.cos(t) * np.cos(phi) * phiDot
    yDot = np.cos(t) * np.sin(phi) + np.sin(t) * np.cos(phi) * phiDot
    zDot = -np.sin(phi) * phiDot
    return np.array([[xDot, yDot, zDot]]).T

def eval_qd_ddot(t):
    """
    Return the second time derivative of qd(t) on S^2.
    qd(t) = (cos t sin phi(t), sin t sin phi(t), cos phi(t)),
    phi(t) = theta_d + amp * sin(freq * t).
    """
    phi = theta_d + amp * np.sin(freq * t)
    phiDot = amp * freq * np.cos(freq * t)
    phiDDot = -amp * freq**2 * np.sin(freq * t)
    
    xDDot = (-np.cos(t) * np.sin(phi)
             - 2*np.sin(t) * np.cos(phi) * phiDot
             + np.cos(t) * (-np.sin(phi) * phiDot**2 + np.cos(phi) * phiDDot))
    
    yDDot = (-np.sin(t) * np.sin(phi)
             + 2*np.cos(t) * np.cos(phi) * phiDot
             + np.sin(t) * (-np.sin(phi) * phiDot**2 + np.cos(phi) * phiDDot))
    
    zDDot = -np.cos(phi) * phiDot**2 - np.sin(phi) * phiDDot
    
    return np.array([[xDDot, yDDot, zDDot]]).T


def k_des(q, v, t, kp = 32, kd = 12):
    """
    Evaluate the tracking control input.
    Uses GCMS controller (pg. 245)
    """
    qd = eval_qd(t)
    qdDot = eval_qd_dot(t)
    qdDDot = eval_qd_ddot(t)
    nablaRnd = qdDDot + np.linalg.norm(qdDot)**2 * qd

    #calculate the PD and ff terms
    Fpd = -kp * cs.hat(q) @ cs.hat(q) @ qd - kd * (v - cs.hat(cs.hat(qd) @ qdDot) @ q)
    Fff = m * (q.T @ cs.hat(qd) @ qdDot)[0, 0] * (cs.hat(q) @ v) + m * cs.hat(cs.hat(qd) @ nablaRnd) @ q 
    return Fpd + Fff

def eval_h0(q):
    """
    Configuration barrier
    """
    return (q.T @ e3)[0, 0] - np.cos(theta_d)

def eval_h(q, v):
    """
    Full-state barrier
    """
    return eval_h0(q) - m/(2 * epsilon) * np.linalg.norm(v - 1/m * (I - q @ q.T) @ e3)**2

def eval_alpha(r):
    """
    Class K function
    """
    return 2*abs(np.cos(theta_d))/m * r

def eval_Lfh(q, v):
    """
    Return the drift term of the Lie derivative
    """
    e = v - 1/m * (I - q @ q.T) @ e3
    return (v.T @ e3 - (q.T @ e3)/epsilon * e.T @ v + (m * g) / epsilon * e.T @ e3)[0, 0]


def eval_Lgh(q, v):
    """
    Return the control term of the Lie derivative
    """
    e = v - 1/m * (I - q @ q.T) @ e3
    return -1/epsilon * e.T

def eval_input(q, v, t, OSQP = False, OD = True):
    """
    Evaluate the input using the closed-form CBF-QP solution
    """
    #Evaluate the CBF terms
    h0, h, Lfh, Lgh = eval_h0(q), eval_h(q, v), eval_Lfh(q, v), eval_Lgh(q, v)
    alpha = eval_alpha(h)

    #Evaluate the desired input
    kDes = k_des(q, v, t)
    
    #use the analytical solution to solve
    a = Lfh + alpha + Lgh @ kDes
    b = np.linalg.norm(Lgh)**2
    if np.linalg.norm(Lgh) < 1e-12:
        lam = 0
    else:
        lam = max(0, -a/b)
    
    #Reproject and return
    return (I - q @ q.T) @ (kDes + Lgh.T * lam)

def simulate(q0 = e3, v0 = np.zeros((3, 1))):
    """
    Run the simulation
    """
    t = np.arange(N) * dt
    q = [q0]
    v = [v0]
    u = []
    h = [eval_h(q[0], v[0])]

    for k in range(N):
        # Control
        u.append(eval_input(q[-1], v[-1], t[k]))

        # Euler integrate
        qk, vk = q[-1], v[-1]
        qkp1 = qk + vk * dt
        vkp1 = vk + dt * (-np.linalg.norm(vk)**2 * qk - g * (I - qk @ qk.T) @ e3 + 1/m * u[-1])

        # Project to sphere
        q.append(qkp1/np.linalg.norm(qkp1))
        v.append((I - qk @ qk.T) @ vkp1)
        
        # Evaluate barrier
        h.append(eval_h(q[-1], v[-1]))

    return t, q, v, u, h


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

    # time grid
    t = np.linspace(0, T, 500)
    
    # phi(t)
    phi = theta_d + amp * np.sin(freq * t)
    
    # curve r(t)
    x = np.cos(t) * np.sin(phi)
    y = np.sin(t) * np.sin(phi)
    z = np.cos(phi)
    ax.plot(x, y, z, "b:")

    # Formatting
    ax.set_title('Trajectory on the Sphere')
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect([1,1,1])
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
    ax.grid(False)

    plt.tight_layout()
    plt.show()


#Run Simulation
q0 = e3 #np.random.uniform(-1, 1, (3, 1)) #cs.calc_Rx(np.pi/4) @ e3
q0 = q0/np.linalg.norm(q0)
v0 = (I - q0 @ q0.T) @ np.random.uniform(-1, 1, (3, 1)) * 2 #(I - q0 @ q0.T) @ np.array([[5, -1, -1]]).T

t, q, v, u, h = simulate(q0, v0)
x = [qk[0, 0] for qk in q]
y = [qk[1, 0] for qk in q]
z = [qk[2, 0] for qk in q]
plot_sphere(x, y, z)

# #plot barrier
# plt.plot(t, h[:-1])
# plt.xlabel("Time")
# plt.ylabel("h")
# plt.show()