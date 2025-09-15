"""
Geometric CBF, implemented for the simple point mass system $m \ddot q = F$.
Here, we use the position constraints $h_0(q) = 1 - q^2$, which keep the system in [-1, 1]
"""
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

#Define system parameters
m = 1
epsilon = 0.5

#Define simulation parameters
dt = 0.01
T = 10
N = int(T//dt) + 1

def eval_h0(q):
    """
    Configuration barrier
    """
    return 1 - q**2

def eval_h(q, v):
    """
    Full-state barrier
    """
    return eval_h0(q) - m/(2 * epsilon) * (v + 2/m * q)**2

def eval_alpha(r):
    """
    Class K function
    """
    return 4/m * r

def eval_Lfh(q, v):
    """
    Return the drift term of the Lie derivative
    """
    return -2 * q * v - 2/epsilon * (v + 2/m * q) * v

def eval_Lgh(q, v):
    """
    Return the control term of the Lie derivative
    """
    return -1/epsilon * (v + 2/m * q)

def eval_input(q, v, OD = False):
    """
    Evaluate the input using the closed-form CBF-QP solution
    """
    #Evaluate the CBF terms
    h0, h, Lfh, Lgh = eval_h0(q), eval_h(q, v), eval_Lfh(q, v), eval_Lgh(q, v)
    alpha = eval_alpha(h)

    if OD == False:
        #Solve the standard CBF-QP
        F = cp.Variable(1)
        cons = [Lfh + Lgh*F >= -alpha]
        prob = cp.Problem(cp.Minimize(cp.sum_squares(F)), cons)
        prob.solve(solver=cp.OSQP)
        return float(F.value)
    else:
        #Solve the OD-CBF-QP
        p = 1
        F = cp.Variable(1)
        omega = cp.Variable(1)
        cons = [Lfh + Lgh*F >= -omega * alpha, omega >= 0]
        prob = cp.Problem(cp.Minimize(cp.sum_squares(F) + p * cp.sum_squares(omega)), cons)
        prob.solve(solver=cp.OSQP)
        return float(F.value)


    
def simulate(q0=0.9, v0=1.0):
    """
    Run the simulation
    """
    t = np.arange(N+1) * dt
    q = np.zeros(N+1)
    v = np.zeros(N+1)
    u = np.zeros(N+1)
    h = np.zeros(N+1)

    q[0], v[0] = q0, v0
    h[0] = eval_h(q[0], v[0])
    u[0] = eval_input(q[0], v[0])

    for k in range(N):
        # Control
        u[k] = eval_input(q[k], v[k])

        # Euler integrate
        q[k+1] = q[k] + v[k] * dt
        v[k+1] = v[k] + (u[k]/m) * dt
        
        # Evaluate barrier
        h[k+1] = eval_h(q[k+1], v[k+1])

    return t, q, v, u, h

# ---- Run & Plot ----
t, q, v, u, h = simulate(q0 = 0.9, v0 = 0.2)

fig, axs = plt.subplots(3)
fig.suptitle('Configuration, Velocity, and CBF')
axs[0].plot(t, q)
axs[0].set_ylabel('q')
axs[1].plot(t, v)
axs[1].set_ylabel('v')
axs[2].plot(t, h)
axs[2].set_ylabel('h')
plt.show()

# Quick textual checks
print(f"min h(t) over simulation: {h.min():.4f}")
print(f"max |q(t)| over simulation: {np.max(np.abs(q)):.4f}")

# Grid for plotting
q_min, q_max = -1.2, 1.2
v_min, v_max = -4.0, 4.0
Nq, Nv = 600, 600

qLin = np.linspace(q_min, q_max, Nq)
vLin = np.linspace(v_min, v_max, Nv)
Q, V = np.meshgrid(qLin, vLin, indexing='xy')

H = eval_h(Q, V)

# Plot the zero contour
fig, ax = plt.subplots(figsize=(6, 5))
cs = ax.contour(Q, V, H, levels=[0.0], linewidths=2)
ax.plot(q, v)
ax.set_title(r"Zero Contour of $h(q,v)=0$")
ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$v$")
ax.set_xlim(q_min, q_max)
ax.set_ylim(v_min, v_max)
ax.grid(True, alpha=0.3)

# Optional: draw vertical lines for the configuration bounds |q| ≤ 1
ax.axvline( 1.0, linestyle="--", linewidth=1, alpha=0.6)
ax.axvline(-1.0, linestyle="--", linewidth=1, alpha=0.6)

plt.tight_layout()
plt.show()

