"""
Implement a sphere CBF controller with a gradient flow-based CBF
"""
import numpy as np
import CalSim as cs

class S2GradFlowCBF:
    """
    Gradient flow CBF on the 2-sphere
    """
    def __init__(self, m, g, epsilon = 0.1, theta_d = np.pi/3, amp = 0.2, freq = 5):
        """
        Sphere gradient flow controller
        """
        #Store physical parameters
        self.m = m
        self.g = g
        self.e3 = np.array([[0, 0, 1]]).T
        self.I = np.eye(3)
    
        #Store CBF parameters
        self.theta_d = theta_d
        self.epsilon = epsilon

        #Store desired trajectory parameters
        self.amp = amp
        self.freq = freq

    def eval_qd(self, t):
        """
        Sinusoidal trajectory that oscillates about the safe set.
        """
        # phi(t)
        phi = self.theta_d + self.amp * np.sin(self.freq * t)
        
        # curve r(t)
        x = np.cos(t) * np.sin(phi)
        y = np.sin(t) * np.sin(phi)
        z = np.cos(phi)
        return np.array([[x, y, z]]).T

    def eval_qd_dot(self, t):
        """
        Return the time derivative of qd(t)
        """
        phi = self.theta_d + self.amp * np.sin(self.freq * t)
        phiDot = self.amp * self.freq * np.cos(self.freq * t)
        xDot = -np.sin(t) * np.sin(phi) + np.cos(t) * np.cos(phi) * phiDot
        yDot = np.cos(t) * np.sin(phi) + np.sin(t) * np.cos(phi) * phiDot
        zDot = -np.sin(phi) * phiDot
        return np.array([[xDot, yDot, zDot]]).T

    def eval_qd_ddot(self, t):
        """
        Return the second time derivative of qd(t) on S^2.
        qd(t) = (cos t sin phi(t), sin t sin phi(t), cos phi(t)),
        phi(t) = self.theta_d + self.amp * sin(self.freq * t).
        """
        phi = self.theta_d + self.amp * np.sin(self.freq * t)
        phiDot = self.amp * self.freq * np.cos(self.freq * t)
        phiDDot = -self.amp * self.freq**2 * np.sin(self.freq * t)
        
        xDDot = (-np.cos(t) * np.sin(phi)
                - 2*np.sin(t) * np.cos(phi) * phiDot
                + np.cos(t) * (-np.sin(phi) * phiDot**2 + np.cos(phi) * phiDDot))
        
        yDDot = (-np.sin(t) * np.sin(phi)
                + 2*np.cos(t) * np.cos(phi) * phiDot
                + np.sin(t) * (-np.sin(phi) * phiDot**2 + np.cos(phi) * phiDDot))
        
        zDDot = -np.cos(phi) * phiDot**2 - np.sin(phi) * phiDDot
        
        return np.array([[xDDot, yDDot, zDDot]]).T


    def k_des(self, q, v, t, kp = 32, kd = 12):
        """
        Evaluate the tracking control input.
        Uses GCMS controller (pg. 245)
        """
        qd = self.eval_qd(t)
        qdDot = self.eval_qd_dot(t)
        qdDDot = self.eval_qd_ddot(t)

        #Calculate the acceleration of the desired trajectory in the round metric
        nablaRnd = qdDDot + np.linalg.norm(qdDot)**2 * qd

        #calculate the PD and ff terms
        Fpd = -kp * cs.hat(q) @ cs.hat(q) @ qd - kd * (v - cs.hat(cs.hat(qd) @ qdDot) @ q)
        Fff = self.m * (q.T @ cs.hat(qd) @ qdDot)[0, 0] * (cs.hat(q) @ v) + self.m * cs.hat(cs.hat(qd) @ nablaRnd) @ q 
        return Fpd + Fff

    def eval_h0(self, q):
        """
        Configuration barrier
        """
        return (q.T @ self.e3)[0, 0] - np.cos(self.theta_d)

    def eval_h(self, q, v):
        """
        Full-state barrier
        """
        return self.eval_h0(q) - (self.m * self.epsilon / 2) * np.linalg.norm(v - 1/self.m * (self.I - q @ q.T) @ self.e3)**2

    def eval_alpha(self, r):
        """
        Class K function
        """
        return 2*abs(np.cos(self.theta_d))/self.m * r

    def eval_Lfh(self, q, v):
        """
        Return the drift term of the Lie derivative
        """
        e = v - 1/self.m * (self.I - q @ q.T) @ self.e3
        return self.epsilon * (v.T @ self.e3 - (q.T @ self.e3) * e.T @ v + self.epsilon * (self.m * self.g) * e.T @ self.e3)[0, 0]


    def eval_Lgh(self, q, v):
        """
        Return the control term of the Lie derivative
        """
        e = v - 1/self.m * (self.I - q @ q.T) @ self.e3
        return -self.epsilon * e.T

    def eval_input(self, x, t):
        """
        Evaluate the input using the closed-form CBF-QP solution
        """
        #Unpack q and v from state
        q, v = x

        #Evaluate the CBF terms
        h0, h, Lfh, Lgh = self.eval_h0(q), self.eval_h(q, v), self.eval_Lfh(q, v), self.eval_Lgh(q, v)
        alpha = self.eval_alpha(h)

        #Evaluate the desired input
        kDes = self.k_des(q, v, t)
        
        #use the analytical solution to solve
        a = Lfh + alpha + Lgh @ kDes
        b = np.linalg.norm(Lgh)**2
        if np.linalg.norm(Lgh) < 1e-14:
            lam = 0
        else:
            lam = max(0, -a/b)
        
        #Reproject and return
        return (self.I - q @ q.T) @ (kDes + Lgh.T * lam)