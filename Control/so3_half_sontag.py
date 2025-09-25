"""
Implement a sphere CBF controller with a gradient flow-based CBF
"""
import numpy as np
import CalSim as cs

class SO3HalfSontag:
    """
    Half-Sontag controller on SO(3)
    """
    def __init__(self, Ic, epsilon = 0.1, theta_d = np.pi/3, amp = 0.2, freq = 5, alpha = 5):
        """
        Sphere gradient flow controller
        """
        #Store physical parameters
        self.Ic = Ic
        self.tildeI = self.Ic[0, 0]
        self.e3 = np.array([[0, 0, 1]]).T
        self.I = np.eye(3)
        self.e1 = np.array([[1, 0, 0]]).T
        self.e2 = np.array([[0, 1, 0]]).T
        self.e3 = np.array([[0, 0, 1]]).T
        self.e3Hat = cs.hat(self.e3)
    
        #Store CBF parameters
        self.theta_d = theta_d
        self.epsilon = epsilon
        self.alpha = alpha #Class K function constant

        #Store desired trajectory parameters
        self.amp = amp
        self.freq = freq

        #Store pseudoinverse of projection matrix for tau calculation
        self.projInv = np.array([[0, -1, 0],
                            [1, 0, 0]])
        self.P = np.eye(3) - self.e3 @ self.e3.T        

    def eval_qd(self, t):
        """
        Sinusoidal trajectory on S2 that oscillates about the safe set.
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

    def geom_PD_S2(self, q, v, mu, t, kp = 10, kd = 5):
        """
        Evaluate the tracking control input on S2, using the geometric PD controller of GCMS
        32 and 12 is strong tracking
        Here, we add an extra momentum term to compensate for the quotient
        """
        qd = self.eval_qd(t)
        qdDot = self.eval_qd_dot(t)
        qdDDot = self.eval_qd_ddot(t)

        #Calculate the acceleration of the desired trajectory in the round metric
        nablaRnd = qdDDot + np.linalg.norm(qdDot)**2 * qd

        #calculate the PD and ff terms
        Fpd = -kp * cs.hat(q) @ cs.hat(q) @ qd - kd * (v - cs.hat(cs.hat(qd) @ qdDot) @ q)
        Fff = mu * cs.hat(q) @ v + self.tildeI * (q.T @ cs.hat(qd) @ qdDot)[0, 0] * (cs.hat(q) @ v) + self.tildeI * cs.hat(cs.hat(qd) @ nablaRnd) @ q 
        return Fpd + Fff
    
    def eval_kPD(self, R, Omega, t):
        """
        Evaluate the nominal tracking controller
        """
        #Compute q and v for the sphere
        q = R @ self.e3
        v = R @ cs.hat(Omega) @ self.e3

        #Compute the (conserved) angular momentum
        mu = (self.e3.T @ self.Ic @ Omega)[0, 0]

        #compute the geometric PD control
        Fpd = self.geom_PD_S2(q, v, mu, t)

        #Now, solve for tau1 and tau2
        TAU12 = self.projInv @ R.T @ Fpd
        return np.array([[TAU12[0, 0], TAU12[1, 0], 0]]).T

    def eval_h0(self, R):
        """
        Configuration barrier
        """
        return (self.e3.T @ R @ self.e3)[0, 0] - np.cos(self.theta_d)
    
    def eval_ab(self, R):
        """
        Evaluate a and b functions for half-sontag
        """
        a = self.eval_alpha(self.eval_h0(R))
        b = 1 / self.tildeI * (self.e3.T @ R @ self.P @ R.T @ self.e3)[0, 0] 
        return a, b
    
    def eval_gradh0(self, R):
        """
        Evaluate the Riemannian gradient of h0 on SO(3)
        """
        return 1 / self.tildeI * R @ cs.hat(self.e3Hat @ R.T @ self.e3)
    
    def eval_lamHS(self, R):
        """
        Evaluate lambda(a(R), b(R))
        """
        #evaluate a and b and gradient
        a, b = self.eval_ab(R)

        #evaluate lambdaHS(a, b)
        if abs(b) < 1e-12:
            return 0
        else:
            return (-a + np.sqrt(a**2 + b**2)) / (2 * b)
    
    def eval_HS(self, R):
        """
        Evaluate the half-sontag controller on SO(3)
        """
        return self.eval_lamHS(R) * self.eval_gradh0(R)
    
    def eval_grad_lam(self, R):
        """
        Evaluate dlamHS/da and dlamHS/db 
        """
        #evaluate a and b
        a, b = self.eval_ab(R)

        #evaluate lambda
        p = self.eval_lamHS(R)

        #calculate the derivative terms (see Max Cohen's paper)
        dFdp = 2 * b * p + a
        dFda = p
        dFdb = p**2 - 1/4

        #calculate and return derivatives (dLam/da and dLam/db)
        return -1/dFdp * dFda, -1/dFdp * dFdb
    
    def eval_kappaDot(self, R, Omega):
        """
        Find the derivative of the half-sontag controller in the Lie algebra;
        i.e. evaluate d/dt (R.T @ kappaHS)
        """
        #find lambdaHS(a(R), b(R))
        lamHS = self.eval_lamHS(R)

        #Calculate the derivative of lambdaHS
        aDot = self.alpha * Omega.T @ self.e3Hat @ R.T @ self.e3
        bDot = 1/self.tildeI * (self.e3.T @ R @ (cs.hat(Omega) @ self.P - self.P @ cs.hat(Omega)) @ R.T @ self.e3)[0, 0]
        dLamda, dLamdb = self.eval_grad_lam(R)
        lamHSDot = dLamda * aDot + dLamdb * bDot

        #Calculate d/dt(R.T @ kappaHS) (derivative in Lie algebra)
        kappaHSLieDot = (-1/self.tildeI * self.e3Hat @ cs.hat(Omega) @ R.T @ self.e3)

        #evaluate and return kappaDot
        return lamHSDot * R.T @ self.eval_gradh0(R) + lamHS * kappaHSLieDot
    
    def eval_eA(self, R, Omega):
        """
        Evaluate projected error in the Lie algebra under isomorhpism with R3
        """
        return self.P @ (Omega - cs.vee_3d(R.T @ self.eval_HS(R)))

    def eval_h(self, R, Omega):
        """
        Full-state barrier
        """
        eA = self.eval_eA(R, Omega)
        return (self.eval_h0(R) - self.epsilon/2 * eA.T @ self.Ic @ eA)[0, 0]

    def eval_alpha(self, r):
        """
        Class K function
        """
        return self.alpha * r

    def eval_Lfh(self, R, Omega):
        """
        Return the drift term of the Lie derivative
        """
        eA = self.eval_eA(R, Omega)
        kappaDot = self.eval_kappaDot(R, Omega)
        return (Omega.T @ self.e3Hat @ R.T @ self.e3 - self.epsilon * eA.T @ (self.tildeI * kappaDot - cs.hat(Omega) @ self.Ic @ Omega))[0, 0]

    def eval_Lgh(self, R, Omega):
        """
        Return the control term of the Lie derivative
        """
        eA = self.eval_eA(R, Omega)
        return -self.epsilon * eA.T 

    def eval_input(self, x, t):
        """
        Evaluate the input using the closed-form CBF-QP solution
        """
        #Unpack R, Omega from state
        R, Omega = x

        #Evaluate the CBF terms
        h0, h, Lfh, Lgh = self.eval_h0(R), self.eval_h(R, Omega), self.eval_Lfh(R, Omega), self.eval_Lgh(R, Omega)
        alpha = self.eval_alpha(h)

        #Evaluate the desired input
        kDes = self.eval_kPD(R, Omega, t)
        
        #use the analytical solution to solve
        a = Lfh + alpha + Lgh @ kDes
        b = np.linalg.norm(Lgh)**2
        if np.linalg.norm(Lgh) < 1e-12:
            lam = 0
        else:
            lam = max(0, -a/b)
        
        #return input
        return kDes + Lgh.T * lam