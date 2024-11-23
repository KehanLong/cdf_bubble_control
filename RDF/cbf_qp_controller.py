import numpy as np
import cvxpy as cp

class CbfQpController:
    def __init__(self, p1=1e0, cbf_rate=1.0):
        # Control Lipschitz penalty
        self.p1 = p1
        # CBF decay rate
        self.rateh = cbf_rate
        
        # Solver status
        self.solve_fail = True
        # Previous control
        self.prev_u = None

    def generate_control(self, nominal_u, cbf_h_val, cbf_h_grad, cbf_t_grad):
        """
        Safety filter using CBF-QP
        Args:
            nominal_u: nominal control input 
            cbf_h_val: CBF value
            cbf_h_grad: gradient of CBF w.r.t. robot configuration
            cbf_t_grad: time derivative of CBF
        Returns:
            Safe control input
        """
        num_joints = len(nominal_u)

        
        if self.prev_u is None:
            self.prev_u = np.zeros(num_joints)

        # Control input variable
        u = cp.Variable(num_joints)

        slack = cp.Variable(1)

        adaptive_rate = self.rateh * (1 + np.exp(-cbf_h_val))

        # CBF constraint: hdot + α(h) ≥ 0
        constraints = [
            cbf_h_grad @ u + cbf_t_grad + adaptive_rate * cbf_h_val >= -slack,  # Safety margin 0.05
            slack >= 0.0,
            cp.abs(u) <= 2.0  # Velocity limits
        ]

        # Objective: Minimize deviation from nominal control
        # obj = cp.Minimize(cp.norm(nominal_u - u) ** 2 + self.p1 * cp.norm(self.prev_u - u) ** 2)
        obj = cp.Minimize(cp.norm(nominal_u - u) ** 2 + 1e3 * slack)

        # Solve QP
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.CLARABEL, verbose=False)
        

        if prob.status == "optimal" or prob.status == "optimal_inaccurate":
            print('cbf_h_val:', cbf_h_val)
            print('cbf_t_grad:', cbf_t_grad)
            print('cbf_constraints:', cbf_h_grad @ u.value + cbf_t_grad + adaptive_rate * cbf_h_val)

        # Check solution
        if prob.status != "optimal":
            self.solve_fail = True
            print("CBF-QP Solver failed:")
            print('cbf_h_val', cbf_h_val)
            print('cbf_h_grad', cbf_h_grad)
            self.prev_u = nominal_u  # Use nominal control if solver fails
            return nominal_u
        else:
            self.solve_fail = False
            self.prev_u = u.value
            return u.value 