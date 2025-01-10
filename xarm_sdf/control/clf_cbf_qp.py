import numpy as np
import cvxpy as cp

class ClfCbfQpController:
    def __init__(self, p1=1e0, p2=1e3, clf_rate=1.0, cbf_rate=1.0, safety_margin=0.1):
        # Control effort and slack variable penalties
        self.p1 = p1  # Control effort penalty
        self.p2 = p2  # CLF slack variable penalty
        # CLF and CBF decay rates
        self.clf_rate = clf_rate
        self.cbf_rate = cbf_rate
        # Safety margin for CBF
        self.safety_margin = safety_margin
        # Previous control
        self.prev_u = None

    def generate_controller(self, current_config, reference_config, h, dh_dtheta):
        """
        Generate control input using CLF-CBF-QP.
        
        Args:
            current_config: Current robot configuration (numpy array)
            reference_config: Reference configuration from governor (numpy array)
            h: CBF value (float)
            dh_dtheta: CBF gradient (numpy array)
        """
        num_links = len(current_config)
        if self.prev_u is None:
            self.prev_u = np.zeros(num_links)
        
        # CLF based on tracking error
        error = current_config - reference_config
        V = 0.5 * np.dot(error, error)
        dV_dtheta = error  # Gradient of quadratic V
        
        # Control input and CLF slack variable
        u = cp.Variable(num_links)
        delta = cp.Variable(1)
        
        # CLF-CBF-QP constraints
        constraints = [
            # CLF condition with slack
            dV_dtheta @ u + self.clf_rate * V <= delta,
            
            # CBF condition
            dh_dtheta @ u + self.cbf_rate * h >= 0,
            
            # Input bounds
            cp.abs(u) <= 2.0
        ]

        # Minimize control effort and CLF slack
        obj = cp.Minimize(self.p2 * cp.norm(delta) ** 2)
        prob = cp.Problem(obj, constraints)

        try:
            # Solve QP
            prob.solve(solver='SCS', verbose=False)

            if prob.status == "optimal" or prob.status == "optimal_inaccurate":
                self.prev_u = u.value
                return u.value
            else:
                print(f"[CLF-CBF-QP] Solver status: {prob.status}")
                self.prev_u = np.zeros(num_links)
                return np.zeros(num_links)
                
        except Exception as e:
            print(f"[CLF-CBF-QP] Solver failed: {str(e)}")
            self.prev_u = np.zeros(num_links)
            return np.zeros(num_links) 