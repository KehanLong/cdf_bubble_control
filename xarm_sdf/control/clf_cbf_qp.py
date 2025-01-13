import numpy as np
import casadi as ca

class ClfCbfQpController:
    def __init__(self, p1=1e0, p2=1e3, clf_rate=1.0, cbf_rate=0.2, safety_margin=0.1):
        self.p1 = p1  # Control effort penalty
        self.p2 = p2  # CLF slack variable penalty
        self.clf_rate = clf_rate
        self.cbf_rate = cbf_rate
        self.safety_margin = safety_margin
        self.prev_u = None
        
        # Setup the optimization solver
        self.setup_solver()
        
    def setup_solver(self):
        # Decision variables
        u = ca.SX.sym('u', 6)  # Control inputs (assuming 6-DOF arm)
        delta = ca.SX.sym('delta')  # CLF slack variable
        
        # Parameters
        current_config = ca.SX.sym('current_config', 6)
        reference_config = ca.SX.sym('reference_config', 6)
        h = ca.SX.sym('h')  # CBF value
        dh_dtheta = ca.SX.sym('dh_dtheta', 6)  # CBF gradient
        
        # CLF computation
        error = current_config - reference_config
        V = 0.5 * ca.dot(error, error)
        dV_dtheta = error
        
        # Constraints
        g = []
        lbg = []
        ubg = []
        
        # CLF constraint with slack
        g.append(ca.dot(dV_dtheta, u) + self.clf_rate * V - delta)
        lbg.append(-ca.inf)
        ubg.append(0)
        
        # CBF constraint
        g.append(ca.dot(dh_dtheta, u) + self.cbf_rate * h)
        lbg.append(0)
        ubg.append(ca.inf)
        
        # Input bounds
        g.append(u)
        lbg.extend([-2.0] * 6)
        ubg.extend([2.0] * 6)
        
        # Objective function
        obj = self.p1 * ca.sumsqr(u) + self.p2 * delta**2
        
        # Create solver
        opts = {
            'print_time': 0,
            'ipopt': {
                'print_level': 0,
                'max_iter': 100,
                'tol': 1e-3,
                'acceptable_tol': 1e-3
            }
        }
        
        nlp = {
            'x': ca.vertcat(u, delta),
            'p': ca.vertcat(current_config, reference_config, h, dh_dtheta),
            'f': obj,
            'g': ca.vertcat(*g)
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.lbg = lbg
        self.ubg = ubg
        
    def generate_controller(self, current_config, reference_config, h, dh_dtheta):
        num_links = len(current_config)
        if self.prev_u is None:
            self.prev_u = np.zeros(num_links)
            
        try:
            # Pack parameters
            p = np.concatenate([
                current_config,
                reference_config,
                [h],
                dh_dtheta
            ])
            
            # Initial guess
            x0 = np.zeros(num_links + 1)  # +1 for delta
            
            # Solve optimization
            sol = self.solver(
                x0=x0,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p
            )
            
            # Extract solution
            x_opt = sol['x'].full().flatten()
            u_new = x_opt[:num_links]
            
            # Add diagnostic prints for CBF
            print("\n=== CBF Diagnostics ===")
            print(f"CBF value (h): {h:.4f}")
            print(f"CBF gradient (dh_dtheta): {dh_dtheta}")
            
            # Calculate and print the safety constraint value
            cbf_constraint = np.dot(dh_dtheta, u_new) + self.cbf_rate * h
            print(f"Safety constraint (should be ≥ 0): {cbf_constraint:.4f}")
            print(f"Is safety constraint satisfied? {cbf_constraint >= 0}")
            print("=====================\n")
            
            self.prev_u = u_new
            return u_new
            
        except Exception as e:
            print(f"[CLF-CBF-QP] Solver failed: {str(e)}")
            return np.zeros(num_links) 