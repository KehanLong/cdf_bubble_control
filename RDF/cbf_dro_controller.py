import numpy as np
import cvxpy as cp

class CbfDroController:
    def __init__(self, p1=1e0, cbf_rate=3.0, wasserstein_r=0.005, epsilon=0.1):
        # Optimizer parameters
        self.p1 = p1
        self.rateh = cbf_rate
        
        # DRO parameters
        self.wasserstein_r = wasserstein_r
        self.epsilon = epsilon
        
        # Solver status
        self.solve_fail = True
        self.prev_u = None

    def generate_control(self, nominal_u, cbf_h_samples, cbf_h_grad_samples, cbf_t_grad_samples):
        """
        Safety filter using DRO CBF-QP
        Args:
            nominal_u: nominal control input
            cbf_h_samples: array of CBF values from sampled points
            cbf_h_grad_samples: array of CBF gradients from sampled points
            cbf_t_grad_samples: array of time derivatives from sampled points
        Returns:
            Safe control input
        """
        num_joints = len(nominal_u)
        N = len(cbf_h_samples)  # Number of samples
        
        if self.prev_u is None:
            self.prev_u = np.zeros(num_joints)

        # Control input and DRO variables
        u = cp.Variable(num_joints)
        si = cp.Variable(N)
        t = cp.Variable(1)

        slack = cp.Variable(1)

        # Reshape for DRO constraints
        one_vector = cp.reshape(np.array([1]), (1, 1), order='C')
        rateh_vector = cp.reshape(np.array([self.rateh]), (1, 1), order='C')
        u_reshaped = cp.reshape(u, (num_joints, 1), order='C')
        stacked_vector = cp.vstack([one_vector, rateh_vector, u_reshaped])

        # Create q_samples for DRO
        q_samples = [
            np.hstack([
                cbf_t_grad_samples[i], 
                cbf_h_samples[i],  # Safety margin of 0.05
                cbf_h_grad_samples[i, :]
            ]) 
            for i in range(N)
        ]

        # DRO CBF constraints
        constraints = [
            self.wasserstein_r * cp.abs(stacked_vector) / self.epsilon <= 
                (t - (1/N) * cp.sum(si) / self.epsilon) * np.ones((num_joints + 2, 1)),
            si >= 0,
            cp.abs(u) <= 3.0,  # Velocity limits
            slack >= 0.0,
        ]
        
        # Add individual sample constraints
        for i in range(N):
            constraints.append(
                si[i] >= t - stacked_vector.T @ q_samples[i] - slack
            )

        # Objective: Minimize deviation from nominal control
        # obj = cp.Minimize(cp.norm(nominal_u - u) ** 2 + self.p1 * cp.norm(self.prev_u - u) ** 2)

        obj = cp.Minimize(cp.norm(nominal_u - u) ** 2 + 1e3 * slack)

        # Solve QP
        prob = cp.Problem(obj, constraints)
        
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)

            if prob.status == "optimal" or prob.status == "optimal_inaccurate":
                print('cbf_h_samples:', cbf_h_samples)
                print('slack:', slack.value)
                

            if prob.status == "optimal" or prob.status == "optimal_inaccurate":
                self.solve_fail = False
                self.prev_u = u.value
                return u.value
            else:
                self.solve_fail = True
                print("-------------------------- SOLVER NOT OPTIMAL -------------------------")
                print("[In solver] solver status: ", prob.status)
                print("[In solver] CBF samples = ", cbf_h_samples)
                print("[In solver] CBF gradients = ", cbf_h_grad_samples)
                print("[In solver] CBF time derivatives = ", cbf_t_grad_samples)
                self.prev_u = nominal_u
                return nominal_u

        except cp.error.SolverError:
            self.solve_fail = True
            print("-------------------------- SOLVER ERROR -------------------------")
            self.prev_u = nominal_u
            return nominal_u 