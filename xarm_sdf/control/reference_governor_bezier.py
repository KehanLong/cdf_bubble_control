import numpy as np

class BezierReferenceGovernor:
    def __init__(self, initial_state, trajectory_data, dt):
        """
        Initialize Bezier Reference Governor.
        
        Args:
            initial_state: Initial robot state
            trajectory_data: Dictionary containing:
                - bezier_curves: List of Bezier curve segments
                - times: Time parameterization
            dt: Time step
        """
        self.dt = dt
        self.s = 0.0  # Progress variable
        self.z = initial_state
        
        # Store Bezier curves and timing
        self.bezier_curves = trajectory_data['bezier_curves']
        self.num_segments = len(self.bezier_curves)
        
        # Parameters for governor dynamics
        self.k = 0.5  # Gain term
        self.zeta = 15  # Power term in dynamics

    def get_segment_and_local_s(self, s):
        """Convert global s ∈ [0,1] to segment index and local s"""
        segment_idx = min(int(s * self.num_segments), self.num_segments - 1)
        local_s = (s * self.num_segments) - segment_idx
        local_s = np.clip(local_s, 0, 1)
        return segment_idx, local_s

    def update(self, z):
        """
        Update reference governor state using the original Bezier curves
        """
        self.z = z
        
        # Get current segment and local parameter
        segment_idx, local_s = self.get_segment_and_local_s(self.s)
        current_curve = self.bezier_curves[segment_idx]
        
        # Query current position and velocity using original Bezier curve
        gamma_s = current_curve.query(local_s).value
        
        # Get derivative curve and evaluate
        derivative_curve = current_curve.derivative()
        gamma_s_dot = derivative_curve.query(local_s).value
        
        # Compute distance to current reference point
        distance_to_ref = np.linalg.norm(z - gamma_s)
        
        # Reference governor dynamics
        s_dot = (self.k / (1 + distance_to_ref)) * (1 - self.s**self.zeta)
        
        # Update progress variable
        self.s += self.dt * s_dot
        self.s = min(1.0, max(0.0, self.s))
        
        # Get reference point and its derivative for output
        segment_idx, local_s = self.get_segment_and_local_s(self.s)
        current_curve = self.bezier_curves[segment_idx]
        
        reference_pos = current_curve.query(local_s).value
        derivative_curve = current_curve.derivative()
        reference_vel = derivative_curve.query(local_s).value * s_dot
        
        return reference_pos, self.s, reference_vel 