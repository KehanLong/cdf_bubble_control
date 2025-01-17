import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap


@jit
def forward_kinematics_jax(q, link_lengths):
    x = jnp.zeros(len(link_lengths) + 1)
    y = jnp.zeros(len(link_lengths) + 1)
    current_angle = 0
    for i in range(len(link_lengths)):
        current_angle += q[i]
        x = x.at[i+1].set(x[i] + link_lengths[i] * jnp.cos(current_angle))
        y = y.at[i+1].set(y[i] + link_lengths[i] * jnp.sin(current_angle))
    return x, y

@jit
def forward_kinematics(q):
    # Initialize position
    x, y = 0.0, 0.0
    current_angle = 0.0
    link_length = 2.0  # Fixed link length

    # Compute the end-effector position
    for angle in q:
        current_angle += angle
        x += link_length * jnp.cos(current_angle)
        y += link_length * jnp.sin(current_angle)

    return jnp.array([x, y])

@jit
def point_to_segment_distance_jax(p, a, b):
    ab = b - a
    ap = p - a
    proj = jnp.dot(ap, ab) / jnp.dot(ab, ab)
    proj = jnp.clip(proj, 0, 1)
    closest = a + proj * ab
    return jnp.linalg.norm(p - closest)

@jax.jit
def point_to_segment_distance_jax(points, segment_start, segment_end):
    """Calculate the minimum distance from points to a line segment using JAX.
    
    Args:
        points: Array of shape (N, 2) containing obstacle points
        segment_start: Array of shape (2,) containing segment start point
        segment_end: Array of shape (2,) containing segment end point
    
    Returns:
        Array of shape (N,) containing distances from each point to the segment
    """
    segment = segment_end - segment_start
    length_sq = jnp.sum(segment**2)
    
    t = jnp.clip(
        jnp.sum((points - segment_start) * segment, axis=1) / length_sq,
        0, 1
    )
    
    # Calculate projection points
    # Shape: (N, 2)
    projections = segment_start[None, :] + t[:, None] * segment[None, :]
    
    # Calculate distances
    # Shape: (N,)
    distances = jnp.linalg.norm(points - projections, axis=1)
    
    return distances

@jax.jit
def compute_robot_distances(configurations, obstacle_points):
    """Compute minimum distances from obstacles to robot segments.
    
    Args:
        configurations: Array of shape (B, 2) or (2,) containing joint angles
        obstacle_points: Array of shape (N, 2) containing obstacle points
    
    Returns:
        Array of shape (B, N) or (N,) containing minimum distances
    """
    # Add batch dimension if single configuration
    if configurations.ndim == 1:
        configurations = configurations[None, :]
    
    def get_segment_positions(q):
        # Base position
        joint1_pos = jnp.array([0., 0.])
        
        # First joint position
        angle1 = q[0]
        joint2_pos = jnp.array([
            2.0 * jnp.cos(angle1),
            2.0 * jnp.sin(angle1)
        ])
        
        # End effector position
        end_pos = forward_kinematics(q)
        
        return joint1_pos, joint2_pos, end_pos
    
    # Vectorize over all configurations
    get_positions_vmap = jax.vmap(get_segment_positions)
    joint1_positions, joint2_positions, end_positions = get_positions_vmap(configurations)
    
    # Compute distances to both segments for all configurations
    segment1_distances = jax.vmap(
        lambda j1, j2: point_to_segment_distance_jax(
            obstacle_points, 
            j1,
            j2
        )
    )(joint1_positions, joint2_positions)
    
    segment2_distances = jax.vmap(
        lambda j2, ee: point_to_segment_distance_jax(
            obstacle_points,
            j2,
            ee
        )
    )(joint2_positions, end_positions)
    
    # Take minimum distance to either segment
    min_distances = jnp.minimum(segment1_distances, segment2_distances)
    
    # Remove batch dimension if input was single configuration
    if configurations.shape[0] == 1:
        min_distances = min_distances[0]
    
    return min_distances
