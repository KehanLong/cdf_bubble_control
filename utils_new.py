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