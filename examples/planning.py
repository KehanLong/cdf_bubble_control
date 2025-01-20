import sys
from pathlib import Path
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  # Insert at beginning of path to ensure local version is used first

# Now import from local sdf_marching package
from sdf_marching.samplers import (
    get_uniform_random, 
    get_rapidly_exploring, 
    get_rapidly_exploring_connect,
)
from sdf_marching.samplers import trace_toward_graph_all
from sdf_marching.overlap import position_to_max_circle_idx
from sdf_marching.cvx import (
    bezier_cost_all,
    edgeseq_to_traj_constraint_bezier
)
from sdf_marching.envs import GazeboDataset
from sdf_marching.discrete import get_shortest_path, epath_to_vpath
from sdf_marching.plotting import get_circle_patch

import cvxpy
import numpy as np
#from sdf_marching.samplers.rrt_jax import get_rapidly_exploring_jax

def create_multigoal_sampler(mins, maxs, goal_configs, p0=0.2, rng=None):
    """
    Creates a sampler that:
    - With probability p0, chooses one of the G goals (each with p0/G probability)
    - With probability 1-p0, returns uniform random sample
    """
    if rng is None:
        rng = np.random.default_rng()
    
    goal_configs = np.array(goal_configs)
    num_goals = len(goal_configs)
    
    def sample_fn(batch_size=100):
        samples = []
        for _ in range(batch_size):
            if rng.random() < p0:
                # Choose one of the G goals randomly
                goal_idx = rng.integers(0, num_goals)
                sample = goal_configs[goal_idx]
            else:
                # Uniform random sampling
                sample = rng.uniform(mins, maxs)
            samples.append(sample)
        
        samples = np.array(samples)
        return samples[0] if batch_size == 1 else samples
    
    return sample_fn

if __name__ == "__main__":
    # Parameters
    num_test_positions = 1000
    epsilon = 0.1
    minimum_radius = 0.1
    max_retry = 100
    max_retry_epsilon = 10000
    inflate_factor = 1.0
    p0 = 0.1  # Probability of sampling from goals

    # Set up the problem
    sdf = GazeboDataset()
    mins = sdf.mins
    maxs = sdf.maxs

    start_position = np.array([0.8, -3.])
    # Define multiple goal positions
    goal_positions = [
        np.array([0.7, -12]),
        np.array([15.0, -0]),
        np.array([10., -10.5])
    ]

    rng = np.random.default_rng(seed=3)

    # Create multi-goal sampler
    sampler = create_multigoal_sampler(
        mins, 
        maxs, 
        goal_positions, 
        p0=p0,
        rng=rng
    )

    planning_method = "rrt"   # or "rrt"

    # Run RRT with custom sampler
    if planning_method == "rrt":    
        overlaps_graph, max_circles, _ = get_rapidly_exploring(
            sdf,
            epsilon,
            minimum_radius,
            num_test_positions,
            mins,
            maxs,
            start_position,
            end_point=goal_positions,
            batch_size=100,
            max_retry=max_retry,
            max_retry_epsilon=max_retry_epsilon,
            inflate_factor=inflate_factor,
            sample_fn=sampler,  # Use our custom sampler
            rng=rng,
            profile=False,
            early_termination=True
        )

        goal_connections = []
        for goal_idx, goal_position in enumerate(goal_positions):
            end_idx = position_to_max_circle_idx(overlaps_graph, goal_position)
            if end_idx < 0:
                print(f"Attempting to repair graph for goal {goal_idx}")
                try:
                    overlaps_graph, end_idx = trace_toward_graph_all(
                        overlaps_graph, 
                        sdf, 
                        epsilon, 
                        minimum_radius, 
                        goal_position
                    )
                    goal_connections.append((end_idx, goal_position))
                    print(f"Successfully connected goal {goal_idx}")
                except Exception as e:
                    print(f"Failed to connect goal {goal_idx}: {e}")
            else:
                goal_connections.append((end_idx, goal_position))
                print(f"Goal {goal_idx} already connected")

    elif planning_method == "rrt_connect":
        overlaps_graph, max_circles, _ = get_rapidly_exploring_connect(
            sdf,
            epsilon,
            minimum_radius,
            num_test_positions,
            mins,
            maxs,
        start_position,
        batch_size=100,
        max_retry=max_retry,
        end_point=goal_positions,
        max_retry_epsilon=max_retry_epsilon,
        inflate_factor=inflate_factor,
        prc=p0,
        rng=rng,
        profile=True,
        early_termination=False
        )
        goal_connections = []
        for goal_idx, goal_position in enumerate(goal_positions):
            end_idx = position_to_max_circle_idx(overlaps_graph, goal_position)
            if end_idx >= 0:  # If goal is connected
                goal_connections.append((end_idx, goal_position))
                print(f"Goal {goal_idx} connected")




    # Unpack the centers and radii
    centers, radii = max_circles

    # Find start connection
    start_idx = position_to_max_circle_idx(overlaps_graph, start_position)
    if start_idx < 0:
        print("Repairing graph for start")
        overlaps_graph, start_idx = trace_toward_graph_all(
            overlaps_graph, 
            sdf, 
            epsilon, 
            minimum_radius, 
            start_position
        )

    """
    Find best path among all connected goals
    """
    overlaps_graph.to_directed()
    best_path = None
    best_cost = float('inf')
    best_goal = None

    for end_idx, goal_position in goal_connections:
        epath = get_shortest_path(
            lambda from_circle, to_circle: from_circle.hausdorff_distance_to(to_circle),
            overlaps_graph,
            start_idx,
            end_idx,
            cost_name="hausdorff_distance",
            return_epath=True,
        )
        
        if epath is not None:
            path_cost = sum(overlaps_graph.es[e]["hausdorff_distance"] for e in epath[0])
            if path_cost < best_cost:
                best_cost = path_cost
                best_path = epath
                best_goal = goal_position

    if best_path is None:
        print("No path found to any goal!")
    else:
        print(f"Found path to goal at {best_goal} with cost {best_cost}")
        
        # Generate trajectory for best path
        bps, constr_bps = edgeseq_to_traj_constraint_bezier(
            overlaps_graph.es[best_path[0]], 
            start_position, 
            best_goal
        )

        cost = bezier_cost_all(bps)
        prob_bezier = cvxpy.Problem(cvxpy.Minimize(cost), constr_bps)
        prob_bezier.solve(verbose=True)

        times = np.linspace(0, 1.0, 50)
        query = np.vstack(list(map(lambda bp: bp.query(times).value, bps)))

        """
        Plotting
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(sdf.obs[:, 0], sdf.obs[:, 1], 'ko')

        # Plot all goals
        for goal_pos in goal_positions:
            ax.plot(goal_pos[0], goal_pos[1], 'k^')

        # Plot bubbles
        for center, radius in zip(centers, radii):
            circle = plt.Circle(center, radius, color='cyan', alpha=0.5)
            ax.add_patch(circle)

        # Plot start
        ax.plot(start_position[0], start_position[1], 'kv')

        # Plot trajectory
        ax.plot(query[:, 0], query[:, 1], 'r-', linewidth=2)

        plt.axis('equal')
        plt.show()
