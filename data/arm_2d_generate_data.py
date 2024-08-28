import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from arm_2d_utils import RobotArm2D, sample_arm_points
from data.arm_2d_config import link_lengths, link_widths
from scipy.spatial import ConvexHull

def generate_training_data(robot_arm, num_samples, num_points_per_link, num_workspace_points, joint_limits, expanded_limits=False):
    dataset = []
    expansion_factor = 0.05

    for _ in range(num_samples):
        # Generate random joint angles within the joint limits
        if expanded_limits:
            lower_limits = [limit[0] - expansion_factor * (limit[1] - limit[0]) for limit in joint_limits]
            upper_limits = [limit[1] + expansion_factor * (limit[1] - limit[0]) for limit in joint_limits]
            angles = [np.random.uniform(lower, upper) for lower, upper in zip(lower_limits, upper_limits)]
        else:
            angles = [np.random.uniform(limit[0], limit[1]) for limit in joint_limits]

        # Sample points on the arm links
        point_clouds = sample_arm_points(robot_arm, angles, num_points_per_link)

        # Convert point clouds to 3D by appending z=0
        point_clouds_3d = [np.hstack((points, np.zeros((points.shape[0], 1)))) for points in point_clouds]

        # Generate random workspace points
        workspace_points = np.random.uniform(-20, 20, (num_workspace_points, 3))
        workspace_points[:, 2] = 0  # Set z-coordinate to 0

        for point in workspace_points:
            distances = []
            for link_points in point_clouds_3d:
                # Calculate the minimum distance between the link and the workspace point
                min_distance = np.min(np.linalg.norm(link_points - point, axis=1))

                # Check if the workspace point is inside the link using convex hull
                hull = ConvexHull(link_points[:, :2])  # Use only x and y coordinates for convex hull
                if point_in_hull(point[:2], hull):
                    min_distance = -min_distance  # Assign negative distance if inside the link

                distances.append(min_distance)

            # Create a dataset entry
            entry = {
                'angles': angles,
                'point': point,
                'distances': distances
            }
            dataset.append(entry)

    return dataset

def point_in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations
    )

def main():
    robot_arm = RobotArm2D(link_lengths, link_widths)
    joint_limits = [(-np.pi, np.pi)] * robot_arm.num_links
    num_configuration_samples = 5
    num_points_per_link = 100

    num_workspace_points = 20

    # Generate training data
    training_data = generate_training_data(robot_arm, num_configuration_samples, num_points_per_link, num_workspace_points, joint_limits, expanded_limits=True)

   
    # Save the datasets
    np.save('training_data.npy', training_data)

if __name__ == "__main__":
    main()