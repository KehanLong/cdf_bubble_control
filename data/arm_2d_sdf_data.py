import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.arm_2d_utils import RobotArm2D
from data.arm_2d_config import link_lengths, link_widths

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from data.dataset import SDFDataset

def visualize_training_data(dataset):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot the square object
    square = plt.Rectangle((-0.5, -0.5), 1, 1, edgecolor='black', facecolor='none', linewidth=2)
    ax.add_patch(square)
    
    # Plot the training points
    # print('dataset_x:', dataset.x)
    # print('y:', dataset.y)
    inside_points = dataset.points[dataset.distances.reshape(-1) < 0]
    outside_points = dataset.points[dataset.distances.reshape(-1) >= 0]
    ax.scatter(inside_points[:, 0], inside_points[:, 1], c='blue', marker='o', label='Inside')
    ax.scatter(outside_points[:, 0], outside_points[:, 1], c='red', marker='x', label='Outside')
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title('Training Data')
    ax.legend()
    plt.show()

def visualize_link_sdf_data(robot_arm, link_sdf_data):
    num_links = robot_arm.num_links
    fig, axes = plt.subplots(1, num_links, figsize=(4 * num_links, 4))

    for i in range(num_links):
        link_length = robot_arm.links[i].length
        link_width = robot_arm.links[i].width
        link_pos1 = np.array([-link_length / 2, -link_width / 2])
        link_pos2 = np.array([link_length / 2, link_width / 2])

        ax = axes[i] if num_links > 1 else axes

        # Plot the rectangle link
        rect = plt.Rectangle(link_pos1, link_length, link_width, edgecolor='b', facecolor='y', alpha=0.5)
        ax.add_patch(rect)

        # Plot the training points
        points = link_sdf_data[i]['points']
        distances = link_sdf_data[i]['distances']
        inside_points = points[distances < 0]
        boundary_points = points[distances == 0]
        outside_points = points[distances > 0]

        ax.scatter(inside_points[:, 0], inside_points[:, 1], c='r', marker='o', label='Inside')
        ax.scatter(boundary_points[:, 0], boundary_points[:, 1], c='b', marker='.', label='Boundary')
        ax.scatter(outside_points[:, 0], outside_points[:, 1], c='g', marker='x', label='Outside')

        ax.set_xlim(2 * link_pos1[0], 2 * link_pos2[0])
        ax.set_ylim(2 * link_pos1[1], 2 * link_pos2[1])
        ax.set_aspect('equal')
        ax.set_title(f'Link {i+1}')
        ax.legend()

    plt.tight_layout()
    plt.show()

def generate_link_sdf_data(link_length, link_width, num_samples):
    # Generate points in the space defined by the link's length and width
    x_coords = np.random.uniform(-link_length, link_length, num_samples)
    y_coords = np.random.uniform(-link_width, link_width, num_samples)
    points = np.column_stack((x_coords, y_coords, np.zeros(num_samples)))

    # Compute distances for each point
    distances = np.zeros((num_samples, 1))
    for i in range(num_samples):
        point = points[i, :2]
        x, y = point[0], point[1]

        # Check if the point is inside or outside the link
        inside = (x >= -link_length / 2) and (x <= link_length / 2) and (y >= -link_width / 2) and (y <= link_width / 2)

        if inside:
            # If inside, compute the negative distance to the nearest edge
            left_dist = x + link_length / 2
            right_dist = link_length / 2 - x
            bottom_dist = y + link_width / 2
            top_dist = link_width / 2 - y
            distances[i] = -min(left_dist, right_dist, bottom_dist, top_dist)
        else:
            # If outside, divide the region into 8 sub-regions
            if x < -link_length / 2 and y >= -link_width / 2 and y <= link_width / 2:
                # Left region
                distances[i] = -x - link_length / 2
            elif x > link_length / 2 and y >= -link_width / 2 and y <= link_width / 2:
                # Right region
                distances[i] = x - link_length / 2
            elif y < -link_width / 2 and x >= -link_length / 2 and x <= link_length / 2:
                # Bottom region
                distances[i] = -y - link_width / 2
            elif y > link_width / 2 and x >= -link_length / 2 and x <= link_length / 2:
                # Top region
                distances[i] = y - link_width / 2
            elif x < -link_length / 2 and y < -link_width / 2:
                # Bottom-left corner
                distances[i] = np.sqrt((x + link_length / 2)**2 + (y + link_width / 2)**2)
            elif x > link_length / 2 and y < -link_width / 2:
                # Bottom-right corner
                distances[i] = np.sqrt((x - link_length / 2)**2 + (y + link_width / 2)**2)
            elif x < -link_length / 2 and y > link_width / 2:
                # Top-left corner
                distances[i] = np.sqrt((x + link_length / 2)**2 + (y - link_width / 2)**2)
            else:
                # Top-right corner
                distances[i] = np.sqrt((x - link_length / 2)**2 + (y - link_width / 2)**2)

    return points, distances


def main():
    robot_arm = RobotArm2D(link_lengths, link_widths)
    num_samples_per_link = 5000

    # Generate SDF training data for each link separately
    for i in range(robot_arm.num_links):
        link_length = robot_arm.links[i].length
        link_width = robot_arm.links[i].width

        points, distances = generate_link_sdf_data(link_length, link_width, num_samples_per_link)

        dataset = SDFDataset(points, distances)

        visualize_training_data(dataset)

        # Save the dataset for each link separately
        data = {
            'points': points,
            'distances': distances
        }
        np.save(f'link{i+1}_sdf_data.npy', data)


if __name__ == "__main__":
    main()