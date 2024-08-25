import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from data.arm_2d_config import link_lengths, link_widths

class Link:
    def __init__(self, length, width):
        self.length = length
        self.width = width
    
class RobotArm2D:
    def __init__(self, link_lengths, link_widths):
        self.links = [Link(length, width) for length, width in zip(link_lengths, link_widths)]
        self.num_links = len(self.links)

    def forward_kinematics(self, angles):
        positions = []
        current_pos = np.zeros(2)
        current_angle = 0

        for i in range(self.num_links):
            link_length = self.links[i].length
            angle = angles[i]
            current_angle += angle

            dx = link_length * np.cos(current_angle)
            dy = link_length * np.sin(current_angle)

            current_pos = current_pos + np.array([dx, dy])
            positions.append(current_pos)

        return np.array(positions)

    def jacobian(self, angles):
        jacobian = []
        current_pos = np.zeros(2)
        
        for i in range(self.num_links):
            link_length = self.links[i].length
            angle = np.sum(angles[:i+1])  # Calculate the cumulative angle up to the current link
            
            jacobian_i = []
            for j in range(i, self.num_links):
                jacobian_ij = np.array([-link_length * np.sin(angle), link_length * np.cos(angle)])
                jacobian_i.append(jacobian_ij)
            
            jacobian.append(jacobian_i)
            current_pos = current_pos + np.array([link_length * np.cos(angle), link_length * np.sin(angle)])
        
        return jacobian

    
def sample_points_on_link(link_pos1, link_pos2, link_length, link_width, num_points):
    # Calculate the center position of the link
    link_center = (link_pos1 + link_pos2) / 2
    
    # Calculate the angle of the link
    link_angle = np.arctan2(link_pos2[1] - link_pos1[1], link_pos2[0] - link_pos1[0])
    
    # Generate random points on the surface of the rectangle
    num_points_per_side = num_points // 4
    x_length = np.random.uniform(-link_length/2, link_length/2, num_points_per_side * 2)
    y_length = np.random.choice([-link_width/2, link_width/2], num_points_per_side * 2)
    x_width = np.random.choice([-link_length/2, link_length/2], num_points_per_side * 2)
    y_width = np.random.uniform(-link_width/2, link_width/2, num_points_per_side * 2)
    
    x = np.concatenate((x_length, x_width))
    y = np.concatenate((y_length, y_width))
    
    # Rotate the points based on the link angle
    rotation_matrix = np.array([[np.cos(link_angle), -np.sin(link_angle)],
                                [np.sin(link_angle), np.cos(link_angle)]])
    rotated_points = np.dot(rotation_matrix, np.vstack((x, y)))
    
    # Translate the points based on the link center position
    translated_points = rotated_points + np.array([[link_center[0]], [link_center[1]]])
    
    return translated_points.T

def sample_arm_points(robot_arm, angles, num_points_per_link):
    positions = robot_arm.forward_kinematics(angles)
    point_clouds = []
    
    for i in range(robot_arm.num_links):
        link_pos1 = positions[i-1] if i > 0 else np.zeros(2)
        link_pos2 = positions[i]
        link_length = robot_arm.links[i].length
        link_width = robot_arm.links[i].width
        
        points = sample_points_on_link(link_pos1, link_pos2, link_length, link_width, num_points_per_link)
        point_clouds.append(points)
    
    return point_clouds

def visualize_arm(robot_arm, angles, point_clouds, goal_endeffector, show_plot = False):
    positions = robot_arm.forward_kinematics(angles)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid(True)

    # Draw the base of the robot arm
    base_radius = 0.1
    base_circle = plt.Circle((0, 0), base_radius, color='k', fill=False)
    ax.add_patch(base_circle)

    # Draw the links and joints of the robot arm
    prev_pos = np.zeros(2)
    for i in range(robot_arm.num_links):
        link_pos = positions[i]
        link_length = robot_arm.links[i].length
        link_width = robot_arm.links[i].width

        # Calculate the angle of the link
        link_angle = np.arctan2(link_pos[1] - prev_pos[1], link_pos[0] - prev_pos[0])

        # Calculate the coordinates of the rectangle corners
        rect_x = prev_pos[0] + link_width/2 * np.sin(link_angle)
        rect_y = prev_pos[1] - link_width/2 * np.cos(link_angle)

        # Draw the link as a rectangle
        link_rect = plt.Rectangle((rect_x, rect_y), link_length, link_width,
                                  angle=np.rad2deg(link_angle),
                                  edgecolor='b', facecolor='y', alpha=0.5)
        ax.add_patch(link_rect)

        # Draw the joint as a circle (except for the end effector)
        if i < robot_arm.num_links - 1:
            joint_radius = 0.05
            joint_circle = plt.Circle(link_pos, joint_radius, color='r', fill=True)
            ax.add_patch(joint_circle)

        prev_pos = link_pos

    # Draw the point clouds to represent obstacles
    for i in range(len(point_clouds)):
        points = point_clouds[i]
        ax.scatter(points[:, 0], points[:, 1], s=10, c=f'C{i}', alpha=0.7)

    ax.plot(goal_endeffector[0], goal_endeffector[1], marker='*', markersize=10, color='red')

    # Set the limits of the plot
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_title('2D Robot Arm')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if show_plot:
        plt.show()
    else:
        return fig


def main():
    robot_arm = RobotArm2D(link_lengths, link_widths)

    # Example configuration (angles in radians)
    angles = [0, np.pi/3, -np.pi/5, -np.pi/4]

    # Sample points on the arm links
    num_points_per_link = 100
    point_clouds = sample_arm_points(robot_arm, angles, num_points_per_link)

    # Visualize the arm and the sampled points
    visualize_arm(robot_arm, angles, point_clouds, goal_endeffector=[0,0], show_plot=True)

if __name__ == "__main__":
    main()