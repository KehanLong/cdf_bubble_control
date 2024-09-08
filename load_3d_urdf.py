import os
import pybullet as p
import pybullet_data
import time
from roboticstoolbox import ERobot
import roboticstoolbox as rtb
from spatialmath import SE3

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the URDF file
urdf_path = os.path.join(current_dir, "xarm", "xarm6_robot.urdf")

# Load the URDF file
robot = ERobot.URDF(urdf_path)

# Define a target end-effector pose
Tep = SE3.Trans(0.6, -0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])

# Solve inverse kinematics
sol = robot.ik_LM(Tep)
q_pickup = sol[0]

# Generate a trajectory from the ready pose to the pickup pose
qt = rtb.jtraj(robot.q, q_pickup, 50)

# Set up PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")

# Load the robot URDF
robotId = p.loadURDF(urdf_path, useFixedBase=1)

# Animate the robot through the trajectory
for q in qt.q:
    for i, angle in enumerate(q):
        p.resetJointState(robotId, i, angle)
    p.stepSimulation()
    time.sleep(0.1)

print("Animation complete. Press Ctrl+C to exit.")
while True:
    time.sleep(0.1)

