import math
import numpy as np
import sys
sys.path.append("../../../")
from utils.visualize import motion_plot

def discrete_time_model(pose, control, td):
    """ Discrete time motion model. It calculates a new pose given the previous
        pose, time discretization, and controls.

        pose      - the previous pose (x, y, orientation)^t
        control   - control input used to calculate motion
                    (linear velocity, angular velocity)
        td        - time discretization """

    shift = (control[1] * td) / 2
    sc = 1
    if shift != 0:
        sc = math.sin(shift)/shift

    return pose + td * np.array([[control[0] * sc * math.cos(pose[2, 0] + shift)], [control[0] * sc * math.sin(pose[2, 0] + shift)], [control[1]]])

if __name__ == "__main__":
    pose1 = np.array([[0], [0], [0]])
    pose_stack1 = pose1
    control1 = [0.1, 0.1]

    pose2 = np.array([[0], [0], [0]])
    pose_stack2 = pose2
    control2 = [1, 0.1]

    dt = 1
    sim_time = 100
    t = 0
    while t < sim_time:
        t += dt
        pose1 = discrete_time_model(pose1, control1, dt)
        pose_stack1 = np.hstack((pose_stack1, pose1))

        pose2 = discrete_time_model(pose2, control2, dt)
        pose_stack2 = np.hstack((pose_stack2, pose2))
        motion_plot(pose_stack1, pose_stack2)
