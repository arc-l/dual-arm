import numpy as np
import math
from dataclasses import dataclass
import os

# paths
root_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(
    root_path, 'motion_planning', 'assets/curobo_configs'
    )
arm0_yml_path = os.path.join(config_path, 'robot/ur5e_0.yml')
arm1_yml_path = os.path.join(config_path, 'robot/ur5e_1.yml')
dual_arm_yml_path = os.path.join(config_path, 'robot/dual_ur5e.yml')
real_time_control_folder = os.path.join(root_path, 'real_time_control')
robot_arms_ip = {0: "172.17.139.100", 1: "172.17.139.103"}


# tolerance
EPSILON = 1e-3

# workspace pad
WORKSPACE_SIZE = 0.60
# real env center in robot 0 coordinate
REAL_CENTER = [-0.0362, -0.471]
# sim env center in world coordinate
SIM_CENTER = [0.5, 0.0]
PNP_ROT = [-0.5877976611933862, 0.8089882123010008, 0.005586555396007588,0.0008787789389623685] # quaternion
# PNP_ROT = [7.07106781e-01, -7.07106781e-01, 4.32978028e-17, 4.32978028e-17]
