import os
import json
import numpy as np
from ur_tools.camera.realsense_camera import Camera
from ur_tools.ur_control.dual_arm import DualArm
from ur_tools.ur_control.ur5_robotiq_2f_85_control import UR_Robotiq_2f_85_Controller
from perception import Perception
from scipy.spatial.transform import Rotation as R
from task import Task
from dual_arms import DualArmEnv
from curobo.types.state import JointState
from helper import save_plan, load_plan 
from real_robot_experiment import plan_process


if __name__ == "__main__":
    plan = load_plan('results/real_robot_experiment_ours_plan.pkl', device='cuda:0')
    # task = Task.read_from_json("data/real_robot_experiment.json")
    # dual_arm_env = DualArmEnv(task, headless=False, batch_size=3, num_buffers=20, pure_curobo=False)
    # dual_arm_env.play(plan)
    plan = plan_process(plan)
    arm1 = UR_Robotiq_2f_85_Controller(robot_ip="172.17.139.100", init_gripper=True, dt=0.02)
    arm2 = UR_Robotiq_2f_85_Controller(robot_ip="172.17.139.103", init_gripper=True, dt=0.02)
    real_dual_arm = DualArm(arm1, arm2)
    real_dual_arm.execute(plan)
