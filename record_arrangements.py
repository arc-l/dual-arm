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

def update_cube_poses_real2sim(camera:Camera):
    perception = Perception(camera=camera)
    perception.stream()
    poses = perception.update_cube_poses(visualize=True)
    print("poses=")
    print(poses)
    # for the poses, add 0.55 at y axis, then rotate 90 degrees around the z axis
    for key, pose in poses.items():
        pose[1] += 0.55  # Add 0.55 to the y axis
        rotation_matrix = R.from_euler('z', 90, degrees=True).as_matrix()
        rotated_pose = rotation_matrix @ pose
        rotated_pose[2] = 0.038/2
        poses[key] = rotated_pose
    return poses

def get_arrangement(save_path):
    camera = Camera(calibrated=True, on_hand=False)
    goals = update_cube_poses_real2sim(camera)
    print("goals:")
    keys = sorted(list(goals.keys()))
    data = {
        'goals': [goals[key].tolist() for key in keys],
        "object_size": 0.038,
        "map_width": 0.6,
        "map_length": 0.6 }
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    arm1 = UR_Robotiq_2f_85_Controller(robot_ip="172.17.139.100", init_gripper=True, dt=0.02)
    arm2 = UR_Robotiq_2f_85_Controller(robot_ip="172.17.139.103", init_gripper=True, dt=0.02)
    real_dual_arm = DualArm(arm1, arm2)
    real_dual_arm.go_home()
    save_dir = "data/icra"
    file_name = "6.json"
    save_path = os.path.join(save_dir, file_name)
    get_arrangement(save_path=save_path)
