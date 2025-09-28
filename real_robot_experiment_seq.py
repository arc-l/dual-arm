import os
import time
import json
import numpy as np
import argparse
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
    # Capture poses 10 times; visualize only the first capture
    pose_runs = []
    for i in range(10):
        # pose_runs.append(perception.update_cube_poses(visualize=(i == 0)))
        pose_runs.append(perception.update_cube_poses(visualize=(i == 9)))

    # Use intersection of keys to ensure consistent averaging
    common_keys = set(pose_runs[0].keys())
    # for pr in pose_runs[1:]:
    #     common_keys &= set(pr.keys())

    # Average poses (in real frame) across runs
    avg_poses = {}
    for key in sorted(common_keys):
        stack = np.stack([pr[key] for pr in pose_runs if key in pr], axis=0)
        avg_poses[key] = stack.mean(axis=0)

    print("poses (averaged)=")
    print(avg_poses)

    # Rotate 90 deg about Z, then subtract 0.55 on x, and set z to half cube height
    rotation_matrix = R.from_euler('z', 90, degrees=True).as_matrix()
    for key, pose in avg_poses.items():
        rotated_pose = rotation_matrix @ pose
        rotated_pose[0] -= 0.55
        rotated_pose[2] = 0.038 / 2
        avg_poses[key] = rotated_pose
    return avg_poses

def plan_process(plan):
    new_plan = []
    for p in plan:
        if type(p) == list:
            # gripper control
            new_plan.append([str(p[0]).split('.')[-1], str(p[1]).split('.')[-1]])
        else:
            # p is curobo.JointState
            # split the JointState into two joint states for arm1 and arm2
            # Add pi/2 to the base joint so the simulation is aligned with real robot setup 
            p:JointState = p
            positions = p.position.cpu().numpy()
            arm1_positions = positions[:, :6]
            arm2_positions = positions[:, 6:]
            arm1_positions[:, 0] += np.pi/2  # Add pi/2 to the base joint
            arm1_positions[:, -1] -= np.pi/2
            arm2_positions[:, 0] += np.pi/2  # Add pi/2 to the base joint
            arm2_positions[:, -1] -= np.pi/2
            new_plan.append([arm1_positions, arm2_positions])
    return new_plan

def get_arrangements():
    camera = Camera(calibrated=True, on_hand=False)
    goals = update_cube_poses_real2sim(camera)
    print("goals:")
    print(goals)
    keys = sorted(list(goals.keys()))
    input("Press Enter to record starts...")
    starts = update_cube_poses_real2sim(camera)
    print("starts:")
    print(starts)
    data = {
        'starts': [starts[key].tolist() for key in keys],
        'goals': [goals[key].tolist() for key in keys],
        "object_size": 0.038,
        "map_width": 0.6,
        "map_length": 0.6 }
    with open("data/real_robot_experiment.json", 'w') as f:
        json.dump(data, f, indent=4)

def get_arrangement(camera):
    goals = update_cube_poses_real2sim(camera)
    print("goals:")
    print(goals)
    keys = sorted(list(goals.keys()))
    goals = [goals[key].tolist() for key in keys]
    return goals

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--letter', default='C')
    args = argparser.parse_args()

    arm1 = UR_Robotiq_2f_85_Controller(robot_ip="172.17.139.100", init_gripper=True, dt=0.02)
    arm2 = UR_Robotiq_2f_85_Controller(robot_ip="172.17.139.103", init_gripper=True, dt=0.02)
    real_dual_arm = DualArm(arm1, arm2)
    real_dual_arm.go_home()
    camera = Camera(calibrated=True, on_hand=False)
    letter = args.letter
    start = get_arrangement(camera)
    goal = json.load(open(f"data/icra/{letter}.json"))['goals']
    data = {
        'starts': start,
        'goals': goal,
        'object_size': 0.045,
        'map_width': 0.45,
        'map_length': 0.45
    }
    with open("data/real_robot_experiment.json", 'w') as f:
        json.dump(data, f, indent=4)
    task = Task.read_from_json("data/real_robot_experiment.json")
    dual_arm_env = DualArmEnv(task, headless=False, batch_size=3, num_buffers=20, pure_curobo=False)
    plan = dual_arm_env.plan()
    save_plan(plan, "real_robot_experiment.json", "ours")
    plan = plan_process(plan)
    real_dual_arm.execute(plan, interpolation_num=5)
