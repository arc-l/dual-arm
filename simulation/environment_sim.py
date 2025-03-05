from logging import critical
import this
import time
import sys
import os
import math
from pybullet_utils import bullet_client
import pybullet as pb
import pybullet_data
import numpy as np
import cameras
from constants import (
    ARM_RADIUS_LIMIT,
    IMAGE_SIZE,
    NUM_ROTATION,
    PIXEL_SIZE,
    WORKSPACE_LIMITS,
    COLOR_SPACE,
    COLOR_MAP,
    blue_lower,
    blue_upper,
    green_lower,
    green_upper,
    red_lower,
    red_upper,
)
import utils
from dataclasses import dataclass
from colorama import Fore
from threading import Thread
import multiprocessing as mp

handoff_position = [0.5, 0.0, 0.7]
handoff_euler = [[0.0, 0.0, 0.0], [math.pi / 2.0, 0.0, math.pi]]


Maximum_Timeout = 100  # general timeout limit for an action

Manipulation_Wait = 0.0 # cost of grasp and release
Manipulation_Speed = 0.0005
class Environment:
    def __init__(self, gui=True, time_step=1 / 240.0):
        """Creates environment with PyBullet.

        Args:
        gui: show environment with PyBullet's built-in display viewer
        time_step: PyBullet physics simulation step speed. Default is 1 / 240.
        """

        self.time_step = time_step
        self.gui = gui
        self.obj_ids = {"fixed": [], "rigid": []}
        self.objects_reward = {}
        self.pixel_size = PIXEL_SIZE
        self.bounds = WORKSPACE_LIMITS
        self.ee_tip_offset = 0.08
        self.buffer_height = 0.1
        self.gripper_open_angle = 0.03
        self.gripper_close_angle = 0.8
        self.gripper_close_angle_threshold = 0.73
        self.agent_cams = cameras.RealSenseD435.CONFIG
        self.oracle_cams = cameras.Oracle.CONFIG
        self.center_control = False  # 0 means no center control, 1 means center control, 2 means
        self.arms_home_joints = [
            np.array([0, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi,
            np.array([-1, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi,
        ]
        self.arms_first_joints = [
            np.array([-0.1, -0.6, 0.5, -0.4, -0.5, 0]) * np.pi,
            np.array([-1.1, -0.6, 0.5, -0.4, -0.5, 0]) * np.pi,
        ]
        self.arms_ik_rest_joints = [
            np.array([0, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi,
            np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi,
        ]
        self.arms_drop_joints0 = [
            np.array([0.5, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi,
            np.array([-0.5, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi,
        ]
        self.arms_drop_joints1 = [
            np.array([0.9, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi,
            np.array([-0.1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi,
        ]

        # Start PyBullet.
        self._pb = bullet_client.BulletClient(connection_mode=pb.GUI if gui else pb.DIRECT)
        self._pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._pb.setTimeStep(time_step)

        if gui:
            self._pb.resetDebugVisualizerCamera(
                cameraDistance=1.2, cameraYaw=510, cameraPitch=-53, cameraTargetPosition=[0.5, 0, 0.23],
            )
            self._pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 1)

    # def init_lock(self):
    #     self.write_state = False
    #     self.

    def reset(self):
        self.obj_ids = {"helper": [], "rigid": []}
        self._pb.resetSimulation()
        self._pb.setGravity(0, 0, -9.8)

        # Temporarily disable rendering to load scene faster.
        if self.gui:
            self._pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        # Load workspace
        self.plane = self._pb.loadURDF("plane.urdf", basePosition=(0, 0, -0.0005), useFixedBase=True)
        current_folder = os.path.dirname(os.path.abspath(__file__))
        self.workspace = self._pb.loadURDF(
            os.path.join(current_folder, "assets/workspace/workspace.urdf"),
            basePosition=(0.50, 0, 0),
            useFixedBase=True,
        )

        # the logging system must manually create constraint and remove, simple save/restore state won't work
        helper0 = self._pb.loadURDF(os.path.join(current_folder, "assets/blocks/cube.urdf"), basePosition=(2, 0, 0))
        helper1 = self._pb.loadURDF(os.path.join(current_folder, "assets/blocks/cube.urdf"), basePosition=(2, 0.1, 0))
        self.obj_ids["helper"].append(helper0)
        self.obj_ids["helper"].append(helper1)

        # self.workspace_box = self._pb.loadURDF(
        #     "assets/workspace/wall50.urdf", basePosition=(0.5, 0, 0.01), useFixedBase=True
        # )
        self._pb.changeDynamics(
            self.plane, -1, lateralFriction=2, restitution=2, linearDamping=2, angularDamping=2,
        )
        self._pb.changeDynamics(
            self.workspace, -1, lateralFriction=2, restitution=2, linearDamping=2, angularDamping=2,
        )

        # self.box1 = self._pb.loadURDF(
        #     "assets/workspace/wall30.urdf", basePosition=(-0.5, 0, 0), useFixedBase=True
        # )
        # self.box2 = self._pb.loadURDF(
        #     "assets/workspace/wall30.urdf", basePosition=(1.5, 0, 0), useFixedBase=True
        # )

        # Load two UR5e
        self.arms = [Arm(*self.setup_ur5e((-0.1, 0, 0))), Arm(*self.setup_ur5e((1.1, 0, 0)))]

        # Move robot to home joint configuration.
        for i in range(len(self.arms)):
            success = self.go_home(i)

        if not success:
            print("Simulation is wrong!")
            sys.exit(-1)

        # Re-enable rendering.
        if self.gui:
            self._pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)

    def setup_rearrangement_task(self, file_name):
        # size = 0.42
        # bound = size / 2
        # num_cell = 3
        # cell_size = size / num_cell

        # if self.gui:
        #     for i in range(num_cell + 1):
        #         self._pb.addUserDebugLine(
        #             [0.5 - bound + i * cell_size, 0 - bound, 0.01],
        #             [0.5 - bound + i * cell_size, 0 + bound, 0.01],
        #             [255, 255, 0],
        #             lifeTime=0,
        #         )
        #         self._pb.addUserDebugLine(
        #             [0.5 - bound, 0 - bound + i * cell_size, 0.01],
        #             [0.5 + bound, 0 - bound + i * cell_size, 0.01],
        #             [255, 255, 0],
        #             lifeTime=0,
        #         )

        success = True

        # Read data
        with open(file_name, "r") as preset_file:
            file_content = preset_file.readlines()
            num_obj = len(file_content)
            obj_files = []
            obj_mesh_colors = []
            obj_positions = []
            obj_orientations = []
            for object_idx in range(num_obj):
                file_content_curr_object = file_content[object_idx].split()
                obj_file = os.path.join("assets", "blocks", file_content_curr_object[0])
                obj_files.append(obj_file)
                obj_positions.append(
                    [
                        float(file_content_curr_object[4]),
                        float(file_content_curr_object[5]),
                        float(file_content_curr_object[6]),
                    ]
                )
                obj_orientations.append(
                    [
                        float(file_content_curr_object[7]),
                        float(file_content_curr_object[8]),
                        float(file_content_curr_object[9]),
                    ]
                )
                obj_mesh_colors.append(
                    [
                        float(file_content_curr_object[1]),
                        float(file_content_curr_object[2]),
                        float(file_content_curr_object[3]),
                    ]
                )

        # Import objects
        for object_idx in range(num_obj):
            curr_mesh_file = obj_files[object_idx]
            object_position = [
                obj_positions[object_idx][0],
                obj_positions[object_idx][1],
                obj_positions[object_idx][2],
            ]
            object_orientation = [
                obj_orientations[object_idx][0],
                obj_orientations[object_idx][1],
                obj_orientations[object_idx][2],
            ]
            object_color = [
                obj_mesh_colors[object_idx][0],
                obj_mesh_colors[object_idx][1],
                obj_mesh_colors[object_idx][2],
                1,
            ]
            body_id = self._pb.loadURDF(curr_mesh_file, object_position, pb.getQuaternionFromEuler(object_orientation),)
            self._pb.changeVisualShape(body_id, -1, rgbaColor=object_color)
            self.add_object_id(body_id)
            for _, info in COLOR_MAP.items():
                if np.abs(info.rgb - object_color[:3]).sum() < 1e-3:
                    self.objects_reward[body_id] = info.reward
                    break
            if body_id not in self.objects_reward:
                sys.exit(f"Color is not matching from {file_name}")

        # give time to stop
        for _ in range(5):
            self._pb.stepSimulation()

        success &= self.wait_static()

    def setup_rearrangement_task_from_dict(self, start_arr, radius):

        success = True

        num_obj = len(start_arr)
        obj_files = []
        obj_mesh_colors = []
        obj_positions = []
        obj_orientations = []
        for object_idx in list(start_arr.keys()):
            obj_file = os.path.join("assets", "blocks", "cylinder.urdf")
            obj_files.append(obj_file)
            obj_positions.append(utils.env_pose_to_sim_pose(start_arr[object_idx]))
            obj_orientations.append([0.0, -0.0, 1.5707964897155762])
            obj_mesh_colors.append([0.30588235, 0.4745098, 0.65490196])

        # Import objects
        for object_idx in range(num_obj):
            current_folder = os.path.dirname(os.path.abspath(__file__))
            curr_mesh_file = os.path.join(current_folder, "assets", "blocks", "cylinder.urdf")
            object_position = utils.env_pose_to_sim_pose(start_arr[object_idx])
            object_orientation = (0.0, -0.0, 1.5707964897155762)
            object_color = [
                0.30588235,
                0.4745098,
                0.65490196,
                1,
            ]
            body_id = self._pb.loadURDF(curr_mesh_file, object_position, pb.getQuaternionFromEuler(object_orientation),)
            self._pb.changeVisualShape(body_id, -1, rgbaColor=object_color)
            self.add_object_id(body_id)
            # for _, info in COLOR_MAP.items():
            #     if np.abs(info.rgb - object_color[:3]).sum() < 1e-3:
            #         self.objects_reward[body_id] = info.reward
            #         break

        # give time to stop
        for _ in range(5):
            self._pb.stepSimulation()

        success &= self.wait_static()

    def setup_ur5e(self, position):
        """Load ur5e in a given position"""
        current_folder = os.path.dirname(os.path.abspath(__file__))
        ur5e = self._pb.loadURDF(
            os.path.join(current_folder, "assets/ur5e/ur5e.urdf"), basePosition=position, useFixedBase=True
        )
        ur5e_joints = []
        for i in range(self._pb.getNumJoints(ur5e)):
            info = self._pb.getJointInfo(ur5e, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "ee_fixed_joint":
                ur5e_ee_id = joint_id
            if joint_type == pb.JOINT_REVOLUTE:
                ur5e_joints.append(joint_id)
        self._pb.enableJointForceTorqueSensor(ur5e, ur5e_ee_id, 1)
        (ee, ee_tip_id,) = self.setup_gripper(ur5e, ur5e_ee_id)

        return (
            ur5e,
            ur5e_ee_id,
            ur5e_joints,
            ee,
            ee_tip_id,
        )

    def setup_gripper(self, ur5e, ur5e_ee_id):
        """Load end-effector: Suction, borrorwed from https://github.com/google-research/ravens/blob/master/ravens/tasks/grippers.py"""

        # Load suction gripper base model (visual only).
        pose = ((0.487, 0.109, 0.438), self._pb.getQuaternionFromEuler((0, -np.pi / 2, 0)))
        current_folder = os.path.dirname(os.path.abspath(__file__))
        base = self._pb.loadURDF(
            os.path.join(current_folder, "assets/ur5e/suction/suction-base.urdf"), pose[0], pose[1],
        )
        self._pb.createConstraint(
            parentBodyUniqueId=ur5e,
            parentLinkIndex=ur5e_ee_id,
            childBodyUniqueId=base,
            childLinkIndex=-1,
            jointType=pb.JOINT_FIXED,
            jointAxis=(0, 0, 1),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0.01),
            childFrameOrientation=pose[1],
        )

        # Load suction tip model (visual and collision) with compliance.
        pose = ((0.487, 0.109, 0.347), self._pb.getQuaternionFromEuler((0, -np.pi / 2, 0)))
        current_folder = os.path.dirname(os.path.abspath(__file__))
        ee = self._pb.loadURDF(os.path.join(current_folder, "assets/ur5e/suction/suction-head.urdf"), pose[0], pose[1],)
        constraint_id = self._pb.createConstraint(
            parentBodyUniqueId=ur5e,
            parentLinkIndex=ur5e_ee_id,
            childBodyUniqueId=ee,
            childLinkIndex=-1,
            jointType=pb.JOINT_FIXED,
            jointAxis=(0, 0, 1),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.08),
            childFrameOrientation=pose[1],
        )
        self._pb.changeConstraint(constraint_id, maxForce=50000)

        info = self._pb.getJointInfo(ee, 0)
        joint_id = info[0]
        joint_name = info[1].decode("utf-8")
        joint_type = info[2]
        ee_tip_id = joint_id
        return ee, ee_tip_id

    def setup_collaborative_task(self, file_name):
        body_ids = []
        success = True
        # Read data
        with open(file_name, "r") as preset_file:
            file_content = preset_file.readlines()
            num_obj = len(file_content)
            obj_files = []
            obj_mesh_colors = []
            obj_positions = []
            obj_orientations = []
            for object_idx in range(num_obj):
                file_content_curr_object = file_content[object_idx].split()
                obj_file = os.path.join("assets", "blocks", file_content_curr_object[0])
                obj_files.append(obj_file)
                obj_positions.append(
                    [
                        float(file_content_curr_object[4]),
                        float(file_content_curr_object[5]),
                        float(file_content_curr_object[6]),
                    ]
                )
                obj_orientations.append(
                    [
                        float(file_content_curr_object[7]),
                        float(file_content_curr_object[8]),
                        float(file_content_curr_object[9]),
                    ]
                )
                obj_mesh_colors.append(
                    [
                        float(file_content_curr_object[1]),
                        float(file_content_curr_object[2]),
                        float(file_content_curr_object[3]),
                    ]
                )

        # Import objects
        for object_idx in range(num_obj):
            curr_mesh_file = obj_files[object_idx]
            object_position = [
                obj_positions[object_idx][0],
                obj_positions[object_idx][1],
                obj_positions[object_idx][2],
            ]
            object_orientation = [
                obj_orientations[object_idx][0],
                obj_orientations[object_idx][1],
                obj_orientations[object_idx][2],
            ]
            object_color = [
                obj_mesh_colors[object_idx][0],
                obj_mesh_colors[object_idx][1],
                obj_mesh_colors[object_idx][2],
                1,
            ]
            body_id = self._pb.loadURDF(curr_mesh_file, object_position, pb.getQuaternionFromEuler(object_orientation),)
            self._pb.changeVisualShape(body_id, -1, rgbaColor=object_color)
            body_ids.append(body_id)
            self.add_object_id(body_id)
            success &= self.wait_static()
            success &= self.wait_static()

        # give time to stop
        for _ in range(5):
            self._pb.stepSimulation()

        return body_ids, success

    def setup_competitive_task(self, file_name):
        success = True

        # Read data
        with open(file_name, "r") as preset_file:
            file_content = preset_file.readlines()
            num_obj = len(file_content)
            obj_files = []
            obj_mesh_colors = []
            obj_positions = []
            obj_orientations = []
            for object_idx in range(num_obj):
                file_content_curr_object = file_content[object_idx].split()
                obj_file = os.path.join("assets", "blocks", file_content_curr_object[0])
                obj_files.append(obj_file)
                obj_positions.append(
                    [
                        float(file_content_curr_object[4]),
                        float(file_content_curr_object[5]),
                        float(file_content_curr_object[6]),
                    ]
                )
                obj_orientations.append(
                    [
                        float(file_content_curr_object[7]),
                        float(file_content_curr_object[8]),
                        float(file_content_curr_object[9]),
                    ]
                )
                obj_mesh_colors.append(
                    [
                        float(file_content_curr_object[1]),
                        float(file_content_curr_object[2]),
                        float(file_content_curr_object[3]),
                    ]
                )

        # Import objects
        for object_idx in range(num_obj):
            curr_mesh_file = obj_files[object_idx]
            object_position = [
                obj_positions[object_idx][0],
                obj_positions[object_idx][1],
                obj_positions[object_idx][2],
            ]
            object_orientation = [
                obj_orientations[object_idx][0],
                obj_orientations[object_idx][1],
                obj_orientations[object_idx][2],
            ]
            object_color = [
                obj_mesh_colors[object_idx][0],
                obj_mesh_colors[object_idx][1],
                obj_mesh_colors[object_idx][2],
                1,
            ]
            body_id = self._pb.loadURDF(curr_mesh_file, object_position, pb.getQuaternionFromEuler(object_orientation),)
            self._pb.changeVisualShape(body_id, -1, rgbaColor=object_color)
            self.add_object_id(body_id)
            for _, info in COLOR_MAP.items():
                if np.abs(info.rgb - object_color[:3]).sum() < 1e-3:
                    self.objects_reward[body_id] = info.reward
                    break
            if body_id not in self.objects_reward:
                sys.exit(f"Color is not matching from {file_name}")

        # give time to stop
        for _ in range(5):
            self._pb.stepSimulation()

        success &= self.wait_static()

        return success

    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [np.linalg.norm(self._pb.getBaseVelocity(i)[0]) for i in self.obj_ids["rigid"]]
        return all(np.array(v) < 5e-3)

    @property
    def is_env_static(self):
        """Return true if all are no longer moving."""
        static_count = 100
        count = 0
        while count < static_count:
            v = [np.linalg.norm(self._pb.getBaseVelocity(i)[0]) for i in self.obj_ids["rigid"]]
            if np.any(np.array(v) > 0.1):
                return False

            for arm_idx in range(len(self.arms)):
                v = np.array(
                    [self._pb.getJointState(self.arms[arm_idx].ur5e, i)[1] for i in self.arms[arm_idx].ur5e_joints]
                )
                if np.any(abs(v) > 0.1):
                    return False
            count += 1
        return True

    @property
    def info(self):
        """Environment info variable with object poses, dimensions, and colors."""

        info = {}  # object id : (position, rotation, dimensions)
        for obj_ids in self.obj_ids.values():
            for obj_id in obj_ids:
                pos, rot = self._pb.getBasePositionAndOrientation(obj_id)
                dim = self._pb.getVisualShapeData(obj_id)[0][3]
                info[obj_id] = (pos, rot, dim)
        return info

    def add_object_id(self, obj_id, category="rigid"):
        """List of (fixed, rigid) objects in env."""
        self.obj_ids[category].append(obj_id)

    def remove_object_id(self, obj_id, category="rigid"):
        """List of (fixed, rigid) objects in env."""
        self.obj_ids[category].remove(obj_id)

    def save_objects(self):
        """Save states of all rigid objects. If this is unstable, could use saveBullet."""
        success = False
        while not success:
            success = self.wait_static()
        object_states = []
        for obj in self.obj_ids["rigid"]:
            pos, orn = self._pb.getBasePositionAndOrientation(obj)
            linVel, angVel = self._pb.getBaseVelocity(obj)
            object_states.append((pos, orn, linVel, angVel))
        return object_states

    def restore_objects(self, object_states):
        """Restore states of all rigid objects. If this is unstable, could use restoreState along with saveBullet."""
        for idx, obj in enumerate(self.obj_ids["rigid"]):
            pos, orn, linVel, angVel = object_states[idx]
            self._pb.resetBasePositionAndOrientation(obj, pos, orn)
            self._pb.resetBaseVelocity(obj, linVel, angVel)
        success = self.wait_static()
        return success

    def save_object(self, start_pose):
        """Save state of target objects. If this is unstable, could use saveBullet."""
        for obj in self.obj_ids["rigid"]:
            pos, orn = self._pb.getBasePositionAndOrientation(obj)
            linVel, angVel = self._pb.getBaseVelocity(obj)
            if np.linalg.norm([pos[0] - start_pose[0], abs(pos[1] - start_pose[1])]) < 0.02:
                object_state = pos, orn, linVel, angVel
                break
        return obj, object_state

    def restore_object(self, obj_info):
        """Restore states of target object. If this is unstable, could use restoreState along with saveBullet."""
        obj_id, object_state = obj_info
        if object_state is not None:
            pos, orn, linVel, angVel = object_state
            self._pb.resetBasePositionAndOrientation(obj_id, pos, orn)
            self._pb.resetBaseVelocity(obj_id, 0, 0)

    def wait_static(self, timeout=Maximum_Timeout):
        """Step simulator asynchronously until objects settle."""
        self._pb.stepSimulation()
        t0 = time.time()
        while (time.time() - t0) < timeout:
            if self.is_static:
                return True
            self._pb.stepSimulation()
        print(Fore.RED + f"Warning: Wait static exceeded {timeout} second timeout. Skipping.")
        return False

    def check_rewards(self, arm_idx):
        for body_id in self.obj_ids["rigid"]:
            found = False
            if arm_idx == 0 and self._pb.getBasePositionAndOrientation(body_id)[0][0] < 0:
                found = True
            elif arm_idx == 1 and self._pb.getBasePositionAndOrientation(body_id)[0][0] > 1:
                found = True

            if found:
                # self._pb.removeBody(body_id)
                # self.remove_object_id(body_id)
                self.arms[arm_idx].reward += self.objects_reward[body_id]
                self._pb.removeUserDebugItem(self.arms[arm_idx].reward_text_id)
                self.arms[arm_idx].reward_text_id = self._pb.addUserDebugText(
                    f"Rewards: {self.arms[arm_idx].reward}",
                    textPosition=[arm_idx, 0, 0.8],
                    textColorRGB=[0.95 - arm_idx * 0.2, 0.56, 0.17],
                    textSize=1.5,
                )
                return self.arms[arm_idx].reward, self.objects_reward[body_id]
        return self.arms[arm_idx].reward, 0

    def clean(self, arm_idx):
        for body_id in self.obj_ids["rigid"]:
            found = False
            if arm_idx == 0 and self._pb.getBasePositionAndOrientation(body_id)[0][0] < 0:
                found = True
            elif arm_idx == 1 and self._pb.getBasePositionAndOrientation(body_id)[0][0] > 1:
                found = True

            if found:
                self._pb.removeBody(body_id)
                self.remove_object_id(body_id)

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def render_camera(self, config):
        """Render RGB-D image with specified camera configuration."""

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = pb.getMatrixFromQuaternion(config["rotation"])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config["position"] + lookdir
        focal_len = config["intrinsics"][0, 0]
        znear, zfar = config["zrange"]
        viewm = pb.computeViewMatrix(config["position"], lookat, updir)
        fovh = (config["image_size"][0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = config["image_size"][1] / config["image_size"][0]
        projm = pb.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = self._pb.getCameraImage(
            width=config["image_size"][1],
            height=config["image_size"][0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=1,
            flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pb.ER_BULLET_HARDWARE_OPENGL,
        )

        # Get color image.
        color_image_size = (config["image_size"][0], config["image_size"][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config["noise"]:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, config["image_size"]))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (config["image_size"][0], config["image_size"][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear)
        depth = (2.0 * znear * zfar) / depth
        if config["noise"]:
            depth += self._random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm

    def __del__(self):
        self._pb.disconnect()

    def get_link_pose(self, body, link):
        result = self._pb.getLinkState(body, link)
        return result[4], result[5]

    # ---------------------------------------------------------------------------
    # Robot Movement Functions
    # ---------------------------------------------------------------------------

    def go_home(self, arm_idx, speed=0.01, check_collision=False, center_control_override=False):
        return self.move_joints(
            arm_idx,
            self.arms_home_joints[arm_idx],
            speed=speed,
            timeout=Maximum_Timeout,
            check_collision=check_collision,
            center_control_override=center_control_override,
        )

    def hold_joints(self, arm_idxs):
        for arm_idx in arm_idxs:
            current_joints = [
                self._pb.getJointState(self.arms[arm_idx].ur5e, i)[0] for i in self.arms[arm_idx].ur5e_joints
            ]
            # for i in range(len(self.arms[arm_idx].ur5e_joints)):
            #     self._pb.resetJointState(
            #         self.arms[arm_idx].ur5e,
            #         self.arms[arm_idx].ur5e_joints[i],
            #         targetValue=current_joints[i],
            #         targetVelocity=0,
            #     )
            self._pb.setJointMotorControlArray(
                bodyIndex=self.arms[arm_idx].ur5e,
                jointIndices=self.arms[arm_idx].ur5e_joints,
                controlMode=pb.POSITION_CONTROL,
                targetPositions=current_joints,
                positionGains=np.ones(len(self.arms[arm_idx].ur5e_joints)),
            )
        self._pb.stepSimulation()
        self._pb.stepSimulation()

    def detect_collisions_of_robots(self, arm_idx):
        points = self._pb.getContactPoints(bodyA=self.arms[arm_idx].ur5e, bodyB=self.arms[1 - arm_idx].ur5e)
        if len(points) > 0:
            print(Fore.RED + f"Warning: Found collisions between two arms.")
            return True
        points = self._pb.getContactPoints(bodyA=self.arms[arm_idx].ee, bodyB=self.arms[1 - arm_idx].ee)
        if len(points) > 0:
            print(Fore.RED + f"Warning: Found collisions between two grippers.")
            return True
        points1 = self._pb.getContactPoints(bodyA=self.arms[arm_idx].ur5e, bodyB=self.arms[1 - arm_idx].ee)
        points2 = self._pb.getContactPoints(bodyA=self.arms[arm_idx].ee, bodyB=self.arms[1 - arm_idx].ur5e)
        if len(points1) > 0 or len(points2) > 0:
            print(Fore.RED + f"Warning: Found collisions between arm and gripper.")
            return True
        if self.arms[arm_idx].activated:
            points1 = self._pb.getContactPoints(
                bodyA=self.arms[arm_idx].grasped_obj_id, bodyB=self.arms[1 - arm_idx].ur5e
            )
            points2 = self._pb.getContactPoints(
                bodyA=self.arms[arm_idx].grasped_obj_id, bodyB=self.arms[1 - arm_idx].ee
            )
            if len(points1) > 0 or len(points2) > 0:
                print(Fore.RED + f"Warning: Found collisions between arm and graped object.")
                return True
        if self.arms[1 - arm_idx].activated:
            points1 = self._pb.getContactPoints(
                bodyA=self.arms[1 - arm_idx].grasped_obj_id, bodyB=self.arms[arm_idx].ur5e
            )
            points2 = self._pb.getContactPoints(
                bodyA=self.arms[1 - arm_idx].grasped_obj_id, bodyB=self.arms[arm_idx].ee
            )
            if len(points1) > 0 or len(points2) > 0:
                print(Fore.RED + f"Warning: Found collisions between arm and graped object.")
                return True
        other_contact = self.is_down_move_and_detect_obstacle(1 - arm_idx)
        this_contact = self.is_down_move_and_detect_obstacle(arm_idx)
        return other_contact or this_contact

    def move_joints(
        self,
        arm_idx,
        target_joints,
        speed=0.01,
        timeout=Maximum_Timeout,
        check_collision=False,
        center_control_override=False,
    ):
        """Move UR5e to target joint configuration."""
        t0 = time.time()
        while (time.time() - t0) < timeout:
            if check_collision:
                # two arms collision checking
                if self.detect_collisions_of_robots(arm_idx):
                    return False
                # if the other arm is in single-thread mode
                if self.center_control and not center_control_override:
                    print(Fore.GREEN + f"{arm_idx} enters center control")
                    while self.center_control:
                        time.sleep(0.1)
                    print(Fore.GREEN + f"{arm_idx} {self.arms[arm_idx].control_state} after center control")
                    if self.arms[arm_idx].control_state == "restart":
                        self.arms[arm_idx].control_state = "free"
                        return False
                    self.arms[arm_idx].control_state = "free"
                # if self.arms[arm_idx].grasped_obj_id != -1:
                #     points = self._pb.getContactPoints(bodyA=self.arms[arm_idx].grasped_obj_id)
                #     if len(points) > 0:
                #         for point in points:
                #             if point[2] in self.obj_ids["rigid"]:
                #                 print(Fore.RED + f"Warning: Found collisions between objects in move_joints.")
                #                 return False

            current_joints = np.array(
                [self._pb.getJointState(self.arms[arm_idx].ur5e, i)[0] for i in self.arms[arm_idx].ur5e_joints]
            )
            pos, _ = self.get_link_pose(self.arms[arm_idx].ee, self.arms[arm_idx].ee_tip_id)
            if pos[2] < 0.005:
                print(Fore.RED + f"Warning: move_joints tip height is {pos[2]}. Skipping.")
                return False
            diff_joints = target_joints - current_joints
            if all(np.abs(diff_joints) < 1e-2):
                # give time to stop
                for _ in range(5):
                    self._pb.stepSimulation()
                return True

            # Move with constant velocity
            norm = np.linalg.norm(diff_joints)
            v = diff_joints / norm if norm > 0 else 0
            step_joints = current_joints + v * speed
            self._pb.setJointMotorControlArray(
                bodyIndex=self.arms[arm_idx].ur5e,
                jointIndices=self.arms[arm_idx].ur5e_joints,
                controlMode=pb.POSITION_CONTROL,
                targetPositions=step_joints,
                positionGains=np.ones(len(self.arms[arm_idx].ur5e_joints)),
            )
            self._pb.stepSimulation()
        print(Fore.RED + f"Warning: move_joints exceeded {timeout} second timeout. Skipping.")
        return False

    def move_ee_pose(
        self, arm_idx, pose, speed=0.01, check_collision=False, center_control_override=False, is_handoff=False
    ):
        """Move UR5e to target end effector pose."""
        target_joints = self.solve_ik(arm_idx, pose, is_handoff=is_handoff)
        return self.move_joints(
            arm_idx,
            target_joints,
            speed,
            timeout=Maximum_Timeout,
            check_collision=check_collision,
            center_control_override=center_control_override,
        )

    def solve_ik(self, arm_idx, pose, is_handoff=False):
        """Calculate joint configuration with inverse kinematics."""
        lowerLimits = [-6.283, -6.283, -3.141, -6.283, -6.283, -6.283]
        upperLimits = [6.283, 6.283, 3.141, 6.283, 6.283, 6.283]
        jointRanges = [12.566, 12.566, 6.282, 12.566, 12.566, 12.566]
        if is_handoff:
            lowerLimits[0] = -4
            upperLimits[0] = -lowerLimits[0]
            jointRanges[0] = abs(lowerLimits[0]) + abs(upperLimits[0])
        joints = self._pb.calculateInverseKinematics(
            bodyUniqueId=self.arms[arm_idx].ur5e,
            endEffectorLinkIndex=self.arms[arm_idx].ur5e_ee_id,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=lowerLimits,
            upperLimits=upperLimits,
            jointRanges=jointRanges,
            restPoses=np.float32(self.arms_ik_rest_joints[arm_idx]).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        joints = np.array(joints, dtype=np.float32)
        # joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def straight_move(
        self,
        arm_idx,
        pose0,
        pose1,
        rot,
        speed=0.01,
        max_force=300,
        detect_force=False,
        is_push=False,
        check_collision=False,
        center_control_override=False,
        is_handoff=False,
    ):
        """Move every 1 cm, keep the move in a straight line instead of a curve. Keep level with rot"""
        step_distance = 0.01  # every 1 cm
        vec = pose1 - pose0
        length = np.linalg.norm(vec)
        vec = vec / length
        n_push = np.int32(np.floor(length / step_distance))  # every 1 cm
        success = True
        for n in range(n_push):
            target = pose0 + vec * n * step_distance
            success = self.move_ee_pose(
                arm_idx,
                (target, rot),
                speed,
                check_collision,
                center_control_override=center_control_override,
                is_handoff=is_handoff,
            )
            if not success:
                return False
            if detect_force:
                force = np.sum(
                    np.abs(np.array(self._pb.getJointState(self.arms[arm_idx].ur5e, self.arms[arm_idx].ur5e_ee_id)[2]))
                )
                if force > max_force:
                    target = target - vec * 2 * step_distance
                    self.move_ee_pose(
                        arm_idx,
                        (target, rot),
                        speed,
                        check_collision,
                        center_control_override=center_control_override,
                        is_handoff=is_handoff,
                    )
                    print(f"Force is {force}, exceed the max force {max_force}")
                    return False
        success = self.move_ee_pose(
            arm_idx,
            (pose1, rot),
            speed / 2,
            check_collision,
            center_control_override=center_control_override,
            is_handoff=is_handoff,
        )
        if not success:
            return False
        if is_push:
            speed /= 5
        success = self.move_ee_pose(
            arm_idx,
            (pose1, rot),
            speed,
            check_collision,
            center_control_override=center_control_override,
            is_handoff=is_handoff,
        )
        return success

    def straight_down_move(
        self,
        arm_idx,
        pose,
        rot,
        step_distance=0.001,
        check_collision=False,
        center_control_override=False,
        is_placing=False,
    ):
        """Move every 1 cm, keep the move in a straight line instead of a curve. Keep level with rot"""
        # time.sleep(1.0)
        step_distance = step_distance * 0.1
        target = pose.copy()
        success = True
        while not self.detect_contact(arm_idx) and success:
            target[2] -= step_distance
            success &= self.move_ee_pose(
                arm_idx,
                (target, rot),
                speed=step_distance,
                check_collision=check_collision,
                center_control_override=center_control_override,
            )
            if is_placing and self.down_move_detect_obstacle(arm_idx):
                success = False

        print(Fore.GREEN + f"{arm_idx} down move {success}")

        return success, target

    def straight_x_move(self, arm_idx, pose, rot, step_distance=0.001, center_control_override=False, is_handoff=False):
        """Move every 1 cm, keep the move in a straight line instead of a curve. Keep level with rot"""
        step_distance = step_distance / 10
        target = pose.copy()
        success = True
        while not self.detect_contact(arm_idx) and success:
            target[0] -= step_distance
            success &= self.move_ee_pose(
                arm_idx,
                (target, rot),
                speed=abs(step_distance),
                center_control_override=center_control_override,
                is_handoff=is_handoff,
            )

        return success, target

    def push(self, arm_idx, pose0, pose1, speed=0.002):
        """Execute pushing primitive.

        Args:
            pose0: SE(3) starting pose.
            pose1: SE(3) ending pose.
            speed: the speed of the planar push.

        Returns:
            success: robot movement success if True.
        """

        print(f"Push from {pose0} to {pose1}")

        # close the gripper
        self.close_gripper(arm_idx, is_slow=False)

        # Adjust push start and end positions.
        pos0 = np.array(pose0, dtype=np.float32)
        pos1 = np.array(pose1, dtype=np.float32)
        pos0[2] += self.ee_tip_offset
        pos1[2] += self.ee_tip_offset
        vec = pos1 - pos0
        length = np.linalg.norm(vec)
        vec = vec / length
        over0 = np.array((pos0[0], pos0[1], pos0[2] + 0.05))
        over0h = np.array((pos0[0], pos0[1], pos0[2] + 0.2))
        over1 = np.array((pos1[0], pos1[1], pos1[2] + 0.05))
        over1h = np.array((pos1[0], pos1[1], pos1[2] + 0.2))

        # Align against push direction.
        theta = np.arctan2(vec[1], vec[0]) + np.pi / 2
        rot = pb.getQuaternionFromEuler([np.pi / 2, np.pi / 2, theta])

        # Execute push.
        success = self.move_joints(arm_idx, self.arms_ik_rest_joints[arm_idx])
        if success:
            success = self.move_ee_pose(arm_idx, (over0h, rot))
        if success:
            success = self.straight_move(arm_idx, over0h, over0, rot, detect_force=True)
        if success:
            success = self.straight_move(arm_idx, over0, pos0, rot, detect_force=True)
        if success:
            success = self.straight_move(arm_idx, pos0, pos1, rot, speed, detect_force=True, is_push=True)
        if success:
            success = self.straight_move(arm_idx, pos1, over1, rot, speed)
        if success:
            success = self.straight_move(arm_idx, over1, over1h, rot)
        self.go_home(arm_idx)

        print(f"Arm {arm_idx} pushes from {pose0} to {pose1}, {success}")

        return success

    def grasp(self, arm_idx, pose, angle, speed=0.005, check_collision=False, center_control_override=False):
        """Execute grasping primitive.

        Args:
            pose: SE(3) grasping pose.
            angle: rotation angle

        Returns:
            success: robot movement success if True.
        """

        # Adjust grasp positions.
        pos = np.array(pose, dtype=np.float32)
        pos[2] = max(pos[2] - 0.04, self.bounds[2][0])
        pos[2] += self.ee_tip_offset + self.buffer_height
        over = pos.copy()
        over[2] += self.buffer_height

        # Align against grasp direction.
        angle = (angle % np.pi) - np.pi / 2
        rot = pb.getQuaternionFromEuler([np.pi / 2, np.pi / 2, -angle])

        # Execute grasp.
        success = True

        if self.is_at_home(arm_idx):
            success = self.move_joints(
                arm_idx,
                self.arms_first_joints[arm_idx],
                speed=speed,
                check_collision=check_collision,
                center_control_override=center_control_override,
            )
        if success:
            success = self.move_ee_pose(
                arm_idx,
                (over, rot),
                speed=speed,
                check_collision=check_collision,
                center_control_override=center_control_override,
            )
        if success:
            success = self.straight_move(
                arm_idx,
                over,
                pos,
                rot,
                speed=speed,
                check_collision=check_collision,
                center_control_override=center_control_override,
            )
        if success:
            success, curr_pos = self.straight_down_move(
                arm_idx,
                pos,
                rot,
                step_distance=Manipulation_Speed,
                check_collision=check_collision,
                center_control_override=center_control_override,
            )
            curr_pos = self.get_link_pose(self.arms[arm_idx].ur5e, self.arms[arm_idx].ur5e_ee_id)[0]
            curr_pos = np.asarray(curr_pos)
        if success:
            time.sleep(Manipulation_Wait)
            self.activate(arm_idx)
            success = self.straight_move(
                arm_idx,
                curr_pos,
                over,
                rot,
                speed=Manipulation_Speed,
                check_collision=check_collision,
                center_control_override=center_control_override,
            )
            success &= self.check_grasp(arm_idx)
        # print(f"Arm {arm_idx} grasps at {pose}, the grasp {success}")

        return success

    def place(self, arm_idx, pose, angle, speed=0.005, check_collision=False, center_control_override=False):

        # Align against grasp direction.
        angle = (angle % np.pi) - np.pi / 2
        rot = pb.getQuaternionFromEuler([np.pi / 2, np.pi / 2, -angle])

        pos = np.array(pose, dtype=np.float32)
        pos[2] = max(pos[2] - 0.04, self.bounds[2][0])
        pos[2] += self.ee_tip_offset + self.buffer_height
        over = pos.copy()
        over[2] += self.buffer_height

        success = True
        curr_pos = self.get_link_pose(self.arms[arm_idx].ur5e, self.arms[arm_idx].ur5e_ee_id)[0]
        curr_pos = np.asarray(curr_pos)
        if curr_pos[2] < over[2]:
            curr_over = curr_pos.copy()
            curr_over[2] = over[2]
            success = self.straight_move(
                arm_idx,
                curr_pos,
                curr_over,
                rot,
                speed=speed,
                check_collision=check_collision,
                center_control_override=center_control_override,
            )
        if success:
            success = self.move_ee_pose(
                arm_idx,
                (over, rot),
                speed=speed,
                check_collision=check_collision,
                center_control_override=center_control_override,
            )
        if success:
            success = self.straight_move(
                arm_idx,
                over,
                pos,
                rot,
                speed=speed,
                check_collision=check_collision,
                center_control_override=center_control_override,
            )
        if success:
            success, curr_pos = self.straight_down_move(
                arm_idx,
                pos,
                rot,
                step_distance=Manipulation_Speed,
                check_collision=check_collision,
                center_control_override=center_control_override,
                is_placing=True,
            )
            curr_pos = self.get_link_pose(self.arms[arm_idx].ur5e, self.arms[arm_idx].ur5e_ee_id)[0]
            curr_pos = np.asarray(curr_pos)
        if success:
            time.sleep(Manipulation_Wait)
            self.release(arm_idx)
            success = self.straight_move(
                arm_idx,
                curr_pos,
                over,
                rot,
                speed=Manipulation_Speed,
                check_collision=check_collision,
                center_control_override=center_control_override,
            )
        return success

    def is_at_home(self, arm_idx):
        current_joints = np.array(
            [self._pb.getJointState(self.arms[arm_idx].ur5e, i)[0] for i in self.arms[arm_idx].ur5e_joints]
        )
        diff_joints = self.arms_home_joints[arm_idx] - current_joints
        if all(np.abs(diff_joints) < 1e-2):
            return True
        else:
            return False

    def pick_and_place(
        self,
        arm_idx,
        pose,
        angle,
        place_pose,
        place_angle,
        speed=0.005,
        check_collision=False,
        center_control_override=False,
    ):
        """Execute grasping primitive.

        Args:
            pose: SE(3) grasping pose.
            angle: rotation angle

        Returns:
            success: robot movement success if True.
        """

        print(f"Arm {arm_idx} picks at {pose}, and places at {place_pose}")

        # Adjust grasp positions.
        pos = np.array(pose, dtype=np.float32)
        pos[2] = max(pos[2] - 0.04, self.bounds[2][0])
        pos[2] += self.ee_tip_offset + self.buffer_height
        over = pos.copy()
        over[2] += self.buffer_height
        # add more height to a later robot, it may help
        # if (
        #     self.get_link_pose(self.arms[arm_idx].ee, self.arms[arm_idx].ee_tip_id)[0]
        #     > self.get_link_pose(self.arms[1 - arm_idx].ee, self.arms[1 - arm_idx].ee_tip_id)[0]
        # ):
        #     over[2] += self.buffer_height * 2

        # Align against grasp direction.
        angle = (angle % np.pi) - np.pi / 2
        rot = pb.getQuaternionFromEuler([np.pi / 2, np.pi / 2, -angle])

        success = True

        if self.is_at_home(arm_idx):
            success = self.move_joints(
                arm_idx,
                self.arms_first_joints[arm_idx],
                speed=speed,
                check_collision=check_collision,
                center_control_override=center_control_override,
            )

        # Execute grasp.
        if not self.arms[arm_idx].activated:
            if success:
                success = self.move_ee_pose(
                    arm_idx,
                    (over, rot),
                    speed=speed,
                    check_collision=check_collision,
                    center_control_override=center_control_override,
                )
            if success:
                success = self.straight_move(
                    arm_idx,
                    over,
                    pos,
                    rot,
                    speed=speed,
                    check_collision=check_collision,
                    center_control_override=center_control_override,
                )
            if success:
                success, curr_pos = self.straight_down_move(
                    arm_idx,
                    pos,
                    rot,
                    step_distance=Manipulation_Speed,
                    check_collision=check_collision,
                    center_control_override=center_control_override,
                )
                curr_pos = self.get_link_pose(self.arms[arm_idx].ur5e, self.arms[arm_idx].ur5e_ee_id)[0]
                curr_pos = np.asarray(curr_pos)
            if success:
                time.sleep(Manipulation_Wait)
                self.activate(arm_idx)
                success = self.straight_move(
                    arm_idx,
                    curr_pos,
                    over,
                    rot,
                    speed=Manipulation_Speed,
                    check_collision=check_collision,
                    center_control_override=center_control_override,
                )
                success &= self.check_grasp(arm_idx)
        else:
            success = True
            curr_pos = self.get_link_pose(self.arms[arm_idx].ur5e, self.arms[arm_idx].ur5e_ee_id)[0]
            curr_pos = np.asarray(curr_pos)
            if curr_pos[2] < over[2]:
                curr_over = curr_pos.copy()
                curr_over[2] = over[2]
                success = self.straight_move(
                    arm_idx,
                    curr_pos,
                    curr_over,
                    rot,
                    speed=speed,
                    check_collision=check_collision,
                    center_control_override=center_control_override,
                )

        # Execute place.
        if success:
            pos = np.array(place_pose, dtype=np.float32)
            pos[2] = max(pos[2] - 0.04, self.bounds[2][0])
            pos[2] += self.ee_tip_offset + self.buffer_height
            over = pos.copy()
            over[2] += self.buffer_height
            # last_over = pos.copy()
            # last_over[2] += 0.02
            # add more height to a later robot, it may help
            # if (
            #     self.get_link_pose(self.arms[arm_idx].ee, self.arms[arm_idx].ee_tip_id)[0]
            #     > self.get_link_pose(self.arms[1 - arm_idx].ee, self.arms[1 - arm_idx].ee_tip_id)[0]
            # ):
            #     over[2] += self.buffer_height * 2
            success = self.move_ee_pose(
                arm_idx,
                (over, rot),
                speed=speed,
                check_collision=check_collision,
                center_control_override=center_control_override,
            )
        if success:
            success = self.straight_move(
                arm_idx,
                over,
                pos,
                rot,
                speed=speed,
                check_collision=check_collision,
                center_control_override=center_control_override,
            )
        if success:
            success, curr_pos = self.straight_down_move(
                arm_idx,
                pos,
                rot,
                step_distance=speed,
                check_collision=check_collision,
                center_control_override=center_control_override,
                is_placing=True,
            )
            curr_pos = self.get_link_pose(self.arms[arm_idx].ur5e, self.arms[arm_idx].ur5e_ee_id)[0]
            curr_pos = np.asarray(curr_pos)
        if success:
            time.sleep(Manipulation_Wait)
            self.release(arm_idx)
            success = self.straight_move(
                arm_idx,
                curr_pos,
                over,
                rot,
                speed=speed,
                check_collision=check_collision,
                center_control_override=center_control_override,
            )

        return success

    def activate(self, arm_idx):
        """Simulate suction using a rigid fixed constraint to contacted object."""

        if not self.arms[arm_idx].activated:
            points = self._pb.getContactPoints(bodyA=self.arms[arm_idx].ee, linkIndexA=0)
            # print(points)
            if points:

                # Handle contact between suction with a rigid object.
                for point in points:
                    obj_id, contact_link = point[2], point[4]
                if obj_id in self.obj_ids["rigid"]:
                    body_pose = self._pb.getLinkState(self.arms[arm_idx].ee, 0)
                    obj_pose = self._pb.getBasePositionAndOrientation(obj_id)
                    world_to_body = self._pb.invertTransform(body_pose[0], body_pose[1])
                    obj_to_body = self._pb.multiplyTransforms(
                        world_to_body[0], world_to_body[1], obj_pose[0], obj_pose[1]
                    )
                    self.arms[arm_idx].contact_constraint = self._pb.createConstraint(
                        parentBodyUniqueId=self.arms[arm_idx].ee,
                        parentLinkIndex=0,
                        childBodyUniqueId=obj_id,
                        childLinkIndex=contact_link,
                        jointType=pb.JOINT_FIXED,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=obj_to_body[0],
                        parentFrameOrientation=obj_to_body[1],
                        childFramePosition=(0, 0, 0),
                        childFrameOrientation=(0, 0, 0, 1),
                    )
                    self._pb.changeConstraint(self.arms[arm_idx].contact_constraint, maxForce=50000)
                    print(Fore.GREEN + f"{arm_idx} activated")

                    self.arms[arm_idx].activated = True
                    self.arms[arm_idx].grasped_obj_id = obj_id

    def release(self, arm_idx):
        """Release gripper object, only applied if gripper is 'activated'.
        If suction off, detect contact between gripper and objects.
        If suction on, detect contact between picked object and other objects.
        To handle deformables, simply remove constraints (i.e., anchors).
        Also reset any relevant variables, e.g., if releasing a rigid, we
        should reset init_grip values back to None, which will be re-assigned
        in any subsequent grasps.
        """
        if self.arms[arm_idx].activated:
            self.arms[arm_idx].activated = False
            self.arms[arm_idx].grasped_obj_id = -1

            # Release gripped rigid object (if any).
            if self.arms[arm_idx].contact_constraint is not None:
                try:
                    self._pb.removeConstraint(self.arms[arm_idx].contact_constraint)
                    self.arms[arm_idx].contact_constraint = None
                    print(Fore.GREEN + f"{arm_idx} released")
                except:
                    print("Warning: release wrong...")
                    pass

    def detect_contact(self, arm_idx):
        """Detects a contact with something."""
        body, link = self.arms[arm_idx].ee, 0
        if self.arms[arm_idx].activated and self.arms[arm_idx].contact_constraint is not None:
            try:
                info = self._pb.getConstraintInfo(self.arms[arm_idx].contact_constraint)
                body, link = info[2], info[3]
            except:
                self.arms[arm_idx].contact_constraint = None
                pass

        # Get all contact points between the suction and a rigid body.
        points = self._pb.getContactPoints(bodyA=body, linkIndexA=link)
        # print(points)
        # exit()
        if self.arms[arm_idx].activated:
            points = [point for point in points if point[2] != self.arms[arm_idx].ee]

        # # We know if len(points) > 0, contact is made with SOME rigid item.
        if points:
            return True

        return False

    def is_down_move_and_detect_obstacle(self, arm_idx):
        """Detects a contact with a rigid object other than grasped one during placing."""
        if self.arms[arm_idx].activated:
            info = self._pb.getConstraintInfo(self.arms[arm_idx].contact_constraint)
            body, link = info[2], info[3]

            points = self._pb.getContactPoints(bodyA=body, linkIndexA=link)
            for point in points:
                if point[2] in self.obj_ids["rigid"]:
                    print(Fore.RED + f"Warning: Found collisions between objects.")
                    return True

            return False
        else:
            return False

    def down_move_detect_obstacle(self, arm_idx):
        """Detects a contact with a rigid object other than grasped one during placing."""
        if not self.arms[arm_idx].activated:
            return False

        info = self._pb.getConstraintInfo(self.arms[arm_idx].contact_constraint)
        body, link = info[2], info[3]

        points = self._pb.getContactPoints(bodyA=body, linkIndexA=link)
        for point in points:
            if point[2] in self.obj_ids["rigid"]:
                print(Fore.RED + f"Warning: Found collisions between objects.")
                return True

        return False

    def check_grasp(self, arm_idx):
        """Check a grasp (object in contact?) for picking success."""

        suctioned_object = None
        if self.arms[arm_idx].contact_constraint is not None:
            suctioned_object = self._pb.getConstraintInfo(self.arms[arm_idx].contact_constraint)[2]
        return suctioned_object is not None

    ############## Handoff ##############
    def handoff(self, arm_idx1, arm_idx2, speed=0.02):
        """Execute grasping primitive.

        Args:
            pose: SE(3) grasping pose.
            angle: rotation angle

        Returns:
            success: robot movement success if True.
        """
        # mp.set_start_method("spawn")

        # compute arm1 handoff pose.
        arm1_pos = np.array(handoff_position, dtype=np.float32)
        if arm_idx1 == 0:
            arm1_pos[0] += 0.02
            arm1_pos[0] -= self.ee_tip_offset + self.buffer_height
            arm1_preparation_pos = arm1_pos.copy()
            arm1_preparation_pos[0] -= 0.1
        else:
            arm1_pos[0] -= 0.02
            arm1_pos[0] += self.ee_tip_offset + self.buffer_height
            arm1_preparation_pos = arm1_pos.copy()
            arm1_preparation_pos[0] += 0.1

        # compute arm2 handoff pose.
        arm2_pos = np.array(handoff_position, dtype=np.float32)
        if arm_idx2 == 0:
            arm2_pos[0] += 0.02
            arm2_pos[0] -= self.ee_tip_offset + self.buffer_height
            arm2_preparation_pos = arm2_pos.copy()
            arm2_preparation_pos[0] -= 0.1
        else:
            arm2_pos[0] -= 0.02
            arm2_pos[0] += self.ee_tip_offset + self.buffer_height
            arm2_preparation_pos = arm2_pos.copy()
            arm2_preparation_pos[0] += 0.1

        # thread 1 for passing arm
        robot1_thread = Thread(
            target=self.pre_hand_off_deliver, args=(arm_idx1, arm1_preparation_pos), kwargs={"speed": speed}
        )
        robot1_thread.daemon = True
        robot1_thread.start()

        # thread 2 for arm2
        robot2_thread = Thread(
            target=self.pre_hand_off_receive, args=(arm_idx2, arm2_preparation_pos), kwargs={"speed": speed}
        )
        robot2_thread.daemon = True
        robot2_thread.start()

        robot1_thread.join()
        robot2_thread.join()

        # arm1 go to handoff pose
        print("Both robots ready to handoff")
        success = self.straight_move(
            arm_idx1,
            arm1_preparation_pos,
            arm1_pos,
            pb.getQuaternionFromEuler(handoff_euler[arm_idx1]),
            speed=speed,
            detect_force=True,
            is_handoff=True,
        )

        # arm2 go to handoff pose
        if success:
            print("robot" + str(arm_idx1) + " is at handoff pose")
            success = self.straight_move(
                arm_idx2,
                arm2_preparation_pos,
                arm2_pos,
                pb.getQuaternionFromEuler(handoff_euler[arm_idx2]),
                speed=speed,
                detect_force=True,
                is_handoff=True,
            )

        # pass object and retrieve object, assume arm_idx2 move toward arm_idx1
        if success:
            if arm_idx2 == 1:
                dist = speed
            else:
                dist = -speed
            self.straight_x_move(
                arm_idx2,
                arm2_pos,
                pb.getQuaternionFromEuler(handoff_euler[arm_idx2]),
                step_distance=dist,
                is_handoff=True,
            )
            print("robot" + str(arm_idx2) + " is at handoff pose")
            self.activate(arm_idx2)
            self.release(arm_idx1)
            # time.sleep(0.1)
            # self.close_gripper(arm_idx2)
            # time.sleep(0.1)
            # self.open_gripper(arm_idx1)

        # thread 1 for arm1
        # Withdraw
        robot1_thread = Thread(
            target=self.post_hand_off_deliver, args=(arm_idx1, arm1_pos, arm1_preparation_pos), kwargs={"speed": speed}
        )
        robot1_thread.daemon = True
        robot1_thread.start()

        # thread 2 for arm2
        # Withdraw
        robot2_thread = Thread(
            target=self.post_hand_off_receive, args=(arm_idx2, arm2_pos, arm2_preparation_pos), kwargs={"speed": speed}
        )
        robot2_thread.daemon = True
        robot2_thread.start()

        robot1_thread.join()
        robot2_thread.join()
        return True

    def pre_hand_off_deliver(self, arm_idx, preparation_pose, speed=0.005):

        # arm1 go to preparation pose
        success = self.move_ee_pose(
            arm_idx, (preparation_pose, pb.getQuaternionFromEuler(handoff_euler[arm_idx])), speed=speed, is_handoff=True
        )
        return success

    def pre_hand_off_receive(self, arm_idx, preparation_pose, speed=0.005):

        success = self.move_ee_pose(
            arm_idx, (preparation_pose, pb.getQuaternionFromEuler(handoff_euler[arm_idx])), speed=speed, is_handoff=True
        )
        return success

    def post_hand_off_deliver(self, arm_idx, handoff_pose, preparation_pose, speed=0.005):

        success = self.straight_move(
            arm_idx,
            handoff_pose,
            preparation_pose,
            pb.getQuaternionFromEuler(handoff_euler[arm_idx]),
            speed,
            detect_force=True,
        )

        return success

    def post_hand_off_receive(self, arm_idx, handoff_pose, preparation_pose, speed=0.005):

        success = self.straight_move(
            arm_idx,
            handoff_pose,
            preparation_pose,
            pb.getQuaternionFromEuler(handoff_euler[arm_idx]),
            speed,
            detect_force=True,
        )

        return success


@dataclass
class Arm:
    ur5e: int
    ur5e_ee_id: int
    ur5e_joints: list
    ee: int
    ee_tip_id: int
    activated: bool = False
    contact_constraint: int = -1
    grasped_obj_id: int = -1
    reward_text_id: int = -1
    reward: int = 0
    control_state: str = "free"


if __name__ == "__main__":
    # Setup environment
    env = Environment()
    env.reset()
    # env.setup_rearrangement_task("test-cases/test-3-3-2.txt")
    # mcts_helper = MCTSHelper(env, "logs_grasp/snapshot-post-020000.reinforcement.pth")

    # # Get images
    # color_image, depth_image, mask_image = utils.get_true_heightmap(env)
    # num_obj = 1
    # for i in np.unique(mask_image):
    #     if i in env.obj_ids["rigid"]:
    #         mask_image[mask_image == i] = num_obj
    #         num_obj += 1
    #     else:
    #         mask_image[mask_image == i] = 0
    # mask_image = mask_image.astype(np.uint8)

    # # Process images
    # size = 210
    # bound = round(size / 2)
    # num_cell = 3
    # cell_size = round(size / num_cell)
    # offset = round((IMAGE_SIZE - size) / 2)
    # color_names = ["red", "green", "blue"]
    # color_lowers = {"red": red_lower, "green": green_lower, "blue": blue_lower}
    # color_uppers = {"red": red_upper, "green": green_upper, "blue": blue_upper}
    # crop_color_image = color_image[
    #     offset : (IMAGE_SIZE - offset), offset : (IMAGE_SIZE - offset), :
    # ]
    # crop_color_image = cv2.cvtColor(crop_color_image, cv2.COLOR_RGB2HSV)
    # crop_mask_image = mask_image[offset : (IMAGE_SIZE - offset), offset : (IMAGE_SIZE - offset)]
    # processed_input = []
    # for im_y in range(num_cell):
    #     for im_x in range(num_cell):
    #         cell_mask_image = crop_mask_image[
    #             im_y * cell_size : (im_y + 1) * cell_size, im_x * cell_size : (im_x + 1) * cell_size
    #         ]
    #         cell_color_image = crop_color_image[
    #             im_y * cell_size : (im_y + 1) * cell_size,
    #             im_x * cell_size : (im_x + 1) * cell_size,
    #             :,
    #         ]
    #         obj_ids = np.unique(cell_mask_image)
    #         cell_info = []
    #         for obj_id in obj_ids:
    #             # background
    #             if obj_id == 0:
    #                 continue
    #             # check color
    #             for name in color_names:
    #                 color_mask = cv2.inRange(
    #                     cell_color_image, color_lowers[name], color_uppers[name]
    #                 )
    #                 if np.sum(color_mask == 255) > 100:
    #                     cell_info.append(name)
    #                     continue
    #         processed_input.append(cell_info)
    # print(processed_input)

    # # Process output
    # # example: arm0 move green in [0, 0] to [2, 1]
    # arm_idx = 0
    # start_cell = [0, 0]
    # end_cell = [2, 1]
    # color_name = "green"
    # # pick
    # cell_mask = np.zeros_like(mask_image)
    # cell_mask[
    #     offset + start_cell[0] * cell_size : offset + (start_cell[0] + 1) * cell_size,
    #     offset + start_cell[1] * cell_size : offset + (start_cell[1] + 1) * cell_size,
    # ] = 255
    # color_mask = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
    # color_mask = cv2.inRange(color_mask, color_lowers[color_name], color_uppers[color_name])
    # mask = np.logical_and(cell_mask == 255, color_mask == 255).astype(np.uint8)
    # mask[mask > 0] = 255
    # for mask_id in np.unique(mask_image[mask == 255]):
    #     single_mask = mask.copy()
    #     single_mask[mask_image != mask_id] = 0
    #     best_pix_ind, grasp_predictions, grasp_q_value = mcts_helper.get_grasp_action(
    #         0.8,
    #         arm_idx,
    #         color_image,
    #         depth_image,
    #         single_mask,
    #         post_checking=True,
    #     )
    #     if np.max(grasp_q_value) > 0.8:
    #         best_rotation_angle = np.deg2rad(best_pix_ind[0] * (360.0 / NUM_ROTATION))
    #         primitive_position = [
    #             best_pix_ind[1] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
    #             best_pix_ind[2] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
    #             depth_image[best_pix_ind[1]][best_pix_ind[2]] + WORKSPACE_LIMITS[2][0],
    #         ]
    #         # success = env.grasp(arm_idx, primitive_position, best_rotation_angle)
    #         break
    # # place
    # if len(processed_input[end_cell[0] * num_cell + end_cell[1]]) == 0:
    #     place_best_pix_ind = [
    #         offset
    #         + end_cell[0] * cell_size
    #         + 17,  # TODO: now assume the grasp pos is the center of object
    #         offset + end_cell[1] * cell_size + 17,
    #     ]
    #     place_best_rotation_angle = best_rotation_angle
    #     place_primitive_position = [
    #         place_best_pix_ind[0] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
    #         place_best_pix_ind[1] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
    #         primitive_position[2] + 0.01,
    #     ]
    # else:
    #     cell_mask = np.zeros_like(mask_image)
    #     cell_mask[
    #         offset + end_cell[0] * cell_size : offset + (end_cell[0] + 1) * cell_size,
    #         offset + end_cell[1] * cell_size : offset + (end_cell[1] + 1) * cell_size,
    #     ] = 255
    #     mask = np.logical_and(cell_mask == 255, mask_image == 0).astype(np.uint8)
    #     mask[mask > 0] = 255
    #     mask = 255 - mask
    #     most_freespace = 0
    #     place_imx, place_imy = -1, -1
    #     for im_y in range(mask.shape[0]):  # TODO: do not iterate over the whole image, or use numba
    #         for im_x in range(mask.shape[1]):
    #             if mask[im_y, im_x] == 0:
    #                 overlap = np.sum(mask[im_y - 12 : im_y + 13, im_x - 12 : im_x + 13] == 0)
    #                 if overlap > most_freespace:
    #                     most_freespace = overlap
    #                     place_imx = im_x
    #                     place_imy = im_y
    #     place_best_pix_ind = [
    #         place_imy,  # TODO: now assume the grasp pos is the center of object
    #         place_imx,
    #     ]
    #     place_best_rotation_angle = best_rotation_angle
    #     place_primitive_position = [
    #         place_best_pix_ind[0] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
    #         place_best_pix_ind[1] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
    #         primitive_position[2] + 0.01,
    #     ]

    # env.pick_and_place(
    #     arm_idx,
    #     primitive_position,
    #     best_rotation_angle,
    #     place_primitive_position,
    #     place_best_rotation_angle,
    # )

    # time.sleep(10)

    # env.move_joints(np.array([0.5, 0, 0.2, 0]))
    # time.sleep(1)

    # env.move_joints(np.array([0.5, 0, 0.2, 1.0]))
    # time.sleep(1)

    # env.go_home()
    # time.sleep(1)

    # push_start = [4.280000000000000471e-01, -3.400000000000000244e-02, 0.01]
    # push_end = [5.020000000000000018e-01, -2.400000000000000244e-02, 0.01]
    # env.push(push_start, push_end)
    # time.sleep(1)
    # env.setup_competitive_task("./test-cases/3-1.txt")

    for i in range(16):
        # best_rotation_angle = np.deg2rad(90 - i * (360.0 / 16))
        # primitive_position = [0.6, 0, 0.01]
        # primitive_position_end = [
        #     primitive_position[0] + 0.1 * np.cos(best_rotation_angle),
        #     primitive_position[1] + 0.1 * np.sin(best_rotation_angle),
        #     0.01,
        # ]
        # env.push(0, primitive_position, primitive_position_end)
        # env._pb.addUserDebugLine(primitive_position, primitive_position_end, lifeTime=0)

        angle = np.deg2rad(i * 360 / 16)
        pos = [0.5, 0, 0.01]
        env.grasp(0, pos, angle)
