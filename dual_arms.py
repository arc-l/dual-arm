import curobo
import os
os.environ['CUROBO_TORCH_CUDA_GRAPH_RESET'] = '1'
import random
from itertools import permutations, product
from packaging import version

import numpy as np
import time
from typing import List
import helper
import torch
def is_cuda_graph_reset_available():
    reset_cuda_graph = os.environ.get("CUROBO_TORCH_CUDA_GRAPH_RESET")
    if reset_cuda_graph is not None:
        if bool(int(reset_cuda_graph)):
            if version.parse(torch.version.cuda) >= version.parse("12.0"):
                return True
        if not bool(int(reset_cuda_graph)):
            return False
    return False
assert is_cuda_graph_reset_available() == True

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Cuboid, Material
from curobo.rollout.rollout_base import Goal
from curobo.geom.sphere_fit import SphereFitType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, MotionGenResult, PoseCostMetric
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig


# toppra
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo

# bullet
import pybullet
import pybullet_data
from pybullet_utils import bullet_client

# DualArmTAMP
from task import Task, TaskPlan, TaskState, GripperState, TaskExecutionResult, ArmState, GripperAction


class DualArmEnv:
    def __init__(self, 
                 task: Task,
                 headless: bool = False,
                 batch_size: int = 3,
                 constrain_grasp_approach: bool = False,
                 num_buffers: int = 20,
                 ):
        self.headless = headless
        self.task = task
        self.num_buffers = num_buffers
        self.spheres = None

        setup_curobo_logger("warn")
        self.tensor_args = TensorDeviceType()
        self.robot_cfg = load_yaml(join_path(get_robot_configs_path(), "dual_ur5e_gripper.yml"))["robot_cfg"]
        gripper_cfg = load_yaml(join_path(get_robot_configs_path(), "gripper/dual_ur5e_gripper.yml"))["gripper_cfg"]
        self.gripper_joint_names = gripper_cfg["gripper_joint_names"]
        self.gripper_grasp_orientations = gripper_cfg["grasp_orientation"]
        self.gripper_multiplier = np.array(gripper_cfg["multiplier"])
        self.gripper_primary_joints = list(gripper_cfg["mimic_joints"].keys())
        self.gripper_primary_indices = []
        self.gripper_mimic_joints_settings = gripper_cfg["mimic_joints"]
        self.gripper_mimic_joint_names = []
        self.gripper_closed = {}
        self.gripper_primary_joints_dict  = {}
        self.gripper_primary_indices_dict = {}       
        
        self.arm_joint_names = []
        for k, v in self.gripper_mimic_joints_settings.items():
            self.gripper_mimic_joint_names.extend(v["joint_names"])
        self.j_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        self.j_names_primary = [name for name in self.j_names if name not in self.gripper_mimic_joint_names]
        print("j_names_primary = ", self.j_names_primary)
        self.default_config = self.robot_cfg["kinematics"]["cspace"]["retract_config"]

        self.objects_curobo = []
        self.objects_pybullet = []
        world_cfg_table = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        )
        world_cfg_table.cuboid[0].pose[2] -= 0.02
        self.world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid)
        self.add_objects_curobo()
        obs = self.objects_curobo + world_cfg_table.cuboid
        # self.world_cfg_curobo = WorldConfig(cuboid=obs, mesh=world_cfg1.mesh)
        self.world_cfg_curobo = WorldConfig(cuboid=obs)
        self.max_velocity = 0.314 * 5
        self.interpolation_dt = 0.02
        self.simulation_dt = 1/200

        single_motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg_curobo,
            self.tensor_args,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
            # collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            interpolation_dt=self.interpolation_dt,
            collision_cache={"obb": 2 * len(obs), "mesh": 2 * len(obs)},
            collision_activation_distance=0.025,
            fixed_iters_trajopt=True,
            maximum_trajectory_dt=0.5,
            num_ik_seeds= 200,
            num_graph_seeds= 64,
            num_trajopt_seeds= 64,
            num_batch_ik_seeds= 64,
            trajopt_tsteps = 32,
            # ik_opt_iters=1000,
            # rotation_threshold=0.5,
            # position_threshold=0.5,
            finetune_trajopt_iters=500,
            high_precision=True,
        )

        self.single_motion_gen = MotionGen(single_motion_gen_config)

        self.ik_solver_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg_curobo,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=200,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
            collision_cache={"obb": len(obs), "mesh": len(obs)},
            regularization=True,
            high_precision=True,
        )
    
        self.ik_solver = IKSolver(self.ik_solver_config)
        # print("warming up...")
        # self.single_motion_gen.warmup(enable_graph=True)

        # print("warming up...")
        # self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False, batch=65)
        self.ee_link_names = self.single_motion_gen.kinematics.link_names
        self.arm1 = self.single_motion_gen.kinematics.ee_link
        self.arm2 = [link for link in self.ee_link_names if link != self.arm1][0]
        print(f"End effector link names: {self.ee_link_names}")
        self.grid_range = 0.2
        self.grid_num = batch_size

        print("Curobo is Ready")
        
        pose_metric = None
        if constrain_grasp_approach:
            pose_metric = PoseCostMetric.create_grasp_approach_metric(
                offset_position=0.2, tstep_fraction=0.8
            )
        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=4,
            max_attempts=10,
            # time_dilation_factor=0.5,
            pose_cost_metric=pose_metric,
            ik_fail_return=100,
        )

        self.single_plan_config = MotionGenPlanConfig(
            enable_graph=True,
            enable_graph_attempt=60,
            max_attempts=60,
            time_dilation_factor=0.5,
            pose_cost_metric=pose_metric,
            ik_fail_return=300,
            num_ik_seeds=200,
            # fail_on_invalid_query=False,
        )

        self.gripper_joint_indices = []

        self.recorded_plan = []
        self.old_task_plan = None
        self.pre_grasp_height = 2 * self.task.object_size
        
        self.preferred_angles = [901, 900]
        self.preferred_angles_combination = list(product(self.preferred_angles, repeat=2))
        self.all_angles = [901, 900, 450, -450, 451, -451]
        self.all_angles_combination = list(product(self.all_angles, repeat=2))
        # the current joint states. Used for headless planning
        self.current_js = JointState.from_position(self.tensor_args.to_device([self.default_config]), self.j_names)
        self.current_js = self.current_js.get_ordered_joint_state(self.j_names_primary)

        # pybullet
        if not self.headless:
            self.sim_step = 0
            self._pb = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            self._client_id = self._pb._client
            self._pb.setAdditionalSearchPath(pybullet_data.getDataPath())
            assets_path = os.path.dirname(os.path.abspath(""))
            self._pb.setAdditionalSearchPath(assets_path)
            self._pb.setAdditionalSearchPath(pybullet_data.getDataPath())
            self._pb.setTimeStep(self.simulation_dt)
            self._pb.configureDebugVisualizer(self._pb.COV_ENABLE_GUI, 0)
            self._pb.configureDebugVisualizer(self._pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            self._pb.configureDebugVisualizer(self._pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            self._pb.configureDebugVisualizer(self._pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

            # target = self._pb.getDebugVisualizerCamera()
            self._pb.resetDebugVisualizerCamera(
                cameraDistance=1.2,
                cameraYaw=0,
                cameraPitch=-45,
                cameraTargetPosition=[0, 0, 0],
            )

        # end of init
        print()

    
    def add_objects_curobo(self):
        starts = self.task.get_starts()
        colors = self.task.get_colors()
        for i in range(self.task.num_objects):
            start = starts[i]
            cube = Cuboid(
                name=f"object_{i}",
                pose=[start[0], start[1], start[2], 1, 0, 0, 0],
                dims=[self.task.object_size, self.task.object_size, self.task.object_size],
                color=[colors[i][0], colors[i][1], colors[i][2], 1]
            )
            self.objects_curobo.append(cube)
    
    def add_objects_pybullet(self):
        starts = self.task.get_starts()
        goals = self.task.get_goals()
        colors = self.task.get_colors()
        goal_height = 0.001
        goal_scale = 1.1
        for i in range(self.task.num_objects):
            object_shape_start = self._pb.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[self.task.object_size/2, self.task.object_size/2, self.task.object_size/2])
            object_visual_start = self._pb.createVisualShape(pybullet.GEOM_BOX, halfExtents=[self.task.object_size/2, self.task.object_size/2, self.task.object_size/2], rgbaColor=[colors[i][0], colors[i][1], colors[i][2], 1])
            object_id_start = self._pb.createMultiBody(0.01, object_shape_start, object_visual_start, basePosition=starts[i])
            self._pb.changeDynamics(object_id_start, -1, lateralFriction=2.0)
            object_visual_goal = self._pb.createVisualShape(pybullet.GEOM_BOX, halfExtents=[self.task.object_size/2 * goal_scale, self.task.object_size/2 * goal_scale, goal_height], rgbaColor=[colors[i][0], colors[i][1], colors[i][2], 0.8])
            object_id_goal = self._pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=object_visual_goal, basePosition=[goals[i][0], goals[i][1], goal_height])
            self.objects_pybullet.append(object_id_start)
            
    def reset(self):
        kin_state = self.single_motion_gen.kinematics.get_state(self.single_motion_gen.get_retract_config().view(1, -1))
        link_poses = kin_state.link_poses
        self.default_ee_pos = {}
        for k, v in link_poses.items():
            t_pos = np.ravel(v.to_list())
            self.default_ee_pos[k] = t_pos[:3]
            self.default_ee_pos[k][2] += 0.15 #TODO: Test this, added 0.15m to z
        
        self.gripper_closed = {self.arm1: False, self.arm2: False}
        #TODO: This kind assignment is risky in the future. Fix it later
        self.gripper_primary_joints_dict = {self.arm1: self.gripper_primary_joints[0], self.arm2: self.gripper_primary_joints[1]}
        self.gripper_primary_indices_dict = {self.arm1: None, self.arm2: None}

        self.arm_joint_names = self.single_motion_gen.kinematics.joint_names


        # pybullet
        if not self.headless:
            self._pb.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
            self._pb.setRealTimeSimulation(0)
            self._pb.setGravity(0, 0, -9.8)

            # Add robot.
            self._pb.loadURDF("plane.urdf", [0, 0, -0.0005])
            self.robot_id = self._pb.loadURDF("assets/ur5e/dual_ur5e_gripper.urdf", [0, 0, 0], flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL, useFixedBase=True)
            self.all_joint_info = [self._pb.getJointInfo(self.robot_id, i) for i in range(self._pb.getNumJoints(self.robot_id))]
            self.joint_ids_all_bullet = [info[0] for info in self.all_joint_info]
            # self.joint_ids_all_bullet = [j[0] for j in self.joint_ids_all_bullet if j[2] == pybullet.JOINT_REVOLUTE]
            self.joint_ids = [self._pb.getJointInfo(self.robot_id, i)[0] for i in range(self._pb.getNumJoints(self.robot_id)) if self._pb.getJointInfo(self.robot_id, i)[2] == pybullet.JOINT_REVOLUTE]
            self.j_names_bullet = [self._pb.getJointInfo(self.robot_id, i)[1].decode("utf-8") for i in range(self._pb.getNumJoints(self.robot_id))]
            self.j_names_to_id = dict(zip(self.j_names_bullet, self.joint_ids_all_bullet))
            self.j_id_to_names = dict(zip(self.joint_ids_all_bullet, self.j_names_bullet))
            self.arm_joints_id_bullet = [self.j_names_to_id[name] for name in self.arm_joint_names]
            self.arm_joint_names_bullet = [self.j_id_to_names[i] for i in self.arm_joints_id_bullet]
            self.gripper_primary_indices = [self.j_names_to_id[name] for name in self.gripper_primary_joints]
            self.gripper_primary_indices_dict[self.arm1] = self.j_names_to_id[self.gripper_primary_joints[0]]
            self.gripper_primary_indices_dict[self.arm2] = self.j_names_to_id[self.gripper_primary_joints[1]]
            self.gripper_pads_id_bullet = {self.arm1: [], self.arm2: []}
            self.ee_name_to_id = {self.arm1: None, self.arm2: None}
            self.attached_constraints = {self.arm1: None, self.arm2: None}
            for i in range(self._pb.getNumJoints(self.robot_id)):
                joint_info = self.all_joint_info[i]
                link_name = joint_info[12].decode("utf-8")
                if "finger_pad" in link_name:
                    if "_1" in link_name:
                        self.gripper_pads_id_bullet[self.arm2].append(i)
                    else:
                        self.gripper_pads_id_bullet[self.arm1].append(i)
                
                if link_name == self.arm1:
                    self.ee_name_to_id[self.arm1] = i
                if link_name == self.arm2:
                    self.ee_name_to_id[self.arm2] = i
            self.gripper_mimic_joints_id_dict = {}
            for gripper_primary_name, settings in self.gripper_mimic_joints_settings.items():
                gripper_primary_id = self.j_names_to_id[gripper_primary_name]
                self.gripper_mimic_joints_id_dict[gripper_primary_id] = {}
                for i in range(len(settings["joint_names"])):
                    mimic_joint_name = settings["joint_names"][i]
                    multiplier = settings["multiplier"][i]
                    mimic_joint_id = self.j_names_to_id[mimic_joint_name]
                    self.gripper_mimic_joints_id_dict[gripper_primary_id][mimic_joint_id] = multiplier
            
            # Move robot to home configuration.
            for i in range(len(self.joint_ids)):
                self._pb.resetJointState(self.robot_id, self.joint_ids[i], self.default_config[i])
            
            #TODO: Debug the constraints if we have time
            # c = pybullet.createConstraint(self.robot_id, self.j_names_to_id['left_inner_knuckle_joint'], self.robot_id, self.j_names_to_id["left_inner_finger_joint"], pybullet.JOINT_POINT2POINT, [0, 0, 0], [0, -0.014, 0.043], [0, -0.034, 0.021])
            # c = pybullet.createConstraint(self.robot_id, self.j_names_to_id['right_inner_knuckle_joint'], self.robot_id, self.j_names_to_id["right_inner_finger_joint"], pybullet.JOINT_POINT2POINT, [0, 0, 0], [0, -0.014, 0.043], [0, -0.034, 0.021])
            # c = pybullet.createConstraint(self.robot_id, self.j_names_to_id['left_inner_knuckle_joint_1'], self.robot_id, self.j_names_to_id["left_inner_finger_joint_1"], pybullet.JOINT_POINT2POINT, [0, 0, 0], [0, -0.014, 0.043], [0, -0.034, 0.021])
            # c = pybullet.createConstraint(self.robot_id, self.j_names_to_id['right_inner_knuckle_joint_1'], self.robot_id, self.j_names_to_id["right_inner_finger_joint_1"], pybullet.JOINT_POINT2POINT, [0, 0, 0], [0, -0.014, 0.043], [0, -0.034, 0.021])
            # for gripper_primary_name, settings in self.gripper_mimic_joints_settings.items():
            #     for i in range(len(settings["joint_names"])):
            #         mimic_joint_name = settings["joint_names"][i]
            #         multiplier = settings["multiplier"][i]
            #         print(f"gripper_primary_name = {gripper_primary_name}, mimic_joint_name = {mimic_joint_name}, multiplier = {multiplier}")
            #         constraint_id = self._pb.createConstraint(
            #             parentBodyUniqueId=self.robot_id,
            #             parentLinkIndex=self.j_names_to_id[gripper_primary_name],
            #             childBodyUniqueId=self.robot_id,
            #             childLinkIndex=self.j_names_to_id[mimic_joint_name],
            #             jointType=pybullet.JOINT_GEAR,
            #             jointAxis=[0, 1, 0],
            #             parentFramePosition=[0, 0, 0],
            #             childFramePosition=[0, 0, 0]
            #         )
            #         self._pb.setJointMotorControl2(
            #             bodyUniqueId=self.robot_id,
            #             jointIndex=self.j_names_to_id[mimic_joint_name],
            #             controlMode=pybullet.VELOCITY_CONTROL,
            #             targetVelocity=0,
            #             force=0
            #         )
            #         self._pb.changeConstraint(
            #             constraint_id,
            #             gearRatio=multiplier,
            #             maxForce=10000,
            #             erp=0.2
            #         )

            # Add workspace.
            plane_shape = self._pb.createCollisionShape(
                pybullet.GEOM_BOX, halfExtents=[0.3, 0.3, 0.001])
            plane_visual = self._pb.createVisualShape(
                pybullet.GEOM_BOX, halfExtents=[0.3, 0.3, 0.001])
            plane_id = self._pb.createMultiBody(
                0, plane_shape, plane_visual, basePosition=[0, 0, 0])
            self._pb.changeVisualShape(
                plane_id, -1, rgbaColor=[0.2, 0.2, 0.2, 1.0])

            # Add objects. You have to add the objects later than the robot and the workspace, or it can be moved by the robot arm acciedentally.
            self.add_objects_pybullet()
            for _ in range(200):
                self._pb.stepSimulation()
            print("reset finished")
        self.sync_isaacsim_curobo()
        print("self.single_motion_gen.world_coll_checker.get_obstacle_names() = ", self.single_motion_gen.world_coll_checker.get_obstacle_names())

    
    def step(self, cnt=1, sleep=True):
        if self.headless:
            return
        self.gripper_update()
        for i in range(cnt):
            self._pb.stepSimulation()
        if sleep:
            time.sleep(self.interpolation_dt)

    def gripper_update(self):
        target_joint_states = []
        target_joint_ids = []
        for gripper_primary_id, gripper_mimic_joints in self.gripper_mimic_joints_id_dict.items(): 
            curr_joint_state = self._pb.getJointState(self.robot_id, gripper_primary_id)
            for mimic_joint_id, multiplier in gripper_mimic_joints.items():
                target_joint_ids.append(mimic_joint_id)
                target_joint_states.append(curr_joint_state[0] * multiplier)
        self._pb.setJointMotorControlArray(
            self.robot_id,
            target_joint_ids,
            pybullet.POSITION_CONTROL,
            targetPositions=target_joint_states,
            positionGains=[1] * len(target_joint_ids),
        )
    
    def grasp_width(self, arm):
        pad1 = np.array(self._pb.getLinkState(self.robot_id, self.gripper_pads_id_bullet[arm][0])[0])
        pad2 = np.array(self._pb.getLinkState(self.robot_id, self.gripper_pads_id_bullet[arm][1])[0])
        dist = np.linalg.norm(pad1 - pad2) - 0.14 
        return dist 
        
    def gripper_control(self, gripper1_action: GripperAction, gripper2_action: GripperAction):
        target_velocities = [
            self.max_velocity if gripper1_action==GripperAction.CLOSE else -self.max_velocity,
            self.max_velocity if gripper2_action==GripperAction.CLOSE else -self.max_velocity
        ]
        if self.gripper_closed[self.arm1] == bool(gripper1_action.value):
            target_velocities[0] = 0
        if self.gripper_closed[self.arm2] == bool(gripper2_action.value):
            target_velocities[1] = 0
        self.gripper_closed[self.arm1] = bool(gripper1_action.value)
        self.gripper_closed[self.arm2] = bool(gripper2_action.value)
        if self.headless:
            return

        self._pb.setJointMotorControlArray(
            self.robot_id,
            self.gripper_primary_indices,
            pybullet.VELOCITY_CONTROL,
            targetVelocities=target_velocities,
            forces=[10, 10],
        )

        old_positions = self._pb.getJointStates(self.robot_id, self.gripper_primary_indices)
        while True:
            self.gripper_update()
            self.step(1, sleep=True)
            new_positions = self._pb.getJointStates(self.robot_id, self.gripper_primary_indices)
            stop1 = False
            stop2 = False
            # we only care about closed ones
            if gripper1_action == GripperAction.CLOSE:
                if self.grasp_width(self.arm1) <= 1.1 * self.task.object_size and self.attached_constraints[self.arm1] is None:
                    obj = self.check_contact(self.arm1)
                    if obj is not None:
                        target_velocities[0] = 0
                        # create fixed constraint
                        pos, orn = self._pb.getLinkState(self.robot_id, self.ee_name_to_id[self.arm1])[:2]
                        obj_pos, obj_orn = self._pb.getBasePositionAndOrientation(obj)
                        parent_frame_pos, parent_frame_orn = self._pb.multiplyTransforms(
                            *self._pb.invertTransform(pos, orn),
                            obj_pos, obj_orn
                        )
                        constraint_id = self._pb.createConstraint(
                            parentBodyUniqueId=self.robot_id,
                            parentLinkIndex=self.ee_name_to_id[self.arm1],
                            childBodyUniqueId=obj,
                            childLinkIndex=-1,
                            jointType=pybullet.JOINT_FIXED,
                            jointAxis=[0, 0, 0],
                            parentFramePosition=parent_frame_pos,
                            parentFrameOrientation=parent_frame_orn,
                            childFramePosition=[0, 0, 0],
                            childFrameOrientation=[0, 0, 0],
                        )
                        self.attached_constraints[self.arm1] = constraint_id
                        print("Constraint created for arm1, obj=", obj)
                        stop1 = True
            elif gripper1_action == GripperAction.OPEN and self.attached_constraints[self.arm1] is not None:
                self._pb.removeConstraint(self.attached_constraints[self.arm1])
                self.attached_constraints[self.arm1] = None
                print("Constraint removed for arm1")
            if gripper2_action == GripperAction.CLOSE and self.attached_constraints[self.arm2] is None:
                if self.grasp_width(self.arm2) <= 1.1 * self.task.object_size:
                    obj = self.check_contact(self.arm2)
                    if obj is not None:
                        target_velocities[1] = 0
                        # create fixed constraint
                        pos, orn = self._pb.getLinkState(self.robot_id, self.ee_name_to_id[self.arm2])[:2]
                        obj_pos, obj_orn = self._pb.getBasePositionAndOrientation(obj)
                        parent_frame_pos, parent_frame_orn = self._pb.multiplyTransforms(
                            *self._pb.invertTransform(pos, orn),
                            obj_pos, obj_orn
                        )
                        constraint_id = self._pb.createConstraint(
                            parentBodyUniqueId=self.robot_id,
                            parentLinkIndex=self.ee_name_to_id[self.arm2],
                            childBodyUniqueId=obj,
                            childLinkIndex=-1,
                            jointType=pybullet.JOINT_FIXED,
                            jointAxis=[0, 0, 0],
                            parentFramePosition=parent_frame_pos,
                            parentFrameOrientation=parent_frame_orn,
                            childFramePosition=[0, 0, 0],
                            childFrameOrientation=[0, 0, 0],
                        )
                        self.attached_constraints[self.arm2] = constraint_id
                        print("Constraint created for arm2, obj=", obj)
                        stop2 = True
            elif gripper2_action == GripperAction.OPEN and self.attached_constraints[self.arm2] is not None:
                self._pb.removeConstraint(self.attached_constraints[self.arm2])
                self.attached_constraints[self.arm2] = None
                print("Constraint removed for arm2")

            self._pb.setJointMotorControlArray(
                self.robot_id,
                self.gripper_primary_indices,
                pybullet.VELOCITY_CONTROL,
                targetVelocities=target_velocities,
                forces=[10, 10],
            )

            converged = all(np.linalg.norm(new_positions[i][0] - old_positions[i][0]) < 0.001 for i in range(len(old_positions)))
            if converged:
                print("Converged")
            if (gripper1_action == GripperAction.CLOSE and stop1 or gripper1_action == GripperAction.OPEN) and \
                (gripper2_action == GripperAction.CLOSE and stop2 or gripper2_action == GripperAction.OPEN) or\
                converged:
                break
            old_positions = new_positions
    
    def check_contact(self, arm):
        contact_points = self._pb.getContactPoints(bodyA=self.robot_id)
        for contact in contact_points:
            link_index = contact[3]  # Link index on robot
            other_body_id = contact[2]  # ID of the other body
            if link_index in self.gripper_pads_id_bullet[arm] and other_body_id != self.robot_id:
                return other_body_id 
        return None
    
    def check_proximity(self, arm):
        ee_pos = np.array(self._pb.getLinkState(self.robot_id, self.ee_name_to_id[arm])[0])
        tool_pos = np.array(self._pb.getLinkState(self.robot_id, self.gripper_pads_id_bullet[arm][0])[0])
        vec = (tool_pos - ee_pos) / np.linalg.norm((tool_pos - ee_pos))
        ee_targ = ee_pos + vec
        ray_data = self._pb.rayTest(ee_pos, ee_targ)[0]
        obj, link, ray_frac = ray_data[0], ray_data[1], ray_data[2]
        return obj, link, ray_frac
    
    def test_gripper_control(self):
        self.reset()
        for i in range(3):
            self.gripper_control(GripperAction.CLOSE, GripperAction.CLOSE)
            self.gripper_control(GripperAction.OPEN, GripperAction.OPEN)
        
        print("Gripper test 0 finished")
        time.sleep(2)
        
        for i in range(3):
            self.gripper_control(GripperAction.CLOSE, GripperAction.OPEN)
            self.gripper_control(GripperAction.OPEN, GripperAction.OPEN)
        
        print("Gripper test 1 finished")
        time.sleep(2)
        
        for i in range(3):
            self.gripper_control(GripperAction.OPEN, GripperAction.CLOSE)
            self.gripper_control(GripperAction.OPEN, GripperAction.OPEN)

        print("Gripper test 2 finished")
        time.sleep(2)
        
        for i in range(3):
            self.gripper_control(GripperAction.CLOSE, GripperAction.CLOSE)
            self.gripper_control(GripperAction.OPEN, GripperAction.OPEN)

        print("Gripper test 3 finished")
        time.sleep(2)
        
        for i in range(3):
            self.gripper_control(GripperAction.CLOSE, GripperAction.OPEN)
            self.gripper_control(GripperAction.OPEN, GripperAction.CLOSE)

        print("Gripper test 4 finished")
        time.sleep(2)
        
        self.gripper_control(GripperAction.OPEN, GripperAction.OPEN)

        
    
    @staticmethod
    def get_pose_grid(n_x, n_y, n_z, max_x, max_y, max_z):
        x = np.linspace(-max_x, max_x, n_x)
        y = np.linspace(-max_y, max_y, n_y)
        z = np.linspace(0, max_z, n_z)
        x, y, z = np.meshgrid(x, y, z, indexing="ij")

        position_arr = np.zeros((n_x * n_y * n_z, 3))
        position_arr[:, 0] = x.flatten()
        position_arr[:, 1] = y.flatten()
        position_arr[:, 2] = z.flatten()
        # add one row for [0, 0, 0]
        position_arr = np.vstack([np.zeros(3), position_arr])
        return position_arr
    
    def ik_solve_batch(self, ik_goal1, ik_goal2):
        link_poses = {}
        link_poses[self.arm2] = ik_goal2
        current_pose = self.get_current_arm_js().position.view(1, -1) # shape: 1 x dof
        # make it to batch size
        current_pose = current_pose.repeat(ik_goal1.position.shape[0], 1)
        seed = current_pose.unsqueeze(1)
        ik_results = self.ik_solver.solve_batch(
            goal_pose=ik_goal1,
            # retract_config=self.tensor_args.to_device(self.default_config),
            retract_config=current_pose,
            seed_config=seed,
            link_poses=link_poses,
        )
        print("ik_batch_solve time: ", ik_results.solve_time)
        print("succeeded: ", len(ik_results.success[ik_results.success]), " out of ", len(ik_results.success))
        return ik_results
    
    def get_grid_goal_poses(self, c_p1, c_p2, angle1=900, angle2=900, prioritize_arm1=True):
        c_rot1 = self.gripper_grasp_orientations[self.arm1][angle1]
        c_rot2 = self.gripper_grasp_orientations[self.arm2][angle2]
        position_offsets = self.tensor_args.to_device(self.get_pose_grid(self.grid_num, self.grid_num, self.grid_num, self.grid_range, self.grid_range, self.grid_range*2))
        ik_goal1 = Pose(
            position=self.tensor_args.to_device(c_p1),
            quaternion=self.tensor_args.to_device(c_rot1),
        )
        ik_goal1 = ik_goal1.repeat(position_offsets.shape[0])
        ik_goal2 = Pose(
            position=self.tensor_args.to_device(c_p2),
            quaternion=self.tensor_args.to_device(c_rot2),
        )
        ik_goal2 = ik_goal2.repeat(position_offsets.shape[0])
        if prioritize_arm1:
            ik_goal2.position += position_offsets
        else:
            ik_goal1.position += position_offsets
        return ik_goal1, ik_goal2

    def ik_solve_single(self, c_p1:np.ndarray, c_p2:np.ndarray|None, arm_angle1=900, arm_angle2:float|None=900, num_solutions=1):
        c_rot = self.gripper_grasp_orientations[self.arm1][arm_angle1]
        ik_goal = Pose(
            position=self.tensor_args.to_device(c_p1),
            quaternion=self.tensor_args.to_device(c_rot),
        )

        link_poses = None
        if c_p2 is not None:
            c_rot = self.gripper_grasp_orientations[self.arm2][arm_angle2]
            link_poses = {}
            link_poses[self.arm2] = Pose(
                position=self.tensor_args.to_device(c_p2),
                quaternion=self.tensor_args.to_device(c_rot),
            )
        
        current_pose = self.get_current_arm_js().position.view(1, -1)
        seed = current_pose.unsqueeze(0)
        ik_results = self.ik_solver.solve_single(
            goal_pose=ik_goal,
            # retract_config=self.tensor_args.to_device(self.default_config),
            return_seeds=num_solutions,
            retract_config=current_pose,
            seed_config=seed,
            link_poses=link_poses,
        )
        return ik_results


    def plan_single_from_curr_js(self, goal_arm1, goal_arm2, arm_angle1, arm_angle2):
        cu_js = self.get_current_arm_js()
        link_poses = {}

        c_rot = self.gripper_grasp_orientations[self.arm1][arm_angle1]
        ik_goal = Pose(
            position=self.tensor_args.to_device(goal_arm1),
            quaternion=self.tensor_args.to_device(c_rot),
        )

        c_rot = self.gripper_grasp_orientations[self.arm2][arm_angle2]

        link_poses[self.arm2] = Pose(
            position=self.tensor_args.to_device(goal_arm2),
            quaternion=self.tensor_args.to_device(c_rot),
        )
        result = self.single_motion_gen.plan_single(
            cu_js, ik_goal, self.single_plan_config.clone(), 
            link_poses=link_poses
        )
        return result
    
    def plan_single(self, task_plan: dict):
        if not self.simulation_app.is_running():
            raise Exception("Simulation is not running")

        cu_js = task_plan['start_js']
        link_poses = {}

        c_p = task_plan[self.arm1][1]
        c_rot = self.gripper_grasp_orientations[self.arm1][900]
        ik_goal = Pose(
            position=self.tensor_args.to_device(c_p),
            quaternion=self.tensor_args.to_device(c_rot),
        )

        c_p = task_plan[self.arm2][1]
        c_rot = self.gripper_grasp_orientations[self.arm2][900]

        link_poses[self.arm2] = Pose(
            position=self.tensor_args.to_device(c_p),
            quaternion=self.tensor_args.to_device(c_rot),
        )
        result = self.single_motion_gen.plan_single(
            cu_js, ik_goal, self.plan_config.clone(), 
            link_poses=link_poses
        )
        return result

    def get_task_execution_result(self, task_plan: TaskPlan):
        task_execution_result = TaskExecutionResult()
        task_execution_result.arm1_assigned_object = task_plan.arm1_assigned_object 
        task_execution_result.arm2_assigned_object = task_plan.arm2_assigned_object
        task_execution_result.arm1_gripper_state = GripperState(task_plan.arm1_gripper_action.value)
        task_execution_result.arm2_gripper_state = GripperState(task_plan.arm2_gripper_action.value)
        task_execution_result.arm1_angle = task_plan.arm1_angle
        task_execution_result.arm2_angle = task_plan.arm2_angle
        task_execution_result.arm1_task_state = TaskState.IN_PROGRESS if task_plan.arm1_to_start else TaskState.COMPLETED
        task_execution_result.arm2_task_state = TaskState.IN_PROGRESS if task_plan.arm2_to_start else TaskState.COMPLETED
        task_execution_result.buffer = task_plan.buffer
        return task_execution_result

    def motion_plan(self, task_plan: TaskPlan):
        needs_naive_plan = True #TODO: Remove this later
        if task_plan.arm1_assigned_object == -1 or task_plan.arm1_assigned_object is None:
            goal_arm1 = self.default_ee_pos[self.arm1].copy()
        else:
            goal_arm1 = self.task.get_start( task_plan.arm1_assigned_object ).copy() if task_plan.arm1_to_start else self.task.get_goal( task_plan.arm1_assigned_object ).copy()
        if task_plan.arm2_assigned_object == -1 or task_plan.arm2_assigned_object is None:
            goal_arm2 = self.default_ee_pos[self.arm2].copy()
        else:
            goal_arm2 = self.task.get_start( task_plan.arm2_assigned_object ).copy() if task_plan.arm2_to_start else self.task.get_goal( task_plan.arm2_assigned_object ).copy()
        
        # safety pre_grasp
        goal_arm1[2] += self.pre_grasp_height
        goal_arm2[2] += self.pre_grasp_height

        task_execution_result = self.get_task_execution_result(task_plan)

        if task_plan.arm1_assigned_object is not None and task_plan.arm2_assigned_object is not None and not needs_naive_plan:
            result = self.plan_single_from_curr_js(goal_arm1=goal_arm1, goal_arm2=goal_arm2, arm_angle1=task_plan.arm1_angle, arm_angle2=task_plan.arm2_angle)
            if not result.success:
                print("Failed to go to start position, using naive waypoints")
                needs_naive_plan = True
                self.check_object_collision_enabled()
            else:
                print("Going to the start position")
                cmd_plan = self.get_plan_single(result)

        if needs_naive_plan:
            self.naive_motion_plan(task_plan)
            result = self.plan_single_from_curr_js(goal_arm1=goal_arm1, goal_arm2=goal_arm2, arm_angle1=task_plan.arm1_angle, arm_angle2=task_plan.arm2_angle)
            if result.success:
                cmd_plan = self.get_plan_single(result)
                cmd_plan = self.toppra_improvement(cmd_plan, speed_scale=0.5, acc_scale=1)
                self.visualize_plan(cmd_plan)
                self.add_plan_to_record(cmd_plan)
                # go down
                arm1_delta_z = -self.pre_grasp_height if task_plan.arm1_assigned_object not in [None, -1] else 0
                arm2_delta_z = -self.pre_grasp_height if task_plan.arm2_assigned_object not in [None, -1] else 0
                cmd_plan = self.move_z_2(arm1_delta_z, arm2_delta_z, task_plan.arm1_angle, task_plan.arm2_angle)
                self.visualize_plan(cmd_plan)
                self.add_plan_to_record(cmd_plan)
                self.gripper_control(task_plan.arm1_gripper_action, task_plan.arm2_gripper_action)
                self.add_plan_to_record([task_plan.arm1_gripper_action, task_plan.arm2_gripper_action])
                # go up
                arm1_delta_z = self.pre_grasp_height if task_plan.arm1_assigned_object not in [None, -1] else 0
                arm2_delta_z = self.pre_grasp_height if task_plan.arm2_assigned_object not in [None, -1] else 0
                cmd_plan = self.move_z_2(arm1_delta_z, arm2_delta_z, task_plan.arm1_angle, task_plan.arm2_angle)
                self.visualize_plan(cmd_plan)
                self.add_plan_to_record(cmd_plan)
            else:
                # one arm by one arm
                #TODO: Check the order of the arms. If arm1's goal is dependent on arm2's goal, then we can't move arm1 first and vice versa.
                tmp_goal_arm1 = self.default_ee_pos[self.arm1]
                tmp_goal_arm2 = self.default_ee_pos[self.arm2]
                # move arm2 to goal, arm1 to retract pose
                result = self.plan_single_from_curr_js(goal_arm1=tmp_goal_arm1, goal_arm2=goal_arm2, arm_angle1=task_plan.arm1_angle, arm_angle2=task_plan.arm2_angle)
                cmd_plan = self.get_plan_single(result)
                cmd_plan = self.toppra_improvement(cmd_plan, speed_scale=0.5, acc_scale=1)
                self.visualize_plan(cmd_plan)
                self.add_plan_to_record(cmd_plan)

                # go down
                arm1_delta_z = 0
                arm2_delta_z = -self.pre_grasp_height if task_plan.arm2_assigned_object not in [None, -1] else 0
                cmd_plan = self.move_z_2(arm1_delta_z, arm2_delta_z, task_plan.arm1_angle, task_plan.arm2_angle)
                self.visualize_plan(cmd_plan)
                self.add_plan_to_record(cmd_plan)

                self.gripper_control(GripperAction(self.gripper_closed[self.arm1]), task_plan.arm2_gripper_action)
                self.add_plan_to_record([GripperAction(self.gripper_closed[self.arm1]), task_plan.arm2_gripper_action])
                # go up
                arm1_delta_z = 0
                arm2_delta_z = self.pre_grasp_height if task_plan.arm2_assigned_object not in [None, -1] else 0
                cmd_plan = self.move_z_2(arm1_delta_z, arm2_delta_z, task_plan.arm1_angle, task_plan.arm2_angle)
                self.visualize_plan(cmd_plan)
                self.add_plan_to_record(cmd_plan)

                # move the other arm
                result = self.plan_single_from_curr_js(goal_arm1=goal_arm1, goal_arm2=tmp_goal_arm2, arm_angle1=task_plan.arm1_angle, arm_angle2=task_plan.arm2_angle)
                cmd_plan = self.get_plan_single(result)
                cmd_plan = self.toppra_improvement(cmd_plan, speed_scale=0.5, acc_scale=1)
                self.visualize_plan(cmd_plan)
                self.add_plan_to_record(cmd_plan)
                # go down
                arm2_delta_z = 0
                arm1_delta_z = -self.pre_grasp_height if task_plan.arm2_assigned_object not in [None, -1] else 0
                cmd_plan = self.move_z_2(arm1_delta_z, arm2_delta_z, task_plan.arm1_angle, task_plan.arm2_angle)
                self.visualize_plan(cmd_plan)
                self.add_plan_to_record(cmd_plan)

                self.gripper_control(task_plan.arm1_gripper_action, GripperAction(self.gripper_closed[self.arm2]))
                self.add_plan_to_record([task_plan.arm1_gripper_action, GripperAction(self.gripper_closed[self.arm2])])
                # go up
                arm2_delta_z = 0
                arm1_delta_z = self.pre_grasp_height if task_plan.arm2_assigned_object not in [None, -1] else 0
                cmd_plan = self.move_z_2(arm1_delta_z, arm2_delta_z, task_plan.arm1_angle, task_plan.arm2_angle)
                self.visualize_plan(cmd_plan)
                self.add_plan_to_record(cmd_plan)

        return task_execution_result
    
    def naive_motion_plan(self, task_plan):
        new_angle1 = task_plan.arm1_angle
        new_angle2 = task_plan.arm2_angle
        ee_pos = self.get_current_ee_pos()
        pos1 = ee_pos[self.arm1]
        pos2 = ee_pos[self.arm2]

        old_angle1 = None
        old_angle2 = None

        angle1 = new_angle1
        angle2 = new_angle2

        if self.old_task_plan is not None:
            old_angle1 = self.old_task_plan.arm1_angle if self.old_task_plan is not None else None
            old_angle2 = self.old_task_plan.arm2_angle if self.old_task_plan is not None else None
            angle1 = old_angle1 if old_angle1 is not None else new_angle1
            angle2 = old_angle2 if old_angle2 is not None else new_angle2
            if old_angle1 in [450, -450] or old_angle2 in [450, -450]:
                match old_angle1:
                    case 450:
                        pos1[1] = max(pos1[1] - 0.3, -0.3)
                    case -450:
                        pos1[1] = min(pos1[1] + 0.3, 0.3)
                match old_angle2:
                    case 450:
                        pos2[1] = min(pos2[1] + 0.3, 0.3)
                    case -450:
                        pos2[1] = max(pos2[1] - 0.3, -0.3)
                
                cmd_plan = self.move_ee(pos1, self.old_task_plan.arm1_angle, pos2, self.old_task_plan.arm2_angle)
                self.visualize_plan(cmd_plan)
                self.add_plan_to_record(cmd_plan)

        # check left and right on the y-axis
        if 0 < pos2[0] - pos1[0] < 0.2:
            if 0 < pos1[1] - pos2[1] < 0.1:
                # arm1 on the left, separate them
                delta_y = 0.3 
                pos1[1] = min(pos1[1] + delta_y, 0.3)
                pos2[1] = max(pos2[1] - delta_y, -0.3)
                cmd_plan = self.move_ee(pos1, angle1, pos2, angle2)
                self.visualize_plan(cmd_plan)
                self.add_plan_to_record(cmd_plan)
                # cmd_plan_total = cmd_plan_total.stack(cmd_plan)
            elif -0.1 < pos1[1] - pos2[1] < 0:
                # arm2 on the left
                delta_y = 0.3 
                pos1[1] = max(pos1[1] - delta_y, -0.3)
                pos2[1] = min(pos2[1] + delta_y, 0.3)
                cmd_plan = self.move_ee(pos1, angle1, pos2, angle2)
                self.visualize_plan(cmd_plan)
                self.add_plan_to_record(cmd_plan)
            
        # move back a little bit to seperate them on x-axis
        if pos2[0] - pos1[0] < 0.3:
            delta_x = 0.3
            pos1[0] = max(pos1[0]-delta_x, -0.3)
            pos2[0] = min(pos1[0]+delta_x, 0.3)
            cmd_plan = self.move_ee(pos1, angle1, pos2, angle2)
            self.visualize_plan(cmd_plan)
            self.add_plan_to_record(cmd_plan)
        
        # align with the goal on the y-axis
        ee_pos = self.get_current_ee_pos()
        pos1 = ee_pos[self.arm1]
        pos2 = ee_pos[self.arm2]

        if task_plan.arm1_assigned_object == -1 or task_plan.arm1_assigned_object is None:
            goal_arm1 = self.default_ee_pos[self.arm1]
        else:
            goal_arm1 = self.task.get_start( task_plan.arm1_assigned_object ) if task_plan.arm1_to_start else self.task.get_goal( task_plan.arm1_assigned_object )
        if task_plan.arm2_assigned_object == -1 or task_plan.arm2_assigned_object is None:
            goal_arm2 = self.default_ee_pos[self.arm2]
        else:
            goal_arm2 = self.task.get_start( task_plan.arm2_assigned_object ) if task_plan.arm2_to_start else self.task.get_goal( task_plan.arm2_assigned_object )
        
        if angle1 in [450, -450] or angle2 in [450, -450]:
            match angle1:
                case 450:
                    pos1[1] = max(min(goal_arm1[1] - 0.3, self.get_current_ee_pos()[self.arm1][1]-0.3), -0.3)
                case -450:
                    pos1[1] = min(max(goal_arm1[1] + 0.3, self.get_current_ee_pos()[self.arm1][1]+0.3), 0.3)
            
            match angle2:
                case 450:
                    pos2[1] = min(max(goal_arm2[1] + 0.3, self.get_current_ee_pos()[self.arm2][1]+0.3), 0.3)
                case -450:
                    pos2[1] = max(min(goal_arm2[1] - 0.3, self.get_current_ee_pos()[self.arm2][1]-0.3), -0.3)

            if not (new_angle1 == old_angle1 and new_angle2 == old_angle2):
                cmd_plan = self.move_ee(pos1, new_angle1, pos2, new_angle2)
                self.visualize_plan(cmd_plan)
                self.add_plan_to_record(cmd_plan)
            
            # align with the goal on the x-axis
            if new_angle1 in [450, -450]:
                pos1[0] = goal_arm1[0]
            else:
                pos1[1] = goal_arm1[1]
            if new_angle2 in [450, -450]:
                pos2[0] = goal_arm2[0]
            else:
                pos2[1] = goal_arm2[1]
            
            cmd_plan = self.move_ee(pos1, new_angle1, pos2, new_angle2)
            self.visualize_plan(cmd_plan)
            self.add_plan_to_record(cmd_plan)

        else:
            pos1[1] = goal_arm1[1]
            pos2[1] = goal_arm2[1]

            cmd_plan = self.move_ee(pos1, new_angle1, pos2, new_angle2)
            self.visualize_plan(cmd_plan)
            self.add_plan_to_record(cmd_plan)
    
    def task_plan(self):
        task_plan = self.task.plan()
        if task_plan.start_goal_pairs is not None and not task_plan.use_buffer:
            print("Pick two from the start_goal_pairs")
            self.find_available_task(task_plan)
        elif task_plan.use_buffer:
            print("Using buffer for one arm")
            self.find_available_task_with_buffers(task_plan)
        elif task_plan.start_goal_pairs is None:
            print("Start and Goal pairs are None")
            if task_plan.arm2_assigned_object is not None and task_plan.arm1_assigned_object == -1:
                print("Finishing up for one arm2")
                if task_plan.arm2_angle is None:
                    self.find_angle_for_single_arm(task_plan, self.arm2)
            elif task_plan.arm1_assigned_object is not None and task_plan.arm2_assigned_object == -1:  
                print("Finishing up for one arm1")
                if task_plan.arm1_angle is None:
                    self.find_angle_for_single_arm(task_plan, self.arm1)
            # elif (not task_plan.arm1_to_start) and (not task_plan.arm2_to_start):
            else:
                print("Finishing up for both arms")
        return task_plan

    def plan(self):
        self.reset()
        while not self.task.finished():
            task_plan = self.task_plan()
            task_execution_result = self.motion_plan(task_plan)
            self.old_task_plan = task_plan
            idx_to_sync = list(range(self.task.num_objects))
            sync_after_open = False
            sync_before_close = False
            if task_plan.arm1_gripper_action == GripperAction.OPEN and task_plan.arm1_assigned_object != -1:
                idx_to_sync.remove(task_plan.arm1_assigned_object)
                sync_after_open = True
            if task_plan.arm2_gripper_action == GripperAction.OPEN:
                idx_to_sync.remove(task_plan.arm2_assigned_object)
                sync_after_open = True
            if task_plan.arm1_gripper_action == GripperAction.CLOSE and task_plan.arm1_assigned_object != -1:
                sync_before_close = True
            if task_plan.arm2_gripper_action == GripperAction.CLOSE:
                sync_before_close = True
            
            if sync_before_close:
                print("Syncing IsaacSim and Curobo before close, idx_to_sync = ", idx_to_sync)
                # self.sync_isaacsim_curobo(idx_to_sync=idx_to_sync)
                self.sync_isaacsim_curobo()

            idx_to_sync = list(range(self.task.num_objects))
            #TODO: Do I need to be concerned that the attaching and detaching happens after the gripper control in self.motion_plan()?
            if task_plan.arm1_gripper_action == GripperAction.CLOSE and task_plan.arm1_assigned_object != -1:
                print(f"Attaching object {task_plan.arm1_assigned_object} to {self.arm1}")
                self.attach_obj(cu_js=self.get_current_arm_js(), cube_name=self.objects_curobo[task_plan.arm1_assigned_object].name, arm_name=self.arm1)
                idx_to_sync.remove(task_plan.arm1_assigned_object)
            if task_plan.arm2_gripper_action == GripperAction.CLOSE:
                print(f"Attaching object {task_plan.arm2_assigned_object} to {self.arm2}")
                self.attach_obj(cu_js=self.get_current_arm_js(), cube_name=self.objects_curobo[task_plan.arm2_assigned_object].name, arm_name=self.arm2)
                idx_to_sync.remove(task_plan.arm2_assigned_object)
            if task_plan.arm1_gripper_action == GripperAction.OPEN and task_plan.arm1_assigned_object != -1:
                print(f"Detaching object {task_plan.arm1_assigned_object} from {self.arm1}")
                self.detach_obj(cube_name=self.objects_curobo[task_plan.arm1_assigned_object].name, arm_name=self.arm1)
            if task_plan.arm2_gripper_action == GripperAction.OPEN:
                print(f"Detaching object {task_plan.arm2_assigned_object} from {self.arm2}")
                self.detach_obj(cube_name=self.objects_curobo[task_plan.arm2_assigned_object].name, arm_name=self.arm2)

            if sync_after_open:
                print("Syncing IsaacSim and Curobo after open, idx_to_sync = ", idx_to_sync)
                # self.sync_isaacsim_curobo(idx_to_sync=idx_to_sync)
                self.sync_isaacsim_curobo()

            self.check_object_collision_enabled()
            self.task.update(task_execution_result)
        return self.recorded_plan
    
    def plan_all(self):
        self.reset()
        task_plans = []
        while not self.task.finished():
            starts = self.task.get_starts().copy()
            task_plan = self.task_plan()
            task_plans.append(task_plan)
            task_execution_result = self.get_task_execution_result(task_plan)
            # we need to assume the motion planning has succeeded and update the start to goal in the curobo world_cfg
            if not task_plan.arm1_assigned_object == -1 or task_plan.arm1_assigned_object is None:
                goal_arm1 = self.task.get_start( task_plan.arm1_assigned_object ).copy() if task_plan.arm1_to_start else self.task.get_goal( task_plan.arm1_assigned_object ).copy()
                starts[task_plan.arm1_assigned_object] = goal_arm1
            if not task_plan.arm2_assigned_object == -1 or task_plan.arm2_assigned_object is None:
                goal_arm2 = self.task.get_start( task_plan.arm2_assigned_object ).copy() if task_plan.arm2_to_start else self.task.get_goal( task_plan.arm2_assigned_object ).copy()
                starts[task_plan.arm2_assigned_object] = goal_arm2
            
            #TODO: Do I need the attach and the detach here?
            idx_to_sync = list(range(self.task.num_objects))
            new_starts = {}
            cuboids = []
            for i in idx_to_sync:
                cube_curobo = self.objects_curobo[i]
                cube_curobo.pose = [starts[i][0], starts[i][1], starts[i][2], 1, 0, 0, 0]
                cuboids.append(cube_curobo)
                new_starts[i] = np.array([starts[i][0], starts[i][1], starts[i][2]])
            self.task.set_starts(new_starts)
            world_cfg_table = WorldConfig.from_dict(
                load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
            )
            world_cfg_table.cuboid[0].pose[2] -= 0.02
            self.world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid)
            obs = cuboids + world_cfg_table.cuboid
            self.world_cfg_curobo = WorldConfig(cuboid=obs)
            self.single_motion_gen.clear_world_cache()
            self.single_motion_gen.update_world(self.world_cfg_curobo)
            self.ik_solver.update_world(self.world_cfg_curobo)
            print("self.single_motion_gen.world_coll_checker.get_obstacle_names() = ", self.single_motion_gen.world_coll_checker.get_obstacle_names())
            self.task.update(task_execution_result)
        return task_plans

    def check_object_collision_enabled(self):
        for obj in self.objects_curobo:
            name = obj.name
            env_idx = 0
            obs_idx = self.single_motion_gen.world_coll_checker.get_obb_idx(name, env_idx)
            # print(f"{name} single_motion_gen collision enabled = ", self.single_motion_gen.world_coll_checker._cube_tensor_list[2][env_idx, obs_idx].cpu().numpy(), end=' ') 
        print()

    def get_current_js(self):
        #TODO: add back the function to get the joint states from pybullet
        cu_js = self.current_js.get_ordered_joint_state(self.arm_joint_names)
        return cu_js
        
    def visualize_plan(self, plan: JointState):
        if plan is None:
            print("Null JointState plan.")
            return
        if plan.shape[0] == 0:
            print("Empty JointState plan.")
            return
        cu_js = plan[-1].get_ordered_joint_state(self.j_names_primary)
        self.current_js = JointState.from_position(self.tensor_args.to_device(cu_js.position.reshape(1, -1)), self.j_names_primary)
        if self.headless:
            return 
        cmd_idx = 0
        while cmd_idx < len(plan.position):
            cmd_state = plan[cmd_idx]
            cu_js = cmd_state.get_ordered_joint_state(self.arm_joint_names)
            position = cu_js.position.view(-1).cpu().numpy()
            self.servoj(position)
            self.step(20, sleep=True)
            cmd_idx += 1

    def put_cubes_to_goal_to_solve_ik(self, ids: List[int]):
        #TODO: comeback here
        cuboids = []
        if self.old_task_plan is None:
            return
        for i in range(self.task.num_objects):
            cube_curobo = self.objects_curobo[i]
            if i in ids:
                p = self.task.get_goal(i)
            else:
                p = self.task.get_start(i)
            cube_curobo.pose = [p[0], p[1], p[2], 1, 0, 0, 0]
            cuboids.append(cube_curobo)
        world_cfg_table = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        )
        world_cfg_table.cuboid[0].pose[2] -= 0.02
        self.world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid)
        obs = cuboids + world_cfg_table.cuboid
        self.world_cfg_curobo = WorldConfig(cuboid=obs)
        self.ik_solver.update_world(self.world_cfg_curobo)
    
    def restore_ik_solver(self):
        cuboids = []
        if self.old_task_plan is None:
            return
        for i in range(self.task.num_objects):
            cube_curobo = self.objects_curobo[i]
            p = self.task.get_start(i)
            cube_curobo.pose = [p[0], p[1], p[2], 1, 0, 0, 0]
            cuboids.append(cube_curobo)
        world_cfg_table = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        )
        world_cfg_table.cuboid[0].pose[2] -= 0.02
        self.world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid)
        obs = cuboids + world_cfg_table.cuboid
        self.world_cfg_curobo = WorldConfig(cuboid=obs)
        self.ik_solver.update_world(self.world_cfg_curobo)

    def find_available_task(self, task_plan: TaskPlan):
        # first, find the avaiable task that uses vertical pick and place
        print("finding vertical pick and place")
        for combo in self.preferred_angles_combination:
            for indices in task_plan.start_goal_pairs:
                start1, goal1, start2, goal2 = self.task.get_start( indices[0] ), self.task.get_goal( indices[0] ), self.task.get_start( indices[1] ), self.task.get_goal( indices[1] )
                angle1, angle2 = combo
                start_ik_result = self.ik_solve_single(start1, start2, angle1, angle2)
                self.put_cubes_to_goal_to_solve_ik(indices)
                goal_ik_result = self.ik_solve_single(goal1, goal2, angle1, angle2)
                self.restore_ik_solver()
                if start_ik_result.success and goal_ik_result.success:
                    # [start_pos, goal_pos, orientation]
                    task_plan.arm1_assigned_object = indices[0]
                    task_plan.arm2_assigned_object = indices[1]
                    task_plan.arm1_angle = angle1
                    task_plan.arm2_angle = angle2
                    return 
        # if no solution with vertical pick and place, try other angles
        print("finding other angles")
        for combo in self.all_angles_combination:
            for indices in task_plan.start_goal_pairs:
                start1, goal1, start2, goal2 = self.task.get_start( indices[0] ), self.task.get_goal( indices[0] ), self.task.get_start( indices[1] ), self.task.get_goal( indices[1] )
                angle1, angle2 = combo
                start_ik_result = self.ik_solve_single(start1, start2, angle1, angle2)
                self.put_cubes_to_goal_to_solve_ik(indices)
                goal_ik_result = self.ik_solve_single(goal1, goal2, angle1, angle2)
                self.restore_ik_solver()
                if start_ik_result.success and goal_ik_result.success:
                    # [start_pos, goal_pos, orientation]
                    task_plan.arm1_assigned_object = indices[0]
                    task_plan.arm2_assigned_object = indices[1]
                    task_plan.arm1_angle = angle1
                    task_plan.arm2_angle = angle2
                    return 
        # if still no solution, do it one by one
        print("finding one by one")
        for combo in self.all_angles_combination:
            for indices in task_plan.start_goal_pairs:
                start1, goal1, start2, goal2 = self.task.get_start( indices[0] ), self.task.get_goal( indices[0] ), self.task.get_start( indices[1] ), self.task.get_goal( indices[1] )
                angle1, angle2 = combo
                default_ee_pos1 = self.default_ee_pos[self.arm1]
                default_ee_pos2 = self.default_ee_pos[self.arm2]
                start_ik_result1 = self.ik_solve_single(default_ee_pos1, start2, angle1, angle2)
                start_ik_result2 = self.ik_solve_single(start1, default_ee_pos2, angle1, angle2)
                # no need to put the cubes to the goal because only one cube is moved. No unpredicted collision
                goal_ik_result1 = self.ik_solve_single(default_ee_pos1, goal2, angle1, angle2)
                goal_ik_result2 = self.ik_solve_single(goal1, default_ee_pos2, angle1, angle2)
                if start_ik_result1.success and start_ik_result2.success and goal_ik_result1.success and goal_ik_result2.success:
                    task_plan.arm1_assigned_object = indices[0]
                    task_plan.arm2_assigned_object = indices[1]
                    task_plan.arm1_angle = angle1
                    task_plan.arm2_angle = angle2
                    return 
        raise RuntimeError("No available task found")
        

    def find_available_task_with_buffers(self, task_plan: TaskPlan):
        buffers = self.task.sample_buffers(self.num_buffers)
        for combo in self.preferred_angles_combination:
            for indices in task_plan.start_goal_pairs:
                goal1_dependent_on_goal2 = self.task.is_direct_dependent(indices[0], indices[1])
                goal2_dependent_on_goal1 = self.task.is_direct_dependent(indices[1], indices[0])
                start1, goal1, start2, goal2 = self.task.get_start( indices[0] ), self.task.get_goal( indices[0] ), self.task.get_start( indices[1] ), self.task.get_goal( indices[1] )
                angle1, angle2 = combo
                start_ik_result = self.ik_solve_single(start1, start2, angle1, angle2)
                if not start_ik_result.success:
                    continue
                ik_goal2 = Pose(
                    position=self.tensor_args.to_device(np.zeros(3)),
                    quaternion=self.tensor_args.to_device(self.gripper_grasp_orientations[self.arm2][angle2]),
                )
                ik_goal2 = ik_goal2.repeat(buffers.shape[0])
                ik_goal1 = Pose(
                    position=self.tensor_args.to_device(np.zeros(3)),
                    quaternion=self.tensor_args.to_device(self.gripper_grasp_orientations[self.arm1][angle1])
                )
                ik_goal1 = ik_goal1.repeat(buffers.shape[0])
                if goal1_dependent_on_goal2 and goal2_dependent_on_goal1:
                    if len(self.task.depends_on(indices[0])) > 1 and len(self.task.depends_on(indices[1])) == 1:
                        ik_goal1.position += self.tensor_args.to_device(buffers)
                        ik_goal2.position += self.tensor_args.to_device(goal2)
                    elif len(self.task.depends_on(indices[1])) > 1 and len(self.task.depends_on(indices[0])) == 1:
                        ik_goal1.position += self.tensor_args.to_device(goal1)
                        ik_goal2.position += self.tensor_args.to_device(buffers) 
                    else:
                        raise RuntimeError("Both goals has multiple dependencies")
                else:
                    if goal1_dependent_on_goal2:
                        ik_goal1.position += self.tensor_args.to_device(goal1)
                        ik_goal2.position += self.tensor_args.to_device(buffers) 
                    else:
                        ik_goal1.position += self.tensor_args.to_device(buffers)
                        ik_goal2.position += self.tensor_args.to_device(goal2)
                goal_ik_results = self.ik_solve_batch(ik_goal1, ik_goal2)
                success_indices = torch.where(goal_ik_results.success)[0]
                if len(success_indices) == 0:
                    continue

                # pick the one buffer that is furthest to all the objects
                available_buffers = buffers[success_indices.cpu().tolist()]
                object_positions = self.get_current_cube_pose()
                furthest_buffer = None
                max_min_dist = -1
                for buffer in available_buffers:
                    dists = [np.linalg.norm(buffer - obj_pos) for obj_pos in object_positions]
                    min_dist = min(dists)
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        furthest_buffer = buffer

                task_plan.buffer = {1: None, 2: None}
                if goal1_dependent_on_goal2 and goal2_dependent_on_goal1:
                    if len(self.task.depends_on(indices[0])) > 1 and len(self.task.depends_on(indices[1])) == 1:
                        task_plan.buffer[1] = furthest_buffer
                    elif len(self.task.depends_on(indices[1])) > 1 and len(self.task.depends_on(indices[0])) == 1:
                        task_plan.buffer[2] = furthest_buffer
                    else:
                        raise RuntimeError("Both goals has multiple dependencies")
                else:
                    if goal1_dependent_on_goal2:
                        task_plan.buffer[2] = furthest_buffer
                    else:
                        task_plan.buffer[1] = furthest_buffer
                task_plan.arm1_assigned_object = indices[0]
                task_plan.arm2_assigned_object = indices[1]
                task_plan.arm1_angle = angle1
                task_plan.arm2_angle = angle2
                print("arm1_assigned_object = ", task_plan.arm1_assigned_object, "arm2_assigned_object = ", task_plan.arm2_assigned_object)
                print("goal1_dependent_on_goal2 = ", goal1_dependent_on_goal2)
                return 
                
    def find_angle_for_single_arm(self, task_plan: TaskPlan, arm: str):
        if arm == self.arm1:
            for angle in self.all_angles:
                start1 = self.default_ee_pos[self.arm1]
                goal1 = self.default_ee_pos[self.arm1]
                angle2 = 900
                start2 = self.default_ee_pos[self.arm2]
                goal2 = self.default_ee_pos[self.arm2]
                if task_plan.arm1_assigned_object != -1 and task_plan.arm1_assigned_object is not None:
                    start1 = self.task.get_start( task_plan.arm1_assigned_object )
                    goal1 = self.task.get_goal( task_plan.arm1_assigned_object )
                
                if task_plan.arm2_assigned_object != -1 and task_plan.arm2_assigned_object is not None:
                    start2 = self.task.get_start( task_plan.arm2_assigned_object )
                    goal2 = self.task.get_goal( task_plan.arm2_assigned_object )
                    angle2 = task_plan.arm2_angle

                start_ik_result = self.ik_solve_single(start1, start2, angle, angle2)
                goal_ik_result = self.ik_solve_single(goal1, goal2, angle, angle2)
                print("start = ", start2, "goal= ", goal2, "angle= ", angle)
                print("start_ik_result = ", start_ik_result.success, "goal_ik_result = ", goal_ik_result.success)
                if start_ik_result.success and goal_ik_result.success: 
                    task_plan.arm1_angle = angle
                    return
        elif arm == self.arm2:
            for angle in self.all_angles:
                start1 = self.default_ee_pos[self.arm1]
                goal1 = self.default_ee_pos[self.arm1]
                start2 = self.default_ee_pos[self.arm2]
                goal2 = self.default_ee_pos[self.arm2]
                angle1 = 900
                if task_plan.arm1_assigned_object != -1 and task_plan.arm1_assigned_object is not None:
                    start1 = self.task.get_start( task_plan.arm1_assigned_object )
                    goal1 = self.task.get_goal( task_plan.arm1_assigned_object )
                    angle1 = task_plan.arm1_angle
                
                if task_plan.arm2_assigned_object != -1 and task_plan.arm2_assigned_object is not None:
                    start2 = self.task.get_start( task_plan.arm2_assigned_object )
                    goal2 = self.task.get_goal( task_plan.arm2_assigned_object )

                start_ik_result = self.ik_solve_single(start1, start2, angle1, angle)
                goal_ik_result = self.ik_solve_single(goal1, goal2, angle1, angle)
                print("start = ", start2, "goal= ", goal2, "angle= ", angle)
                print("start_ik_result = ", start_ik_result.success, "goal_ik_result = ", goal_ik_result.success)
                if start_ik_result.success and goal_ik_result.success: 
                    task_plan.arm2_angle = angle
                    return
            
    def find_available_task_for_single_arm(self, task_plan: TaskPlan):
        for angle in self.all_angles:
            for indices in task_plan.start_goal_pairs:
                start = self.task.get_start( indices[0] )
                goal = self.task.get_goal( indices[1] )
                start_ik_result = self.ik_solve_single(start, None, angle, None)
                goal_ik_result = self.ik_solve_single(goal, None, angle, None)
                print("start = ", start, "goal= ", goal, "angle= ", angle) 
                print("start_ik_result = ", start_ik_result.success, "goal_ik_result = ", goal_ik_result.success)
                if start_ik_result.success and goal_ik_result.success:
                    # [start_pos, goal_pos, orientation]
                    task_plan.arm1_assigned_object = indices[0]
                    task_plan.arm1_angle = angle
                    task_plan.arm2_assigned_object = indices[1]
                    task_plan.arm2_angle = 900
                    return
    
    def remove_gripper_dof_from_js(self, js: JointState):
        js = js.get_ordered_joint_state(self.arm_joint_names)
        return js
        
    def get_plan_single(self, result: MotionGenResult):
        cmd_plan = result.get_interpolated_plan()
        cmd_plan = self.single_motion_gen.get_full_js(cmd_plan)
        if self.headless:
            cmd_plan = cmd_plan.get_ordered_joint_state(self.j_names_primary)
        return cmd_plan

    def get_current_arm_js(self):
        if self.headless:
            cu_js = self.remove_gripper_dof_from_js(self.current_js)
            return cu_js
        positions = self._pb.getJointStates(
            self.robot_id,
            self.arm_joints_id_bullet
        )
        cu_js = JointState.from_position(
            self.tensor_args.to_device(np.array([joint[0] for joint in positions]).reshape(1, -1)),
            self.arm_joint_names
        )
        return cu_js
    
    def attach_obj(
        self,
        cu_js: JointState,
        cube_name: str,
        arm_name: str
    ) -> None:
        self.single_motion_gen.attach_objects_to_robot(
            cu_js,
            [cube_name],
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            # sphere_fit_type=SphereFitType.SAMPLE_SURFACE,
            # world_objects_pose_offset=Pose.from_list([0, 0, 0.01, 1, 0, 0, 0], self.tensor_args),
            link_name=arm_name
        )
        self.single_motion_gen.world_coll_checker.enable_obstacle(enable=False, name=cube_name)

    def detach_obj(self, arm_name: str, cube_name: str) -> None:
        self.single_motion_gen.detach_object_from_robot(link_name=arm_name)
        self.single_motion_gen.world_coll_checker.enable_obstacle(enable=True, name=cube_name)

    def test_different_grasp_angles(self):
        self.reset()
        angles = [900, 450, -450, 901, 451, -451]
        # angles = [901, 451, -451]
        cube_positions = self.get_current_cube_pose()
        pos1 = cube_positions[0]
        pos2 = cube_positions[1]
        pos1[2] += self.pre_grasp_height
        pos2[2] += self.pre_grasp_height

        for angle in angles:
            result = self.plan_single_from_curr_js(pos1, pos2, 900, angle)
            if not result.success:
                print("Failed to go to start position")
                return
            else:
                print("Going to the start position")
            
            cmd_plan = self.get_plan_single(result)
            self.visualize_plan(cmd_plan)
            cmd = self.move_z(0.3, 900, angle)
            self.visualize_plan(cmd)

        for angle in angles:
            result = self.plan_single_from_curr_js(pos1, pos2, angle, 900)
            if not result.success:
                print("Failed to go to start position")
                return
            else:
                print("Going to the start position")
            
            cmd_plan = self.get_plan_single(result)
            self.visualize_plan(cmd_plan)
            cmd = self.move_z(0.3, angle, 900)
            self.visualize_plan(cmd)
    
    def get_current_cube_pose(self):
        if self.headless:
            return self.task.get_starts()
        cube_positions = []
        for cube_id in self.objects_pybullet:
            cube_position, cube_orientation = self._pb.getBasePositionAndOrientation(cube_id) 
            cube_position = np.array(cube_position)
            cube_positions.append(cube_position)
        return cube_positions 

    def sync_isaacsim_curobo(self, idx_to_sync: list = None):
        if idx_to_sync is None:
            idx_to_sync = list(range(self.task.num_objects))
        new_starts = {}
        cuboids = []
        if not self.headless:
            for i in idx_to_sync:
                cube_bullet = self.objects_pybullet[i]
                cube_curobo = self.objects_curobo[i]
                p, q = self._pb.getBasePositionAndOrientation(cube_bullet)
                cube_curobo.pose = [p[0], p[1], p[2], q[0], q[1], q[2], q[3]]
                cuboids.append(cube_curobo)
                new_starts[i] = np.array([p[0], p[1], p[2]])
        else:
            if self.old_task_plan is None:
                return
            arm1_id = self.old_task_plan.arm1_assigned_object
            arm2_id = self.old_task_plan.arm2_assigned_object
            if arm1_id != -1 and arm1_id is not None:
                new_starts[arm1_id] = self.task.get_goal(arm1_id)
            if arm2_id != -1 and arm2_id is not None:
                new_starts[arm2_id] = self.task.get_goal(arm2_id)
            for i in idx_to_sync:
                cube_curobo = self.objects_curobo[i]
                p = self.task.get_start(i)
                cube_curobo.pose = [p[0], p[1], p[2], 1, 0, 0, 0]
                cuboids.append(cube_curobo)

        self.task.set_starts(new_starts)
        world_cfg_table = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        )
        world_cfg_table.cuboid[0].pose[2] -= 0.02
        self.world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid)
        obs = cuboids + world_cfg_table.cuboid
        self.world_cfg_curobo = WorldConfig(cuboid=obs)
        self.single_motion_gen.clear_world_cache()
        self.single_motion_gen.update_world(self.world_cfg_curobo)
        self.ik_solver.update_world(self.world_cfg_curobo)
        # print("self.single_motion_gen.world_coll_checker.get_obstacle_names() = ", self.single_motion_gen.world_coll_checker.get_obstacle_names())
    
    def save_scene_as_mesh(self):
        mesh = self.single_motion_gen.world_coll_checker.get_mesh_in_bounding_box(
            cuboid=Cuboid(name="test", dims=[2,2,2], pose=[0, 0, -0.1,1,0,0,0]),
            voxel_size=0.01,
            run_marching_cubes=True
        )
        mesh.save_as_mesh(file_path="results/scene.obj")

    def interpolate_joint_states(self, start_joints, end_joints):
        """
        Interpolates between two joint states without exceeding the maximum velocity.
        Parameters:
            start_joints (list or np.array): Initial joint angles (radians).
            end_joints (list or np.array): Target joint angles (radians).
        Returns:
            np.array: Interpolated joint positions at each time step.
        """
        delta_joints = torch.abs(end_joints - start_joints)
        time_required = delta_joints / self.max_velocity
        total_time = torch.max(time_required)
        num_steps = int(torch.ceil(total_time / self.interpolation_dt))
        times = self.tensor_args.to_device(torch.linspace(0, 1, num_steps).view(-1, 1))
        interpolated_trajectories = start_joints + (end_joints - start_joints) * times
        if interpolated_trajectories.shape[0] == 0:
            print("interpolated_trajectories.shape[0] == 0")
        return interpolated_trajectories
    
    def servoj(self, joints):
        """Move to target joint positions with position control."""
        self._pb.setJointMotorControlArray(
        bodyIndex=self.robot_id,
        jointIndices=self.arm_joints_id_bullet,
        controlMode=pybullet.POSITION_CONTROL,
        targetPositions=joints,
        positionGains=[0.05] * len(self.arm_joints_id_bullet),
        )
        
    def move_ee(self, pos1, angle1, pos2, angle2):
        ik_result = self.ik_solve_single(pos1, pos2, angle1, angle2, num_solutions=1)
        curr_js = self.get_current_arm_js()
        curr_js = self.single_motion_gen.get_full_js(curr_js)
        # if self.headless:
        curr_js = curr_js.get_ordered_joint_state(self.j_names_primary)
        start_js = curr_js.position.squeeze(0).view(-1)
        goal_js = ik_result.js_solution.get_ordered_joint_state(self.j_names_primary).position.squeeze(0).view(-1)
        traj = self.interpolate_joint_states(curr_js.position, goal_js)
        cmd_plan = JointState.from_position(traj, self.j_names_primary)
        return cmd_plan
    
    def move_z(self, delta, angle1, angle2):
        ee_pos = self.get_current_ee_pos()
        pos1 = ee_pos[self.arm1]
        pos2 = ee_pos[self.arm2]
        pos1[2] += delta
        pos2[2] += delta
        return self.move_ee(pos1, angle1, pos2, angle2)
    
    def move_z_2(self, delta1, delta2, angle1, angle2):
        ee_pos = self.get_current_ee_pos()
        pos1 = ee_pos[self.arm1]
        pos2 = ee_pos[self.arm2]
        pos1[2] += delta1
        pos2[2] += delta2
        return self.move_ee(pos1, angle1, pos2, angle2)
    
    def move_x(self, delta, angle1, angle2):
        ee_pos = self.get_current_ee_pos()
        pos1 = ee_pos[self.arm1]
        pos2 = ee_pos[self.arm2]
        pos1[0] += delta
        pos2[0] -= delta
        return self.move_ee(pos1, angle1, pos2, angle2)
    
    def move_y(self, delta, angle1, angle2):
        ee_pos = self.get_current_ee_pos()
        pos1 = ee_pos[self.arm1]
        pos2 = ee_pos[self.arm2]
        pos1[1] += delta
        pos2[1] -= delta
        return self.move_ee(pos1, angle1, pos2, angle2)

    def get_current_ee_pos(self):
        cu_js = self.get_current_js()
        kin_state = self.single_motion_gen.kinematics.get_state(cu_js.position.view(1, -1))
        link_poses = kin_state.link_poses
        ee_pos = {}
        for k, v in link_poses.items():
            t_pos = np.ravel(v.to_list())
            ee_pos[k] = t_pos[:3]
        return ee_pos
    
    def test_move_ee(self):
        self.reset()
        pos1 = self.default_ee_pos[self.arm1]
        pos2 = self.default_ee_pos[self.arm2]
        print("initial: pos1= ", pos1, "pos2= ", pos2)
        ee_pos = self.get_current_ee_pos()
        print("pos1 = ", ee_pos[self.arm1], "pos2 = ", ee_pos[self.arm2])
        angle = -450
        cmd_plan = self.move_ee(pos1, angle, pos2, angle)
        self.visualize_plan(cmd_plan)
        ee_pos = self.get_current_ee_pos()
        print("pos1 = ", ee_pos[self.arm1], "pos2 = ", ee_pos[self.arm2])
        time.sleep(2)
        delta_z = -0.2
        print("moving z")
        cmd_plan = self.move_z(delta_z, angle, angle)
        self.visualize_plan(cmd_plan)
        print("moved z")
        ee_pos = self.get_current_ee_pos()
        print("pos1 = ", ee_pos[self.arm1], "pos2 = ", ee_pos[self.arm2])
        time.sleep(2)
        delta_x = 0.2
        print("moving x")
        cmd_plan = self.move_x(delta_x, angle, angle)
        self.visualize_plan(cmd_plan)
        ee_pos = self.get_current_ee_pos()
        print("pos1 = ", ee_pos[self.arm1], "pos2 = ", ee_pos[self.arm2])
        time.sleep(2)
        delta_y = 0.2
        print("moving y")
        cmd_plan = self.move_y(delta_y, angle, angle)
        self.visualize_plan(cmd_plan)
        ee_pos = self.get_current_ee_pos()
        print("pos1 = ", ee_pos[self.arm1], "pos2 = ", ee_pos[self.arm2])
        self.display(reset=False)
    
    def toppra_improvement(self, traj: JointState, speed_scale=0.5, acc_scale=1.0):
        waypoints_ = traj.position.cpu().numpy()
        joint_names = traj.joint_names
        max_accelerations = np.array([[-40.0*acc_scale, 40.0*acc_scale]]*traj[0].shape[0])
        max_velocities = np.array([[-np.pi * speed_scale, np.pi * speed_scale]]*traj[0].shape[0])
        path = ta.SplineInterpolator(np.linspace(0, 1, traj.shape[0]), waypoints_)
        pc_vel = constraint.JointVelocityConstraint(max_velocities)
        pc_acc = constraint.JointAccelerationConstraint(max_accelerations)
        instance = algo.TOPPRA([pc_vel, pc_acc], path, solver_wrapper='seidel')
        jnt_traj = instance.compute_trajectory(0, 0) 
        if jnt_traj is None:
            print("Toppra failed")
            return traj
        toppra_duration = jnt_traj.duration
        traj = jnt_traj.eval(np.linspace(0, toppra_duration, int(toppra_duration/(self.interpolation_dt))))
        traj = self.tensor_args.to_device(traj)
        traj = JointState.from_position(traj.view(-1,waypoints_.shape[1]), joint_names=joint_names)
        return traj
    
    def play(self, recorded_plan):
        self.reset()
        for section in recorded_plan:
            if type(section) == list:
                self.gripper_control(section[0], section[1])
            else:
                self.visualize_plan(section)

    def add_plan_to_record(self, cmd_plan:JointState|List):
        if isinstance(cmd_plan, list):
            self.recorded_plan.append(cmd_plan)
        else:
            assert cmd_plan.joint_names is not None
            self.recorded_plan.append(cmd_plan)

# debug
if __name__ == '__main__':
    test_grasp_angles = True
    test_plan_all = False
    if test_grasp_angles:
        num_objects:int = 2  
        map_width:float = 0.6 
        map_length:float = 0.6 
        object_size:float = 0.05 
        task = Task(num_objects, map_width, map_length, object_size)
        # task.generate()
        task.generate_fixed_two_objects(offset=0.)
    if test_plan_all:
        task = Task.read_from_json("data/random_objects_4.json")

    dual_arm = DualArmEnv(task,
                          False,
                          batch_size = 3,
                          constrain_grasp_approach=False,
                          )
    dual_arm.reset()
    if test_grasp_angles:
        dual_arm.test_different_grasp_angles()
        # dual_arm.plan()
    if test_plan_all:
        task_plans = dual_arm.plan_all()
        print(task_plans)
    