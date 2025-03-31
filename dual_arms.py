import curobo
import os
os.environ['CUROBO_TORCH_CUDA_GRAPH_RESET'] = '1'
import random
from itertools import permutations, product
from packaging import version

import numpy as np
import time
import isaacsim
from typing import List
from omni.isaac.kit import SimulationApp
# from isaacsim import SimulationApp
if __name__ == '__main__':
    simulation_app = SimulationApp({
                    "headless": False,
                    "width": "1920",
                    "height": "1080",
    })

import carb
from helper import add_extensions, add_robot_to_scene
# from isaacsim.core.api.world import World
# from isaacsim.core.api.objects import cuboid, sphere, DynamicCuboid
# from isaacsim.core.api.materials.physics_material import PhysicsMaterial 
# import isaacsim.core.utils.prims as prims_utils
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere, DynamicCuboid
from omni.isaac.core.materials import PhysicsMaterial
import omni.isaac.core.utils.prims as prims_utils

########### OV #################
# from isaacsim.core.utils.types import ArticulationAction
from omni.isaac.core.utils.types import ArticulationAction

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
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Cuboid, Material
from curobo.rollout.rollout_base import Goal
from curobo.geom.sphere_fit import SphereFitType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, MotionGenResult, PoseCostMetric
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

from pxr import Usd, UsdPhysics, Sdf, UsdGeom, PhysxSchema, Gf, UsdShade

# toppra
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo

# DualArmTAMP
from task import Task, TaskPlan, TaskState, GripperState, TaskExecutionResult, ArmState, GripperAction
from dependency_graph import DependencyGraph


class DualArmEnv:
    def __init__(self, 
                 task: Task,
                 simulation_app: SimulationApp,
                 batch_size: int = 3,
                 constrain_grasp_approach: bool = False,
                 visualize_spheres: bool = False,
                 num_buffers: int = 20,
                 ):
        self.simulation_app = simulation_app
        self.task = task
        self.spheres = None
        self.visualize_spheres = visualize_spheres
        self.num_buffers = num_buffers
        self.spheres = None
        self.world = World(stage_units_in_meters=1.0)
        self.stage = self.world.stage
        xform = self.stage.DefinePrim("/World", "Xform")
        self.stage.SetDefaultPrim(xform)
        self.stage.DefinePrim("/curobo", "Xform")
        setup_curobo_logger("warn")
        self.usd_help = UsdHelper()
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
        
        self.common_js_names = []
        self.common_js_indices = []
        self.arm_joint_names = []
        for k, v in self.gripper_mimic_joints_settings.items():
            self.gripper_mimic_joint_names.extend(v["joint_names"])
        self.j_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        self.j_names_primary = [name for name in self.j_names if name not in self.gripper_mimic_joint_names]
        print("j_names_primary = ", self.j_names_primary)
        self.default_config = self.robot_cfg["kinematics"]["cspace"]["retract_config"]
        self.robot, self.robot_prim_path = add_robot_to_scene(self.robot_cfg, self.world)
        self.set_material_for_finger_pads()
        self.dof_names = []

        # Check screen boundaries
        self.articulation_controller = self.robot.get_articulation_controller()
        self.objects_curobo = []
        self.objects_isaacsim_dynamic = []
        self.objects_isaacsim_visual = []
        world_cfg_table = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        )
        world_cfg_table.cuboid[0].pose[2] -= 0.02
        # world_cfg1 = WorldConfig.from_dict(
        #     load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        # ).get_mesh_world()
        # world_cfg1.mesh[0].name += "_mesh"
        # world_cfg1.mesh[0].pose[2] = -10.5
        # self.world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)
        self.world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid)
        self.add_objects_curobo()
        obs = self.objects_curobo + world_cfg_table.cuboid
        # self.world_cfg_curobo = WorldConfig(cuboid=obs, mesh=world_cfg1.mesh)
        self.world_cfg_curobo = WorldConfig(cuboid=obs)
        self.max_velocity = 0.314 * 5
        self.interpolation_dt = 0.02

        # ik_seed = np.random.randint(0, 10000)
        # graph_seed = np.random.randint(0, 10000)

        motion_gen_config = MotionGenConfig.load_from_robot_config(
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
            num_graph_seeds=1,
            num_ik_seeds=100,
            num_batch_ik_seeds=32,
            num_trajopt_seeds= 32,
            trajopt_tsteps=64,
            finetune_trajopt_iters=500,
        )

        self.motion_gen = MotionGen(motion_gen_config)

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
            # collision_checker_type=CollisionCheckerType.MESH,
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
        add_extensions(self.simulation_app, None)
        
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

        self.usd_help.load_stage(self.stage)
        self.usd_help.add_world_to_stage(self.world_cfg, base_frame="/World", enable_physics=False)
        self.world.scene.add_default_ground_plane()
        self.gripper_joint_indices = []

        self.recorded_plan = []
        self.old_task_plan = None
        self.pre_grasp_height = 2 * self.task.object_size
        
        self.preferred_angles = [901, 900]
        self.preferred_angles_combination = list(product(self.preferred_angles, repeat=2))
        self.all_angles = [901, 900, 450, -450, 451, -451]
        self.all_angles_combination = list(product(self.all_angles, repeat=2))
        # end of init


    
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


    
    def get_ready(self, reset=False, autostart=False):
        i = 0
        while self.simulation_app.is_running():
            self.world.step(render=True)
            if not self.world.is_playing():
                if autostart:
                    self.world.play()
                if i % 100 == 0:
                    print("**** Click Play to start simulation *****")
                i += 1
                continue

            step_index = self.world.current_time_step_index
            if step_index <= 2 or reset:
                self.reset()
                reset=False

            if step_index < 20:
                continue
            return

    
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

    def test_ik_solve_batch(self):
        self.get_ready()
        cube = self.objects_isaacsim_dynamic[0]
        c_p1, _ = cube.get_world_pose()
        cube = self.objects_isaacsim_dynamic[1]
        c_p2, _ = cube.get_world_pose()
        ik_goal1, ik_goal2 = self.get_grid_goal_poses(c_p1, c_p2, 900, 900)
        ik_results = self.ik_solve_batch(ik_goal1, ik_goal2)
        
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
    

    def plan_batch_from_curr_js_all(self, goal_arm1, goal_arm2, angle1=900, angle2=900, prioritize_arm1=True):
        if not self.simulation_app.is_running():
            raise Exception("Simulation is not running")

        self.world.step(render=True)

        ik_goal1, ik_goal2 = self.get_grid_goal_poses(goal_arm1, goal_arm2, angle1, angle2, prioritize_arm1)

        # filter out valid goals with feasible IKs

        ik_results = self.ik_solve_batch(ik_goal1=ik_goal1, ik_goal2=ik_goal2)
        ik_goal1 = ik_goal1[torch.where(ik_results.success)[0]]
        ik_goal2 = ik_goal2[torch.where(ik_results.success)[0]]

        sim_js = self.robot.get_joints_state()
        sim_js_positions = sim_js.positions.reshape(1, -1)
        # repeat cu_js
        cu_js = JointState.from_position(self.tensor_args.to_device(sim_js_positions), self.dof_names)
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
        cu_js = JointState.from_position(cu_js.position.repeat(ik_goal1.position.shape[0], 1), self.dof_names)
        
        link_poses = {}
        link_poses[self.arm2] = ik_goal2
        results = self.motion_gen.plan_batch(
            cu_js,
            ik_goal1,
            self.plan_config.clone(),
            link_poses=link_poses,
            )

        print("plan time: ", results.solve_time)
        print("succeeded:", torch.count_nonzero(results.success).item(), " out of ", len(results.success))
        return results
    
    def plan_batch_from_curr_js(self, goal_arm1, goal_arm2, angle1=900, angle2=900, prioritize_arm1=True):
        results = self.plan_batch_from_curr_js_all(goal_arm1, goal_arm2, angle1, angle2, prioritize_arm1=prioritize_arm1)
        if results.success[0]:
            print("Original Plan successful")
            return results.optimized_plan[0]
        else:
            print("Picking a random successful plan")
            # randomly pick one from the successful ones
            success_indices = torch.where(results.success)[0]
            if len(success_indices) == 0:
                return None
            idx = random.choice(success_indices)
            return results.optimized_plan[idx]

    def test_plan_batch(self):
        self.get_ready()
        
        cube = self.objects_isaacsim_dynamic[0]
        c_p1, _ = cube.get_world_pose()
        cube = self.objects_isaacsim_dynamic[1]
        c_p2, _ = cube.get_world_pose()

        results = self.plan_batch_from_curr_js_all(c_p1, c_p2)
        for i in range(len(results.success)):
            if results.success[i]:
                cmd_plan = results.optimized_plan[i]
                cmd_plan = self.motion_gen.get_full_js(cmd_plan)
                # get only joint names that are in both:
                if self.common_js_names == [] and self.common_js_indices == []:
                    for x in self.dof_names:
                        if x in cmd_plan.joint_names:
                            self.common_js_indices.append(self.robot.get_dof_index(x))
                            self.common_js_names.append(x)
                
                cmd_plan = cmd_plan.get_ordered_joint_state(self.common_js_names)
                cmd_idx = 0
                reached = False

                while self.simulation_app.is_running():
                    self.world.step(render=True)
                    # if torch.any(results.success):
                    cmd_state = cmd_plan[cmd_idx]
                    # get full dof state
                    # positions, velocities = self.get_joint_positions_with_static_gripper(cmd_state)
                    gripper_primary_pos_dict = self.get_primary_gripper_position_dict()
                    positions = cmd_state.position.cpu().numpy()
                    for gripper_primary_name in self.gripper_primary_joints:
                        idx = self.common_js_names.index(gripper_primary_name)
                        positions[idx] = gripper_primary_pos_dict[gripper_primary_name]

                    velocities = cmd_state.velocity.cpu().numpy()
                    art_action = ArticulationAction(
                        joint_positions=positions,
                        joint_velocities=velocities,
                        joint_indices=self.common_js_indices
                    )

                    # set desired joint angles obtained from IK:
                    self.articulation_controller.apply_action(art_action)
                    cmd_idx += 1
                    for _ in range(2):
                        self.world.step(render=False)
                    if cmd_idx >= len(cmd_plan.position):
                        cmd_idx = 0
                        cmd_plan = None
                        reached = True
                        break
                
                    # if reached and not self.gripper_closed:
                    #     self.close_gripper()
                    #     return


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
        result = self.motion_gen.plan_single(
            # cu_js.unsqueeze(0), ik_goal, self.plan_config.clone(), 
            cu_js, ik_goal, self.plan_config.clone(), 
            link_poses=link_poses
        )
        return result

    
    def draw_task_config(self):
        self.task.draw()
    
    def get_max_joint_velocity(self):
        sim_js = self.robot.get_joints_state()
        velocities = sim_js.velocities
        max_vel = np.max(np.abs(velocities))
        return max_vel
    
    def get_gripper_positions(self):
        """
            Get the gripper positions
        """
        sim_js = self.robot.get_joints_state()
        gripper_positions = sim_js.positions[self.gripper_joint_indices]
        return gripper_positions

    def get_primary_gripper_positions(self):
        """
            Get the gripper positions
        """
        sim_js = self.robot.get_joints_state()
        gripper_positions = sim_js.positions[self.gripper_primary_indices]
        return gripper_positions

    def get_primary_gripper_position_dict(self):
        """
            Get the gripper positions
        """
        sim_js = self.robot.get_joints_state()
        gripper_positions = sim_js.positions[self.gripper_primary_indices]
        return dict(zip(self.gripper_primary_joints, gripper_positions))
    

    def close_gripper(self, close_arm1: bool = True, close_arm2: bool = True):
        if close_arm1 and self.gripper_closed[self.arm1]:
            close_arm1 = False
        if close_arm2 and self.gripper_closed[self.arm2]:
            close_arm2 = False
        
        if not close_arm1 and not close_arm2:
            return

        dof_upper = self.robot.dof_properties["upper"]
        gripper_closed_pos = dof_upper[self.gripper_primary_indices[0]]
        gripper_pos = gripper_closed_pos * np.ones(close_arm1 + close_arm2)
        old_gripper_positions = self.get_gripper_positions()
        gripper_indices = []
        if close_arm1:
            gripper_indices.append(self.gripper_primary_indices_dict[self.arm1])
        if close_arm2:
            gripper_indices.append(self.gripper_primary_indices_dict[self.arm2])
        while self.simulation_app.is_running():
            # get full dof state
            art_action = ArticulationAction(
                joint_positions = gripper_pos,
                # joint_velocities=gripper_vel,
                joint_indices=gripper_indices
            )
            # set desired joint angles obtained from IK:
            self.articulation_controller.apply_action(art_action)
            for _ in range(2):
                self.world.step(render=False)
            self.world.step(render=True)
            new_gripper_positions = self.get_gripper_positions()
            if np.allclose(new_gripper_positions, old_gripper_positions, rtol=1e-3, atol=1e-3) :
                if close_arm1:
                    self.gripper_closed[self.arm1] = True
                    print("Gripper1 closed.")
                if close_arm2:
                    self.gripper_closed[self.arm2] = True
                    print("Gripper2 closed.")
                break
            old_gripper_positions = new_gripper_positions


    def open_gripper(self, open_arm1: bool = True, open_arm2: bool = True):
        if open_arm1 and not self.gripper_closed[self.arm1]:
            open_arm1 = False
        if open_arm2 and not self.gripper_closed[self.arm2]:
            open_arm2 = False
        
        if not open_arm1 and not open_arm2:
            return

        gripper_pos = np.zeros(open_arm1 + open_arm2)
        old_gripper_positions = self.get_gripper_positions()
        gripper_indices = []
        if open_arm1:
            gripper_indices.append(self.gripper_primary_indices_dict[self.arm1])
        if open_arm2:
            gripper_indices.append(self.gripper_primary_indices_dict[self.arm2])
        while self.simulation_app.is_running():
            # get full dof state
            art_action = ArticulationAction(
                joint_positions = gripper_pos,
                # joint_velocities=gripper_vel,
                joint_indices=gripper_indices
            )
            # set desired joint angles obtained from IK:
            self.articulation_controller.apply_action(art_action)
            for _ in range(2):
                self.world.step(render=False)
            self.world.step(render=True)
            new_gripper_positions = self.get_gripper_positions()
            if np.allclose(new_gripper_positions, old_gripper_positions, rtol=1e-3, atol=1e-3): 
                if open_arm1:
                    self.gripper_closed[self.arm1] = False
                    print("Gripper1 opened.")
                if open_arm2:
                    self.gripper_closed[self.arm2] = False
                    print("Gripper2 opened.")
                break
            old_gripper_positions = new_gripper_positions
    
    def gripper_control(self, gripper1_action: GripperAction, gripper2_action: GripperAction):
        dof_upper = self.robot.dof_properties["upper"]
        gripper_closed_pos = dof_upper[self.gripper_primary_indices[0]]
        gripper1_pos = gripper1_action.value * gripper_closed_pos
        gripper2_pos = gripper2_action.value * gripper_closed_pos
        gripper_primary_positions = self.get_primary_gripper_positions()
        if gripper1_action == GripperAction.CLOSE and self.gripper_closed[self.arm1]:
            gripper1_pos = gripper_primary_positions[0]
        if gripper2_action == GripperAction.CLOSE and self.gripper_closed[self.arm2]:
            gripper2_pos = gripper_primary_positions[1]
        gripper_pos = np.array([gripper1_pos, gripper2_pos])
        old_gripper_positions = self.get_gripper_positions()
        while self.simulation_app.is_running():
            # get full dof state
            art_action = ArticulationAction(
                joint_positions = gripper_pos,
                # joint_velocities=gripper_vel,
                joint_indices=self.gripper_primary_indices
            )
            # set desired joint angles obtained from IK:
            self.articulation_controller.apply_action(art_action)
            for _ in range(2):
                self.world.step(render=False)
            self.world.step(render=True)
            new_gripper_positions = self.get_gripper_positions()
            if np.allclose(new_gripper_positions, old_gripper_positions, rtol=1e-3, atol=1e-3):
                self.gripper_closed[self.arm1] = bool(gripper1_action.value)
                self.gripper_closed[self.arm2] = bool(gripper2_action.value)
                break
            old_gripper_positions = new_gripper_positions

    
    def test_gripper_close_open(self):
        """
            Test the gripper closing
        """
        self.get_ready()
        for i in range(5):
            self.close_gripper(True, False)
            self.open_gripper(True, False)
        
        for i in range(5):
            self.close_gripper(False, True)
            self.open_gripper(False, True)
        
        for i in range(5):
            self.close_gripper(True, True)
            self.open_gripper(True, True)
    
    def test_gripper_control(self):
        self.get_ready()
        for i in range(3):
            self.gripper_control(GripperAction.CLOSE, GripperAction.OPEN)
            self.gripper_control(GripperAction.OPEN, GripperAction.OPEN)
        
        for i in range(3):
            self.gripper_control(GripperAction.OPEN, GripperAction.CLOSE)
            self.gripper_control(GripperAction.OPEN, GripperAction.OPEN)
        
        for i in range(3):
            self.gripper_control(GripperAction.CLOSE, GripperAction.CLOSE)
            self.gripper_control(GripperAction.OPEN, GripperAction.OPEN)
        
        for i in range(3):
            self.gripper_control(GripperAction.CLOSE, GripperAction.OPEN)
            self.gripper_control(GripperAction.OPEN, GripperAction.CLOSE)
        
        self.gripper_control(GripperAction.OPEN, GripperAction.OPEN)
    
    def test_grasp(self):
        self.get_ready()
        angle1 = 900
        angle2 = -450 
        cube = self.objects_isaacsim_dynamic[1]
        c_p1, _ = cube.get_world_pose()
        cube = self.objects_isaacsim_dynamic[0]
        c_p2, _ = cube.get_world_pose()
        c_p1[2] += 0.1
        c_p2[2] += 0.1
        result = self.plan_single_from_curr_js(c_p1, c_p2, angle1, angle2)
        cmd_plan = self.get_plan_single(result)
        self.visualize_plan(cmd_plan)
        self.visualize_plan(self.move_z(-0.1, angle1, angle2))
        self.gripper_control(GripperAction.CLOSE, GripperAction.CLOSE)
        self.visualize_plan(self.move_z(0.1, angle1, angle2))
    
    def reset(self):
        self.world.reset()
        # set gravity
        physics_scene_path = "/physicsScene"
        physics_scene = self.stage.GetPrimAtPath(physics_scene_path)
        # Ensure the Physics Scene prim is valid
        if physics_scene.IsValid():
            # Set the gravity direction (e.g., (0, 0, -1) for downward gravity along the Z-axis)
            gravity_direction_attr = physics_scene.GetAttribute("physics:gravityDirection")
            if not gravity_direction_attr:
                gravity_direction_attr = physics_scene.CreateAttribute("physics:gravityDirection", Gf.Vec3f)
            gravity_direction_attr.Set(Gf.Vec3f(0.0, 0.0, -1.0))

            # Set the gravity magnitude (e.g., 9.81 for Earth's gravity)
            gravity_magnitude_attr = physics_scene.GetAttribute("physics:gravityMagnitude")
            if not gravity_magnitude_attr:
                gravity_magnitude_attr = physics_scene.CreateAttribute("physics:gravityMagnitude", float)
            gravity_magnitude_attr.Set(5.0)
        else:
            print(f"Physics Scene prim at {physics_scene_path} not found.")


        idx_list = [self.robot.get_dof_index(x) for x in self.j_names]
        # idx_list = [self.robot.get_dof_index(x) for x in self.j_names_primary]
        self.robot.set_joint_positions(self.default_config, idx_list)
        self.robot.set_joint_velocities(np.zeros(len(idx_list)), idx_list)
        kin_state = self.single_motion_gen.kinematics.get_state(self.single_motion_gen.get_retract_config().view(1, -1))
        link_poses = kin_state.link_poses
        self.default_ee_pos = {}
        for k, v in link_poses.items():
            t_pos = np.ravel(v.to_list())
            self.default_ee_pos[k] = t_pos[:3]
            self.default_ee_pos[k][2] += 0.15 #TODO: Test this, added 0.15m to z
        # t_pos = np.ravel(kin_state.ee_pose.to_list())
        # self.default_ee_pos = t_pos[:3]
        # self.default_ee_rot = t_pos[3:]
        sim_js = self.robot.get_joints_state()
        self.dof_names = self.robot.dof_names
        for pair in zip(self.dof_names, sim_js.positions):
            print(pair)
        
        if self.common_js_names == [] and self.common_js_indices == []:
            for x in self.dof_names:
                if x in self.j_names_primary:
                    self.common_js_indices.append(self.robot.get_dof_index(x))
                    self.common_js_names.append(x)

        if self.visualize_spheres: 
            self.visualize_robot_spheres()

        self.robot._articulation_view.set_max_efforts(
            values=np.array([5000 for i in range(len(idx_list))]), 
            joint_indices=idx_list
        )

        self.gripper_joint_indices = []
        for joint_name in self.gripper_joint_names:
            dof_index = self.robot.get_dof_index(joint_name)
            self.gripper_joint_indices.append(dof_index)

        self.gripper_primary_indices = []
        for joint_name in self.gripper_primary_joints:
            dof_index = self.robot.get_dof_index(joint_name)
            self.gripper_primary_indices.append(dof_index)
            
        
        self.gripper_closed = {self.arm1: False, self.arm2: False}
        #TODO: This kind assignment is risky in the future. Fix it later
        self.gripper_primary_joints_dict = {self.arm1: self.gripper_primary_joints[0], self.arm2: self.gripper_primary_joints[1]}
        self.gripper_primary_indices_dict = {self.arm1: None, self.arm2: None}
        for arm, joint_name in self.gripper_primary_joints_dict.items():
            self.gripper_primary_indices_dict[arm] = self.robot.get_dof_index(joint_name)
        
        self.arm_joint_names = [name for name in self.j_names_primary if name not in self.gripper_primary_joints]

        self.robot._articulation_view.set_gains(
            kps = 100 * np.ones(len(self.gripper_joint_indices)), 
            kds= 100 * np.ones(len(self.gripper_joint_indices)), 
            joint_indices=self.gripper_joint_indices
            )
        print("self.robot.dof_properties = ", self.robot.dof_properties)
        self.add_objects_isaacsim()
        self.sync_isaacsim_curobo()
        print("self.single_motion_gen.world_coll_checker.get_obstacle_names() = ", self.single_motion_gen.world_coll_checker.get_obstacle_names())
        print("       self.motion_gen.world_coll_checker.get_obstacle_names() = ", self.single_motion_gen.world_coll_checker.get_obstacle_names())
    
    def add_objects_isaacsim(self):
        starts = self.task.get_starts()
        goals = self.task.get_goals()
        colors = self.task.get_colors()
        if len(self.objects_isaacsim_dynamic)>0:
            # remove all the prims
            for i in range(len(self.objects_isaacsim_dynamic)):
                prims_utils.delete_prim(self.objects_isaacsim_dynamic[i].prim_path)
        for i in range(self.task.num_objects):
            material = PhysicsMaterial(
                prim_path=f"/World/physics_material/obstacle{i}",
                dynamic_friction=100,
                static_friction=100,
                restitution=0.2,
            )
            start = starts[i]
            goal = goals[i]
            cube_isaac = cuboid.DynamicCuboid(
            # cube_isaac = cuboid.VisualCuboid(
                "/World/obstacle" + str(i),
                position = [start[0], start[1], start[2]],
                orientation = [1, 0, 0, 0],
                color = colors[i],
                size=self.task.object_size,
                physics_material=material
            )
            self.objects_isaacsim_dynamic.append(cube_isaac)

            cube_isaac = cuboid.VisualCuboid(
                "/World/visual_goals" + str(i),
                position = [goal[0], goal[1], self.task.object_size/2 * 0.1],
                orientation = [1, 0, 0, 0],
                color = colors[i], 
                size=self.task.object_size,
                scale=[1.1, 1.1, 0.1]
                # visual_material=material
            )
            self.add_opacity_material(
                material_name=f"material_goal{i}",
                object_path=cube_isaac.prim_path,
                color = [colors[i][0], colors[i][1], colors[i][2], 0.8],
                obj_prim=prims_utils.get_prim_at_path(f"/World/visual_goals{i}")
            )

            self.objects_isaacsim_visual.append(cube_isaac)
    
    def set_material_for_finger_pads(self):
        pad_path = ["left_inner_finger_pad", "right_inner_finger_pad", "left_inner_finger_pad_1", "right_inner_finger_pad_1"]
        for i, path in enumerate(pad_path):
            self.add_physics_material(
                material_name=f"material_pad{i}",
                object_path=path,
                dynamic_friction=10,
                static_friction=10,
                restitution=0.2,
                obj_prim=prims_utils.get_prim_at_path(f"/ur5e_robot/{path}")
            )
        
    def motion_plan(self, task_plan: TaskPlan):
        task_execution_result = TaskExecutionResult()
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
        # if goal_arm1[2] >= self.default_ee_pos[self.arm1][2]+self.pre_grasp_height:
        #     goal_arm1[2] -= self.pre_grasp_height
        # if goal_arm2[2] >= self.default_ee_pos[self.arm2][2]+self.pre_grasp_height:
        #     goal_arm2[2] -= self.pre_grasp_height

        task_execution_result.arm1_assigned_object = task_plan.arm1_assigned_object 
        task_execution_result.arm2_assigned_object = task_plan.arm2_assigned_object
        task_execution_result.arm1_gripper_state = GripperState(task_plan.arm1_gripper_action.value)
        task_execution_result.arm2_gripper_state = GripperState(task_plan.arm2_gripper_action.value)
        task_execution_result.arm1_angle = task_plan.arm1_angle
        task_execution_result.arm2_angle = task_plan.arm2_angle
        task_execution_result.arm1_task_state = TaskState.IN_PROGRESS if task_plan.arm1_to_start else TaskState.COMPLETED
        task_execution_result.arm2_task_state = TaskState.IN_PROGRESS if task_plan.arm2_to_start else TaskState.COMPLETED
        task_execution_result.buffer = task_plan.buffer

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
                self.recorded_plan.append(cmd_plan)
                self.visualize_plan(cmd_plan)
                self.recorded_plan.append([task_plan.arm1_gripper_action, task_plan.arm2_gripper_action])
                # go down
                arm1_delta_z = -self.pre_grasp_height if task_plan.arm1_assigned_object not in [None, -1] else 0
                arm2_delta_z = -self.pre_grasp_height if task_plan.arm2_assigned_object not in [None, -1] else 0
                cmd_plan = self.move_z_2(arm1_delta_z, arm2_delta_z, task_plan.arm1_angle, task_plan.arm2_angle)
                self.recorded_plan.append(cmd_plan)
                self.visualize_plan(cmd_plan)

                self.gripper_control(task_plan.arm1_gripper_action, task_plan.arm2_gripper_action)
                # go up
                arm1_delta_z = 2 * self.pre_grasp_height if task_plan.arm1_assigned_object not in [None, -1] else 0
                arm2_delta_z = 2 * self.pre_grasp_height if task_plan.arm2_assigned_object not in [None, -1] else 0
                cmd_plan = self.move_z_2(arm1_delta_z, arm2_delta_z, task_plan.arm1_angle, task_plan.arm2_angle)
                self.recorded_plan.append(cmd_plan)
                self.visualize_plan(cmd_plan)
            else:
                # one arm by one arm
                tmp_goal_arm1 = self.default_ee_pos[self.arm1]
                tmp_goal_arm2 = self.default_ee_pos[self.arm2]
                # move arm2 to goal, arm1 to retract pose
                result = self.plan_single_from_curr_js(goal_arm1=tmp_goal_arm1, goal_arm2=goal_arm2, arm_angle1=task_plan.arm1_angle, arm_angle2=task_plan.arm2_angle)
                cmd_plan = self.get_plan_single(result)
                cmd_plan = self.toppra_improvement(cmd_plan, speed_scale=0.5, acc_scale=1)
                self.recorded_plan.append(cmd_plan)
                self.visualize_plan(cmd_plan)

                # go down
                arm1_delta_z = 0
                arm2_delta_z = -self.pre_grasp_height if task_plan.arm2_assigned_object not in [None, -1] else 0
                cmd_plan = self.move_z_2(arm1_delta_z, arm2_delta_z, task_plan.arm1_angle, task_plan.arm2_angle)
                self.recorded_plan.append(cmd_plan)
                self.visualize_plan(cmd_plan)

                self.gripper_control(GripperAction(self.gripper_closed[self.arm1]), task_plan.arm2_gripper_action)
                # go up
                arm1_delta_z = 0
                arm2_delta_z = 2 * self.pre_grasp_height if task_plan.arm2_assigned_object not in [None, -1] else 0
                cmd_plan = self.move_z_2(arm1_delta_z, arm2_delta_z, task_plan.arm1_angle, task_plan.arm2_angle)
                self.recorded_plan.append(cmd_plan)
                self.visualize_plan(cmd_plan)

                # move the other arm
                result = self.plan_single_from_curr_js(goal_arm1=goal_arm1, goal_arm2=tmp_goal_arm2, arm_angle1=task_plan.arm1_angle, arm_angle2=task_plan.arm2_angle)
                cmd_plan = self.get_plan_single(result)
                cmd_plan = self.toppra_improvement(cmd_plan, speed_scale=0.5, acc_scale=1)
                self.recorded_plan.append(cmd_plan)
                self.visualize_plan(cmd_plan)
                # go down
                arm2_delta_z = 0
                arm1_delta_z = -self.pre_grasp_height if task_plan.arm2_assigned_object not in [None, -1] else 0
                cmd_plan = self.move_z_2(arm1_delta_z, arm2_delta_z, task_plan.arm1_angle, task_plan.arm2_angle)
                self.recorded_plan.append(cmd_plan)
                self.visualize_plan(cmd_plan)

                self.gripper_control(task_plan.arm1_gripper_action, GripperAction(self.gripper_closed[self.arm2]))
                # go up
                arm2_delta_z = 0
                arm1_delta_z = 2 * self.pre_grasp_height if task_plan.arm2_assigned_object not in [None, -1] else 0
                cmd_plan = self.move_z_2(arm1_delta_z, arm2_delta_z, task_plan.arm1_angle, task_plan.arm2_angle)
                self.recorded_plan.append(cmd_plan)
                self.visualize_plan(cmd_plan)

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
                self.recorded_plan.append(cmd_plan)
                self.visualize_plan(cmd_plan)

        # check left and right on the y-axis
        if 0 < pos2[0] - pos1[0] < 0.2:
            if 0 < pos1[1] - pos2[1] < 0.1:
                # arm1 on the left, separate them
                delta_y = 0.3 
                pos1[1] = min(pos1[1] + delta_y, 0.3)
                pos2[1] = max(pos2[1] - delta_y, -0.3)
                cmd_plan = self.move_ee(pos1, angle1, pos2, angle2)
                self.recorded_plan.append(cmd_plan)
                self.visualize_plan(cmd_plan)
                # cmd_plan_total = cmd_plan_total.stack(cmd_plan)
            elif -0.1 < pos1[1] - pos2[1] < 0:
                # arm2 on the left
                delta_y = 0.3 
                pos1[1] = max(pos1[1] - delta_y, -0.3)
                pos2[1] = min(pos2[1] + delta_y, 0.3)
                cmd_plan = self.move_ee(pos1, angle1, pos2, angle2)
                self.recorded_plan.append(cmd_plan)
                self.visualize_plan(cmd_plan)
            
        # move back a little bit to seperate them on x-axis
        if pos2[0] - pos1[0] < 0.3:
            delta_x = 0.3
            pos1[0] = max(pos1[0]-delta_x, -0.3)
            pos2[0] = min(pos1[0]+delta_x, 0.3)
            cmd_plan = self.move_ee(pos1, angle1, pos2, angle2)
            self.recorded_plan.append(cmd_plan)
            self.visualize_plan(cmd_plan)
        
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

            cmd_plan = self.move_ee(pos1, new_angle1, pos2, new_angle2)
            self.recorded_plan.append(cmd_plan)
            self.visualize_plan(cmd_plan)
            
            # align with the goal on the x-axis
            if new_angle1 in [450, -450]:
                pos1[0] = goal_arm1[0]
            if new_angle2 in [450, -450]:
                pos2[0] = goal_arm2[0]
            
            cmd_plan = self.move_ee(pos1, new_angle1, pos2, new_angle2)
            self.recorded_plan.append(cmd_plan)
            self.visualize_plan(cmd_plan)

        else:
            pos1[1] = goal_arm1[1]
            pos2[1] = goal_arm2[1]

            cmd_plan = self.move_ee(pos1, new_angle1, pos2, new_angle2)
            self.recorded_plan.append(cmd_plan)
            self.visualize_plan(cmd_plan)
    

    def batch_motion_plan(self, task_plan):
        task_execution_result = TaskExecutionResult()
        # if no available task found for both arms, sample intermediate goals for one arm
        angle1 = task_plan.arm1_angle
        angle2 = task_plan.arm2_angle
        if task_plan.arm1_assigned_object == -1 or task_plan.arm1_assigned_object is None:
            goal_arm1 = self.default_ee_pos[self.arm1]
        else:
            goal_arm1 = self.task.get_start( task_plan.arm1_assigned_object ) if task_plan.arm1_to_start else self.task.get_goal( task_plan.arm1_assigned_object )
        if task_plan.arm2_assigned_object == -1 or task_plan.arm2_assigned_object is None:
            goal_arm2 = self.default_ee_pos[self.arm2]
        else:
            goal_arm2 = self.task.get_start( task_plan.arm2_assigned_object ) if task_plan.arm2_to_start else self.task.get_goal( task_plan.arm2_assigned_object )
        prioritize_arm1 = task_plan.prioritize_arm1
        results = self.plan_batch_from_curr_js_all(goal_arm1, goal_arm2, angle1, angle2, prioritize_arm1=prioritize_arm1)
        if results.success[0]:
            print("Original Plan successful")
            task_execution_result.arm1_task_state = TaskState.IN_PROGRESS if task_plan.arm1_to_start else TaskState.COMPLETED
            task_execution_result.arm2_task_state = TaskState.IN_PROGRESS if task_plan.arm2_to_start else TaskState.COMPLETED
            cmd_plan = results.optimized_plan[0]
            cmd_plan = self.get_plan_batch(cmd_plan)
        else:
            print("Picking a random successful plan")
            # randomly pick one from the successful ones
            success_indices = torch.where(results.success)[0]
            if len(success_indices) == 0:
                return
            idx = random.choice(success_indices)
            # cmd_plan = results.optimized_plan[idx]
            # cmd_plan = self.get_plan_batch(cmd_plan)
            result = MotionGenResult(
                success=True,
                optimized_plan=results.optimized_plan[idx],
                optimized_dt=results.optimized_dt[idx],
                position_error=results.position_error[idx],
                interpolated_plan=results.interpolated_plan[idx],
                interpolation_dt=results.interpolation_dt,
                path_buffer_last_tstep=[results.path_buffer_last_tstep[idx]],
            )
            result.retime_trajectory(time_dilation_factor=0.5)
            cmd_plan = self.get_plan_single(result)
            if prioritize_arm1:
                task_execution_result.arm1_task_state = TaskState.IN_PROGRESS if task_plan.arm1_to_start else TaskState.COMPLETED
                task_execution_result.arm2_task_state = TaskState.IN_PROGRESS
                task_execution_result.arm2_at_intermediate = True
                task_execution_result.arm1_at_intermediate = False
                task_plan.arm2_gripper_action = task_plan.arm2_gripper_action.reverse()
                print("Prioritizing arm1, reversing arm2 gripper action")
            else:
                task_execution_result.arm1_task_state = TaskState.IN_PROGRESS 
                task_execution_result.arm2_task_state = TaskState.IN_PROGRESS if task_plan.arm2_to_start else TaskState.COMPLETED
                task_execution_result.arm1_at_intermediate = True
                task_execution_result.arm2_at_intermediate = False
                task_plan.arm1_gripper_action = task_plan.arm1_gripper_action.reverse()
                print("Prioritizing arm2, reversing arm1 gripper action")
        task_execution_result.arm1_assigned_object = task_plan.arm1_assigned_object 
        task_execution_result.arm2_assigned_object = task_plan.arm2_assigned_object
        task_execution_result.arm1_gripper_state = GripperState(task_plan.arm1_gripper_action.value)
        task_execution_result.arm2_gripper_state = GripperState(task_plan.arm2_gripper_action.value)
        task_execution_result.arm1_angle = task_plan.arm1_angle
        task_execution_result.arm2_angle = task_plan.arm2_angle
        task_execution_result.buffer = task_plan.buffer
        return cmd_plan, task_execution_result
    
    def plan(self, autostart=False):
        self.get_ready(reset=True, autostart=autostart)
        while not self.task.finished():
            task_plan = self.task.plan()
            task_execution_result = TaskExecutionResult()
            if task_plan.start_goal_pairs is not None and not task_plan.use_buffer:
                print("Pick two from the start_goal_pairs")
                self.find_available_task(task_plan)
                task_execution_result = self.motion_plan(task_plan)
            elif task_plan.use_buffer:
                print("Using buffer for one arm")
                self.find_available_task_with_buffers(task_plan)
                task_execution_result = self.motion_plan(task_plan)
            elif task_plan.start_goal_pairs is None:
                print("Start and Goal pairs are None")
                if task_plan.arm2_assigned_object is not None and task_plan.arm1_assigned_object == -1:
                    print("Finishing up for one arm2")
                    if task_plan.arm2_angle is None:
                        self.find_angle_for_single_arm(task_plan, self.arm2)
                    task_execution_result = self.motion_plan(task_plan)
                elif task_plan.arm1_assigned_object is not None and task_plan.arm2_assigned_object == -1:  
                    print("Finishing up for one arm1")
                    if task_plan.arm1_angle is None:
                        self.find_angle_for_single_arm(task_plan, self.arm1)
                    task_execution_result = self.motion_plan(task_plan)
                # elif (not task_plan.arm1_to_start) and (not task_plan.arm2_to_start):
                else:
                    print("Finishing up for both arms")
                    task_execution_result = self.motion_plan(task_plan)
            
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
            self.old_task_plan = task_plan
            self.task.update(task_execution_result)
        return self.recorded_plan
    
    def check_object_collision_enabled(self):
        print("Single_motion_gen")
        for obj in self.objects_curobo:
            name = obj.name
            env_idx = 0
            obs_idx = self.single_motion_gen.world_coll_checker.get_obb_idx(name, env_idx)
            print(f"{name} single_motion_gen collision enabled = ", self.single_motion_gen.world_coll_checker._cube_tensor_list[2][env_idx, obs_idx].cpu().numpy(), end=' ') 
        print()
        for obj in self.objects_curobo:
            name = obj.name
            env_idx = 0
            obs_idx = self.single_motion_gen.world_coll_checker.get_obb_idx(name, env_idx)
            print(f"{name} motion_gen collision enabled = ", self.motion_gen.world_coll_checker._cube_tensor_list[2][env_idx, obs_idx].cpu().numpy())
        print()

    def get_current_js(self):
        sim_js = self.robot.get_joints_state()
        sim_js_positions = sim_js.positions.reshape(1, -1)
        cu_js = JointState.from_position(self.tensor_args.to_device(sim_js_positions), self.dof_names)
        cu_js = cu_js.get_ordered_joint_state(self.single_motion_gen.kinematics.joint_names)
        return cu_js
        
    def visualize_robot_spheres(self):
        sim_js = self.get_current_arm_js()
        sph_list = self.single_motion_gen.kinematics.get_robot_as_spheres(self.tensor_args.to_device(sim_js.position))

        if self.spheres is None:
            self.spheres = []
            # create spheres:
            for si, s in enumerate(sph_list[0]):
                sp = sphere.VisualSphere(
                    prim_path="/curobo/robot_sphere_" + str(si),
                    position=np.ravel(s.position),
                    radius=float(s.radius),
                    color=np.array([0, 0.8, 0.2]),
                )
                self.spheres.append(sp)
        else:
            for si, s in enumerate(sph_list[0]):
                self.spheres[si].set_world_pose(position=np.ravel(s.position))
                self.spheres[si].set_radius(float(s.radius))


            
    def visualize_plan(self, plan: JointState):
        if plan is None:
            print("Null JointState plan.")
            return
        cmd_idx = 0
        while self.simulation_app.is_running():
            self.world.step(render=True)
            if not self.world.is_playing():
                continue
            cmd_state = plan[cmd_idx]
            # get full dof state
            positions = self.fix_gripper(cmd_state)
            velocities = cmd_state.velocity.cpu().numpy()
            art_action = ArticulationAction(
                joint_positions=positions,
                joint_velocities=velocities,
                joint_indices=self.common_js_indices
            )

            step_index = self.world.current_time_step_index

            if self.visualize_spheres and step_index % 2 == 0:
                self.visualize_robot_spheres()
            # set desired joint angles obtained from IK:
            self.articulation_controller.apply_action(art_action)
            cmd_idx += 1
            for _ in range(5):
                self.world.step(render=False)
            if cmd_idx >= len(plan.position):
                return
    
    def find_available_task(self, task_plan: TaskPlan):
        # first, find the avaiable task that uses vertical pick and place
        print("finding vertical pick and place")
        for combo in self.preferred_angles_combination:
            for indices in task_plan.start_goal_pairs:
                start1 = self.task.get_start( indices[0] )
                goal1 = self.task.get_goal( indices[0] )
                start2 = self.task.get_start( indices[1] )
                goal2 = self.task.get_goal( indices[1] )
                angle1, angle2 = combo
                start_ik_result = self.ik_solve_single(start1, start2, angle1, angle2)
                goal_ik_result = self.ik_solve_single(goal1, goal2, angle1, angle2)
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
                start1 = self.task.get_start( indices[0] )
                goal1 = self.task.get_goal( indices[0] )
                start2 = self.task.get_start( indices[1] )
                goal2 = self.task.get_goal( indices[1] )
                angle1, angle2 = combo
                start_ik_result = self.ik_solve_single(start1, start2, angle1, angle2)
                goal_ik_result = self.ik_solve_single(goal1, goal2, angle1, angle2)
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
                start1 = self.task.get_start( indices[0] )
                goal1 = self.task.get_goal( indices[0] )
                start2 = self.task.get_start( indices[1] )
                goal2 = self.task.get_goal( indices[1] )
                angle1, angle2 = combo
                default_ee_pos1 = self.default_ee_pos[self.arm1]
                default_ee_pos2 = self.default_ee_pos[self.arm2]
                start_ik_result1 = self.ik_solve_single(default_ee_pos1, start2, angle1, angle2)
                start_ik_result2 = self.ik_solve_single(start1, default_ee_pos2, angle1, angle2)
                goal_ik_result1 = self.ik_solve_single(default_ee_pos1, goal2, angle1, angle2)
                goal_ik_result2 = self.ik_solve_single(goal1, default_ee_pos2, angle1, angle2)
                if start_ik_result1.success and start_ik_result2.success and goal_ik_result1.success and goal_ik_result2.success:
                    # [start_pos, goal_pos, orientation]
                    task_plan.arm1_assigned_object = indices[0]
                    task_plan.arm2_assigned_object = indices[1]
                    task_plan.arm1_angle = angle1
                    task_plan.arm2_angle = angle2
                    return 
        

    
    def find_available_task_with_buffers(self, task_plan: TaskPlan):
        #TODO: make this buffer size configurable
        buffers = self.task.sample_buffers(self.num_buffers)
        for combo in self.preferred_angles_combination:
            for indices in task_plan.start_goal_pairs:
                goal1_dependent_on_goal2 = self.task.is_direct_dependent(indices[0], indices[1])
                start1 = self.task.get_start( indices[0] )
                goal1 = self.task.get_goal( indices[0] )
                start2 = self.task.get_start( indices[1] )
                goal2 = self.task.get_goal( indices[1] )
                angle1, angle2 = combo
                start_ik_result = self.ik_solve_single(start1, start2, angle1, angle2)
                if not start_ik_result.success:
                    continue
                # use buffer for goal2
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
                object_positions, _ = self.get_current_cube_pose()
                furthest_buffer = None
                max_min_dist = -1
                for buffer in available_buffers:
                    dists = [np.linalg.norm(buffer - obj_pos) for obj_pos in object_positions]
                    min_dist = min(dists)
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        furthest_buffer = buffer

                idx = random.choice(success_indices)
                task_plan.buffer = {1: None, 2: None}
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
            #TODO: fix this
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
        
    def get_robot_dof_indices(self, joint_names: list):
        return [self.robot.get_dof_index(name) for name in joint_names]

    def get_plan_batch(self, cmd_plan):
        cmd_plan = self.motion_gen.get_full_js(cmd_plan)
        # get only joint names that are in both:
        if self.common_js_names == [] and self.common_js_indices == []:
            for x in self.dof_names:
                if x in cmd_plan.joint_names:
                    self.common_js_indices.append(self.robot.get_dof_index(x))
                    self.common_js_names.append(x)
    
        cmd_plan = cmd_plan.get_ordered_joint_state(self.common_js_names)
        return cmd_plan

    def get_plan_single(self, result: MotionGenResult):
        cmd_plan = result.get_interpolated_plan()
        cmd_plan = self.single_motion_gen.get_full_js(cmd_plan)
        # get only joint names that are in both:
        if self.common_js_names == [] and self.common_js_indices == []:
            for x in self.dof_names:
                if x in cmd_plan.joint_names:
                    self.common_js_indices.append(self.robot.get_dof_index(x))
                    self.common_js_names.append(x)
    
        cmd_plan = cmd_plan.get_ordered_joint_state(self.common_js_names)
        return cmd_plan

    def fix_gripper(self, js: JointState):
        gripper_primary_pos_dict = self.get_primary_gripper_position_dict()
        positions = js.position.cpu().numpy()
        for arm, gripper_primary_name in self.gripper_primary_joints_dict.items():
            idx = self.common_js_names.index(gripper_primary_name)
            positions[idx] = gripper_primary_pos_dict[gripper_primary_name] + 0.002 * self.gripper_closed[arm]
            # positions[idx] = gripper_primary_pos_dict[gripper_primary_name] 
        return positions
    
    def get_current_arm_js(self):
        sim_js = self.robot.get_joints_state()
        sim_js_positions = sim_js.positions.reshape(1, -1)
        cu_js = JointState.from_position(self.tensor_args.to_device(sim_js_positions), self.dof_names)
        cu_js = self.remove_gripper_dof_from_js(cu_js)
        return cu_js
    
    def attach_obj(
        self,
        cu_js: JointState,
        cube_name: str,
        arm_name: str
    ) -> None:
        for mg in [self.single_motion_gen, self.motion_gen]:
            mg.attach_objects_to_robot(
                cu_js,
                [cube_name],
                sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
                # sphere_fit_type=SphereFitType.SAMPLE_SURFACE,
                # world_objects_pose_offset=Pose.from_list([0, 0, 0.01, 1, 0, 0, 0], self.tensor_args),
                link_name=arm_name
            )
            mg.world_coll_checker.enable_obstacle(enable=False, name=cube_name)
        if self.visualize_spheres:
            self.visualize_robot_spheres()

    def detach_obj(self, arm_name: str, cube_name: str) -> None:
        for mg in [self.single_motion_gen, self.motion_gen]:
            mg.detach_object_from_robot(link_name=arm_name)
            mg.world_coll_checker.enable_obstacle(enable=True, name=cube_name)
        if self.visualize_spheres:
            self.visualize_robot_spheres()

    def add_opacity_material(
        self,
        material_name: str,
        object_path: str,
        color: List[float],
        obj_prim: Usd.Prim,
        material: Material = Material(),
    ):
        mat_path = join_path(object_path, material_name)
        material_usd = UsdShade.Material.Define(self.stage, mat_path)
        pbrShader = UsdShade.Shader.Define(self.stage, join_path(mat_path, "PbrShader"))
        pbrShader.CreateIdAttr("UsdPreviewSurface")
        pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(material.roughness)
        pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(material.metallic)
        pbrShader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(color[:3]))
        pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(color[:3]))
        pbrShader.CreateInput("baseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(color[:3]))
        # somehow we need this emissiveColor here
        pbrShader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(color[:3]))

        pbrShader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(color[3])
        material_usd.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), "surface")
        obj_prim.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
        UsdShade.MaterialBindingAPI(obj_prim).Bind(material_usd)
        return material_usd
    
    def add_physics_material(
        self,
        material_name: str,
        object_path: str,
        dynamic_friction: float,
        static_friction: float,
        restitution: float,
        obj_prim: Usd.Prim,
    ):
        mat_path = join_path(object_path, material_name)
        material_api = UsdPhysics.MaterialAPI.Apply(obj_prim.GetPrim())
        material_api.CreateDynamicFrictionAttr(dynamic_friction)
        material_api.CreateStaticFrictionAttr(static_friction)
        material_api.CreateRestitutionAttr(restitution)

    def test_different_grasp_angles(self):
        self.get_ready()
        angles = [900, 450, -450, 901, 451, -451]
        # angles = [901, 451, -451]
        cube_positions, _ = self.get_current_cube_pose()
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
        positions = []
        rotations = []
        for i in range(self.task.num_objects):
            cube_isaac = self.objects_isaacsim_dynamic[i]
            cube_curobo = self.objects_curobo[i]
            p, q = cube_isaac.get_world_pose()
            positions.append(p)
            rotations.append(q)
        return positions, rotations

    def sync_isaacsim_curobo(self, idx_to_sync: list = None):
        cuboids = []
        if idx_to_sync is None:
            idx_to_sync = list(range(self.task.num_objects))
        new_starts = {}
        for i in idx_to_sync:
            cube_isaac = self.objects_isaacsim_dynamic[i]
            cube_curobo = self.objects_curobo[i]
            p, q = cube_isaac.get_world_pose()
            cube_curobo.pose = [p[0], p[1], p[2], q[0], q[1], q[2], q[3]]
            cuboids.append(cube_curobo)
            new_starts[i] = np.array([p[0], p[1], p[2]])
        self.task.set_starts(new_starts)
        world_cfg_table = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        )
        world_cfg_table.cuboid[0].pose[2] -= 0.02
        # world_cfg1 = WorldConfig.from_dict(
        #     load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        # ).get_mesh_world()
        # world_cfg1.mesh[0].name += "_mesh"
        # world_cfg1.mesh[0].pose[2] = -10.5
        # self.world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)
        self.world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid)
        obs = cuboids + world_cfg_table.cuboid
        # self.world_cfg_curobo = WorldConfig(cuboid=obs, mesh=world_cfg1.mesh)
        self.world_cfg_curobo = WorldConfig(cuboid=obs)
        self.motion_gen.clear_world_cache()
        self.single_motion_gen.clear_world_cache()
        self.motion_gen.update_world(self.world_cfg_curobo)
        self.single_motion_gen.update_world(self.world_cfg_curobo)
        self.ik_solver.update_world(self.world_cfg_curobo)
        print("self.single_motion_gen.world_coll_checker.get_obstacle_names() = ", self.single_motion_gen.world_coll_checker.get_obstacle_names())
        print("       self.motion_gen.world_coll_checker.get_obstacle_names() = ", self.single_motion_gen.world_coll_checker.get_obstacle_names())
        # mesh = self.single_motion_gen.world_coll_checker.get_mesh_in_bounding_box(
        #     cuboid=Cuboid(name="test", dims=[2,2,2], pose=[0, 0, -0.1,1,0,0,0]),
        #     voxel_size=0.01,
        #     run_marching_cubes=True
        # )
        # mesh.save_as_mesh(file_path="results/scene.obj")


    def display(self, reset=False):
        if reset:
            self.get_ready(reset=reset)
        while self.simulation_app.is_running():
            self.world.step(render=True)
    
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
        return interpolated_trajectories
        
    def move_ee(self, pos1, angle1, pos2, angle2):
        ik_result = self.ik_solve_single(pos1, pos2, angle1, angle2, num_solutions=1)
        curr_js = self.get_current_arm_js()
        curr_js = self.single_motion_gen.get_full_js(curr_js)
        if self.common_js_names == [] and self.common_js_indices == []:
            for x in self.dof_names:
                if x in curr_js.joint_names:
                    self.common_js_indices.append(self.robot.get_dof_index(x))
                    self.common_js_names.append(x)
        curr_js = curr_js.get_ordered_joint_state(self.common_js_names)
        start_js = curr_js.position.squeeze(0).view(-1)
        goal_js = ik_result.js_solution.get_ordered_joint_state(self.common_js_names).position.squeeze(0).view(-1)
        # idx_list = [self.robot.get_dof_index(x) for x in self.common_js_names]
        # goal_js_display = goal_js.cpu().tolist()
        # self.robot.set_joint_positions(goal_js_display, idx_list)
        # self.robot.set_joint_velocities(np.zeros(len(idx_list)), idx_list)
        # for _ in range(100):
        #     self.world.step(render=True)
        traj = self.interpolate_joint_states(curr_js.position, goal_js)
        cmd_plan = JointState.from_position(traj, self.common_js_names)
        # self.visualize_plan(cmd_plan)
        return cmd_plan
    
    def move_z(self, delta, angle1, angle2):
        ee_pos = self.get_current_ee_pos()
        pos1 = ee_pos[self.arm1]
        pos2 = ee_pos[self.arm2]
        print("move_z: pos1 = ", pos1, "pos2 = ", pos2)
        pos1[2] += delta
        pos2[2] += delta
        return self.move_ee(pos1, angle1, pos2, angle2)
    
    def move_z_2(self, delta1, delta2, angle1, angle2):
        ee_pos = self.get_current_ee_pos()
        pos1 = ee_pos[self.arm1]
        pos2 = ee_pos[self.arm2]
        print("move_z: pos1 = ", pos1, "pos2 = ", pos2)
        pos1[2] += delta1
        pos2[2] += delta2
        return self.move_ee(pos1, angle1, pos2, angle2)
    
    def move_x(self, delta, angle1, angle2):
        ee_pos = self.get_current_ee_pos()
        pos1 = ee_pos[self.arm1]
        pos2 = ee_pos[self.arm2]
        print("move_x: pos1 = ", pos1, "pos2 = ", pos2)
        pos1[0] += delta
        pos2[0] -= delta
        return self.move_ee(pos1, angle1, pos2, angle2)
    
    def move_y(self, delta, angle1, angle2):
        ee_pos = self.get_current_ee_pos()
        pos1 = ee_pos[self.arm1]
        pos2 = ee_pos[self.arm2]
        print("move_y: pos1 = ", pos1, "pos2 = ", pos2)
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
        self.get_ready()
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
        traj = JointState.from_position(traj.view(-1,waypoints_.shape[1]))
        return traj
    
    def play(self, recorded_plan):
        self.get_ready(reset=True)
        for section in recorded_plan:
            if type(section) == list:
                self.gripper_control(section[0], section[1])
            else:
                self.visualize_plan(section)


# debug
if __name__ == '__main__':
    num_objects:int = 4  
    map_width:float = 0.6 
    map_length:float = 0.6 
    object_size:float = 0.05 
    task = Task(num_objects, map_width, map_length, object_size)
    # task.generate()
    task.generate_fixed_two_objects(offset=0.)
    # task.generate_chained_two_objects()
    # task.generate_fixed_cycled_objects()
    # task.generate_fixed_two_objects_to_test_grasp_orientation()
    # task.generate_fixed_two_to_test_batch_planner()
    # task.generate_fixed_four_objects()
    # task.generate_chained_four_objects()
    # task.generate_fixed_six_objects()
    # task = Task.read_from_json("data/task_random_9_objects_17.json")
    dual_arm = DualArmEnv(task,
                          simulation_app,
                          batch_size = 3,
                          constrain_grasp_approach=False,
                          visualize_spheres=False,
                          )
    # dual_arm.reset()
    # dual_arm.test_ik_solve_batch()
    # dual_arm.test_plan_batch()
    # dual_arm.display()
    dual_arm.test_different_grasp_angles()
    # dual_arm.test_gripper_control()
    # dual_arm.test_grasp()
    # dual_arm.test_move_ee()
    # plan = dual_arm.plan()
    # dual_arm.play(plan)
    
