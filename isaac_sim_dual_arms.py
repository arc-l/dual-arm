import curobo
import os, json
os.environ['CUROBO_TORCH_CUDA_GRAPH_RESET'] = '1'
import random
from itertools import permutations, product
from packaging import version

import numpy as np
import time
import isaacsim
from typing import List
from omni.isaac.kit import SimulationApp
import helper
from helper import save_plan, load_plan 
if __name__ == '__main__':
    if helper.HEADLESS:
        simulation_app = None
    else:
        simulation_app = SimulationApp({
                        "headless": False,
                        "width": "1920",
                        "height": "1080",
        })

if not helper.HEADLESS:
    from pxr import Usd, UsdPhysics, Sdf, UsdGeom, PhysxSchema, Gf, UsdShade
    from curobo_isaacsim_helper import add_extensions, add_robot_to_scene
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
    from curobo.util.usd_helper import UsdHelper

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

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

from curobo.geom.types import WorldConfig, Cuboid, Material
from curobo.types.robot import JointState, RobotConfig
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml

from dual_arms import DualArmEnv
from task import Task, TaskPlan, TaskState, GripperState, TaskExecutionResult, ArmState, GripperAction


class IsaacSimDualArms(DualArmEnv):
    def __init__(self, 
            task: Task,
            simulation_app: SimulationApp|None,
            headless: bool = True,
            batch_size: int = 3,
            num_buffers: int = 20,
            pure_curobo: bool = False,
            ):
        super().__init__(task, batch_size=batch_size, headless=headless, num_buffers=num_buffers, pure_curobo=pure_curobo)
        self.simulation_app = simulation_app
        if simulation_app is None:
            self.headless_isaac = True
        else:
            self.headless_isaac = False
        
        if not self.headless_isaac:
            self.world = World(stage_units_in_meters=1.0)
            self.stage = self.world.stage
            xform = self.stage.DefinePrim("/World", "Xform")
            self.stage.SetDefaultPrim(xform)
            self.stage.DefinePrim("/curobo", "Xform")
            self.usd_help = UsdHelper()
            self.robot, self.robot_prim_path = add_robot_to_scene(self.robot_cfg, self.world)
            self.set_material_for_finger_pads()
            # Check screen boundaries
            self.dof_names = []
            self.articulation_controller = self.robot.get_articulation_controller()
            add_extensions(self.simulation_app, None)
            self.usd_help.load_stage(self.stage)
            self.usd_help.add_world_to_stage(self.world_cfg, base_frame="/World", enable_physics=False)
            self.world.scene.add_default_ground_plane()
            self.common_js_names = []
            self.common_js_indices = []
            self.objects_isaacsim_dynamic = []
            self.objects_isaacsim_visual = []

    
    def add_objects_isaacsim(self):
        starts = self.task.get_original_starts()
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
    
    def get_ready(self, reset=False, autostart=False):
        if self.headless_isaac:
            self.reset_isaac()
            return
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
                self.reset_isaac()
                reset=False

            if step_index < 20:
                continue
            return
    
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
    
    def gripper_control_isaac(self, gripper1_action: GripperAction, gripper2_action: GripperAction):
        if self.headless_isaac:
            self.gripper_closed[self.arm1] = bool(gripper1_action.value)
            self.gripper_closed[self.arm2] = bool(gripper2_action.value)
            return
        dof_upper = self.robot.dof_properties["upper"]
        gripper_closed_pos = dof_upper[self.gripper_primary_indices[0]] * 0.6
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

    def test_gripper_control_isaac(self):
        self.get_ready()
        for i in range(3):
            self.gripper_control_isaac(GripperAction.CLOSE, GripperAction.OPEN)
            self.gripper_control_isaac(GripperAction.OPEN, GripperAction.OPEN)

        for i in range(3):
            self.gripper_control_isaac(GripperAction.OPEN, GripperAction.CLOSE)
            self.gripper_control_isaac(GripperAction.OPEN, GripperAction.OPEN)

        for i in range(3):
            self.gripper_control_isaac(GripperAction.CLOSE, GripperAction.CLOSE)
            self.gripper_control_isaac(GripperAction.OPEN, GripperAction.OPEN)

        for i in range(3):
            self.gripper_control_isaac(GripperAction.CLOSE, GripperAction.OPEN)
            self.gripper_control_isaac(GripperAction.OPEN, GripperAction.CLOSE)

        self.gripper_control_isaac(GripperAction.OPEN, GripperAction.OPEN)

    def reset_isaac(self):
        if not self.headless_isaac:
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
            self.robot.set_joint_positions(self.default_config, idx_list)
            self.robot.set_joint_velocities(np.zeros(len(idx_list)), idx_list)
            self.dof_names = self.robot.dof_names
            self.robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), 
                joint_indices=idx_list
            )
            if self.common_js_names == [] and self.common_js_indices == []:
                for x in self.dof_names:
                    if x in self.j_names_primary:
                        self.common_js_indices.append(self.robot.get_dof_index(x))
                        self.common_js_names.append(x)

        kin_state = self.single_motion_gen.kinematics.get_state(self.single_motion_gen.get_retract_config().view(1, -1))
        link_poses = kin_state.link_poses
        self.default_ee_pos = {}
        for k, v in link_poses.items():
            t_pos = np.ravel(v.to_list())
            self.default_ee_pos[k] = t_pos[:3]
            self.default_ee_pos[k][2] += 0.15 #TODO: Test this, added 0.15m to z
        
        if not self.headless_isaac:
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

        if not self.headless_isaac:
            for arm, joint_name in self.gripper_primary_joints_dict.items():
                self.gripper_primary_indices_dict[arm] = self.robot.get_dof_index(joint_name)
            self.robot._articulation_view.set_gains(
                kps = 100 * np.ones(len(self.gripper_joint_indices)), 
                kds= 100 * np.ones(len(self.gripper_joint_indices)), 
                joint_indices=self.gripper_joint_indices
                )
            print("self.robot.dof_properties = ", self.robot.dof_properties)
            self.add_objects_isaacsim()
        
        # self.arm_joint_names = [name for name in self.j_names_primary if name not in self.gripper_primary_joints]
        # self.arm_joint_names = self.single_motion_gen.kinematics.joint_names

        # self.sync_bullet_curobo()
        # print("self.single_motion_gen.world_coll_checker.get_obstacle_names() = ", self.single_motion_gen.world_coll_checker.get_obstacle_names())
    
    def visualize_plan_isaac(self, plan: JointState):
        if plan is None:
            print("Null JointState plan.")
            return

        if plan.joint_names != self.j_names_primary:
            reordered = plan.get_ordered_joint_state(self.arm_joint_names)
            positions = reordered.position
            if positions.dim() == 1:
                positions = positions.unsqueeze(0)
            pad = torch.zeros(
                (positions.shape[0], len(self.gripper_primary_joints)),
                dtype=positions.dtype,
                device=positions.device,
            )
            positions = torch.cat([positions, pad], dim=-1)
            joint_names = list(self.arm_joint_names) + list(self.gripper_primary_joints)
            plan = JointState.from_position(positions, joint_names)

        if self.headless_isaac:
            cu_js = plan[-1].get_ordered_joint_state(self.j_names_primary)
            self.current_js = JointState.from_position(self.tensor_args.to_device(cu_js.position.reshape(1, -1)), self.j_names_primary)
            return

        cmd_idx = 0
        while self.simulation_app.is_running():
            self.world.step(render=True)
            if not self.world.is_playing():
                continue

            cmd_state = plan[cmd_idx]
            if cmd_state.joint_names != self.j_names_primary:
                # cmd_state = cmd_state.get_ordered_joint_state(self.j_names_primary)
                # cmd_state = self.single_motion_gen.get_full_js(cmd_state)
                cmd_state = cmd_state.get_ordered_joint_state(self.common_js_names)
            # get full dof state
            positions = self.fix_gripper(cmd_state)
            velocities = cmd_state.velocity.cpu().numpy()
            art_action = ArticulationAction(
                joint_positions=positions,
                joint_velocities=velocities,
                joint_indices=self.common_js_indices
            )

            step_index = self.world.current_time_step_index

            # set desired joint angles obtained from IK:
            self.articulation_controller.apply_action(art_action)
            cmd_idx += 1
            for _ in range(5):
                self.world.step(render=False)
            if cmd_idx >= len(plan.position):
                return
    
    def add_opacity_material(
        self,
        material_name: str,
        object_path: str,
        color: List[float],
        obj_prim, #Usd.Prim
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
        obj_prim, #Usd.Prim
        ):
        mat_path = join_path(object_path, material_name)
        material_api = UsdPhysics.MaterialAPI.Apply(obj_prim.GetPrim())
        material_api.CreateDynamicFrictionAttr(dynamic_friction)
        material_api.CreateStaticFrictionAttr(static_friction)
        material_api.CreateRestitutionAttr(restitution)

    def display(self, reset=False):
        if reset:
            self.get_ready(reset=reset)
        while self.simulation_app.is_running():
            self.world.step(render=True)
    
    def play(self, recorded_plan, reset_bullet=False):
        if reset_bullet:
            self.reset()
        self.get_ready(reset=True)
        for section in recorded_plan:
            if type(section) == list:
                self.gripper_control_isaac(section[0], section[1])
            else:
                self.visualize_plan_isaac(section)

    def fix_gripper(self, js: JointState):
        gripper_primary_pos_dict = self.get_primary_gripper_position_dict()
        positions = js.position.cpu().numpy()
        for arm, gripper_primary_name in self.gripper_primary_joints_dict.items():
            idx = self.common_js_names.index(gripper_primary_name)
            positions[idx] = gripper_primary_pos_dict[gripper_primary_name] + 0.002 * self.gripper_closed[arm]
        return positions
    

if __name__ == "__main__":
    start = json.load(open(f"data/icra/A.json"))['goals']
    goal = json.load(open(f"data/icra/R.json"))['goals']
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
    dual_arms_env = IsaacSimDualArms(task, simulation_app, headless=False, pure_curobo=False)
    # plan = dual_arms_env.plan()
    # save_plan(plan, "real_robot_experiment.json", "ours")
    plan = load_plan("results/real_robot_experiment_ours_plan.pkl", device='cuda:0')
    dual_arms_env.play(plan, reset_bullet=True)
