#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

# Standard Library
from typing import Dict, List

# Third Party
import numpy as np
from matplotlib import cm
import isaacsim
# from isaacsim.core.api.world import World
# from isaacsim.core.api.materials.omni_pbr import OmniPBR
# from isaacsim.core.api.objects import cuboid
# from isaacsim.core.api.robots import Robot
from omni.isaac.core import World
from omni.isaac.core.materials import OmniPBR
from omni.isaac.core.objects import cuboid
from omni.isaac.core.robots import Robot
from pxr import UsdPhysics, Usd

# CuRobo
from curobo.util.logger import log_warn
from curobo.util.usd_helper import set_prim_transform

ISAAC_SIM_23 = False
try:
    # Third Party
    from omni.isaac.urdf import _urdf  # isaacsim 2022.2
except ImportError:
    # Third Party
    from omni.importer.urdf import _urdf  # isaac sim 2023.1 or above

    ISAAC_SIM_23 = True

# from isaacsim.asset.importer.urdf import _urdf

# Standard Library
from typing import Optional

# Third Party
from omni.isaac.core.utils.extensions import enable_extension
import omni.isaac.core.utils.prims as prims_utils

# CuRobo
from curobo.util_file import get_assets_path, get_filename, get_path_of_dir, join_path

import torch
from curobo.types.state import JointState

def joint_state_to_cpu(joint_state):
    """Convert all tensor attributes of a JointState to CPU."""
    position=joint_state.position.cpu() if isinstance(joint_state.position, torch.Tensor) else joint_state.position
    joint_names=joint_state.joint_names,
    cpu_joint_state = JointState.from_position(position, joint_names)
    return cpu_joint_state

import pickle
import os

def save_plan(plan, input_filename, method):
    """Save a list of JointStates to a pickle file after moving them to CPU."""
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)  # Create the results directory if it doesn't exist

    # Convert each JointState to CPU
    cpu_plan = []
    for js in plan:
        if type(js) == JointState:
            cpu_plan.append(joint_state_to_cpu(js))
        else:
            cpu_plan.append(js)

    # Extract filename without extension
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    output_filename = os.path.join(results_dir, f"{base_name}_{method}_plan.pkl")

    with open(output_filename, "wb") as file:
        pickle.dump(cpu_plan, file)

    print(f"Plan saved to {output_filename}")

def load_plan(filename, device='cpu'):
    """Load a pickled plan and move JointStates to the specified device."""
    with open(filename, "rb") as file:
        cpu_plan = pickle.load(file)

    # Define a helper to move JointState to the desired device
    def joint_state_to_device(joint_state, device):
        position=joint_state.position.to(device) if isinstance(joint_state.position, torch.Tensor) else joint_state.position
        joint_names=joint_state.joint_names,
        return JointState.from_position(position, joint_names)

    device_plan = [joint_state_to_device(js, device) if type(js)==JointState else js for js in cpu_plan]
    return device_plan


def add_extensions(simulation_app, headless_mode: Optional[str] = None):
    ext_list = [
        "omni.kit.asset_converter",
        "omni.kit.tool.asset_importer",
        "omni.isaac.asset_browser",
        # "isaacsim.asset.browser"
    ]
    if headless_mode is not None:
        log_warn("Running in headless mode: " + headless_mode)
        ext_list += ["omni.kit.livestream." + headless_mode]
    [enable_extension(x) for x in ext_list]
    simulation_app.update()

    return True


############################################################
def add_robot_to_scene(
    robot_config: Dict,
    my_world: World,
    load_from_usd: bool = False,
    subroot: str = "",
    robot_name: str = "robot",
    position: np.array = np.array([0, 0, 0]),
):

    if not load_from_usd:
        if robot_config["kinematics"]["urdf_path"] is None or robot_config["kinematics"]["urdf_path"] == "":
            raise ValueError("URDF path is not provided")

    if robot_config["kinematics"]["usd_path"] is None or robot_config["kinematics"]["usd_path"] == "":
        urdf_interface = _urdf.acquire_urdf_interface()
        # Set the settings in the import config
        import_config = _urdf.ImportConfig()
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.fix_base = True
        import_config.make_default_prim = True
        import_config.self_collision = False
        import_config.create_physics_scene = True
        import_config.import_inertia_tensor = False
        # import_config.default_drive_strength = 1047.19751
        import_config.default_drive_strength = 10000
        import_config.default_position_drive_damping = 52.35988
        import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
        # import_config.default_drive_type = _urdf.UrdfJointDriveType.JOINT_DRIVE_FORCE
        import_config.distance_scale = 1
        import_config.density = 0.0

        asset_path = get_assets_path()
        if (
            "external_asset_path" in robot_config["kinematics"]
            and robot_config["kinematics"]["external_asset_path"] is not None
        ):
            asset_path = robot_config["kinematics"]["external_asset_path"]
        full_path = join_path(asset_path, robot_config["kinematics"]["urdf_path"])
        robot_path = get_path_of_dir(full_path)
        filename = get_filename(full_path)
        imported_robot = urdf_interface.parse_urdf(robot_path, filename, import_config)
        dest_path = subroot
        robot_path = urdf_interface.import_robot(
            robot_path,
            filename,
            imported_robot,
            import_config,
            dest_path,
        )

        base_link_name = robot_config["kinematics"]["base_link"]

        robot_p = Robot(
            prim_path=robot_path + "/" + base_link_name,
            name=robot_name,
        )

        robot_prim = robot_p.prim
        stage = robot_prim.GetStage()
        linkp = stage.GetPrimAtPath(robot_path)
        set_prim_transform(linkp, [position[0], position[1], position[2], 1, 0, 0, 0])
    else:
        usd_path = robot_config["kinematics"]["usd_path"]
        asset_path = get_assets_path()
        full_path = join_path(asset_path, usd_path)
        print("full_path", full_path)
        # robot_path = "/" + usd_path.split("/")[-1].split(".")[0]
        robot_path = subroot + "/robot"
        robot_prim = prims_utils.create_prim(
            prim_path = robot_path,
            prim_type = "Xform",
            usd_path = full_path,
        )
        robot_p = Robot(
            prim_path=robot_path,
            name=robot_name,
        )
        stage = robot_prim.GetStage()
        linkp = stage.GetPrimAtPath(robot_path)
        set_prim_transform(linkp, [position[0], position[1], position[2], 1, 0, 0, 0])

    robot = my_world.scene.add(robot_p)
    return robot, robot_path


class VoxelManager:
    def __init__(
        self,
        num_voxels: int = 5000,
        size: float = 0.02,
        color: List[float] = [1, 1, 1],
        prefix_path: str = "/World/curobo/voxel_",
        material_path: str = "/World/looks/v_",
    ) -> None:
        self.cuboid_list = []
        self.cuboid_material_list = []
        self.disable_idx = num_voxels
        for i in range(num_voxels):
            target_material = OmniPBR("/World/looks/v_" + str(i), color=np.ravel(color))

            cube = cuboid.VisualCuboid(
                prefix_path + str(i),
                position=np.array([0, 0, -10]),
                orientation=np.array([1, 0, 0, 0]),
                size=size,
                visual_material=target_material,
            )
            self.cuboid_list.append(cube)
            self.cuboid_material_list.append(target_material)
            cube.set_visibility(True)

    def update_voxels(self, voxel_position: np.ndarray, color_axis: int = 0):
        max_index = min(voxel_position.shape[0], len(self.cuboid_list))

        jet = cm.get_cmap("hot")  # .reversed()
        z_val = voxel_position[:, 0]

        jet_colors = jet(z_val)

        for i in range(max_index):
            self.cuboid_list[i].set_visibility(True)

            self.cuboid_list[i].set_local_pose(translation=voxel_position[i])
            self.cuboid_material_list[i].set_color(jet_colors[i][:3])

        for i in range(max_index, len(self.cuboid_list)):
            self.cuboid_list[i].set_local_pose(translation=np.ravel([0, 0, -10.0]))

            # self.cuboid_list[i].set_visibility(False)

    def clear(self):
        for i in range(len(self.cuboid_list)):
            self.cuboid_list[i].set_local_pose(translation=np.ravel([0, 0, -10.0]))

