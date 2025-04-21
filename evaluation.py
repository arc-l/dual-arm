import isaacsim
from omni.isaac.kit import SimulationApp
from helper import save_plan, load_plan, HEADLESS
import helper
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

from dual_arms import DualArmEnv
from baseline_dual_arm import Baseline
import argparse, json, os, pickle
from curobo.types.state import JointState
from task import Task

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse command-line arguments for method and input file.")
    
    parser.add_argument(
        "--method",
        choices=["baseline", "ours"],
        required=True,
        help="Specify the method to use: 'baseline' or 'ours'."
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input JSON file."
    )

    args = parser.parse_args()

    # Validate JSON file existence
    if not os.path.isfile(args.input):
        parser.error(f"The input file '{args.input}' does not exist.")
    
    return args

def get_duration(plan):
    duration = 0
    for js in plan:
        if type(js) == JointState:
            duration += js.position.shape[0] * 0.02
        else:
            duration += 0.02
    return duration

if __name__ == "__main__":
    args = parse_arguments()
    
    
    task = Task.read_from_json(args.input)
    
    if args.method == "baseline":
        baseline = Baseline(task, simulation_app)
        plan = baseline.plan(autostart=False, use_our_planner=True)
        baseline.play(plan)
    else:
        # Call our method
        dual_arm_env = DualArmEnv(task, simulation_app, batch_size=3, constrain_grasp_approach=False, visualize_spheres=False)
        plan = dual_arm_env.plan(autostart=False)

    duration = get_duration(plan)
    print("Plan duration:", duration)
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_filename = os.path.join("results", f"{base_name}_{args.method}_duration.txt")
    with open(output_filename, "w") as file:
        file.write(f"{duration}\n")
    # Save the plan
    save_plan(plan, args.input, args.method)
