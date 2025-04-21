import isaacsim
from omni.isaac.kit import SimulationApp
if __name__ == '__main__':
    import helper
    helper.HEADLESS = False 
    simulation_app = SimulationApp({
                    "headless": False,
                    "width": "1920",
                    "height": "1080",
    })

from dual_arms import DualArmEnv
from baseline_dual_arm import Baseline
import argparse, json, os, pickle
import torch
import curobo
from curobo.types.state import JointState
from helper import save_plan, load_plan
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
        planner = Baseline(task, simulation_app)
    else:
        # Call our method
        planner = DualArmEnv(task, simulation_app, batch_size=3, constrain_grasp_approach=False, visualize_spheres=False)
    
    # change the input json file's extension to .pkl
    # and change it to results folder from data folder
    plan_path = args.input.replace('.json', f'_{args.method}_plan.pkl').replace('data', 'results')

    plan = load_plan(plan_path, device='cuda:0')
    planner.play(plan)

