import isaacsim
from dual_arms_test import DualArmEnv
from baseline_dual_arm import Baseline
import argparse, os
from curobo.types.state import JointState
from helper import load_plan
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
        planner = Baseline(task, None)
    else:
        # Call our method
        planner = DualArmEnv(task, headless=False, batch_size=3, constrain_grasp_approach=False)
    
    # change the input json file's extension to .pkl
    # and change it to results folder from data folder
    plan_path = args.input.replace('.json', f'_{args.method}_plan.pkl').replace('data', 'results')

    plan = load_plan(plan_path, device='cuda:0')
    planner.play(plan)

