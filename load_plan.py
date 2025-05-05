import isaacsim
from dual_arms import DualArmEnv
from baseline_dual_arm import Baseline
import argparse, os
from curobo.types.state import JointState
from helper import load_plan, compute_max_joint_speed
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
    plan_path = args.input.replace('.json', f'_{args.method}_plan.pkl').replace('data', 'results')

    plan = load_plan(plan_path, device='cuda:0')
    max_speed, state_index, time_index, joint_index = compute_max_joint_speed(plan, time_step=0.02)
    print(f"Max joint speed: {max_speed}")
    print(f"State index: {state_index}")
    print(f"Time index: {time_index}")
    print(f"Joint index: {joint_index}")
    exit(0)
    
    
    task = Task.read_from_json(args.input)
    
    if args.method == "baseline":
        planner = Baseline(task, headless=False, display=True)
    else:
        # Call our method
        planner = DualArmEnv(task, headless=False, batch_size=3, constrain_grasp_approach=False)
    
    # change the input json file's extension to .pkl
    # and change it to results folder from data folder
    
    # input("Press Enter to continue...")
    planner.play(plan)

