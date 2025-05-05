from helper import save_plan, load_plan 
import helper

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
        baseline = Baseline(task, headless=True)
        plan = baseline.plan(use_our_planner=False)
        # baseline.play(plan)
    else:
        # Call our method
        dual_arm_env = DualArmEnv(task, headless=False, batch_size=3, constrain_grasp_approach=False, num_buffers=20)
        plan = dual_arm_env.plan()

    duration = get_duration(plan)
    print("Plan duration:", duration)
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_filename = os.path.join("results", f"{base_name}_{args.method}_duration.txt")
    with open(output_filename, "w") as file:
        file.write(f"{duration}\n")
    # Save the plan
    save_plan(plan, args.input, args.method)
