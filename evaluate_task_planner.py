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
        choices=["baseline", "ours", "baseline_with_our_tp", "ours_with_baseline_tp",  "our_tp_with_curobo"],
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
    baseline = Baseline(task, headless=False, display=False)
    if args.method == "baseline":
        action_list = baseline.baseline_task_plan()
    else:
        # Call our method
        action_list = baseline.plan_all()

    if action_list is None:
        print("No plan found.")
        exit(1)

    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_filename = os.path.join("results", f"{base_name}_{args.method}_num_actions.txt")
    with open(output_filename, "w") as file:
        file.write(str(len(action_list[0])+len(action_list[1])))
    print(f"Number of actions saved to {output_filename}")
    action_filename = os.path.join("results", f"{base_name}_{args.method}_actions.json")
    with open(action_filename, "w") as file:
        json.dump(action_list, file, indent=4)
    print(f"Plan saved to {action_filename}")
