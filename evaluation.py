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
        choices=["baseline", "ours", "baseline_with_our_tp", "ours_with_baseline_tp",  "our_tp_with_curobo", "baseline_tp_with_curobo"],
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
        baseline = Baseline(task, headless=False, display=False)
        plan = baseline.plan(use_our_tp=False, use_our_mp=False)
        # baseline.play(plan)
    elif args.method == "baseline_with_our_tp":
        baseline = Baseline(task, headless=False, display=False)
        plan = baseline.plan(use_our_tp=True, use_our_mp=False)
    elif args.method == "ours_with_baseline_tp":
        print("Using baseline tp + our mp")
        baseline = Baseline(task, headless=True, display=True)
        plan = baseline.plan(use_our_tp=False, use_our_mp=True)
    elif args.method == "baseline_tp_with_curobo":
        print("Using baseline tp + our mp")
        baseline = Baseline(task, headless=True, display=True)
        baseline.pure_curobo = True
        plan = baseline.plan(use_our_tp=False, use_our_mp=True)
    elif args.method == "our_tp_with_curobo":
        dual_arm_env = DualArmEnv(task, headless=False, batch_size=3, num_buffers=20, pure_curobo=True)
        plan = dual_arm_env.plan()
    else:
        # Call our method
        dual_arm_env = DualArmEnv(task, headless=False, batch_size=3, num_buffers=20, pure_curobo=False)
        plan = dual_arm_env.plan()

    if plan is None:
        print("No plan found.")
        exit(1)

    duration = get_duration(plan)
    print("Plan duration:", duration)
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_filename = os.path.join("results", f"{base_name}_{args.method}_duration.txt")
    with open(output_filename, "w") as file:
        file.write(f"{duration}\n")
    # Save the plan
    save_plan(plan, args.input, args.method)
