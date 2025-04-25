import pickle
import os
from curobo.types.state import JointState
import torch

HEADLESS = False 

def joint_state_to_cpu(joint_state):
    """Convert all tensor attributes of a JointState to CPU."""
    position=joint_state.position.cpu() if isinstance(joint_state.position, torch.Tensor) else joint_state.position
    joint_names=joint_state.joint_names
    if joint_names is None:
        print("joint_names is None")
    cpu_joint_state = JointState.from_position(position, joint_names)
    return cpu_joint_state

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
        joint_names=joint_state.joint_names
        if joint_names is None:
            print("joint_names is None")
        return JointState.from_position(position, joint_names)

    device_plan = [joint_state_to_device(js, device) if type(js)==JointState else js for js in cpu_plan]
    return device_plan

