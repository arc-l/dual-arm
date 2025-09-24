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


def compute_max_joint_speed(plan, time_step=0.1):
    """
    Compute the maximum joint speed across the plan.

    Args:
        plan: List of JointState objects with `.position` as torch.Tensor.
        time_step: Time between consecutive JointStates (in seconds).

    Returns:
        max_speed: Maximum joint speed (float).
        joint_index: Index of the joint with the max speed.
        step_index: Index of the time step at which it occurred.
    """
    max_speed = 0.0
    joint_index = -1
    time_index = -1
    state_index = -1

    for s_idx, state in enumerate(plan):
        if not isinstance(state, JointState):
            continue
        pos = state.position  # Should be shape [T, D]
        if not isinstance(pos, torch.Tensor) or pos.ndim != 2:
            continue  # Skip malformed entries

        vel = torch.diff(pos, dim=0) / time_step  # shape [T-1, D]
        abs_vel = torch.abs(vel)
        local_max, flat_idx = torch.max(abs_vel.view(-1), dim=0)
        if local_max.item() > max_speed:
            max_speed = local_max.item()
            t_idx, j_idx = divmod(flat_idx.item(), abs_vel.shape[1])
            time_index = t_idx
            joint_index = j_idx
            state_index = s_idx

    return max_speed, state_index, time_index, joint_index