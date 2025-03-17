import sys
sys.path.append('.')
from .central_control import central_control
import os


from constants import (
    ARM_RADIUS_LIMIT,
    IMAGE_SIZE,
    NUM_ROTATION,
    PIXEL_SIZE,
    WORKSPACE_LIMITS,
    COLOR_SPACE,
    COLOR_MAP,
    blue_lower,
    blue_upper,
    green_lower,
    green_upper,
    red_lower,
    red_upper,
)

from environment_sim import Environment

def simulation(start_arr:dict, goal_arr:dict, radius:float, dual_action_list:dict, exp_text=''):
    update_cylinder_urdf(radius)

    # Setup environment
    env = Environment(gui=True)
    env.reset()
    env.setup_rearrangement_task_from_dict(start_arr, radius)


    # input("start")
    control = central_control(env, dual_action_list, 0.005, start_arr, goal_arr, exp_text=exp_text)
    return control.total_count, control.execution_time
    # input("finished")


def update_cylinder_urdf(radius):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    urdf_file = os.path.join(current_folder, 'assets', 'blocks', 'cylinder.urdf')
    scale = (radius/1000.0)/(0.044/1.0)
    a_file = open(urdf_file, "r")
    list_of_lines = a_file.readlines()
    list_of_lines[16] = "         <mesh filename=\"cylinder.obj\" scale=\"" +str(scale) + " " + str(scale) + " 1\"/>\n"
    list_of_lines[22] = "         <mesh filename=\"cylinder.obj\" scale=\"" +str(scale) + " " + str(scale) + " 1\"/>\n"
    a_file = open(urdf_file, "w")
    a_file.writelines(list_of_lines)
    a_file.close()
