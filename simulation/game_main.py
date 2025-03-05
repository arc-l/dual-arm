import time
import numpy as np
from central_control import central_control
import utils
import os

import cv2
import imutils


import sys
import math
import matplotlib.pyplot as plt

from mcts_utils import MCTSHelper
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



from solvers.A_star_solver import A_star_solver
from solvers.Single_Arm_A_star_solver import Single_Arm_A_star_solver
from solvers.high_level_solver import high_level_search
from tools.animation import plan_animation
from tools.show_arrangements import show_arrangements
from tools.generate_arrangements import load_arrangement



size = 210
num_cell = 3
cell_size = round(size / num_cell)
offset = round((IMAGE_SIZE - size) / 2)
color_names = ["red", "green", "blue"]
color_lowers = {"red": red_lower, "green": green_lower, "blue": blue_lower}
color_uppers = {"red": red_upper, "green": green_upper, "blue": blue_upper}




def main():

    # Setup environment
    env = Environment(gui=True)
    env.reset()
    env.setup_rearrangement_task("test-cases/test1.txt")
    mcts_helper = MCTSHelper(env, "logs_grasp/snapshot-post-020000.reinforcement.pth")

    while True:
        # Get images
        color_image, depth_image, mask_image = get_images(env)

        # Process images
        processed_input = process_images(color_image, mask_image)
        # TODO: Yuhan, this is the state information for you
        print(processed_input)

        # Process output
        # TODO: Yuhan, these four variables should be from you
        # Arm 0 to move
        arm_idx = 0
        start_cell = [0, 0]
        end_cell = [2, 1]
        color_name = "green"
        (
            arm_idx,
            primitive_position,
            best_rotation_angle,
            place_primitive_position,
            place_best_rotation_angle,
        ) = process_output(
            arm_idx,
            start_cell,
            end_cell,
            color_name,
            color_image,
            depth_image,
            mask_image,
            processed_input,
            mcts_helper,
        )
        env.pick_and_place(
            arm_idx,
            primitive_position,
            best_rotation_angle,
            place_primitive_position,
            place_best_rotation_angle,
        )

        # Arm 1 to move
        arm_idx = 1
        start_cell = [0, 1]
        end_cell = [2, 0]
        color_name = "red"
        color_image, depth_image, mask_image = get_images(env)
        processed_input = process_images(color_image, mask_image)
        (
            arm_idx,
            primitive_position,
            best_rotation_angle,
            place_primitive_position,
            place_best_rotation_angle,
        ) = process_output(
            arm_idx,
            start_cell,
            end_cell,
            color_name,
            color_image,
            depth_image,
            mask_image,
            processed_input,
            mcts_helper,
        )
        env.pick_and_place(
            arm_idx,
            primitive_position,
            best_rotation_angle,
            place_primitive_position,
            place_best_rotation_angle,
        )


def test():
    # do planning
    start_arr, goal_arr, radius, action_list = planning()
    
    update_cylinder_urdf(radius)

    # Setup environment
    env = Environment(gui=True)
    env.reset()
    env.setup_rearrangement_task_from_dict(start_arr, radius)


    input("start")
    control = central_control(env, action_list, 0.005, start_arr, goal_arr)

    input("finished")


def update_cylinder_urdf(radius):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    urdf_file = os.path.join(current_folder, 'assets', 'blocks', 'cylinder.urdf')
    scale = (radius/1000.0)/(0.044/1.0)
    a_file = open(urdf_file, "r")
    list_of_lines = a_file.readlines()
    list_of_lines[16] = "         <mesh filename=\"cylinder.obj\" scale=\"" +str(scale) + " " + str(scale) + " 1\"/>\n"

    a_file = open(urdf_file, "w")
    a_file.writelines(list_of_lines)
    a_file.close()



def planning():
    # scale of each region
    Width = 1000
    Height = 1000
    numObjs = 5
    Density = 0.2
    radius = math.sqrt(Density*Width*Height/numObjs/math.pi)
    start_arr = load_arrangement(numObjs, Density, 15)
    goal_arr = load_arrangement(numObjs, Density, 3)
    print(start_arr)
    print(goal_arr)
    planner = A_star_solver(Width, Height, numObjs, radius, start_arr, goal_arr)
    # planner = Single_Arm_A_star_solver(Width, Height, numObjs, radius, start_arr, goal_arr)
    planner.A_star()
    action_list = planner.action_list
    # action_list = high_level_search(A_star_solver, Width, Height, numObjs, radius, start_arr, goal_arr)
    print(action_list)
    makespan = max([len(e) for e in action_list.values()])
    print(makespan)
    # show_arrangements(Height, Width, start_arr, goal_arr, radius)
    # plan_animation(Height, Width, start_arr, goal_arr, radius, action_list)
    return start_arr, goal_arr, radius, action_list


if __name__ == "__main__":
    # main()
    test()
    # planning()
