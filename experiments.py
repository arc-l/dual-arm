import math
import timeout_decorator
import time
import datetime
import os
import sys
import random
import argparse
from colorama import init
from numpy import random

from solvers.high_level_solver import high_level_search
from tools.animation import plan_animation
from tools.show_arrangements import show_arrangements
from tools.generate_arrangements import load_arrangement
from high_level_planners.Bi_Directional_Search_Tree_Planner import Bi_Directional_Search_Tree_Planner
# from simulation.simulation import simulation
from motion_planning.simulation import simulation
from motion_planning.robot_arm import ControlMode
from motion_planning.plan_recorder import Plan_Recorder
from tools.util import write_to_log,write_title,write_a_plan, write_a_traj, read_a_traj, get_real_arrangement, load_real_arrangement,real_env_to_plan_env
from constants import WORKSPACE_SIZE



init(autoreset=True)

TimeLimit = 1000.0

def run_simulation(args):
    control_mode = ControlMode(args.control_mode)
    print(f"control_mode: {control_mode}")

    texts = [
        # '0.1 15 05 1.0 50 51'
        '0.1 4 05 1.0 00 01'
        # '0.1 6 05 1.0 00 01'
        # '0.1 8 05 1.0 00 01'
        # '0.2 5 05 1.0 02 19',  # compact example
        # '0.2 05 00 0.5 02 19', # handoff example
        # '0.2 05 00 0.75 01 04', 
        # '0.2 10 3 0.75 3 15'
    ] # experiment code (density, numObjs, trial, overlap_rate, ID1, ID2)
    results = []
    for text in texts:
        words = text.split()
        Density, numObjs, _, overlap_rate, ID1, ID2 = [float(e) for e in words]
        numObjs, ID1, ID2 = int(numObjs), int(ID1), int(ID2)
        start_arr = load_arrangement(numObjs, Density, overlap_rate, ID1)
        goal_arr = load_arrangement(numObjs, Density, overlap_rate, ID2)
        Height = 1000
        Width = 2*Height/(1+overlap_rate)
        radius = math.sqrt(Density * Width * Height / numObjs / math.pi)
        radius = min(0.036*1000/WORKSPACE_SIZE,radius) # the opened hand is 8.5cm, and give some margin
        time_step_size = 0.02 # 1/240.0
        speed = 0.3141592653589793
        # method = 'baseline_v2'
        method = "baseline"
        # method = 'bato'
        # method = 'cuto_v2'

        # overlap_rate = 0.3
        print("start_arr", start_arr)
        print("goal_arr", goal_arr)
        start = time.time()
        planner = Bi_Directional_Search_Tree_Planner(
            start_arr, 
            goal_arr, 
            Height, 
            Width, 
            radius, 
            sampling=True, 
            overlap_rate=overlap_rate, 
            num_arms=2,
            makespan='MC'
        )
        action_list = planner.action_list
        action_list = plan2schedule(action_list)
        print(action_list)
        start = time.time()
        try:
            result_dict, control = timeout_simulation(
                Width, Height,start_arr, goal_arr, radius, action_list, 
                overlap_rate = overlap_rate, speed=speed, 
                time_step_size=time_step_size, control_mode=control_mode, 
                gui=args.gui, method=method, debug = True)
            return control
        except timeout_decorator.timeout_decorator.TimeoutError:
            result_dict = {}
        for k,v in result_dict.items():
            if k == 'raw_plan':
                continue
            print(k, v)
    return None
            
            


@timeout_decorator.timeout(TimeLimit)
def timeout_simulation(Width, Height, start_arr, goal_arr, radius, action_list, overlap_rate=1.0, speed=0.002,instance_info=[],label_txt='test', **kwargs):
    return simulation(Width, Height, start_arr, goal_arr, radius, action_list, overlap_rate=overlap_rate, speed=speed, instance_info=instance_info,label_txt=label_txt, **kwargs)
    

def plan2schedule(plan):
    '''
    Input: [(objID,start_pose, goal_pose)]
    Output: [(obj, start_mode, goal_mode, pose1, pose2, start_time, end_time)]
    '''
    schedule = {}
    
    for arm in range(2):
        schedule[arm] = []
        for id_, task in enumerate(plan[arm]):
            if  task == None:
                continue
            objID, start_pose, goal_pose = task
            if start_pose == "H":
                start_mode = 'H'
                start_pose = None
            else:
                start_mode = 'P'
            if goal_pose == 'H':
                goal_mode = 'H'
                goal_pose = None
            else:
                goal_mode = 'P'
            schedule[arm].append(
                (objID, start_mode,goal_mode, start_pose, goal_pose, id_, id_)
            )
    return schedule


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--control_mode", type=int, default=0, help="Control Mode for motors. 0: Position, 1: Velocity, 2: Torque")
    argparser.add_argument("--gui", action='store_true', help="Show GUI")
    args = argparser.parse_args()
    run_simulation(args)
    