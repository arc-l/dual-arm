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
from real_time_control.real_camera.camera import Camera
from tools.util import write_to_log,write_title,write_a_plan, write_a_traj, read_a_traj, get_real_arrangement, load_real_arrangement,real_env_to_plan_env
from constants import WORKSPACE_SIZE



init(autoreset=True)

TimeLimit = 1000.0
# TimeLimit = 10

def run_simulation_from_real(args, start_arr:dict=None, goal_arr:dict=None, radius:int=42):
    control_mode = ControlMode(args.control_mode)
    print(f"control_mode: {control_mode}")

    texts = [
        '0.2 5 05 1.0 02 19',  # compact example
        # '0.2 05 00 0.5 02 19', # handoff example
        # '0.2 05 00 0.75 01 04', 
        # '0.2 10 3 0.75 3 15'
    ] # experiment code (density, numObjs, trial, overlap_rate, ID1, ID2)
    results = []
    for text in texts:
        words = text.split()
        Density, numObjs, _, overlap_rate, ID1, ID2 = [float(e) for e in words]
        numObjs, ID1, ID2 = int(numObjs), int(ID1), int(ID2)
        # case 1, fully overlapping
        # start_arr = {0: [892.7968767601278, 729.8344376754679], 7: [667.0224216014623, 546.5473462582706]}
        # goal_arr = {0: [749.4758312162808, 676.6277667539638], 7: [174.9507293537881, 683.3685411907734]}
        # # case 2, handoff
        # start_arr = {0: [135.50576330107648, 864.4942366989235]}
        # goal_arr = {0: [1101.0228122820904, 611.1705909840796]}
        start_arr = start_arr
        goal_arr = goal_arr
        # start_arr = load_arrangement(numObjs, Density, overlap_rate, ID1)
        # goal_arr = load_arrangement(numObjs, Density, overlap_rate, ID2)
        Height = 1000
        Width = 2*Height/(1+overlap_rate)
        # radius = math.sqrt(Density * Width * Height / numObjs / math.pi)
        # radius = min(0.036*1000/WORKSPACE_SIZE,radius) # the opened hand is 8.5cm, and give some margin
        radius = radius
        time_step_size = 0.002 # 1/240.0
        speed = 0.3141592653589793
        method = 'baseline'
        # method = 'cuto'

        # overlap_rate = 0.3
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
        # case 1, fully overlapping
        # action_list = {
        #     0: [(0, 'P', 'P', [892.7968767601278, 729.8344376754679], [749.4758312162808, 676.6277667539638], 0, 0)], 
        #     # 1: [(7, 'P', 'P', [667.0224216014623, 546.5473462582706], [174.9507293537881, 683.3685411907734], 0, 0)]
        #     1: []
        #     }
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
            print(k, v)
    return None


def run_simulation(args):
    control_mode = ControlMode(args.control_mode)
    print(f"control_mode: {control_mode}")

    texts = [
        # '0.1 15 05 1.0 50 51'
        # '0.1 4 05 1.0 00 01'
        # '0.1 6 05 1.0 00 01'
        '0.1 8 05 1.0 00 01'
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
        # case 1, fully overlapping
        # start_arr = {0: [892.7968767601278, 729.8344376754679], 7: [667.0224216014623, 546.5473462582706]}
        # goal_arr = {0: [749.4758312162808, 676.6277667539638], 7: [174.9507293537881, 683.3685411907734]}
        # # case 2, handoff
        # start_arr = {0: [135.50576330107648, 864.4942366989235]}
        # goal_arr = {0: [1101.0228122820904, 611.1705909840796]}
        start_arr = load_arrangement(numObjs, Density, overlap_rate, ID1)
        goal_arr = load_arrangement(numObjs, Density, overlap_rate, ID2)
        print(start_arr, goal_arr)
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
        # case 1, fully overlapping
        # action_list = {
        #     0: [(0, 'P', 'P', [892.7968767601278, 729.8344376754679], [749.4758312162808, 676.6277667539638], 0, 0)], 
        #     # 1: [(7, 'P', 'P', [667.0224216014623, 546.5473462582706], [174.9507293537881, 683.3685411907734], 0, 0)]
        #     1: []
        #     }
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
    



def motion_planning_experiments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--control_mode", type=int, default=0, help="Control Mode for motors. 0: Position, 1: Velocity, 2: Torque")
    argparser.add_argument("--gui", action='store_true', help="Show GUI")
    args = argparser.parse_args()
    control_mode = ControlMode(args.control_mode)
    print(f"control_mode: {control_mode}")
    if control_mode != ControlMode.POSITION:
        raise NotImplementedError("Only position control is supported for now.")

    '''simulatipn parameters'''
    time_step_size = 0.02
    speed = 0.3141592653589793

    numObjs_list = [6,9,12,15,18]
    overlap_rate_list = [0.5, 0.75, 1.0]
    Density_list = [0.2]
    method_list = [
        'baseline_v2',
        'cuto_v2',
        'bato_v2',
        # 'single-cuto',
        # 'single-bl',
        ]
    
    num_trials = 20

    # generate an unique experiment ID
    timestamp = time.time()
    timestamp_value = datetime.datetime.fromtimestamp(timestamp)
    experimentID = timestamp_value.strftime("%Y-%m-%d-%H-%M-%S")

    # arrangement_bank = restore_arrangementIDs()
    
    for method in method_list:
        write_title(
            experimentID,
            'D, n, trial, overlap, IDs[0], IDs[1], comp_time, motion_plan_time, exe_time, arm0_len, arm1_len',
            method
        )
    for trial in range(4,num_trials):
        for numObjs in numObjs_list:
            for overlap_rate in overlap_rate_list:
                for Density in Density_list:
                    ID1, ID2 = random.choice(range(20), size=2, replace=False)
                    # ID1, ID2 = arrangement_bank[Density][numObjs][overlap_rate][trial]
                    start_arr = load_arrangement(numObjs, Density, overlap_rate, ID1)
                    goal_arr = load_arrangement(numObjs, Density, overlap_rate, ID2)
                    Height = 1000
                    Width = 2*Height/(1+overlap_rate)
                    radius = math.sqrt(Density * Width * Height / numObjs / math.pi)
                    radius = min(0.036*1000/WORKSPACE_SIZE,radius) # the opened hand is 8.5cm, and give some margin
                    instance_info = [ experimentID, Density, numObjs, trial, overlap_rate, ID1, ID2]
                    for method in method_list:
                        if method in ['single-cuto', 'single-bl']:
                            num_arms = 1
                        else:
                            num_arms = 2 
                        plan_start = time.time()
                        planner = Bi_Directional_Search_Tree_Planner(
                            start_arr, 
                            goal_arr, 
                            Height, 
                            Width, 
                            radius, 
                            sampling=True, 
                            overlap_rate=overlap_rate, 
                            num_arms=num_arms,
                            makespan='MC'
                        )
                        task_plan_time = time.time() - plan_start
                        action_list = planner.action_list
                        action_list = plan2schedule(action_list)
                        try:
                            if method == 'bato_v2':
                                raw_plan = read_a_traj(instance_info, 'baseline_v2')
                            else:
                                raw_plan = None
                            result_dict = timeout_simulation(
                                Width, Height,start_arr, goal_arr, radius, action_list, 
                                overlap_rate = overlap_rate, speed=speed, 
                                time_step_size=time_step_size, control_mode=control_mode, 
                                gui=args.gui, method=method, raw_plan=raw_plan)
                        # except timeout_decorator.timeout_decorator.TimeoutError:
                        except Exception as e:
                            # print(e)
                            result_dict = {
                                'motion planning time': -1.0, 
                                'execution time': -1.0, 
                                'plan length': [-1.0, -1.0], 
                                # 'control': None
                                'raw_plan': None,
                                }
                        except KeyboardInterrupt:
                            result_dict = {
                                'motion planning time': -1.0, 
                                'execution time': -1.0, 
                                'plan length': [-1.0, -1.0], 
                                # 'control': None
                                'raw_plan': None,
                                } 
                            input("Press Enter to continue...")
                        motion_plan_time = result_dict['motion planning time']
                        execution_time = result_dict['execution time']
                        arm0_len, arm1_len = result_dict['plan length']
                        traj = result_dict['raw_plan']
                        result_data = [task_plan_time, motion_plan_time, execution_time, arm0_len, arm1_len]
                        recording( instance_info, result_data, action_list, traj, method)
    

def batch_recovery():
    '''
    add experiment results for a specific trial
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--control_mode", type=int, default=0, help="Control Mode for motors. 0: Position, 1: Velocity, 2: Torque")
    argparser.add_argument("--gui", action='store_true', help="Show GUI")
    args = argparser.parse_args()
    control_mode = ControlMode(args.control_mode)
    print(f"control_mode: {control_mode}")
    if control_mode != ControlMode.POSITION:
        raise NotImplementedError("Only position control is supported for now.")

    '''simulatipn parameters'''
    time_step_size = 0.002 # 1/240.0
    speed = 0.3141592653589793

    task_tuples = [
        (0.2, 10, 19, 0.5),
    ] # experiment code (density, numObjs, trial, overlap_rate)

    method_list = ['baseline','cuto']
    # generate an unique experiment ID
    timestamp = time.time()
    timestamp_value = datetime.datetime.fromtimestamp(timestamp)
    experimentID = timestamp_value.strftime("%Y-%m-%d-%H-%M-%S")
    
    for method in method_list:
        write_title(
            experimentID,
            'D, n, trial, overlap, IDs[0], IDs[1], comp_time, motion_plan_time, exe_time, arm0_len, arm1_len',
            method
        )

    for task in task_tuples:
        Density, numObjs, trial, overlap_rate = task
        ID1, ID2 = random.choice(range(20), size=2, replace=False)
        start_arr = load_arrangement(numObjs, Density, overlap_rate, ID1)
        goal_arr = load_arrangement(numObjs, Density, overlap_rate, ID2)
        Height = 1000
        Width = 2*Height/(1+overlap_rate)
        radius = math.sqrt(Density * Width * Height / numObjs / math.pi)
        radius = min(0.036*1000/WORKSPACE_SIZE,radius) # the opened hand is 8.5cm, and give some margin
        instance_info = [ experimentID, Density, numObjs, trial, overlap_rate, ID1, ID2]
        for method in method_list:
            plan_start = time.time()
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
            task_plan_time = time.time() - plan_start
            action_list = planner.action_list
            action_list = plan2schedule(action_list)
            try:
                result_dict = timeout_simulation(
                    Width, Height,start_arr, goal_arr, radius, action_list, 
                    overlap_rate = overlap_rate, speed=speed, 
                    time_step_size=time_step_size, control_mode=control_mode, 
                    gui=args.gui, method=method)
            # except timeout_decorator.timeout_decorator.TimeoutError:
            except Exception as e:
                print(e)
            result_dict = {
                'motion planning time': -1.0, 
                'execution time': -1.0, 
                'plan length': [-1.0, -1.0], 
                # 'control': None,
                'raw_plan': None,
                }
            motion_plan_time = result_dict['motion planning time']
            execution_time = result_dict['execution time']
            arm0_len, arm1_len = result_dict['plan length']
            traj = result_dict['raw_plan']
            result_data = [task_plan_time, motion_plan_time, execution_time, arm0_len, arm1_len]
            recording( instance_info, result_data, action_list, traj, method)


def batch_rerun():
    '''
    recover instances for specific methods
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--control_mode", type=int, default=0, help="Control Mode for motors. 0: Position, 1: Velocity, 2: Torque")
    argparser.add_argument("--gui", action='store_true', help="Show GUI")
    args = argparser.parse_args()
    control_mode = ControlMode(args.control_mode)
    print(f"control_mode: {control_mode}")
    if control_mode != ControlMode.POSITION:
        raise NotImplementedError("Only position control is supported for now.")

    '''simulatipn parameters'''
    time_step_size = 0.02 # 1/240.0
    speed = 0.3141592653589793

    task_tuples = [
        (0.2, 15, 0, 1.0, 2, 13, 'cuto_v2'),
        (0.2, 18, 0, 0.5, 6, 10, 'cuto_v2'),
        (0.2, 18, 0, 1.0, 2, 9, 'cuto_v2'),
        (0.2, 6, 1, 0.75, 10, 13, 'cuto_v2'),
        (0.2, 15, 1, 1.0, 14, 5, 'cuto_v2'),
        (0.2, 18, 1, 1.0, 5, 11, 'cuto_v2'),
        (0.2, 12, 2, 0.75, 10, 13, 'cuto_v2'),
        (0.2, 12, 2, 1.0, 11, 18, 'cuto_v2'),
        (0.2, 15, 2, 0.5, 3, 7, 'cuto_v2'),
        (0.2, 15, 2, 1.0, 9, 15, 'cuto_v2'),
        (0.2, 18, 2, 0.5, 17, 14, 'cuto_v2'),
        (0.2, 18, 2, 0.75, 9, 15, 'cuto_v2'),
        (0.2, 18, 2, 1.0, 15, 4, 'cuto_v2'),
    ] # experiment code (density, numObjs, trial, overlap_rate, ID1, ID2, method_list)

    method_list = [
        # 'baseline_v2', 'bato_v2', 
        'cuto_v2'
                   ]
    # generate an unique experiment ID
    timestamp = time.time()
    timestamp_value = datetime.datetime.fromtimestamp(timestamp)
    experimentID = timestamp_value.strftime("%Y-%m-%d-%H-%M-%S")
    
    for method in method_list:
        write_title(
            experimentID,
            'D, n, trial, overlap, IDs[0], IDs[1], comp_time, motion_plan_time, exe_time, arm0_len, arm1_len',
            method
        )

    for task in task_tuples:
        Density, numObjs, trial, overlap_rate, ID1, ID2, method = task
        # ID1, ID2 = random.choice(range(20), size=2, replace=False)
        start_arr = load_arrangement(numObjs, Density, overlap_rate, ID1)
        goal_arr = load_arrangement(numObjs, Density, overlap_rate, ID2)
        Height = 1000
        Width = 2*Height/(1+overlap_rate)
        radius = math.sqrt(Density * Width * Height / numObjs / math.pi)
        radius = min(0.036*1000/WORKSPACE_SIZE,radius) # the opened hand is 8.5cm, and give some margin
        instance_info = [ experimentID, Density, numObjs, trial, overlap_rate, ID1, ID2]
        # for method in method_list:
        plan_start = time.time()
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
        task_plan_time = time.time() - plan_start
        action_list = planner.action_list
        action_list = plan2schedule(action_list)
        try:
            if method == 'bato_v2':
                raw_plan = read_a_traj(instance_info, 'baseline_v2')
            else:
                raw_plan = None
            result_dict = timeout_simulation(
                Width, Height,start_arr, goal_arr, radius, action_list, 
                overlap_rate = overlap_rate, speed=speed, 
                time_step_size=time_step_size, control_mode=control_mode, 
                gui=args.gui, method=method, raw_plan=raw_plan)
        # except timeout_decorator.timeout_decorator.TimeoutError:
        except Exception as e:
            print(e)
            result_dict = {
                'motion planning time': -1.0, 
                'execution time': -1.0, 
                'plan length': [-1.0, -1.0], 
                # 'control': None,
                'raw_plan': None,
                }
        except KeyboardInterrupt:
            result_dict = {
                'motion planning time': -1.0, 
                'execution time': -1.0, 
                'plan length': [-1.0, -1.0], 
                # 'control': None
                'raw_plan': None,
                } 
            input("Press Enter to continue...")
        motion_plan_time = result_dict['motion planning time']
        execution_time = result_dict['execution time']
        arm0_len, arm1_len = result_dict['plan length']
        traj = result_dict['raw_plan']
        result_data = [task_plan_time, motion_plan_time, execution_time, arm0_len, arm1_len]
        recording( instance_info, result_data, action_list, traj, method)



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


def restore_arrangementIDs():
    arrangementID_bank = {}
    File = os.path.join(os.path.abspath(os.path.dirname(__file__)), "logs", "cuto", "collections.txt")

    with open(File, "r") as f:
        for line in list(f.readlines())[1:]:
            words = line.split()
            Density = float(words[0])
            numObjs = int(float(words[1]))
            trial = int(float(words[2]))
            overlap_rate = float(words[3])
            # Time = float(words[3])
            # ratio = float(words[4])

            ID1 = int(float(words[4]))
            ID2 = int(float(words[5]))

            if Density not in arrangementID_bank:
                arrangementID_bank[Density] = {}
            if numObjs not in arrangementID_bank[Density]:
                arrangementID_bank[Density][numObjs] = {}
            if overlap_rate not in arrangementID_bank[Density][numObjs]:
                arrangementID_bank[Density][numObjs][overlap_rate] = {}
            if trial not in arrangementID_bank[Density][numObjs][overlap_rate]:
                arrangementID_bank[Density][numObjs][overlap_rate][trial] = [ID1, ID2]

    # print arrangementID_bank
    return arrangementID_bank


def recording(instance_info, result_data, plan, traj, text):
    write_a_plan(instance_info, plan, text)
    write_to_log(instance_info, result_data, text)
    write_a_traj(instance_info, traj, text)
    print("results: ", text, result_data)


def count_handoffs(dual_action_list):
    ''' count handoff number in the computed plan '''
    handoff_num = 0
    for action in dual_action_list[0]:
        start = action[1]
        goal = action[2]
        for mode in [start,goal]:
            if mode == 'H':
                handoff_num += 1
    return handoff_num


def real_test():
    '''
    real robot test
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--control_mode", type=int, default=0, help="Control Mode for motors. 0: Position, 1: Velocity, 2: Torque")
    argparser.add_argument("--gui", action='store_true', help="Show GUI")
    args = argparser.parse_args()
    control_mode = ControlMode(args.control_mode)
    print(f"control_mode: {control_mode}")
    
    overlap_rate = 1.0
    # get start poses
    # get_real_arrangement('start', Camera())
    real_start_arr = load_real_arrangement('start')
    sim_start_arr = {
        k:real_env_to_plan_env(v) for k,v in real_start_arr.items()
    }
    
    # get goal poses
    real_goal_arr = load_real_arrangement('goal')
    sim_goal_arr = {
        k:real_env_to_plan_env(v) for k,v in real_goal_arr.items()
    }

    Height = 1000
    Width = 2*Height/(1+overlap_rate)
    radius = 0.024/WORKSPACE_SIZE*Height # radius of bounding discs in planning 
    time_step_size = 0.002 # 1/240.0
    speed = 0.6283185307179586 # 0.3141592653589793

    start = time.time()
    planner = Bi_Directional_Search_Tree_Planner(
        sim_start_arr, 
        sim_goal_arr, 
        Height, 
        Width, 
        radius, 
        sampling=True, 
        overlap_rate=overlap_rate, 
        num_arms=2,
        makespan='MC'
    )
    action_list = planner.action_list
    if action_list[0][0][0] != 1:
        action_list[0], action_list[1] = action_list[1], action_list[0]
    action_list = plan2schedule(action_list)
    start = time.time()
    try:
        conflicts, execution_time, sim_control = timeout_simulation(
            Width, Height,sim_start_arr, sim_goal_arr, radius, action_list, 
            overlap_rate = overlap_rate, speed=speed, 
            time_step_size=time_step_size, control_mode=control_mode, 
            gui=args.gui)
    except timeout_decorator.timeout_decorator.TimeoutError:
        conflicts, execution_time = -1, -1
    
    


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--control_mode", type=int, default=0, help="Control Mode for motors. 0: Position, 1: Velocity, 2: Torque")
    argparser.add_argument("--gui", action='store_true', help="Show GUI")
    args = argparser.parse_args()
    run_simulation(args)
    # motion_planning_experiments()
    # real_test()
    # get_real_arrangement('start')
    # plan_recorder = Plan_Recorder()
    # plan_recorder.read_record()
    # plan_recorder.plot_plan(1/240.0)
    

