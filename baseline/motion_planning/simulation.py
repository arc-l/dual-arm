import os, sys
import math
import xml.etree.ElementTree as ET
from baseline.motion_planning.central_control import central_control
from baseline.motion_planning.robot_arm import ControlMode

from baseline.constants import WORKSPACE_SIZE, EPSILON, arm0_yml_path, arm1_yml_path, dual_arm_yml_path
from baseline.tools.util import print_progress_bar
import matplotlib.pyplot as plt
import yaml
import pickle
import datetime
import time


import sys
import numpy as np
sys.path.append('.')



def simulation(
        Width: float, Height: float, start_arr: dict, goal_arr: dict, radius: float,
         dual_action_list: dict, overlap_rate=1.0, speed=0.3141592653589793, instance_info=[],
        label_txt='test', time_step_size=1/240.0, control_mode=ControlMode.POSITION, gui=True,
        method='baseline', debug=False, raw_plan=None
        ):
    '''
    return result_dict = {
        'motion planning time': float, 
        'execution time': float, # execution time of the motion plan
        'plan length': List[float], # l2 norms of robot motions
        'control': central_control,
        'raw_plan': traj
        }
    '''
    update_cube_urdf(radius)
    ws_x, ws_y, robot0_position, robot1_position = get_ws_info(overlap_rate)
    # robot1_position = (robot1_position[0], robot1_position[1]+0.015, robot1_position[2])
    update_ws_urdf(ws_x, ws_y)
    
    # Setup environment
    control = central_control(
        Width, Height, start_arr, goal_arr,
        radius, dual_action_list, robot0_position,
        robot1_position, overlap_rate=overlap_rate,
        speed=speed,time_step_size=time_step_size,
        control_mode=control_mode,
        gui=gui
    )
    
    result_dict = {
        'motion planning time': 0.0, 
        'execution time': 0.0, 
        'plan length': [0.0, 0.0], 
        'raw_plan': None,
        'recorded_plan': None
        }
    motion_plan_start = time.time()
    if method in ['baseline', 'single-bl']:
        baseline_motion_planning(control, toppra_improve=False)
        if debug:
            input('ready to execute')
        control.record_replay()
        plan, exe_time = control.recorder.record, control.execution_time
        dist = control.recorder.compute_plan_length(ord=2)
        # control.recorder.plot_plan(time_step_size)
    elif method == 'bato': # baseline with toppra improvement
        baseline_motion_planning(control, toppra_improve=True)
        if debug:
            input('ready to execute')
        control.record_replay()
        plan, exe_time = control.recorder.record, control.execution_time
        dist = control.recorder.compute_plan_length(ord=2)
        # control.recorder.plot_plan(time_step_size)
    elif method == 'baseline_v2':
        baseline_motion_planning(control, toppra_improve=False)
        plan = control.recorder.record
        control.constraint_speeds()
        improved_plan = control.recorder.record
        exe_time = len(improved_plan[0])*control.time_step
        dist = control.recorder.compute_plan_length(ord=2)
        if debug:
            input('ready to execute')
            control.record_replay()
    elif method == 'bato_v2':
        control.recorder.record = raw_plan
        if raw_plan is None:
            raise ValueError('raw_plan is None')
        print('toppra improvement')
        control.recorder.record = control.recorder.toppra_improvement(acc_scale=1,speed_scale=control.speed/3.14,dt=control.time_step)
        plan = control.recorder.record
        control.constraint_speeds()
        improved_plan = control.recorder.record
        exe_time = len(improved_plan[0])*control.time_step
        dist = control.recorder.compute_plan_length(ord=2)
        if debug:
            input('ready to execute')
            control.record_replay()
    result_dict['motion planning time'] = time.time() - motion_plan_start
    result_dict['execution time'] = exe_time
    result_dict['plan length'] = [dist[0], dist[1]]
    result_dict['recorded_plan'] = control.recorder.record
    return result_dict, control
    

def baseline_motion_planning(control: central_control, toppra_improve=False):
    '''
    baseline motion planning
    '''
    control.update()
    # control.test()
    # write_motion_plan(instance_info, control.recorder.record, label_txt)
    if toppra_improve:
        print('toppra improvement')
        control.recorder.record = control.recorder.toppra_improvement(acc_scale=1,speed_scale=control.speed/3.14,dt=control.time_step)
    return control.recorder.record, control.execution_time
    
        
def update_cylinder_urdf(radius):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    scale = (radius/1000.0)/(0.0225/0.6)
    urdf_file = os.path.join(current_folder, 'assets', 'blocks', 'cylinder.urdf')
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        # Check if this is the mesh you want to modify
        if mesh.get('filename') == 'cylinder.obj':
            mesh.set('scale', f'{scale} {scale} 1')  # Set your new scale here
    # Save the modified XML back to a file
    tree.write(urdf_file)

    urdf_file = os.path.join(current_folder, 'assets', 'blocks', 'collision_free_cylinder.urdf')
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        # Check if this is the mesh you want to modify
        if mesh.get('filename') == 'cylinder.obj':
            mesh.set('scale', f'{scale} {scale} 1')  # Set your new scale here
    # Save the modified XML back to a file
    tree.write(urdf_file)

def update_cube_urdf(radius):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    scale = (radius/1000.0)/(0.0225/0.6)
    scale /= math.sqrt(2)
    # radius = radius/math.sqrt(2)
    scale = 1.0
    urdf_file = os.path.join(current_folder, 'assets', 'blocks', 'cube.urdf')
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        # Check if this is the mesh you want to modify
        if mesh.get('filename') == 'cube.obj':
            mesh.set('scale', f'{scale} {scale} 1')  # Set your new scale here
    # Save the modified XML back to a file
    tree.write(urdf_file)

    urdf_file = os.path.join(current_folder, 'assets', 'blocks', 'collision_free_cube.urdf')
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        # Check if this is the mesh you want to modify
        if mesh.get('filename') == 'cube.obj':
            mesh.set('scale', f'{scale} {scale} {scale}')  # Set your new scale here
            # mesh.set('scale', f'{scale} {scale} 1')  # Set your new scale here
    # Save the modified XML back to a file
    tree.write(urdf_file)

def update_ws_urdf(Width, Height):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    urdf_file = os.path.join(current_folder, 'assets', 'workspace', 'workspace.urdf')
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    scalex = Width/30.0
    scaley = Height/30.0
    for mesh in root.findall(".//mesh"):
        # Check if this is the mesh you want to modify
        if mesh.get('filename') == 'plane.obj':
            mesh.set('scale', f'{scalex} {scaley} 1')  # Set your new scale here
            # break  # Remove this if you want to modify all meshes with plane.obj
    # Save the modified XML back to a file
    tree.write(urdf_file)


def get_ws_info(overlap_rate: float):
    workspace_x = 2*WORKSPACE_SIZE/(1+overlap_rate)
    workspace_y = WORKSPACE_SIZE
    robot0_position = (
        0.25-WORKSPACE_SIZE/(1+overlap_rate),
        0.0, 0.0
    )
    robot1_position = (
        0.75+WORKSPACE_SIZE/(1+overlap_rate),
        0.0, 0.0
    )
    return workspace_x, workspace_y, robot0_position, robot1_position
