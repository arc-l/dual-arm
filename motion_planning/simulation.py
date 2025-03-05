import os, sys
import math
import xml.etree.ElementTree as ET
from motion_planning.utils import quaternion_representation_transform
from motion_planning.central_control import central_control
from motion_planning.plan_recorder import Arm_State
from motion_planning.utils import env_pose_to_sim_pose
from motion_planning.robot_arm import ControlMode
from motion_planning.test import test_curobo_single_arm_motion_gen


from motion_planning.curobo_planning import Motion_Gen
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from constants import WORKSPACE_SIZE, EPSILON, arm0_yml_path, arm1_yml_path, dual_arm_yml_path
from motion_planning.plan_recorder import Plan_Recorder
from curobo.types.robot import JointState
from curobo.types.math import Pose
from tools.util import print_progress_bar
import matplotlib.pyplot as plt
import torch
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
    # update_cylinder_urdf(radius) # update object size
    update_cube_urdf(radius)
    ws_x, ws_y, robot0_position, robot1_position = get_ws_info(overlap_rate)
    robot1_position = (robot1_position[0], robot1_position[1]+0.015, robot1_position[2])
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
        # 'control': control
        }
    # get_robot_samples(control)
    # test_curobo_single_motion_gen(control=control)
    # test_more_moves(control=control)
    # execute_single_arm(control=control)
    motion_plan_start = time.time()
    if method in ['cuto', 'single-cuto']:
        horizon = 10
        curobo_tamp(control,horizon)
        if debug:
            input('ready to execute')
        control.record_replay()
        plan, exe_time = control.recorder.record, control.execution_time
        dist = control.recorder.compute_plan_length(ord=2)
        # control.recorder.plot_plan(time_step_size)
    elif method in ['baseline', 'single-bl']:
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
    elif method == 'cuto_v2':
        control.time_step *= 1/10.0
        horizon = 1
        curobo_tamp(control,horizon)
        if debug:
            input('ready to execute')
            control.record_replay()
        plan = control.recorder.record
        exe_time = len(plan[0])*control.time_step*horizon
        dist = control.recorder.compute_plan_length(ord=2)
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
    # result_dict['control'] = control
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
    

def curobo_tamp(control:central_control, horizon=10):
    '''
    1. get subgoals
    2. compute IK
    3. for current two subgoal IKs
        3.1 plan individual motions
        3.2 if individual motions collide, plan dual motions
            3.2.1 the next subgoal (nearest or priority)
            3.2.2 sample collision-free 12 dof joint state for traj opt
    '''
    curobo_dt = control.time_step*horizon
    
    robot_positions = np.array([
        control.arms[0].base_position,
        control.arms[1].base_position
    ])
    
    # reset speed scale to 1
    for arm_yml_path in [arm0_yml_path, arm1_yml_path, dual_arm_yml_path]:
        with open(arm_yml_path, 'r') as f:
            arm_config = yaml.safe_load(f)
        arm_config['robot_cfg']['kinematics']['cspace']['velocity_scale'] = 0.5
        with open(arm_yml_path, 'w') as f:
            yaml.dump(arm_config, f)
    
    planner = Motion_Gen(robot_positions=robot_positions, curobo_dt=curobo_dt)
    
    
    dual_action_list = {
        arm_id:control.arms[arm_id].action_list for arm_id in range(2)
    }
    wait_handoff = [False, False] # whether the arm is waiting for handoff 
    wait_for_clearance = [False, False] # wait until the placement/grasping is feasible
    pnp_time = control.arms[0].pnp_steps # pick and place time
    pnp_time_left = [pnp_time, pnp_time] # time left for pick and place
    
    
    # start planning
    complete_traj = None # a jointstate object for 12 dof traj
    dual_trajs = [] # a list of jointstate objects for 12 dof trajs
    # start state
    curr_single_states = [
        JointState.from_position(
            torch.tensor(control.arms[arm_id].arm_home_joints, dtype=torch.float32, device='cuda:0').view(1, -1)
        ) if len(dual_action_list[arm_id])>0 else None for arm_id in range(2)
    ]
    object_current_pose_np = {
        k:np.array(env_pose_to_sim_pose(control.start_arr[k],control.overlap_rate)) for k in control.start_arr.keys()
        }
    # update object pose, away from the workspace, avoid collision between robot and object
    for object_idx, pos in object_current_pose_np.items():
        pos = list(pos)
        pos[0] += 5
        control._pb.resetBasePositionAndOrientation(
            control.object_body_ids[object_idx],
            pos,
            control._pb.getQuaternionFromEuler([0.0, -0.0, 1.5707964897155762]),
        )
    for _ in range(500):
        control.stepSimulation()
    next_subgoals = [None, None]
    need_update_goal = set([0,1]) # robots need update goal
    total_iterations = sum([len(dual_action_list[arm_id]) for arm_id in range(2)])
    while True:
        print_progress_bar(total_iterations-sum([len(dual_action_list[arm_id]) for arm_id in range(2)]), total_iterations, prefix='Progress:', suffix='Complete', length=50)
        # update next subgoals and do IK if needed
        for arm_id in need_update_goal:
            if len(dual_action_list[arm_id])==0: # no more subgoals
                next_subgoals[arm_id] = None
                continue
            action = dual_action_list[arm_id][0]
            # check whether the action is feasible
            if action.comment == 'to_goal': # whether the goal position is collision free
                curr_obj_body_id = action.bodyID
                for curr_obj in control.object_body_ids.keys():
                    if control.object_body_ids[curr_obj] == curr_obj_body_id:
                        break
                else:
                    raise ValueError('object not found')
                goal_np = np.array((action.ee_pose[0][0], action.ee_pose[0][1]))
                FEASIBLE = True
                for other_obj in object_current_pose_np.keys():
                    if other_obj == curr_obj:
                        continue
                    elif object_current_pose_np[other_obj] is None:
                        continue
                    if np.linalg.norm(goal_np-object_current_pose_np[other_obj][:2]) < 1.8*control.radius:
                        FEASIBLE = False
                        break
                if not FEASIBLE:
                    wait_for_clearance[arm_id] = True
                    next_subgoals[arm_id] = None
                    continue
            elif action.comment == 'to_current': # whether the object is at the position
                curr_obj_body_id = action.bodyID
                for curr_obj in control.object_body_ids.keys():
                    if control.object_body_ids[curr_obj] == curr_obj_body_id:
                        break
                else:
                    raise ValueError('object not found')
                
                target_np = np.array((action.ee_pose[0][0], action.ee_pose[0][1]))
                FEASIBLE = True
                if object_current_pose_np[curr_obj] is None:
                    FEASIBLE = False
                elif np.linalg.norm(target_np-object_current_pose_np[curr_obj][:2]) > control.radius:
                    FEASIBLE = False
                if not FEASIBLE:
                    wait_for_clearance[arm_id] = True
                    next_subgoals[arm_id] = None
                    continue
            action = dual_action_list[arm_id].pop(0)
            if action.joint_goal is not None: # IK already computed
                next_subgoals[arm_id] = action
            elif action.ee_pose is not None: # compute IK
                curobo_pose = Pose(
                    torch.tensor(action.ee_pose[0], device='cuda:0').view(1, -1),
                    torch.tensor(quaternion_representation_transform(action.ee_pose[1]), device='cuda:0').view(1, -1),
                )
                if action.status == 'M': # go straight up or down
                    ik_sol = planner.single_ik_solve(
                        planner.ik_solvers[arm_id], curobo_pose, curr_single_states[arm_id].position
                        # planner.ik_solvers[arm_id], curobo_pose, control.arms[arm_id].arm_first_joints
                    )
                else:
                    ik_sol = planner.single_ik_solve(
                        # planner.ik_solvers[arm_id], curobo_pose, curr_single_states[arm_id].position
                        planner.ik_solvers[arm_id], curobo_pose, control.arms[arm_id].arm_first_joints
                    )
                action.joint_goal = ik_sol.view(-1).cpu().numpy()
                next_subgoals[arm_id] = action
            elif wait_handoff[arm_id]:
                next_subgoals[arm_id] = None
            elif action.status in ['activate', 'deactivate', 'handoff']:
                action.joint_goal = curr_single_states[arm_id].position.view(-1).cpu().numpy()
                next_subgoals[arm_id] = action
            else:
                next_subgoals[arm_id] = None
        if (next_subgoals[0] is None )and (next_subgoals[1] is None) and (len(dual_action_list[0])==0) and (len(dual_action_list[1])==0): # all tasks finished
            break
        need_update_goal = set() # reset update goal
        
        # whether a robot is waiting for handoff
        for i in range(2):
            if next_subgoals[i] and next_subgoals[i].status=='handoff': # just reached the handoff prepare pose
                wait_handoff[i] = True
        
        print('next_subgoals')
        print('0: ', next_subgoals[0].comment if next_subgoals[0] else None)
        print('1: ', next_subgoals[1].comment if next_subgoals[1] else None)
        # get individual trajs
        single_trajs = []
        for arm_id in range(2):
            if next_subgoals[arm_id] is None:
                traj = curr_single_states[arm_id]
                single_trajs.append(traj)
                continue
            if next_subgoals[arm_id].status in ['activate', 'deactivate']:
                traj = JointState.from_position(
                    curr_single_states[arm_id].position.repeat(pnp_time_left[arm_id],1).view(pnp_time_left[arm_id],-1)
                    )
            elif next_subgoals[arm_id].status == 'M': # M: manipulation, pre-grasp <-> grasp
                control.arms[arm_id].move_to_joint(curr_single_states[arm_id].position.view(-1).cpu().numpy(), 3.14) # move to pose with high speed
                control.arms[1-arm_id].move_to_joint(control.arms[1-arm_id].arm_home_joints, 3.14) # move to pose with high speed
                for _ in range(500):
                    control.stepSimulation()
                
                
                traj = JointState.from_position(
                    curr_single_states[arm_id].position.view(1,-1)
                    )
                manipulation_ready = False
                while not manipulation_ready:
                    control.arms[arm_id].next_joint_state(next_subgoals[arm_id].joint_goal, control.speed*horizon)
                    control.arms[1-arm_id].next_joint_state(control.arms[1-arm_id].arm_home_joints, control.speed*horizon)
                    for _ in range(horizon):
                        control.stepSimulation()
                    curr_position = control.arms[arm_id].get_joint_state()
                    diff_joints = np.linalg.norm(curr_position - next_subgoals[arm_id].joint_goal)
                    manipulation_ready = diff_joints < EPSILON
                    curr_js = JointState.from_position(
                        torch.tensor(curr_position, dtype=torch.float32, device='cuda:0').view(1,-1),
                    )
                    traj = traj.stack(curr_js)
                    
            elif next_subgoals[arm_id].status == 'handoff': # idle and wait for handoff
                traj = JointState.from_position(
                    curr_single_states[arm_id].position.repeat(pnp_time,1).view(pnp_time,-1) # default idle time, does not matter
                    )
            else: # curobo planning 
                next_js = JointState.from_position(
                    torch.tensor(next_subgoals[arm_id].joint_goal, dtype=torch.float32, device='cuda:0').view(1,-1),
                )    
                PLAN_SUCCESS = False
                while not PLAN_SUCCESS:
                    result = planner.trajopt_solve(
                        planner.motion_gens[arm_id],
                        curr_single_states[arm_id],
                        next_js
                    )
                    print("TrajOpt Solution for arm {}: ".format(arm_id), result.success, result.status, result.solve_time)
                    PLAN_SUCCESS = result.success
                    # if not result.success:
                    #     test_move_to_single_goal(control, curr_single_states[arm_id], arm_id)
                    #     test_move_to_single_goal(control, next_js, arm_id)
                # traj = result.interpolated_solution.trim_trajectory(
                #         0, result.path_buffer_last_tstep[0]
                #     )
                traj = result.get_interpolated_plan()
            single_trajs.append(traj)

        # choose next dual subgoal
        if wait_handoff[0] and wait_handoff[1]:  # both arms are waiting for handoff
            wait_handoff = [False, False]
            # execute handoff
            for arm_id in range(2):
                if next_subgoals[arm_id].comment == 'handoff delivery':
                    deliver_arm = control.arms[arm_id]
                if next_subgoals[arm_id].comment == 'handoff receive':
                    receive_arm = control.arms[arm_id]
            current_object = next_subgoals[deliver_arm.arm_id].bodyID
            control.curobo_handoff_coordination(
                current_object=current_object, deliver_arm=deliver_arm, receive_arm=receive_arm, horizon=horizon,
                curr_joint_states={
                    0:curr_single_states[0].position.view(-1).cpu().numpy(), 
                    1:curr_single_states[1].position.view(-1).cpu().numpy()}
                )
            need_update_goal = set([0,1])
            curr_single_states = [
                JointState.from_position(torch.tensor(control.arms[arm_id].arm_first_joints, dtype=torch.float32, device='cuda:0').view(1, -1))
                for arm_id in range(2)
            ]
            continue
        elif (next_subgoals[0] is None) and (not wait_for_clearance[0]):  # arm 0 finished or waiting for handoff
            print('arm 0 finished or waiting for handoff')
            reaching_goal_arm_id = 1
            dual_traj = JointState(position=torch.cat(
                [curr_single_states[0].position.repeat(single_trajs[1].position.shape[0],
                                                       1),
                 single_trajs[1].position],
                dim=-1).view(-1, 12),
                velocity=torch.cat(
                [curr_single_states[0].velocity.repeat(single_trajs[1].velocity.shape[0],
                                                       1),
                 single_trajs[1].velocity],
                dim=-1).view(-1, 12),
                acceleration=torch.cat(
                [curr_single_states[0].acceleration.repeat(
                    single_trajs[1].acceleration.shape[0],
                    1),
                 single_trajs[1].acceleration],
                dim=-1).view(-1, 12),
                jerk=torch.cat(
                [curr_single_states[0].jerk.repeat(single_trajs[1].jerk.shape[0],
                                                   1),
                 single_trajs[1].jerk],
                dim=-1).view(-1, 12))
            curr_single_states[reaching_goal_arm_id] = JointState.from_position(
                torch.tensor(next_subgoals[reaching_goal_arm_id].joint_goal, dtype=torch.float32, device='cuda:0').view(1,-1),
            )    
        elif (next_subgoals[1] is None) and (not wait_for_clearance[1]):  # arm 1 finished or waiting for handoff
            print('arm 1 finished or waiting for handoff')
            reaching_goal_arm_id = 0
            dual_traj = JointState(position=torch.cat(
                [single_trajs[0].position, curr_single_states[1].position.repeat(
                    single_trajs[0].position.shape[0],
                    1)],
                dim=-1).view(-1, 12),
                velocity=torch.cat(
                [single_trajs[0].velocity, curr_single_states[1].velocity.repeat(
                    single_trajs[0].velocity.shape[0],
                    1)],
                dim=-1).view(-1, 12),
                acceleration=torch.cat(
                [single_trajs[0].acceleration, curr_single_states[1].acceleration.repeat(
                    single_trajs[0].acceleration.shape[0],
                    1)],
                dim=-1).view(-1, 12),
                jerk=torch.cat(
                [single_trajs[0].jerk, curr_single_states[1].jerk.repeat(
                    single_trajs[0].jerk.shape[0],
                    1)],
                dim=-1).view(-1, 12))
            curr_single_states[reaching_goal_arm_id] = JointState.from_position(
                torch.tensor(next_subgoals[reaching_goal_arm_id].joint_goal, dtype=torch.float32, device='cuda:0').view(1,-1),
            )    
        else: # both arms are in execution
            if wait_for_clearance[0] or wait_handoff[0]:
                next_time_step = single_trajs[1].position.shape[0]
                reaching_goal_arm_id = 1
            elif wait_for_clearance[1] or wait_handoff[1]:
                next_time_step = single_trajs[0].position.shape[0]
                reaching_goal_arm_id = 0
            elif single_trajs[0].position.shape[0] < single_trajs[1].position.shape[0]: # traj 0 is shorter
                next_time_step = single_trajs[0].position.shape[0]
                reaching_goal_arm_id = 0
            else:  # traj 1 is shorter
                next_time_step = single_trajs[1].position.shape[0]
                reaching_goal_arm_id = 1
            
            # in case the deprioritized traj is shorter
            if single_trajs[1-reaching_goal_arm_id].position.shape[0] < next_time_step:
                single_trajs[1-reaching_goal_arm_id] = single_trajs[1-reaching_goal_arm_id].stack(
                    JointState.from_position(
                        single_trajs[1-reaching_goal_arm_id].position[-1].view(1,-1).repeat(next_time_step-single_trajs[1-reaching_goal_arm_id].position.shape[0],1)
                    )
                )
            
            # collision check
            if not detect_collisions([traj[:next_time_step] for traj in single_trajs],planner): # no collision
                print('single trajs have no collision')
                dual_traj = JointState(
                    position=torch.cat(
                        [single_trajs[0].position[: next_time_step],
                         single_trajs[1].position[: next_time_step]],
                        dim=-1),
                    velocity=torch.cat(
                        [single_trajs[0].velocity[: next_time_step],
                         single_trajs[1].velocity[: next_time_step]],
                        dim=-1),
                    acceleration=torch.cat(
                        [single_trajs[0].acceleration[: next_time_step],
                         single_trajs[1].acceleration[: next_time_step]],
                        dim=-1),
                    jerk=torch.cat(
                        [single_trajs[0].jerk[: next_time_step],
                         single_trajs[1].jerk[: next_time_step]],
                        dim=-1))
                curr_single_states = [
                    JointState.from_position(single_trajs[0][[next_time_step-1]].position.view(1,-1)),
                    JointState.from_position(single_trajs[1][[next_time_step-1]].position.view(1,-1))
                ]

            else:  # collision detected
                # in the case of collision, we need to figure out which arm has the priority
                # by default, the arm with the shorter traj has the priority
                # however, if the longer traj is prioritized, we need to swap the priority
                
                if (next_subgoals[reaching_goal_arm_id] and next_subgoals[reaching_goal_arm_id].prioritized) or (
                    not (next_subgoals[1-reaching_goal_arm_id] and next_subgoals[1-reaching_goal_arm_id].prioritized)
                    ): 
                    # not swapping priority if the shorter arm is prioritized or the other arm is not prioritized
                    pass
                else: # swap priority
                    reaching_goal_arm_id = 1-reaching_goal_arm_id
                    next_time_step = single_trajs[reaching_goal_arm_id].position.shape[0]
                    # in case the deprioritized traj is shorter
                    if single_trajs[1-reaching_goal_arm_id].position.shape[0] < next_time_step:
                        single_trajs[1-reaching_goal_arm_id] = single_trajs[1-reaching_goal_arm_id].stack(
                            JointState.from_position(
                                single_trajs[1-reaching_goal_arm_id].position[-1].view(1,-1).repeat(next_time_step-single_trajs[1-reaching_goal_arm_id].position.shape[0],1)
                            )
                        )
                
                print('single trajs have collision')
                curr_dual_state = JointState(
                    position=torch.cat([curr_single_states[0].position, curr_single_states[1].position], dim=-1),
                    velocity=torch.cat([curr_single_states[0].velocity, curr_single_states[1].velocity], dim=-1),
                    acceleration=torch.cat([curr_single_states[0].acceleration, curr_single_states[1].acceleration], dim=-1),
                    jerk=torch.cat([curr_single_states[0].jerk, curr_single_states[1].jerk], dim=-1)
                )
                if next_subgoals[reaching_goal_arm_id].comment in ['down', 'activate', 'deactivate', 'up']: # during manipulation
                    planner.dual_priorities = reaching_goal_arm_id
                    dual_traj = manipulation_dual_planning(planner=planner, prioritized_arm=reaching_goal_arm_id,
                        mani_traj=single_trajs[reaching_goal_arm_id][:next_time_step], curr_js=curr_dual_state)
                else:
                    # change the distance weight of arms
                    # planner.update_dual_cspace_weight(reaching_goal_arm_id)
                    PLAN_SUCCESS = False
                    while not PLAN_SUCCESS:
                        if planner.dual_priorities is not None:
                            planner.restore_default_dual_planning()
                            planner.dual_priorities = None
                        dual_state = JointState.from_position(
                            joint_state_sampling(
                            arm_id=1-reaching_goal_arm_id,
                            planner=planner,
                            other_arm_state=JointState.from_position(
                                torch.tensor(next_subgoals[reaching_goal_arm_id].joint_goal, dtype=torch.float32, device='cuda:0').view(1,-1),
                            ),
                            curr_js=curr_single_states[1-reaching_goal_arm_id],
                            ref_js=single_trajs[1-reaching_goal_arm_id][next_time_step-1],    
                        ))
                        result = planner.trajopt_solve(planner.motion_gens[2], curr_dual_state, dual_state)
                        print("TrajOpt Solution for dual arms: ", result.success, result.solve_time)
                        PLAN_SUCCESS = result.success
                    # if not result.success:
                    #     test_move_to_dual_goal(control, curr_dual_state)
                    #     test_move_to_dual_goal(control, dual_state)
                    # dual_traj = result.interpolated_solution.trim_trajectory(
                    #     1, result.path_buffer_last_tstep[0]
                    # )
                    dual_traj = result.get_interpolated_plan()
                curr_single_states = [
                    JointState.from_position(dual_traj[-1].position[:6].view(1,-1)),
                    JointState.from_position(dual_traj[-1].position[6:].view(1,-1))
                ]
        print('arm', reaching_goal_arm_id, 'is prioritized')
        # control.arms[0].set_joint_state(dual_traj[0].position[:6].view(-1).cpu().numpy())
        # control.arms[1].set_joint_state(dual_traj[0].position[6:].view(-1).cpu().numpy())
        # for _ in range(100):
        #     control.stepSimulation()
        # input('execute to next subgoal')
        # for waypt_id in range(dual_traj.position.shape[0]):
        #     for arm_id in range(2):
        #         joint_state = dual_traj[waypt_id][arm_id*6:(arm_id+1)*6]
        #         control.arms[arm_id].move_to_joint(joint_state)
        #     for _ in range(10):
        #         control.stepSimulation()
        # input('finish')
        # a=1

        if complete_traj is None:
            complete_traj = dual_traj
        else:
            complete_traj = complete_traj.stack(dual_traj)
        dual_trajs.append(dual_traj)
        if (
            ((next_subgoals[0] is None) or (next_subgoals[0].status not in ['activate', 'deactivate'])) and
            ((next_subgoals[1] is None) or (next_subgoals[1].status not in ['activate', 'deactivate']))
            ):
            dual_traj = planner.toppra_improvement(dual_traj)
        control.record_curobo_traj(dual_traj, next_subgoals=next_subgoals, horizon=horizon)



        # update utility
        pnp_time_left[reaching_goal_arm_id] = pnp_time # reset pnp time
        if not wait_handoff[reaching_goal_arm_id]: # not waiting for handoff
            need_update_goal = set([reaching_goal_arm_id]) # update reaching goal arm
            # update object pose
            if next_subgoals[reaching_goal_arm_id].comment == 'activate':
                for obj_id in control.object_body_ids.keys():
                    if control.object_body_ids[obj_id] == next_subgoals[reaching_goal_arm_id].bodyID:
                        break
                else:
                    raise ValueError('object not found')
                object_current_pose_np[obj_id] = None
                # reset position in pybullet, throw the object away from the workspace
                # control._pb.resetBasePositionAndOrientation(
                #     control.object_body_ids[obj_id],[4*(reaching_goal_arm_id-0.5),0,0.05],
                #     control._pb.getQuaternionFromEuler([0.0, -0.0, 1.5707964897155762]),
                # )
                # for _ in range(500):
                #     control.stepSimulation()
            elif next_subgoals[reaching_goal_arm_id].comment == 'deactivate':
                # ee_pose = control.arms[reaching_goal_arm_id].get_ee_qua_pose()
                # ee_pose = dual_action_list[reaching_goal_arm_id][0].ee_pose
                for obj_id in control.object_body_ids.keys():
                    if control.object_body_ids[obj_id] == next_subgoals[reaching_goal_arm_id].bodyID:
                        break
                else:
                    raise ValueError('object not found')
                object_current_pose_np[obj_id] = np.array((
                    dual_action_list[reaching_goal_arm_id][0].ee_pose[0][0],
                    dual_action_list[reaching_goal_arm_id][0].ee_pose[0][1],
                    0.023000000044703484
                    )) # the x,y,z position
                # reset position in pybullet
                # control._pb.resetBasePositionAndOrientation(
                #     control.object_body_ids[obj_id],object_current_pose_np[obj_id],
                #     control._pb.getQuaternionFromEuler([0.0, -0.0, 1.5707964897155762]),
                # )
                # for _ in range(500):
                #     control.stepSimulation()
        if (next_subgoals[1-reaching_goal_arm_id]) and (next_subgoals[1-reaching_goal_arm_id].status in ['activate', 'deactivate']): # update remaining pnp time
            pnp_time_left[1-reaching_goal_arm_id] = max(pnp_time_left[1-reaching_goal_arm_id] - next_time_step, 1)
        for arm_id in range(2):
            if wait_for_clearance[arm_id]:
                wait_for_clearance[arm_id] = False
                need_update_goal.add(arm_id)
            
    # input('ready to execute')
    # control.record_replay()
    # for traj in dual_trajs:
    #     for waypt_id in range(traj.position.shape[0]):
    #         for arm_id in range(2):
    #             joint_state = traj[waypt_id][arm_id*6:(arm_id+1)*6]
    #             control.arms[arm_id].move_to_joint(joint_state)
    #         for _ in range(10):
    #             control.stepSimulation()
    #     input('next')
    # return control.recorder.record, control.execution_time
    control.recorder.write_record()
    return 0,0


def execute_single_arm(control:central_control,speed_scale=0.1, arm_id=0):
    horizon = 1
    
    methods = [
        # 'curobo', 
        'curobo_toppra', 
        # 'baseline'
        ]
    
    control.speed = 3.14*speed_scale
    control.arms[0].speed = 3.14*speed_scale
    control.arms[1].speed = 3.14*speed_scale
    
    for method in methods:
        traj_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'real_time_control',
            method+'_traj0_'+str(int(10*speed_scale))+'.pkl'
        )    
        traj = pickle.load(open(traj_file, 'rb'))
        
        timestr = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')
        dims = traj.position.shape[1]
        _, axs = plt.subplots(dims, 1)
        timesteps = [i * control.time_step for i in range(2*traj.position.shape[0])]
        for joint_id in range(dims):
            Y = traj.position[:,joint_id].cpu().numpy().reshape(-1)
            Y = list(Y)
            Y = Y + Y[-1::-1]
            axs[joint_id].plot(timesteps, Y, label='desired')
        
        
        for waypt_id in range(len(traj)):
            joint_state = traj[waypt_id]
            control.arms[arm_id].move_to_joint(joint_state)
            for _ in range(1):
                control.stepSimulation()
            arm_states = dict()
            arm_states[arm_id] = Arm_State(
                tuple(),traj[waypt_id].position.cpu().numpy().reshape(-1),
                None,'T','G',np.zeros(6),
                np.zeros(6),np.zeros(6)
            )
            arm_states[1-arm_id] = Arm_State(
                tuple(),
                control.arms[1-arm_id].get_joint_state(),
                None,'T','G',np.zeros(6),np.zeros(6),
                np.zeros(6)
            )
            control.recorder.add_new_step(arm_states)
        control.record_replay()
        for joint_id in range(6):
                Y = []
                for state in control.recorder.record[arm_id]:
                    Y.append(state.joint_state[joint_id])
                Y = Y + Y[-1::-1]
                axs[6*arm_id + joint_id].plot(timesteps, Y, label='sim exec')
        # control.recorder.plot_plan(control.time_step)
        plt.legend()
        plt.savefig(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'figures',
            method+'_traj_'+timestr+'.png'
        ))
        plt.cla()
        plt.clf()
        control.recorder = Plan_Recorder()
        control.reset_env()
        
    exit()
        
            

def joint_state_sampling(arm_id: int, planner: Motion_Gen, other_arm_state: JointState, curr_js: JointState, ref_js: JointState,**kwargs):
    '''
    sample joint state for traj opt
    '''
    # sample 6 dof joint state based on curr_js and next_goal_js
    # check collision
    # if collision, sample again
    # if no collision, return
    radius = 0.05  # search radius
    while True:
        for _ in range(100):
            sampled_joint = (torch.rand((1, 6), device='cuda:0')*2*radius-radius).view(1, -1)+ref_js.position
            sampled_joint = torch.clamp(sampled_joint, -6.2, 6.2)
            sampled_js = JointState.from_position(sampled_joint)
            if arm_id == 0:
                dual_joint_position = torch.cat([sampled_js.position, other_arm_state.position], dim=-1)
            else:
                dual_joint_position = torch.cat([other_arm_state.position, sampled_js.position], dim=-1)
            d_world, d_self = planner.dual_coll_checker.get_world_self_collision_distance_from_joints(
                dual_joint_position)
            if torch.max(d_world)+torch.max(d_self) > 0.0:
                # collision detected
                continue
            elif torch.max(d_world)+torch.max(d_self) <= 0.0:  # no collision
                return dual_joint_position
            else:
                raise ValueError('collision detection error')
        radius *= 2
        if radius > 0.3: # try to go back to the retract config
            diff_joint = planner.motion_gens[arm_id].get_retract_config().view(1, -1)-curr_js.position
            for i in range(1,30):
                sampled_joint = curr_js.position + diff_joint*(i+1)/30.0
                sampled_js = JointState.from_position(sampled_joint)
                if arm_id == 0:
                    dual_joint_position = torch.cat([sampled_js.position, other_arm_state.position], dim=-1)
                else:
                    dual_joint_position = torch.cat([other_arm_state.position, sampled_js.position], dim=-1)
                d_world, d_self = planner.dual_coll_checker.get_world_self_collision_distance_from_joints(
                    dual_joint_position)
                if torch.max(d_world)+torch.max(d_self) > 0.0:
                    # collision detected
                    continue
                elif torch.max(d_world)+torch.max(d_self) <= 0.0:  # no collision
                    return dual_joint_position
            else:
                raise ValueError('retract config collision detection error')
            
            
def manipulation_dual_planning(planner: Motion_Gen, prioritized_arm:int, mani_traj: JointState, curr_js: JointState):
    '''
    deal with collision during manipulation
    [start, start]->[start, retract] => [start, start]->[start, coll-free-from-traj]
    start->goal
    start->coll-free->stay
    '''
    # update robot config
    if prioritized_arm == 0:
        locked_joints = {
            'shoulder_pan_joint': curr_js.position[0,0],
            'shoulder_lift_joint': curr_js.position[0,1],
            'elbow_joint': curr_js.position[0,2],
            'wrist_1_joint': curr_js.position[0,3],
            'wrist_2_joint': curr_js.position[0,4],
            'wrist_3_joint': curr_js.position[0,5]
            }
    elif prioritized_arm == 1:
        locked_joints = {
            'shoulder_pan_joint_1': curr_js.position[0,6],
            'shoulder_lift_joint_1': curr_js.position[0,7],
            'elbow_joint_1': curr_js.position[0,8],
            'wrist_1_joint_1': curr_js.position[0,9],
            'wrist_2_joint_1': curr_js.position[0,10],
            'wrist_3_joint_1': curr_js.position[0,11]
            }
    else:
        raise Exception("priority must be 0 or 1")
    # not sure which one is necessary
    planner.robot_cfgs[2].kinematics.kinematics_parser.lock_joints = locked_joints
    planner.robot_cfgs[2].kinematics.kinematics_config.lock_joints = locked_joints
    
    planner.motion_gen_configs[2] = MotionGenConfig.load_from_robot_config(
        planner.robot_cfgs[2], planner.world_cfg, planner.tensor_args,
        **planner.motion_gen_config_params
        )
    
    # Free up the cache
    torch.cuda.empty_cache()
    
    planner.motion_gens[2] = MotionGen(planner.motion_gen_configs[2])
    print('warming up dual arm motion gen')
    planner.motion_gens[2].warmup(planner.motion_gen_warmup_params)
    planner.dual_graph_warmup()
    print('warmup dual arm motion gen done')
    # [start, start]->[start, retract]
    retract_config = planner.motion_gens[1-prioritized_arm].get_retract_config().view(1, -1)
    if prioritized_arm == 0:
        goal_dual_state = JointState.from_position(
            torch.cat([curr_js.position[0,:6].view(1,-1), retract_config],dim=-1))
    else:
        goal_dual_state = JointState.from_position(
            torch.cat([retract_config, curr_js.position[0,6:].view(1,-1)],dim=-1))
    PLANNING_SUCCESS = False
    while not PLANNING_SUCCESS:
        result = planner.trajopt_solve(planner.motion_gens[2], curr_js, goal_dual_state)
        print("TrajOpt Solution for dual arms: ", result.success, result.solve_time)
        if result.success:
            PLANNING_SUCCESS = True
    dual_traj = result.get_interpolated_plan()
    
    # get the earliest time step that is collision free from mani traj
    rough_traj = mani_traj
    for time_step in range(dual_traj.shape[0]):
        if prioritized_arm == 0:
            checking_traj = torch.cat([rough_traj.position, dual_traj[time_step].position[6:].view(1,-1).repeat((rough_traj.shape[0],1))],dim=-1).view(1,-1,12)
        else:
            checking_traj = torch.cat([dual_traj[time_step].position[:6].view(1,-1).repeat((rough_traj.shape[0],1)), rough_traj.position],dim=-1).view(1,-1,12)
        d_world, d_self = planner.dual_coll_checker.get_world_self_collision_distance_from_joint_trajectory(checking_traj)
        if torch.max(d_world)+torch.max(d_self) > 0.0: # collision detected
            continue
        else: # no collision
            break
    yielding_traj = dual_traj[:time_step+1,(1-prioritized_arm)*6:(2-prioritized_arm)*6]
    
    # velocity tuning
    dual_traj = velocity_tuning(planner, mani_traj, yielding_traj, prioritized_arm)
    return dual_traj

        
def velocity_tuning(planner:Motion_Gen, primary_traj:JointState, yielding_traj:JointState, prioritized_arm:int):
    '''
    velocity tuning for dual arm
    '''
    # init dual traj
    if prioritized_arm == 0:
        dual_traj = JointState(
            position=torch.cat([primary_traj.position[0].view(1,-1), yielding_traj.position[0].view(1,-1)],dim=-1),
            velocity=torch.cat([primary_traj.velocity[0].view(1,-1), yielding_traj.velocity[0].view(1,-1)],dim=-1),
            acceleration=torch.cat([primary_traj.acceleration[0].view(1,-1), yielding_traj.acceleration[0].view(1,-1)],dim=-1),
            jerk=torch.cat([primary_traj.jerk[0].view(1,-1), yielding_traj.jerk[0].view(1,-1)],dim=-1)
            )
    else:
        dual_traj = JointState(
            position=torch.cat([yielding_traj.position[0].view(1,-1), primary_traj.position[0].view(1,-1)],dim=-1),
            velocity=torch.cat([yielding_traj.velocity[0].view(1,-1), primary_traj.velocity[0].view(1,-1)],dim=-1),
            acceleration=torch.cat([yielding_traj.acceleration[0].view(1,-1), primary_traj.acceleration[0].view(1,-1)],dim=-1),
            jerk=torch.cat([yielding_traj.jerk[0].view(1,-1), primary_traj.jerk[0].view(1,-1)],dim=-1))
    
    primary_step, yielding_step = 1, 0
    while primary_step < primary_traj.shape[0]: # the prioritized arm is still moving to goal
        NO_COLLISION = False
        while not NO_COLLISION:
            # combine for collision check
            if prioritized_arm == 0:
                checking_joint = torch.cat([primary_traj.position[primary_step].view(1,-1), yielding_traj.position[yielding_step].view(1,-1)],dim=-1).view(1,1,-1)
            else:
                checking_joint = torch.cat([yielding_traj.position[yielding_step].view(1,-1), primary_traj.position[primary_step].view(1,-1)],dim=-1).view(1,1,-1)
            d_world, d_self = planner.dual_coll_checker.get_world_self_collision_distance_from_joint_trajectory(checking_joint)
            NO_COLLISION = (torch.max(d_world)+torch.max(d_self) <= 0.00) # no hard collision detected considering the error coming from the roughness
            if not NO_COLLISION:
                yielding_step += 1
                assert yielding_step <= yielding_traj.shape[0]-1
            else:
                coll_free_state = JointState.from_position(checking_joint.view(1,-1))
        dual_traj = dual_traj.stack(coll_free_state)
        primary_step += 1
    return dual_traj
        
        
def detect_collisions(single_trajs: list, planner: Motion_Gen):
    '''
    detect path conflicts between two trajs
    '''
    traj_lens = [traj.position.shape[0] for traj in single_trajs]
    for time_step in range(max(traj_lens)):
        traj_js = [single_trajs[i].position[min(time_step, traj_lens[i]-1)].view(1, -1) for i in range(2)]
        dof_state = torch.cat(traj_js, dim=-1)
        d_world, d_self = planner.dual_coll_checker.get_world_self_collision_distance_from_joints(dof_state)
        # potential alternatives
        # planner.dual_coll_checker.get_world_self_collision_distance_from_joint_trajectory(dof_state)
        if torch.max(d_world)+torch.max(d_self) > 0.0:
            # collision detected
            return True
    return False


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
    radius = radius/math.sqrt(2)
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


def write_motion_plan(instance_info, plan, labeltxt):
    if len(instance_info) >= 1:
        experimentID = instance_info[0]
    else:
        experimentID = 'test'
    folder = os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
        "motion_plans",
        labeltxt,
        experimentID
    )
    if not os.path.exists(folder):
        os.makedirs(folder)
    added_text = ''
    for t in (instance_info[1:5]):
        added_text += str(t) + '_'

    with open(
        os.path.join(
            folder, added_text+'.pkl'
        ), 'wb'
    ) as output:
        pickle.dump(plan, output, pickle.HIGHEST_PROTOCOL)


def get_robot_samples(control:central_control):
    '''
    
    '''
    pnp_rot = [-0.5877976611933862, 0.8089882123010008, 0.005586555396007588,0.0008787789389623685]
    num_samples = 4
    control.arms[0].move_to_joint(control.arms[0].arm_first_joints)
    control.arms[1].move_to_joint(control.arms[1].arm_first_joints)
    
    for _ in range(1000):
        control.stepSimulation()
    
    sampled_joints = {
        0:[], 1:[]
    }
    # input('press enter to sample')
    for arm_id in range(2):
        if arm_id == 0:
            # robot 0
            x_min = 0.5-(2-control.overlap_rate)*WORKSPACE_SIZE/2.0
            x_max = 0.5+(control.overlap_rate*WORKSPACE_SIZE/(1+control.overlap_rate))
        else: # robot 1
            x_min = 0.5-(control.overlap_rate*WORKSPACE_SIZE/(1+control.overlap_rate))
            x_max = 0.5+(2-control.overlap_rate)*WORKSPACE_SIZE/2.0
        
        y_min = -WORKSPACE_SIZE/2.0
        y_max = WORKSPACE_SIZE/2.0
        
        x_pts = np.linspace(x_min, x_max, num_samples)
        y_pts = np.linspace(y_min, y_max, num_samples)
        
        for x in x_pts:
            for y in y_pts:
                for z in [0.3,0.4]:
                    pos = [x, y, z]
                    j_state = control.arms[arm_id].solve_ik((pos, pnp_rot))
                    sampled_joints[arm_id].append(j_state)
                    
                    # control.arms[arm_id].move_to_joint(j_state)
                    
                    # for _ in range(1000):
                    #     control.stepSimulation()
                    # input('next sample')
                    
                    # control.arms[arm_id].move_to_joint(control.arms[arm_id].arm_first_joints)
                    
                    # for _ in range(1000):
                    #     control.stepSimulation()
                
    
    # print('sampled_joints', len(sampled_joints[0]), len(sampled_joints[1]))
    pickle.dump(sampled_joints, open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sampled_joints.pkl'), 
        'wb'))
    return sampled_joints
