'''
test functions for pybullet simulation
'''

from motion_planning.utils import quaternion_representation_transform
from motion_planning.central_control import central_control
from motion_planning.plan_recorder import Arm_State
from motion_planning.utils import env_pose_to_sim_pose, Colors

from motion_planning.curobo_planning import Motion_Gen
from constants import EPSILON, real_time_control_folder, arm0_yml_path, dual_arm_yml_path
from motion_planning.plan_recorder import Plan_Recorder
from curobo.types.robot import JointState
from curobo.types.math import Pose
from typing import List
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo 
import torch
import pickle as pkl
import os
import time
import yaml

import sys
import numpy as np
sys.path.append('.')


def test_more_moves(control: central_control):
    '''
    test motion planning with curobo
    '''
    # pick and place orientation
    pnp_ori = control.arms[0].get_ee_qua_pose()[1]
    pnp_ori = list(pnp_ori)
    pnp_ori = [pnp_ori[3], pnp_ori[0], pnp_ori[1], pnp_ori[2]]  # xyzw -> wxyz
    # print('pnp_ori',pnp_ori)
    # exit()
    planner = Motion_Gen()

    # dual arm planning
    dual_start_state = JointState.from_position(
        torch.cat([
            torch.tensor(control.arms[i].get_joint_state().reshape((1, -1)),
                         dtype=torch.float32, device='cuda:0')
            for i in range(len(control.arms))
        ], dim=1)
    )
    link_poses_list = [
        {
            'tool0': Pose(
                (torch.tensor([[0.6, -0.2, 0.3]], device='cuda:0')),
                torch.tensor(pnp_ori, device='cuda:0').unsqueeze(0)
            ),
            'tool1': Pose(
                (torch.tensor([[0.4, 0.2, 0.3]], device='cuda:0')),
                torch.tensor(pnp_ori, device='cuda:0').unsqueeze(0)
            )
        },
        {
            'tool0': Pose(
                (torch.tensor([[0.1, -0.2, 0.8]], device='cuda:0')),
                torch.tensor(pnp_ori, device='cuda:0').unsqueeze(0)
            ),
            'tool1': Pose(
                # (torch.tensor([[0.5, -0.2, 0.3]], device='cuda:0')),
                (torch.tensor([[0.4, 0.2, 0.35]], device='cuda:0')),
                torch.tensor(pnp_ori, device='cuda:0').unsqueeze(0)
            )
        },
        {
            'tool0': Pose(
                # (torch.tensor([[0.5, -0.2, 0.3]], device='cuda:0')),
                (torch.tensor([[0.5, -0.2, 0.2]], device='cuda:0')),
                torch.tensor(pnp_ori, device='cuda:0').unsqueeze(0)
            ),
            'tool1': Pose(
                # (torch.tensor([[0.5, 0.2, 0.3]], device='cuda:0')),
                (torch.tensor([[0.4, 0.2, 0.3]], device='cuda:0')),
                torch.tensor(pnp_ori, device='cuda:0').unsqueeze(0)
            )
        },
        {
            'tool0': Pose(
                (torch.tensor([[0.4, -0.2, 0.1]], device='cuda:0')),
                torch.tensor(pnp_ori, device='cuda:0').unsqueeze(0)
            ),
            'tool1': Pose(
                # (torch.tensor([[0.6, -0.2, 0.3]], device='cuda:0')),
                (torch.tensor([[0.4, 0.2, 0.35]], device='cuda:0')),
                torch.tensor(pnp_ori, device='cuda:0').unsqueeze(0)
            )
        },
    ]

    input("start")
    for i in range(len(link_poses_list)):
        link_poses = link_poses_list[i]
        start = time.time()
        result = planner.motion_gen(
                                    planner.motion_gens[2], 
                                    dual_start_state, 
                                    link_poses['tool0'], 
                                    link_poses=link_poses
                                    )
        print("result.success=", result.success)
        traj = result.get_interpolated_plan()
        print(f"planning time: {time.time()-start}")
        for waypt_id in range(len(traj)):
            for arm_id in range(2):
                joint_state = traj[waypt_id][arm_id*6:(arm_id+1)*6]
                control.arms[arm_id].move_to_joint(joint_state)
            for _ in range(200):
                control.stepSimulation()
        input(f"end{i}")
        dual_start_state = JointState.from_position(
            torch.cat([
                torch.tensor(control.arms[i].get_joint_state().reshape((1, -1)),
                            dtype=torch.float32, device='cuda:0')
                for i in range(len(control.arms))
            ], dim=1)
        )

def dual_arm_motion_planning(control: central_control, planner, target:List[List], method, speed_scale, start_state, acc=None, horizon=1):
    pnp_ori = control.arms[0].get_ee_qua_pose()[1]
    pnp_ori = list(pnp_ori)
    pnp_ori = [pnp_ori[3], pnp_ori[0], pnp_ori[1], pnp_ori[2]]  # xyzw -> wxyz
    link_poses = {
        'tool0': Pose(
            (torch.tensor([target[0]], device='cuda:0')),
            torch.tensor(pnp_ori, device='cuda:0').unsqueeze(0)
        ),
        'tool1': Pose(
            (torch.tensor([target[1]], device='cuda:0')),
            torch.tensor(pnp_ori, device='cuda:0').unsqueeze(0)
        ),
    }
    curobo_pose0 = Pose(
            (torch.tensor([target[0]], device='cuda:0')),
            torch.tensor(pnp_ori, device='cuda:0').unsqueeze(0)
        )
    curobo_pose1 = Pose(
            (torch.tensor([target[1]], device='cuda:0')),
            torch.tensor(pnp_ori, device='cuda:0').unsqueeze(0)
        )
    if method == 'curobo':
        result = planner.motion_gen(
            planner.motion_gens[2], 
            start_state, 
            link_poses['tool0'],
            link_poses=link_poses 
            ) # dual arm result
        print("result.success=", result.success)
        traj1 = result.get_interpolated_plan() # single arm trajectory
        print(traj1)
    elif method == 'curobo_toppra':
        result = planner.motion_gen(
                                    planner.motion_gens[2], 
                                    start_state, 
                                    link_poses['tool0'],
                                    link_poses=link_poses
                                    ) # dual arm result
        traj1 = result.optimized_plan
        traj1 = planner.toppra_improvement(traj1, speed_scale=speed_scale)
    elif method == 'baseline':
        control.speed = speed_scale*3.14
        control.arms[0].speed = speed_scale*3.14
        control.arms[1].speed = speed_scale*3.14
        control.arms[0].move_to_joint(start_state)
        for _ in range(5):
            control.stepSimulation()
        ik_sol0 = planner.single_ik_solve(
            planner.ik_solvers[0], curobo_pose0, control.arms[0].arm_first_joints
        )
        ik_sol1 = planner.single_ik_solve(
            planner.ik_solvers[1], curobo_pose1, control.arms[1].arm_first_joints
        )
        control.arms[0].move_to_joint(ik_sol0)
        control.arms[1].move_to_joint(ik_sol1)
        traj1 = JointState.from_position(
            torch.tensor(start_state.position, dtype=torch.float32, device='cuda:0').view(1,-1)
        )
        ik_sol_np0 = ik_sol0.view(-1).cpu().numpy()
        curr_np0 = control.arms[0].get_joint_state()
        ik_sol_np1 = ik_sol1.view(-1).cpu().numpy() 
        curr_np1 = control.arms[1].get_joint_state()
        while np.linalg.norm(curr_np0 - ik_sol_np0) > EPSILON:
            for _ in range(horizon):
                control.stepSimulation()
            curr_np0 = control.arms[0].get_joint_state()
            traj1 = traj1.stack(
                JointState.from_position(
                    torch.tensor(torch.cat([curr_np0.reshape((1, -1)), curr_np1.reshape((1, -1))], dim=1),
                                dtype=torch.float32, device='cuda:0').view(1,-1))
            )
    return traj1

def test_curobo_dual_arm_motion_gen(control: central_control):
    '''
    test motion planning with curobo
    '''
    horizon = 1
    # speed_scale = [1.0, 0.5, 0.2, 0.1]
    # speed_scale = [0.5, 0.2, 0.1]
    speed_scale = [0.5]
    acc = 40
    methods = [
        'curobo', 
        'curobo_toppra', 
        'baseline'
        ]
    targets = [
        [0.5, 0.4, 0.35],
        [0.5, -0.4, 0.35],
    ]
    dual_start_state = JointState.from_position(
        torch.cat([
            torch.tensor(control.arms[i].get_joint_state().reshape((1, -1)),
                         dtype=torch.float32, device='cuda:0')
            for i in range(len(control.arms))
        ], dim=1)
    )
    for scale in speed_scale:
        with open(dual_arm_yml_path, 'r') as f:
            arm_config = yaml.safe_load(f)
        arm_config['robot_cfg']['kinematics']['cspace']['velocity_scale'] = scale
        with open(dual_arm_yml_path, 'w') as f:
            yaml.dump(arm_config, f)
        planner = Motion_Gen(curobo_dt=0.002*horizon)
        for method in methods:
            print(Colors.GREEN+ "generating traj for method {} speed {}".format(method, scale)+Colors.RESET)
            start = time.time()
            traj1 = dual_arm_motion_planning(control, planner=planner, target=targets, start_state=dual_start_state, method=method, speed_scale=scale, acc=acc, horizon=horizon)
            end = time.time()
            print(Colors.CYAN+ "Done generating traj for method {} speed {} in {} seconds".format(method, scale, end-start)+Colors.RESET)
            with open(os.path.join(real_time_control_folder,f"{method}_traj_{int(scale*10)}_overlap_rate={control.overlap_rate:.6f}.pkl"), 'wb') as f:
                pkl.dump(traj1, f)
            print(Colors.MAGENTA, "All saved traj length: ", traj1.shape, Colors.RESET)
            # for waypt_id in range(len(traj1)):
            #     for arm_id in range(2):
            #         joint_state = traj1[waypt_id][arm_id*6:(arm_id+1)*6]
            #         control.arms[arm_id].move_to_joint(joint_state)
            #     for _ in range(200):
            #         control.stepSimulation()

    exit()

def test_curobo_single_arm_motion_gen(control: central_control):
    '''
    test motion planning with curobo
    '''
    horizon = 1
    speed_scale = [1.0, 0.5, 0.2, 0.1]
    # speed_scale = [0.5, 0.2, 0.1]
    # speed_scale = [0.1]
    acc = 40
    methods = [
        'curobo', 
        'curobo_toppra', 
        'baseline'
        ]
    targets = [
        [0.5, 0.2, 0.35],
        [0.5, -0.2, 0.35],
        [0.3, 0.2, 0.35],
        [0.3, -0.2, 0.35],
        [0.4, 0, 0.35],
    ]
    single_start_state = JointState.from_position(
        torch.cat([torch.tensor(control.arms[0].get_joint_state().reshape((1, -1)),dtype=torch.float32, device='cuda:0')], dim=1))

    for scale in speed_scale:
        with open(arm0_yml_path, 'r') as f:
            arm_config = yaml.safe_load(f)
        arm_config['robot_cfg']['kinematics']['cspace']['velocity_scale'] = scale
        with open(arm0_yml_path, 'w') as f:
            yaml.dump(arm_config, f)
        planner = Motion_Gen(curobo_dt=0.002*horizon)
        for i in range(len(targets)): 
            for method in methods:
                print(Colors.GREEN+ "generating traj for method {} target {} speed {}".format(method, i, scale)+Colors.RESET)
                start = time.time()
                traj1 = single_arm_motion_planning(control, planner=planner, target=targets[i], start_state=single_start_state, method=method, speed_scale=scale, acc=acc, horizon=horizon)
                end = time.time()
                print(Colors.CYAN+ "Done generating traj for method {} target {} speed {} in {} seconds".format(method, i, scale, end-start)+Colors.RESET)
                ee_pos = []
                # for waypt_id in range(len(traj1)):
                #     joint_state = traj1[waypt_id].position[:6]
                #     ee_pos.append(control.arms[0].get_pos(joint_state))
                # with open(os.path.join(real_time_control_folder,f"{method}_traj{i}_{int(scale*10)}_overlap_rate={control.overlap_rate:.6f}_ee.pkl"), 'wb') as f:
                #     pkl.dump(ee_pos, f) 
                with open(os.path.join(real_time_control_folder,f"{method}_traj{i}_{int(scale*10)}_overlap_rate={control.overlap_rate:.6f}.pkl"), 'wb') as f:
                    pkl.dump(traj1, f)
                print(Colors.MAGENTA, "All saved traj length: ", traj1.shape, Colors.RESET)

    exit()



def single_arm_motion_planning(control: central_control, planner, target:List[List]|List[float], method, speed_scale, start_state, acc=None, horizon=1):
    pnp_ori = control.arms[0].get_ee_qua_pose()[1]
    pnp_ori = list(pnp_ori)
    pnp_ori = [pnp_ori[3], pnp_ori[0], pnp_ori[1], pnp_ori[2]]  # xyzw -> wxyz
    link_poses = {
        'tool0': Pose(
            (torch.tensor([target], device='cuda:0')),
            torch.tensor(pnp_ori, device='cuda:0').unsqueeze(0)
        ),
    }
    curobo_pose = Pose(
            (torch.tensor([target], device='cuda:0')),
            torch.tensor(pnp_ori, device='cuda:0').unsqueeze(0)
        )
    ik_sol = planner.single_ik_solve(
        planner.ik_solvers[0], curobo_pose, control.arms[0].arm_first_joints
    )
    if method == 'curobo':
        result = planner.motion_gen(
            planner.motion_gens[0], 
            start_state, 
            link_poses['tool0']
            ) # single result
        traj1 = result.get_interpolated_plan() # single arm trajectory
    elif method == 'curobo_toppra':
        result = planner.motion_gen(
                                    planner.motion_gens[0], 
                                    start_state, 
                                    link_poses['tool0']
                                    ) # single result
        traj1 = result.optimized_plan
        # traj1 = result.get_interpolated_plan() # single arm trajectory
        traj1 = planner.toppra_improvement(traj1, speed_scale=speed_scale)
    elif method == 'baseline':
        control.speed = speed_scale*3.14
        control.arms[0].speed = speed_scale*3.14
        control.arms[1].speed = speed_scale*3.14
        control.arms[0].move_to_joint(start_state)
        for _ in range(5):
            control.stepSimulation()
        control.arms[0].move_to_joint(ik_sol)
        traj1 = JointState.from_position(
            torch.tensor(start_state.position, dtype=torch.float32, device='cuda:0').view(1,-1)
        )
        ik_sol_np = ik_sol.view(-1).cpu().numpy()
        curr_np = control.arms[0].get_joint_state()
        while np.linalg.norm(curr_np - ik_sol_np) > EPSILON:
            for _ in range(horizon):
                control.stepSimulation()
            curr_np = control.arms[0].get_joint_state()
            traj1 = traj1.stack(
                JointState.from_position(
                    torch.tensor(curr_np,
                                dtype=torch.float32, device='cuda:0').view(1,-1))
            )
    return traj1


def execute_traj(control: central_control, dual_traj: JointState):
    control.arms[0].move_to_joint(dual_traj[0].position[:6].view(-1).cpu().numpy(),3.14)
    control.arms[1].move_to_joint(dual_traj[0].position[6:].view(-1).cpu().numpy(),3.14)
    for _ in range(1000):
        control.stepSimulation()
    input('execute to next subgoal')
    for waypt_id in range(dual_traj.position.shape[0]):
        for arm_id in range(2):
            joint_state = dual_traj[waypt_id][arm_id*6:(arm_id+1)*6]
            control.arms[arm_id].move_to_joint(joint_state)
        for _ in range(1):
            control.stepSimulation()
    input('finish')


def test_curobo_motion_gen(control: central_control):
    '''
    test motion planning with curobo
    '''
    # pick and place orientation
    pnp_ori = control.arms[0].get_ee_qua_pose()[1]
    pnp_ori = list(pnp_ori)
    pnp_ori = [pnp_ori[3], pnp_ori[0], pnp_ori[1], pnp_ori[2]]  # xyzw -> wxyz
    # print('pnp_ori',pnp_ori)
    # exit()
    planner = Motion_Gen()

    # dual arm planning
    dual_start_state = JointState.from_position(
        torch.cat([
            torch.tensor(control.arms[i].get_joint_state().reshape((1, -1)),
                         dtype=torch.float32, device='cuda:0')
            for i in range(len(control.arms))
        ], dim=1)
    )
    
    single_start_state = JointState.from_position(
        torch.cat([
            torch.tensor(control.arms[i].get_joint_state().reshape((1, -1)),
                         dtype=torch.float32, device='cuda:0')
            for i in range(1)
        ], dim=1)
    )
    
    link_poses = {
        'tool0': Pose(
            (torch.tensor([[0.5, 0.2, 0.35]], device='cuda:0')),
            torch.tensor(pnp_ori, device='cuda:0').unsqueeze(0)
        ),
        'tool1': Pose(
            (torch.tensor([[0.5, -0.2, 0.35]], device='cuda:0')),
            torch.tensor(pnp_ori, device='cuda:0').unsqueeze(0)
        )

    }

    import time

    control.arms[0].open_gripper()
    control.arms[1].close_gripper(0.3)

    for _ in range(50):
        control.stepSimulation()

    result = planner.motion_gen(planner.motion_gens[2], dual_start_state, link_poses['tool0'], link_poses=link_poses)
    
    result1 = planner.motion_gen(planner.motion_gens[0], single_start_state, link_poses['tool0']) # single result
    
    # planner.motion_gen and 
    traj = result.get_interpolated_plan()
    traj1 = result1.get_interpolated_plan() # single arm trajectory
    
    print("this is traj1", traj1)

    
    for waypt_id in range(len(traj)):
        for arm_id in range(2):
            joint_state = traj[waypt_id][arm_id*6:(arm_id+1)*6]
            control.arms[arm_id].move_to_joint(joint_state)
        for _ in range(50):
            control.stepSimulation()

    # import time

    # control.arms[0].open_gripper()
    # control.arms[1].close_gripper(0.3)

    # for _ in range(50):
    #     control.stepSimulation()

    # time.sleep(5)


    # swap pose
    dual_start_state = JointState.from_position(
        torch.cat([
            torch.tensor(control.arms[i].get_joint_state().reshape((1, -1)),
                         dtype=torch.float32, device='cuda:0')
            for i in range(len(control.arms))
        ], dim=1)
    )
    link_poses = {
        'tool0': Pose(
            (torch.tensor([[0.5, -0.2, 0.35]], device='cuda:0')),
            torch.tensor(pnp_ori, device='cuda:0').unsqueeze(0)
        ),
        'tool1': Pose(
            (torch.tensor([[0.5, 0.2, 0.35]], device='cuda:0')),
            torch.tensor(pnp_ori, device='cuda:0').unsqueeze(0)
        )

    }
    input('start swapping arms')
    result = planner.motion_gen(planner.motion_gens[2], dual_start_state, link_poses['tool0'], link_poses=link_poses)
    traj = result.get_interpolated_plan()
    print(traj.shape)
    for waypt_id in range(len(traj)):
        for arm_id in range(2):
            # joint_state = traj[waypt_id].position[arm_id*6:(arm_id+1)*6]
            joint_state = traj[waypt_id][arm_id*6:(arm_id+1)*6]
            control.arms[arm_id].move_to_joint(joint_state)
        for _ in range(50):
            control.stepSimulation()


def test_trajopt(control: central_control):
    horizon = 1
    curobo_dt = control.time_step*horizon
    
    robot_positions = np.array([
        control.arms[0].base_position,
        control.arms[1].base_position
    ])
    planner = Motion_Gen(robot_positions=robot_positions, curobo_dt=curobo_dt)
    
    curr_dual_state = JointState.from_position(
        torch.tensor([[-0.0145, -1.5752,  1.2707, -1.3978, -1.4125,  0.1762, -3.5021, -1.7091,
          1.8227, -1.6927, -1.5784,  3.0951]], device='cuda:0')
    )
    dual_state = JointState.from_position(
        torch.tensor([[-0.0053, -2.1693,  1.4608, -0.9105, -1.5127,  0.0646, -3.4651, -0.9411,
          0.8497, -1.4875, -1.5787,  3.1322]], device='cuda:0')
    )
    
    PLAN_SUCCESS = False
    while not PLAN_SUCCESS:
        result = planner.trajopt_solve(planner.motion_gens[2], curr_dual_state, dual_state)
        print("TrajOpt Solution for dual arms: ", result.success, result.status, result.solve_time)
        PLAN_SUCCESS = result.success
    execute_traj(control, result.get_interpolated_plan())
    # execute_traj(control, result.graph_plan[1])
    exit()
    dual_traj = result.optimized_plan
    for i in range(3):
        PLAN_SUCCESS = False
        while not PLAN_SUCCESS:
            result = planner.dual_traj_improvement(dual_traj)
            print(i+1, "th TrajOpt Solution for dual arms: ", result.success, result.solve_time)
            PLAN_SUCCESS = result.success

        # dual_traj = result.get_interpolated_plan()
        execute_traj(control,result.interpolated_solution.trim_trajectory(0, result.path_buffer_last_tstep[0]))
        dual_traj = result.raw_solution


def test_planning_speed(control: central_control):
    '''
    test whether the traveling speed is the same
    '''
    arm_id = 0
    horizon = 10
    curobo_dt = control.time_step*horizon
    
    robot_positions = np.array([
        control.arms[0].base_position,
        control.arms[1].base_position
    ])
    planner = Motion_Gen(robot_positions=robot_positions, curobo_dt=curobo_dt)
    pnp_rot = [-0.5877976611933862, 0.8089882123010008, 0.005586555396007588,0.0008787789389623685]
    waypoints = []
    for i in range(10):
        if i%2==0:
            waypoints.append([0.5, 0.2, 0.35])
        else:
            waypoints.append([0.5, -0.2, 0.35])
    control.arms[arm_id].move_to_joint(control.arms[arm_id].arm_first_joints)
    print(control.arms[arm_id].arm_first_joints)
    for _ in range(10000):
        control.stepSimulation() 
    curr_single_state = JointState.from_position(
        torch.tensor(control.arms[arm_id].arm_first_joints.reshape((1, -1)),
    ))
    '''
    Curobo
    '''
    curobo_total_duration = 0
    for pos in waypoints:
        curobo_pose = Pose(
            torch.tensor(pos, device='cuda:0').view(1, -1),
            torch.tensor(quaternion_representation_transform(pnp_rot), device='cuda:0').view(1, -1),
        )
        ik_sol = planner.single_ik_solve(
            planner.ik_solvers[arm_id], curobo_pose, curr_single_state.position
        )
        next_js = JointState.from_position(
            torch.tensor(ik_sol.view(1,-1), dtype=torch.float32, device='cuda:0').view(1,-1),
        )
        success = False
        while not success:
            result = planner.trajopt_solve(
                    planner.motion_gens[arm_id],
                    curr_single_state,
                    next_js
                )
            print("TrajOpt Solution for arm {}: ".format(arm_id), result.success, result.solve_time)
            success = result.success
        traj = result.get_interpolated_plan()
        curobo_duration = (traj.position.shape[0]-1)*curobo_dt
        traj = traj.position.cpu().numpy()
        
        time_steps, _ = traj.shape
        for i in range(time_steps):
            arm_states = dict()
            arm_states[arm_id] = Arm_State(
                tuple(),
                traj[i],
                None,
                'T',
                'G',
                np.zeros(6),
                np.zeros(6),
                np.zeros(6)
            )
            arm_states[1-arm_id] = Arm_State(
                tuple(),
                control.arms[1-arm_id].get_joint_state(),
                None,
                'T',
                'G',
                np.zeros(6),
                np.zeros(6),
                np.zeros(6)
            )
            control.recorder.add_new_step(arm_states)
        latest_dist = control.recorder.compute_plan_length(ord=2)[1]
        print('******************')
        curr_single_state = next_js
    # input('ready')
    curobo_plan_size = len(control.recorder.record[0])
    control.record_replay()
    curobo_time = control.execution_time
    curobo_length = control.recorder.compute_plan_length(ord=2)
    control.recorder.plot_plan(control.time_step)
    print('curobo execution time', curobo_time)
    print("curobo time", curobo_total_duration)
    print('curobo length', curobo_length)
    
    
    
    '''
    Toppra
    '''
    control.recorder = Plan_Recorder()
    control.reset_env()
    planner = Motion_Gen(robot_positions=robot_positions, curobo_dt=curobo_dt)
    control.arms[arm_id].move_to_joint(control.arms[arm_id].arm_first_joints)
    for _ in range(10000):
        control.stepSimulation()
    curr_single_state = JointState.from_position(
        torch.tensor(control.arms[arm_id].arm_first_joints.reshape((1, -1)),
    ))
    toppra_total_duration = 0
    for pos in waypoints:
        prev_dist = control.recorder.compute_plan_length(ord=2)[1]
        curobo_pose = Pose(
            torch.tensor(pos, device='cuda:0').view(1, -1),
            torch.tensor(quaternion_representation_transform(pnp_rot), device='cuda:0').view(1, -1),
        )
        ik_sol = planner.single_ik_solve(
            planner.ik_solvers[arm_id], curobo_pose, curr_single_state.position
        )
        next_js = JointState.from_position(
            torch.tensor(ik_sol.view(1,-1), dtype=torch.float32, device='cuda:0').view(1,-1),
        )
        success = False
        while not success:
            result = planner.trajopt_solve(
                    planner.motion_gens[arm_id],
                    curr_single_state,
                    next_js
                )
            print("TrajOpt Solution for arm {}: ".format(arm_id), result.success, result.solve_time)
            success = result.success
        traj = result.get_interpolated_plan()
        curobo_duration = (traj.position.shape[0]-1)*curobo_dt
        waypoints_ = traj.position.cpu().numpy()
        max_accelerations = np.array([[-40.0, 40.0]]*traj[0].shape[0])
        max_velocities = np.array([[-np.pi * 0.1, np.pi * 0.1]]*traj[0].shape[0])
        path = ta.SplineInterpolator(np.linspace(0, 1, traj.shape[0]), waypoints_)
        pc_vel = constraint.JointVelocityConstraint(max_velocities)
        pc_acc = constraint.JointAccelerationConstraint(max_accelerations)
        instance = algo.TOPPRA([pc_vel, pc_acc], path, solver_wrapper='seidel')
        jnt_traj = instance.compute_trajectory(0, 0) 
        toppra_duration = jnt_traj.duration
        traj = jnt_traj.eval(np.linspace(0, toppra_duration, int(toppra_duration/control.time_step/horizon)))
        
        toppra_total_duration += toppra_duration
        # traj = torch.tensor(traj, dtype=torch.float32, device='cuda')
        time_steps, _ = traj.shape
        for i in range(time_steps):
            arm_states = dict()
            arm_states[arm_id] = Arm_State(
                tuple(),
                traj[i],
                None,
                'T',
                'G',
                np.zeros(6),
                np.zeros(6),
                np.zeros(6)
            )
            arm_states[1-arm_id] = Arm_State(
                tuple(),
                control.arms[1-arm_id].get_joint_state(),
                None,
                'T',
                'G',
                np.zeros(6),
                np.zeros(6),
                np.zeros(6)
            )
            control.recorder.add_new_step(arm_states)
        latest_dist = control.recorder.compute_plan_length(ord=2)[1]
        print('******************')
        prev_dist = latest_dist
        curr_single_state = next_js
    # input('ready')
    toppra_plan_size = len(control.recorder.record[0])
    control.record_replay()
    toppra_time = control.execution_time
    toppra_length = control.recorder.compute_plan_length(ord=2)
    control.recorder.plot_plan(control.time_step)
    print('toppra execution time', toppra_time)
    print("Toppra time", toppra_total_duration)
    print('Toppra length', toppra_length)
    # print('opt dist', opt_dist)
    # exit()
    
    
    
    '''
    Baseline
    '''
    control.recorder = Plan_Recorder()
    control.reset_env()
    planner = Motion_Gen(robot_positions=robot_positions, curobo_dt=curobo_dt)
    control.arms[arm_id].move_to_joint(control.arms[arm_id].arm_first_joints)
    for _ in range(10000):
        control.stepSimulation()
    curr_single_state = JointState.from_position(
        torch.tensor(control.arms[arm_id].arm_first_joints.reshape((1, -1)),
                     dtype=torch.float32, device='cuda:0')
    )
    opt_dist = 0
    for pos in waypoints:
        curobo_pose = Pose(
            torch.tensor(pos, device='cuda:0').view(1, -1),
            torch.tensor(quaternion_representation_transform(pnp_rot), device='cuda:0').view(1, -1),
        )
        ik_sol = planner.single_ik_solve(
            planner.ik_solvers[arm_id], curobo_pose, curr_single_state.position
        )
        ready = False
        while not ready:
            for _ in range(horizon):
                ready = control.arms[arm_id].next_joint_state(ik_sol.view(-1).cpu().numpy(), control.arms[arm_id].speed)
                control.stepSimulation()
            control.recorder.add_new_step(control.update_arm_state())
        dist = np.linalg.norm(curr_single_state.position.cpu().numpy().reshape(-1)-ik_sol.view(-1).cpu().numpy())
        opt_dist += dist
        curr_single_state = JointState.from_position(ik_sol.view(1,-1))
    baseline_plan_size = len(control.recorder.record[0])
    control.recorder.plot_plan(control.time_step)
    # input('ready')
    control.record_replay()
    baseline_time = control.execution_time
    control.recorder.compute_distance()
    baseline_length = control.recorder.compute_plan_length(ord=2)
    print('curobo execution time', curobo_time)
    print('toppra execution time', toppra_time)
    print('baseline execution time', baseline_time)
    print('curobo plan num waypoints', curobo_plan_size)
    print('toppra plan num waypoints', toppra_plan_size)
    print('baseline plan num waypoints', baseline_plan_size)
    print('curobo traj l2 length', curobo_length)
    print('toppra traj l2 length', toppra_length)
    print('baseline traj l2 length', baseline_length)
    print('straight line dist', opt_dist)
    exit()
    return control.recorder.record, control.execution_time


def test_pick_n_place(control:central_control):
    ''''''



