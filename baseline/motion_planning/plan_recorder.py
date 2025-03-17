import pickle as pickle
import os
from typing import Dict,List,Any
from dataclasses import dataclass
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import datetime
import copy
from copy import deepcopy

import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo 
from tools.util import Colors

@dataclass
class Arm_State:
    pose:tuple
    joint_state:np.ndarray
    involved_object:Any
    motion_status:str = 'T' # T:travel, M:pick or place, A:activate, D:deactivate
    mode:str = 'G' # A:avoid, G: to a goal joint state
    velocity:np.ndarray=np.zeros(6)
    acceleration:np.ndarray=np.zeros(6)
    jerk:np.ndarray=np.zeros(6)
    


class Plan_Recorder(object):
    def __init__(self) -> None:
        self.record = {
            0:[],
            1:[]
            }

    def write_record(self):
        ''' write record '''
        with open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'recorded_plan.pkl'
                ),'wb'
            ) as output:
            pickle.dump(self.record, output, pickle.HIGHEST_PROTOCOL)


    def read_record(self):
        ''' read record '''
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'recorded_plan.pkl'
            ), 'rb'
        ) as input_:
            self.record = pickle.load(input_)

    def add_new_step(self, arm_states:Dict[int,Arm_State]):
        ''' add new joint states for both arms '''
        for armID in self.record.keys():
            assert type(arm_states[armID]) == Arm_State
            assert type(arm_states[armID].joint_state) == ndarray
            self.record[armID].append(arm_states[armID])
    
    def check_correctness_of_record(self):
        for armID in self.record.keys():
            for i, state in enumerate(self.record[armID]):
                assert type(state) == Arm_State
                assert type(state.joint_state) == np.ndarray
            

    def compute_plan_length(self, ord=2):
        ''' compute the length of the plan '''
        total_length = {arm_id:0 for arm_id in self.record.keys()}
        if len(self.record[0]) == 0:
            return total_length
        current_joint_state = {arm_id:self.record[arm_id][0].joint_state for arm_id in self.record.keys()}
        for armID in self.record.keys():
            for current_state in self.record[armID]:
                dist = np.linalg.norm(current_state.joint_state-current_joint_state[armID], ord=ord)
                total_length[armID] += dist
                current_joint_state[armID] = current_state.joint_state
        return total_length


    def plot_plan(self,dt):
        ''' plot the plan '''
        timestr = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')
        dims = 2*self.record[0][0].joint_state.shape[0]
        _, axs = plt.subplots(dims, 1)
        timesteps = [i * dt for i in range(len(self.record[0]))]
        for arm_id in range(2):
            for joint_id in range(6):
                Y = []
                for state in self.record[arm_id]:
                    Y.append(state.joint_state[joint_id])
                axs[6*arm_id + joint_id].plot(timesteps, Y)
        
        plt.legend()
        if not os.path.exists(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'figures'
        )):
            os.makedirs(os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'figures'
            ))
        plt.savefig(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'figures',
            'sim_exec_'+timestr+'.png'
        ))
        # plt.show()
        
        
    def toppra_improvement(self, acc_scale=1.0, speed_scale=1.0, dt=0.002):
        ''' 
        use toppra to improve the recorded trajectory 
        still under construction
        dt: desired output dt
        '''
        
        # indicate if the robot is in the activation or deactivation mode
        # When robot.motion_status='I' sometimes it is picking/placing (the states following states of 'A' or 'D' motion status), 
        # sometimes it is waiting or finished
        # We cannot improve the trajectory when the robot is in the activation or deactivation mode 
        pnp = [False, False]
        
        
        new_record = {arm_id:[] for arm_id in self.record.keys()}
        # the current trajectory pending for improvement
        for curr_traj in self.dual_traj_decoupling():
            for arm_id in [0, 1]:
                if curr_traj[arm_id][0].motion_status in ['A', 'D']:
                    pnp[arm_id] = True
                elif curr_traj[arm_id][0].motion_status in ['T', 'M']:
                    # next "I" state is not in the activation or deactivation mode
                    pnp[arm_id] = False
            if pnp[0] or pnp[1] or (len(curr_traj[0])<=2):
                new_record[0].extend(curr_traj[0])
                new_record[1].extend(curr_traj[1])
            else: # eligible for improvement
                waypoints_ = np.concatenate(
                    [
                        np.concatenate([curr_traj[0][i].joint_state, curr_traj[1][i].joint_state]).reshape(1, -1)
                        for i in range(len(curr_traj[0]))
                        ], axis=0
                )
                path = ta.SplineInterpolator(np.linspace(0, 1, waypoints_.shape[0]), waypoints_)
                # constraints for toppra traj
                max_accelerations = np.array([[-40.0*acc_scale, 40.0*acc_scale]]*waypoints_.shape[1])
                max_velocities = np.array([[-np.pi * speed_scale, np.pi * speed_scale]]*waypoints_.shape[1])
                pc_vel = constraint.JointVelocityConstraint(max_velocities)
                pc_acc = constraint.JointAccelerationConstraint(max_accelerations)
                instance = algo.TOPPRA([pc_vel, pc_acc], path, solver_wrapper='seidel')
                jnt_traj = instance.compute_trajectory(0, 0) 
                if jnt_traj is None:
                    print("Toppra failed")
                    new_record[0].extend(curr_traj[0])
                    new_record[1].extend(curr_traj[1])
                    continue
                toppra_duration = jnt_traj.duration
                traj = jnt_traj.eval(np.linspace(0, toppra_duration, int(toppra_duration/dt)))
                r0_traj = [copy.deepcopy(curr_traj[0][0]) for i in range(traj.shape[0])]
                r1_traj = [copy.deepcopy(curr_traj[1][0]) for i in range(traj.shape[0])]
                for i in range(traj.shape[0]):
                    r0_traj[i].joint_state = traj[i][:6]
                    r1_traj[i].joint_state = traj[i][6:]
                new_record[0].extend(r0_traj)
                new_record[1].extend(r1_traj)
        return new_record
                    

    def dual_traj_decoupling(self):
        '''
        decouple traj to sub-trajs where each sub-traj has the same motion status for each arm
        '''
        current_motion_status = [None, None]
        current_sub_traj = {0:[], 1:[]}
        for step in range(len(self.record[0])):
            if (self.record[0][step].motion_status != current_motion_status[0]) or (self.record[1][step].motion_status != current_motion_status[1]):
                # motion status changes
                if current_sub_traj[0] != []:
                    # generate new traj
                    yield current_sub_traj
                    # reset the current sub-traj
                    current_sub_traj = {0:[self.record[0][step]], 1:[self.record[1][step]]}
                else: # the first step
                    for arm_id in self.record.keys():
                        current_sub_traj[arm_id].append(self.record[arm_id][step])
                # update the current motion status
                current_motion_status = [self.record[0][step].motion_status, self.record[1][step].motion_status]
            else:
                # add the current step to the current sub-traj
                for arm_id in self.record.keys():
                    current_sub_traj[arm_id].append(self.record[arm_id][step])
        if current_sub_traj[0] != []:
            yield current_sub_traj

    
    
if __name__ == '__main__':
    plan_recorder = Plan_Recorder()
    plan_recorder.read_record()
    plan_recorder.plot_plan(1/240.0)