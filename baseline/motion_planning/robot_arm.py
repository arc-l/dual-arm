import copy
import numpy as np
import sys
import os
import time
import math
import pybullet as pb
from typing import List

# from curobo.types.robot import JointState

from baseline.motion_planning.plan_recorder import Arm_State
from baseline.constants import EPSILON


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from baseline.tools.util import poly_disk
from baseline.motion_planning.utils import env_pose_to_sim_pose, Task, Action
from enum import Enum

class ControlMode(Enum):
    POSITION = 0
    VELOCITY = 1
    TORQUE = 2

class robot_arm(object):
    def __init__(
        self, arm_id, Width, Height, start_arr, goal_arr, radius, 
        task_list, pb_env, object_body_ids, position, overlap_rate,
        speed = 0.3141592653589793, pnp_steps = 2, control_mode=ControlMode.POSITION,
        time_step=1/240.0
        ) -> None:
        '''
        Robot behaviour control system
        Input: arm id in pybullet, Width, Height, start_arr, goal_arr, radius
        task_list: [(obj, start_mode, goal_mode, pose1, pose2, start_time, end_time)]
        speed: moving speed
        '''
        self.arm_id = arm_id
        self.start_arr = start_arr
        self.goal_arr = goal_arr
        self.Width = Width
        self.Height = Height
        self.radius = radius
        self.current_object = None
        self._pb = pb_env
        self.object_body_ids = object_body_ids
        self.base_position = position
        self.overlap_rate = overlap_rate
        self.speed = speed
        self.pnp_steps = pnp_steps
        self.control_mode = control_mode
        self.time_step = time_step
        self.clearance = 0.15

        self.task_list = task_list
        self.current_joint_goal = None
        self.ee_goal = None
        self.yielding = False
        self.status = 'T' # T: traveling, M: manipulation, I: idle
        self.detailed_status = None
        self.remaining_idle_time_steps = 0
        self.finished = False
        self.current_action = None
        
        self.stop = False # finished
        self.pause = False # handoff
        self.is_colliding = False # currently close to the other robot
        self.to_rest = False # current action is to the ready(first) pose, don't repeat
        # T:travel, A:activate, D:deactivate, I:idle (for recording)
        # updated in activate, deactivate, and the controller
        self.motion_status = 'T' 
        self.num_conflicts = 0 # counted with improving traj


        ''' arm properties '''
        (
            self.ur, self.ur_ee_id, self.ur_joints, 
            self.ee, self.ee_tip_id
            ) = self.setup_ur5e(self.base_position)
        # (
        #     self.ur, self.ur_ee_id, self.ur_joints, 
        #     self.ee, self.ee_tip_id
        #     ) = self.setup_ur10e(self.base_position)
        self.activated: bool = False
        self.contact_constraint: int = -1
        self.arm_home_joints = None
        self.arm_first_joints = None
        self.arm_ik_rest_joints = None
        self.arm_avoid_joints = None
        self.arm_yielding_joints = None
        self.arm_home_pos = None
        self.arm_first_pos = None
        self.arm_ik_rest_pos = None
        self.arm_avoid_pos = None
        self.arm_yielding_pos = None
        self.handoff_preparation_position = [0.5, 0.0, 0.7]
        self.handoff_position = [0.5, 0.0, 0.7]
        self.handoff_euler = None
        self.TheOtherArm = None
        self.ee_tip_offset = 0.11 
        self.buffer_height = 0.05 # 0.04

        self.task_list = [ t for t in self.task_list if t != None]
        self.action_list = []
        # print('arm', self.arm_id, "ready")
        

    def update(self):
        '''
        ready for stepSimulation
        1. High level modes:
            (1). T: traveling: 
                a. whether need to yield
                b. whether need to wait for clearance of obstacles
                c. next joint state
                d. if hit the key point
                    i. assign next task
                    ii. idle to be stable
            (2). M: vertical moves(grasping and releasing)
                a. next joint state
            (3). I: idle for a moment
                a. time step remains
                b. if finished
                    i. jump to next task
        '''
        if self.pause:
            return
        if self.finished:
            return
        if self.status == 'T': # travelling
            self.traveling_reasoning()
        elif self.status == 'M':
            self.manipulation_reasoning()
        else:
            self.idle_reasoning()

    def switch_status(self):
        '''change status, next goal '''
        if len(self.action_list) == 0:
            self.finished = True
            return
        self.current_action = self.action_list.pop(0)
        if self.to_rest:
            self.to_rest = False
        self.status = self.current_action.status
        self.current_object = self.current_action.bodyID
        self.current_joint_goal = self.current_action.joint_goal
        self.ee_goal = self.current_action.ee_pose
        self.current_goal = self.current_action.goal_position
        self.start_time = self.current_action.start_time
        self.end_time = self.current_action.end_time
        self.detailed_status = self.current_action.comment
        if self.status in ['T','M']:
            if type(self.current_joint_goal)!=np.ndarray:
                self.current_joint_goal = self.solve_ik(self.ee_goal)
        if self.status == 'activate':
            self.activate(self.current_object)
            self.freeze(self.pnp_steps) # activate vaccum
        elif self.status == 'deactivate':
            self.deactivate(self.current_object)
            self.freeze(self.pnp_steps) # deactivate vaccum
        elif self.status == 'handoff':
            self.pause = True
        return 

    def task_list_2_action_list(self):
        '''
        Input task_list = [(obj, start_mode, goal_mode, pose1, pose2, start_time, end_time)]
        Output action_list = [
            (status, bodyID, joint_goal, ee_pose, goal_position, start_time, end_time, 'pnp_transit'), 
            ('activate', bodyID, None, ee_pose, None, start_time, end_time,None), 
            ('deactivate', bodyID, None, ee_pose, None, start_time, end_time, None),
            ('handoff', bodyID, None, ee_pose, None, start_time, end_time, None),
            ('handoff', None, None, ee_pose, None, start_time, end_time, None),
        ]
        '''
        # pnp_rot = pb.getQuaternionFromEuler([np.pi / 2, np.pi / 2, np.pi / 2])
        # pnp_rot = [-0.5877976611933862, 0.8089882123010008, 0.005586555396007588,0.0008787789389623685]
        # pnp_rot= [7.07106781e-01, -7.07106781e-01, 4.32978028e-17, 4.32978028e-17]
        pnp_rot = [1.0, 0.0, 0.0, 0.0]
        handoff_rot = pb.getQuaternionFromEuler(self.handoff_euler)
        self.action_list = []
        self.action_list.append(Action('T', None, self.arm_first_joints, self.arm_first_pos, None, 0, 0, 'to_home',arm=self.arm_id))
        for task in self.task_list:
            obj, start_mode, goal_mode, pose1, pose2, start_time, goal_time = task
            obj_body_id = self.object_body_ids[obj]
            if pose1 != None:
                pose1 = self.manipulation_position(env_pose_to_sim_pose(pose1,self.overlap_rate))
            if pose2 != None:
                pose2 = self.manipulation_position(env_pose_to_sim_pose(pose2,self.overlap_rate))
            if (start_mode != 'H') and (goal_mode != 'H'): # pick and place
                # transit to the grasp pose
                ee_goal = (self.manipulation_ready_position(pose1), pnp_rot)
                self.action_list.append(Action('T', obj_body_id, None, ee_goal, pose1, start_time, goal_time, 'to_current',arm=self.arm_id))
                
                # go down
                ee_goal = ( pose1, pnp_rot)
                self.action_list.append(Action('M', obj_body_id,None, ee_goal, None, start_time, goal_time, 'down',arm=self.arm_id, prioritized=True))

                # activate
                self.action_list.append(Action('activate', obj_body_id, None,None, None, start_time, goal_time, 'activate',arm=self.arm_id, prioritized=True))

                # go up
                ee_goal = ( self.manipulation_ready_position(pose1), pnp_rot)
                self.action_list.append(Action('M', obj_body_id,None, ee_goal, None, start_time, goal_time, 'up',arm=self.arm_id, prioritized=True))

                # transfer to the placing pose
                ee_goal = ( self.manipulation_ready_position(pose2), pnp_rot)
                self.action_list.append(Action('T', obj_body_id,None, ee_goal, pose2, start_time, goal_time,'to_goal',arm=self.arm_id))

                # go down
                ee_goal = (pose2, pnp_rot)
                self.action_list.append(Action('M', obj_body_id,None, ee_goal, None, start_time, goal_time, 'down',arm=self.arm_id, prioritized=True))

                # deactivate
                self.action_list.append(Action('deactivate', obj_body_id,None, None, None, start_time, goal_time, 'deactivate',arm=self.arm_id, prioritized=True))

                # go up
                ee_goal = ( self.manipulation_ready_position(pose2), pnp_rot)
                self.action_list.append(Action('M', obj_body_id,None, ee_goal, None, start_time, goal_time, 'up',arm=self.arm_id, prioritized=True))
            elif start_mode == 'H': # handoff receive
                # handoff transit 
                ee_goal = ( self.handoff_preparation_position, handoff_rot)
                self.action_list.append(Action('T', obj_body_id,None, ee_goal, None, start_time, goal_time, 'to_handoff',arm=self.arm_id))

                # handoff
                self.action_list.append(Action('handoff', None,None, None, None, start_time, goal_time,'handoff receive',arm=self.arm_id, prioritized=True))

                # transfer to the placing pose
                ee_goal = ( self.manipulation_ready_position(pose2), pnp_rot)
                self.action_list.append(Action('T', obj_body_id,None, ee_goal, pose2, start_time, goal_time,'to_pose',arm=self.arm_id))

                # go down
                ee_goal = (pose2, pnp_rot)
                self.action_list.append(Action('M', obj_body_id,None, ee_goal, None, start_time, goal_time, 'down',arm=self.arm_id, prioritized=True))

                # deactivate
                self.action_list.append(Action('deactivate', obj_body_id, None,None, None, start_time, goal_time, 'deactivate', arm=self.arm_id, prioritized=True))

                # go up
                ee_goal = ( self.manipulation_ready_position(pose2), pnp_rot)
                self.action_list.append(Action('M', obj_body_id,None, ee_goal, None, start_time, goal_time, 'up',arm=self.arm_id, prioritized=True))
            else: # handoff delivery
                # transit 
                ee_goal = ( self.manipulation_ready_position(pose1), pnp_rot)
                self.action_list.append(Action('T', obj_body_id,None, ee_goal, pose1, start_time, goal_time,'to_pose',arm=self.arm_id))
                
                # go down
                ee_goal = ( pose1, pnp_rot)
                self.action_list.append(Action('M', obj_body_id,None, ee_goal, None, start_time, goal_time, 'down',arm=self.arm_id, prioritized=True))

                # activate
                self.action_list.append(Action('activate', obj_body_id, None,None, None, start_time, goal_time,'activate',arm=self.arm_id, prioritized=True))

                # go up
                ee_goal = ( self.manipulation_ready_position(pose1), pnp_rot)
                self.action_list.append(Action('M', obj_body_id,None, ee_goal, None, start_time, goal_time,'up',arm=self.arm_id, prioritized=True))

                # transfer to handoff
                ee_goal = ( self.handoff_preparation_position, handoff_rot)
                self.action_list.append(Action('T', obj_body_id,None, ee_goal, None, start_time, goal_time,'to_handoff',arm=self.arm_id))

                # handoff
                self.action_list.append(Action('handoff', obj_body_id,None, None, None, start_time, goal_time,'handoff delivery',arm=self.arm_id, prioritized=True))

        # transit 
        # self.action_list.append(('T', None, self.arm_first_joints,self.arm_first_pos, None, float('inf'), float('inf'),'to_home'))
        self.action_list.append(Action('T', None, self.arm_home_joints,self.arm_home_pos, None, float('inf'), float('inf'),'to_home',arm=self.arm_id))


    ''' traveling functions '''
    def traveling_reasoning(self):
        ''' traveling reasoning 
        a. whether need to yield
        b. whether need to wait for clearance of obstacles
        c. next joint state
        '''        
        if self.yielding:
            '''
            collision avoidance logic:
            1. whether it is too close
            2. if too close 
                move towards the home state. 
            3. when conflict is cleared
                move to the ready pose
            '''
            TooClose, target_joint_state = self.distance_detection()
            # need to yield to the other robot
            if TooClose:
                # if self.TheOtherArm.status == 'M': # manipulating, then stop
                #     return
                if not self.is_colliding: # just detected the risk
                    self.is_colliding = True
                    self.current_joint_goal = self.arm_first_joints
                    self.ee_goal = self.arm_first_pos
                    if not self.to_rest:
                        self.action_list.insert(0, self.current_action)
                        self.to_rest = True
                else: # detected risk already
                    self.current_joint_goal = self.arm_first_joints
                    self.ee_goal = self.arm_first_pos
                # self.next_joint_state(self.arm_avoid_joints,self.speed*5.0)
                self.next_joint_state(self.arm_yielding_joints,self.speed)
                # self.straight_next_joint_state(self.arm_yielding_joints,self.arm_yielding_pos, self.speed*5.0)
                return
        self.is_colliding = False
        # manipulation is available
        if self.manipulation_availability():
            ready = self.next_joint_state(self.current_joint_goal, self.speed)
            # ready = self.straight_next_joint_state(self.current_joint_goal, self.ee_goal, self.speed)
            if ready:
                self.freeze(0) # ready to grasp or release
            return
        else: # wait for clearance
            # if not self.to_rest:
            #     self.action_list.insert(0, self.current_action)
            #     self.to_rest = True
            self.next_joint_state( self.arm_yielding_joints, self.speed)
            # self.straight_next_joint_state(self.arm_first_joints, self.arm_first_pos, self.speed)
            # self.next_joint_state( self.arm_avoid_joints, self.speed)
            return 


    def distance_detection(self)->tuple:
        '''
        Output: 
        TooClose: boolean
        First joint Angle Increase: float
        '''
        closest_pair = None
        shortest_dist = float('inf')
        GroupA = [
            self.ur,self.ee
        ]
        GroupB = [
            self.TheOtherArm.ur,self.TheOtherArm.ee
        ]
        if self.activated:
            GroupA.append(self.current_object)
        if self.TheOtherArm.activated:
            GroupB.append(self.TheOtherArm.current_object)
        for bodyA in GroupA:
            for bodyB in GroupB:
                closest_points = self._pb.getClosestPoints(
                    bodyA, 
                    bodyB, 
                    self.clearance
                    )
                for pair in closest_points:
                    if pair[8] <= shortest_dist:
                        shortest_dist = pair[8]
                        closest_pair = pair

        current_ee_pose = self._pb.getLinkState(self.ee, 0)

        if closest_pair != None:
            obj_point = closest_pair[5]
            obs_point = closest_pair[6]
            direction = np.array(obj_point) - np.array(obs_point)
            norm_ = np.linalg.norm(direction)
            if norm_ > 0:
                direction = direction/norm_
            else:
                direction = np.array(self.base_position) - np.array(current_ee_pose[0])
                norm_ = np.linalg.norm(direction)
                direction = direction/norm_
            target_ee_position = np.array(current_ee_pose[0]) + 0.05*direction
            target_joint = self.solve_ik((target_ee_position, self._pb.getEulerFromQuaternion(current_ee_pose[1])))
            return True, target_joint
        else:
            return False, 0

    def manipulation_availability(self)->bool:
        ''' check whether the current task is doable so far '''
        # transfering, whether there is collision
        # goal mode is handoff
        if self.detailed_status == 'to_handoff':
            return True
        if self.detailed_status == 'to_home':
            return True
        if self.activated:
            for body_id in self.object_body_ids.values():
                if body_id == self.current_object:
                    continue
                if (body_id == self.TheOtherArm.current_object) and (self.TheOtherArm.activated) and (self.TheOtherArm.status == 'T'):
                    continue
                obj_pose = self._pb.getBasePositionAndOrientation(body_id)[0]
                if np.linalg.norm(
                    np.array(self.current_goal[:2])-np.array(obj_pose[:2])
                ) <= 1.8*self.radius:
                    return False
            else:
                return True
        else: # transit, whether there is an object there
                obj_pose = self._pb.getBasePositionAndOrientation(self.current_object)[0]
                if np.linalg.norm(
                    np.array(self.current_goal[:2])-np.array(obj_pose[:2])
                ) <= self.radius:
                    return True
                else:
                    return False


    ''' Manipulation reasoning '''
    def manipulation_reasoning(self):
        '''manipulation_reasoning '''
        ready = self.next_joint_state( self.current_joint_goal, self.speed)
        # ready = self.straight_next_joint_state(self.current_joint_goal, self.ee_goal, self.speed)
        if ready:
            self.freeze(0) # at the grasp or release pose

    ''' idle reasoning '''
    def idle_reasoning(self):
        ''' idle reasoning '''
        if self.remaining_idle_time_steps >0:
            self.remaining_idle_time_steps -= 1
        else:
            self.switch_status()

    def freeze(self, stop_steps):
        ''' wait a moment to stop '''
        self.status = 'I'
        self.remaining_idle_time_steps = stop_steps

    ''' sync with the world '''
    def update_priority(self, is_prioritized):
        ''' switch priority '''
        self.yielding = not is_prioritized
    

    ''' pybullet functions '''
    def next_joint_state(self, target_joints, speed):
        ''' update joint state 
        if hit the key point
            return True
        '''
        step_size = speed * self.time_step
        current_joints = np.array([
                self._pb.getJointState(self.ur, i)[0] for i in self.ur_joints
            ])
        current_joints = np.array(current_joints)
        target_joints = np.array(target_joints)
        diff_joints = target_joints - current_joints
        norm = np.linalg.norm(diff_joints,ord=2)
        if norm < EPSILON:
            return True
        
        v = diff_joints / norm if norm > 0 else 0
        if norm >= step_size:
            step_joints = current_joints + v * step_size
        else:
            step_joints = current_joints + v * norm
        for joint_id in range(len(self.ur_joints)):
            self._pb.setJointMotorControl2(
                bodyUniqueId=self.ur,
                jointIndex=self.ur_joints[joint_id],
                controlMode=pb.POSITION_CONTROL,
                targetPosition=step_joints[joint_id],
                positionGain=1,
                maxVelocity=self.speed,
            )
        return False

    def straight_next_joint_state(self, target_joint_state, target_goal_ee_pos,joint_speed, step=0.1):
        ''' straight line movement '''
        current_ee_pos = self.get_ee_qua_pose()
        # if np.linalg.norm(
        #         np.array(current_ee_pos[0])-np.array(target_goal_ee_pos[0])
        #     ) <= step:
        #     return self.next_joint_state(target_joint_state, joint_speed)
        current_ee_pos = np.array(
            list(current_ee_pos[0])+list(current_ee_pos[1])
        )
        target_ee_pos = np.array(
            list(target_goal_ee_pos[0])+list(target_goal_ee_pos[1])
        )
        diff_ee_pos = target_ee_pos - current_ee_pos
        norm = np.linalg.norm(diff_ee_pos[:3])
        
        v = diff_ee_pos[:3] / norm if norm > 0 else 0
        if norm >= step:
            step_pos = current_ee_pos.copy()
            step_pos[:3] = current_ee_pos[:3] + v * step
            step_pos[3:] = current_ee_pos[3:] + diff_ee_pos[3:] * step/norm
        else:
            step_pos = target_ee_pos
        step_pos = (step_pos[:3], step_pos[3:])
        step_joint_state = self.solve_ik(step_pos)
        return self.next_joint_state(step_joint_state, joint_speed)
        

    def manipulation_ready_position(self, pose):
        ''' get the position over the object '''
        return (
            pose[0], pose[1], pose[2]+self.ee_tip_offset+self.buffer_height*2
        )

    def manipulation_position(self, pose):
        ''' get the position over the object '''
        return (
            pose[0], pose[1], pose[2]+self.ee_tip_offset+self.buffer_height
        )

    def get_ee_qua_pose(self):
        return tuple(
            pb.getLinkState(
                self.ur, self.ur_ee_id
            )[:2]
        )

    def solve_ik(self, pose, is_handoff=False):
        """Calculate joint configuration with inverse kinematics."""
        if self.arm_id == 1:
            lowerLimits = [-4.5, -3.141, -3.141, -6.283, -6.283, -6.283]
            upperLimits = [-1.5, 6.283, 3.141, 6.283, 6.283, 6.283]
            jointRanges = [3, 9.424, 6.282, 12.566, 12.566, 12.566]
            # lowerLimits = [-6.283, -6.283, -3.141, -6.283, -6.283, -6.283]
            # upperLimits = [6.283, 6.283, 3.141, 6.283, 6.283, 6.283]
            # jointRanges = [12.566, 12.566, 6.282, 12.566, 12.566, 12.566]
        else:
            lowerLimits = [-1.5, -6.283, -3.141, -6.283, -6.283, -6.283]
            upperLimits = [1.5, 6.283, 3.141, 6.283, 6.283, 6.283]
            jointRanges = [3, 12.566, 6.282, 12.566, 12.566, 12.566]
        joints = self._pb.calculateInverseKinematics(
            bodyUniqueId=self.ur,
            endEffectorLinkIndex=self.ur_ee_id,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=lowerLimits,
            upperLimits=upperLimits,
            jointRanges=jointRanges,
            restPoses=np.float32(self.arm_first_joints).tolist(),
            maxNumIterations=100000,
            residualThreshold=1e-3,
        )
        joints = np.array(joints, dtype=np.float32)
        return joints

    def activate(self,obj_body_id=0):
        ''' add constraints to the attach the ee and a rigid object '''
        if not self.activated:
            body_pose = self._pb.getBasePositionAndOrientation(self.ee)
            obj_pose = self._pb.getBasePositionAndOrientation(obj_body_id)
            world_to_body = self._pb.invertTransform(body_pose[0], body_pose[1])
            obj_to_body = self._pb.multiplyTransforms(
                world_to_body[0], world_to_body[1], obj_pose[0], obj_pose[1]
            )
            self.contact_constraint = self._pb.createConstraint(
                        parentBodyUniqueId=self.ee,
                        parentLinkIndex=-1,
                        childBodyUniqueId=obj_body_id,
                        childLinkIndex=-1,
                        jointType=pb.JOINT_FIXED,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=obj_to_body[0],
                        parentFrameOrientation=obj_to_body[1],
                        childFramePosition=(0, 0, 0),
                        childFrameOrientation=(0, 0, 0, 1),
                    )
            self._pb.changeConstraint(self.contact_constraint, maxForce=500)
            self.activated = True
            self.motion_status = 'A'
            self.close_gripper(1-(2*self.radius)/8.5+0.2) # plus a small offset to make sure it is grasped
            

    def deactivate(self,obj_body_id):
        ''' remove constraints to the attach the ee and a rigid object '''
        if self.activated:
            self.activated = False
            if self.contact_constraint != -1:
                self._pb.removeConstraint(self.contact_constraint)
                self.contact_constraint = -1
            self.motion_status = 'D'
            self.open_gripper()

    ''' simulation '''
    def setup_ur5e(self, position):
        """Load ur5e in a given position"""
        current_folder = os.path.dirname(os.path.abspath(__file__))
        ur5e = self._pb.loadURDF(
            os.path.join(current_folder, "assets/ur5e/ur5e.urdf"), basePosition=position, useFixedBase=True
        )
        ur5e_joints = []
        for i in range(self._pb.getNumJoints(ur5e)):
            info = self._pb.getJointInfo(ur5e, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            # if joint_name == "ee_fixed_joint":
            if joint_name == "flange-tool0":
                ur5e_ee_id = joint_id
            if joint_type == pb.JOINT_REVOLUTE:
                ur5e_joints.append(joint_id)
        self._pb.enableJointForceTorqueSensor(ur5e, ur5e_ee_id, 1)
        (ee, ee_tip_id,) = self.setup_gripper(ur5e, ur5e_ee_id)

        return (
            ur5e,
            ur5e_ee_id,
            ur5e_joints,
            ee,
            ee_tip_id,
        )


    def setup_ur10e(self, position):
        """Load ur10e in a given position"""
        current_folder = os.path.dirname(os.path.abspath(__file__))
        ur = self._pb.loadURDF(
            os.path.join(current_folder, "assets/ur10e/ur10e.urdf"), basePosition=position, useFixedBase=True
        )
        ur_joints = []
        for i in range(self._pb.getNumJoints(ur)):
            info = self._pb.getJointInfo(ur, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "flange-tool0":
                ur_ee_id = joint_id
            if joint_type == pb.JOINT_REVOLUTE:
                ur_joints.append(joint_id)
        self._pb.enableJointForceTorqueSensor(ur, ur_ee_id, 1)
        (ee, ee_tip_id,) = self.setup_gripper(ur, ur_ee_id)

        return (
            ur,
            ur_ee_id,
            ur_joints,
            ee,
            ee_tip_id,
        )


    def open_gripper(self, is_slow=False):
        self._move_gripper(self.gripper_angle_open, is_slow=is_slow)

    def close_gripper(self, close_angle=0, is_slow=True):
        if close_angle<=0: # close angle must be greater than 0
            self._move_gripper(self.gripper_angle_close, is_slow=is_slow)
        self._move_gripper(close_angle, is_slow=is_slow)

    def _move_gripper(self, target_angle, timeout=3, is_slow=False):
        t0 = time.time()
        prev_angle = self._pb.getJointState(
            self.ee, self.gripper_main_joint
        )[0]

        if is_slow:
            self._pb.setJointMotorControl2(
                self.ee,
                self.gripper_main_joint,
                pb.VELOCITY_CONTROL,
                targetVelocity=1 if target_angle > 0.5 else -1,
                maxVelocity=1 if target_angle > 0.5 else -1,
                force=3,
                # physicsClientId=self._client_id,
            )
            self._pb.setJointMotorControl2(
                self.ee,
                self.gripper_mimic_joints["right_outer_knuckle_joint"],
                pb.VELOCITY_CONTROL,
                targetVelocity=1 if target_angle > 0.5 else -1,
                maxVelocity=1 if target_angle > 0.5 else -1,
                force=3,
                # physicsClientId=self._client_id,
            )
            for _ in range(10):
                self._pb.stepSimulation()
            while (time.time() - t0) < timeout:
                current_angle = self._pb.getJointState(self.ee, self.gripper_main_joint)[0]
                diff_angle = abs(current_angle - prev_angle)
                if diff_angle < 1e-4:
                    break
                prev_angle = current_angle
                for _ in range(10):
                    self._pb.stepSimulation()
        # maintain the angles
        self._pb.setJointMotorControl2(
            self.ee,
            self.gripper_main_joint,
            pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=3.1,
        )
        self._pb.setJointMotorControl2(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=3.1,
        )
        for _ in range(30):
            self._pb.stepSimulation()


    def setup_gripper(self, ur, ur_ee_id):
        ori = (0,0,0) # (0, -np.pi / 2, 0)
        ee_position, _ = ((0.487, 0.109, 0.438), self._pb.getQuaternionFromEuler(ori))
        current_folder = os.path.dirname(os.path.abspath(__file__))
        self.ee = self._pb.loadURDF(
            os.path.join(current_folder, "assets/ur5e/gripper/robotiq_2f_85.urdf"),
            ee_position,
            self._pb.getQuaternionFromEuler((0, -np.pi / 2, 0)),
        )
        self._pb.setCollisionFilterGroupMask(self.ee, -1, collisionFilterGroup=0, collisionFilterMask=0)
        self.ee_tip_offset = 0.1625
        self.gripper_angle_open = 0.00 # 0.03
        self.gripper_angle_close = 0.8
        self.gripper_angle_close_threshold = 0.73
        self.gripper_mimic_joints = {
            "left_inner_finger_joint": -1,
            "left_inner_knuckle_joint": -1,
            "right_outer_knuckle_joint": -1,
            "right_inner_finger_joint": -1,
            "right_inner_knuckle_joint": -1,
        }
        for i in range(self._pb.getNumJoints(self.ee)):
            info = self._pb.getJointInfo(self.ee, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "finger_joint":
                self.gripper_main_joint = joint_id
            elif joint_name == "dummy_center_fixed_joint":
                self.ee_tip_id = joint_id
            elif "finger_pad_joint" in joint_name:
                self._pb.changeDynamics(
                    self.ee, joint_id, lateralFriction=10
                )
                self.ee_finger_pad_id = joint_id
            elif joint_type == pb.JOINT_REVOLUTE:
                self.gripper_mimic_joints[joint_name] = joint_id
                # Keep the joints static
                self._pb.setJointMotorControl2(
                    self.ee, joint_id, pb.VELOCITY_CONTROL, targetVelocity=0, force=0,
                )
        self.ee_constraint = self._pb.createConstraint(
            parentBodyUniqueId=ur,
            parentLinkIndex=ur_ee_id,
            childBodyUniqueId=self.ee,
            childLinkIndex=-1,
            jointType=pb.JOINT_FIXED,
            jointAxis=(0, 0, 1),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.02),
            childFrameOrientation=pb.getQuaternionFromEuler((0, 0, 0)),
            # physicsClientId=self._client_id,
        )
        self._pb.changeConstraint(self.ee_constraint, maxForce=10000)
        self._pb.enableJointForceTorqueSensor(self.ee, self.gripper_main_joint, 1)

        # Set up mimic joints in robotiq gripper: left
        c = self._pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["left_inner_finger_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self._pb.changeConstraint(c, gearRatio=1, erp=0.8, maxForce=10000)
        c = self._pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["left_inner_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self._pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=10000)
        # Set up mimic joints in robotiq gripper: right
        c = self._pb.createConstraint(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee,
            self.gripper_mimic_joints["right_inner_finger_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self._pb.changeConstraint(c, gearRatio=1, erp=0.8, maxForce=10000)
        c = self._pb.createConstraint(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee,
            self.gripper_mimic_joints["right_inner_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self._pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=10000)
        # Set up mimic joints in robotiq gripper: connect left and right
        c = self._pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            # physicsClientId=self._client_id, # what is this physicsClientId?
        )
        self._pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=1000)
        self.open_gripper()
        return self.ee, self._pb.getJointInfo(self.ee, 0)

    def setup_gripper_old(self, ur, ur_ee_id):
        """Load end-effector: Suction, borrorwed from https://github.com/google-research/ravens/blob/master/ravens/tasks/grippers.py"""

        ori = (0,0,0) # (0, -np.pi / 2, 0)

        # Load suction gripper base model (visual only).
        pose = ((0.487, 0.109, 0.438), self._pb.getQuaternionFromEuler(ori))
        current_folder = os.path.dirname(os.path.abspath(__file__))
        base = self._pb.loadURDF(
            os.path.join(current_folder, "assets/ur5e/suction/suction-base.urdf"), pose[0], pose[1],
        )
        self._pb.createConstraint(
            parentBodyUniqueId=ur,
            parentLinkIndex=ur_ee_id,
            childBodyUniqueId=base,
            childLinkIndex=-1,
            jointType=pb.JOINT_FIXED,
            jointAxis=(0, 0, 1),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0.01),
            childFrameOrientation=pose[1],
        )

        # Load suction tip model (visual and collision) with compliance.
        pose = ((0.487, 0.109, 0.347), self._pb.getQuaternionFromEuler(ori))
        current_folder = os.path.dirname(os.path.abspath(__file__))
        ee = self._pb.loadURDF(os.path.join(current_folder, "assets/ur5e/suction/suction-head.urdf"), pose[0], pose[1],)
        constraint_id = self._pb.createConstraint(
            parentBodyUniqueId=ur,
            parentLinkIndex=ur_ee_id,
            childBodyUniqueId=ee,
            childLinkIndex=-1,
            jointType=pb.JOINT_FIXED,
            jointAxis=(0, 0, 1),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.08),
            childFrameOrientation=pose[1],
        )
        self._pb.changeConstraint(constraint_id, maxForce=50000)

        info = self._pb.getJointInfo(ee, 0)
        joint_id = info[0]
        joint_name = info[1].decode("utf-8")
        joint_type = info[2]
        ee_tip_id = joint_id
        return ee, ee_tip_id
    
    def set_joint_state(self, joint_state):
        print('##################')
        print('This function reset the joint state, should not be used in the middle of a simulation process')
        print("Please use move_to_joint instead")
        print('##################')
        for i in range(len(self.ur_joints)):
                self._pb.resetJointState(
                    self.ur,
                    self.ur_joints[i],
                    targetValue=joint_state[i],
                    targetVelocity=0,
                )
        
    def get_joint_state(self):
        ''' get current joint state '''
        return np.array([
                self._pb.getJointState(self.ur, i)[0] for i in self.ur_joints
            ])
    
    def move_to_joint(self, joint_state, speed = None):
        if speed is None:
            speed = self.speed
        # if type(joint_state) != JointState:
        #     joint_state = JointState.from_position(
        #         torch.tensor(joint_state, dtype=torch.float32, device='cuda:0').view(1,-1)
        #         )
        match self.control_mode:
            case ControlMode.POSITION:
                # self._pb.setJointMotorControlArray(
                #     bodyIndex=self.ur,
                #     jointIndices=self.ur_joints,
                #     controlMode=pb.POSITION_CONTROL,
                #     targetPositions=joint_state.position.view(-1),
                #     positionGains=np.ones(len(self.ur_joints)),
                # )
                for joint_id in range(len(self.ur_joints)):
                    self._pb.setJointMotorControl2(
                        bodyUniqueId=self.ur,
                        jointIndex=self.ur_joints[joint_id],
                        controlMode=pb.POSITION_CONTROL,
                        # targetPosition=joint_state.position.view(-1)[joint_id],
                        targetPosition=joint_state[joint_id],
                        positionGain=1,
                        maxVelocity=speed,
                    )
            case ControlMode.VELOCITY:
                self._pb.setJointMotorControlArray(
                    bodyIndex=self.ur,
                    jointIndices=self.ur_joints,
                    controlMode=pb.VELOCITY_CONTROL,
                    # targetPositions=joint_state.position.view(-1),
                    targetPosition=joint_state[joint_id],
                    targetVelocities=joint_state.velocity.view(-1),
                    positionGains=np.ones(len(self.ur_joints)),
                )
            case ControlMode.TORQUE:
                torque = self._pb.calculateInverseDynamics(self.ur, joint_state.position, joint_state.velocity, joint_state.acceleration)
                self._pb.setJointMotorControlArray(
                    bodyIndex=self.ur,
                    jointIndices=self.ur_joints,
                    controlMode=pb.TORQUE_CONTROL,
                    force=torque, 
                    positionGains=np.ones(len(self.ur_joints)),
                )

    
    def get_pos(self, joint_state):
        ''' joint state -> pose state '''
        self.move_to_joint(joint_state,3.14)
        for _ in range(500):
            self._pb.stepSimulation()
        return self.get_ee_qua_pose()


    ''' motion improvement '''
    def motion_improvement(self, record: List[Arm_State])->List[Arm_State]:
        ''' given a record of motions, improve some motions '''
        self.home_to_ready_traj = self.get_traj(
            self.arm_yielding_joints, self.arm_first_joints,self.speed
        )
        self.ready_to_home_traj = copy.deepcopy(self.home_to_ready_traj)
        self.ready_to_home_traj.reverse()
        new_record = []
        for traj in self.separate_trajs(record):
            new_record.extend(
                self.check_current_traj(traj)
            )
        assert len(new_record) == len(record)
        return new_record

    def get_traj(self, current_joints, target_joints, speed)->List[Arm_State]:
        ''' compute spent time to move the joints and get traj to do so '''
        joint_state_list = [current_joints]
        while 1:
            diff_joints = target_joints - current_joints
            norm = np.linalg.norm(diff_joints)
            if norm < EPSILON:
                break
            
            v = diff_joints / norm if norm > 0 else 0
            if norm >= speed:
                step_joints = current_joints + v * speed
            else:
                step_joints = current_joints + v * norm
            joint_state_list.append(step_joints)
            current_joints = step_joints
        traj = []
        for joint in joint_state_list:
            while 1:
                finish = self.next_joint_state(joint, speed)
                self._pb.stepSimulation()
                if finish:
                    break
            for _ in range(5):
                self._pb.stepSimulation()
            traj.append(
                Arm_State(
                    self.get_ee_qua_pose(), joint, None, 'T', 'G'
                )
            )
        return traj


    def get_straight_traj(self, current_joints, target_joints, speed)->List[Arm_State]:
        # straight joint state movement
        while 1:
            finish = self.next_joint_state(target_joints, speed)
            self._pb.stepSimulation()
            if finish:
                break
        target_pos = self.get_ee_qua_pose()
        while 1:
            finish = self.next_joint_state(current_joints, speed)
            self._pb.stepSimulation()
            if finish:
                break
        current_pos = self.get_ee_qua_pose()
        traj = [
            Arm_State(
                current_pos, current_joints, None, 'T', 'G'
            )
        ]
        Finish = False
        while not Finish:
            Finish = self.straight_next_joint_state(target_joints, target_pos, speed)
            self._pb.stepSimulation()
            traj.append(
                Arm_State(
                    self.get_ee_qua_pose(), self.get_joint_state(), None, 'T', 'G'
                )
            )
        return traj
        


    def get_traj_with_certain_length(self, current_joints, target_joints, length:int)->List[Arm_State]:
        ''' compute spent time to move the joints and get traj to do so '''
        while 1:
            finish = self.next_joint_state(target_joints, self.speed)
            self._pb.stepSimulation()
            if finish:
                break
        while 1:
            finish = self.next_joint_state(current_joints, self.speed)
            self._pb.stepSimulation()
            if finish:
                break
        current_pos = self.get_ee_qua_pose()
        target_pos = np.array(list(target_pos[0])+list(target_pos[1]))
        current_pos = np.array(list(current_pos[0])+list(current_pos[1]))
        diff_pos = target_pos - current_pos
        step_pos = diff_pos/(length-1)
        pos_list = [
            current_pos + i * step_pos
            for i in range(length)
        ]
        traj = []
        for pos in pos_list:
            pos = (pos[:3], pos[3:])
            joint = self.solve_ik(pos)
            self.set_joint_state(joint)
            for _ in range(50):
                self._pb.stepSimulation()
            traj.append(
                Arm_State(
                    pos, joint, None, 'T', 'G'
                )
            )
        return traj

    def get_straight_traj_with_certain_length(self, current_joints, target_joints, length:int)->List[Arm_State]:
        ''' compute spent time to move the joints and get traj to do so '''
        while 1:
            finish = self.next_joint_state(target_joints, self.speed)
            self._pb.stepSimulation()
            if finish:
                break
        while 1:
            finish = self.next_joint_state(current_joints, self.speed)
            self._pb.stepSimulation()
            if finish:
                break
        current_pos = self.get_ee_qua_pose()
        target_pos = np.array(list(target_pos[0])+list(target_pos[1]))
        current_pos = np.array(list(current_pos[0])+list(current_pos[1]))
        diff_pos = target_pos - current_pos
        step_pos = diff_pos/(length-1)
        pos_list = [
            current_pos + i * step_pos
            for i in range(length)
        ]
        traj = []
        for pos in pos_list:
            pos = (pos[:3], pos[3:])
            joint = self.solve_ik(pos)
            self.set_joint_state(joint)
            for _ in range(50):
                self._pb.stepSimulation()
            traj.append(
                Arm_State(
                    pos, joint, None, 'T', 'G'
                )
            )
        return traj


    def separate_trajs(self,record:List[Arm_State]):
        ''' separate trajs based on motion status ''' 
        start_index = 0
        end_index = 0
        for i, state in enumerate(record):
            if state.mode == 'A':
                self.num_conflicts += 1
            if state.motion_status in ['A', 'D']: # manipulation
                yield record[start_index:end_index+1]
                start_index = end_index + 1
                end_index += 1
            else:
                end_index += 1
        yield record[start_index:end_index+1]

    
    def check_current_traj(self, traj:List[Arm_State])->List[Arm_State]:
        ''' detect the segment that needs to modify '''
        # start_improve = self.get_first_avoid_end(traj)
        end_improve = self.get_last_approach_goal(traj)
        # no avoid or only one avoid
        if (end_improve==-1): # no avoid
            return traj
        else:
            start_improve = self.get_the_first_place_to_go_home(traj)
            new_traj = self.improve_traj(traj, start_improve, end_improve)
        return new_traj

    def get_the_first_place_to_go_home(self, traj:List[Arm_State])->int:
        ''' 
        get the first place that finish the last manipulation
        assuming it needs to avoid
        '''
        go_home_index = 0
        for index, state in enumerate(traj):
            if state.motion_status == 'T':
                return go_home_index
            else:
                go_home_index += 1

    def get_first_avoid_end(self, traj:List[Arm_State])->int:
        ''' get the fist time in the traj that finish avoidance '''
        avoiding = False
        avoid_index = -1
        for i, state in enumerate(traj):
            if avoiding and state.mode == 'A': # avoiding
                avoid_index = i
            elif avoiding : # end avoiding
                break
            elif state.mode == 'A': # start avoiding
                avoiding = True
            else: # haven't met the first avoid
                pass
        return avoid_index


    def get_last_approach_goal(self, traj:List[Arm_State])->int:
        ''' get the last attempt to approach the goal pose '''
        avoid_index = -2
        for i, state in enumerate(traj):
            if state.mode == 'A':
                avoid_index = i
        return avoid_index + 1 
        

    def improve_traj(self, traj:List[Arm_State], start_index, end_index)->List[Arm_State]:
        ''' 
        improve a certain traj, remove back and forth 
        get start joint state
        compute go start-ready-home path
        if not enough time, speed up to make sure start-ready exist
        '''
        start_joint_state = traj[start_index].joint_state
        start_to_ready = self.get_traj(
            start_joint_state, self.arm_first_joints, self.speed
        )
        improve_length = end_index - start_index
        cost_from_ready_to_home = len(self.ready_to_home_traj)-2
        if improve_length >= 2*(len(self.ready_to_home_traj)-2)+len(start_to_ready):
            new_traj =  traj[:start_index] + start_to_ready +self.ready_to_home_traj[1:-1]+[
                Arm_State( self.arm_yielding_pos, self.arm_yielding_joints, None, 'T', 'A')
                for _ in range(- start_index - len(start_to_ready) - 2*cost_from_ready_to_home + end_index)
            ] + self.home_to_ready_traj[1:-1] + traj[end_index:]
        elif improve_length >= len(start_to_ready):
            shrink_steps = (improve_length-len(start_to_ready)) //2
            stay_steps = (improve_length-len(start_to_ready)) % 2
            new_traj = traj[:start_index]+start_to_ready+self.ready_to_home_traj[1:(shrink_steps+1)]+[
                Arm_State(self.ready_to_home_traj[shrink_steps+1].pose, self.ready_to_home_traj[shrink_steps+1].joint_state,None, 'T','A')
                for _ in range(stay_steps)
            ] + self.home_to_ready_traj[-(shrink_steps+1):-1] + traj[end_index:]
        else:
            faster_go_back = self.get_traj_with_certain_length(
                start_joint_state, 
                self.arm_first_joints, 
                improve_length
                )
            new_traj = traj[:start_index]+faster_go_back+traj[end_index:]
        assert len(new_traj) == len(traj)
        return new_traj
        
    
    
                    

