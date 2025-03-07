import math
import numpy as np
import pybullet as p
from constants import WORKSPACE_SIZE, SIM_CENTER


class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"  # Resets the color to default


def env_pose_to_sim_pose( pose, overlap_rate):
        '''
        from 1000*1000 planning env to current env
        Input: (x,y)
        Output: (x,y,z)
        '''
        ws_center = SIM_CENTER
        size = WORKSPACE_SIZE
        return (
            pose[0]/1000*size+(ws_center[0]-size/(1+overlap_rate)),
            pose[1]/1000*size+(ws_center[1]-size/2),
            # 0.023000000044703484
            0.053
        )
        

def sim_pose_to_env_pose( pose, overlap_rate):
    '''
    from current env to 1000*1000 planning env
    2d/3d->2d(x,y)
    '''
    ws_center = SIM_CENTER
    size = WORKSPACE_SIZE
    return (
        (pose[0]-(ws_center[0]-size/(1+overlap_rate)))/size*1000,
        (pose[1]-(ws_center[1]-size/2))/size*1000
        
    )
    

class Task:
    def __init__(self, obj=0, start_mode='P', goal_mode='P', pose1=(0,0), pose2=(0,0), start_time=0, end_time=0):
        self.obj = obj
        self.start_mode = start_mode
        self.goal_mode = goal_mode
        self.pose1 = pose1
        self.pose2 = pose2
        self.start_time = start_time
        self.end_time = end_time

    def __repr__(self):
        return f"Task(obj:{self.obj}, start_mode:{self.start_mode}, goal_mode:{self.goal_mode}, pose1:{self.pose1}, pose2:{self.pose2}, start_time:{self.start_time}, end_time:{self.end_time})\n"

    def __get_item__(self, index):
        return [self.obj, self.start_mode, self.goal_mode, self.pose1, self.pose2, self.start_time, self.end_time][index]
            

    def __set_item__(self, index, value):
        if index == 0:
            self.obj = value
        elif index == 1:
            self.start_mode = value
        elif index == 2:
            self.goal_mode = value
        elif index == 3:
            self.pose1 = value
        elif index == 4:
            self.pose2 = value
        elif index == 5:
            self.start_time = value
        elif index == 6:
            self.end_time = value
        else:
            raise ValueError("index out of range")

class Action:
    def __init__(
        self, 
        status, # activate, deactivate, handoff
        bodyID=None, # body ID in hand 
        joint_goal=None, # if IK is computed or no need IK
        ee_pose=None, 
        goal_position=None, # (x,y)
        start_time=0, 
        end_time=0, 
        comment='pnp_transit',
        arm=0,
        prioritized=False # if this action should be executed immediately after the previous action
        ):
        self.status = status
        self.bodyID = bodyID
        self.joint_goal = joint_goal
        self.ee_pose = ee_pose
        self.goal_position = goal_position
        self.start_time = start_time
        self.end_time = end_time
        self.comment = comment
        self.arm = arm
        self.prioritized = prioritized
        
    def __repr__(self):
        return f"Action(status:{self.status}, bodyID:{self.bodyID}, joint_goal:{self.joint_goal}, ee_pose:{self.ee_pose}, goal_position:{self.goal_position}, start_time:{self.start_time}, end_time:{self.end_time}, comment:{self.comment}, arm:{self.arm})\n"
        
    def __get_index__(self,index):
        return [self.status, self.bodyID, self.joint_goal, self.ee_pose, self.goal_position, self.start_time, self.end_time, self.comment, self.arm][index]
    
    def __set_index__(self,index,value):
        if index == 0:
            self.status = value
        elif index == 1:
            self.bodyID = value
        elif index == 2:
            self.joint_goal = value
        elif index == 3:
            self.ee_pose = value
        elif index == 4:
            self.goal_position = value
        elif index == 5:
            self.start_time = value
        elif index == 6:
            self.end_time = value
        elif index == 7:
            self.comment = value
        elif index == 8:
            self.arm = value
        else:
            raise ValueError("index out of range")
        
def quaternion_representation_transform(quaternion):
    '''
    transform quaternion from xyzw to wxyz
    '''
    return [quaternion[3],quaternion[0],quaternion[1],quaternion[2]]

