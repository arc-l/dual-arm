import math
import sys
from tools.SCC_decomposition import SCC_decomposition, Graph_Decomposition, Topological_SCC_ordering
from tools.util import construct_DG

class high_level_solver(object):
    def __init__(self, Width, Height, numObjs, radius, start_arr, goal_arr, num_arms=2, overlap_rate=0.3) -> None:
        
        ''' Input
        Width, Height, numObjs, radius, start_arr, goal_arr, num_arms=2, 
        each pose: (region_id, x, y)
        '''
        # scale of each region
        self.Width = Width
        self.Height = Height
        self.numObjs = numObjs
        self.radius = radius
        self.start_arr = start_arr
        self.goal_arr = goal_arr
        self.num_arms = num_arms
        self.overlap_rate = overlap_rate
        self.handoff_pose = (self.Width/2, self.Height/2)
        self.rest_pose = {
            0: (-self.radius, Height/2.0),
            1: (Width+self.radius, Height/2.0)
        }
        self.construct_constant_dict()

        ''' Output
        action_list[arm_id]=[( obj_id, pose1, pose2)]
        '''
        self.action_list = {k:[] for k in range(num_arms) }

    def construct_constant_dict(self):
        v = 1.0
        Max_dist = math.sqrt((self.Width)**2+(self.Height)**2)
        self.cost_constants = {
            'transit_v': 10.0*v,
            'transfer_v': v,
            'from_to_buffer': 1.0 * Max_dist/v,
            'grasp': 1.0 * Max_dist/v,
            'release': 1.0 * Max_dist/v
        }

def high_level_search(solver, Width, Height, numObjs, radius, start_arr, goal_arr, num_arms=2):
    '''
    In the handoff scenario, decomposing SCC will not guarantee optimality
    '''
    DG = construct_DG(start_arr, goal_arr, radius)
    action_list = {k:[] for k in range(num_arms) }
    # SCC Decomposition
    Partition = SCC_decomposition(DG, DG.keys())
    SCC_ordering = Topological_SCC_ordering(DG, Partition)
    for SCC_index in SCC_ordering:
        SCC = Partition[SCC_index]
        SCC_list = list(SCC)
        current_start_arr = {k:v for k,v in start_arr.items() if k in SCC_list}
        current_goal_arr = {k:v for k,v in goal_arr.items() if k in SCC_list}
        current_solver = solver(Width, Height, len(current_start_arr), radius, current_start_arr, current_goal_arr, num_arms=2)
        current_solver.A_star()
        for k in action_list.keys():
            action_list[k].extend(current_solver.action_list[k])
            
    return action_list



