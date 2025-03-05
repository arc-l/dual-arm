from math import sqrt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from primitive_plan.Labeled_DFS_DP import DFS_DP_Search
# from primitive_plan.FVS_DP_Solver import FVS_Plan_Search
# from primitive_plan.random_order import random_order
# from buffer_allocation.Buffer_Generation import Buffer_Generation
from buffer_allocation.Buffer_Sampling import Buffer_Sampling
from buffer_allocation.Robust_Buffer_Sampling import Buffer_Sampling as Robust_Buffer_Sampling
from tools.util import deepcopy_dict
# from solvers.A_star_solver import A_star_solver
from solvers.A_star_solver_v2 import A_star_solver as A_star_solver
from solvers.Single_Arm_A_star_solver import Single_Arm_A_star_solver
from solvers.A_star_Dist_Opt_solver import A_star_solver as KC_A_star_solver
from solvers.Single_Arm_A_star_Dist_Opt_solver import Single_Arm_A_star_solver as KC_Single_Arm_A_star_solver
from solvers.MC_greedy import Greedy_Solver as MC_Greedy
from solvers.Split_Single import Split_Single as Split_Single

class Labeled_Partial_Optimizer(object):
    '''
    Input: sampling: use sampling based buffer generation or not 
    '''
    def __init__(self, start_arr, goal_arr, Height, Width, radius, sampling = True, primitive_plan = 'RB', robust_sampling = False, overlap_rate = 0.3, num_arms = 2, makespan='MC'):
        self.Height = Height
        self.Width = Width
        self.radius = radius
        self.sampling = sampling
        self.primitive_plan = primitive_plan
        self.robust_sampling = robust_sampling
        self.overlap_rate = overlap_rate

        self.new_arr = {} # resulting arrangement
        self.transitions = []
        self.isfeasible = False
        self.num_collision = 0
        self.max_node = 0
        self.total_node = 0
        
        epsilon = 1e-4
        real_start_arr = {}
        real_goal_arr = {}

        for obj in start_arr.keys():
            # check whether they are at the goal
            if (sqrt((start_arr[obj][0]-goal_arr[obj][0])**2+
                (start_arr[obj][1]-goal_arr[obj][1])**2) >= epsilon):
                real_start_arr[obj] = start_arr[obj]
                real_goal_arr[obj] = goal_arr[obj]
        
        if num_arms == 2:
            if makespan == 'MC':
                planner = A_star_solver(
                    self.Width, self.Height, len(real_start_arr), self.radius, 
                    real_start_arr, real_goal_arr, num_arms=2, overlap_rate=self.overlap_rate
                    )
            elif makespan == 'KC':
                planner = KC_A_star_solver(
                    self.Width, self.Height, len(real_start_arr), self.radius, 
                    real_start_arr, real_goal_arr, num_arms=2, overlap_rate=self.overlap_rate
                    )
            elif makespan == 'GD':
                planner = MC_Greedy(
                    self.Width, self.Height, len(real_start_arr), self.radius, 
                    real_start_arr, real_goal_arr, num_arms=2, overlap_rate=self.overlap_rate
                    )
            elif makespan == 'Split':
                planner = Split_Single(
                    self.Width, self.Height, len(real_start_arr), self.radius, 
                    real_start_arr, real_goal_arr, num_arms=2, overlap_rate=self.overlap_rate
                    )
            else:
                raise ValueError('wrong makespan method')

        elif num_arms == 1:
            if makespan == 'MC':
                planner = Single_Arm_A_star_solver(
                    self.Width, self.Height, len(real_start_arr), self.radius, 
                    real_start_arr, real_goal_arr
                    )
                self.overlap_rate = 1.0
            elif makespan == 'KC':
                planner = KC_Single_Arm_A_star_solver(
                    self.Width, self.Height, len(real_start_arr), self.radius, 
                    real_start_arr, real_goal_arr, num_arms=2, overlap_rate=self.overlap_rate
                    )
            else:
                planner = MC_Greedy(
                    self.Width, self.Height, len(real_start_arr), self.radius, 
                    real_start_arr, real_goal_arr, num_arms=1, overlap_rate=self.overlap_rate
                    )
        else:
            raise ValueError('wrong number of arms')

        planner.A_star()


        # DG = self.construct_DG(real_start_arr, real_goal_arr)

        # if self.primitive_plan == 'RB':
        #     _, action_sequence, self.max_node, self.total_node = DFS_DP_Search(DG)
        # elif self.primitive_plan == 'FVS':
        #     action_sequence = FVS_Plan_Search(DG)
        # elif self.primitive_plan == 'random':
        #     action_sequence = random_order(DG)

        # print action_sequence
        # for action in action_sequence:
        #     print action

        # generate "arm_actions" dual-action_sequences
        # arm_actions = {
        #     0:[(obj1, 'S'/'H'/'0'/'1', 'G'/'H'/'0'/'1')],
        #     1:[None, (obj1, 'S', 'G')]
        # }

        arm_actions = planner.action_list

        Problem_Partition = self.problem_partitioning( arm_actions)

        self.new_arr, self.transitions, self.isfeasible = self.call_subproblems(
            arm_actions, 
            Problem_Partition, 
            start_arr, 
            goal_arr
            )

    
    def problem_partitioning(self, arm_action:dict)->list:
        '''
        Assuming that there is no obj moving from buffer to buffer
        '''
        Partitions = [] # a list of subproblems
        current_buffers = [] # a list of objects that need buffers in this subproblem
        partial_action_sequence = {
            0:[],
            1:[]
        } # a dict of lists of actions for a subproblem
        print(arm_action)
        for time_step in range(len(arm_action[0])):
            arm0_action = arm_action[0][time_step]
            arm1_action = arm_action[1][time_step]
            if (
                (arm0_action!= None) and (arm0_action[2] not in {'H', 'G'})
                ) or (
                (arm1_action!= None) and (arm1_action[2] not in {'H', 'G'})
                ): # inside a subproblem or going into a subproblem
                if ( (arm0_action!= None) and (arm0_action[2] not in {'H', 'G'})): # move to buffer
                    current_buffers.append(arm0_action[0])
                if ((arm1_action!= None) and (arm1_action[2] not in {'H', 'G'})): # move to buffer
                    current_buffers.append(arm1_action[0])
                partial_action_sequence[0].append(arm0_action)
                partial_action_sequence[1].append(arm1_action)
            elif (len(current_buffers)==0): # outside a problem
                continue
            else: # (len(current_buffers)>0) inside a problem or going out of a subp
                partial_action_sequence[0].append(arm0_action)
                partial_action_sequence[1].append(arm1_action)
            if ((arm0_action != None) and (arm0_action[0] in current_buffers) and (arm0_action[1] in {'0', '1'})):
                current_buffers.remove(arm0_action[0])
            if ((arm1_action != None) and (arm1_action[0] in current_buffers) and (arm1_action[1] in {'0', '1'})):
                current_buffers.remove(arm1_action[0])
            if (len(current_buffers)==0) and (len(partial_action_sequence)>0): # just gone out a subp
                Partitions.append(partial_action_sequence)
                partial_action_sequence = {
                    0:[],
                    1:[]
                }
        return Partitions
            
    def call_subproblems(self, arm_actions:dict, Problem_Partition:list, start_arr, goal_arr):
        current_index = 0
        dual_action_list = {0:[], 1:[]} # {0: [(objID, start_pose, buffer_pose), None], 1: [None, (objID, start_pose, buffer_pose)]}
        isfeasible = True
        current_arr = deepcopy_dict(start_arr)
        if len(Problem_Partition) == 0:
            for time_step in range(len(arm_actions[0])):
                arm0_action = arm_actions[0][time_step]
                arm1_action = arm_actions[1][time_step]
                if arm0_action == None:
                    dual_action_list[0].append(arm0_action)
                else:
                    objID = arm0_action[0]
                    if arm0_action[1] == 'S':
                        start_pose = start_arr[objID]
                    else: # 'H'
                        start_pose = 'H'
                    if arm0_action[2] == 'G':
                        goal_pose = goal_arr[objID]
                    else:
                        goal_pose = 'H'
                    dual_action_list[0].append((objID, start_pose, goal_pose))
                if arm1_action == None:
                    dual_action_list[1].append(arm1_action)
                else:
                    objID = arm1_action[0]
                    if arm1_action[1] == 'S':
                        start_pose = start_arr[objID]
                    else: # 'H'
                        start_pose = 'H'
                    if arm1_action[2] == 'G':
                        goal_pose = goal_arr[objID]
                    else:
                        goal_pose = 'H'
                    dual_action_list[1].append((objID, start_pose, goal_pose))
            return goal_arr, dual_action_list, isfeasible
        
        goal_objs = []        
        for subproblem in Problem_Partition:
            if subproblem[0][0] != None:
                start_index = arm_actions[0].index(subproblem[0][0])
            else:
                start_index = arm_actions[1].index(subproblem[1][0])
            if subproblem[0][-1] != None:
                end_index = arm_actions[0].index(subproblem[0][-1])
            else:
                end_index = arm_actions[1].index(subproblem[1][-1])

            while current_index < start_index: # move objects to the goal before solving the first subproblem
                arm0_action = arm_actions[0][current_index]
                arm1_action = arm_actions[1][current_index]
                if arm0_action == None:
                    dual_action_list[0].append(arm0_action)
                else:
                    objID = arm0_action[0]
                    if arm0_action[1] == 'S':
                        start_pose = start_arr[objID]
                    else: # 'H'
                        start_pose = 'H'
                    if arm0_action[2] == 'G':
                        goal_pose = goal_arr[objID]
                        goal_objs.append(objID)
                    else:
                        goal_pose = 'H'
                    dual_action_list[0].append((objID, start_pose, goal_pose))
                    if goal_pose != 'H':
                        current_arr[objID] = goal_pose
                if arm1_action == None:
                    dual_action_list[1].append(arm1_action)
                else:
                    objID = arm1_action[0]
                    if arm1_action[1] == 'S':
                        start_pose = start_arr[objID]
                    else: # 'H'
                        start_pose = 'H'
                    if arm1_action[2] == 'G':
                        goal_pose = goal_arr[objID]
                        goal_objs.append(objID)
                    else:
                        goal_pose = 'H'
                    dual_action_list[1].append((objID, start_pose, goal_pose))
                    if goal_pose != 'H':
                        current_arr[objID] = goal_pose
                current_index += 1

            # current_arr = {}
            # for obj in start_arr.keys():
            #     if obj in goal_objs:
            #         current_arr[obj] = goal_arr[obj]
            #     else:
            #         current_arr[obj] = start_arr[obj]
            partial_schedule, partial_buffers = self.process_constraints(subproblem)
            # TODO: only Buffer Sampling supported so far
            if self.robust_sampling:
                BG = Robust_Buffer_Sampling(
                    subproblem, partial_buffers, partial_schedule, 
                    current_arr, goal_arr, self.Height, self.Width, self.radius
                )
            elif self.sampling:
                BG = Buffer_Sampling(
                    subproblem, partial_buffers, partial_schedule, 
                    current_arr, goal_arr, self.Height, self.Width, self.radius, 
                    overlap_rate = self.overlap_rate
                )
            else:
                BG = Buffer_Generation(subproblem, partial_buffers, partial_schedule, 
                current_arr, goal_arr, self.Height, self.Width, self.radius
                )


            if self.sampling:
                self.num_collision += BG.num_collision

            if self.robust_sampling:
                action_list = action_list + BG.action_list
                isfeasible = BG.isfeasible
                current_arr = BG.final_arr
                if isfeasible:
                    current_index = end_index+1
                else:
                    break
            else:
                buffer_locations = BG.buffer_locations
                # print "buffer locations"
                # for key, value in buffer_locations.items():
                #     print key, value

                if BG.trouble_action_index == float('inf'):
                    isfeasible = True 
                else:
                    isfeasible = False

                for time_step in range(len(subproblem[0])):
                    if time_step == BG.trouble_action_index:
                        break
                    for arm in subproblem.keys():
                        real_action = self.primitive_action_to_exact_action(
                            subproblem[arm][time_step], 
                            buffer_locations,
                            start_arr,
                            goal_arr
                            )
                        dual_action_list[arm].append(
                            real_action
                        )
                        if (real_action != None) and (type(real_action[2]) is not str):
                            current_arr[real_action[0]] = real_action[2]
                if isfeasible:
                    current_index = end_index+1
                else:
                    break
        if isfeasible:
            while current_index < len(arm_actions[0]): # move objects to the goal after solving subproblems
                arm0_action = arm_actions[0][current_index]
                arm1_action = arm_actions[1][current_index]
                if arm0_action == None:
                    dual_action_list[0].append(arm0_action)
                else:
                    objID = arm0_action[0]
                    if arm0_action[1] == 'S':
                        start_pose = start_arr[objID]
                    else: # 'H'
                        start_pose = 'H'
                    if arm0_action[2] == 'G':
                        goal_pose = goal_arr[objID]
                        goal_objs.append(objID)
                    else:
                        goal_pose = 'H'
                    dual_action_list[0].append((objID, start_pose, goal_pose))
                    if goal_pose != 'H':
                        current_arr[objID] = goal_pose
                if arm1_action == None:
                    dual_action_list[1].append(arm1_action)
                else:
                    objID = arm1_action[0]
                    if arm1_action[1] == 'S':
                        start_pose = start_arr[objID]
                    else: # 'H'
                        start_pose = 'H'
                    if arm1_action[2] == 'G':
                        goal_pose = goal_arr[objID]
                        goal_objs.append(objID)
                    else:
                        goal_pose = 'H'
                    dual_action_list[1].append((objID, start_pose, goal_pose))
                    if goal_pose != 'H':
                        current_arr[objID] = goal_pose
                current_index += 1

        return current_arr, dual_action_list, isfeasible

    def primitive_action_to_exact_action(self,action, buffer_locations:dict, start_arr, goal_arr):
        if action == None:
            return None
        primitive_poses = [action[1], action[2]]
        obj = action[0]
        real_poses = []
        for pose in primitive_poses:
            if pose == 'S':
                real_poses.append(start_arr[obj])
            elif pose == 'G':
                real_poses.append((goal_arr[obj]))
            elif pose == 'H':
                real_poses.append('H')
            else: # buffer
                real_poses.append(buffer_locations[obj])
        return tuple([obj]+real_poses)

            
    def process_constraints(self, arm_actions:dict):
        time_stamp = 0
        obj_schedule = {} # dict[obj_id] = [starting_timestamp, goal_timestamp]
        buffers = []
        for time_stamp in range(len(arm_actions[0])):
            arm0_action = arm_actions[0][time_stamp]
            arm1_action = arm_actions[1][time_stamp]
            actions = []
            if (arm0_action != None) and (arm1_action != None) and (arm0_action[0]==arm1_action[0]): #handoff
                if arm0_action[1] == 'H':
                    action = (arm0_action[0], arm1_action[1], arm0_action[2])
                else:
                    action = (arm0_action[0], arm0_action[1], arm1_action[2])
                actions.append(action)
            else:
                if arm0_action != None:
                    actions.append(arm0_action)
                if arm1_action != None:
                    actions.append(arm1_action)
            for action in actions:
                if action[2] in {'0','1'}:
                    obj_schedule[action[0]] = [time_stamp, -1]
                    buffers.append(action[0])
                elif action[0] in obj_schedule:
                    obj_schedule[action[0]][1] = time_stamp
                else:
                    obj_schedule[action[0]] = [time_stamp, time_stamp]
            time_stamp += 1
        return obj_schedule, buffers

    def construct_DG(self, start_arr, goal_arr):
        DG = {}
        for goal_obj, goal_center in goal_arr.items():
            DG[goal_obj] = set()
            for start_obj, start_center in start_arr.items():
                if start_obj == goal_obj:
                    continue
                self.num_collision += 1
                if (sqrt((start_center[0]-goal_center[0])**2+
                (start_center[1]-goal_center[1])**2) <= 2*self.radius):
                    DG[goal_obj].add(start_obj)
        return DG
        