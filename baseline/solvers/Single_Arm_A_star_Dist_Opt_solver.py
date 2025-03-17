import itertools
import sys
import math
import copy
import numpy as np

# sys.path.append("/home/kai/Documents/Kai_Gao/dual_arm/limited_reachability_makespan_opt/")
from solvers.high_level_solver import high_level_solver
from tools.util import construct_DG


class arr_status(object):
    def __init__(self, id) -> None:
        self.id = id
        self.g = float('inf')
        self.h = float('inf')
        self.f = float('inf')
        self.in_open = False
        self.parent = None
        self.holding = None # (armID, obj, target)
        self.just_placed = None # (armID, objID)
        self.holding_time = 0 # holding time so far
        self.estimated_time = 0 # total time to move the obj to the target
        self.gap = 0 # makespan difference arm0-arm1
        self.last_finish_positions = {0:(0,0), 1:(0,0)}


class Single_Arm_A_star_solver(high_level_solver):
    ''''''
    def __init__(self, Width, Height, numObjs, radius, start_arr, goal_arr, num_arms=2, overlap_rate=0.3) -> None:
        super().__init__(Width, Height, numObjs, radius, start_arr, goal_arr, num_arms=num_arms, overlap_rate=overlap_rate)

        self.DG = construct_DG(start_arr, goal_arr, radius)

    def A_star(self):
        '''
        A star search
        '''
        self.state_dict = {} # key: compressed_state, value: arr_status class
        self.open_list = [] # [compressed_state] decrease order
        self.parents = {}
        # state: objs at S/G/0/1/T, T means moving(transfer or transit), 'arm0'/'arm1': E/G/H/0/1, E means empty
        start_state = {k:'S' for k in self.start_arr.keys()}
        start_state['arm0'] = 'E' # empty
        start_state['arm1'] = 'E' # empty
        compressed_start_state = self.compress_state_dict(start_state)
        start_status = arr_status(0)
        self.state_dict[compressed_start_state] = start_status
        start_status.h = self.heuristic(compressed_start_state)
        start_status.g = 0
        start_status.f = start_status.h
        start_status.in_open = True
        start_status.last_finish_positions = {
            0:self.rest_pose[0],
            1:self.rest_pose[1]
        }
        self.open_list.append( compressed_start_state)
        FindGoal = False # Found goal pose
        while (len(self.open_list) >0) and (not FindGoal):
            # pop out (from the list bottom) a state to develop
            current_cpd_state = self.open_list.pop()
            self.state_dict[current_cpd_state].in_open = False
            if self.is_goal(self.decompress_state(current_cpd_state)):
                FindGoal = True
                continue
            for new_cpd_state, pending_status in self.generate_new_nodes(current_cpd_state):
                # visited state
                if new_cpd_state in self.state_dict.keys():
                    new_status = self.state_dict[new_cpd_state]
                    if new_status.g > pending_status.g: # found a shorter path
                        pending_status.id = new_status.id
                        pending_status.h = new_status.h
                        pending_status.f = pending_status.g + pending_status.h
                        pending_status.parent = current_cpd_state
                        if new_status.in_open: # in the open set
                            self.resort_in_open_list(new_cpd_state, pending_status) # can have a higher ranking
                        else: # revisit a node
                            self.state_dict[new_cpd_state] = pending_status
                            self.add_new_state_to_open(new_cpd_state)
                            pending_status.in_open = True
                    else: # not a better parent
                        continue
                        
                else: # new state
                    pending_status.id = len(self.state_dict)
                    self.state_dict[new_cpd_state] = pending_status
                    pending_status.h = self.heuristic(new_cpd_state)
                    pending_status.f = pending_status.g + pending_status.h
                    pending_status.in_open = True
                    pending_status.parent = current_cpd_state
                    self.add_new_state_to_open(new_cpd_state)
        if FindGoal:
            self.action_list = self.construct_solution()
        else:
            print("no feasible solution")


    def construct_solution(self):
        '''
        Backtrack to get the state path
        return actions{0:[(21, 'S', 'G')], 1:[(45, '0', 'H')]}
        '''
        goal_state = {k:'G' for k in self.start_arr.keys()}
        goal_state['arm0'] = 'E'
        goal_state['arm1'] = 'E'
        compressed_goal_state = self.compress_state_dict(goal_state)
        assert compressed_goal_state in self.state_dict
        state_list = []
        action_list = []
        # backtrack to get the state list
        current_cpd_state = compressed_goal_state
        current_status = self.state_dict[current_cpd_state]
        state_list.append(current_cpd_state)
        action_list.append(current_status.just_placed)
        while current_status.parent != None:
            current_state = current_status.parent
            state_list.append(current_state)
            current_status = self.state_dict[current_state]
            if current_status.just_placed != action_list[-1]:
                action_list.append(current_status.just_placed)
        state_list.reverse()
        action_list.reverse()
        while None in action_list:
            action_list.remove(None)
        # construct arm action list
        arm_actions = {arm:[] for arm in range(self.num_arms)}
        current_obj_state = {k:'S' for k in self.start_arr.keys()}
        for _, obj, target in action_list:
            arm_actions[0].append((obj, current_obj_state[obj], target))
            arm_actions[1].append(None)
            current_obj_state[obj] = target
        return arm_actions
    
    def add_new_state_to_open(self, state):   
        """ Add new node to open list based on f value """
        state_f = self.state_dict[state].f
        high = len(self.open_list) - 1
        low = 0
        mid = 0
        while high >= low:
            mid = (high+low)//2
            # print(mid)
            # print(state_f, self.state_dict[self.open_list[mid]].f)
            if state_f < self.state_dict[self.open_list[mid]].f:
                low = mid + 1
                mid = low
            elif state_f > self.state_dict[self.open_list[mid]].f:
                high = mid - 1
                mid = high
            else:
                break
            # print(low,mid,high)
        # print("_____________________")
        if high < low:
            mid = max(0,low)
        self.open_list.insert(mid, state)
        return

    def remove_state_from_open(self, state):   
        """ Remove a node from open list based on f value """
        
        state_f = self.state_dict[state].f
        high = len(self.open_list) - 1
        low = 0
        mid = 0
        while high >= low:
            mid = (high+low)//2
            if state_f == self.state_dict[self.open_list[mid]].f:
                break
            elif state_f < self.state_dict[self.open_list[mid]].f:
                low = mid + 1
            elif state_f > self.state_dict[self.open_list[mid]].f:
                high = mid - 1
        if high < low:
            mid = low
        # get state:
        if state == self.open_list[mid]:
            del self.open_list[mid]
            return
        # go low:
        low = mid - 1
        while state_f <= self.state_dict[self.open_list[low]].f:
            if state == self.open_list[low]:
                del self.open_list[low]
                return
            low -= 1
            if low < 0:
                break
        # go high:
        high = mid + 1
        while state_f >= self.state_dict[self.open_list[high]].f:
            if state == self.open_list[high]:
                del self.open_list[high]
                return
            high += 1
            if high >= len(self.open_list):
                break
        raise TypeError('Impossible removal situation')
        


    def resort_in_open_list(self, state, status):
        self.remove_state_from_open(state)
        status.in_open = True
        self.state_dict[state] = status
        self.add_new_state_to_open(state)

    def compress_state_dict(self, state):
        return tuple([state[obj] for obj in sorted(list(self.start_arr.keys()))]
        +[state['arm0'], state['arm1']])

    def decompress_state(self, compressed_state):
        state_dict = {}
        for i, k in enumerate(sorted(list(self.start_arr.keys()))):
            state_dict[k] = compressed_state[i]
        state_dict['arm0'] = compressed_state[-2]
        state_dict['arm1'] = compressed_state[-1]
        return state_dict


    def is_goal(self, new_state):
        for k, v in new_state.items():
            if type(k) == str:
                continue
            if v != 'G':
                return False
        else:
            return True
    

    def generate_new_nodes(self, cpd_current_state):
        ''' children of the current node 
        '''
        current_state = self.decompress_state(cpd_current_state)
        current_status = self.state_dict[cpd_current_state]
        
        for obj, status in current_state.items():
            if(type(obj) is str): # key is an arm
                continue
            if status == 'T':
                raise TypeError('empty hands but this obj is holded')
            if status == 'G': # at goal already
                continue
            if self.goal_available(obj, current_state): # move to goal
                yield self.generate_new_state_from_action(
                    current_state, 
                    current_status, 
                    (0, obj, 'G')
                    )
            elif status == 'S': # move to buffer if at start
                yield self.generate_new_state_from_action(
                    current_state, 
                    current_status, 
                    (0, obj, str(0))
                    )


    def generate_new_state_from_action(self, prev_state, prev_status:arr_status, action:tuple):
        '''
        prev_status: arr_status
        action: (armID, objID, target)
        Output:
        new state and new status
        '''
        new_arm, new_obj, target = action
        new_arm_current_pose = prev_status.last_finish_positions[new_arm]
        if prev_state[new_obj] == 'S':
            new_obj_current_pose = self.start_arr[new_obj]
        elif prev_state[new_obj] == 'T': # handoffing
            new_obj_current_pose = 'H'
        else: # buffer
            new_obj_current_pose = prev_state[new_obj]
        if target == 'G':
            target_pose = self.goal_arr[new_obj]
        else: # buffer or handoff
            target_pose = target
        # compute cost
        new_action_cost = self.cost_estimation(
            new_arm_current_pose,
            new_obj_current_pose,
            target_pose
            )
        
        
        # consider handoff first
        new_state = copy.deepcopy(prev_state)
        new_status = copy.deepcopy(prev_status)
        new_state[new_obj] = target
        new_status.g = prev_status.g + new_action_cost
        
        new_status.just_placed = action
        new_status.estimated_time = new_action_cost
        new_status.last_finish_positions = {
            0:target_pose,
            1:self.rest_pose[1]
        }
        
        if self.is_goal(new_state):
            new_status.g += self.compute_final_transit(prev_status, new_status)

        
        return self.compress_state_dict(new_state), new_status
    
    def compute_final_transit(self, prev_status, new_status):
        last_arm, last_obj, _ = new_status.just_placed
        last_arm_cost = self.transit_cost(
            self.goal_arr[last_obj], 
            self.rest_pose[last_arm]
        )
        return last_arm_cost
        


    def cost_estimation(self, current_pose, pose1, pose2):
        # pose can be H, 0, 1, (x,y)
        if current_pose == 'H': # don't specialize current pose
            current_pose = self.handoff_pose
        cost = 0
        # manipulation
        if pose1 == 'H': # from handoff
            cost += self.cost_constants['release']
        elif pose2 == 'H':
            cost += self.cost_constants['grasp']
        else:
            cost += self.cost_constants['grasp'] + self.cost_constants['release']
        # transit
        if (type(current_pose) is str) or ((type(pose1) is str) and pose1 != 'H'): # buffer involved
            cost += self.cost_constants['from_to_buffer']
        elif pose1 == 'H':
            cost += self.transit_cost(current_pose, self.handoff_pose)
        else:
            cost += self.transit_cost(current_pose, pose1)
        # transfer
        if (type(pose1) is str) or ((type(pose2) is str) and pose2 != 'H'): # buffer involved
            cost += self.cost_constants['from_to_buffer']
        elif pose2 == 'H': # go to handoff
            cost += self.transfer_cost(pose1, self.handoff_pose)
        else:
            cost += self.transfer_cost(pose1, pose2)
        return cost
        

    def transfer_cost(self, pose1, pose2):
        return np.linalg.norm(np.array(pose1)-np.array(pose2))/self.cost_constants['transfer_v']

    def transit_cost(self, pose1, pose2):
        return np.linalg.norm(np.array(pose1)-np.array(pose2))/self.cost_constants['transit_v']



    def goal_available(self, obj:int, current_state):
        for obs in self.DG[obj]:
            if current_state[obs] == 'S':
                return False
        else:
            return True

        

    def heuristic(self, current_cpd_state):
        # return 0
        # current_status = self.state_dict[current_cpd_state]
        # time that each hand has to use
        arm_counts = 0
        current_state = self.decompress_state(current_cpd_state)
        for obj, state in current_state.items():
            cost = self.cost_constants['release']
            if type(obj) == str:
                continue
            if state == 'G':
                continue
            elif state == 'T':
                continue
            elif state == 'S':
                cost += self.transfer_cost(self.start_arr[obj], self.goal_arr[obj]) + self.cost_constants['grasp'] 
            else: # at a buffer
                cost += self.cost_constants['from_to_buffer'] + self.cost_constants['grasp']
            arm_counts += cost
        return arm_counts


