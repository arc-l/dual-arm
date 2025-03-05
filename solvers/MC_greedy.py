import copy
import itertools
import sys
import math
import numpy as np
import random
from solvers.high_level_solver import high_level_solver
from tools.util import construct_DG


class arr_status(object):
    def __init__(self, id) -> None:
        self.id = id
        self.g = float('inf')
        self.h = float('inf')
        self.f = float('inf')
        self.in_open = False
        self.in_closed = False
        self.parent = None



class Greedy_Solver(high_level_solver):
    def __init__(self, Width, Height, numObjs, radius, start_arr, goal_arr, num_arms=2, overlap_rate=0.3) -> None:
        super().__init__(Width, Height, numObjs, radius, start_arr, goal_arr, num_arms=num_arms, overlap_rate=overlap_rate)

        self.DG = construct_DG(start_arr, goal_arr, radius)

    def greedy_search(self):
        '''
        greedy search for the schedule
        If some buffer object have available goal, then move to goal
        Else: pick the nearest object, to goal if possible, else, pick to goal
        If one arm is looking for a handoff, 
        then the other arm needs to do handoff rather than pick the nearest
        '''
        self.state_dict = {} # key: compressed_state, value: arr_status class
        start_state = {k:'S' for k in self.start_arr.keys()}
        compressed_start_state = self.compress_state_dict(start_state)
        start_status = arr_status(0)
        self.state_dict[compressed_start_state] = start_status
        self.current_cpd_state = compressed_start_state
        self.current_buffers = set() # current objects at buffers
        self.current_arm_positions = {
            0:[0, self.Height/2.0],
            1:[self.Width, self.Height/2.0]
        }
        FindGoal = False # Found goal pose
        while not FindGoal:
            if self.is_goal(self.current_cpd_state):
                FindGoal = True
                continue
            objects_to_move = {
                0:[], 1:[] # [(dist1,obj1), (dist2,obj2)] a list with at most two objects and the dist to the obj
            }
            current_state = self.decompress_state(self.current_cpd_state)
            current_at_start = set(
                [k for k, v in current_state.items() if v =='S']
            )
            for armID in range(self.num_arms):
                buffer_objects = self.get_available_buffer_objects(armID)
                for obj in buffer_objects:
                    objects_to_move[armID].append((0,obj))
                if len(buffer_objects) >= 2:
                    continue
                objects_to_move[armID].extend(
                    self.get_nearest_objects(armID, current_at_start)
                    )
            # select objects
            selected_object = {i:None for i in range(self.num_arms)}
            selection_order = list(range(self.num_arms))
            random.shuffle(selection_order)
            for armID in selection_order:
                for _,obj in objects_to_move[armID]:
                    if obj not in selected_object.values():
                        selected_object[armID] = obj
                        break
                    
            # whether it is a handoff
            is_handoff = False
            handoff_object = None
            for obj in selected_object.values():
                if obj == None:
                    continue
                start_region = set()
                status = current_state[obj]
                if status == "S":
                    start_region = self.specify_region(self.start_arr[obj])
                else: # at buffer
                    start_region = {int(float(status))}
                goal_region = self.specify_region(self.goal_arr[obj])
                itst_region = start_region.intersection(goal_region)
                if len(itst_region) == 0: # need handoff
                    is_handoff = True
                    handoff_object = obj
                    break
            new_state = copy.deepcopy(current_state)
            if (is_handoff) and (self.num_arms==2):
                goal_region = list(self.specify_region(self.goal_arr[handoff_object]))[0]
                if self.goal_available(handoff_object, current_state):
                    new_state[handoff_object] = 'G'
                    if (handoff_object, goal_region) in self.current_buffers:
                        self.current_buffers.remove((handoff_object, goal_region))
                    self.current_arm_positions[goal_region] = self.goal_arr[handoff_object]
                    self.current_arm_positions[1-goal_region] = (self.Width/2.0, self.Height/2.0)
                else:
                    new_state[handoff_object] = str(goal_region)
                    self.current_buffers.add((handoff_object, goal_region))
                    self.current_arm_positions[goal_region] = self.start_arr[handoff_object]
                    self.current_arm_positions[1-goal_region] = (self.Width/2.0, self.Height/2.0)

            else:
                for armID in range(self.num_arms):
                    obj = selected_object[armID]
                    if obj == None:
                        continue
                    if self.goal_available(obj, current_state):
                        new_state[obj] = 'G'
                        if (obj,armID) in self.current_buffers:
                            self.current_buffers.remove((obj,armID))
                        self.current_arm_positions[armID] = self.goal_arr[obj]
                    else:
                        new_state[obj] = str(armID)
                        self.current_buffers.add((obj,armID))
                        self.current_arm_positions[armID] = self.start_arr[obj]
            new_status = arr_status(len(self.state_dict))
            new_status.parent = self.current_cpd_state
            self.state_dict[self.compress_state_dict(new_state)] = new_status            
            self.current_cpd_state = self.compress_state_dict(new_state)
        if FindGoal:
            self.action_list = self.construct_solution()


        
    def get_available_buffer_objects(self, armID:int):
        ''' get buffer objects whose goal positions are available '''
        current_state = self.decompress_state(self.current_cpd_state)
        possible_buffer_objects = list()
        for obj,regionID in self.current_buffers:
            if regionID != armID:
                continue
            if self.goal_available(obj, current_state):
                possible_buffer_objects.append(obj)
        return possible_buffer_objects[:2]


    def get_nearest_objects(self, armID:int, start_objects:set):
        ''' get nearest start object for an arm '''
        distance_list = []
        current_position = self.current_arm_positions[armID]
        for obj in start_objects:
            pos = self.start_arr[obj]
            if armID not in self.specify_region(pos): # start position not accessible
                continue
            elif (armID not in self.specify_region(self.goal_arr[obj])) and ((1-armID) in self.specify_region(self.start_arr[obj])):
                # this object belongs to the other arm
                continue
            dist = np.linalg.norm(
                np.array(current_position)-pos
            )
            distance_list.append((dist, obj))
        distance_list.sort()
        return distance_list[:2]

        
    def check_handoff_options(self):
        ''' obtain objects for handoffs '''
            

    def A_star(self):
        '''
        use the shell of A star search
        '''
        self.greedy_search()

    def construct_solution(self):
        '''
        Backtrack to get the state path
        return actions{0:[(21, 'S', 'G')], 1:[(45, '0', 'H')]}
        '''
        goal_state = {k:'G' for k in self.start_arr.keys()}
        compressed_goal_state = self.compress_state_dict(goal_state)
        assert compressed_goal_state in self.state_dict
        state_list = []
        # backtrack to get the state list
        current_cpd_state = compressed_goal_state
        current_status = self.state_dict[current_cpd_state]
        state_list.append(current_cpd_state)
        while current_status.parent != None:
            current_state = current_status.parent
            state_list.append(current_state)
            current_status = self.state_dict[current_state]
        state_list.reverse()
        # construct arm action list
        arm_actions = {arm:[] for arm in range(2)}
        # print('-----------------')
        # for state in state_list:
        #     print(self.decompress_state(state))
        for i in range(len(state_list)-1):
            previous_state = state_list[i]
            new_state = state_list[i+1]
            new_actions = self.state_transition_to_actions(previous_state, new_state)
            for arm_id, actions in new_actions.items():
                arm_actions[arm_id].extend(actions)
        return arm_actions
    

    def state_transition_to_actions(self, previous_state, new_state):
        '''
        Input: two compressed states
        Output: actions:{0:[(21, 'S', 'G')], 1:[(45, '0', 'H')]}
        '''
        arm_actions = {k:[] for k in range(2)}
        prev_state_dict = self.decompress_state(previous_state)
        new_state_dict = self.decompress_state(new_state)
        # objects that are moved in the transition
        diff_obj = [k for k in prev_state_dict.keys() 
        if prev_state_dict[k]!=new_state_dict[k]
        ]
        action_options = {obj:[] for obj in diff_obj} # {obj:[[(armID, prev_status, new_status)]]}
        for obj in diff_obj:
            prev_status = prev_state_dict[obj]
            new_status = new_state_dict[obj]
            # get a set of arms that are involved
            arm_option_list = []
            for status in [prev_status, new_status]:
                if status == 'S':
                    pose = self.start_arr[obj]
                    arm_option_list.append(self.specify_region(pose))
                elif status == 'G':
                    pose = self.goal_arr[obj]
                    arm_option_list.append(self.specify_region(pose))
                else:
                    arm_option_list.append({int(status)}) # if at buffer, which buffer(int)
            intersection = arm_option_list[0].intersection(arm_option_list[1])
            if len(intersection) == 2: # both status are at overlapping region
                action_options[obj] = [
                    [(0, prev_status, new_status)],
                    [(1, prev_status, new_status)]
                ]
            elif len(intersection) == 1: # only one arm can execute the action
                action_options[obj] = [
                    [(list(intersection)[0], prev_status, new_status)]
                ]
            elif len(intersection) == 0:
                action_options[obj] = [[
                    (list(arm_option_list[0])[0], prev_status, 'H'),
                    (list(arm_option_list[1])[0], 'H', new_status),
                ]]
        priority_objs = sorted(diff_obj, key = lambda e: len(action_options[e]))
        Success = True # whether we can get valid actions
        for obj in priority_objs:
            for actions in action_options[obj]:
                # detect used arms
                unavailable_arms = [action[0] for action in actions if len(arm_actions[action[0]])>0]
                if unavailable_arms != []:
                    Success = False
                    continue
                Success = True
                for action in actions:
                    arm_actions[action[0]].append(tuple([obj]+list(action[1:])))
                break
        for arm in arm_actions.keys():
            if arm_actions[arm] == []:
                arm_actions[arm] = [None]
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
    

    def resort_in_open_list(self, state):
        self.remove_state_from_open(state)
        self.add_new_state_to_open(state)

    def compress_state_dict(self, state):
        return tuple([state[obj] for obj in sorted(list(state.keys()))])

    def decompress_state(self, compressed_state):
        state_dict = {}
        for i, k in enumerate(sorted(list(self.start_arr.keys()))):
            state_dict[k] = compressed_state[i]
        return state_dict


    def is_goal(self, new_state):
        new_state = self.decompress_state(new_state)
        for v in new_state.values():
            if v != 'G':
                return False
        else:
            return True
    

    def generate_new_nodes(self, cpd_current_state):
        ''' children of the current node '''
        # objects that are able to do handoffs
        handoff_options = []
        # objects that are able to swap to goal at the same time even with dependencies
        swap_options = [] # [[(arm1, obj1, 'G'), (arm2, obj2, 'B')] ]
        # besides handoffs, each arm has other action options, None means doing nothing
        arm_options = {id : [None] for id in range(self.num_arms)}
        current_state = self.decompress_state(cpd_current_state)
        for obj, status in current_state.items():
            if status == 'G': # we can do nothing for objs at goal
                continue
            elif status == 'S':
                start_arm_option_set = self.specify_region(self.start_arr[obj])
                goal_arm_options_set = self.specify_region(self.goal_arr[obj])
                intersection = start_arm_option_set.intersection(goal_arm_options_set)
                if intersection == set(): # need handoff and at start pose
                    # inter regional start: handoff or to buffer
                    handoff_options.append(obj)
                    buffer_id = list(start_arm_option_set)[0]
                    arm_options[buffer_id].append((obj, str(buffer_id)))
                else: # no need handoff
                    # don't need handoff: check goal availability
                    obs_set = self.check_obstacle_set(obj, current_state)
                    if (len(obs_set) == 0): # goal pose avaliable
                        for valid_arm in intersection:
                            arm_options[valid_arm].append((obj, 'G'))
                    elif (len(obs_set) == 1):
                        for valid_arm in intersection:
                            arm_options[valid_arm].append((obj, str(valid_arm)))
                        obs = list(obs_set)[0]
                        obs_start_option_set = self.specify_region(self.start_arr[obs])
                        obs_goal_option_set = self.specify_region(self.goal_arr[obs])
                        obs_intersection = obs_start_option_set.intersection(obs_goal_option_set)
                        for obj_arm in intersection:
                            obs_arm = 1-obj_arm
                            # obs to goal
                            if (obs_arm in obs_intersection) and ({obj}.issuperset(self.check_obstacle_set(obs, current_state))):
                                swap_options.append(
                                    ((obj_arm, obj, 'G'), (obs_arm, obs, 'G'))
                                )
                            else:
                                # move to buffer
                                # whether the obs arm can move obs to its buffer
                                if obs_intersection == {obj_arm}:
                                    continue
                                # goal not available
                                if (obs_arm in obs_start_option_set):
                                    swap_options.append(
                                        ((obj_arm, obj, 'G'), (obs_arm, obs, str(obs_arm)))
                                    )
                        

                    else:
                        for valid_arm in intersection:
                            arm_options[valid_arm].append((obj, str(valid_arm)))
            else: # at a buffer
                goal_arm_options_set = self.specify_region(self.goal_arr[obj])
                buffer_id = int(status)
                intersection = {buffer_id}.intersection(goal_arm_options_set)
                if intersection==set(): # need handoff and at buffer
                    # inter regional start: handoff
                    if self.goal_available(obj, current_state): # goal pose avaliable
                        handoff_options.append(obj)
                else: # no need handoff
                    # don't need handoff: check goal availability
                    obs_set = self.check_obstacle_set(obj, current_state)
                    if (len(obs_set) == 0): # goal pose avaliable
                        arm_options[buffer_id].append((obj, 'G'))
                    elif (len(obs_set) == 1):
                        obj_arm = buffer_id
                        obs = list(obs_set)[0]
                        obs_start_option_set = self.specify_region(self.start_arr[obs])
                        obs_goal_option_set = self.specify_region(self.goal_arr[obs])
                        obs_intersection = obs_start_option_set.intersection(obs_goal_option_set)
                        obs_arm = 1-obj_arm
                        # obs to goal
                        if (obs_arm in obs_intersection) and ({obj}.issuperset(self.check_obstacle_set(obs, current_state))):
                            swap_options.append(
                                ((obj_arm, obj, 'G'), (obs_arm, obs, 'G'))
                            )
                        else:
                            # move to buffer
                            # whether the obs arm can move obs to its buffer
                            if obs_intersection == {obj_arm}:
                                continue
                            # goal not available
                            if (obs_arm in obs_start_option_set):
                                swap_options.append(
                                    ((obj_arm, obj, 'G'), (obs_arm, obs, str(obs_arm)))
                                )
        # new states after handoffs
        for obj in handoff_options:
            # deepcopy
            new_state = {k:value for k, value in current_state.items()}
            if self.goal_available(obj, current_state):
                new_state[obj] = 'G'
            else:
                # handoff objs must be at disjointed regions
                goal_arm = list(self.specify_region(self.goal_arr[obj]))[0]
                new_state[obj] = str(goal_arm)
            yield self.compress_state_dict(new_state)
        # TODO: only support two arms 
        # new states after swapping
        for option in swap_options:
            new_state = {k:value for k, value in current_state.items()}
            for action in option:
                new_state[action[1]] = action[2]
            yield self.compress_state_dict(new_state)
        for options in itertools.product(arm_options[0], arm_options[1]):
            if options == (None, None): # doesn't make sense if both robots do nothing
                continue
            if (options[0] != None) and (options[1] != None) and (options[0][0] == options[1][0]): # move the same obj
                continue
            new_state = {k:value for k, value in current_state.items()}
            for action in options:
                if action != None:
                    new_state[action[0]] = action[1]
            yield self.compress_state_dict(new_state)

                
    
    def goal_available(self, obj:int, current_state):
        for obs in self.DG[obj]:
            if current_state[obs] == 'S':
                return False
        else:
            return True

    def check_obstacle_set(self, obj:int, current_state):
        return set([obs for obs in self.DG[obj] if current_state[obs] == 'S'])

    def heuristic(self, current_state):
        # number of actions that each hand has to execute
        arm_counts = {id:0 for id in range(self.num_arms)}
        flexible_actions = 0 # actions that can be done by either arm
        current_state = self.decompress_state(current_state)
        for obj, state in current_state.items():
            if state == 'G':
                continue
            elif state == 'S':
                current_arm_option_set = self.specify_region(self.start_arr[obj])
            else: # at a buffer
                current_arm_option_set = {int(state)}
            goal_arm_option_set = self.specify_region(self.goal_arr[obj])
            intersection = current_arm_option_set.intersection(goal_arm_option_set)
            if len(intersection) == 2: # flexible action
                flexible_actions += 1
            elif len(intersection) == 1:
                arm_counts[list(intersection)[0]] += 1
            elif len(intersection) == 0:
                arm_counts[0] += 1
                arm_counts[1] += 1
            else:
                raise ValueError(' intersection error')
        arm_max = max(arm_counts[0], arm_counts[1])
        arm_min = min(arm_counts[0], arm_counts[1])
        if (arm_max - arm_min) >= flexible_actions:
            return arm_max
        else:
            return math.ceil((arm_max+arm_min+flexible_actions)/2.0)


    def specify_region(self, pose):
        ''' specify reachability '''
        available_set = {}
        if pose[0] < (0.5*(1-self.overlap_rate)*self.Width):
            available_set = {0}
        elif pose[0] > (0.5*(1+self.overlap_rate)*self.Width):
            available_set = {1}
        else:
            available_set = {0,1}
        return available_set.intersection(set(list(range(self.num_arms))))


class open_list(object):
    def __init__(self):
        '''
        init Dict[value,List[tuple]]
        '''
        self.open_dict = {}
        self.ordered_keys = [] # reversed
    
    def is_empty(self)->bool:
        ''' check empty '''
        return len(self.ordered_keys)==0
    
    def add_element(self, state:tuple, value:int):
        ''' append '''
        if value in self.open_dict:
            self.open_dict[value].append(state)
        else:
            self.open_dict[value] = [state]
            self.add_value_to_list(value)

    def remove_element(self, state:tuple, value:int):
        ''' remove '''
        if len(self.open_dict[value]) >=2:
            self.open_dict[value].remove(state)
        else:
            del self.open_dict[value]
            self.remove_list_element(value)
            
            

    def pop(self):
        state = self.open_dict[self.ordered_keys[-1]][0]
        self.remove_element(state, self.ordered_keys[-1])
        return state


    def add_value_to_list(self,value:int):
        high = len(self.ordered_keys) - 1
        low = 0
        mid = 0
        while high >= low:
            mid = (high+low)//2
            if value < self.ordered_keys[mid]:
                low = mid + 1
                mid = low
            elif value > self.ordered_keys[mid]:
                high = mid - 1
                mid = high
            else:
                break
            # print(low,mid,high)
        # print("_____________________")
        if high < low:
            mid = max(0,low)
        self.ordered_keys.insert(mid, value)
        return

    def get_list_index(self,value)->int:
        high = len(self.ordered_keys) - 1
        low = 0
        mid = 0
        # [low, high]
        while high >= low:
            mid = (high+low)//2
            if value < self.ordered_keys[mid]:
                low = mid + 1
                mid = low
            elif value > self.ordered_keys[mid]:
                high = mid - 1
                mid = high
            else:
                return mid
    
    def remove_list_element(self,value):
        index = self.get_list_index(value)
        del self.ordered_keys[index]
        