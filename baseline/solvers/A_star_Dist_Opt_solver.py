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


class A_star_solver(high_level_solver):
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
        current_state = compressed_goal_state
        current_status = self.state_dict[current_state]
        state_list.append(current_state)
        while current_status.parent != None:
            action_list.append(
                self.get_action_from_state_transition(
                    self.decompress_state(current_status.parent),
                    self.decompress_state(current_state)
                    )
            )
            current_state = current_status.parent
            state_list.append(current_state)
            current_status = self.state_dict[current_state]
        state_list.reverse()
        action_list.reverse()
        while None in action_list:
            action_list.remove(None)
        # construct arm action list
        arm_actions = {arm:[] for arm in range(self.num_arms)}
        current_obj_state = {k:'S' for k in self.start_arr.keys()}
        last_finish = {0:-1,1:-1}
        last_action_start_time = 0
        for i,action in enumerate(action_list):
            arm, obj, target = action
            arm_actions[arm].append((obj, current_obj_state[obj], target, last_finish[arm]+1, i))
            last_action_start_time = max(last_finish[arm]+1, last_action_start_time)
            current_obj_state[obj] = target
            last_finish[arm] = i
        # scheduled_arm_actions = {0:[None]*(last_action_start_time+1), 1:[None]*(last_action_start_time+1)}
        finish_moment = max(last_finish[0],last_finish[1])
        scheduled_arm_actions = {0:[None]*(finish_moment+1), 1:[None]*(finish_moment+1)}
        for arm in arm_actions.keys():
            for action in arm_actions[arm]:
                # start = action[3]
                # scheduled_arm_actions[arm][start] = action[:3]
                finish = action[4]
                scheduled_arm_actions[arm][finish] = action[:3]
        # parallize handoffs
        for armID in scheduled_arm_actions.keys():
            removed_indices = []
            for i, action in enumerate(scheduled_arm_actions[armID]):
                if action == None:
                    continue
                if action[1] == 'H':
                    # move from handoff
                    removed_indices.append(i-1)
                elif action[2] == 'H':
                    removed_indices.append(i+1)
            scheduled_arm_actions[armID] = [
                scheduled_arm_actions[armID][i] 
                for i in range(len(scheduled_arm_actions[armID]))
                if i not in removed_indices
            ]

        return scheduled_arm_actions
    
    def get_action_from_state_transition(self, prev_state:dict,new_state:dict):
        '''
        return action (armID, objID, 'G')
        cases:
        1. just started
            obj:S->obj:T
            arm:E->arm:G
        2. finish the action
            obj1:T->obj1:G
            obj2:S->obj2:T
            arm:G->arm:E
            arm:E->arm:G
        3. finish a quick task
            obj1:S->obj1:G
            obj2:T->obj2:T
            arm:E->arm:E
            arm:G->arm:G
        4. handoff
            obj1:T->obj1:T
            arm:H->arm:E
            arm:E->arm:G
        '''
        obj = None
        armID = None
        target = None
        # check arms
        if (prev_state['arm0'] != new_state['arm0']) and (new_state['arm0'] == 'E'): # case 2,4
            armID = 0
            target = prev_state['arm0']
        if (prev_state['arm1'] != new_state['arm1']) and (new_state['arm1'] == 'E'): # case 2,4
            armID = 1
            target = prev_state['arm1']
        if armID != None: # case 2,4
            for k,status in prev_state.items():
                if status == 'T':
                    obj = k
                    break
            else:
                raise ValueError('there is no object for the action')
            return (armID, obj, target)
        else: # case 1,3
            for k in prev_state.keys():
                if type(k) == str: # arm
                    continue
                if (prev_state[k] != new_state[k]) and (new_state[k]!= 'T'): # changed and not just started
                    obj = k
                    target = new_state[k]
                    break
            else: # case 1
                return None
            # case 3
            available_arms = set([0,1])
            for status in [prev_state[obj], new_state[obj]]:
                if status == 'S':
                    available_arms = available_arms.intersection(self.specify_region(self.start_arr[obj]))
                elif status == 'G':
                    available_arms = available_arms.intersection(self.specify_region(self.goal_arr[obj]))
                elif status in ['0','1']:
                    available_arms = available_arms.intersection(set([int(status)]))
            if prev_state['arm0'] != 'E':
                available_arms = available_arms.difference(set([0]))
            if prev_state['arm1'] != 'E':
                available_arms = available_arms.difference(set([1]))
            
            assert len(available_arms) > 0
            armID = list(available_arms)[0]
            return (armID, obj, target)
            

        

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
        if current_status.holding != None: # holding an obj
            holding_arm, holding_obj, target = current_status.holding
            empty_arm = 1-holding_arm
            if target == 'H': # handoff sth
                if self.goal_available(
                    holding_obj, 
                    current_state
                    ): # handoff to goal
                    yield self.generate_new_state_from_action(
                        current_state, 
                        current_status, 
                        (empty_arm, holding_obj, 'G')
                        )
                else: # handoff to buffer
                    yield self.generate_new_state_from_action(
                        current_state, 
                        current_status, 
                        (empty_arm, holding_obj, str(empty_arm))
                        )
            else: # pick-n-place objs
                # think about the action of the other arm
                # do nothing
                new_state = copy.deepcopy(current_state)
                new_state['arm'+str(holding_arm)] = 'E'
                new_state[holding_obj] = target # not 'H'
                new_status = copy.deepcopy(current_status)
                new_status.holding = None
                new_status.just_placed = current_status.holding
                new_status.holding_time = 0
                new_status.estimation_time = 0
                if target == 'G':
                    holding_arm_current_pose = self.goal_arr[holding_obj]
                else:
                    holding_arm_current_pose = target # buffer
                new_status.last_finish_positions[holding_arm] = holding_arm_current_pose
                if self.is_goal(new_state):
                    new_status.g += self.compute_final_transit(current_status, new_status)
                yield self.compress_state_dict(new_state), new_status

                for obj, status in current_state.items():
                    if(type(obj) is str): # key is an arm
                        continue
                    if status == 'G': # at goal already
                        continue
                    if status == 'T': # the other arm is holding it
                        continue
                    elif (status == 'S'):
                        # obj at start
                        current_arm_option_set = self.specify_region(self.start_arr[obj])
                    else: # at buffer
                        current_arm_option_set = {int(status)}
                    goal_arm_option_set = self.specify_region(self.goal_arr[obj])
                    intersection = current_arm_option_set.intersection(goal_arm_option_set)
                    if (intersection == set()) and (empty_arm in current_arm_option_set): # need handoff
                        if (status not in {'G', 'T', 'S'} and (not self.goal_available(obj, current_state))):
                            continue
                        # execute a handoff
                        yield self.generate_new_state_from_action(
                            current_state, 
                            current_status, 
                            (empty_arm, obj, 'H')
                            )
                        if status == 'S':
                            yield self.generate_new_state_from_action(
                                current_state, 
                                current_status, 
                                (empty_arm, obj, str(empty_arm))
                                )
                    elif( empty_arm in intersection): # reachable pick and place
                        if self.goal_available(
                            obj, 
                            current_state
                            ): # to goal
                            yield self.generate_new_state_from_action(
                                current_state, 
                                current_status, 
                                (empty_arm, obj, 'G')
                                )
                        elif status == 'S': # handoff to buffer
                            yield self.generate_new_state_from_action(
                                current_state, 
                                current_status, 
                                (empty_arm, obj, str(empty_arm))
                            )

        else: # neither arm is holding obj
            for obj, status in current_state.items():
                if(type(obj) is str): # key is an arm
                    continue
                if status == 'T':
                    raise TypeError('empty hands but this obj is holded')
                if status == 'G': # at goal already
                    continue
                if status == 'S': # at start
                    current_arm_option_set = self.specify_region(self.start_arr[obj])
                else:
                    current_arm_option_set = {int(status)}
                goal_arm_option_set = self.specify_region(self.goal_arr[obj])
                intersection = current_arm_option_set.intersection(goal_arm_option_set)
                if intersection == set(): # need handoff
                    current_arm = list(current_arm_option_set)[0]
                    # execute a handoff
                    yield self.generate_new_state_from_action(
                        current_state, 
                        current_status, 
                        (current_arm, obj, 'H')
                        )
                    if status == 'S':
                        yield self.generate_new_state_from_action(
                            current_state, 
                            current_status, 
                            (current_arm, obj, str(current_arm))
                            )
                elif self.goal_available(obj, current_state): # move to goal
                    for arm in intersection:
                        yield self.generate_new_state_from_action(
                            current_state, 
                            current_status, 
                            (arm, obj, 'G')
                            )
                elif status == 'S': # move to buffer if at start
                    for arm in intersection:
                        yield self.generate_new_state_from_action(
                            current_state, 
                            current_status, 
                            (arm, obj, str(arm))
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
        else:
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
        if target == 'H': # if passing handoff
            new_state[new_obj] = 'T'
            new_state['arm'+str(new_arm)] = 'H'
            new_status.holding = action
            if prev_status.holding != None: # there was an obj in the other hand when the new action was executed
                old_arm, old_obj, old_target = prev_status.holding
                if old_obj != new_obj:
                    new_state[old_obj] = old_target # old target must not be 'H'
                if new_arm != old_arm:
                    new_state['arm'+str(old_arm)] = 'E'

                new_status.just_placed = prev_status.holding
                # determine the transit cost of the other arm to do the handoff
                if old_target == 'G':
                    old_arm_current_pose = self.goal_arr[old_obj]
                else:
                    old_arm_current_pose = old_target # buffer
            else:
                old_arm_current_pose = prev_status.last_finish_positions[1-new_arm]
                new_status.just_placed = prev_status.just_placed
            if old_arm_current_pose == 'H':
                old_arm_transit_cost = 0
            elif type(old_arm_current_pose) is str:
                old_arm_transit_cost = self.cost_constants['from_to_buffer']
            else:
                old_arm_transit_cost = self.transit_cost(
                    old_arm_current_pose,
                    self.handoff_pose
                )
            left_arm_cost = 0
            right_arm_cost = 0
            if prev_status.gap >= 0:
                right_arm_cost += prev_status.gap
            else:
                left_arm_cost -= prev_status.gap
            if new_arm == 0:
                left_arm_cost += new_action_cost
                right_arm_cost += old_arm_transit_cost
            else:
                left_arm_cost += old_arm_transit_cost
                right_arm_cost += new_action_cost            
            new_status.g = prev_status.g + max(left_arm_cost, right_arm_cost)
            new_status.holding_time = new_action_cost
            new_status.estimated_time = new_action_cost
            new_status.gap = 0
            new_status.last_finish_positions = {
                0:self.handoff_pose,
                1:self.handoff_pose
            }
        elif (prev_status.holding != None): # new action when the other arm is holding
            old_arm, old_obj, old_target = prev_status.holding
            if new_action_cost >= (prev_status.estimated_time-prev_status.holding_time): # new action take longer to execute
                new_state['arm'+str(old_arm)] = 'E'
                new_state['arm'+str(new_arm)] = target
                if old_target != 'H': # new obj != old obj
                    new_state[old_obj] = old_target
                new_state[new_obj] = 'T'
                new_status.g = prev_status.g + (new_action_cost-(prev_status.estimated_time-prev_status.holding_time))
                new_status.holding = action
                new_status.just_placed = prev_status.holding
                new_status.estimated_time = new_action_cost
                if new_arm == 0:
                    new_status.holding_time = max(0,-prev_status.gap)
                    new_status.gap = new_status.estimated_time - new_status.holding_time
                else:
                    new_status.holding_time = max(0,prev_status.gap)
                    new_status.gap = -(new_status.estimated_time - new_status.holding_time)
                if old_target == 'G':
                    old_arm_current_pose = self.goal_arr[old_obj]
                else:
                    old_arm_current_pose = old_target # buffer
                new_status.last_finish_positions = {
                    old_arm:old_arm_current_pose,
                    new_arm:prev_status.last_finish_positions[new_arm]
                }
            else: # old action takes longer to execute
                new_state[new_obj] = target
                new_status.g = prev_status.g
                new_status.just_placed = action
                new_status.holding_time = prev_status.holding_time + new_action_cost
                if new_arm == 0:
                    new_status.gap = new_status.estimated_time - new_status.holding_time
                else:
                    new_status.gap = -(new_status.estimated_time - new_status.holding_time)
                new_status.last_finish_positions = {
                    old_arm:prev_status.last_finish_positions[old_arm],
                    new_arm:target_pose
                }
        else: # no holding
            new_state['arm'+str(new_arm)] = target
            new_state[new_obj] = 'T'
            if (new_arm == 0) and (prev_status.gap>=0):
                new_status.gap = prev_status.gap + new_action_cost
                new_status.g = prev_status.g + new_action_cost
            elif (new_arm == 0) and (prev_status.gap<0):
                new_status.gap = new_action_cost + prev_status.gap
                new_status.g = prev_status.g + max(0,new_status.gap)
            elif (new_arm == 1) and (prev_status.gap>=0):
                new_status.gap = prev_status.gap - new_action_cost
                new_status.g = prev_status.g + max(0, -new_status.gap)
            else: # (new_arm == 1) and (prev_status.gap<0)
                new_status.gap = prev_status.gap - new_action_cost
                new_status.g = prev_status.g + new_action_cost
            
            new_status.holding = action
            new_status.just_placed = prev_status.just_placed
            new_status.estimated_time = new_action_cost
            if new_arm == 0:
                new_status.holding_time = 0
                new_status.gap = new_status.estimated_time - new_status.holding_time
            else:
                new_status.holding_time = 0
                new_status.gap = -(new_status.estimated_time - new_status.holding_time)
            new_status.last_finish_positions = copy.deepcopy(prev_status.last_finish_positions)
        
        if self.is_goal(new_state):
            new_status.g += self.compute_final_transit(prev_status, new_status)
        
        return self.compress_state_dict(new_state), new_status
    
    def compute_final_transit(self, prev_status, new_status):
        last_arm, last_obj, _ = new_status.just_placed
        last_arm_cost = self.transit_cost(
            self.goal_arr[last_obj], 
            self.rest_pose[last_arm]
        )
        other_arm = 1-last_arm
        other_arm_cost = -abs(new_status.gap)
        # get last obj of the other arm
        Found = False
        while 1:
            if prev_status.just_placed == None:
                return last_arm_cost
            arm,obj,target = prev_status.just_placed
            if arm == other_arm:
                Found = True
                if target == 'H':
                    other_arm_cost += self.transit_cost(
                        self.handoff_pose, self.rest_pose[other_arm]
                    )
                else:
                    other_arm_cost += self.transit_cost(
                        self.handoff_pose, self.rest_pose[other_arm]
                    )
                return max(last_arm_cost, other_arm_cost)
            if prev_status.parent == None:
                return last_arm_cost
            prev_status = self.state_dict[prev_status.parent]
        


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
        arm_counts = {id:0 for id in range(self.num_arms)}
        flexible_actions = 0 # actions that can be done by either arm
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
                current_arm_option_set = self.specify_region(self.start_arr[obj])
                cost += self.transfer_cost(self.start_arr[obj], self.goal_arr[obj]) + self.cost_constants['grasp'] 
            elif state == 'H':
                current_arm_option_set = {0,1}
                cost += self.transfer_cost(self.handoff_pose, self.goal_arr[obj])
            else: # at a buffer
                current_arm_option_set = {int(state)}
                cost += self.cost_constants['from_to_buffer'] + self.cost_constants['grasp']
            goal_arm_option_set = self.specify_region(self.goal_arr[obj])
            intersection = current_arm_option_set.intersection(goal_arm_option_set)
            if len(intersection) == 2: # flexible action
                flexible_actions += cost
            elif len(intersection) == 1:
                arm_counts[list(intersection)[0]] += cost
            elif len(intersection) == 0:
                pass_arm = list(current_arm_option_set)[0]
                receive_arm = list(goal_arm_option_set)[0]
                if state == 'S':
                    arm_counts[pass_arm] += self.transfer_cost(self.start_arr[obj], self.handoff_pose) + self.cost_constants['grasp']
                else:
                    arm_counts[pass_arm] += self.cost_constants['from_to_buffer'] + self.cost_constants['grasp']
                arm_counts[receive_arm] += self.transfer_cost(self.handoff_pose, self.goal_arr[obj]) + self.cost_constants['release']
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
        prop = pose[0]/self.Width
        if prop < (0.5*(1-self.overlap_rate)):
            return {0}
        elif prop > (0.5*(1+self.overlap_rate)):
            return {1}
        else:
            return {0,1}

