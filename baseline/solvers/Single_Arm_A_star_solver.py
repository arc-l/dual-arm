import itertools
import sys
import math
# sys.path.append("/home/kai/Documents/Kai_Gao/dual_arm/limited_reachability/")
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



class Single_Arm_A_star_solver(high_level_solver):
    '''
    rearrangement in a two-region scene, but only one robot moves in each time step
    '''
    def __init__(self, Width, Height, numObjs, radius, start_arr, goal_arr) -> None:
        super().__init__(Width, Height, numObjs, radius, start_arr, goal_arr, num_arms=2, overlap_rate=0.3)

        self.DG = construct_DG(start_arr, goal_arr, radius)

    def A_star(self):
        '''
        A star search
        '''
        self.state_dict = {} # key: compressed_state, value: arr_status class
        self.open_list = [] # [compressed_state] decrease order
        self.parents = {}
        start_state = {k:'S' for k in self.start_arr.keys()}
        compressed_start_state = self.compress_state_dict(start_state)
        start_status = arr_status(0)
        self.state_dict[compressed_start_state] = start_status
        start_status.h = self.heuristic(compressed_start_state)
        start_status.g = 0
        start_status.f = start_status.h
        start_status.in_open = True
        self.open_list.append( compressed_start_state)
        self.state_dict[compressed_start_state].in_open = True
        FindGoal = False # Found goal pose
        while (len(self.open_list) >0) and (not FindGoal):
            # pop out (from the list bottom) a state to develop
            current_cpd_state = self.open_list.pop()
            current_g = self.state_dict[current_cpd_state].g
            self.state_dict[current_cpd_state].in_open = False
            self.state_dict[current_cpd_state].in_closed = True
            if self.is_goal(current_cpd_state):
                FindGoal = True
                continue
            for new_cpd_state in self.generate_new_nodes(current_cpd_state):
                # visited state
                if new_cpd_state in self.state_dict.keys():
                    new_status = self.state_dict[new_cpd_state]
                    # in the closed set
                    if new_status.in_closed :
                        continue
                    # in the open set
                    if new_status.in_open:
                        # found a shorter path
                        if new_status.g > current_g + 1:
                            new_status.g = current_g + 1
                            self.remove_state_from_open(new_cpd_state)
                            new_status.f = new_status.g + new_status.h # h has been computed
                            self.add_new_state_to_open(new_cpd_state)
                            # self.resort_in_open_list(new_cpd_state) # can have a higher ranking
                            new_status.parent = current_cpd_state
                        else: # not a better parent
                            continue
                    else: # should not happen, but add it the the open list for safety
                        raise TypeError('Unexpected situation')
                        new_status.parent = current_cpd_state
                        new_status.in_open = True
                        self.add_new_state_to_open(new_cpd_state)
                else: # new state
                    status = arr_status(len(self.state_dict))
                    self.state_dict[new_cpd_state] = status
                    status.g = current_g + 1
                    status.h = self.heuristic(new_cpd_state)
                    status.f = status.g + status.h
                    status.in_open = True
                    status.parent = current_cpd_state
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
        arm_actions = {arm:[] for arm in range(self.num_arms)}
        # print('-----------------')
        # for state in state_list:
        #     print(self.decompress_state(state))
        for i in range(len(state_list)-1):
            previous_state = state_list[i]
            new_state = state_list[i+1]
            new_actions = self.state_transition_to_actions(previous_state, new_state)
            arm_actions[0].extend(new_actions)
            arm_actions[1].append(None)
        return arm_actions
    

    def state_transition_to_actions(self, previous_state, new_state):
        '''
        Input: two compressed states
        Output: actions:{0:[(21, 'S', 'G')], 1:[(45, '0', 'H')]}
        '''
        arm_actions = []
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
            arm_actions.append((obj, prev_status, new_status))
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
        # besides handoffs, each arm has other action options, None means doing nothing
        arm_options = []
        current_state = self.decompress_state(cpd_current_state)
        for obj, status in current_state.items():
            if status == 'G': # we can do nothing for objs at goal
                continue
            elif self.goal_available(obj, current_state):
                arm_options.append((obj, 'G'))
            else:
                arm_options.append((obj,'0'))
        for action in arm_options:
            # deepcopy
            new_state = {k:value for k, value in current_state.items()}
            new_state[action[0]] = action[1]
            yield self.compress_state_dict(new_state)

                
    
    def goal_available(self, obj:int, current_state):
        for obs in self.DG[obj]:
            if current_state[obs] == 'S':
                return False
        else:
            return True

        

    def heuristic(self, current_state):
        # number of actions that each hand has to execute
        current_state = self.decompress_state(current_state)
        return sum([1 for v in current_state.values() if v != 'G'])
        


    def specify_region(self, pose):
        ''' specify reachability '''
        prop = pose[0]/self.Width
        if pose[0] < (0.5*(1-self.overlap_rate)*self.Width):
            return {0}
        elif pose[0] > (0.5*(1+self.overlap_rate)*self.Width):
            return {1}
        else:
            return {0,1}