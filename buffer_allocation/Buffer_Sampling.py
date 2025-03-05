from random import random, shuffle, uniform

class Buffer_Sampling(object):
    def __init__(self, partial_arm_actions :dict, partial_buffers, partial_schedule, start_arr, goal_arr, Height, Width, radius, overlap_rate = 1.0):
        self.partial_arm_actions = partial_arm_actions
        self.partial_buffers = partial_buffers
        self.partial_schedule = partial_schedule
        self.start_arr = start_arr
        self.goal_arr = goal_arr
        self.Height = Height
        self.Width = Width
        self.radius = radius
        self.overlap_rate = overlap_rate

        # debug
        self.num_collision = 0

        # Increamentally add constraints
        self.buffer_locations, self.trouble_action_index = self.partial_optimization()

    def partial_optimization(self):
        current_buffers = [] #[(armid, objid)]
        current_at_start = []
        current_at_goal = []
        obs_dict = {}
        buffer_locations = {}

        current_at_start = list(self.start_arr.keys())

        for time_step in range(len(self.partial_arm_actions[0])):
            arm0_action = self.partial_arm_actions[0][time_step]
            arm1_action = self.partial_arm_actions[1][time_step]
            actions = []
            if (arm0_action != None) and (arm1_action != None) and (arm0_action[0]==arm1_action[0]): #handoff
                if arm0_action[1] == 'H': # 0 moves obj from handoff
                    action = (0, arm0_action[0], arm1_action[1], arm0_action[2])
                else:
                    action = (1, arm0_action[0], arm0_action[1], arm1_action[2])
                actions.append(action)
            else:
                if arm0_action != None:
                    actions.append(tuple([0]+list(arm0_action)))
                if arm1_action != None:
                    actions.append(tuple([1]+list(arm1_action)))
            if len(actions) == 1:
                action = actions[0]
                current_objID = action[1]
                if action[3] in {'0', '1'}: # from start to buffer
                    current_buffers.append((action[0],current_objID))
                    # buffer_locations[current_objID] = self.start_arr[current_objID]
                    buffer_locations[current_objID] = self.reachable_buffer_sampling(action[0])
                    obs_dict[current_objID] = []
                    for obj_id in current_at_start:
                        if obj_id == current_objID:
                            continue
                        else:
                            obs_dict[current_objID].append(self.start_arr[obj_id])
                    for obj_id in current_at_goal:
                        obs_dict[current_objID].append(self.goal_arr[obj_id])
                elif current_objID in self.partial_buffers: # from buffer to goal
                    current_at_goal.append(current_objID)
                    for _, obj_id in current_buffers:
                        if obj_id == current_objID:
                            continue
                        else:
                            obs_dict[obj_id].append(self.goal_arr[current_objID])
                else: # from start to goal
                    current_at_goal.append(current_objID)
                    for _, obj_id in current_buffers:
                        obs_dict[obj_id].append(self.goal_arr[current_objID])
            
            else: # two actions:
                if (arm0_action[2] in {'0', '1'}) and (arm1_action[2] in {'0', '1'}): # both to buffer
                    obj0 = arm0_action[0]
                    obj1 = arm1_action[0]
                    current_buffers.append((0,obj0))
                    current_buffers.append((1,obj1))
                    buffer_locations[obj0] = self.reachable_buffer_sampling(0)
                    buffer_locations[obj1] = self.reachable_buffer_sampling(1)
                    obs_dict[obj0] = []
                    obs_dict[obj1] = []
                    for obj_id in current_at_start:
                        if obj_id in {obj1, obj0}:
                            continue
                        else:
                            obs_dict[obj0].append(self.start_arr[obj_id])
                            obs_dict[obj1].append(self.start_arr[obj_id])
                    for obj_id in current_at_goal:
                        obs_dict[obj0].append(self.goal_arr[obj_id])
                        obs_dict[obj1].append(self.goal_arr[obj_id])
                elif ((arm0_action[2] == 'G') and (arm1_action[2] == 'G')):
                    obj0 = arm0_action[0]
                    obj1 = arm1_action[0]
                    current_at_goal.append(obj0)
                    current_at_goal.append(obj1)
                    for _, obj_id in current_buffers:
                        if obj_id in {obj0, obj1}:
                            continue
                        else:
                            obs_dict[obj_id].append(self.goal_arr[obj0])
                            obs_dict[obj_id].append(self.goal_arr[obj1])
                else: # one to buffer, one to goal
                    if arm0_action[2] == 'G':
                        to_goal_arm = 0
                        to_goal_obj = arm0_action[0]
                        to_buffer_arm = 1
                        to_buffer_obj = arm1_action[0]
                    else:
                        to_goal_arm = 1
                        to_goal_obj = arm1_action[0]
                        to_buffer_arm = 0
                        to_buffer_obj = arm0_action[0]
                    current_buffers.append((to_buffer_arm, to_buffer_obj))
                    buffer_locations[to_buffer_obj] = self.reachable_buffer_sampling(to_buffer_arm)
                    obs_dict[to_buffer_obj] = []
                    for obj_id in current_at_start:
                        if obj_id in {to_buffer_obj, to_goal_obj}:
                            continue
                        else:
                            obs_dict[to_buffer_obj].append(self.start_arr[obj_id])
                    for obj_id in current_at_goal:
                        obs_dict[to_buffer_obj].append(self.goal_arr[obj_id])
                    current_at_goal.append(to_goal_obj)
                    for _, obj_id in current_buffers:
                        if obj_id == to_goal_obj:
                            continue
                        else:
                            obs_dict[obj_id].append(self.goal_arr[to_goal_obj])

            # check feasibility
            isfeasible, new_buffer_location_dict = self.buffer_sampling(current_buffers, buffer_locations, obs_dict)
            
            if not isfeasible:
                return buffer_locations, time_step
            else:
                # update buffer locations
                for buffer in new_buffer_location_dict.keys():
                    buffer_locations[buffer] = new_buffer_location_dict[buffer]
                # update obj status
                for action in actions:
                    if (action[3] in {'0', '1'}): # from start to buffer
                        current_at_start.remove(action[1])
                    elif action[1] in self.partial_buffers: # from buffer to goal
                        for i, arm_buffer in enumerate(current_buffers):
                            if arm_buffer[1] == action[1]:
                                del current_buffers[i]
                        del obs_dict[action[1]]
                        for _, obj_id in current_buffers:
                            obs_dict[obj_id].append(buffer_locations[action[1]])
                    else: # from start to goal
                        current_at_start.remove(action[1])
        return buffer_locations, float('inf')

    def buffer_sampling(self, buffers, buffer_current_location_dict, obs_dict):
        isfeasible = False

        # # check previous locations
        # new_buffer_dict = {}
        # for bufferID in buffers:
        #     obs_list = obs_dict[bufferID] + list(new_buffer_dict.values())
        #     if not self.disk_collision_check(buffer_current_location_dict[bufferID], obs_list):
        #         new_buffer_dict[bufferID] = buffer_current_location_dict[bufferID]
        #     else:
        #         break
        # else:
        #     isfeasible = True

        timeout = 100
        while (timeout > 0) and (not isfeasible):
            timeout -= 1
            previous_buffer_feasible = True
            new_buffer_dict = {}
            shuffle(buffers)
            for armID, bufferID in buffers:
                if not previous_buffer_feasible: # previous buffer is not placed
                    break
                previous_buffer_feasible = False
                inner_timeout = 10
                obs_list = obs_dict[bufferID] + list(new_buffer_dict.values())
                
                # check the current position
                if not self.disk_collision_check(buffer_current_location_dict[bufferID], obs_list):
                    previous_buffer_feasible = True 
                    new_buffer_dict[bufferID] = buffer_current_location_dict[bufferID]
                    continue
                # generate new positions
                while inner_timeout >= 0:
                    inner_timeout -= 1
                    new_location = self.reachable_buffer_sampling(armID)
                    if not self.disk_collision_check(new_location, obs_list):
                        previous_buffer_feasible = True 
                        new_buffer_dict[bufferID] = new_location
                        break
            isfeasible = previous_buffer_feasible # the last buffer is feasible
        
        return isfeasible, new_buffer_dict
    
    def disk_collision_check(self, obj, obs_list):
        isColliding = False
        for obs in obs_list:
            self.num_collision += 1
            if ((obj[0]-obs[0])**2+(obj[1]-obs[1])**2<=4*self.radius**2):
                isColliding = True 
                break
        return isColliding

    def reachable_buffer_sampling(self, armID):
        if armID == 0:
            return (
                uniform(self.radius, min(self.Width*0.5*(1+self.overlap_rate), self.Width-self.radius)),
                uniform(self.radius, self.Height-self.radius)
            )
        else:
            return (
                uniform(max(self.radius, self.Width*0.5*(1-self.overlap_rate)), self.Width-self.radius),
                uniform(self.radius, self.Height-self.radius)
            )
            