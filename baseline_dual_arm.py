import curobo
from curobo.types.state import JointState
import os
os.environ['CUROBO_TORCH_CUDA_GRAPH_RESET'] = '1'
import random
import torch

import numpy as np
import time
from dual_arms import DualArmEnv
from baseline.motion_planning.simulation import simulation
from baseline.high_level_planners.Bi_Directional_Search_Tree_Planner import Bi_Directional_Search_Tree_Planner
from baseline.experiments import plan2schedule
from baseline.motion_planning.robot_arm import ControlMode
from task import Task, GripperAction

class Baseline(DualArmEnv):
    def __init__(self, task: Task):
        super().__init__(task, headless=True, batch_size = 3, constrain_grasp_approach=False)
    
    @staticmethod
    def convert_to_task_planner_space(pos):
        pos = 1000 * pos/0.6 + 500
        pos = pos.tolist()[:2]
        return pos

    def prepare_input(self):
        starts = self.task.get_original_starts().copy()
        goals = self.task.get_goals().copy()
        start_arr = {}
        goal_arr = {}
        for i in range(len(starts)):
            start_arr[i] = self.convert_to_task_planner_space(starts[i])
            goal_arr[i] = self.convert_to_task_planner_space(goals[i])
        return start_arr, goal_arr
    
    def plan_all(self):
        task_plans = super().plan_all()
        start_arr, goal_arr = self.prepare_input()
        action_list = {0:[], 1:[]}
        for plan in task_plans:
            if plan.arm1_gripper_action==GripperAction.OPEN:
                continue
            if plan.arm1_assigned_object != -1:
                start = start_arr[plan.arm1_assigned_object]
                goal = goal_arr[plan.arm1_assigned_object] 
                if plan.buffer is not None and plan.buffer[1] is not None:
                    goal = self.convert_to_task_planner_space(plan.buffer[1])
                    start_arr[plan.arm1_assigned_object] = self.convert_to_task_planner_space(plan.buffer[1])
                action_list[0].append((
                    plan.arm1_assigned_object,
                    start,
                    goal,
                ))
            else:
                action_list[0].append(None)
            if plan.arm2_assigned_object != -1:
                start = start_arr[plan.arm2_assigned_object]
                goal = goal_arr[plan.arm2_assigned_object] 
                if plan.buffer is not None and plan.buffer[2] is not None:
                    goal = self.convert_to_task_planner_space(plan.buffer[2])
                    start_arr[plan.arm2_assigned_object] = self.convert_to_task_planner_space(plan.buffer[2])
                action_list[1].append((
                    plan.arm2_assigned_object,
                    start,
                    goal
                ))
            else:
                action_list[1].append(None)

        return action_list



    def plan(self, use_our_planner=False):
        self.reset()
        start_arr, goal_arr = self.prepare_input()
        print("start_arr", start_arr)
        print("goal_arr", goal_arr)
        Height = 1000
        Width = 1000
        control_mode = ControlMode.POSITION
        # size = 37.5 # 0.0225 / 0.6 * 1000
        size = 41.7 * np.sqrt(2) # 0.025 / 0.6 * 1000 * np.sqrt(2)
        time_step_size = 0.02
        speed = 0.3141592653589793 * 5
        method = 'baseline_v2'
        if use_our_planner:
            action_list = self.plan_all()
        else:
            planner = Bi_Directional_Search_Tree_Planner(
                start_arr, 
                goal_arr, 
                Height, 
                Width, 
                size, 
                sampling=True, 
                overlap_rate=1.0, 
                num_arms=2,
                makespan='MC'
            )
            action_list = planner.action_list
        action_list = plan2schedule(action_list)
        result_dict, control = simulation(
            Width, Height,start_arr, goal_arr, size, action_list, 
            overlap_rate = 1.0, speed=speed, 
            time_step_size=time_step_size, control_mode=control_mode, 
            gui=True, method=method, debug = True)
        
        recorded_plan = result_dict['recorded_plan']
        recorded_plan = self.post_process(recorded_plan)
        return recorded_plan
        
    
    def post_process(self, recorded_plan):
        plan = []
        cmd_plan = []
        gripper_closed = {self.arm1: False, self.arm2: False}
        for time_step in range(len(recorded_plan[0])):
            # recorded_plan[0][time_step].joint_state[0] -= np.pi/2
            # recorded_plan[1][time_step].joint_state[0] += np.pi
            dual_position = torch.tensor(
                np.concatenate([recorded_plan[i][time_step].joint_state for i in range(2)],axis=-1), 
                dtype=torch.float32, device='cuda:0'
                ).view(1,-1)
            
            cmd_plan.append(dual_position)
            if recorded_plan[0][time_step].motion_status in ["A", "D"] or recorded_plan[1][time_step].motion_status in ["A", "D"]:
                cmd_plan = torch.cat(cmd_plan, dim=0)
                cmd_plan = JointState.from_position(self.tensor_args.to_device(cmd_plan), self.arm_joint_names)
                cmd_plan = self.single_motion_gen.get_full_js(cmd_plan)
                # get only joint names that are in both:
                if self.common_js_names == [] and self.common_js_indices == []:
                    for x in self.dof_names:
                        if x in cmd_plan.joint_names:
                            self.common_js_indices.append(self.robot.get_dof_index(x))
                            self.common_js_names.append(x)
            
                cmd_plan = cmd_plan.get_ordered_joint_state(self.common_js_names)
                plan.append(cmd_plan)
                cmd_plan = []
                
                grasp_control = []
                match recorded_plan[0][time_step].motion_status:
                    case "A":
                        grasp_control.append(GripperAction.CLOSE)
                    case "D":
                        grasp_control.append(GripperAction.OPEN)
                    case _:
                        if gripper_closed[self.arm1]:
                            grasp_control.append(GripperAction.CLOSE)
                        else:
                            grasp_control.append(GripperAction.OPEN)
                match recorded_plan[1][time_step].motion_status:
                    case "A":
                        grasp_control.append(GripperAction.CLOSE)
                    case "D":
                        grasp_control.append(GripperAction.OPEN)
                    case _:
                        if gripper_closed[self.arm2]:
                            grasp_control.append(GripperAction.CLOSE)
                        else:
                            grasp_control.append(GripperAction.OPEN)
                
                plan.append(grasp_control)
                if grasp_control[0] == GripperAction.CLOSE:
                    gripper_closed[self.arm1] = True
                else:
                    gripper_closed[self.arm1] = False
                if grasp_control[1] == GripperAction.CLOSE:
                    gripper_closed[self.arm2] = True
                else:
                    gripper_closed[self.arm2] = False
        return plan

if __name__ == '__main__':
    # task = Task.read_from_json('data/task.json')
    num_objects:int = 4  
    map_width:float = 0.6 
    map_length:float = 0.6 
    object_size:float = 0.045 
    task = Task(num_objects, map_width, map_length, object_size)
    # task.generate(cycled=True)
    # task.generate_fixed_two_objects(offset=0.)
    # task.generate_chained_two_objects()
    task.generate_fixed_cycled_objects(radius=0.2)
    # task.generate_fixed_two_objects_to_test_grasp_orientation()
    # task.generate_fixed_two_to_test_batch_planner()
    # task.generate_fixed_four_objects()
    # task.generate_chained_four_objects()
    # task.generate_fixed_six_objects()
    dual_arm = Baseline(task)
    recorded_plan = dual_arm.plan()
    dual_arm.play(recorded_plan=recorded_plan)
