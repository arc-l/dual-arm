import curobo
import timeout_decorator
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
from task import Task, GripperAction, TaskPlan

class Baseline(DualArmEnv):
    def __init__(self, task: Task, headless=True, display=False):
        self.headless_baseline = headless
        super().__init__(task, headless=not display, batch_size = 3)
    
    def convert_to_baseline_task_planner_space(self, pos):
        pos = 1000 * pos/self.task.map_width + 500
        pos = pos.tolist()[:2]
        return pos
    
    def convert_to_motion_planner_space(self, pos):
        res = np.zeros(3)
        res[0] = (pos[0] - 500) * self.task.map_width / 1000
        res[1] = (pos[1] - 500) * self.task.map_width / 1000
        res[2] = self.task.object_size/2
        return res

    def prepare_input(self):
        starts = self.task.get_original_starts().copy()
        goals = self.task.get_goals().copy()
        start_arr = {}
        goal_arr = {}
        for i in range(len(starts)):
            start_arr[i] = self.convert_to_baseline_task_planner_space(starts[i])
            goal_arr[i] = self.convert_to_baseline_task_planner_space(goals[i])
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
                    goal = self.convert_to_baseline_task_planner_space(plan.buffer[1])
                    start_arr[plan.arm1_assigned_object] = self.convert_to_baseline_task_planner_space(plan.buffer[1])
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
                    goal = self.convert_to_baseline_task_planner_space(plan.buffer[2])
                    start_arr[plan.arm2_assigned_object] = self.convert_to_baseline_task_planner_space(plan.buffer[2])
                action_list[1].append((
                    plan.arm2_assigned_object,
                    start,
                    goal
                ))
            else:
                action_list[1].append(None)

        return action_list


    def baseline_task_plan(self):
        start_arr, goal_arr = self.prepare_input()
        Height = 1000
        Width = 1000
        size = 41.7 * np.sqrt(2) # 0.025 / 0.6 * 1000 * np.sqrt(2)
        size = self.task.object_size/2 / self.task.map_length * 1000 # 0.025 / 0.6 * 1000 * np.sqrt(2)
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
        return action_list


    @timeout_decorator.timeout(1000)
    def plan(self, use_our_tp=False, use_our_mp=False):
        self.reset()
        start_arr, goal_arr = self.prepare_input()
        Height = 1000
        Width = 1000
        control_mode = ControlMode.POSITION
        # size = 37.5 # 0.0225 / 0.6 * 1000
        size = 41.7 * np.sqrt(2) # 0.025 / 0.6 * 1000 * np.sqrt(2)
        size = self.task.object_size/2 / self.task.map_length * 1000 # 0.025 / 0.6 * 1000 * np.sqrt(2)
        time_step_size = 0.02
        speed = 0.3141592653589793 * 5
        method = 'baseline_v2'
        if use_our_tp:
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
        if use_our_mp:
            self.motion_plan(action_list)
            return self.recorded_plan
        else:
            result_dict, control = simulation(
                Width, Height,start_arr, goal_arr, size, action_list, 
                overlap_rate = 1.0, speed=speed, 
                time_step_size=time_step_size, control_mode=control_mode, 
                gui=not self.headless_baseline, method=method, debug = True)
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
                cmd_plan = cmd_plan.get_ordered_joint_state(self.j_names_primary)
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


    def motion_plan(self, action_list):
        for i in range(max(len(action_list[0]), len(action_list[1]))):
            # transfer the action to a task_plan - to go to the start
            if i < len(action_list[0]):
                arm1_object = action_list[0][i][0]
            else:
                arm1_object = None

            if i < len(action_list[1]): 
                arm2_object = action_list[1][i][0]
            else:
                arm2_object = None

            task_plan_to_start = TaskPlan(
                start_goal_pairs=None if (arm1_object is None or arm2_object is None) else [(arm1_object, arm2_object)],
                arm1_assigned_object=arm1_object if arm1_object is not None else None,
                arm2_assigned_object=arm2_object if arm1_object is not None else None,
                arm1_gripper_action=GripperAction.CLOSE,
                arm2_gripper_action=GripperAction.CLOSE,
                arm1_to_start=True,
                arm2_to_start=True,
                buffer=None,
                use_buffer=False,
                arm1_angle=900 if arm1_object is None else None,
                arm2_angle=900 if arm2_object is None else None
            )
            task_plan_to_start = super().filter_task_plans(task_plan_to_start)
            # place the object in the goal
            task_execution_result = super().motion_plan(task_plan_to_start)
            self.old_task_plan = task_plan_to_start
            if task_execution_result is None:
                print("No plan found.")
                exit(1)
            self.task.update(task_execution_result)
        
            if arm1_object is not None:
                self.task.buffer_dict[arm1_object] = self.convert_to_motion_planner_space(action_list[0][i][4])
            if arm2_object is not None:
                self.task.buffer_dict[arm2_object] = self.convert_to_motion_planner_space(action_list[1][i][4])

            task_plan_to_goal = TaskPlan(
                start_goal_pairs=None if (arm1_object is None or arm2_object is None) else [(arm1_object, arm2_object)],
                arm1_assigned_object=arm1_object if arm1_object is not None else arm2_object,
                arm2_assigned_object=arm2_object if arm1_object is not None else None,
                arm1_gripper_action=GripperAction.OPEN,
                arm2_gripper_action=GripperAction.OPEN,
                arm1_angle = task_plan_to_start.arm1_angle,
                arm2_angle = task_plan_to_start.arm2_angle,
                arm1_to_start=False,
                arm2_to_start=False,
                buffer=None,
                use_buffer=False,
            )
            task_execution_result = super().motion_plan(task_plan_to_goal)
            self.old_task_plan = task_plan_to_goal
            self.task.update(task_execution_result)
        
        return self.recorded_plan








if __name__ == '__main__':
    # task = Task.read_from_json('data/task.json')
    num_objects:int = 4  
    map_width:float = 0.6 
    map_length:float = 0.6 
    object_size:float = 0.045 
    task = Task(num_objects, map_width, map_length, object_size)
    task = Task.read_from_json('data/cycled_objects_4.json')
    # task.generate(cycled=True)
    # task.generate_fixed_two_objects(offset=0.)
    # task.generate_chained_two_objects()
    # task.generate_fixed_cycled_objects(radius=0.2)
    # task.generate_fixed_two_objects_to_test_grasp_orientation()
    # task.generate_fixed_two_to_test_batch_planner()
    # task.generate_fixed_four_objects()
    # task.generate_chained_four_objects()
    # task.generate_fixed_six_objects()
    dual_arm = Baseline(task, display=False, headless=False)
    recorded_plan = dual_arm.plan()
    dual_arm.play(recorded_plan=recorded_plan)
