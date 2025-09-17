import numpy as np
import os
import random
from copy import deepcopy
from itertools import permutations
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dependency_graph import DependencyGraph
from dataclasses import dataclass
from enum import Enum
import json
import os
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.oxml import parse_xml
from pptx.enum.shapes import MSO_CONNECTOR_TYPE

class TaskState(Enum):
    IN_PROGRESS = 1
    COMPLETED = 2

class GripperState(Enum):
    OPEN = 0
    CLOSE = 1
    def reverse(self):
        if self == GripperState.OPEN:
            return GripperState.CLOSE
        else:
            return GripperState.OPEN
    
    def reverse_action(self):
        if self == GripperState.OPEN:
            return GripperAction.CLOSE
        else:
            return GripperAction.OPEN

class GripperAction(Enum):
    OPEN = 0 
    CLOSE = 1
    def reverse(self):
        if self == GripperAction.OPEN:
            return GripperAction.CLOSE
        else:
            return GripperAction.OPEN
    
    def reverse_state(self):
        if self == GripperAction.OPEN:
            return GripperState.CLOSE
        else:
            return GripperState.OPEN

@dataclass
class TaskPlan:
    start_goal_pairs: list = None
    arm1_gripper_action: GripperAction = GripperAction.OPEN
    arm2_gripper_action: GripperAction = GripperAction.OPEN
    arm1_to_start: bool = True
    arm2_to_start: bool = True
    arm1_assigned_object: int = None
    arm2_assigned_object: int = None
    arm1_angle: int = None
    arm2_angle: int = None
    use_buffer: bool = False
    prioritize_arm1: bool = True
    buffer: dict = None

@dataclass
class TaskExecutionResult:
    arm1_gripper_state: GripperState  = None
    arm2_gripper_state: GripperState = None
    arm1_task_state: TaskState = None 
    arm2_task_state: TaskState = None 
    arm1_assigned_object: int = None
    arm2_assigned_object: int = None
    arm1_angle: int = None
    arm2_angle: int = None
    buffer: dict = None
    arm2_at_intermediate: bool = False
    arm1_at_intermediate: bool = False

@dataclass
class ArmState:
    gripper_state: GripperState = GripperState.OPEN
    task_state: TaskState = None
    assigned_object: int = None
    angle: int = None
    at_intermediate = False

class Task:
    def __init__(self, num_objects:int, map_width:float, map_length:float, object_size:float):
        self.num_objects = num_objects
        self.map_width = map_width
        self.map_length = map_length
        self.object_size = object_size
        self.starts = None
        self.goals = None
        self.original_starts = None
        self.colors = plt.cm.rainbow(np.linspace(0, 1, self.num_objects))
        # self.generate()
        self.start_dict = None
        self.goal_dict = None
        self.arm1_state = ArmState()
        self.arm2_state = ArmState()
        self.buffer_dict = {i: None for i in range(self.num_objects)}

    def build_start_goal_dict(self):
        self.start_dict = {}
        self.goal_dict = {}
        for i in self.idx_to_be_rearranged:
            self.start_dict[i] = self.starts[i]
            self.goal_dict[i] = self.goals[i] if self.buffer_dict[i] is None else self.buffer_dict[i]
    
    def generate_fixed_two_objects(self, offset:float = 0.0)->None:
        self.num_objects = 2
        self.buffer_dict = {i: None for i in range(self.num_objects)}
        self.starts = [
            np.array([-offset, -self.map_length/2 + 2 * self.object_size, self.object_size/2]), 
            np.array([offset, self.map_length/2 - 2 * self.object_size, self.object_size/2])
            ]
        self.original_starts = deepcopy(self.starts)
        self.goals = [
            np.array([-offset, self.map_length/2 - 2 * self.object_size, self.object_size/2]),
            np.array([offset, -self.map_length/2 + 2 * self.object_size, self.object_size/2]), 
            ]
        
        self.colors = plt.cm.rainbow(np.linspace(0, 1, self.num_objects))
        self.init()
    
    def generate_fixed_two_to_test_batch_planner(self)->None:
        self.num_objects = 2
        self.buffer_dict = {i: None for i in range(self.num_objects)}
        self.starts = [
            np.array([-self.map_length/2 + 2 * self.object_size, 0, self.object_size/2]), 
            np.array([self.map_length/2 - 2 * self.object_size, 0, self.object_size/2])
            ]
        self.original_starts = deepcopy(self.starts)
        self.goals = [
            np.array([self.map_length/2 - 2 * self.object_size, 0, self.object_size/2]),
            np.array([-self.map_length/2 + 2 * self.object_size, 0, self.object_size/2]), 
            ]
        
        self.colors = plt.cm.rainbow(np.linspace(0, 1, self.num_objects))
        self.init()
            
    
    def generate_chained_two_objects(self):
        self.num_objects = 2
        self.buffer_dict = {i: None for i in range(self.num_objects)}
        # self.starts = [
        #     np.array([0, -self.map_length/2 + 2 * self.object_size, self.object_size/2]), 
        #     np.array([0, self.map_length/2 - 2 * self.object_size, self.object_size/2])
        #     ]
        # self.goals = [
        #     np.array([0, self.map_length/2 - 2 * self.object_size, self.object_size/2]),
        #     np.array([0, 0, self.object_size/2]), 
        #     ]
        self.starts = [
            np.array([-self.map_width/2 + 2 * self.object_size, -self.map_width/2 + 2 * self.object_size, self.object_size/2]), 
            np.array([self.map_width/2 - 2 * self.object_size, -self.map_width/2 + 2 * self.object_size, self.object_size/2])
        ]
        self.original_starts = deepcopy(self.starts)
        self.goals = [
            np.array([self.map_width/2 - 2 * self.object_size, -self.map_width/2 + 2 * self.object_size, self.object_size/2]),
            np.array([self.map_width/2 - 2 * self.object_size, self.map_width/2 - 2 * self.object_size, self.object_size/2]),
        ]
        self.colors = plt.cm.rainbow(np.linspace(0, 1, self.num_objects))
        self.init()
        
    
    def generate_fixed_two_objects_to_test_grasp_orientation(self, offset:float = 0.0)->None:
        self.num_objects = 2
        self.buffer_dict = {i: None for i in range(self.num_objects)}
        self.starts = [
            np.array([-offset, -self.map_length/2 + 2 * self.object_size, self.object_size/2]), 
            np.array([offset, self.map_length/2 - 2 * self.object_size, self.object_size/2])
            ]
        self.goals = [
            np.array([0, 0.6* self.object_size, self.object_size/2]),
            np.array([0, -0.6 * self.object_size, self.object_size/2]), 
        ]
        # self.goals = [
        #     np.array([0.6* self.object_size, 0, self.object_size/2]),
        #     np.array([-0.6 * self.object_size, 0, self.object_size/2]), 
        # ]

        self.starts, self.goals = self.goals, self.starts
        self.original_starts = deepcopy(self.starts)
        self.colors = plt.cm.rainbow(np.linspace(0, 1, self.num_objects))
        self.init()

    def generate_fixed_two_objects_to_test_ee_sampling(self, offset:float = 0.0)->None:
        self.num_objects = 2
        self.buffer_dict = {i: None for i in range(self.num_objects)}
        self.starts = [
            np.array([-offset, -self.map_length/2 + 2 * self.object_size, self.object_size/2]), 
            np.array([offset, self.map_length/2 - 2 * self.object_size, self.object_size/2])
            ]
        self.goals = [
            np.array([0.6* self.object_size, 0, self.object_size/2]),
            np.array([-0.6 * self.object_size, 0, self.object_size/2]), 
        ]

        self.starts, self.goals = self.goals, self.starts
        self.original_starts = deepcopy(self.starts)
        self.colors = plt.cm.rainbow(np.linspace(0, 1, self.num_objects))
        self.init()
    
    def generate_fixed_cycled_objects(self)->None:
        self.starts = []
        self.goals = []
        for i in range(self.num_objects):
            x = np.cos(2 * np.pi * i / self.num_objects) * (self.map_width/2- 2 * self.object_size)
            y = np.sin(2 * np.pi * i / self.num_objects) * (self.map_length/2 - 2 * self.object_size)
            start = np.array([x, y, self.object_size/2])
            self.starts.append(start)
        for i in range(self.num_objects):
            goal = self.starts[(i+1)%self.num_objects]
            self.goals.append(goal)
        self.original_starts = deepcopy(self.starts)
        self.colors = plt.cm.rainbow(np.linspace(0, 1, self.num_objects))
        self.init()
    
    def generate_fixed_four_objects(self)->None:
        self.num_objects = 4
        self.buffer_dict = {i: None for i in range(self.num_objects)}
        self.starts = [
            np.array([-self.map_width/2 + 2 * self.object_size, -self.map_length/2 + 2 * self.object_size, self.object_size/2]), 
            np.array([self.map_width/2 - 2 * self.object_size, -self.map_length/2 + 2 * self.object_size, self.object_size/2]),
            np.array([self.map_width/2 - 2 * self.object_size, self.map_length/2 - 2 * self.object_size, self.object_size/2]),
            np.array([-self.map_width/2 + 2 * self.object_size, self.map_length/2 - 2 * self.object_size, self.object_size/2])
        ]
        self.goals = []
        for i in range(len(self.starts)):
            goal = (self.starts[i] + self.starts[(i+1)%len(self.starts)])/2
            self.goals.append(goal)
        
        self.colors = plt.cm.rainbow(np.linspace(0, 1, self.num_objects))
        self.original_starts = deepcopy(self.starts)
        self.init()
    
    
    def generate_chained_four_objects(self)->None:
        self.num_objects = 4
        self.buffer_dict = {i: None for i in range(self.num_objects)}
        self.starts = [
            np.array([-self.map_width/2 + 3 * self.object_size, -self.map_length/2 + 3 * self.object_size, self.object_size/2]), 
            np.array([self.map_width/2 - 3 * self.object_size, -self.map_length/2 + 3 * self.object_size, self.object_size/2]),
            np.array([self.map_width/2 - 3 * self.object_size, self.map_length/2 - 3 * self.object_size, self.object_size/2]),
            np.array([-self.map_width/2 + 3 * self.object_size, self.map_length/2 - 3 * self.object_size, self.object_size/2])
        ]
        self.goals = []
        for i in range(len(self.starts)):
            self.goals.append(self.starts[(i+1)%len(self.starts)])
        
        self.goals[-1] = np.array([0, 0, self.object_size/2])
        self.original_starts = deepcopy(self.starts)
        self.colors = plt.cm.rainbow(np.linspace(0, 1, self.num_objects))
        self.init()
        

    def generate_fixed_six_objects(self)->None:
        self.num_objects = 6
        self.buffer_dict = {i: None for i in range(self.num_objects)}
        self.starts = [
            np.array([-self.map_width/2 + 2 * self.object_size, -self.map_length/2 + 2 * self.object_size, self.object_size/2]), 
            np.array([0, -self.map_length/2 + 2 * self.object_size, self.object_size/2]),
            np.array([self.map_width/2 - 2 * self.object_size, -self.map_length/2 + 2 * self.object_size, self.object_size/2]),
            np.array([self.map_width/2 - 2 * self.object_size, self.map_length/2 - 2 * self.object_size, self.object_size/2]),
            np.array([0, self.map_length/2 - 2 * self.object_size, self.object_size/2]),
            np.array([-self.map_width/2 + 2 * self.object_size, self.map_length/2 - 2 * self.object_size, self.object_size/2]),
        ]
        self.goals = []
        for i in range(len(self.starts)):
            goal = (self.starts[i] + self.starts[(i+1)%len(self.starts)])/2
            self.goals.append(goal)
        self.original_starts = deepcopy(self.starts)
        
        self.colors = plt.cm.rainbow(np.linspace(0, 1, self.num_objects))
        self.init()
        
    def generate_running_example(self)->None:
        # one cycle, one chain, some independent objects
        cycle_starts = [
            np.array([-self.map_width/2 + 2 * self.object_size, -self.map_length/2 + 2 * self.object_size, self.object_size/2]), 
            np.array([self.map_width/2 - 2 * self.object_size, -self.map_length/2 + 2 * self.object_size, self.object_size/2]),
            np.array([self.map_width/2 - 2 * self.object_size, self.map_length/2 - 2 * self.object_size, self.object_size/2]),
            np.array([-self.map_width/2 + 2 * self.object_size, self.map_length/2 - 2 * self.object_size, self.object_size/2]), # cycles object
        ]
        for i in range(len(cycle_starts)):
            cycle_starts[i][:-1] += (np.random.rand(2)-0.5) * 0.2
            cycle_starts[i] = np.clip(cycle_starts[i], -self.map_width/2 + self.object_size, self.map_width/2 - self.object_size)
        # cycle_starts = []
        # cnt = 0
        # for i in range(4):
        #     start = self.random_position()
        #     while self.check_collision(cycle_starts, start, 0.1):
        #         start = self.random_position()
        #         if cnt > 1000:
        #             raise RuntimeError('Cannot generate task configuration')
        #         cnt += 1
        #     cycle_starts.append(start)
        cycle_goals = cycle_starts[1:] + [cycle_starts[0]]
        chain_starts = []
        for i in range(len(cycle_starts)-1):
            goal = (cycle_starts[i] + cycle_starts[(i+1)%len(cycle_starts)])/2
            chain_starts.append(goal)
        chain_goals = [np.array([0,0, self.object_size/2])] + chain_starts[0:-1]

        independent_starts = []
        for i in range(len(chain_starts)-1):
            independent_start = (chain_starts[i] + chain_starts[(i+1)%len(chain_starts)])/2
            independent_starts.append(independent_start)

        independent_goals = [np.array([-1 * start[0], start[1], start[2]]) for start in independent_starts] 
        independent_starts, independent_goals = independent_goals, independent_starts

        self.starts = deepcopy(cycle_starts) + deepcopy(chain_starts) + deepcopy(independent_starts)
        self.goals = deepcopy(cycle_goals) + deepcopy(chain_goals) + deepcopy(independent_goals)
        self.starts[8][0] = self.starts[3][0]
        self.starts[8][1] = self.starts[3][1] - self.object_size / 3. * 4
        self.goals[2][1] = (self.starts[8][1] + self.starts[3][1]) / 2

        
        self.num_objects = len(self.starts)
        self.buffer_dict = {i: None for i in range(self.num_objects)}
        self.colors = plt.cm.rainbow(np.linspace(0, 1, self.num_objects))
        self.original_starts = deepcopy(self.starts)
        self.init()

    
    def random_position(self):
        x, y = 2 * (np.random.rand(2)-0.5) * np.array([self.map_width/2 - self.object_size, self.map_length/2 - self.object_size])
        pos = np.array([x, y, self.object_size/2])
        return pos
    
    def generate_random_postions(self, num_objects, collision_avoidance_list=None, clearance=0.0):
        cnt = 0
        results = []
        if collision_avoidance_list is None:
            collision_avoidance_list = []
        else:
            collision_avoidance_list = deepcopy(collision_avoidance_list)
        # Generate random unique start positions
        for i in range(num_objects):
            pos = self.random_position()
            while self.check_collision(collision_avoidance_list, pos, clearance=clearance):
                pos = self.random_position()
                if cnt > 1000:
                    raise RuntimeError('Cannot generate task configuration')
                cnt += 1
            results.append(pos)
            collision_avoidance_list.append(pos)
        return results
    
    def generate(self, gen_type="random", clearance=0.0) -> None:
        """Generate random start and goal positions with optional cycling behavior."""
        self.starts = []
        self.goals = []
        self.colors = plt.cm.rainbow(np.linspace(0, 1, self.num_objects))

        match gen_type:
            case "random":
                self.starts = self.generate_random_postions(self.num_objects, clearance=clearance)
                self.goals = self.generate_random_postions(self.num_objects, clearance=clearance)
            case "cycled":
                self.starts = self.generate_random_postions(self.num_objects, clearance=clearance)
                # Create a cyclic goal configuration
                self.goals = self.starts[1:] + [self.starts[0]]
            case "double_cycle":
                assert self.num_objects > 4, "Number of objects must be greater than 4 for double cycle."
                # generate the knot of the cycle
                start1 = self.random_position()
                start2 = deepcopy(start1)
                if start1[0] + 4/3 * self.object_size < self.map_width/2 - self.object_size:
                    start2[0] += 4/3 * self.object_size
                else:
                    start2[0] -= 4/3 * self.object_size
                
                # ajust the y position of start1 and start2. Only one while loop should run
                while start1[1] + 2/3 * self.object_size > self.map_length/2 - self.object_size:
                    start1[1] -= 2 * self.object_size
                while start1[1] - 2/3 * self.object_size < -self.map_length/2 + self.object_size:
                    start1[1] += 2 * self.object_size
                
                goal1 = deepcopy(start1)
                goal1[0] = (start1[0] + start2[0])/2
                goal2 = deepcopy(goal1)
                goal1[1] += 2/3 * self.object_size
                goal2[1] -= 2/3 * self.object_size
                
                collision_avoidance_list = [start1, start2]

                # pick a cycle size for the first cycle
                cycle_size1 = np.random.randint(2, self.num_objects - 2)
                cycle_size2 = self.num_objects - cycle_size1
                cycle_starts1 = self.generate_random_postions(cycle_size1, collision_avoidance_list=collision_avoidance_list, clearance=clearance)
                cycle_starts2 = self.generate_random_postions(cycle_size2, collision_avoidance_list=cycle_starts1+collision_avoidance_list, clearance=clearance)
                # pick one object from cycle 1
                cycle_id_1 = np.random.randint(0, cycle_size1)
                cycle_id_2 = np.random.randint(0, cycle_size2)
                cycle_starts1[cycle_id_1] = start1
                cycle_starts2[cycle_id_2] = start2
                print(f"cycle_size1 = {cycle_size1}, cycle_size2 = {cycle_size2}")
                cycle_goals1 = cycle_starts1[1:] + [cycle_starts1[0]]
                cycle_goals2 = cycle_starts2[1:] + [cycle_starts2[0]]
                cycle_goals1[(cycle_id_1-1)%cycle_size1] = goal1
                cycle_goals2[(cycle_id_2-1)%cycle_size2] = goal2

                self.starts = cycle_starts1 + cycle_starts2
                self.goals = cycle_goals1 + cycle_goals2
            case "mixed":
                # sample a cycle size that is > 4 and < num_objects
                self.num_objects = 12
                self.buffer_dict = {i: None for i in range(self.num_objects)}
                print("Warning: num_objects is set to 12 for mixed generation")
                self.colors = plt.cm.rainbow(np.linspace(0, 1, self.num_objects))
                cycle_number = np.random.randint(2, self.num_objects - 4)
                cycle_starts = self.generate_random_postions(cycle_number, clearance=clearance)
                cycle_goals = cycle_starts[1:] + [cycle_starts[0]]
                # sampel a chain size
                chain_number = np.random.randint(2, self.num_objects - cycle_number - 1)
                chain_starts = self.generate_random_postions(chain_number, collision_avoidance_list=cycle_starts, clearance=clearance)
                chain_goals = self.generate_random_postions(1, collision_avoidance_list=cycle_starts+chain_starts, clearance=clearance)
                chain_goals = chain_starts[1:] + [chain_goals[0]]
                # sample a number of independent objects
                independent_number = self.num_objects - cycle_number - chain_number
                independent_starts = self.generate_random_postions(independent_number, collision_avoidance_list=cycle_starts+chain_starts, clearance=clearance)
                independent_goals = self.generate_random_postions(independent_number, collision_avoidance_list=cycle_goals+chain_goals, clearance=clearance)
                self.starts = cycle_starts + chain_starts + independent_starts
                self.goals = cycle_goals + chain_goals + independent_goals
            case _:
                raise ValueError(f"Unknown generation type: {gen_type}")

        self.original_starts = self.starts.copy()
        self.init()

    def check_collision_between_cubes(self, pos1: np.ndarray, pos2: np.ndarray, clearance=0)->bool:
        # check if two rectangles are colliding
        # pos1 and pos2 are the center of the rectangles
        x1, y1, z1 = pos1
        x2, y2, z2 = pos2
        if abs(x1 - x2) < self.object_size + clearance and abs(y1 - y2) < self.object_size + clearance and abs(z1 - z2) < self.object_size + clearance:
            return True
        return False
    
    def check_collision(self, positions:list, new_position: np.ndarray, clearance = 0)->bool:
        if len(positions) == 0:
            return False
        for position in positions:
            if self.check_collision_between_cubes(position, new_position, clearance=clearance):
                return True
        return False

    
    def draw(self):
        print(self.get_colors())
        # draw starts and goals in rectangles
        # add texture to goals
        fig, ax = plt.subplots()
        for i in range(self.num_objects):
            start = self.starts[i ]
            goal = self.goals[i]
            ax.add_patch(plt.Rectangle((start[0], start[1]), self.object_size, self.object_size, facecolor=self.colors[i], alpha=1.0, edgecolor='black'))
            ax.add_patch(plt.Rectangle((goal[0], goal[1]), self.object_size, self.object_size, facecolor=self.colors[i], alpha=1.0, hatch='/', edgecolor='black'))
        
        legend_handles = [
            patches.Patch(facecolor='none', edgecolor='black', label='Start'),
            patches.Patch(facecolor='none', edgecolor='black', hatch='/', label='Goal')
        ]

        plt.legend(handles=legend_handles)
        plt.xlim([-self.map_width/2, self.map_width/2])
        plt.ylim([-self.map_length/2., self.map_length/2.])
        plt.axis('equal') 
        plt.show()
    
    def draw_triptych(self, save_path: str):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['Start Positions', 'Goal Positions', 'Start & Goal Overlay']
        goal_scale = 1.2
        scaled_size = self.object_size * goal_scale

        for i, ax in enumerate(axes):
            table_rect = patches.FancyBboxPatch(
                (-task.map_width / 2, -task.map_length / 2),
                task.map_width,
                task.map_length,
                boxstyle="round,pad=0.02",  # Rounded corners; change to "square" if preferred
                edgecolor='black',
                facecolor='lightgray',  # Light gray table color
                linewidth=2,
                # zorder=0  # Behind everything else
            )
            ax.add_patch(table_rect)
            ax.set_xlim([-task.map_width / 2, task.map_width / 2])
            ax.set_ylim([-task.map_length / 2, task.map_length / 2])
            ax.set_title(titles[i])
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:  # Only start
                for j in range(self.num_objects):
                    start = self.starts[j]
                    goal = self.goals[j]
                    color = self.colors[j]
                    ax.add_patch(plt.Rectangle(
                        (start[0]-self.object_size/2, start[1]-self.object_size/2),
                        self.object_size,
                        self.object_size,
                        facecolor=color,
                        edgecolor='black',
                        alpha=1.0
                    ))
            elif i == 1:  # Only goal
                for j in range(self.num_objects):
                    start = self.starts[j]
                    goal = self.goals[j]
                    color = self.colors[j]
                    ax.add_patch(plt.Rectangle(
                        (goal[0] - scaled_size/2, goal[1] - scaled_size/2),
                        scaled_size,
                        scaled_size,
                        facecolor=color,
                        edgecolor='black',
                        hatch='/',
                        alpha=1.0
                    ))
            else:  # Overlay both
                for j in range(self.num_objects):
                    goal = self.goals[j]
                    color = self.colors[j]
                    ax.add_patch(plt.Rectangle(
                        (goal[0] - scaled_size/2, goal[1] - scaled_size/2),
                        scaled_size,
                        scaled_size,
                        facecolor=color,
                        edgecolor='black',
                        hatch='/',
                        alpha=1.0
                    ))
                for j in range(self.num_objects):
                    start = self.starts[j]
                    color = self.colors[j]
                    ax.add_patch(plt.Rectangle(
                        (start[0]-self.object_size/2, start[1]-self.object_size/2),
                        self.object_size,
                        self.object_size,
                        facecolor=color,
                        edgecolor='black',
                        alpha=0.95
                    ))

        # Shared legend
        legend_elements = [
            patches.Patch(facecolor='gray', edgecolor='black', label='Start'),
            patches.Patch(facecolor='gray', edgecolor='black', hatch='/', label='Goal')
        ]
        axes[2].legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
        plt.close(fig)
    


    def draw_triptych_to_ppt(self, ppt_path="triptych_slide.pptx", slide_title="Start, Goal, and Overlay"):
        """
        Draws a triptych (start, goal, overlay) of the task into a PowerPoint slide.
        Uses a fixed visual table size and scales all positions and object sizes proportionally.
        """
        if os.path.exists(ppt_path):
            print(f"Opening existing presentation: {ppt_path}")
            prs = Presentation(ppt_path)
        else:
            print(f"Creating new presentation: {ppt_path}")
            prs = Presentation()

        slide = prs.slides.add_slide(prs.slide_layouts[5])
        width, height = prs.slide_width, prs.slide_height

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), width - Inches(1), Inches(0.5))
        tf = title_box.text_frame
        tf.text = slide_title
        tf.paragraphs[0].font.size = Pt(24)

        # Layout setup
        section_width = width / 3 - Inches(0.5)
        section_margin = Inches(0.25)

        # VISUAL SIZE SETTINGS
        visual_table_size_inch = 3.0  # how large to draw the table
        scale = visual_table_size_inch / self.map_width  # scale factor for positions
        visual_object_size = self.object_size * scale  # scaled size of each square
        rect_size = Inches(visual_object_size)
        visual_table_size_inch = Inches(visual_table_size_inch)

        def draw_scene(scene_type, x_offset):
            # Draw the table (centered at 0,0)
            table_center_x = x_offset + section_width / 2
            table_center_y = height / 2
            table_left = table_center_x - visual_table_size_inch / 2
            table_top = height / 2 - visual_table_size_inch / 2
            table_rect = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE,
                table_left, table_top,
                visual_table_size_inch, visual_table_size_inch
            )
            table_fill = table_rect.fill
            table_fill.solid()
            table_fill.fore_color.rgb = RGBColor(230, 230, 230)
            table_rect.line.color.rgb = RGBColor(180, 180, 180)

            for i in range(self.num_objects):
                color = tuple(int(c * 255) for c in self.colors[i][:3])
                start = self.get_start(i)
                goal = self.get_goal(i)

                def draw_box(xy, origin, label, size_scale):
                    x, y = xy[0], xy[1]
                    left = origin[0] + Inches(scale * x) - rect_size / 2 * size_scale
                    top =  origin[1] - Inches(scale * y) - rect_size / 2 * size_scale
                    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, rect_size * size_scale, rect_size * size_scale)
                    fill = shape.fill
                    fill.solid()
                    fill.fore_color.rgb = RGBColor(*color)
                    shape.line.color.rgb = RGBColor(0, 0, 0)
                    shape.line.width = Pt(1)

                    label_box = slide.shapes.add_textbox(left, top, rect_size * size_scale, rect_size * size_scale)
                    label_box.text = str(label)
                    p = label_box.text_frame.paragraphs[0]
                    p.font.size = Pt(10)
                    p.alignment = 1  # center

                if scene_type in ['goal', 'overlay']:
                    draw_box(goal, (table_center_x, table_center_y), i, 1.2)
                if scene_type in ['start', 'overlay']:
                    draw_box(start, (table_center_x, table_center_y), i, 1.0)

        draw_scene('start', section_margin)
        draw_scene('goal', section_margin + section_width + section_margin)
        draw_scene('overlay', section_margin + 2 * (section_width + section_margin))

        prs.save(ppt_path)
        print(f"Saved triptych slide to: {ppt_path}")

    def draw_start_goal_dependency_to_ppt(self, ppt_path="triptych_depgraph.pptx", slide_title="Start, Goal, and Dependency Graph"):
        """
        Draws start, goal, and dependency graph views into a PowerPoint slide using elbow connectors.
        """
        if os.path.exists(ppt_path):
            print(f"Opening existing presentation: {ppt_path}")
            prs = Presentation(ppt_path)
        else:
            print(f"Creating new presentation: {ppt_path}")
            prs = Presentation()

        slide = prs.slides.add_slide(prs.slide_layouts[5])
        width, height = prs.slide_width, prs.slide_height

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), width - Inches(1), Inches(0.5))
        tf = title_box.text_frame
        tf.text = slide_title
        tf.paragraphs[0].font.size = Pt(24)

        section_width = width / 3 - Inches(0.5)
        section_margin = Inches(0.25)

        visual_table_size_inch = 3.0
        scale = visual_table_size_inch / self.map_width
        rect_size = Inches(self.object_size * scale)
        visual_table_size_inch = Inches(visual_table_size_inch)

        def get_edge_center(from_pt, to_pt):
            dx, dy = to_pt[0] - from_pt[0], to_pt[1] - from_pt[1]
            if abs(dx) > abs(dy):
                return (from_pt[0] + np.sign(dx) * self.object_size / 2, from_pt[1])
            else:
                return (from_pt[0], from_pt[1] + np.sign(dy) * self.object_size / 2)

        def draw_scene(scene_type, x_offset):
            center_x = x_offset + section_width / 2
            center_y = height / 2

            # Table
            table_left = center_x - visual_table_size_inch / 2
            table_top = center_y - visual_table_size_inch / 2
            table_rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, table_left, table_top,
                                                visual_table_size_inch, visual_table_size_inch)
            table_fill = table_rect.fill
            table_fill.solid()
            table_fill.fore_color.rgb = RGBColor(230, 230, 230)
            table_rect.line.color.rgb = RGBColor(180, 180, 180)

            # Object Rectangles
            for i in range(self.num_objects):
                color = tuple(int(c * 255) for c in self.colors[i][:3])
                start = self.get_start(i)
                goal = self.get_goal(i)

                def draw_box(xy, label):
                    x, y = xy[0], xy[1]
                    left = center_x + Inches(scale * x) - rect_size / 2
                    top = center_y - Inches(scale * y) - rect_size / 2
                    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, rect_size, rect_size)
                    fill = shape.fill
                    fill.solid()
                    fill.fore_color.rgb = RGBColor(*color)
                    shape.line.color.rgb = RGBColor(0, 0, 0)
                    shape.line.width = Pt(1)
                    label_box = slide.shapes.add_textbox(left, top, rect_size, rect_size)
                    label_box.text = str(label)
                    p = label_box.text_frame.paragraphs[0]
                    p.font.size = Pt(10)
                    p.alignment = 1

                if scene_type in ['start', 'dependency']:
                    draw_box(start, i)
                if scene_type == 'goal':
                    draw_box(goal, i)

            # Dependency arrows
            if scene_type == 'dependency':
                for dependent, interfering_set in self.dependency_graph.dependency_graph.items():
                    dest = self.start_dict[dependent]
                    for interfering in interfering_set:
                        source = self.start_dict[interfering]
                        src_edge = get_edge_center(source, dest)
                        dst_edge = get_edge_center(dest, source)

                        sx = center_x + Inches(scale * src_edge[0])
                        sy = center_y - Inches(scale * src_edge[1])
                        dx = center_x + Inches(scale * dst_edge[0])
                        dy = center_y - Inches(scale * dst_edge[1])
                        print(f"sx: {sx}, sy: {sy}, dx: {dx}, dy: {dy}")

                        if sx == dx and sy == dy:
                            print("❌ Skipping degenerate connector")
                            return

                        if sx < 0 or sy < 0 or dx < 0 or dy < 0:
                            print("❌ Skipping connector with negative coords")
                            return
 

                        connector = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, sx, sy, dx, dy)
                        connector.line.color.rgb = RGBColor(0, 0, 0)
                        connector.line.width = Pt(2)
                        connector.line.end_arrowhead = True
                        line_elem = connector.line._get_or_add_ln()
                        # line_elem.append(parse_xml("""
                        #         <a:headEnd type="arrow" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"/>
                        # """))
                        line_elem.append(parse_xml("""
                            <a:tailEnd type="arrow" w="lg" len="lg"
                            xmlns:a="http://schemas.openxmlformats.org/drawingml/2023/main"/>
                        """))


        draw_scene('start', section_margin)
        draw_scene('goal', section_margin + section_width + section_margin)
        draw_scene('dependency', section_margin + 2 * (section_width + section_margin))

        prs.save(ppt_path)
        print(f"Saved triptych with dependency graph to: {ppt_path}")
 
    def get_starts(self):
        return self.starts
    
    def get_original_starts(self):
        return self.original_starts
    
    def get_goals(self):
        return self.goals
    
    def get_colors(self):
        return self.colors[:, :3]
    
    def init(self):
        self.idx_to_be_rearranged = []
        for i in range(self.num_objects):
            if np.linalg.norm(self.starts[i] - self.goals[i]) > 1e-3:
                self.idx_to_be_rearranged.append(i)
        self.build_start_goal_dict()
        self.dependency_graph = DependencyGraph(self.start_dict, self.goal_dict, self.object_size)   

    def get_independent_sets(self):
        return self.dependency_graph.get_independent_sets()

    def all_arms_available(self):
        return self.arm1_state.task_state == None and self.arm2_state.task_state == None
    
    def all_arms_busy(self):
        return self.arm1_state.task_state != None and self.arm2_state.task_state != None
    
    def one_arm_busy(self):
        return self.arm1_state.task_state != None or self.arm2_state.task_state != None
    
    def finished(self):
        return len(self.idx_to_be_rearranged) == 0
    
    def depends_on(self, index):
        # check if the index is in the dependency graph
        if index in self.dependency_graph.dependency_graph:
            return self.dependency_graph.depends_on(index)
        return {}
    

    def plan(self):
        if self.all_arms_available():
            if len(self.idx_to_be_rearranged) == 1:
                # last one to finish
                print("Last one to finish.")
                task_plan = TaskPlan(
                    arm1_gripper_action=GripperAction.CLOSE,
                    arm2_gripper_action=GripperAction.CLOSE,
                    start_goal_pairs=None,
                    arm1_to_start=True,
                    arm2_to_start=True,
                    arm1_assigned_object=-1,
                    arm2_assigned_object=self.idx_to_be_rearranged[0],
                    arm1_angle=900
                )
                return task_plan

            start_goal_pairs = []
            if len(self.dependency_graph.independent_sets) > 1 or self.dependency_graph.has_chains():
                if len(self.dependency_graph.independent_sets) > 1:
                    # pick two independent sets
                    print("Pick two independent sets.")
                    idx_to_be_rearranged = self.dependency_graph.independent_sets
                    start_goal_pairs += list(permutations(idx_to_be_rearranged, 2))
                    # random.shuffle(start_goal_pairs)
                
                if self.dependency_graph.has_chains():
                    # pick one chain
                    print("Pick one chain.")
                    chain = self.dependency_graph.get_chains()[0]
                    print("chain: ", chain)
                    assert len(chain) > 1, "Chain has to have more than one object"
                    start_goal_pairs += [(chain[-1], chain[-2]), (chain[-2], chain[-1])]

                # sort the start and the goal pairs based on the sum of the start-start distance and the goal-goal distance
                start_goal_pairs.sort(key=lambda x: np.linalg.norm(self.start_dict[x[0]] - self.start_dict[x[1]]) + np.linalg.norm(self.goal_dict[x[0]] - self.goal_dict[x[1]]))
                task_plan = TaskPlan(
                    arm1_gripper_action=GripperAction.CLOSE,
                    arm2_gripper_action=GripperAction.CLOSE,
                    start_goal_pairs=start_goal_pairs,
                    arm1_to_start=True,
                    arm2_to_start=True,
                    arm1_assigned_object=None,
                    arm2_assigned_object=None
                    )
                return task_plan
                
            if self.dependency_graph.has_cycles():
                # pick one cycle
                print("Pick one cycle to break")
                for i in range(len(self.dependency_graph.cycles)):
                    cycle = self.dependency_graph.get_cycles()[0]
                    assert len(cycle) > 1, "Cycle has to have more than one object"
                    if len(cycle) == 2:
                        # switch them
                        print("The cycle has two objects. Switch two objects.")
                        print("cycle: ", cycle)
                        use_buffer = False
                        # cnt the number of the objects that have more than 1 dependencies
                        cnt = sum([len(self.depends_on(i)) > 1 for i in cycle])
                        if cnt == 1:
                            use_buffer = True
                        if cnt == 2:
                            continue # there's no way to rearrange this
                        task_plan = TaskPlan(
                            arm1_gripper_action=GripperAction.CLOSE,
                            arm2_gripper_action=GripperAction.CLOSE,
                            start_goal_pairs=[(cycle[0], cycle[1]), (cycle[1], cycle[0])],
                            arm1_to_start=True,
                            arm2_to_start=True,
                            arm1_assigned_object=None,
                            arm2_assigned_object=None,
                            use_buffer=use_buffer
                        )
                        return task_plan
                    else:
                        # need to break the cycle, need a buffer position to store the object. Need a temporary goal
                        print("The cycle has more than two objects. Break the cycle.")
                        print("cycle: ", cycle)
                        for i in range(len(cycle)):
                            # pick two consecutive objects
                            if not len(self.depends_on( cycle[i] )) > 1:
                                start_goal_pairs.append((cycle[i], cycle[(i+1)%len(cycle)]))
                                start_goal_pairs.append((cycle[(i+1)%len(cycle)], cycle[i]))
                        task_plan = TaskPlan(
                            arm1_gripper_action=GripperAction.CLOSE,
                            arm2_gripper_action=GripperAction.CLOSE,
                            start_goal_pairs=start_goal_pairs,
                            arm1_to_start=True,
                            arm2_to_start=True,
                            arm1_assigned_object=None,
                            arm2_assigned_object=None,
                            use_buffer=True
                        )
                        return task_plan


        elif self.all_arms_busy():
            # keep doing it
            print(f"Both arms are busy, keep finishing up. Arm1: {self.arm1_state.assigned_object}, Arm2: {self.arm2_state.assigned_object}")
            task_plan = TaskPlan(
                arm1_gripper_action=self.arm1_state.gripper_state.reverse_action(),
                arm2_gripper_action=self.arm2_state.gripper_state.reverse_action(),
                start_goal_pairs=None,
                arm1_to_start=False if self.arm1_state.gripper_state == GripperState.CLOSE else True,
                arm2_to_start=False if self.arm2_state.gripper_state == GripperState.CLOSE else True,
                arm1_assigned_object=self.arm1_state.assigned_object,
                arm2_assigned_object=self.arm2_state.assigned_object,
                arm1_angle=self.arm1_state.angle,
                arm2_angle=self.arm2_state.angle,
                prioritize_arm1=False if self.arm2_state.at_intermediate else True
            )
            return task_plan
        
        elif self.one_arm_busy():
            # finish the task for one arm and assign task for the other arm
            print("One Arm is busy and one arm is available.")
            if len(self.dependency_graph.independent_sets) > 0:
                print("Picking one random independent object and keep finishing the other arm.")
                start_goal_pairs = []
                if self.arm1_state.task_state == None:
                    for i in self.dependency_graph.independent_sets:
                        if i == self.arm2_state.assigned_object:
                            continue
                        start_goal_pairs.append((i, self.arm2_state.assigned_object))
                else:
                    for i in self.dependency_graph.independent_sets:
                        if i == self.arm1_state.assigned_object:
                            continue
                        start_goal_pairs.append((self.arm1_state.assigned_object, i))
                if start_goal_pairs == []:
                    start_goal_pairs = None
                else:
                    random.shuffle(start_goal_pairs)
                    
                task_plan = TaskPlan(
                    arm1_gripper_action=self.arm1_state.gripper_state.reverse_action(),
                    arm2_gripper_action=self.arm2_state.gripper_state.reverse_action(),
                    start_goal_pairs=start_goal_pairs,
                    arm1_to_start=False if self.arm1_state.gripper_state == GripperState.CLOSE else True,
                    arm2_to_start=False if self.arm2_state.gripper_state == GripperState.CLOSE else True,
                    arm1_assigned_object=self.arm1_state.assigned_object if self.arm1_state.assigned_object is not None else None,
                    arm2_assigned_object=self.arm2_state.assigned_object if self.arm2_state.assigned_object is not None else None,
                    arm1_angle=self.arm1_state.angle if self.arm1_state.angle is not None else None,
                    arm2_angle=self.arm2_state.angle if self.arm2_state.angle is not None else None,
                    prioritize_arm1=False if self.arm2_state.at_intermediate else True
                )
                return task_plan
            if self.dependency_graph.has_chains():
                # pick one chain
                print("Pick one chain.")
                chain = self.dependency_graph.get_chains()[0]
                print("chain: ", chain)
                assert len(chain) > 1, "Chain has to have more than one object"
                start_goal_pairs = []
                if self.arm1_state.task_state == None:
                    start_goal_pairs.append((chain[-1], self.arm2_state.assigned_object))
                else:
                    start_goal_pairs.append((self.arm1_state.assigned_object, chain[-1]))
                print("Pick the end of the chain and keep finishing the other arm.")
                task_plan = TaskPlan(
                    arm1_gripper_action=self.arm1_state.gripper_state.reverse_action(),
                    arm2_gripper_action=self.arm2_state.gripper_state.reverse_action(),
                    start_goal_pairs=start_goal_pairs,
                    arm1_to_start=False if self.arm1_state.gripper_state == GripperState.CLOSE else True,
                    arm2_to_start=False if self.arm2_state.gripper_state == GripperState.CLOSE else True,
                    arm1_assigned_object=self.arm1_state.assigned_object if self.arm1_state.assigned_object is not None else None,
                    arm2_assigned_object=self.arm2_state.assigned_object if self.arm2_state.assigned_object is not None else None,
                    arm1_angle=self.arm1_state.angle if self.arm1_state.angle is not None else None,
                    arm2_angle=self.arm2_state.angle if self.arm2_state.angle is not None else None,
                    prioritize_arm1=False if self.arm2_state.at_intermediate else True
                )
                return task_plan
            if self.dependency_graph.has_cycles():
                print("Pick one cycle to break")
                cycle = self.dependency_graph.get_cycles()[0]
                assert len(cycle) > 1, "Cycle has to have more than one object"
                start_goal_pairs = []
                if self.arm1_state.task_state == None:
                    for i in range(len(cycle)):
                        start_goal_pairs.append((cycle[i], self.arm2_state.assigned_object))
                else:
                    for i in range(len(cycle)):
                        start_goal_pairs.append((self.arm1_state.assigned_object, cycle[i]))
                task_plan = TaskPlan(
                    arm1_gripper_action=self.arm1_state.gripper_state.reverse_action(),
                    arm2_gripper_action=self.arm2_state.gripper_state.reverse_action(),
                    start_goal_pairs=start_goal_pairs,
                    arm1_to_start=False if self.arm1_state.gripper_state == GripperState.CLOSE else True,
                    arm2_to_start=False if self.arm2_state.gripper_state == GripperState.CLOSE else True,
                    arm1_assigned_object=self.arm1_state.assigned_object if self.arm1_state.assigned_object is not None else None,
                    arm2_assigned_object=self.arm2_state.assigned_object if self.arm2_state.assigned_object is not None else None,
                    arm1_angle=self.arm1_state.angle if self.arm1_state.angle is not None else None,
                    arm2_angle=self.arm2_state.angle if self.arm2_state.angle is not None else None,
                    prioritize_arm1=False if self.arm2_state.at_intermediate else True,
                    use_buffer=True
                )
        return None

    def update(self, result: TaskExecutionResult):
        if result.arm1_task_state == TaskState.COMPLETED:
            self.arm1_state.task_state = None
            self.arm1_state.assigned_object = None
            self.arm1_state.gripper_state = result.arm1_gripper_state
            self.arm1_state.angle = None
            if result.arm1_assigned_object != -1 and result.arm1_assigned_object is not None:
                if self.buffer_dict[result.arm1_assigned_object] is not None:
                    self.starts[result.arm1_assigned_object] = self.buffer_dict[result.arm1_assigned_object]
                    self.buffer_dict[result.arm1_assigned_object] = None
                else:
                    self.idx_to_be_rearranged.remove(result.arm1_assigned_object)
        elif result.arm1_task_state == TaskState.IN_PROGRESS:
            self.arm1_state.task_state = TaskState.IN_PROGRESS
            self.arm1_state.gripper_state = result.arm1_gripper_state
            self.arm1_state.assigned_object = result.arm1_assigned_object
            self.arm1_state.angle = result.arm1_angle
            if result.arm1_at_intermediate:
                self.arm1_state.at_intermediate = True
        if result.arm2_task_state == TaskState.COMPLETED:
            self.arm2_state.task_state = None
            self.arm2_state.assigned_object = None
            self.arm2_state.gripper_state = result.arm2_gripper_state
            self.arm2_state.angle = None
            if result.arm2_assigned_object != -1 and result.arm2_assigned_object is not None:
                if self.buffer_dict[result.arm2_assigned_object] is not None:
                    self.starts[result.arm2_assigned_object] = self.buffer_dict[result.arm2_assigned_object]
                    self.buffer_dict[result.arm2_assigned_object] = None
                else:
                    self.idx_to_be_rearranged.remove(result.arm2_assigned_object)
        elif result.arm2_task_state == TaskState.IN_PROGRESS:
            self.arm2_state.task_state = TaskState.IN_PROGRESS
            self.arm2_state.gripper_state = result.arm2_gripper_state
            self.arm2_state.assigned_object = result.arm2_assigned_object
            self.arm2_state.angle = result.arm2_angle
            if result.arm2_at_intermediate:
                self.arm2_state.at_intermediate = True
        
        if result.buffer is not None:
            if result.buffer[1] is not None:
                self.buffer_dict[result.arm1_assigned_object] = result.buffer[1]
            else:
                self.buffer_dict[result.arm2_assigned_object] = result.buffer[2]

        print("Updated task state.")
        print(f"Arm1: {self.arm1_state}")
        print(f"Arm2: {self.arm2_state}")
        self.build_start_goal_dict()
        self.dependency_graph = DependencyGraph(self.start_dict, self.goal_dict, self.object_size)   
        # self.draw_four_panel(None)
    
    def get_start(self, i: int):
        return self.starts[i]
    
    def set_starts(self, pos_dict: dict):
        for i in pos_dict:
            self.starts[i] = pos_dict[i]
        # self.init()
        self.build_start_goal_dict()
        

    def get_goal(self, i: int):
        return self.goals[i] if self.buffer_dict[i] is None else self.buffer_dict[i]
    
    def sample_buffers(self, buffer_num:int):
        buffers = []
        for i in range(buffer_num):
            buffer = self.random_position()
            while self.check_collision(self.starts + self.goals, buffer, clearance=self.object_size/2):
                buffer = self.random_position()
            buffers.append(buffer)
        buffers = np.array(buffers)
        return buffers
    
    def is_dependent(self, i:int, j:int):
        return self.dependency_graph.is_dependent(i, j)

    def is_direct_dependent(self, i:int, j:int):
        return self.dependency_graph.is_directly_dependent(i, j)
    
    def write_to_json(self, filename: str):
        data = {
            'starts': [start.tolist() for start in self.starts],
            'goals': [goal.tolist() for goal in self.goals],
            'object_size': self.object_size,
            'map_width': self.map_width,
            'map_length': self.map_length
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def read_from_json(filename: str):
        with open(filename, 'r') as f:
            data = json.load(f)

        task = Task(len(data['starts']), data['map_width'], data['map_length'], data['object_size'])
        task.starts = [np.array(start) for start in data['starts']]
        task.goals = [np.array(goal) for goal in data['goals']]
        task.original_starts = task.starts.copy()
        task.init()
        return task

    def draw_dependency_graph(task, save_path: str):
        """
        Draws the dependency graph using edge-to-edge arrows with enhanced styling.
        Arrows start and end at the rectangle edges closest to each other.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        # Draw the table surface as a light gray square background
        table_rect = patches.FancyBboxPatch(
            (-task.map_width / 2, -task.map_length / 2),
            task.map_width,
            task.map_length,
            boxstyle="round,pad=0.02",  # Rounded corners; change to "square" if preferred
            edgecolor='black',
            facecolor='lightgray',  # Light gray table color
            linewidth=2,
            # zorder=0  # Behind everything else
        )
        ax.add_patch(table_rect)
        ax.set_xlim([-task.map_width / 2, task.map_width / 2])
        ax.set_ylim([-task.map_length / 2, task.map_length / 2])
        ax.set_aspect('equal')
        ax.set_title("Dependency Graph (Edge-to-Edge Arrows)")

        object_size = task.object_size * 1.2

        def get_edge_center(from_pt, to_pt):
            """Return the edge center (on the square at from_pt) that's closest to to_pt."""
            dx, dy = to_pt[0] - from_pt[0], to_pt[1] - from_pt[1]
            if abs(dx) > abs(dy):
                return (from_pt[0] + np.sign(dx) * object_size / 2, from_pt[1])
            else:
                return (from_pt[0], from_pt[1] + np.sign(dy) * object_size / 2)

        # Draw objects and label them
        for obj_id, start in task.start_dict.items():
            lower_left = (start[0] - object_size / 2, start[1] - object_size / 2)
            color = task.colors[obj_id]
            rect = patches.Rectangle(lower_left, object_size, object_size,
                                    facecolor=color, edgecolor='black', alpha=1.0)
            ax.add_patch(rect)
            ax.text(start[0], start[1], str(obj_id), color='black', fontsize=40,
                    ha='center', va='center')

        # Draw dependency arrows
        for dependent, interfering_set in task.dependency_graph.dependency_graph.items():
            dest = task.start_dict[dependent]
            for interfering in interfering_set:
                source = task.start_dict[interfering]
                src_edge = get_edge_center(source, dest)
                dst_edge = get_edge_center(dest, source)

                ax.annotate(
                    "",
                    xy=dst_edge, xycoords='data',
                    xytext=src_edge, textcoords='data',
                    arrowprops=dict(
                        arrowstyle="->",
                        color='black',
                        lw=1.5,
                        mutation_scale=40,
                        connectionstyle="arc3,rad=0.3"
                    )
                )

        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved edge-to-edge dependency graph to: {save_path}")
        plt.close(fig)

    
    def draw_start_goal_pair(task, save_path=None):
        """
        Draws two subplots: one showing the start positions and one showing the goal positions.
        Each object is labeled with its ID.
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        titles = ['Start Positions', 'Goal Positions']
        goal_scale = 1.2
        scaled_size = task.object_size * goal_scale

        for ax, title, is_goal in zip(axes, titles, [False, True]):
            # Draw the table surface as a light gray square background
            table_rect = patches.FancyBboxPatch(
                (-task.map_width / 2, -task.map_length / 2),
                task.map_width,
                task.map_length,
                boxstyle="round,pad=0.02",  # Rounded corners; change to "square" if preferred
                edgecolor='black',
                facecolor='lightgray',  # Light gray table color
                linewidth=2,
                # zorder=0  # Behind everything else
            )
            ax.add_patch(table_rect)
            ax.set_xlim([-task.map_width / 2, task.map_width / 2])
            ax.set_ylim([-task.map_length / 2, task.map_length / 2])
            ax.set_aspect('equal')
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

            for i in range(task.num_objects):
                color = task.colors[i]
                pos = task.get_goal(i) if is_goal else task.get_start(i)
                # size = scaled_size if is_goal else task.object_size
                size = scaled_size 
                lower_left = (pos[0] - size / 2, pos[1] - size / 2)
                hatch = '/' if is_goal else None
                alpha = 1.0

                ax.add_patch(plt.Rectangle(
                    lower_left,
                    size,
                    size,
                    facecolor=color,
                    edgecolor='black',
                    hatch=hatch,
                    alpha=alpha
                ))

                ax.text(pos[0], pos[1], str(i), color='black', fontsize=30,
                        ha='center', va='center')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()
    
    def draw_all(self, save_path=None, overlay=False):
        fig_num = 4 if overlay else 3
        fig, axes = plt.subplots(1, fig_num, figsize=(6 * fig_num, 6))
        titles = ['Start Configurations', 'Goal Configurations', 'Start & Goal Overlay', 'Dependency Graph'] if overlay else ['Start Configurations', 'Goal Configurations', 'Dependency Graph']
        goal_scale = 1.05
        scaled_size = self.object_size * goal_scale
        fig_cnt = 0

        def draw_table(ax, title, facecolor='lightgray'):
            table_rect = patches.FancyBboxPatch(
                (-self.map_width / 2, -self.map_length / 2),
                self.map_width,
                self.map_length,
                boxstyle="round,pad=0.02",
                edgecolor='black',
                facecolor=facecolor,
                linewidth=2,
            )
            ax.add_patch(table_rect)
            ax.set_xlim([-self.map_width / 2, self.map_width / 2])
            ax.set_ylim([-self.map_length / 2, self.map_length / 2])
            ax.set_aspect('equal')
            ax.set_title(title, fontsize=30, fontname='Times New Roman')
            ax.set_xticks([])
            ax.set_yticks([])

        def draw_objects(ax, positions, colors, size, hatch=None, label_prefix="", label_show=True, label_fontsize=30, label_weight='normal'):
            for i, pos in positions.items():
                color = colors[i]
                lower_left = (pos[0] - size / 2, pos[1] - size / 2)
                ax.add_patch(patches.Rectangle(
                    lower_left, size, size,
                    facecolor=color,
                    edgecolor='black',
                    hatch=hatch,
                    alpha=1.0
                ))
                if label_show:
                    ax.text(
                        pos[0], pos[1]-0.005, f"{label_prefix}{i}",
                        color='black', fontsize=label_fontsize,
                        ha='center', va='center', fontname='Times New Roman', fontweight=label_weight
                    )

        for i, ax in enumerate(axes):
            # For the dependency graph panel, use a very light gray background
            if titles[i] == 'Dependency Graph':
                draw_table(ax, titles[i], facecolor='#f5f5f5')
            else:
                draw_table(ax, titles[i])

        # Start
        def _pixels_to_data_x(ax, px):
            # Convert pixel width to data units along X
            (x0_disp, y0_disp) = ax.transData.transform((0, 0))
            (x1_data, _) = ax.transData.inverted().transform((x0_disp + px, y0_disp))
            (x0_data, _) = ax.transData.inverted().transform((x0_disp, y0_disp))
            return x1_data - x0_data

        draw_objects(axes[fig_cnt], self.start_dict, self.colors, self.object_size, label_weight="bold")
        fig_cnt += 1

        # Goal: remove hatches; additionally, draw start configuration beneath (no labels/colors)
        # Draw start outlines first (beneath goals)
        start_px_inflate = _pixels_to_data_x(axes[fig_cnt], 2)
        start_size_main = self.object_size + start_px_inflate
        for i, pos in self.start_dict.items():
            lower_left = (pos[0] - start_size_main / 2, pos[1] - start_size_main / 2)
            axes[fig_cnt].add_patch(patches.Rectangle(
                lower_left, scaled_size, scaled_size,
                facecolor='white', edgecolor='white', linewidth=2.0, alpha=1.0, zorder=1
            ))
        # Now draw goals on top, no hatches
        draw_objects(axes[fig_cnt], self.goal_dict, self.colors, self.object_size * 0.95, hatch=None, label_prefix="", label_show=True, label_fontsize=30, label_weight='bold')
        fig_cnt += 1

        # Overlay
        if overlay:
            draw_objects(axes[fig_cnt], self.goal_dict, self.colors, scaled_size, hatch='/', label_prefix='G')
            start_px_inflate_ov = _pixels_to_data_x(axes[fig_cnt], 2)
            start_size_ov = self.object_size + start_px_inflate_ov
            draw_objects(axes[fig_cnt], self.start_dict, self.colors, start_size_ov, label_prefix='S')
            fig_cnt += 1

        # Dependency Graph: draw circular nodes with larger bold labels
        node_radius = self.object_size / 2 * 1.1 
        for i, pos in self.start_dict.items():
            color = self.colors[i]
            circ = patches.Circle((pos[0], pos[1]), radius=node_radius, facecolor=color, edgecolor='black', alpha=1.0)
            axes[fig_cnt].add_patch(circ)
            axes[fig_cnt].text(pos[0], pos[1]-0.005, f"{i}", color='black', fontsize=30, fontweight='bold',
                               ha='center', va='center', fontname='Times New Roman')

        # Build a mapping from edge to color based on cycles
        edge_color_map = {}
        available_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
        available_colors = ['red', 'blue', 'green', 'black']
        rad_list = [0.1, 0.3, 0.5, 0.2]
        rad_color_dict = dict(zip(available_colors, rad_list))
        for idx, cycle in enumerate(self.dependency_graph.cycles):
            color = available_colors[idx % len(available_colors)]
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]  # cycle around
                if (u, v) not in edge_color_map:
                    edge_color_map[(u, v)] = [color] 
                    edge_color_map[(v, u)] = [color] 
                else:
                    edge_color_map[(u, v)].append(color)
                    edge_color_map[(v, u)].append(color)

        # Now draw arrows
        for dependent, interfering_set in self.dependency_graph.dependency_graph.items():
            dest = self.start_dict[dependent]
            for interfering in interfering_set:
                source = self.start_dict[interfering]

                # For circular nodes, compute edge points along the line connecting centers
                vec = np.array([dest[0] - source[0], dest[1] - source[1]])
                norm = np.linalg.norm(vec) if np.linalg.norm(vec) != 0 else 1.0
                direction = vec / norm
                src_edge = (source[0] + direction[0] * node_radius, source[1] + direction[1] * node_radius)
                dst_edge = (dest[0] - direction[0] * node_radius, dest[1] - direction[1] * node_radius)

                # Color based on cycles
                colors = edge_color_map.get((interfering, dependent), ['black'])
                for color in colors:
                    axes[fig_cnt].annotate(
                        "",
                        xy=src_edge, xycoords='data',
                        xytext=dst_edge, textcoords='data',
                        arrowprops=dict(
                            arrowstyle="->",
                            color=color,
                            lw=1.5,
                            mutation_scale=20,
                            connectionstyle=f"arc3,rad={rad_color_dict[color]}"
                        )
                    )
    
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved 4-panel view to: {save_path}")
            plt.close(fig)
        else:
            plt.show()


def generate_dataset(num_samples: int, folder: str = 'data', object_size=0.05, clearance=0.0, gen_type="random"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for i in range(num_samples):
        # Generate random configuration
        task = Task(i, 0.6, 0.6, object_size=object_size)
        task.generate(gen_type, clearance=clearance)
        filename = f"{folder}/task_random_{i}_objects_{i}.json"
        task.write_to_json(filename)
        print(f"Saved: {filename}")


# debug
if __name__ == '__main__':
    task = Task(6, 0.6, 0.6, 0.05) 
    task.generate_running_example()
    # task.generate("mixed", clearance=0.05)
    # print("task.cycles: ", task.dependency_graph.get_cycles())
    # print("task.chains: ", task.dependency_graph.get_chains())
    # print("task.independent_sets: ", task.dependency_graph.get_independent_sets())
    task.write_to_json("data/task_running_example.json")
    # task = Task.read_from_json("data/task_running_example.json")
    task.draw_all("results/four_panel_running_example.png", overlay=True)
    # task.draw_triptych_to_ppt("results/test.pptx", slide_title="Running Example")
    task.draw_start_goal_dependency_to_ppt("results/test2.pptx", slide_title="Start, Goal and Dependency Graph")
    exit()

    task = Task.read_from_json("data/random_objects_18.json")
    task.draw_all("results/four_panel.png")
    print("task.cycles: ", task.dependency_graph.get_cycles())
    print("task.chains: ", task.dependency_graph.get_chains())
    print("task.independent_sets: ", task.dependency_graph.get_independent_sets())
