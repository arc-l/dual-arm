import numpy as np
import random
from itertools import permutations
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dependency_graph import DependencyGraph
from dataclasses import dataclass
from enum import Enum

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
        self.colors = None
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
        
        self.colors = plt.cm.rainbow(np.linspace(0, 1, self.num_objects))
        self.init()
        
    
    def random_position(self):
        x, y = 2 * (np.random.rand(2)-0.5) * np.array([self.map_width/2 - self.object_size, self.map_length/2 - self.object_size])
        pos = np.array([x, y, self.object_size/2])
        return pos
    
    def generate(self)->None:
        # generate random start and goal positions with the boundary
        # all starts have to be collision free with each other
        self.starts = []
        self.goals = []
        # each start and goal has different colors
        # interpolate color uniformly
        self.colors = plt.cm.rainbow(np.linspace(0, 1, self.num_objects))
        cnt = 0
        for i in range(self.num_objects):
            start = self.random_position() 
            while self.check_collision(self.starts, start):
                start = self.random_position()
                if cnt > 1000:
                    raise RuntimeError('Cannot generate task configuration')
                cnt += 1
            self.starts.append(start)
        
        cnt = 0
        for i in range(self.num_objects):
            goal = self.random_position()
            while self.check_collision(self.goals, goal):
                goal = self.random_position()
                if cnt > 1000:
                    raise RuntimeError('Cannot generate task configuration')
                cnt += 1
            self.goals.append(goal)
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
    
    def get_starts(self):
        return self.starts
    
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

    def plan(self):
        if self.all_arms_available():
            if len(self.dependency_graph.independent_sets) > 1:
                # pick two independent sets
                print("Pick two independent sets.")
                idx_to_be_rearranged = self.dependency_graph.independent_sets
                start_goal_pairs = list(permutations(idx_to_be_rearranged, 2))
                random.shuffle(start_goal_pairs)
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
                
            if self.dependency_graph.has_chains():
                # pick one chain
                print("Pick one chain.")
                chain = self.dependency_graph.get_chains()[0]
                print("chain: ", chain)
                assert len(chain) > 1, "Chain has to have more than one object"
                start_goal_pairs = [(chain[-1], chain[-2]), (chain[-2], chain[-1])]
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
                cycle = self.dependency_graph.get_cycles()[0]
                assert len(cycle) > 1, "Cycle has to have more than one object"
                if len(cycle) == 2:
                    # switch them
                    print("The cycle has two objects. Switch two objects.")
                    print("cycle: ", cycle)
                    task_plan = TaskPlan(
                        arm1_gripper_action=GripperAction.CLOSE,
                        arm2_gripper_action=GripperAction.CLOSE,
                        start_goal_pairs=[(cycle[0], cycle[1]), (cycle[1], cycle[0])],
                        arm1_to_start=True,
                        arm2_to_start=True,
                        arm1_assigned_object=None,
                        arm2_assigned_object=None
                    )
                    return task_plan
                else:
                    # need to break the cycle, need a buffer position to store the object. Need a temporary goal
                    print("The cycle has more than two objects. Break the cycle.")
                    print("cycle: ", cycle)
                    start_goal_pairs = []
                    for i in range(len(cycle)):
                        start_goal_pairs.append((cycle[i], cycle[(i+1)%len(cycle)]))
                        start_goal_pairs.append((cycle[(i+1)%len(cycle)], cycle[i]))
                    # random.shuffle(start_goal_pairs)
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
            print("Both arms are busy, keep finishing up.")
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
            if result.arm1_assigned_object != -1:
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
            if result.arm2_assigned_object != -1:
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


# debug
if __name__ == '__main__':
    task = Task(10, 0.6, 0.6, 0.05)
    task.generate()
    # task.generate_fixed_six_objects()
    # task.generate_fixed_two_objects()
    print("Dependency Graph")
    print(task.dependency_graph.get_dependency_graph())
    print(task.get_independent_sets())
    print("Has cycles: ", task.dependency_graph.has_cycles())
    cycles = task.dependency_graph.get_cycles()
    print("Cycles: ", cycles)
    task.draw()

