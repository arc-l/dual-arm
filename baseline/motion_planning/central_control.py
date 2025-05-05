from pybullet_utils import bullet_client
import pybullet_data
import pybullet as pb
import os
import numpy as np
import math
import time
import copy
from copy import deepcopy
from typing import Dict, List
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from baseline.motion_planning.utils import env_pose_to_sim_pose, Action
from baseline.motion_planning.robot_arm import robot_arm, ControlMode
from baseline.constants import WORKSPACE_SIZE, EPSILON, PNP_ROT
from baseline.motion_planning.plan_recorder import Arm_State, Plan_Recorder
from baseline.tools.util import Colors


class central_control(object):
    ''' centrol controler for rearrangement '''
    def __init__(self, Width, Height, start_arr, goal_arr, radius, dual_action_list, 
                 robot0_position, robot1_position, overlap_rate=0.0, speed = 0.3141592653589793, 
                 time_step_size=1/240.0, control_mode:ControlMode=ControlMode.POSITION, gui:bool=True
                 ) -> None:
        self.start_arr = copy.deepcopy(start_arr)
        self.goal_arr = goal_arr
        self.Width = Width
        self.Height = Height
        self.radius = radius/min(Width,Height)*WORKSPACE_SIZE
        self.robot0_position = robot0_position
        self.robot1_position = robot1_position
        self.overlap_rate = overlap_rate
        self.dual_action_list = dual_action_list
        self.control_mode = control_mode
        self.execution_time = 0
        self.num_conflicts = 0
        self.recorder = Plan_Recorder()

        ''' pybullet parameters '''
        self.time_step = time_step_size # 1/240.0
        self.pnp_steps = 1 # number of timesteps to do a pick or place
        self.gui = gui
        self.speed = speed
        self.arms_home_joints = [
            np.array([0, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi, # [ 0., -2.51327412, 1.57079633, -0.62831853, -1.57079633, 0.]
            np.array([0, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi, # [ -3.1415926., -2.51327412, 1.57079633, -0.62831853, -1.57079633, 0.]
        ]
        self.arms_first_joints = [
            np.array([-0., -2.1633484,   1.7050244,  -1.1028812,  -1.5651518,  0]),
            np.array([0, -2.1607058,  1.7094864, -1.1291537, -1.5776776, 0]),
        ]
        self.arms_ik_rest_joints = [
            np.array([0, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi,
            np.array([0, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi,
        ]
        self.arms_avoid_joints = [
            np.array([-0.87030536, -2.2162654,   1.3529284,  -0.69680774, -1.570098,   -0.5554225]),
            np.array([-0.841997,   -2.3679318,   1.7499532,  -0.96347654, -1.5787702,  -3.525451])
        ]
        self.handoff_position = [0.5, 0.0, 0.7]
        self.handoff_euler = [
            [-math.pi / 2.0, 0.0, -math.pi / 2.0], # [0.0, math.pi / 2.0, 0.0],
            [0.0, -math.pi / 2.0, 0.0]
            ]

        self.setup_env()
        # print('robot 0')
        # print(self.arms[0].action_list)
        # print('robot 1')
        # print(self.arms[1].action_list)


    def __del__(self):
        try:
            self._pb.disconnect()
        except:
            pass

    def setup_env(self):
        '''setup Pybullet env and assign robot controller '''
        
        # Start PyBullet.
        self._pb = bullet_client.BulletClient(connection_mode=pb.GUI if self.gui else pb.DIRECT)
        self._pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._pb.setTimeStep(self.time_step)

        if self.gui:
            self._pb.resetDebugVisualizerCamera(
                cameraDistance=1.2, cameraYaw=510, cameraPitch=-53, cameraTargetPosition=[0.5, 0, 0.23],
            )

        self.reset_env()

    def reset_env(self):
        ''' reset Pybullet env '''
        self._pb.resetSimulation()
        self._pb.setTimeStep(self.time_step)
        print("time step", self.time_step)
        self._pb.setGravity(0, 0, -9.8)

        # # Temporarily disable rendering to load scene faster.
        # if self.gui:
        #     self._pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
        # add environment objs
        self.object_body_ids = self.setup_rearrangement_task_from_dict(self.start_arr, self.goal_arr)
        
        # Load two UR5e
        self.setup_arms()
          # ready pose
        self.arms[0].move_to_joint(self.arms_first_joints[0], 3.14) # move to joint with high speed
        self.arms[1].move_to_joint(self.arms_first_joints[1], 3.14) # move to joint with high speed
        for i in range(500):
            self.stepSimulation()
        # Load workspace
        self.plane = self._pb.loadURDF("plane.urdf", basePosition=(0, 0, -0.0005), useFixedBase=True)
        current_folder = os.path.dirname(os.path.abspath(__file__))
        self.workspace = self._pb.loadURDF(
            os.path.join(current_folder, "assets/workspace/workspace.urdf"),
            basePosition=(0.50, 0, 0.0),
            useFixedBase=True,
        )

        pb.addUserDebugLine(
            (0.5-(self.overlap_rate*WORKSPACE_SIZE/(1+self.overlap_rate)), -WORKSPACE_SIZE/2.0,0.001),
            (0.5-(self.overlap_rate*WORKSPACE_SIZE/(1+self.overlap_rate)), WORKSPACE_SIZE/2.0,0.001),
            (1,1,1),
            lineWidth=3
        )

        pb.addUserDebugLine(
            (0.5+(self.overlap_rate*WORKSPACE_SIZE/(1+self.overlap_rate)), -WORKSPACE_SIZE/2.0,0.001),
            (0.5+(self.overlap_rate*WORKSPACE_SIZE/(1+self.overlap_rate)), WORKSPACE_SIZE/2.0,0.001),
            (1,1,1),
            lineWidth=3
        )

        # self._pb.changeDynamics(
        #     self.plane, -1, lateralFriction=2, restitution=0, linearDamping=2, angularDamping=2,
        # )
        # self._pb.changeDynamics(
        #     self.workspace, -1, lateralFriction=2, restitution=0, linearDamping=2, angularDamping=2,
        # )

      


        # # Re-enable rendering.
        # if self.gui:
        #     self._pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)

      
        self.update_ready_pose()
        
        # reset object position
        for object_idx in self.start_arr.keys():
            self._pb.resetBasePositionAndOrientation(
                self.object_body_ids[object_idx],
                env_pose_to_sim_pose(self.start_arr[object_idx],self.overlap_rate),
                pb.getQuaternionFromEuler([0.0, -0.0, 1.5707964897155762]),
            )
        for _ in range(500):
            self.stepSimulation()
        
            
    def getColorMap(self, obj_list):
        # set colormap
        gist_ncar = plt.get_cmap('gist_ncar')
        cNorm = colors.Normalize(vmin=0, vmax=1)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=gist_ncar)

        color_values = np.linspace(0.1, 0.9, len(obj_list))
        color_pool = {} 
        for ii, obj in enumerate(obj_list):
            color_pool[obj] = scalarMap.to_rgba(color_values[ii])[:3]

        return color_pool
        
    def setup_rearrangement_task_from_dict(self, start_arr, goal_arr):
        color_pool = self.getColorMap(list(start_arr.keys()))
        object_body_ids = {}

        success = True

        num_obj = len(start_arr)
        obj_files = []
        obj_mesh_colors = []
        obj_positions = []
        obj_orientations = []
        for object_idx in list(start_arr.keys()):
            obj_file = os.path.join("assets", "blocks", "cube.urdf")
            obj_files.append(obj_file)
            obj_positions.append(env_pose_to_sim_pose(start_arr[object_idx],self.overlap_rate))
            obj_orientations.append([0.0, -0.0, 1.5707964897155762])
            obj_mesh_colors.append([0.30588235, 0.4745098, 0.65490196])

        # Import objects
        for object_idx in start_arr.keys():
            current_folder = os.path.dirname(os.path.abspath(__file__))
            curr_mesh_file = os.path.join(current_folder, "assets", "blocks", "cube.urdf")
            object_position = env_pose_to_sim_pose(start_arr[object_idx],self.overlap_rate)
            object_orientation = (0.0, -0.0, 1.5707964897155762)
            object_color = list(color_pool[object_idx])+[1.0]
            body_id = self._pb.loadURDF(curr_mesh_file, object_position, pb.getQuaternionFromEuler(object_orientation),)
            self._pb.changeVisualShape(body_id, -1, rgbaColor=object_color)
            self._pb.changeDynamics(body_id, -1, lateralFriction=2, restitution=0, linearDamping=2, angularDamping=2,)
            # for _, info in COLOR_MAP.items():
            #     if np.abs(info.rgb - object_color[:3]).sum() < 1e-3:
            #         self.objects_reward[body_id] = info.reward
            #         break
            object_body_ids[object_idx] = body_id
        
        # Import objects
        for object_idx in start_arr.keys():
            current_folder = os.path.dirname(os.path.abspath(__file__))
            curr_mesh_file = os.path.join(current_folder, "assets", "blocks", "collision_free_cube.urdf")
            object_position = env_pose_to_sim_pose(goal_arr[object_idx],self.overlap_rate)
            object_orientation = (0.0, -0.0, 1.5707964897155762)
            object_color = list(color_pool[object_idx])+[0.5]
            body_id = self._pb.loadURDF(curr_mesh_file, object_position, pb.getQuaternionFromEuler(object_orientation),)
            self._pb.changeVisualShape(body_id, -1, rgbaColor=object_color)
            # for _, info in COLOR_MAP.items():
            #     if np.abs(info.rgb - object_color[:3]).sum() < 1e-3:
            #         self.objects_reward[body_id] = info.reward
            #         break
            # object_body_ids[object_idx] = body_id

        # give time to stop
        for _ in range(500):
            self.stepSimulation()

        return object_body_ids


    ''' sync between arms '''
    def setup_arms(self):
        self.arms = [
            robot_arm(
                arm_id=0, 
                Width=self.Width, 
                Height=self.Height,
                start_arr=self.start_arr, 
                goal_arr=self.goal_arr, 
                radius=self.radius, 
                task_list=self.dual_action_list[0], 
                pb_env=self._pb, 
                object_body_ids=self.object_body_ids, 
                position=self.robot0_position, 
                overlap_rate=self.overlap_rate, 
                speed=self.speed, 
                pnp_steps=self.pnp_steps, 
                control_mode=self.control_mode,
                time_step=self.time_step
                ),
            robot_arm(
                arm_id=1, 
                Width=self.Width, 
                Height=self.Height,
                start_arr=self.start_arr, 
                goal_arr=self.goal_arr, 
                radius=self.radius, 
                task_list=self.dual_action_list[1], 
                pb_env=self._pb, 
                object_body_ids=self.object_body_ids, 
                position=self.robot1_position, 
                overlap_rate=self.overlap_rate, 
                speed=self.speed, 
                pnp_steps=self.pnp_steps, 
                control_mode=self.control_mode,
                time_step=self.time_step,
                orientation=self._pb.getQuaternionFromEuler([0.0, -0.0, 3.141592653589793])
                )
        ]
        
        for armID, arm in enumerate(self.arms):
            arm.arm_home_joints = self.arms_home_joints[armID]
            arm.arm_first_joints = self.arms_first_joints[armID]
            arm.arm_ik_rest_joints = self.arms_ik_rest_joints[armID]
            arm.arm_avoid_joints = self.arms_avoid_joints[armID]
            arm.TheOtherArm = self.arms[1-armID]
            arm.handoff_position = copy.deepcopy(self.handoff_position)
            arm.handoff_euler = self.handoff_euler[armID]
            if armID == 0:
                arm.handoff_position[0] += 0.00
                arm.handoff_position[0] -= (arm.ee_tip_offset + arm.buffer_height)
                arm.handoff_preparation_position = copy.deepcopy(arm.handoff_position)
                arm.handoff_preparation_position[0] -= 0.1
            else:
                arm.handoff_position[0] -= 0.00
                arm.handoff_position[0] += (arm.ee_tip_offset + arm.buffer_height)
                arm.handoff_preparation_position = copy.deepcopy(arm.handoff_position)
                arm.handoff_preparation_position[0] += 0.1
        self.get_yielding_pose()
        self.get_key_positions()
        for armID, arm in enumerate(self.arms):
            arm.task_list_2_action_list()

    def update(self):
        ''' when one robot is not finished, update and step sim '''
        robot0 = self.arms[0]
        robot1 = self.arms[1]
        # input('start')
        # planning = False
        planning = True
        if planning:
            robot0.switch_status()
            robot1.switch_status()
            while (not robot0.finished) or (not robot1.finished):
                if robot0.pause and robot1.pause:
                    self.handoff_coordination()
                    robot0.pause = False
                    robot1.pause = False
                    continue
                robot0.update()
                robot1.update()
                self.recorder.add_new_step(
                    self.update_arm_state()
                )
                self.stepSimulation()
                
                self.update_priority()
            self.recorder.write_record()
        self.recorder.read_record()
        new_records = dict()
        for armID in range(2):
            new_records[armID] = self.arms[armID].motion_improvement(self.recorder.record[armID])
        self.recorder.record = new_records
        self.recorder.write_record()
        # count number of conflicts
        self.num_conflicts = (robot0.num_conflicts + robot1.num_conflicts)*self.time_step
        # input("ready")
        # self.record_replay()

    def update_ready_pose(self):
        ''' update_ready pose based on overlap rate '''
        # pnp_rot = pb.getQuaternionFromEuler([np.pi / 2, np.pi / 2, np.pi / 2])
        pnp_rot = [-0.5877976611933862, 0.8089882123010008, 0.005586555396007588,0.0008787789389623685]
        robot0 = self.arms[0]
        robot1 = self.arms[1]
        # robot 0
        joint_state = robot0.arm_first_joints
        finish = False
        while not finish:
            finish = robot0.next_joint_state(joint_state, self.speed)
            self.stepSimulation()
        x_pos = min(
            0.5-WORKSPACE_SIZE*(1/(1+self.overlap_rate)), # border
            0.5-(self.overlap_rate*WORKSPACE_SIZE/(1+self.overlap_rate)) # the other arm's reachable border
            # 0.5-(self.overlap_rate*WORKSPACE_SIZE/(1+self.overlap_rate))
            )
        pos = ((x_pos, 0.0, 0.588), pnp_rot)
        robot0.arm_first_joints = robot0.solve_ik(pos)
        # print('ready')
        # print('robot0',robot0.arm_first_joints)

        # robot 1
        joint_state = robot1.arm_first_joints
        finish = False
        while not finish:
            finish = robot1.next_joint_state(joint_state, self.speed)
            self.stepSimulation()
        x_pos = max(
            0.5+WORKSPACE_SIZE*(1/(1+self.overlap_rate)), # border
            0.5+(self.overlap_rate*WORKSPACE_SIZE/(1+self.overlap_rate)) # the other arm reachable border
            # 0.5+(self.overlap_rate*WORKSPACE_SIZE/(1+self.overlap_rate))
            )
        pos = ((x_pos, 0.0, 0.588), pnp_rot)
        robot1.arm_first_joints = robot1.solve_ik(pos)
        # print('robot1',robot1.arm_first_joints)

    def get_yielding_pose(self):
        ''' get yielding pose '''
        # pnp_rot = pb.getQuaternionFromEuler([np.pi / 2, np.pi / 2, np.pi / 2])
        pnp_rot = PNP_ROT
        robot0 = self.arms[0]
        robot1 = self.arms[1]
        
        # ### until 0219 yield = home
        # robot0.arm_yielding_joints = robot0.arm_home_joints
        # robot1.arm_yielding_joints = robot1.arm_home_joints
        # return 

        ### yield to edge
        # robot 0
        joint_state = robot0.arm_first_joints
        finish = False
        while not finish:
            finish = robot0.next_joint_state(joint_state, self.speed)
            self.stepSimulation()
        x_pos = self.robot0_position[0]+0.15
        pos = ((x_pos, 0.0, 0.7), pnp_rot)
        robot0.arm_yielding_joints = robot0.solve_ik(pos)

        # robot 1
        joint_state = robot1.arm_first_joints
        finish = False
        while not finish:
            finish = robot1.next_joint_state(joint_state, self.speed)
            self.stepSimulation()
        x_pos = self.robot1_position[0]-0.15
        pos = ((x_pos, 0.0, 0.588), pnp_rot)
        robot1.arm_yielding_joints = robot1.solve_ik(pos)
        # print('avoid')
        # print('robot0',robot0.arm_yielding_joints)
        # print('robot1',robot1.arm_yielding_joints)

    def get_key_positions(self):
        for armID in range(2):
            robot = self.arms[armID]
            robot.arm_home_pos = robot.get_pos(robot.arm_home_joints)
            robot.arm_first_pos = robot.get_pos(robot.arm_first_joints)
            robot.arm_ik_rest_pos = robot.get_pos(robot.arm_ik_rest_joints)
            robot.arm_avoid_pos = robot.get_pos(robot.arm_avoid_joints)
            robot.arm_yielding_pos = robot.get_pos(robot.arm_yielding_joints)
            


    def test(self):
        print('execution time', self.execution_time)
        robot0 = self.arms[0]
        robot1 = self.arms[1]
        # self.attach_objects()
        # input('start')
        pnp_rot = pb.getQuaternionFromEuler([np.pi / 2, np.pi / 2, np.pi / 2])
        joint_state = np.array([ -0.12180982, -0.77299047, 1.2969291, -2.0945518, -1.5707891, -0.12179157])
        finish = False
        while not finish:
            finish = robot0.next_joint_state(joint_state, self.speed)
            self.stepSimulation()
        joint_state = robot1.arm_first_joints
        finish = False
        while not finish:
            finish = robot1.next_joint_state(joint_state, self.speed)
            self.stepSimulation()
        pos = self._pb.getLinkState(robot0.ur5e, robot0.ur5e_ee_id)
        print('first',pos)
        input('finish')
        return
        pos = (np.array(robot0.handoff_position)+np.array([0.00,0.0,0.0]), pb.getQuaternionFromEuler(robot0.handoff_euler))
        joint_state = robot0.solve_ik(pos)
        finish = False
        while not finish:
            finish = robot0.next_joint_state(joint_state, self.speed)
            self.stepSimulation()
        pos = (np.array(robot1.handoff_position)+np.array([-0.00,0.0,0.0]), pb.getQuaternionFromEuler(robot1.handoff_euler))
        joint_state = robot1.solve_ik(pos)
        finish = False
        while not finish:
            finish = robot1.next_joint_state(joint_state, self.speed)
            self.stepSimulation()
        pos = (np.array(robot1.handoff_position)+np.array([-0.01,0.0,0.0]), pb.getQuaternionFromEuler(robot1.handoff_euler))
        joint_state = robot1.solve_ik(pos)
        finish = False
        while not finish:
            finish = robot1.next_joint_state(joint_state, self.speed)
            self.stepSimulation()
        # pos = ((0.65,0.20,0.2),pnp_rot)
        # joint_state = robot1.solve_ik(pos)
        # finish = False
        # while not finish:
        #     finish = robot1.next_joint_state(joint_state, self.speed)
        #     self.stepSimulation()
        for _ in range(50):
            self.stepSimulation()
        pos = self._pb.getLinkState(robot0.ur5e, robot0.ur5e_ee_id)
        print('final',pos)
        for _ in range(50):
            self.stepSimulation()
        pos = self._pb.getLinkState(robot1.ur5e, robot1.ur5e_ee_id)
        print('final',pos)
        # print('joint_state',joint_state)
        input('finish')


    def attach_objects(self):
        ''' attach objects before collision check '''
        current_folder = os.path.dirname(os.path.abspath(__file__))
        curr_mesh_file = os.path.join(current_folder, "assets", "blocks", "cylinder.urdf")
        object_orientation = (0.0, -0.0, 1.5707964897155762)
        object_color = [
            0.30588235,
            0.4745098,
            0.65490196,
            1,
        ]
        for arm in [self.arms[0]]:
            pos = self._pb.getLinkState(arm.ee, arm.ee_tip_id)
            object_position = list(pos[0])
            object_position[2] -= arm.buffer_height - 0.01
            body_id = self._pb.loadURDF(curr_mesh_file, tuple(object_position), pb.getQuaternionFromEuler(object_orientation),)
            self._pb.changeVisualShape(body_id, -1, rgbaColor=object_color)
            arm.activate(body_id)
            arm.current_object = body_id


    def update_priority(self):
        ''' update yielding property '''
        robot0 = self.arms[0]
        robot1 = self.arms[1]
        robot0_height = self._pb.getLinkState(robot0.ee, 0)[0][2]
        robot1_height = self._pb.getLinkState(robot1.ee, 0)[0][2]
        robot0_x = self._pb.getLinkState(robot0.ee, 0)[0][0]
        robot1_x = self._pb.getLinkState(robot1.ee, 0)[0][0]
        if robot0.finished:
            robot0.yielding = True
            robot1.yielding = False
        elif robot1.finished:
            robot0.yielding = False
            robot1.yielding = True
        # elif robot0.start_time <= robot1.start_time:
        elif robot0_height <= robot1_height:
            # start time of tasks
            robot0.yielding = False
            robot1.yielding = True
        else:
            robot0.yielding = True
            robot1.yielding = False
        if robot1_x < robot0_x+0.0: # deadlock is possible
            robot0.clearance = 0.2
            robot1.clearance = 0.2
        else:
            robot0.clearance = 0.1
            robot1.clearance = 0.1

    def curobo_handoff_coordination(
            self, current_object: int = None, deliver_arm:robot_arm = None, receive_arm:robot_arm = None, 
            horizon=1, curr_joint_states:Dict=None
            ):
        
        
        if curr_joint_states is None:
            curr_joint_states = {
                0: self.arms[0].get_joint_state(),
                1: self.arms[1].get_joint_state()
            }


        deliver_arm.move_to_joint(curr_joint_states[deliver_arm.arm_id],speed=3.14)
        receive_arm.move_to_joint(curr_joint_states[receive_arm.arm_id],speed=3.14)
        for i in range(500):
            self.stepSimulation()

        deliver_handoff_rot = pb.getQuaternionFromEuler(deliver_arm.handoff_euler)
        deliver_joint_pose = deliver_arm.solve_ik(
            (deliver_arm.handoff_position, deliver_handoff_rot)
        )
        receive_handoff_rot = pb.getQuaternionFromEuler(receive_arm.handoff_euler)
        receive_joint_pose = receive_arm.solve_ik(
            (receive_arm.handoff_position, receive_handoff_rot)
        )

        deliver_preparation_joint_pose = curr_joint_states[deliver_arm.arm_id]
        receive_preparation_joint_pose = curr_joint_states[receive_arm.arm_id]

        # set the joints to the preparation pose
        deliver_arm.move_to_joint(deliver_preparation_joint_pose, 3.14)
        receive_arm.move_to_joint(receive_preparation_joint_pose, 3.14)
        for i in range(500):
            self.stepSimulation()
            
        # go to position
        deliver_ready = False
        receive_ready = False
        while (not deliver_ready) or (not receive_ready):
            deliver_arm.next_joint_state(deliver_joint_pose, deliver_arm.speed*horizon)
            receive_arm.next_joint_state(receive_joint_pose, receive_arm.speed*horizon)
            for _ in range(horizon):
                self.stepSimulation()
            arm_state_dict = dict()
            for armID in range(2):
                curr_state = self.arms[armID].get_joint_state()
                arm_state_dict[armID] = Arm_State(
                    self.arms[armID].get_ee_qua_pose(),
                    curr_state,
                    current_object,'T','G',
                )
                if armID == deliver_arm.arm_id:
                    diff_joints = np.linalg.norm(self.arms[armID].get_joint_state()-deliver_joint_pose)
                    if diff_joints < EPSILON:
                        deliver_ready = True
                elif armID == receive_arm.arm_id:
                    diff_joints = np.linalg.norm(self.arms[armID].get_joint_state()-receive_joint_pose)
                    if diff_joints < EPSILON:
                        receive_ready = True
            self.recorder.add_new_step(arm_state_dict)
                
        
        #exchange
        for i in range(self.pnp_steps//2):
            for _ in range(horizon):
                self.stepSimulation()
            arm_state_dict = dict()
            for armID in range(2):
                curr_state = self.arms[armID].get_joint_state()
                if armID == deliver_arm.arm_id:
                    motion_status = 'T'
                elif armID == receive_arm.arm_id:
                    motion_status = 'A'
                arm_state_dict[armID] = Arm_State(
                    self.arms[armID].get_ee_qua_pose(),
                    curr_state,
                    current_object,motion_status,'G',
                )
            self.recorder.add_new_step(arm_state_dict)
            
        for i in range(self.pnp_steps//2+self.pnp_steps%2):
            for _ in range(horizon):
                self.stepSimulation()
            arm_state_dict = dict()
            for armID in range(2):
                curr_state = self.arms[armID].get_joint_state()
                if armID == deliver_arm.arm_id:
                    motion_status = 'D'
                elif armID == receive_arm.arm_id:
                    motion_status = 'T'
                arm_state_dict[armID] = Arm_State(
                    self.arms[armID].get_ee_qua_pose(),
                    curr_state,
                    current_object,motion_status,'G',
                )
            self.recorder.add_new_step(arm_state_dict)

        # withdraw
        deliver_ready = False
        receive_ready = False
        while (not deliver_ready) or (not receive_ready):
            deliver_arm.next_joint_state(deliver_preparation_joint_pose, deliver_arm.speed*horizon)
            receive_arm.next_joint_state(receive_preparation_joint_pose, receive_arm.speed*horizon)
            for _ in range(horizon):
                self.stepSimulation()
            arm_state_dict = dict()
            for armID in range(2):
                curr_state = self.arms[armID].get_joint_state()
                arm_state_dict[armID] = Arm_State(
                    self.arms[armID].get_ee_qua_pose(),
                    curr_state,
                    current_object,'T','G',
                )
                if armID == deliver_arm.arm_id:
                    diff_joints = np.linalg.norm(self.arms[armID].get_joint_state()-deliver_preparation_joint_pose)
                    if diff_joints < EPSILON:
                        deliver_ready = True
                elif armID == receive_arm.arm_id:
                    diff_joints = np.linalg.norm(self.arms[armID].get_joint_state()-receive_preparation_joint_pose)
                    if diff_joints < EPSILON:
                        receive_ready = True
            self.recorder.add_new_step(arm_state_dict)

        # to rest
        to_rest = [False, False]
        while not (to_rest[0] and to_rest[1]):
            deliver_arm.next_joint_state(deliver_arm.arm_first_joints,deliver_arm.speed*horizon)
            receive_arm.next_joint_state(receive_arm.arm_first_joints,receive_arm.speed*horizon)
            for _ in range(horizon):
                self.stepSimulation()
            arm_state_dict = dict()
            for armID in range(2):
                curr_state = self.arms[armID].get_joint_state()
                arm_state_dict[armID] = Arm_State(
                    self.arms[armID].get_ee_qua_pose(),
                    curr_state,
                    current_object,'T','G',
                )
                diff_joints = np.linalg.norm(self.arms[armID].get_joint_state()-self.arms[armID].arm_first_joints, ord=np.inf)
                if diff_joints < EPSILON:
                    to_rest[armID] = True
            self.recorder.add_new_step(arm_state_dict)
        
        
    def handoff_coordination(self, curr_joint_states:Dict=None):
        
        if curr_joint_states is None:
            curr_joint_states = {
                0: self.arms[0].get_joint_state(),
                1: self.arms[1].get_joint_state()
            }

        if self.arms[0].current_object == None:
            deliver_arm:robot_arm = self.arms[1]
            receive_arm:robot_arm = self.arms[0]
        else:
            deliver_arm:robot_arm = self.arms[0]
            receive_arm:robot_arm = self.arms[1]

        deliver_arm.move_to_joint(curr_joint_states[deliver_arm.arm_id],speed=3.14)
        receive_arm.move_to_joint(curr_joint_states[receive_arm.arm_id],speed=3.14)
        for i in range(500):
            self.stepSimulation()

        deliver_handoff_rot = pb.getQuaternionFromEuler(deliver_arm.handoff_euler)
        deliver_joint_pose = deliver_arm.solve_ik(
            (deliver_arm.handoff_position, deliver_handoff_rot)
        )
        receive_handoff_rot = pb.getQuaternionFromEuler(receive_arm.handoff_euler)
        receive_joint_pose = receive_arm.solve_ik(
            (receive_arm.handoff_position, receive_handoff_rot)
        )

        deliver_preparation_joint_pose = curr_joint_states[deliver_arm.arm_id]
        receive_preparation_joint_pose = curr_joint_states[receive_arm.arm_id]

        # set the joints to the preparation pose
        deliver_arm.move_to_joint(deliver_preparation_joint_pose, 3.14)
        receive_arm.move_to_joint(receive_preparation_joint_pose, 3.14)
        for i in range(500):
            self.stepSimulation()


        # go to position
        deliver_ready = False
        receive_ready = False
        while (not deliver_ready) or (not receive_ready):
            if not deliver_ready:
                deliver_ready = deliver_arm.next_joint_state(deliver_joint_pose, self.speed)
            if not receive_ready:
                receive_ready = receive_arm.next_joint_state(receive_joint_pose, self.speed)
            self.recorder.add_new_step(self.update_arm_state())
            self.stepSimulation()
        
        # for i in range(5):
        #     self.recorder.add_new_step(self.update_arm_state())
        #     self.stepSimulation()
        
        #exchange
        receive_arm.activate(deliver_arm.current_object)
        for i in range(self.pnp_steps//2):
            self.recorder.add_new_step(self.update_arm_state())
            self.stepSimulation()
        deliver_arm.deactivate(deliver_arm.current_object)
        for i in range(self.pnp_steps//2+self.pnp_steps%2):
            self.recorder.add_new_step(self.update_arm_state())
            self.stepSimulation()


        
        # withdraw
        deliver_ready = False
        receive_ready = False
        while (not deliver_ready) or (not receive_ready):
            if not deliver_ready:
                deliver_ready = deliver_arm.next_joint_state(deliver_preparation_joint_pose, self.speed)
            if not receive_ready:
                receive_ready = receive_arm.next_joint_state(receive_preparation_joint_pose, self.speed)
            self.recorder.add_new_step(self.update_arm_state())
            self.stepSimulation()

        # to rest
        deliver_rest_ready = False
        receive_rest_ready = False
        while (not deliver_rest_ready) or (not receive_rest_ready):
            if not deliver_rest_ready:
                deliver_rest_ready = deliver_arm.next_joint_state(deliver_arm.arm_first_joints, self.speed)
            if not receive_rest_ready:
                receive_rest_ready = receive_arm.next_joint_state(receive_arm.arm_first_joints, self.speed)
            self.recorder.add_new_step(self.update_arm_state())
            self.stepSimulation()
        
        # for i in range(5):
        #     self.recorder.add_new_step(self.update_arm_state())
        #     self.stepSimulation()


    def stepSimulation(self, real_time=False):
        self._pb.stepSimulation()
        if real_time:
            time.sleep(self.time_step)
        self.execution_time += self.time_step


    def clear_time(self):
        self.execution_time = 0


    ''' for recording '''
    def update_arm_state(self):
        ''' update arm state '''
        arm_state_dict = dict()
        for armID in range(2):
            if self.arms[armID].motion_status in ['A', 'D']:
                if self.arms[armID].current_object == None:  # handoff exchange
                    current_object = self.arms[armID].TheOtherArm.current_object
                else:
                    current_object = self.arms[armID].current_object
            else:
                current_object = self.arms[armID].current_object
            if self.arms[armID].to_rest:
                current_mode = "A"
            else:
                current_mode = "G"
            arm_state_dict[armID] = Arm_State(
                self.arms[armID].get_ee_qua_pose(),
                self.arms[armID].get_joint_state(),
                current_object,
                self.arms[armID].motion_status,
                current_mode
            )
            self.arms[armID].motion_status = self.arms[armID].status
        return arm_state_dict      

    
    def record_replay(self):
        self.reset_env()
        self.clear_time()
        horizon = 1
        for time_step in range(len(self.recorder.record[0])):
            if time_step % horizon == 0:
                for armID in self.recorder.record.keys():
                    arm_state:Arm_State = self.recorder.record[armID][time_step]
                #     # arm_js = JointState(
                #     #     position=torch.tensor(arm_state.joint_state, dtype=torch.float32, device='cuda:0').view(-1,1),
                #     #     velocity=torch.tensor(arm_state.velocity, dtype=torch.float32, device='cuda:0').view(-1,1),
                #     #     acceleration=torch.tensor(arm_state.acceleration, dtype=torch.float32, device='cuda:0').view(-1,1),
                #     #     jerk=torch.tensor(arm_state.jerk, dtype=torch.float32, device='cuda:0').view(-1,1),
                #     # )
                #     # self.arms[armID].move_to_joint(arm_js)
                    if arm_state.motion_status == 'A':
                        self.arms[armID].activate(arm_state.involved_object)
                    elif arm_state.motion_status == 'D':
                        self.arms[armID].deactivate(arm_state.involved_object)
                # dual_position = torch.tensor(
                #     np.concatenate([self.recorder.record[i][time_step].joint_state for i in range(2)],axis=-1), 
                #     dtype=torch.float32, device='cuda:0'
                #     ).view(-1,1)
                # dual_velocity = torch.tensor(
                #     np.concatenate([self.recorder.record[i][time_step].velocity for i in range(2)],axis=-1), 
                #     dtype=torch.float32, device='cuda:0'
                #     ).view(-1,1)
                # dual_acceleration = torch.tensor(
                #     np.concatenate([self.recorder.record[i][time_step].acceleration for i in range(2)],axis=-1), 
                #     dtype=torch.float32, device='cuda:0'
                #     ).view(-1,1)
                # dual_jerk = torch.tensor(
                #     np.concatenate([self.recorder.record[i][time_step].jerk for i in range(2)],axis=-1), 
                #     dtype=torch.float32, device='cuda:0'
                #     ).view(-1,1)
                # dual_js = JointState(
                #     position=dual_position,
                #     velocity=dual_velocity,
                #     acceleration=dual_acceleration,
                #     jerk=dual_jerk,
                # )
                dual_js = np.concatenate([self.recorder.record[i][time_step].joint_state for i in range(2)],axis=-1)
                self.move_to_dual_state(dual_js, real_time=True)
    
    
    def move_to_dual_state(self, dual_state, real_time=False):
        prev_dist = float('inf')
        for arm_id in range(2):
            self.arms[arm_id].move_to_joint(dual_state[(arm_id*6):((arm_id+1)*6)], speed=self.speed)
        
        target_dual_state = dual_state
        while True:
            for _ in range(1):
                self.stepSimulation(real_time=real_time)
            ''' check distance '''
            current_dual_state = np.concatenate([self.arms[arm_id].get_joint_state() for arm_id in range(2)], axis=-1).reshape(-1)
            distance = np.linalg.norm(current_dual_state-target_dual_state, ord=2)
            if distance < EPSILON: # at neighborhood
                return True
            if prev_dist - distance < 0: # if distance is increasing
                return False
            prev_dist = distance
            
            
    def constraint_speeds(self):
        interpolate = lambda v1, v2, t: (1 - t) * v1 + t * v2
        # for arm_id in range(2):
        #     record = self.recorder.record[arm_id]
        i = 0
        while i < len(self.recorder.record[0]) - 1:
            velocity0 = np.abs((self.recorder.record[0][i + 1].joint_state - self.recorder.record[0][i].joint_state) / self.time_step)
            velocity1 = np.abs((self.recorder.record[1][i + 1].joint_state - self.recorder.record[1][i].joint_state) / self.time_step)
            if any(velocity0 > self.speed*1.05) or any(velocity1 > self.speed*1.05):
                interpolation_number = max( 
                    int(np.ceil(np.max(np.abs(velocity0)/self.speed)) ),
                    int(np.ceil(np.max(np.abs(velocity1)/self.speed)) ),
                )
                print(Colors.green(f"Interpolating arm between {i} and {i + 1}, interpolation number = {interpolation_number}"))
                new_state0 = [
                    interpolate(self.recorder.record[0][i].joint_state, 
                                self.recorder.record[0][i + 1].joint_state, 
                                j/interpolation_number
                                ) 
                    for j in range(1, interpolation_number)
                    ]
                new_state1 = [
                    interpolate(self.recorder.record[1][i].joint_state, 
                                self.recorder.record[1][i + 1].joint_state, 
                                j/interpolation_number
                                ) 
                    for j in range(1, interpolation_number)
                    ]
                for j in range(0, len(new_state0)):
                    self.recorder.record[0].insert(i + j + 1, deepcopy(self.recorder.record[0][i]))
                    self.recorder.record[0][i + j + 1].joint_state = new_state0[j]
                    self.recorder.record[1].insert(i + j + 1, deepcopy(self.recorder.record[1][i]))
                    self.recorder.record[1][i + j + 1].joint_state = new_state1[j]
                i += int(interpolation_number)
            else:
                i += 1

