import os
import numpy as np
# Third Party
import torch
import pickle
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo 



# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml, get_assets_path
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, MotionGenResult

from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.trajopt import TrajOptSolver, TrajOptSolverConfig
from curobo.rollout.rollout_base import Goal
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.util.trajectory import get_batch_interpolated_trajectory
from curobo.wrap.reacher.evaluator import TrajEvaluator, TrajEvaluatorConfig



config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'assets/curobo_configs'
    )

class Motion_Gen:

    def __init__(self, robot_positions=None, curobo_dt=0.01) -> None:
        self.do_motion_gen = True # initialize motion_gen, if not, the warm-up time will be saved
        if robot_positions is None:
            robot_positions = np.array([[-0.05, 0.0, 0.0], [1.05, 0.0, 0.0]])
        self.update_curobo_robot_pos(robot_positions)
        
        self.tensor_args = TensorDeviceType()
        world_file = "collision_table.yml"
        robot_files = ["ur5e_0.yml","ur5e_1.yml","dual_ur5e.yml"]
        self.world_cfg = WorldConfig.from_dict(load_yaml(join_path(config_path, 'world/' + world_file)))
        self.robot_cfgs = [
            RobotConfig.from_dict(
            # load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
            load_yaml(join_path(config_path, 'robot/'+robot_file))["robot_cfg"]
            )
            for robot_file in robot_files
        ]
        self.curobo_dt = curobo_dt
        
        self.set_default_params()
        
        if self.do_motion_gen:
            self.motion_gen_configs = []
            for i in range(3):
                self.motion_gen_configs.append(
                    MotionGenConfig.load_from_robot_config(
                    self.robot_cfgs[i],
                    self.world_cfg,
                    self.tensor_args,
                    **self.motion_gen_config_params
                )
                )
            self.motion_gens = [
                MotionGen(motion_gen_config) for motion_gen_config in self.motion_gen_configs
            ]
            
            print('warming up motion gen')
            for motion_gen in self.motion_gens:
                motion_gen.warmup(**self.motion_gen_warmup_params)
            self.dual_graph_warmup()
            print('warmup motion gen done')
        # Indicate the current prioritizied arm. 
        # The prioritized arm will be exactly 
        # following its straight line path. 
        # Should be None or 0 or 1
        self.dual_priorities = None 
        ik_configs = []
        for i in range(3):
            ik_configs.append(
                IKSolverConfig.load_from_robot_config(
                self.robot_cfgs[i],
                self.world_cfg,
                **self.ik_config_params
                )
            )
        self.ik_solvers = [
            IKSolver(ik_config) for ik_config in ik_configs
            ]
        
        # trajopt solver
        trajopt_configs = []
        for i in range(3):
            trajopt_configs.append(
                TrajOptSolverConfig.load_from_robot_config(
                    self.robot_cfgs[i], self.world_cfg, self.tensor_args, **self.trajopt_config_params
                    )
            )

        self.trajopt_solvers = [
            TrajOptSolver(trajopt_config) 
            for trajopt_config in trajopt_configs
        ]

        # collision checker
        dual_collision_config = RobotWorldConfig.load_from_config(self.robot_cfgs[2], self.world_cfg,
                                        **self.dual_coll_checker_config_params)
        self.dual_coll_checker = RobotWorld(dual_collision_config)



    def set_default_params(self):
        ## default config parameters
        self.motion_gen_config_params = {
            'interpolation_dt':self.curobo_dt,
            'interpolation_steps':10000,
            "trajopt_tsteps":44,
            'collision_checker_type':CollisionCheckerType.PRIMITIVE,
            'use_cuda_graph':True,
            'num_trajopt_seeds':4,
            'num_graph_seeds':12,
            'collision_cache':{"obb": 30, "mesh": 10},
            'collision_activation_distance':0.025,
            # 'velocity_scale':0.1,
            # 'acceleration_scale':1.0,
            'trajopt_dt':0.2,
            'js_trajopt_dt':0.2,
            'maximum_trajectory_dt':10.0,
            'fixed_iters_trajopt':True,
            'store_debug_in_result':False,
            # 'trim_steps':[1,None],
            'optimize_dt':True,
            'minimize_jerk':True,
            # 'traj_evaluator_config':TrajEvaluatorConfig(cost_weight=0.01)
        }
        
        self.motion_gen_warmup_params = {
            'enable_graph':True, 'warmup_js_trajopt':True
        }
        
        self.ik_config_params = {
            'rotation_threshold':0.005,
            'position_threshold':0.001,
            'num_seeds':20,
            'self_collision_check':False,
            'self_collision_opt':False,
            'tensor_args':self.tensor_args,
            'use_cuda_graph':False,
        }

        self.trajopt_config_params = {
            'use_cuda_graph':False,
            'optimize_dt':True,
            'traj_tsteps':32,
            'interpolation_dt':self.curobo_dt,
        }

        self.dual_coll_checker_config_params = {
            'collision_activation_distance':0.0
        }

        # motion gen plan config parameters
        self.plan_config_params = {
            'enable_graph':True,
            'enable_graph_attempt':4,
            'max_attempts':4,
            'enable_finetune_trajopt':False,
        }


    def motion_gen(self, motion_gen:MotionGen, start_joint_state, goal_ee_pose, **kwargs):
        '''
        not used since the automatically generated IKs are not desired
        '''
        if not self.do_motion_gen:
            raise Exception("Motion Gen is choosed not to be initialized. Set self.do_motion_gen = True if you want to use it.")
        # Standard Library
        PLOT = False
        
        # exit()
        result = motion_gen.plan_single(
            start_joint_state, goal_ee_pose, 
            MotionGenPlanConfig(**self.plan_config_params), **kwargs)
        traj = result.get_interpolated_plan()
        # print(traj)
        # print("Trajectory Generated: ", result.success, result.optimized_dt.item())
        if PLOT:
            self.plot_traj(traj, result.interpolation_dt)
            # plt.save("test.png")
            # plt.close()
            # traj = result.debug_info["opt_solution"].position.view(-1,7)
            # plot_traj(traj.cpu().numpy())
        return result


    def plot_traj(self, trajectory, dt):
        # Third Party
        import matplotlib.pyplot as plt

        _, axs = plt.subplots(4, 1)
        q = trajectory.position.cpu().numpy()
        qd = trajectory.velocity.cpu().numpy()
        qdd = trajectory.acceleration.cpu().numpy()
        qddd = trajectory.jerk.cpu().numpy()
        timesteps = [i * dt for i in range(q.shape[0])]
        for i in range(q.shape[-1]):
            axs[0].plot(timesteps, q[:, i], label=str(i))
            axs[1].plot(timesteps, qd[:, i], label=str(i))
            axs[2].plot(timesteps, qdd[:, i], label=str(i))
            axs[3].plot(timesteps, qddd[:, i], label=str(i))

        plt.legend()
        # plt.savefig("test.png")
        plt.show()


    def single_ik_solve(self, ik_solver, goal:Pose, retract_config, **kwargs):
        retract_tensor = torch.tensor(retract_config, dtype=torch.float32, device='cuda:0').view(1,1,-1)
        result = ik_solver.solve_single(goal, seed_config=retract_tensor, retract_config=retract_tensor, **kwargs)
        # print("IK Solution: ", result.success, result.js_solution)
        return result.js_solution.position


    def trajopt_solve(self, motion_gen:MotionGen, js_start, js_goal, **kwargs):
        if type(js_goal) == JointState:
            js_goal_state = js_goal
        else:
            js_goal_state = JointState.from_position(
                torch.tensor(js_goal, dtype=torch.float32, device='cuda:0').view(1,-1)
            )
        if (type(js_start) == JointState):
            js_start_state = js_start
        else:
            js_start_state = JointState.from_position(
                torch.tensor(js_start, dtype=torch.float32, device='cuda:0').view(1,-1))
        result = motion_gen.plan_single_js(
            js_start_state, js_goal_state,
            plan_config = MotionGenPlanConfig(**self.plan_config_params), **kwargs)
        return result


    def dual_traj_improvement(self,dual_traj):
        '''
        improve motion gen result 
        '''
        goal_pose = Pose()
        js_goal_state = JointState.from_position(dual_traj.position[[-1]])
        js_start_state = JointState.from_position(dual_traj.position[[0]])
        opt_goal = Goal(goal_pose=goal_pose, goal_state=js_goal_state, current_state=js_start_state)
        seed_traj = dual_traj[1:-3]
        result = self.trajopt_solvers[2].solve_single(goal=opt_goal,seed_traj=seed_traj,num_seeds=1)
        return result


    def update_dual_cspace_weight(self, priority:int):
        '''Update the cspace weight of the dual arm planning'''
        reduced_weight = 0.1 # tradeoff between the two arms
        
        if priority is None:
            return False
        if priority not in [0,1]:
            raise Exception("priority must be 0 or 1")
        if self.dual_priorities == priority:
            return False # not updated
        self.dual_priorities = priority
        cspace_weights = np.ones((2,6),dtype=np.float32)
        cspace_weights[1-priority,:] = reduced_weight
        self.robot_cfgs[2].cspace.cspace_distance_weight = torch.tensor(
            torch.tensor(cspace_weights.reshape(-1)), device='cuda:0'
            )
        self.motion_gen_configs[2] = MotionGenConfig.load_from_robot_config(
                self.robot_cfgs[2],
                self.world_cfg,
                self.tensor_args,
                **self.motion_gen_config_params
            )
        self.motion_gens[2] = MotionGen(self.motion_gen_configs[2])
        print('warming up dual arm motion gen')
        self.motion_gens[2].warmup(**self.motion_gen_warmup_params)
        print('warmup dual arm motion gen done')
        return True


    def restore_default_dual_planning(self):
        ''' Restore dual motion planning with default robot config '''
        robot_file = "dual_ur5e.yml"
        self.robot_cfgs[2] = RobotConfig.from_dict(
            # load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
            load_yaml(join_path(config_path, 'robot/'+robot_file))["robot_cfg"]
            )
        
        if self.do_motion_gen:
            self.motion_gen_configs[2] = MotionGenConfig.load_from_robot_config(
                self.robot_cfgs[2],self.world_cfg,self.tensor_args,**self.motion_gen_config_params
            )
            self.motion_gens[2] = MotionGen(self.motion_gen_configs[2])
            
            # Free up the cache
            torch.cuda.empty_cache()
            
            print('warming up default dual motion gen')
            self.motion_gens[2].warmup(**self.motion_gen_warmup_params)
            self.dual_graph_warmup()
            print('warmup motion gen done')


    def trajopt_solve_1220(self, trajopt_solver:TrajOptSolver, js_start, js_goal, **kwargs):
        if type(js_goal) == JointState:
            js_goal_state = js_goal
        else:
            js_goal_state = JointState.from_position(
                torch.tensor(js_goal, dtype=torch.float32, device='cuda:0').view(1,-1)
            )
        if type(js_start) == JointState:
            js_start_state = js_start
        else:
            js_start_state = JointState.from_position(
                torch.tensor(js_start, dtype=torch.float32, device='cuda:0').view(1,-1))
        goal_pose = Pose()
        opt_goal = Goal(goal_pose=goal_pose, goal_state=js_goal_state, current_state=js_start_state)
        result = trajopt_solver.solve_single(opt_goal)
        return result


    def update_curobo_robot_pos(self, positions):
        '''
        positions = np.array([[x1,y1,z1], [x2,y2,z2]])
        '''
        x1,y1,z1 = positions[0]
        x2,y2,z2 = positions[1]
        
        print("this is get assets path", get_assets_path())

        dual_urdf_file = os.path.join(get_assets_path(), 'robot', 'ur_description', 'dual_ur5e.urdf')
        a_file = open(dual_urdf_file, "r")
        list_of_lines = a_file.readlines()
        list_of_lines[74] = "    <origin rpy=\" 0 0 0\" xyz=\""+str(x1)+" "+str(y1)+" "+str(z1)+"\"/>\n"
        list_of_lines[82] = "    <origin rpy=\" 0 0 0\" xyz=\""+str(x2)+" "+str(y2)+" "+str(z2)+"\"/>\n"
        a_file = open(dual_urdf_file, "w")
        a_file.writelines(list_of_lines)
        a_file.close()
        
        urdf0_file = os.path.join(get_assets_path(), 'robot', 'ur_description', 'ur5e_0.urdf')
        a_file = open(urdf0_file, "r")
        list_of_lines = a_file.readlines()
        list_of_lines[122] = "    <origin rpy=\" 0 0 0\" xyz=\""+str(x1)+" "+str(y1)+" "+str(z1)+"\"/>\n"
        a_file = open(urdf0_file, "w")
        a_file.writelines(list_of_lines)
        a_file.close()
        
        
        urdf1_file = os.path.join(get_assets_path(), 'robot', 'ur_description', 'ur5e_1.urdf')
        a_file = open(urdf1_file, "r")
        list_of_lines = a_file.readlines()
        list_of_lines[122] = "    <origin rpy=\" 0 0 0\" xyz=\""+str(x2)+" "+str(y2)+" "+str(z2)+"\"/>\n"
        a_file = open(urdf1_file, "w")
        a_file.writelines(list_of_lines)
        a_file.close()


    def dual_graph_warmup(self):
        '''
        warmup the dual arm graph planner by adding nodes to the graph
        '''
        assist_joint_states = pickle.load(open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),'sampled_joints.pkl'), 
            'rb'))
        
        joint_states_tensor = torch.cat(
            [
                torch.cat([
                    torch.tensor(joint0, dtype=torch.float32, device='cuda:0').view(1,-1),
                    torch.tensor(joint1, dtype=torch.float32, device='cuda:0').view(1,-1)
                    ],dim=-1)
                for joint0 in assist_joint_states[0]
                for joint1 in assist_joint_states[1]
            ], dim=0
        ).view(-1, 12)
        feasible_js = self.motion_gens[2].graph_planner.get_feasible_sample_set(joint_states_tensor)
        feasible_js = torch.cat(
            [feasible_js, torch.ones((feasible_js.shape[0],1), device='cuda:0')], dim=-1
        )
        number_of_nodes = feasible_js.shape[0]
        dof = self.motion_gens[2].graph_planner.dof
        k_nn = self.motion_gens[2].graph_planner.k_nn
        if self.motion_gens[2].graph_planner.i + number_of_nodes >= self.motion_gens[2].graph_planner.path.shape[0]:
            raise ValueError(
                "Path memory buffer is too small", self.motion_gens[2].graph_planner.path.shape[0], self.motion_gens[2].graph_planner.i + number_of_nodes
            )
        self.motion_gens[2].graph_planner.path[self.motion_gens[2].graph_planner.i : self.motion_gens[2].graph_planner.i + number_of_nodes, : dof + 1] = feasible_js
        for i in range(number_of_nodes):
            self.motion_gens[2].graph_planner.path[self.motion_gens[2].graph_planner.i + i, dof + 1] = self.motion_gens[2].graph_planner.i + i

        self.motion_gens[2].graph_planner.i = self.motion_gens[2].graph_planner.i + number_of_nodes
        sample_nodes = feasible_js[:, : self.motion_gens[2].graph_planner.dof]
        self.motion_gens[2].graph_planner.connect_nodes(sample_nodes, lazy=False, k_nn=k_nn)
    
    
    def toppra_improvement(self,traj:JointState, speed_scale=0.1, acc_scale=1.0):
        waypoints_ = traj.position.cpu().numpy()
        max_accelerations = np.array([[-40.0*acc_scale, 40.0*acc_scale]]*traj[0].shape[0])
        max_velocities = np.array([[-np.pi * speed_scale, np.pi * speed_scale]]*traj[0].shape[0])
        path = ta.SplineInterpolator(np.linspace(0, 1, traj.shape[0]), waypoints_)
        pc_vel = constraint.JointVelocityConstraint(max_velocities)
        pc_acc = constraint.JointAccelerationConstraint(max_accelerations)
        instance = algo.TOPPRA([pc_vel, pc_acc], path, solver_wrapper='seidel')
        jnt_traj = instance.compute_trajectory(0, 0) 
        if jnt_traj is None:
            print("Toppra failed")
            return traj
        toppra_duration = jnt_traj.duration
        traj = jnt_traj.eval(np.linspace(0, toppra_duration, int(toppra_duration/(self.curobo_dt))))
        traj = JointState.from_position(torch.tensor(traj, dtype=torch.float32, device='cuda:0').view(-1,waypoints_.shape[1]))
        return traj
        
    

    
if __name__ == "__main__":
    planner = Motion_Gen()
    start_joint_state = JointState(
        torch.tensor([[0.0, -2.51327412, 1.57079633, -0.62831853, -1.57079633, 0.0]],device='cuda:0'),
        torch.tensor([[0., -0., 0., -0., -0., 0.]],device='cuda:0'),
        torch.tensor([[0., -0., 0., -0., -0., 0.]],device='cuda:0')
    )
    goal_ee_pose = Pose(
        torch.tensor([[0.7356781260560767, 0.13790066260528078, 0.3430000000447035]],device='cuda:0'),
        torch.tensor([[ 5.551115123125784e-17, 0.7071067811865476, 5.551115123125784e-17, 0.7071067811865476]],device='cuda:0')
    )
    
    result = planner.motion_gen(planner.motion_gens[0],start_joint_state, goal_ee_pose)
    print(f"type result={type(result)}")
    print(f"result.shape={result.shape}")
    

