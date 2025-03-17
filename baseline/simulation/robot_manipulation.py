import sys
import os
from colorama import Fore
from threading import Lock
import time

import pybullet as pb
import numpy as np

from .utils import env_pose_to_sim_pose


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environment_sim import Environment, handoff_position, handoff_euler


class robot_manipulation(object):
    """
    Individual robot operations
    """

    def __init__(
        self,
        env: Environment,
        plan: list,
        arm_idx: int,
        start_arr: dict,
        goal_arr: dict,
        state: int,
        obj_state: list,
        state_lock: Lock,
        tie_arm_idx: list,
        log_count: list,
        log_id: list,
        logging: bool,
        count_plan: list,
    ) -> None:
        self.env = env
        self.plan = plan
        self.arm_idx = arm_idx
        self.start_arr = start_arr
        self.goal_arr = goal_arr
        self.state = state
        self.obj_state = obj_state  # format [[a0_obj_info, a0_gripper_act], [a1_obj_info, a1_gripper_act]]
        self.state_lock = state_lock
        self.tie_arm_idx = tie_arm_idx  # avoid one arm keeps doing job while the other is waiting all the time
        self.log_count = log_count
        self.log_id = log_id
        self.logging = logging
        self.ready = False
        self.paused = True
        self.finished = False
        self.handoff_obj = None
        self.pass_obj = True
        self.grasp_pose = None
        self.place_pose = None
        self.speed = 0.002
        self.collision_count = {}
        self.count_plan = count_plan

    def execution(self):
        """
        decide what to do next
        """
        success = False
        while True:
            if not self.paused:
                extra_wait = False
                self.env.arms[self.arm_idx].control_state = "free"
                if len(self.plan) == 0:  # finished all actions in the plan
                    self.env.go_home(self.arm_idx, speed=self.speed, check_collision=True)
                    self.finished = True
                    self.paused = True
                    self.tie_arm_idx[0] = 1 - self.arm_idx
                    print(Fore.GREEN + "robot " + str(self.arm_idx) + "has finished tasks.")
                    return
                next_action = self.plan.pop(0)
                if next_action == None:
                    print(Fore.RED + "why the plan has none inside?")
                    time.sleep(3)
                    continue
                obj_id, start_pose, goal_pose = next_action
                print(Fore.GREEN + f"{self.arm_idx} {obj_id} {start_pose} {goal_pose}")
                if start_pose == "H":  # pick up something from handoff
                    self.handoff_obj = obj_id
                    self.pass_obj = False
                    self.plan.insert(0, (obj_id, "G", goal_pose))
                    success = self.prepare_for_handoff(self.speed, check_collision=True)
                    # time.sleep(1)

                    if success:
                        with self.state_lock:
                            self.set_center_control(True)
                            print(Fore.GREEN + f"{self.arm_idx} success lock - prepare")
                            other_collides = self.env.is_down_move_and_detect_obstacle(1 - self.arm_idx)
                            print(f"other {other_collides}")
                            if other_collides:
                                success = False
                            if self.logging:
                                self.log(success)
                            if not success:
                                self.env.arms[1 - self.arm_idx].control_state = "restart"
                                state_id = self.state[0]
                                self.check_and_restore(state_id)
                                self.env.hold_joints([self.arm_idx, 1 - self.arm_idx])
                                if self.logging:
                                    self.log(success)
                                success = self.env.go_home(
                                    1 - self.arm_idx,
                                    speed=self.speed,
                                    check_collision=True,
                                    center_control_override=True,
                                )
                                if success:
                                    success = self.prepare_for_handoff(
                                        self.speed, check_collision=True, center_control_override=True
                                    )
                                if self.logging:
                                    self.log(success)
                                if not success:
                                    print(Fore.RED + "??? prepare_for_handoff has deadlock")
                                    input("wait")
                                self.collision_count[str(self.count_plan)] = 1
                            if success:
                                self.paused = True
                                self.ready = True
                                self.state[0] = self.env._pb.saveState()
                                self.record_obj_state()
                            print(Fore.GREEN + f"{self.arm_idx} success unlock - prepare")
                            self.set_center_control(False)
                    else:
                        if self.arm_idx == self.tie_arm_idx[0]:
                            with self.state_lock:
                                self.set_center_control(True)
                                self.env.arms[self.arm_idx].control_state = "restart"
                                print(Fore.GREEN + f"{self.arm_idx} no success lock - prepare")
                                state_id = self.state[0]
                                self.check_and_restore(state_id)
                                self.env.hold_joints([self.arm_idx, 1 - self.arm_idx])
                                if self.logging:
                                    self.log(success)
                                success = self.env.go_home(
                                    1 - self.arm_idx,
                                    speed=self.speed,
                                    check_collision=True,
                                    center_control_override=True,
                                )
                                if success:
                                    success = self.prepare_for_handoff(
                                        self.speed, check_collision=True, center_control_override=True
                                    )
                                if self.logging:
                                    self.log(success)
                                if success:
                                    self.paused = True
                                    self.ready = True
                                    self.state[0] = self.env._pb.saveState()
                                    self.record_obj_state()
                                else:
                                    self.check_and_restore(state_id)
                                    self.env.hold_joints([self.arm_idx, 1 - self.arm_idx])
                                    if self.logging:
                                        self.log(success)
                                    go_home_success = self.env.go_home(
                                        self.arm_idx,
                                        speed=self.speed,
                                        check_collision=True,
                                        center_control_override=True,
                                    )
                                    if self.logging:
                                        self.log(go_home_success)
                                    if go_home_success:
                                        self.state[0] = self.env._pb.saveState()
                                        self.record_obj_state()
                                    else:
                                        print(Fore.RED + f"Error: prepare_for_handoff could have deadlock...")
                                        input("press to continue")
                                    self.plan.pop(0)
                                self.tie_arm_idx[0] = 1 - self.tie_arm_idx[0]
                                self.collision_count[str(self.count_plan)] = 1
                                print(Fore.GREEN + f"{self.arm_idx} no success unlock - prepare")
                                self.set_center_control(False)
                        else:
                            print(Fore.GREEN + f"{self.arm_idx} wait for tie")
                            while self.arm_idx != self.tie_arm_idx[0]:
                                time.sleep(0.1)
                elif goal_pose == "H":  # pass something to the other side
                    self.handoff_obj = obj_id
                    self.pass_obj = True
                    success = self.grasp(env_pose_to_sim_pose(start_pose), 0, speed=self.speed, check_collision=True)
                    if success:
                        success = self.prepare_for_handoff(self.speed, check_collision=True)
                    # time.sleep(1)

                    if success:
                        with self.state_lock:
                            self.set_center_control(True)
                            print(Fore.GREEN + f"{self.arm_idx} success lock - pass")
                            other_collides = self.env.is_down_move_and_detect_obstacle(1 - self.arm_idx)
                            print(f"other {other_collides}")
                            if other_collides:
                                success = False
                            if self.logging:
                                self.log(success)
                            if not success:
                                self.env.arms[self.arm_idx].control_state = "restart"
                                state_id = self.state[0]
                                self.check_and_restore(state_id)
                                self.env.hold_joints([self.arm_idx, 1 - self.arm_idx])
                                if self.logging:
                                    self.log(success)
                                success = self.env.go_home(
                                    1 - self.arm_idx,
                                    speed=self.speed,
                                    check_collision=True,
                                    center_control_override=True,
                                )
                                if success:
                                    success = self.env.grasp(
                                        self.arm_idx,
                                        env_pose_to_sim_pose(start_pose),
                                        0,
                                        speed=self.speed,
                                        check_collision=True,
                                        center_control_override=True,
                                    )
                                if success:
                                    success = self.prepare_for_handoff(
                                        self.speed, check_collision=True, center_control_override=True
                                    )
                                if self.logging:
                                    self.log(success)
                                if not success:
                                    print(Fore.RED + "??? pass something to the other side has deadlock")
                                    input("wait")
                                self.collision_count[str(self.count_plan)] = 1
                            if success:
                                self.paused = True
                                self.ready = True
                                self.state[0] = self.env._pb.saveState()
                                self.record_obj_state()
                            print(Fore.GREEN + f"{self.arm_idx} success unlock - pass")
                            self.set_center_control(False)
                    else:
                        if self.arm_idx == self.tie_arm_idx[0]:
                            with self.state_lock:
                                self.set_center_control(True)
                                self.env.arms[self.arm_idx].control_state = "restart"
                                print(Fore.GREEN + f"{self.arm_idx} no success lock - pass")
                                state_id = self.state[0]
                                self.check_and_restore(state_id)
                                self.env.hold_joints([self.arm_idx, 1 - self.arm_idx])
                                if self.logging:
                                    self.log(success)
                                success = self.env.go_home(
                                    1 - self.arm_idx,
                                    speed=self.speed,
                                    check_collision=True,
                                    center_control_override=True,
                                )
                                if success:
                                    success = self.env.grasp(
                                        self.arm_idx,
                                        env_pose_to_sim_pose(start_pose),
                                        0,
                                        speed=self.speed,
                                        check_collision=True,
                                        center_control_override=True,
                                    )
                                if success:
                                    success = self.prepare_for_handoff(
                                        self.speed, check_collision=True, center_control_override=True
                                    )
                                if self.logging:
                                    self.log(success)
                                if success:
                                    self.paused = True
                                    self.ready = True
                                    self.state[0] = self.env._pb.saveState()
                                    self.record_obj_state()
                                else:
                                    self.check_and_restore(state_id)
                                    self.env.hold_joints([self.arm_idx, 1 - self.arm_idx])
                                    if self.logging:
                                        self.log(success)
                                    go_home_success = self.env.go_home(
                                        self.arm_idx,
                                        speed=self.speed,
                                        check_collision=True,
                                        center_control_override=True,
                                    )
                                    if self.logging:
                                        self.log(go_home_success)
                                    if go_home_success:
                                        self.state[0] = self.env._pb.saveState()
                                        self.record_obj_state()
                                    else:
                                        print(Fore.RED + f"Error: pass something to the other side has deadlock...")
                                        input("press to continue")
                                self.tie_arm_idx[0] = 1 - self.tie_arm_idx[0]
                                self.collision_count[str(self.count_plan)] = 1
                                self.set_center_control(False)
                        else:
                            print(Fore.GREEN + f"{self.arm_idx} wait for tie")
                            while self.arm_idx != self.tie_arm_idx[0]:
                                time.sleep(0.1)
                elif start_pose == "G":
                    # when the robot receive objs from handoffs, it will add this action to the plan,
                    # asking the robot to place object to the goal
                    self.place_pose = env_pose_to_sim_pose(goal_pose)
                    success = self.env.place(self.arm_idx, self.place_pose, 0, speed=self.speed, check_collision=True)
                    # time.sleep(1)

                    if success:
                        with self.state_lock:
                            self.set_center_control(True)
                            print(Fore.GREEN + f"{self.arm_idx} success lock - place")
                            other_collides = self.env.is_down_move_and_detect_obstacle(1 - self.arm_idx)
                            print(f"other {other_collides}")
                            if other_collides:
                                success = False
                            if self.logging:
                                self.log(success)
                            if not success:
                                self.env.arms[self.arm_idx].control_state = "restart"
                                state_id = self.state[0]
                                self.check_and_restore(state_id)
                                self.env.hold_joints([self.arm_idx, 1 - self.arm_idx])
                                if self.logging:
                                    self.log(success)
                                success = self.env.go_home(
                                    1 - self.arm_idx,
                                    speed=self.speed,
                                    check_collision=True,
                                    center_control_override=True,
                                )
                                if success:
                                    success = self.env.place(
                                        self.arm_idx,
                                        self.place_pose,
                                        0,
                                        speed=self.speed,
                                        check_collision=True,
                                        center_control_override=True,
                                    )
                                if self.logging:
                                    self.log(success)
                                if not success:
                                    print(Fore.RED + "??? place has deadlock")
                                    input("wait")
                                self.collision_count[str(self.count_plan)] = 1
                            if success:
                                self.state[0] = self.env._pb.saveState()
                                self.record_obj_state()
                            self.set_center_control(False)
                    else:
                        if self.arm_idx == self.tie_arm_idx[0]:
                            with self.state_lock:
                                self.env.arms[self.arm_idx].control_state = "restart"
                                self.set_center_control(True)
                                print(Fore.GREEN + f"{self.arm_idx} no success lock - place")
                                state_id = self.state[0]
                                self.check_and_restore(state_id)
                                self.env.hold_joints([self.arm_idx, 1 - self.arm_idx])
                                if self.logging:
                                    self.log(success)
                                if self.env.is_down_move_and_detect_obstacle(self.arm_idx):
                                    success = self.env.go_home(
                                        self.arm_idx,
                                        speed=self.speed,
                                        check_collision=True,
                                        center_control_override=True,
                                    )
                                    extra_wait = True
                                else:
                                    success = self.env.go_home(
                                        1 - self.arm_idx,
                                        speed=self.speed,
                                        check_collision=True,
                                        center_control_override=True,
                                    )
                                    if success:
                                        success = self.env.place(
                                            self.arm_idx,
                                            self.place_pose,
                                            0,
                                            speed=self.speed,
                                            check_collision=True,
                                            center_control_override=True,
                                        )
                                if self.logging:
                                    self.log(success)
                                if success:
                                    self.state[0] = self.env._pb.saveState()
                                    self.record_obj_state()
                                else:
                                    self.check_and_restore(state_id)
                                    self.env.hold_joints([self.arm_idx, 1 - self.arm_idx])
                                    if self.logging:
                                        self.log(success)
                                    go_home_success = self.env.go_home(
                                        self.arm_idx,
                                        speed=self.speed,
                                        check_collision=True,
                                        center_control_override=True,
                                    )
                                    if self.logging:
                                        self.log(go_home_success)
                                    if go_home_success:
                                        self.state[0] = self.env._pb.saveState()
                                        self.record_obj_state()
                                    else:
                                        print(Fore.RED + f"Error: place could have deadlock...")
                                        input("press to continue")
                                self.tie_arm_idx[0] = 1 - self.tie_arm_idx[0]
                                self.collision_count[str(self.count_plan)] = 1
                                self.set_center_control(False)
                        else:
                            print(Fore.GREEN + f"{self.arm_idx} wait for tie")
                            while self.arm_idx != self.tie_arm_idx[0]:
                                time.sleep(0.1)
                else:  # pick-n-place
                    self.grasp_pose = env_pose_to_sim_pose(start_pose)
                    self.place_pose = env_pose_to_sim_pose(goal_pose)
                    success = self.pick_n_place(
                        self.grasp_pose, 0, self.place_pose, 0, speed=self.speed, check_collision=True
                    )
                    # TODO: it could be a case where one robot success, and checked the other is collision free. But since the other is still running, it then could collide.
                    # it can be solved by adding more protection, like once we enter state_lock, the other thread should stop.
                    # time.sleep(1)

                    if success:
                        with self.state_lock:
                            self.set_center_control(True)
                            print(Fore.GREEN + f"{self.arm_idx} success lock - pick and place")
                            other_collides = self.env.is_down_move_and_detect_obstacle(1 - self.arm_idx)
                            print(f"other {other_collides}")
                            if other_collides:
                                success = False
                            if self.logging:
                                self.log(success)
                            if not success:
                                self.env.arms[self.arm_idx].control_state = "restart"
                                state_id = self.state[0]
                                self.check_and_restore(state_id)
                                self.env.hold_joints([self.arm_idx, 1 - self.arm_idx])
                                if self.logging:
                                    self.log(success)
                                success = self.env.go_home(
                                    1 - self.arm_idx,
                                    speed=self.speed,
                                    check_collision=True,
                                    center_control_override=True,
                                )
                                if success:
                                    success = self.env.pick_and_place(
                                        self.arm_idx,
                                        self.grasp_pose,
                                        0,
                                        self.place_pose,
                                        0,
                                        speed=self.speed,
                                        check_collision=True,
                                        center_control_override=True,
                                    )
                                if self.logging:
                                    self.log(success)
                                if not success:
                                    print(Fore.RED + "??? pick-n-place has deadlock")
                                    input("wait")
                                self.collision_count[str(self.count_plan)] = 1
                            if success:
                                print(Fore.GREEN + f"{self.arm_idx} success record")
                                self.state[0] = self.env._pb.saveState()
                                self.record_obj_state()
                            self.set_center_control(False)
                    else:
                        if self.arm_idx == self.tie_arm_idx[0]:
                            with self.state_lock:
                                self.set_center_control(True)
                                self.env.arms[self.arm_idx].control_state = "restart"
                                print(Fore.GREEN + f"{self.arm_idx} no success lock - pick and place")
                                state_id = self.state[0]
                                self.check_and_restore(state_id)
                                self.env.hold_joints([self.arm_idx, 1 - self.arm_idx])
                                if self.logging:
                                    self.log(success)
                                if self.env.is_down_move_and_detect_obstacle(self.arm_idx):
                                    success = self.env.go_home(
                                        self.arm_idx,
                                        speed=self.speed,
                                        check_collision=True,
                                        center_control_override=True,
                                    )
                                    extra_wait = True
                                else:
                                    success = self.env.go_home(
                                        1 - self.arm_idx,
                                        speed=self.speed,
                                        check_collision=True,
                                        center_control_override=True,
                                    )
                                    if success:
                                        success = self.env.pick_and_place(
                                            self.arm_idx,
                                            self.grasp_pose,
                                            0,
                                            self.place_pose,
                                            0,
                                            speed=self.speed,
                                            check_collision=True,
                                            center_control_override=True,
                                        )
                                if self.logging:
                                    self.log(success)
                                if success:
                                    self.state[0] = self.env._pb.saveState()
                                    self.record_obj_state()
                                else:  # target is occupied
                                    self.check_and_restore(state_id)
                                    self.env.hold_joints([self.arm_idx, 1 - self.arm_idx])
                                    if self.logging:
                                        self.log(success)
                                    go_home_success = self.env.go_home(
                                        self.arm_idx,
                                        speed=self.speed,
                                        check_collision=True,
                                        center_control_override=True,
                                    )
                                    if self.logging:
                                        self.log(go_home_success)
                                    if go_home_success:
                                        self.state[0] = self.env._pb.saveState()
                                        self.record_obj_state()
                                    else:
                                        print(Fore.RED + f"Error: pick_n_place could have deadlock...")
                                        input("press to continue")
                                self.tie_arm_idx[0] = 1 - self.tie_arm_idx[0]
                                self.collision_count[str(self.count_plan)] = 1
                                self.set_center_control(False)
                                print(Fore.GREEN + f"{self.arm_idx} no success unlock - pick and place")
                        else:
                            print(Fore.GREEN + f"{self.arm_idx} wait for tie")
                            while self.arm_idx != self.tie_arm_idx[0]:
                                time.sleep(0.1)
                if not success:
                    self.plan.insert(0, next_action)
                if extra_wait:
                    print(Fore.GREEN + f"extra wait {self.arm_idx}")
                    time.sleep(5)
            else:
                time.sleep(1.0)

    def log(self, success):
        # self.env._pb.stepSimulation()
        self.env._pb.stopStateLogging(self.log_id[0])
        print(f"stop Log-{self.log_count[0]:02}.mp4")
        if success:
            self.log_count[0] += 1
        else:
            os.remove(f"videos/Log-{self.log_count[0]:02}.mp4")
        log_id = self.env._pb.startStateLogging(
            self.env._pb.STATE_LOGGING_VIDEO_MP4, f"videos/Log-{self.log_count[0]:02}.mp4"
        )
        print(f"start Log-{self.log_count[0]:02}.mp4")
        self.log_id[0] = log_id

    def set_center_control(self, controlling):
        self.env.center_control = controlling
        time.sleep(1)

    def record_obj_state(self):
        obj_info = None
        # if self.env.arms[self.arm_idx].activated:
        #     obj_info = self.env.arms[self.arm_idx].grasped_obj_id, None
        # else:
        #     obj_info = self.env.save_object(env_pose_to_sim_pose(pose))
        self.obj_state[self.arm_idx] = [obj_info, self.env.arms[self.arm_idx].activated]
        self.obj_state[1 - self.arm_idx][1] = self.env.arms[1 - self.arm_idx].activated

    def check_and_restore(self, state_id):
        print("check and restore")

        if not self.obj_state[1 - self.arm_idx][1]:
            self.env.release(1 - self.arm_idx)
        if not self.obj_state[self.arm_idx][1]:
            self.env.release(self.arm_idx)

        self.env._pb.restoreState(state_id)

        # if not self.obj_state[1 - self.arm_idx][1]:
        #     self.env.restore_object(self.obj_state[1 - self.arm_idx][0])
        # if not self.obj_state[self.arm_idx][1]:
        #     self.env.restore_object(self.obj_state[self.arm_idx][0])

    def pick_n_place(
        self, pose, angle, place_pose, place_angle, speed=0.005, check_collision=True, center_control_override=False
    ) -> bool:
        success = self.env.pick_and_place(
            self.arm_idx,
            pose,
            angle,
            place_pose,
            place_angle,
            speed=speed,
            check_collision=check_collision,
            center_control_override=center_control_override,
        )
        return success

    def grasp(self, pose, angle, speed=0.005, check_collision=False, center_control_override=False) -> bool:
        success = self.env.grasp(
            self.arm_idx,
            pose,
            angle,
            speed=speed,
            check_collision=check_collision,
            center_control_override=center_control_override,
        )
        return success

    def place(self, pose, angle, speed=0.005, check_collision=False, center_control_override=False) -> bool:
        success = self.env.place(
            self.arm_idx,
            pose,
            angle,
            speed=speed,
            check_collision=check_collision,
            center_control_override=center_control_override,
        )
        return success

    def prepare_for_handoff(self, speed=0.005, check_collision=False, center_control_override=False) -> bool:

        # compute handoff poses
        arm1_pos = np.array(handoff_position, dtype=np.float32)
        if self.arm_idx == 0:
            arm1_pos[0] += 0.02
            arm1_pos[0] -= self.env.ee_tip_offset + self.env.buffer_height
            arm1_preparation_pos = arm1_pos.copy()
            arm1_preparation_pos[0] -= 0.1
        else:
            arm1_pos[0] -= 0.02
            arm1_pos[0] += self.env.ee_tip_offset + self.env.buffer_height
            arm1_preparation_pos = arm1_pos.copy()
            arm1_preparation_pos[0] += 0.1

        # success = self.env.move_joints(
        #     self.arm_idx, self.env.arms_ik_rest_joints[self.arm_idx], speed=speed, check_collision=check_collision
        # )
        success = self.env.move_ee_pose(
            self.arm_idx,
            (arm1_preparation_pos, pb.getQuaternionFromEuler(handoff_euler[self.arm_idx])),
            speed=speed,
            check_collision=check_collision,
            center_control_override=center_control_override,
            is_handoff=True,
        )
        print("arm " + str(self.arm_idx) + " preparation " + str(success))
        return success

