from threading import Thread, Lock
from queue import Queue
import time
import multiprocessing as mp
import sys
import glob
from tkinter.tix import Tree
import traceback
import timeout_decorator

from .robot_manipulation import robot_manipulation
from environment_sim import Environment
from merge_videos import merge_video

class central_control(object):
    """
    A central control system to ask robots move or stay and also coordinate robots to do handoffs
    """

    def __init__(self, env: Environment, action_list: dict, speed: float, start_arr: dict, goal_arr: dict, exp_text=''):
        # logs = glob.glob("*.txt")
        # logs = sorted(logs)[:]
        # for log in logs:
        #     logId = env._pb.startStateLogging(env._pb.STATE_REPLAY_ALL_COMMANDS, log)
        #     print(logId)
        #     while env._pb.isConnected() and not env.is_env_static:
        #         time.sleep(1.0 / 240.0)
        #         # env._pb.stepSimulation()
        #     env._pb.stopStateLogging(logId)
        #     input("next")
        # exit()

        # action_list = {0: [], 1: [(3, 'S', 'G')]}
        mp.set_start_method("spawn")

        cleanup_prev_videos()

        self.env = env
        self.state = [self.env._pb.saveState()]
        self.obj_state = [[None, False], [None, False]]
        self.state_lock = Lock()
        self.tie_arm_idx = [0]
        self.log_count = [0]
        self.logging = True
        if self.logging:
            self.log_id = [
                self.env._pb.startStateLogging(
                    self.env._pb.STATE_LOGGING_VIDEO_MP4, f"videos/Log-{self.log_count[0]:02}.mp4"
                )
            ]
        else:
            self.log_id = []
        self.count_plan = action_list
        self.total_count = -1
        self.execution_time = -1
        # self.env.init_lock()

        self.robot1 = robot_manipulation(
            env,
            action_list[0],
            0,
            start_arr,
            goal_arr,
            self.state,
            self.obj_state,
            self.state_lock,
            self.tie_arm_idx,
            self.log_count,
            self.log_id,
            self.logging,
            self.count_plan,
        )
        self.robot2 = robot_manipulation(
            env,
            action_list[1],
            1,
            start_arr,
            goal_arr,
            self.state,
            self.obj_state,
            self.state_lock,
            self.tie_arm_idx,
            self.log_count,
            self.log_id,
            self.logging,
            self.count_plan,
        )
        self.speed = speed

        try:
            robot1_thread = Thread(target=self.robot1.execution, args=())
            robot1_thread.daemon = True
            robot1_thread.start()

            robot2_thread = Thread(target=self.robot2.execution, args=())
            robot2_thread.daemon = True
            robot2_thread.start()

            handoff_thread = Thread(target=self.execute_handoff, args=())
            handoff_thread.daemon = True
            handoff_thread.start()

            self.robot1.paused = False
            self.robot2.paused = False

            while robot1_thread.is_alive() or robot2_thread.is_alive() or handoff_thread.is_alive():
                robot1_thread.join(1)
                robot2_thread.join(1)
                handoff_thread.join(1)

            # success = self.robot2.pick_n_place(
            #     (0.5694027192608125, 0.03462598922540139, 0.023000000044703484),
            #     0,
            #     (0.3443546100813887, -0.09357594491594684, 0.023000000044703484),
            #     0,
            #     speed=self.speed,
            #     check_collision=True,
            # )
            # self.robot2.log(success)
            # self.robot2.go_home()
            # self.robot2.log(success)
            # state_id = self.env._pb.saveState()
            # contact_constraint = self.env._pb.createConstraint(
            #     parentBodyUniqueId=self.env.arms[0].ee,
            #     parentLinkIndex=0,
            #     childBodyUniqueId=9,
            #     childLinkIndex=-1,
            #     jointType=self.env._pb.JOINT_FIXED,
            #     jointAxis=(0, 0, 0),
            #     parentFramePosition=(0, 0, 0),
            #     parentFrameOrientation=(0, 0, 0, 1),
            #     childFramePosition=(0, 0, 0),
            #     childFrameOrientation=(0, 0, 0, 1),
            # )
            # self.env._pb.removeConstraint(contact_constraint)
            # self.env._pb.restoreState(state_id)
            # self.robot1.log(True)
            # success = self.robot1.pick_n_place(
            #     (0.7926140039735514, -0.23198765584064623, 0.023000000044703484),
            #     0,
            #     (0.6067761784253975, 0.0226640421866503, 0.023000000044703484),
            #     0,
            #     speed=self.speed,
            #     check_collision=True,
            # )
            # self.robot1.go_home()
            # self.robot1.log(success)

            if self.logging:
                self.env._pb.stopStateLogging(self.log_id[0])

            r1_count = list(self.robot1.collision_count.keys())
            r2_count = list(self.robot2.collision_count.keys())
            total_count = set(r1_count + r2_count)
            total_count = len(total_count)
            print(f"total collision is {total_count}")
            self.total_count = total_count
            self.execution_time = merge_video(exp_text)

        except KeyboardInterrupt:
            print("Ctrl+C pressed...")
            sys.exit(1)

        except timeout_decorator.timeout_decorator.TimeoutError:
            raise timeout_decorator.timeout_decorator.TimeoutError

        except Exception as e:
            print(e)
            traceback.print_exc()

    def execute_handoff(self):
        while True:
            if self.robot1.ready and self.robot2.ready:
                if self.robot1.pass_obj:
                    passing_robot = self.robot1
                    receiving_robot = self.robot2
                else:
                    passing_robot = self.robot2
                    receiving_robot = self.robot1
                self.env.handoff(passing_robot.arm_idx, receiving_robot.arm_idx, speed=self.speed)
                self.robot1.paused = False
                self.robot2.paused = False
                self.robot1.ready = False
                self.robot2.ready = False
                time.sleep(0.1)
            elif self.robot1.finished and self.robot2.finished:
                print("shut down controller")
                return
            time.sleep(1)


def cleanup_prev_videos():
    import os
    dir_name = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'videos'
    )
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    files_in_directory = os.listdir(dir_name)
    filtered_files = [file for file in files_in_directory if file.endswith(".mp4")]
    print(filtered_files)
    for file in filtered_files:
        path_to_file = os.path.join(dir_name, file)
        os.remove(path_to_file)
