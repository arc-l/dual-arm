import os
import time
import numpy as np
import pybullet
import pybullet_data
from pybullet_utils import bullet_client
import matplotlib.pyplot as plt

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Cuboid, Material
from curobo.rollout.rollout_base import Goal
from curobo.geom.sphere_fit import SphereFitType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, MotionGenResult, PoseCostMetric
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

# @markdown Global constants: pick and place objects, colors, workspace bounds

class Robotiq2F85:
    """Gripper handling for Robotiq 2F85."""

    def __init__(self, bulletClient, robot, tool):
        self._pb = bulletClient
        self.robot = robot
        self.tool = tool
        pos = [0.1339999999999999, -0.49199999999872496, 0.5]
        rot = self._pb.getQuaternionFromEuler([np.pi, 0, np.pi])
        urdf = "assets/robotiq_2f_85/robotiq_2f_85.urdf"
        self.body = self._pb.loadURDF(urdf, pos, rot)
        self.n_joints = self._pb.getNumJoints(self.body)
        self.activated = False

        # Connect gripper base to robot tool.
        self._pb.createConstraint(self.robot, tool, self.body, 0, jointType=self._pb.JOINT_FIXED, jointAxis=[0, 0, 0], parentFramePosition=[
                                  0, 0, 0], childFramePosition=[0, 0, -0.07], childFrameOrientation=pybullet.getQuaternionFromEuler([0, 0, np.pi / 2]))

        # Set friction coefficients for gripper fingers.
        for i in range(self._pb.getNumJoints(self.body)):
            self._pb.changeDynamics(self.body, i, lateralFriction=10.0,
                                    spinningFriction=1.0, rollingFriction=1.0, frictionAnchor=True)

        self.motor_joint = 1

    # Control joint positions by enforcing hard contraints on gripper behavior.
    # Set one joint as the open/close motor joint (other joints should mimic).
    def update(self):
        currj = [self._pb.getJointState(self.body, i)[0]
                 for i in range(self.n_joints)]
        indj = [6, 3, 8, 5, 10]
        targj = [currj[1], -currj[1], -currj[1], currj[1], currj[1]]
        self._pb.setJointMotorControlArray(
            self.body, indj, pybullet.POSITION_CONTROL, targj, positionGains=np.ones(5))

    # Close gripper fingers.
    def activate(self):
        self._pb.setJointMotorControl2(
            self.body, self.motor_joint, pybullet.VELOCITY_CONTROL, targetVelocity=1, force=10)
        self.activated = True

    # Open gripper fingers.
    def release(self):
        self._pb.setJointMotorControl2(
            self.body, self.motor_joint, pybullet.VELOCITY_CONTROL, targetVelocity=-1, force=10)
        self.activated = False

    # If activated and object in gripper: check object contact.
    # If activated and nothing in gripper: check gripper contact.
    # If released: check proximity to surface (disabled).
    def detect_contact(self):
        obj, _, ray_frac = self.check_proximity()
        if self.activated:
            empty = self.grasp_width() < 0.01
            cbody = self.body if empty else obj
            if obj == self.body or obj == 0:
                return False
            return self.external_contact(cbody)
    #   else:
    #     return ray_frac < 0.14 or self.external_contact()

    # Return if body is in contact with something other than gripper
    def external_contact(self, body=None):
        if body is None:
            body = self.body
        pts = self._pb.getContactPoints(bodyA=body)
        pts = [pt for pt in pts if pt[2] != self.body]
        return len(pts) > 0  # pylint: disable=g-explicit-length-test

    def check_grasp(self):
        while self.moving():
            time.sleep(0.001)
        success = self.grasp_width() > 0.01
        return success

    def grasp_width(self):
        lpad = np.array(self._pb.getLinkState(self.body, 4)[0])
        rpad = np.array(self._pb.getLinkState(self.body, 9)[0])
        dist = np.linalg.norm(lpad - rpad) - 0.047813
        return dist

    def check_proximity(self):
        ee_pos = np.array(self._pb.getLinkState(self.robot, self.tool)[0])
        tool_pos = np.array(self._pb.getLinkState(self.body, 0)[0])
        vec = (tool_pos - ee_pos) / np.linalg.norm((tool_pos - ee_pos))
        ee_targ = ee_pos + vec
        ray_data = self._pb.rayTest(ee_pos, ee_targ)[0]
        obj, link, ray_frac = ray_data[0], ray_data[1], ray_data[2]
        return obj, link, ray_frac


class DualArmEnv:
    def __init__(self, dt=1/240, home_joints=None):
        self.dt = dt
        self.sim_step = 0
        self._pb = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        self._client_id = self._pb._client
        self._pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        assets_path = os.path.dirname(os.path.abspath(""))
        self._pb.setAdditionalSearchPath(assets_path)
        self._pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._pb.setTimeStep(self.dt)

        # target = self._pb.getDebugVisualizerCamera()
        self._pb.resetDebugVisualizerCamera(
            cameraDistance=1,
            cameraYaw=45,
            cameraPitch=-25,
            cameraTargetPosition=[0, -0.5, 0],
        )

        # Joint angles: (J0, J1, J2, J3, J4, J5).
        # self.home_joints = (np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, 3 * np.pi / 2, 0)
        self.home_joints = home_joints
        # (RX, RY, RZ) rotation in Euler angles.
        self.home_ee_euler = (np.pi, 0, np.pi)
        self.ee_link_id = 8  # Link ID of UR5 end effector.
        self.gripper = None

    def reset(self):
        self._pb.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
        self._pb.setGravity(0, 0, -9.8)

        # Temporarily disable rendering to load URDFs faster.
        # self._pb.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        # Add robot.
        self._pb.loadURDF("plane.urdf", [0, 0, -0.0005])
        self.robot_id = self._pb.loadURDF("assets/ur5e/dual_ur5e_gripper.urdf", [0, 0, 0], flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        # For forward kinematics.
        self.ghost_id = self._pb.loadURDF("assets/ur5e/dual_ur5e_gripper.urdf", [0, 0, -10])
        self.joint_ids = [self._pb.getJointInfo(self.robot_id, i) for i in range(self._pb.getNumJoints(self.robot_id))]
        self.joint_ids = [j[0] for j in self.joint_ids if j[2] == pybullet.JOINT_REVOLUTE]

        # Move robot to home configuration.
        for i in range(len(self.joint_ids)):
            # self._pb.resetJointState(self.robot_id, self.joint_ids[i], self.home_joints[i])
            if self.home_joints is None:
                self._pb.resetJointState(self.robot_id, self.joint_ids[i], 0)
            else:
                self._pb.resetJointState(self.robot_id, self.joint_ids[i], self.home_joints[i])


        # Add gripper.
        self.gripper = Robotiq2F85(self._pb, self.robot_id, self.ee_link_id)
        self.gripper.release()

        # Add workspace.
        plane_shape = self._pb.createCollisionShape(
            pybullet.GEOM_BOX, halfExtents=[0.3, 0.3, 0.001])
        plane_visual = self._pb.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=[0.3, 0.3, 0.001])
        plane_id = self._pb.createMultiBody(
            0, plane_shape, plane_visual, basePosition=[0, 0, 0])
        self._pb.changeVisualShape(
            plane_id, -1, rgbaColor=[0.2, 0.2, 0.2, 1.0])

        for _ in range(10000):
            if self.gripper is not None:
                self.gripper.update()
            self._pb.stepSimulation()

    def servoj(self, joints):
        """Move to target joint positions with position control."""
        self._pb.setJointMotorControlArray(
            bodyIndex=self.robot_id,
            jointIndices=self.joint_ids,
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=joints,
            positionGains=[0.01]*6)

    def movep(self, position):
		#TODO: Make this handle multiple end effectors: calculateInverseKinematics2
        """Move to target end effector position."""
        joints = self._pb.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.tip_link_id,
            targetPosition=position,
            targetOrientation=self._pb.getQuaternionFromEuler(
                self.home_ee_euler),
            maxNumIterations=100)
        self.servoj(joints)

if __name__ == "__main__":
    home_joints = np.array([
        0, -2.2, 1.9, -1.383, -1.57, 0.00, # first arm
        0, -2.2, 1.9, -1.383, -1.57, 0.00, # second arm
    ])
    env = DualArmEnv(1/240, home_joints=home_joints)
    time.sleep(1)

    np.random.seed(42)
    env.reset()

    time.sleep(100)
