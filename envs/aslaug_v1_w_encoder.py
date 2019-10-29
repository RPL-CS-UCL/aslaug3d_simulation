from gym import spaces
import numpy as np
import pybullet as pb
import pybullet_data
import os
import random
from . import aslaug_base

from sklearn.externals import joblib
from keras.models import load_model

class AslaugEnv(aslaug_base.AslaugBaseEnv):

    # Overvation: [sp_pos_ee[3], mb_vel_r[3], link_pos[3*n_joints],
    #              joint_pos[3], joint_vel[3]]
    # Action: [mb_d_vel_r[3], joint_d_vel[3]]
    # State: [mb_pos_w[3], mb_vel_w[3], joint_pos[3], joint_vel[3]]
    def __init__(self, folder_name="", gui=False):
        # Common params
        version = "v1"
        self.folder_name = folder_name

        self.scaler = joblib.load("test/autoencoder/scaler.bin")
        self.encoder = load_model("test/autoencoder/encoder.h5")

        params = {
            "joints": {
                "joint_names": ['panda_joint{}'.format(i+1) for i in [0, 3]],
                "init_states": [-np.pi/2, np.pi/2, np.pi/2, -1.75, -np.pi/2,
                                np.pi, np.pi/4],
                "base_link_name": "panda_link0",
                "ee_link_name": "grasp_loc",
                "link_names": (['panda_link{}'.format(i)
                                for i in [1, 3, 4, 5, 7]]
                               + ['grasp_loc']),
                "link_mag": [0.33, 0.67, 0.75, 1.16, 1.28, 1.56],
                "vel_mag": 1.0,
                "acc_mag": 0.75
            },
            "base": {
                "vel_mag": np.array([0.4, 0.4, 0.75]),
                "acc_mag": np.array([1.5, 1.5, 0.5])
            },
            "setpoint": {
                "hold_time": 2.0,
                "tol_lin_mag": 0.2,
                "tol_ang_mag": np.pi,
                "continious_mode": True
            },
            "reward": {
                "fac_goal_dis_lin": 10.0,
                "fac_goal_dis_ang": 0.0,
                "fac_ang_vel": -2.0,
                "fac_sp_hold": 20.0,
                "rew_timeout": -5.0,
                "rew_joint_limits": -20.0,
                "rew_collision": -25.0,
                "rew_goal_reached": 35.0
            },
            "world": {
                "tau": 0.02,
                "timeout": 30.0,
                "size": 20.0,
                "action_discretization": 7,
                "n_bookcases": 12,
                "corridor_width": 3.0,
            },
            "sensors": {
                "lidar": {
                    "n_scans": 41,
                    "ang_mag": np.pi/2,
                    "range": 5.0,
                    "link_id1": "front_laser",
                    "link_id2": "rear_laser"
                }
            }
        }

        # Initialize super class
        super().__init__(version, params, gui=gui, init_seed=None)

    def setup_action_observation_spaces(self):
        # Define action space
        # [delta_mb_vel_r[3], delta_joint_vel[n_joints]]
        accel_lims_mb = self.p["base"]["acc_mag"]
        acc_lim_joints = (self.n_joints*[self.p["joints"]["acc_mag"]])
        highs_a = (self.p["world"]["tau"]
                   * np.concatenate((accel_lims_mb, acc_lim_joints)))
        lows_a = -highs_a
        n_d = self.p["world"]["action_discretization"]
        self.action_space = spaces.MultiDiscrete(lows_a.shape[0]*[n_d])
        self.actions = np.linspace(lows_a, highs_a, n_d)
        # Define observation space
        # Overvation: [sp_pos_ee[6], mb_vel_r[3], link_pos[6*n_links+1],
        #              joint_pos[n_joints], joint_vel[n_joints],
        #              scan[n_scans]]
        high_sp = np.array([self.p["world"]["size"]]*2 + [1.5] + 3*[np.pi])
        low_sp = -high_sp
        high_mb = np.array(self.p["base"]["vel_mag"])
        low_mb = -high_mb

        high_lp = []
        low_lp = []
        for v in self.p["joints"]["link_mag"]:
            high_lp += [v, v, v, np.pi, np.pi, np.pi]
            low_lp += [-v, -v, -0.3, -np.pi, -np.pi, -np.pi]
        high_lp = np.array(high_lp)
        low_lp = np.array(low_lp)
        high_j_p = self.joint_limits[self.actuator_selection, 1]
        low_j_p = self.joint_limits[self.actuator_selection, 0]
        high_j_v = np.array([self.p["joints"]["vel_mag"]]*self.n_joints)
        low_j_v = -high_j_v
        rng = self.p["sensors"]["lidar"]["range"]
        high_scan = rng * np.ones(self.p["sensors"]["lidar"]["n_scans"])
        low_scan = 0.1*high_scan
        high_o = np.concatenate((high_sp, high_mb, high_lp, high_j_p,
                                 high_j_v, high_scan))
        low_o = np.concatenate((low_sp, low_mb, low_lp, low_j_p,
                                low_j_v, low_scan))

        self.obs_slicing = None
        # for e in (high_sp, high_mb, high_lp, high_j_p, high_j_v, high_scan):
        #     self.obs_slicing.append(self.obs_slicing[-1] + e.shape[0])

        low_o = np.zeros(16)
        high_o = np.array([3.6635242, 5.5809293, 5.4702263, 8.838332,  4.162554,  0.5,
                           5.030792,  4.180762,  5.237934,  0.11989237, 4.3316593, 6.1497827,
                           5.670298,  7.7763414, 8.392844,  6.320906])
        self.observation_space = spaces.Box(low_o, high_o)

    def calculate_reward(self):
        # Introducte reward variable
        reward = 0.0
        done = False
        info = {}

        # Penalize if joint limit reached and end episode
        if self.check_joint_limits_reached():
            reward += self.p["reward"]["rew_joint_limits"]
            info["done_reason"] = "joint_limits_reached"
            done = True

        # Penalize collisions and end episode
        if self.check_collision():
            reward += self.p["reward"]["rew_collision"]
            info["done_reason"] = "collision"
            done = True

        # Penalize velocity in move base rotation
        mb_ang_vel = self.get_base_vels()[2]
        reward += np.abs(mb_ang_vel)*self.tau*self.p["reward"]["fac_ang_vel"]

        # # Penalize velocity of joints
        # j_vel_mag = np.array([self.p["joints"]["vel_mag"]]*self.n_joints)
        # j_pos, j_vel = self.get_joint_states()
        # reward -= 1.1*self.tau*(np.abs(j_vel)/j_vel_mag).sum()
        # Calculate goal distance
        eucl_dis, eucl_ang = self.calculate_goal_distance()

        # Calculate intermediate reward
        delta_eucl_dis = self.last_eucl_dis - eucl_dis
        delta_eucl_ang = self.last_eucl_ang - eucl_ang
        reward += (self.scl_eucl_dis
                   * self.p["reward"]["fac_goal_dis_lin"] * delta_eucl_dis)
        reward += (self.scl_eucl_ang
                   * self.p["reward"]["fac_goal_dis_ang"] * delta_eucl_ang)
        self.last_eucl_dis, self.last_eucl_ang = eucl_dis, eucl_ang

        # Check if goal reached
        if eucl_dis <= self.p["setpoint"]["tol_lin_mag"] and \
                eucl_ang <= self.p["setpoint"]["tol_ang_mag"]:

            if self.sp_hold_time >= self.p["setpoint"]["hold_time"]:
                if self.p["setpoint"]["continious_mode"]:
                    # Spawn new random setpoint
                    sp_pos = random.sample(self.possible_sp_pos, 1)[0]
                    self.move_sp(sp_pos)
                    self.sp_hold_time = 0.0
                    self.step_no = 0
                else:
                    done = True
                    info["done_reason"] = "success"
                reward += self.p["reward"]["rew_goal_reached"]

            self.sp_hold_time += self.tau
            reward += self.tau*self.p["reward"]["fac_sp_hold"]
        else:
            reward -= self.sp_hold_time*self.p["reward"]["fac_sp_hold"]
            self.sp_hold_time = 0.0

        return reward, done, info

    def calculate_observation(self):
        sp_pose_ee = self.get_ee_sp_transform()
        link_pose_r = self.get_link_states(self.link_mapping)
        j_pos, j_vel = self.get_joint_states(self.actuator_selection)
        mb_vel_w = self.get_base_vels()
        scan = self.get_lidar_scan()
        obs = np.concatenate((sp_pose_ee, mb_vel_w, link_pose_r.flatten(),
                              j_pos, j_vel, scan))

        obs_scl = self.scaler.transform(np.array([obs]))
        enc = self.encoder.predict(obs_scl)
        return enc[0, :]

    def reset(self, init_state=None, init_setpoint_state=None,
              init_obstacle_grid=None, init_obstacle_locations=None):

        # Reset internal parameters
        self.step_no = 0
        self.state = {"base_vel": np.array([0.0, 0.0, 0.0]),
                      "joint_vel": np.array(7*[0.0])}

        # Reset environment
        for i in range(len(self.joint_mapping)):
            pb.resetJointState(self.robotId, self.joint_mapping[i],
                               self.p["joints"]["init_states"][i],
                               0.0, self.clientId)

        j0 = self.np_random.uniform(self.joint_limits[0, 0],
                                    self.joint_limits[0, 1])
        j3 = self.np_random.uniform(self.joint_limits[3, 0],
                                    self.joint_limits[3, 1])
        pb.resetJointState(self.robotId, self.joint_mapping[0],
                           j0,
                           0.0, self.clientId)
        pb.resetJointState(self.robotId, self.joint_mapping[3],
                           j3,
                           0.0, self.clientId)
        pb.resetBaseVelocity(self.robotId, [0, 0, 0], [0, 0, 0], self.clientId)

        robot_pos = (0, 0, 0.02)
        robot_init_yaw = np.pi/2 + np.random.uniform(-np.pi/4, np.pi/4)
        robot_ori = pb.getQuaternionFromEuler([np.pi/2, 0, robot_init_yaw])
        pb.resetBasePositionAndOrientation(self.robotId, robot_pos, robot_ori,
                                           self.clientId)

        # Reorder bookcases
        possible_sp_pos = []
        pos = np.array([1.0, -self.p["world"]["corridor_width"]/2, np.pi/2])
        for i in range(int(len(self.bookcaseIds)/2.0)):
            bookcaseId = self.bookcaseIds[i]
            possible_sp_pos += self.move_bookcase(bookcaseId, pos, sp_layers=[1])
            pos[0] += 1.1 + self.np_random.uniform(0, 0.2)

        pos = np.array([1.0, self.p["world"]["corridor_width"]/2, -np.pi/2])
        for i in range(int(len(self.bookcaseIds)/2.0), len(self.bookcaseIds)):
            bookcaseId = self.bookcaseIds[i]
            possible_sp_pos += self.move_bookcase(bookcaseId, pos, sp_layers=[1])
            pos[0] += 1.2 + self.np_random.uniform(0, 0.2)
        self.possible_sp_pos = possible_sp_pos

        # Spawn random setpoint
        sp_pos = random.sample(self.possible_sp_pos, 1)[0]
        self.move_sp(sp_pos)

        # Initialize reward state variables
        self.last_eucl_dis, self.last_eucl_ang = self.calculate_goal_distance()
        self.scl_eucl_dis = 1/self.last_eucl_dis
        self.scl_eucl_ang = 1/self.last_eucl_ang
        self.sp_hold_time = 0.0

        # Calculate observation and return
        obs = self.calculate_observation()
        return obs

    def spawn_robot(self):
        # Spawn robot
        robot_pos = [0, 0, 0.02]
        robot_ori = pb.getQuaternionFromEuler([0, 0, 0])
        dirname = os.path.dirname(__file__)
        model_path = os.path.join(dirname, '../urdf/robot/aslaug.urdf')
        return pb.loadURDF(model_path, robot_pos, robot_ori,
                           useFixedBase=True,
                           physicsClientId=self.clientId)

    def spawn_setpoint(self):
        # Spawn setpoint
        mug_pos = [5, 2, 0.0]
        mug_ori = pb.getQuaternionFromEuler([0, 0, 0])
        dirname = os.path.dirname(__file__)
        model_path = os.path.join(dirname,
                                  '../urdf/beer_rothaus/beer_rothaus.urdf')
        spId = pb.loadURDF(model_path, mug_pos, mug_ori,
                           useFixedBase=True,
                           physicsClientId=self.clientId)

        # Spawn setpoint marker
        mug_pos = [5, 3, 0.0]
        mug_ori = pb.getQuaternionFromEuler([0, 0, 0])
        self.markerId = pb.loadURDF("sphere2red.urdf", mug_pos, mug_ori,
                                    globalScaling=0.2, useFixedBase=True,
                                    physicsClientId=self.clientId)
        return spId

    def spawn_additional_objects(self):
        # Spawn bounding box
        mug_pos = [0, 0, 0.0]
        mug_ori = pb.getQuaternionFromEuler([0, 0, 0])
        dirname = os.path.dirname(__file__)
        model_path = os.path.join(dirname,
                                  '../urdf/bounding_box/bounding_box.urdf')
        self.bbId = pb.loadURDF(model_path, mug_pos, mug_ori,
                                useFixedBase=True,
                                physicsClientId=self.clientId)

    def calculate_goal_distance(self):
        sp_pose_ee = self.get_ee_sp_transform()
        eucl_dis = np.linalg.norm(sp_pose_ee[1:3])  # Ignore x coord
        eucl_ang = np.linalg.norm(sp_pose_ee[3:6])
        return eucl_dis, eucl_ang