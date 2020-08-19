import tensorflow as tf
from stable_baselines.common.policies import ActorCriticPolicy
from tensorflow.keras.layers import Lambda
from gym import spaces
import torch
import numpy as np 
import sys, os 
sys.path.insert(0,"./lidar_autoencoder/")
from lidar_autoencoder import *

class AslaugPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, **kwargs):

        if "obs_slicing" in kwargs and kwargs["obs_slicing"] is not None:
            obs_slicing = kwargs["obs_slicing"]
        else:
            obs_slicing = [0, 6, 9, 57, 64, 71, 272, 473]
        self.obs_slicing = obs_slicing
        
        n_scans = 201
        self.n_scans=n_scans
        latent_dim = 128
        
        self.obs_slicing = [obs_slicing[i] for i in range(6)] + [obs_slicing[5]+latent_dim] 
        

        ob_space_low =  ob_space.low[:obs_slicing[5]].tolist()
        ob_space_low = np.array(ob_space_low + 128*[0.0])
        ob_space_high =  ob_space.high[:obs_slicing[5]].tolist()
        ob_space_high = np.array(ob_space_high + 128*[5.0])
        
        ob_space = spaces.Box(ob_space_low, ob_space_high)

        self.LAE = LidarAutoencoder(
            n_scans = n_scans*2,
            latent_dim = latent_dim,
            capacity=2)#.to("cpu")
        lae_path = "./lidar_autoencoder/cp_LAE_{:}_{:}.pth".format(n_scans,latent_dim)
        self.LAE.load_state_dict(torch.load(lae_path))

        self.LAE.eval()

        super(AslaugPolicy, self).__init__(sess, ob_space, ac_space,
                                           n_env, n_steps, n_batch,
                                           reuse=reuse, scale=False)

        # Scale observation
        with tf.variable_scope("observation_scaling", reuse=reuse):
            obs_avg = tf.constant((self.ob_space.high+self.ob_space.low)/2.0,
                                  name="obs_avg")
            obs_dif = tf.constant((self.ob_space.high - self.ob_space.low),
                                  name="obs_diff")
            shifted = tf.math.subtract(self.processed_obs, obs_avg)
            proc_obs = tf.math.divide(shifted, obs_dif)

        # Create network
        with tf.variable_scope("model/inputs", reuse=reuse):
            lrelu = tf.nn.leaky_relu
            o = obs_slicing
            in_sp = self.crop(1, o[0], o[1])(proc_obs)
            in_mb = self.crop(1, o[1], o[2])(proc_obs)
            in_lp = self.crop(1, o[2], o[3])(proc_obs)
            in_jp = self.crop(1, o[3], o[4])(proc_obs)
            in_jv = self.crop(1, o[4], o[5])(proc_obs)

            in_scans_latent = self.crop(1, o[5], o[5]+latent_dim)(proc_obs)
            #in_sc1 = self.crop(1, o[5], o[6])(proc_obs)
            #in_sc2 = self.crop(1, o[6], o[7])(proc_obs)
            #in_scans_latent = tf.keras.layers.Concatenate(name="in_scans_latent")([in_sc1, in_sc2])
            

        with tf.variable_scope("model/scan_block", reuse=reuse):
            sc_2 = tf.layers.Dense(64, activation=lrelu, name="sc_2")(in_scans_latent)
            sc_out = tf.layers.Dense(64, activation=lrelu, name="sc_out")(sc_2)

        with tf.variable_scope("model/combination_block", reuse=reuse):
            c_0 = tf.keras.layers.Concatenate(name="c_0")([sc_out, in_sp,
                                                           in_mb, in_lp, in_jp,
                                                           in_jv])
            c_1 = tf.layers.Dense(512, activation=lrelu, name="c_1")(c_0)
            c_2 = tf.layers.Dense(256, activation=lrelu, name="c_2")(c_1)
            c_3 = tf.layers.Dense(256, activation=lrelu, name="c_3")(c_2)
            c_4 = tf.layers.Dense(128, activation=lrelu, name="c_4")(c_3)
            c_5 = tf.layers.Dense(128, activation=lrelu, name="c_5")(c_4)
            c_6 = tf.layers.Dense(64, activation=lrelu, name="c_6")(c_5)
            c_7 = tf.layers.Dense(64, activation=lrelu, name="c_7")(c_6)
            c_out = tf.layers.Dense(32, activation=lrelu, name="c_out")(c_7)

        with tf.variable_scope("model/actor_critic_block", reuse=reuse):
            vf_0 = tf.layers.Dense(64, activation=lrelu,
                                   name="vf_0")(c_out)
            vf_1 = tf.layers.Dense(32, activation=lrelu,
                                   name="vf_1")(vf_0)
            vf_latent = tf.layers.Dense(16, activation=lrelu,
                                        name="vf_latent")(vf_1)
            pi_0 = tf.layers.Dense(64, activation=lrelu,
                                   name="vf_0")(c_out)
            pi_latent = tf.layers.Dense(32, activation=lrelu,
                                        name="vf_f_1")(pi_0)

            value_fn = tf.layers.Dense(1, name='vf')(vf_latent)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent,
                                                           vf_latent,
                                                           init_scale=1.0)


        self._value_fn = value_fn
        # self._initial_state = None
        self._setup_init()

    def encode_obs (self, obs):
        scans = obs[:,self.obs_slicing[5]:]
        scans = torch.Tensor(scans).view(obs.shape[0],1,self.n_scans*2)
        scans = scans#.to("cpu")
        ret, lat = self.LAE(scans)
        lat = lat.detach().numpy()
        obs = np.concatenate((obs[:,:self.obs_slicing[5]], lat), axis=1)
        return obs

    def step(self, obs, state=None, mask=None, deterministic=False):
        obs = self.encode_obs(obs)
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action,
                                                    self.value_flat,
                                                    self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action,
                                                    self.value_flat,
                                                    self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        obs = self.encode_obs(obs)
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        obs = self.encode_obs(obs)
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

    def crop(self, dimension, start, end):
        # Crops (or slices) a Tensor on a given dimension from start to end
        def func(x):
            if dimension == 0:
                return x[start: end]
            if dimension == 1:
                return x[:, start: end]
            if dimension == 2:
                return x[:, :, start: end]
            if dimension == 3:
                return x[:, :, :, start: end]
            if dimension == 4:
                return x[:, :, :, :, start: end]
        return Lambda(func)
