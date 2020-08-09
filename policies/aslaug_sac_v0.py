import tensorflow as tf
from stable_baselines.sac.policies import *
from tensorflow.keras.layers import Lambda


class CustomPolicySAC(SACPolicy):
    """
    Policy object that implements a DDPG-like actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param layer_norm: (bool) enable layer normalisation
    :param reg_weight: (float) Regularization loss weight for the policy parameters
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """
    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, layers=None,
        cnn_extractor=nature_cnn, feature_extraction="cnn", reg_weight=0.0,
        layer_norm=False, act_fun=tf.nn.relu, 
        obs_slicing=[0, 6, 9, 57, 64, 71, 272, 473], **kwargs):
        super(CustomPolicySAC, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                                reuse=reuse, scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)
        self.layer_norm = layer_norm
        self.feature_extraction = feature_extraction
        self.cnn_kwargs = kwargs
        self.cnn_extractor = self.make_feature_extractor #cnn_extractor
        self.reuse = reuse
        if layers is None:
            layers = [256, 128, 64, 32, 16]
        self.layers = layers
        self.reg_loss = None
        self.reg_weight = reg_weight
        self.entropy = None
        self.obs_slicing = obs_slicing

        assert len(layers) >= 1, "Error: must have at least one hidden layer for the policy."

        self.activ_fn = act_fun

    def make_feature_extractor(self, obs=None, reuse=True, scope="ft_extr"):
        if obs is None:
            obs = self.processed_obs
        # Scale observation
        obs_avg = tf.constant((self.ob_space.high+self.ob_space.low)/2.0,
                              name="obs_avg")
        obs_dif = tf.constant((self.ob_space.high - self.ob_space.low),
                              name="obs_diff")
        shifted = tf.math.subtract(self.processed_obs, obs_avg)
        proc_obs = tf.math.divide(shifted, obs_dif)
        if self.obs_slicing is not None:
            obs_slicing = self.obs_slicing
        else:
            obs_slicing = [0, 6, 9, 57, 64, 71, 272, 473]
        # Create network
        lrelu = tf.nn.leaky_relu
        o = obs_slicing
        in_sp = self.crop(1, o[0], o[1])(proc_obs)
        in_mb = self.crop(1, o[1], o[2])(proc_obs)
        in_lp = self.crop(1, o[2], o[3])(proc_obs)
        in_jp = self.crop(1, o[3], o[4])(proc_obs)
        in_jv = self.crop(1, o[4], o[5])(proc_obs)
        in_sc1 = self.crop(1, o[5], o[6])(proc_obs)
        in_sc2 = self.crop(1, o[6], o[7])(proc_obs)

        sl_1 = tf.layers.Conv1D(2, 11, activation=lrelu, name="sc_1")
        sl_2 = tf.layers.Conv1D(4, 7, activation=lrelu, name="sc_2")
        sl_3 = tf.layers.Conv1D(8, 3, activation=lrelu, name="sc_3")
        sl_4 = tf.layers.MaxPooling1D(3, 3, name="sc_4")
        sl_5 = tf.layers.Conv1D(8, 7, activation=lrelu, name="sc_5")
        sl_6 = tf.layers.Conv1D(8, 5, activation=lrelu, name="sc_6")
        sl_7 = tf.layers.MaxPooling1D(3, 3, name="sc_7")
        sl_8 = tf.layers.Conv1D(8, 3, activation=lrelu, name="sc_8")
        sl_9 = tf.layers.Flatten(name="sc_9")
        sl_10 = tf.layers.Dense(128, activation=lrelu, name="sc_10")
        sl_11 = tf.layers.Dense(64, activation=lrelu, name="s1_11")
        sl_out = tf.layers.Dense(64, activation=lrelu, name="s1_out")

        s1_0 = tf.expand_dims(in_sc1, -1)
        s1_1 = sl_1(s1_0)
        s1_2 = sl_2(s1_1)
        s1_3 = sl_3(s1_2)
        s1_4 = sl_4(s1_3)
        s1_5 = sl_5(s1_4)
        s1_6 = sl_6(s1_5)
        s1_7 = sl_7(s1_6)
        s1_8 = sl_8(s1_7)
        s1_9 = sl_9(s1_8)
        s1_10 = sl_10(s1_9)
        s1_11 = sl_11(s1_10)
        s1_out = sl_out(s1_11)

        s2_0 = tf.expand_dims(in_sc2, -1)
        s2_1 = sl_1(s2_0)
        s2_2 = sl_2(s2_1)
        s2_3 = sl_3(s2_2)
        s2_4 = sl_4(s2_3)
        s2_5 = sl_5(s2_4)
        s2_6 = sl_6(s2_5)
        s2_7 = sl_7(s2_6)
        s2_8 = sl_8(s2_7)
        s2_9 = sl_9(s2_8)
        s2_10 = sl_10(s2_9)
        s2_11 = sl_11(s2_10)
        s2_out = sl_out(s2_11)

        sc_0 = tf.keras.layers.Concatenate(name="sc_0")([s1_out, s2_out])
        sc_1 = tf.layers.Dense(128, activation=lrelu, name="sc_1")(sc_0)
        sc_2 = tf.layers.Dense(64, activation=lrelu, name="sc_2")(sc_1)
        sc_out = tf.layers.Dense(64, activation=lrelu, name="sc_out")(sc_2)

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

        feature_extractor = c_out
        return feature_extractor

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                pi_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                pi_h = tf.layers.flatten(obs)

            pi_h = mlp(pi_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)

            self.act_mu = mu_ = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None)
            # Important difference with SAC and other algo such as PPO:
            # the std depends on the state, so we cannot use stable_baselines.common.distribution
            log_std = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None)

        # Regularize policy output (not used for now)
        # reg_loss = self.reg_weight * 0.5 * tf.reduce_mean(log_std ** 2)
        # reg_loss += self.reg_weight * 0.5 * tf.reduce_mean(mu ** 2)
        # self.reg_loss = reg_loss

        # OpenAI Variation to cap the standard deviation
        # activation = tf.tanh # for log_std
        # log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        # Original Implementation
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

        self.std = std = tf.exp(log_std)
        # Reparameterization trick
        pi_ = mu_ + tf.random_normal(tf.shape(mu_)) * std
        logp_pi = gaussian_likelihood(pi_, mu_, log_std)
        self.entropy = gaussian_entropy(log_std)
        # MISSING: reg params for log and mu
        # Apply squashing and account for it in the probability
        deterministic_policy, policy, logp_pi = apply_squashing_func(mu_, pi_, logp_pi)
        self.policy = policy
        self.deterministic_policy = deterministic_policy

        return deterministic_policy, policy, logp_pi

    def make_critics(self, obs=None, action=None, reuse=False, scope="values_fn",
                     create_vf=True, create_qf=True):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                critics_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                critics_h = tf.layers.flatten(obs)

            if create_vf:
                # Value function
                with tf.variable_scope('vf', reuse=reuse):
                    vf_h = mlp(critics_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    value_fn = tf.layers.dense(vf_h, 1, name="vf")
                self.value_fn = value_fn

            if create_qf:
                # Concatenate preprocessed state and action
                qf_h = tf.concat([critics_h, action], axis=-1)

                # Double Q values to reduce overestimation
                with tf.variable_scope('qf1', reuse=reuse):
                    qf1_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    qf1 = tf.layers.dense(qf1_h, 1, name="qf1")

                with tf.variable_scope('qf2', reuse=reuse):
                    qf2_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    qf2 = tf.layers.dense(qf2_h, 1, name="qf2")

                self.qf1 = qf1
                self.qf2 = qf2

        return self.qf1, self.qf2, self.value_fn

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run(self.deterministic_policy, {self.obs_ph: obs})
        return self.sess.run(self.policy, {self.obs_ph: obs})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run([self.act_mu, self.std], {self.obs_ph: obs})


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


class AslaugPolicySAC(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, **kwargs):
        super(AslaugPolicySAC, self).__init__(sess, ob_space, ac_space,
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

        if "obs_slicing" in kwargs and kwargs["obs_slicing"] is not None:
            obs_slicing = kwargs["obs_slicing"]
        else:
            obs_slicing = [0, 6, 9, 57, 64, 71, 272, 473]
        # Create network
        with tf.variable_scope("model/inputs", reuse=reuse):
            lrelu = tf.nn.leaky_relu
            o = obs_slicing
            in_sp = self.crop(1, o[0], o[1])(proc_obs)
            in_mb = self.crop(1, o[1], o[2])(proc_obs)
            in_lp = self.crop(1, o[2], o[3])(proc_obs)
            in_jp = self.crop(1, o[3], o[4])(proc_obs)
            in_jv = self.crop(1, o[4], o[5])(proc_obs)
            in_sc1 = self.crop(1, o[5], o[6])(proc_obs)
            in_sc2 = self.crop(1, o[6], o[7])(proc_obs)

        with tf.variable_scope("model/scan_block", reuse=reuse):

            sl_1 = tf.layers.Conv1D(2, 11, activation=lrelu, name="sc_1")
            sl_2 = tf.layers.Conv1D(4, 7, activation=lrelu, name="sc_2")
            sl_3 = tf.layers.Conv1D(8, 3, activation=lrelu, name="sc_3")
            sl_4 = tf.layers.MaxPooling1D(3, 3, name="sc_4")
            sl_5 = tf.layers.Conv1D(8, 7, activation=lrelu, name="sc_5")
            sl_6 = tf.layers.Conv1D(8, 5, activation=lrelu, name="sc_6")
            sl_7 = tf.layers.MaxPooling1D(3, 3, name="sc_7")
            sl_8 = tf.layers.Conv1D(8, 3, activation=lrelu, name="sc_8")
            sl_9 = tf.layers.Flatten(name="sc_9")
            sl_10 = tf.layers.Dense(128, activation=lrelu, name="sc_10")
            sl_11 = tf.layers.Dense(64, activation=lrelu, name="s1_11")
            sl_out = tf.layers.Dense(64, activation=lrelu, name="s1_out")

            s1_0 = tf.expand_dims(in_sc1, -1)
            s1_1 = sl_1(s1_0)
            s1_2 = sl_2(s1_1)
            s1_3 = sl_3(s1_2)
            s1_4 = sl_4(s1_3)
            s1_5 = sl_5(s1_4)
            s1_6 = sl_6(s1_5)
            s1_7 = sl_7(s1_6)
            s1_8 = sl_8(s1_7)
            s1_9 = sl_9(s1_8)
            s1_10 = sl_10(s1_9)
            s1_11 = sl_11(s1_10)
            s1_out = sl_out(s1_11)

            s2_0 = tf.expand_dims(in_sc2, -1)
            s2_1 = sl_1(s2_0)
            s2_2 = sl_2(s2_1)
            s2_3 = sl_3(s2_2)
            s2_4 = sl_4(s2_3)
            s2_5 = sl_5(s2_4)
            s2_6 = sl_6(s2_5)
            s2_7 = sl_7(s2_6)
            s2_8 = sl_8(s2_7)
            s2_9 = sl_9(s2_8)
            s2_10 = sl_10(s2_9)
            s2_11 = sl_11(s2_10)
            s2_out = sl_out(s2_11)

            sc_0 = tf.keras.layers.Concatenate(name="sc_0")([s1_out, s2_out])
            sc_1 = tf.layers.Dense(128, activation=lrelu, name="sc_1")(sc_0)
            sc_2 = tf.layers.Dense(64, activation=lrelu, name="sc_2")(sc_1)
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

    def step(self, obs, state=None, mask=None, deterministic=False):
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
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
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
