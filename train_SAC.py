import sys
rem_path = "/opt/ros/kinetic/lib/python2.7/dist-packages"
if rem_path in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")

from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC
from spinup import sac_pytorch as SU_sac

from millify import millify
import numpy as np


from importlib import import_module
import argparse
import os
import shutil
import yaml
from util.tb_logging import Logger

pre_path = ""
#pre_path = "/home/ucabds6/projects/mppl_rl/aslaug3d_simulation/"
            

class AslaugTrainer:
    def __init__(self):
        # Define internal counters
        self.counter = {"n_steps": 0, "model_idx": 0, "cl_idx": 0,
                        "info_idx": 0, "ADR_idx": 1, "ADR_lvl": 0,
                        "last_adaption": None, "last_adr_idx": 0,
                        "logger": None}

        # Parse arguments
        self.parse_args()

        # Prepare directory structure
        self.prepare_directory()

        # Prepare curriculum learning
        self.cl_list = self.prepare_curriculum_learning(self.args['cl'])

        # Load parameters
        with open(pre_path+"params_sac.yaml") as f:
            params_all = yaml.load(f)
        self.learning_params = params_all["learning_params"]
        self.env_params = params_all["environment_params"]

        # Prepare learning function
        lp = self.learning_params
        if type(lp["learning_rate"]) in [list, tuple]:
            lr_params = lp["learning_rate"]
            lp["learning_rate"] = self.create_custom_lr(*lr_params)
        if type(lp["cliprange"]) in [list, tuple]:
            lr_params = lp["cliprange"]
            lp["cliprange"] = self.create_custom_lr(*lr_params)

        # Inizialize gyms as vector environments
        aslaug_mod = import_module("envs." + self.model_name)
        
        def create_gym(): return aslaug_mod.AslaugEnv(params=self.env_params)
        #env = create_gym() #SubprocVecEnv([create_gym for i in range(self.args['n_cpu'])])
        env = DummyVecEnv([create_gym])
        #env = DummyVecEnv([create_gym for i in range(self.args['n_cpu'])])
        g_env = create_gym()

        # Obtain observation slicing for neural network adaption
        obs_slicing = g_env.obs_slicing if hasattr(g_env,
                                                   "obs_slicing") else None
        lidar_calib = np.array(g_env.get_lidar_calibration())
        np.save(pre_path+"data/saved_models/{}/lidar_calib.npy".format(self.folder_name), lidar_calib)

        SAC_params = {}
        SAC_params['gamma'] = 0.999
        SAC_params['learning_rate']= lp['learning_rate']
        SAC_params['learning_starts']=1000
        SAC_params['ent_coef']= lp['ent_coef']
        SAC_params["buffer_size"] = 100 #400e3
        
        policy_params = {}#{"obs_slicing": obs_slicing}
        policy_params['n_env'] = 1
        policy_params['n_steps'] = 1
        policy_params['n_batch'] = None
        policy_params["obs_slicing"] = obs_slicing
                          
        # Prepare model, either new or proceeding training (pt)
        if self.args['pt'] is None:
            # MlpPolicy
            model = SAC(self.policy, env, verbose=1,
                         tensorboard_log=pre_path+"data/tb_logs/{}".format(self.folder_name),
                         policy_kwargs=policy_params,
                         **SAC_params)
                
        else:
            pfn, pep = self.args['pt'].split(":")
            model_path = pre_path+"data/saved_models/{}/aslaug_{}_{}.pkl".format(pfn, self.args['version'], pep)
            tb_log_path = pre_path+"data/tb_logs/{}".format(self.folder_name)
            model = SAC.load(model_path, env=env, verbose=1,
                              tensorboard_log=tb_log_path,
                              policy_kwargs=policy_params,
                              **SAC_params)
        self.model = model

    def train(self):
        # Print number of trainable weights
        n_els = np.sum([x.shape.num_elements()*x.trainable
                        for x in self.model.get_parameter_list()])
        print("Number of trainable weights: {}".format(n_els))
        # Start learning
        self.model.learn(total_timesteps=int(self.args['steps']),
                         callback=self.callback)

        # Save model
        self.model.save(self.dir_path + self.model_name + ".pkl")
        
        #model.learn(total_timesteps=50000, log_interval=10)
        #model.save("sac_pendulum")

    def parse_args(self):
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("-s", "--steps",
                            help="Define # steps for learning.",
                            default=10e6)
        parser.add_argument("-n", "--n_cpu",
                            help="Define # processes to use.",
                            default=1)
        parser.add_argument("-v", "--version",
                            help="Set env version.",
                            default="v0")
        parser.add_argument("-f", "--folder",
                            help="Name the folder to save models.",
                            default="None")
        parser.add_argument("-p", "--policy",
                            help="Define policy to use (import path).",
                            default="policies.aslaug_sac_v0.CustomPolicySAC")
        parser.add_argument("-cp", "--check_point",
                            help="# steps in between model checkpoints.",
                            default=500e3)
        parser.add_argument("-cl", "--curriculum_learning", action='append',
                            help="Enable curriculum learning. Example to \
                            adjust parameter reward.r1 from 1 to 5 in 3M \
                            steps, starting at 1M: -cl reward.r1:1:5:1e6:3e6.")
        parser.add_argument("-pt", "--proceed_training",
                            help="Specify model from which training shall be \
                            proceeded. Format: folder_name:episode")
        args = parser.parse_args()

        self.args = {"n_cpu": int(float(args.n_cpu)), "version": args.version,
                     "steps": int(float(args.steps)),
                     "policy_arg": args.policy, "n_cp": args.check_point,
                     "folder_name": args.folder,
                     "cl": args.curriculum_learning,
                     "pt": args.proceed_training}

        # Define model name
        self.model_name = "aslaug_{}".format(self.args['version'])

        # Load module for policy
        policy_mod_name = ".".join(self.args['policy_arg'].split(".")[:-1])
        self.policy_name = self.args['policy_arg'].split(".")[-1]
        self.policy_mod = import_module(policy_mod_name)
        self.policy = getattr(self.policy_mod, self.policy_name)
        # Define folder name
        if self.args['folder_name'] == "None":
            self.folder_name = self.model_name
        else:
            self.folder_name = self.args['folder_name']

        # Define directory to save files to
        self.dir_path = pre_path+"data/saved_models/{}/".format(self.folder_name)
        # self.prepare_directory()

    def prepare_directory(self):
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)
        elif len(os.listdir(self.dir_path)) == 0:
            print("Directory exists, but is empty. Proceeding.")
        else:
            print("Attention, {} already exists.".format(self.folder_name))
            resp = input("Move it [m], delete it [r] or cancel [c]: ")
            if resp == 'c':
                exit()
            elif resp == 'r':
                shutil.rmtree(self.dir_path)
                os.mkdir(self.dir_path)
            elif resp == 'm':
                resp = input("Enter new folder name for \
                             {}: ".format(self.folder_name))
                shutil.move(self.dir_path,
                            pre_path+"data/saved_models/{}/".format(resp))
                os.mkdir(self.dir_path)
            else:
                print("Can't understand your expression.")
                exit()

        # Copy model to directory
        shutil.copy(pre_path+"envs/{}.py".format(self.model_name),
                    self.dir_path + self.model_name + ".py")

        # Save command to file for reference
        text_file = open(os.path.join(self.dir_path, 'cmd.txt'), "w")
        text_file.write(" ".join(sys.argv))
        text_file.close()

        # Save learning params to file
        params_file = pre_path+"data/saved_models/{}/params_sac.yaml".format(self.folder_name)
        shutil.copy(pre_path+"params_sac.yaml", params_file)

        # Copy policy to models folder
        shutil.copy(self.policy_mod.__file__,
                    pre_path+"data/saved_models/{}/{}.py".format(self.folder_name,
                                                        self.policy_name))

    def prepare_curriculum_learning(self, cl):
        # Prepare curriculum learning
        if cl is not None:
            cl_list = []
            for clstr in cl:
                (cl_param, cl_start, cl_end,
                 cl_begin, cl_steps) = clstr.split(":")
                cl_list.append({"param": cl_param, "start": float(cl_start),
                                "end": float(cl_end), "begin": float(cl_begin),
                                "steps": float(cl_steps)})
        else:
            cl_list = None
        return cl_list

    # Prepare custom learning rate function
    def create_custom_lr(self, lr_max, lr_min, a, b):
        m = (lr_max - lr_min) / (a - b)
        c = lr_max - m * a

        return lambda x: np.min([lr_max, np.max([lr_min, m * x + c])])

    # Please excuse the horrible formatting
    def callback(self, _locals, _globals):

        n_cp_simple = 0
        self.counter['n_steps'] += 1
        if self.counter['logger'] is None:
            sac_id = 1
            sac_path = pre_path+'data/tb_logs/{}/SAC_{}'.format(self.folder_name,
                                                        sac_id+1)
            while os.path.exists(sac_path):
                sac_id += 1
                sac_path = pre_path+'data/tb_logs/{}/SAC_{}'.format(self.folder_name,
                                                            sac_id+1)
            sac_path = pre_path+'data/tb_logs/{}/SAC_{}/addons'.format(
                self.folder_name, sac_id)
            self.counter['logger'] = Logger(sac_path)
        
        if (self.counter['n_steps'] / float(self.args['n_cp'])
                >= self.counter['model_idx']):
            n_cp_simple = millify(
                float(self.counter['model_idx']) * float(self.args['n_cp']),
                precision=6)
            suffix = "_{}.pkl".format(n_cp_simple)
            cp_name = self.model_name + suffix
            self.model.save(self.dir_path + cp_name)
            self.counter['model_idx'] += 1
            data = {"version": self.args['version'],
                    "model_path": self.dir_path + cp_name}
            with open('latest.json', 'w') as outfile:
                yaml.dump(data, outfile)
            print("Stored model at episode {}.".format(n_cp_simple))


        if (self.cl_list is not None and self.counter['n_steps'] / 25000.0
                >= self.counter['cl_idx']):
            self.counter['cl_idx'] += 1
            self.perform_CL()

        if self.counter['n_steps'] / 5000.0 >= self.counter['info_idx']:
            self.counter['info_idx'] += 1
            msr_avg = np.average(
              self.model.env.env_method("get_success_rate"))
            self.counter['logger'].log_scalar('metrics/success_rate', msr_avg,
                                              self.counter['n_steps'])

        if self.counter['n_steps'] / 25000.0 >= self.counter['ADR_idx'] \
                and len(self.env_params['adr']['adaptions']) > 0:
            self.counter['ADR_idx'] += 1
            self.perform_ADR()

    # Please excuse the horrible formatting
    def perform_ADR(self):
        avg = np.average(self.model.env.env_method("get_success_rate"))
        print("Average success rate: {}".format(avg))
        for level in range(len(self.env_params['adr']['adaptions'])):
            for adaption in self.env_params['adr']['adaptions'][level]:
                val = np.average(self.model.env.env_method(
                    "get_param", adaption['param']))
                self.counter['logger'].log_scalar(
                    'ADR/{}/{}'.format(level, adaption['param']), val,
                    self.counter['n_steps'])

        if avg >= self.env_params['adr']['success_threshold']:
            to_adapt = []
            if self.counter['ADR_lvl'] \
                    < len(self.env_params['adr']['adaptions']):
                adaptions = self.env_params['adr']['adaptions']
                for adapts in adaptions[self.counter['ADR_lvl']]:
                    val = np.average(self.model.env.env_method(
                        "get_param", adapts['param']))
                    if val != adapts['end']:
                        to_adapt.append(adapts)

            if len(to_adapt) > 0:
                rnd_idx = np.random.randint(len(to_adapt))
                adapts = to_adapt[rnd_idx]
                self.counter['last_adaption'] = adapts
                val = np.average(self.model.env.env_method(
                    "get_param", adapts['param']))
                dval = +(adapts['end']-adapts['start'])/adapts['steps']
                val = max(min(adapts['end'], adapts['start']), min(
                    max(adapts['end'], adapts['start']), val + dval))

                print("Setting {} to {}(+)".format(adapts['param'], val))
                self.model.env.env_method("set_param", adapts['param'], val)
            else:
                self.counter['ADR_lvl'] = min(
                    self.counter['ADR_lvl'] + 1,
                    len(self.env_params['adr']['adaptions']))
        if avg <= self.env_params['adr']['fail_threshold']:
            if self.counter['last_adaption'] is not None:
                adaptions = self.env_params['adr']['adaptions']
                if self.counter['last_adaption'] \
                        not in adaptions[self.counter['ADR_lvl']]:
                    self.counter['ADR_lvl'] = max(
                        0, self.counter['ADR_lvl'] - 1)
                else:
                    val = np.average(self.model.env.env_method(
                        "get_param", self.counter['last_adaption']['param']))
                    dval = (-(self.counter['last_adaption']['end']
                              - self.counter['last_adaption']['start'])
                            / self.counter['last_adaption']['steps'])
                    val_h1 = max(self.counter['last_adaption']['end'],
                                 self.counter['last_adaption']['start'])
                    val = max(min(self.counter['last_adaption']['end'],
                                  self.counter['last_adaption']['start']),
                              min(val_h1, val + dval))
                    print("Setting {} to {}(-)\
                          ".format(self.counter['last_adaption']['param'],
                                   val))
                    self.model.env.env_method(
                        "set_param", self.counter['last_adaption']['param'],
                        val)
                    self.counter['last_adaption'] = None

    # Please excuse the horrible formatting
    def perform_CL(self):
        for cl_entry in self.cl_list:
            cl_val = (cl_entry["start"]
                      + (cl_entry["end"]
                         - cl_entry["start"])
                      * (self.counter['n_steps']-cl_entry["begin"])
                      / cl_entry["steps"])

            cl_h1 = cl_entry["begin"] + cl_entry["steps"]
            if cl_val >= min(cl_entry["start"], cl_entry["end"]) \
                    and cl_val <= max(cl_entry["start"], cl_entry["end"]) \
                    and self.counter['n_steps'] >= cl_entry["begin"] \
                    and self.counter['n_steps'] <= cl_h1:
                self.model.env.env_method("set_param",
                                          cl_entry["param"], cl_val)
                print("Modifying param {} to {}".format(cl_entry["param"],
                                                        cl_val))
                self.counter['logger'].log_scalar(
                    'CL/{}'.format(cl_entry['param']), cl_val,
                    self.counter['n_steps'])


def main():
    trainer = AslaugTrainer()
    trainer.train()
    
if __name__ == '__main__':
    main()
