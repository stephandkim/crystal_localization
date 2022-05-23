from datetime import datetime
import argparse
import json
import numpy as np
import torch as th
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import random
from stable_baselines3.dqn.policies import DQNPolicy, MlpPolicy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import time
import os
import src


class CustomDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, net_arch=src.config.NET_ARCH)


def get_args():
    time_now = datetime.now().strftime('%Y%m%d_%H%M%S')

    parser = argparse.ArgumentParser(description='Training a RL agent for localizing xtals.')
    # parser.add_argument('--id', help='Identifier of this run', type=str, default=time_now)
    # parser.add_argument('--save_path_parent', help='Parent path for saving data', type=str, default='data/')
    # parser.add_argument('--num_env', help='Number of environments', type=int, default=1)
    # parser.add_argument('--image_path', help='Path for images', type=str, default='images/set2/')
    #
    # parser.add_argument('--total_timesteps', help='Total timesteps', type=int, default=1_000_000)
    # parser.add_argument('--save_freq', help='Model save frequency', type=int, default=1_000)
    # parser.add_argument('--target_update_interval', help='Target update interval', type=int, default=10_000)

    args = parser.parse_args()

    args.id = '20220523_1'
    args.save_path_parent = '/scratch/network/stephank/'
    # args.save_path_parent = 'data/'
    args.num_env = 12
    args.image_path = 'images/set2/'

    args.total_timesteps = 1_000_000
    args.save_freq = 10_000
    args.target_update_interval = 10_0000

    args.save_path = args.save_path_parent + args.id
    args.image_names = [n[:-4] for n in os.listdir(args.image_path) if n[-4:] == '.npy']
    args.algo = 'DQN'
    args.net_arch = src.config.NET_ARCH
    print(args)

    return args


def make_env(images, rank, save_replay=False, seed_val=0, random_seed=False):
    def _init():
        env = src.XtalEnv(images=images, save_replay=save_replay)
        env = Monitor(env=env)
        if random_seed:
            env.seed()
        else:
            env.seed(seed_val + rank)
        return env
    # set_random_seed(seed_val)
    return _init


def train(args):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    with open(args.save_path + '/parameters.json', 'w') as outfile:
        json.dump(vars(args), outfile, indent=4)
        outfile.close()

    images = [np.load(args.image_path + image_name + '.npy') for image_name in args.image_names]
    env = src.XtalEnv(images=images)
    env.reset(start_full_image=False)

    env = SubprocVecEnv([make_env(images=images, rank=0, random_seed=True) for i in range(args.num_env)])
    model = DQN(policy=CustomDQNPolicy,
                env=env,
                verbose=2,
                tensorboard_log=args.save_path,
                target_update_interval=args.target_update_interval,
                )
    checkpoint_callback = CheckpointCallback(save_freq=args.save_freq,
                                             save_path=args.save_path,
                                             name_prefix='model'
                                             )
    start = time.time()
    model.learn(total_timesteps=args.total_timesteps,
                callback=checkpoint_callback,
                tb_log_name='run',
                reset_num_timesteps=True
                )
    end = time.time()
    args.time_taken = end - start

    with open(args.save_path + '/time_taken.json', 'w') as outfile:
        json.dump({'time_taken': args.time_taken}, outfile, indent=4)
        outfile.close()

    print(args)

if __name__ == '__main__':
    args = get_args()
    train(args)
