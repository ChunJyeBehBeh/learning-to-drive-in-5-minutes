# Code adapted from https://github.com/araffin/rl-baselines-zoo
# Author: Antonin Raffin
# python enjoy.py --algo sac -vae vae-level-0-dim-32.pkl --exp-id 1 -n 5000
# python enjoy.py --algo sac --exp-id 15 -n 5000
# python enjoy.py --algo ddpg -vae logs/vae-64.pkl --exp-id 24 -n 5000
import argparse
import os
import time
import random

import gym
import numpy as np
from stable_baselines.common import set_global_seeds

from utils.utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', help='Log folder', type=str, default='logs')
parser.add_argument('--algo', help='RL Algorithm', default='sac',
                    type=str, required=False, choices=list(ALGOS.keys()))
parser.add_argument('-n', '--n-timesteps', help='number of timesteps', default=1000,
                    type=int)
parser.add_argument('--exp-id', help='Experiment ID (-1: no exp folder, 0: latest)', default=0,
                    type=int)
parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                    type=int)
parser.add_argument('--no-render', action='store_true', default=False,
                    help='Do not render the environment (useful for tests)')
parser.add_argument('--deterministic', action='store_true', default=False,
                    help='Use deterministic actions')
parser.add_argument('--norm-reward', action='store_true', default=False,
                    help='Normalize reward if applicable (trained with VecNormalize)')
parser.add_argument('--seed', help='Random generator seed', type=int, default=41)
parser.add_argument('--reward-log', help='Where to log reward', default='', type=str)
parser.add_argument('-vae', '--vae-path', help='Path to saved VAE', type=str, default='')
parser.add_argument('-best', '--best-model', action='store_true', default=False,
                    help='Use best saved model of that experiment (if it exists)')
parser.add_argument('--level', help='Level index', type=int, default=0)
args = parser.parse_args()

algo = args.algo
folder = args.folder
ENV_ID = "DonkeyVae-v0-level-{}".format(args.level)

if args.exp_id == 0:
    args.exp_id = get_latest_run_id(os.path.join(folder, algo), ENV_ID)
    print('Loading latest experiment, id={}'.format(args.exp_id))

# Sanity checks
if args.exp_id > 0:
    log_path = os.path.join(folder, algo, '{}_{}'.format(ENV_ID, args.exp_id))
else:
    log_path = os.path.join(folder, algo)

best_path = ''
if args.best_model:
    best_path = '_best'

model_path = os.path.join(log_path, "{}{}.pkl".format(ENV_ID, best_path))

assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)  # log_path: logs\ddpg
assert os.path.isfile(model_path), "No model found for {} on {}, path: {}".format(algo, ENV_ID, model_path)

set_global_seeds(args.seed)

stats_path = os.path.join(log_path, ENV_ID)

hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward)
hyperparams['vae_path'] = args.vae_path

log_dir = args.reward_log if args.reward_log != '' else None

print("!!!!!!!!!!",stats_path)
env = create_test_env(args.level, stats_path=stats_path, seed=args.seed, log_dir=log_dir,
                      hyperparams=hyperparams)

print("Loaded Model path from {}".format(model_path))
model = ALGOS[algo].load(model_path)

obs = env.reset()

# Force deterministic for SAC and DDPG
deterministic = args.deterministic or algo in ['ddpg', 'sac']
if args.verbose >= 1:
    print("Deterministic actions: {}".format(deterministic))

running_reward = 0.0
ep_len = 0
throttle_store = np.array([])
steering_store = np.array([])

for step in range(args.n_timesteps):
    action, _ = model.predict(obs, deterministic=deterministic)
    # Clip Action to avoid out of bound errors
    if isinstance(env.action_space, gym.spaces.Box):
        action = np.clip(action, env.action_space.low, env.action_space.high)
    
    throttle_store = np.append(throttle_store,action[0][0])
    steering_store = np.append(steering_store,action[0][1])
    obs, reward, done, infos = env.step(action)
    if not args.no_render:
        env.render('human')
    running_reward += reward[0]
    ep_len += 1

    if done and args.verbose >= 1 or step+1 == args.n_timesteps:
        # NOTE: for env using VecNormalize, the mean reward
        # is a normalized reward when `--norm_reward` flag is passed
        print("Episode Reward: {:.2f}".format(running_reward))
        print("Episode Length", ep_len)

        value_to_save = {'throttle':throttle_store,
                            'steering':steering_store
        }
        np.savez("result_processing\\result_{}".format(args.exp_id),**value_to_save)

        if(step+1<args.n_timesteps):
            print("Episode stop! Comment the [break] in if done to test more than one episode!")       
        
        running_reward = 0.0
        ep_len = 0
        
        # https://github.com/tleyden/learning-to-drive-in-5-minutes/commit/f32592be64238509d595b798b37402000fce9444
        # print("Regenerating track")
        # donkeyEnv = env.envs[0].env
        # road_styles = range(5)
        # donkeyEnv.regen_road(rand_seed=int(time.time()), road_style=random.choice(road_styles))

env.reset()
env.envs[0].env.exit_scene()
# Close connection does work properly for now
# env.envs[0].env.close_connection()
time.sleep(0.5)
