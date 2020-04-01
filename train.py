# Code adapted from https://github.com/araffin/rl-baselines-zoo
# Author: Antonin Raffin
# python train.py --algo sac -vae vae-level-0-dim-32.pkl -n 250000
# python train.py --algo ddpg -vae vae-level-0-dim-32.pkl -n 250000
# python train.py --algo ddpg -n 7500000
# python train.py --algo sac -vae logs/vae-64.pkl -n 250000
import argparse
import os
import time
from collections import OrderedDict
from pprint import pprint

import numpy as np
import yaml
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv
from stable_baselines.ddpg import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
# from stable_baselines.ppo2.ppo2 import constfn
from stable_baselines.common.schedules import constfn
from stable_baselines.gail import ExpertDataset

from config import MIN_THROTTLE, MAX_THROTTLE, FRAME_SKIP,\
    MAX_CTE_ERROR, SIM_PARAMS, N_COMMAND_HISTORY, Z_SIZE, MAX_STEERING_DIFF
from utils.utils import make_env, ALGOS, linear_schedule, get_latest_run_id, load_vae, create_callback
from teleop.teleop_client import TeleopEnv
from teleop.recorder import Recorder

import time
import datetime


sim_path = "E:\BEH FYP\projects\donkey_window\DonkeySim.exe"
os.environ['DONKEY_SIM_PATH'] = sim_path
os.environ['DONKEY_SIM_PORT'] = '9091'
os.environ['DONKEY_SIM_HEADLESS'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='log', type=str)
parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training',
                    default='', type=str)
parser.add_argument('--algo', help='RL Algorithm', default='sac',
                    type=str, required=False, choices=list(ALGOS.keys()))
parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=-1,
                    type=int)
parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1,
                    type=int)
parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
parser.add_argument('-vae', '--vae-path', help='Path to saved VAE', type=str, default='')
parser.add_argument('--save-vae', action='store_true', default=False,
                    help='Save VAE')
parser.add_argument('--seed', help='Random generator seed', type=int, default=50)

parser.add_argument('--level', help='Level index', type=int, default=0)

parser.add_argument('--level', help='Level index', type=int, default=0)
parser.add_argument('--random-features', action='store_true', default=False,
                    help='Use random features')
parser.add_argument('--teleop', action='store_true', default=False,
                    help='Use teleoperation for training')
parser.add_argument('-pretrain', '--pretrain-path', type=str,
                    help='Path to an expert dataset for pretraining')
parser.add_argument('--n-epochs', type=int, default=50,
                    help='Number of epochs when doing pretraining')
parser.add_argument('--batch-size', type=int, default=64,
                    help='Minibatch size when doing pretraining')
parser.add_argument('--traj-limitation', type=int, default=-1,
                    help='The number of trajectory to use (if -1, load all)')
args = parser.parse_args()

set_global_seeds(args.seed)
ENV_ID = "DonkeyVae-v0-level-{}".format(args.level)

if args.trained_agent != "":
    assert args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent), \
        "The trained_agent must be a valid path to a .pkl file"

tensorboard_log = None if args.tensorboard_log == '' else os.path.join(args.tensorboard_log, ENV_ID)

print("=" * 10, ENV_ID, args.algo, "=" * 10)
print("Tensorboard Directory ------->",tensorboard_log)

vae = None
if args.vae_path != '':
    print("Loading VAE ...")
    vae = load_vae(args.vae_path)
elif args.random_features:
    print("Randomly initialized VAE")
    vae = load_vae(z_size=Z_SIZE)
    # Save network
    args.save_vae = True
else:
    print("Learning from pixels...")

# Load hyperparameters from yaml file
with open('hyperparams/{}.yml'.format(args.algo), 'r') as f:
    hyperparams = yaml.load(f)['DonkeyVae-v0']

# Sort hyperparams that will be saved
saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])
# save vae path
saved_hyperparams['vae_path'] = args.vae_path
if vae is not None:
    saved_hyperparams['z_size'] = vae.z_size

# Save simulation params
for key in SIM_PARAMS:
    saved_hyperparams[key] = eval(key)
saved_hyperparams['seed'] = args.seed
pprint(saved_hyperparams)

# Compute and create log path
log_path = os.path.join(args.log_folder, args.algo)
print("------------")
print(log_path,ENV_ID)          # logs\sac DonkeyVae-v0-level-0  
print(get_latest_run_id(log_path, ENV_ID) + 1)  # 1
print("------------")
save_path = os.path.join(log_path, "{}_{}".format(ENV_ID, get_latest_run_id(log_path, ENV_ID) + 1))
params_path = os.path.join(save_path, ENV_ID)
os.makedirs(params_path, exist_ok=True)

# Create learning rate schedules for ppo2 and sac
if args.algo in ["ppo2", "sac"]:
    for key in ['learning_rate', 'cliprange']:
        if key not in hyperparams:
            continue
        if isinstance(hyperparams[key], str):
            schedule, initial_value = hyperparams[key].split('_')
            initial_value = float(initial_value)
            hyperparams[key] = linear_schedule(initial_value)
        elif isinstance(hyperparams[key], float):
            hyperparams[key] = constfn(hyperparams[key])
        else:
            raise ValueError('Invalid valid for {}: {}'.format(key, hyperparams[key]))

# Should we overwrite the number of timesteps?
if args.n_timesteps > 0:
    n_timesteps = args.n_timesteps
else:
    n_timesteps = int(hyperparams['n_timesteps'])
del hyperparams['n_timesteps']

normalize = False
normalize_kwargs = {}
if 'normalize' in hyperparams.keys():
    normalize = hyperparams['normalize']
    if isinstance(normalize, str):
        normalize_kwargs = eval(normalize)
        normalize = True
    del hyperparams['normalize']

if 'policy_kwargs' in hyperparams.keys():
    hyperparams['policy_kwargs'] = eval(hyperparams['policy_kwargs'])
    print("!!!->",eval(hyperparams['policy_kwargs']))
    exit()

if not args.teleop:
    env = DummyVecEnv([make_env(args.level, args.seed, vae=vae, teleop=args.teleop)])
else:
    env = make_env(args.level, args.seed, vae=vae, teleop=args.teleop,
                   n_stack=hyperparams.get('frame_stack', 1))()

if normalize:
    if hyperparams.get('normalize', False) and args.algo in ['ddpg']:
        print("WARNING: normalization not supported yet for DDPG")
    else:
        print("Normalizing input and return")
        env = VecNormalize(env, **normalize_kwargs)

# Optional Frame-stacking
n_stack = 1
if hyperparams.get('frame_stack', False):
    n_stack = hyperparams['frame_stack']
    if not args.teleop:
        env = VecFrameStack(env, n_stack)
    print("Stacking {} frames".format(n_stack))
    del hyperparams['frame_stack']

# Parse noise string for DDPG and SAC
if args.algo in ['ddpg', 'sac'] and hyperparams.get('noise_type') is not None:
    noise_type = hyperparams['noise_type'].strip()
    noise_std = hyperparams['noise_std']
    n_actions = env.action_space.shape[0]
    if 'adaptive-param' in noise_type:
        hyperparams['param_noise'] = AdaptiveParamNoiseSpec(initial_stddev=noise_std,
                                                            desired_action_stddev=noise_std)
    elif 'normal' in noise_type:
        hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions),
                                                        sigma=noise_std * np.ones(n_actions))
    elif 'ornstein-uhlenbeck' in noise_type:
        hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                                   sigma=noise_std * np.ones(n_actions))
    else:
        raise RuntimeError('Unknown noise type "{}"'.format(noise_type))
    print("Applying {} noise with std {}".format(noise_type, noise_std))
    del hyperparams['noise_type']
    del hyperparams['noise_std']

if args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent):
    # Continue training
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Loading pretrained agent")
    # Policy should not be changed
    del hyperparams['policy']

    model = ALGOS[args.algo].load(args.trained_agent, env=env,
                                  tensorboard_log=tensorboard_log, verbose=1, **hyperparams)

    exp_folder = args.trained_agent.split('.pkl')[0]
    if normalize:
        print("Loading saved running average")
        env.load_running_average(exp_folder)
else:
    print("xxxxxxxxxxxxx")
    # Train an agent from scratch
    model = ALGOS[args.algo](env=env, tensorboard_log=tensorboard_log, verbose=1, **hyperparams)

if args.pretrain_path is not None:
    print("Petraining model for {} epochs".format(args.n_epochs))
    if os.path.isdir(args.pretrain_path):
        args.pretrain_path = os.path.join(args.pretrain_path, 'expert_dataset.npz')
    assert args.pretrain_path.endswith('.npz') and os.path.isfile(args.pretrain_path), "Invalid pretain path: {}".format(args.pretrain_path)
    expert_dataset = np.load(args.pretrain_path)
    # Convert dataset if needed
    if vae is not None:
        print("Converting to vae latent space...")
        expert_dataset = Recorder.convert_obs_to_latent_vec(expert_dataset, vae, N_COMMAND_HISTORY)
    # Create the dataloader and petrain (Behavior-Cloning)
    dataset = ExpertDataset(traj_data=expert_dataset,
                            traj_limitation=args.traj_limitation, batch_size=args.batch_size)

    # Fill the replay buffer
    if args.algo == "sac":
        print("Filling replay buffer")
        for i in range(len(expert_dataset['obs']) - 1):
            done = expert_dataset['episode_starts'][i + 1]
            obs, next_obs = expert_dataset['obs'][i], expert_dataset['obs'][i + 1]
            action, reward = expert_dataset['actions'][i], expert_dataset['rewards'][i]
            model.replay_buffer.add(obs, action, reward, next_obs, float(done))
        # Initialize the value fn
        model.n_updates = 0
        for _ in range(10):
            model.optimize(max(model.batch_size, model.learning_starts), None, model.learning_rate(1))
    else:
        # TODO: pretrain also the std to match the one from the dataset
        model.pretrain(dataset, n_epochs=args.n_epochs)
    del dataset

# Teleoperation mode:
# we don't wrap the environment with a monitor or in a vecenv
if args.teleop:
    assert args.algo == "sac", "Teleoperation mode is not yet implemented for {}".format(args.algo)
    env = TeleopEnv(env, is_training=True)
    model.set_env(env)
    env.model = model

kwargs = {}
if args.log_interval > -1:
    kwargs = {'log_interval': args.log_interval}

if args.algo == 'sac' or args.algo == 'ddpg' or args.algo == 'ppo2':

    kwargs.update({'callback': create_callback(args.algo,
                                               os.path.join(save_path, ENV_ID + "_best"),
                                               verbose=1)})

# Beh
def check_array(input_array,value):
    '''
    Last 5 array exceed certain value
    '''
    input_array = input_array[input_array>0]        # Delete all the negative rewards
    for i in input_array[-5:]:
        if i < value:
            return False
    return True

log_txt = open('log_training.txt',"a")
start = datetime.datetime.now()
model.learn(n_timesteps, **kwargs)
total_episode_reward_store = model.total_episode_reward
print("Length of episode: {}".format(len(total_episode_reward_store)))

x = 1
threshold = 2000
# while(round(float(np.mean(total_episode_reward_store[-5:])), 1) < threshold and not check_array(total_episode_reward_store,threshold)):
#     x = x + 1
#     print("Mean of Last 5 episode: {}".format(round(float(np.mean(model.total_episode_reward[-5:])), 1)))
#     print("Not Hitting the Target! Additional Training for n_timesteps: 5000")
#     model.learn(n_timesteps, **kwargs)
#     total_episode_reward_store = np.append(total_episode_reward_store,model.total_episode_reward)
#     print("Length of episode: {}".format(len(total_episode_reward_store)))

total_episode_reward_store = total_episode_reward_store[total_episode_reward_store>0]       # Delete all the negative rewards
end = datetime.datetime.now()
print("Training End!")

if vae!=None:
    vae_used = "Yes"
else:
    vae_used = "No"

timestr = time.strftime("%Y%m%d-%H%M%S")

if args.algo == 'ddpg':
    value_to_save ={'total_episode_reward_store':total_episode_reward_store,
                'throttle_mean':model.throttle_mean,
                'throttle_min_max':model.throttle_min_max,
                'steering_diff':model.steering_diff,
                'step_episode_store':model.step_episode_store}

    np.savez("result_processing\\Episode_reward_{}_{}_{}_{}".format(args.algo,vae_used,x*args.n_timesteps,timestr),**value_to_save)
else:
    np.savez("result_processing\\Episode_reward_{}_{}_{}_{}".format(args.algo,vae_used,x*args.n_timesteps,timestr),total_episode_reward_store)

log_txt.write("\n{} || {} || VAE: {} || n_timesteps: {} || Training time: {}\n".format(time.ctime(time.time()),args.algo,vae_used,x*args.n_timesteps,end-start))
log_txt.close()

# Beh

if args.teleop:
    env.wait()
    env.exit()
    time.sleep(0.5)
else:
    # Close the connection properly
    env.reset()
    if isinstance(env, VecFrameStack):
        env = env.venv
    # HACK to bypass Monitor wrapper
    env.envs[0].env.exit_scene()

print("Saving model to {}".format(save_path))
# Save trained model
model.save(os.path.join(save_path, ENV_ID),cloudpickle = True)
# Save hyperparams
with open(os.path.join(params_path, 'config.yml'), 'w') as f:
    yaml.dump(saved_hyperparams, f)

if args.save_vae and vae is not None:
    print("Saving VAE")
    vae.save(os.path.join(params_path, 'vae'))

if normalize:
    # Unwrap
    if isinstance(env, VecFrameStack):
        env = env.venv
    # Important: save the running average, for testing the agent we need that normalization
    env.save_running_average(params_path)
