from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines import SAC
from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy
from stable_baselines.common.policies import register_policy
import tensorflow as tf
import gym

# There already exists an environment generator
# that will make and wrap atari environments correctly
env = gym.make("MountainCarContinuous-v0")

class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                              layers=[32, 32],
                                              act_fun=tf.nn.elu,
                                              feature_extraction="mlp")
register_policy('CustomSACPolicy', CustomSACPolicy)

model = SAC('CustomSACPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

# Close the processes
env.close()