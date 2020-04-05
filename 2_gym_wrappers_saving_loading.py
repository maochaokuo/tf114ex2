import gym
from stable_baselines import A2C, SAC, PPO2, TD3

import os

# Create save dir
save_dir = "/tmp/gym/"
os.makedirs(save_dir, exist_ok=True)

model = PPO2('MlpPolicy', 'Pendulum-v0', verbose=0).learn(8000)
# The model will be saved under PPO2_tutorial.zip
model.save(save_dir + "/PPO2_tutorial")

# sample an observation from the environment
obs = model.env.observation_space.sample()

# Check prediction before saving
print("pre saved", model.predict(obs, deterministic=True))

del model # delete trained model to demonstrate loading

loaded_model = PPO2.load(save_dir + "/PPO2_tutorial")
# Check that the prediction is the same after loading (for the same observation)
print("loaded", loaded_model.predict(obs, deterministic=True))

import os
from stable_baselines.common.vec_env import DummyVecEnv

# Create save dir
save_dir = "/tmp/gym/"
os.makedirs(save_dir, exist_ok=True)

model = A2C('MlpPolicy', 'Pendulum-v0', verbose=0, gamma=0.9, n_steps=20).learn(8000)
# The model will be saved under A2C_tutorial.zip
model.save(save_dir + "/A2C_tutorial")

del model # delete trained model to demonstrate loading

# load the model, and when loading set verbose to 1
loaded_model = A2C.load(save_dir + "/A2C_tutorial", verbose=1)

# show the save hyperparameters
print("loaded:", "gamma =", loaded_model.gamma, "n_steps =", loaded_model.n_steps)

# as the environment is not serializable, we need to set a new instance of the environment
loaded_model.set_env(DummyVecEnv([lambda: gym.make('Pendulum-v0')]))
# and continue training
loaded_model.learn(8000)


class CustomWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super(CustomWrapper, self).__init__(env)

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        return obs

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info


class TimeLimitWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    :param max_steps: (int) Max number of steps per episode
    """

    def __init__(self, env, max_steps=100):
        # Call the parent constructor, so we can access self.env later
        super(TimeLimitWrapper, self).__init__(env)
        self.max_steps = max_steps
        # Counter of steps per episode
        self.current_step = 0

    def reset(self):
        """
        Reset the environment
        """
        # Reset the counter
        self.current_step = 0
        return self.env.reset()

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        self.current_step += 1
        obs, reward, done, info = self.env.step(action)
        # Overwrite the done signal when
        if self.current_step >= self.max_steps:
            done = True
            # Update the info dict to signal that the limit was exceeded
            info['time_limit_reached'] = True
        return obs, reward, done, info

from gym.envs.classic_control.pendulum import PendulumEnv

# Here we create the environment directly because gym.make() already wrap the environement in a TimeLimit wrapper otherwise
env = PendulumEnv()
# Wrap the environment
env = TimeLimitWrapper(env, max_steps=100)

obs = env.reset()
done = False
n_steps = 0
while not done:
  # Take random actions
  random_action = env.action_space.sample()
  obs, reward, done, info = env.step(random_action)
  n_steps += 1

print(n_steps, info)

import numpy as np


class NormalizeActionWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Retrieve the action space
        action_space = env.action_space
        assert isinstance(action_space,
                          gym.spaces.Box), "This wrapper only works with continuous action space (spaces.Box)"
        # Retrieve the max/min values
        self.low, self.high = action_space.low, action_space.high

        # We modify the action space, so all actions will lie in [-1, 1]
        env.action_space = gym.spaces.Box(low=-1, high=1, shape=action_space.shape, dtype=np.float32)

        # Call the parent constructor, so we can access self.env later
        super(NormalizeActionWrapper, self).__init__(env)

    def rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))

    def reset(self):
        """
        Reset the environment
        """
        # Reset the counter
        return self.env.reset()

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        # Rescale action from [-1, 1] to original [low, high] interval
        rescaled_action = self.rescale_action(action)
        obs, reward, done, info = self.env.step(rescaled_action)
        return obs, reward, done, info

original_env = gym.make("Pendulum-v0")

print(original_env.action_space.low)
for _ in range(10):
  print(original_env.action_space.sample())

env = NormalizeActionWrapper(gym.make("Pendulum-v0"))

print(env.action_space.low)

for _ in range(10):
  print(env.action_space.sample())

from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv

env = Monitor(gym.make('Pendulum-v0'), filename=None, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

model = A2C("MlpPolicy", env, verbose=1).learn(int(1000))

normalized_env = Monitor(gym.make('Pendulum-v0'), filename=None, allow_early_resets=True)
# Note that we can use multiple wrappers
normalized_env = NormalizeActionWrapper(normalized_env)
normalized_env = DummyVecEnv([lambda: normalized_env])

model_2 = A2C("MlpPolicy", normalized_env, verbose=1).learn(int(1000))

from stable_baselines.common.vec_env import VecNormalize, VecFrameStack

env = DummyVecEnv([lambda: gym.make("Pendulum-v0")])
normalized_vec_env = VecNormalize(env)

obs = normalized_vec_env.reset()
for _ in range(10):
  action = [normalized_vec_env.action_space.sample()]
  obs, reward, _, _ = normalized_vec_env.step(action)
  print(obs, reward)


class MyMonitorWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super(MyMonitorWrapper, self).__init__(env)
        # === YOUR CODE HERE ===#
        # Initialize the variables that will be used
        # to store the episode length and episode reward

        # ====================== #

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        # === YOUR CODE HERE ===#
        # Reset the variables

        # ====================== #
        return obs

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        obs, reward, done, info = self.env.step(action)
        # === YOUR CODE HERE ===#
        # Update the current episode reward and episode length

        # ====================== #

        #if done:
        # === YOUR CODE HERE ===#
        # Store the episode length and episode reward in the info dict

        # ====================== #
        return obs, reward, done, info

# To use LunarLander, you need to install box2d box2d-kengz (pip) and swig (apt-get)
#!pip install box2d box2d-kengz

env = gym.make("LunarLander-v2")
# === YOUR CODE HERE ===#
# Wrap the environment

# Reset the environment

# Take random actions in the enviromnent and check
# that it returns the correct values after the end of each episode

# ====================== #

