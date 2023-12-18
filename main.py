import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

class Env(gym.Env):
    def __init__(self):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low = -1, 
            high = 1, 
            shape=(10, ), 
            dtype = np.float64
        )

        self.action_space = gym.spaces.Box(
            low = -1, 
            high = 1, 
            shape=(10, ), 
            dtype = np.float64
        )

    def mae(self, x, y):
        return np.mean(np.abs(x - y))

    def step(self, action):
        self.reward = -1 * self.mae(self.current_state, action)

        obs = self.observation_space.sample()
        self.current_state = obs
        return obs, self.reward, False, False, {}

    def reset(self, seed = None, options = None):
        self.current_state = self.observation_space.sample()
        return self.current_state, {}
    
    def render(self):
        print(f"Reward: {self.reward:.2f}")

if __name__ == '__main__':
    env = Env()

    model = PPO(
        "MlpPolicy", 
        env, 
        use_sde = True,
        tensorboard_log = 'log',
        verbose = 1
    )
    
    model.learn(
        total_timesteps = 1000000
    )

    ## test
    # env = Env()
    # obs, _ = env.reset()
    # act = model.predict(obs, deterministic = True)[0]

    # plt.plot(obs, '-o', label = 'observation')
    # plt.plot(act, '-o', label = 'action')
    # plt.legend()
    # plt.show()