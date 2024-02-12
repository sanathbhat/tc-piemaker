import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config.dqn_agent_config import PieMakerDQNConfig

class PieMakerDQN:
    TRAINING_EPISODES = 1
    TRAINING_EPISODE_STEPS = 1000
    def __init__(self):
        self.env = gym.make("PieMakerEnv-v0")
        self.vec_env = DummyVecEnv([lambda: self.env])
        self.hyper_params = PieMakerDQNConfig.hyper_parameters
        self.model = DQN(env=self.vec_env, verbose=1, **self.hyper_params)

    def train(self, episodes=TRAINING_EPISODES, episode_steps=TRAINING_EPISODE_STEPS):
        self.model.learn(total_timesteps=episodes*episode_steps, log_interval=4)
        self.model.save("models/piemaker_dqn")

    def test(self):
        self.model.load("models/piemaker_dqn")
        total_reward = 0.0
        obs, info = self.env.reset()
        print(info)
        while True:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            print(info)

            if terminated or truncated:
                return total_reward

if __name__ == '__main__':
    agent = PieMakerDQN()
    epochs = 10

    for _ in range(epochs):
        agent.train()
