import sys

import gymnasium as gym
from gymnasium import spaces, register
import numpy as np


class PieMakerEnv(gym.Env):
    ILLEGAL_ACTION_REWARD = -100
    DISCARD_REWARD = 0
    ROTATE_REWARD = 0
    TRANSFER_BASE_REWARD = 0  # additional fixed reward for successful transfers

    def __init__(self):
        super(PieMakerEnv, self).__init__()
        # global limits for space definitions
        self.maxS = 6
        self.maxT = 6
        self.maxC = 6
        self.maxN = 10

        # Run params: will be initialized in reset()
        self.T = 0  # Number of target plates
        self.S = 0  # Number of source plates
        self.C = 0  # Number of pie flavours
        self.N = 0  # Number of pieces in each pie
        self.P = 0  # Plate fill ratio
        self.sources = np.array([[-1] * self.maxN for _ in range(self.maxS)])
        self.targets = np.array([[-1] * self.maxN for _ in range(self.maxT)])
        self.meta_params = None

        # Action spaces
        self.action1_space = spaces.Box(low=1, high=self.maxS, shape=(1,), dtype=np.int32)
        self.action2_space = spaces.Box(low=1, high=self.maxT + 3, shape=(1,), dtype=np.int32)  # 6 targets + D + R + L
        self.raw_action_space = spaces.Tuple(
            [self.action1_space, self.action2_space])  # "<s> <t>", "<s> D", "<s> R", "<s> L"

        # Observation space
        self.source_obs = spaces.Box(low=-1, high=self.maxC, shape=(self.maxS, self.maxN), dtype=np.int32)
        self.target_obs = spaces.Box(low=-1, high=self.maxC, shape=(self.maxT, self.maxN), dtype=np.int32)
        self.s_obs = spaces.Box(low=1, high=self.maxS, shape=(1,), dtype=np.int32)
        self.t_obs = spaces.Box(low=1, high=self.maxT, shape=(1,), dtype=np.int32)
        self.c_obs = spaces.Box(low=1, high=self.maxC, shape=(1,), dtype=np.int32)
        self.n_obs = spaces.Box(low=4, high=self.maxN, shape=(1,), dtype=np.int32)
        self.pfr_obs = spaces.Box(low=0.05, high=0.3, shape=(1,), dtype=np.float32)
        self.raw_observation_space = spaces.Tuple(
            [self.source_obs, self.target_obs, self.s_obs, self.t_obs, self.c_obs, self.n_obs, self.pfr_obs])

        # Flattened Action, Observation spaces that will be used in RL
        self.action_space = spaces.flatten_space(self.raw_action_space)
        self.observation_space = spaces.flatten_space(self.raw_observation_space)

        # print("Flattened action space dimensions = ", spaces.flatdim(self.raw_action_space))
        # print("Flattened observation space dimensions = ", spaces.flatdim(self.raw_observation_space))

        self.state = None
        self.target_flavor = None  # target plate flavors: [1, self.C] = Single flavor, -1 = Mixed flavors, 0 = Empty
        self.turns = 0

    def step(self, action):
        # Returns observation, reward, terminated, truncated, info
        # Implement the logic to handle an action
        self.turns += 1
        # Update the state, calculate the reward, and check if the episode is done
        reward = 0

        # translate and output to console for tester interaction
        non_target_a2s = ["D", "R", "L"]
        a1, a2 = str(action[0]), str(action[1]) if action[1] <= self.maxT else non_target_a2s[action[1] - self.maxT - 1]
        print(f"{a1} {a2}")
        sys.stdout.flush()

        # Code here to update self.state, reward based on the action
        s = action[0] - 1  # action[0] ~ [1,6] but s: [0,5]

        if s >= self.S:
            return self.state, PieMakerEnv.ILLEGAL_ACTION_REWARD, False, True, {"TotalTurns": self.turns,
                                                                                "Status": f"Truncated: Invalid source plate {s}"}
        # Action: s D
        if a2 == "D":
            reward += PieMakerEnv.DISCARD_REWARD
        # Action: s R
        elif a2 == "R":
            self.sources[s][:self.N] = np.roll(self.sources[s][:self.N], -1)
        # Action: s L
        elif a2 == "L":
            self.sources[s][:self.N] = np.roll(self.sources[s][:self.N], 1)
        # Action: s t
        else:
            t = action[1] - 1
            if t >= self.T:
                return self.state, PieMakerEnv.ILLEGAL_ACTION_REWARD, False, True, {"TotalTurns": self.turns,
                                                                                    "Status": f"Truncated: Invalid target plate {t}"}
            if not ((can_transfer := self._is_valid_transfer(s, t))[0]):
                return self.state, PieMakerEnv.ILLEGAL_ACTION_REWARD, False, True, {"TotalTurns": self.turns,
                                                                                    "Status": f"Truncated: Invalid transfer: {s} {t}"}

            reward += can_transfer[1]

            # empty target plate
            for n in range(self.N):
                self.targets[t][n] = 0

        # get new source plate s if previous was discarded or transferred
        if a2 not in ("R", "L"):
            new_source = input()
            for n in range(self.N):
                self.sources[s][n] = int(new_source[n])

        self.state = np.concatenate((self.sources.flatten(), self.targets.flatten(), self.meta_params))
        done = self.turns < 1000
        return self.state, reward, done, False, {"TotalTurns": self.turns, "Status": "Terminated: Reached 1000 turns"}

    def reset(self, *, seed=None, options=None):
        # Reset the environment state and return the initial observation and info
        # Run params
        self.T = int(input())
        self.S = int(input())
        self.C = int(input())
        self.N = int(input())
        self.P = float(input())

        # Create source, target with max size. Leave padding with -1 beyond current S, T, N limits
        for s in range(self.S):
            source_str = input()
            for n in range(self.N):
                self.sources[s][n] = int(source_str[s][n])

        for t in range(self.T):
            for n in range(self.N):
                self.targets[t][n] = 0

        self.target_flavor = [0] * self.T

        self.meta_params = np.array([self.S, self.T, self.C, self.N, self.P])

        self.state = np.concatenate((self.sources.flatten(), self.targets.flatten(), self.meta_params))
        return self.state, {
            "StartState": {"sources": self.sources, "targets": self.targets, "S": self.S, "T": self.T, "C": self.C,
                           "N": self.N, "P": self.P}}

    def close(self):
        # Send terminate action to console
        print("-1")
        sys.stdout.flush()
        pass

    def render(self):
        raise NotImplementedError

    '''
    Checks if source plate s can be transferred to target plate t. If transfer is valid, then returns the effective 
    reward obtained from the transfer. Also updates target flavor.
    :returns bool, float
    '''

    def _is_valid_transfer(self, s, t):
        source_plate, target_plate = self.sources[s], self.targets[t]
        k = 0
        target_count = 0
        for i in range(self.N):
            if target_plate[i] > 0:
                target_count += 1

            if source_plate[i] > 0:
                if target_plate[i] > 0:
                    return False, -1
                k += 1
                target_count += 1

                if self.target_flavor[t] == 0:  # if target fully empty
                    self.target_flavor[t] = source_plate[i]
                elif self.target_flavor[t] > 0 and self.target_flavor[t] != source_plate[i]:
                    # if target has single flavor and is different from current source pie
                    self.target_flavor[t] = -1

        reward = k + 0.0
        if target_count == self.N:
            if self.target_flavor[t] > 0:
                reward += self.N ** 2
            else:
                reward += self.N

        return True, reward

    # def _init_state(self, source_plates_strs):
    #     # Initialize state
    #     source_plates_init = [[-1] * self.maxN for _ in range(self.maxS)]
    #     for s in range(self.S):
    #         for n in range(self.N):
    #             source_plates_init[s][n] = int(source_plates_strs[s][n])
    #
    #     target_plates_init = [[-1] * self.maxN for _ in range(self.maxT)]
    #     for t in range(self.T):
    #         for n in range(self.N):
    #             target_plates_init[t][n] = 0
    #
    #     meta_params_init = [self.S, self.T, self.C, self.N - 3, self.P]
    #
    #     self.state = np.concatenate((np.array(source_plates_init).flatten(),
    #                                  np.array(target_plates_init).flatten(),
    #                                  np.array(meta_params_init)))
    #
    #     assert self.observation_space.contains(self.state)
    #     print("PieMaker Environment initialized successfully")


# Env registration
register(
    id='PieMakerEnv-v0',
    entry_point='piemaker:PieMakerEnv',
)

# # Main training loop
# for episode in range(num_episodes):
#     total_reward = 0
#     done = False
#
#     while not done:
#         action = model.predict_action(state)  # Model decides the action
#         next_state, reward, done = env.step(action)  # Execute the action in the environment
#         model.update(state, action, reward, next_state, done)  # Update the model
#         state = next_state
#         total_reward += reward
#
#     print(f"Episode {episode}: Total Reward: {total_reward}")
