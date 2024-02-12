from gymnasium import spaces
import numpy as np
# a = spaces.MultiDiscrete([2,2])
# valid_samples = [[0,0], [1,0], [0,1], [1,1]]
#
# # this works
# for i in range(100):
#     sample = a.sample()
#     assert list(sample) in valid_samples, "{} is not a valid sample of {}".format(sample, valid_samples)
#
# af = spaces.flatten_space(a)
# # this will fail because a sample will be drawn with a value of 2 in one of the elements
# for i in range(100):
#     flat_sample = af.sample()
#     assert list(flat_sample) in valid_samples, "{} is not a valid flat sample of {}".format(flat_sample, valid_samples)

# maxS, maxT = 6, 6
# action1_space = spaces.Box(low=1, high=maxS, shape=(1,), dtype=np.int32)
# action2_space = spaces.Box(low=1, high=maxT + 3, shape=(1,), dtype=np.int32)  # 6 targets + 3 for D, R, L
# raw_action_space = spaces.Tuple([action1_space, action2_space])  # "<s> <t>", "<s> D", "<s> R", "<s> L"
# action_space = spaces.flatten_space(raw_action_space)
#
# for _ in range(54):
#     action = action_space.sample()
#     a, b = str(action[0]), str(action[1]) if action[1] <= maxT else ["D", "R", "L"][action[1]-maxT-1]
#
#     print(a, b)

# a = np.array([1, 2, 3, 4, 5, 6, -1, -1, -1, -1])
# print(a)
# a[:6] = np.roll(a[:6], 1)
# print(a)

class Config:
    params = {
        "a": 5,
        "b": 10
    }

class Test:
    def __init__(self, a, b):
        self.a = a
        self.b = b

t = Test(**Config.params)
print(t.a, t.b)