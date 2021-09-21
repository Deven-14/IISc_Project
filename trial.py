import gym
from collections import deque
import random
import numpy as np

env = gym.make('CartPole-v1')

memory = deque(maxlen=200)

state = env.reset()
state = state[:]
action = random.randrange(env.action_space.n)
next_state, reward, done, info = env.step(action)
memory.append((state, action, reward, next_state, done))

state = env.reset()
state = np.reshape(state, -1)
action = random.randrange(env.action_space.n)
next_state, reward, done, info = env.step(action)
memory.append((state, action, reward, next_state, done))

f = random.sample(memory, 2)

a, b, c, d, e = zip(*f)

print(a)# b, c, d, e, sep='\n')



print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
print(np.zeros((2, env.action_space.n)))

print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
print(type(np.zeros((2, 4))), type(state))

state = np.zeros((2, 4))
next_state = np.zeros((2, 4))
action, reward, done = [], [], []

# do this before prediction
# for speedup, this could be done on the tensor level
# but easier to understand using a loop
for i in range(2):
    state[i] = f[i][0]
    action.append(f[i][1])
    reward.append(f[i][2])
    next_state[i] = f[i][3]
    done.append(f[i][4])

print(state)

env.close()

a = np.array([1, 2, 3])
print(a)
b = np.array(state)
print(b)
