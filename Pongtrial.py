import gym

env = gym.make('Pong-v0')

state = env.reset()

print(state)

env.close()
