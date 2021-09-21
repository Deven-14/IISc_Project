import gym

env = gym.make('CartPole-v0')
env.reset()

done = False

while not done:
    env.render()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

env.close()
