import os
import gym
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


def build_model(learning_rate, input_shape, action_size):
    input_layer = Input(input_shape)

    hidden_layer1 = Dense(512, input_shape=input_shape, activation='relu')(input_layer) #512
    hidden_layer2 = Dense(256, activation='relu')(hidden_layer1) #256
    hidden_layer3 = Dense(64, activation='relu')(hidden_layer2) #64
    output_layer = Dense(action_size, activation='linear')(hidden_layer3)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

    model.summary()
    return model


class Agent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = deque(maxlen=2000)

        self.learning_rate = 0.001 #0.00025
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 500 #1000

        self.model = build_model(self.learning_rate, (self.state_size, ), self.action_size)

    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = np.reshape(state, [1, self.state_size])
            a = self.model.predict(state)
            return np.argmax(a)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        sample_batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, done = zip(*sample_batch)
        states = np.array(states)
        next_states = np.array(next_states)
        
        targets = self.model.predict(states)
        next_targets = self.model.predict(next_states)
        
        for i in range(self.batch_size):
            if done[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_targets[i])

        self.model.fit(states, targets, batch_size=self.batch_size, verbose=0)


class CartPole:

    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.episodes = 1000
        self.agent = Agent(self.state_size, self.action_size)

    def load(self, name):
        self.agent.model = load_model(name)

    def save(self, name):
        self.agent.model.save(name)

    def run(self):
        for e in range(self.episodes):
            state = self.env.reset()
            done = False
            for i in range(1, 502):
                self.env.render()
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print(f"episode: {e}/{self.episodes},score: {i}, e: {self.agent.epsilon:.2}")
                    if i == 500:
                        print("Successfully completed")
                        self.save("cartpole-dqn.h5")
                        return
                    break
                self.agent.replay()

    def test(self):
        self.load("cartpole-dqn.h5")
        for e in range(3):#self.episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, info = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
            else:
                print(f"episode: {e}/{self.episodes},score: {i}")
        
def main():
    cartpole = CartPole()
    cartpole.run()
    #cartpole.test()

if __name__ == "__main__":
    main()
