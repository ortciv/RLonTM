# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from env import TMSimEnv
from Partition import Partition
from Task import Task
from Generation import Generation
from keras.models import load_model
EPISODES = 1000000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def _build_model2(self):
        # encoder style NN for Deep-Q Learning Model by ort
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = TMSimEnv()
    partition_list = {}
    # for i in range(3):
    #     partition_list[i]=Partition(i, 0.2*(i+1))
    g = Generation()
    partition_list =g.generate_partitions(20)
    print ('Number of partitions:'+str(len(partition_list)))
    env.make(partition_list, 0.6)
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32



    wFile = open('Result_Diff_Task.txt','w')
    for e in range(EPISODES):
        state = env.reset()
        r = 0
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, m = env.step(action)
            #print m
            #print m+' and the reward is: '+str(reward)
            r += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:                
                model_result = env.get_unit_ratio()
                bf_result = env.simulate_best_fit()
                if e % 100 == 0:
                    print("episode: {}/{}, score: {}, e: {:.5}"
                          .format(e, EPISODES, r, agent.epsilon))
                    print ("Unit ratio provided by the model:"+str(model_result))
                    print ("Unit ratio provided by Best-Fit: "+str(bf_result))
                               
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 100 == 0:
            wFile.write(str(model_result)+',' +str(bf_result)+','+str(r)+','+str(agent.epsilon)+'\n')
            wFile.flush()

    wFile.close()

        #     agent.save("./save/cartpole-dqn.h5")
