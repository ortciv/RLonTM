"""
Policy Gradient, Reinforcement Learning.

The cart pole example

Using:
Tensorflow: 1.0
gym: 0.8.0
python: python3
"""

from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

from Model import *

DISPLAY_REWARD_THRESHOLD = 400

env = Model(20, 0.6)


n_actions = env.get_action_size()
n_features = env.get_state_size()

RL = PolicyGradient(
    n_actions=n_actions,
    n_features = n_features,
    learning_rate = 0.02,
    reward_decay = 0.99
)

for i_episode in range(3000):
    observation = env.reset()

    while True:
        action = RL.choose_action(observation)

        observation_,reward,done,info = env.step(action)

        RL.store_transition(observation,action,reward)


        if done:
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()

            #if i_episode == 0:
            #    plt.plot(vt)    # plot the episode vt
            #    plt.xlabel('episode steps')
            #    plt.ylabel('normalized state-action value')
            #    plt.show()
            break

        observation = observation_

