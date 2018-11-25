"""
solving pendulum using actor-critic model
"""

import gym
import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf

import random
from collections import deque
from env import TMSimEnv
from Generation import Generation
# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
class ActorCritic:
	def __init__(self, env, sess):
		self.env  = env
		self.sess = sess

		self.learning_rate = 0.001
		self.epsilon = 1.0
		self.epsilon_decay = .995
		self.gamma = .95
		self.tau   = .125
		self.state_size = env.get_state_size()
		self.action_size = 1

		# ===================================================================== #
		#                               Actor Model                             #
		# Chain rule: find the gradient of chaging the actor network params in  #
		# getting closest to the final value network predictions, i.e. de/dA    #
		# Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
		# ===================================================================== #

		self.memory = deque(maxlen=2000)#dequeue is a data structure that can be extended from both sides
		self.actor_state_input, self.actor_model = self.create_actor_model()
		_, self.target_actor_model = self.create_actor_model()

		self.actor_critic_grad = tf.placeholder(tf.float32, 
			[None, self.action_size]) # where we will feed de/dC (from critic)
		
		actor_model_weights = self.actor_model.trainable_weights
		self.actor_grads = tf.gradients(self.actor_model.output, 
			actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
		grads = zip(self.actor_grads, actor_model_weights)
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

		# ===================================================================== #
		#                              Critic Model                             #
		# ===================================================================== #		

		self.critic_state_input, self.critic_action_input, \
			self.critic_model = self.create_critic_model()
		_, _, self.target_critic_model = self.create_critic_model()

		self.critic_grads = tf.gradients(self.critic_model.output, 
			self.critic_action_input) # where we calcaulte de/dC for feeding above
		
		# Initialize for later gradient calculations
		self.sess.run(tf.initialize_all_variables())

	# ========================================================================= #
	#                              Model Definitions                            #
	# ========================================================================= #

	def create_actor_model(self): #change the model here, init can stay
		state_input = Input(shape=(self.state_size,)) #define the input shape
		h1 = Dense(24, activation='relu')(state_input)#Input and Dense are Keras.Layers
		h2 = Dense(48, activation='relu')(h1)
		h3 = Dense(24, activation='relu')(h2)
		output = Dense(self.action_size, activation='relu')(h3)
		
		model = Model(input=state_input, output=output)
		adam  = Adam(lr=0.001)
		model.compile(loss="mse", optimizer=adam)
		return state_input, model

	def create_critic_model(self): #change here as well
		state_input = Input(shape=(self.state_size,))#Input and Dense are Keras.Layers
		state_h1 = Dense(24, activation='relu')(state_input)
		state_h2 = Dense(48)(state_h1)
		
		action_input = Input(shape=(self.action_size,))
		action_h1    = Dense(48)(action_input)
		
		merged    = Add()([state_h2, action_h1])#merge state_h2 and action_h1 as a new layer
		merged_h1 = Dense(24, activation='relu')(merged)
		output = Dense(1, activation='relu')(merged_h1)
		model  = Model(input=[state_input,action_input], output=output)
		
		adam  = Adam(lr=0.001)#Adam is self-adaptive,can fit better
		model.compile(loss="mse", optimizer=adam)
		return state_input, action_input, model

	# ========================================================================= #
	#                               Model Training                              #
	# ========================================================================= #

	def remember(self, cur_state, action, reward, new_state, done):
		self.memory.append([cur_state, action, reward, new_state, done])

	def _train_actor(self, samples): 
		for sample in samples:
			cur_state, action, reward, new_state, _ = sample
			predicted_action = self.actor_model.predict(cur_state)
			grads = self.sess.run(self.critic_grads, feed_dict={
				self.critic_state_input:  cur_state,
				self.critic_action_input: predicted_action
			})[0] #just like training a hash_map, returns the value of attr (mapping gradient)

			self.sess.run(self.optimize, feed_dict={
				self.actor_state_input: cur_state,
				self.actor_critic_grad: grads
			})
            
	def _train_critic(self, samples):
		for sample in samples:
			cur_state, action, reward, new_state, done = sample
			if not done:
				target_action = self.target_actor_model.predict(new_state)
				future_reward = self.target_critic_model.predict(
					[new_state, target_action])[0][0]
				reward += self.gamma * future_reward
			#cur_state = np.reshape(cur_state,(1,self.state_size))
			#action = np.array([action])
			#action = np.atleast_2d(action)
			#action = np.reshape(action,(self.action_size,))
			cur_state = np.array(cur_state)
			reward = np.array(reward)
			reward = np.reshape(reward, (1,))
			self.critic_model.fit([cur_state, action], reward, verbose=0)
		
	def train(self): #how to train
		batch_size = 32
		if len(self.memory) < batch_size: #make sure it wont trespass the memory
			return

		rewards = []
		samples = random.sample(self.memory, batch_size)#obtain batch_size-long pieces from the memory
		self._train_critic(samples)
		self._train_actor(samples)

	# ========================================================================= #
	#                         Target Model Updating                             #
	# ========================================================================= #

	def _update_actor_target(self):
		actor_model_weights  = self.actor_model.get_weights()
		actor_target_weights = self.target_critic_model.get_weights()
		
		for i in range(len(actor_target_weights)):
			actor_target_weights[i] = actor_model_weights[i]
		self.target_critic_model.set_weights(actor_target_weights)

	def _update_critic_target(self):
		critic_model_weights  = self.critic_model.get_weights()
		critic_target_weights = self.critic_target_model.get_weights()
		
		for i in range(len(critic_target_weights)):
			critic_target_weights[i] = critic_model_weights[i]
		self.critic_target_model.set_weights(critic_target_weights)		

	def update_target(self):
		self._update_actor_target()
		self._update_critic_target()

	# ========================================================================= #
	#                              Model Predictions                            #
	# ========================================================================= #

	def act(self, cur_state):
		self.epsilon *= self.epsilon_decay
		if np.random.random() < self.epsilon:
			return self.env.action_sample()
		return self.actor_model.predict(cur_state)

def main():
	sess = tf.Session()
	K.set_session(sess)
	env = TMSimEnv()
	g = Generation()
	partition_list = g.generate_partitions(20)
	env.make(partition_list, 0.6)
	actor_critic = ActorCritic(env, sess)

	num_trials = 10000
	trial_len  = 500

	cur_state = env.reset()
	#action = env.action_sample()
	counter = 1
	wFile = open("Actor_critic_Model.txt", "w")
	state_size = env.get_state_size()
	action_size = env.get_action_size()
	while True:
		cur_state = env.reset()
		cur_state = np.reshape(cur_state, [1, state_size])
		total_reward = 0
		for time in range(500):
			action = actor_critic.act(cur_state)
			#action = np.reshape(action, [1, action_size])

			new_state, reward, done, _ = env.step(action)
			total_reward += reward
			new_state = np.reshape(new_state, [1, state_size])

			actor_critic.remember(cur_state, action, reward, new_state, done)
			actor_critic.train()
			cur_state = new_state
			if done:
				model_result = env.get_unit_ratio()
				bf_result = env.simulate_best_fit()
				output = str(counter)+', '+str(total_reward)+', '+str(model_result)+', '+str(bf_result)+', '+str(actor_critic.epsilon)
				print output
				wFile.write(output+'\n')
				wFile.flush()
				counter += 1
				break

	wFile.close()
if __name__ == "__main__":
	main()