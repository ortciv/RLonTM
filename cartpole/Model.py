from Elements import *
from Scheduler import Scheduler
from Generation import Generation
class Model:
	def __init__(self, bin_sum_size, load_ratio):
		'''
		Args:
			bin_set: 			type list of bin, takes in the bins
			item_set: 			type list of items, takes in the items, assume items inside sorted by the arrival time already
			state_now:			type list of float, the state of the system now, includes the capacity of bins and the size of next item
			is_schedulable:		type bool, the state of the system now, showing whether the system misses any deadline or not
			total_item:			type int, number of items processed
			item_success:		type int, number of items successfully mapped
			item_counter:		type int, the index of next item
			load_ratio:			type float, the sum of sizes of items/ the sum of sizes of bins
			bin_sum_size		type float, the sum of sizes of bins
			state_size			type int, the size of the state
			action_size			type int, the range of the action
		'''
		self._load_ratio = load_ratio
		self._bin_sum_size = bin_sum_size
		g = Generation()
		self._bin_set = g.generate_bins(bin_sum_size) #randomly generate a bin_set that does not change during the learning
		self._item_set = []
		self._state_now = []
		self._item_counter = 0
		self._scheduler = Scheduler('best_fit')
		self._action_size = len(self._bin_set)
		self._state_size = self._action_size+1

	def reset(self):
		g = Generation()
		self._item_set = g.generate_items(self._bin_sum_size*self._load_ratio)#reset the item set
		self._state_now = []
		self._item_counter = 0
		for i in range(len(self._bin_set)):
			self._bin_set[i]._capacity = self._bin_set[i]._size
			self._state_now.append(self._bin_set[i]._capacity)
		self._state_now.append(self._item_set[self._item_counter]._size)
		self._item_counter+=1
		return self._state_now

	def step(self, action):
		if action<0 or action > len(self._bin_set):
			return self._state_now, -1, True, "Ends"
		item_now = self._item_set[self._item_counter]
		self._item_counter += 1
		reward = 0
		if self._bin_set[action]._capacity < item_now._size:
			reward = -5
			message = "A non-fittable chocie is made."
			return self._state_now, reward, True, message
		else:
			bf_choice = self._scheduler.schedule(item_now, self._bin_set)
			done = False
			if bf_choice == action:
				reward = 1
			else:
				reward = -1
			message = 'Item assigned to Bin '+str(action)
			self._bin_set[action]._capacity -= item_now._size
			self._state_now[action] -= item_now._size
			if self._item_counter>= len(self._item_set):
				done = True
			else:
				self._state_now[len(self._state_now)-1] = self._item_set[self._item_counter]._size
			return self._state_now, reward, done, message
	def get_state_size(self):
		return self._state_size
	def get_action_size(self):
		return self._action_size
