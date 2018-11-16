from Model import Model
import random
class TMSimEnv:
	def __init__():
		self._partition_list = []
		self._load_ratio = 0
		self._task_list = []
		self._state_size = 0
		self._action_size = 0

	#make is the initiliazation given by the user, the partition list is given, if no load ratio is given, then randomly generate one
	def make(partitions, load_ratio = -1):
		self._partition_list = partitions
		if load_ratio>=0:
			self._load_ratio = load_ratio
		else:
			self._load_ratio = random.random() #generates a load_ratio smaller than 1, could be 0
		self._state_size = len(partitions)+1
		self._action_size = len(partitions)
	#initialize the simulation
	def reset():
		g = Generation()
		state = []
		total_af = 0
		for p in self._partition_list:
			total_af += p._af
			p._af_remain = p._af
			state.append(p._af_remain)
		self._task_list = g.generate_tasks(total_af*self._load_ratio)
		return state
