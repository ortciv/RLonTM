from Model import Model
from Generation import Generation
import random
import copy
from Scheduler import Scheduler
import numpy as np
class TMSimEnv:
	def __init__(self):
		self._partition_list = []
		self._load_ratio = 0
		self._task_list = []
		self._state_size = 0
		self._action_size = 0
		self._total_af = 0

	#make is the initiliazation given by the user, the partition list is given, if no load ratio is given, then randomly generate one
	def make(self,partitions, load_ratio = -1):
		self._partition_list = partitions
		for _,p in partitions.items():
			self._total_af += p._af
		if load_ratio>=0:
			self._load_ratio = load_ratio
		else:
			self._load_ratio = random.random() #generates a load_ratio smaller than 1, could be 0
		self._state_size = len(partitions)+1
		self._action_size = len(partitions)

	#initialize the simulation
	def reset(self):
		g = Generation()
		self._task_list =g.generate_tasks(self._total_af*self._load_ratio)
		#print 'Number of tasks:'+str(len(self._task_list))
		tempP = copy.deepcopy(self._partition_list)
		tempT = copy.deepcopy(self._task_list)
		self. _model = Model()
		return self._model.reset( tempT, tempP)

	def step(self,action):
		return self._model.step(action)

	def get_state_size(self):
		return self._state_size

	def get_action_size(self):
		return self._action_size
    
	def get_random_action(self):
		return [random.randrange(self._action_size)]

	def get_unit_ratio(self):
		return self._model.get_unit_ratio()
	def get_pro_ratio(self):
		return self._model.get_pro_ratio()
	def get_val_ratio(self):
		return self._model.get_val_ratio()
	def is_schedulable(self):
		return self._model.is_schedulable()
	def simulate_best_fit(self):
		tl = copy.deepcopy(self._task_list)
		pl = copy.deepcopy(self._partition_list)
		m = Model()
		m.reset(tl, pl)
		s = Scheduler('best_fit')
		for task in tl:
			action = s.schedule(task, pl)
			_,_,done,_ = m.step(action)
			if done:
				break
		return m.get_unit_ratio()