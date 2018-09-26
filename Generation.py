from Model import Model
import random
import math
from Task import Task
from Partition import Partition
class Generation:

	def __init__(self):
		self._max_num = 10000
		self._min_num = 10
		self._max_wcet = 100

	def gen_kato_utilizations(self, target_val, min_val, max_val):
		'''
			This function is modified from the function gen_kato_utilizations in class simso.generator.task_generator.
		'''
		vals = []
		total_val = 0
		# Classic UUniFast algorithm:
		while total_val < target_val:
			val = random.uniform(min_val, max_val)
			if val + total_val> target_val:
 				val = target_val - total_val
			total_val += val
			vals.append(val)
		return vals

	def generate_tasks(self, target_util, leave = False, per_ratio = 0.5):
		'''
		 
		 Args:
			  - target_util: Total utilization to reach.
			  - leave: whether tasks leave or not
			  - per_ratio: the ratio of periodic tasks.
		'''
		task_set =[]
		utils = self.gen_kato_utilizations(target_util,0, 1)#generate utilizations based on the number of tasks generated
		num = len(utils)
		for i in range(num):
			util_now = utils[i]
			wcet = random.randint(1, self._max_wcet)
			period = wcet/util_now
			deadline = period
			#arrival = -math.log(1.0 - random.random())
			arrival = random.randint(0, 3000)
			leaving = -1
			if leave:
				dice = random.random()
				if dice>= per_ratio:
					#leaving = -math.log(1.0 - random.random())
					leaving = random.randint(0,3000)
					if arrival+period > leaving:
						leaving = arrival+period
					#print 'arrival: '+str(arrival) +' and leaving: '+str(leaving)
			v = random.randint(1, 5)
			task_now = Task(i, arrival, wcet, deadline, period, v, leaving)
			task_set.append(task_now)
		return task_set
	def generate_partitions(self, target_af):
		'''
		 
		 Args:
			  - `target_af`: Total af of all partitions to reach.
		'''
		partition_set = {}
		afs = self.gen_kato_utilizations(target_af,0, 1)#generate utilizations based on the number of tasks generated
		num = len(afs)
		for i in range(num):
			partition_now = Partition(i, afs[i])#only generates regular partitions
			#print afs[i]
			partition_set[i] = partition_now
		return partition_set
