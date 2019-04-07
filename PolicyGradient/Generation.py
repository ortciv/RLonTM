import random
import math
from Elements import Item
from Elements import Bin
class Generation:

	def __init__(self):
		self._max_time = 10000
		self._min_num = 10


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

	def generate_items(self, target_size):
		'''
		 
		 Args:
			  - target_util: Total size of the item set to reach.
		'''
		item_set =[]
		sizes = self.gen_kato_utilizations(target_size,0, 1)#generate utilizations based on the number of tasks generated
		num = len(sizes)
		for i in range(num):
			arrival = random.randint(0, self._max_time)
			item_now = Item(i, arrival, sizes[i])
			item_set.append(item_now)
		item_set = sorted(item_set, key = lambda temp: temp._arrival)
		return item_set
	def generate_bins(self, target_size):
		'''
		 
		 Args:
			  - `target_af`: Total af of all partitions to reach.
		'''
		bin_set = []
		sizes = self.gen_kato_utilizations(target_size,0, 1)#generate utilizations based on the number of tasks generated
		num = len(sizes)
		for i in range(num):
			bin_now = Bin(i, sizes[i])#only generates regular partitions
			#print afs[i]
			bin_set.append(bin_now)
		return bin_set
