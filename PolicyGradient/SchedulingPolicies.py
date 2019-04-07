from Elements import *
class SchedulingPolicies:

	'''
	This class is defined to hold all sorts of scheduling policies invoked by the class Model.
	Policies:
	1. best_fit
	2. worst_fit
	3. first_fit
	'''
	def best_fit(self, item, bin_set):
		#calculate the partition that the task best fits
		#print 'Best_Fit invoked'
		p_id = -1
		closest_gap = -1
		for p in bin_set:
			gap_now = p._capacity - item._size
			if gap_now<0:
				continue
			if closest_gap<0 or gap_now<closest_gap:
				p_id = p._id
				closest_gap = gap_now
		#print p_id	
		#print 'Sum af now: '+str(sum_af)
		if p_id== -1:
			return p_id
		#print 'Inside scheduling for BF: '+str(partition_dict[p_id]._af_remain)
		return p_id 

	'''
	def worst_fit(self, task, partition_dict):
		#return id of the partition that the task fits worst
		p_id = -1
		furthest_gap = -1
		for _, p in partition_dict.items():
				gap_now = p._af_remain - task._utilization
				if gap_now < 0: #task doesn't fit
						continue 
				if furthest_gap < 0 or gap_now > furthest_gap:
						p_id = p._id
						furthest_gap = gap_now
		if p_id== -1:
			return p_id
		#print 'Inside scheduling for BF: '+str(partition_dict[p_id]._af_remain)
		return p_id
	def first_fit(self, task, partition_dict):
		#return id of the first partition that the task fits
		p_id = -1
		for _, p in partition_dict.items():
				gap_now = p._af_remain - task._utilization
				if gap_now < 0: #task doesn't fit
						continue
				else:
						p_id = p._id
						break
		if p_id== -1:
			return p_id
		#print 'Inside scheduling for FF: '+str(partition_dict[p_id]._af_remain)
		return p_id
	def DABF(self, task, partition_dict):
		timeNow = task._arrival#could be different if the simulation of the global queue is added
		if task._leaving < 0:
			smallest = -1
			smallest_id = -1
			temp_partition = {}
			for _,p in partition_dict.items():
				p._AS += p._accomplishment_weight * (timeNow - p._last_time)
				if p._af_remain >= task._utilization and (smallest == -1 or p._AS < smallest):
					temp_partition = {}
					temp_partition[p._id] = p
					smallest = p._AS
					smallest_id = p._id
				elif p._af_remain>= task._utilization and p._AS == smallest:
					temp_partition[p._id] = p
			if len(temp_partition)>0:
				#print len(temp_partition)
				p_id = self.best_fit(task, temp_partition)
				#print 'Inside scheduling for DABF: '+str(partition_dict[p_id]._af_remain)
				#partition_dict[p_id]._af_remain -= task._utilization
				#print 'Inside after:   '+str(partition_dict[p_id]._af_remain)
				return p_id
			else:
				return -1
		else:
			largest = -1
			largest_id = -1
			temp_partition = {}
			for _,p in partition_dict.items():
				p._AS += p._accomplishment_weight * (timeNow - p._last_time)
				if p._af_remain >= task._utilization and (largest == -1 or p._AS > largest):
					temp_partition = {}
					temp_partition[p._id] = p
					largest = p._AS
					largest_id = p._id
				elif p._af_remain>= task._utilization and p._AS == largest:
					temp_partition[p._id] = p
			#handle the picked partition, update af_remain and weight
			if len(temp_partition)>0:
				#print len(temp_partition)
				p_id = self.best_fit(task, temp_partition)
				#print 'Inside scheduling for DABF: '+str(partition_dict[p_id]._af_remain)
				#partition_dict[p_id]._af_remain -= task._utilization
				partition_dict[p_id]._accomplishment_weight += task._utilization/float(task._leaving - task._arrival)
				#print 'Inside after:   '+str(partition_dict[p_id]._af_remain)
				return p_id
			else:
				return -1
		'''
	'''def DABF_test(self, task, partition_dict):
		candidate_set = {}
		closest_gap = -1
		if task._leaving<=0:
			l_c = 0
		else:
			l_c = float(task._leaving)
		for _,p in partition_dict.items():
			if p._af_remain < task._utilization:
				continue
			gap = abs(l_c - p._DS)
			if closest_gap<0 or gap<closest_gap:
				candidate_set = {}
				candidate_set[p._id] = p
				closest_gap = gap
			elif gap==closest_gap:
				candidate_set[p._id] = p
		if len(candidate_set)>0:
			p_id = self.best_fit(task, candidate_set)
			partition_dict[p_id]._DS = (partition_dict[p_id]._DS*partition_dict[p_id]._task_num + l_c)/float(partition_dict[p_id]._task_num + 1)
			partition_dict[p_id]._task_num += 1
			return p_id
		else:
			return -1'''
