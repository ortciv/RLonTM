from Task import Task
from Partition import Partition
from Scheduler import Scheduler
import logging
class Model:
	def __init__(self, scheduler, task_list, partition_list):
		'''
		Args:
			--scheduler: type Scheduler, takes in the scheduler 
			--task_list: type list of Task, takes in the tasks
			--partition_list, type dict of partitions, the index of a partition is its id
			--sch_util, type float, total utilization of tasks scheduled successfully
		'''
		self._scheduler = scheduler
		self._task_list = task_list
		self._partition_list = partition_list
		#print type(self._partition_list)
		self._is_schedulable = True
		self._total_util = 0
		self._sch_util = 0
		self._total_num = 0
		self._sch_num = 0
		self._total_val = 0
		self._sch_val = 0
	def run_model(self, log_flag = False):
		#runs the simulation, each leaving time and arrival time will be a critical time spot to make decision or updates
	        '''wFile = open("debug.txt",'a')	
		for _,p in self._partition_list.items():
			wFile.write(str(p._af_remain)+' ')
		wFile.write('\n')
		wFile.flush()'''
		critical_time = []
		#print len(self._partition_list)
		#extract arrival and leaving time and sort them
		for task in self._task_list:
			critical_time.append(task._arrival)
			if task._leaving>0:
				critical_time.append(task._leaving)
		critical_time = list(set(critical_time))
		critical_time.sort()#get rid of duplicates and sort

		mapping = {} #mapping is used to record the  map from task to partition
		to_leave_tasks = [] #records the tasks that will leave, it should be sorted by the leaving time
		self._task_list.sort(key = lambda x: x._arrival) #sort task_list by the arrival time
		if log_flag:
			logging.basicConfig(filename='mapping.log',level=logging.INFO)
		task_counter = 0
		leaving_counter = 0
		for t in critical_time:
			#print 'time now is: '+str(t)
			#print type(self._partition_list)	
			while task_counter< len(self._task_list) and self._task_list[task_counter]._arrival == t:
				task_now = self._task_list[task_counter]
				self._total_util += task_now._utilization
				self._total_num += 1
				self._total_val += task_now._value
				p_id = self._scheduler.schedule(task_now, self._partition_list)#invoke schedulers to schedule the task now, if not allocated successfully, return -1
				
				if p_id>=0:
					mapping[task_now._id] = p_id
					#print 'Outside scheduling:'+str(self._partition_list[p_id]._af_remain)
					self._partition_list[p_id]._af_remain -= task_now._utilization #update records and partition states
					#print 'Outside scheduling2:'+str(self._partition_list[p_id]._af_remain)
					#self._partition_list[p_id]._task_num += 1
					if log_flag:
						logging.info('Task '+str(task_now._id)+' is allocated to partition '+ str(p_id)+' at time '+str(t))
					if task_now._leaving>0:
						#print 'Leaving task: '+str(task_now._id)+' in: '+str(task_now._leaving)
						to_leave_tasks.append(task_now)
						to_leave_tasks.sort(key = lambda x: x._leaving)#append the task to leaving and resort it.
					self._sch_util += task_now._utilization
					self._sch_num += 1
					self._sch_val += task_now._value
				else:
					#if this task cannot be scheduled
					if log_flag:
						logging.info('Task '+str(task_now._id)+' is not schedulable')
					self._is_schedulable = False
				task_counter += 1
			#check the leaving time of tasks, update af_remain in partitions in time
			while leaving_counter< len(to_leave_tasks) and to_leave_tasks[leaving_counter]._leaving == t:
				task_now = to_leave_tasks[leaving_counter]
				if task_now._id not in mapping:
					print 'weird exception for'
					leaving_counter += 1
					continue
				p_id = mapping[task_now._id]
				#print '1: '+str(self._partition_list[p_id]._af_remain)
				self._partition_list[p_id]._af_remain += task_now._utilization
				#print '2: '+str(self._partition_list[p_id]._af_remain)
				#print 'task '+str(task_now._id)+'leaves from '+str(p_id)+'releasing '+str(task_now._utilization)+' at time '+str(t)
				leaving_counter += 1
	def is_schedulable(self):
		#returns whether last simulation is schedulable or not
		return self._is_schedulable
	def get_pro_ratio(self):
		#print 'Total utilization is:'+str(self._total_util)
		if self._total_util == 0:
			return 0
		return (float)(self._sch_util)/(self._total_util)
	def get_unit_ratio(self):
		if self._total_num == 0:
			return 0
		return float(self._sch_num)/(self._total_num)
	def get_val_ratio(self):
		if self._total_val == 0:
			return 0
		return float(self._sch_val)/(self._total_val)
'''
#test code

task_list = []
t1 = Task(1,1,2,3,4,-1)
t2 = Task(2,5,2,7,9,-1)
t3 = Task(3,6,3,9,9, -1)
task_list.append(t1)
task_list.append(t2)
task_list.append(t3)
p1 = Partition(1,0.5)
p2 = Partition(2,1)
partition_list = {}
partition_list[1] = p1
partition_list[2] = p2
scheduler = Scheduler('best_fit')
model = Model(scheduler, task_list, partition_list)
model.run_model()
'''
