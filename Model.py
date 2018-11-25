from Task import Task
from Partition import Partition
from Scheduler import Scheduler
import logging
class Model:
	def __init__(self):
		'''
		Args:
			--scheduler: type Scheduler, takes in the scheduler 
			--task_list: type list of Task, takes in the tasks
			--partition_list, type dict of partitions, the index of a partition is its id
			--sch_util, type float, total utilization of tasks scheduled successfully
		'''
		self._scheduler = Scheduler('best_fit')
		self._task_list = []
		self._partition_list = []
		#print type(self._partition_list)
		self._is_schedulable = True
		self._total_util = 0
		self._sch_util = 0
		self._total_num = 0
		self._sch_num = 0
		self._total_val = 0
		self._sch_val = 0
		self._state_now = []

		self._critical_time = []
		self._mapping = {} #mapping is used to record the  map from task to partition
		self._to_leave_tasks = [] #records the tasks that will leave, it should be sorted by the leaving time
		self._task_counter = 0
		self._leaving_counter = 0
		self._time_now = 0 #did not consider the extreme conditions where no tasks are passed.


	def reset(self, task_list, partition_list):
		self.__init__()
		#extract arrival and leaving time and sort them
		self._task_list = task_list
		self._partition_list = partition_list
		for _,p in self._partition_list.items(): #needs to use a way to enumerate the dict!!! partition_list is a list!!
			self._state_now.append(p._af)
			p._af_remain = p._af

		self._state_now.append(0)#the last one is the task passed in.
		for task in self._task_list:
			self._critical_time.append(task._arrival)
			if task._leaving>0:
				self._critical_time.append(task._leaving)
		self._critical_time = list(set(self._critical_time))
		self._critical_time.sort()#get rid of duplicates and sort

		self._task_list.sort(key = lambda x: x._arrival) #sort task_list by the arrival time
		
		for task_now in self._task_list:
			self._total_num += 1
			self._total_val += task_now._value
			self._total_util += task_now._utilization

		#execute to the first place where an action is needed, since no arrival leads to no leaving, so just initialize the counters
		
		self._state_now[len(self._state_now)-1] = self._task_list[self._task_counter]._utilization
		return self._state_now

	def handle_leaving(self):
		while self._leaving_counter< len(self._to_leave_tasks) and self._to_leave_tasks[leaving_counter]._leaving == self._critical_time[self._time_now]:
			task_now = self._to_leave_tasks[self._leaving_counter]
			if task_now._id not in self._mapping:
				print 'weird exception for'
				leaving_counter += 1
				continue
			p_id = self._mapping[task_now._id]
			#print '1: '+str(self._partition_list[p_id]._af_remain)
			self._partition_list[p_id]._af_remain += task_now._utilization
			#print '2: '+str(self._partition_list[p_id]._af_remain)
			#print 'task '+str(task_now._id)+'leaves from '+str(p_id)+'releasing '+str(task_now._utilization)+' at time '+str(t)
			self._leaving_counter += 1

	#accomplish the action, return the reward and continue the actions until next action required.
	def step(self, action):
		task_now = self._task_list[self._task_counter]
		if action<0 and self._scheduler.schedule(task_now, self._partition_list)<0:
			return self._state_now, 1, True, "Ends" #proper ends
		elif action<0 or action > len(self._partition_list):
			return self._state_now, -5, True, "Ends" #illegal inputs

		reward = 0
		if self._partition_list[action]._af_remain < task_now._utilization:
			reward = -2 #should we punish a non-fittable choice?
			message = "A non-fittable choice is made."
			#print message
			return self._state_now, reward, True, message
		else:
			bf_choice = self._scheduler.schedule(task_now, self._partition_list)
			#print str(self._state_now) + ':  '+str(action)
			if bf_choice == action:
				reward = 1 #same value for each task now
			else:
				reward = -1
			message = 'Task assigned to'+str(action)
			self._mapping[task_now._id] = action
			self._partition_list[action]._af_remain -= task_now._utilization
			self._state_now[action] = self._partition_list[action]._af_remain
			if task_now._leaving>0:
				self._to_leave_tasks.append(task_now)
				self._to_leave_tasks.sort(key = lambda x: x._leaving)#append the task to leaving and resort it.
			self._sch_util += task_now._utilization
			self._sch_num += 1
			self._sch_val += task_now._value
		#handle the leaving tasks anyway
		self._task_counter += 1
		#handle the situation where task_counter reaches the end here
		if self._task_counter >=len(self._task_list):
			#print 'Done  '+str(len(self._task_list))
			done = True
			message = 'Ends'
			while self._time_now < len(self._critical_time):
				self.handle_leaving()
				self._time_now += 1
			self._state_now[len(self._state_now)-1] = 0#end the process, set the task utilization as 0
			return self._state_now, reward, done, message
		else:
			while self._task_list[self._task_counter]._arrival > self._critical_time[self._time_now]:
				self.handle_leaving()
				self._time_now += 1
			self._state_now[len(self._state_now)-1] = self._task_list[self._task_counter]._utilization
			done = False
			return self._state_now, reward, done, message




	def run_model(self, log_flag = False):
		#runs the simulation, each leaving time and arrival time will be a critical time spot to make decision or updates
	        '''wFile = open("debug.txt",'a')	
		for _,p in self._partition_list.items():
			wFile.write(str(p._af_remain)+' ')
		wFile.write('\n')
		wFile.flush()'''
		
		#print len(self._partition_list)


		

		if log_flag:
			logging.basicConfig(filename='mapping.log',level=logging.INFO)
		done = False
		while not done:
			p_id = self._scheduler.schedule(self._task_list[self._task_counter], self._partition_list)
			state,reward,done,message = self.step(p_id)
		# task_counter = 0
		# leaving_counter = 0
		# for t in self._critical_time:
		# 	#print 'time now is: '+str(t)
		# 	#print type(self._partition_list)	
		# 	while task_counter< len(self._task_list) and self._task_list[task_counter]._arrival == t:
		# 		task_now = self._task_list[task_counter]
		# 		self._total_util += task_now._utilization
		# 		self._total_num += 1
		# 		self._total_val += task_now._value
		# 		p_id = self._scheduler.schedule(task_now, self._partition_list)#invoke schedulers to schedule the task now, if not allocated successfully, return -1
				
		# 		if p_id>=0:
		# 			self._mapping[task_now._id] = p_id
		# 			#print 'Outside scheduling:'+str(self._partition_list[p_id]._af_remain)
		# 			self._partition_list[p_id]._af_remain -= task_now._utilization #update records and partition states
		# 			#print 'Outside scheduling2:'+str(self._partition_list[p_id]._af_remain)
		# 			#self._partition_list[p_id]._task_num += 1
		# 			if log_flag:
		# 				logging.info('Task '+str(task_now._id)+' is allocated to partition '+ str(p_id)+' at time '+str(t))
		# 			if task_now._leaving>0:
		# 				#print 'Leaving task: '+str(task_now._id)+' in: '+str(task_now._leaving)
		# 				self._to_leave_tasks.append(task_now)
		# 				self._to_leave_tasks.sort(key = lambda x: x._leaving)#append the task to leaving and resort it.
		# 			self._sch_util += task_now._utilization
		# 			self._sch_num += 1
		# 			self._sch_val += task_now._value
		# 		else:
		# 			#if this task cannot be scheduled
		# 			if log_flag:
		# 				logging.info('Task '+str(task_now._id)+' is not schedulable')
		# 			self._is_schedulable = False
		# 		task_counter += 1
		# 	#check the leaving time of tasks, update af_remain in partitions in time
		# 	while leaving_counter< len(self._to_leave_tasks) and self._to_leave_tasks[leaving_counter]._leaving == t:
		# 		task_now = self._to_leave_tasks[leaving_counter]
		# 		if task_now._id not in self._mapping:
		# 			print 'weird exception for'
		# 			leaving_counter += 1
		# 			continue
		# 		p_id = self._mapping[task_now._id]
		# 		#print '1: '+str(self._partition_list[p_id]._af_remain)
		# 		self._partition_list[p_id]._af_remain += task_now._utilization
		# 		#print '2: '+str(self._partition_list[p_id]._af_remain)
		# 		#print 'task '+str(task_now._id)+'leaves from '+str(p_id)+'releasing '+str(task_now._utilization)+' at time '+str(t)
		# 		leaving_counter += 1
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
