from SchedulingPolicies import SchedulingPolicies
import logging

def a(k):
	print(k)
class Scheduler:

	def __init__(self, scheduler_name):
		'''
		Args:
			--scheduler_name: type string, the name of scheduling algorithm
		'''
		self._scheduler_name = scheduler_name
		self._scheduling_policies = SchedulingPolicies()

	def schedule(self, task, partition_list):
		#eval(self._scheduler_name)(0) #try getattr with class name passed in as well

		if  callable(getattr(self._scheduling_policies, self._scheduler_name)):
			#The input is not in right form check the code in Model.py and SchedulingPolicies.py the type of the partition_list is somehow changed from dict to list
			#print type(task)
			#print type(partition_list)
			result = getattr(self._scheduling_policies, self._scheduler_name)(task, partition_list)
			return result
		else:
			logging.warning('No such scheduler!' + self._scheduler_name)
'''
#test code
s = Scheduler('best_fit')
t = Task(1,2,3,5,5)
p1 = Partition(2, 0.7)
p2 = Partition(1, 0.6)
partition_list = []

partition_list.append(p1)
partition_list.append(p2)
print(s.schedule(t, partition_list))
'''