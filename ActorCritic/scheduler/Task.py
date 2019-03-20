class Task:


	def __init__(self, id, arrival, wcet, deadline, period,value=1, leaving = -1):
		'''
		Args:
			--id:			type string, the id of a task
			--arrival: 	  			type float, arrival time of a task
			--wcet:					type float, worst case execution time
			--deadline: 		type float, the relative deadline of a task
			--period: 			type float, the period of a task
			--leaving:		type float, the leaving time of a task, default to be -1 if not applied
			--value: 		type int, the value of the task set by user
		'''
		self._id = id
		self._arrival = arrival
		self._wcet = wcet
		self._deadline = deadline
		self._period = period
		self._leaving = leaving
		self._utilization = (float)(self._wcet)/(min(self._deadline, self._period))
		self._value = value
