class Item:

	def __init__(self, id, arrival, size):
		'''
		Args:
			--id: 				type int, the id of the item
			--arrival:			type float, arrival time of a task
			--size:				type float, the size of a task
		'''

		self._id = id
		self._arrival = arrival
		self._size = size
	def print_info(self):
		print("Item #"+str(self._id)+" that arrives at "+str(self._arrival)+" and its size is: "+str(self._size))

class Bin:

	def __init__(self, id, size):
		'''
		Args:
			--id:					type int, the id of the bin
			--capacity:				type float, the capacity of the bin now
			--size:					type float, the initial capacity of the bin without any items
		'''
		self._id = id
		self._size = size
		self._capacity = size

	def print_info(self):
		print("Bin #"+str(self._id)+"'s size is: "+str(self._size)+" and the capacity now is: "+str(self._capacity))
