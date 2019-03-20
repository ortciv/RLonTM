class Partition:

	def __init__(self, id, af, regularity = 1):
		'''
		Args;
			--id:			type string, the id of the partition
			--af: 			type float, availability factor of the partition
			--regularity:	type int, the regularity of the partition
		'''
		self._id = id
		self._af = af
		self._regularity = regularity
		self._af_remain = self._af
		self._accomplishment_weight = 0
		self._last_time = 0
		self._AS = 0
