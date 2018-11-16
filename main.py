from Generation import Generation
from Scheduler import Scheduler
from Model import Model
from Task import Task
from Partition import Partition
import copy
def simulate(total_density, total_af, wFile):

	repeat_times = 10000
	#repeat_times = 1	
	schedulers = []

	schedulers.append(Scheduler('best_fit'))
	schedulers.append(Scheduler('worst_fit'))
	schedulers.append(Scheduler('first_fit'))
	num_ratio = [0 for k in range(len(schedulers))]
	util_ratio= [0 for k in range(len(schedulers))]
	val_ratio= [0 for k in range(len(schedulers))]
	schedulability = [0 for k in range(len(schedulers))]

	for i in range(repeat_times):
		if i%100000==0:
			print '100000 repeats made'
		g = Generation()
#		task_list = g.generate_tasks(total_density, True)
		task_list = g.generate_tasks(total_density, False)
		partition_list = g.generate_partitions(total_af)
		for policy in range(len(schedulers)):
			#initialize partitions
			for _, p in partition_list.items():
				p._af_remain = p._af
			#model = Model(schedulers[policy], temp_tasks, temp_partitions)
			model = Model(schedulers[policy],task_list, partition_list)
			model.run_model()
			if model.is_schedulable():
				schedulability[policy] += 1
			num_ratio[policy] += model.get_unit_ratio()
			util_ratio[policy] += model.get_pro_ratio()
			val_ratio[policy] += model.get_val_ratio()
	print 'Schedulability and raio when utilization ratio = '+str(total_density/float(total_af))+':'

	x = total_af
	for policy in range(len(schedulers)):
		schedulability[policy] /= (float)(repeat_times)
		num_ratio[policy] /= (float)(repeat_times)
		val_ratio[policy] /= (float)(repeat_times)
		util_ratio[policy] /= (float)(repeat_times)
		print 'For policy '+str(policy)+', schedulability is: '+str(schedulability[policy])+', and ratio is: '+str(num_ratio[policy])+', '+str(val_ratio[policy])+', '+str(util_ratio[policy])
		wFile.write('('+str(x)+','+str(schedulability[policy])+')'+';')
		wFile.write('('+str(x)+','+str(num_ratio[policy])+')'+';')
		wFile.write('('+str(x)+','+str(val_ratio[policy])+')'+';')
		wFile.write('('+str(x)+','+str(util_ratio[policy])+')'+'\n')



if __name__ == "__main__":
	total_af = 20
	wFile = open('RPM_epsilon_change_ratio = 0.2.txt','w')
	for total_af in range(10,110,10):
		utilization_ratio = 0.9
		total_density = total_af*utilization_ratio
		simulate(total_density, total_af, wFile)	


