from Generation import *
from Model import Model
g = Generation()

'''
bin_set = g.generate_bins(20)
item_set = g.generate_items(20)

for item in item_set:
	item.print_info()

'''

env = Model(10,0.5)
env.reset()
state, reward, done, msg = env.step(0)
print(state)
print(msg)
