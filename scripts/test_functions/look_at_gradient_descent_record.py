import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import copy

recordname1 = r'D:\Howie_FM2_Brain_Data\gradient_descent_record.npy'
windownum = 69

test_record = np.load(recordname1, allow_pickle=True)

stage_list = np.zeros(len(test_record))
for nn in range(len(test_record)):
	stage_list[nn] = copy.deepcopy(test_record[nn]['stage'])

# nsteps1 = 30
# nsteps2 = 4
# nsteps3 = 1

nsteps1 = len(np.where(stage_list == 1)[0])
nsteps2 = len(np.where(stage_list == 2)[0])
nsteps3 = len(np.where(stage_list == 3)[0])

stage1_length = np.zeros(nsteps2)
for nn in range(nsteps2):
	aa = nsteps1+nn
	index = test_record[aa]['stage2_base'].astype(int)
	stage1_length[nn] = len(test_record[index]['R2avg_record'])

index = test_record[nsteps1+nsteps2]['stage3_base'].astype(int)
stage2_length = stage1_length[index] + len(test_record[index+nsteps1]['R2avg_record'])

plt.close(windownum)
fig = plt.figure(windownum)

for nn in range(nsteps1):
	plt.plot(range(len(test_record[nn]['R2avg_record'])),test_record[nn]['R2avg_record'],'-r')

for nn in range(nsteps2):
	plt.plot(range(len(test_record[nn+nsteps1]['R2avg_record'])) + stage1_length[nn],test_record[nn+nsteps1]['R2avg_record'],'-g')

plt.plot(np.array(range(len(test_record[nsteps2+nsteps1]['R2avg_record']))) + stage2_length, test_record[nsteps2+nsteps1]['R2avg_record'],'-b')

