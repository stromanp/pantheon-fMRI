
color1 = [1,0,0]
color2 = [0,0,1]
marker1 = 'o'
marker2 = 'o'


interpolate_values = False
offset_starting_value = True

show_time_periods = True
timeperiod1 = [60.,70.]
timeperiod2 = [120.,150.]

if datatype1 == 'brain':
	clusterdef_name1 = r'D:\Howie_FM2_Brain_Data\allstim_cluster_def_brain_Jan28_2024_V3.npy'
	regiondata_name1 = r'D:\Howie_FM2_Brain_Data\{}{}_region_data_allstim_Jan28_2024_V5.npy'.format(grouptype1,stimtype1)
	TR1 = 2.0
	cnum_list1 = [0, 2, 3, 2, 3, 2, 0, 3, 3, 3, 1, 2, 0, 2]
else:
	clusterdef_name1 = r'E:\FM2021data\allstim_equal_cluster_def_Jan22_2024_V3.npy'
	regiondata_name1 = r'E:\FM2021data\{}{}_equal_region_data_Jan23_2024_V5.npy'.format(grouptype1,stimtype1)
	TR1 = 6.75
	cnum_list1 = [2, 2, 3, 0, 2, 2, 4, 1, 2, 4]

if datatype2 == 'brain':
	clusterdef_name2 = r'D:\Howie_FM2_Brain_Data\allstim_cluster_def_brain_Jan28_2024_V3.npy'
	regiondata_name2 = r'D:\Howie_FM2_Brain_Data\{}{}_region_data_allstim_Jan28_2024_V5.npy'.format(grouptype2,stimtype2)
	TR2 = 2.0
	cnum_list2 = [0, 2, 3, 2, 3, 2, 0, 3, 3, 3, 1, 2, 0, 2]
else:
	clusterdef_name2 = r'E:\FM2021data\allstim_equal_cluster_def_Jan22_2024_V3.npy'
	regiondata_name2 = r'E:\FM2021data\{}{}_equal_region_data_Jan23_2024_V5.npy'.format(grouptype2,stimtype2)
	TR2 = 6.75
	cnum_list2 = [2, 2, 3, 0, 2, 2, 4, 1, 2, 4]

toffset = 0.

#--------copy values and sort out what to plot----------
# cnum1 = copy.deepcopy(cnum_brain)
# cnum2 = copy.deepcopy(cnum_ccbs)
#
# clusterdef_name1 = copy.deepcopy(clusterdef_brain_name)
# clusterdef_name2 = copy.deepcopy(regiondata_ccbs_name)
#
# regiondata_name1 = copy.deepcopy(regiondata_brain_name)
# regiondata_name2 = copy.deepcopy(clusterdef_ccbs_name)

clusterdef1 = np.load(clusterdef_name1, allow_pickle=True).flat[0]
clusterdef2 = np.load(clusterdef_name2, allow_pickle=True).flat[0]

regiondata1 = np.load(regiondata_name1, allow_pickle=True).flat[0]
regiondata2 = np.load(regiondata_name2, allow_pickle=True).flat[0]

regiondata1 = regiondata1['region_properties']
regiondata2 = regiondata2['region_properties']

rname_list1 = [regiondata1[xx]['rname'] for xx in range(len(regiondata1))]
rname_index1 = rname_list1.index(region_name)
cnum1 = cnum_list1[rname_index1]
print('1) getting data for region {},  index number {}'.format(region_name, rname_index1))

rname_list2 = [regiondata2[xx]['rname'] for xx in range(len(regiondata2))]
rname_index2 = rname_list2.index(region_name)
cnum2 = cnum_list2[rname_index2]
print('2) getting data for region {},  index number {}'.format(region_name, rname_index2))

tc1 = regiondata1[rname_index1]['tc']
tsize1 = regiondata1[rname_index1]['tsize']
nruns_per_person1 = regiondata1[rname_index1]['nruns_per_person']
total_nruns1 = np.sum(nruns_per_person1)
nclusters1, tsize_total1 = np.shape(tc1)
NP1 = len(nruns_per_person1)

tc2 = regiondata2[rname_index2]['tc']
tsize2 = regiondata2[rname_index2]['tsize']
nruns_per_person2 = regiondata2[rname_index2]['nruns_per_person']
total_nruns2 = np.sum(nruns_per_person2)
nclusters2, tsize_total2 = np.shape(tc2)
NP2 = len(nruns_per_person2)



tc1_avg = np.mean(np.reshape(tc1, (nclusters1, total_nruns1, tsize1)),axis=1)
tc2_avg = np.mean(np.reshape(tc2, (nclusters2, total_nruns2, tsize2)),axis=1)

tc1_sem = np.std(np.reshape(tc1, (nclusters1, total_nruns1, tsize1)),axis=1)/np.sqrt(NP1)
tc2_sem = np.std(np.reshape(tc2, (nclusters2, total_nruns2, tsize2)),axis=1)/np.sqrt(NP2)

# cc_grid1 = np.zeros((nclusters_ccbs, nclusters_brain))
# cc_grid2 = np.zeros((nclusters_ccbs, nclusters_brain))
# for bb in range(nclusters_brain):


# check
# toffset = -2.2
# t1 = np.array(range(tsize1))*TR1 + TR1/2.0
# t2 = np.array(range(tsize2))*TR2 + TR2/2.0 + toffset
# f = interpolate.interp1d(t1, tc1_avg[cnum1, :], fill_value='extrapolate')
# tc1_avg_interp = f(t2)
# f = interpolate.interp1d(t1, tc1_sem[cnum1, :], fill_value='extrapolate')
# tc1_sem_interp = f(t2)
# R = np.corrcoef(tc2_avg[cnum2, :], tc1_avg_interp)
# print('toffset = {:.4f}'.format(toffset))
# print('   correlation betwen TC2 and interpolated TC1 is {:.5f}'.format(R[0, 1]))
# Z = np.arctanh(R[0, 1]) * np.sqrt(tsize2 - 3)
# p = 1.0 - stats.norm.cdf(Z)
# print('   Z = {:.5f}   p = {:.5f}'.format(Z,p))





# now interpolate to the same number of volumes
toffset = 0.0
t1 = np.array(range(tsize1))*TR1 + TR1/2.0
t2 = np.array(range(tsize2))*TR2 + TR2/2.0 + toffset

if interpolate_values:
	f = interpolate.interp1d(t1, tc1_avg[cnum1,:], fill_value = 'extrapolate')
	tc1_avg_interp = f(t2)
	f = interpolate.interp1d(t1, tc1_sem[cnum1,:], fill_value = 'extrapolate')
	tc1_sem_interp = f(t2)

	# for aa in range(nclusters_ccbs):
# CC = np.corrcoef(tc_ccbs_avg[cnum_ccbs, :], tc_brain_avg_interp)

initial_volumes = 5

if offset_starting_value:
	if interpolate_values:
		startingval1 = np.mean(tc1_avg_interp[:initial_volumes])
	else:
		startingval1 = np.mean(tc1_avg[cnum1,:initial_volumes])
	startingval2 = np.mean(tc2_avg[cnum2, :initial_volumes])
	intensity_offset = startingval1 - startingval2
else:
	intensity_offset = 0.0


# get size of initial rise
tbaseline = list(range(int(timeperiod1[1]), int(timeperiod2[0]),2)) + list(range(int(timeperiod2[1]),270,2))
cbaseline = list((np.array(tbaseline)/2.0).astype(int))

if interpolate_values:
	startingval1 = np.mean(tc1_avg_interp[:initial_volumes])
	# startingval1_sem = np.std(tc1_avg_interp[:initial_volumes])/np.sqrt(initial_volumes)
	startingval1_sem = np.std(tc1_avg_interp[:initial_volumes])
	baselineval1 = np.mean(tc1_avg_interp[cbaseline])
	# baselineval1_sem = np.std(tc1_avg_interp[cbaseline])/np.sqrt(len(cbaseline))
	baselineval1_sem = np.std(tc1_avg_interp[cbaseline])
else:
	startingval1 = np.mean(tc1_avg[cnum1,:initial_volumes])
	# startingval1_sem = np.std(tc1_avg[cnum1,:initial_volumes])/np.sqrt(initial_volumes)
	startingval1_sem = np.std(tc1_avg[cnum1,:initial_volumes])
	baselineval1 = np.mean(tc1_avg[cnum1,cbaseline])
	# baselineval1_sem = np.std(tc1_avg[cnum1,cbaseline])/np.sqrt(len(cbaseline))
	baselineval1_sem = np.std(tc1_avg[cnum1,cbaseline])

startingval2 = np.mean(tc2_avg[cnum2,:initial_volumes])
# startingval2_sem = np.std(tc2_avg[cnum2,:initial_volumes])/np.sqrt(initial_volumes)
startingval2_sem = np.std(tc2_avg[cnum2,:initial_volumes])
baselineval2 = np.mean(tc2_avg[cnum2,cbaseline])
# baselineval2_sem = np.std(tc2_avg[cnum2,cbaseline])/np.sqrt(len(cbaseline))
baselineval2_sem = np.std(tc2_avg[cnum2,cbaseline])

print('{} {} {}:   change from initial to baseline = {:.4f} {} {:.4f}'.format(region_name, grouptype1, stimtype1, baselineval1-startingval1,chr(177),baselineval1_sem))
print('{} {} {}:   change from initial to baseline = {:.4f} {} {:.4f}'.format(region_name, grouptype2, stimtype2, baselineval2-startingval2,chr(177),baselineval2_sem))
