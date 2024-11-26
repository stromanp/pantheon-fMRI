import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pydatabase
import pyclustering

networkmodel = r'E:\FM2021data\network_model_June2023_SAPM.xlsx'

DBname = r'E:\FM2021data\FMS2_database_July27_2022b.xlsx'
DBnum = [253,256,258,260, 261]   # data from another person
DBnum = [0,1,3,6]   # get the data for the first person in the set
prefix = 'xptc'
region = 'Thalamus'
nclusters = 5

# get info about data
filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix, mode='list')
nruns_per_person = np.zeros(NP).astype(int)
for nn in range(NP):
	nruns_per_person[nn] = len(filename_list[nn])
nruns_total = np.sum(nruns_per_person)


clusterdefname = r'E:\FM2021data\allstim_equalsize_cluster_def_Dec2023.npy'
clusterdefname = r'E:\FM2021data\allstim_cluster_def_Jan2024.npy'

clusterdef = np.load(clusterdefname, allow_pickle=True).flat[0]
cluster_properties = clusterdef['cluster_properties']
template_img = clusterdef['template_img']
regionmap_img = clusterdef['regionmap_img']
rnamelist = [cluster_properties[xx]['rname'] for xx in range(len(cluster_properties))]
index = rnamelist.index(region)
cx,cy,cz = cluster_properties[index]['cx'], cluster_properties[index]['cy'], cluster_properties[index]['cz']
IDX = cluster_properties[index]['IDX']
nvox = len(cx)

# reload data for one person
nvolmask = 0
region_properties_test = pyclustering.load_cluster_data(cluster_properties, DBname, DBnum, prefix, nvolmask, networkmodel)
nregions0 = len(region_properties_test)
rnamelist0 = [region_properties_test[xx]['rname'] for xx in range(nregions0)]
index0 = rnamelist0.index(region_name)
tc0 = region_properties_test[index0]['tc']

# reload data for one person
nvolmask = 0
region_properties_raw = pyclustering.load_cluster_data(cluster_properties, DBname, DBnum, 'ptc', nvolmask, networkmodel)
nregions0b = len(region_properties_raw)
rnamelist0b = [region_properties_raw[xx]['rname'] for xx in range(nregions0b)]
index0b = rnamelist0.index(region_name)
tc0b = region_properties_raw[index0b]['tc']


# compare with original data
regiondataname1 = r'E:\FM2021data\HCstim_region_data_Dec2023.npy'
regiondata1 = np.load(regiondataname1, allow_pickle=True).flat[0]
region_properties1 = regiondata1['region_properties']
nregions1 = len(region_properties1)
rnamelist1 = [region_properties1[xx]['rname'] for xx in range(nregions1)]
index1 = rnamelist1.index(region_name)
tc1 = region_properties1[index1]['tc']
nruns_per_person1 = region_properties1[index1]['nruns_per_person']
tsize1 = region_properties1[index1]['tsize']
nclusters1, tsize_full1 = np.shape(tc1)
NP1 = len(nruns_per_person1)


# compare with brain data
regiondatanameb = r'D:\Howie_FM2_Brain_Data\HCstim_region_data_Dec2023.npy'
regiondatab = np.load(regiondatanameb, allow_pickle=True).flat[0]
region_propertiesb = regiondatab['region_properties']
nregionsb = len(region_propertiesb)
rnamelistb = [region_propertiesb[xx]['rname'] for xx in range(nregionsb)]
indexb = rnamelistb.index(region_name)
tcb = region_propertiesb[indexb]['tc']
nruns_per_personb = region_propertiesb[indexb]['nruns_per_person']
tsizeb = region_propertiesb[indexb]['tsize']
nclustersb, tsize_fullb = np.shape(tcb)
NPb = len(nruns_per_personb)
nruns_totalb = nruns_per_personb[0]
tsize_fullb = nruns_totalb*tsizeb
# tcb_avg = np.mean(np.reshape(tcb[:,:tsize_fullb],(nclusters, nruns_totalb, tsizeb)),axis=1)  # first person in set
tcb_avg = np.mean(np.reshape(tcb[:,-tsize_fullb:],(nclusters, nruns_totalb, tsizeb)),axis=1)   # last person in set
TR = 6.75
TRb = 2.0
timelist = np.array(range(tsize1)) * TR + TR / 2.0
timelistb = np.array(range(tsizeb)) * TRb + TRb / 2.0
tcb_avg_interp = np.zeros((nclusters,tsize1))
for cc in range(nclusters):
	f = interpolate.interp1d(timelistb, tcb_avg[cc,:], fill_value='extrapolate')
	tcb_avg_interp[cc,:] = f(timelist)

windownum = 200
plt.close(windownum)
fig = plt.figure(windownum)
plt.plot(tc1[0,:160], tc0[0,:160],'og')

windownum = 120
plt.close(windownum)
fig = plt.figure(windownum)
plt.plot(range(160), tc0[0,:160],'-ob')

plt.close(windownum-1)
fig = plt.figure(windownum-1)
plt.plot(timelist, tcb_avg_interp[cnum,:],'-g')
plt.plot(timelistb, tcb_avg[cnum,:],'-b')

plt.close(windownum+10)
fig = plt.figure(windownum+10)
plt.plot(range(160), tc0b[0,:160],'-xb')

plt.close(windownum+1)
fig = plt.figure(windownum+1)
plt.plot(range(160), tc1[0,:160],'-or')


tsize = region_properties_test[0]['tsize']
tc0_avg = np.mean(np.reshape(tc0,(nclusters, nruns_total, tsize)),axis=1)
tc0b_avg = np.mean(np.reshape(tc0b,(nclusters, nruns_total, tsize)),axis=1)

windownum = 110
plt.close(windownum+5)
cnum = 1
fig = plt.figure(windownum+5)
plt.plot(range(tsize), tc0_avg[cnum,:],'-or')
plt.plot(range(tsize), 0.2*tc0b_avg[cnum,:],'-b')
plt.plot(range(tsize), tcb_avg_interp[cnum,:],'-g')

plt.close(windownum+6)
fig = plt.figure(windownum+6)
plt.plot(tc0_avg[cnum,:], 0.2*tc0b_avg[cnum,:],'xb')

plt.close(windownum+7)
fig = plt.figure(windownum+7)
plt.plot(range(tsize1), tcb_avg_interp[cnum,:],'-g')





nvolmask = 0
for nn, name in enumerate(filename_list[0]):
	input_img = nib.load(name)
	input_data = input_img.get_fdata()
	if nn == 0:
		xs,ys,zs,tsize = np.shape(input_data)
		ts = nruns_per_person[0]*tsize

	roi_data1 = input_data[cx, cy, cz, :]  # check the size of this
	# mask out the initial volumes, if wanted
	if nvolmask > 0:
		for tt in range(nvolmask): roi_data1[:, tt] = roi_data1[:, nvolmask]

	# # convert to signal change from the average----------------
	# if data have been cleaned they are already percent signal changes
	mean_data = np.mean(roi_data1, axis=1)
	mean_data = np.repeat(mean_data[:, np.newaxis], tsize, axis=1)
	roi_data1 = roi_data1 - mean_data

	if nn == 0:
		person_data = copy.deepcopy(roi_data1)
	else:
		person_data = np.concatenate((person_data, roi_data1), axis=1)

# identify crazy voxels
rdtemp = person_data.reshape(nvox, tsize, nruns_total, order='F').copy()
varcheck2 = np.var(rdtemp, axis=1)
typicalvar2 = np.median(varcheck2)
varcheckthresh = 3.0
varlimit = varcheckthresh * typicalvar2

cv, cp = np.where(varcheck2 > varlimit)  # voxels with crazy variance
if len(cv) > 0:
	for vv in range(len(cv)):
		rdtemp[cv[vv], :, cp[vv]] = np.zeros(tsize)
	print('---------------!!!!!----------------------');
	print('Variance check found {} crazy voxels'.format(len(cv)))
	print('---------------!!!!!----------------------\n');
else:
	print('Variance check did not find any crazy voxels');
person_data2 = rdtemp.reshape(nvox, ts, order='F').copy()

tc = np.zeros([nclusters, ts])
tc_sem = np.zeros([nclusters, ts])
for cc in range(nclusters):
	# dd = np.where(IDX == cc)[0]
	dd = [i for i in range(len(IDX)) if IDX[i] == cc]
	temp_data = person_data2[dd, :]  # check the size of this
	tc[cc, :] = np.mean(temp_data, axis=0)
	tc_sem[cc, :] = np.std(temp_data, axis=0) / np.sqrt(nvox)
roi_data = copy.deepcopy(tc)
roi_data_sem = copy.deepcopy(tc_sem)



nvolmask = 2
mode = 'concatenate'
region_data_check = pyclustering.load_data_from_region(filename_list, nvolmask, mode, cx, cy, cz)
roi_data2 = np.zeros(np.shape(roi_data))
for cc in range(nclusters):
	dd = np.where(IDX == cc)[0]
	temp_data = region_data_check[dd, :]
	temp_data2 = np.mean(temp_data, axis=0)
	# for ttt in range(nvolmask):
	# 	temp_data2[ttt] = temp_data2[nvolmask]
	# temp_data2 = (temp_data2 - np.mean(temp_data2)) / np.mean(temp_data2)

	roi_data2[cc, :] = temp_data2


region_properties2 = pyclustering.load_cluster_data(cluster_properties, DBname, DBnum, prefix, nvolmask, networkmodel)
tc3 = copy.deepcopy(region_properties2[index]['tc'])

windownum = 20
plt.close(windownum)
fig = plt.figure(windownum)
plt.plot(range(160), tc1[0,:160],'-ob')

plt.close(windownum-1)
fig = plt.figure(windownum-1)
plt.plot(range(160), tc3[0,:160],'-ob')

plt.close(windownum+1)
fig = plt.figure(windownum+1)
plt.plot(range(160), roi_data[0,:],'-xr')

plt.close(windownum+2)
fig = plt.figure(windownum+2)
plt.plot(range(160), roi_data2[0,:],'-xr')

plt.close(windownum+3)
fig = plt.figure(windownum+3)
plt.plot(tc1[0,:160], tc3[0,:160],'og')



# more tests
DBname = r'E:\FM2021data\FMS2_database_July27_2022b.xlsx'
DBnum = [0,1,3,6]   # get the data for the first person in the set
DBnum = [253,256,258,260, 261]   # data from another person
prefix = 'xptc'
nclusters = 5

nmaskvol = 0
# get info about data
filename_list1, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, 'xptc', mode='list')
name = filename_list1[0][1]
input_img1 = nib.load(name)
input_data1 = input_img1.get_fdata()
for tt in range(nmaskvol):
	input_data1[:,:,:,tt] = copy.deepcopy(input_data1[:,:,:,nmaskvol])

filename_list2, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, 'ptc', mode='list')
name = filename_list2[0][1]
input_img2 = nib.load(name)
input_data2 = input_img2.get_fdata()
for tt in range(nmaskvol):
	input_data2[:,:,:,tt] = copy.deepcopy(input_data2[:,:,:,nmaskvol])

filename_list3, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, 'tc', mode='list')
name = filename_list3[0][1]
input_img3 = nib.load(name)
input_data3 = input_img3.get_fdata()
for tt in range(nmaskvol):
	input_data3[:,:,:,tt] = copy.deepcopy(input_data3[:,:,:,nmaskvol])

filename_list4, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, 'c', mode='list')
name = filename_list4[0][1]
input_img4 = nib.load(name)
input_data4 = input_img4.get_fdata()
for tt in range(nmaskvol):
	input_data4[:,:,:,tt] = copy.deepcopy(input_data4[:,:,:,nmaskvol])


filename_list5, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, '', mode='list')
name = filename_list5[0][1]
input_img5 = nib.load(name)
input_data5 = input_img5.get_fdata()
for tt in range(nmaskvol):
	input_data5[:,:,:,tt] = copy.deepcopy(input_data5[:,:,:,nmaskvol])


x,y,z = 11,36,141
x,y,z = 12, 58, 199
tc1 = input_data1[x,y,z,:]
tc2 = input_data2[x,y,z,:]
tc2_scale = 100.0*tc2/np.mean(tc2)

x,y,z = 4,58,96
x,y,z = 4,60,133   # first person
x,y,z = 4,60,139   # last person
tc3 = input_data3[x,y,z,:]
tc3_scale = 100.0*tc3/np.mean(tc3)
tc4 = input_data4[x,y,z,:]
tc4_scale = 100.0*tc4/np.mean(tc4)
tc5 = input_data5[x,y,z,:]
tc5_scale = 100.0*tc5/np.mean(tc5)

plt.close(105)
fig = plt.figure(105)
plt.plot(range(40),tc1,'-xr')
plt.plot(range(40),tc2_scale,'-xb')
plt.plot(range(40),tc3_scale,'-og')
plt.plot(range(40),tc4_scale,'-oy')
plt.plot(range(40),tc5_scale,'-ok')



# now test SAPM data versus original data
region_name ='Thalamus'
numperson = 14

# region data
regiondataname = r'E:\FM2021data\HCstim_region_data_Dec2023.npy'
regiondata = np.load(regiondataname, allow_pickle=True).flat[0]
region_properties = regiondata['region_properties']
nregions = len(region_properties)
rnamelist = [region_properties[xx]['rname'] for xx in range(nregions)]

index = rnamelist.index(region_name)
tcr = region_properties[index]['tc']
nruns_per_person = region_properties[index]['nruns_per_person']
tsize = region_properties[index]['tsize']

t1 = np.sum(nruns_per_person[:numperson])*tsize
t2 = np.sum(nruns_per_person[:(numperson+1)])*tsize
tc_region = np.mean(np.reshape(tcr[:,t1:t2],(nclusters,nruns_per_person[numperson],tsize)),axis = 1)

# parameter data
paramsname = r'E:\FM2021data\HCstim_1432043142_params.npy'
params = np.load(paramsname, allow_pickle=True).flat[0]
tcdata_centered = copy.deepcopy(params['tcdata_centered'])
tcdata_centered_original = copy.deepcopy(params['tcdata_centered_original'])
nclusterlist = copy.deepcopy(params['nclusterlist'])
nclusterstotal = np.sum(nclusterlist)
tc_params = np.mean(np.reshape(tcdata_centered[:,t1:t2],(nclusterstotal,nruns_per_person[numperson],tsize)),axis =1)


# data saved with results
SAPMresultsname = r'E:\FM2021data\HCstim_1432043142_results_corr.npy'
SAPMresults_load = np.load(SAPMresultsname, allow_pickle=True)
Sinput = SAPMresults_load[numperson]['Sinput']
tc_results = np.mean(np.reshape(Sinput[index,:],(nruns_per_person[numperson],tsize)),axis=0)

cnum = 2
cpnum = np.sum(nclusterlist[:index]) + cnum

plt.close(300)
fig = plt.figure(300)
plt.plot(range(tsize),5.7*tc_region[cnum,:],'-xr')
plt.plot(range(tsize),tc_results,'-ob')
plt.plot(range(tsize),tc_params[cpnum,:],'-k')


fig = plt.figure(105)
plt.plot(range(tsize),100+5.0*tc_params[cpnum,:],'-xk')