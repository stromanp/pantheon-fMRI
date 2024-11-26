# sys.path.extend([r'C:\Users\Stroman\PycharmProjects\pantheon\venv'])
# sys.path.extend([r'C:\Users\Stroman\PycharmProjects\pantheon\venv\test_functions'])

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyclustering
import load_templates
import copy
import image_operations_3D as i3d
import os
import pandas as pd
import nibabel as nib
from scipy import interpolate
import py2ndlevelanalysis

datatype = 'brain'

region_name = 'IC'
region_name = 'Hippocampus'
region_name = 'Hypothalamus'
region_name = 'LC'
region_name = 'Thalamus'
region_name = 'PAG'
region_name = 'FOrb'
region_name = 'PAG'
region_name = 'IC'
windownum = 102


datatype1 = copy.deepcopy(datatype)
grouptype1 = 'PVD'
stimtype1 = 'nostim'

datatype2 = copy.deepcopy(datatype)
grouptype2 = 'PVD'
stimtype2 = 'stim'

color1 = [0.5,0.5,0.5]
color2 = [0,0,0]
marker1 = 'o'
marker2 = 'o'


interpolate_values = False

show_time_periods = True
timeperiod1 = [60.,70.]
timeperiod2 = [120.,150.]

if datatype1 == 'brain':
	clusterdef_name1 = r'F:\PVD_study\SAPMresults\allstim_cluster_def_brain_Jan28_2024_V3.npy'
	regiondata_name1 = r'F:\PVD_study\SAPMresults\{}{}_allstim_V5.npy'.format(grouptype1,stimtype1)
	TR1 = 3.0
	cnum_list1 = [0, 2, 3, 2, 3, 2, 0, 3, 3, 3, 1, 2, 0, 2]
else:
	print('not ready for cord data ...')
	# clusterdef_name1 = r'E:\FM2021data\allstim_equal_cluster_def_Jan22_2024_V3.npy'
	# regiondata_name1 = r'E:\FM2021data\{}{}_equal_region_data_Jan23_2024_V5.npy'.format(grouptype1,stimtype1)
	# TR1 = 6.75
	# cnum_list1 = [2, 2, 3, 0, 2, 2, 4, 1, 2, 4]

if datatype2 == 'brain':
	clusterdef_name2 = r'F:\PVD_study\SAPMresults\allstim_cluster_def_brain_Jan28_2024_V3.npy'
	regiondata_name2 = r'F:\PVD_study\SAPMresults\{}{}_allstim_V5.npy'.format(grouptype2,stimtype2)
	TR2 = 3.0
	cnum_list2 = [0, 2, 3, 2, 3, 2, 0, 3, 3, 3, 1, 2, 0, 2]
else:
	print('not ready for cord data ...')
	# clusterdef_name2 = r'E:\FM2021data\allstim_equal_cluster_def_Jan22_2024_V3.npy'
	# regiondata_name2 = r'E:\FM2021data\{}{}_equal_region_data_Jan23_2024_V5.npy'.format(grouptype2,stimtype2)
	# TR2 = 6.75
	# cnum_list2 = [2, 2, 3, 0, 2, 2, 4, 1, 2, 4]

toffset = 0.

#--------copy values and sort out what to plot----------
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

# reorganize
tc1r = np.reshape(tc1, (nclusters1, total_nruns1, tsize1))
tc1_per_person = np.zeros((nclusters1,NP1,tsize1))
for pp in range(NP1):
	r1 = np.sum(nruns_per_person1[:pp])
	r2 = np.sum(nruns_per_person1[:(pp+1)])
	tc1_per_person[:,pp,:] = np.mean(tc1r[:,r1:r2,:], axis = 1)

tc2r = np.reshape(tc2, (nclusters2, total_nruns2, tsize2))
tc2_per_person = np.zeros((nclusters2,NP2,tsize2))
for pp in range(NP2):
	r1 = np.sum(nruns_per_person2[:pp])
	r2 = np.sum(nruns_per_person2[:(pp+1)])
	tc2_per_person[:,pp,:] = np.mean(tc2r[:,r1:r2,:], axis = 1)

tc1_avg = np.mean(tc1_per_person,axis=1)
tc2_avg = np.mean(tc2_per_person,axis=1)

tc1_sem = np.std(tc1_per_person,axis=1)/np.sqrt(NP1)
tc2_sem = np.std(tc2_per_person,axis=1)/np.sqrt(NP2)


# tc1_avg = np.mean(np.reshape(tc1, (nclusters1, total_nruns1, tsize1)),axis=1)
# tc2_avg = np.mean(np.reshape(tc2, (nclusters2, total_nruns2, tsize2)),axis=1)
#
# tc1_sem = np.std(np.reshape(tc1, (nclusters1, total_nruns1, tsize1)),axis=1)/np.sqrt(NP1)
# tc2_sem = np.std(np.reshape(tc2, (nclusters2, total_nruns2, tsize2)),axis=1)/np.sqrt(NP2)

# cc_grid1 = np.zeros((nclusters_ccbs, nclusters_brain))
# cc_grid2 = np.zeros((nclusters_ccbs, nclusters_brain))
# for bb in range(nclusters_brain):


# now interpolate to the same number of volumes
t1 = np.array(range(tsize1))*TR1 + TR1/2.0
t2 = np.array(range(tsize2))*TR2 + TR2/2.0 + toffset

if interpolate_values:
	f = interpolate.interp1d(t1, tc1_avg[cnum1,:], fill_value = 'extrapolate')
	tc1_avg_interp = f(t2)
	f = interpolate.interp1d(t1, tc1_sem[cnum1,:], fill_value = 'extrapolate')
	tc1_sem_interp = f(t2)

	# for aa in range(nclusters_ccbs):
# CC = np.corrcoef(tc_ccbs_avg[cnum_ccbs, :], tc_brain_avg_interp)

plt.close(windownum)
fig = plt.figure(windownum)
plt.plot(t2,tc2_avg[cnum2,:],linestyle='-',marker = marker2, color = color2)

y1 = list(tc2_avg[cnum2,:] + tc2_sem[cnum2,:])
y2 = list(tc2_avg[cnum2,:] - tc2_sem[cnum2,:])
yy = y1 + y2[::-1]
xx = list(t2) + list(t2)[::-1]
plt.fill(xx, yy, facecolor=color2, edgecolor='None', alpha=0.2)

if show_time_periods:
	ymin1 = np.min(yy)
	ymax1 = np.max(yy)

if interpolate_values:
	plt.plot(t2,tc1_avg_interp,linestyle='-',marker = marker1, color = color1)

	y1 = list(tc1_avg_interp + tc1_sem_interp[cnum2, :])
	y2 = list(tc1_avg_interp - tc1_sem_interp[cnum2, :])
	yy = y1 + y2[::-1]
	xx = list(t2) + list(t2)[::-1]
else:
	plt.plot(t1,tc1_avg[cnum1,:],linestyle='-',marker = marker1, color = color1)

	y1 = list(tc1_avg[cnum1,:] + tc1_sem[cnum1,:])
	y2 = list(tc1_avg[cnum1,:] - tc1_sem[cnum1,:])
	yy = y1 + y2[::-1]
	xx = list(t1) + list(t1)[::-1]

plt.fill(xx, yy, facecolor=color1, edgecolor='None', alpha=0.2)

if show_time_periods:
	ymin2 = np.min(yy)
	ymax2 = np.max(yy)

	ymin = np.min([ymin1, ymin2])
	ymax = np.max([ymax1, ymax2])

	tp = copy.deepcopy(timeperiod1)
	# ymin = np.min(yy)
	# ymax = np.max(yy)
	xx = [tp[0], tp[1], tp[1], tp[0]]
	yy = [ymin, ymin, ymax, ymax]
	plt.fill(xx, yy, facecolor=[1,1,0], edgecolor='None', alpha=0.2)

	tp = copy.deepcopy(timeperiod2)
	# ymin = np.min([ymin1, ymin2])
	# ymax = np.max([ymax1, ymax2])
	xx = [tp[0], tp[1], tp[1], tp[0]]
	# yy = [ymin, ymin, ymax, ymax]
	plt.fill(xx, yy, facecolor=[0,1,0.3], edgecolor='None', alpha=0.2)


# # fit a curve to the first minute of data
# expfit1,m1,b1 = fit_exp_rise(t1,tc1_avg[cnum1, :], 22.0)
# expfit2,m2,b2 = fit_exp_rise(t2,tc2_avg[cnum2, :], 22.0)
#
# print('1. exp scale: {:.3f}    offset {:.3f} '.format(m1, b1))
# print('2. exp scale: {:.3f}    offset {:.3f} '.format(m2, b2))
#
# plt.plot(t1, expfit1, linestyle='-', linewidth = 3, marker='', color=color1)
# plt.plot(t2, expfit2, linestyle='-', linewidth = 3, marker='', color=color2)



def fit_exp_func(t,tc):
	nt = len(tc)
	nt2 = np.floor(nt/2.0).astype(int)
	offset = np.mean(tc[-nt2:])
	if tc[0] < 0:
		t1 = np.where((tc-offset) >= -1e-2)[0][0]
		m, b = np.polyfit(t[:t1], np.log(-1.0*(np.array((tc[:t1]-offset)))), 1)
		b0 = -1.0*np.exp(b)
		fit = b0*np.exp(t*m) + offset
	else:
		t1 = np.where((tc-offset) >= 1e-2)[0][0]
		m, b = np.polyfit(t[:t1], np.log(np.array((tc[:t1]-offset))), 1)
		b0 = np.exp(b)
		fit = b0*np.exp(t*m) + offset

	return fit, m, b0, offset


def fit_exp_rise(t,tc,Trate):
	nt = len(tc)
	modelexp = np.exp(-t/Trate)   # exp decay function
	G = np.concatenate((modelexp[:,np.newaxis], np.ones(nt)[:,np.newaxis]),axis=1)

	# tc = G @ b
	b = np.linalg.inv(G.T @ G) @ (G.T @ tc)
	fit = G @ b

	m = b[0]
	b0 = b[1]
	return fit, m, b0


def look_at_cluster_data():
	datatype = 'HC'
	windownum = 7

	if datatype == 'PVD':
		regiondataname = r'F:\PVD_study\SAPMresults/PVDstim_allstim_V5.npy'
		covariatesname = r'F:\PVD_study\SAPMresults/PVDstim_02323203331202_results_covariates.npy'

		regiondataname2 = r'F:\PVD_study\SAPMresults/PVDnostim_allstim_V5.npy'
		covariatesname2 = r'F:\PVD_study\SAPMresults/PVDnostim_02323203331202_results_covariates.npy'

	else:
		regiondataname = r'F:\PVD_study\SAPMresults/HCstim_allstim_V5.npy'
		covariatesname = r'F:\PVD_study\SAPMresults/HCstim_02323203331202_results_covariates.npy'

		regiondataname2 = r'F:\PVD_study\SAPMresults/HCnostim_allstim_V5.npy'
		covariatesname2 = r'F:\PVD_study\SAPMresults/HCnostim_02323203331202_results_covariates.npy'

	TR = 3.0

	data = np.load(regiondataname, allow_pickle=True).flat[0]
	regiondata = data['region_properties']
	rnamelist = [regiondata[xx]['rname'] for xx in range(len(regiondata))]

	covariates = np.load(covariatesname, allow_pickle=True).flat[0]
	lastpainindex = covariates['GRPcharacteristicslist'].index('lastpain')
	firstpainindex = covariates['GRPcharacteristicslist'].index('firstpain')
	tempindex = covariates['GRPcharacteristicslist'].index('temperature')

	lastpains = covariates['GRPcharacteristicsvalues'][lastpainindex,:]
	firstpains = covariates['GRPcharacteristicsvalues'][firstpainindex,:]
	temperatures = covariates['GRPcharacteristicsvalues'][tempindex,:]

	tc = regiondata[0]['tc']
	tsize = regiondata[0]['tsize']
	nruns_per_person = regiondata[0]['nruns_per_person']
	total_nruns = np.sum(nruns_per_person)
	nclusters, tsize_total = np.shape(tc)
	NP = len(nruns_per_person)


	ncluster_list = [np.shape(regiondata[xx]['tc'])[0] for xx in range(len(regiondata))]
	tc_all = np.zeros((np.sum(ncluster_list), tsize))
	tc_person = np.zeros((np.sum(ncluster_list), NP, tsize))
	tc_runs_per_person = np.zeros((np.sum(ncluster_list), NP, np.max(nruns_per_person), tsize))
	cluster_count = 0
	for rr in range(len(regiondata)):
		for cc in range(ncluster_list[rr]):
			tc_all[cluster_count, :] = np.mean(np.reshape(regiondata[rr]['tc'][cc, :], (total_nruns, tsize)), axis=0)
			for nn in range(NP):
				r1 = np.sum(nruns_per_person[:nn]).astype(int)
				r2 = np.sum(nruns_per_person[:(nn+1)]).astype(int)
				tc_person[cluster_count,nn, :] = np.mean(np.reshape(regiondata[rr]['tc'][cc, (r1*tsize):(r2*tsize)], (nruns_per_person[nn], tsize)), axis=0)
				tc_runs_per_person[cluster_count,nn,:nruns_per_person[nn], :] = np.reshape(regiondata[rr]['tc'][cc, (r1*tsize):(r2*tsize)], (nruns_per_person[nn], tsize))

			cluster_count += 1


#--------second set of data for comparison-------------------------
	data2 = np.load(regiondataname2, allow_pickle=True).flat[0]
	regiondata2 = data2['region_properties']

	covariates2 = np.load(covariatesname2, allow_pickle=True).flat[0]
	lastpainindex = covariates2['GRPcharacteristicslist'].index('lastpain')
	firstpainindex = covariates2['GRPcharacteristicslist'].index('firstpain')
	tempindex = covariates2['GRPcharacteristicslist'].index('temperature')

	lastpains2 = covariates2['GRPcharacteristicsvalues'][lastpainindex,:]
	firstpains2 = covariates2['GRPcharacteristicsvalues'][firstpainindex,:]
	temperatures2 = covariates2['GRPcharacteristicsvalues'][tempindex,:]

	tc2 = regiondata2[0]['tc']
	tsize2 = regiondata2[0]['tsize']
	nruns_per_person2 = regiondata2[0]['nruns_per_person']
	total_nruns2 = np.sum(nruns_per_person2)
	nclusters2, tsize_total2 = np.shape(tc2)
	NP2 = len(nruns_per_person2)

	tc_all2 = np.zeros((np.sum(ncluster_list), tsize2))
	tc_person2 = np.zeros((np.sum(ncluster_list), NP2, tsize2))
	tc_runs_per_person2 = np.zeros((np.sum(ncluster_list), NP2, np.max(nruns_per_person2), tsize2))
	cluster_count2 = 0
	for rr in range(len(regiondata2)):
		for cc in range(ncluster_list[rr]):
			tc_all2[cluster_count2, :] = np.mean(np.reshape(regiondata2[rr]['tc'][cc, :], (total_nruns2, tsize2)), axis=0)
			for nn in range(NP2):
				r1 = np.sum(nruns_per_person2[:nn]).astype(int)
				r2 = np.sum(nruns_per_person2[:(nn+1)]).astype(int)
				tc_person2[cluster_count2,nn, :] = np.mean(np.reshape(regiondata2[rr]['tc'][cc, (r1*tsize2):(r2*tsize2)], (nruns_per_person2[nn], tsize2)), axis=0)
				tc_runs_per_person2[cluster_count2,nn,:nruns_per_person2[nn], :] = np.reshape(regiondata2[rr]['tc'][cc, (r1*tsize2):(r2*tsize2)], (nruns_per_person2[nn], tsize2))
			cluster_count2 += 1
# ---------end of second set of data-------------------------------


	timeperiod1 = [60., 70.]
	timeperiod2 = [120., 150.]
	ttfill1 = np.array([timeperiod1[0], timeperiod1[1], timeperiod1[1], timeperiod1[0]])/TR
	ttfill2 = np.array([timeperiod2[0], timeperiod2[1], timeperiod2[1], timeperiod2[0]])/TR

	plt.close(windownum)
	fig, axs = plt.subplots(2,3,num=windownum, sharey=True)
	cols = np.array([[1,0,0],[1,0.5,0],[1,1,0],[0,0,1],[0,0.5,1],[0,1,1]])
	nc1, tsize = np.shape(tc_all)
	regionlist = [3,5,7,9,10,12]
	for nn,rr in enumerate([3,5,7,9,10,12]):
		rownum = np.floor(nn/2).astype(int)
		colnum = nn % 2
		# axs[colnum,rownum] = plt.subplot(2,3,nn+1)
		col = cols[nn,:]
		# for cc in range(ncluster_list[rr]):
		mm1 = np.sum(ncluster_list[:(rr-1)])
		mm2 = np.sum(ncluster_list[:(rr)])
		tc_avg = np.mean(tc_all[mm1:mm2,:],axis=0)
		tc_sem = np.std(tc_all[mm1:mm2,:],axis=0)/np.sqrt(ncluster_list[rr])

		tc_avg2 = np.mean(tc_all2[mm1:mm2,:],axis=0)
		tc_sem2 = np.std(tc_all2[mm1:mm2,:],axis=0)/np.sqrt(ncluster_list[rr])


		#---------second set for comparison------------------------
		axs[colnum,rownum].plot(range(tsize2), tc_avg2, linestyle = '-', color = [0.5,0.5,0.5])
		t = np.array(range(tsize2))
		terrfill2 = np.concatenate((t,t[::-1]))
		yerrp2 = tc_avg2 + tc_sem2
		yerrm2 = tc_avg2 - tc_sem2
		yerrfill2 = np.concatenate((yerrp2,yerrm2[::-1]))
		axs[colnum,rownum].fill(terrfill2,yerrfill2, facecolor=(0.5, 0.5, 0.5), edgecolor='None', alpha=0.2)
		#---------end of second set--------------------------------

		# -------------plot the main data-----------------------------------
		# axs[colnum,rownum].errorbar(range(tsize), tc_avg, tc_sem, linestyle = '-', color = [0,0,0])
		axs[colnum,rownum].plot(range(tsize), tc_avg, linestyle = '-', color = [0,0,0])
		axs[colnum,rownum].annotate('{}'.format(rnamelist[rr]), xy=(.025, .975), xycoords='axes fraction',
					horizontalalignment='left', verticalalignment='top', fontsize=10)
		t = np.array(range(tsize))
		terrfill = np.concatenate((t,t[::-1]))
		yerrp = tc_avg + tc_sem
		yerrm = tc_avg - tc_sem
		yerrfill = np.concatenate((yerrp,yerrm[::-1]))
		axs[colnum,rownum].fill(terrfill,yerrfill, facecolor=(0, 0, 0), edgecolor='None', alpha=0.2)
		#------------end of plotting main data------------------------------

		ymax = np.max(tc_avg + tc_sem)
		ymin = np.min(tc_avg - tc_sem)
		ymax = 0.6
		ymin = -0.4
		yyfill = [ymin, ymin, ymax, ymax]
		axs[colnum,rownum].fill(ttfill1,yyfill, facecolor=(1, 1, 0), edgecolor='None', alpha=0.2)
		axs[colnum,rownum].fill(ttfill2,yyfill, facecolor=(0, 1, 0), edgecolor='None', alpha=0.2)


# --------separated by covariate values-----------------------------
	plt.close(windownum+12)
	fig, axs = plt.subplots(2,3,num=windownum+12, sharey=True)

	tlimit = 135
	ncols = 3
	cols = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
	if ncols > 4:
		red = np.linspace(1,-1,ncols)
		red[red < 0] = 0.
		green = 1.0 - np.abs(np.linspace(1,-1,ncols))
		blue = np.linspace(-1,1,ncols)
		blue[blue < 0] = 0.
		cols = np.concatenate((red[:,np.newaxis], green[:,np.newaxis], blue[:,np.newaxis]),axis=1)

	nc1, tsize = np.shape(tc_all)
	# for rr,regionnum in enumerate([3,5,7,9,10,12]):
	for rr,regionnum in enumerate([3,5,7,9,10,12]):
		rownum = np.floor(rr/3).astype(int)
		colnum = rr % 3

		# for cc in range(ncluster_list[rr]):
		mm1 = np.sum(ncluster_list[:(regionnum-1)])
		mm2 = np.sum(ncluster_list[:(regionnum)])

		# for nn in range(np.max(nruns_per_person)):
		# 	# tc_runs_per_person = np.zeros((np.sum(ncluster_list), NP, np.max(nruns_per_person), tsize))
		# 	cc = [xx for xx in range(NP) if (nruns_per_person[xx] > nn) ]
		# 	tc = np.mean(tc_runs_per_person[:,cc,nn,:],axis=1)
		# 	tc_avg = np.mean(tc[mm1:mm2,:],axis=0)
		# 	tc_sem = np.std(tc[mm1:mm2,:],axis=0)/np.sqrt(ncluster_list[regionnum])

		tc_avg_set = np.zeros((tsize,NP))
		tc_sem_set = np.zeros((tsize,NP))
		tc_avg2_set = np.zeros((tsize2,NP2))
		tc_sem2_set = np.zeros((tsize2,NP2))

		for nn in range(NP):
			tc = np.mean(tc_runs_per_person[:,nn,:nruns_per_person[nn],:],axis=1)
			tc_avg_set[:,nn] = np.mean(tc[mm1:mm2,:],axis=0)
			tc_sem_set[:,nn] = np.std(tc[mm1:mm2,:],axis=0)/np.sqrt(ncluster_list[regionnum])

		covariatevalues = copy.deepcopy(compass)
		# covcutoffs = [5, 14]  # wpi
		covcutoffs = [16, 40]  # compass
		# covcutoffs = [10, 22]  # sds
		# covcutoffs = [40, 63]  # fiqr
		# covcutoffs = [5, 20]  # pcs
		# covcutoffs = [30, 38]  # sta1
		# covcutoffs = [32, 44.5]  # sta2
		covname = 'COMPASS'

		vv = np.where(np.isnan(covariatevalues) == False)[0]
		xxsort = vv[np.argsort(covariatevalues[vv])]

		p1 = np.floor(len(vv)/3).astype(int)
		p2 = np.floor(2*len(vv)/3).astype(int)
		p1 = np.where(covariatevalues[xxsort] < covcutoffs[0])[0]
		p2 = np.where( (covariatevalues[xxsort] >= covcutoffs[0]) & (covariatevalues[xxsort] < covcutoffs[1]) )[0]
		p3 = np.where(covariatevalues[xxsort] >= covcutoffs[1])[0]

		cov_avg = [np.mean(covariatevalues[xxsort[p1]]), np.mean(covariatevalues[xxsort[p2]]), np.mean(covariatevalues[xxsort[p3]])]
		cov_std = [np.std(covariatevalues[xxsort[p1]]), np.std(covariatevalues[xxsort[p2]]), np.std(covariatevalues[xxsort[p3]])]
		print('region {}  {} {:.3f} {} {:.3f}  {:.3f} {} {:.3f}  {:.3f} {} {:.3f}'.format(rnamelist[regionnum], covname,
					np.mean(covariatevalues[xxsort[p1]]), chr(177), np.std(covariatevalues[xxsort[p1]]),
					np.mean(covariatevalues[xxsort[p2]]), chr(177), np.std(covariatevalues[xxsort[p2]]),
			  		np.mean(covariatevalues[xxsort[p3]]), chr(177), np.std(covariatevalues[xxsort[p3]]) ) )

		tc_avg = np.zeros((tsize,3))
		tc_sem = np.zeros((tsize,3))
		tc_avg2 = np.zeros((tsize2,3))
		tc_sem2 = np.zeros((tsize2,3))

		tc_avg[:,0] = np.mean(tc_avg_set[:, xxsort[p1]], axis=1)
		tc_sem[:,0] = np.std(tc_avg_set[:, xxsort[p1]], axis=1)/np.sqrt(len(p1))
		tc_avg[:,1] = np.mean(tc_avg_set[:, xxsort[p2]], axis=1)
		tc_sem[:,1] = np.std(tc_avg_set[:, xxsort[p2]], axis=1)/np.sqrt(len(p2))
		tc_avg[:,2] = np.mean(tc_avg_set[:, xxsort[p3]], axis=1)
		tc_sem[:,2] = np.std(tc_avg_set[:, xxsort[p3]], axis=1)/np.sqrt(len(p3))

		# for nn in range(3):
		# 	tc = np.mean(tc_runs_per_person[:,nn,:nruns_per_person[nn],:],axis=1)
		# 	tc_avg_set[:,nn] = np.mean(tc[mm1:mm2,:],axis=0)
		# 	tc_sem_set[:,nn] = np.std(tc[mm1:mm2,:],axis=0)/np.sqrt(ncluster_list[regionnum])
		# xxsort = np.argsort(painsens)

		#
		# for nn in range(NP2):
		# 	tc = np.mean(tc_runs_per_person2[:,nn,:nruns_per_person2[nn],:],axis=1)
		# 	tc_avg2_set[:,nn] = np.mean(tc[mm1:mm2,:],axis=0)
		# 	tc_sem2_set[:,nn] = np.std(tc[mm1:mm2,:],axis=0)/np.sqrt(ncluster_list[regionnum])

		for nn in range(3):
			# tc = np.mean(tc_runs_per_person[:,nn,:nruns_per_person[nn],:],axis=1)
			# tc_avg = np.mean(tc[mm1:mm2,:],axis=0)
			# tc_sem = np.std(tc[mm1:mm2,:],axis=0)/np.sqrt(ncluster_list[regionnum])

			#---------second set for comparison------------------------
			# cc = [xx for xx in range(NP2) if (nruns_per_person2[xx] > nn) ]
			# tc = np.mean(tc_runs_per_person2[:,cc,nn,:],axis=1)
			# tc_avg2 = np.mean(tc[mm1:mm2,:],axis=0)
			# tc_sem2 = np.std(tc[mm1:mm2,:],axis=0)/np.sqrt(ncluster_list[regionnum])

			# tc = np.mean(tc_runs_per_person2[:,nn,:nruns_per_person2[nn],:],axis=1)
			# tc_avg2 = np.mean(tc[mm1:mm2,:],axis=0)
			# tc_sem2 = np.std(tc[mm1:mm2,:],axis=0)/np.sqrt(ncluster_list[regionnum])

			# axs.plot(range(tsize2), tc_avg2, linestyle = '-', color = (cols[nn,:]/2.0))
			#---------end of second set--------------------------------

			# -------------plot the main data-----------------------------------
			# axs[colnum,rownum].errorbar(range(tsize), tc_avg, tc_sem, linestyle = '-', color = [0,0,0])
			axs[rownum, colnum].plot(range(tlimit), tc_avg[:tlimit,nn], linestyle = '-', color = cols[nn,:])

			axs[rownum, colnum].annotate('{:.3f}'.format(cov_avg[nn]), xy = (-1.0, tc_avg[0,nn]),
										 xycoords='data', xytext = (-10,0), textcoords = 'offset points',
										 horizontalalignment='left', verticalalignment='top', fontsize=8)

			#------------end of plotting main data------------------------------
			if nn == 0:
				axs[rownum, colnum].annotate('{}'.format(rnamelist[regionnum]), xy=(.025, .975),
											 xycoords='axes fraction',
											 horizontalalignment='left', verticalalignment='top', fontsize=10)

				ymax = np.max(tc_avg + tc_sem)
				ymin = np.min(tc_avg - tc_sem)
				yyfill = [ymin, ymin, ymax, ymax]
				if tlimit > timeperiod1[0]/TR:
					axs[rownum, colnum].fill(ttfill1,yyfill, facecolor=(1, 0, 0), edgecolor='None', alpha=0.1)
				if tlimit > timeperiod2[0]/TR:
					axs[rownum, colnum].fill(ttfill2,yyfill, facecolor=(0, 1, 0), edgecolor='None', alpha=0.1)

#--------end of separated by covariates-----------------------


	plt.close(windownum+1)
	fig = plt.figure(windownum+1)
	cols = np.array([[1,0,0],[1,0.5,0],[1,1,0],[0,0,1],[0,0.5,1],[0,1,1]])
	nc1, tsize = np.shape(tc_all)
	for nn,rr in enumerate([3,5,7,9,10,12]):
		col = cols[nn,:]
		for cc in range(ncluster_list[rr]):
			mm1 = np.sum(ncluster_list[:(rr-1)])+cc
			plt.plot(range(tsize), tc_all[mm1,:], linestyle = '-', color = col)


	# correlation between pain ratings and initial intensity
	cov = copy.deepcopy(compass)
	cov2 = copy.deepcopy(compass2)

	nc1, tsize = np.shape(tc_all)
	corrlist = np.zeros(nc1)
	fitlist = np.zeros((nc1,3))
	savedata = []

	# xx = np.where(cov < 60)[0]   # for STA1 and STA2
	# xx2 = np.where(cov2 < 60)[0]

	xx = np.where(cov < 100)[0]   # for compass
	xx2 = np.where(cov2 < 100)[0]

	for nn in range(nc1):
		y = np.mean(tc_person[nn,:,:5],axis=1)
		my = np.mean(y[xx])
		cc = np.corrcoef(cov[xx],y[xx])
		m,b = np.polyfit(cov[xx],y[xx],1)
		fit = m*cov[xx] + b
		r2 = 1 - np.sum( (y[xx] - fit)**2)/np.sum((y[xx]-my)**2)
		corrlist[nn] = cc[0,1]
		fitlist[nn,:] = [r2,m,b]

		# second group
		y2 = np.mean(tc_person2[nn,:,:5],axis=1)
		my2 = np.mean(y2[xx2])
		cc2 = np.corrcoef(cov2[xx2],y2[xx2])
		m2,b2 = np.polyfit(cov2[xx2],y2[xx2],1)
		fit2 = m2*cov2[xx2] + b2
		r22 = 1 - np.sum( (y2[xx2] - fit2)**2)/np.sum((y2[xx2]-my2)**2)

		savedata.append({'cov':cov, 'y':y, 'region':nn, 'r2':r2, 'm':m, 'b':b, 'corr':cc[0,1], 'xx':xx,
						 'cov2':cov2, 'y2':y2, 'r22':r22, 'm2':m2, 'b2':b2})

	ncols = NP
	red = np.linspace(1,-1,ncols)
	red[red < 0] = 0.
	green = 1.0 - np.abs(np.linspace(-1,1,ncols))
	blue = np.linspace(-1,1,ncols)
	blue[blue < 0] = 0.
	cols = np.concatenate((red[:,np.newaxis], green[:,np.newaxis], blue[:,np.newaxis]), axis=1)

	nlist = np.zeros(len(ncluster_list), dtype = int)
	for nn in range(len(ncluster_list)):
		r1 = np.sum(ncluster_list[:nn]).astype(int)
		r2 = np.sum(ncluster_list[:(nn+1)])
		rlist = list(range(r1,r2))
		subcorr = corrlist[r1:r2]
		x = np.argmax(np.abs(subcorr))
		print('max corr region {} is {:.3f}'.format(rnamelist[nn],subcorr[x]))
		nlist[nn] = rlist[x]
	nlist = np.array(nlist)

	nclusterend = np.array([0] + list(np.cumsum(ncluster_list)))
	# [2, 7, 9, 12, 24, 25, 44, 47, 83, 89]
	for nn in nlist:
		cc = np.where(nn >= nclusterend)[0]
		rnum = cc[-1]
		cnum = nn - nclusterend[rnum]
		windownum2 = copy.deepcopy(nn)
		# plt.close(windownum2)
		fig,ax = plt.subplots(num = windownum2)
		# ax = fig.add_axes([0,0,1,1])
		# fig.suptitle('{} {}'.format(rnamelist[rnum],cnum))
		cov = savedata[nn]['cov']
		y = savedata[nn]['y']
		m = savedata[nn]['m']
		b = savedata[nn]['b']
		r2 = savedata[nn]['r2']
		fit = m*cov+b

		cov2 = savedata[nn]['cov2']
		y2 = savedata[nn]['y2']
		m2 = savedata[nn]['m2']
		b2 = savedata[nn]['b2']
		r22 = savedata[nn]['r22']
		fit2 = m2*cov2+b2

		# ancova
		statstype = 'ANCOVA'
		formula_key1 = 'C(Group)'
		formula_key2 = 'cov'
		formula_key3 = 'C(Group):cov'
		atype = 2

		ancova_p = np.zeros(3)
		ancova_table, p_MeoG, p_MeoC, p_intGC = py2ndlevelanalysis.run_ANOVA_or_ANCOVA2(y, y2, cov, cov2, covname,
															formula_key1, formula_key2,formula_key3, atype)
		ancova_p = np.array([p_MeoG, p_MeoC, p_intGC])

		plt.plot(cov2, y2, linestyle='', marker = 'o',color = [0.5,0.5,0.5])
		plt.plot(cov2, fit2, linestyle='-', marker = '',color = [0.5,0.5,0.5])

		plt.plot(cov, y, linestyle='', marker = 'o',color = [0,0,0])
		plt.plot(cov, fit, linestyle='-', marker = '',color = [0,0,0])

		if m > 0:
			ax.annotate('{} {}  R2a = {:.3f}   R2b = {:.3f}'.format(rnamelist[rnum],cnum, r2, r22), xy=(.025, .975),
										 xycoords='axes fraction', horizontalalignment='left', verticalalignment='top', fontsize=10)

			ax.annotate('MeoG: {:.3e}  MeoC: {:.3e}  Int.: {:.3e}'.format(p_MeoG, p_MeoC, p_intGC), xy=(.025, .935),
										 xycoords='axes fraction', horizontalalignment='left', verticalalignment='top', fontsize=10)
		else:
			ax.annotate('{} {}  R2a = {:.3f}   R2b = {:.3f}'.format(rnamelist[rnum],cnum, r2, r22), xy=(.975, .975),
										 xycoords='axes fraction', horizontalalignment='right', verticalalignment='top', fontsize=10)
			ax.annotate('MeoG: {:.3e}  MeoC: {:.3e}  Int.: {:.3e}'.format(p_MeoG, p_MeoC, p_intGC), xy=(.975, .935),
										 xycoords='axes fraction', horizontalalignment='right', verticalalignment='top', fontsize=10)

	plt.close(windownum2+1)
	fig = plt.figure(windownum2+1)
	xx = np.argsort(cov)
	for pp in range(NP):
		plt.plot(range(tsize), tc_person[nn,pp,:], linestyle='-',color = cols[xx[pp],:])


# xls = pd.ExcelFile(DBname, engine='openpyxl')
# df1 = pd.read_excel(xls, 'datarecord')
# xlfields = df1.keys()
#
# wpi_vals = df1['wpi']
# participantid = df1['participantid']
# studygroup = df1['studygroup']
# COMPASS = df1['COMPASS']
