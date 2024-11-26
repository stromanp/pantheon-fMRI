# sys.path.extend([r'C:\Users\Stroman\PycharmProjects\pantheon\venv'])
# sys.path.extend([r'C:\Users\Stroman\PycharmProjects\pantheon\venv\test_functions'])

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyclustering
import pydatabase
import load_templates
import copy
import image_operations_3D as i3d
import os
import pandas as pd
import nibabel as nib
from scipy import interpolate
import py2ndlevelanalysis


region_name = 'FOrb'
region_name = 'IC'
region_name = 'Hypothalamus'
region_name = 'Hippocampus'
region_name = 'LC'
region_name = 'Thalamus'
region_name = 'PAG'
windownum = 107

datatype1 = 'brain'
grouptype1 = 'FM'
stimtype1 = 'stim'

SNRIgroup = ['HW2018_002_FMS_Stim', 'HW2018_003_FMS_Stim', 'HW2018_007_FMS_Stim', 'HW2018_007redo_FMS_Stim', 'HW2018_011_FMS_Stim', 'HW2018_013_FMS_Stim',
       'HW2018_016_FMS_Stim', 'HW2018_023_FMS_Stim', 'HW2018_025_FMS_Stim', 'HW2018_027_FMS_Stim', 'HW2018_035_FMS_Stim']

color1 = [1,0,0]
color2 = [0,0,1]
marker1 = 'o'
marker2 = 'o'


interpolate_values = False

show_time_periods = True
timeperiod1 = [60.,70.]
timeperiod2 = [120.,150.]
TR = 2.0

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

toffset = 0.

#--------copy values and sort out what to plot----------

clusterdef1 = np.load(clusterdef_name1, allow_pickle=True).flat[0]
regiondata1 = np.load(regiondata_name1, allow_pickle=True).flat[0]

regiondata1 = regiondata1['region_properties']

DBname = regiondata1[0]['DBname']
DBnum = regiondata1[0]['DBnum']
namelist, dbnumlist, NP = pydatabase.get_datanames_by_person(DBname, DBnum, 'xptc')
dbkeylist = dbnumlist.keys()
SNRIdata = [pidname in SNRIgroup for pidname in dbkeylist]

group1 = [xx for xx in range(len(SNRIdata)) if SNRIdata[xx] == True]
group2 = [xx for xx in range(len(SNRIdata)) if SNRIdata[xx] == False]


rname_list1 = [regiondata1[xx]['rname'] for xx in range(len(regiondata1))]
rname_index1 = rname_list1.index(region_name)
cnum1 = cnum_list1[rname_index1]
print('1) getting data for region {},  index number {}'.format(region_name, rname_index1))

tc1 = regiondata1[rname_index1]['tc']
tsize1 = regiondata1[rname_index1]['tsize']
nruns_per_person1 = regiondata1[rname_index1]['nruns_per_person']
total_nruns1 = np.sum(nruns_per_person1)
nclusters1, tsize_total1 = np.shape(tc1)
NP1 = len(nruns_per_person1)

# need to use only the runs for each group ...
cumulative_nruns = np.concatenate((np.array([0]), np.cumsum(nruns_per_person1)),axis=0)
group1_runs = []
for xx in group1:
	s1 = cumulative_nruns[xx]
	s2 = cumulative_nruns[xx+1]
	oneperson = [aa for aa in range(s1,s2)]
	group1_runs += oneperson

group2_runs = []
for xx in group2:
	s1 = cumulative_nruns[xx]
	s2 = cumulative_nruns[xx+1]
	oneperson = [aa for aa in range(s1,s2)]
	group2_runs += oneperson

tc1r = np.reshape(tc1, (nclusters1, total_nruns1, tsize1))

tc1_avg = np.mean(tc1r[:,group1_runs,:],axis=1)
tc1_sem = np.std(tc1r[:,group1_runs,:],axis=1)/np.sqrt(len(group1))

tc2_avg = np.mean(tc1r[:,group2_runs,:],axis=1)
tc2_sem = np.std(tc1r[:,group2_runs,:],axis=1)/np.sqrt(len(group2))

tt = np.array(range(tsize1))*TR

plt.close(windownum)
fig = plt.figure(windownum)
plt.plot(tt,tc2_avg[cnum1,:],linestyle='-',marker = marker2, color = color2)

y1 = list(tc2_avg[cnum1,:] + tc2_sem[cnum1,:])
y2 = list(tc2_avg[cnum1,:] - tc2_sem[cnum1,:])
yy = y1 + y2[::-1]
xx = list(tt) + list(tt)[::-1]
plt.fill(xx, yy, facecolor=color2, edgecolor='None', alpha=0.2)

if show_time_periods:
	ymin1 = np.min(yy)
	ymax1 = np.max(yy)


plt.plot(tt,tc1_avg[cnum1,:],linestyle='-',marker = marker1, color = color1)

y1 = list(tc1_avg[cnum1,:] + tc1_sem[cnum1,:])
y2 = list(tc1_avg[cnum1,:] - tc1_sem[cnum1,:])
yy = y1 + y2[::-1]
xx = list(tt) + list(tt)[::-1]

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


def look_at_cluster_data(regiondataname):
	datatype = 'FM'
	windownum = 1

	if datatype == 'FM':
		regiondataname = r'D:/Howie_FM2_Brain_Data/FMstim_region_data_allstim_Jan28_2024_V5_19.npy'
		covariatesname = r'D:/Howie_FM2_Brain_Data/FMstim_02323203331202_V5_19_results_covariates.npy'

		regiondataname2 = r'D:/Howie_FM2_Brain_Data/HCstim_region_data_allstim_Jan28_2024_V5.npy'
		covariatesname2 = r'D:/Howie_FM2_Brain_Data/HCstim_02323203331202_V5_results_covariates.npy'

	else:
		regiondataname = r'D:/Howie_FM2_Brain_Data/HCstim_region_data_allstim_Jan28_2024_V5.npy'
		covariatesname = r'D:/Howie_FM2_Brain_Data/HCstim_02323203331202_V5_results_covariates.npy'

		regiondataname2 = r'D:/Howie_FM2_Brain_Data/FMstim_region_data_allstim_Jan28_2024_V5_19.npy'
		covariatesname2 = r'D:/Howie_FM2_Brain_Data/FMstim_02323203331202_V5_19_results_covariates.npy'
	TR = 2.0

	data = np.load(regiondataname, allow_pickle=True).flat[0]
	regiondata = data['region_properties']
	rnamelist = [regiondata[xx]['rname'] for xx in range(len(regiondata))]

	covariates = np.load(covariatesname, allow_pickle=True).flat[0]
	painindex = covariates['GRPcharacteristicslist'].index('painrating')
	tempindex = covariates['GRPcharacteristicslist'].index('temperature')
	wpiindex = covariates['GRPcharacteristicslist'].index('wpi')
	compindex = covariates['GRPcharacteristicslist'].index('COMPASS')
	sdsindex = covariates['GRPcharacteristicslist'].index('SDS')
	fiqrindex = covariates['GRPcharacteristicslist'].index('FIQR_SIQR')
	sta1index = covariates['GRPcharacteristicslist'].index('STAI_Y_1')
	sta2index = covariates['GRPcharacteristicslist'].index('STAI_Y_2')
	pcsindex = covariates['GRPcharacteristicslist'].index('PCS_Total')

	painratings = covariates['GRPcharacteristicsvalues'][painindex,:]
	temperatures = covariates['GRPcharacteristicsvalues'][tempindex,:]
	wpi = covariates['GRPcharacteristicsvalues'][wpiindex,:]
	compass = covariates['GRPcharacteristicsvalues'][compindex,:]
	sds = covariates['GRPcharacteristicsvalues'][sdsindex,:]
	fiqr = covariates['GRPcharacteristicsvalues'][fiqrindex,:]
	sta1 = covariates['GRPcharacteristicsvalues'][sta1index,:]
	sta2 = covariates['GRPcharacteristicsvalues'][sta2index,:]
	pcs = covariates['GRPcharacteristicsvalues'][pcsindex,:]


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
	painindex = covariates2['GRPcharacteristicslist'].index('painrating')
	tempindex = covariates2['GRPcharacteristicslist'].index('temperature')
	wpiindex = covariates2['GRPcharacteristicslist'].index('wpi')
	compindex = covariates2['GRPcharacteristicslist'].index('COMPASS')
	# sdsindex = covariates2['GRPcharacteristicslist'].index('SDS')
	fiqrindex = covariates2['GRPcharacteristicslist'].index('FIQR_SIQR')
	sta1index = covariates2['GRPcharacteristicslist'].index('STAI_Y_1')
	sta2index = covariates2['GRPcharacteristicslist'].index('STAI_Y_2')
	pcsindex = covariates2['GRPcharacteristicslist'].index('PCS_Total')

	painratings2 = covariates2['GRPcharacteristicsvalues'][painindex,:]
	temperatures2 = covariates2['GRPcharacteristicsvalues'][tempindex,:]
	wpi2 = covariates2['GRPcharacteristicsvalues'][wpiindex,:]
	compass2 = covariates2['GRPcharacteristicsvalues'][compindex,:]
	# sds2 = covariates2['GRPcharacteristicsvalues'][sdsindex,:]
	fiqr2 = covariates2['GRPcharacteristicsvalues'][fiqrindex,:]
	sta12 = covariates2['GRPcharacteristicsvalues'][sta1index,:]
	sta22 = covariates2['GRPcharacteristicsvalues'][sta2index,:]
	pcs2 = covariates2['GRPcharacteristicsvalues'][pcsindex,:]

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
		axs[colnum,rownum].fill(terrfill2,yerrfill2, facecolor=(0, 0, 0.5), edgecolor='None', alpha=0.2)
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
		axs[colnum,rownum].fill(terrfill,yerrfill, facecolor=(0, 0, 1), edgecolor='None', alpha=0.2)
		#------------end of plotting main data------------------------------

		ymax = np.max(tc_avg + tc_sem)
		ymin = np.min(tc_avg - tc_sem)
		yyfill = [ymin, ymin, ymax, ymax]
		axs[colnum,rownum].fill(ttfill1,yyfill, facecolor=(1, 0, 0), edgecolor='None', alpha=0.2)
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
