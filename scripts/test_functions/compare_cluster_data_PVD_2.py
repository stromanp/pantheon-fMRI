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
import scipy.stats as stats

datatype = 'brain'

region_name = 'Hippocampus'
region_name = 'Hypothalamus'
region_name = 'PAG'
region_name = 'LC'
region_name = 'FOrb'
region_name = 'IC'
region_name = 'Thalamus'

windownum = 126
figsize = (10,3.5)

# datatype1 = copy.deepcopy(datatype)
# grouptype1 = copy.deepcopy(grouptype)
grouptype1 = 'HC'
datatype1 = 'brain'

# datatype2 = copy.deepcopy(datatype)
# grouptype2 = copy.deepcopy(grouptype)
grouptype2 = 'FM'
datatype2 = 'brain'

stimtype1 = 'nostim'
stimtype2 = 'stim'

region_name_list = ['IC','Hypothalamus','Amygdala','Thalamus','PAG','LC']

grouptype_list = ['PVD','HC']
region_name_list = ['IC','Hypothalamus','Amygdala','Thalamus','PAG','LC']

# grouptype_list = ['PVD']
# region_name_list = ['Thalamus']

for grouptype in grouptype_list:
	for region_name in region_name_list:

		grouptype1 = copy.deepcopy(grouptype)
		grouptype2 = copy.deepcopy(grouptype)

		figoutputname = r'F:\PVD_study\SAPMresults\compare_stim_rest_{}{}_Sept2024.svg'.format(grouptype1,region_name)
		figoutputname_strat = r'F:\PVD_study\SAPMresults\compare_stim_rest_{}{}_painsens_strat_Sept2024.svg'.format(grouptype1,region_name)

		color1_plot = [0.4,0.4,0.4]
		color1_fill = [0.4,0.4,0.4]
		color2_plot = [0,0,0]
		color2_fill = [0.,0.,0.]
		marker1 = ' '
		marker2 = ' '


		interpolate_values = False
		offset_starting_value = False
		stratify_results = False

		fixed_vertical_scale = True
		ylims = [-0.6,0.6]

		show_time_periods = True
		timeperiod1 = [60.,70.]
		timeperiod2 = [120.,150.]

		if datatype1 == 'brain':
			clusterdef_name1 = r'F:\PVD_study\SAPMresults\allstim_cluster_def_brain_Jan28_2024_V3.npy'
			regiondata_name1 = r'F:\PVD_study\SAPMresults\{}{}_allstim_V5.npy'.format(grouptype1, stimtype1)

			# results_name1 = r'D:\Howie_FM2_Brain_Data\{}{}_02323203331202_V5_results_corr.npy'.format(grouptype1,stimtype1)
			TR1 = 3.0
			cnum_list1 = [0, 2, 3, 2, 3, 2, 0, 3, 3, 3, 1, 2, 0, 2]

			if grouptype2 == 'HC':
				covariates_file1 = r'F:\PVD_study\SAPMresults\HCstim_02323203331202_results_covariates.npy'
			else:
				covariates_file1 = r'F:\PVD_study\SAPMresults\PVDstim_02323203331202_results_covariates.npy'

		else:
			print('not ready for cord data ...')
			# results_name1 = r'E:\FM2021data\{}{}_2230224124_V5_results_corr.npy'.format(grouptype1,stimtype1)
			TR1 = 6.75
			cnum_list1 = [2, 2, 3, 0, 2, 2, 4, 1, 2, 4]
			covariates_file1 = []

		if datatype2 == 'brain':
			clusterdef_name2 = r'F:\PVD_study\SAPMresults\allstim_cluster_def_brain_Jan28_2024_V3.npy'
			regiondata_name2 = r'F:\PVD_study\SAPMresults\{}{}_allstim_V5.npy'.format(grouptype2, stimtype2)
			# results_name2 = r'D:\Howie_FM2_Brain_Data\{}{}_02323203331202_V5_results_corr.npy'.format(grouptype2,stimtype2)
			TR2 = 3.0
			cnum_list2 = [0, 2, 3, 2, 3, 2, 0, 3, 3, 3, 1, 2, 0, 2]

			if grouptype2 == 'HC':
				covariates_file2 = r'F:\PVD_study\SAPMresults\HCstim_02323203331202_results_covariates.npy'
			else:
				covariates_file2 = r'F:\PVD_study\SAPMresults\PVDstim_02323203331202_results_covariates.npy'

		else:
			print('not ready for cord data ...')
			# results_name2 = r'E:\FM2021data\{}{}_2230224124_V5_results_corr.npy'.format(grouptype2,stimtype2)
			TR2 = 6.75
			cnum_list2 = [2, 2, 3, 0, 2, 2, 4, 1, 2, 4]
			covariates_file2 = []

		toffset = 0.

		if stratify_results:
			covdata1 = np.load(covariates_file1, allow_pickle = True).flat[0]
			covdata2 = np.load(covariates_file2, allow_pickle = True).flat[0]

			x1 = covdata1['GRPcharacteristicslist'].index('painrating')
			x2 = covdata2['GRPcharacteristicslist'].index('painrating')
			painrating1 = covdata1['GRPcharacteristicsvalues'][x1]
			painrating2 = covdata1['GRPcharacteristicsvalues'][x2]

			x1 = covdata1['GRPcharacteristicslist'].index('temperature')
			x2 = covdata2['GRPcharacteristicslist'].index('temperature')
			temperature1 = covdata1['GRPcharacteristicsvalues'][x1]
			temperature2 = covdata1['GRPcharacteristicsvalues'][x2]

			painsens1 = painrating1/temperature1
			painsens2 = painrating2/temperature2

			nstrat = np.floor(len(painsens1)/3.).astype(int)
			x = np.argsort(painsens1)
			g1 = x[:nstrat]
			g2 = x[nstrat:(2*nstrat)]
			g3 = x[(2*nstrat):]
			m1 = np.mean(painsens1[g1])
			s1 = np.std(painsens1[g1])/np.sqrt(len(g1))
			m2 = np.mean(painsens1[g2])
			s2 = np.std(painsens1[g2])/np.sqrt(len(g2))
			m3 = np.mean(painsens1[g3])
			s3 = np.std(painsens1[g3])/np.sqrt(len(g3))
			print('stratified groups  low {:.3f}{}{:.3f} med {:.3f}{}{:.3f} high {:.3f}{}{:.3f}'.format(m1,chr(177),s1, m2,chr(177),s2, m3,chr(177),s3))

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

		# results_data1 = np.load(results_name1, allow_pickle=True)
		# results_data2 = np.load(results_name2, allow_pickle=True)

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
		tc1_per_person = np.zeros((nclusters1, NP1, tsize1))
		for pp in range(NP1):
			r1 = np.sum(nruns_per_person1[:pp])
			r2 = np.sum(nruns_per_person1[:(pp + 1)])
			tc1_per_person[:, pp, :] = np.mean(tc1r[:, r1:r2, :], axis=1)

		tc2r = np.reshape(tc2, (nclusters2, total_nruns2, tsize2))
		tc2_per_person = np.zeros((nclusters2, NP2, tsize2))
		for pp in range(NP2):
			r1 = np.sum(nruns_per_person2[:pp])
			r2 = np.sum(nruns_per_person2[:(pp + 1)])
			tc2_per_person[:, pp, :] = np.mean(tc2r[:, r1:r2, :], axis=1)

		tc1_avg = np.mean(tc1_per_person, axis=1)
		tc2_avg = np.mean(tc2_per_person, axis=1)

		tc1_sem = np.std(tc1_per_person, axis=1) / np.sqrt(NP1)
		tc2_sem = np.std(tc2_per_person, axis=1) / np.sqrt(NP2)

		# tc1_avg = np.mean(np.reshape(tc1, (nclusters1, total_nruns1, tsize1)),axis=1)
		# tc2_avg = np.mean(np.reshape(tc2, (nclusters2, total_nruns2, tsize2)),axis=1)
		#
		# tc1_sem = np.std(np.reshape(tc1, (nclusters1, total_nruns1, tsize1)),axis=1)/np.sqrt(NP1)
		# tc2_sem = np.std(np.reshape(tc2, (nclusters2, total_nruns2, tsize2)),axis=1)/np.sqrt(NP2)


		# # fit results
		# fit_results_perperson1 = np.zeros((NP1,tsize1))
		# for pp in range(NP1):
		# 	# Minput = results_data1[pp]['Minput']
		# 	# Sconn = results_data1[pp]['Sconn']
		# 	# fit_calc = Minput @ Sconn
		# 	fit_temp = copy.deepcopy(results_data1[pp]['Sinput_original'])
		# 	fit_results_perperson1[pp,:] = np.mean(np.reshape(fit_temp[rname_index1,:],(nruns_per_person1[pp],tsize1)),axis=0)
		# fit_tc1_avg = np.mean(fit_results_perperson1,axis = 0)
		# fit_tc1_sem = np.std(fit_results_perperson1,axis = 0)/np.sqrt(NP1)
		#
		# fit_results_perperson2 = np.zeros((NP2,tsize2))
		# for pp in range(NP2):
		# 	fit_temp = copy.deepcopy(results_data2[pp]['Sinput_original'])
		# 	fit_results_perperson2[pp,:] = np.mean(np.reshape(fit_temp[rname_index2,:],(nruns_per_person2[pp],tsize2)),axis=0)
		# fit_tc2_avg = np.mean(fit_results_perperson2,axis = 0)
		# fit_tc2_sem = np.std(fit_results_perperson2,axis = 0)/np.sqrt(NP2)
		#
		# # use fit data
		# use_Sinput_original_data = True
		# if use_Sinput_original_data:
		# 	tc1_avg = np.repeat(fit_tc1_avg[np.newaxis,:],nclusters1,axis=0)
		# 	tc1_sem = np.repeat(fit_tc1_sem[np.newaxis,:],nclusters1,axis=0)
		# 	tc2_avg = np.repeat(fit_tc2_avg[np.newaxis,:],nclusters2,axis=0)
		# 	tc2_sem = np.repeat(fit_tc2_sem[np.newaxis,:],nclusters2,axis=0)


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
		step = np.floor(TR1).astype(int)
		tbaseline = list(range(30,int(timeperiod1[0]),step)) + list(range(int(timeperiod1[1]), int(timeperiod2[0]),step)) + list(range(int(timeperiod2[1]),270,step))
		cbaseline = list((np.array(tbaseline)/TR1).astype(int))

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


		plt.close(windownum)
		fig = plt.figure(windownum, figsize = figsize)
		plt.plot(t2,tc2_avg[cnum2,:] + intensity_offset,linestyle='-',marker = marker2, color = color2_plot)
		plt.xticks(np.arange(0, 300, 30))

		y1 = list(tc2_avg[cnum2,:] + intensity_offset + tc2_sem[cnum2,:])
		y2 = list(tc2_avg[cnum2,:] + intensity_offset - tc2_sem[cnum2,:])
		yy = y1 + y2[::-1]
		xx = list(t2) + list(t2)[::-1]
		plt.fill(xx, yy, facecolor=color2_fill, edgecolor='None', alpha=0.2)

		if show_time_periods:
			ymin1 = np.min(yy)
			ymax1 = np.max(yy)

		if interpolate_values:
			plt.plot(t2,tc1_avg_interp,linestyle='-',marker = marker1, color = color1_plot)
			plt.xticks(np.arange(0, 300, 30))
			R = np.corrcoef(tc2_avg[cnum2,:], tc1_avg_interp)
			print('correlation betwen TC2 and interpolated TC1 is {:.3f}'.format(R[0,1]))
			Z = np.arctanh(R[0,1])*np.sqrt(tsize2-3)
			Zthresh = stats.norm.ppf(1 - np.array([1.0, 0.05, 0.01, 0.001]))
			p = 1.0 - stats.norm.cdf(Z)

			y1 = list(tc1_avg_interp + tc1_sem_interp)
			y2 = list(tc1_avg_interp - tc1_sem_interp)
			yy = y1 + y2[::-1]
			xx = list(t2) + list(t2)[::-1]
		else:
			plt.plot(t1,tc1_avg[cnum1,:],linestyle='-',marker = marker1, color = color1_plot)
			plt.xticks(np.arange(0, 300, 30))

			y1 = list(tc1_avg[cnum1,:] + tc1_sem[cnum1,:])
			y2 = list(tc1_avg[cnum1,:] - tc1_sem[cnum1,:])
			yy = y1 + y2[::-1]
			xx = list(t1) + list(t1)[::-1]

		plt.fill(xx, yy, facecolor=color1_fill, edgecolor='None', alpha=0.2)

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

		if fixed_vertical_scale:
			plt.ylim(ylims[0],ylims[1])

		plt.savefig(figoutputname)


		#-------------------------------------------------------------------
		if stratify_results:

			fit_tc1_avg_1 = np.mean(fit_results_perperson1[g1,:], axis=0)
			fit_tc1_sem_1 = np.std(fit_results_perperson1[g1,:], axis=0) / np.sqrt(len(g1))

			fit_tc1_avg_2 = np.mean(fit_results_perperson1[g2,:], axis=0)
			fit_tc1_sem_2 = np.std(fit_results_perperson1[g2,:], axis=0) / np.sqrt(len(g2))

			fit_tc1_avg_3 = np.mean(fit_results_perperson1[g3,:], axis=0)
			fit_tc1_sem_3 = np.std(fit_results_perperson1[g3,:], axis=0) / np.sqrt(len(g3))

			# now interpolate to the same number of volumes
			toffset = 0.0
			t1 = np.array(range(tsize1)) * TR1 + TR1 / 2.0
			t2 = np.array(range(tsize2)) * TR2 + TR2 / 2.0 + toffset


			initial_volumes = 5
			offset_starting_value = False
			if offset_starting_value:
				startingval1 = np.mean(fit_tc1_avg_1[:initial_volumes])
				startingval2 = np.mean(fit_tc1_avg_2[:initial_volumes])
				startingval3 = np.mean(fit_tc1_avg_3[:initial_volumes])
				refval = np.max(np.array([startingval1, startingval2, startingval3]))

				intensity_offset1 = startingval1 - refval
				intensity_offset2 = startingval2 - refval
				intensity_offset3 = startingval3 - refval
			else:
				intensity_offset1 = 0.0
				intensity_offset2 = 0.0
				intensity_offset3 = 0.0

			# get size of initial rise
			tbaseline = list(range(30, int(timeperiod1[0]), 2)) + list(
				range(int(timeperiod1[1]), int(timeperiod2[0]), 2)) + list(range(int(timeperiod2[1]), 270, 2))
			cbaseline = list((np.array(tbaseline) / 2.0).astype(int))

			startingval1 = np.mean(fit_tc1_avg_1[:initial_volumes])
			startingval1_sem = np.std(fit_tc1_avg_1[:initial_volumes])
			baselineval1 = np.mean(fit_tc1_avg_1[cbaseline])
			baselineval1_sem = np.std(fit_tc1_avg_1[cbaseline])

			startingval2 = np.mean(fit_tc1_avg_2[:initial_volumes])
			startingval2_sem = np.std(fit_tc1_avg_2[:initial_volumes])
			baselineval2 = np.mean(fit_tc1_avg_2[cbaseline])
			baselineval2_sem = np.std(fit_tc1_avg_2[cbaseline])

			startingval3 = np.mean(fit_tc1_avg_3[:initial_volumes])
			startingval3_sem = np.std(fit_tc1_avg_3[:initial_volumes])
			baselineval3 = np.mean(fit_tc1_avg_3[cbaseline])
			baselineval3_sem = np.std(fit_tc1_avg_3[cbaseline])

			print('{} {} {}: group1 {:.3f} {} {:.3f} change from initial to baseline = {:.4f} {} {:.4f}'.format(region_name, grouptype1,
							stimtype1, m1, chr(177), s1, baselineval1 - startingval1, chr(177), baselineval1_sem))
			print('{} {} {}: group2 {:.3f} {} {:.3f} change from initial to baseline = {:.4f} {} {:.4f}'.format(region_name, grouptype1,
							stimtype1, m2, chr(177), s2, baselineval2 - startingval2, chr(177), baselineval2_sem))
			print('{} {} {}: group3 {:.3f} {} {:.3f} change from initial to baseline = {:.4f} {} {:.4f}'.format(region_name, grouptype1,
							stimtype1, m3, chr(177), s3, baselineval3 - startingval3, chr(177), baselineval3_sem))

			stratcolor1 = [1,0,0]
			stratcolor2 = [0,1,0]
			stratcolor3 = [0,0,1]
			plt.close(windownum+10)
			fig = plt.figure(windownum+10, figsize=figsize)

			# group1
			plt.plot(t1, fit_tc1_avg_1 + intensity_offset1, linestyle='-', marker=marker1, color=stratcolor1)
			plt.xticks(np.arange(0, 300, 30))

			y1 = list(fit_tc1_avg_1 + intensity_offset1 + fit_tc1_sem_1)
			y2 = list(fit_tc1_avg_1 + intensity_offset1 - fit_tc1_sem_1)
			yy = y1 + y2[::-1]
			xx = list(t2) + list(t2)[::-1]
			plt.fill(xx, yy, facecolor=stratcolor1, edgecolor='None', alpha=0.2)

			if show_time_periods:
				ymin1 = np.min(yy)
				ymax1 = np.max(yy)

			# group2
			plt.plot(t1, fit_tc1_avg_2 + intensity_offset2, linestyle='-', marker=marker1, color=stratcolor2)
			plt.xticks(np.arange(0, 300, 30))

			y1 = list(fit_tc1_avg_2 + intensity_offset2 + fit_tc1_sem_2)
			y2 = list(fit_tc1_avg_2 + intensity_offset1 - fit_tc1_sem_2)
			yy = y1 + y2[::-1]
			xx = list(t2) + list(t2)[::-1]
			plt.fill(xx, yy, facecolor=stratcolor2, edgecolor='None', alpha=0.2)

			if show_time_periods:
				ymin2 = np.min(yy)
				ymax2 = np.max(yy)

			# group3
			plt.plot(t1, fit_tc1_avg_3 + intensity_offset3, linestyle='-', marker=marker1, color=stratcolor3)
			plt.xticks(np.arange(0, 300, 30))

			y1 = list(fit_tc1_avg_3 + intensity_offset3 + fit_tc1_sem_3)
			y2 = list(fit_tc1_avg_3 + intensity_offset3 - fit_tc1_sem_3)
			yy = y1 + y2[::-1]
			xx = list(t2) + list(t2)[::-1]
			plt.fill(xx, yy, facecolor=stratcolor3, edgecolor='None', alpha=0.2)

			if show_time_periods:
				ymin3 = np.min(yy)
				ymax3 = np.max(yy)

			if show_time_periods:
				ymin = np.min([ymin1, ymin2, ymin3])
				ymax = np.max([ymax1, ymax2, ymax3])

				tp = copy.deepcopy(timeperiod1)
				xx = [tp[0], tp[1], tp[1], tp[0]]
				yy = [ymin, ymin, ymax, ymax]
				plt.fill(xx, yy, facecolor=[1,1,0], edgecolor='None', alpha=0.2)

				tp = copy.deepcopy(timeperiod2)
				xx = [tp[0], tp[1], tp[1], tp[0]]
				plt.fill(xx, yy, facecolor=[0,1,0.3], edgecolor='None', alpha=0.2)

			plt.xticks(np.arange(0, 300, 30))

			if fixed_vertical_scale:
				plt.ylim(ylims[0], ylims[1])

			plt.savefig(figoutputname_strat)


# # fit a curve to the first minute of data
# expfit1,m1,b1 = fit_exp_rise(t1,tc1_avg[cnum1, :], 22.0)
# expfit2,m2,b2 = fit_exp_rise(t2,tc2_avg[cnum2, :], 22.0)
#
# print('1. exp scale: {:.3f}    offset {:.3f} '.format(m1, b1))
# print('2. exp scale: {:.3f}    offset {:.3f} '.format(m2, b2))
#
# plt.plot(t1, expfit1, linestyle='-', linewidth = 3, marker='', color=color1)
# plt.plot(t2, expfit2, linestyle='-', linewidth = 3, marker='', color=color2)





#
# temp_tc1 = tc1_avg[cnum1,:]
# tt1 = np.where(temp_tc1 >= -1e-2)[0][0]
#
# temp_tc2 = tc2_avg[cnum2,:]
# tt2 = np.where(temp_tc2 >= -1e-2)[0][0]
#
# plt.close(windownum+10)
# fig = plt.figure(windownum+10)
# plt.plot(t1[:tt1],np.log(np.abs(np.array(temp_tc1[:tt1]))),linestyle='',marker = marker1, color = color1)
# plt.plot(t2[:tt2],np.log(np.abs(temp_tc2[:tt2])),linestyle='',marker = marker2, color = color2)
#
# m, b = np.polyfit(t1[:tt1],np.log(np.abs(np.array(temp_tc1[:tt1]))), 1)
# fit1 = m*t1[:tt1] + b
# plt.plot(t1[:tt1],fit1,linestyle='-',marker = '', color = color1)
#
# m, b = np.polyfit(t2[:tt2],np.log(np.abs(np.array(temp_tc2[:tt2]))), 1)
# fit2 = m*t2[:tt2] + b
# plt.plot(t2[:tt2],fit2,linestyle='-',marker = '', color = color2)
#
#
# linfit2 = -1.0*np.exp(b)*np.exp(t2*m)


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
	fig = plt.figure(windownum+1, figsize = figsize)
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
	fig = plt.figure(windownum2+1, figsize = figsize)
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
