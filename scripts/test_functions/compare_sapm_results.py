
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import nibabel as nib
import copy
from scipy import interpolate



def compare_SAPM_timecourses(SAPMresultsname1, SAPMparamsname1, rname1, TR1,
							 SAPMresultsname2, SAPMparamsname2, rname2, TR2, relative_time_shift = 0, windownum = 20):

	results1 = np.load(SAPMresultsname1, allow_pickle=True)
	params1 = np.load(SAPMparamsname1, allow_pickle=True).flat[0]
	NP1 = len(results1)
	rnamelist1 = params1['rnamelist']
	nregions1 = params1['nregions']
	nruns_per_person1 = params1['nruns_per_person']
	fintrinsic_count1 = params1['fintrinsic_count']
	vintrinsic_count1 = params1['vintrinsic_count']
	Nintrinsic1 = fintrinsic_count1 + vintrinsic_count1
	nbeta1, tsize_full1 = np.shape(results1[0]['Sconn'])
	ncon1 = nbeta1 - Nintrinsic1

	results2 = np.load(SAPMresultsname2, allow_pickle=True)
	params2 = np.load(SAPMparamsname2, allow_pickle=True).flat[0]
	NP2 = len(results2)
	rnamelist2 = params2['rnamelist']
	nregions2 = params2['nregions']
	nruns_per_person2 = params2['nruns_per_person']
	fintrinsic_count2 = params2['fintrinsic_count']
	vintrinsic_count2 = params2['vintrinsic_count']
	Nintrinsic2 = fintrinsic_count2 + vintrinsic_count2
	nbeta2, tsize_full2 = np.shape(results2[0]['Sconn'])
	ncon2 = nbeta2 - Nintrinsic2

	rnamelist1_ext = copy.deepcopy(rnamelist1)
	for nn in range(Nintrinsic1):
		rnamelist1_ext += ['latent{}'.format(nn)]

	rnamelist2_ext = copy.deepcopy(rnamelist2)
	for nn in range(Nintrinsic2):
		rnamelist2_ext += ['latent{}'.format(nn)]

	rnum1 = rnamelist1_ext.index(rname1)
	rnum2 = rnamelist2_ext.index(rname2)

	nruns_total1 = np.sum(nruns_per_person1)
	nruns_total2 = np.sum(nruns_per_person2)

	Sinput1 = results1[0]['Sinput']
	nr1, tsize_total1 = np.shape(Sinput1)
	tsize1 = (tsize_total1 / nruns_per_person1[0]).astype(int)

	Sinput2 = results2[0]['Sinput']
	nr2, tsize_total2 = np.shape(Sinput2)
	tsize2 = (tsize_total2 / nruns_per_person2[0]).astype(int)

	Sinput1_total = np.zeros((tsize1,NP1))
	Soutput1_total = np.zeros((tsize1,NP1))
	Sinput2_total = np.zeros((tsize2,NP2))
	Soutput2_total = np.zeros((tsize2,NP2))

	for nperson in range(NP1):
		nruns1 = nruns_per_person1[nperson]

		Sinput1 = results1[nperson]['Sinput']
		fit1 = results1[nperson]['fit']
		Sconn1 = results1[nperson]['Sconn']

		tc_input1 = np.mean(np.reshape(Sinput1, (nr1, nruns1, tsize1)), axis=1)
		tc_output1 = np.mean(np.reshape(Sconn1, (nr1+Nintrinsic1, nruns1, tsize1)), axis=1)

		if rnum1 < nr1:
			Sinput1_total[:, nperson] = tc_input1[rnum1,:]
		else:
			Sinput1_total[:, nperson] = tc_output1[rnum1,:]

		Soutput1_total[:, nperson] = tc_output1[rnum1,:]

	for nperson in range(NP2):
		nruns2 = nruns_per_person2[nperson]

		Sinput2 = results2[nperson]['Sinput']
		fit2 = results2[nperson]['fit']
		Sconn2 = results2[nperson]['Sconn']

		tc_input2 = np.mean(np.reshape(Sinput2, (nr2, nruns2, tsize2)), axis=1)
		tc_output2 = np.mean(np.reshape(Sconn2, (nr2 + Nintrinsic2, nruns2, tsize2)), axis=1)

		if rnum2 < nr2:
			Sinput2_total[:, nperson] = tc_input2[rnum2, :]
		else:
			Sinput2_total[:, nperson] = tc_output2[rnum2, :]

		Soutput2_total[:, nperson] = tc_output2[rnum2, :]

	Sinput1_avg = np.mean(Sinput1_total,axis=1)
	Soutput1_avg = np.mean(Soutput1_total,axis=1)
	Sinput2_avg = np.mean(Sinput2_total,axis=1)
	Soutput2_avg = np.mean(Soutput2_total,axis=1)

	# now interpolate to the same number of volumes
	t1 = np.array(range(tsize1))*TR1 + TR1/2.0
	t2 = np.array(range(tsize2))*TR2 + TR2/2.0 + relative_time_shift

	f = interpolate.interp1d(t1, Sinput1_avg, fill_value = 'extrapolate')
	Sinput1_avg_interp = f(t2)

	CC = np.corrcoef(Sinput2_avg, Sinput1_avg_interp)
	print('input correlation is {:.4f}'.format(CC[0,1]))

	CC1 = np.corrcoef(Sinput2_avg[:17], Sinput1_avg_interp[:17])
	print('   input correlation 0-17 is {:.4f}'.format(CC1[0,1]))
	CC1 = np.corrcoef(Sinput2_avg[:23], Sinput1_avg_interp[:23])
	print('   input correlation 0-23 is {:.4f}'.format(CC1[0,1]))
	CC1 = np.corrcoef(Sinput2_avg[7:23], Sinput1_avg_interp[7:23])
	print('   input correlation 7-23 is {:.4f}'.format(CC1[0,1]))
	CC1 = np.corrcoef(Sinput2_avg[12:29], Sinput1_avg_interp[12:29])
	print('   input correlation 12-29 is {:.4f}'.format(CC1[0,1]))
	CC1 = np.corrcoef(Sinput2_avg[23:], Sinput1_avg_interp[23:])
	print('   input correlation 23-39 is {:.4f}'.format(CC1[0,1]))

	f = interpolate.interp1d(t1, Soutput1_avg, fill_value='extrapolate')
	Soutput1_avg_interp = f(t2)

	CC = np.corrcoef(Soutput2_avg, Soutput1_avg_interp)
	print('output correlation is {:.4f}'.format(CC[0, 1]))

	CC1 = np.corrcoef(Soutput2_avg[:17], Soutput1_avg_interp[:17])
	print('   output correlation 0-17 is {:.4f}'.format(CC1[0,1]))
	CC1 = np.corrcoef(Soutput2_avg[:23], Soutput1_avg_interp[:23])
	print('   output correlation 0-23 is {:.4f}'.format(CC1[0,1]))
	CC1 = np.corrcoef(Soutput2_avg[7:23], Soutput1_avg_interp[7:23])
	print('   output correlation 7-23 is {:.4f}'.format(CC1[0,1]))
	CC1 = np.corrcoef(Soutput2_avg[12:29], Soutput1_avg_interp[12:29])
	print('   output correlation 12-29 is {:.4f}'.format(CC1[0,1]))
	CC1 = np.corrcoef(Soutput2_avg[23:], Soutput1_avg_interp[23:])
	print('   output correlation 23-39 is {:.4f}'.format(CC1[0,1]))

	plot_input = True
	if plot_input:
		plt.close(windownum)
		fig = plt.figure(windownum)
		plt.plot(range(tsize2), Sinput1_avg_interp, '-xr')
		plt.plot(range(tsize2), Sinput2_avg, '-b')

	plot_output = True
	if plot_output:
		plt.close(windownum+1)
		fig = plt.figure(windownum+1)
		plt.plot(range(tsize2), Soutput1_avg_interp, '-xr')
		plt.plot(range(tsize2), Soutput2_avg, '-b')





def compare_cluster_data(clusterdataname1, clusterdataname2, rname1, TR1, rname2, TR2, relative_time_shift):

	results1 = np.load(SAPMresultsname1, allow_pickle=True)
	params1 = np.load(SAPMparamsname1, allow_pickle=True).flat[0]
	NP1 = len(results1)
	rnamelist1 = params1['rnamelist']
	nregions1 = params1['nregions']
	nruns_per_person1 = params1['nruns_per_person']
	fintrinsic_count1 = params1['fintrinsic_count']
	vintrinsic_count1 = params1['vintrinsic_count']
	Nintrinsic1 = fintrinsic_count1 + vintrinsic_count1
	nbeta1, tsize_full1 = np.shape(results1[0]['Sconn'])
	ncon1 = nbeta1 - Nintrinsic1

	results2 = np.load(SAPMresultsname2, allow_pickle=True)
	params2 = np.load(SAPMparamsname2, allow_pickle=True).flat[0]
	NP2 = len(results2)
	rnamelist2 = params2['rnamelist']
	nregions2 = params2['nregions']
	nruns_per_person2 = params2['nruns_per_person']
	fintrinsic_count2 = params2['fintrinsic_count']
	vintrinsic_count2 = params2['vintrinsic_count']
	Nintrinsic2 = fintrinsic_count2 + vintrinsic_count2
	nbeta2, tsize_full2 = np.shape(results2[0]['Sconn'])
	ncon2 = nbeta2 - Nintrinsic2

	rnamelist1_ext = copy.deepcopy(rnamelist1)
	for nn in range(Nintrinsic1):
		rnamelist1_ext += ['latent{}'.format(nn)]

	rnamelist2_ext = copy.deepcopy(rnamelist2)
	for nn in range(Nintrinsic2):
		rnamelist2_ext += ['latent{}'.format(nn)]

	rnum1 = rnamelist1_ext.index(rname1)
	rnum2 = rnamelist2_ext.index(rname2)

	nruns_total1 = np.sum(nruns_per_person1)
	nruns_total2 = np.sum(nruns_per_person2)

	Sinput1 = results1[0]['Sinput']
	nr1, tsize_total1 = np.shape(Sinput1)
	tsize1 = (tsize_total1 / nruns_per_person1[0]).astype(int)

	Sinput2 = results2[0]['Sinput']
	nr2, tsize_total2 = np.shape(Sinput2)
	tsize2 = (tsize_total2 / nruns_per_person2[0]).astype(int)

	Sinput1_total = np.zeros((tsize1,NP1))
	Soutput1_total = np.zeros((tsize1,NP1))
	Sinput2_total = np.zeros((tsize2,NP2))
	Soutput2_total = np.zeros((tsize2,NP2))

	for nperson in range(NP1):
		nruns1 = nruns_per_person1[nperson]

		Sinput1 = results1[nperson]['Sinput']
		fit1 = results1[nperson]['fit']
		Sconn1 = results1[nperson]['Sconn']

		tc_input1 = np.mean(np.reshape(Sinput1, (nr1, nruns1, tsize1)), axis=1)
		tc_output1 = np.mean(np.reshape(Sconn1, (nr1+Nintrinsic1, nruns1, tsize1)), axis=1)

		if rnum1 < nr1:
			Sinput1_total[:, nperson] = tc_input1[rnum1,:]
		else:
			Sinput1_total[:, nperson] = tc_output1[rnum1,:]

		Soutput1_total[:, nperson] = tc_output1[rnum1,:]

	for nperson in range(NP2):
		nruns2 = nruns_per_person2[nperson]

		Sinput2 = results2[nperson]['Sinput']
		fit2 = results2[nperson]['fit']
		Sconn2 = results2[nperson]['Sconn']

		tc_input2 = np.mean(np.reshape(Sinput2, (nr2, nruns2, tsize2)), axis=1)
		tc_output2 = np.mean(np.reshape(Sconn2, (nr2 + Nintrinsic2, nruns2, tsize2)), axis=1)

		if rnum2 < nr2:
			Sinput2_total[:, nperson] = tc_input2[rnum2, :]
		else:
			Sinput2_total[:, nperson] = tc_output2[rnum2, :]

		Soutput2_total[:, nperson] = tc_output2[rnum2, :]

	Sinput1_avg = np.mean(Sinput1_total,axis=1)
	Soutput1_avg = np.mean(Soutput1_total,axis=1)
	Sinput2_avg = np.mean(Sinput2_total,axis=1)
	Soutput2_avg = np.mean(Soutput2_total,axis=1)

	# now interpolate to the same number of volumes
	t1 = np.array(range(tsize1))*TR1 + TR1/2.0
	t2 = np.array(range(tsize2))*TR2 + TR2/2.0 + relative_time_shift

	f = interpolate.interp1d(t1, Sinput1_avg, fill_value = 'extrapolate')
	Sinput1_avg_interp = f(t2)

	CC = np.corrcoef(Sinput2_avg, Sinput1_avg_interp)
	print('input correlation is {:.4f}'.format(CC[0,1]))

	CC1 = np.corrcoef(Sinput2_avg[:17], Sinput1_avg_interp[:17])
	print('   input correlation 0-17 is {:.4f}'.format(CC1[0,1]))
	CC1 = np.corrcoef(Sinput2_avg[:23], Sinput1_avg_interp[:23])
	print('   input correlation 0-23 is {:.4f}'.format(CC1[0,1]))
	CC1 = np.corrcoef(Sinput2_avg[7:23], Sinput1_avg_interp[7:23])
	print('   input correlation 7-23 is {:.4f}'.format(CC1[0,1]))
	CC1 = np.corrcoef(Sinput2_avg[12:29], Sinput1_avg_interp[12:29])
	print('   input correlation 12-29 is {:.4f}'.format(CC1[0,1]))
	CC1 = np.corrcoef(Sinput2_avg[23:], Sinput1_avg_interp[23:])
	print('   input correlation 23-39 is {:.4f}'.format(CC1[0,1]))

	f = interpolate.interp1d(t1, Soutput1_avg, fill_value='extrapolate')
	Soutput1_avg_interp = f(t2)

	CC = np.corrcoef(Soutput2_avg, Soutput1_avg_interp)
	print('output correlation is {:.4f}'.format(CC[0, 1]))

	CC1 = np.corrcoef(Soutput2_avg[:17], Soutput1_avg_interp[:17])
	print('   output correlation 0-17 is {:.4f}'.format(CC1[0,1]))
	CC1 = np.corrcoef(Soutput2_avg[:23], Soutput1_avg_interp[:23])
	print('   output correlation 0-23 is {:.4f}'.format(CC1[0,1]))
	CC1 = np.corrcoef(Soutput2_avg[7:23], Soutput1_avg_interp[7:23])
	print('   output correlation 7-23 is {:.4f}'.format(CC1[0,1]))
	CC1 = np.corrcoef(Soutput2_avg[12:29], Soutput1_avg_interp[12:29])
	print('   output correlation 12-29 is {:.4f}'.format(CC1[0,1]))
	CC1 = np.corrcoef(Soutput2_avg[23:], Soutput1_avg_interp[23:])
	print('   output correlation 23-39 is {:.4f}'.format(CC1[0,1]))

	plot_input = True
	if plot_input:
		plt.close(20)
		fig = plt.figure(20)
		plt.plot(range(tsize2), Sinput1_avg_interp, '-xr')
		plt.plot(range(tsize2), Sinput2_avg, '-b')

	plot_output = True
	if plot_output:
		plt.close(21)
		fig = plt.figure(21)
		plt.plot(range(tsize2), Soutput1_avg_interp, '-xr')
		plt.plot(range(tsize2), Soutput2_avg, '-b')


def compare_region_data(regiondataname1, regiondataname2, TR1, TR2, region_name, relative_time_shift):
	regiondataname1 = r'E:\FM2021data\HCstim_region_data_Dec2023.npy'
	regiondataname2 = r'D:\Howie_FM2_Brain_Data\HCstim_region_data_Dec2023.npy'

	TR1 = 6.75
	TR2 = 2.0

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

	tc_data1 = np.zeros((nclusters1,tsize1,NP1))
	for cc in range(nclusters1):
		for nn in range(NP1):
			t1 = np.sum(nruns_per_person1[:nn])*tsize1
			t2 = np.sum(nruns_per_person1[:nn+1])*tsize1
			one_tc = np.mean(np.reshape(tc1[cc,t1:t2],(nruns_per_person1[nn],tsize1)),axis=0)
			tc_data1[cc,:,nn] = copy.deepcopy(one_tc)


	regiondata2 = np.load(regiondataname2, allow_pickle=True).flat[0]
	region_properties2 = regiondata2['region_properties']
	nregions2 = len(region_properties2)
	rnamelist2 = [region_properties2[xx]['rname'] for xx in range(nregions2)]
	index2 = rnamelist2.index(region_name)
	tc2 = region_properties2[index2]['tc']
	nruns_per_person2 = region_properties2[index2]['nruns_per_person']
	tsize2 = region_properties2[index2]['tsize']
	nclusters2, tsize_full2 = np.shape(tc2)
	NP2 = len(nruns_per_person2)

	tc_data2 = np.zeros((nclusters2,tsize2,NP2))
	tc_data2_interp = np.zeros((nclusters2,tsize1,NP2))

	for cc in range(nclusters2):
		for nn in range(NP2):
			t1 = np.sum(nruns_per_person2[:nn])*tsize2
			t2 = np.sum(nruns_per_person2[:nn+1])*tsize2
			one_tc = np.mean(np.reshape(tc2[cc,t1:t2],(nruns_per_person2[nn],tsize2)),axis=0)

			# interpolate to the same size

			timelist1 = np.array(range(tsize1)) * TR1 + TR1 / 2.0
			timelist2 = np.array(range(tsize2)) * TR2 + TR2 / 2.0 + relative_time_shift

			f = interpolate.interp1d(timelist2, one_tc, fill_value='extrapolate')
			one_tc_interp = f(timelist1)

			tc_data2[cc,:,nn] = copy.deepcopy(one_tc)
			tc_data2_interp[cc,:,nn] = copy.deepcopy(one_tc_interp)

	# compare clusters
	nn = 2
	tc1 = tc_data1[:,:,nn]
	tc2 = tc_data2_interp[:,:,nn]

	tc1 = np.mean(tc_data1, axis=2)
	tc2 = np.mean(tc_data2_interp, axis=2)

	CCgrid = np.zeros((nclusters1,nclusters2))
	for n1 in range(nclusters1):
		for n2 in range(nclusters2):
			cc = np.corrcoef(tc1[n1,5:], tc2[n2,5:])
			CCgrid[n1,n2] = cc[0,1]



def main():
	# "V2" for the brain data was an improved run
	# use V2 for brain data
	rname1 = 'PAG'
	# rname1 = 'Hypothalamus'
	SAPMresultsname1 = r'D:\Howie_FM2_Brain_Data\HCstim_31323042031224_results_corr.npy'
	SAPMparamsname1 = r'D:\Howie_FM2_Brain_Data\HCstim_31323042031224_params.npy'

	# SAPMresultsname1 = r'D:\Howie_FM2_Brain_Data\FMstim_31323042031224_results_corr.npy'
	# SAPMparamsname1 = r'D:\Howie_FM2_Brain_Data\FMstim_31323042031224_params.npy'
	TR1 = 2.0

	rname2 = 'PAG'
	SAPMresultsname2 = r'E:\FM2021data\HCstim_1432043142_results_corr.npy'
	SAPMparamsname2 = r'E:\FM2021data\HCstim_1432043142_params.npy'

	# SAPMresultsname2 = r'E:\FM2021data\FMstim_1432043142_results_corr.npy'
	# SAPMparamsname2 = r'E:\FM2021data\FMstim_1432043142_params.npy'
	TR2 = 6.75

	relative_time_shift = -6.75
	relative_time_shift = 0.
	windownum = 58
	compare_SAPM_timecourses(SAPMresultsname1, SAPMparamsname1, rname1, TR1,
							 SAPMresultsname2, SAPMparamsname2, rname2, TR2, relative_time_shift, windownum)

