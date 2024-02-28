# setup program to compare two sets of SAPM results
import numpy as np
import matplotlib.pyplot as plt
import pysapm
import os
import copy
import pydatabase
import py2ndlevelanalysis
import pandas as pd
import scipy.stats as stats

# comparison type - choices are:  'ancova', 'unpaired_difference', 'paired_difference'
#                  'paired_difference_correlation_w_cov1', 'paired_difference_correlation_w_delta_cov'

#specify the type of analysis
comparison_type = 'ancova'
descriptor = 'ancova_FMstim_v_HCstim_firstpainrating'
#
# comparison_type = 'unpaired_difference'
# descriptor = 'unpaired_diff_FMstim_v_HCstim'

pthresh = 0.05 / 32.0
pthresh = 0.05
covname = 'firstpainrating'
covlimit = 1000.0

datagroup = 'brain'
# specify the data names and locations
# brain data
if datagroup == 'brain':
	datadir = r'D:/Howie_FM2_Brain_Data'
	# group1
	group1_resultsname = r'FMstim_02323203331202_V5_19_results_corr.npy'
	group1_paramsname = r'FMstim_02323203331202_V5_19_params.npy'
	group1_covname = r'FMstim_02323203331202_V5_19_results_covariates.npy'

	group2_resultsname = r'HCstim_02323203331202_V5_results_corr.npy'
	group2_paramsname = r'HCstim_02323203331202_V5_params.npy'
	group2_covname = r'HCstim_02323203331202_V5_results_covariates.npy'


	# group2
	# group2_resultsname = r'FMrest_02323203331202_V5_results_corr.npy'
	# group2_paramsname = r'FMrest_02323203331202_V5_params.npy'
	# group2_covname = r'FMrest_02323203331202_V5_results_covariates.npy'

	# group2_resultsname = r'HCrest_02323203331202_V5_results_corr.npy'
	# group2_paramsname = r'HCrest_02323203331202_V5_params.npy'
	# group2_covname = r'HCrest_02323203331202_V5_results_covariates.npy'

else:
	datadir = r'E:\FM2021data'

	group1_resultsname = r'FMstim_2230224124_V5b_results_corr.npy'
	group1_paramsname = r'FMstim_2230224124_V5b_params.npy'
	group1_covname = r'FMstim_1441234122_V3_results_covariates.npy'

	group2_resultsname = r'HCstim_2230224124_V5b_results_corr.npy'
	group2_paramsname = r'HCstim_2230224124_V5b_params.npy'
	group2_covname = r'HCstim_1432043142_results_covariates.npy'

#-----------------------------------------------------------------------------------
# probably no need to change anything below this line
#-----------------------------------------------------------------------------------
# load the data - group1
gr1 = os.path.join(datadir, group1_resultsname)
group1_results = np.load(gr1, allow_pickle=True)

gp1 = os.path.join(datadir, group1_paramsname)
group1_params = np.load(gp1, allow_pickle=True).flat[0]

gc1 = os.path.join(datadir, group1_covname)
group1_cov = np.load(gc1, allow_pickle=True).flat[0]

# load the data - group2
gr2 = os.path.join(datadir, group2_resultsname)
group2_results = np.load(gr2, allow_pickle=True)

gp2 = os.path.join(datadir, group2_paramsname)
group2_params = np.load(gp2, allow_pickle=True).flat[0]

gc2 = os.path.join(datadir, group2_covname)
group2_cov = np.load(gc2, allow_pickle=True).flat[0]

#------------------------------------------------------------
# compare group results - with many options

# load the covariates data for each group
x1 = group1_cov['GRPcharacteristicslist'].index(covname)
x2 = group2_cov['GRPcharacteristicslist'].index(covname)

cov1 = group1_cov['GRPcharacteristicsvalues'][x1,:]
cov2 = group2_cov['GRPcharacteristicsvalues'][x2,:]

xx1 = np.where(cov1 < covlimit)[0]
xx2 = np.where(cov2 < covlimit)[0]

cov1 = np.array(cov1).astype(float)
cov2 = np.array(cov2).astype(float)

#-----------------------------------------------------------------------------------
# load some information about the network and the data
nregions = group1_params['nregions']
vintrinsic_count = group1_params['vintrinsic_count']
fintrinsic_count = group1_params['fintrinsic_count']
tsize = group1_params['tsize']
beta_list = group1_params['beta_list']
rnamelist = group1_params['rnamelist']
NP1 = len(group1_results)
NP2 = len(group2_results)

#-----------------------------------------------------------------------------------
# load the B values for the group, for each connection
B1 = np.array([group1_results[x]['betavals'] for x in range(NP1)])
B2 = np.array([group2_results[x]['betavals'] for x in range(NP2)])

#-----------------------------------------------------------------------------------
# create a list of names for each connection, for looking at the results later on
cname_list = []
nconnections = len(beta_list)
for aa in range(nconnections):
	if beta_list[aa]['pair'][0] >= nregions:
		sname = 'int{}'.format(beta_list[aa]['pair'][0] - nregions)
	else:
		sname = rnamelist[beta_list[aa]['pair'][0]][:4]
	tname = rnamelist[beta_list[aa]['pair'][1]][:4]
	cname = '{}-{}'.format(sname, tname)
	cname_list += [cname]


#-----------------------------------------------------------------------------------
# 1) group average difference
#-----------------------------------------------------------------------------------
if comparison_type == 'unpaired_difference':
	B1mean = np.mean(B1,axis=0)
	B1var = np.var(B1,axis=0)
	B2mean = np.mean(B2,axis=0)
	B2var = np.var(B2,axis=0)
	valuetext1 = ['{:.3f} {} {:.3f}'.format(B1mean[x],chr(177), np.sqrt(B1var[x]/NP1)) for x in range(nconnections)]
	valuetext2 = ['{:.3f} {} {:.3f}'.format(B2mean[x],chr(177), np.sqrt(B2var[x]/NP1)) for x in range(nconnections)]

	Sp = np.sqrt( ((NP1-1)*B1var + (NP2-1)*B2var)/(NP1+NP2-2) )
	T = (B1mean-B2mean)/(Sp * np.sqrt(1/NP1 + 1/NP2))   # independent two-sample T-test assuming approximately equal variance

	si = np.argsort(T)
	si = si[::-1]
	Ts = copy.deepcopy(T[si])
	valuetext1s = copy.deepcopy(np.array(valuetext1)[si])
	valuetext2s = copy.deepcopy(np.array(valuetext2)[si])
	cname_lists = copy.deepcopy(np.array(cname_list)[si])

	Tthresh = stats.t.ppf(1 - pthresh, NP1+NP2-2)
	if np.isnan(Tthresh):  Tthresh = 0.0
	ps = 1-stats.t.cdf(Ts,NP1+NP2-2)

	pthresh_list = np.repeat(pthresh, nconnections)
	Tthresh_list = np.repeat(Tthresh, nconnections)

	c = np.where(Ts > Tthresh)[0]
	if len(c) > 0:
		textoutputs = {'connection': np.array(cname_lists)[c], 'B1': np.array(valuetext1s)[c], 'B2': np.array(valuetext2s)[c],
					   'T': np.array(Ts)[c], 'T thresh': np.array(Tthresh_list)[c], 'p': np.array(ps)[c], 'p thresh': np.array(pthresh_list)[c]}

		df = pd.DataFrame(textoutputs)
		xlname = os.path.join(datadir, descriptor + '.xlsx')
		if os.path.isfile(xlname):
			mode = 'a'
		else:
			mode = 'w'
		with pd.ExcelWriter(xlname, engine = 'openpyxl', mode=mode) as writer:
			df.to_excel(writer, sheet_name='unpaired_diff')
		outputname = xlname
		print('wrote {} values to unpaired_diff sheet in {}'.format(len(c),xlname))
	else:
		print('no signficant unpaired differences detected')


#-----------------------------------------------------------------------------------
# 2) paired group difference
#-----------------------------------------------------------------------------------
if comparison_type == 'paired_difference':
	if NP1 == NP2:
		dBmean = np.mean(B1-B2, axis=0)
		dBsem = np.std(B1-B2, axis=0)/np.sqrt(NP1)
		T = dBmean/(dBsem + 1.0e-20)
		valuetext1 = ['{:.3f} {} {:.3f}'.format(B1mean[x],chr(177), np.sqrt(B1var[x]/NP1)) for x in range(nconnections)]
		valuetext2 = ['{:.3f} {} {:.3f}'.format(B2mean[x],chr(177), np.sqrt(B2var[x]/NP1)) for x in range(nconnections)]

		si = np.argsort(T)
		si = si[::-1]
		Ts = copy.deepcopy(T[si])
		valuetext1s = copy.deepcopy(np.array(valuetext1)[si])
		valuetext2s = copy.deepcopy(np.array(valuetext2)[si])
		cname_lists = copy.deepcopy(np.array(cname_list)[si])

		Tthresh = stats.t.ppf(1 - pthresh, NP1 + NP2 - 2)
		if np.isnan(Tthresh):  Tthresh = 0.0
		ps = 1 - stats.t.cdf(Ts, NP1 + NP2 - 2)

		pthresh_list = np.repeat(pthresh, nconnections)
		Tthresh_list = np.repeat(Tthresh, nconnections)

		c = np.where(Ts > Tthresh)[0]
		if len(c) > 0:
			textoutputs = {'connection': np.array(cname_lists)[c], 'B1': np.array(valuetext1s)[c],
						   'B2': np.array(valuetext2s)[c],
						   'T': np.array(Ts)[c], 'Tthresh': np.array(Tthresh_list)[c], 'p': np.array(ps)[c], 'p thresh': np.array(pthresh_list)[c]}

			df = pd.DataFrame(textoutputs)
			xlname = os.path.join(datadir, descriptor + '.xlsx')
			if os.path.isfile(xlname):
				mode = 'a'
			else:
				mode = 'w'
			with pd.ExcelWriter(xlname, engine='openpyxl', mode=mode) as writer:
				df.to_excel(writer, sheet_name='paired_diff')
			outputname = xlname
			print('wrote {} values to paired_diff sheet in {}'.format(len(c), xlname))
		else:
			print('no significant paired differences detected')

	else:
		print('NP1 = {} and NP2 = {}.   A paired comparison is not possible with unequal group sizes'.format(NP1,NP2))


#-----------------------------------------------------------------------------------
# 2.5) paired group difference
#-----------------------------------------------------------------------------------
if comparison_type == 'paired_difference_correlation_w_cov1':
	if NP1 == NP2:
		dB = B1-B2
		dC = cov1-cov2

		dBmean = np.mean(dB,axis=0)
		dBvar = np.var(dB,axis=0)

		Z = np.zeros(nconnections)
		for nn in range(nconnections):
			rr = np.corrcoef(dB[:,nn],cov1)
			Z[nn] = np.arctanh(rr[0,1])*np.sqrt(NP1-3)

		valuetext1 = ['{:.3f} {} {:.3f}'.format(dBmean[x],chr(177), np.sqrt(dBvar[x]/NP1)) for x in range(nconnections)]
		# valuetext2 = ['{:.3f}'.format(cov1[x]) for x in range(nconnections)]

		si = np.argsort(np.abs(Z))
		si = si[::-1]
		Zs = copy.deepcopy(Z[si])
		valuetext1s = copy.deepcopy(np.array(valuetext1)[si])
		# valuetext2s = copy.deepcopy(np.array(valuetext2)[si])
		cname_lists = copy.deepcopy(np.array(cname_list)[si])

		Zthresh = stats.norm.ppf(1 - pthresh)
		if np.isnan(Zthresh):  Zthresh = 0.0
		ps = 1 - stats.norm.cdf(Zs)

		pthresh_list = np.repeat(pthresh, nconnections)
		Tthresh_list = np.repeat(Zthresh, nconnections)

		c = np.where(np.abs(Zs) > Zthresh)[0]
		if len(c) > 0:
			textoutputs = {'connection': np.array(cname_lists)[c], 'dB avg': np.array(valuetext1s)[c],
						   'Z': np.array(Zs)[c], 'Zthresh': np.array(Zthresh_list)[c], 'p': np.array(ps)[c], 'p thresh': np.array(pthresh_list)[c]}

			df = pd.DataFrame(textoutputs)
			xlname = os.path.join(datadir, descriptor + '.xlsx')
			if os.path.isfile(xlname):
				mode = 'a'
			else:
				mode = 'w'
			with pd.ExcelWriter(xlname, engine='openpyxl', mode=mode) as writer:
				df.to_excel(writer, sheet_name='paired_diff_corr')
			outputname = xlname
			print('wrote {} values to paired_diff_corr sheet in {}'.format(len(c), xlname))
		else:
			print('no significant paired differences detected')

	else:
		print('NP1 = {} and NP2 = {}.   A paired comparison is not possible with unequal group sizes'.format(NP1,NP2))


#-----------------------------------------------------------------------------------
# 2.75) paired group difference 3
#-----------------------------------------------------------------------------------
if comparison_type == 'paired_difference_correlation_w_delta_cov':
	if NP1 == NP2:
		dB = B1-B2
		dC = cov1-cov2

		dBmean = np.mean(dB,axis=0)
		dBvar = np.var(dB,axis=0)

		Z = np.zeros(nconnections)
		for nn in range(nconnections):
			rr = np.corrcoef(dB[:,nn],dC)
			Z[nn] = np.arctanh(rr[0,1])*np.sqrt(NP1-3)

		valuetext1 = ['{:.3f} {} {:.3f}'.format(dBmean[x],chr(177), np.sqrt(dBvar[x]/NP1)) for x in range(nconnections)]
		valuetext2 = ['{:.3f} {} {:.3f}'.format(dC,chr(177), np.sqrt(dC/NP1))]

		si = np.argsort(np.abs(Z))
		si = si[::-1]
		Zs = copy.deepcopy(Z[si])
		valuetext1s = copy.deepcopy(np.array(valuetext1)[si])
		# valuetext2s = copy.deepcopy(np.array(valuetext2)[si])
		cname_lists = copy.deepcopy(np.array(cname_list)[si])

		Zthresh = stats.norm.ppf(1 - pthresh)
		if np.isnan(Zthresh):  Zthresh = 0.0
		ps = 1 - stats.norm.cdf(Zs)

		pthresh_list = np.repeat(pthresh, nconnections)
		Tthresh_list = np.repeat(Zthresh, nconnections)

		c = np.where(np.abs(Zs) > Zthresh)[0]
		if len(c) > 0:
			textoutputs = {'connection': np.array(cname_lists)[c], 'dB avg': np.array(valuetext1s)[c],
						   'dcov avg': np.array(valuetext2),
						   'Z': np.array(Zs)[c], 'Zthresh': np.array(Zthresh_list)[c], 'p': np.array(ps)[c], 'p thresh': np.array(pthresh_list)[c]}

			df = pd.DataFrame(textoutputs)
			xlname = os.path.join(datadir, descriptor + '.xlsx')
			if os.path.isfile(xlname):
				mode = 'a'
			else:
				mode = 'w'
			with pd.ExcelWriter(xlname, engine='openpyxl', mode=mode) as writer:
				df.to_excel(writer, sheet_name='paired_diff_vs_dcov')
			outputname = xlname
			print('wrote {} values to paired_diff_vs_dcov sheet in {}'.format(len(c), xlname))
		else:
			print('no significant paired differences detected')

	else:
		print('NP1 = {} and NP2 = {}.   A paired comparison is not possible with unequal group sizes'.format(NP1,NP2))



#-----------------------------------------------------------------------------------
# 3) ANCOVA
#-----------------------------------------------------------------------------------
if comparison_type == 'ancova':
	if 'STAI' in covname:
		covnameshort = 'anx'
	else:
		covnameshort = covname

	statstype = 'ANCOVA'
	formula_key1 = 'C(Group)'
	formula_key2 = covnameshort
	formula_key3 = 'C(Group):' + covnameshort
	atype = 2

	pthresh_list = np.repeat(pthresh, nconnections)

	ancova_p = np.zeros((nconnections, 3))
	for aa in range(nconnections):
		group1_B = B1[:,aa]
		group2_B = B2[:,aa]
		ancova_table, p_MeoG, p_MeoC, p_intGC = py2ndlevelanalysis.run_ANOVA_or_ANCOVA2(group1_B[xx1], group2_B[xx2], cov1[xx1], cov2[xx2], covnameshort,
																	formula_key1, formula_key2, formula_key3, atype)
		ancova_p[aa, :] = np.array([p_MeoG, p_MeoC, p_intGC])


	valuetext1 = ['{:.3f} {} {:.3f}'.format(np.mean(B1[:,x]),chr(177), np.std(B1[:,x])/np.sqrt(NP1)) for x in range(nconnections)]
	valuetext2 = ['{:.3f} {} {:.3f}'.format(np.mean(B2[:,x]),chr(177), np.std(B2[:,x])/np.sqrt(NP2)) for x in range(nconnections)]

	# main effect of group
	p = ancova_p[:,0]
	si = np.argsort(p)
	ps = copy.deepcopy(p[si])
	valuetext1s = copy.deepcopy(np.array(valuetext1)[si])
	valuetext2s = copy.deepcopy(np.array(valuetext2)[si])
	cname_lists = copy.deepcopy(np.array(cname_list)[si])

	c = np.where(ps < pthresh)[0]
	if len(c) > 0:
		textoutputs = {'connection': np.array(cname_lists)[c], 'B1': np.array(valuetext1s)[c], 'B2': np.array(valuetext2s)[c],
					   'p': np.array(ps)[c], 'p thresh': np.array(pthresh_list)[c]}

		# p, f = os.path.split(SAPMresultsname)
		df = pd.DataFrame(textoutputs)
		xlname = os.path.join(datadir, descriptor + '.xlsx')
		if os.path.isfile(xlname):
			mode = 'a'
		else:
			mode = 'w'
		with pd.ExcelWriter(xlname, engine = 'openpyxl', mode=mode) as writer:
			df.to_excel(writer, sheet_name='MeoG')
		outputname = xlname
		print('wrote {} values to MeoG sheet in {}'.format(len(c),xlname))
	else:
		print('no signficant main effects of Group detected')

	# main effect of covariate
	p = ancova_p[:,1]
	si = np.argsort(p)
	ps = copy.deepcopy(p[si])
	valuetext1s = copy.deepcopy(np.array(valuetext1)[si])
	valuetext2s = copy.deepcopy(np.array(valuetext2)[si])
	cname_lists = copy.deepcopy(np.array(cname_list)[si])

	c = np.where(ps < pthresh)[0]
	if len(c) > 0:
		textoutputs = {'connection': np.array(cname_lists)[c], 'B1': np.array(valuetext1s)[c], 'B2': np.array(valuetext2s)[c],
					   'p': np.array(ps)[c], 'p thresh': np.array(pthresh_list)[c]}

		# p, f = os.path.split(SAPMresultsname)
		df = pd.DataFrame(textoutputs)
		xlname = os.path.join(datadir, descriptor + '.xlsx')
		if os.path.isfile(xlname):
			mode = 'a'
		else:
			mode = 'w'
		with pd.ExcelWriter(xlname, engine = 'openpyxl', mode=mode) as writer:
			df.to_excel(writer, sheet_name='MeoC')
		outputname = xlname
		print('wrote {} values to MeoC sheet in {}'.format(len(c),xlname))
	else:
		print('no signficant main effects of Covariates detected')

	# interaction effect group x covariate
	p = ancova_p[:,2]
	si = np.argsort(p)
	ps = copy.deepcopy(p[si])
	valuetext1s = copy.deepcopy(np.array(valuetext1)[si])
	valuetext2s = copy.deepcopy(np.array(valuetext2)[si])
	cname_lists = copy.deepcopy(np.array(cname_list)[si])

	c = np.where(ps < pthresh)[0]
	if len(c) > 0:
		textoutputs = {'connection': np.array(cname_lists)[c], 'B1': np.array(valuetext1s)[c], 'B2': np.array(valuetext2s)[c],
					   'p': np.array(ps)[c], 'p thresh': np.array(pthresh_list)[c]}

		# p, f = os.path.split(SAPMresultsname)
		df = pd.DataFrame(textoutputs)
		xlname = os.path.join(datadir, descriptor + '.xlsx')
		if os.path.isfile(xlname):
			mode = 'a'
		else:
			mode = 'w'
		with pd.ExcelWriter(xlname, engine = 'openpyxl', mode=mode) as writer:
			df.to_excel(writer, sheet_name='Interaction')
		outputname = xlname
		print('wrote {} values to Interaction sheet in {}'.format(len(c),xlname))
	else:
		print('no signficant interaction effects detected')