# setup program to compare two sets of SAPM results
import numpy as np
import matplotlib.pyplot as plt
import pysapm
import os
import copy
import pydatabase
import py2ndlevelanalysis
import pandas as pd

# specify the data names and locations
datadir = r'E:\SAPMresults_Dec2022'
covname = 'painrating'
descriptor = 'ANCOVA_High_vs_Low'

# group1
group1_resultsname = r'High_3242423012_v2_results.npy'
group1_paramsname = r'High_3242423012_v2_params.npy'
group1_covname = r'High_covariates.npy'

# group2
group2_resultsname = r'Low_3242423012_v2_results.npy'
group2_paramsname = r'Low_3242423012_v2_params.npy'
group2_covname = r'Low_covariates.npy'


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

# 1) group average difference

# 2) paired group difference

#-----------------------------------------------------------------------------------
# 3) ANCOVA
#-----------------------------------------------------------------------------------
# load the covariates data for each group
x1 = group1_cov['GRPcharacteristicslist'].index(covname)
x2 = group2_cov['GRPcharacteristicslist'].index(covname)

cov1 = group1_cov['GRPcharacteristicsvalues'][x1,:]
cov2 = group2_cov['GRPcharacteristicsvalues'][x2,:]

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
# setup for ANCOVA------------------------------------------------
statstype = 'ANCOVA'
formula_key1 = 'C(Group)'
formula_key2 = covname
formula_key3 = 'C(Group):' + covname
atype = 2

ancova_p = np.zeros((nconnections, 3))
for aa in range(nconnections):
	group1_B = B1[:,aa]
	group2_B = B2[:,aa]
	ancova_table, p_MeoG, p_MeoC, p_intGC = py2ndlevelanalysis.run_ANOVA_or_ANCOVA2(group1_B, group2_B, cov1, cov2, covname,
																formula_key1, formula_key2, formula_key3, atype)
	ancova_p[aa, :] = np.array([p_MeoG, p_MeoC, p_intGC])


valuetext1 = ['{:.3f} {} {:.3f}'.format(np.mean(B1[:,x]),chr(177), np.std(B1[:,x])/np.sqrt(NP1)) for x in range(nconnections)]
valuetext2 = ['{:.3f} {} {:.3f}'.format(np.mean(B2[:,x]),chr(177), np.std(B2[:,x])/np.sqrt(NP2)) for x in range(nconnections)]

pthresh = 0.05/32.0
pthresh = 0.05
pthresh_list = np.repeat(pthresh,nconnections)

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
	with pd.ExcelWriter(xlname, engine = 'openpyxl', mode='a') as writer:
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
	with pd.ExcelWriter(xlname, engine = 'openpyxl', mode='a') as writer:
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
	with pd.ExcelWriter(xlname, engine = 'openpyxl', mode='a') as writer:
		df.to_excel(writer, sheet_name='Interaction')
	outputname = xlname
	print('wrote {} values to Interaction sheet in {}'.format(len(c),xlname))
else:
	print('no signficant interaction effects detected')