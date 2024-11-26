# test problems with SAPM
import numpy as np
import matplotlib.pyplot as plt
import pysapm
import os
import copy
import pydatabase
import random
import scipy.stats as stats
import pysem
import pandas as pd
import sklearn

nperson = 0

datadir = r'E:\SAPMresults_Dec2022'
savefilename = r'E:\SAPMresults_Dec2022\temp_params_save.npy'
reload_old_results = False
save_new_results = True
windowbasenum = 300

resultsname = r'E:\SAPMresults_Dec2022\Pain_3242423012_norm_results.npy'
paramsname = r'E:\SAPMresults_Dec2022\Pain_3242423012_norm_params.npy'
SAPMresults_norm = np.load(resultsname, allow_pickle=True)
Sinput_norm = SAPMresults_norm[nperson]['Sinput']
Sconn_norm = SAPMresults_norm[nperson]['Sconn']
Mintrinsic_norm = SAPMresults_norm[nperson]['Mintrinsic']

resultsname = r'E:\SAPMresults_Dec2022\Pain_3242423012_results.npy'
paramsname = r'E:\SAPMresults_Dec2022\Pain_3242423012_params.npy'
SAPMresults = np.load(resultsname, allow_pickle=True)
Sinput = SAPMresults[nperson]['Sinput']
Sconn = SAPMresults[nperson]['Sconn']
Mintrinsic = SAPMresults[nperson]['Mintrinsic']

# resultsname = r'E:\SAPMresults_Dec2022\Pain_3242423012_norm_results.npy'
# paramsname = r'E:\SAPMresults_Dec2022\Pain_3242423012_norm_params.npy'

regiondataname = r'E:\SAPMresults_Dec2022\Pain_regiondata2.npy'
clusterdataname = r'E:/SAPMresults_Dec2022\Pain_equalsize_cluster_def.npy'
networkfile = r'E:\SAPMresults_Dec2022\network_model_March2023_SAPM.xlsx'
DBname = r'E:\graded_pain_database_May2022.xlsx'
timepoint = 'all'
epoch = 'all'
betascale=0.1
DBname = r'E:\graded_pain_database_May2022.xlsx'

cnums = [3, 2, 4, 2, 4, 2, 3, 0, 1, 2]

pysapm.prep_data_sem_physio_model_SO_V2(networkfile, regiondataname, clusterdataname, paramsname, timepoint, epoch,
								 fullgroup=False, normalizevar=False, filter_tcdata=False)

SAPMresults_load = np.load(resultsname, allow_pickle=True)
SAPMparams = np.load(paramsname, allow_pickle=True).flat[0]
beta_list = SAPMparams['beta_list']
ctarget = SAPMparams['ctarget']
csource = SAPMparams['csource']
dtarget = SAPMparams['dtarget']
dsource = SAPMparams['dsource']
rnamelist = SAPMparams['rnamelist']
network = SAPMparams['network']

betanamelist = SAPMparams['betanamelist']
nruns_per_person = SAPMparams['nruns_per_person']
nclusterstotal = SAPMparams['nclusterstotal']
rnamelist = SAPMparams['rnamelist']
nregions = SAPMparams['nregions']
cluster_properties = SAPMparams['cluster_properties']
fintrinsic_count = SAPMparams['fintrinsic_count']
vintrinsic_count = SAPMparams['vintrinsic_count']
sem_region_list = SAPMparams['sem_region_list']
nclusterlist = SAPMparams['nclusterlist']
tsize = SAPMparams['tsize']
tplist_full = SAPMparams['tplist_full']
tcdata_centered = SAPMparams['tcdata_centered']
fintrinsic_region = SAPMparams['fintrinsic_region']
latent_flag = SAPMparams['latent_flag']
reciprocal_flag = SAPMparams['reciprocal_flag']
Mconn = SAPMparams['Mconn']
Minput = SAPMparams['Minput']
Nlatent = vintrinsic_count + fintrinsic_count


# xls = pd.ExcelFile(DBname, engine='openpyxl')
# df1 = pd.read_excel(xls, 'paradigm1_BOLD')
# del df1['Unnamed: 0']  # get rid of the unwanted header column
# fields = list(df1.keys())
# paradigm = df1['paradigms_BOLD']
# timevals = df1['time']
# paradigm_centered = paradigm - np.mean(paradigm)
# fintrinsic_base = copy.deepcopy(paradigm_centered)


# load some data, setup some parameters...
network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = pysapm.load_network_model_w_intrinsics(
	networkfile)
ncluster_list = np.array([nclusterlist[x]['nclusters'] for x in range(len(nclusterlist))])
cluster_name = [nclusterlist[x]['name'] for x in range(len(nclusterlist))]
not_latent = [x for x in range(len(cluster_name)) if 'intrinsic' not in cluster_name[x]]
ncluster_list = ncluster_list[not_latent]
full_rnum_base = [np.sum(ncluster_list[:x]) for x in range(len(ncluster_list))]
namelist = [cluster_name[x] for x in not_latent]
namelist += ['Rtotal']
namelist += ['R ' + cluster_name[x] for x in not_latent]

clusterlist = np.array(cnums) + full_rnum_base

epochnum = 0
tp = tplist_full[epochnum][nperson]['tp']
tcdata_centered = SAPMparams['tcdata_centered']
Sinput = []
# Sinput_scalefactor = np.zeros(len(clusterlist))
for nc, cval in enumerate(clusterlist):
	tc1 = tcdata_centered[cval, tp]
	# Sinput_scalefactor[nc] = np.var(tc1)
	# tc1 /= np.var(tc1)
	Sinput.append(tc1)
Sinput = np.array(Sinput)

# setup fixed intrinsic based on the model paradigm
# need to account for timepoint and epoch....
epoch = tsize
if fintrinsic_count > 0:
	if epoch >= tsize:
		et1 = 0
		et2 = tsize
	else:
		if np.floor(epoch / 2).astype(int) == np.ceil(epoch / 2).astype(int):  # even numbered epoch
			et1 = (timepoint - np.floor(epoch / 2)).astype(int)
			et2 = (timepoint + np.floor(epoch / 2)).astype(int)
		else:
			et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
			et2 = (timepoint + np.floor(epoch / 2)).astype(int)
	if et1 < 0: et1 = 0
	if et2 > tsize: et2 = tsize
	epoch = et2 - et1

	ftemp = fintrinsic_base[0,et1:et2]
	fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])

# nbeta, nbeta = np.shape(SAPMresults_load[0]['Mconn'])
# Nlatent = nbeta - nregions
rnamelist_ext = copy.deepcopy(rnamelist)
for ll in range(Nlatent):
	rnamelist_ext += ['int{}'.format(ll)]

# Sinput = Minput@Sconn
# Sconn = Mconn@Sconn
# Sconn = Meigv@Mintrinsic

initial_alpha = 1e-2
initial_Lweight = 1e-2
initial_dval = 0.1
# nitermax = 300
alpha_limit = 1.0e-5
repeat_limit = 2
betalimit = 3.0

dval = copy.deepcopy(initial_dval)
Lweight = copy.deepcopy(initial_Lweight)
alpha = copy.deepcopy(initial_alpha)
alphabint = copy.deepcopy(initial_alpha)

ntime, NP = np.shape(tplist_full)
Nintrinsics = vintrinsic_count + fintrinsic_count

Sinput = SAPMresults_load[nperson]['Sinput']

nr,tsize = np.shape(Sinput)
# scale variance
v2 = np.std(Sinput, axis=1)
v2 = np.repeat(v2[:,np.newaxis], tsize, axis=1)
# Sinput /= v2
#
# also center
mu2 = np.mean(Sinput, axis=0)
mu2 = np.repeat(mu2[np.newaxis, :], nr, axis=0)
# Sinput -= mu2

beta_int1 = 0.0

deltavals = np.ones(len(dtarget))
betavals = 0.1*np.random.randn(len(ctarget))
Minput[dtarget,dsource] = copy.deepcopy(deltavals)
Mconn[ctarget,csource] = copy.deepcopy(betavals)

if reload_old_results:
	oldresults = np.load(savefilename, allow_pickle=True).flat[0]
	betavals = oldresults['betavals']
	deltavals = oldresults['deltavals']
	beta_int1 = oldresults['beta_int1']

lastgood_betavals = copy.deepcopy(betavals)
lastgood_deltavals = copy.deepcopy(deltavals)
lastgood_beta_int1 = copy.deepcopy(beta_int1)

# windownum = 80
# for nn in range(10):
# 	windownum = 80+nn
# 	plt.close(windownum)
# 	fig = plt.figure(windownum)
# 	plt.plot(range(200), Sinput[nn, :], '-r')
# 	plt.plot(range(200), Sinput_norm[nn, :], '-b')

# Sinput = copy.deepcopy(Sinput_norm)

# # add in constant term
# mu2 = np.sum(Sinput, axis=0)
# Sinput = np.concatenate((Sinput,-mu2[np.newaxis,:]),axis=0)
# nregions,ntotal = np.shape(Minput)
# Minput2 = np.zeros((nregions+2,ntotal+2))
# Minput2[:nregions,:nregions] = Minput[:nregions,:nregions]
# Minput2[nregions,nregions+1] = 1.0
# Minput2[:nregions,-Nintrinsic:] = Minput[:nregions,-Nintrinsic:]
#
# Mconn2 = np.zeros((ntotal+2,ntotal+2))
# Mconn2[:nregions,:nregions] = Mconn[:nregions,:nregions]
# Mconn2[nregions,nregions+1] = 1.0
# Mconn2[nregions+1,nregions+1] = 1.0
# Mconn2[-Nintrinsic:,:nregions] = Mconn[-Nintrinsic:,:nregions]
# Mconn2[:nregions,-Nintrinsic:] = Mconn[:nregions,-Nintrinsic:]
# Mconn2[-Nintrinsic:,-Nintrinsic:] = Mconn[-Nintrinsic:,-Nintrinsic:]
# Nintrinsic+=1
#
# Minput = copy.deepcopy(Minput2)
# Mconn = copy.deepcopy(Mconn2)
# ctarget,csource = np.where(Mconn != 0)
# c = np.where(ctarget < nregions)
# ctarget = ctarget[c]
# csource = csource[c]

# use only constant term----------------------------
#---------------------------------------------------
# nr,tsize = np.shape(Sinput)
# mu2 = np.mean(Sinput,axis=0)
# Sinput = np.repeat(mu2[np.newaxis,:],nr,axis=0)

# PCA of Sinput
pca = sklearn.decomposition.PCA()
pca.fit(Sinput)
components = pca.components_
loadings = pca.transform(Sinput)
mu2 = np.mean(Sinput, axis=0)
loadings = np.concatenate((np.ones((nr, 1)), loadings), axis=1)
components = np.concatenate((mu2[np.newaxis, :], components), axis=0)
# test_fit = loadings @ components

countmax = 300
count = 0
keep_going = True
ssqdrecord = np.zeros(countmax)
results_record = []
runcount = 0
while keep_going and count < countmax:
	betavals, deltavals, beta_int1, fit, dssq_db, dssq_dd, dssq_dbeta1, ssqd_original, ssqd, alpha, alphabint = \
		pysapm.update_betavals_V2(Sinput, components, Minput, Mconn, betavals, deltavals, betalimit,
						   ctarget, csource, dtarget, dsource, dval, fintrinsic_count,
						   vintrinsic_count, beta_int1, fintrinsic1, Lweight, alpha, alphabint,
						   latent_flag=latent_flag)

	# betavalsn, deltavalsn, beta_int1n, fitn, dssq_dbn, dssq_ddn, dssq_dbeta1n, ssqd_originaln, ssqdn, alphan, alphabintn = \
	# 	pysapm.update_betavals_V2(Sinput_norm, components, Minput, Mconn, betavals, deltavals, betalimit,
	# 					   ctarget, csource, dtarget, dsource, dval, fintrinsic_count,
	# 					   vintrinsic_count, beta_int1, fintrinsic1, Lweight, alpha, alphabint,
	# 					   latent_flag=latent_flag)

	ssqdrecord[count] = ssqd

	if ssqd > ssqd_original:
		alpha *= 0.5
		alphabint *= 0.5
		betavals = copy.deepcopy(lastgood_betavals)  # no improvement, so don't update
		deltavals = copy.deepcopy(lastgood_deltavals)
		beta_int1 = copy.deepcopy(lastgood_beta_int1)
		runcount = 0
	else:
		lastgood_betavals = copy.deepcopy(betavals)
		lastgood_deltavals = copy.deepcopy(deltavals)
		lastgood_beta_int1 = copy.deepcopy(beta_int1)
		runcount += 1
		if runcount > 3:
			alpha *= 1.3
			alphabint *= 1.3
			runcount = 0

	Mconn[ctarget, csource] = copy.deepcopy(betavals)
	Minput[dtarget, dsource] = copy.deepcopy(deltavals)
	fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
															 vintrinsic_count, beta_int1, fintrinsic1)
	ssqd = pysapm.sapm_error_function_V2(Sinput, fit, Lweight, betavals, deltavals, beta_int1, Minput, components, Mintrinsic, Meigv)

	err_total = Sinput - fit
	Smean = np.mean(Sinput)
	errmean = np.mean(err_total)

	R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
	R2avg = np.mean(R2list)
	R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)

	# Sinput_sim, Soutput_sim = network_sim(Sinput_full, Soutput_full, Minput, Moutput)
	results_record.append({'Sinput': Sinput, 'fit': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})

	ssqchange = ssqd - ssqd_original
	# if np.abs(ssqchange) < 1e-5: converging = False

	print('SAPM  count {} alpha {:.3e}  ssqd {:.2f} change {:.3f}  R2 avg {:.3f}  R2 total {:.3f}'.format(count, alpha,
											ssqd, ssqchange, R2avg, R2total))

	count += 1

results = {'betavals':betavals, 'deltavals':deltavals, 'beta_int1':beta_int1, 'Mintrinsic':Mintrinsic, 'Sinput':Sinput, 'fit':fit, 'Meigv': Meigv}
if save_new_results:
	np.save(savefilename, results)

nn=3
for nn in range(10):
	tt = countmax-1
	windownum = windowbasenum+nn
	plt.close(windownum)
	fig = plt.figure(windownum)
	plt.plot(range(200), results_record[tt]['Sinput'][nn, :], '-xr')
	plt.plot(range(200), results_record[tt]['fit'][nn, :], '-b')

#
# conn_name_list = ['{}-{}'.format(rnamelist_ext[csource[x]][:4], rnamelist_ext[ctarget[x]][:4]) for x in
# 				  range(len(ctarget))]
# plt.errorbar(conn_name_list, Mconn_avg, yerr=Mconn_sem, marker='o', linestyle='none', markeredgecolor='none')
# ax.set_xticklabels(labels=conn_name_list, rotation='vertical', fontsize=7)
#
# plt.plot(conn_name_list, np.zeros(len(conn_name_list)),'-k')


# PCA approach--------------------------------------------------------
#---------------------------------------------------------------------
nr,tsize = np.shape(Sinput)

pca = sklearn.decomposition.PCA()
pca.fit(Sinput)
components = pca.components_
evr = pca.explained_variance_ratio_

mu2 = np.mean(Sinput, axis=0)
# mu2 = np.repeat(mu2[np.newaxis, :], nr, axis=0)

loadings = pca.transform(Sinput)

loadings = np.concatenate((np.ones((nr,1)),loadings),axis=1)
components = np.concatenate((mu2[np.newaxis, :], components),axis=0)

fit_check = loadings @ components

# PCA of Sinput_avg
pcan = sklearn.decomposition.PCA()
pcan.fit(Sinput_norm)
componentsn = pcan.components_
evrn = pcan.explained_variance_ratio_

mu2n = np.mean(Sinput_norm, axis=0)
# mu2n = np.repeat(mu2n[np.newaxis, :], 10, axis=0)

loadingsn = pca.transform(Sinput_norm)

loadingsn = np.concatenate((np.ones((nr,1)),loadingsn),axis=1)
componentsn = np.concatenate((mu2n[np.newaxis, :], componentsn),axis=0)

fit_checkn = loadingsn @ componentsn

# PC loadings for intrinsics
w = Mintrinsic[1:,:] @ components.T @ np.linalg.inv(components @ components.T)

# PC loadings for Sinput
a = (Sinput) @ components.T @ np.linalg.inv(components @ components.T)

# this needs to be true
atest = Minput @ Meigv[:,1:] @ w[:2]