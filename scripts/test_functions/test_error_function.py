import numpy as np
import matplotlib.pyplot as plt
import pysapm
import pysem
import os
import copy
import pydatabase
import time

SAPMresultsname = r'E:/SAPMresults_Dec2022\AllPain0_3242423012_results.npy'
SAPMparametersname = r'E:/SAPMresults_Dec2022\AllPain0_3242423012_params.npy'
regiondataname = r'E:/SAPMresults_Dec2022\allpainconditions_regiondata2.npy'


cnums = [3, 2, 4, 2, 4, 2, 3, 0, 1, 2]
clusterdataname = r'E:/SAPMresults_Dec2022\Pain_equalsize_cluster_def.npy'
networkfile = r'E:\SAPMresults_Dec2022\network_model_April2023_SAPM_2L.xlsx'
DBname = r'E:\graded_pain_database_May2022.xlsx'
timepoint = 'all'
epoch = 'all'
betascale=0.1


#--------------

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

# starting values
cnums_original = copy.deepcopy(cnums)
excelsheetname = 'clusters'

# run the analysis with SAPM
clusterlist = np.array(cnums) + full_rnum_base

pysapm.prep_data_sem_physio_model_SO_V2(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch,
								 fullgroup=False, normalizevar=True, filter_tcdata=False)


# output = pysapm.sem_physio_model1_V2(clusterlist, fintrinsic_base, SAPMresultsname, SAPMparametersname,
# 							  fixed_beta_vals=[], betascale=betascale, normalizevar=False)


#-------------sem_physio_model1_V2-----------------------------------------
verbose = True
normalizevar=True
fixed_beta_vals=[]

betascale = 0.1
nitermax = 250
initial_nitermax_stage1 = 10
initial_nsteps_stage1 = 10

initial_alpha = 1e-1
initial_Lweight = 10.
initial_dval = 0.05
# nitermax = 300
alpha_limit = 1.0e-5
repeat_limit = 2

SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
# load the data values
betanamelist = SAPMparams['betanamelist']
beta_list = SAPMparams['beta_list']
nruns_per_person = SAPMparams['nruns_per_person']
nclusterstotal = SAPMparams['nclusterstotal']
rnamelist = SAPMparams['rnamelist']
nregions = SAPMparams['nregions']
cluster_properties = SAPMparams['cluster_properties']
cluster_data = SAPMparams['cluster_data']
network = SAPMparams['network']
fintrinsic_count = SAPMparams['fintrinsic_count']
vintrinsic_count = SAPMparams['vintrinsic_count']
sem_region_list = SAPMparams['sem_region_list']
nclusterlist = SAPMparams['nclusterlist']
tsize = SAPMparams['tsize']
tplist_full = SAPMparams['tplist_full']
tcdata_centered = SAPMparams['tcdata_centered']
tcdata_centered_original = SAPMparams['tcdata_centered_original']
ctarget = SAPMparams['ctarget']
csource = SAPMparams['csource']
dtarget = SAPMparams['dtarget']
dsource = SAPMparams['dsource']
fintrinsic_region = SAPMparams['fintrinsic_region']
Mconn = SAPMparams['Mconn']
Minput = SAPMparams['Minput']
timepoint = SAPMparams['timepoint']
epoch = SAPMparams['epoch']
latent_flag = SAPMparams['latent_flag']
reciprocal_flag = SAPMparams['reciprocal_flag']

ntime, NP = np.shape(tplist_full)
Nintrinsics = vintrinsic_count + fintrinsic_count
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# repeat the process for each participant-----------------------------------------------------------------
betalimit = 3.0
epochnum = 0
SAPMresults = []
first_pass_results = []
second_pass_results = []
beta_init_record = []
for nperson in range(1):  # test with data from one person
	if verbose:
		print('starting person {} at {}'.format(nperson, time.ctime()))
	tp = tplist_full[epochnum][nperson]['tp']
	tsize_total = len(tp)
	nruns = nruns_per_person[nperson]

	# get tc data for each region/cluster
	rnumlist = []
	clustercount = np.cumsum(nclusterlist)
	for aa in range(len(clusterlist)):
		x = np.where(clusterlist[aa] < clustercount)[0]
		rnumlist += [x[0]]

	Sinput = []
	# Sinput_scalefactor = np.zeros(len(clusterlist))
	for nc, cval in enumerate(clusterlist):
		tc1 = tcdata_centered[cval, tp]
		Sinput.append(tc1)
	Sinput = np.array(Sinput)

	if normalizevar:
		Sinput_original = []
		# Sinput_scalefactor = np.zeros(len(clusterlist))
		for nc, cval in enumerate(clusterlist):
			tc1 = tcdata_centered_original[cval, tp]
			Sinput_original.append(tc1)
		Sinput_original = np.array(Sinput_original)
	else:
		Sinput_original = copy.deepcopy(Sinput)

	print('--------setup stage-----------------------------------')
	print('std of normalized data:  {}'.format(np.std(Sinput, axis=1)))
	print('std of original data:  {}'.format(np.std(Sinput_original, axis=1)))
	print('------------------------------------------------------')

	# setup fixed intrinsic based on the model paradigm
	# need to account for timepoint and epoch....
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

		ftemp = fintrinsic_base[0, et1:et2]
		fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])
		print('shape of fintrinsic1 is {}'.format(np.shape(fintrinsic1)))
		if np.var(ftemp) > 1.0e-3:
			Sint = Sinput[fintrinsic_region, :]
			Sint = Sint - np.mean(Sint)
			# need to add constant to fit values
			G = np.concatenate((fintrinsic1[np.newaxis, :], np.ones((1, tsize_total))), axis=0)
			b, fit, R2, total_var, res_var = pysem.general_glm(Sint, G)
			beta_int1 = b[0]
		else:
			beta_int1 = 0.0
	else:
		beta_int1 = 0.0
		fintrinsic1 = []

	lastgood_beta_int1 = copy.deepcopy(beta_int1)

	# initialize beta values-----------------------------------
	nbeta = len(csource)
	if isinstance(betascale, str):
		if betascale == 'shotgun':
			beta_initial = pysapm.betaval_init_shotgun(initial_Lweight, csource, ctarget, Sinput, Minput, Mconn,
												fintrinsic_count,
												vintrinsic_count, beta_int1, fintrinsic1, nreps=10000)
			beta_initial = beta_initial[np.newaxis, :]
			nitermax_stage1 = 0
		else:
			# read saved beta_initial values
			b = np.load(betascale, allow_pickle=True).flat[0]
			beta_initial = b['beta_initial']
			beta_initial = beta_initial[np.newaxis, :]
			nitermax_stage1 = 0
		nsteps_stage1 = 1
	# beta_initial[0,latent_flag > 0] = 1.0
	else:
		nsteps_stage1 = copy.deepcopy(initial_nsteps_stage1)
		beta_initial = betascale * np.random.randn(nsteps_stage1, nbeta)

		# np.save(r'E:\SAPMresults_Dec2022\beta_initial_test_values.npy',{'beta_initial':beta_initial})
		# print('beta_initial = {}'.format(beta_initial))
		# beta_setup = np.load(r'E:\SAPMresults_Dec2022\beta_initial_test_values.npy', allow_pickle=True).flat[0]
		# beta_initial = beta_setup['beta_initial']
		# print('beta_initial = {}'.format(beta_initial))

		# beta_initial[:,latent_flag > 0] = 1.0
		nitermax_stage1 = copy.deepcopy(initial_nitermax_stage1)

	# initialize deltavals
	delta_initial = np.ones(len(dtarget))
	deltascale = np.std(Sinput, axis=1)
	meanscale = np.mean(deltascale)
	for rr in range(len(dtarget)):
		delta_initial[rr] = deltascale[dtarget[rr]] / deltascale[dsource[rr]]  # make initial deltavals proportional to std's of regions

	# initialize
	results_record = []
	ssqd_record = []

	# stage 1 - test the initial betaval settings
	stage1_ssqd = np.zeros(nsteps_stage1)
	stage1_results = []
	for ns in range(nsteps_stage1):
		ssqd_record_stage1 = []
		beta_init_record.append({'beta_initial': beta_initial[ns, :]})

		# initalize Sconn
		betavals = copy.deepcopy(beta_initial[ns, :])  # initialize beta values at zero
		lastgood_betavals = copy.deepcopy(betavals)
		# deltavals = np.ones(len(dsource))
		deltavals = copy.deepcopy(delta_initial)
		lastgood_deltavals = copy.deepcopy(deltavals)

		alphalist = initial_alpha * np.ones(nbeta)
		alphabint = copy.deepcopy(initial_alpha)
		alpha = copy.deepcopy(initial_alpha)
		Lweight = copy.deepcopy(initial_Lweight)
		dval = copy.deepcopy(initial_dval)

		# # starting point for optimizing intrinsics with given betavals----------------------------------------------------
		Mconn[ctarget, csource] = copy.deepcopy(betavals)
		Minput[dtarget, dsource] = copy.deepcopy(deltavals)
		fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
																 vintrinsic_count, beta_int1, fintrinsic1)
		ssqd, error, costfactor = pysapm.sapm_error_function_V2(Sinput, fit, Lweight,
									  betavals)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

		ssqd_starting = copy.deepcopy(ssqd)
		ssqd_old = copy.deepcopy(ssqd)
		ssqd_record += [ssqd]

		iter = 0
		converging = True
		dssq_record = np.ones(3)
		dssq_count = 0
		sequence_count = 0

		while alpha > alpha_limit and iter < nitermax_stage1 and converging:
			iter += 1
			# betavals, beta_int1, fit, updatebflag, updatebintflag, dssq_db, dssq_dbeta1, ssqd, alphalist, alphabint = \
			#     update_betavals_sequentially(Sinput, Minput, Mconn, betavals, ctarget, csource, dval,
			#                                         fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight,
			#                                         alphalist, alphabint, latent_flag)

			betavals, deltavals, beta_int1, fit, dssq_db, dssq_dd, dssq_dbeta1, ssqd_original, ssqd, alpha, alphabint = \
				pysapm.update_betavals_V2(Sinput, Minput, Mconn, betavals, deltavals, betalimit,
								   ctarget, csource, dtarget, dsource, dval, fintrinsic_count,
								   vintrinsic_count, beta_int1, fintrinsic1, Lweight, alpha, alphabint,
								   latent_flag=latent_flag)

			ssqd_record_stage1 += [ssqd]

			if ssqd > ssqd_original:
				alpha *= 0.5
				alphabint *= 0.5
				betavals = copy.deepcopy(lastgood_betavals)  # no improvement, so don't update
				deltavals = copy.deepcopy(lastgood_deltavals)
				beta_int1 = copy.deepcopy(lastgood_beta_int1)
			else:
				lastgood_betavals = copy.deepcopy(betavals)
				lastgood_deltavals = copy.deepcopy(deltavals)
				lastgood_beta_int1 = copy.deepcopy(beta_int1)

			Mconn[ctarget, csource] = copy.deepcopy(betavals)
			Minput[dtarget, dsource] = copy.deepcopy(deltavals)
			fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
																	 vintrinsic_count, beta_int1, fintrinsic1)
			ssqd, error, costfactor = pysapm.sapm_error_function_V2(Sinput, fit, Lweight,
										  betavals)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

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

			if verbose:
				print(
					'SAPM  {} stage1 pass {} iter {} alpha {:.3e}  ssqd {:.2f} change {:.3f}  percent {:.1f}  R2 avg {:.3f}  R2 total {:.3f}'.format(
						nperson, ns, iter, alpha, ssqd, ssqchange, 100. * ssqd / ssqd_starting, R2avg, R2total))
			ssqd_old = copy.deepcopy(ssqd)
		# now repeat it ...
		stage1_ssqd[ns] = ssqd
		stage1_results.append({'betavals': betavals, 'deltavals': deltavals})

	# get the best betavals from stage1 so far ...
	x = np.argmin(stage1_ssqd)
	betavals = stage1_results[x]['betavals']
	deltavals = stage1_results[x]['deltavals']

	# stage 2
	# # starting point for optimizing intrinsics with given betavals----------------------------------------------------
	if verbose: print('starting stage 2 ....')
	lastgood_betavals = copy.deepcopy(betavals)
	alpha = copy.deepcopy(initial_alpha)
	alphabint = copy.deepcopy(initial_alpha)
	Lweight = copy.deepcopy(initial_Lweight)
	dval = copy.deepcopy(initial_dval)

	# # starting point for optimizing intrinsics with given betavals----------------------------------------------------
	Mconn[ctarget, csource] = copy.deepcopy(betavals)
	Minput[dtarget, dsource] = copy.deepcopy(deltavals)
	fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
															 vintrinsic_count, beta_int1, fintrinsic1)
	ssqd, error, costfactor = pysapm.sapm_error_function_V2(Sinput, fit, Lweight, betavals)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

	ssqd_starting = copy.deepcopy(ssqd)
	ssqd_old = copy.deepcopy(ssqd)
	ssqd_record += [ssqd]

	iter = 0
	converging = True
	dssq_record = np.ones(3)
	dssq_count = 0
	sequence_count = 0
	ssqd_stage2_record = []
	costfactor_stage2_record = []

	while alpha > alpha_limit and iter < nitermax and converging:
		iter += 1
		betavals, deltavals, beta_int1, fit, dssq_db, dssq_dd, dssq_dbeta1, ssqd_original, ssqd, alpha, alphabint = \
			pysapm.update_betavals_V2(Sinput, Minput, Mconn, betavals, deltavals, betalimit,
							   ctarget, csource, dtarget, dsource, dval, fintrinsic_count,
							   vintrinsic_count, beta_int1, fintrinsic1, Lweight, alpha, alphabint,
							   latent_flag=latent_flag)

		ssqd_record_stage1 += [ssqd]

		if ssqd > ssqd_original:
			alpha *= 0.5
			alphabint *= 0.5
			betavals = copy.deepcopy(lastgood_betavals)  # no improvement, so don't update
			deltavals = copy.deepcopy(lastgood_deltavals)
			beta_int1 = copy.deepcopy(lastgood_beta_int1)
			sequence_count = 0
		else:
			lastgood_betavals = copy.deepcopy(betavals)
			lastgood_deltavals = copy.deepcopy(deltavals)
			lastgood_beta_int1 = copy.deepcopy(beta_int1)
			sequence_count += 1
			if sequence_count > 3:
				alpha *= 1.3
				alphabint *= 1.3
				sequence_count = 0

		Mconn[ctarget, csource] = copy.deepcopy(betavals)
		Minput[dtarget, dsource] = copy.deepcopy(deltavals)
		fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
																 vintrinsic_count, beta_int1, fintrinsic1)
		ssqd, error, costfactor = pysapm.sapm_error_function_V2(Sinput, fit, Lweight, betavals)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv
		ssqd_stage2_record += [ssqd]
		costfactor_stage2_record += [costfactor]

		err_total = Sinput - fit
		Smean = np.mean(Sinput)
		errmean = np.mean(err_total)

		R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
		R2avg = np.mean(R2list)
		R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)

		results_record.append({'Sinput': Sinput, 'fit': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})

		ssqchange = ssqd - ssqd_original
		# if np.abs(ssqchange) < 1e-5: converging = False

		if verbose:
			print(
				'SAPM  {} beta vals:  iter {} alpha {:.3e}  ssqd {:.2f} change {:.3f}  percent {:.1f}  R2 avg {:.3f}  R2 total {:.3f}'.format(
					nperson, iter, alpha, ssqd, ssqchange, 100. * ssqd / ssqd_starting, R2avg, R2total))
		ssqd_old = copy.deepcopy(ssqd)
	# now repeat it ...

	if normalizevar:
		# the data have been fit to data with normalized variance ... now use this to determine the
		# fit parameters for the original non-normalized data
		print('------------------------------------------------------')
		print('std of normalized data:  {}'.format(np.std(Sinput, axis=1)))
		print('std of original data:  {}'.format(np.std(Sinput_original, axis=1)))
		print('------------------------------------------------------')

		SAPMconversion = pysapm.sem_physio_model_incremental_change(Sinput, Sinput_original, betavals, deltavals, Minput,
															 Mconn, Mintrinsic, betalimit, ctarget, csource, dtarget,
															 dsource,
															 fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1,
															 latent_flag=latent_flag,
															 verbose=True)
		Sinput = copy.deepcopy(Sinput_original)
		betavals = copy.deepcopy(SAPMconversion['betavals'])
		deltavals = copy.deepcopy(SAPMconversion['deltavals'])
		deltavals = copy.deepcopy(SAPMconversion['deltavals'])
		Minput = copy.deepcopy(SAPMconversion['Minput'])
		Mconn = copy.deepcopy(SAPMconversion['Mconn'])
		R2avg = copy.deepcopy(SAPMconversion['R2avg'])
		R2total = copy.deepcopy(SAPMconversion['R2total'])

	# fit the results now to determine output signaling from each region
	Mconn[ctarget, csource] = copy.deepcopy(betavals)
	Minput[dtarget, dsource] = copy.deepcopy(deltavals)
	fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count,
															 beta_int1, fintrinsic1)
	Sconn = Meigv @ Mintrinsic  # signalling over each connection

	entry = {'Sinput': Sinput, 'Sconn': Sconn, 'beta_int1': beta_int1, 'Mconn': Mconn, 'Minput': Minput,
			 'R2total': R2total, 'R2avg': R2avg, 'Mintrinsic': Mintrinsic, 'fintrinsic_count': fintrinsic_count,
			 'vintrinsic_count': vintrinsic_count,
			 'Meigv': Meigv, 'betavals': betavals, 'deltavals': deltavals, 'fintrinsic1': fintrinsic1,
			 'clusterlist': clusterlist,
			 'fintrinsic_base': fintrinsic_base, 'ssqd_stage2_record':ssqd_stage2_record, 'costfactor_stage2_record':costfactor_stage2_record}

	# person_results.append(entry)
	SAPMresults.append(copy.deepcopy(entry))

	stoptime = time.ctime()

#-------------------end-------------------------------------------
NP = len(SAPMresults)
R2list = np.zeros(len(SAPMresults))
R2list2 = np.zeros(len(SAPMresults))
ssqd = np.zeros(len(SAPMresults))
costfactor = np.zeros(len(SAPMresults))
for nperson in range(NP):
	R2list[nperson] = SAPMresults[nperson]['R2avg']
	R2list2[nperson] = SAPMresults[nperson]['R2total']
	ssqd[nperson] = SAPMresults[nperson]['ssqd_stage2_record'][-1]
	costfactor[nperson] = SAPMresults[nperson]['costfactor_stage2_record'][-1]
# R2list[nperson] = SAPMresults[nperson][0]['R2total']
print('SAPM parameters computed for {} data sets'.format(NP))
print('R2 values averaged {:.3f} {} {:.3f}'.format(np.mean(R2list), chr(177), np.std(R2list)))
print('average R2 values ranged from {:.3f} to {:.3f}'.format(np.min(R2list), np.max(R2list)))
print('Total R2 values were {:.3f} {} {:.3f}'.format(np.mean(R2list2), chr(177), np.std(R2list2)))
print('Total ssqd values were {:.3f} {} {:.3f}'.format(np.mean(ssqd), chr(177), np.std(ssqd)))
print('Total costfactor values were {:.3f} {} {:.3f}'.format(np.mean(costfactor), chr(177), np.std(costfactor)))