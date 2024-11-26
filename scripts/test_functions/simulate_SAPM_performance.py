# simulate random networks and see which ones result in biased (non-zero) B values
# sys.path.extend([r'C:\Users\Stroman\PycharmProjects\pantheon\venv'])

import numpy as np
import os
import copy
import pysapm
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import scipy

# import sklearn.neural_network as NN
from tensorflow.keras import models, layers
import tensorflow as tf

import pybasissets
import pysapm

from sklearn.linear_model import LinearRegression
import sklearn
from sklearn.decomposition import PCA
import time



def generate_random_networkfile(networkmodel_name, nregions, fintrinsic_count, vintrinsic_count, tsize, nclusters = 5):
	# network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = \
	# 		pysapm.load_network_model_w_intrinsics(networkfile)

	# generate a network with 2 to 4 inputs per region
	# with 2 regions having variable latent input
	# ntotal = len(sem_region_list)
	# nregions = ntotal - vintrinsic_count - fintrinsic_count

	Nlatents = fintrinsic_count + vintrinsic_count
	ntotal = nregions + fintrinsic_count + vintrinsic_count
	sem_region_list = ['region{}'.format(x) for x in range(nregions)]
	nclusterlist = nclusters*np.ones(nregions)

	# simulate fintrinsic_base
	nstim = 2
	tperiod = np.floor(tsize/(2*nstim+1)).astype(int)
	fintrinsic_base = np.zeros(tsize)
	for nn in range(nstim):
		t1 = (2*nn+1)*tperiod
		t2 = (2*nn+2)*tperiod
		fintrinsic_base[t1:t2] = 1
	fintrinsic_base = np.array(fintrinsic_base)
	fintrinsic_base -= np.mean(fintrinsic_base)

	ninput_options = list(range(2,5))
	ninput_record = np.zeros(nregions)
	network_model = []
	for nn in range(nregions):
		ninput = np.random.choice(ninput_options, replace=False)
		ninput_record[nn] = copy.deepcopy(ninput)
		target = sem_region_list[nn]
		source_options = list(range(nregions))
		i = source_options.index(nn)
		del source_options[i]
		sourcenums = np.random.choice(source_options,ninput, replace=False)
		sources = [sem_region_list[x] for x in sourcenums]
		entry = {'target':target, 'sources':sources, 'ninput':ninput}
		network_model.append(entry)

	# add latents
	latent_inputs = np.random.choice(list(range(nregions)),Nlatents, replace=False)
	if fintrinsic_count > 0:
		latents_names = ['fintrinsic1']
	else:
		latents_names = []
	for nn in range(vintrinsic_count):
		latents_names += ['vintrinsic{}'.format(nn+1)]
	for nn in range(Nlatents):
		sources = copy.deepcopy(network_model[latent_inputs[nn]]['sources'])
		sources += [latents_names[nn]]
		network_model[latent_inputs[nn]]['sources'] = copy.deepcopy(sources)
		ninput_record[latent_inputs[nn]] += 1


	# write out the network-------------------------------------------
	# 'connections' sheet in Excel file ...
	ninput_max = np.max(ninput_record).astype(int)

	columntitles = ['target']
	for nn in range(ninput_max):
		columntitles += ['source{}'.format(nn+1)]
	rowtitles = []
	for nn in range(nregions):
		rowtitles += ['{}'.format(nn)]

	M = np.zeros((nregions,ninput_max+1)).astype(str)
	for nn in range(nregions):
		M[nn,0] = network_model[nn]['target']
		for mm in range(ninput_max):
			if mm < ninput_record[nn]:
				M[nn, mm+1] = network_model[nn]['sources'][mm]
			else:
				M[nn, mm+1] = ''

	df_connections = pd.DataFrame(M, columns=columntitles, index=rowtitles)

	# 'nclusters' sheet in Excel file ...
	columntitles = ['name', 'nclusters']
	rowtitles = []
	for nn in range(nregions+Nlatents):
		rowtitles += ['{}'.format(nn)]

	M = np.zeros((nregions+Nlatents,2)).astype(str)
	for nn in range(nregions):
		M[nn,0] = sem_region_list[nn]
		M[nn,1] = nclusters
	for nn in range(Nlatents):
		M[nregions+nn,0] = latents_names[nn]
		M[nregions+nn,1] = 1

	df_nclusters = pd.DataFrame(M, columns=columntitles, index=rowtitles)

	# 'fintrinsic1' sheet in Excel file ...
	paradigm = {'time':list(range(tsize))  , 'paradigms_BOLD':fintrinsic_base}
	df_fintrinsic1 = pd.DataFrame(paradigm)

	with pd.ExcelWriter(networkmodel_name) as writer:
		df_connections.to_excel(writer, sheet_name='connections')
		df_nclusters.to_excel(writer, sheet_name='nclusters')
		df_fintrinsic1.to_excel(writer, sheet_name='fintrinsic1')

	return networkmodel_name


def generate_simulated_data_and_results(networkmodel_name, tsize, nruns, TR, normalize_var = True):
	network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = pysapm.load_network_model_w_intrinsics(networkmodel_name)

	# example results
	resultsfilename = r'E:\SAPMresults_Dec2022\Pain_1202023213_test_results.npy'
	results = np.load(resultsfilename, allow_pickle=True)
	ndata_samples = len(results)
	x = np.random.choice(ndata_samples, 1)[0]
	betavals_sample = copy.deepcopy(results[x]['betavals'])
	deltavals_sample = copy.deepcopy(results[x]['deltavals'])
	Mintrinsic_sample = copy.deepcopy(results[x]['Mintrinsic'])
	ni, tsizei = np.shape(Mintrinsic_sample)
	if tsizei > tsize:
		Mintrinsic_sample = Mintrinsic_sample[:,:tsize]
	if tsizei < tsize:
		rep = np.floor(tsize/tsizei).astype(int) + 1
		Mtemp = np.zeros((ni,tsize))
		for vv in range(ni):
			temp = list(Mintrinsic_sample[vv,:])*rep
			Mtemp[vv,:] = np.array(temp[:tsize])
		Mintrinsic_sample = copy.deepcopy(Mtemp)


	Nintrinsic = fintrinsic_count + vintrinsic_count
	beta_list = []
	nbeta = 0
	targetnumlist = []
	beta_id = []
	sourcelist = []
	for nn in range(len(network)):
		target = network[nn]['targetnum']
		sources = network[nn]['sourcenums']
		targetnumlist += [target]
		for mm in range(len(sources)):
			source = sources[mm]
			sourcelist += [source]
			betaname = '{}_{}'.format(source, target)
			entry = {'name': betaname, 'number': nbeta, 'pair': [source, target]}
			beta_list.append(entry)
			beta_id += [1000 * source + target]
			nbeta += 1

	ncon = nbeta - Nintrinsic

	# reorder to put intrinsic inputs at the end-------------
	beta_list2 = []
	beta_id2 = []
	x = np.where(np.array(sourcelist) < nregions)[0]
	for xx in x:
		beta_list2.append(beta_list[xx])
		beta_id2 += [beta_id[xx]]
	for sn in range(nregions, nregions + Nintrinsic):
		x = np.where(np.array(sourcelist) == sn)[0]
		for xx in x:
			beta_list2.append(beta_list[xx])
			beta_id2 += [beta_id[xx]]

	for nn in range(len(beta_list2)):
		beta_list2[nn]['number'] = nn

	beta_list = beta_list2
	beta_id = beta_id2

	beta_pair = []
	# Mconn = np.zeros((nbeta, nbeta))
	Mconn = np.zeros((nregions + Nintrinsic, nregions + Nintrinsic))

	count = 0
	for nn in range(len(network)):
		target = network[nn]['targetnum']
		sources = network[nn]['sourcenums']
		for mm in range(len(sources)):
			source = sources[mm]
			conn1 = beta_id.index(source * 1000 + target)

			count += 1
			beta_pair.append([target, source])
			Mconn[target, source] = count

			if source >= nregions:  # intrinsic input
				# conn2 = conn1
				# Mconn[conn1, conn2] = 1  # set the intrinsic beta values
				Mconn[source, source] = 1  # set the intrinsic beta values

	# prep to index Mconn for updating beta values
	beta_pair = np.array(beta_pair)
	ctarget = beta_pair[:, 0]
	csource = beta_pair[:, 1]

	latent_flag = np.zeros(len(ctarget))
	found_latent_list = []
	for nn in range(len(ctarget)):
		# if csource[nn] >= ncon  and ctarget[nn] < ncon:
		if csource[nn] >= nregions and ctarget[nn] < nregions:
			found_latent_list += [csource[nn]]
			occurence = np.count_nonzero(found_latent_list == csource[nn])
			latent_flag[nn] = csource[nn] - nregions + 1

	reciprocal_flag = np.zeros(len(ctarget))
	for nn in range(len(beta_list)):
		pair1 = beta_list[nn]['pair']
		for mm in range(len(beta_list)):
			if mm != nn:
				pair2 = beta_list[mm]['pair']
				if (pair1[0] == pair2[1]) & (pair1[1] == pair2[0]):
					reciprocal_flag[nn] = 1
					reciprocal_flag[mm] = 1


	# setup Minput matrix--------------------------------------------------------------
	# Sconn = Mconn @ Sconn    # propagate the intrinsic inputs through the network
	# Sinput = Minput @ Mconn
	# Minput = np.zeros((nregions, nbeta))  # mixing of connections to model the inputs to each region
	Minput = np.zeros((nregions, nregions + Nintrinsic))  # mixing of connections to model the inputs to each region
	betanamelist = [beta_list[a]['name'] for a in range(nbeta)]
	for nn in range(len(network)):
		target = network[nn]['targetnum']
		sources = network[nn]['sourcenums']
		for mm in range(len(sources)):
			source = sources[mm]
			betaname = '{}_{}'.format(source, target)
			x = betanamelist.index(betaname)
			# Minput[target, x] = 1
			Minput[target, source] = 1

	# flag which Minput values can be varied
	# (keep one output from each region at 1, vary the other outputs)
	Dvarflag = copy.deepcopy(Minput)
	for nn in range(nregions):
		onesource = copy.deepcopy(Minput[:, nn])
		c = np.where(onesource > 0)[0]
		if len(c) > 0: onesource[c[0]] = 0
		Dvarflag[:, nn] = copy.deepcopy(onesource)
	for nn in range(nregions, nregions + Nintrinsic):
		Dvarflag[:, nn] = 0
	dtarget, dsource = np.where(Dvarflag > 0)

	# now simulate data to match a set of chosen B and D values
	#----------------------------------------------------------
	betavals = (np.random.normal(0,1,len(ctarget)))

	# betavals = copy.deepcopy(betavals_sample)
	# scalei = 1.0 + 1e-6 * np.random.randn(len(betavals))
	# betavals *= scalei

	c = np.where(latent_flag > 0)[0]
	betavals[c] = 1.0 	# + 0.05*np.random.normal(0,1,len(c))

	# deltavals = 1.0 + 0.0*np.random.normal(0,1,len(dtarget))
	deltavals = np.ones(len(dtarget))
	c = np.where(dsource > 0)[0]
	deltavals[c] = 1.0  # latent inputs

	Mconn[ctarget,csource] = betavals
	Minput[dtarget,dsource] = deltavals

	# generate models of changes in metabolic demand, then convolve with HRF

	# Mintrinsic_base = np.random.normal(0,1,(vintrinsic_count, tsize))
	Mintrinsic_base = generate_intrinsic_input_models(vintrinsic_count, tsize, nruns, linear_indep = False)
	Mintrinsic_base -= np.repeat(np.mean(Mintrinsic_base, axis = 1)[:, np.newaxis], tsize, axis = 1)

	# convolve with HRF
	hrf = pybasissets.HRF(TR)
	Mintrinsic = copy.deepcopy(Mintrinsic_base)
	for vv in range(vintrinsic_count):
		Mintrinsic[vv,:] = np.convolve(Mintrinsic[vv,:], hrf, mode='same')
	Mintrinsic -= np.repeat(np.mean(Mintrinsic, axis = 1)[:, np.newaxis], tsize, axis = 1)

	if fintrinsic_count > 0:
		fintrinsic1 = np.array(list(fintrinsic_base[0,:]) * nruns_per_person)[np.newaxis,:]
		Mintrinsic = np.concatenate((fintrinsic1, Mintrinsic),axis=0)
		Mintrinsic_base = np.concatenate((fintrinsic1, Mintrinsic_base),axis=0)

	# replace with sample from actual data
	# Mintrinsic = copy.deepcopy(Mintrinsic_sample)
	# Mintrinsic_base = copy.deepcopy(Mintrinsic_sample)
	# ni,tsizei = np.shape(Mintrinsic_base)
	# scalei = 1.0 + 0.1*np.random.randn(ni)
	# for vv in range(ni):
	# 	Mintrinsic[vv,:] *= scalei[vv]
	# 	Mintrinsic_base[vv,:] *= scalei[vv]

	e, v = np.linalg.eig(Mconn)  # Mconn is nbeta x nbeta   where nbeta = ncon + Nintrinsic
	Meigv = np.real(v[:, -Nintrinsic:])
	# Meigv = (Gram_Schmidt_orthogonalization(Meigv.T)).T   # make them a set of linearly indpendent eigvenvectors

	Sconn = Meigv @ Mintrinsic

	if normalize_var:
		nr,tsize = np.shape(Sconn)
		tcdata_std = np.zeros(nr)
		# # normalize the data to have the same variance, for each person
		for nn in range(nr):
			tcdata_std[nn] = np.std(Sconn[nn, :])
			Sconn[nn, :] /= tcdata_std[nn]

	Sinput = Minput @ Sconn

	Sconn_base = Meigv @ Mintrinsic_base

	if normalize_var:
		nr, tsize = np.shape(Sconn_base)
		tcdata_std = np.zeros(nr)
		# # normalize the data to have the same variance, for each person
		for nn in range(nr):
			tcdata_std[nn] = np.std(Sconn_base[nn, :])
			Sconn_base[nn, :] /= tcdata_std[nn]

	Sinput_base = Minput @ Sconn_base

	dataset = {'Sinput':Sinput, 'Sconn':Sconn, 'Sinput_base':Sinput_base, 'Sconn_base':Sconn_base,
			   'Minput':Minput, 'Mconn':Mconn, 'Mintrinsic':Mintrinsic, 'Mintrinsic_base':Mintrinsic_base,
			   'ctarget':ctarget, 'csource':csource, 'dtarget':dtarget, 'dsource':dsource, 'fintrinsic_base':fintrinsic_base,
			   'latent_flag':latent_flag, 'reciprocal_flag':reciprocal_flag, 'beta_list':beta_list}

	return dataset


def generate_intrinsic_input_models(vintrinsic, tsize, nruns, linear_indep = False):
	Mintrinsic_base = np.zeros((vintrinsic,tsize))
	tsize_small = np.floor(tsize/nruns).astype(int)

	nblocks_max = np.floor(tsize_small/8).astype(int)
	nblocks = np.random.choice(list(range(1,nblocks_max)), vintrinsic, replace=False)
	for vv in range(vintrinsic):
		block_length = np.floor(tsize_small/(2*nblocks[vv]+1)).astype(int)
		gap_length = np.floor((tsize_small - block_length*nblocks[vv])/(nblocks[vv]+1)).astype(int)
		isi = gap_length+block_length
		nmax = np.floor((tsize_small-gap_length)/isi).astype(int)
		tstart = (gap_length + np.array(range(nmax))*isi).astype(int)
		# tstart = np.random.choice(tsize-block_length, nblocks, replace=False)

		v1 = np.zeros(tsize_small)
		for nn in range(nblocks[vv]):
			t1 = tstart[nn]
			t2 = t1+block_length
			v1[t1:t2] = 1.0

		v1 = np.array(list(v1)*nruns)
		v2 = np.zeros(tsize)
		v2[:len(v1)] = v1
		v2-= np.mean(v2)

		Mintrinsic_base[vv,:] = copy.deepcopy(v2)

	if linear_indep:
		# make linearly independent
		pca = PCA(n_components=vintrinsic)
		pca.fit(Mintrinsic_base)
		Mintrinsic_base = pca.components_

	return Mintrinsic_base



def generate_sim_clusterdata(clusterdataname, nregions, nclusters = 5):
	sem_region_list = ['region{}'.format(x) for x in range(nregions)]
	nv = 10
	nvox = nv*nv   # arbitrary for simulation
	cluster_properties = []
	for nn in range(nregions):
		cx,cy = np.mgrid[0:nv,0:nv]
		cx = cx.flatten()
		cy = cy.flatten()
		cz = nn*np.ones(nvox)
		IDX = list(range(nclusters))*np.ceil(nvox/nclusters).astype(int)
		IDX = np.reshape(IDX,(np.ceil(nvox/nclusters).astype(int),nclusters))
		IDX = (IDX.T).flatten()
		entry = {'cx':cx, 'cy':cy, 'cz':cz, 'IDX':IDX, 'nclusters':nclusters, 'rname':sem_region_list[nn], 'regionindex':nn, 'regionnum':nn, 'occurrence':0}
		cluster_properties.append(entry)

	clusterdef = {'cluster_properties':cluster_properties, 'template_img':[], 'regionmap_img':[]}
	np.save(clusterdataname, clusterdef)

	return clusterdataname





def convert_results(simnums, simdatanametag):
	resultsdir = r'E:/sim_data3'

	sim_record = []
	for simnum in simnums:
		SAPMresultsname = os.path.join(resultsdir, 'sim_results_sim{}_corr.npy'.format(simnum))
		SAPMparametersname = os.path.join(resultsdir, 'sim_params_sim{}.npy'.format(simnum))

		results = np.load(SAPMresultsname, allow_pickle=True)
		params = np.load(SAPMparametersname, allow_pickle=True).flat[0]
		csource = params['csource']
		ctarget = params['ctarget']
		dsource = params['dsource']
		dtarget = params['dtarget']
		Minput = results[0]['Minput']
		Mconn = results[0]['Mconn']

		NP = len(results)
		for nn in range(NP):
			betavals = results[nn]['betavals']
			deltavals = results[nn]['deltavals']
			if nn == 0:
				betarecord = copy.deepcopy(betavals[:,np.newaxis])
				deltarecord = copy.deepcopy(deltavals[:,np.newaxis])
			else:
				betarecord = np.concatenate((betarecord, betavals[:,np.newaxis]),axis=1)
				deltarecord = np.concatenate((deltarecord, deltavals[:,np.newaxis]),axis=1)
		beta_mean = np.mean(betarecord, axis=1)
		beta_std = np.std(betarecord, axis=1)
		delta_mean = np.mean(deltarecord, axis=1)
		delta_std = np.std(deltarecord, axis=1)

		Minput[ctarget,csource] = 1
		Mconn[ctarget,csource] = 1

		entry = {'Minput':Minput, 'Mconn':Mconn, 'beta_mean':beta_mean, 'beta_std':beta_std,
				 'delta_mean':delta_mean, 'delta_std':delta_std, 'csource':csource, 'ctarget':ctarget}
		sim_record.append(entry)

	simdataname = os.path.join(resultsdir, 'simdata_' + simdatanametag + '.npy')
	np.save(simdataname, sim_record)

	print('saved data in {}'.format(simdataname))


def find_red_flags(Minput):
	nregion, ntotal = np.shape(Minput)
	Nintrinsic = ntotal-nregion
	vnums = [x for x in range(nregion) if (Minput[x,-(Nintrinsic-1):] == 1).any() ]

	flagcount = 0
	for nn in range(nregion):
		if (Minput[nn,vnums] == 1).all():
			flagcount += 1


	# flag2 - feedback loops
	flagcount2 = 0
	looprecord = []
	for vv in vnums:
		# regions receiving input from region with latent input
		targets1 = np.where(Minput[:,vv] > 0)[0]
		for tt1 in targets1:
			targets2 = np.where(Minput[:,tt1] > 0)[0]
			for tt2 in targets2:
				returnloop  = np.where(Minput[:,tt2] > 0)[0]
				if vv in returnloop:
					flagcount2 += 1   # found a feed-back loop
					looprecord.append({'loop':[vv,tt1,tt2]})

	# flag3 common inputs
	flagcount3 = 0
	for vv in vnums:
		sources1 = np.where(Minput[vv, :nregion] > 0)[0]
		# common inputs
		for vv2 in sources1:
			ccount = 0
			for rr in range(nregion):
				sources2 = np.where(Minput[rr, :nregion] > 0)[0]
				if (vv in sources2) and (vv2 in sources2):
					ccount += 1
			if ccount > 1:
				flagcount3 += (ccount-1)


	# flag4 multiple close inputs from latents
	flagcount4 = 0
	latcount = np.zeros(len(vnums))
	for vv in vnums:
		sources1 = np.where(Minput[vv, :nregion] > 0)[0]
		for ss in sources1:
			if ss in vnums:
				cc = np.where(vnums == ss)[0]
				latcount[cc] += 1
	for ss in latcount:
		if ss > 0:
			flagcount4 += 10**(ss-1)
	flagcount4 = int(flagcount4)

	# flag5 imbalance of network
	flagcount5 = 0
	sourcecount = np.zeros(nregion)
	for rr in range(nregion):
		sources = np.where(Minput[rr, :nregion] > 0)[0]
		sourcecount[rr] = len(sources)
	flagcount5 = np.sum((sourcecount - np.mean(sourcecount))**2)


	# flag6 imbalance of sources
	flagcount6 = 0
	sourcecount = np.zeros(nregion)
	for rr in range(nregion):
		scount = np.sum(Minput[:,rr])
		sourcecount[rr] = scount
	flagcount6 = np.sum((sourcecount - np.mean(sourcecount))**2)

	# flag7 the number of times a region with latent input is a source for another region
	flagcount6 = 0
	sourcecount = np.zeros(len(vnums))
	for ii,vv in enumerate(vnums):
		count = np.sum(Minput[:, vv])
		sourcecount[ii] = count
	maxsourcecount = np.max(sourcecount)
	minsourcecount = np.min(sourcecount)
	flagcount7 = maxsourcecount**2 - minsourcecount**2


	# flag8 a region with latent input is never a source for other regions
	flagcount8 = 0
	sourcecount = np.zeros(len(vnums))
	for ii,vv in enumerate(vnums):
		count = np.sum(Minput[:, vv])
		sourcecount[ii] = count
	flagcount8 = np.min(sourcecount)

	return flagcount, flagcount2, flagcount3, flagcount4, flagcount5, flagcount6, flagcount7, flagcount8




# def run_null_test_on_network(nsims, networkmodel, cnums, regiondataname, clusterdataname, timepoint = 'all', epoch = 'all', betascale = 0.1):
# 	resultsdir, networkfilename = os.path.split(networkmodel)
# 	networkbasename, ext = os.path.splitext(networkfilename)
#
# 	covariatesname = []
# 	null_regiondataname, null_covariates = pysapm.generate_null_data_set(regiondataname, covariatesname, npeople=nsims, variable_variance = False)
#
# 	SAPMresultsname = os.path.join(resultsdir,'null_results.npy')
# 	SAPMparametersname = os.path.join(resultsdir,'null_params.npy')
#
# 	pysapm.SAPMrun_V2(cnums, null_regiondataname, clusterdataname, SAPMresultsname, SAPMparametersname, networkmodel, timepoint,
# 				epoch, betascale = betascale, reload_existing = False, multiple_output = False)
#
# 	# compile stats distributions for each connection
# 	results = np.load(SAPMresultsname, allow_pickle=True)
# 	params = np.load(SAPMparametersname, allow_pickle=True).flat[0]
# 	csource = params['csource']
# 	ctarget = params['ctarget']
# 	rnamelist = params['rnamelist']
# 	fintrinsic_count = params['fintrinsic_count']
# 	vintrinsic_count = params['vintrinsic_count']
# 	rnamelist_full = copy.deepcopy(rnamelist)
# 	if fintrinsic_count > 0: rnamelist_full += ['latent0']
# 	for nn in range(vintrinsic_count): rnamelist_full += ['latent{}'.format(fintrinsic_count+nn)]
# 	ncon = len(results[0]['betavals'])
# 	betavals = np.zeros((ncon,nsims))
# 	for nn in range(nsims): betavals[:,nn] = results[nn]['betavals']
# 	bstats = []
# 	for nn in range(ncon):
# 		conname = '{}-{}'.format(rnamelist_full[csource[nn]], rnamelist_full[ctarget[nn]])
# 		b = copy.deepcopy(betavals[nn,:])
# 		entry = {'name':conname, 'mean':np.mean(b), 'std':np.std(b), 'skewness':scipy.stats.skew(b), 'kurtosis':scipy.stats.kurtosis(b)}
# 		bstats.append(entry)
#
# 	df = pd.DataFrame(bstats)
# 	xlname = os.path.join(resultsdir,networkbasename + '_bstats.xlsx')
# 	with pd.ExcelWriter(xlname) as writer:
# 		df.to_excel(writer, sheet_name='B stats')
#
# 	return xlname
#



#----------------------------------------------------------------------------------
# primary function--------------------------------------------------------------------
def sem_physio_modelV3_copy(Sinput, tsize, Minput, Mintrinsic, Mconn, nruns_per_person, vintrinsic_count, fintrinsic_count, network,
	 					ctarget, csource, dtarget, dsource, timepoint, epoch, latent_flag, reciprocal_flag,
                      	betascale = 0.1, Lweight = 1.0, normalize_var=False, nitermax = 250, verbose = True, initial_nitermax_stage1 = 15,
                      	initial_nsteps_stage1 = 15, beta_initial_vals = []):

	include_stage1b = False

	if fintrinsic_count > 0:
		fintrinsic1 = copy.deepcopy(Mintrinsic[0,:])
	else:
		fintrinsic1 = []

	starttime = time.ctime()
	# initialize gradient-descent parameters--------------------------------------------------------------
	initial_alpha = 1e-3
	initial_Lweight = copy.deepcopy(Lweight)
	initial_dval = 0.05
	alpha_limit = 1.0e-6
	repeat_limit = 2

	# SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
	# # load the data values
	# betanamelist = SAPMparams['betanamelist']
	# beta_list = SAPMparams['beta_list']
	# nruns_per_person = SAPMparams['nruns_per_person']
	# nclusterstotal = SAPMparams['nclusterstotal']
	# rnamelist = SAPMparams['rnamelist']
	# nregions = SAPMparams['nregions']
	# cluster_properties = SAPMparams['cluster_properties']
	# cluster_data = SAPMparams['cluster_data']
	# network = SAPMparams['network']
	# fintrinsic_count = SAPMparams['fintrinsic_count']
	# vintrinsic_count = SAPMparams['vintrinsic_count']
	# sem_region_list = SAPMparams['sem_region_list']
	# nclusterlist = SAPMparams['nclusterlist']
	# tsize = SAPMparams['tsize']
	# tplist_full = SAPMparams['tplist_full']
	# tcdata_centered = SAPMparams['tcdata_centered']
	# tcdata_centered_original = SAPMparams['tcdata_centered_original']
	# ctarget = SAPMparams['ctarget']
	# csource = SAPMparams['csource']
	# dtarget = SAPMparams['dtarget']
	# dsource = SAPMparams['dsource']
	# fintrinsic_region = SAPMparams['fintrinsic_region']
	# Mconn = SAPMparams['Mconn']
	# Minput = SAPMparams['Minput']
	# timepoint = SAPMparams['timepoint']
	# epoch = SAPMparams['epoch']
	# latent_flag = SAPMparams['latent_flag']
	# reciprocal_flag = SAPMparams['reciprocal_flag']
	# DBname = SAPMparams['DBname']
	# DBnum = SAPMparams['DBnum']

	regular_flag = 1-latent_flag   # flag where connections are not latent

	# ntime, NP = np.shape(tplist_full)
	Nintrinsics = vintrinsic_count + fintrinsic_count

	ncomponents_to_fit = copy.deepcopy(nregions)
	#---------------------------------------------------------------------------------------------------------
	#---------------------------------------------------------------------------------------------------------
	# repeat the process for each participant-----------------------------------------------------------------
	betalimit = 3.0
	epochnum = 0
	SAPMresults = []
	first_pass_results = []
	second_pass_results = []
	beta_init_record = []

	# one person at a time with this method
	# for nperson in range(NP):
	nruns = nruns_per_person

	# get principal components of Sinput--------------------------
	nr = np.shape(Sinput)[0]
	pca = sklearn.decomposition.PCA()
	pca.fit(Sinput)
	components = pca.components_
	loadings = pca.transform(Sinput)
	mu2 = np.mean(Sinput, axis=0)
	loadings = np.concatenate((np.ones((nr, 1)), loadings), axis=1)
	components = np.concatenate((mu2[np.newaxis, :], components), axis=0)

	beta_int1 = 0.1
	lastgood_beta_int1 = copy.deepcopy(beta_int1)

	# initialize beta values-----------------------------------
	nbeta = len(csource)
	if isinstance(betascale,str):
		if betascale == 'shotgun':
			beta_initial = betaval_init_shotgun(initial_Lweight, csource, ctarget, Sinput, Minput, Mconn, components,
								loadings, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1,
												ncomponents_to_fit, nreps=10000)

			beta_initial = beta_initial[np.newaxis,:]
			nitermax_stage1 = 0
		else:
			# read saved beta_initial values
			b = np.load(betascale,allow_pickle=True).flat[0]
			beta_initial = b['beta_initial']
			beta_initial = beta_initial[np.newaxis,:]
			nitermax_stage1 = 0
		nsteps_stage1 = 1
	else:
		nsteps_stage1 = copy.deepcopy(initial_nsteps_stage1)
		beta_initial = betascale*np.random.randn(nsteps_stage1,nbeta)
		nregion,ntotal = np.shape(Minput)
		cl = np.where(latent_flag > 0)[0]
		beta_initial[:,cl] = 1.0

		if len(beta_initial_vals) == nbeta:
			beta_initial[0,:] = copy.deepcopy(beta_initial_vals)

		nitermax_stage1 = copy.deepcopy(initial_nitermax_stage1)

	# initialize deltavals
	delta_initial = np.ones(len(dtarget))
	deltascale = np.std(Sinput,axis=1)
	meanscale = np.mean(deltascale)

	# initialize
	results_record = []
	ssqd_record = []

	# stage 1 - test the initial betaval settings
	if include_stage1b:
		stage1_ssqd = np.zeros(2*nsteps_stage1)
	else:
		stage1_ssqd = np.zeros(nsteps_stage1)
	stage1_results = []
	for ns in range(nsteps_stage1):
		ssqd_record_stage1 = []
		beta_init_record.append({'beta_initial':beta_initial[ns,:]})

		# initalize Sconn
		betavals = copy.deepcopy(beta_initial[ns,:]) # initialize beta values at zero
		lastgood_betavals = copy.deepcopy(betavals)
		deltavals = copy.deepcopy(delta_initial)
		lastgood_deltavals = copy.deepcopy(deltavals)

		alphalist = initial_alpha*np.ones(nbeta)
		alphabint = copy.deepcopy(initial_alpha)
		alpha = copy.deepcopy(initial_alpha)
		Lweight = copy.deepcopy(initial_Lweight)
		dval = copy.deepcopy(initial_dval)

		# # starting point for optimizing intrinsics with given betavals----------------------------------------------------
		Mconn[ctarget,csource] = copy.deepcopy(betavals)
		Minput[dtarget, dsource] = copy.deepcopy(deltavals)
		fit, loadings_fit, W, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method_V3(Sinput, components, loadings, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, ncomponents_to_fit)
		# Soutput = Meigv @ Mintrinsic  # signalling over each connection
		ssqd, error, error2, costfactor = pysapm.sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

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

			betavals, deltavals, beta_int1, fit, dssq_db, dssq_dd, dssq_dbeta1, ssqd_original, ssqd, alpha, alphabint = \
												pysapm.update_betavals_V3(Sinput, components, loadings, Minput, Mconn, betavals,
												deltavals, betalimit, ctarget, csource, dtarget, dsource,
												dval,fintrinsic_count,
												vintrinsic_count, beta_int1,fintrinsic1, Lweight, regular_flag, alpha,alphabint,
												ncomponents_to_fit, latent_flag=latent_flag)  # kappavals, ktarget, ksource,
			betavals[latent_flag > 0] = 1.0

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

			fit, loadings_fit, W, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method_V3(Sinput, components, loadings, Minput,
												Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1,
												ncomponents_to_fit)
			# Soutput = Meigv @ Mintrinsic  # signalling over each connection
			ssqd, error, error2, costfactor = pysapm.sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

			err_total = Sinput - fit
			Smean = np.mean(Sinput)
			errmean = np.mean(err_total)

			R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
			R2avg = np.mean(R2list)
			R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)

			results_record.append({'Sinput': Sinput, 'fit': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})

			ssqchange = ssqd - ssqd_original

			if verbose:
				nperson = 0
				print('SAPM  {} stage1 pass {} iter {} alpha {:.3e}  ssqd {:.2f} error {:.2f} error2 {:.2f} change {:.3f}  percent {:.1f}  R2 avg {:.3f}  R2 total {:.3f}'.format(nperson,
								ns, iter, alpha, ssqd, error, error2, ssqchange, 100.*ssqd/ssqd_starting, R2avg, R2total))
			ssqd_old = copy.deepcopy(ssqd)
			# now repeat it ...
		stage1_ssqd[ns] = ssqd
		stage1_results.append({'betavals':betavals, 'deltavals':deltavals})


	if include_stage1b:
		# # stage 1 part b
		for ns in range(nsteps_stage1):
			ssqd_record_stage1 = []
			bi = beta_initial[ns,:]
			bi[reciprocal_flag > 0] *= -1.0   # invert all the betavals for reciprocal connections and test them
			beta_init_record.append({'beta_initial':bi})

			# initalize Sconn
			betavals = copy.deepcopy(bi) # initialize beta values at zero
			lastgood_betavals = copy.deepcopy(betavals)
			deltavals = copy.deepcopy(delta_initial)
			lastgood_deltavals = copy.deepcopy(deltavals)

			alphalist = initial_alpha*np.ones(nbeta)
			alphabint = copy.deepcopy(initial_alpha)
			alpha = copy.deepcopy(initial_alpha)
			Lweight = copy.deepcopy(initial_Lweight)
			dval = copy.deepcopy(initial_dval)

			# # starting point for optimizing intrinsics with given betavals----------------------------------------------------
			Mconn[ctarget,csource] = copy.deepcopy(betavals)
			Minput[dtarget, dsource] = copy.deepcopy(deltavals)
			fit, loadings_fit, W, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method_V3(Sinput, components, loadings, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, ncomponents_to_fit)
			# Soutput = Meigv @ Mintrinsic  # signalling over each connection
			ssqd, error, error2, costfactor = pysapm.sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

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

				betavals, deltavals, beta_int1, fit, dssq_db, dssq_dd, dssq_dbeta1, ssqd_original, ssqd, alpha, alphabint = \
													pysapm.update_betavals_V3(Sinput, components, loadings, Minput, Mconn, betavals,
													deltavals, betalimit, ctarget, csource, dtarget, dsource,
													dval,fintrinsic_count,
													vintrinsic_count, beta_int1,fintrinsic1, Lweight, regular_flag, alpha,alphabint,
													ncomponents_to_fit, latent_flag=latent_flag)  # kappavals, ktarget, ksource,

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

				fit, loadings_fit, W, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method_V3(Sinput, components, loadings, Minput,
													Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1,
													ncomponents_to_fit)
				# Soutput = Meigv @ Mintrinsic  # signalling over each connection
				ssqd, error, error2, costfactor = pysapm.sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

				err_total = Sinput - fit
				Smean = np.mean(Sinput)
				errmean = np.mean(err_total)

				R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
				R2avg = np.mean(R2list)
				R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)

				results_record.append({'Sinput': Sinput, 'fit': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})

				ssqchange = ssqd - ssqd_original

				if verbose:
					nperson = 0
					print('SAPM  {} stage1b pass {} iter {} alpha {:.3e}  ssqd {:.2f} error {:.2f} error2 {:.2f} change {:.3f}  percent {:.1f}  R2 avg {:.3f}  R2 total {:.3f}'.format(nperson,
									ns, iter, alpha, ssqd, error, error2, ssqchange, 100.*ssqd/ssqd_starting, R2avg, R2total))
				ssqd_old = copy.deepcopy(ssqd)
				# now repeat it ...
			stage1_ssqd[ns+nsteps_stage1] = ssqd
			stage1_results.append({'betavals':betavals, 'deltavals':deltavals})


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

	fit, loadings_fit, W, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method_V3(Sinput, components, loadings, Minput,
											Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1,
											ncomponents_to_fit)
	# Soutput = Meigv @ Mintrinsic  # signalling over each connection
	ssqd, error, error2, costfactor = pysapm.sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv


	ssqd_starting = copy.deepcopy(ssqd)
	ssqd_old = copy.deepcopy(ssqd)
	ssqd_record += [ssqd]

	iter = 0
	converging = True
	dssq_record = np.ones(3)
	dssq_count = 0
	sequence_count = 0

	while alpha > alpha_limit and iter < nitermax and converging:
		iter += 1
		betavals, deltavals, beta_int1, fit, dssq_db, dssq_dd, dssq_dbeta1, ssqd_original, ssqd, alpha, alphabint = \
			pysapm.update_betavals_V3(Sinput, components, loadings, Minput, Mconn, betavals, deltavals, betalimit,
							   ctarget, csource, dtarget, dsource, dval, fintrinsic_count,
							   vintrinsic_count, beta_int1, fintrinsic1, Lweight, regular_flag, alpha, alphabint,
							   ncomponents_to_fit, latent_flag=latent_flag)   #, kappavals, ktarget, ksource

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

		fit, loadings_fit, W, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method_V3(Sinput, components, loadings, Minput,
											Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1,
											ncomponents_to_fit)
		# Soutput = Meigv @ Mintrinsic  # signalling over each connection
		ssqd, error, error2, costfactor = pysapm.sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

		err_total = Sinput - fit
		Smean = np.mean(Sinput)
		errmean = np.mean(err_total)

		R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
		R2avg = np.mean(R2list)
		R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)

		results_record.append({'Sinput': Sinput, 'fit': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})

		ssqchange = ssqd - ssqd_original

		if verbose:
			nperson = 0
			print('SAPM  {} beta vals:  iter {} alpha {:.3e}  ssqd {:.2f} error {:.2f} error2 {:.2f} change {:.3f}  percent {:.1f}  R2 avg {:.3f}  R2 total {:.3f}'.format(
					nperson,iter, alpha, ssqd, error, error2, ssqchange, 100. * ssqd / ssqd_starting, R2avg, R2total))
		ssqd_old = copy.deepcopy(ssqd)
		# now repeat it ...

	# fit the results now to determine output signaling from each region
	Mconn[ctarget, csource] = copy.deepcopy(betavals)
	Minput[dtarget, dsource] = copy.deepcopy(deltavals)
	fit, loadings_fit, W, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method_V3(Sinput, components, loadings, Minput,
										Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1,
										ncomponents_to_fit)

	Sconn = Meigv @ Mintrinsic    # signalling over each connection

	entry = {'Sinput':Sinput, 'Sconn':Sconn, 'beta_int1':beta_int1, 'Mconn':Mconn, 'Minput':Minput,
			 'fit':fit, 'loadings_fit':loadings_fit, 'W':W, 'loadings':loadings, 'components':components,
			 'R2total':R2total, 'R2avg':R2avg, 'Mintrinsic':Mintrinsic, 'fintrinsic_count':fintrinsic_count, 'vintrinsic_count':vintrinsic_count,
			 'Meigv':Meigv, 'betavals':betavals, 'deltavals':deltavals, 'fintrinsic1':fintrinsic1}

	SAPMresults = copy.deepcopy(entry)

	return SAPMresults


def Gram_Schmidt_orthogonalization(V):
	# take a set of vectors and make an orthogonal set out of them
	nv,tsize = np.shape(V)
	U = np.zeros((nv,tsize))  # new set
	for nn in range(nv):
		U[nn,:] = copy.deepcopy(V[nn,:])
		if nn > 0:
			projections = np.zeros(tsize)
			for mm in range(nn):
				proj = U[mm,:] * np.dot(U[mm,:],V[nn,:])/np.dot(U[mm,:],U[mm,:])
				projections += proj
			U[nn,:] -= projections
	return U


# setup to train a network
def run_SAPM_on_model():
	resultsdir = r'E:/network_training'
	nsamples = 5
	nregions = 10
	tsize = 200
	fintrinsic_count = 1
	vintrinsic_count = 2
	nclusters = 5
	networkmodel_name = os.path.join(resultsdir, 'random_network.xlsx')
	TR = 3.0

	simresults_name = os.path.join(resultsdir, 'sim_results.npy')
	simresults_name_base = os.path.join(resultsdir, 'sim_results_base.npy')

	sim_results = []
	sim_results_base = []
	for nn in range(nsamples):
		tsize = 40
		# networkmodel_name = generate_random_networkfile(networkmodel_name, nregions, fintrinsic_count, vintrinsic_count, tsize, nclusters)
		networkmodel_name = r'E:\SAPMresults_Dec2022\network_model_June2023_SAPM_test2.xlsx'

		tsize = 200
		nruns_per_person = 5
		dataset = generate_simulated_data_and_results(networkmodel_name, tsize, nruns_per_person, TR)

		# test the same data repeatedly
		# if nn == 0:
		# 	saved_dataset = copy.deepcopy(dataset)
		# else:
		# 	dataset = copy.deepcopy(saved_dataset)

		Mconn = copy.deepcopy(dataset['Mconn'])
		ctarget = copy.deepcopy(dataset['ctarget'])
		csource = copy.deepcopy(dataset['csource'])

		# run SAPM with the simulated network and data ...
		# create a network model
		timepoint = 'all'
		epoch = 'all'
		betascale = 1e-2

		SAPMresultsname = os.path.join(resultsdir, 'sim_results_sim{}.npy'.format(nn))
		SAPMparametersname = os.path.join(resultsdir, 'sim_params_sim{}.npy'.format(nn))

		Sinput = copy.deepcopy(dataset['Sinput'])
		Minput = copy.deepcopy(dataset['Minput'])
		Sconn = copy.deepcopy(dataset['Sconn'])
		Mconn = copy.deepcopy(dataset['Mconn'])
		Mintrinsic = copy.deepcopy(dataset['Mintrinsic'])
		ctarget = copy.deepcopy(dataset['ctarget'])
		csource = copy.deepcopy(dataset['csource'])
		dtarget = copy.deepcopy(dataset['dtarget'])
		dsource = copy.deepcopy(dataset['dsource'])
		beta_list = copy.deepcopy(dataset['beta_list'])
		latent_flag = copy.deepcopy(dataset['latent_flag'])
		reciprocal_flag = copy.deepcopy(dataset['reciprocal_flag'])

		Sinput_base = copy.deepcopy(dataset['Sinput_base'])
		Sconn_base = copy.deepcopy(dataset['Sconn_base'])
		Mintrinsic_base = copy.deepcopy(dataset['Mintrinsic_base'])

		nregions, tsize_full = np.shape(Sinput)
		noise_level = 0.05
		SinputN = Sinput + noise_level*np.random.randn(nregions,tsize_full)
		Sinput_baseN = Sinput_base + noise_level*np.random.randn(nregions,tsize_full)

		Mconn_original = copy.deepcopy(Mconn)
		Minput_input = copy.deepcopy(Minput)
		Mconn_input = copy.deepcopy(Mconn)
		Mconn_input[ctarget,csource] = 0.0
		print('\n\nsim {} of {}'.format(nn,nsamples))
		SAPMresults = sem_physio_modelV3_copy(SinputN, tsize, Minput, Mintrinsic, Mconn, nruns_per_person, vintrinsic_count,
							fintrinsic_count, networkmodel_name, ctarget, csource, dtarget, dsource, timepoint, epoch, latent_flag, reciprocal_flag,
							betascale=1e-2, Lweight=1.0, normalize_var=False, nitermax=500, verbose=True,
							initial_nitermax_stage1= 50,
							initial_nsteps_stage1= 15, beta_initial_vals = [])
		# beta_initial_vals = 0.3 * Mconn[ctarget, csource]


		# SAPMresults_base = sem_physio_modelV3_copy(Sinput_baseN, tsize, Minput, Mintrinsic_base, Mconn, nruns_per_person, vintrinsic_count,
		# 					fintrinsic_count, networkmodel_name, ctarget, csource, dtarget, dsource, timepoint, epoch, latent_flag, reciprocal_flag,
		# 					betascale=0.1, Lweight=1.0, normalize_var=False, nitermax=300, verbose=True,
		# 					initial_nitermax_stage1=60,
		# 					initial_nsteps_stage1=60, beta_initial_vals = [])

		SAPMresults['Mconn_model'] = copy.deepcopy(Mconn_original)
		SAPMresults['Minput_model'] = copy.deepcopy(Minput_input)
		SAPMresults['Mintrinsic_model'] = copy.deepcopy(Mintrinsic)
		SAPMresults['Sconn_model'] = copy.deepcopy(Sconn)
		SAPMresults['ctarget'] = copy.deepcopy(ctarget)
		SAPMresults['csource'] = copy.deepcopy(csource)
		SAPMresults['dtarget'] = copy.deepcopy(dtarget)
		SAPMresults['dsource'] = copy.deepcopy(dsource)
		SAPMresults['Sinput_model'] = copy.deepcopy(SinputN)

		# SAPMresults_base['Mconn_model'] = copy.deepcopy(Mconn_original)
		# SAPMresults_base['Minput_model'] = copy.deepcopy(Minput)
		# SAPMresults_base['Mintrinsic_model'] = copy.deepcopy(Mintrinsic_base)
		# SAPMresults_base['Sconn_model'] = copy.deepcopy(Sconn_base)
		# SAPMresults_base['Sinput_model'] = copy.deepcopy(Sinput_baseN)
		# SAPMresults_base['ctarget'] = copy.deepcopy(ctarget)
		# SAPMresults_base['csource'] = copy.deepcopy(csource)
		# SAPMresults_base['dtarget'] = copy.deepcopy(dtarget)
		# SAPMresults_base['dsource'] = copy.deepcopy(dsource)

		sim_results.append(SAPMresults)
		# sim_results_base.append(SAPMresults_base)

	np.save(simresults_name, sim_results)
	# np.save(simresults_name_base, sim_results_base)

	DBvals_record = []
	DBvals_input_record = []
	corrO_record = []
	corrI_record = []
	for nn in range(nsamples):
		results = sim_results[nn]
		Mconn = copy.deepcopy(results['Mconn'])
		Mconn_input = copy.deepcopy(results['Mconn_model'])
		Minput = copy.deepcopy(results['Minput'])
		Minput_input = copy.deepcopy(results['Minput_model'])
		ctarget = copy.deepcopy(results['ctarget'])
		csource = copy.deepcopy(results['csource'])
		dtarget = copy.deepcopy(results['dtarget'])
		dsource = copy.deepcopy(results['dsource'])
		DBvals = Mconn[ctarget,csource]
		DBvals_input = Mconn_input[ctarget,csource]
		Dvals = Minput[dtarget,dsource]
		Dvals_input = Minput_input[dtarget,dsource]

		if nn == 0:
			DBvals_record = copy.deepcopy(DBvals.flatten())[:,np.newaxis]
			DBvals_input_record = copy.deepcopy(DBvals_input.flatten())[:,np.newaxis]
		else:
			DBvals_record = np.concatenate((DBvals_record, DBvals.flatten()[:,np.newaxis]),axis=0)
			DBvals_input_record = np.concatenate((DBvals_input_record, DBvals_input.flatten()[:,np.newaxis]),axis=0)

		Sinput = copy.deepcopy(results['fit'])
		Sinput_model = copy.deepcopy(results['Sinput_model'])
		Sconn = copy.deepcopy(results['Sconn'])
		Sconn_model = copy.deepcopy(results['Sconn_model'])
		Mintrinsic = copy.deepcopy(results['Mintrinsic'])
		Mintrinsic_model = copy.deepcopy(results['Mintrinsic_model'])

		nr,tsize = np.shape(Sinput)
		corrO = np.zeros(nr)
		corrI = np.zeros(nr)
		for rr in range(nr):
			cc = np.corrcoef(Sconn[rr,:],Sconn_model[rr,:])
			corrO[rr] = cc[0,1]
			cc = np.corrcoef(Sinput[rr,:],Sinput_model[rr,:])
			corrI[rr] = cc[0,1]

		corrO_record += [corrO.flatten()]
		corrI_record += [corrI.flatten()]

	DBvals_input_record = np.array(DBvals_input_record)
	DBvals_record = np.array(DBvals_record)
	corrO_record = np.array(corrO_record)
	corrI_record = np.array(corrI_record)

	wnum = 40
	plt.close(wnum)
	fig = plt.figure(wnum)
	ndb = len(ctarget)
	cols = np.zeros((nsamples,3))
	red = 1.0 - 2.0*np.array(range(nsamples))/(nsamples-1)
	red[red < 0] = 0.0
	green = 1.0-np.abs(2.0*np.array(range(nsamples))/(nsamples-1)-1.0)
	green[green < 0] = 0.0
	blue = 2.0*np.array(range(nsamples))/(nsamples-1)-1.0
	blue[blue < 0] = 0.0
	cols = np.concatenate((red[:,np.newaxis], green[:,np.newaxis], blue[:,np.newaxis]),axis=1)
	for nn in range(nsamples):
		n1 = nn*ndb
		n2 = n1+ndb
		plt.plot(DBvals_input_record[n1:n2].flatten(), DBvals_record[n1:n2].flatten(), linestyle = 'none', marker = 'o', color = cols[nn,:])

	plt.close(wnum+1)
	fig = plt.figure(wnum+1)
	plt.hist(corrO_record.flatten(),40)

	plt.close(wnum+2)
	fig = plt.figure(wnum+2)
	plt.hist(corrI_record.flatten(),40)

	# find which connections have bad results
	Ddiff = DBvals_record - DBvals_input_record
	x = np.argsort(np.abs(Ddiff.flatten()))
	cnum = x[-20:] % len(csource)
	pnum = np.floor(x[-20:]/len(csource)).astype(int)

	csort = np.sort(cnum)
	for nn in range(len(csort)):
		print('{}'.format(beta_list[csort[nn]]['name']))

	CC = np.corrcoef(DBvals_record.flatten(), DBvals_input_record.flatten())
	nv = len(DBvals_record.flatten())
	G = np.concatenate((DBvals_input_record.flatten()[:,np.newaxis], np.ones((nv,1))),axis=1)
	b = np.linalg.inv(G.T @ G) @ G.T @ DBvals_record.flatten()
	print('DB fit vs DB model:   correlation = {:.4f}    linear fit slope = {:.3f}  intercept = {:.3f}'.format(CC[0,1],b[0],b[1]))
