# simulate random networks and see which ones result in biased (non-zero) B values
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


def run_simulated_networks(nsimulations, starting_number):
	resultsdir = r'E:/sim_data3'

	datadir = r'E:/SAPMresults_Dec2022'
	regiondataname = r'E:/SAPMresults_Dec2022\allpainconditions_regiondata2.npy'
	covariatesname = r'E:/SAPMresults_Dec2022\allpain_covariates.npy'
	clusterdataname = r'E:/sim_data\sim_cluster_def.npy'
	DBname = r'E:\graded_pain_database_May2022.xlsx'

	nregions = 10
	clusterdataname = generate_sim_clusterdata(clusterdataname, nregions, nclusters=5)

	for simnum in range(starting_number, nsimulations + starting_number):
		# run many simulations
		null_regiondataname, null_covariates = pysapm.generate_null_data_set(regiondataname, covariatesname, npeople=50, variable_variance = False)
		p,f = os.path.split(null_regiondataname)
		f2,e = os.path.splitext(f)
		new_regiondataname = os.path.join(resultsdir,f2 + 'sim{}'.format(simnum) + e)
		shutil.copyfile(null_regiondataname,new_regiondataname)

		new_covariatesname = os.path.join(resultsdir,'null_cov_sim{}'.format(simnum) + e)
		shutil.copyfile(null_covariates,new_covariatesname)

		# create a network model
		tsize = 40
		fintrinsic_count = 1
		vintrinsic_count = 3
		networkmodel_name = os.path.join(resultsdir, 'network_model_sim{}.xlsx'.format(simnum))
		networkmodel_name = generate_random_networkfile(networkmodel_name, nregions, fintrinsic_count, vintrinsic_count, tsize, nclusters = 5)

		timepoint = 'all'
		epoch = 'all'
		betascale=0.1

		SAPMresultsname = os.path.join(resultsdir,'sim_results_sim{}'.format(simnum) + e)
		SAPMparametersname = os.path.join(resultsdir,'sim_params_sim{}'.format(simnum) + e)

		cnums = [3, 2, 4, 2, 4, 2, 3, 0, 1, 2]
		print('using data in {}'.format(new_regiondataname))
		pysapm.SAPMrun_V2(cnums, new_regiondataname, clusterdataname, SAPMresultsname, SAPMparametersname, networkmodel_name, DBname, timepoint,
					epoch, betascale = 0.1, reload_existing = False, multiple_output = False)



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


def train_neural_network(simdataname):
	train_portion = 0.8
	test_portion = 0.2

	sim_record = np.load(simdataname, allow_pickle=True)
	nsim = len(sim_record)

	ntrain = np.floor(nsim*train_portion).astype(int)
	ntest = nsim-ntrain

	ctrain = np.random.choice(list(range(nsim)), ntrain,replace=False)
	ctest = [x for x in range(nsim) if x not in ctrain]

	nregion, ntotal = np.shape(sim_record[0]['Minput'])
	ncon_record = np.zeros(nsim)
	for nn in range(nsim):
		ncon_record[nn] = len(sim_record[nn]['beta_mean'])

	nconmax = np.max(ncon_record).astype(int)
	sizeM = nregion*nregion
	sizeM_full = nregion*ntotal

	NP = 50

	X = np.zeros((nsim,sizeM))
	Y = np.zeros((nsim,sizeM))
	T = np.zeros((nsim,sizeM))
	X_full = np.zeros((nsim,sizeM_full))
	Y_full = np.zeros((nsim,sizeM_full))
	T_full = np.zeros((nsim,sizeM_full))
	flagcount = np.zeros(nsim)
	flagcount2 = np.zeros(nsim)
	flagcount3 = np.zeros(nsim)
	for nn in range(nsim):
		Mconn = copy.deepcopy(sim_record[nn]['Mconn'])
		Tconn = copy.deepcopy(sim_record[nn]['Mconn'])
		csource = copy.deepcopy(sim_record[nn]['csource'])
		ctarget = copy.deepcopy(sim_record[nn]['ctarget'])
		Mconn[ctarget,csource] = copy.deepcopy(sim_record[nn]['beta_mean'])
		Tconn[ctarget,csource] = sim_record[nn]['beta_mean']/(sim_record[nn]['beta_std']/np.sqrt(NP) + 1.0e-4)

		flagcount[nn], flagcount2[nn], flagcount3[nn] = find_red_flags(sim_record[nn]['Minput'])

		X[nn,:] = np.reshape(sim_record[nn]['Minput'][:nregion,:nregion],(1,sizeM))
		Y[nn,:] = np.reshape(Mconn[:nregion,:nregion],(1,sizeM))
		T[nn,:] = np.reshape(Tconn[:nregion,:nregion],(1,sizeM))
		X_full[nn,:] = np.reshape(sim_record[nn]['Minput'][:nregion,:ntotal],(1,sizeM_full))
		T_full[nn,:] = np.reshape(Tconn[:nregion,:ntotal],(1,sizeM_full))

	Xtrain = X[ctrain,:]
	Ytrain = Y[ctrain,:]
	Ttrain = T[ctrain,:]

	Xtest = X[ctest,:]
	Ytest = Y[ctest,:]
	Ttest = T[ctest,:]

	# mlp = NN.MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=500)
	# mlp.fit(X_train, y_train)

	model = models.Sequential()
	model.add(layers.Dense(2*sizeM, input_shape = (sizeM,) , activation = 'tanh'))  # , activation = 'tanh')
	model.add(layers.Dense(4*sizeM , activation = 'tanh'))
	model.add(layers.Dense(4*sizeM , activation = 'tanh'))
	model.add(layers.Dense(2*sizeM , activation = 'tanh'))
	model.add(layers.Dense(sizeM , activation = 'tanh'))


	# opt = tf.keras.optimizers.Ftrl(learning_rate=1e-2, l1_regularization_strength=1e-3)
	opt = tf.keras.optimizers.SGD(learning_rate=1e-1)
	model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

	model.fit(Xtrain, Ytrain, epochs=2000, batch_size=10)

	_, accuracy = model.evaluate(Xtrain, Ttrain)
	print('Accuracy: %.2f' % (accuracy * 100))

	_, accuracy = model.evaluate(Xtest, Ttest)
	print('Accuracy: %.2f' % (accuracy * 100))

	predictions_train = model.predict(Xtrain)
	predictions_test = model.predict(Xtest)

	wnum = 35
	cc = 1
	plt.close(wnum)
	fig = plt.figure(wnum)
	plt.plot(T[ctrain,cc],predictions_train[:,cc],'ob')
	plt.plot(T[ctest,cc],predictions_test[:,cc],'or')

	wnum = 36
	cc = 37
	plt.close(wnum)
	fig = plt.figure(wnum)
	plt.plot(T[ctrain,cc],predictions_train[:,cc],'ob')
	plt.plot(T[ctest,cc],predictions_test[:,cc],'or')

	wnum = 37
	cc = 75
	plt.close(wnum)
	fig = plt.figure(wnum)
	plt.plot(T[ctrain,cc],predictions_train[:,cc],'ob')
	plt.plot(T[ctest,cc],predictions_test[:,cc],'or')



	# try to categorize networks
	aa = np.argsort(T_full.flatten())
	flagrecord = np.zeros((len(aa),5))
	for cc in range(len(aa)):
		ii = copy.deepcopy(aa[cc])

		ra = np.floor(ii/sizeM_full).astype(int)
		ca = ii % sizeM_full
		tnum = np.floor(ca/ntotal).astype(int)
		snum = ca % ntotal

		Minput = np.reshape(X_full[ra,:],(nregion,ntotal))
		Nintrinsic = ntotal-nregion
		vnums = [x for x in range(nregion) if (Minput[x,-(Nintrinsic-1):] == 1).any() ]

		# Ttemp = np.reshape(T_full[ra,:],(nregion,ntotal))
		# T_full[ra,ca]
		# Ttemp[tnum,snum]

		flagcount, flagcount2, flagcount3, flagcount4, flagcount5 = find_red_flags(Minput)
		flagrecord[cc,:] = [flagcount, flagcount2, flagcount3, flagcount4, flagcount5]

	c = np.where(T_full.flatten() != 0)[0]
	flagrecord2 = flagrecord[c,:]



	# try to categorize networks
	Tmax = np.max(np.abs(T_full),axis=1)
	aa = np.argsort(Tmax)
	flagrecord = np.zeros((len(aa),8))
	for cc in range(len(aa)):
		ra = copy.deepcopy(aa[cc])

		Minput = np.reshape(X_full[ra,:],(nregion,ntotal))
		Nintrinsic = ntotal-nregion
		vnums = [x for x in range(nregion) if (Minput[x,-(Nintrinsic-1):] == 1).any() ]

		flagcount, flagcount2, flagcount3, flagcount4, flagcount5, flagcount6, flagcount7, flagcount8 = find_red_flags(Minput)
		flagrecord[cc,:] = [flagcount, flagcount2, flagcount3, flagcount4, flagcount5, flagcount6, flagcount7, flagcount8]




def run_null_test_on_network(nsims, networkmodel, cnums, regiondataname, clusterdataname, timepoint = 'all', epoch = 'all', betascale = 0.1):
	resultsdir, networkfilename = os.path.split(networkmodel)
	networkbasename, ext = os.path.splitext(networkfilename)

	covariatesname = []
	null_regiondataname, null_covariates = pysapm.generate_null_data_set(regiondataname, covariatesname, npeople=nsims, variable_variance = False)

	SAPMresultsname = os.path.join(resultsdir,'null_results.npy')
	SAPMparametersname = os.path.join(resultsdir,'null_params.npy')

	pysapm.SAPMrun_V2(cnums, null_regiondataname, clusterdataname, SAPMresultsname, SAPMparametersname, networkmodel, timepoint,
				epoch, betascale = betascale, reload_existing = False, multiple_output = False)

	# compile stats distributions for each connection
	results = np.load(SAPMresultsname, allow_pickle=True)
	params = np.load(SAPMparametersname, allow_pickle=True).flat[0]
	csource = params['csource']
	ctarget = params['ctarget']
	rnamelist = params['rnamelist']
	fintrinsic_count = params['fintrinsic_count']
	vintrinsic_count = params['vintrinsic_count']
	rnamelist_full = copy.deepcopy(rnamelist)
	if fintrinsic_count > 0: rnamelist_full += ['latent0']
	for nn in range(vintrinsic_count): rnamelist_full += ['latent{}'.format(fintrinsic_count+nn)]
	ncon = len(results[0]['betavals'])
	betavals = np.zeros((ncon,nsims))
	for nn in range(nsims): betavals[:,nn] = results[nn]['betavals']
	bstats = []
	for nn in range(ncon):
		conname = '{}-{}'.format(rnamelist_full[csource[nn]], rnamelist_full[ctarget[nn]])
		b = copy.deepcopy(betavals[nn,:])
		entry = {'name':conname, 'mean':np.mean(b), 'std':np.std(b), 'skewness':scipy.stats.skew(b), 'kurtosis':scipy.stats.kurtosis(b)}
		bstats.append(entry)

	df = pd.DataFrame(bstats)
	xlname = os.path.join(resultsdir,networkbasename + '_bstats.xlsx')
	with pd.ExcelWriter(xlname) as writer:
		df.to_excel(writer, sheet_name='B stats')

	return xlname

