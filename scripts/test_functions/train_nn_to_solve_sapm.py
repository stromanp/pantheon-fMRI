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


def generate_simulated_data_and_results(networkmodel_name, tsize):
	network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = pysapm.load_network_model_w_intrinsics(networkmodel_name)

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
	for nn in range(len(ctarget)):
		spair = beta_list[csource[nn]]['pair']
		tpair = beta_list[ctarget[nn]]['pair']
		if spair[0] == tpair[1]:
			reciprocal_flag[nn] = 1

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
	betavals = np.tanh(np.random.normal(0,1,len(ctarget)))
	# c = np.where(latent_flag > 0)[0]
	# betavals[c] = 1.0 + 0.05*np.random.normal(0,1,len(c))
	deltavals = 1.0 + 0.2*np.random.normal(0,1,len(dtarget))
	c = np.where(dsource > 0)[0]
	deltavals[c] = 1.0  # latent inputs
	Mconn[ctarget,csource] = betavals
	Minput[dtarget,dsource] = deltavals
	Mintrinsic_var = np.random.normal(0,1,(vintrinsic_count, tsize))
	Mintrinsic_var -= np.repeat(np.mean(Mintrinsic_var, axis = 1)[:, np.newaxis], tsize, axis = 1)
	if fintrinsic_count > 0:
		Mintrinsic = np.concatenate((fintrinsic_base, Mintrinsic_var),axis=0)
	else:
		Mintrinsic = copy.deepcopy(Mintrinsic_var)

	e, v = np.linalg.eig(Mconn)  # Mconn is nbeta x nbeta   where nbeta = ncon + Nintrinsic
	Meigv = np.real(v[:, -Nintrinsic:])

	Sconn = Meigv @ Mintrinsic
	Sinput = Minput @ Sconn
	Sconn = Mconn @ Sconn

	dataset = {'Sinput':Sinput, 'Sconn':Sconn, 'Minput':Minput, 'Mconn':Mconn, 'Mintrinsic':Mintrinsic,
			   'ctarget':ctarget, 'csource':csource, 'dtarget':dtarget, 'dsource':dsource}

	return dataset


def train_neural_network(trainingdata_name):

	#--------------------------------------------------
	resultsdir = r'E:/network_training'
	nsamples = 10000
	nregions = 10
	tsize = 200
	fintrinsic_count = 0
	vintrinsic_count = 2
	nclusters = 5
	networkmodel_name = os.path.join(resultsdir, 'random_network.xlsx')

	trainingdata_name = os.path.join(resultsdir, 'training_data.npy')
	#------------------------------------------------------------

	train_portion = 0.7
	test_portion = 0.3

	tsize_max = 200

	train_record = np.load(trainingdata_name, allow_pickle=True)
	nsim = len(train_record)

	ntrain = np.floor(nsim*train_portion).astype(int)
	ntest = nsim-ntrain

	ctrain = np.random.choice(list(range(nsim)), ntrain,replace=False)
	ctest = [x for x in range(nsim) if x not in ctrain]

	nregions, ntotal = np.shape(train_record[0]['Minput'])

	sizeD = nregions*ntotal
	sizeB = ntotal*ntotal
	sizeS = nregions*tsize_max
	sizeC = nregions*nregions
	sizeC2 = sizeC*sizeC

	# X inputs to the neural network are:
	# 1) Sinput  [nregions x tsize]
	# 2) variance/covariance grid for Sinput
	# 3) flags showing which elements of Mconn have non-zero values

	# Y inputs - showing correct results:
	# 1) betavals
	# 2) deltavals

	Xsize = sizeD+sizeC
	# Ysize = sizeB+sizeD
	Ysize = sizeB
	X = np.zeros((nsim,Xsize))
	Y = np.zeros((nsim,Ysize))

	for nn in range(nsim):
		Mconn = copy.deepcopy(train_record[nn]['Mconn'])
		Minput = copy.deepcopy(train_record[nn]['Minput'])
		Sinput = copy.deepcopy(train_record[nn]['Sinput'])
		ctarget = copy.deepcopy(train_record[nn]['ctarget'])
		csource = copy.deepcopy(train_record[nn]['csource'])
		dtarget = copy.deepcopy(train_record[nn]['dtarget'])
		dsource = copy.deepcopy(train_record[nn]['dsource'])

		Mflag = np.zeros(np.shape(Minput))
		Mflag[ctarget,csource] = 1.0

		Cgrid = Sinput @ Sinput.T/tsize
		cgrid_flat = np.reshape(Cgrid, (nregions*nregions,1))
		Cgrid2 = cgrid_flat @ cgrid_flat.T

		Sinput2 = np.zeros((nregions, tsize_max))
		Sinput2[:,:tsize] = copy.deepcopy(Sinput)

		X[nn,:sizeD] = np.reshape(Mflag,(1,sizeD))
		X[nn,sizeD:] = np.reshape(Cgrid,(1,sizeC))
		# X[nn,(sizeD+sizeC):] = np.reshape(Cgrid2,(1,sizeC2))

		# Y[nn,:sizeD] = np.reshape(Minput,(1,sizeD))
		Y[nn,:] = np.reshape(Mconn,(1,sizeB))

	Xtrain = X[ctrain,:]
	Ytrain = Y[ctrain,:]

	Xtest = X[ctest,:]
	Ytest = Y[ctest,:]

	# mlp = NN.MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=500)
	# mlp.fit(X_train, y_train)

	model = models.Sequential()
	model.add(layers.Dense(2*Xsize, input_shape = (Xsize,) , activation = 'tanh'))   # activation = 'tanh'
	model.add(layers.Dense(2*Xsize , activation = 'tanh'))
	model.add(layers.Dense(Ysize))

	# opt = tf.keras.optimizers.Ftrl(learning_rate=1e-2, l1_regularization_strength=1e-3)
	# opt = tf.keras.optimizers.SGD(learning_rate=0.05)
	model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy','mse', 'mae'])

	model.fit(Xtrain, Ytrain, epochs=5, batch_size=32)

	_, accuracy, mse, mae = model.evaluate(Xtrain, Ytrain)
	print('Accuracy: %.2f' % (accuracy * 100))

	_, accuracy, mse, mae = model.evaluate(Xtest, Ytest)
	print('Accuracy: %.2f' % (accuracy * 100))

	savename = os.path.join(resultsdir, 'model1.keras')
	model.save(savename)
	# del model
	# # Recreate the exact same model purely from the file:

	predictions_train = model.predict(Xtrain)
	predictions_test = model.predict(Xtest)

	sample_number = 102  # random pick
	Yactual = np.reshape(Ytest[sample_number,:],(ntotal,ntotal))
	Ypredict = np.reshape(predictions_test[sample_number,:],(ntotal,ntotal))
	wnum = 35
	plt.close(wnum)
	fig = plt.figure(wnum)
	plt.plot(Yactual,Ypredict,'ob')


	# #------------------------------------------------------------
	# from tensorflow import keras
	# from tensorflow.keras import layers
	# from keras.utils import plot_model
	# from keras.models import Model
	# from keras.layers import Input
	# from keras.layers import Dense
	# # from keras.layers.recurrent import LSTM
	#
	#
	# Xsize = sizeD+sizeS+sizeC
	# Ysize = sizeB+sizeD
	# X1 = np.zeros((nsim,nregions,tsize))
	# X2 = np.zeros((nsim,nregions,ntotal))
	# X3 = np.zeros((nsim,nregions,nregions))
	# Y1 = np.zeros((nsim,nregions,ntotal))
	# Y2 = np.zeros((nsim,ntotal,ntotal))
	#
	# for nn in range(nsim):
	# 	Mconn = copy.deepcopy(train_record[nn]['Mconn'])
	# 	Minput = copy.deepcopy(train_record[nn]['Minput'])
	# 	Sinput = copy.deepcopy(train_record[nn]['Sinput'])
	# 	ctarget = copy.deepcopy(train_record[nn]['ctarget'])
	# 	csource = copy.deepcopy(train_record[nn]['csource'])
	# 	dtarget = copy.deepcopy(train_record[nn]['dtarget'])
	# 	dsource = copy.deepcopy(train_record[nn]['dsource'])
	#
	# 	Mflag = np.zeros(np.shape(Minput))
	# 	Mflag[ctarget,csource] = 1.0
	#
	# 	Cgrid = Sinput @ Sinput.T/tsize
	#
	# 	Sinput2 = np.zeros((nregions, tsize_max))
	# 	Sinput2[:,:tsize] = copy.deepcopy(Sinput)
	#
	# 	X1[nn,:,:] = copy.deepcopy(Sinput2)
	# 	X2[nn,:,:] = copy.deepcopy(Mflag)
	# 	X3[nn,:,:] = copy.deepcopy(Cgrid)
	#
	# 	Y1[nn,:,:] = copy.deepcopy(Minput)
	# 	Y2[nn,:,:] = copy.deepcopy(Mconn)
	#
	# Xtrain1 = X1[ctrain,:]
	# Xtrain2 = X2[ctrain,:]
	# Xtrain3 = X3[ctrain,:]
	# Ytrain1 = Y1[ctrain,:]
	# Ytrain2 = Y2[ctrain,:]
	#
	# Xtest1 = X1[ctest,:]
	# Xtest2 = X2[ctest,:]
	# Xtest3 = X3[ctest,:]
	# Ytest1 = Y1[ctest,:]
	# Ytest2 = Y2[ctest,:]
	#
	# visible1 = layers.Input(shape=(nregions, tsize_max))
	# visible2 = layers.Input(shape=(nregions, ntotal))
	# visible3 = layers.Input(shape=(nregions, nregions))
	# # hidden1 = LSTM(Xsize)(visible1)
	# hidden1 = layers.Dense(2*Xsize, activation='linear')(visible1, visible2, visible3)
	# hidden2 = layers.Dense(2*Xsize, activation='linear')(hidden1)
	# hidden3 = layers.Dense(2*Xsize, activation='linear')(hidden2)
	# hidden4 = layers.Dense(2*Xsize, activation='linear')(hidden3)
	# output = layers.Dense(shape = (ntotal,ntotal), activation='linear')(hidden4)
	# model = model(inputs=(visible1,visible2,visible3), outputs=output)
	#
	# model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
	#
	# model.fit(x=[Xtrain1, Xtrain2, Xtrain3], y=Ytrain2,
	# 	validation_data=([Xtest1, Xtest2, Xtest3], Ytest2),
	# 	epochs=100, batch_size=8)


	# test_scores = model.evaluate(x_test, y_test, verbose=2)
	# print("Test loss:", test_scores[0])
	# print("Test accuracy:", test_scores[1])
	#
	# #---------------another example-----------------------------
	# encoder_input = keras.Input(shape=(28, 28, 1), name="img")
	# x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
	# x = layers.Conv2D(32, 3, activation="relu")(x)
	# x = layers.MaxPooling2D(3)(x)
	# x = layers.Conv2D(32, 3, activation="relu")(x)
	# x = layers.Conv2D(16, 3, activation="relu")(x)
	# encoder_output = layers.GlobalMaxPooling2D()(x)
	#
	# encoder = keras.Model(encoder_input, encoder_output, name="encoder")
	# encoder.summary()
	#
	# x = layers.Reshape((4, 4, 1))(encoder_output)
	# x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
	# x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
	# x = layers.UpSampling2D(3)(x)
	# x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
	# decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)
	#
	# autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
	# autoencoder.summary()
	# #---------------------------------------------------------------------------
	# # model.save("path_to_my_model.keras")
	# # del model
	# # # Recreate the exact same model purely from the file:
	# # model = keras.models.load_model("path_to_my_model.keras")
	# #----------------------------------------------------------
	#
	#
	#
	#
	# _, accuracy = model.evaluate(Xtrain, Ytrain)
	# print('Accuracy: %.2f' % (accuracy * 100))
	#
	# _, accuracy = model.evaluate(Xtest, Ytest)
	# print('Accuracy: %.2f' % (accuracy * 100))
	#
	# predictions_train = model.predict(Xtrain)
	# predictions_test = model.predict(Xtest)
	#
	# # wnum = 35
	# # cc = 1
	# # plt.close(wnum)
	# # fig = plt.figure(wnum)
	# # plt.plot(T[ctrain,cc],predictions_train[:,cc],'ob')
	# # plt.plot(T[ctest,cc],predictions_test[:,cc],'or')



# setup to train a network
def setup_to_train_network():
	resultsdir = r'E:/network_training'
	nsamples = 10000
	nregions = 10
	tsize = 200
	fintrinsic_count = 0
	vintrinsic_count = 2
	nclusters = 5
	networkmodel_name = os.path.join(resultsdir, 'random_network.xlsx')

	trainingdata_name = os.path.join(resultsdir, 'training_data.npy')
	generate_training_data = True
	if generate_training_data:
		train_record = []
		for nn in range(nsamples):
			networkmodel_name = generate_random_networkfile(networkmodel_name, nregions, fintrinsic_count, vintrinsic_count, tsize, nclusters)
			dataset = generate_simulated_data_and_results(networkmodel_name, tsize)
			train_record.append(dataset)

		np.save(trainingdata_name, train_record)

	# now train a network
	train_neural_network(trainingdata_name)







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

