import numpy as np
import matplotlib.pyplot as plt
import pysapm
import pysem
import os
import copy
import pydatabase
import time

resultsnamelist = [r'E:/SAPMresults_Dec2022\Null1000_2Lt_3242423012_results_corr.npy',
					r'E:/SAPMresults_Dec2022\Null_2Lt_3242423012_results_corr.npy' ,
					r'E:/SAPMresults_Dec2022\AllPain_2Lt_3242423012_results_corr.npy']

SAPMparametersname = r'E:/SAPMresults_Dec2022\AllPain_2Lt_3242423012_params.npy'
regiondataname = r'E:/SAPMresults_Dec2022\allpainconditions_regiondata2.npy'

clusterdataname = r'E:/SAPMresults_Dec2022\Pain_equalsize_cluster_def.npy'
networkfile = r'E:\SAPMresults_Dec2022\network_model_April2023_SAPM_V2.xlsx'
DBname = r'E:\graded_pain_database_May2022.xlsx'


#-----------------load results----------------------------

params = np.load(SAPMparametersname, allow_pickle=True).flat[0]

record = []

for nn in range(len(resultsnamelist)):
	results = np.load(resultsnamelist[nn], allow_pickle=True)
	dbetavals = np.array([[results[x]['betavals']] for x in range(len(results))])
	dbetavals = dbetavals[:,0,:]
	NP,nregions = np.shape(dbetavals)
	dbmean = np.mean(dbetavals,axis=0)
	dbsem = np.std(dbetavals,axis=0)/np.sqrt(NP)
	dbT = dbmean/(dbsem + 1.0e-20)

	deltavals = np.array([[results[x]['deltavals']] for x in range(len(results))])
	deltavals = deltavals[:,0,:]
	dmean = np.mean(deltavals,axis=0)
	dsem = np.std(deltavals,axis=0)/np.sqrt(NP)
	dT = dmean/(dsem + 1.0e-20)

	betavals = dbetavals/(deltavals + 1.0e-6)
	betavals[np.abs(betavals) > 1e3] = 0.0
	bmean = np.mean(betavals,axis=0)
	bsem = np.std(betavals,axis=0)/np.sqrt(NP)
	bT = bmean/(bsem + 1.0e-20)

	R2avg = np.array([[results[x]['R2avg']] for x in range(len(results))])
	R2total = np.array([[results[x]['R2total']] for x in range(len(results))])
	entry = {'dbetavals': dbetavals, 'dbmean': dbmean, 'dbsem': dbsem, 'dbT': dbT,
			 'deltavals':deltavals, 'dmean':dmean, 'dsem':dsem, 'dT':dT,
			'betavals': betavals, 'bmean': bmean, 'bsem': bsem, 'bT': bT,
			 'R2avg':R2avg, 'R2total':R2total}

	record.append(entry)

NP,ncon = np.shape(record[0]['dbetavals'])

bins = np.linspace(0.2,0.6,50)
wn = 30
plt.close(wn)
fig = plt.figure(wn)
plt.hist(record[2]['R2avg'], bins = bins, alpha = 0.5, fc = (0,0,1))
plt.hist(record[0]['R2avg'], bins = bins, alpha = 0.5, fc = (1,0,0))

plt.close(wn+2)
fig = plt.figure(wn+2)
plt.plot(list(range(ncon)), record[0]['dbetavals'].T, 'ob' )
plt.plot(list(range(ncon)), record[2]['dbetavals'].T, 'or', alpha = 0.5 )

plt.close(wn+3)
fig = plt.figure(wn+3)
plt.plot(list(range(ncon)), record[0]['betavals'].T, 'ob' )
plt.plot(list(range(ncon)), record[2]['betavals'].T, 'or', alpha = 0.5 )

plt.close(wn+4)
fig = plt.figure(wn+4)
plt.plot(list(range(ncon)), record[0]['deltavals'].T, 'ob' )
plt.plot(list(range(ncon)), record[2]['deltavals'].T, 'or', alpha = 0.5 )


bins2 = np.linspace(-5,5,50)
plt.close(wn+5)
fig = plt.figure(wn+5)
plt.hist(record[0]['bT'], bins = bins2, alpha = 1.0, fc = (0,0,1))
plt.hist(record[2]['bT'], bins = bins2, alpha = 0.5, fc = (1,0,0))

bins2 = np.linspace(-5,5,50)
plt.close(wn+6)
fig = plt.figure(wn+6)
plt.hist(record[0]['dbT'], bins = bins2, alpha = 1.0, fc = (0,0,1))
plt.hist(record[2]['dbT'], bins = bins2, alpha = 0.5, fc = (1,0,0))