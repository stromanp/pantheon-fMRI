# sys.path.append(r'C:\Users\Stroman\PycharmProjects\pantheon\venv')
import numpy as np
import matplotlib.pyplot as plt
import pysapm
import os
import copy
import pydatabase

datadir = r'E:\SAPMresults_Dec2022'

networkfile = r'E:\SAPMresults_Dec2022\network_model_Jan2023.xlsx'
DBname = r'E:\graded_pain_database_May2022.xlsx'

rname = os.path.join(datadir,'AllPain_equal_regiondata.npy')
rnameL = os.path.join(datadir,'AllPain_equal_lowR2_regiondata.npy')
rnameH = os.path.join(datadir,'AllPain_equal_highR2_regiondata.npy')

regiondata = np.load(rname,allow_pickle=True).flat[0]
region_properties = regiondata['region_properties']
DBnum = regiondata['DBnum']

resultsname = os.path.join(datadir,'AllPainE_1202023213_results.npy')
resultsnameL = os.path.join(datadir,'LowR2_1202023213_results.npy')
resultsnameH = os.path.join(datadir,'HighR2_1202023213_results.npy')

paramsname = os.path.join(datadir,'AllPainE_1202023213_params.npy')
paramsnameL = os.path.join(datadir,'LowR2_1202023213_params.npy')
paramsnameH = os.path.join(datadir,'HighR2_1202023213_params.npy')

results = np.load(resultsnameH,allow_pickle=True)
R2list = np.array([results[x]['R2avg'] for x in range(len(results))])
R2list2 = np.array([results[x]['R2total'] for x in range(len(results))])

R2diff = np.abs(R2list-R2list2)
x = np.argmin(R2diff)
x = np.argmax(R2list)
print('R2avg = {:.3f}  R2total = {:.3f}'.format(R2list[x],R2list2[x]))

Sinput = results[x]['Sinput']
Sconn = results[x]['Sconn']
Minput = results[x]['Minput']
Mconn = results[x]['Mconn']
fintrinsic_count = results[x]['fintrinsic_count']
vintrinsic_count = results[x]['vintrinsic_count']
beta_int1 = results[x]['beta_int1']
fintrinsic1 = results[x]['fintrinsic1']
Nintrinsic = fintrinsic_count+vintrinsic_count
fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count,
														 beta_int1, fintrinsic1)

nr,tsizefull = np.shape(Sinput)


# show results in different formats
SAPMparams = np.load(paramsnameH, allow_pickle=True).flat[0]
network = SAPMparams['network']
beta_list = SAPMparams['beta_list']
betanamelist = SAPMparams['betanamelist']
nruns_per_person = SAPMparams['nruns_per_person']
rnamelist = SAPMparams['rnamelist']
fintrinsic_count = SAPMparams['fintrinsic_count']
fintrinsic_region = SAPMparams['fintrinsic_region']
vintrinsic_count = SAPMparams['vintrinsic_count']
nclusterlist = SAPMparams['nclusterlist']
tplist_full = SAPMparams['tplist_full']
tcdata_centered = SAPMparams['tcdata_centered']
ctarget = SAPMparams['ctarget']
csource = SAPMparams['csource']
tsize = SAPMparams['tsize']


nruns = nruns_per_person[x]

Sinput_avg = np.mean(np.reshape(Sinput, (nr, nruns, tsize)), axis=1)
Sinput_sem = np.std(np.reshape(Sinput, (nr, nruns, tsize)), axis=1) / np.sqrt(nruns)
Sconn_avg = np.mean(np.reshape(Sconn, (nr+Nintrinsic, nruns, tsize)), axis=1)
Sconn_sem = np.std(np.reshape(Sconn, (nr+Nintrinsic, nruns, tsize)), axis=1) / np.sqrt(nruns)
fit_avg = np.mean(np.reshape(fit, (nr, nruns, tsize)), axis=1)
fit_sem = np.std(np.reshape(fit, (nr, nruns, tsize)), axis=1) / np.sqrt(nruns)


windownum = 101
plt.close(windownum)
fig = plt.figure(windownum)
for rr in range(nr):
	fig.add_subplot(10,1,rr+1)
	plt.plot(range(tsizefull),Sinput[rr,:],'-xr')
	plt.plot(range(tsizefull),fit[rr,:],'-b')


windownum = 102
plt.close(windownum)
fig = plt.figure(windownum)
for rr in range(nr):
	fig.add_subplot(10,1,rr+1)
	plt.plot(range(tsize),Sinput_avg[rr,:],'-xr')
	plt.plot(range(tsize),fit_avg[rr,:],'-b')


window = 103
plt.close(window)
target = 'C6RD'
nametag1 = 'oneperson'
outputname = pysapm.plot_region_inputs_average(window, target, nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem,
										Sconn_avg,Sconn_sem, beta_list, rnamelist, betanamelist, Mconn, datadir)

