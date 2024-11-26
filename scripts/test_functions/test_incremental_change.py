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


datadir = r'E:\SAPMresults_Dec2022'
savefilename = r'E:\SAPMresults_Dec2022\temp_params_save.npy'
reload_old_results = False
save_new_results = True
windowbasenum = 300

resultsname_norm = r'E:\SAPMresults_Dec2022\AllPain_Norm_3242423012_results.npy'
paramsname_norm = r'E:\SAPMresults_Dec2022\AllPain_Norm_3242423012_params.npy'

resultsname = r'E:\SAPMresults_Dec2022\AllPain_3242423012_results.npy'
paramsname = r'E:\SAPMresults_Dec2022\AllPain_3242423012_params.npy'

nperson = 2
# normalized results and data
SAPMresults_norm = np.load(resultsname_norm, allow_pickle=True)
SAPMparams_norm = np.load(paramsname_norm, allow_pickle=True).flat[0]
Sinput_norm = SAPMresults_norm[nperson]['Sinput']
Sconn_norm = SAPMresults_norm[nperson]['Sconn']
Mintrinsic_norm = SAPMresults_norm[nperson]['Mintrinsic']

# original results and data
SAPMresults = np.load(resultsname, allow_pickle=True)
Sinput = SAPMresults[nperson]['Sinput']
Sconn = SAPMresults[nperson]['Sconn']

Sinput_goal = copy.deepcopy(Sinput)
Sinput = copy.deepcopy(Sinput_norm)

nr,tsize = np.shape(Sinput)

betavals = SAPMresults_norm[nperson]['betavals']
deltavals = SAPMresults_norm[nperson]['deltavals']
Minput = SAPMresults_norm[nperson]['Minput']
Mconn = SAPMresults_norm[nperson]['Mconn']
Mintrinsic = SAPMresults_norm[nperson]['Mintrinsic']
fintrinsic1 = SAPMresults_norm[nperson]['fintrinsic1']
beta_int1 = SAPMresults_norm[nperson]['beta_int1']
ctarget = SAPMparams_norm['ctarget']
csource = SAPMparams_norm['csource']
dtarget = SAPMparams_norm['dtarget']
dsource = SAPMparams_norm['dsource']
fintrinsic_count = SAPMparams_norm['fintrinsic_count']
vintrinsic_count = SAPMparams_norm['vintrinsic_count']
latent_flag = SAPMparams_norm['latent_flag']

fit_norm, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count,
                                                         beta_int1, fintrinsic1)
betalimit = 2.0

test_results = pysapm.sem_physio_model_incremental_change(Sinput, Sinput_goal, betavals, deltavals, Minput, Mconn, Mintrinsic,
                            betalimit, ctarget, csource, dtarget, dsource, fintrinsic_count, vintrinsic_count,
                            beta_int1, fintrinsic1, latent_flag=latent_flag, verbose = True)

wn = 30
plt.close(wn)
fig = plt.figure(wn)
rr=0
plt.plot(range(tsize),Sinput[rr,:],'-xr')
plt.plot(range(tsize),fit_norm[rr,:],'-b')

plt.close(wn+1)
fig = plt.figure(wn+1)
plt.plot(range(tsize),Sinput_goal[rr,:],'-xr')
plt.plot(range(tsize),test_results['fit'][rr,:],'-b')


plt.close(95)
fig = plt.figure(95)
plt.plot(range(200),results1[40]['Sinput'][rr,:],'-xr')
plt.plot(range(200),fit[rr,:],'-b')



# compare some results------------------------------------------------------------
resultsnameN = r'E:\SAPMresults_Dec2022\AllPainN_3242423012_results.npy'
resultsname0 = r'E:\SAPMresults_Dec2022\AllPain0_3242423012_results.npy'
resultsname0b = r'E:\SAPMresults_Dec2022\AllPain0b_3242423012_results.npy'
resultsname0c = r'E:\SAPMresults_Dec2022\AllPain0c_3242423012_results.npy'
resultsname3 = r'E:\SAPMresults_Dec2022\AllPain3_3242423012_results.npy'
resultsN = np.load(resultsnameN, allow_pickle=True)
results0 = np.load(resultsname0, allow_pickle=True)
results0b = np.load(resultsname0b, allow_pickle=True)
results0c = np.load(resultsname0c, allow_pickle=True)
results3 = np.load(resultsname3, allow_pickle=True)

person = 0
fitN = resultsN[person]['Minput'] @ resultsN[person]['Meigv'] @ resultsN[person]['Mintrinsic']
fit0 = results0[person]['Minput'] @ results0[person]['Meigv'] @ results0[person]['Mintrinsic']
fit0b = results0b[person]['Minput'] @ results0b[person]['Meigv'] @ results0b[person]['Mintrinsic']
fit0c = results0c[person]['Minput'] @ results0c[person]['Meigv'] @ results0c[person]['Mintrinsic']
fit3 = results3[person]['Minput'] @ results3[person]['Meigv'] @ results3[person]['Mintrinsic']

rr=1
wn = 13
plt.close(wn)
fig = plt.figure(wn)
plt.plot(range(200),resultsN[person]['Sinput'][rr,:],'-xr')
plt.plot(range(200),fitN[rr,:],'-b')

plt.close(wn+1)
fig = plt.figure(wn+1)
plt.plot(range(200),results0[person]['Sinput'][rr,:],'-xr')
plt.plot(range(200),fit0[rr,:],'-b')

plt.close(wn+2)
fig = plt.figure(wn+2)
plt.plot(range(200),results0b[person]['Sinput'][rr,:],'-xr')
plt.plot(range(200),fit0b[rr,:],'-b')

plt.close(wn+3)
fig = plt.figure(wn+3)
plt.plot(range(200),results0c[person]['Sinput'][rr,:],'-xr')
plt.plot(range(200),fit0c[rr,:],'-b')

plt.close(wn+4)
fig = plt.figure(wn+4)
plt.plot(range(200),results3[person]['Sinput'][rr,:],'-xr')
plt.plot(range(200),fit3[rr,:],'-b')
