# function to analyze cluster data using PCA
import numpy as np
import os
from sklearn.decomposition import PCA
import pandas as pd
import pysapm
import matplotlib.pyplot as plt

# groupnum = 6
cnums = [0, 2, 4, 0, 0, 2, 1, 2, 1, 3]

datadir = r'E:\SAPMresults_Dec2022'
# nametag = '_1340023413'
# resultsbase = ['RSnostim','Sens', 'Low', 'Pain','High', 'RSstim', 'allpain']
# covnamebase = ['RSnostim','Sens', 'Low', 'Pain2','High', 'RSstim2', 'allpain']

# nresults = len(resultsbase)
# resultsnames = [resultsbase[x]+nametag+'_results.npy' for x in range(nresults)]
# paramsnames = [resultsbase[x]+nametag+'_params.npy' for x in range(nresults)]
# covnames = [covnamebase[x]+'_covariates.npy' for x in range(nresults)]
# regiondatanames = [resultsbase[x]+'_regiondata2.npy' for x in range(nresults)]
clusterdataname = r'E:/SAPMresults_Dec2022\Pain_equalsize_cluster_def.npy'
networkfile = r'E:\SAPMresults_Dec2022\network_model_Jan2023.xlsx'
DBname = r'E:\graded_pain_database_May2022.xlsx'

network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count = pysapm.load_network_model_w_intrinsics(networkfile)
SAPMparametersname = r'E:\SAPMresults_Dec2022\Pain_0240021213_params.npy'
regiondataname = r'E:\SAPMresults_Dec2022\Pain_regiondata2.npy'
covdataname = r'E:\SAPMresults_Dec2022\Pain2_covariates.npy'

resultsname = r'E:\SAPMresults_Dec2022\Pain_0240021213_results.npy'
results = np.load(resultsname, allow_pickle=True)
Mintrinsic = results[0]['Mintrinsic']
Meigv = results[0]['Meigv']
Mconn = results[0]['Mconn']
Minput = results[0]['Minput']
Sinput = results[0]['Sinput']
Sconn = results[0]['Sconn']
Mintrinsic = results[0]['Mintrinsic']

# load paradigm data--------------------------------------------------------------------
xls = pd.ExcelFile(DBname, engine='openpyxl')
df1 = pd.read_excel(xls, 'paradigm1_BOLD')
del df1['Unnamed: 0']  # get rid of the unwanted header column
fields = list(df1.keys())
paradigm = df1['paradigms_BOLD']
timevals = df1['time']
paradigm_centered = paradigm - np.mean(paradigm)
dparadigm = np.zeros(len(paradigm))
dparadigm[1:] = np.diff(paradigm_centered)

# get cluster info and setup for saving information later
cluster_data = np.load(clusterdataname, allow_pickle=True).flat[0]
# cluster_properties = cluster_data['cluster_properties']
cluster_properties = pysapm.load_filtered_cluster_properties(clusterdataname, networkfile)
nregions = len(cluster_properties)
nclusterlist = [cluster_properties[i]['nclusters'] for i in range(nregions)]
rnamelist = [cluster_properties[i]['rname'] for i in range(nregions)]
namelist_addon = ['R ' + n for n in rnamelist]
namelist = rnamelist + namelist_addon

# ---------------------
# prep_data_sem_physio_model(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch)
SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
ctarget = SAPMparams['ctarget']
csource = SAPMparams['csource']
tcdata = SAPMparams['tcdata_centered']  # data for all regions/clusters concatenated along time dimension for all runs
betanamelist = SAPMparams['betanamelist']
beta_list = SAPMparams['beta_list']
rnamelist = SAPMparams['rnamelist']
nregions = SAPMparams['nregions']

con_names = []
ncon = len(betanamelist)
for nn in range(ncon):
    pair = beta_list[nn]['pair']
    if pair[0] >= nregions:
        sname = 'int{}'.format(pair[0]-nregions)
    else:
        sname = rnamelist[pair[0]]
    tname = rnamelist[pair[1]]
    name = '{}-{}'.format(sname[:4],tname[:4])
    con_names += [name]

connection_names = []
for nn in range(len(csource)):
    source_pair = beta_list[csource[nn]]['pair']
    target_pair = beta_list[ctarget[nn]]['pair']
    if source_pair[0] >= nregions:
        sname = 'int{}'.format(source_pair[0]-nregions)
    else:
        sname = rnamelist[source_pair[0]]
    mname = rnamelist[target_pair[0]]
    tname = rnamelist[target_pair[1]]
    name = '{}-{}-{}'.format(sname[:4],mname[:4],tname[:4])
    connection_names += [name]
#------------------------------------------------------



# need to get principal components for each region to model the clusters as a continuum

nclusters_total, tsize_total = np.shape(tcdata)
component_data = np.zeros(np.shape(tcdata))
average_data = np.zeros(np.shape(tcdata))
ncmax = np.max(nclusterlist)
original_loadings = np.zeros((nregions, ncmax, ncmax))
weights = np.zeros((nregions, ncmax))

# want PCA for all timecourse data in all clusters in each person
region_data1 = np.load(regiondataname, allow_pickle=True).flat[0]
region_properties = region_data1['region_properties']
tsize = region_properties[0]['tsize']
nruns_per_person = region_properties[0]['nruns_per_person']

timepoint = 'all'
# setup index lists---------------------------------------------------------------------------
# timepoints for full runs----------------------------------------------
if timepoint == 'all':
    epoch = tsize
    timepoint = np.floor(tsize/2)

tplist_full = []
if epoch >= tsize:
    et1 = 0
    et2 = tsize
else:
    if np.floor(epoch/2).astype(int) == np.ceil(epoch/2).astype(int):   # even numbered epoch
        et1 = (timepoint - np.floor(epoch / 2)).astype(int)
        et2 = (timepoint + np.floor(epoch / 2)).astype(int)
    else:
        et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
        et2 = (timepoint + np.floor(epoch / 2)).astype(int)
if et1 < 0: et1 = 0
if et2 > tsize: et2 = tsize
epoch = et2-et1

dtsize = tsize - 1  # for using deriviation of tc wrt time
tplist1 = []
nclusterstotal, tsizetotal = np.shape(tcdata)
tcdata_centered = copy.deepcopy(tcdata)
NP = len(nruns_per_person)
for nn in range(NP):
    r1 = sum(nruns_per_person[:nn])
    r2 = sum(nruns_per_person[:(nn + 1)])
    tp = []  # initialize list
    tpoints = []
    for ee2 in range(r1, r2):
        # tp = list(range((ee2 * tsize), (ee2 * tsize + tsize)))
        tp = list(range((ee2*tsize+et1),(ee2*tsize+et2)))
        tpoints += tp  # concatenate lists
        temp = np.mean(tcdata[:, tp], axis=1)
        temp_mean = np.repeat(temp[:, np.newaxis], epoch, axis=1)
        tcdata_centered[:, tp] = tcdata[:, tp] - temp_mean  # center each epoch, in each person
    tplist1.append({'tp': tpoints})
tplist_full.append(tplist1)

epochnum = 0
nperson = 0
tp = tplist_full[epochnum][nperson]['tp']
Sinput = tcdata_centered[:,tp]

# compile Sinput average per person
Sinput_avg = np.zeros((NP*nclusterstotal,tsize))
for nn in range(NP):
    tp = tplist_full[epochnum][nn]['tp']
    Stemp = tcdata_centered[:,tp]
    Stempr = np.reshape(Stemp,(nclusterstotal,nruns_per_person[nn],tsize))
    Stemp_avg = np.mean(Stempr,axis=1)
    c1 = nn*nclusterstotal
    c2 = (nn+1)*nclusterstotal
    Sinput_avg[c1:c2,:] = copy.deepcopy(Stemp_avg)

# PCA of Sinput
nregions,tsizefull = np.shape(Sinput)
Sin_std = np.repeat(np.std(Sinput,axis=1)[:,np.newaxis],tsizefull,axis=1)
Sinput_norm = Sinput/Sin_std
nstates = copy.deepcopy(nregions)
pca = PCA(n_components=nstates)
pca.fit(Sinput_norm)
S_pca_ = pca.fit(Sinput_norm).transform(Sinput_norm)

components = pca.components_
evr = pca.explained_variance_ratio_
# get loadings
mu = np.mean(Sinput_norm, axis=0)
mu = np.repeat(mu[np.newaxis, :], nstates, axis=0)

loadings = pca.transform(Sinput_norm)
fit_check = (loadings @ components) + mu
nterms = 3
terms = [0,1,2]
fit_check_reduced = (loadings[:,terms] @ components[terms,:]) + mu

for nn in range(nregions):
	R2_0 = 1.0 - np.sum((Sinput_norm[nn,:]-fit_check[nn,:])**2)/np.sum(Sinput_norm[nn,:]**2)
	R2_1 = 1.0 - np.sum((Sinput_norm[nn,:]-fit_check_reduced[nn,:])**2)/np.sum(Sinput_norm[nn,:]**2)
	R2_2 = 1.0 - np.sum((Sinput_norm[nn,:]-mu[nn,:])**2)/np.sum(Sinput_norm[nn,:]**2)
	print('person {}   region {}   R2full {:.3f}   R2reduced {:.3f}   R2mean {:.3f}'.format(nperson,nn,R2_0,R2_1,R2_2))

# normalize the loadings for comparison
nclusterstotal
load_norm = np.repeat(np.linalg.norm(loadings,axis=1)[:,np.newaxis], nclusterstotal, axis =1)
loadings_normalized = loadings/load_norm



# PCA of Sinput_avg
nstates = copy.deepcopy(tsize)
pca2 = PCA(n_components=nstates)
pca2.fit(Sinput_avg)
S2_pca_ = pca2.fit(Sinput_avg).transform(Sinput_avg)

components2 = pca2.components_
evr2 = pca2.explained_variance_ratio_
# get loadings
mu2 = np.mean(Sinput_avg, axis=0)
mu2 = np.repeat(mu2[np.newaxis, :], NP*nclusterstotal, axis=0)

loadings2 = pca2.transform(Sinput_avg)
fit_check2 = (loadings2 @ components2) + mu2
nterms = 10
fit_check_reduced2 = (loadings2[:,:nterms] @ components2[:nterms,:]) + mu2


windownumber = 11
regionnum = 1
plt.close(windownumber)
fig = plt.figure(windownumber)
plt.plot(range(tsize),Sinput_avg[regionnum,:],'-xr')
plt.plot(range(tsize),fit_check2[regionnum,:],'-b')
plt.plot(range(tsize),fit_check_reduced2[regionnum,:],'-g')
plt.plot(range(tsize),mu2[regionnum,:],'-y')
