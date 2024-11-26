# test_CCrecord_display
# copied to split_results_CCrecord so that first efforts are not lost
# sys.path.append(r'C:\Users\Stroman\PycharmProjects\pyspinalfmri3\venv')

import numpy as np
import matplotlib.pyplot as plt
import py2ndlevelanalysis
import copy
import pyclustering
import pydisplay
import time
import pysem
from sklearn.cluster import KMeans
import scipy.stats as stats
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
from mpl_toolkits import mplot3d

from sklearn.cluster import KMeans




#-----------------------------------------------------------------------
def setBoxColors(bp):
    group1col = 'red'
    plt.setp(bp['boxes'][0], color=group1col)
    plt.setp(bp['caps'][0], color=group1col)
    plt.setp(bp['caps'][1], color=group1col)
    plt.setp(bp['whiskers'][0], color=group1col)
    plt.setp(bp['whiskers'][1], color=group1col)
    plt.setp(bp['medians'][0], color=group1col)

    group2col = 'blue'
    plt.setp(bp['boxes'][1], color=group2col)
    plt.setp(bp['caps'][2], color=group2col)
    plt.setp(bp['caps'][3], color=group2col)
    plt.setp(bp['whiskers'][2], color=group2col)
    plt.setp(bp['whiskers'][3], color=group2col)
    plt.setp(bp['medians'][1], color=group2col)

#-----------------------------------------------------------------------
# main program

Tthresh_connection1 = 4.0
Tthresh_connection2 = 3.0
groupsizelimit = 0.2

outputdir = r'D:/threat_safety_python/SEMresults'

plt.close(3)
plt.close(4)
#-----------------------------------------------------------------------------------------
networkfile = r'D:/threat_safety_python/network_possible_connections.xlsx'
network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkfile)

Zsavedataname = r'D:\threat_safety_python\Zgridsplit_saved_data.npy'
networksavename = r'D:\threat_safety_python\Zgridsplit_3level_trace.npy'

reload_saved_data = False
if reload_saved_data:
    saveddata = np.load(Zsavedataname, allow_pickle=True).flat[0]
    data1 = saveddata['data1']
    covariates1 = saveddata['covariates1']
    covariates2 = saveddata['covariates2']
    region_data1 = saveddata['region_data1']
    Zsplit = saveddata['Zsplit']
else:
    region_data_name1 = r'D:/threat_safety_python/SEMresults/threat_safety_regiondata_allthreat55.npy'
    region_data1 = np.load(region_data_name1, allow_pickle=True).flat[0]

    filename1 = r'D:/threat_safety_python/SEMresults/SEMresults_2source_record_allthreat55.npy'
    data1 = np.load(filename1, allow_pickle=True).flat[0]

    # get covariates
    settingsfile = r'C:\Users\Stroman\PycharmProjects\pyspinalfmri3\venv\base_settings_file.npy'
    settings = np.load(settingsfile, allow_pickle=True).flat[0]
    covariates1 = settings['GRPcharacteristicsvalues'][0]  # gender
    covariates2 = settings['GRPcharacteristicsvalues'][1].astype(float)  # painrating

    # saveddata = {'data1':data1, 'covariates1':covariates1, 'covariates2':covariates2, 'region_data1':region_data1}
    # np.save(CCsavedataname,saveddata)

#----------------------------------------------------------------
if split_by_pain:
    cov4split = covariates2
    covariatelabel = 'pain ratings'

if split_by_sex:
    cov4split = np.zeros(len(covariates2))
    covariatelabel = 'sex'
    for nn in range(len(covariates2)):
        if covariates1[nn] == 'Female':
            cov4split[nn] = 1
        else:
            cov4split[nn] = -1
#----------------------------------------------------------------

region_properties = region_data1['region_properties']
# # dict_keys(['tc', 'tc_sem', 'nruns_per_person', 'tsize'])
cluster_properties = data1['cluster_properties']
beta2 = copy.deepcopy(data1['beta2'])
Zgrid2 = copy.deepcopy(data1['Zgrid2'])
CCrecord = copy.deepcopy(data1['CCrecord'])
ntarget,nsource1,nsource2,ntime,NP,nc = np.shape(beta2)

# setup lists
g1 = np.where(covariates1 == 'Female')[0]
g2 = np.where(covariates1 == 'Male')[0]
groupnames = ['Female', 'Male']

nregions = len(cluster_properties)
nclusterlist = [cluster_properties[i]['nclusters'] for i in range(nregions)]
rnamelist = [cluster_properties[i]['rname'] for i in range(nregions)]
nclusterstotal = np.sum(nclusterlist)

tsize = region_properties[0]['tsize']
nruns_per_person = region_properties[0]['nruns_per_person']
NP = len(nruns_per_person)  # number of people in the data set


#  fit parameters etc can be determined entirely from the variance/covariance grid between every pair of regions
# b1 = (Ct1 - Ct2 * C12 / V2) / (V1 - (C12 ** 2) / V2)
# b2 = (Ct2 - Ct1 * C12 / V1) / (V2 - (C12 ** 2) / V1)
# fit_check = b1 * tc1 + b2 * tc2
# R2ts1 = 2 * b1 * Ct1 / Vt - b1 ** 2 * V1 / Vt
# R2ts2 = 2 * b2 * Ct2 / Vt - b2 ** 2 * V2 / Vt
# R2s1s2 = -2 * b1 * b2 * C12 / Vt
# R2check = R2ts1 + R2ts2 + R2s1s2
# can expand this to 2-source 3D volume for each participant
# take that data and see how it compares with pain behaviors, personal characteristics etc.

# find out where the beta values split between positive and negative values to give different behavioral effects

timepoint = 0
target = 1    # CRRD1 is first target
tname, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, target)

connection_list = ['DRt','NRM','NGC','LC']
sregions = [rnamelist.index(rname) for rname in connection_list]
exclude_sregions = np.setdiff1d(list(range(len(rnamelist))),sregions)
source_exclude = []
for ss in exclude_sregions:
    t1 = np.sum(nclusterlist[:ss]).astype(int)
    t2 = np.sum(nclusterlist[:(ss + 1)])
    source_exclude += list(range(t1,t2))

t1 = np.sum(nclusterlist[:tnumber]).astype(int)
t2 = np.sum(nclusterlist[:(tnumber + 1)])
target_exclude = list(range(t1, t2))
exclude = np.unique(target_exclude+source_exclude)
# tn = np.setdiff1d(list(range(ntarget)),target_exclude)

Zvalgrid = np.zeros((NP,nsource1*nsource2))
for pp in range(NP):  # pick one person
    Z = copy.deepcopy(Zgrid2[target,:,:,timepoint,pp])
    Z[:,exclude] = 0.
    Z[exclude,:] = 0.
    Zvalgrid[pp,:] = np.reshape(Z,(1,nsource1*nsource2))

# for nn in range(nsource1*nsource2):
#     if np.var(Zvalgrid[:,nn]) > 0:
#         keeplist += nn

keeplist = [nn for nn in range(nsource1*nsource2) if np.var(Zvalgrid[:,nn]) > 0]

kmeans = KMeans(n_clusters=2, random_state=0).fit(Zvalgrid[:,keeplist])
IDX = kmeans.labels_
cluster_tc = kmeans.cluster_centers_

g1 = np.where(IDX == 0)[0]
g2 = np.where(IDX == 1)[0]

Z = copy.deepcopy(Zgrid2[target,:,:,timepoint,:])
Z1 = np.mean(Z[:,:,g1],axis = 2)
Z2 = np.mean(Z[:,:,g2],axis = 2)

plt.close(20)
fig = plt.figure(20, figsize=(10, 8), dpi=100)
plt.subplot(1,2,1)
plt.imshow(Z1)

plt.subplot(1,2,2)
plt.imshow(Z2)

#----------------------------------------------------------
target = 1
timepoint = 1
Zplot_grid = np.zeros((10,NP))
for sourcenum in range(1,10):
    sourcename = rnamelist[sourcenum]
    t1 = np.sum(nclusterlist[:sourcenum]).astype(int)
    t2 = np.sum(nclusterlist[:(sourcenum + 1)])
    ssvals = range(t1,t2)

    Z = copy.deepcopy(Zgrid2[target,:,:,timepoint,:])
    N = np.zeros((NP,7))
    Zplot = []
    Rplot = []
    for pp in range(NP):
        Z2 = copy.deepcopy(Z[:,:,pp])
        Zb = np.zeros((5,nsource1))
        for ss in range(t1): Zb[:,ss] = Z2[ss,t1:t2]
        for ss in range(t2,nsource1): Zb[:,ss] = Z2[t1:t2,ss]

        x = np.argmax(Zb)
        s1b,s2 = np.unravel_index(x,(5,nsource1))
        s1 = ssvals[s1b]
        if s1 > s2:
            a = s1
            s1 = s2
            s2 = a

        if s1 in ssvals:
            connum = 0
        else:
            connum = 1

        Zp = Zgrid2[target,s1,s2,timepoint,pp]

        Ct1 = CCrecord[timepoint,pp,target,s1]
        Ct2 = CCrecord[timepoint,pp,target,s2]
        C12 = CCrecord[timepoint,pp,s1,s2]
        Vt = CCrecord[timepoint,pp,target,target]
        V1 = CCrecord[timepoint,pp,s1,s1]
        V2 = CCrecord[timepoint,pp,s2,s2]

        b1 = (Ct1 - Ct2 * C12 / V2) / (V1 - (C12 ** 2) / V2)
        b2 = (Ct2 - Ct1 * C12 / V1) / (V2 - (C12 ** 2) / V1)
        # fit_check = b1 * tc1 + b2 * tc2
        R2ts1 = 2 * b1 * Ct1 / Vt - b1 ** 2 * V1 / Vt
        R2ts2 = 2 * b2 * Ct2 / Vt - b2 ** 2 * V2 / Vt
        R2s1s2 = -2 * b1 * b2 * C12 / Vt
        R2check = R2ts1 + R2ts2 + R2s1s2

        N[pp,:] = [target,s1,s2,Zp,R2ts1,R2ts2,R2s1s2]

        if connum == 0:
            Zplot += [R2ts1]
            Rplot += [s1]
        else:
            Zplot += [R2ts2]
            Rplot += [s2]

        print('{}  s1 {}  s2 {}  Z = {:.2f}  R2ts1 {:.3f}  R2ts2 {:.3f}  R2s1s2 {:.3f}'.format(pp,s1,s2,Zp,R2ts1,R2ts2,R2s1s2))
    print('source = {}'.format(sourcename))
    Zplot_grid[sourcenum,:] = np.array(Zplot)

    plt.close(sourcenum+1)
    fig = plt.figure(sourcenum+1, figsize=(10, 6), dpi=100)
    plt.bar(range(NP),Zplot,tick_label = Rplot)
    plt.ylim(-0.1,0.4)
