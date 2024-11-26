# test_CCrecord_display
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


outputdir = r'D:/threat_safety_python/SEMresults'
networkfile = r'D:/threat_safety_python/network_possible_connections.xlsx'
network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkfile)

CCsavedataname = r'D:\threat_safety_python\conn_investigation_saved_data.npy'
reload_saved_data = True
if reload_saved_data:
    saveddata = np.load(CCsavedataname, allow_pickle=True).flat[0]
    data1 = saveddata['data1']
    covariates1 = saveddata['covariates1']
    covariates2 = saveddata['covariates2']
    region_data1 = saveddata['region_data1']
    paradigm = saveddata['paradigm']
    timevals = saveddata['timevals']
else:
    DBname = r'D:/threat_safety_python/threat_safety_database.xlsx'
    xls = pd.ExcelFile(DBname, engine='openpyxl')
    df1 = pd.read_excel(xls, 'paradigm1_BOLD')
    del df1['Unnamed: 0']  # get rid of the unwanted header column
    fields = list(df1.keys())
    paradigm = df1['paradigms_BOLD']
    timevals = df1['time']

    region_data_name1 = r'D:/threat_safety_python/SEMresults/threat_safety_regiondata_allthreat55.npy'
    region_data1 = np.load(region_data_name1, allow_pickle=True).flat[0]

    filename1 = r'D:/threat_safety_python/SEMresults/SEMresults_2source_record_allthreat55.npy'
    data1 = np.load(filename1, allow_pickle=True).flat[0]

    # get covariates
    settingsfile = r'C:\Users\Stroman\PycharmProjects\pyspinalfmri3\venv\base_settings_file.npy'
    settings = np.load(settingsfile, allow_pickle=True).flat[0]
    covariates1 = settings['GRPcharacteristicsvalues'][0]  # gender
    covariates2 = settings['GRPcharacteristicsvalues'][1].astype(float)  # painrating

    saveddata = {'data1':data1, 'covariates1':covariates1, 'covariates2':covariates2, 'region_data1':region_data1, 'paradigm':paradigm, 'timevals':timevals}
    np.save(CCsavedataname,saveddata)

region_properties = region_data1['region_properties']
# # dict_keys(['tc', 'tc_sem', 'nruns_per_person', 'tsize'])
cluster_properties = data1['cluster_properties']
#
# # setup lists
# NP = len(covariates1)
# g1 = np.where(covariates1 == 'Female')[0]
# g2 = np.where(covariates1 == 'Male')[0]
# groupnames = ['Female', 'Male']
#
# # over-ride this
# beta2 = data1['beta2']
# b2test = beta2[target,source1,source2,timepoint,:,nc]
# g1 = np.where(b2test > 0)[0]
# g2 = np.where(b2test <= 0)[0]
# groupnames = ['pos_beta', 'neg_beta']


# exclude = []
# gx = np.setdiff1d(list(range(NP)),exclude)
# g1x = np.setdiff1d(g1,exclude)
# g2x = np.setdiff1d(g2,exclude)

nregions = len(cluster_properties)
nclusterlist = [cluster_properties[i]['nclusters'] for i in range(nregions)]
rnamelist = [cluster_properties[i]['rname'] for i in range(nregions)]
nclusterstotal = np.sum(nclusterlist)

tsize = region_properties[0]['tsize']
nruns_per_person = region_properties[0]['nruns_per_person']
NP = len(nruns_per_person)  # number of people in the data set

tcdata = []
for i in range(nregions):
    tc = region_properties[i]['tc']
    # nc = nclusterlist[i]
    if i == 0:
        tcdata = tc
    else:
        tcdata = np.append(tcdata, tc, axis=0)


# setup index lists---------------------------------------------------------------------------
# nruns_per_person = region_data1['region_properties'][0]['nruns_per_person']
# tsize = region_data1['region_properties'][0]['tsize']
timepoints = [13, 20]
epoch = 7
tplist = []
tcdata_centered = copy.deepcopy(tcdata)
for ee in range(len(timepoints)):
    et1 = (timepoints[ee] - np.floor(epoch / 2)).astype(int) - 1
    et2 = (timepoints[ee] + np.floor(epoch / 2)).astype(int)
    tplist1 = []
    for nn in range(NP):
        r1 = sum(nruns_per_person[:nn])
        r2 = sum(nruns_per_person[:(nn + 1)])
        tp = []  # initialize list
        tpoints = []
        for ee2 in range(r1, r2):
            tp = list(range((ee2 * tsize + et1), (ee2 * tsize + et2)))
            tpoints = tpoints + tp  # concatenate lists
            temp = np.mean(tcdata[:, tp],axis=1)
            temp_mean = np.repeat(temp[:, np.newaxis], epoch, axis=1)
            tcdata_centered[:, tp] = tcdata[:, tp] - temp_mean   # center each epoch, in each person
        tplist1.append({'tp': tpoints})
    tplist.append(tplist1)


# timepoints for full runs----------------------------------------------
tplist_full = []
tcdata_centered = copy.deepcopy(tcdata)
et1 = 0
et2 = tsize
tplist1 = []
for nn in range(NP):
    r1 = sum(nruns_per_person[:nn])
    r2 = sum(nruns_per_person[:(nn + 1)])
    tp = []  # initialize list
    tpoints = []
    for ee2 in range(r1, r2):
        tp = list(range((ee2 * tsize), (ee2 * tsize + tsize)))
        tpoints = tpoints + tp  # concatenate lists
        temp = np.mean(tcdata[:, tp],axis=1)
        temp_mean = np.repeat(temp[:, np.newaxis], tsize, axis=1)
        tcdata_centered[:, tp] = tcdata[:, tp] - temp_mean   # center each epoch, in each person
    tplist1.append({'tp': tpoints})
tplist_full.append(tplist1)


CCrecord = data1['CCrecord']
beta2 = data1['beta2']
Zgrid2 = data1['Zgrid2']
ntime,NP,ntarget,nsource = np.shape(CCrecord)
ntarget, nsource1, nsource2, ntime, NP, ncon = np.shape(beta2)

# beta vals with one source and model paradigm as sources
beta2_intrinsic = np.zeros((ntarget, nsource1, NP, ncon))
beta2_full = np.zeros((ntarget, nsource1, nsource2, NP, ncon))

CCrecord_intrinsic = np.zeros((ntarget, NP))
CCrecord_full = np.zeros((ntarget,nsource1, NP))

R2grid_full = np.zeros((ntarget,nsource1,nsource2,NP,4))
R2grid_intrinsic = np.zeros((ntarget,nsource1,NP,4))

paradigm_centered = paradigm - np.mean(paradigm)
Vparadigm = np.var(paradigm_centered)
timepoint = 0
for nn in range(NP):
    print('person {} of {}  {}'.format(nn+1,NP,time.ctime()))
    for tt in range(ntarget):
        tname, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, tt)
        tp = tplist_full[timepoint][nn]['tp']
        tct = tcdata_centered[tt, tp]
        paradigm1 = np.array(list(paradigm_centered)*nruns_per_person[nn])
        c = np.cov(tct, paradigm1)
        CCrecord_intrinsic[tt,nn] = c[0,1]

        for ss1 in range(nsource1):
            sname1, scluster1, snumber1 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, ss1)
            if snumber1 != tnumber:
                tp = tplist_full[timepoint][nn]['tp']
                tct = tcdata_centered[tt, tp]
                tc1 = tcdata_centered[ss1, tp]

                c = np.cov(tct,tc1)
                CCrecord_full[tt,ss1,nn] = c[0,1]
                CCrecord_full[ss1,tt,nn] = c[0,1]
                CCrecord_full[tt,tt,nn] = c[0,0]
                CCrecord_full[ss1,ss1,nn] = c[1,1]

    for tt in range(ntarget):
        tname, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, tt)
        for ss1 in range(nsource1):
            sname1, scluster1, snumber1 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, ss1)
            if snumber1 != tnumber:
                # intrinsic
                Ct1 = CCrecord_full[tt, ss1, nn]
                Ctp = CCrecord_intrinsic[tt, nn]
                C1p = CCrecord_intrinsic[ss1, nn]
                Vt = CCrecord_full[tt,tt,nn]
                V1 = CCrecord_full[ss1,ss1,nn]
                b1 = (Ct1 - Ctp * C1p / Vparadigm) / (V1 - (C1p ** 2) / Vparadigm)
                b2 = (Ctp - Ct1 * C1p / V1) / (Vparadigm - (C1p ** 2) / V1)
                beta2_intrinsic[tt, ss1, nn, 0] = b1
                beta2_intrinsic[tt, ss1, nn, 1] = b2

                R2ts1 = 2 * b1 * Ct1 / Vt - b1 ** 2 * V1 / Vt
                R2tp = 2 * b2 * Ctp / Vt - b2 ** 2 * Vparadigm / Vt
                R2s1p = -2 * b1 * b2 * C1p / Vt
                R2total = R2ts1 + R2tp + R2s1p
                R2grid_intrinsic[tt, ss1, nn, :] = [R2ts1, R2tp, R2s1p, R2total]

                for ss2 in range(nsource2):
                    sname2, scluster2, snumber2 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, ss2)
                    if snumber2 != snumber1 and snumber2 != tnumber:
                        Ct1 = CCrecord_full[tt,ss1,nn]
                        Ct2 = CCrecord_full[tt,ss2,nn]
                        C12 = CCrecord_full[ss1,ss2,nn]
                        Vt = CCrecord_full[tt,tt,nn]
                        V1 = CCrecord_full[ss1,ss1,nn]
                        V2 = CCrecord_full[ss2,ss2,nn]

                        b1 = (Ct1 - Ct2 * C12 / V2) / (V1 - (C12 ** 2) / V2)
                        b2 = (Ct2 - Ct1 * C12 / V1) / (V2 - (C12 ** 2) / V1)
                        beta2_full[tt,ss1,ss2,nn,0] = b1
                        beta2_full[tt,ss1,ss2,nn,1] = b2

                        R2ts1 = 2 * b1 * Ct1 / Vt - b1 ** 2 * V1 / Vt
                        R2ts2 = 2 * b2 * Ct2 / Vt - b2 ** 2 * V2 / Vt
                        R2s1s2 = -2 * b1 * b2 * C12 / Vt
                        R2total = R2ts1 + R2ts2 + R2s1s2
                        R2grid_full[tt,ss1,ss2,nn,:] = [R2ts1,R2ts2,R2s1s2,R2total]

# pick a connection and compare-----------------------------------------------
target, source1, source2, ncon = (1, 8, 41, 1)     # C6RD1 target,  PBN1 source1, DRt3 source2
tname, tcluster, tnumber= py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, target)
sname1, scluster1, snumber12 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, source1)
sname2, scluster2, snumber2 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, source2)

b1full = beta2_full[target,source1,source2,:,0]
b2full = beta2_full[target,source1,source2,:,1]

if ncon == 0:
    source = source1
else:
    source = source2
    temp = b1full
    b1full = b2full
    b2full = temp
    temp_name = sname1
    temp_cluster = scluster1
    sname1 = sname2
    scluster1 = scluster2
    sname2 = temp_name
    scluster2 = temp_cluster

b1int = beta2_intrinsic[target,source,:,0]
b2int = beta2_intrinsic[target,source,:,1]

R2full = R2grid_full[target,source1,source2,:,3]
R2intrinsic = R2grid_intrinsic[target,source1,:,3]

x = np.argmax(R2grid_intrinsic[1,:,:,1])
a,b = np.unravel_index(x,(ntarget,NP))
R2 = R2grid_intrinsic[1,a,b,:]
R2

nn = 0
print('target {}{}   sources {}{}  {}{}'.format(tname,tcluster,sname1,scluster1,sname2,scluster2))
print('b1 = {:.2f}  b2 = {:.2f}   R2 = {:.4f}'.format(b1full[nn],b2full[nn],R2full[nn]))
print('target {}{}   source {}{}  and model paradigm'.format(tname,tcluster,sname1,scluster1))
print('b1 = {:.2f}  b2 = {:.2f} (paradigm)   R2 = {:.4f}'.format(b1int[nn],b2int[nn],R2intrinsic[nn]))

#  fit
tp = tplist_full[timepoint][nn]['tp']
tct = tcdata_centered[target, tp]
tc1 = tcdata_centered[source1, tp]
tc2 = tcdata_centered[source2, tp]

paradigm1 = np.array(list(paradigm_centered)*nruns_per_person[nn])
tfit = b1full[nn]*tc1 + b2full[nn]*tc2
tintfit = b1int[nn]*tc1 + b2int[nn]*paradigm1

fig = plt.figure(22)
plt.subplot(2,1,1)
plt.plot(tct,'-b')
plt.plot(tfit,'-r')
plt.subplot(2,1,2)
plt.plot(tct,'-b')
plt.plot(tintfit,'-g')
