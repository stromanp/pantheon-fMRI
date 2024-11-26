# test_CCrecord_display
# sys.path.append(r'C:\Users\Stroman\PycharmProjects\pyspinalfmri3\venv')

# test_connectivity_with_derivatives.py

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

# load data
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


# CCrecord with derivatives
paradigm_centered = paradigm - np.mean(paradigm)
CCdiffrecord = np.zeros((NP, ntarget, nsource1))
CCparadigm = np.zeros((NP,ntarget))
timepoint = 0
for nn in range(NP):
    print('person number {}   {}'.format(nn, time.ctime()))
    paradigm1 = np.array(list(paradigm_centered) * nruns_per_person[nn])
    dparadigm = np.diff(paradigm1)
    tp = tplist_full[timepoint][nn]['tp']

    for target in range(ntarget):
        tname, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, target)
        tct = tcdata_centered[target, tp]
        dtct = np.diff(tct)
        rval = np.corrcoef(dtct,dparadigm)
        CCparadigm[nn,target] = rval[0,1]

        for source1 in range(nsource1):
            sname1, scluster1, snumber1 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, source1)
            if snumber1 != tnumber:
                tc1 = tcdata_centered[source1, tp]
                dtc1 = np.diff(tc1)
                rval = np.corrcoef(dtct,dtc1)
                CCdiffrecord[nn,target,source1] = rval[0,1]

temp = copy.deepcopy(CCdiffrecord)
temp[:,45:50,:] = 0.
temp[:,:,45:50] = 0.
x = np.argmax(temp)
n,a,b = np.unravel_index(x,(NP,ntarget,nsource1))
tname, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, a)
sname, scluster, snumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, b)
print('{}{} - {}{}'.format(tname,tcluster,sname,scluster))

CCdiffrecord[n,a,b]
plt.close(26)
fig = plt.figure(26), plt.plot(CCdiffrecord[:,a,b],'-xr')

CCdiffmean = np.mean(CCdiffrecord,axis=0)
a,b = (39,15)
plt.close(27)
fig = plt.figure(27), plt.plot(CCdiffrecord[:,a,b],'-xr')
tname, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, a)
sname, scluster, snumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, b)
print('{}{} - {}{}'.format(tname,tcluster,sname,scluster))


fitgrid = np.zeros((NP, ntarget, nsource1, nsource2, 7))
timepoint = 0
for nn in range(NP):
    print('person number {}   {}'.format(nn, time.ctime()))
    paradigm1 = np.array(list(paradigm_centered) * nruns_per_person[nn])
    tp = tplist_full[timepoint][nn]['tp']

    for target in range(ntarget):
        tname, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, target)
        tct = tcdata_centered[target, tp]

        for source1 in range(nsource1):
            sname1, scluster1, snumber1 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, source1)
            if snumber1 != tnumber:
                tc1 = tcdata_centered[source1, tp]
                for source2 in range(nsource2):
                    sname2, scluster2, snumber2 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, source2)
                    if snumber2 != tnumber and snumber2 != snumber1:

                        tc2 = tcdata_centered[source2, tp]
                        dtct = np.diff(tct)
                        dtc1 = np.diff(tc1)
                        dtc2 = np.diff(tc2)

                        # covariates needs to be nc x NP
                        sourcetc = np.concatenate((dtc1[:,np.newaxis],dtc2[:,np.newaxis]),axis=1)
                        targettc = dtct[:,np.newaxis]
                        b, bsem, R2, Z, Rcorrelation, Zcorrelation = py2ndlevelanalysis.GLMregression(targettc.T, sourcetc.T, axis=1)
                        T = b/bsem

                        fitgrid[nn, target, source1,source2,:] = [R2[0], b[0][0], b[0][1], b[0][2], T[0][0], T[0][1], T[0][2]]

g = copy.deepcopy(fitgrid)
g[:,45:50,:,:,:] = 0
g[:,:,45:50,:,:] = 0
g[:,:,:,45:50,:] = 0
x = np.argmax(g[:,:,:,:,4])
n,a,b,c = np.unravel_index(x,(NP,ntarget,nsource1,nsource2))
tname, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, a)
sname1, scluster1, snumber1 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, b)
sname2, scluster2, snumber2 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, c)

rval = fitgrid[n,a,b,c,0]
tval = fitgrid[n,a,b,c,4]
print('max T: person {}  target {}{}  sources {}{} and {}{}   R = {:.3f}  T = {:.3f}'.format(n,tname,tcluster,sname1,scluster1,sname2,scluster2,rval,tval))


# sort lists
g1 = np.where(covariates1 == 'Female')[0]
pg1 = covariates2[g1]
x = np.argsort(pg1)
g1s = g1[x]

g2 = np.where(covariates1 == 'Male')[0]
pg2 = covariates2[g2]
x = np.argsort(pg2)
g2s = g2[x]
gorder = np.concatenate(g1s,g2s)

# check results-------------------------------------------------------------------------
# split by sex---------------------------------
Tg1 = np.mean(fitgrid[g1,:,:,:,1],axis=0)/(np.std(fitgrid[g1,:,:,:,1],axis=0)/np.sqrt(len(g1)) + 1.0e-20)
Tg2 = np.mean(fitgrid[g2,:,:,:,1],axis=0)/(np.std(fitgrid[g2,:,:,:,1],axis=0)/np.sqrt(len(g2)) + 1.0e-20)

temp = copy.deepcopy(Tg1)
exclude = np.setdiff1d(list(range(ntarget)),range(5))
temp[exclude,:,:] = 0.
temp[:,45:50,:] = 0.
temp[:,:,45:50] = 0.
x = np.argmax(temp)
a,b,c = np.unravel_index(x,(ntarget,nsource1,nsource2))
tname, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, a)
sname1, scluster1, snumber1 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, b)
sname2, scluster2, snumber2 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, c)
tval = Tg1[a,b,c]
print('Female:  max T: target {}{}  sources {}{} and {}{}  T = {:.3f}'.format(tname,tcluster,sname1,scluster1,sname2,scluster2,tval))

x = np.argmin(temp)
a,b,c = np.unravel_index(x,(ntarget,nsource1,nsource2))
tname, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, a)
sname1, scluster1, snumber1 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, b)
sname2, scluster2, snumber2 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, c)
tval = Tg1[a,b,c]
print('Female:  min T: target {}{}  sources {}{} and {}{}  T = {:.3f}'.format(tname,tcluster,sname1,scluster1,sname2,scluster2,tval))


temp = copy.deepcopy(Tg2)
exclude = np.setdiff1d(list(range(ntarget)),range(5))
temp[exclude,:,:] = 0.
temp[:,45:50,:] = 0.
temp[:,:,45:50] = 0.
x = np.argmax(temp)
a,b,c = np.unravel_index(x,(ntarget,nsource1,nsource2))
tname, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, a)
sname1, scluster1, snumber1 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, b)
sname2, scluster2, snumber2 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, c)
tval = Tg2[a,b,c]
print('Male:  max T: target {}{}  sources {}{} and {}{}  T = {:.3f}'.format(tname,tcluster,sname1,scluster1,sname2,scluster2,tval))

x = np.argmin(temp)
a,b,c = np.unravel_index(x,(ntarget,nsource1,nsource2))
tname, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, a)
sname1, scluster1, snumber1 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, b)
sname2, scluster2, snumber2 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, c)
tval = Tg2[a,b,c]
print('Male:  min T: target {}{}  sources {}{} and {}{}  T = {:.3f}'.format(tname,tcluster,sname1,scluster1,sname2,scluster2,tval))






target = 3
source = 30   # 30 for NTS0,  27 for NRM2
tname, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, target)
sname1, scluster1, snumber1 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, source)
check = fitgrid[:,target,source,:,1]

plt.close(24)
fig = plt.figure(24), plt.imshow(check[gorder,:])
plt.close(25)
fig = plt.figure(25), plt.plot(check[gorder,20],'-xr')




meanR = np.mean(fitgrid[:,:,:,:,0],axis=0)
x = np.argmax(meanR)
a,b,c = np.unravel_index(x,(ntarget,nsource1,nsource2))
tname, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, a)
sname1, scluster1, snumber1 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, b)
sname2, scluster2, snumber2 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, c)
rval = meanR[a,b,c]
print('max mean correlation:  target {}{}  sources {}{} and {}{}   R = {:.3f}'.format(tname,tcluster,sname1,scluster1,sname2,scluster2,rval))

temp = copy.deepcopy(meanR)
for aa in range(50):
    x = np.argmax(temp)
    a,b,c = np.unravel_index(x,(ntarget,nsource1,nsource2))
    tname, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, a)
    sname1, scluster1, snumber1 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, b)
    sname2, scluster2, snumber2 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, c)
    rval = meanR[a, b, c]
    print('{} correlation:  target {}{}  sources {}{} and {}{}   R = {:.3f}'.format(
        aa,tname, tcluster, sname1,scluster1, sname2, scluster2, rval))
    # eliminate redundant results
    temp[a,:,:] = 0.0


temp = copy.deepcopy(fitgrid[:,:,:,:,0])
tt=1
tname, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, tt)

for nn in range(NP):
    x = np.argmax(temp[nn,tt,:,:])
    b,c = np.unravel_index(x,(nsource1,nsource2))
    sname1, scluster1, snumber1 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, b)
    sname2, scluster2, snumber2 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, c)
    rval = temp[nn,tt, b, c]

    print('{} correlation:  target {}{}  sources {}{} and {}{}   R = {:.3f}'.format(
        nn, tname, tcluster, sname1, scluster1, sname2, scluster2, rval))

