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


write_figures = False
outputdir = r'D:/threat_safety_python/SEMresults'
# setup connections to look at
target, source1, source2, timepoint, nc = (31,36,40,1,0)    # look for inputs to NTS1 now (other side of NTS)
target, source1, source2, timepoint, nc = (4,30,37,0,0)     # C6RD target, NTS0 source1
target, source1, source2, timepoint, nc = (31,36,45,0,0)    # look for inputs to NTS1 now (other side of NTS)

target, source1, source2, timepoint, nc = (4,30,37,0,0)     # C6RD target, NTS0 source1
target, source1, source2, timepoint, nc = (4,30,37,1,0)     # C6RD target, NTS0 source1 repeat during other period

target, source1, source2, timepoint, nc = (30,38,49,1,0)    # look for inputs to NTS0 from PAG4
target, source1, source2, timepoint, nc = (30,38,49,0,0)    # look for inputs to NTS0 from PAG4, during other period

target, source1, source2, timepoint, nc = (43,23,30,1,1)    # look for PBN3 target, NTS0 source1
target, source1, source2, timepoint, nc = (43,23,30,0,1)    # look for PBN3 target, NTS0 source1, during other period

target, source1, source2, timepoint, nc = (12,39,43,1,1)   # hypothalamus2 target from PBN3 source
target, source1, source2, timepoint, nc = (12,39,43,0,1)   # hypothalamus2 target from PBN3 source, during other period
target, source1, source2, timepoint, nc = (12,43,45,0,0)   # hypothalamus2 target from PBN3 source, during other period


target, source1, source2, timepoint, nc = (41, 4, 8, 0, 0)     # PBN1 target, C6RD4 source1, DRt3 source2, before stim

plt.close(2)
plt.close(3)
plt.close(4)
plt.close(5)
plt.close(6)
plt.close(7)
#-----------------------------------------------------------------------------------------
networkfile = r'D:/threat_safety_python/network_possible_connections.xlsx'
network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkfile)

CCsavedataname = r'D:\threat_safety_python\CCinvestigation_saved_data.npy'
reload_saved_data = True
if reload_saved_data:
    saveddata = np.load(CCsavedataname, allow_pickle=True).flat[0]
    data1 = saveddata['data1']
    covariates1 = saveddata['covariates1']
    covariates2 = saveddata['covariates2']
    region_data1 = saveddata['region_data1']
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

    saveddata = {'data1':data1, 'covariates1':covariates1, 'covariates2':covariates2, 'region_data1':region_data1}
    np.save(CCsavedataname,saveddata)

region_properties = region_data1['region_properties']
# # dict_keys(['tc', 'tc_sem', 'nruns_per_person', 'tsize'])
cluster_properties = data1['cluster_properties']

# setup lists
NP = len(covariates1)
g1 = np.where(covariates1 == 'Female')[0]
g2 = np.where(covariates1 == 'Male')[0]
groupnames = ['Female', 'Male']

# over-ride this
beta2 = data1['beta2']
b2test = beta2[target,source1,source2,timepoint,:,nc]
g1 = np.where(b2test > 0)[0]
g2 = np.where(b2test <= 0)[0]
groupnames = ['pos_beta', 'neg_beta']


# for target, source1, source2, timepoint, nc = (34,24,28,1,0)     # NTS0 target, NGC4 source1, NRM3 source2, during stim
# np.mean(covariates2[g1xx]), np.std(covariates2[g1xx])/np.sqrt(len(g1xx))
# Out[224]: (50.0390625, 1.5568691263045937)
# np.mean(covariates2[g2xx]), np.std(covariates2[g2xx])/np.sqrt(len(g2xx))
# Out[225]: (45.85434782608696, 2.576516957404106)


# exclude = [16, 42]    # exclude values for unusually high variance, and unbelievable pain ratings
exclude = []
gx = np.setdiff1d(list(range(NP)),exclude)
g1x = np.setdiff1d(g1,exclude)
g2x = np.setdiff1d(g2,exclude)

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


#-------load results-----------------------------------------------------------------------------
# CCrecord is the variance/covariance grid
CCrecord = data1['CCrecord']   # size is ntime x NP x nregions x nregions
ntime, NP, ntargets, nsources = np.shape(CCrecord)
beta2 = data1['beta2']
Zgrid2 = data1['Zgrid2']
Zgrid1_1 = data1['Zgrid1_1']
Zgrid1_2 = data1['Zgrid1_2']

# to here, the data needed are loaded ...
# beta2    ntarget x nsource1 x nsource2 x ntime x NP x nc
# CCrecord  ntime x NP x nsource1 x nsource2
# Zgrid2  ntarget x nsource1 x nsource2 x ntime x NP


# get the names of regions for writing the results-------------------------------------------------------
tname, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, target)
sname1, scluster1, snumber1 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, source1)
sname2, scluster2, snumber2 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, source2)
if len(tname) > 3:  tname = tname[:3]
if len(sname1) > 3:  sname1 = sname1[:3]
if len(sname2) > 3:  sname2 = sname2[:3]
tagname = '{}{}_{}{}_{}{}_{}_{}_{}_{}_{}_'.format(tname,tcluster,sname1,scluster1,sname2,scluster2,target, source1, source2, timepoint, nc)
print(tagname)

if nc == 0:
    # focus on source1 only if nc == 0
    Z1 = Zgrid2[target, source1,:,timepoint,:]    # find which source2 gives the best fit, for target and source1
    Z2 = Zgrid2[target, :, source1,timepoint,:]    # find which source2 gives the best fit, for target and source1
    Z = Z1  # values at source1 and higher
    Z[:source1,:] = Z2[:source1,:]  # values lower than source1
    # Z[:,15] = 0.   # exclude person 15

    b1 = beta2[target,source1,:,timepoint,:,0]
    b2 = beta2[target,:,source1,timepoint,:,1]
    b = b1  # values at source1 and higher
    b[:source1,:] = b2[:source1,:]  # values lower than source1
    # b[:,15] = 0.   # exclude person 15
    bsource1 = b

    b1 = beta2[target,source1,:,timepoint,:,1]
    b2 = beta2[target,:,source1,timepoint,:,0]
    b = b1  # values at source1 and higher
    b[:source1,:] = b2[:source1,:]  # values lower than source1
    # b[:,15] = 0.   # exclude person 15
    bsource2 = b
else:
    # focus on source2 only if nc == 1
    Z1 = Zgrid2[target, :, source2,timepoint,:]    # find which source2 gives the best fit, for target and source1
    Z2 = Zgrid2[target, source2, :,timepoint,:]    # find which source2 gives the best fit, for target and source1
    Z = Z2  # values at source2 and higher
    Z[:source2,:] = Z1[:source2,:]  # values lower than source2
    # Z[:,15] = 0.   # exclude person 15

    b1 = beta2[target,:, source2,timepoint,:,0]
    b2 = beta2[target,source2,:,timepoint,:,1]
    b = b2  # values at source2 and higher
    b[:source2,:] = b1[:source2,:]  # values lower than source1
    # b[:,15] = 0.   # exclude person 15
    bsource1 = b

    b1 = beta2[target,:,source2,timepoint,:,1]
    b2 = beta2[target,source2,:,timepoint,:,0]
    b = b2  # values at source2 and higher
    b[:source2,:] = b1[:source2,:]  # values lower than source2
    # b[:,15] = 0.   # exclude person 15
    bsource2 = b



# check the correlation between Z and covariate as a function of source2---------------------------------------------
Rcheck = np.zeros(50)
RcheckG1 = np.zeros(50)
RcheckG2 = np.zeros(50)

RcheckB = np.zeros(50)
RcheckBG1 = np.zeros(50)
RcheckBG2 = np.zeros(50)
for nn in range(50):
    if np.var(Z[nn,:]) > 0:
        r = np.corrcoef(Z[nn,gx],covariates2[gx])
        Rcheck[nn] = r[0,1]
        r = np.corrcoef(Z[nn,g1x],covariates2[g1x])
        RcheckG1[nn] = r[0,1]
        r = np.corrcoef(Z[nn,g2x],covariates2[g2x])
        RcheckG2[nn] = r[0,1]

        if nc == 0:
            r = np.corrcoef(bsource1[nn,gx],covariates2[gx])
            RcheckB[nn] = r[0,1]
            r = np.corrcoef(bsource1[nn,g1x],covariates2[g1x])
            RcheckBG1[nn] = r[0,1]
            r = np.corrcoef(bsource1[nn,g2x],covariates2[g2x])
            RcheckBG2[nn] = r[0,1]
        else:
            r = np.corrcoef(bsource2[nn,gx],covariates2[gx])
            RcheckB[nn] = r[0,1]
            r = np.corrcoef(bsource2[nn,g1x],covariates2[g1x])
            RcheckBG1[nn] = r[0,1]
            r = np.corrcoef(bsource2[nn,g2x],covariates2[g2x])
            RcheckBG2[nn] = r[0,1]

if nc == 0:
    figname = 'ZcorrCov_vS2'
else:
    figname = 'ZcorrCov_vS1'
fig = plt.figure(2, figsize=(10, 8), dpi=100)
plt.plot(Rcheck,'-xr')
plt.plot(RcheckG1,'-',color=[0.7,0.5,0])
plt.plot(RcheckG2,'-',color=[0.3,0.7,0])

# check the average Z as a function of source2
Acheck = np.mean(Z,axis=1)
plt.plot(Acheck,'-ob')
plt.suptitle(tagname+figname)
if write_figures:
    outputname = os.path.join(outputdir,tagname+figname+'.eps')
    plt.savefig(outputname, format='eps')

if nc == 0:
    figname = 'BcorrCov_vS2'
else:
    figname = 'BcorrCov_vS1'
fig = plt.figure(3, figsize=(10, 8), dpi=100)
plt.plot(RcheckB,'-or')
plt.plot(RcheckBG1,'-',color=[0.7,0.5,0])
plt.plot(RcheckBG2,'-',color=[0.3,0.7,0])

# check the average b as a function of source2
if nc == 0:
    Acheck = np.mean(bsource1,axis=1)
else:
    Acheck = np.mean(bsource2,axis=1)
plt.plot(Acheck,'-ob')

plt.suptitle(tagname+figname)
if write_figures:
    outputname = os.path.join(outputdir,tagname+figname+'.eps')
    plt.savefig(outputname, format='eps')


# check how each source contributes to the fit-------------------------------------------------
Rts1_tc = np.zeros(NP)
Rts2_tc = np.zeros(NP)
R2s1s2_tc = np.zeros(NP)
Rtotal_tc = np.zeros(NP)
Rts1_cc = np.zeros(NP)
Rts2_cc = np.zeros(NP)
R2s1s2_cc = np.zeros(NP)
Rtotal_cc = np.zeros(NP)

# check values across people
for nn in range(NP):
    # from CCrecord....
    Ct1 = CCrecord[timepoint,nn,target,source1]
    Ct2 = CCrecord[timepoint,nn,target,source2]
    C12 = CCrecord[timepoint,nn,source1,source2]
    Vt = CCrecord[timepoint,nn,target,target]
    V1 = CCrecord[timepoint,nn,source1,source1]
    V2 = CCrecord[timepoint,nn,source2,source2]
    b1 = beta2[target,source1,source2,timepoint,nn,0]
    b2 = beta2[target,source1,source2,timepoint,nn,1]
    b2calc = beta2[target,source1,source2,timepoint,nn,:]

    R2ts1 = 2 * b2calc[0] * Ct1 / Vt - b2calc[0] ** 2 * V1 / Vt
    R2ts2 = 2 * b2calc[1] * Ct2 / Vt - b2calc[1] ** 2 * V2 / Vt
    R2s1s2 = -2 * b2calc[0] * b2calc[1] * C12 / Vt
    R2check = R2ts1 + R2ts2 + R2s1s2
    print('based on CCrecord:  R2ts1 = {:.3f}  R2ts2 = {:.3f}  R2s1s2 = {:.3f}  R2total = {:.3f}'.format(R2ts1,R2ts2,R2s1s2,R2check))
    Rts1_cc[nn] = R2ts1
    Rts2_cc[nn] = R2ts2
    R2s1s2_cc[nn] = R2s1s2
    Rtotal_cc[nn] = R2check

    # from timecourse data ....
    # need the timecourse data for the fit
    tp = tplist[timepoint][nn]['tp']
    tct = tcdata_centered[target, tp]
    tc1 = tcdata_centered[source1, tp]
    tc2 = tcdata_centered[source2, tp]
    b2calc = beta2[target,source1,source2,timepoint,nn,:]

    fit = b2calc[0] * tc1 + b2calc[1] * tc2
    R2 = 1 - np.sum((tct - fit) ** 2) / np.sum(tct ** 2)

    cc = np.cov(tct, tc1)
    Ct1 = cc[0, 1]
    Vt = cc[0, 0]
    V1 = cc[1, 1]
    cc = np.cov(tct, tc2)
    Ct2 = cc[0, 1]
    V2 = cc[1, 1]
    C12 = np.cov(tc1, tc2)[0, 1]

    b1 = (Ct1 - Ct2 * C12 / V2) / (V1 - (C12 ** 2) / V2)
    b2 = (Ct2 - Ct1 * C12 / V1) / (V2 - (C12 ** 2) / V1)
    fit_check = b1 * tc1 + b2 * tc2

    R2ts1 = 2 * b1 * Ct1 / Vt - b1 ** 2 * V1 / Vt
    R2ts2 = 2 * b2 * Ct2 / Vt - b2 ** 2 * V2 / Vt
    R2s1s2 = -2 * b1 * b2 * C12 / Vt
    R2check = R2ts1 + R2ts2 + R2s1s2
    print('based on tc data:  R2ts1 = {:.3f}  R2ts2 = {:.3f}  R2s1s2 = {:.3f}  R2total = {:.3f}'.format(R2ts1,R2ts2,R2s1s2,R2check))
    Rts1_tc[nn] = R2ts1
    Rts2_tc[nn] = R2ts2
    R2s1s2_tc[nn] = R2s1s2
    Rtotal_tc[nn] = R2check


figname = 'Rvals_per_person'
fig = plt.figure(4, figsize=(10, 8), dpi=100)
plt.subplot(3,2,1)
plt.plot(covariates2[g1x],Rtotal_tc[g1x],'o',color = [1,0,0])
plt.plot(covariates2[g2x],Rtotal_tc[g2x],'o',color = [0,1,0])
plt.subplot(3,2,2)
plt.plot(covariates2[g1x],Rts1_tc[g1x],'x',color=[1,0,0])
plt.plot(covariates2[g2x],Rts1_tc[g2x],'x',color=[0,1,0])
plt.subplot(3,2,3)
plt.plot(covariates2[g1x],Rts2_tc[g1x],'x',color=[1,0,0.5])
plt.plot(covariates2[g2x],Rts2_tc[g2x],'x',color=[0.5,0,1])
plt.subplot(3,2,4)
plt.plot(covariates2[g1x],R2s1s2_tc[g1x],'*',color=[1,0,0])
plt.plot(covariates2[g2x],R2s1s2_tc[g2x],'*',color=[0,1,0])
plt.subplot(3,2,5)
plt.plot(Rts1_tc[g1x],Rts2_tc[g1x],'o',color=[1,0,0])
plt.plot(Rts1_tc[g2x],Rts2_tc[g2x],'o',color=[0,1,0])

plt.suptitle(tagname+figname)
if write_figures:
    outputname = os.path.join(outputdir,tagname+figname+'.eps')
    plt.savefig(outputname, format='eps')


figname = 'R1_vs_R2_vs_Cov_per_person'
fig = plt.figure(5, figsize=(10, 8), dpi=100)
ax = plt.axes(projection = '3d')
ax.plot3D(Rts1_tc[g1x],Rts2_tc[g1x], covariates2[g1x],'o',color=[1,0,0])
ax.plot3D(Rts1_tc[g2x],Rts2_tc[g2x], covariates2[g2x],'o',color=[0,1,0])
ax.view_init(60, 35)
plt.suptitle(tagname+figname)
if write_figures:
    outputname = os.path.join(outputdir,tagname+figname+'.eps')
    plt.savefig(outputname, format='eps')


# now check how the beta values vary with pain ratings, and between genders--------------------------
figname = 'BetavCov'
# b2calc = beta2[target,source1,source2,timepoint,:,:]
fig = plt.figure(6, figsize=(10, 8), dpi=100)

plt.subplot(2,2,1)
# b1 = b2calc[:,0]
# b2 = b2calc[:,1]
if nc == 0:
    b1 = bsource1[source2,:]
    b2 = bsource2[source2,:]
else:
    b1 = bsource1[source1,:]
    b2 = bsource2[source1,:]
check1 = np.where(np.abs(b1)>2)[0]
check2 = np.where(np.abs(b2)>2)[0]
gxx = gx
g1xx = g1x
g2xx = g2x
if len(check1) > 0:
    gxx = np.setdiff1d(gx,check1)
    g1xx = np.setdiff1d(g1x,check1)
    g2xx = np.setdiff1d(g2x,check1)
if len(check2) > 0:
    gxx = np.setdiff1d(gx,check2)
    g1xx = np.setdiff1d(g1x,check2)
    g2xx = np.setdiff1d(g2x,check2)

p1,fit1,R21 = pydisplay.simple_GLMfit(covariates2[gxx], b1[gxx])
p2,fit2,R22 = pydisplay.simple_GLMfit(covariates2[gxx], b2[gxx])
if nc == 0:
    plt.plot(covariates2[gxx],b1[gxx],'o',color = [1,0,0])
    plt.plot(covariates2[gxx],fit1,'-',color = [1,0,0])
else:
    plt.plot(covariates2[gxx],b2[gxx],'o',color = [0,0,1])
    plt.plot(covariates2[gxx],fit2,'-',color = [0,0,1])
R2text = 'Group: R2 = {:.3f}'.format(R21)
x = np.min(covariates2[gxx])
if p1[0] > 0:
    if nc == 0:
        y = np.min(b1[gxx])
    else:
        y = np.min(b2[gxx])
else:
    if nc == 0:
        y = np.max(b1[gxx])
    else:
        y = np.max(b2[gxx])
plt.text(x,y,R2text,color = [1,0,0])
print('fit to b1:  R2 = {:.3f}      fit to b2:  R2 = {:.3f}'.format(R21,R22))

plt.subplot(2,2,2)
p1,fit1,R21 = pydisplay.simple_GLMfit(covariates2[g1xx], b1[g1xx])
p2,fit2,R22 = pydisplay.simple_GLMfit(covariates2[g1xx], b2[g1xx])

if nc == 0:
    plt.plot(covariates2[g1xx],b1[g1xx],'o',color = [1,0,0])
    plt.plot(covariates2[g1xx],fit1,'-',color = [1,0,0])
    R2text = '{}: R2 = {:.3f}'.format(groupnames[0],R21)
else:
    plt.plot(covariates2[g1xx],b2[g1xx],'o',color = [0,0,1])
    plt.plot(covariates2[g1xx],fit2,'-',color = [0,0,1])
    R2text = '{}: R2 = {:.3f}'.format(groupnames[0],R22)
x = np.min(covariates2[g1xx])
if p1[0] > 0:
    y = np.min(b1[g1xx])
else:
    y = np.max(b1[g1xx])
plt.text(x,y,R2text,color = [1,0,0])
print('{}:   fit to b1:  R2 = {:.3f}      fit to b2:  R2 = {:.3f}'.format(groupnames[0],R21,R22))

plt.subplot(2,2,3)
p1,fit1,R21 = pydisplay.simple_GLMfit(covariates2[g2xx], b1[g2xx])
p2,fit2,R22 = pydisplay.simple_GLMfit(covariates2[g2xx], b2[g2xx])

if nc == 0:
    plt.plot(covariates2[g2xx],b1[g2xx],'o',color = [1,0,0])
    plt.plot(covariates2[g2xx],fit1,'-',color = [1,0,0])
    R2text = '{}: R2 = {:.3f}'.format(groupnames[1],R21)
else:
    plt.plot(covariates2[g2xx],b2[g2xx],'o',color = [0,0,1])
    plt.plot(covariates2[g2xx],fit2,'-',color = [0,0,1])
    R2text = '{}: R2 = {:.3f}'.format(groupnames[1],R22)
x = np.min(covariates2[g2xx])
if p1[0] > 0:
    y = np.min(b1[g2xx])
else:
    y = np.max(b1[g2xx])
plt.text(x,y,R2text,color = [1,0,0])
print('{}:   fit to b1:  R2 = {:.3f}      fit to b2:  R2 = {:.3f}'.format(groupnames[1],R21,R22))

# compare stats values with regular analysis
# correlation----------------------------------------------------------

pthreshold = 5.2e-5
Zthresh = stats.norm.ppf(1-pthreshold)
b = beta2[target,source1,source2,timepoint,:,nc]
# entire group
r = np.corrcoef(b[gxx],covariates2[gxx])
R = r[0,1]
Z = np.arctanh(R)*np.sqrt(len(gxx)-3)
p = 1-stats.norm.cdf(np.abs(Z))
# group1
r = np.corrcoef(b[g1xx],covariates2[g1xx])
Rf = r[0,1]
Zf = np.arctanh(Rf)*np.sqrt(len(g1xx)-3)
pf = 1-stats.norm.cdf(np.abs(Zf))
# group2
r = np.corrcoef(b[g2xx],covariates2[g2xx])
Rm = r[0,1]
Zm = np.arctanh(Rm)*np.sqrt(len(g2xx)-3)
pm = 1-stats.norm.cdf(np.abs(Zm))

# compare beta values
bfmean = np.mean(b[g1xx])
bfsem = np.std(b[g1xx])/np.sqrt(len(g1xx))
bmmean = np.mean(b[g2xx])
bmsem = np.std(b[g2xx])/np.sqrt(len(g2xx))
T,p = stats.ttest_ind(b[g1xx],b[g2xx])

# compare covariate values
cfmean = np.mean(covariates2[g1xx])
cfsem = np.std(covariates2[g1xx])/np.sqrt(len(g1xx))
cmmean = np.mean(covariates2[g2xx])
cmsem = np.std(covariates2[g2xx])/np.sqrt(len(g2xx))
Tc,pc = stats.ttest_ind(covariates2[g1xx],covariates2[g2xx])


text1 = 'whole group:  R = {:.3f}  Z = {:.2f}   p = {:.3e}'.format(R,Z,p)
text2 = '{}:  R = {:.3f}  Z = {:.2f}   p = {:.3e}'.format(groupnames[0],Rf,Zf,pf)
text3 = '{}:  R = {:.3f}  Z = {:.2f}   p = {:.3e}'.format(groupnames[1],Rm,Zm,pm)
text4 = 'Zthreshold for p = {:.3e} is {:.2f}'.format(pthreshold,Zthresh)
text4b = 'beta {}: {:.2f} ({:.2f})  {}: {:.2f} ({:.2f}) p = {:.2e}'.format(groupnames[0][0],bfmean,bfsem,groupnames[1][0],bmmean,bmsem,p)
text4c = 'cov {}: {:.2f} ({:.2f})  {}: {:.2f} ({:.2f}) p = {:.2e}'.format(groupnames[0][0],cfmean,cfsem,groupnames[1][0],cmmean,cmsem,pc)

print(text1)
print(text2)
print(text3)
print(text4)
print(text4b)
print(text4c)

# ANCOVA ---------------------------------------------------
beta = b[gxx]
group = covariates1[gxx]
cov = covariates2[gxx]

atype = 2

d = {'beta': beta, 'Group': group, 'painrating': cov}
df = pd.DataFrame(data=d)

formula = 'beta ~ C(Group) + painrating + C(Group):painrating'

model = ols(formula, data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=atype)

p_MeoG = anova_table['PR(>F)']['C(Group)']
p_MeoC = anova_table['PR(>F)']['painrating']
p_intGC = anova_table['PR(>F)']['C(Group):painrating']
text5 = 'p_MeoG = {:.2e}  p_MeoC = {:.2e}  p_intGC = {:.2e}'.format(p_MeoG,p_MeoC,p_intGC)
print(text5)

#------make a place to hold some text
plt.subplot(2,2,4)
plt.plot([0,10],[0,10],'xr')
fs=7
plt.text(0.01,7,text1, fontsize=fs)
plt.text(0.01,6,text2, fontsize=fs)
plt.text(0.01,5,text3, fontsize=fs)
plt.text(0.01,4,text4, fontsize=fs)
plt.text(0.01,3,text4b, fontsize=fs)
plt.text(0.01,2,text4c, fontsize=fs)
plt.text(0.01,1,text5, fontsize=fs)

plt.suptitle(tagname+figname)
if write_figures:
    outputname = os.path.join(outputdir,tagname+figname+'.eps')
    plt.savefig(outputname, format='eps')


# check on CCrecord-----------------------------------------------

figname = 'CCrecord_for_target'
b2calc = beta2[target,source1,source2,timepoint,:,:]
fig = plt.figure(7, figsize=(10, 8), dpi=100)

ntime,NP,nregions1,nregions2 = np.shape(CCrecord)
CCcorrM = np.zeros((ntime,nregions1,nregions2))
CCcorrF = np.zeros((ntime,nregions1,nregions2))
for tt in range(ntime):
    for s1 in range(nregions1):
        for s2 in range(nregions2):
            c = CCrecord[tt,:,s1,s2]
            r = np.corrcoef(c[g1xx],covariates2[g1xx])
            CCcorrF[tt,s1,s2] = r[0,1]
            r = np.corrcoef(c[g2xx],covariates2[g2xx])
            CCcorrM[tt,s1,s2] = r[0,1]

CCmeanF = np.mean(CCrecord[:,g1xx,:,:],axis=1)
CCmeanM = np.mean(CCrecord[:,g2xx,:,:],axis=1)
for s1 in range(nregions1):  # mask out the variance values
    CCmeanF[:,s1,s1] = 0.
    CCmeanM[:,s1,s1] = 0.
# plot for selected target region

plt.plot(CCcorrF[timepoint,target,:], '-', color=[1, 0, 0])
plt.plot(CCcorrM[timepoint,target,:], '-', color=[0, 0, 1])
plt.plot(CCmeanF[timepoint,target,:], 'o-', color=[1, 0, 0])
plt.plot(CCmeanM[timepoint,target,:], 'o-', color=[0, 0, 1])

plt.suptitle(tagname+figname)
if write_figures:
    outputname = os.path.join(outputdir,tagname+figname+'.eps')
    plt.savefig(outputname, format='eps')