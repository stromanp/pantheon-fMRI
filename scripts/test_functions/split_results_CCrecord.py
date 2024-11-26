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

CCsavedataname = r'D:\threat_safety_python\CCinvestigation_saved_data_pain.npy'
networksavename = r'D:\threat_safety_python\network_3level_trace_pain.npy'

CCsavedataname = r'D:\threat_safety_python\CCinvestigation_saved_data_sex.npy'
networksavename = r'D:\threat_safety_python\network_3level_trace_sex.npy'

split_by_pain = False
split_by_sex = True

reload_saved_data = False
if reload_saved_data:
    saveddata = np.load(CCsavedataname, allow_pickle=True).flat[0]
    data1 = saveddata['data1']
    covariates1 = saveddata['covariates1']
    covariates2 = saveddata['covariates2']
    region_data1 = saveddata['region_data1']
    beta_split = saveddata['beta_split']
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
beta2 = data1['beta2']
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
#
if not reload_saved_data:
    # trace networks with divisions based on beta values---------------------------------
    grouplimit = NP*groupsizelimit
    beta_split = np.zeros((ntarget,nsource1,nsource2,ntime,nc))
    for nt in range(ntime):
        for tt in range(ntarget):
            print('ntime = {}  ntarget = {}   {}'.format(nt,tt,time.ctime()))
            for s1 in range(nsource1):
                for s2 in range(nsource2):
                    for cc in range(nc):
                        b2test = beta2[tt, s1, s2, nt, :, cc]
                        g1 = np.where(b2test > 0)[0]
                        g2 = np.where(b2test <= 0)[0]
                        cov1 = cov4split[g1]
                        cov2 = cov4split[g2]
                        if len(g1) > grouplimit  and len(g2) > grouplimit:
                            T, p = stats.ttest_ind(cov1, cov2, equal_var=False)
                            beta_split[tt, s1, s2, nt, cc] = T

        saveddata = {'data1':data1, 'covariates1':covariates1, 'covariates2':covariates2, 'cov4split':cov4split, 'region_data1':region_data1, 'beta_split':beta_split}
        np.save(CCsavedataname,saveddata)


# find the significant connections and sort them in order of significance
x = np.where(np.abs(beta_split) > Tthresh_connection1)
nn,nv = np.shape(x)
Tlist = []
coordslist = []
for v in range(nv):
    target = x[0][v]
    source1 = x[1][v]
    source2 = x[2][v]
    timepoint = x[3][v]
    cc = x[4][v]
    T = beta_split[target,source1,source2,timepoint,cc]
    Tlist += [T]
    coordslist.append({'target':target, 'source1':source1, 'source2':source2 ,'timepoint':timepoint, 'cc':cc})
xsort = np.argsort(np.abs(Tlist))
print('Found {} connections with T value > {} for first connection'.format(len(xsort),Tthresh_connection1))

for x in xsort:
    entry = coordslist[x]
    T = Tlist[x]
    target,source1,source2,timepoint,cc = (entry['target'],entry['source1'],entry['source2'],entry['timepoint'],entry['cc'])
    # get the names of regions for writing the results-------------------------------------------------------
    tname, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, target)
    sname1, scluster1, snumber1 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, source1)
    sname2, scluster2, snumber2 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, source2)
    if len(tname) > 3:  tname = tname[:3]
    if len(sname1) > 3:  sname1 = sname1[:3]
    if len(sname2) > 3:  sname2 = sname2[:3]
    tagname = '{}{}_{}{}_{}{}_{}_{}_{}_{}_{}_'.format(tname, tcluster, sname1, scluster1, sname2, scluster2, target,
                                                      source1, source2, timepoint, cc)
    print('{}   T= {:.2f}'.format(tagname,T))

# keep a record of starting connections
xsort1 = xsort
Tlist1 = Tlist
coordslist1 = coordslist

network_list = []
for x1 in xsort1:
    # choose one connection
    entry = coordslist1[x1]
    target1, source1, source2, timepoint, cc = (entry['target'], entry['source1'], entry['source2'], entry['timepoint'], entry['cc'])
    tname1, tcluster1, tnumber1 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, target1)
    t1 = np.sum(nclusterlist[:tnumber1]).astype(int)
    t2 = np.sum(nclusterlist[:(tnumber1+1)])
    target1_exclude = list(range(t1,t2))
    b2test = beta2[target1, source1, source2, timepoint, :, cc]
    g1 = np.where(b2test > 0)[0]
    g2 = np.where(b2test <= 0)[0]
    cov1 = cov4split[g1]
    cov2 = cov4split[g2]
    Tc1, pc1 = stats.ttest_ind(cov1, cov2, equal_var=False)
    T1, p1 = stats.ttest_ind(b2test[g1], b2test[g2], equal_var=False)

    connection1 = entry

    #---------------next connection--------------------------------------
    # check inputs to relevant source
    if cc == 0:
        target2 = source1
    else:
        target2 = source2
    tname2, tcluster2, tnumber2 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, target2)
    t1 = np.sum(nclusterlist[:tnumber2]).astype(int)
    t2 = np.sum(nclusterlist[:(tnumber2+1)]).astype(int)
    target2_exclude = list(range(t1,t2))
    beta_split2 = np.zeros((nsource1,nsource2,2))
    for s1 in range(nsource1):
        for s2 in range(nsource2):
            for cc in range(2):
                b2test = beta2[target2, s1, s2, timepoint, :, cc]
                b1 = b2test[g1]
                b2 = b2test[g2]
                if np.var(b1) > 0 and np.var(b2) > 0:
                    T, p = stats.ttest_ind(b1, b2, equal_var=False)
                    beta_split2[s1,s2,cc] = T

    # exclude clusters in target1 or target2 regions
    target_exclude_list = target1_exclude + target2_exclude
    beta_mask = np.ones((nsource1,nsource2,2))
    for cc in range(2):
        beta_mask[target_exclude_list,:,cc] = 0
        beta_mask[:,target_exclude_list,cc] = 0

    # find the 2nd order connections and sort them in terms of signfiicance
    x2 = np.where(np.abs(beta_split2*beta_mask) > Tthresh_connection2)
    nn,nv = np.shape(x2)
    Tlist = []
    coordslist = []
    for v in range(nv):
        source1 = x2[0][v]
        source2 = x2[1][v]
        cc = x2[2][v]
        T = beta_split2[source1,source2,cc]
        Tlist += [T]
        coordslist.append({'target':target2, 'source1':source1, 'source2':source2 ,'timepoint':timepoint, 'cc':cc})
    xsort2 = np.argsort(np.abs(Tlist))
    Tlist2 = Tlist
    coordslist2 = coordslist
    print('Found {} connections with T value > {} for second connection'.format(len(xsort2),Tthresh_connection2))

    for x2 in xsort2:
        entry = coordslist2[x2]
        T2 = Tlist2[x2]
        target2,source1,source2,timepoint,cc = (entry['target'],entry['source1'],entry['source2'],entry['timepoint'],entry['cc'])
        # get the names of regions for writing the results-------------------------------------------------------
        tname, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, target2)
        sname1, scluster1, snumber1 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, source1)
        sname2, scluster2, snumber2 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, source2)
        if len(tname) > 3:  tname = tname[:3]
        if len(sname1) > 3:  sname1 = sname1[:3]
        if len(sname2) > 3:  sname2 = sname2[:3]
        tagname = '{}{}_{}{}_{}{}_{}_{}_{}_{}_{}_'.format(tname, tcluster, sname1, scluster1, sname2, scluster2, target,
                                                          source1, source2, timepoint, cc)
        # print('{}   T= {:.2f}'.format(tagname,T))

        # check values
        b2test = beta2[target2, source1, source2, timepoint, :, cc]
        b1 = b2test[g1]
        b2 = b2test[g2]
        T2, p2 = stats.ttest_ind(b1, b2, equal_var=False)
        cov1 = cov4split[g1]
        cov2 = cov4split[g2]
        Tc2, pc2 = stats.ttest_ind(cov1, cov2, equal_var=False)
        # print('T = {:.2f}   p = {:.3e}'.format(T2,p2))
        # print('Tc = {:.2f}   pc = {:.3e}'.format(Tc2,pc2))
        # print('b1: {:.2f} ({:.2f})  b2: {:.2f} ({:.2f})'.format(np.mean(b1),np.std(b1)/np.sqrt(len(b1)),np.mean(b2),np.std(b2)/np.sqrt(len(b2))))

        connection2 = entry

        #---------------insert for 3 connections=---------------------------------------
        # ---------------next connection - level 3 --------------------------------------
        # check inputs to relevant source
        if cc == 0:
            target3 = source1
        else:
            target3 = source2
        tname3, tcluster3, tnumber3 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, target3)
        t1 = np.sum(nclusterlist[:tnumber3]).astype(int)
        t2 = np.sum(nclusterlist[:(tnumber3 + 1)]).astype(int)
        target3_exclude = list(range(t1, t2))
        beta_split3 = np.zeros((nsource1, nsource2, 2))
        for s1 in range(nsource1):
            for s2 in range(nsource2):
                for cc in range(2):
                    b2test = beta2[target3, s1, s2, timepoint, :, cc]
                    b1 = b2test[g1]
                    b2 = b2test[g2]
                    if np.var(b1) > 0 and np.var(b2) > 0:
                        T, p = stats.ttest_ind(b1, b2, equal_var=False)
                        beta_split3[s1, s2, cc] = T

        # exclude clusters in target1 or target2 regions
        target_exclude_list = target1_exclude + target2_exclude + target3_exclude
        beta_mask = np.ones((nsource1, nsource2, 2))
        for cc in range(2):
            beta_mask[target_exclude_list, :, cc] = 0
            beta_mask[:, target_exclude_list, cc] = 0

        # find the 2nd order connections and sort them in terms of signfiicance
        x3 = np.where(np.abs(beta_split3 * beta_mask) > Tthresh_connection2)
        nn, nv = np.shape(x3)
        Tlist = []
        coordslist = []
        for v in range(nv):
            source1 = x3[0][v]
            source2 = x3[1][v]
            cc = x3[2][v]
            T = beta_split3[source1, source2, cc]
            Tlist += [T]
            coordslist.append(
                {'target': target3, 'source1': source1, 'source2': source2, 'timepoint': timepoint,
                 'cc': cc})
        xsort3 = np.argsort(np.abs(Tlist))
        Tlist3 = Tlist
        coordslist3 = coordslist
        print('Found {} connections with T value > {} for third connection'.format(len(xsort3), Tthresh_connection2))

        for x3 in xsort3:
            entry = coordslist3[x3]
            T3 = Tlist3[x3]
            target3, source1, source2, timepoint, cc = (entry['target'], entry['source1'], entry['source2'], entry['timepoint'], entry['cc'])
            # get the names of regions for writing the results-------------------------------------------------------
            tname, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, target3)
            sname1, scluster1, snumber1 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, source1)
            sname2, scluster2, snumber2 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, source2)
            if len(tname) > 3:  tname = tname[:3]
            if len(sname1) > 3:  sname1 = sname1[:3]
            if len(sname2) > 3:  sname2 = sname2[:3]
            tagname = '{}{}_{}{}_{}{}_{}_{}_{}_{}_{}_'.format(tname, tcluster, sname1, scluster1,
                                                              sname2, scluster2, target,
                                                              source1, source2, timepoint, cc)
            # print('{}   T= {:.2f}'.format(tagname,T))

            # check values
            b2test = beta2[target3, source1, source2, timepoint, :, cc]
            b1 = b2test[g1]
            b2 = b2test[g2]
            T3, p3 = stats.ttest_ind(b1, b2, equal_var=False)
            cov1 = cov4split[g1]
            cov2 = cov4split[g2]
            Tc3, pc3 = stats.ttest_ind(cov1, cov2, equal_var=False)
            # print('T = {:.2f}   p = {:.3e}'.format(T2,p2))
            # print('Tc = {:.2f}   pc = {:.3e}'.format(Tc2,pc2))
            # print('b1: {:.2f} ({:.2f})  b2: {:.2f} ({:.2f})'.format(np.mean(b1),np.std(b1)/np.sqrt(len(b1)),np.mean(b2),np.std(b2)/np.sqrt(len(b2))))

            connection3 = entry

            #--------------end of insert for 3 connections------------------------
            # network so far--------------------------------
            if cc == 0:
                source = source1
            else:
                source = source2

            tname1, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, target1)
            tname2, tcluster2, tnumber2 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, target2)
            tname3, tcluster3, tnumber3 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, target3)
            sname, scluster2, snumber2 = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, source)
            tagname = '{}{}_{}{}_{}{}_{}{}_time{}_T({:.2f})_T({:.2f})_T({:.2f})_Tc({:.2f})'.format(tname1, tcluster, tname2, tcluster2, tname3, tcluster3, sname, scluster2,timepoint,T1,T2,T3,Tc1)
            # print(tagname)

            entry = {'tname1':tname1,'tcluster1':tcluster,'tname2':tname2,'tcluster2':tcluster2,'tname3':tname3,'tcluster3':tcluster3,'source':sname,'scluster':scluster2,
                     'timepoint':timepoint, 'Tc1':Tc1,'pc1':pc1,'T1':T1,'p1':p1,'T2':T2,'p2':p2,'T3':T3,'p3':p3,'description':tagname, 'g1':g1, 'g2':g2,
                     'target1':target1, 'target2':target2, 'target3':target3, 'source':source,
                     'connection1':connection1, 'connection2':connection2, 'connection3':connection3}
            network_list.append(entry)


# eliminate redundant results
nvals = len(network_list)
network_list_short = []

keylist = {'tname1', 'tcluster1', 'tname2', 'tcluster2', 'tname3', 'tcluster3', 'source', 'scluster','timepoint'}
working_list = copy.deepcopy(network_list)

for nn in range(nvals):
    ee = working_list[nn]
    if not ee['description'] == 'removed':
        ee1 = {k:ee[k] for k in keylist}

        check = np.zeros(nvals)
        for mm in range(nvals):
            ee = working_list[mm]
            if not ee['description'] == 'removed':
                ee2 = {k:ee[k] for k in keylist}
                if ee1 == ee2: check[mm] = 1

        cc = np.where(check==1)[0]
        if len(cc) == 0:
            network_list_short.append(ee1)
        else:
            totalT = np.zeros(len(cc))
            for mm in range(len(cc)):
                ee = network_list[cc[mm]]
                totalT[mm] = np.abs(ee['T1']) + np.abs(ee['T2'])+ np.abs(ee['T3'])
            cc2 = np.argmax(totalT)
            cbest = cc[cc2]
            network_list_short.append(network_list[cbest])
        # remove tested values from the list
        working_list[nn]['description'] = 'removed'
        for ccc in cc: working_list[ccc]['description'] = 'removed'


np.save(networksavename, {'network_list':network_list,'network_list_short':network_list_short})


# plot results in bar charts

# first, find connections with the same base connection used for splitting groups
keylist = {'target', 'source1', 'source2', 'cc', 'timepoint'}
foundconn = np.zeros(len(network_list_short))
grouplist = []
timepointlist = []
for nn in range(len(network_list_short)):
    if foundconn[nn] == 0:
        foundconn[nn] = 1
        connection1 = network_list_short[nn]['connection1']
        timepoint = connection1['timepoint']
        timepointlist += [timepoint]
        basecon = {k: connection1[k] for k in keylist}

        foundnext = np.zeros(len(network_list_short))
        for mm in range(len(network_list_short)):
            if foundconn[mm] == 0:
                connection1b = network_list_short[mm]['connection1']
                nextcon = {k: connection1b[k] for k in keylist}
                if nextcon == basecon: foundnext[mm] = 1
        clist = [nn] + list(np.where(foundnext==1)[0])
        group = {'clist':clist, 'timepoint':timepoint}
        for c in clist:
            foundconn[c] = 1
        grouplist.append(group)


for tt in range(ntime):
    glist = [k for k in range(len(timepointlist)) if timepointlist[k] == tt]
    nplots = len(glist)
    fig = plt.figure(3+tt, figsize=(6, 9), dpi=100)

    for pcount, gg in enumerate(glist):
        ax = plt.subplot(nplots,1,pcount+1)
        clist = grouplist[gg]['clist']
        plotrecord = []
        connid = []
        ppos_list = []
        boxcount = 0
        plotlabellist = []
        for nn in clist:
            connlist = []
            connlist += [network_list_short[nn]['connection1']]
            connlist += [network_list_short[nn]['connection2']]
            connlist += [network_list_short[nn]['connection3']]
            g1 = network_list_short[nn]['g1']
            g2 = network_list_short[nn]['g2']
            Tc = network_list_short[nn]['Tc1']

            for netstep in range(3):
                con = connlist[netstep]
                target, source1, source2, timepoint, cc = (con['target'], con['source1'], con['source2'], con['timepoint'], con['cc'])
                b2test = beta2[target, source1, source2, timepoint, :, cc]
                if Tc > 0:
                    b1 = b2test[g1]
                    b2 = b2test[g2]
                    cov1 = cov4split[g1]
                    cov2 = cov4split[g2]
                else:
                    b1 = b2test[g2]   # swap groups so that high value covariate group is always group1
                    b2 = b2test[g1]
                    cov1 = cov4split[g2]
                    cov2 = cov4split[g1]

                if cc == 0:
                    source = source1
                else:
                    source = source2
                tname, tcluster, tnumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, target)
                sname, scluster, snumber = py2ndlevelanalysis.get_cluster_info(rnamelist, nclusterlist, source)
                if len(tname) > 3: tname = tname[:4]
                if len(sname) > 3: sname = sname[:4]

                plotlabel = '{}{}-->{}{}'.format(sname,scluster,tname,tcluster)

                if boxcount == 0:   # show the covariates first
                    boxcount += 1
                    ppos = (boxcount - 1) * 3 + 1
                    maxval = np.max(np.abs(np.concatenate((cov1, cov2))))
                    if maxval > 5:
                        f = np.ceil((np.log10(maxval))).astype(int)
                        scale = 10**f
                        addlabel = 'x{}'.format(scale)
                    else:
                        addlabel = ''
                        scale = 1

                    onecat = [cov1/scale, cov2/scale]
                    bp = ax.boxplot(onecat, positions=[ppos, ppos + 1], widths=0.6, notch=True, showfliers=False)
                    setBoxColors(bp)
                    ppos_list.append(ppos + 0.5)
                    plotlabellist = [covariatelabel + addlabel]


                # keep a record of the connections already plotted (do not duplicate)
                id = timepoint*1e5 + target*1e4 + source1*1e2 + source2
                if not id in connid:
                    boxcount += 1
                    ppos = (boxcount - 1) * 3 + 1
                    onecat = [b1, b2]
                    bp = ax.boxplot(onecat, positions=[ppos, ppos + 1], widths=0.6, notch=True, showfliers=False)
                    setBoxColors(bp)
                    ppos_list.append(ppos + 0.5)
                    plotlabellist += [plotlabel]

                    connid += [id]

        labelfont = 6
        ax.set_xticks(ppos_list)
        ax.set_xticklabels(plotlabellist, rotation=0, fontsize=labelfont)
        plt.tight_layout()


