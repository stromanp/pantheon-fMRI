
import numpy as np
import load_templates
import os
import pandas as pd
import pysapm
import pysem
import time
import copy
import matplotlib.pyplot as plt

SAPMresultsdir = r'E:\SAPMresults2_Oct2022\binit_comparison'
namelist = ['Sens','Low','RSnostim','Pain','High','RSstim']
nametag = r'_0310013210_all_results'
runtype = ['', '2', '3', '4']

ngroups = len(namelist)
nruns = len(runtype)

# pick a data set
gn = 3
rn = 0


filename = namelist[gn] + nametag + runtype[rn] + '.npy'
SAPMresultsname = os.path.join(SAPMresultsdir, filename)
r = np.load(SAPMresultsname, allow_pickle = True)
NP = len(r)
betavals = np.array([r[x]['betavals'] for x in range(NP)])
bmean = np.mean(betavals,axis=0)
bstd = np.std(betavals,axis=0)
fintrinsic_base = r[0]['fintrinsic_base']

#-----------info from parameters-----------------------
nametag2 = r'_0310013210_all_params'
filename = namelist[0] + nametag2 + '.npy'
SAPMparametersname = os.path.join(SAPMresultsdir, filename)

# dict_keys(['betanamelist', 'beta_list', 'nruns_per_person', 'nclusterstotal', 'rnamelist',
#            'nregions', 'cluster_properties', 'cluster_data', 'network', 'fintrinsic_count',
#            'vintrinsic_count', 'fintrinsic_region', 'sem_region_list', 'nclusterlist', 'tsize',
#            'tplist_full', 'tcdata_centered', 'ctarget', 'csource', 'Mconn', 'Minput',
#            'timepoint', 'epoch', 'latent_flag'])

p = np.load(SAPMparametersname, allow_pickle=True).flat[0]
betanamelist = p['betanamelist']
beta_list = p['beta_list']
ctarget = p['ctarget']
csource = p['csource']
nregions = p['nregions']
rnamelist = p['rnamelist']
fintrinsic_count = p['fintrinsic_count']
vintrinsic_count = p['vintrinsic_count']

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

clusterlist = r[0]['clusterlist']

# xls = pd.ExcelFile(DBname, engine='openpyxl')
# df1 = pd.read_excel(xls, 'paradigm1_BOLD')
# del df1['Unnamed: 0']  # get rid of the unwanted header column
# fields = list(df1.keys())
# paradigm = df1['paradigms_BOLD']
# timevals = df1['time']
# paradigm_centered = paradigm - np.mean(paradigm)
# dparadigm = np.zeros(len(paradigm))
# dparadigm[1:] = np.diff(paradigm_centered)


# def sem_physio_model(clusterlist, fintrinsic_base, SAPMresultsname, SAPMparametersname, fixed_beta_vals = [], betascale = 0.0, verbose = True):

SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
# load the data values
betanamelist = SAPMparams['betanamelist']
beta_list = SAPMparams['beta_list']
nruns_per_person = SAPMparams['nruns_per_person']
nclusterstotal = SAPMparams['nclusterstotal']
rnamelist = SAPMparams['rnamelist']
nregions = SAPMparams['nregions']
cluster_properties = SAPMparams['cluster_properties']
cluster_data = SAPMparams['cluster_data']
network = SAPMparams['network']
fintrinsic_count = SAPMparams['fintrinsic_count']
vintrinsic_count = SAPMparams['vintrinsic_count']
sem_region_list = SAPMparams['sem_region_list']
nclusterlist = SAPMparams['nclusterlist']
tsize = SAPMparams['tsize']
tplist_full = SAPMparams['tplist_full']
tcdata_centered = SAPMparams['tcdata_centered']
ctarget = SAPMparams['ctarget']
csource = SAPMparams['csource']
fintrinsic_region = SAPMparams['fintrinsic_region']
Mconn = SAPMparams['Mconn']
Minput = SAPMparams['Minput']
timepoint = SAPMparams['timepoint']
epoch = SAPMparams['epoch']
latent_flag = SAPMparams['latent_flag']

ntime, NP = np.shape(tplist_full)
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
# repeat the process for one participant with a huge number of starting beta values-----------------------

repeat_betaval_record = []
nrepeat_checks = 4
for nrc in range(nrepeat_checks):
    windownum = 80+nrc
    best_betaval_record = []
    for nperson in range(NP):

        epochnum = 0
        tp = tplist_full[epochnum][nperson]['tp']
        tsize_total = len(tp)
        nruns = nruns_per_person[nperson]

        # get tc data for each region/cluster
        rnumlist = []
        clustercount = np.cumsum(nclusterlist)
        for aa in range(len(clusterlist)):
            x = np.where(clusterlist[aa] < clustercount)[0]
            rnumlist += [x[0]]

        Sinput = []
        for cval in clusterlist:
            tc1 = tcdata_centered[cval, tp]
            Sinput.append(tc1)
        Sinput = np.array(Sinput)
        # Sinput is size:  nregions x tsize_total

        # setup fixed intrinsic based on the model paradigm
        # need to account for timepoint and epoch....
        if fintrinsic_count > 0:
            if epoch >= tsize:
                et1 = 0
                et2 = tsize
            else:
                if np.floor(epoch / 2).astype(int) == np.ceil(epoch / 2).astype(int):  # even numbered epoch
                    et1 = (timepoint - np.floor(epoch / 2)).astype(int)
                    et2 = (timepoint + np.floor(epoch / 2)).astype(int)
                else:
                    et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
                    et2 = (timepoint + np.floor(epoch / 2)).astype(int)
            if et1 < 0: et1 = 0
            if et2 > tsize: et2 = tsize
            epoch = et2 - et1

            ftemp = fintrinsic_base[et1:et2]
            fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])
            if np.var(ftemp) > 1.0e-3:
                Sint = Sinput[fintrinsic_region,:]
                Sint = Sint - np.mean(Sint)
                # need to add constant to fit values
                G = np.concatenate((fintrinsic1[np.newaxis, :],np.ones((1,tsize_total))),axis=0)
                b, fit, R2, total_var, res_var = pysem.general_glm(Sint, G)
                beta_int1 = b[0]
            else:
                beta_int1 = 0.0
        else:
            beta_int1 = 0.0
            fintrinsic1 = []

        betascale = 0.5
        search_record = []
        Lweight = 1e-2

        nreps = 10000
        for rr in range(nreps):
            # initialize beta values at random values-----------------------------------
            betavals = betascale*np.random.randn(len(csource)) # initialize beta values at zero
            Mconn[ctarget,csource] = betavals
            fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
            cost = np.mean(np.abs(betavals))  # L1 regularization
            ssqd = err + Lweight * cost
            R2 = 1 - np.sum((Sinput-fit)**2)/np.sum(Sinput**2)
            search_record.append({'betavals':betavals, 'ssqd':ssqd, 'R2':R2})

        ssqd_list = np.array([search_record[x]['ssqd'] for x in range(nreps)])
        R2_list = np.array([search_record[x]['R2'] for x in range(nreps)])
        b_list = np.array([search_record[x]['betavals'] for x in range(nreps)])
        windownum1 = 10
        plt.close(windownum1)
        fig = plt.figure(windownum1)
        plt.hist(ssqd_list, 500)

        # check values
        x = ssqd_list.argmin()
        best_betavals = b_list[x,:]
        best_betaval_record.append({'best_betavals':best_betavals})
        print('{} best betavals gives ssqd = {:.2f}  R2 = {:.2f}'.format(nperson, ssqd_list[x], R2_list[x]))

        # compare with other values
        init_file1 = os.path.join(SAPMresultsdir, namelist[gn]+'_init1.npy')
        init_file2 = os.path.join(SAPMresultsdir, namelist[gn]+'_init2.npy')

        b = np.load(init_file1, allow_pickle=True).flat[0]
        binit1 = b['beta_initial']
        b = np.load(init_file2, allow_pickle=True).flat[0]
        binit2 = b['beta_initial']

        dist1 = [np.sqrt(np.sum( (binit1 - b_list[x,:])**20)) for x in range(nreps)]
        x = np.argmin(dist1)
        print('{} closest match to binit1 gives ssqd = {:.2f}'.format(nperson, ssqd_list[x]))

        dist2 = [np.sqrt(np.sum( (binit2 - b_list[x,:])**20)) for x in range(nreps)]
        x = np.argmin(dist2)
        print('{} closest match to binit2 gives ssqd = {:.2f}'.format(nperson, ssqd_list[x]))


    best_betavals = np.array([best_betaval_record[x]['best_betavals'] for x in range(NP)])
    bestb_avg = np.mean(best_betavals,axis=0)
    bestb_std = np.std(best_betavals,axis=0)

    plt.close(windownum)
    fig = plt.figure(windownum)
    fig.subplots_adjust(bottom=0.3)
    ax = fig.add_subplot()
    ind = np.arange(len(bestb_avg))
    plt.errorbar(connection_names,bestb_avg, bestb_std,linestyle='none',marker = 'o')
    plt.pause(0.1)
    locs, labels = plt.xticks()
    plt.xticks(locs,labels,rotation=90,fontsize=8)

    repeat_betaval_record.append({'best_betavals':best_betavals})