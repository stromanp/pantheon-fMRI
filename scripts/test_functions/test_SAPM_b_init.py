# sys.path.append(r'C:\Users\Stroman\PycharmProjects\pantheon\venv')

import numpy as np
import os
import pandas as pd
import time
import copy
import matplotlib.pyplot as plt
import pysapm

SAPMresultsdir = r'E:\SAPMresults2_Oct2022\binit_comparison'
namelist = ['Sens','Low','RSnostim','Pain','High','RSstim']
nametag = r'_0310013210_all_results'
runtype = ['', '2', '3', '4']

ngroups = len(namelist)
nruns = len(runtype)

rn = 3
betaval_data = []
for gn in range(ngroups):
    filename = namelist[gn] + nametag + runtype[rn] + '.npy'
    SAPMresultsname = os.path.join(SAPMresultsdir, filename)
    r = np.load(SAPMresultsname, allow_pickle = True)
    NP = len(r)
    betavals = np.array([r[x]['betavals'] for x in range(NP)])
    bmean = np.mean(betavals,axis=0)
    bstd = np.std(betavals,axis=0)
    betaval_data.append({'betavals':betavals, 'bmean':bmean, 'bstd':bstd})



#-----------info from parameters-----------------------
nametag2 = r'_0310013210_all_params'
filename = namelist[0] + nametag2 + '.npy'
SAPMparamsname = os.path.join(SAPMresultsdir, filename)

# dict_keys(['betanamelist', 'beta_list', 'nruns_per_person', 'nclusterstotal', 'rnamelist',
#            'nregions', 'cluster_properties', 'cluster_data', 'network', 'fintrinsic_count',
#            'vintrinsic_count', 'fintrinsic_region', 'sem_region_list', 'nclusterlist', 'tsize',
#            'tplist_full', 'tcdata_centered', 'ctarget', 'csource', 'Mconn', 'Minput',
#            'timepoint', 'epoch', 'latent_flag'])

p = np.load(SAPMparamsname, allow_pickle=True).flat[0]
betanamelist = p['betanamelist']
beta_list = p['beta_list']
ctarget = p['ctarget']
csource = p['csource']
nregions = p['nregions']
rnamelist = p['rnamelist']
fintrinsic_count = p['fintrinsic_count']
vintrinsic_count = p['vintrinsic_count']
latent_flag = p['latent_flag']
reciprocal_flag = p['reciprocal_flag']

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

# Mgrid0 = np.array([ [np.corrcoef(M1[x,0,:],M2[y,0,:])[0,1] for x in range(4)] for y in range(4)])
# cgrid = np.array([ [np.corrcoef(b1[x,:],b2[y,:])[0,1] for x in range(4)] for y in range(4)])

# plot betavals for different groups
windownum = 84
plt.close(windownum)
fig = plt.figure(windownum)
fig.subplots_adjust(bottom=0.3)
ax = fig.add_subplot()
ind = np.arange(len(betaval_data[0]['bmean']))
for nn in range(ngroups):
    plt.errorbar(connection_names,betaval_data[nn]['bmean'], betaval_data[0]['bstd'],linestyle='none',marker = 'o')
plt.pause(0.1)
locs, labels = plt.xticks()
plt.xticks(locs,labels,rotation=90,fontsize=8)



# look at different results for one person
person = 9
gn = 3

# dict_keys(['Sinput', 'Sconn', 'beta_int1', 'Mconn', 'Minput', 'R2total',
#            'Mintrinsic', 'Meigv', 'betavals', 'fintrinsic1', 'clusterlist',
#            'fintrinsic_base'])

betavals = []
R2total = []
Sinput = []
Sconn = []
Mconn =[]
Minput = []
beta_int1 = []
fintrinsic1 = []
fit = []
for rn in range(4):
    filename = namelist[gn] + nametag + runtype[rn] + '.npy'
    SAPMresultsname = os.path.join(SAPMresultsdir, filename)
    r = np.load(SAPMresultsname, allow_pickle = True)
    NP = len(r)
    betavals += [r[person]['betavals']]
    R2total += [r[person]['R2total']]
    Sinput += [r[person]['Sinput']]
    Sconn += [r[person]['Sconn']]
    Mconn += [r[person]['Mconn']]
    Minput += [r[person]['Minput']]
    beta_int1 += [r[person]['beta_int1']]
    fintrinsic1 += [r[person]['fintrinsic1']]

    fit_single, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(r[person]['Sinput'], r[person]['Minput'], r[person]['Mconn'], fintrinsic_count,
                                                                    vintrinsic_count, r[person]['beta_int1'], r[person]['fintrinsic1'])
    fit += [fit_single]

betavals = np.array(betavals)
R2total = np.array(R2total)
Sinput = np.array(Sinput)
Sconn = np.array(Sconn)
Mconn = np.array(Mconn)
Minput = np.array(Minput)
fintrinsic1 = np.array(fintrinsic1)
fit = np.array(fit)


# test code section
for nr in range(4):
    Nintrinsic = fintrinsic_count + vintrinsic_count
    e,v = np.linalg.eig(Mconn[nr,:,:])    # Mconn is nbeta x nbeta   where nbeta = ncon + Nintrinsic
    Meigv = np.real(v[:,-Nintrinsic:])
    # scale to make the term corresponding to each intrinsic = 1
    for aa in range(Nintrinsic):
        Meigv[:,aa] = Meigv[:,aa]/Meigv[(-Nintrinsic+aa),aa]
    M1 = Minput[nr,:,:] @ Meigv
    Mintrinsic = np.linalg.inv(M1.T @ M1) @ M1.T @ Sinput[nr,:,:]

    if fintrinsic_count > 0:
        Mintrinsic[0,:] = beta_int1[nr]*fintrinsic1[nr,:]

    fit = Minput[nr,:,:] @ Meigv @ Mintrinsic
    err = np.sum((Sinput[nr,:,:] - fit)**2)
    err0 = np.sum((Sinput[nr,0,:] - fit[0,:])**2)
    print('nr = {}   err = {:.2f}   err0 = {:.2f}   R2 = {:.3f}'.format(nr,err, err0, R2total[nr]))