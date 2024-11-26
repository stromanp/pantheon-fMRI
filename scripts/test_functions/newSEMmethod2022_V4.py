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
from sklearn.decomposition import PCA
import scipy.stats as stats
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
from mpl_toolkits import mplot3d


def sub2ind(vsize, indices):
    # give all the combinations for the values that are allowed to vary
    ndims = len(vsize)
    w = 0
    for nn in range(ndims):
        w += indices[nn]*np.prod(vsize[:nn])
    return w

def all_flat_indices_nfixed(vsize, fixedindices, fixedvals):
    # give all the combinations for the values that are allowed to vary
    ndims = len(vsize)
    vsize2 = copy.deepcopy(vsize)
    vsize2[fixedindices] = 1
    nc = np.prod(vsize2)
    w = np.zeros(nc).astype(int)
    # convert to original vsize indices
    for nn in range(nc):
        # x = np.array(np.unravel_index(nn,vsize2))
        x = ind2sub_ndims(vsize2, nn)
        x[fixedindices] = fixedvals
        w[nn] = sub2ind(vsize, x)
    return w


def load_network_model_w_intrinsics(networkmodel):
    xls = pd.ExcelFile(networkmodel, engine = 'openpyxl')
    dnet = pd.read_excel(xls, 'connections')
    dnet.pop('Unnamed: 0')   # remove this blank field from the beginning
    dnclusters = pd.read_excel(xls, 'nclusters')

    vintrinsic_count = 0
    fintrinsic_count = 0

    nregions = len(dnclusters)
    ntargets, ncols = dnet.shape
    nsources_max = ncols-1

    sem_region_list = []
    ncluster_list = []
    for nn in range(nregions):
        sem_region_list.append(dnclusters.loc[nn,'name'])
        cname = dnclusters.loc[nn,'name']
        if 'vintrinsic' in cname:  vintrinsic_count += 1
        if 'fintrinsic' in cname:  fintrinsic_count += 1
        entry = {'name':dnclusters.loc[nn,'name'],'nclusters':dnclusters.loc[nn,'nclusters']}
        ncluster_list.append(entry)

    network = []
    for nn in range(ntargets):
        targetname = dnet.loc[nn,'target']
        targetnum = sem_region_list.index(targetname)
        sourcelist = []
        sourcenumlist = []
        for ss in range(nsources_max):
            tag = 'source'+str(ss+1)
            try:
                sourcename = dnet.loc[nn,tag]
                if not str(sourcename) == 'nan':
                    sourcelist.append(sourcename)
                    sourcenum = sem_region_list.index(sourcename)
                    sourcenumlist.append(sourcenum)
            except:
                print('source {} for target {} ignored in network definition - invalid source'.format(sourcename,targetname))

        entry = {'target':targetname, 'sources':sourcelist, 'targetnum':targetnum, 'sourcenums':sourcenumlist}
        network.append(entry)

    return network, ncluster_list, sem_region_list, fintrinsic_count, vintrinsic_count


def gradients_in_vintrinsics(Sinput, Sconn, fintrinsic1, vintrinsics, beta_int1,
                             Minput, Mconn, dvali, fintrinsic_count, vintrinsic_count):
    nregions, tsize_full = np.shape(Sinput)
    ncon, tsize_full = np.shape(Sconn)
    nv,nt = np.shape(vintrinsics)
    nI = nv*nt
    dssq_dI = np.zeros((nv,nt))

    II = copy.deepcopy(vintrinsics)
    Sinput_full = np.array(Sinput)
    Sconn_full = np.array(Sconn)
    if fintrinsic_count > 0:
        Sinput_full = np.concatenate((Sinput_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
        Sconn_full = np.concatenate((Sconn_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
    if vintrinsic_count > 0:
        Sinput_full = np.concatenate((Sinput_full, vintrinsics), axis=0)
        Sconn_full = np.concatenate((Sconn_full, vintrinsics), axis=0)

    fit, Sconn_full = network_eigenvalue_method(Sconn_full, Minput, Mconn, ncon)
    Sconn = Sconn_full[:ncon,:]

    err = Sinput_full[:nregions, :] - fit[:nregions, :]
    cost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(II)) + np.sum(np.abs(betavals))
    ssqd = np.sum(err ** 2) + Lweight * cost  # L1 regularization

    for nn in range(nI):
        II = copy.deepcopy(vintrinsics)
        aa,bb = np.unravel_index(nn, (nv,nt))
        II[aa,bb] += dvali

        Sin_full = np.array(Sinput)
        S_full = np.array(Sconn)
        if fintrinsic_count > 0:
            Sin_full = np.concatenate((Sin_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
            S_full = np.concatenate((S_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
        Sin_full = np.concatenate((Sin_full, II), axis=0)
        S_full = np.concatenate((S_full, II), axis=0)

        fit, S_full = network_eigenvalue_method(S_full, Minput, Mconn, ncon)

        err = Sin_full[:nregions, :] - fit[:nregions, :]
        cost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(II)) + np.sum(np.abs(betavals))
        ssqdp = np.sum(err ** 2) + Lweight * cost  # L1 regularization
        dssq_dI[aa,bb] = (ssqdp - ssqd) / dvali

    return dssq_dI, ssqd


def gradients_in_beta1(Sinput, Sconn, fintrinsic1, vintrinsics, beta_int1, Minput, Mconn,
                       dval, fintrinsic_count, vintrinsic_count):
    nregions,tsize_full = np.shape(Sinput)
    ncon,tsize_full = np.shape(Sconn)
    dint = copy.deepcopy(beta_int1)

    Sin_full = np.array(Sinput)
    S_full = np.array(Sconn)
    if fintrinsic_count > 0:
        Sin_full = np.concatenate((Sin_full, dint * fintrinsic1[np.newaxis, :]), axis=0)
        S_full = np.concatenate((S_full, dint * fintrinsic1[np.newaxis, :]), axis=0)
    if vintrinsic_count > 0:
        Sin_full = np.concatenate((Sin_full, vintrinsics), axis=0)
        S_full = np.concatenate((S_full, vintrinsics), axis=0)

    fit, S_full = network_eigenvalue_method(S_full, Minput, Mconn, ncon)
    Soutput = S_full[:ncon,:]

    err = Sin_full[:nregions, :] - fit[:nregions, :]
    cost = np.sum(np.abs(dint)) + np.sum(np.abs(vintrinsics)) + np.sum(np.abs(betavals))
    ssqd = np.sum(err ** 2) + Lweight * cost  # L1 regularization

    dint += dval
    Sin_full = np.array(Sinput)
    S_full = np.array(Sconn)
    if fintrinsic_count > 0:
        Sin_full = np.concatenate((Sin_full, dint * fintrinsic1[np.newaxis, :]), axis=0)
        S_full = np.concatenate((S_full, dint * fintrinsic1[np.newaxis, :]), axis=0)
    if vintrinsic_count > 0:
        Sin_full = np.concatenate((Sin_full, vintrinsics), axis=0)
        S_full = np.concatenate((S_full, vintrinsics), axis=0)

    fit, S_full = network_eigenvalue_method(S_full, Minput, Mconn, ncon)

    err = Sin_full[:nregions, :] - fit[:nregions, :]
    # cost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(betavals)) + np.sum(np.abs(intrinsic2))
    cost = np.sum(np.abs(dint)) + np.sum(np.abs(vintrinsics)) + np.sum(np.abs(betavals))
    ssqdp = np.sum(err ** 2) + Lweight * cost  # L1 regularization
    dssq_dbeta1 = (ssqdp - ssqd) / dval

    return dssq_dbeta1, ssqd


def gradients_for_betavals(Sinput, Sconn, fintrinsic1, vintrinsics, beta_int1, Minput, Mconn,
                           betavals, ctarget, csource, dval, fintrinsic_count, vintrinsic_count):
    nregions, tsize_full = np.shape(Sinput)
    ncon, tsize_full = np.shape(Sconn)
    nbetavals = len(betavals)

    # initialize
    Sinput_full = np.array(Sinput)
    Sconn_full = np.array(Sconn)
    if fintrinsic_count > 0:
        Sinput_full = np.concatenate((Sinput_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
        Sconn_full = np.concatenate((Sconn_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
    if vintrinsic_count > 0:
        Sinput_full = np.concatenate((Sinput_full, vintrinsics), axis=0)
        Sconn_full = np.concatenate((Sconn_full, vintrinsics), axis=0)

    Mconn[ctarget, csource] = betavals

    fit, Sconn_temp = network_eigenvalue_method(Sconn_full, Minput, Mconn, ncon)
    Sconn = Sconn_temp[:ncon,:]

    err = Sinput_full[:nregions, :] - fit[:nregions, :]
    cost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(vintrinsics)) + np.sum(np.abs(betavals))
    ssqd = np.sum(err ** 2) + Lweight * cost  # L1 regularization

    # gradients for betavals
    # cost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(betavals)) + np.sum(np.abs(intrinsic2))
    dssq_db = np.zeros(np.shape(betavals))
    for nn in range(nbetavals):
        b = copy.deepcopy(betavals)
        b[nn] += dval
        Mconn[ctarget, csource] = b

        fit, Sconn_temp = network_eigenvalue_method(Sconn_full, Minput, Mconn, ncon)

        err = Sinput_full[:nregions, :] - fit[:nregions, :]
        cost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(vintrinsics)) + np.sum(np.abs(b))
        ssqdp = np.sum(err ** 2) + Lweight * cost  # L1 regularization
        dssq_db[nn] = (ssqdp - ssqd) / dval

    return dssq_db, ssqd


def network_eigenvalue_method(Sconn_full, Minput, Mconn, ncon):
    #
    # Soutput_full = Moutput @ Soutput_full
    # find Soutput_working with given starting values
    # the form of Moutput is a block matrix with the upper nregions x nregions section
    # giving the beta values
    # the lower nintrinsic x nintrinsic portion is an identity matrix
    # and the upper right nregions x nintrinsic porition is the mixing from the intrinsics
    # to the regions
    # This form ensures that there are are number of eigenvalues = 1, and the number
    # is equal to nintrinsic
    # the corresponding eigenvectors have non-zero values for the intrinsic inputs and for
    # other regions only if there is mixing between them
    nr,nt = np.shape(Sconn_full)
    nintrinsics = nr-ncon

    det = np.linalg.det(Mconn)
    w,v = np.linalg.eig(Mconn)

    # Moutput @ v[:,a] = w[a]*v[:,a]

    # check that intrinsics have eigenvalues = 1 (or close to it)
    # assume that the eigenvalues, eigenvectors are always ordered the same as Moutput
    check = np.zeros(nintrinsics)
    tol = 1e-4
    for nn in range(nintrinsics):
        check[nn] = np.abs(w[nn+ncon]-1.0) < tol

    if np.sum(check) < nintrinsics:
        print('--------------------------------------------------------------------------------')
        print('Error:  network_eigenvalue_method:  solution to network fitting cannot be found!')
        print('--------------------------------------------------------------------------------')
    else:
        # M v = a v
        fit1 = np.zeros((ncon,nt))
        for nn in range(nintrinsics):
            # do this for each intrinsic:
            eval = np.real(w[nn+ncon])
            evec = np.real(v[:,nn+ncon])
            for tt in range(nt):
                scale = Sconn_full[nn+ncon,tt]/evec[nn+ncon]
                fit1[:ncon,tt] += evec[:ncon]*scale

        Sconn_working = copy.deepcopy(Sconn_full)
        Sconn_working[:ncon] = fit1
        fit = Minput @ Sconn_working

    return fit, Sconn_working


def get_overall_num(nclusterlist, regionnum, clusternum):
    if isinstance(regionnum,list):
        number = [np.sum(nclusterlist[:regionnum[aa]]) + clusternum[aa] for aa in range(len(regionnum))]
    if isinstance(regionnum,np.ndarray):
        number = [np.sum(nclusterlist[:regionnum[aa]]) + clusternum[aa] for aa in range(len(regionnum))]
    if isinstance(regionnum,int):
        number = np.sum(nclusterlist[:regionnum]) + clusternum
    return number

#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
def ind2sub_ndims(vsize,index):
    # mlist = ind2sub_ndims(vsize, ind)
    ndims = len(vsize)
    m = np.zeros(ndims).astype(int)
    for nn in range(ndims):
        if nn == 0:
            m[0] = np.mod(index,vsize[0])
        else:
            m[nn] = np.mod( np.floor(index/np.prod(vsize[:nn])), vsize[nn])
    return m


# temporary-------------------------
# get covariates
# settingsfile = r'C:\Users\Stroman\PycharmProjects\pyspinalfmri3\venv\base_settings_file.npy'
# settings = np.load(settingsfile, allow_pickle=True).flat[0]
# covariates1 = settings['GRPcharacteristicsvalues'][0]  # gender
# covariates2 = settings['GRPcharacteristicsvalues'][1].astype(float)  # painrating

#----------------------------------------------------------------------------------
# main function--------------------------------------------------------------------
starttime = time.ctime()
# main function
outputdir = r'D:/threat_safety_python/SEMresults'
SEMresultsname = os.path.join(outputdir,'SEMresults_newmethod_3.npy')
SEMparameterssname = os.path.join(outputdir,'SEMparameters_newmethod_3.npy')
networkfile = r'D:/threat_safety_python/network_model_with_3intrinsics.xlsx'
network, ncluster_list, sem_region_list, fintrinsic_count, vintrinsic_count = load_network_model_w_intrinsics(networkfile)

# load data--------------------------------------------------------------------
DBname = r'D:/threat_safety_python/threat_safety_database.xlsx'
xls = pd.ExcelFile(DBname, engine='openpyxl')
df1 = pd.read_excel(xls, 'paradigm1_BOLD')
del df1['Unnamed: 0']  # get rid of the unwanted header column
fields = list(df1.keys())
paradigm = df1['paradigms_BOLD']
timevals = df1['time']
paradigm_centered = paradigm - np.mean(paradigm)
dparadigm = np.zeros(len(paradigm))
dparadigm[1:] = np.diff(paradigm_centered)

region_data_name1 = r'D:/threat_safety_python/SEMresults/threat_safety_regiondata_allthreat55.npy'
clustername = r'D:/threat_safety_python/SEMresults/threat_safety_clusterdata.npy'

region_data1 = np.load(region_data_name1, allow_pickle=True).flat[0]
region_properties = region_data1['region_properties']

cluster_data = np.load(clustername, allow_pickle=True).flat[0]
cluster_properties = cluster_data['cluster_properties']

nregions = len(cluster_properties)
nclusterlist = [cluster_properties[i]['nclusters'] for i in range(nregions)]
rnamelist = [cluster_properties[i]['rname'] for i in range(nregions)]
nclusterstotal = np.sum(nclusterlist)

tsize = region_properties[0]['tsize']
nruns_per_person = region_properties[0]['nruns_per_person']
nruns_total = np.sum(nruns_per_person)
NP = len(nruns_per_person)  # number of people in the data set

tcdata = []
for i in range(nregions):
    tc = region_properties[i]['tc']
    if i == 0:
        tcdata = tc
    else:
        tcdata = np.append(tcdata, tc, axis=0)


# setup index lists---------------------------------------------------------------------------
# timepoints for full runs----------------------------------------------
tplist_full = []
dtplist_full = []
et1 = 0
et2 = tsize
dtsize = tsize-1  # for using deriviation of tc wrt time
tplist1 = []
dtplist1 = []
nclusterstotal,tsizetotal = np.shape(tcdata)
tcdata_centered = copy.deepcopy(tcdata)
dtcdata_centered = np.zeros((nclusterstotal,nruns_total*tsize))
for nn in range(NP):
    r1 = sum(nruns_per_person[:nn])
    r2 = sum(nruns_per_person[:(nn + 1)])
    tp = []  # initialize list
    dtp = []  # initialize list
    tpoints = []
    dtpoints = []
    for ee2 in range(r1, r2):
        tp = list(range((ee2 * tsize), (ee2 * tsize + tsize)))
        # dtp = list(range((ee2 * dtsize), (ee2 * dtsize + dtsize)))
        tpoints = tpoints + tp  # concatenate lists
        # dtpoints = dtpoints + dtp  # concatenate lists
        temp = np.mean(tcdata[:, tp],axis=1)
        temp_mean = np.repeat(temp[:, np.newaxis], tsize, axis=1)
        tcdata_centered[:, tp] = tcdata[:, tp] - temp_mean   # center each epoch, in each person
        dtcdata_centered[:, tp[1:]] = np.diff(tcdata[:, tp])   # 1st dervitive of timecourse wrt time (estimated)
    tplist1.append({'tp': tpoints})
    # dtplist1.append({'tp': dtpoints})
tplist_full.append(tplist1)
# dtplist_full.append(dtplist1)


# compute grid of dSsource/dStarget--------------------------------------------------------------------
timepoint = 0
dSdSgrid = np.zeros((nclusterstotal, nclusterstotal, NP,2))
for nn in range(NP):
    tp = tplist_full[timepoint][nn]['tp']
    tsize_total = len(tp)
    for ss in range(nclusterstotal):
        dss = dtcdata_centered[ss,tp]
        for tt in range(nclusterstotal):
            dtt = dtcdata_centered[tt,tp]
            dsdt = dss/(dtt+1.0e-20)
            stdval = np.std(dsdt)
            dsdt[np.abs(dsdt) > 3.0*stdval] = 0.0
            dSsdSt = np.mean(dsdt)
            dSsdSt_sem = np.std(dsdt)/np.sqrt(tsize_total)
            dSdSgrid[ss,tt,nn,0] = dSsdSt
            dSdSgrid[ss,tt,nn,1] = dSsdSt_sem
T = dSdSgrid[:, :, :, 0] / (dSdSgrid[:, :, :, 1] + 1.0e-20)

# network mask for T grid etc.--------------------------------------------------------------------
mask = np.zeros((nclusterstotal, nclusterstotal))
for nn in range(len(network)):
    target = network[nn]['targetnum']
    t1 = np.sum(nclusterlist[:target]).astype(int)
    t2 = np.sum(nclusterlist[:(target + 1)])
    sources = network[nn]['sourcenums']
    for mm in range(len(sources)):
        if sources[mm] < nregions:
            s1 = np.sum(nclusterlist[:sources[mm]]).astype(int)
            s2 = np.sum(nclusterlist[:(sources[mm] + 1)])
            mask[s1:s2, t1:t2] = 1

for nn in range(NP):
    dSdSgrid[:, :, nn, 0] *= mask
    dSdSgrid[:, :, nn, 1] *= mask
    T[:, :, nn] *= mask

Tlim = np.abs(T) > 2
Tcount = np.sum(Tlim, axis=2)   # count of how many people have significant estimated beta values for each connection


# setup matrices for modeling network --------------------------------------------------------------------
# new model:  model the timecourse for each connection
# find the number of connections:  nbeta
# Minput and Moutput are (nbeta + Nintrinsic) x (nbeta + Nintrinsic)
#
Nintrinsic = fintrinsic_count + vintrinsic_count
nregions = len(rnamelist)

beta_list = []
nbeta = 0
targetnumlist = []
beta_id = []
sourcelist = []
for nn in range(len(network)):
    target = network[nn]['targetnum']
    sources = network[nn]['sourcenums']
    targetnumlist += [target]
    for mm in range(len(sources)):
        source = sources[mm]
        sourcelist += [source]
        betaname = '{}_{}'.format(source,target)
        entry = {'name':betaname, 'number':nbeta, 'pair':[source,target]}
        beta_list.append(entry)
        beta_id += [1000*source + target]
        nbeta += 1

ncon = nbeta-Nintrinsic

# recorder to put intrinsic inputs at the end-------------
beta_list2 = []
beta_id2 = []
x = np.where(np.array(sourcelist) < nregions)[0]
for xx in x:
    beta_list2.append(beta_list[xx])
    beta_id2 += [beta_id[xx]]
for sn in range(nregions,nregions+Nintrinsic):
    x = np.where(np.array(sourcelist) == sn)[0]
    for xx in x:
        beta_list2.append(beta_list[xx])
        beta_id2 += [beta_id[xx]]

for nn in range(len(beta_list2)):
    beta_list2[nn]['number'] = nn

beta_list = beta_list2
beta_id = beta_id2

beta_pair = []
Mconn = np.zeros((nbeta,nbeta))
count = 0
for nn in range(len(network)):
    target = network[nn]['targetnum']
    sources = network[nn]['sourcenums']
    for mm in range(len(sources)):
        source = sources[mm]
        conn1 = beta_id.index(source*1000 + target)
        if source >= nregions:   # intrinsic input
            conn2 = conn1
            Mconn[conn1,conn2] = 1   # set the intrinsic beta values
        else:
            x = targetnumlist.index(source)
            source_sources = network[x]['sourcenums']
            for nn in range(len(source_sources)):
                ss1 = source_sources[nn]
                conn2 = beta_id.index(ss1*1000 + source)
                beta_pair.append([conn1, conn2])
                count += 1
                Mconn[conn1,conn2] = count

# prep to index Mconn for updating beta values
beta_pair = np.array(beta_pair)
ctarget = beta_pair[:,0]
csource = beta_pair[:,1]

# setup Minput matrix--------------------------------------------------------------
# Sconn = Mconn @ Sconn    # propagate the intrinsic inputs through the network
# Sinput = Minput @ Mconn
Minput = np.zeros((nregions,nbeta))   # mixing of connections to model the inputs to each region
betanamelist = [beta_list[a]['name'] for a in range(nbeta)]
for nn in range(len(network)):
    target = network[nn]['targetnum']
    sources = network[nn]['sourcenums']
    for mm in range(len(sources)):
        source = sources[mm]
        betaname = '{}_{}'.format(source,target)
        x = betanamelist.index(betaname)
        Minput[target,x] = 1

# save parameters for looking at results later
SEMparams = {'betanamelist': betanamelist, 'beta_list':beta_list, 'nruns_per_person': nruns_per_person,
             'nclusterstotal': nclusterstotal, 'rnamelist': rnamelist, 'nregions': nregions,
             'cluster_properties': cluster_properties, 'cluster_data': cluster_data,
             'network': network, 'fintrinsic_count': fintrinsic_count, 'vintrinsic_count': vintrinsic_count,
             'sem_region_list': sem_region_list, 'ncluster_list': ncluster_list, 'tsize': tsize}
np.save(SEMparameterssname, SEMparams)


# initialize gradient-descent parameters--------------------------------------------------------------
initial_alpha = 1e-3
initial_alphai = 1e-3
initial_alphab = 1e-3
initial_Lweight = 1.0
initial_dval = 0.05
initial_dvali = 0.05


#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
# repeat the process for each participant-----------------------------------------------------------------
betalimit = 3.0
timepoint = 0
SEMresults = []
beta_init_record = []
# person_list = [0,7,10]
# for nperson in person_list:
for nperson in range(NP):
    print('starting person {} at {}'.format(nperson,time.ctime()))
    tp = tplist_full[timepoint][nperson]['tp']
    tsize_total = len(tp)
    nruns = nruns_per_person[nperson]

    # get tc data for each region/cluster
    clusterlist = [3,5,13,15,20,25,34,38,43,45]   # picked by "best" function above, with C6RD 3 as starting point
    clusterlist = [4, 9, 14, 15, 20, 28, 32, 35, 41, 47]   # picked by PCA method below
    rnumlist = []
    clustercount = np.cumsum(nclusterlist)
    for aa in range(len(clusterlist)):
        x = np.where(clusterlist[aa] < clustercount)[0]
        rnumlist += [x[0]]

    Sinput = []
    for cval in clusterlist:
        tc1 = tcdata_centered[cval, tp]
        Sinput.append(tc1)
    # Sinput is size:  nregions x tsize_total

    if fintrinsic_count > 0:
        beta_int1 = 0.1    # start the magnitude of intrinsic1 at a small value
        fintrinsic1 = np.array(list(paradigm_centered) * nruns_per_person[nperson])
    else:
        beta_int1 = 0.0

    if vintrinsic_count > 0:
        vintrinsics = np.zeros((vintrinsic_count, tsize_total))    # initialize unknown intrinsic with small random values
        # for v in range(vintrinsic_count):
        #     vtemp = np.mean(Sinput,axis=0)
        #     vtemp = center_tc_by_run(vtemp, nruns)  # keep all tc values at a mean of zero
        #     vintrinsics[v,:] = vtemp

    # initialize beta values based on dSdSgrid and T value of it, for each source-target pair in the network
    # 1) optimize intrinsic1 and intrinsic2 for these beta values
    # 2) optimze betavals for given intrinsic1 and intrinsic2 values
    # 3) repeat

    # initialize beta values-----------------------------------
    beta_initial = np.zeros(len(csource))
    for nn in range(len(csource)):
        # check how the connection relates to the next connection in the network
        xsource = beta_id[csource[nn]]
        sc1 = np.floor(xsource/1000).astype(int)
        tc1 = np.mod(xsource,1000).astype(int)
        x1 = rnumlist.index(tc1)
        cluster1 = clusterlist[x1]

        xtarget = beta_id[ctarget[nn]]
        sc2 = np.floor(xtarget/1000).astype(int)
        tc2 = np.mod(xtarget,1000).astype(int)
        x2 = rnumlist.index(tc2)
        cluster2 = clusterlist[x2]

        beta = dSdSgrid[cluster1,cluster2,nperson,0]
        betaT = T[cluster1,cluster2,nperson]
        # if np.abs(betaT) > 0.5:
        #     beta_initial[nn] = beta
        beta_initial[nn] = beta

    # for testing
    # beta_initial = np.random.randn(len(csource))

    beta_initial[beta_initial >= betalimit] = betalimit
    beta_initial[beta_initial <= -betalimit] = -betalimit
    beta_init_record.append({'beta_initial':beta_initial})

    # initalize Sconn
    Sconn = np.zeros((ncon,tsize_total))   # initialize
    betavals = copy.deepcopy(beta_initial) # initialize beta values at zero

    bigitermax = 6
    bigiter = 0
    results_record = []
    lastgood_vintrinsics = copy.deepcopy(vintrinsics)
    lastgood_beta_int1 = copy.deepcopy(beta_int1)
    lastgood_betavals = copy.deepcopy(betavals)
    ssqd_record = []
    while bigiter < bigitermax:
        bigiter += 1
        # on each iteration....
        # first determine values of:  beta_int1, intrinsic2
        # then determine values of: betavals
        alpha = initial_alpha
        alphai = initial_alphai
        alphab = initial_alphab
        Lweight = initial_Lweight
        dval = initial_dval
        dvali = initial_dvali

        # initialize values at the start of each pass------------------
        Sinput_full = copy.deepcopy(np.array(Sinput))
        Sconn_full = copy.deepcopy(Sconn)
        if fintrinsic_count > 0:
            Sinput_full = np.concatenate((Sinput_full,beta_int1*fintrinsic1[np.newaxis,:]),axis=0)
            Sconn_full = np.concatenate((Sconn_full,beta_int1*fintrinsic1[np.newaxis,:]),axis=0)
        if vintrinsic_count > 0:
            Sinput_full = np.concatenate((Sinput_full,vintrinsics),axis=0)
            Sconn_full = np.concatenate((Sconn_full,vintrinsics),axis=0)
        Mconn[ctarget,csource] = betavals

        # starting point for optimizing intrinsics with given betavals----------------------------------------------------
        fit, Sconn_full = network_eigenvalue_method(Sconn_full, Minput, Mconn, ncon)

        Sconn = Sconn_full[:ncon,:]
        err = Sinput_full[:nregions,:] - fit[:nregions,:]
        cost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(vintrinsics)) + np.sum(np.abs(betavals))
        ssqd = np.sum(err ** 2) + Lweight * cost  # L1 regularization
        if bigiter == 1:  ssqd_starting = ssqd
        ssqd_record += [ssqd]

        nitermax = 50
        alpha_limit = 1.0e-5

        iter = 0
        vintrinsics_record = []
        converging = True
        dssq_record = np.ones(3)
        dssq_count = 0
        while alphai > alpha_limit  and iter < nitermax  and converging:
            iter += 1

            # gradients for vintrinsics
            # initialize
            Sinput_full = copy.deepcopy(np.array(Sinput))
            Sconn_full = copy.deepcopy(Sconn)
            if fintrinsic_count > 0:
                Sinput_full = np.concatenate((Sinput_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
                Sconn_full = np.concatenate((Sconn_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
            if vintrinsic_count > 0:
                Sinput_full = np.concatenate((Sinput_full, vintrinsics), axis=0)
                Sconn_full = np.concatenate((Sconn_full, vintrinsics), axis=0)
            Mconn[ctarget,csource] = betavals

            # gradients in intrinsic2
            # dssq_dI, ssqd = gradients_in_intrinsic2(Sinput, Soutput, intrinsic1, intrinsic2, beta_int1, Minput, Moutput, dvali)
            dssq_dI, ssqd = gradients_in_vintrinsics(Sinput, Sconn, fintrinsic1, vintrinsics, beta_int1,
                                     Minput, Mconn, dvali, fintrinsic_count, vintrinsic_count)
            ssqd_record += [ssqd]
            dssq_dbeta1, ssqd = gradients_in_beta1(Sinput, Sconn, fintrinsic1, vintrinsics, beta_int1, Minput, Mconn, dval, fintrinsic_count, vintrinsic_count)
            ssqd_record += [ssqd]

            # apply the changes
            vintrinsics -= alphai * dssq_dI
            beta_int1 -= alphab * dssq_dbeta1

            vintrinsics_record.append({'i2':vintrinsics})

            Sinput_full = np.array(Sinput)
            Sconn_full = np.array(Sconn)
            if fintrinsic_count > 0:
                Sinput_full = np.concatenate((Sinput_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
                Sconn_full = np.concatenate((Sconn_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
            if vintrinsic_count > 0:
                Sinput_full = np.concatenate((Sinput_full, vintrinsics), axis=0)
                Sconn_full = np.concatenate((Sconn_full, vintrinsics), axis=0)

            Mconn[ctarget,csource] = betavals
            fit, Sconn_temp = network_eigenvalue_method(Sconn_full, Minput, Mconn, ncon)

            S = Sinput_full[:nregions, :]
            err = S - fit[:nregions, :]
            Smean = np.mean(S)
            errmean = np.mean(err)
            R2total = 1 - np.sum((err - errmean) ** 2) / np.sum((S - Smean) ** 2)

            cost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(vintrinsics)) + np.sum(np.abs(betavals))
            ssqd_new = np.sum(err ** 2) + Lweight * cost  # L1 regularization

            # Sinput_sim, Soutput_sim = network_sim(Sinput_full, Soutput_temp, Minput, Moutput)
            results_record.append({'Sinput':fit, 'Soutput':Sconn_temp})

            if ssqd_new >= ssqd:
                alphab *= 0.5
                alphai *= 0.5
                # revert back to last good values
                vintrinsics = copy.deepcopy(lastgood_vintrinsics)
                beta_int1 = copy.deepcopy(lastgood_beta_int1)
                dssqd = ssqd - ssqd_new
                dssq_record = np.ones(3)   # reset the count
                dssq_count = 0
                print('intrinsics:  iter {} alpha {:.3e}  delta ssq {:.4f} - no update'.format(iter,alphai,-dssqd))
            else:
                # save the good values
                lastgood_vintrinsics = copy.deepcopy(vintrinsics)
                lastgood_beta_int1 = copy.deepcopy(beta_int1)
                Sconn = Sconn_temp[:ncon,:]

                dssqd = ssqd - ssqd_new
                ssqd = ssqd_new

                dssq_count += 1
                dssq_count = np.mod(dssq_count, 3)
                dssq_record[dssq_count] = 100.0*dssqd/ssqd_starting
                if np.max(dssq_record) < 0.01:  converging = False

                print('intrinsics:  iter {} alpha {:.3e}  delta ssq {:.4f}  relative: {:.1f} percent  R2 {:.3f}'.format(iter,alphai,-dssqd,100.0*ssqd/ssqd_starting, R2total))

        # starting point for optimizing betavals ---------------------------------------------------------------
        alpha = initial_alpha
        alphai = initial_alphai
        alphab = initial_alphab
        Lweight = initial_Lweight
        dval = initial_dval
        dvali = initial_dvali

        # initialize values at the start of each pass------------------
        Sinput_full = np.array(Sinput)
        Sconn_full = np.array(Sconn)
        if fintrinsic_count > 0:
            Sinput_full = np.concatenate((Sinput_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
            Sconn_full = np.concatenate((Sconn_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
        if vintrinsic_count > 0:
            Sinput_full = np.concatenate((Sinput_full, vintrinsics), axis=0)
            Sconn_full = np.concatenate((Sconn_full, vintrinsics), axis=0)

        Mconn[ctarget,csource] = betavals

        # starting point for optimizing betavals with given intrinsics----------------------------------------------------
        # fit, Soutput_full = network_fit(Soutput_full, Minput, Moutput)
        # fit, Soutput_full = network_approach_method(Soutput_full, Minput, Moutput, nregions)
        # fit, Soutput_full = network_descent_L1(Soutput_full, Minput, Moutput, nregions)   # update with more accurate method again
        fit, Sconn_full = network_eigenvalue_method(Sconn_full, Minput, Mconn, ncon)
        Sconn = Sconn_full[:ncon,:]

        # fit, Soutput_temp = network_fit(Soutput_full, Minput, Moutput)   # use the quick method for estimating ssqd start
        # fit, Soutput_temp = network_approach_method(Soutput_full, Minput, Moutput, nregions)
        fit, Sconn_temp = network_eigenvalue_method(Sconn_full, Minput, Mconn, ncon)
        err = Sinput_full[:nregions,:] - fit[:nregions,:]
        cost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(vintrinsics)) + np.sum(np.abs(betavals))
        ssqd = np.sum(err ** 2) + Lweight * cost  # L1 regularization

        nitermax = 50
        alpha_limit = 1.0e-6

        iter = 0
        converging = True
        dssq_record = np.ones(3)
        dssq_count = 0
        while alpha > alpha_limit  and iter < nitermax and converging:
            iter += 1
            Sinput_full = copy.deepcopy(np.array(Sinput))
            Sconn_full = copy.deepcopy(Sconn)
            if fintrinsic_count > 0:
                Sinput_full = np.concatenate((Sinput_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
                Sconn_full = np.concatenate((Sconn_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
            if vintrinsic_count > 0:
                Sinput_full = np.concatenate((Sinput_full, vintrinsics), axis=0)
                Sconn_full = np.concatenate((Sconn_full, vintrinsics), axis=0)

            Mconn[ctarget,csource] = betavals

            dssq_db, ssqd = gradients_for_betavals(Sinput, Sconn, fintrinsic1, vintrinsics, beta_int1, Minput, Mconn,
                                             betavals, ctarget, csource, dval, fintrinsic_count, vintrinsic_count)
            ssqd_record += [ssqd]

            # apply the changes
            betavals -= alpha * dssq_db

            betavals[betavals >= betalimit] = betalimit
            betavals[betavals <= -betalimit] = -betalimit

            Mconn[ctarget,csource] = betavals
            fit, Sconn_temp = network_eigenvalue_method(Sconn_full, Minput, Mconn, ncon)

            S = Sinput_full[:nregions, :]
            err = S - fit[:nregions, :]
            Smean = np.mean(S)
            errmean = np.mean(err)
            R2total = 1 - np.sum((err - errmean) ** 2) / np.sum((S - Smean) ** 2)

            cost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(vintrinsics)) + np.sum(np.abs(betavals))
            ssqd_new = np.sum(err ** 2) + Lweight * cost  # L1 regularization

            # Sinput_sim, Soutput_sim = network_sim(Sinput_full, Soutput_full, Minput, Moutput)
            results_record.append({'Sinput':fit, 'Soutput':Sconn_temp})

            if ssqd_new >= ssqd:
                alpha *= 0.5
                # revert back to last good values
                betavals = copy.deepcopy(lastgood_betavals)
                dssqd = ssqd - ssqd_new
                dssq_record = np.ones(3) # reset the count
                dssq_count = 0
                print('beta vals:  iter {} alpha {:.3e}  delta ssq > 0  - no update'.format(iter, alpha))
            else:
                # save the good values
                lastgood_betavals = copy.deepcopy(betavals)
                Sconn= Sconn_temp[:ncon,:]

                dssqd = ssqd - ssqd_new
                ssqd = ssqd_new

                dssq_count += 1
                dssq_count = np.mod(dssq_count, 3)
                dssq_record[dssq_count] = 100.0*dssqd/ssqd_starting
                if np.max(dssq_record) < 0.01:  converging = False

                print('beta vals:  iter {} alpha {:.3e}  delta ssq {:.4f}  relative: {:.1f} percent  R2 {:.3f}'.format(iter,alpha,-dssqd,100.0*ssqd/ssqd_starting, R2total))
    # now repeat it ...

    # show results
    betavals = copy.deepcopy(lastgood_betavals)
    vintrinsics = copy.deepcopy(lastgood_vintrinsics)
    beta_int1 = copy.deepcopy(lastgood_beta_int1)

    Sinput_full = copy.deepcopy(np.array(Sinput))
    Sconn_full = copy.deepcopy(Sconn)
    if fintrinsic_count > 0:
        Sinput_full = np.concatenate((Sinput_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
        Sconn_full = np.concatenate((Sconn_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
    if vintrinsic_count > 0:
        Sinput_full = np.concatenate((Sinput_full, vintrinsics), axis=0)
        Sconn_full = np.concatenate((Sconn_full, vintrinsics), axis=0)

    Mconn[ctarget, csource] = betavals

    fit, Sconn_full = network_eigenvalue_method(Sconn_full, Minput, Mconn, ncon)
    S = Sinput_full[:nregions, :]
    err = S- fit[:nregions, :]
    Smean = np.mean(S)
    errmean = np.mean(err)
    R2total = 1 - np.sum((err-errmean)**2)/np.sum((S-Smean)**2)

    regionnum1 = 0
    regionnum2 = 7
    window1 = 24
    window2 = 25

    nruns = nruns_per_person[nperson]
    tsize = (tsize_total/nruns).astype(int)

    tc = Sinput_full[regionnum1,:]
    tc1 = np.mean(np.reshape(tc, (nruns, tsize)), axis=0)
    tc = fit[regionnum1,:]
    tcf1 = np.mean(np.reshape(tc, (nruns, tsize)), axis=0)

    plt.close(window1)
    fig = plt.figure(window1, figsize=(12.5, 3.5), dpi=100)
    plt.plot(range(tsize),tc1,'-ob')
    plt.plot(range(tsize),tcf1,'-xr')

    tc = Sinput_full[regionnum2,:]
    tc2 = np.mean(np.reshape(tc, (nruns, tsize)), axis=0)
    tc = fit[regionnum2,:]
    tcf2 = np.mean(np.reshape(tc, (nruns, tsize)), axis=0)

    plt.close(window2)
    fig = plt.figure(window2, figsize=(12.5, 3.5), dpi=100)
    plt.plot(range(tsize),tc2,'-ob')
    plt.plot(range(tsize),tcf2,'-xr')

    # plot intrinsic2
    tc = vintrinsics[0,:]
    tc1 = np.mean(np.reshape(tc, (nruns, tsize)), axis=0)
    tc = vintrinsics[1,:]
    tc2 = np.mean(np.reshape(tc, (nruns, tsize)), axis=0)

    plt.close(20)
    fig = plt.figure(20, figsize=(12.5, 3.5), dpi=100)
    plt.subplot(2,1,1)
    plt.plot(range(tsize),tc1,'-og')
    plt.subplot(2,1,2)
    plt.plot(range(tsize),tc2,'-og')

    columns = [name[:3] +' in' for name in betanamelist]
    rows = [name[:3] for name in betanamelist]
    df = pd.DataFrame(Mconn,columns = columns, index = rows)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    pd.options.display.float_format = '{:.2f}'.format
    print(df)

    R1 = np.corrcoef(Sinput_full[regionnum1, :], fit[regionnum1, :])
    Z1 = np.arctanh(R1[0, 1]) * np.sqrt(tsize_total-3)
    results_text1 = 'person {} region {}   R = {:.3f}  Z = {:.2f}'.format(nperson, regionnum1, R1[0, 1], Z1)
    print(results_text1)

    R2 = np.corrcoef(Sinput_full[regionnum2, :], fit[regionnum2, :])
    Z2 = np.arctanh(R2[0, 1]) * np.sqrt(tsize_total-3)
    results_text2 = 'person {} region {}   R = {:.3f}  Z = {:.2f}'.format(nperson, regionnum2, R2[0, 1], Z2)
    print(results_text2)

    entry = {'Sinput':Sinput_full, 'Sconn':Sconn_full, 'beta_int1':beta_int1, 'Mconn':Mconn, 'Minput':Minput,
             'rtext1':results_text1, 'rtext2':results_text2, 'R2total':R2total}
    SEMresults.append(copy.deepcopy(entry))

    stoptime = time.ctime()

np.save(SEMresultsname, SEMresults)
print('finished SEM at {}'.format(time.ctime()))
print('     started at {}'.format(starttime))


#------for checking results---------------------
check_results = False
if check_results:
    # reload parameters if needed--------------------------------------------------------
    settingsfile = r'C:\Users\Stroman\PycharmProjects\pyspinalfmri3\venv\base_settings_file.npy'
    settings = np.load(settingsfile, allow_pickle=True).flat[0]
    covariates1 = settings['GRPcharacteristicsvalues'][0]  # gender
    covariates2 = settings['GRPcharacteristicsvalues'][1].astype(float)  # painrating

    SEMparams = np.load(SEMparameterssname, allow_pickle=True).flat[0]
    # SEMparams = {'betanamelist': betanamelist, 'beta_list':beta_list, 'nruns_per_person': nruns_per_person,
    #              'nclusterstotal':nclusterstotal, 'rnamelist':rnamelist, 'nregions':nregions,
    #             'cluster_properties':cluster_properties, 'cluster_data':cluster_data,
    #              'network':network, 'fintrinsic_count':fintrinsic_count, 'vintrinsic_count':vintrinsic_count,
    #              'sem_region_list':sem_region_list, 'ncluster_list':ncluster_list, 'tsize':tsize}

    network = SEMparams['network']
    beta_list = SEMparams['beta_list']
    betanamelist = SEMparams['betanamelist']
    nruns_per_person = SEMparams['nruns_per_person']
    rnamelist = SEMparams['rnamelist']
    fintrinsic_count = SEMparams['fintrinsic_count']
    vintrinsic_count = SEMparams['vintrinsic_count']
    Nintrinsic = fintrinsic_count + vintrinsic_count
    # end of reloading parameters if needed--------------------------------------------------------

    # load the SEM results
    SEMresults_load = np.load(SEMresultsname, allow_pickle=True)

    # for nperson in range(NP):
    person_list = [41, 48, 32, 21, 10]

    NP = len(SEMresults_load)
    resultscheck = np.zeros((NP,4))
    nbeta,tsize_full = np.shape(SEMresults_load[0]['Sconn'])
    ncon = nbeta-Nintrinsic

    for nperson in person_list:
        nruns = nruns_per_person[nperson]
        Sinput = SEMresults_load[nperson]['Sinput']
        Sconn= SEMresults_load[nperson]['Sconn']
        Minput = SEMresults_load[nperson]['Minput']
        Mconn = SEMresults_load[nperson]['Mconn']
        beta_int1 = SEMresults_load[nperson]['beta_int1']
        R2total = SEMresults_load[nperson]['R2total']
        # fit, Soutput_temp = network_fit(Soutput, Minput, Moutput)
        # fit, Soutput_temp = network_descent_L1(Soutput, Minput, Moutput, nregions)
        # fit, Soutput_temp = network_approach_method(Soutput, Minput, Moutput, nregions)
        fit, Sconn_temp = network_eigenvalue_method(Sconn, Minput, Mconn, ncon)
        tsize = (tsize_total/nruns).astype(int)

        region1 = 0
        region2 = 5
        region3 = 7
        nametag2 = r'_cord_NRM_PAG'

        target = 'NRM'
        nametag1 = r'NRMinput'
        rtarget = rnamelist.index(target)
        m = Minput[rtarget,:]
        sources = np.where(m == 1)[0]
        rsources = [beta_list[ss]['pair'][0] for ss in sources]

        target2 = 'NGC'
        rtarget2 = rnamelist.index(target2)

        window1 = 24
        window2 = 25

        # inputs to NRM as example
        plt.close(window1)
        # fig1 = plt.figure(window1, figsize=(12, 9), dpi=100)
        fig1, ((ax1,ax2), (ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,sharey=True, figsize=(12, 9), dpi=100, num = window1)
        # ax1 = fig.add_subplot(3,2,4, sharey = True)

        tc = Sinput[rtarget, :]
        tc1 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = fit[rtarget, :]
        tcf = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)

        ax4.plot(tc1, '-ob')
        ax4.plot(tcf, '-xr')
        ax4.set_title('target input {}'.format(rnamelist[rtarget]))

        tc = Sinput[rtarget2, :]
        tc2 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = fit[rtarget2, :]
        tc2f = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)

        ax2.plot(tc2, '-ob')
        ax2.plot(tc2f, '-xr')
        ax2.set_title('target input {}'.format(rnamelist[rtarget2]))


        tc = Sconn[sources[0], :]
        tc1 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = Sconn[sources[1], :]
        tc2 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = Sconn[sources[2], :]
        tc3 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)

        ax1.plot(tc1, '-xr')
        ax1.set_title('source output {} {}'.format(betanamelist[sources[0]], rnamelist[rsources[0]]))
        ax3.plot(tc2, '-xr')
        ax3.set_title('source output {} {}'.format(betanamelist[sources[1]], rnamelist[rsources[1]]))
        ax5.plot(tc3, '-xr')
        ax5.set_title('source output {} {}'.format(betanamelist[sources[2]], rnamelist[rsources[2]]))

        p,f = os.path.split(SEMresultsname)
        svgname = os.path.join(p,'Person{}_'.format(nperson) + nametag1 + '.svg')
        plt.savefig(svgname)

        # show C6RD, NRM, and PAG as examples (inputs real and fit)
        plt.close(window2)
        # fig2 = plt.figure(window2, figsize=(12, 6), dpi=100)
        fig2, (ax1b,ax2b,ax3b) = plt.subplots(3,sharey=False, figsize=(12, 6), dpi=100, num = window2)
        # ax1 = fig.add_subplot(3,2,4, sharey = True)

        tc = Sinput[region1, :]
        tc1 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = fit[region1, :]
        tcf1 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = Sinput[region2, :]
        tc2 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = fit[region2, :]
        tcf2 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = Sinput[region3, :]
        tc3 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = fit[region3, :]
        tcf3 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)

        ax1b.plot(tc1, '-ob')
        ax1b.plot(tcf1, '-xr')
        ax1b.set_title('target {}'.format(rnamelist[region1]))
        ax2b.plot(tc2, '-ob')
        ax2b.plot(tcf2, '-xr')
        ax2b.set_title('target {}'.format(rnamelist[region2]))
        ax3b.plot(tc3, '-ob')
        ax3b.plot(tcf3, '-xr')
        ax3b.set_title('target {}'.format(rnamelist[region3]))

        p,f = os.path.split(SEMresultsname)
        svgname = os.path.join(p,'Person{}_'.format(nperson) + nametag2 + '.svg')
        plt.savefig(svgname)


        R1 = np.corrcoef(Sinput[region1, :], fit[region1, :])
        Z1 = np.arctanh(R1[0,1]) * np.sqrt(tsize_total-3)
        print('person {} region {}   R = {:.3f}  Z = {:.2f}'.format(nperson, region1,R1[0,1],Z1))
        resultscheck[nperson,0] = R1[0,1]
        resultscheck[nperson,1] = Z1

        R2 = np.corrcoef(Sinput[region2, :], fit[region2, :])
        Z2 = np.arctanh(R2[0,1]) * np.sqrt(tsize_total-3)
        print('person {} region {}   R = {:.3f}  Z = {:.2f}'.format(nperson, region2,R2[0,1],Z2))
        resultscheck[nperson,2] = R2[0,1]
        resultscheck[nperson,3] = Z2

        # vintrinsics
        tc = Sinput[nregions+fintrinsic_count,:]
        tc1 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = Sinput[nregions+fintrinsic_count+1,:]
        tc2 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)

        plt.close(4)
        fig = plt.figure(4, figsize=(12.5, 3.5), dpi=100)
        plt.subplot(2,1,1)
        plt.plot(tc1, '-og')
        plt.subplot(2,1,2)
        plt.plot(tc2, '-og')

        #
        columns = [name[:3] + ' in' for name in betanamelist]
        rows = [name[:3] for name in betanamelist]

        df = pd.DataFrame(Mconn, columns=columns, index=rows)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        pd.options.display.float_format = '{:.2f}'.format
        print(df)

        p,f = os.path.split(SEMresultsname)
        xlname = os.path.join(p,'Person{}_Moutput_v3.xlsx'.format(nperson))
        df.to_excel(xlname)

    Mrecord = np.zeros((nbeta,nbeta,NP))
    R2record = np.zeros(NP)
    for nperson in range(NP):
        Sinput = SEMresults_load[nperson]['Sinput']
        Sconn = SEMresults_load[nperson]['Sconn']
        Minput = SEMresults_load[nperson]['Minput']
        Mconn = SEMresults_load[nperson]['Mconn']
        R2total = SEMresults_load[nperson]['R2total']
        Mrecord[:,:,nperson] = Mconn
        R2record[nperson] = R2total

    mpos = np.zeros((nbeta,nbeta,NP))
    mneg = np.zeros((nbeta,nbeta,NP))
    mpos[Mrecord > 0] = 1
    mneg[Mrecord < 0] = 1
    mpos = np.sum(mpos, axis = 2)
    mneg= np.sum(mneg, axis = 2)
    Mcount = mpos - mneg
    x = np.argmax(np.abs(Mcount[:ncon,:ncon]))
    aa,bb = np.unravel_index(x,np.shape(Mcount[:ncon,:ncon]))


    Rrecord = np.zeros((ncon,ncon))
    R2record = np.zeros((ncon,ncon))
    for aa in range(ncon):
        for bb in range(ncon):
            m = Mrecord[aa,bb,:]
            if np.var(m) > 0:
                R = np.corrcoef(covariates2, m)
                Rrecord[aa,bb] = R[0,1]
                b, fit, R2, total_var, res_var = pysem.general_glm(m[np.newaxis, :], covariates2[np.newaxis, :])
                R2record[aa,bb] = R2

    x = np.argmax(np.abs(R2record))
    aa,bb = np.unravel_index(x, np.shape(R2record))
    # aa,bb = (6,7)
    m = Mrecord[aa,bb,:]
    plt.close(35)
    fig = plt.figure(35), plt.plot(covariates2, m, 'ob')
    b, fit, R2, total_var, res_var = pysem.general_glm(m[np.newaxis, :], covariates2[np.newaxis, :])
    plt.plot(covariates2, fit[0, :], '-b')



# # test matrix concepts
# n1 = 4
# n2 = 2
# M = np.zeros((n1+n2,n1+n2))
# M[:n1,:n1] = np.random.randn(n1,n1)
# M[n1:,n1:] = np.eye(n2,n2)
# # mixing part is crtical:
# M[2,n1+1] = 1
#
# # upper part
# w1,v1 = np.linalg.eig(M[:n1,:n1])
#
# # lower part
# w2,v2 = np.linalg.eig(M[n1:,n1:])
#
# # total
# w3,v3 = np.linalg.eig(M)


check_connections = False
if check_connections:
    nregions = len(nclusterlist)
    ncombo_set = np.floor(nregions/2).astype(int)
    nleaveout = nregions-ncombo_set
    nclusterlist = np.array(nclusterlist)

    list1 = list(range(ncombo_set))
    ncombinations = np.prod(nclusterlist)
    ncombinations1 = np.prod(nclusterlist[list1])

    # list3 = np.sort(np.random.choice(nregions,ncombo_set, replace=False))
    # list4 = np.sort(np.random.choice(nregions,ncombo_set, replace=False))

    EVR1 = np.zeros((NP,ncombinations1,3))
    EVR2 = np.zeros((NP,ncombinations,3))
    nkeep = 100
    xlist = np.zeros((NP,nkeep))  # keep a record of best 1st round picks for each person

    for nperson in range(NP):
        starttime = time.ctime()
        print('starting person {} at {}'.format(nperson, time.ctime()))
        tp = tplist_full[timepoint][nperson]['tp']
        tcdata_centered_person = tcdata_centered[:, tp]
        tsize_total = len(tp)
        nruns = nruns_per_person[nperson]

        # set 1
        cnums = np.zeros(ncombo_set).astype(int)
        full_rnum_base = get_overall_num(nclusterlist, list1, cnums)
        full_rnum_base = np.array(full_rnum_base).astype(int)

        print('     part 1 at {}'.format(time.ctime()))
        for nc in range(ncombinations1):
            cnums = ind2sub_ndims(nclusterlist[list1], nc)
            clusterlist = np.array(cnums) + full_rnum_base
            Sinput = tcdata_centered_person[clusterlist,:]

            pca = PCA(n_components=3)
            pca.fit(Sinput)
            EVR1[nperson,nc,:] = pca.explained_variance_ratio_

        # save a record of the best finds so far
        evr_values = EVR1[nperson,:,0]
        x = np.argsort(-evr_values)
        xlist[nperson,:] = x[:nkeep]

    print('collect the best starts for each person  {}'.format(time.ctime()))
    x2 = xlist[:,0]

    fixedindices = list1
    full_rnum_base = get_overall_num(nclusterlist, list(range(nregions)), np.zeros(nregions))
    full_rnum_base = np.array(full_rnum_base).astype(int)
    for nperson in range(NP):
        print('starting person {} at {}'.format(nperson, time.ctime()))
        # search through the top starting combinations
        tp = tplist_full[timepoint][nperson]['tp']
        tcdata_centered_person = tcdata_centered[:, tp]
        for ss, x in enumerate(x2):
            cnums = ind2sub_ndims(nclusterlist[list1], x)
            fixedvals = cnums
            w = all_flat_indices_nfixed(nclusterlist, fixedindices, fixedvals)
            for nc in w:
                cnums = ind2sub_ndims(nclusterlist, nc)
                clusterlist = np.array(cnums) + full_rnum_base
                Sinput = tcdata_centered_person[clusterlist,:]

                pca = PCA(n_components=3)
                pca.fit(Sinput)
                EVR2[nperson,nc,:] = pca.explained_variance_ratio_

    # look for the best combination based on whole set
    p,f = os.path.split(SEMresultsname)
    EVRname = os.path.join(p,'explained_variance_PCA_2.npy')
    np.save(EVRname, EVR2)

    count = np.count_nonzero(EVR2[:,:,0],axis=0)
    totalval = np.sum(EVR2[:,:,0],axis=0)
    x = np.where(count < 4)[0]
    totalval[x] = 0    # exclude values with too few samples
    nonzeroavg = totalval/(count + 1e-6)

    x = np.argmax(nonzeroavg)     # find where the average of the samles is the greatest value
    cnums = ind2sub_ndims(nclusterlist, x)

    # check cnums result------------------------
    full_rnum_base = get_overall_num(nclusterlist, list(range(nregions)), np.zeros(nregions))
    full_rnum_base = np.array(full_rnum_base).astype(int)
    clusterlist = np.array(cnums) + full_rnum_base

    EVRcheck = np.zeros((NP,3))
    for nperson in range(NP):
        tp = tplist_full[timepoint][nperson]['tp']
        tcdata_centered_person = tcdata_centered[:, tp]
        Sinput = tcdata_centered_person[clusterlist, :]
        pca = PCA(n_components=3)
        pca.fit(Sinput)
        EVRcheck[nperson, :] = pca.explained_variance_ratio_


    EVRname = os.path.join(p,'explained_variance_check.npy')
    np.save(EVRname, EVRcheck)


check_Moutput = False
if check_Moutput:
    # check it
    SEMresults_load = np.load(SEMresultsname, allow_pickle=True)
    NP = len(SEMresults_load)
    Moutput = SEMresults_load[0]['Moutput']
    nr1, nr2 = np.shape(Moutput)
    Mrecord = np.zeros((nr1,nr2,NP))
    for nperson in range(NP):
        Sinput = SEMresults_load[nperson]['Sinput']
        Soutput = SEMresults_load[nperson]['Soutput']
        Minput = SEMresults_load[nperson]['Minput']
        Moutput = SEMresults_load[nperson]['Moutput']
        Mrecord[:,:,nperson] = Moutput

    Mpos = np.zeros(np.shape(Mrecord))
    Mpos[Mrecord > 0] = 1
    Mneg = np.zeros(np.shape(Mrecord))
    Mneg[Mrecord < 0] = 1
    Mposneg = np.sum(Mpos, axis = 2) - np.sum(Mneg, axis = 2)

    columns = [name[:3] + ' in' for name in rnamelist]
    columns += ['int1 in', 'int2 in']
    rows = [name[:3] for name in rnamelist]
    rows += ['int1', 'int2']

    df = pd.DataFrame(Mposneg, columns=columns, index=rows)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    pd.options.display.float_format = '{:.0f}'.format
    print(df)

    p, f = os.path.split(SEMresultsname)
    xlname = os.path.join(p, 'Moutput_pos_neg_counts.xlsx')
    df.to_excel(xlname)
