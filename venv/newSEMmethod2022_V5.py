# sys.path.append(r'C:\Users\Stroman\PycharmProjects\pyspinalfmri3\venv')

# calculate eigenvectors and intrinsic inputs
# then calculate matrix of beta values from those
# Sinput = Minput @ Sconn
#  Sconn = Mconn @ Sconn    - eigenvalue problem
# for every time point, each connection value is determined by a scaled value of the
# sum of the eigenvectors
# so Sconn =  Meigv @ Mintrinsics
#
# new concept for V5
#   - calculate the intrinsic inputs based on the choice of beta values to test
#   Mconn is the mixing matrix of betavalues for each connection in the network
#   Sconn = Mconn @ Sconn
#   Meigv is the matrix of eigenvectors of Mconn, corrsponding to each intrinsic input
#       - these vectors are scaled so the value corresponding to the intrinsic input is equal to 1
#   Mintrinsics is the matrix of intrinsic values
#   Meigv is size [ncon x Nintrinsic]
#   Mintrinsic is size [Nintrinsic x tsize]
#   Sconn = Meigv @ Mintrinsics
#   Sinput = Minput @ Sconn
#   therefore  Mintrinsics = inv(Meigv.T @ Minput.T @ Minput @ Meigv) @ Meigv.T @ Minput.T @ Sinput

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
    keylist = dnet.keys()
    for nn in range(len(keylist)):
        if 'Unnamed' in keylist[nn]:
            dnet.pop(keylist[nn])   # remove any blank fields
    dnclusters = pd.read_excel(xls, 'nclusters')

    vintrinsic_count = 0
    fintrinsic_count = 0

    nregions = len(dnclusters)
    ntargets, ncols = dnet.shape
    nsources_max = ncols-1

    sem_region_list = []
    nclusterlist = []
    for nn in range(nregions):
        sem_region_list.append(dnclusters.loc[nn,'name'])
        cname = dnclusters.loc[nn,'name']
        if 'vintrinsic' in cname:  vintrinsic_count += 1
        if 'fintrinsic' in cname:  fintrinsic_count += 1
        entry = {'name':dnclusters.loc[nn,'name'],'nclusters':dnclusters.loc[nn,'nclusters']}
        nclusterlist.append(entry)

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

    return network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count


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


def gradients_for_betavals(Sinput, Minput, Mconn, betavals, ctarget, csource, dval, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight):
    # calculate change in error term with small changes in betavalues
    nbetavals = len(betavals)
    Mconn[ctarget, csource] = betavals
    fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
    cost = np.sum(np.abs(betavals**2))
    ssqd = err + Lweight * cost  # L2 regularization

    # gradients for betavals
    dssq_db = np.zeros(nbetavals)
    for nn in range(nbetavals):
        b = copy.deepcopy(betavals)
        b[nn] += dval
        Mconn[ctarget, csource] = b
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
        cost = np.sum(np.abs(b**2))
        ssqdp = err + Lweight * cost  # L2 regularization
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


def network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1):
    # # Sinput[:nregions,:] = Minput @ Meigv @ Mintrinsic   #  [nr x tsize] = [nr x ncon] @ [ncon x Nint] @ [Nint x tsize]
    # # Mintrinsic = np.linalg.inv(Meigv.T @ Minput.T @ Minput @ Meigv) @ Meigv.T @ Minput.T @ Sin
    # fit based on eigenvectors alone, with intrinsic values calculated
    Nintrinsic = fintrinsic_count + vintrinsic_count
    e,v = np.linalg.eig(Mconn)    # Mconn is nbeta x nbeta   where nbeta = ncon + Nintrinsic
    Meigv = np.real(v[:,-Nintrinsic:])
    # scale to make the term corresponding to each intrinsic = 1
    for aa in range(Nintrinsic):
        Meigv[:,aa] = Meigv[:,aa]/Meigv[(-Nintrinsic+aa),aa]
    M1 = Minput @ Meigv
    Mintrinsic = np.linalg.inv(M1.T @ M1) @ M1.T @ Sinput

    if fintrinsic_count > 0:
        Mintrinsic[0,:] = beta_int1*fintrinsic1

    fit = Minput @ Meigv @ Mintrinsic
    err = np.sum((Sinput - fit)**2)

    return fit, Mintrinsic, Meigv, err


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

#---------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def prep_data_sem_physio_model(networkfile, regiondataname, clusterdataname, SEMparametersname):

    outputdir, f = os.path.split(SEMparametersname)
    network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count = load_network_model_w_intrinsics(networkfile)

    fintrinsic_region = []
    if fintrinsic_count > 0:  # find which regions have fixed intrinsic input
        for nn in range(len(network)):
            sources = network[nn]['sources']
            if 'fintrinsic1' in sources:
                fintrinsic_region = network[nn]['targetnum']  # only one region should have this input

    region_data1 = np.load(regiondataname, allow_pickle=True).flat[0]
    region_properties = region_data1['region_properties']

    cluster_data = np.load(clusterdataname, allow_pickle=True).flat[0]
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
    et1 = 0
    et2 = tsize
    dtsize = tsize - 1  # for using deriviation of tc wrt time
    tplist1 = []
    nclusterstotal, tsizetotal = np.shape(tcdata)
    tcdata_centered = copy.deepcopy(tcdata)
    for nn in range(NP):
        r1 = sum(nruns_per_person[:nn])
        r2 = sum(nruns_per_person[:(nn + 1)])
        tp = []  # initialize list
        tpoints = []
        for ee2 in range(r1, r2):
            tp = list(range((ee2 * tsize), (ee2 * tsize + tsize)))
            tpoints = tpoints + tp  # concatenate lists
            temp = np.mean(tcdata[:, tp], axis=1)
            temp_mean = np.repeat(temp[:, np.newaxis], tsize, axis=1)
            tcdata_centered[:, tp] = tcdata[:, tp] - temp_mean  # center each epoch, in each person
        tplist1.append({'tp': tpoints})
    tplist_full.append(tplist1)

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
            betaname = '{}_{}'.format(source, target)
            entry = {'name': betaname, 'number': nbeta, 'pair': [source, target]}
            beta_list.append(entry)
            beta_id += [1000 * source + target]
            nbeta += 1

    ncon = nbeta - Nintrinsic

    # recorder to put intrinsic inputs at the end-------------
    beta_list2 = []
    beta_id2 = []
    x = np.where(np.array(sourcelist) < nregions)[0]
    for xx in x:
        beta_list2.append(beta_list[xx])
        beta_id2 += [beta_id[xx]]
    for sn in range(nregions, nregions + Nintrinsic):
        x = np.where(np.array(sourcelist) == sn)[0]
        for xx in x:
            beta_list2.append(beta_list[xx])
            beta_id2 += [beta_id[xx]]

    for nn in range(len(beta_list2)):
        beta_list2[nn]['number'] = nn

    beta_list = beta_list2
    beta_id = beta_id2

    beta_pair = []
    Mconn = np.zeros((nbeta, nbeta))
    count = 0
    for nn in range(len(network)):
        target = network[nn]['targetnum']
        sources = network[nn]['sourcenums']
        for mm in range(len(sources)):
            source = sources[mm]
            conn1 = beta_id.index(source * 1000 + target)
            if source >= nregions:  # intrinsic input
                conn2 = conn1
                Mconn[conn1, conn2] = 1  # set the intrinsic beta values
            else:
                x = targetnumlist.index(source)
                source_sources = network[x]['sourcenums']
                for nn in range(len(source_sources)):
                    ss1 = source_sources[nn]
                    conn2 = beta_id.index(ss1 * 1000 + source)
                    beta_pair.append([conn1, conn2])
                    count += 1
                    Mconn[conn1, conn2] = count
                    # alternative:
                    # if ss1 >= nregions: # intrinsic input to source, which will be scaled
                    #     Mconn[conn1, conn2] = 1    # keep this beta value fixed at 1
                    # else:
                    #     beta_pair.append([conn1, conn2])
                    #     count += 1
                    #     Mconn[conn1, conn2] = count

    # prep to index Mconn for updating beta values
    beta_pair = np.array(beta_pair)
    ctarget = beta_pair[:, 0]
    csource = beta_pair[:, 1]

    # setup Minput matrix--------------------------------------------------------------
    # Sconn = Mconn @ Sconn    # propagate the intrinsic inputs through the network
    # Sinput = Minput @ Mconn
    Minput = np.zeros((nregions, nbeta))  # mixing of connections to model the inputs to each region
    betanamelist = [beta_list[a]['name'] for a in range(nbeta)]
    for nn in range(len(network)):
        target = network[nn]['targetnum']
        sources = network[nn]['sourcenums']
        for mm in range(len(sources)):
            source = sources[mm]
            betaname = '{}_{}'.format(source, target)
            x = betanamelist.index(betaname)
            Minput[target, x] = 1

    # save parameters for looking at results later
    SEMparams = {'betanamelist': betanamelist, 'beta_list': beta_list, 'nruns_per_person': nruns_per_person,
                 'nclusterstotal': nclusterstotal, 'rnamelist': rnamelist, 'nregions': nregions,
                 'cluster_properties': cluster_properties, 'cluster_data': cluster_data,
                 'network': network, 'fintrinsic_count': fintrinsic_count, 'vintrinsic_count': vintrinsic_count,
                 'fintrinsic_region':fintrinsic_region, 'sem_region_list': sem_region_list,
                 'nclusterlist': nclusterlist, 'tsize': tsize, 'tplist_full': tplist_full,
                 'tcdata_centered': tcdata_centered, 'ctarget':ctarget ,'csource':csource,
                 'Mconn':Mconn, 'Minput':Minput}
    np.save(SEMparametersname, SEMparams)


#----------------------------------------------------------------------------------
# primary function--------------------------------------------------------------------
def sem_physio_model(clusterlist, paradigm_centered, SEMresultsname, SEMparametersname):
    starttime = time.ctime()

    # initialize gradient-descent parameters--------------------------------------------------------------
    initial_alpha = 1e-3
    initial_Lweight = 1e-6
    initial_dval = 0.01
    betascale = 0.0

    SEMparams = np.load(SEMparametersname, allow_pickle=True).flat[0]
    # load the data values
    betanamelist = SEMparams['betanamelist']
    beta_list = SEMparams['beta_list']
    nruns_per_person = SEMparams['nruns_per_person']
    nclusterstotal = SEMparams['nclusterstotal']
    rnamelist = SEMparams['rnamelist']
    nregions = SEMparams['nregions']
    cluster_properties = SEMparams['cluster_properties']
    cluster_data = SEMparams['cluster_data']
    network = SEMparams['network']
    fintrinsic_count = SEMparams['fintrinsic_count']
    vintrinsic_count = SEMparams['vintrinsic_count']
    sem_region_list = SEMparams['sem_region_list']
    nclusterlist = SEMparams['nclusterlist']
    tsize = SEMparams['tsize']
    tplist_full = SEMparams['tplist_full']
    tcdata_centered = SEMparams['tcdata_centered']
    ctarget = SEMparams['ctarget']
    csource = SEMparams['csource']
    fintrinsic_region = SEMparams['fintrinsic_region']
    Mconn = SEMparams['Mconn']
    Minput = SEMparams['Minput']

    ntime, NP = np.shape(tplist_full)


    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    # repeat the process for each participant-----------------------------------------------------------------
    betalimit = 3.0
    timepoint = 0
    SEMresults = []
    beta_init_record = []
    for nperson in range(NP):
        print('starting person {} at {}'.format(nperson,time.ctime()))
        tp = tplist_full[timepoint][nperson]['tp']
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
        if fintrinsic_count > 0:
            fintrinsic1 = np.array(list(paradigm_centered) * nruns_per_person[nperson])
            Sint = Sinput[fintrinsic_region,:]
            Sint = Sint - np.mean(Sint)
            b, fit, R2, total_var, res_var = pysem.general_glm(Sint[np.newaxis,:], fintrinsic1[np.newaxis,:])
            beta_int1 = b[0]

            beta_int1 = 1.0   # fix it and see what happens
        else:
            beta_int1 = 0.0

        # # this is no longer needed
        # if vintrinsic_count > 0:
        #     vintrinsics = np.zeros((vintrinsic_count, tsize_total))    # initialize unknown intrinsic with small random values

        # new concept-------------------------------------------
        # 1) test a set of betavalues
        # 2) fit to get the intrinsics that would go along with them
        # 3) see how that solution fits the data
        # e,v = np.linalg.eig(Mconn)    # Mconn is nbeta x nbeta   where nbeta = ncon + Nintrinsic
        # Meigv = v[:,-Nintrinsic:]
        # # Sinput[:nregions,:] = Minput @ Meigv @ Mintrinsic   #  [nr x tsize] = [nr x ncon] @ [ncon x Nint] @ [Nint x tsize]
        # # Mintrinsic = np.linalg.inv(Meigv.T @ Minput.T @ Minput @ Meigv) @ Meigv.T @ Minput.T @ Sin
        # Sin = Sinput[:nregions, :]
        # M1 = Minput @ Meigv
        # Mintrinsic = np.linalg.inv(M1.T @ M1) @ M1.T @ Sin
        #
        # v1 = np.mean(np.reshape(np.real(Mintrinsic[1, :]), (5, 40)), axis=0)
        # v2 = np.mean(np.reshape(np.real(Mintrinsic[2, :]), (5, 40)), axis=0)

        # initialize beta values-----------------------------------
        beta_initial = np.zeros(len(csource))
        # beta_initial = np.random.randn(len(csource))
        beta_initial = betascale*np.ones(len(csource))
        beta_init_record.append({'beta_initial':beta_initial})

        # initalize Sconn
        betavals = copy.deepcopy(beta_initial) # initialize beta values at zero
        lastgood_betavals = copy.deepcopy(betavals)

        results_record = []
        ssqd_record = []

        alpha = initial_alpha
        Lweight = initial_Lweight
        dval = initial_dval

        Mconn[ctarget,csource] = betavals

        # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
        # fit, Sconn_full = network_eigenvalue_method(Sconn_full, Minput, Mconn, ncon)

        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
        cost = np.sum(np.abs(betavals**2))
        ssqd = err + Lweight * cost  # L2 regularization
        ssqd_starting = ssqd
        ssqd_record += [ssqd]

        nitermax = 500
        alpha_limit = 1.0e-5

        iter = 0
        # vintrinsics_record = []
        converging = True
        dssq_record = np.ones(3)
        dssq_count = 0
        sequence_count = 0
        while alpha > alpha_limit  and iter < nitermax  and converging:
            iter += 1
            # gradients in betavals
            Mconn[ctarget, csource] = betavals
            fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
            dssq_db, ssqd = gradients_for_betavals(Sinput, Minput, Mconn, betavals, ctarget, csource, dval, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight)
            ssqd_record += [ssqd]

            # apply the changes
            betavals -= alpha * dssq_db

            # betavals[betavals >= betalimit] = betalimit
            # betavals[betavals <= -betalimit] = -betalimit

            Mconn[ctarget, csource] = betavals
            fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
            cost = np.sum(np.abs(betavals**2))
            ssqd_new = err + Lweight * cost  # L2 regularization

            err_total = Sinput - fit
            Smean = np.mean(Sinput)
            errmean = np.mean(err_total)
            R2total = 1 - np.sum((err_total - errmean) ** 2) / np.sum((Sinput - Smean) ** 2)

            # Sinput_sim, Soutput_sim = network_sim(Sinput_full, Soutput_full, Minput, Moutput)
            results_record.append({'Sinput': fit, 'Mintrinsic': Mintrinsic, 'Meigv':Meigv})

            if ssqd_new >= ssqd:
                alpha *= 0.5
                # revert back to last good values
                betavals = copy.deepcopy(lastgood_betavals)
                dssqd = ssqd - ssqd_new
                dssq_record = np.ones(3)  # reset the count
                dssq_count = 0
                sequence_count = 0
                print('beta vals:  iter {} alpha {:.3e}  delta ssq > 0  - no update'.format(iter, alpha))
            else:
                # save the good values
                lastgood_betavals = copy.deepcopy(betavals)

                dssqd = ssqd - ssqd_new
                ssqd = ssqd_new

                sequence_count += 1
                if sequence_count > 5:
                    alpha *= 1.5
                    sequence_count = 0

                dssq_count += 1
                dssq_count = np.mod(dssq_count, 3)
                # dssq_record[dssq_count] = 100.0 * dssqd / ssqd_starting
                dssq_record[dssq_count] = dssqd
                if np.max(dssq_record) < 0.1:  converging = False

            print('beta vals:  iter {} alpha {:.3e}  delta ssq {:.4f}  relative: {:.1f} percent  '
                  'R2 {:.3f}'.format(iter,alpha, -dssqd,100.0 * ssqd / ssqd_starting, R2total))
            # now repeat it ...


        # fit the results now to determine output signaling from each region
        Mconn[ctarget, csource] = betavals
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
        Sconn = Meigv @ Mintrinsic    # signalling over each connection

        regionlist = [0, 7]
        results_text = display_SEM_results_1person(nperson, Sinput, fit, regionlist, nruns, tsize, windowlist=[24, 25])

        entry = {'Sinput':Sinput, 'Sconn':Sconn, 'beta_int1':beta_int1, 'Mconn':Mconn, 'Minput':Minput,
                 'rtext1':results_text[0], 'rtext2':results_text[1], 'R2total':R2total, 'Mintrinsic':Mintrinsic,
                 'Meigv':Meigv, 'betavals':betavals}
        SEMresults.append(copy.deepcopy(entry))

        stoptime = time.ctime()

    np.save(SEMresultsname, SEMresults)
    print('finished SEM at {}'.format(time.ctime()))
    print('     started at {}'.format(starttime))



#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
def display_SEM_results(settingsfile, SEMparametersname, SEMresultsname, person_list = [41, 48, 32, 21, 10]):
    # reload parameters if needed--------------------------------------------------------
    settingsfile = r'C:\Users\Stroman\PycharmProjects\pyspinalfmri3\venv\base_settings_file.npy'
    settings = np.load(settingsfile, allow_pickle=True).flat[0]
    covariates1 = settings['GRPcharacteristicsvalues'][0]  # gender
    covariates2 = settings['GRPcharacteristicsvalues'][1].astype(float)  # painrating

    SEMparams = np.load(SEMparametersname, allow_pickle=True).flat[0]
    # SEMparams = {'betanamelist': betanamelist, 'beta_list':beta_list, 'nruns_per_person': nruns_per_person,
    #              'nclusterstotal':nclusterstotal, 'rnamelist':rnamelist, 'nregions':nregions,
    #             'cluster_properties':cluster_properties, 'cluster_data':cluster_data,
    #              'network':network, 'fintrinsic_count':fintrinsic_count, 'vintrinsic_count':vintrinsic_count,
    #              'sem_region_list':sem_region_list, 'nclusterlist':nclusterlist, 'tsize':tsize}

    network = SEMparams['network']
    beta_list = SEMparams['beta_list']
    betanamelist = SEMparams['betanamelist']
    nruns_per_person = SEMparams['nruns_per_person']
    rnamelist = SEMparams['rnamelist']
    fintrinsic_count = SEMparams['fintrinsic_count']
    vintrinsic_count = SEMparams['vintrinsic_count']
    nclusterlist = SEMparams['nclusterlist']
    tplist_full = SEMparams['tplist_full']
    tcdata_centered = SEMparams['tcdata_centered']
    Nintrinsic = fintrinsic_count + vintrinsic_count

    # end of reloading parameters if needed--------------------------------------------------------

    # load the SEM results
    SEMresults_load = np.load(SEMresultsname, allow_pickle=True)

    # for nperson in range(NP)
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
        Mintrinsic = SEMresults_load[nperson]['Mintrinsic']
        Meigv = SEMresults_load[nperson]['Meigv']
        beta_int1 = SEMresults_load[nperson]['beta_int1']
        betavals = SEMresults_load[nperson]['betavals']
        R2total = SEMresults_load[nperson]['R2total']

        # Mconn[ctarget, csource] = betavals
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                                 vintrinsic_count, beta_int1, fintrinsic1)
        # Sconn = Meigv @ Mintrinsic  # signalling over each connection

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

        # intrinsics
        tc = Mintrinsic[0,:]
        tc0 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = Mintrinsic[fintrinsic_count,:]
        tc1 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = Mintrinsic[fintrinsic_count+1,:]
        tc2 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)

        plt.close(4)
        fig = plt.figure(4, figsize=(12.5, 3.5), dpi=100)
        plt.subplot(3,1,1)
        plt.plot(tc0, '-og')
        plt.subplot(3,1,2)
        plt.plot(tc1, '-og')
        plt.subplot(3,1,3)
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
    R2totalrecord = np.zeros(NP)
    for nperson in range(NP):
        Sinput = SEMresults_load[nperson]['Sinput']
        Sconn = SEMresults_load[nperson]['Sconn']
        Minput = SEMresults_load[nperson]['Minput']
        Mconn = SEMresults_load[nperson]['Mconn']
        R2total = SEMresults_load[nperson]['R2total']
        Mrecord[:,:,nperson] = Mconn
        R2totalrecord[nperson] = R2total

    Mrecord_mean = np.mean(Mrecord,axis=2)
    M_T = np.mean(Mrecord,axis=2)/(np.std(Mrecord,axis=2)/np.sqrt(NP)  + 1e-20)
    M_T[np.abs(M_T) > 1e10] = 0

    mpos = np.zeros((nbeta,nbeta,NP))
    mneg = np.zeros((nbeta,nbeta,NP))
    mpos[Mrecord > 0] = 1
    mneg[Mrecord < 0] = 1
    mpos = np.sum(mpos, axis = 2)
    mneg= np.sum(mneg, axis = 2)
    Mcount = mpos - mneg
    x = np.argmax(np.abs(Mcount[:ncon,:ncon]))
    aa,bb = np.unravel_index(x,np.shape(Mcount[:ncon,:ncon]))

    # separate by sex
    g1 = np.where(covariates1 == 'Female')[0]
    g2 = np.where(covariates1 == 'Male')[0]

    M1 = Mrecord[:,:,g1]
    M2 = Mrecord[:,:,g2]
    Tsexdiff = np.zeros((ncon,ncon))
    psexdiff = np.zeros((ncon,ncon))
    for aa in range(ncon):
        for bb in range(ncon):
            a = M1[aa,bb,:]
            b = M2[aa,bb,:]
            if (np.var(a) > 0)  &  (np.var(b) > 0):
                t, p = stats.ttest_ind(a, b, equal_var=False)
                Tsexdiff[aa,bb] = t
                psexdiff[aa,bb] = p

    aa,bb = np.where(np.abs(Tsexdiff) > 2.0)
    for xx in range(len(aa)):
        c1 = betanamelist[aa[xx]]
        c2 = betanamelist[bb[xx]]
        conn1 = beta_list[aa[xx]]['pair']
        conn2 = beta_list[bb[xx]]['pair']
        T = Tsexdiff[aa[xx],bb[xx]]
        print('difference in {}-{} effect on {}-{}   T = {:.2f}'.format(rnamelist[conn2[0]],rnamelist[conn2[1]],rnamelist[conn1[0]], rnamelist[conn1[1]],T))

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

    plt.close(36)
    fig = plt.figure(36), plt.plot(covariates2[g1], m[g1], 'or')
    bf, fitf, R2f, total_var, res_var = pysem.general_glm(m[np.newaxis, g1], covariates2[np.newaxis, g1])
    plt.plot(covariates2[g1], fitf[0, :], '-r')

    plt.plot(covariates2[g2], m[g2], 'ob')
    bm, fitm, R2m, total_var, res_var = pysem.general_glm(m[np.newaxis, g2], covariates2[np.newaxis, g2])
    plt.plot(covariates2[g2], fitm[0, :], '-b')


#--------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
def show_Mconn_properties(settingsfile, SEMparametersname, SEMresultsname):
    settings = np.load(settingsfile, allow_pickle=True).flat[0]
    covariates1 = settings['GRPcharacteristicsvalues'][0]  # gender
    covariates2 = settings['GRPcharacteristicsvalues'][1].astype(float)  # painrating

    SEMparams = np.load(SEMparametersname, allow_pickle=True).flat[0]
    network = SEMparams['network']
    beta_list = SEMparams['beta_list']
    betanamelist = SEMparams['betanamelist']
    nruns_per_person = SEMparams['nruns_per_person']
    rnamelist = SEMparams['rnamelist']
    fintrinsic_count = SEMparams['fintrinsic_count']
    vintrinsic_count = SEMparams['vintrinsic_count']
    nclusterlist = SEMparams['nclusterlist']
    tplist_full = SEMparams['tplist_full']
    tcdata_centered = SEMparams['tcdata_centered']
    Nintrinsic = fintrinsic_count + vintrinsic_count
    # end of reloading parameters-------------------------------------------------------

    # load the SEM results
    SEMresults_load = np.load(SEMresultsname, allow_pickle=True)

    # for nperson in range(NP)
    NP = len(SEMresults_load)
    resultscheck = np.zeros((NP, 4))
    nbeta, tsize_full = np.shape(SEMresults_load[0]['Sconn'])
    ncon = nbeta - Nintrinsic

    Mrecord = np.zeros((nbeta, nbeta, NP))
    R2totalrecord = np.zeros(NP)
    for nperson in range(NP):
        Sinput = SEMresults_load[nperson]['Sinput']
        Sconn = SEMresults_load[nperson]['Sconn']
        Minput = SEMresults_load[nperson]['Minput']
        Mconn = SEMresults_load[nperson]['Mconn']
        R2total = SEMresults_load[nperson]['R2total']
        Mrecord[:, :, nperson] = Mconn
        R2totalrecord[nperson] = R2total

    Mrecord_mean = np.mean(Mrecord, axis=2)
    M_T = np.mean(Mrecord, axis=2) / (np.std(Mrecord, axis=2) / np.sqrt(NP) + 1e-20)
    M_T[np.abs(M_T) > 1e10] = 0

    mpos = np.zeros((nbeta, nbeta, NP))
    mneg = np.zeros((nbeta, nbeta, NP))
    mpos[Mrecord > 0] = 1
    mneg[Mrecord < 0] = 1
    mpos = np.sum(mpos, axis=2)
    mneg = np.sum(mneg, axis=2)
    Mcount = mpos - mneg
    x = np.argmax(np.abs(Mcount[:ncon, :ncon]))
    aa, bb = np.unravel_index(x, np.shape(Mcount[:ncon, :ncon]))

    # separate by sex
    g1 = np.where(covariates1 == 'Female')[0]
    g2 = np.where(covariates1 == 'Male')[0]

    M1 = Mrecord[:, :, g1]
    M2 = Mrecord[:, :, g2]
    Tsexdiff = np.zeros((ncon, ncon))
    psexdiff = np.zeros((ncon, ncon))
    for aa in range(ncon):
        for bb in range(ncon):
            a = M1[aa, bb, :]
            b = M2[aa, bb, :]
            if (np.var(a) > 0) & (np.var(b) > 0):
                t, p = stats.ttest_ind(a, b, equal_var=False)
                Tsexdiff[aa, bb] = t
                psexdiff[aa, bb] = p

    aa, bb = np.where(np.abs(Tsexdiff) > 2.0)
    for xx in range(len(aa)):
        c1 = betanamelist[aa[xx]]
        c2 = betanamelist[bb[xx]]
        conn1 = beta_list[aa[xx]]['pair']
        conn2 = beta_list[bb[xx]]['pair']
        T = Tsexdiff[aa[xx], bb[xx]]
        print('difference in {}-{} effect on {}-{}   T = {:.2f}'.format(rnamelist[conn2[0]], rnamelist[conn2[1]],
                                                                        rnamelist[conn1[0]], rnamelist[conn1[1]],
                                                                        T))
    Rrecord = np.zeros((ncon, ncon))
    R2record = np.zeros((ncon, ncon))
    for aa in range(ncon):
        for bb in range(ncon):
            m = Mrecord[aa, bb, :]
            if np.var(m) > 0:
                R = np.corrcoef(covariates2[g1], m[g1])
                Rrecord[aa, bb] = R[0, 1]
                G = np.concatenate((covariates2[np.newaxis, g1], np.ones((1,len(g1)))), axis = 0)
                b, fit, R2, total_var, res_var = pysem.general_glm(m[np.newaxis, g1], G)
                R2record[aa, bb] = R2

    x = np.argsort(-np.abs(Rrecord.flatten()))
    number = 0
    aa, bb = np.unravel_index(x[number], np.shape(Rrecord))
    # aa,bb = (6,7)
    m = Mrecord[aa, bb, :]
    plt.close(35)
    fig = plt.figure(35), plt.plot(covariates2, m, 'ob')
    G = np.concatenate((covariates2[np.newaxis, :], np.ones((1,NP))), axis = 0)
    b, fit, R2, total_var, res_var = pysem.general_glm(m[np.newaxis, :], G)
    plt.plot(covariates2, fit[0, :], '-b')

    plt.close(36)
    fig = plt.figure(36), plt.plot(covariates2[g1], m[g1], 'or')
    G = np.concatenate((covariates2[np.newaxis, g1], np.ones((1,len(g1)))), axis = 0)
    bf, fitf, R2f, total_var, res_var = pysem.general_glm(m[np.newaxis, g1], G)
    plt.plot(covariates2[g1], fitf[0, :], '-r')

    plt.plot(covariates2[g2], m[g2], 'ob')
    G = np.concatenate((covariates2[np.newaxis, g2], np.ones((1,len(g2)))), axis = 0)
    bm, fitm, R2m, total_var, res_var = pysem.general_glm(m[np.newaxis, g2], G)
    plt.plot(covariates2[g2], fitm[0, :], '-b')


    columns = [name + ' in' for name in betanamelist]
    rows = [name for name in betanamelist]
    df = pd.DataFrame(Mconn, columns=columns, index=rows)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    pd.options.display.float_format = '{:.2f}'.format
    print(df)

    p, f = os.path.split(SEMresultsname)
    xlname = os.path.join(p, 'Mconn.xlsx')
    df.to_excel(xlname)


def write_Mconn_values(Mconn, betanamelist, rnamelist, beta_list):
    nregions = len(rnamelist)
    nr1, nr2 = np.shape(Mconn)
    text_record = []
    for n1 in range(nr1):
        tname = betanamelist[n1]
        tpair = beta_list[n1]['pair']
        if tpair[0] >= nregions:
            ts = 'int{}'.format(tpair[0]-nregions)
        else:
            ts = rnamelist[tpair[0]]
        tt = rnamelist[tpair[1]]
        text1 = '{}-{} input from '.format(ts[:3],tt[:3])
        for n2 in range(nr2):
            if np.abs(Mconn[n1,n2]) > 0:
                sname = betanamelist[n2]
                spair = beta_list[n2]['pair']
                if spair[0] >= nregions:
                    ss = 'int{}'.format(spair[0]-nregions)
                else:
                    ss = rnamelist[spair[0]]
                st = rnamelist[spair[1]]
                texts = '{}-{} {:.2f}  '.format(ss[:3],st[:3], Mconn[n1,n2])
                text1 += texts
        print(text1)
        text_record += [text1]
    return text_record


def estimate_best_connections(Nintrinsics, nclusterlist, tplist_full, tcdata_centered, nruns_per_person):
    # look for the combination of clusters that are best explained by Nintrinsics terms
    nregions = len(nclusterlist)   # initial set to test
    ncombo_set = np.floor(nregions/2).astype(int)
    nleaveout = nregions-ncombo_set
    nclusterlist = np.array(nclusterlist)

    ntime, NP = np.shape(tplist_full)

    list1 = list(range(ncombo_set))
    ncombinations = np.prod(nclusterlist)
    ncombinations1 = np.prod(nclusterlist[list1])

    EVR1 = np.zeros((NP,ncombinations1,Nintrinsics))
    EVR2 = np.zeros((NP,ncombinations,Nintrinsics))
    nkeep = 100
    xlist = np.zeros((NP,nkeep))  # keep a record of best 1st round picks for each person

    timepoint = 0
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
        evr_values = np.sum(EVR1[nperson,:,:],axis=1)
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

                pca = PCA(n_components=Nintrinsics)
                pca.fit(Sinput)
                EVR2[nperson,nc,:] = pca.explained_variance_ratio_

    # look for the best combination based on whole set
    p,f = os.path.split(SEMresultsname)
    EVRname = os.path.join(p,'explained_variance_PCA.npy')
    np.save(EVRname, EVR2)

    EVR2sum = np.sum(EVR2[:,:,:2], axis = 2)
    count = np.count_nonzero(EVR2sum,axis=0)
    totalval = np.sum(EVR2sum,axis=0)
    x = np.where(count < 4)[0]
    totalval[x] = 0    # exclude values with too few samples
    nonzeroavg = totalval/(count + 1e-6)

    x = np.argsort(-nonzeroavg)     # find where the average of the samples is the greatest value
    cnums = ind2sub_ndims(nclusterlist, x[0])

    # check cnums result------------------------
    full_rnum_base = get_overall_num(nclusterlist, list(range(nregions)), np.zeros(nregions))
    full_rnum_base = np.array(full_rnum_base).astype(int)
    clusterlist = np.array(cnums) + full_rnum_base

    EVRcheck = np.zeros((NP,Nintrinsics))
    for nperson in range(NP):
        tp = tplist_full[timepoint][nperson]['tp']
        tcdata_centered_person = tcdata_centered[:, tp]
        Sinput = tcdata_centered_person[clusterlist, :]
        pca = PCA(n_components=Nintrinsics)
        pca.fit(Sinput)
        EVRcheck[nperson, :] = pca.explained_variance_ratio_

    EVRname = os.path.join(p,'explained_variance_check.npy')
    np.save(EVRname, EVRcheck)




def display_Mconn_properties(SEMresultsname, rnamelist, betanamelist):
    SEMresults_load = np.load(SEMresultsname, allow_pickle=True)
    NP = len(SEMresults_load)
    Mconn = SEMresults_load[0]['Mconn']
    nr1, nr2 = np.shape(Mconn)
    Mrecord = np.zeros((nr1,nr2,NP))
    for nperson in range(NP):
        Sinput = SEMresults_load[nperson]['Sinput']
        Sconn= SEMresults_load[nperson]['Sconn']
        Minput = SEMresults_load[nperson]['Minput']
        Mconn = SEMresults_load[nperson]['Mconn']
        Mrecord[:,:,nperson] = Mconn

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


def SEM_dsource_dtarget(network, nclusterlist, tplist_full, nclusterstotal, tcdata_centered, nruns_per_person, tsize):
    # compute grid of dSsource/dStarget--------------------------------------------------------------------
    timepoint = 0
    NP = len(nruns_per_person)
    nclusterstotal, tsizetotal = np.shape(tcdata_centered)
    dtcdata_centered = np.zeros((nclusterstotal, tsizetotal))

    for nperson in range(NP):
        tp = tplist_full[timepoint][nperson]['tp']
        nruns = nruns_per_person[nperson]
        for ee2 in range(nruns):
            t1 = ee2*tsize
            t2 = (ee2+1)*tsize
            tp1 = tp[t1:t2]
            dtcdata_centered[:, tp1[1:]] = np.diff(tcdata_centered[:, tp1])

    dSdSgrid = np.zeros((nclusterstotal, nclusterstotal, NP, 2))
    for nn in range(NP):
        tp = tplist_full[timepoint][nn]['tp']
        tsize_total = len(tp)
        for ss in range(nclusterstotal):
            dss = dtcdata_centered[ss, tp]
            for tt in range(nclusterstotal):
                dtt = dtcdata_centered[tt, tp]
                dsdt = dss / (dtt + 1.0e-20)
                stdval = np.std(dsdt)
                dsdt[np.abs(dsdt) > 3.0 * stdval] = 0.0
                dSsdSt = np.mean(dsdt)
                dSsdSt_sem = np.std(dsdt) / np.sqrt(tsize_total)
                dSdSgrid[ss, tt, nn, 0] = dSsdSt
                dSdSgrid[ss, tt, nn, 1] = dSsdSt_sem
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
    Tcount = np.sum(Tlim, axis=2)  # count of how many people have significant estimated beta values for each connection

    return dtcdata_centered, T, Tcount



def display_SEM_results_1person(nperson, Sinput, fit, regionlist, nruns, tsize, windowlist = [24]):
    # show results
    err = Sinput - fit
    Smean = np.mean(Sinput)
    errmean = np.mean(err)
    R2total = 1 - np.sum((err-errmean)**2)/np.sum((Sinput-Smean)**2)
    tsize_total = nruns*tsize

    show_nregions = len(regionlist)
    if len(windowlist) < show_nregions:
        w = windowlist[0]
        windowlist = [w+a for a in range(show_nregions)]

    results_text_output = []
    for rr in range(show_nregions):
        regionnum = regionlist[rr]
        windownum = windowlist[rr]

        tc = Sinput[regionnum,:]
        tc1 = np.mean(np.reshape(tc, (nruns, tsize)), axis=0)
        tc = fit[regionnum,:]
        tcf1 = np.mean(np.reshape(tc, (nruns, tsize)), axis=0)

        plt.close(windownum)
        fig = plt.figure(windownum, figsize=(12.5, 3.5), dpi=100)
        plt.plot(range(tsize),tc1,'-ob')
        plt.plot(range(tsize),tcf1,'-xr')

        R1 = np.corrcoef(Sinput[regionnum, :], fit[regionnum, :])
        Z1 = np.arctanh(R1[0, 1]) * np.sqrt(tsize_total - 3)
        results_text = 'person {} region {}   R = {:.3f}  Z = {:.2f}'.format(nperson, regionnum, R1[0, 1], Z1)
        print(results_text)

        results_text_output += [results_text]

    return results_text_output


def display_matrix(M,columntitles,rowtitles):

    # columns = [name[:3] +' in' for name in betanamelist]
    # rows = [name[:3] for name in betanamelist]

    df = pd.DataFrame(M,columns = columntitles, index = rorowtitlesws)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    pd.options.display.float_format = '{:.2f}'.format
    print(df)


def run_SEM():
    # main function
    settingsfile = r'C:\Users\Stroman\PycharmProjects\pyspinalfmri3\venv\base_settings_file.npy'

    outputdir = r'D:/threat_safety_python/SEMresults'
    SEMresultsname = os.path.join(outputdir, 'SEMresults_newmethod_5.npy')
    SEMparametersname = os.path.join(outputdir, 'SEMparameters_newmethod_5.npy')
    networkfile = r'D:/threat_safety_python/network_model3_with_3intrinsics.xlsx'

    # load paradigm data--------------------------------------------------------------------
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

    regiondataname = r'D:/threat_safety_python/SEMresults/threat_safety_regiondata_allthreat55.npy'
    clusterdataname = r'D:/threat_safety_python/SEMresults/threat_safety_clusterdata.npy'

    clusterlist = [4, 9, 14, 15, 20, 28, 32, 35, 41, 47]   # picked by PCA method below
    clusterlist = [4, 5, 14, 15, 20, 28, 32, 35, 41, 47]   # picked by PCA method with 2 intrinsics
    clusterlist = [1, 5, 14, 15, 20, 28, 32, 35, 41, 47]  # picked 2nd by PCA method with 2 intrinsics

    # rnamelist = ['C6RD',  'DRt', 'Hypothalamus','LC', 'NGC',
    #                'NRM', 'NTS', 'PAG', 'PBN', 'Thalamus']
    # from other SEM results:
    # cnums = [4, 3, 2, 0, 3, 3, 0, 3, 3, 1]
    # C6RD 4,  DRt  3, Hypothalamus 2 or 1,  PBN 3, NGC 3
    # NRM 3,  NTS 0, LC 0, PAG  3 or 4 or 2, Thalamus 1 or 3
    # full_rnum_base = get_overall_num(nclusterlist, list(range(nregions)), np.zeros(nregions))
    # full_rnum_base = np.array(full_rnum_base).astype(int)
    # clusterlist = np.array(cnums) + full_rnum_base

    clusterlist = [4,8,12,15,23,28,30,38,43,46]   # from prior SEM analysis

    prep_data_sem_physio_model(networkfile, regiondataname, clusterdataname, SEMparametersname)
    output = sem_physio_model(clusterlist, paradigm_centered, SEMresultsname, SEMparametersname)
