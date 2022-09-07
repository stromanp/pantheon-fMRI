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
import random
import draw_sapm_diagram2 as dsd2
import copy
import multiprocessing as mp
# import parallel_functions as pf

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
    # include beta_int1
    nbetavals = len(betavals)
    Mconn[ctarget, csource] = betavals
    fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
    # cost = np.sum(np.abs(betavals**2))  # L2 regularization
    cost = np.sum(np.abs(betavals))  # L1 regularization
    ssqd = err + Lweight * cost

    # gradients for betavals
    dssq_db = np.zeros(nbetavals)
    for nn in range(nbetavals):
        b = copy.deepcopy(betavals)
        b[nn] += dval
        Mconn[ctarget, csource] = b
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
        # cost = np.sum(np.abs(b**2))  # L2 regularization
        cost = np.sum(np.abs(b))  # L1 regularization
        ssqdp = err + Lweight * cost
        dssq_db[nn] = (ssqdp - ssqd) / dval

    # gradients for beta_int1
    b = copy.deepcopy(beta_int1)
    b += dval
    Mconn[ctarget, csource] = betavals
    fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, b, fintrinsic1)
    # cost = np.sum(np.abs(b**2)) # L2 regularization
    cost = np.sum(np.abs(b))  # L1 regularization
    ssqdp = err + Lweight * cost
    dssq_dbeta1 = (ssqdp - ssqd) / dval

    return dssq_db, ssqd, dssq_dbeta1


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

#---------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def prep_data_sem_physio_model(networkfile, regiondataname, clusterdataname, SAPMparametersname, fullgroup = False):

    outputdir, f = os.path.split(SAPMparametersname)
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
    epoch = tsize
    timepoint = np.floor(tsize/2)

    tplist_full = []
    if epoch >= tsize:
        et1 = 0
        et2 = tsize
    else:
        et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
        et2 = (timepoint + np.floor(epoch / 2)).astype(int)
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
            # tp = list(range((ee2 * tsize), (ee2 * tsize + tsize)))
            tp = list(range((ee2*tsize+et1),(ee2*tsize+et2)))
            tpoints = tpoints + tp  # concatenate lists
            temp = np.mean(tcdata[:, tp], axis=1)
            temp_mean = np.repeat(temp[:, np.newaxis], epoch, axis=1)
            tcdata_centered[:, tp] = tcdata[:, tp] - temp_mean  # center each epoch, in each person
        tplist1.append({'tp': tpoints})
    tplist_full.append(tplist1)


    if fullgroup:
        # special case to fit the full group together
        # treat the whole group like one person
        tpgroup_full = []
        tpgroup = []
        tp = []
        for nn in range(NP):
            tp += tplist_full[0][nn]['tp']   # concatenate timepoint lists
        tpgroup.append({'tp': tp})
        tpgroup_full.append(tpgroup)
        tplist_full = copy.deepcopy(tpgroup_full)
        nruns_per_person = [np.sum(nruns_per_person)]


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

    # reorder to put intrinsic inputs at the end-------------
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

    latent_flag = np.zeros(len(ctarget))
    found_latent_list = []
    for nn in range(len(ctarget)):
        if csource[nn] >= ncon:
            if not csource[nn] in found_latent_list:
                latent_flag[nn] = 1
                found_latent_list += [csource[nn]]

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
    SAPMparams = {'betanamelist': betanamelist, 'beta_list': beta_list, 'nruns_per_person': nruns_per_person,
                 'nclusterstotal': nclusterstotal, 'rnamelist': rnamelist, 'nregions': nregions,
                 'cluster_properties': cluster_properties, 'cluster_data': cluster_data,
                 'network': network, 'fintrinsic_count': fintrinsic_count, 'vintrinsic_count': vintrinsic_count,
                 'fintrinsic_region':fintrinsic_region, 'sem_region_list': sem_region_list,
                 'nclusterlist': nclusterlist, 'tsize': tsize, 'tplist_full': tplist_full,
                 'tcdata_centered': tcdata_centered, 'ctarget':ctarget ,'csource':csource,
                 'Mconn':Mconn, 'Minput':Minput, 'timepoint':timepoint, 'epoch':epoch, 'latent_flag':latent_flag}
    print('saving SAPM parameters to file: {}'.format(SAPMparametersname))
    np.save(SAPMparametersname, SAPMparams)


#----------------------------------------------------------------------------------
# primary function--------------------------------------------------------------------
def sem_physio_model(clusterlist, fintrinsic_base, SAPMresultsname, SAPMparametersname, fixed_beta_vals = [], verbose = True):
    starttime = time.ctime()

    # initialize gradient-descent parameters--------------------------------------------------------------
    initial_alpha = 1e-3
    initial_Lweight = 1e-4
    initial_dval = 0.01
    betascale = 0.0

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
    # repeat the process for each participant-----------------------------------------------------------------
    betalimit = 3.0
    epochnum = 0
    SAPMresults = []
    beta_init_record = []
    for nperson in range(NP):
        print('starting person {} at {}'.format(nperson,time.ctime()))
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
                et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
                et2 = (timepoint + np.floor(epoch / 2)).astype(int)

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

        lastgood_beta_int1 = copy.deepcopy(beta_int1)

        # initialize beta values-----------------------------------
        beta_initial = np.zeros(len(csource))
        # beta_initial = np.random.randn(len(csource))
        beta_initial = betascale*np.ones(len(csource))

        # limit the beta values related to intrinsic inputs to positive values
        for aa in range(len(beta_initial)):
            if latent_flag[aa] > 0:
                # if beta_initial[aa] < 0:  beta_initial[aa] = 0.0
                beta_initial[aa] = 1.0

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
        # cost = np.sum(np.abs(betavals**2)) # L2 regularization
        cost = np.sum(np.abs(betavals))  # L1 regularization
        ssqd = err + Lweight * cost
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
        while alpha > alpha_limit and iter < nitermax and converging:
            iter += 1
            # gradients in betavals and beta_int1
            Mconn[ctarget, csource] = betavals
            fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                                     vintrinsic_count, beta_int1, fintrinsic1)
            dssq_db, ssqd, dssq_dbeta1 = gradients_for_betavals(Sinput, Minput, Mconn, betavals, ctarget, csource, dval,
                                                                fintrinsic_count, vintrinsic_count, beta_int1,
                                                                fintrinsic1, Lweight)
            ssqd_record += [ssqd]

            # fix some beta values at zero, if specified
            if len(fixed_beta_vals) > 0:
                dssq_db[fixed_beta_vals] = 0

            # apply the changes
            betavals -= alpha * dssq_db
            beta_int1 -= alpha * dssq_dbeta1

            # limit the beta values related to intrinsic inputs to positive values
            for aa in range(len(betavals)):
                if latent_flag[aa] > 0:
                    # if betavals[aa] < 0:  betavals[aa] = 0.0
                    betavals[aa] = 1.0

            # betavals[betavals >= betalimit] = betalimit
            # betavals[betavals <= -betalimit] = -betalimit

            Mconn[ctarget, csource] = betavals
            fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                                     vintrinsic_count, beta_int1, fintrinsic1)
            # cost = np.sum(np.abs(betavals**2))  # L2 regularization
            cost = np.sum(np.abs(betavals))  # L1 regularization
            ssqd_new = err + Lweight * cost

            err_total = Sinput - fit
            Smean = np.mean(Sinput)
            errmean = np.mean(err_total)
            R2total = 1 - np.sum((err_total - errmean) ** 2) / np.sum((Sinput - Smean) ** 2)

            # Sinput_sim, Soutput_sim = network_sim(Sinput_full, Soutput_full, Minput, Moutput)
            results_record.append({'Sinput': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})

            if ssqd_new >= ssqd:
                alpha *= 0.5
                # revert back to last good values
                betavals = copy.deepcopy(lastgood_betavals)
                beta_int1 = copy.deepcopy(lastgood_beta_int1)
                dssqd = ssqd - ssqd_new
                dssq_record = np.ones(3)  # reset the count
                dssq_count = 0
                sequence_count = 0
                print('beta vals:  iter {} alpha {:.3e}  delta ssq > 0  - no update'.format(iter, alpha))
            else:
                # save the good values
                lastgood_betavals = copy.deepcopy(betavals)
                lastgood_beta_int1 = copy.deepcopy(beta_int1)

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
                  'R2 {:.3f}'.format(iter, alpha, -dssqd, 100.0 * ssqd / ssqd_starting, R2total))
            # now repeat it ...

        # fit the results now to determine output signaling from each region
        Mconn[ctarget, csource] = betavals
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
        Sconn = Meigv @ Mintrinsic    # signalling over each connection

        # regionlist = [0, 7]
        # if verbose:
        #     results_text = display_SEM_results_1person(nperson, Sinput, fit, regionlist, nruns, epoch, windowlist=[24, 25])
        # else:
        #     results_text = ['silent mode','silent mode']

        entry = {'Sinput':Sinput, 'Sconn':Sconn, 'beta_int1':beta_int1, 'Mconn':Mconn, 'Minput':Minput,
                 'R2total':R2total, 'Mintrinsic':Mintrinsic,
                 'Meigv':Meigv, 'betavals':betavals, 'fintrinsic1':fintrinsic1, 'clusterlist':clusterlist,
                 'fintrinsic_base':fintrinsic_base}
        SAPMresults.append(copy.deepcopy(entry))

        stoptime = time.ctime()

    np.save(SAPMresultsname, SAPMresults)
    print('finished SAPM at {}'.format(time.ctime()))
    print('     started at {}'.format(starttime))

    return SAPMresultsname

#
# #
# # gradient descent method per person
# Does this need to be in a separate module?
#------------------------------------------------------------------------
#------------------------------------------------------------------------
# gradient descent method per person
def gradient_descent_per_person(data):
    # print('running gradient_descent_per_person (in pysapm.py)')
    nperson = data['nperson']
    tsize = data['tsize']
    tplist_full = data['tplist_full']
    nruns_per_person = data['nruns_per_person']
    nclusterlist = data['nclusterlist']
    Minput = data['Minput']
    fintrinsic_count = data['fintrinsic_count']
    fintrinsic_region = data['fintrinsic_region']
    vintrinsic_count = data['vintrinsic_count']
    epoch = data['epoch']
    timepoint = data['timepoint']
    tcdata_centered = data['tcdata_centered']
    ctarget = data['ctarget']
    csource = data['csource']
    latent_flag = data['latent_flag']
    Mconn = data['Mconn']
    ntime = data['ntime']
    NP = data['NP']
    component_data = data['component_data']
    average_data = data['average_data']
    epochnum = data['epochnum']
    fintrinsic_base = data['fintrinsic_base']
    PCloadings = data['PCloadings']
    initial_alpha = data['initial_alpha']
    initial_Lweight = data['initial_Lweight']
    initial_dval = data['initial_dval']
    alpha_limit = data['alpha_limit']
    nitermax = data['nitermax']
    fixed_beta_vals = data['fixed_beta_vals']
    verbose = data['verbose']

    #
    # tsize = SAPMparams['tsize']
    # tplist_full = SAPMparams['tplist_full']
    # nruns_per_person = SAPMparams['nruns_per_person']
    # nclusterlist = SAPMparams['nclusterlist']
    # Minput = SAPMparams['Minput']
    # fintrinsic_count = SAPMparams['fintrinsic_count']
    # fintrinsic_region = SAPMparams['fintrinsic_region']
    # vintrinsic_count = SAPMparams['vintrinsic_count']
    # epoch = SAPMparams['epoch']
    # timepoint = SAPMparams['timepoint']
    # tcdata_centered = SAPMparams['tcdata_centered']
    # ctarget = SAPMparams['ctarget']
    # csource = SAPMparams['csource']
    # latent_flag = SAPMparams['latent_flag']
    # Mconn = SAPMparams['Mconn']
    #
    # ntime, NP = np.shape(tplist_full)

    # component_data = PCparams['components']
    # average_data = PCparams['average']

    # if verbose: print('starting person {} at {}'.format(nperson, time.ctime()))
    tp = tplist_full[epochnum][nperson]['tp']
    tsize_total = len(tp)
    nruns = nruns_per_person[nperson]

    # PCparams = {'components': component_data, 'loadings': original_loadings}
    Sinput = []
    for rval in range(len(nclusterlist)):
        r1 = np.sum(nclusterlist[:rval]).astype(int)
        r2 = np.sum(nclusterlist[:(rval + 1)]).astype(int)
        L = PCloadings[r1:r2]
        L = np.repeat(L[:, np.newaxis], tsize_total, axis=1)
        C = component_data[r1:r2, tp]
        tc1 = np.sum(L * C, axis=0) + average_data[r1, tp]
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
            et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
            et2 = (timepoint + np.floor(epoch / 2)).astype(int)

        ftemp = fintrinsic_base[et1:et2]
        fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])
        if np.var(ftemp) > 1.0e-3:
            Sint = Sinput[fintrinsic_region, :]
            Sint = Sint - np.mean(Sint)
            # need to add constant to fit values
            G = np.concatenate((fintrinsic1[np.newaxis, :], np.ones((1, tsize_total))), axis=0)
            b, fit, R2, total_var, res_var = pysem.general_glm(Sint, G)
            beta_int1 = b[0]
        else:
            beta_int1 = 0.0
    else:
        beta_int1 = 0.0
        fintrinsic1 = []

    lastgood_beta_int1 = copy.deepcopy(beta_int1)

    # initialize beta values-----------------------------------
    beta_initial = np.zeros(len(csource))
    # beta_initial = np.random.randn(len(csource))
    betascale = 0.0
    beta_initial = betascale * np.ones(len(csource))

    # limit the beta values related to intrinsic inputs to positive values
    for aa in range(len(beta_initial)):
        if latent_flag[aa] > 0:
            # if beta_initial[aa] < 0:  beta_initial[aa] = 0.0
            beta_initial[aa] = 1.0

    # beta_init_record.append({'beta_initial': beta_initial})

    # initalize Sconn
    betavals = copy.deepcopy(beta_initial)  # initialize beta values at zero
    lastgood_betavals = copy.deepcopy(betavals)

    results_record = []
    ssqd_record = []

    alpha = initial_alpha
    Lweight = initial_Lweight
    dval = initial_dval

    Mconn[ctarget, csource] = betavals

    # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
    # fit, Sconn_full = network_eigenvalue_method(Sconn_full, Minput, Mconn, ncon)

    fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count,
                                                             beta_int1, fintrinsic1)
    # cost = np.sum(np.abs(betavals**2)) # L2 regularization
    cost = np.sum(np.abs(betavals))  # L1 regularization
    ssqd = err + Lweight * cost
    ssqd_starting = ssqd
    ssqd_record += [ssqd]

    # nitermax = 100
    # alpha_limit = 1.0e-4

    iter = 0
    # vintrinsics_record = []
    converging = True
    dssq_record = np.ones(3)
    dssq_count = 0
    sequence_count = 0
    while alpha > alpha_limit and iter < nitermax and converging:
        iter += 1
        # gradients in betavals and beta_int1
        Mconn[ctarget, csource] = betavals
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                                 vintrinsic_count, beta_int1, fintrinsic1)
        dssq_db, ssqd, dssq_dbeta1 = gradients_for_betavals(Sinput, Minput, Mconn, betavals, ctarget, csource, dval,
                                                            fintrinsic_count, vintrinsic_count, beta_int1,
                                                            fintrinsic1, Lweight)
        ssqd_record += [ssqd]

        # fix some beta values at zero, if specified
        if len(fixed_beta_vals) > 0:
            dssq_db[fixed_beta_vals] = 0

        # apply the changes
        betavals -= alpha * dssq_db
        beta_int1 -= alpha * dssq_dbeta1

        # limit the beta values related to intrinsic inputs to positive values
        for aa in range(len(betavals)):
            if latent_flag[aa] > 0:
                # if betavals[aa] < 0:  betavals[aa] = 0.0
                betavals[aa] = 1.0

        # betavals[betavals >= betalimit] = betalimit
        # betavals[betavals <= -betalimit] = -betalimit

        Mconn[ctarget, csource] = betavals
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                                 vintrinsic_count, beta_int1, fintrinsic1)
        # cost = np.sum(np.abs(betavals**2))  # L2 regularization
        cost = np.sum(np.abs(betavals))  # L1 regularization
        ssqd_new = err + Lweight * cost

        err_total = Sinput - fit
        Smean = np.mean(Sinput)
        errmean = np.mean(err_total)
        R2total = 1 - np.sum((err_total - errmean) ** 2) / np.sum((Sinput - Smean) ** 2)
        if R2total < 0: R2total = 0.0

        # Sinput_sim, Soutput_sim = network_sim(Sinput_full, Soutput_full, Minput, Moutput)
        results_record.append({'Sinput': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})

        if ssqd_new >= ssqd:
            alpha *= 0.5
            # revert back to last good values
            betavals = copy.deepcopy(lastgood_betavals)
            beta_int1 = copy.deepcopy(lastgood_beta_int1)
            dssqd = ssqd - ssqd_new
            dssq_record = np.ones(3)  # reset the count
            dssq_count = 0
            sequence_count = 0
            if verbose: print('beta vals:  iter {} alpha {:.3e}  delta ssq > 0  - no update'.format(iter, alpha))
        else:
            # save the good values
            lastgood_betavals = copy.deepcopy(betavals)
            lastgood_beta_int1 = copy.deepcopy(beta_int1)

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

        if verbose: print('beta vals:  iter {} alpha {:.3e}  delta ssq {:.4f}  relative: {:.1f} percent  '
                          'R2 {:.3f}'.format(iter, alpha, -dssqd, 100.0 * ssqd / ssqd_starting, R2total))
        # now repeat it ...

    # fit the results now to determine output signaling from each region
    Mconn[ctarget, csource] = betavals
    fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count,
                                                             beta_int1, fintrinsic1)
    Sconn = Meigv @ Mintrinsic  # signalling over each connection

    entry = {'Sinput': Sinput, 'Sconn': Sconn, 'beta_int1': beta_int1, 'Mconn': Mconn, 'Minput': Minput,
             'R2total': R2total, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv, 'betavals': betavals,
             'fintrinsic1': fintrinsic1, 'PCloadings': PCloadings, 'fintrinsic_base': fintrinsic_base}

    return entry

#----------------------------------------------------------------------------------
# --------------------------------------------------------------------
def sem_physio_model_PCAclusters(PCparams, PCloadings, fintrinsic_base, SAPMresultsname,
                                 SAPMparametersname, nitermax = 250, alpha_limit = 1e-5,
                                 subsample = [1,0], fixed_beta_vals = [], verbose = False,
                                 nprocessors = 8):
    starttime = time.ctime()

    # instead of working with specific clusters, this version uses a mix of clusters
    # as a continuum, in order to find the optimal clusters
    # principal components information about clusters are contained in:
    # PCparams = {'components': component_data, 'loadings': original_loadings}
    # how the components are mixed for each region are contained in PCloadings

    # initialize gradient-descent parameters--------------------------------------------------------------
    initial_alpha = 1e-3
    initial_Lweight = 1e-4
    initial_dval = 0.01
    betascale = 0.0

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

    tplist_full = SAPMparams['tplist_full']
    ntime, NP = np.shape(tplist_full)
    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    # repeat the process for each participant-----------------------------------------------------------------
    betalimit = 3.0
    epochnum = 0
    SAPMresults = []
    beta_init_record = []

    # data for gradient_descent_per_person
    ntime, NP = np.shape(SAPMparams['tplist_full'])
    fixed_beta_vals = []
    verbose = False

    data = {'nperson':0,
            'tsize':SAPMparams['tsize'],
            'tplist_full':SAPMparams['tplist_full'],
            'nruns_per_person':SAPMparams['nruns_per_person'],
            'nclusterlist':SAPMparams['nclusterlist'],
            'Minput':SAPMparams['Minput'],
            'fintrinsic_count':SAPMparams['fintrinsic_count'],
            'fintrinsic_region':SAPMparams['fintrinsic_region'],
            'vintrinsic_count':SAPMparams['vintrinsic_count'],
            'epoch':SAPMparams['epoch'],
            'timepoint':SAPMparams['timepoint'],
            'tcdata_centered':SAPMparams['tcdata_centered'],
            'ctarget':SAPMparams['ctarget'],
            'csource':SAPMparams['csource'],
            'latent_flag':SAPMparams['latent_flag'],
            'Mconn':SAPMparams['Mconn'],
            'ntime':ntime,
            'NP':NP,
            'component_data':PCparams['components'],
            'average_data':PCparams['average'],
            'epochnum' :epochnum,
            'fintrinsic_base' :fintrinsic_base,
            'PCloadings' :PCloadings,
            'initial_alpha' :initial_alpha,
            'initial_Lweight' :initial_Lweight,
            'initial_dval' :initial_dval,
            'alpha_limit' :alpha_limit,
            'nitermax' :nitermax,
            'fixed_beta_vals' :fixed_beta_vals,
            'verbose' :verbose }

    # setup iterable input parameters
    input_data = []
    for nperson in range(subsample[1], NP, subsample[0]):
        oneval = copy.deepcopy(data)
        oneval['nperson'] = nperson
        input_data.append(oneval)
    p,f = os.path.split(SAPMparametersname)
    search_data_name = os.path.join(p,'cluster_search_data.npy')

    startpool = time.time()
    if nprocessors <= 1:
        SAPMresults = [gradient_descent_per_person(input_data[n]) for n in range(len(input_data))]
    else:
        pool = mp.Pool(nprocessors)
        # print('runnning gradient_descent_per_person ... (with {} processors)'.format(nprocessors))
        # SAPMresults = pool.map(pf.gradient_descent_per_person, input_data)
        SAPMresults = pool.map(gradient_descent_per_person, input_data)
        pool.close()
    donepool = time.time()
    # print('time to run gradient-descent with {} processors:  {:.1f} sec'.format(nprocessors, donepool-startpool))

    stoptime = time.ctime()

    if verbose:
        print('finished SAPM at {}'.format(time.ctime()))
        print('     started at {}'.format(starttime))

    return SAPMresults, search_data_name

def mod_tplist_for_bootstrap(tplist_full, epoch, modtype, percent_replace = 0, tsize =40):
    # modtype can be 'random', 'allodds', 'allevens', 'firsthalf', 'lasthalf'
    tplist = copy.deepcopy(tplist_full[epoch])
    NP = len(tplist)
    stilllooking = True
    if modtype == 'allodds':
        for nn in range(NP):
            tp = copy.deepcopy(tplist[nn]['tp'])
            nt = len(tp)
            tpb = copy.deepcopy(tp)
            for tt in range(0,nt,2): tpb[tt] = tpb[tt+1]
            tplist[nn]['tp'] = tpb
        stilllooking = False

    if modtype == 'allevens':
        for nn in range(NP):
            tp = copy.deepcopy(tplist[nn]['tp'])
            nt = len(tp)
            tpb = copy.deepcopy(tp)
            for tt in range(1,nt,2): tpb[tt] = tpb[tt-1]
            tplist[nn]['tp'] = tpb
        stilllooking = False

    if modtype == 'firsthalf':
        for nn in range(NP):
            tp = copy.deepcopy(tplist[nn]['tp'])
            nt = len(tp)
            tpb = copy.deepcopy(tp)
            tt = np.floor(nt/2).astype(int)
            tpb[-tt:] = tpb[:tt]
            tplist[nn]['tp'] = tpb
        stilllooking = False

    if modtype == 'lasthalf':
        for nn in range(NP):
            tp = copy.deepcopy(tplist[nn]['tp'])
            nt = len(tp)
            tpb = copy.deepcopy(tp)
            tt = np.floor(nt/2).astype(int)
            tpb[:tt] = tpb[-tt:]
            tplist[nn]['tp'] = tpb
        stilllooking = False

    if modtype == 'oddruns':
        for nn in range(NP):
            tp = copy.deepcopy(tplist[nn]['tp'])
            nt = len(tp)
            nruns = np.floor(nt/tsize).astype(int)
            tpb = copy.deepcopy(tp)
            replaceruns = list(range(0,nruns,2))
            for rr in replaceruns:
                if nruns > (rr+1):
                    tr1 = rr*tsize
                    tr2 = (rr+1)*tsize
                    tt1 = (rr+1)*tsize
                    tt2 = (rr+2)*tsize
                    tpb[tr1:tr2] = tpb[tt1:tt2]
            tplist[nn]['tp'] = tpb
        stilllooking = False

    if modtype == 'evenruns':
        for nn in range(NP):
            tp = copy.deepcopy(tplist[nn]['tp'])
            nt = len(tp)
            nruns = np.floor(nt/tsize).astype(int)
            tpb = copy.deepcopy(tp)
            replaceruns = list(range(1,nruns,2))
            for rr in replaceruns:
                tr1 = rr*tsize
                tr2 = (rr+1)*tsize
                tt1 = (rr-1)*tsize
                tt2 = rr*tsize
                tpb[tr1:tr2] = tpb[tt1:tt2]
            tplist[nn]['tp'] = tpb
        stilllooking = False

    if modtype == 'random' or stilllooking:
        for nn in range(NP):
            tp = copy.deepcopy(tplist[nn]['tp'])
            nt = len(tp)
            if percent_replace <= 0:
                nreplace = 1
            else:
                nreplace = np.floor(percent_replace*nt/100.0).astype(int)
            tpb = copy.deepcopy(tp)
            ntlist = list(range(nt))
            treplace = random.sample(ntlist,nreplace)
            ntlist2 = [x for x in ntlist if x not in treplace]
            twith = random.sample(ntlist2,nreplace)
            for tt in range(nreplace): tpb[treplace[tt]] = tpb[twith[tt]]
            tplist[nn]['tp'] = tpb
        stilllooking = False

    tplist_full2 = copy.deepcopy(tplist_full)
    tplist_full2[epoch] = tplist
    return tplist_full2


def loadings_gradients(beta, PCparams,PCloadings,paradigm_centered,SAPMresultsname,SAPMparametersname,subsample, nprocessors, Lweight = 1.0e-3):
    SAPMresults, search_data_name = sem_physio_model_PCAclusters(PCparams, PCloadings, paradigm_centered, SAPMresultsname,
                                              SAPMparametersname, nitermax = 100, alpha_limit = 1e-5,
                                              subsample = subsample, fixed_beta_vals = [], verbose = False, nprocessors = nprocessors)
    nclusters_total = len(PCloadings)

    # cost function
    R2list = np.array([SAPMresults[x]['R2total'] for x in range(len(SAPMresults))])
    basecost = np.sum(1 - R2list) + Lweight * np.sum(np.abs(PCloadings))

    # gradients in PCloadings
    load_gradients = np.zeros(nclusters_total)
    gradcalcstart = time.time()
    for aa in range(nclusters_total):
        testload = copy.deepcopy(PCloadings)
        testload[aa] += beta
        SAPMresults, search_data_name = sem_physio_model_PCAclusters(PCparams, testload, paradigm_centered, SAPMresultsname,
                                                  SAPMparametersname, nitermax = 100, alpha_limit = 1e-5, subsample = subsample,
                                                  fixed_beta_vals = [], verbose = False, nprocessors = nprocessors)

        # cost function
        R2list = np.array([SAPMresults[x]['R2total'] for x in range(len(SAPMresults))])
        testcost = np.sum(1 - R2list) + Lweight * np.sum(np.abs(PCloadings))

        load_gradients[aa] = (testcost - basecost) / beta
    gradcalcend = time.time()
    print('calculating load gradients took {:.1f} seconds'.format(gradcalcend - gradcalcstart))

    return load_gradients, basecost



# gradient descent method to find best clusters------------------------------------
def SAPM_cluster_search(outputdir, SAPMresultsname, SAPMparametersname, networkfile, DBname, regiondataname, clusterdataname, nprocessors, samplesplit, initial_clusters = []):

    # need inputs:
    # outputdir, SAPMresultsname, SAPMparametersname, networkfile
    # DBname, regiondataname, clusterdataname

    # basedir = r'E:\FM2021data'
    if not os.path.exists(outputdir): os.mkdir(outputdir)

    # outputdir = basedir
    # SAPMresultsname = os.path.join(outputdir, 'SEMphysio_model5.npy')
    # SAPMparametersname = os.path.join(outputdir, 'SEMparameters_model5.npy')
    # networkfile = r'E:/network_model_5cluster_v5_w_3intrinsics.xlsx'

    # load paradigm data--------------------------------------------------------------------
    # DBname = r'E:\FM2021data\FMS2_database_July27_2022b.xlsx'
    xls = pd.ExcelFile(DBname, engine='openpyxl')
    df1 = pd.read_excel(xls, 'paradigm1_BOLD')
    del df1['Unnamed: 0']  # get rid of the unwanted header column
    fields = list(df1.keys())
    paradigm = df1['paradigms_BOLD']
    timevals = df1['time']
    paradigm_centered = paradigm - np.mean(paradigm)
    dparadigm = np.zeros(len(paradigm))
    dparadigm[1:] = np.diff(paradigm_centered)

    # regiondataname = r'E:\FM2021data\HCstim_region_data.npy'
    # clusterdataname = r'E:\FM2021data\FM2021_cluster_definition.npy'

    # get cluster info and setup for saving information later
    cluster_data = np.load(clusterdataname, allow_pickle=True).flat[0]
    cluster_properties = cluster_data['cluster_properties']
    nregions = len(cluster_properties)
    nclusterlist = [cluster_properties[i]['nclusters'] for i in range(nregions)]
    rnamelist = [cluster_properties[i]['rname'] for i in range(nregions)]
    namelist_addon = ['R '+n for n in rnamelist]
    namelist = rnamelist + namelist_addon

    # ---------------------
    prep_data_sem_physio_model(networkfile, regiondataname, clusterdataname, SAPMparametersname)
    SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
    tcdata = SAPMparams['tcdata_centered']  # data for all regions/clusters concatenated along time dimension for all runs
    # need to get principal components for each region to model the clusters as a continuum

    nclusters_total, tsize_total = np.shape(tcdata)
    component_data = np.zeros(np.shape(tcdata))
    average_data = np.zeros(np.shape(tcdata))
    ncmax = np.max(nclusterlist)
    original_loadings = np.zeros((nregions,ncmax,ncmax))
    for regionnum in range(nregions):
        r1 = sum(nclusterlist[:regionnum])
        r2 = sum(nclusterlist[:(regionnum + 1)])

        nstates = nclusterlist[regionnum]  # the number to look at
        pca = PCA(n_components = nstates)
        tcdata_region = tcdata[r1:r2,:]
        pca.fit(tcdata_region)
        S_pca_ = pca.fit(tcdata_region).transform(tcdata_region)

        components = pca.components_
        # use components in SAPM in place of original region data

        # get loadings
        mu = np.mean(tcdata_region, axis=0)
        mu = np.repeat(mu[np.newaxis, :], nstates, axis=0)

        loadings = pca.transform(tcdata_region)
        fit_check = (loadings @ components) + mu

        component_data[r1:r2,:] = components
        average_data[r1:r2,:] = mu
        original_loadings[regionnum,:nstates,:nstates] = loadings

    # scale component_data to make original_loadings near maximum of 1
    PCscalefactor = original_loadings.max()
    original_loadings /= PCscalefactor
    component_data *= PCscalefactor
    PCparams = {'components':component_data, 'average':average_data, 'loadings':original_loadings}

    # for one set of PCloadings
    Lweight = 1.0e-6
    beta = 0.01
    alpha = 1e-2
    initial_alpha = copy.deepcopy(alpha)
    alphalimit = 1e-5
    maxiter = 20
    subsample = [samplesplit,0]  # [2,0] use every 2nd data set, starting with 0

    PCloadings = 1e-4*np.random.randn(nclusters_total)

    if len(initial_clusters) == nregions:
        for aa in range(nregions):
            L = original_loadings[aa,:,:]
            cluster = initial_clusters[aa]
            r1 = sum(nclusterlist[:aa])
            r2 = sum(nclusterlist[:(aa + 1)])
            PCloadings[r1:r2] = L[cluster,:]

    lastgood_PCloadings = copy.deepcopy(PCloadings)

    # gradient descent to find best cluster combination
    iter = 0
    costrecord = []
    print('starting gradient descent search of clusters at {}'.format(time.ctime()))
    recalculate_load_gradients = True
    runcount = 0
    while (alpha > alphalimit) and (iter < maxiter):
        # subsample[1] = iter % 2   # vary which data sets are used out of the subsample
        iter += 1
        # gradients in PCloadings
        if recalculate_load_gradients:
            load_gradients, basecost = loadings_gradients(beta, PCparams, PCloadings, paradigm_centered, SAPMresultsname, SAPMparametersname, subsample, nprocessors, Lweight)
        else:
            print('not calculating load gradients')
        PCloadings -= alpha*load_gradients

        SAPMresults, search_data_name = sem_physio_model_PCAclusters(PCparams, PCloadings, paradigm_centered,
                            SAPMresultsname, SAPMparametersname, nitermax = 100, alpha_limit = 1e-5,
                            subsample = subsample, fixed_beta_vals = [], verbose = False, nprocessors = nprocessors)

        # cost function
        R2list = np.array([SAPMresults[x]['R2total'] for x in range(len(SAPMresults))])
        newcost = np.sum(1-R2list) + Lweight*np.sum(np.abs(PCloadings))
        R2cost_portion = np.sum(1-R2list)
        L1cost_portion = Lweight*np.sum(np.abs(PCloadings))

        costrecord += [basecost]

        if newcost < basecost:
            lastgood_PCloadings = copy.deepcopy(PCloadings)
            recalculate_load_gradients = True
            runcount += 1
            if runcount > 2:
                alpha = np.min([initial_alpha, 1.5*alpha])
            print('iter {}  new cost = {:.3e}  base cost = {:.3e}  delta cost = {:.3e}  alpha = {:.2e}   {}'.format(iter,newcost, basecost, newcost-basecost,alpha,time.ctime()))
        else:
            PCloadings = copy.deepcopy(lastgood_PCloadings)
            alpha *= 0.5
            recalculate_load_gradients = False
            runcount = 0
            print('iter {} - no improvement   new cost = {:.3e}  base cost = {:.3e}  R2 portion = {:.2e}   L1 portion = {:.2e}  alpha = {:.2e}   {}'.format(iter,newcost, basecost,R2cost_portion,L1cost_portion,alpha,time.ctime()))

        # save results on each iteration in case the user wants to abort the run...
        results = {'costrecord':costrecord, 'PCloadings':PCloadings, 'original_loadings':original_loadings, 'PCscalefactor':PCscalefactor}
        outputname = os.path.join(outputdir, 'GDresults2.npy')
        np.save(outputname, results)

        # peek at results
        best_clusters = np.zeros(nregions)
        for region in range(nregions):
            L = original_loadings[region, :, :]
            r1 = sum(nclusterlist[:region])
            r2 = sum(nclusterlist[:(region + 1)])
            p = PCloadings[r1:r2]

            # look for best match
            nclusters = nclusterlist[region]
            d = np.zeros(nclusters)
            for cc in range(nclusters):
                d[cc] = np.sqrt(np.sum((L[cc, :] - p) ** 2))
            x = np.argmin(d)
            best_clusters[region] = x
            best_clusters = best_clusters.astype(int)
        print('\nbest cluster set so far is : {}'.format(best_clusters))


    # look at final results in more detail---------------------------
    best_clusters = np.zeros(nregions)
    for region in range(nregions):
        print('\noriginal loadings region {}'.format(region))
        L = original_loadings[region, :, :]
        nclusters = nclusterlist[region]
        outputstring = ''
        for cc in range(nclusters):
            for dd in range(nclusters):
                outputstring += '{:.3f} '.format(L[cc,dd])
            print(outputstring)

        r1 = sum(nclusterlist[:region])
        r2 = sum(nclusterlist[:(region + 1)])
        print('\nPCloadings region {}'.format(region))
        p = PCloadings[r1:r2]
        outputstring = ''
        for cc in range(nclusters):
            outputstring += '{:.3f} '.format(p[cc])
        print(outputstring)

        # look for best match
        nclusters = nclusterlist[region]
        d = np.zeros(nclusters)
        for cc in range(nclusters):
            d[cc] = np.sqrt(np.sum((L[cc,:]-p)**2))
        print('\ndistance between PCloadings and original {}'.format(region))
        outputstring = ''
        for cc in range(nclusters):
            outputstring += '{:.3f} '.format(d[cc])
        print(outputstring)

        x = np.argmin(d)
        best_clusters[region] = x
        best_clusters = best_clusters.astype(int)
    print('\nbest cluster set is : {}'.format(best_clusters))

    return best_clusters
    #
    # # test result
    # full_rnum_base =  np.array([0,5,10,15,20,25,30,35,40,45])
    # clusterlist = best_clusters + full_rnum_base
    # prep_data_sem_physio_model(networkfile, regiondataname, clusterdataname, SAPMparametersname)
    # output = sem_physio_model(clusterlist, paradigm_centered, SAPMresultsname, SAPMparametersname)
    #
    # SAPMresults = np.load(output, allow_pickle=True)
    # R2list3 = [SAPMresults[aa]['R2total'] for aa in range(len(SAPMresults))]
    #
    # # compare with previous results
    # cnums = [1, 4, 1, 3, 1, 0, 4, 2, 1, 2]  # FMstim, from the last TWO GD attempts
    # cnums = [1, 0, 4, 0, 0, 0, 4, 4, 2, 2]  # HCstim, last GD attempt
    # clusterlist = cnums + full_rnum_base
    # prep_data_sem_physio_model(networkfile, regiondataname, clusterdataname, SAPMparametersname)
    # output = sem_physio_model(clusterlist, paradigm_centered, SAPMresultsname, SAPMparametersname)
    #
    # SAPMresults2 = np.load(output, allow_pickle=True)
    # R2list2 = [SAPMresults2[aa]['R2total'] for aa in range(len(SAPMresults2))]
    #
    # print('current results:  best cluster set: {}'.format(best_clusters))
    # print('previous results:  best cluster set: {}'.format(cnums))
    # print('compare R2 values: ')
    # print('new R2:     old R2:')
    # for aa in range(len(R2list3)):
    #     print('{:.3f}     {:.3f}'.format(R2list3[aa],R2list2[aa]))



# main program
def SAPMrun(cnums, regiondataname, clusterdataname, SAPMresultsname, SAPMparametersname, networkfile, DBname, reload_existing = False):
    # required inputs:
    # cnums
    # outputdir
    # covariatesfile (?)
    # SAPMresultsname
    # SAPMparametersname
    # networkfile
    # DBname
    # regiondataname
    # clusterdataname
    # excelfilename

    # load paradigm data--------------------------------------------------------------------
    # DBname = r'D:/threat_safety_python/threat_safety_database.xlsx'
    xls = pd.ExcelFile(DBname, engine='openpyxl')
    df1 = pd.read_excel(xls, 'paradigm1_BOLD')
    del df1['Unnamed: 0']  # get rid of the unwanted header column
    fields = list(df1.keys())
    paradigm = df1['paradigms_BOLD']
    timevals = df1['time']
    paradigm_centered = paradigm - np.mean(paradigm)
    dparadigm = np.zeros(len(paradigm))
    dparadigm[1:] = np.diff(paradigm_centered)

    # load some data, setup some parameters...
    network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count = load_network_model_w_intrinsics(networkfile)
    ncluster_list = np.array([nclusterlist[x]['nclusters'] for x in range(len(nclusterlist))])
    cluster_name = [nclusterlist[x]['name'] for x in range(len(nclusterlist))]
    not_latent = [x for x in range(len(cluster_name)) if 'intrinsic' not in cluster_name[x]]
    ncluster_list = ncluster_list[not_latent]
    full_rnum_base = [np.sum(ncluster_list[:x]) for x in range(len(ncluster_list))]
    namelist = [cluster_name[x] for x in not_latent]
    namelist += ['Rtotal']
    namelist += ['R ' + cluster_name[x] for x in not_latent]

    # full_rnum_base =  np.array([0,5,10,15,20,25,30,35,40,45])
    #
    # namelist = ['C6RD',  'DRt', 'Hypothalamus','LC', 'NGC', 'NRM', 'NTS', 'PAG', 'PBN', 'Thalamus',
    #         'Rtotal', 'R C6RD',  'R DRt', 'R Hyp','R LC', 'R NGC', 'R NRM', 'R NTS', 'R PAG',
    #         'R PBN', 'R Thal']

    # starting values
    cnums_original = copy.deepcopy(cnums)
    excelsheetname = 'clusters'
    # fname = 'fixed_C6RD{}.xlsx'.format(cord_cluster)
    # excelfilename = os.path.join(outputdir, fname)

    # run the analysis with SAPM
    clusterlist = np.array(cnums) + full_rnum_base
    prep_data_sem_physio_model(networkfile, regiondataname, clusterdataname, SAPMparametersname)
    output = sem_physio_model(clusterlist, paradigm_centered, SAPMresultsname, SAPMparametersname)

    SAPMresults = np.load(output, allow_pickle=True)
    NP = len(SAPMresults)
    R2list =np.zeros(len(SAPMresults))
    for nperson in range(NP):
        R2list[nperson] = SAPMresults[nperson]['R2total']
    print('SAPM parameters computed for {} data sets'.format(NP))
    print('R2 values averaged {:.3f} {} {:.3f}'.format(np.mean(R2list),chr(177),np.std(R2list)))
    print('R2 values ranged from {:.3f} to {:.3f}'.format(np.min(R2list),np.max(R2list)))



    # the rest of this should be in a separate function for displaying selected results-------------
    #
    # if reload_existing:
    #     output = SAPMresultsname
    # else:
    #     output = sem_physio_model(clusterlist, paradigm_centered, SAPMresultsname, SAPMparametersname)
    #
    # SAPMresults = np.load(output, allow_pickle=True).flat[0]
    #
    # windowoffset = 0
    # yrange = []
    # yrange2 = []
    # Rtextlist, Rvallist = show_SEM_timecourse_results(covariatesfile, SAPMparametersname, SAPMresultsname, paradigm_centered, group,
    #                                 windowoffset, yrange, yrange2)
    #
    # yrange = [-0.6, 0.6]
    # yrange2 = [1.6, 0.7, 0.8, 0.6, 0.9, 0.8, 0.5]   # for Feb2022C
    # windowoffset = 0
    #
    # # group = 'all'
    # show_SEM_average_beta_for_groups(covariatesfile, SAPMparametersname, SAPMresultsname, paradigm_centered, group,
    #                                  windowoffset=0)
    #
    # # show a specific connection
    # connection_name = 'PBN-LC-DRt'
    #
    # # settings = np.load(settingsfile, allow_pickle=True).flat[0]
    # # covariates = settings['GRPcharacteristicsvalues'][0].astype(float)  # painrating
    #
    # covariatesdata = np.load(covariatesfile, allow_pickle=True).flat[0]
    # if 'painrating' in covariatesdata['GRPcharacteristicslist']:
    #     x = covariatesdata['GRPcharacteristicslist'].index('painrating')
    #     covariates = covariatesdata['GRPcharacteristicsvalues'][x].astype(float)
    # else:
    #     covariates = []
    #
    # plot_correlated_results(SAPMresultsname, SAPMparametersname, connection_name, covariates, figurenumber = 1)
    #



#----------------------------------------------------------------------------------------
#
#    FUNCTIONS FOR DISPLAYING RESULTS IN VARIOUS FORMATS
#
#----------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
def display_SAPM_results(covariatesfile, SAPMparametersname, SAPMresultsname, person_list = [41, 48, 32, 21, 10]):
    # reload parameters if needed--------------------------------------------------------
    # settings = np.load(settingsfile, allow_pickle=True).flat[0]
    # covariates1 = settings['GRPcharacteristicsvalues'][0]  # gender
    # covariates2 = settings['GRPcharacteristicsvalues'][1].astype(float)  # painrating

    covariatesdata = np.load(covariatesfile, allow_pickle=True).flat[0]
    if 'gender' in covariatesdata['GRPcharacteristicslist']:
        x = covariatesdata['GRPcharacteristicslist'].index('gender')
        covariates1 = covariatesdata['GRPcharacteristicsvalues'][x]
    else:
        covariates1 = []
    if 'painrating' in covariatesdata['GRPcharacteristicslist']:
        x = covariatesdata['GRPcharacteristicslist'].index('painrating')
        covariates2 = covariatesdata['GRPcharacteristicsvalues'][x].astype(float)
    else:
        covariates2 = []


    SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
    # SAPMparams = {'betanamelist': betanamelist, 'beta_list':beta_list, 'nruns_per_person': nruns_per_person,
    #              'nclusterstotal':nclusterstotal, 'rnamelist':rnamelist, 'nregions':nregions,
    #             'cluster_properties':cluster_properties, 'cluster_data':cluster_data,
    #              'network':network, 'fintrinsic_count':fintrinsic_count, 'vintrinsic_count':vintrinsic_count,
    #              'sem_region_list':sem_region_list, 'nclusterlist':nclusterlist, 'tsize':tsize}

    network = SAPMparams['network']
    beta_list = SAPMparams['beta_list']
    betanamelist = SAPMparams['betanamelist']
    nruns_per_person = SAPMparams['nruns_per_person']
    rnamelist = SAPMparams['rnamelist']
    fintrinsic_count = SAPMparams['fintrinsic_count']
    vintrinsic_count = SAPMparams['vintrinsic_count']
    nclusterlist = SAPMparams['nclusterlist']
    tplist_full = SAPMparams['tplist_full']
    tcdata_centered = SAPMparams['tcdata_centered']
    Nintrinsic = fintrinsic_count + vintrinsic_count
    timepoint = SAPMparams['timepoint']
    epoch = SAPMparams['epoch']

    # end of reloading parameters if needed--------------------------------------------------------

    # load the SAPM results
    SAPMresults_load = np.load(SAPMresultsname, allow_pickle=True)

    # for nperson in range(NP)
    NP = len(SAPMresults_load)
    resultscheck = np.zeros((NP,4))
    nbeta,tsize_full = np.shape(SAPMresults_load[0]['Sconn'])
    ncon = nbeta-Nintrinsic

    for nperson in person_list:
        nruns = nruns_per_person[nperson]
        Sinput = SAPMresults_load[nperson]['Sinput']
        Sconn= SAPMresults_load[nperson]['Sconn']
        Minput = SAPMresults_load[nperson]['Minput']
        Mconn = SAPMresults_load[nperson]['Mconn']
        Mintrinsic = SAPMresults_load[nperson]['Mintrinsic']
        Meigv = SAPMresults_load[nperson]['Meigv']
        beta_int1 = SAPMresults_load[nperson]['beta_int1']
        betavals = SAPMresults_load[nperson]['betavals']
        R2total = SAPMresults_load[nperson]['R2total']
        fintrinsic1 = SAPMresults_load[nperson]['fintrinsic1']

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

        p,f = os.path.split(SAPMresultsname)
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

        p,f = os.path.split(SAPMresultsname)
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

        p,f = os.path.split(SAPMresultsname)
        xlname = os.path.join(p,'Person{}_Moutput_v3.xlsx'.format(nperson))
        df.to_excel(xlname)

    Mrecord = np.zeros((nbeta,nbeta,NP))
    R2totalrecord = np.zeros(NP)
    for nperson in range(NP):
        Sinput = SAPMresults_load[nperson]['Sinput']
        Sconn = SAPMresults_load[nperson]['Sconn']
        Minput = SAPMresults_load[nperson]['Minput']
        Mconn = SAPMresults_load[nperson]['Mconn']
        R2total = SAPMresults_load[nperson]['R2total']
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
def show_Mconn_properties(covariatesfile, SAPMparametersname, SAPMresultsname):
    # settings = np.load(settingsfile, allow_pickle=True).flat[0]
    # covariates1 = settings['GRPcharacteristicsvalues'][0]  # gender
    # covariates2 = settings['GRPcharacteristicsvalues'][1].astype(float)  # painrating

    covariatesdata = np.load(covariatesfile, allow_pickle=True).flat[0]
    if 'gender' in covariatesdata['GRPcharacteristicslist']:
        x = covariatesdata['GRPcharacteristicslist'].index('gender')
        covariates1 = covariatesdata['GRPcharacteristicsvalues'][x]
    else:
        covariates1 = []
    if 'painrating' in covariatesdata['GRPcharacteristicslist']:
        x = covariatesdata['GRPcharacteristicslist'].index('painrating')
        covariates2 = covariatesdata['GRPcharacteristicsvalues'][x].astype(float)
    else:
        covariates2 = []

    SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
    network = SAPMparams['network']
    beta_list = SAPMparams['beta_list']
    betanamelist = SAPMparams['betanamelist']
    nruns_per_person = SAPMparams['nruns_per_person']
    rnamelist = SAPMparams['rnamelist']
    fintrinsic_count = SAPMparams['fintrinsic_count']
    vintrinsic_count = SAPMparams['vintrinsic_count']
    nclusterlist = SAPMparams['nclusterlist']
    tplist_full = SAPMparams['tplist_full']
    tcdata_centered = SAPMparams['tcdata_centered']
    Nintrinsic = fintrinsic_count + vintrinsic_count
    # end of reloading parameters-------------------------------------------------------

    # load the SAPM results
    SAPMresults_load = np.load(SAPMresultsname, allow_pickle=True)

    # for nperson in range(NP)
    NP = len(SAPMresults_load)
    resultscheck = np.zeros((NP, 4))
    nbeta, tsize_full = np.shape(SAPMresults_load[0]['Sconn'])
    ncon = nbeta - Nintrinsic

    Mrecord = np.zeros((nbeta, nbeta, NP))
    R2totalrecord = np.zeros(NP)
    for nperson in range(NP):
        Sinput = SAPMresults_load[nperson]['Sinput']
        Sconn = SAPMresults_load[nperson]['Sconn']
        Minput = SAPMresults_load[nperson]['Minput']
        Mconn = SAPMresults_load[nperson]['Mconn']
        R2total = SAPMresults_load[nperson]['R2total']
        fintrinsic1 = SAPMresults_load[nperson]['fintrinsic1']
        Mrecord[:, :, nperson] = Mconn
        R2totalrecord[nperson] = R2total

    Mrecord_mean = np.mean(Mrecord, axis=2)
    Mrecord_sem = np.std(Mrecord, axis=2) / np.sqrt(NP)
    M_T = np.mean(Mrecord, axis=2) / (Mrecord_sem + 1e-20)
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
    Tsexdiff = np.zeros((nbeta, nbeta))
    psexdiff = np.zeros((nbeta, nbeta))
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
    Rrecord = np.zeros((nbeta, nbeta))
    R2record = np.zeros((nbeta, nbeta))
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
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.options.display.float_format = '{:.2f}'.format

    # write out Mrecord_mean, Mrecord_sem, M_T, Tsexdiff, R2record
    # text_mean = write_Mconn_values(Mrecord_mean, Mrecord_sem, betanamelist, rnamelist, beta_list)

    labeltext, valuetext, Ttext, T, Tthresh = write_Mconn_values2(Mrecord_mean, Mrecord_sem, NP, betanamelist, rnamelist, beta_list, format='f', pthresh=0.05)

    text_T = write_Mconn_values(M_T, [], betanamelist, rnamelist, beta_list)
    text_SD = write_Mconn_values(Tsexdiff, [], betanamelist, rnamelist, beta_list)
    text_R2pain = write_Mconn_values(R2record, [], betanamelist, rnamelist, beta_list)

    p, f = os.path.split(SAPMresultsname)

    df = pd.DataFrame(Mrecord_mean, columns=columns, index=rows)
    xlname = os.path.join(p, 'Mrecord_mean.xlsx')
    df.to_excel(xlname)

    df = pd.DataFrame(Mrecord_sem, columns=columns, index=rows)
    xlname = os.path.join(p, 'Mrecord_sem.xlsx')
    df.to_excel(xlname)

    df = pd.DataFrame(M_T, columns=columns, index=rows)
    xlname = os.path.join(p, 'Mrecord_T.xlsx')
    df.to_excel(xlname)

    df = pd.DataFrame(Tsexdiff, columns=columns, index=rows)
    xlname = os.path.join(p, 'Mrecord_Tsexdiffs.xlsx')
    df.to_excel(xlname)

    df = pd.DataFrame(R2record, columns=columns, index=rows)
    xlname = os.path.join(p, 'Mrecord_R2pain.xlsx')
    df.to_excel(xlname)

    # ANCOVA group vs pain rating
    statstype = 'ANCOVA'
    formula_key1 = 'C(Group)'
    formula_key2 = 'pain'
    formula_key3 = 'C(Group):' + 'pain'
    atype = 2

    cov1 = covariates2[g1]
    cov2 = covariates2[g2]
    ancova_p = np.ones((nbeta,nbeta,3))
    for aa in range(ncon):
        for bb in range(ncon):
            m = Mrecord[aa, bb, :]
            if np.var(m) > 0:
                b1 = m[g1]
                b2 = m[g2]
                anova_table, p_MeoG, p_MeoC, p_intGC = py2ndlevelanalysis.run_ANOVA_or_ANCOVA2(b1, b2, cov1, cov2, 'pain', formula_key1,
                                                                            formula_key2, formula_key3, atype)
                ancova_p[aa,bb, :] = np.array([p_MeoG, p_MeoC, p_intGC])


    pd.options.display.float_format = '{:.2e}'.format
    df = pd.DataFrame(ancova_p[:,:,0], columns=columns, index=rows)
    xlname = os.path.join(p, 'Mancova_MeoG.xlsx')
    df.to_excel(xlname)
    df = pd.DataFrame(ancova_p[:,:,1], columns=columns, index=rows)
    xlname = os.path.join(p, 'Mancova_MeoP.xlsx')
    df.to_excel(xlname)
    df = pd.DataFrame(ancova_p[:,:,2], columns=columns, index=rows)
    xlname = os.path.join(p, 'Mancova_IntGP.xlsx')
    df.to_excel(xlname)

    text_MeoG = write_Mconn_values(ancova_p[:,:,0], [], betanamelist, rnamelist, beta_list, format = 'e', minthresh = 0.0, maxthresh = 0.05)
    text_MeoP = write_Mconn_values(ancova_p[:,:,1], [], betanamelist, rnamelist, beta_list, format = 'e', minthresh = 0.0, maxthresh = 0.05)
    text_IntGP = write_Mconn_values(ancova_p[:,:,2], [], betanamelist, rnamelist, beta_list, format = 'e', minthresh = 0.0, maxthresh = 0.05)



def show_SAPM_timecourse_results(covariatesfile, SAPMparametersname, SAPMresultsname, paradigm_centered, group='all', windowoffset = 0, yrange = [], yrange2 = []):
    # if len(yrange) > 0:
    #     setylim = True
    #     ymin = yrange[0]
    #     ymax = yrange[1]
    # else:
    #     setylim = False

    # settings = np.load(settingsfile, allow_pickle=True).flat[0]
    # covariates1 = settings['GRPcharacteristicsvalues'][0]  # gender
    # covariates2 = settings['GRPcharacteristicsvalues'][1].astype(float)  # painrating

    covariatesdata = np.load(covariatesfile, allow_pickle=True).flat[0]
    if 'gender' in covariatesdata['GRPcharacteristicslist']:
        x = covariatesdata['GRPcharacteristicslist'].index('gender')
        covariates1 = covariatesdata['GRPcharacteristicsvalues'][x]
    else:
        covariates1 = []
    if 'painrating' in covariatesdata['GRPcharacteristicslist']:
        x = covariatesdata['GRPcharacteristicslist'].index('painrating')
        covariates2 = covariatesdata['GRPcharacteristicsvalues'][x].astype(float)
    else:
        covariates2 = []


    SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
    network = SAPMparams['network']
    beta_list = SAPMparams['beta_list']
    betanamelist = SAPMparams['betanamelist']
    nruns_per_person = SAPMparams['nruns_per_person']
    rnamelist = SAPMparams['rnamelist']
    fintrinsic_count = SAPMparams['fintrinsic_count']
    fintrinsic_region = SAPMparams['fintrinsic_region']
    vintrinsic_count = SAPMparams['vintrinsic_count']
    nclusterlist = SAPMparams['nclusterlist']
    tplist_full = SAPMparams['tplist_full']
    tcdata_centered = SAPMparams['tcdata_centered']
    ctarget = SAPMparams['ctarget']
    csource = SAPMparams['csource']
    tsize = SAPMparams['tsize']
    timepoint = SAPMparams['timepoint']
    epoch = SAPMparams['epoch']
    Nintrinsic = fintrinsic_count + vintrinsic_count
    # end of reloading parameters-------------------------------------------------------

    # load the SAPM results
    SAPMresults_load = np.load(SAPMresultsname, allow_pickle=True)

    # for nperson in range(NP)
    NP = len(SAPMresults_load)
    resultscheck = np.zeros((NP, 4))
    nbeta, tsize_full = np.shape(SAPMresults_load[0]['Sconn'])
    ncon = nbeta - Nintrinsic

    if epoch >= tsize:
        et1 = 0
        et2 = tsize
    else:
        et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
        et2 = (timepoint + np.floor(epoch / 2)).astype(int)
    ftemp = paradigm_centered[et1:et2]

    Mrecord = np.zeros((nbeta, nbeta, NP))
    R2totalrecord = np.zeros(NP)
    for nperson in range(NP):
        Sinput = SAPMresults_load[nperson]['Sinput']
        Sconn = SAPMresults_load[nperson]['Sconn']
        Minput = SAPMresults_load[nperson]['Minput']
        Mconn = SAPMresults_load[nperson]['Mconn']
        beta_int1 = SAPMresults_load[nperson]['beta_int1']
        R2total = SAPMresults_load[nperson]['R2total']
        Meigv = SAPMresults_load[nperson]['Meigv']
        betavals = SAPMresults_load[nperson]['betavals']
        # fintrinsic1 = SAPMresults_load[nperson]['fintrinsic1']

        nruns = nruns_per_person[nperson]
        fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])

        # ---------------------------------------------------
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                                 vintrinsic_count, beta_int1, fintrinsic1)

        nr, tsize_total = np.shape(Sinput)
        tsize = (tsize_total / nruns).astype(int)
        nbeta,tsize2 = np.shape(Sconn)

        if nperson == 0:
            Sinput_total = np.zeros((nr,tsize, NP))
            Sconn_total = np.zeros((nbeta,tsize, NP))
            fit_total = np.zeros((nr,tsize, NP))

        tc = Sinput
        tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
        Sinput_total[:,:,nperson] = tc1

        tc = Sconn
        tc1 = np.mean(np.reshape(tc, (nbeta, nruns, tsize)), axis=1)
        Sconn_total[:,:,nperson] = tc1

        tc = fit
        tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
        fit_total[:,:,nperson] = tc1

        Mrecord[:, :, nperson] = Mconn
        R2totalrecord[nperson] = R2total


    if len(covariates1) > 1 and len(covariates2) > 1:
        # ancova sex x pain rating---------------------------------------
        # ANCOVA group vs pain rating
        statstype = 'ANCOVA'
        formula_key1 = 'C(Group)'
        formula_key2 = 'pain'
        formula_key3 = 'C(Group):' + 'pain'
        atype = 2

        # separate by sex
        g1 = np.where(covariates1 == 'Female')[0]
        g2 = np.where(covariates1 == 'Male')[0]

        cov1 = covariates2[g1]
        cov2 = covariates2[g2]
        ancova_p = np.ones((nbeta,nbeta,3))
        ttest_p = np.ones((nbeta,nbeta,2))
        for aa in range(ncon):
            for bb in range(ncon):
                m = Mrecord[aa, bb, :]
                if np.var(m) > 0:
                    b1 = m[g1]
                    b2 = m[g2]
                    anova_table, p_MeoG, p_MeoC, p_intGC = py2ndlevelanalysis.run_ANOVA_or_ANCOVA2(b1, b2, cov1, cov2, 'pain', formula_key1,
                                                                                formula_key2, formula_key3, atype)
                    ancova_p[aa,bb, :] = np.array([p_MeoG, p_MeoC, p_intGC])

                    if (np.var(b1) > 0) & (np.var(b2) > 0):
                        t, p = stats.ttest_ind(b1, b2, equal_var=False)
                        ttest_p[aa,bb,:] = np.array([p,t])

        columns = [name[:3] + ' in' for name in betanamelist]
        rows = [name[:3] for name in betanamelist]

        p, f = os.path.split(SAPMresultsname)
        pd.options.display.float_format = '{:.2e}'.format
        df = pd.DataFrame(ancova_p[:,:,0], columns=columns, index=rows)
        xlname = os.path.join(p, 'Mancova_MeoG.xlsx')
        df.to_excel(xlname)
        df = pd.DataFrame(ancova_p[:,:,1], columns=columns, index=rows)
        xlname = os.path.join(p, 'Mancova_MeoP.xlsx')
        df.to_excel(xlname)
        df = pd.DataFrame(ancova_p[:,:,2], columns=columns, index=rows)
        xlname = os.path.join(p, 'Mancova_IntGP.xlsx')
        df.to_excel(xlname)
        df = pd.DataFrame(ttest_p[:,:,0], columns=columns, index=rows)
        xlname = os.path.join(p, 'Ttest_sexdiffs.xlsx')
        df.to_excel(xlname)


        print('\nMain effect of group:')
        text_MeoG = write_Mconn_values(ancova_p[:,:,0], [], betanamelist, rnamelist, beta_list, format = 'e', minthresh = 0.0, maxthresh = 0.05)
        print('\nMain effect of pain ratings:')
        text_MeoP = write_Mconn_values(ancova_p[:,:,1], [], betanamelist, rnamelist, beta_list, format = 'e', minthresh = 0.0, maxthresh = 0.05)
        print('\nInteraction group x pain:')
        text_IntGP = write_Mconn_values(ancova_p[:,:,2], [], betanamelist, rnamelist, beta_list, format = 'e', minthresh = 0.0, maxthresh = 0.05)
        print('\n\n')

        print('\nT-test sex differences:')
        text_T = write_Mconn_values(ttest_p[:,:,0], [], betanamelist, rnamelist, beta_list, format = 'e', minthresh = 0.0, maxthresh = 0.05)
        print('\n\n')


        # compare groups with T-tests

        # set the group
        g = list(range(NP))
        gtag = '_' + group

        if group.lower() == 'male':
            g = g2
            gtag = '_Male'

        if group.lower() == 'female':
            g = g1
            gtag = '_Female'



    # set the group
    g = list(range(NP))
    gtag = '_'+group
    Sinput_avg = np.mean(Sinput_total[:, :, g], axis=2)
    Sinput_sem = np.std(Sinput_total[:, :, g], axis=2) / np.sqrt(len(g))
    Sconn_avg = np.mean(Sconn_total[:, :, g], axis=2)
    Sconn_sem = np.std(Sconn_total[:, :, g], axis=2) / np.sqrt(len(g))
    fit_avg = np.mean(fit_total[:, :, g], axis=2)
    fit_sem = np.std(fit_total[:, :, g], axis=2) / np.sqrt(len(g))

    if len(covariates2) > 1:
        # regression based on pain ratings
        p = covariates2[np.newaxis, g]
        p -= np.mean(p)
        pmax = np.max(np.abs(p))
        p /= pmax
        G = np.concatenate((np.ones((1, len(g))),p), axis=0) # put the intercept term first
        Sinput_reg = np.zeros((nr,tsize,4))
        fit_reg = np.zeros((nr,tsize,4))
        Sconn_reg = np.zeros((nbeta,tsize,4))
        # Sinput_R2 = np.zeros((nr,tsize,2))
        # fit_R2 = np.zeros((nr,tsize,2))
        # Sconn_R2 = np.zeros((nbeta,tsize,2))
        for tt in range(tsize):
            for nn in range(nr):
                m = Sinput_total[nn,tt,g]
                b, fit, R2, total_var, res_var = pysem.general_glm(m, G)
                Sinput_reg[nn,tt,:2] = b
                Sinput_reg[nn,tt,-2:] = [R2, np.sign(R2)*np.arctanh(np.sqrt(np.abs(R2)))*np.sqrt(len(g)-3)]

                m = fit_total[nn,tt,g]
                b, fit, R2, total_var, res_var = pysem.general_glm(m, G)
                fit_reg[nn,tt,:2] = b
                fit_reg[nn,tt,-2:] = [R2, np.sign(R2)*np.arctanh(np.sqrt(np.abs(R2)))*np.sqrt(len(g)-3)]

            for nn in range(nbeta):
                m = Sconn_total[nn,tt,g]
                b, fit, R2, total_var, res_var = pysem.general_glm(m, G)
                Sconn_reg[nn,tt,:2] = b
                Sconn_reg[nn,tt,-2:] = [R2, np.sign(R2)*np.arctanh(np.sqrt(np.abs(R2)))*np.sqrt(len(g)-3)]

            # need to save Sinput_reg, Sinput_R2, etc., somewhere for later use....

            # regression of Mrecord with pain ratings
            # glm_fit
            Mregression = np.zeros((nbeta,nbeta,3))
            # Mregression1 = np.zeros((nbeta,nbeta,3))
            # Mregression2 = np.zeros((nbeta,nbeta,3))
            p = covariates2[np.newaxis, g]
            p -= np.mean(p)
            pmax = np.max(np.abs(p))
            p /= pmax
            G = np.concatenate((np.ones((1, len(g))), p), axis=0)  # put the intercept term first
            for aa in range(nbeta):
                for bb in range(nbeta):
                    m = Mrecord[aa,bb,g]
                    if np.var(m) > 0:
                        b, fit, R2, total_var, res_var = pysem.general_glm(m[np.newaxis,:], G)
                        Mregression[aa,bb,:] = [b[0,0],b[0,1],R2]

            print('\n\nMconn regression with pain ratings')
            # rtext = write_Mconn_values(Mregression[:,:,1], Mregression[:,:,2], betanamelist, rnamelist, beta_list, format='f', minthresh=0.0001, maxthresh=0.0)
            labeltext, inttext, slopetext, R2text, R2, R2thresh = write_Mreg_values(Mregression[:,:,1], Mregression[:,:,2], betanamelist, rnamelist, beta_list, format='f', Rthresh=0.1)

    # average Mconn values
    Mconn_avg = np.mean(Mrecord[:,:,g],axis = 2)
    Mconn_sem = np.std(Mrecord[:,:,g],axis = 2)/np.sqrt(len(g))
    # rtext = write_Mconn_values(Mconn_avg, Mconn_sem, betanamelist, rnamelist, beta_list,
    #                            format='f', minthresh=0.0001, maxthresh=0.0)

    pthresh = 0.05
    Tthresh = stats.t.ppf(1 - pthresh, NP - 1)

    print('\n\nAverage Mconn values')
    labeltext, valuetext, Ttext, T, Tthresh = write_Mconn_values2(Mconn_avg, Mconn_sem, NP, betanamelist, rnamelist, beta_list, format='f', pthresh=0.05)

    Rtextlist = [' ']*10
    Rvallist = [0]*10

    windownum_offset = windowoffset
    outputdir, f = os.path.split(SAPMresultsname)
    # only show 3 regions in each plot for consistency in sizing
    # show some regions
    window2 = 25 + windownum_offset
    regionlist = [3,6,8]
    nametag = r'LC_NTS_PBN' + gtag
    svgname, Rtext, Rvals = plot_region_fits(window2, regionlist, nametag, Sinput_avg, Sinput_sem, fit_avg, fit_sem, rnamelist, outputdir, yrange)
    for n,x in enumerate(regionlist):
        print('Rtext = {}  Rvals = {}'.format(Rtext[n],Rvals[n]))
        Rtextlist[x] = Rtext[n]
        Rvallist[x] = Rvals[n]

    # show some regions
    window2 = 33 + windownum_offset
    regionlist = [0,5,1]
    nametag = r'cord_NRM_DRt' + gtag
    svgname, Rtext, Rvals = plot_region_fits(window2, regionlist, nametag, Sinput_avg, Sinput_sem, fit_avg, fit_sem, rnamelist, outputdir, yrange)
    for n,x in enumerate(regionlist):
        Rtextlist[x] = Rtext[n]
        Rvallist[x] = Rvals[n]

    window2b = 35 + windownum_offset
    regionlist = [0,5,4]
    nametag = r'cord_NRM_NGC' + gtag
    svgname, Rtext, Rvals = plot_region_fits(window2b, regionlist, nametag, Sinput_avg, Sinput_sem, fit_avg, fit_sem, rnamelist, outputdir, yrange)
    for n,x in enumerate(regionlist):
        Rtextlist[x] = Rtext[n]
        Rvallist[x] = Rvals[n]

    window2c = 36 + windownum_offset
    regionlist = [7,2,9]
    nametag = r'PAG_Hyp_Thal' + gtag
    svgname, Rtext, Rvals = plot_region_fits(window2c, regionlist, nametag, Sinput_avg, Sinput_sem, fit_avg, fit_sem, rnamelist, outputdir, yrange)
    for n,x in enumerate(regionlist):
        Rtextlist[x] = Rtext[n]
        Rvallist[x] = Rvals[n]


    # figure 0--------------------------------------------
    # inputs to C6RD
    window3 = 0 + windownum_offset
    target = 'C6RD'
    nametag1 = r'C6RDinput' + gtag
    if len(yrange2) > 0:
        ylim = yrange2[0]
        yrangethis = [-ylim,ylim]
    else:
        yrangethis = []
    plot_region_inputs_regression(window3, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    plot_region_inputs_average(window3+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    # yrange = [-1.5,1.5]
    # yrange = []
    # yrange2 = [-1.5,1.5]
    # yrange2 = []

    # show results
    # figure 1   -  inputs to NRM
    window1 = 1 + windownum_offset
    target = 'NRM'
    nametag1 = r'NRMinput' + gtag
    if len(yrange2) > 0:
        ylim = yrange2[1]
        yrangethis = [-ylim,ylim]
    else:
        yrangethis = []
    plot_region_inputs_regression(window1, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    plot_region_inputs_average(window1+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    # figure 2--------------------------------------------
    # inputs to NTS
    window4 = 2 + windownum_offset
    target = 'NTS'
    nametag1 = r'NTSinput' + gtag
    if len(yrange2) > 0:
        ylim = yrange2[2]
        yrangethis = [-ylim,ylim]
    else:
        yrangethis = []
    plot_region_inputs_regression(window4, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    plot_region_inputs_average(window4+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    # figure 3--------------------------------------------
    # inputs to NGC
    window7 = 3 + windownum_offset
    target = 'NGC'
    nametag1 = r'NGCinput' + gtag
    if len(yrange2) > 0:
        ylim = yrange2[3]
        yrangethis = [-ylim,ylim]
    else:
        yrangethis = []
    plot_region_inputs_regression(window7, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    plot_region_inputs_average(window7+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    # figure 4--------------------------------------------
    # inputs to PAG
    window5 = 4 + windownum_offset
    target = 'PAG'
    nametag1 = r'PAGinput' + gtag
    if len(yrange2) > 0:
        ylim = yrange2[4]
        yrangethis = [-ylim,ylim]
    else:
        yrangethis = []
    plot_region_inputs_regression(window5, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    plot_region_inputs_average(window5+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    # figure 5   -  inputs to PBN
    window11 = 5 + windownum_offset
    target = 'PBN'
    nametag1 = r'PBNinput' + gtag
    if len(yrange2) > 0:
        ylim = yrange2[5]
        yrangethis = [-ylim,ylim]
    else:
        yrangethis = []
    plot_region_inputs_regression(window11, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    plot_region_inputs_average(window11+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    # figure 6   -  inputs to LC
    window12 = 6 + windownum_offset
    target = 'LC'
    nametag1 = r'LCinput' + gtag
    if len(yrange2) > 0:
        ylim = yrange2[6]
        yrangethis = [-ylim,ylim]
    else:
        yrangethis = []
    plot_region_inputs_regression(window12, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    plot_region_inputs_average(window12+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    if len(covariates1) > 1 and len(covariates2) > 1:
        # plot a specific connection
        window6 = 29 + windownum_offset
        target = 'C6RD-PAG'
        source = 'NTS-C6RD'

        target = 'NTS-PBN'
        source = 'PBN-NTS'

        c = target.index('-')
        betaname_target = '{}_{}'.format(rnamelist.index(target[:c]),rnamelist.index(target[(c+1):]))
        c = source.index('-')
        betaname_source = '{}_{}'.format(rnamelist.index(source[:c]),rnamelist.index(source[(c+1):]))

        rt = betanamelist.index(betaname_target)
        rs = betanamelist.index(betaname_source)
        m = Mrecord[rt,rs,:]
        G = np.concatenate((covariates2[np.newaxis,g1],np.ones((1,len(g1)))), axis = 0)
        b1, fit1, R21, total_var, res_var = pysem.general_glm(m[g1], G)
        G = np.concatenate((covariates2[np.newaxis,g2],np.ones((1,len(g2)))), axis = 0)
        b2, fit2, R22, total_var, res_var = pysem.general_glm(m[g2], G)

        plt.close(window6)
        fig = plt.figure(window6)
        ax = fig.add_axes([0.1,0.1,0.8,0.8])
        plt.plot(covariates2[g1],m[g1], marker = 'o', linestyle = 'None', color = (0.1,0.9,0.1))
        plt.plot(covariates2[g1],fit1, marker = 'None', linestyle = '-', color = (0.1,0.9,0.1))
        plt.plot(covariates2[g2],m[g2], marker = 'o', linestyle = 'None', color = (1.0,0.5,0.))
        plt.plot(covariates2[g2],fit2, marker = 'None', linestyle = '-', color = (1.0,0.5,0.))

        ax.annotate('{} input to {}'.format(source,target), xy=(.025, .975), xycoords='axes  fraction', color = 'r',
                            horizontalalignment='left', verticalalignment='top', fontsize=10)
        ax.annotate('Female R2={:.3f}'.format(R21), xy=(.025, .025), xycoords='axes  fraction', color = (0.1,0.9,0.1),
                            horizontalalignment='left', verticalalignment='bottom', fontsize=10)
        ax.annotate('Male R2={:.3f}'.format(R22), xy=(.975, .025), xycoords='axes  fraction', color = (1.0,0.5,0.),
                            horizontalalignment='right', verticalalignment='bottom', fontsize=10)

    return Rtextlist, Rvallist



def show_SAPM_timecourses_covariates(discrete_covariate, continuous_covariate, SAPMparametersname, SAPMresultsname, paradigm_centered, regionlists_for_display, inputs_to_display, group='all', windowoffset = 0, yrange = [], yrange2 = []):
    covariates1 = discrete_covariate
    covariates2 = continuous_covariate

    SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
    network = SAPMparams['network']
    beta_list = SAPMparams['beta_list']
    betanamelist = SAPMparams['betanamelist']
    nruns_per_person = SAPMparams['nruns_per_person']
    rnamelist = SAPMparams['rnamelist']
    fintrinsic_count = SAPMparams['fintrinsic_count']
    fintrinsic_region = SAPMparams['fintrinsic_region']
    vintrinsic_count = SAPMparams['vintrinsic_count']
    nclusterlist = SAPMparams['nclusterlist']
    tplist_full = SAPMparams['tplist_full']
    tcdata_centered = SAPMparams['tcdata_centered']
    ctarget = SAPMparams['ctarget']
    csource = SAPMparams['csource']
    tsize = SAPMparams['tsize']
    timepoint = SAPMparams['timepoint']
    epoch = SAPMparams['epoch']
    Nintrinsic = fintrinsic_count + vintrinsic_count
    # end of reloading parameters-------------------------------------------------------

    # load the SAPM results
    SAPMresults_load = np.load(SAPMresultsname, allow_pickle=True)

    # for nperson in range(NP)
    NP = len(SAPMresults_load)
    resultscheck = np.zeros((NP, 4))
    nbeta, tsize_full = np.shape(SAPMresults_load[0]['Sconn'])
    ncon = nbeta - Nintrinsic

    if epoch >= tsize:
        et1 = 0
        et2 = tsize
    else:
        et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
        et2 = (timepoint + np.floor(epoch / 2)).astype(int)
    ftemp = paradigm_centered[et1:et2]

    Mrecord = np.zeros((nbeta, nbeta, NP))
    R2totalrecord = np.zeros(NP)
    for nperson in range(NP):
        Sinput = SAPMresults_load[nperson]['Sinput']
        Sconn = SAPMresults_load[nperson]['Sconn']
        Minput = SAPMresults_load[nperson]['Minput']
        Mconn = SAPMresults_load[nperson]['Mconn']
        beta_int1 = SAPMresults_load[nperson]['beta_int1']
        R2total = SAPMresults_load[nperson]['R2total']
        Meigv = SAPMresults_load[nperson]['Meigv']
        betavals = SAPMresults_load[nperson]['betavals']
        # fintrinsic1 = SAPMresults_load[nperson]['fintrinsic1']

        nruns = nruns_per_person[nperson]
        fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])

        # ---------------------------------------------------
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                                 vintrinsic_count, beta_int1, fintrinsic1)

        nr, tsize_total = np.shape(Sinput)
        tsize = (tsize_total / nruns).astype(int)
        nbeta,tsize2 = np.shape(Sconn)

        if nperson == 0:
            Sinput_total = np.zeros((nr,tsize, NP))
            Sconn_total = np.zeros((nbeta,tsize, NP))
            fit_total = np.zeros((nr,tsize, NP))

        tc = Sinput
        tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
        Sinput_total[:,:,nperson] = tc1

        tc = Sconn
        tc1 = np.mean(np.reshape(tc, (nbeta, nruns, tsize)), axis=1)
        Sconn_total[:,:,nperson] = tc1

        tc = fit
        tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
        fit_total[:,:,nperson] = tc1

        Mrecord[:, :, nperson] = Mconn
        R2totalrecord[nperson] = R2total



    # ancova sex x pain rating---------------------------------------
    # ANCOVA group vs pain rating
    statstype = 'ANCOVA'
    formula_key1 = 'C(COV1)'
    formula_key2 = 'COV2'
    formula_key3 = 'C(COV1):' + 'COV2'
    atype = 2

    c1groups = np.unique(covariates1)
    # separate by category
    g1 = np.where(covariates1 == c1groups[0])[0]
    g2 = np.where(covariates1 == c1groups[1])[0]

    cov1 = covariates2[g1]
    cov2 = covariates2[g2]
    ancova_p = np.ones((nbeta,nbeta,3))
    ttest_p = np.ones((nbeta,nbeta,2))
    for aa in range(ncon):
        for bb in range(ncon):
            m = Mrecord[aa, bb, :]
            if np.var(m) > 0:
                b1 = m[g1]
                b2 = m[g2]
                anova_table, p_MeoG, p_MeoC, p_intGC = py2ndlevelanalysis.run_ANOVA_or_ANCOVA2(b1, b2, cov1, cov2, 'pain', formula_key1,
                                                                            formula_key2, formula_key3, atype)
                ancova_p[aa,bb, :] = np.array([p_MeoG, p_MeoC, p_intGC])

                if (np.var(b1) > 0) & (np.var(b2) > 0):
                    t, p = stats.ttest_ind(b1, b2, equal_var=False)
                    ttest_p[aa,bb,:] = np.array([p,t])

    columns = [name[:3] + ' in' for name in betanamelist]
    rows = [name[:3] for name in betanamelist]

    p, f = os.path.split(SAPMresultsname)
    pd.options.display.float_format = '{:.2e}'.format
    df = pd.DataFrame(ancova_p[:,:,0], columns=columns, index=rows)
    xlname = os.path.join(p, 'Mancova_MeoG.xlsx')
    df.to_excel(xlname)
    df = pd.DataFrame(ancova_p[:,:,1], columns=columns, index=rows)
    xlname = os.path.join(p, 'Mancova_MeoP.xlsx')
    df.to_excel(xlname)
    df = pd.DataFrame(ancova_p[:,:,2], columns=columns, index=rows)
    xlname = os.path.join(p, 'Mancova_IntGP.xlsx')
    df.to_excel(xlname)
    df = pd.DataFrame(ttest_p[:,:,0], columns=columns, index=rows)
    xlname = os.path.join(p, 'Ttest_groupdiffs.xlsx')
    df.to_excel(xlname)


    print('\nMain effect of COV1:')
    text_MeoG = write_Mconn_values(ancova_p[:,:,0], [], betanamelist, rnamelist, beta_list, format = 'e', minthresh = 0.0, maxthresh = 0.05)
    print('\nMain effect of COV2:')
    text_MeoP = write_Mconn_values(ancova_p[:,:,1], [], betanamelist, rnamelist, beta_list, format = 'e', minthresh = 0.0, maxthresh = 0.05)
    print('\nInteraction COV1 x COV2:')
    text_IntGP = write_Mconn_values(ancova_p[:,:,2], [], betanamelist, rnamelist, beta_list, format = 'e', minthresh = 0.0, maxthresh = 0.05)
    print('\n\n')

    print('\nT-test group differences:')
    text_T = write_Mconn_values(ttest_p[:,:,0], [], betanamelist, rnamelist, beta_list, format = 'e', minthresh = 0.0, maxthresh = 0.05)
    print('\n\n')


    # compare groups with T-tests

    # set the group
    g = list(range(NP))
    gtag = '_'+group

    if group.lower() == c1groups[1]:
        g = g2
        gtag = '_' + c1groups[1]

    if group.lower() == c1groups[0]:
        g = g1
        gtag = '_' + c1groups[0]

    Sinput_avg = np.mean(Sinput_total[:,:,g], axis = 2)
    Sinput_sem = np.std(Sinput_total[:,:,g], axis = 2)/np.sqrt(len(g))
    Sconn_avg = np.mean(Sconn_total[:,:,g], axis = 2)
    Sconn_sem = np.std(Sconn_total[:,:,g], axis = 2)/np.sqrt(len(g))
    fit_avg = np.mean(fit_total[:,:,g], axis = 2)
    fit_sem = np.std(fit_total[:,:,g], axis = 2)/np.sqrt(len(g))

    # regression based on COV2 (separate by group?)
    p = covariates2[np.newaxis, g]
    p -= np.mean(p)
    pmax = np.max(np.abs(p))
    p /= pmax
    G = np.concatenate((np.ones((1, len(g))),p), axis=0) # put the intercept term first
    Sinput_reg = np.zeros((nr,tsize,4))
    fit_reg = np.zeros((nr,tsize,4))
    Sconn_reg = np.zeros((nbeta,tsize,4))
    # Sinput_R2 = np.zeros((nr,tsize,2))
    # fit_R2 = np.zeros((nr,tsize,2))
    # Sconn_R2 = np.zeros((nbeta,tsize,2))
    for tt in range(tsize):
        for nn in range(nr):
            m = Sinput_total[nn,tt,g]
            b, fit, R2, total_var, res_var = pysem.general_glm(m, G)
            Sinput_reg[nn,tt,:2] = b
            Sinput_reg[nn,tt,-2:] = [R2, np.sign(R2)*np.arctanh(np.sqrt(np.abs(R2)))*np.sqrt(len(g)-3)]

            m = fit_total[nn,tt,g]
            b, fit, R2, total_var, res_var = pysem.general_glm(m, G)
            fit_reg[nn,tt,:2] = b
            fit_reg[nn,tt,-2:] = [R2, np.sign(R2)*np.arctanh(np.sqrt(np.abs(R2)))*np.sqrt(len(g)-3)]

        for nn in range(nbeta):
            m = Sconn_total[nn,tt,g]
            b, fit, R2, total_var, res_var = pysem.general_glm(m, G)
            Sconn_reg[nn,tt,:2] = b
            Sconn_reg[nn,tt,-2:] = [R2, np.sign(R2)*np.arctanh(np.sqrt(np.abs(R2)))*np.sqrt(len(g)-3)]

    # need to save Sinput_reg, Sinput_R2, etc., somewhere for later use....

    # regression of Mrecord with pain ratings
    # glm_fit
    Mregression = np.zeros((nbeta,nbeta,3))
    # Mregression1 = np.zeros((nbeta,nbeta,3))
    # Mregression2 = np.zeros((nbeta,nbeta,3))
    p = covariates2[np.newaxis, g]
    p -= np.mean(p)
    pmax = np.max(np.abs(p))
    p /= pmax
    G = np.concatenate((np.ones((1, len(g))), p), axis=0)  # put the intercept term first
    for aa in range(nbeta):
        for bb in range(nbeta):
            m = Mrecord[aa,bb,g]
            if np.var(m) > 0:
                b, fit, R2, total_var, res_var = pysem.general_glm(m[np.newaxis,:], G)
                Mregression[aa,bb,:] = [b[0,0],b[0,1],R2]

    print('\n\nMconn regression with COV2')
    # rtext = write_Mconn_values(Mregression[:,:,1], Mregression[:,:,2], betanamelist, rnamelist, beta_list, format='f', minthresh=0.0001, maxthresh=0.0)
    labeltext, inttext, slopetext, R2text, R2, R2thresh = write_Mreg_values(Mregression[:,:,1], Mregression[:,:,2], betanamelist, rnamelist, beta_list, format='f', Rthresh=0.1)

    # average Mconn values
    Mconn_avg = np.mean(Mrecord[:,:,g],axis = 2)
    Mconn_sem = np.std(Mrecord[:,:,g],axis = 2)/np.sqrt(len(g))
    # rtext = write_Mconn_values(Mconn_avg, Mconn_sem, betanamelist, rnamelist, beta_list,
    #                            format='f', minthresh=0.0001, maxthresh=0.0)


    pthresh = 0.05
    Tthresh = stats.t.ppf(1 - pthresh, NP - 1)

    print('\n\nAverage Mconn values')
    labeltext, valuetext, Ttext, T, Tthresh = write_Mconn_values2(Mconn_avg, Mconn_sem, NP, betanamelist, rnamelist, beta_list, format='f', pthresh=0.05)

    Rtextlist = [' ']*10
    Rvallist = [0]*10

    # plot timecourses for selected regions
    n_to_display = len(regionlists_for_display)
    Nplots = np.ceil(n_to_display/3).astype(int)
    RL = np.concatenate((regionlists_for_display,regionlists_for_display))

    for np in range(Nplots):
        outputdir, f = os.path.split(SAPMresultsname)
        # only show 3 regions in each plot for consistency in sizing
        # show some regions
        window1 = windowoffset + np
        regionlist = RL[np*3:(np+1)*3]
        nametag = rnamelist[regionlist[0]] + '_' + rnamelist[regionlist[1]]  + '_' + rnamelist[regionlist[2]]
        nametag += gtag
        svgname, Rtext, Rvals = plot_region_fits(window1, regionlist, nametag, Sinput_avg, Sinput_sem, fit_avg, fit_sem, rnamelist, outputdir, yrange)
        for n,x in enumerate(regionlist):
            print('Rtext = {}  Rvals = {}'.format(Rtext[n],Rvals[n]))
            Rtextlist[x] = Rtext[n]
            Rvallist[x] = Rvals[n]

    # plot inputs to selected regions
    n_to_display = len(inputs_to_display)
    for np in range(n_to_display):
        window1 = windowoffset + np + 50
        target = inputs_to_display[np]
        nametag1 = target + 'input' + gtag

        if len(yrange2) > 0:
            ylim = yrange2[0]
            yrangethis = [-ylim,ylim]
        else:
            yrangethis = []
        plot_region_inputs_regression(window1, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

        plot_region_inputs_average(window1+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                                   Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    return Rtextlist, Rvallist


def show_SAPM_timecourses(SAPMparametersname, SAPMresultsname, paradigm_centered, regionlists_for_display, inputs_to_display, windowoffset = 0, yrange = [], yrange2 = []):
    import numpy as np

    print('testing  sqrt of 5 is {}'.format(np.sqrt(5)))

    # show the results
    SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
    network = SAPMparams['network']
    beta_list = SAPMparams['beta_list']
    betanamelist = SAPMparams['betanamelist']
    nruns_per_person = SAPMparams['nruns_per_person']
    rnamelist = SAPMparams['rnamelist']
    fintrinsic_count = SAPMparams['fintrinsic_count']
    fintrinsic_region = SAPMparams['fintrinsic_region']
    vintrinsic_count = SAPMparams['vintrinsic_count']
    nclusterlist = SAPMparams['nclusterlist']
    tplist_full = SAPMparams['tplist_full']
    tcdata_centered = SAPMparams['tcdata_centered']
    ctarget = SAPMparams['ctarget']
    csource = SAPMparams['csource']
    tsize = SAPMparams['tsize']
    timepoint = SAPMparams['timepoint']
    epoch = SAPMparams['epoch']
    Nintrinsic = fintrinsic_count + vintrinsic_count
    # end of reloading parameters-------------------------------------------------------

    # load the SAPM results
    SAPMresults_load = np.load(SAPMresultsname, allow_pickle=True)

    # for nperson in range(NP)
    NP = len(SAPMresults_load)
    resultscheck = np.zeros((NP, 4))
    nbeta, tsize_full = np.shape(SAPMresults_load[0]['Sconn'])
    ncon = nbeta - Nintrinsic

    if epoch >= tsize:
        et1 = 0
        et2 = tsize
    else:
        et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
        et2 = (timepoint + np.floor(epoch / 2)).astype(int)
    ftemp = paradigm_centered[et1:et2]

    Mrecord = np.zeros((nbeta, nbeta, NP))
    R2totalrecord = np.zeros(NP)
    for nperson in range(NP):
        Sinput = SAPMresults_load[nperson]['Sinput']
        Sconn = SAPMresults_load[nperson]['Sconn']
        Minput = SAPMresults_load[nperson]['Minput']
        Mconn = SAPMresults_load[nperson]['Mconn']
        beta_int1 = SAPMresults_load[nperson]['beta_int1']
        R2total = SAPMresults_load[nperson]['R2total']
        Meigv = SAPMresults_load[nperson]['Meigv']
        betavals = SAPMresults_load[nperson]['betavals']
        # fintrinsic1 = SAPMresults_load[nperson]['fintrinsic1']

        nruns = nruns_per_person[nperson]
        fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])

        # ---------------------------------------------------
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                                 vintrinsic_count, beta_int1, fintrinsic1)

        nr, tsize_total = np.shape(Sinput)
        tsize = (tsize_total / nruns).astype(int)
        nbeta,tsize2 = np.shape(Sconn)

        if nperson == 0:
            Sinput_total = np.zeros((nr,tsize, NP))
            Sconn_total = np.zeros((nbeta,tsize, NP))
            fit_total = np.zeros((nr,tsize, NP))

        tc = Sinput
        tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
        Sinput_total[:,:,nperson] = tc1

        tc = Sconn
        tc1 = np.mean(np.reshape(tc, (nbeta, nruns, tsize)), axis=1)
        Sconn_total[:,:,nperson] = tc1

        tc = fit
        tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
        fit_total[:,:,nperson] = tc1

        Mrecord[:, :, nperson] = Mconn
        R2totalrecord[nperson] = R2total

    Sinput_avg = np.mean(Sinput_total,axis=2)
    Sinput_sem = np.std(Sinput_total,axis=2)/np.sqrt(NP)

    Sconn_avg = np.mean(Sconn_total,axis=2)
    Sconn_sem = np.std(Sconn_total,axis=2)/np.sqrt(NP)

    fit_avg = np.mean(fit_total,axis=2)
    fit_sem = np.std(fit_total,axis=2)/np.sqrt(NP)

    Mconn_avg = np.mean(Mrecord,axis=2)
    Mconn_sem = np.std(Mrecord,axis=2)/np.sqrt(NP)

    pthresh = 0.05
    Tthresh = stats.t.ppf(1 - pthresh, NP - 1)

    print('\n\nAverage Mconn values')
    labeltext, valuetext, Ttext, T, Tthresh = write_Mconn_values2(Mconn_avg, Mconn_sem, NP, betanamelist, rnamelist, beta_list, format='f', pthresh=0.05)

    Rtextlist = [' ']*len(rnamelist)
    Rvallist = [0]*len(rnamelist)

    # plot timecourses for selected regions
    n_to_display = len(regionlists_for_display)
    Nplots = np.ceil(n_to_display/3).astype(int)
    RL = np.concatenate((regionlists_for_display,regionlists_for_display))

    for np in range(Nplots):
        outputdir, f = os.path.split(SAPMresultsname)
        # only show 3 regions in each plot for consistency in sizing
        # show some regions
        window1 = windowoffset + np
        regionlist = RL[np*3:(np+1)*3]
        nametag = rnamelist[regionlist[0]] + '_' + rnamelist[regionlist[1]]  + '_' + rnamelist[regionlist[2]]
        svgname, Rtext, Rvals = plot_region_fits(window1, regionlist, nametag, Sinput_avg, Sinput_sem, fit_avg, fit_sem, rnamelist, outputdir, yrange)
        for n,x in enumerate(regionlist):
            print('Rtext = {}  Rvals = {}'.format(Rtext[n],Rvals[n]))
            Rtextlist[x] = Rtext[n]
            Rvallist[x] = Rvals[n]

    # plot inputs to selected regions
    n_to_display = len(inputs_to_display)
    for np in range(n_to_display):
        window1 = windowoffset + np + 50
        target = inputs_to_display[np]
        nametag1 = rnamelist[target] + 'input'

        if len(yrange2) > 0:
            ylim = yrange2[0]
            yrangethis = [-ylim,ylim]
        else:
            yrangethis = []

        plot_region_inputs_average(window1, rnamelist[target],nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                                   Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    return Rtextlist, Rvallist


#-------------compare groups----------------------------------

# def show_SAPM_timecourse_results_compare_groups(covariatesfile, SAPMparametersname, SAPMresultsname, paradigm_centered, group='all',
#                                 windowoffset=0, yrange = []):
#     if len(yrange) > 0:
#         setylim = True
#         ymin = yrange[0]
#         ymax = yrange[1]
#     else:
#         setylim = False
#
#     # settings = np.load(settingsfile, allow_pickle=True).flat[0]
#     # covariates1 = settings['GRPcharacteristicsvalues'][0]  # gender
#     # covariates2 = settings['GRPcharacteristicsvalues'][1].astype(float)  # painrating
#
#     covariatesdata = np.load(covariatesfile, allow_pickle=True).flat[0]
#     if 'gender' in covariatesdata['GRPcharacteristicslist']:
#         x = covariatesdata['GRPcharacteristicslist'].index('gender')
#         covariates1 = covariatesdata['GRPcharacteristicsvalues'][x]
#     else:
#         covariates1 = []
#     if 'painrating' in covariatesdata['GRPcharacteristicslist']:
#         x = covariatesdata['GRPcharacteristicslist'].index('painrating')
#         covariates2 = covariatesdata['GRPcharacteristicsvalues'][x].astype(float)
#     else:
#         covariates2 = []
#
#
#     SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
#     network = SAPMparams['network']
#     beta_list = SAPMparams['beta_list']
#     betanamelist = SAPMparams['betanamelist']
#     nruns_per_person = SAPMparams['nruns_per_person']
#     rnamelist = SAPMparams['rnamelist']
#     fintrinsic_count = SAPMparams['fintrinsic_count']
#     fintrinsic_region = SAPMparams['fintrinsic_region']
#     vintrinsic_count = SAPMparams['vintrinsic_count']
#     nclusterlist = SAPMparams['nclusterlist']
#     tplist_full = SAPMparams['tplist_full']
#     tcdata_centered = SAPMparams['tcdata_centered']
#     ctarget = SAPMparams['ctarget']
#     csource = SAPMparams['csource']
#     tsize = SAPMparams['tsize']
#     timepoint = SAPMparams['timepoint']
#     epoch = SAPMparams['epoch']
#     Nintrinsic = fintrinsic_count + vintrinsic_count
#     # end of reloading parameters-------------------------------------------------------
#
#     # load the SAPM results
#     SAPMresults_load = np.load(SAPMresultsname, allow_pickle=True)
#
#     # for nperson in range(NP)
#     NP = len(SAPMresults_load)
#     resultscheck = np.zeros((NP, 4))
#     nbeta, tsize_full = np.shape(SAPMresults_load[0]['Sconn'])
#     ncon = nbeta - Nintrinsic
#
#     if epoch >= tsize:
#         et1 = 0
#         et2 = tsize
#     else:
#         et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
#         et2 = (timepoint + np.floor(epoch / 2)).astype(int)
#     ftemp = paradigm_centered[et1:et2]
#
#     Mrecord = np.zeros((nbeta, nbeta, NP))
#     R2totalrecord = np.zeros(NP)
#     for nperson in range(NP):
#         Sinput = SAPMresults_load[nperson]['Sinput']
#         Sconn = SAPMresults_load[nperson]['Sconn']
#         Minput = SAPMresults_load[nperson]['Minput']
#         Mconn = SAPMresults_load[nperson]['Mconn']
#         beta_int1 = SAPMresults_load[nperson]['beta_int1']
#         R2total = SAPMresults_load[nperson]['R2total']
#         Meigv = SAPMresults_load[nperson]['Meigv']
#         betavals = SAPMresults_load[nperson]['betavals']
#         # fintrinsic1 = SAPMresults_load[nperson]['fintrinsic1']
#
#         nruns = nruns_per_person[nperson]
#         fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])
#
#         # ---------------------------------------------------
#         fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
#                                                                  vintrinsic_count, beta_int1, fintrinsic1)
#
#         nr, tsize_total = np.shape(Sinput)
#         tsize = (tsize_total / nruns).astype(int)
#         nbeta, tsize2 = np.shape(Sconn)
#
#         if nperson == 0:
#             Sinput_total = np.zeros((nr, tsize, NP))
#             Sconn_total = np.zeros((nbeta, tsize, NP))
#             fit_total = np.zeros((nr, tsize, NP))
#
#         tc = Sinput
#         tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
#         Sinput_total[:, :, nperson] = tc1
#
#         tc = Sconn
#         tc1 = np.mean(np.reshape(tc, (nbeta, nruns, tsize)), axis=1)
#         Sconn_total[:, :, nperson] = tc1
#
#         tc = fit
#         tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
#         fit_total[:, :, nperson] = tc1
#
#         Mrecord[:, :, nperson] = Mconn
#         R2totalrecord[nperson] = R2total
#
#     # ancova sex x pain rating---------------------------------------
#     # ANCOVA group vs pain rating
#     statstype = 'ANCOVA'
#     formula_key1 = 'C(Group)'
#     formula_key2 = 'pain'
#     formula_key3 = 'C(Group):' + 'pain'
#     atype = 2
#
#     # separate by sex
#     g1 = np.where(covariates1 == 'Female')[0]
#     g2 = np.where(covariates1 == 'Male')[0]
#
#     cov1 = covariates2[g1]
#     cov2 = covariates2[g2]
#     ancova_p = np.ones((nbeta, nbeta, 3))
#     for aa in range(ncon):
#         for bb in range(ncon):
#             m = Mrecord[aa, bb, :]
#             if np.var(m) > 0:
#                 b1 = m[g1]
#                 b2 = m[g2]
#                 anova_table, p_MeoG, p_MeoC, p_intGC = py2ndlevelanalysis.run_ANOVA_or_ANCOVA2(b1, b2, cov1, cov2,
#                                                                                                'pain', formula_key1,
#                                                                                                formula_key2,
#                                                                                                formula_key3, atype)
#                 ancova_p[aa, bb, :] = np.array([p_MeoG, p_MeoC, p_intGC])
#
#     columns = [name[:3] + ' in' for name in betanamelist]
#     rows = [name[:3] for name in betanamelist]
#
#     p, f = os.path.split(SAPMresultsname)
#     pd.options.display.float_format = '{:.2e}'.format
#     df = pd.DataFrame(ancova_p[:, :, 0], columns=columns, index=rows)
#     xlname = os.path.join(p, 'Mancova_MeoG.xlsx')
#     df.to_excel(xlname)
#     df = pd.DataFrame(ancova_p[:, :, 1], columns=columns, index=rows)
#     xlname = os.path.join(p, 'Mancova_MeoP.xlsx')
#     df.to_excel(xlname)
#     df = pd.DataFrame(ancova_p[:, :, 2], columns=columns, index=rows)
#     xlname = os.path.join(p, 'Mancova_IntGP.xlsx')
#     df.to_excel(xlname)
#
#     print('\nMain effect of group:')
#     text_MeoG = write_Mconn_values(ancova_p[:, :, 0], [], betanamelist, rnamelist, beta_list, format='e', minthresh=0.0,
#                                    maxthresh=0.05)
#     print('\nMain effect of pain ratings:')
#     text_MeoP = write_Mconn_values(ancova_p[:, :, 1], [], betanamelist, rnamelist, beta_list, format='e', minthresh=0.0,
#                                    maxthresh=0.05)
#     print('\nInteraction group x pain:')
#     text_IntGP = write_Mconn_values(ancova_p[:, :, 2], [], betanamelist, rnamelist, beta_list, format='e',
#                                     minthresh=0.0, maxthresh=0.05)
#     print('\n\n')
#
#     # set the group
#     g = list(range(NP))
#     gtag = '_all'
#     g2tag = '_Male'
#     g1tag = '_Female'
#     Mdata = []
#
#     for gnum in range(3):
#         if gnum == 0:  gg = g
#         if gnum == 1:  gg = g1
#         if gnum == 2:  gg = g2
#         # do this for each group---------------------------
#         # regression of Mrecord with pain ratings
#         # glm_fit
#         Mregression = np.zeros((nbeta, nbeta, 3))
#         p = covariates2[np.newaxis, gg]
#         p -= np.mean(p)
#         pmax = np.max(np.abs(p))
#         p /= pmax
#         G = np.concatenate((np.ones((1, len(gg))), p), axis=0)  # put the intercept term first
#         for aa in range(nbeta):
#             for bb in range(nbeta):
#                 m = Mrecord[aa, bb, gg]
#                 if np.var(m) > 0:
#                     b, fit, R2, total_var, res_var = pysem.general_glm(m[np.newaxis, :], G)
#                     Mregression[aa, bb, :] = [b[0, 0], b[0, 1], R2]
#
#         # average Mconn values
#         Mconn_avg = np.mean(Mrecord[:, :, gg], axis=2)
#         Mconn_sem = np.std(Mrecord[:, :, gg], axis=2) / np.sqrt(len(gg))
#         # rtext = write_Mconn_values(Mconn_avg, Mconn_sem, betanamelist, rnamelist, beta_list,
#         #                            format='f', minthresh=0.0001, maxthresh=0.0)
#         Tvals = Mconn_avg/(Mconn_sem + 1.0e-10)
#         entry = {'Mreg':Mregression, 'Mconn_avg':Mconn_avg, 'Mconn_sem':Mconn_sem, 'Tvals':Tvals}
#
#         pthresh = 0.05
#         Tthresh = stats.t.ppf(1 - pthresh, NP - 1)
#         if gnum == 0:
#             Ttemp = np.abs(Tvals) > Tthresh
#             Tsigflag = copy.deepcopy(Ttemp)
#             Rtemp = np.abs(Mregression[:,:,2]) > 0.1
#             Rsigflag = copy.deepcopy(Rtemp)
#         else:
#             Ttemp = np.abs(Tvals) > Tthresh
#             Rtemp = np.abs(Mregression[:,:,2]) > 0.1
#             Tsigflag += Ttemp
#             Rsigflag += Rtemp
#
#         Mdata.append(entry)
#
#
#     for gnum in range(3):
#         if gnum == 0:  tag = gtag
#         if gnum == 1:  tag = g1tag
#         if gnum == 2:  tag = g2tag
#
#         Mregression = Mdata[gnum]['Mreg']
#         Mconn_avg = Mdata[gnum]['Mconn_avg']
#         Mconn_sem = Mdata[gnum]['Mconn_sem']
#
#         descriptor = '{} Mconn regression with pain ratings'.format(tag)
#         print('\n\n{}'.format(descriptor))
#         # rtext = write_Mconn_values(Mregression[:,:,1], Mregression[:,:,2], betanamelist, rnamelist, beta_list, format='f', minthresh=0.0001, maxthresh=0.0)
#
#         reg_pthresh = 0.0001
#         Zthresh = stats.norm.ppf(1 - reg_pthresh)
#         Rthresh = np.tanh(Zthresh/np.sqrt(NP-3))
#         R2thresh = Rthresh**2
#         print('for p = {:.2e}  Z = {:.2f}   R = {:.3f}  R2 = {:.3f} NP = {}'.format(reg_pthresh, Zthresh,Rthresh,R2thresh, NP))
#
#         R2thresh = 0.1
#         format = 'f'
#         labeltext, inttext, slopetext, R2text, R2, R2thresh = write_Mreg_values(Mregression[:, :, 1], Mregression[:, :, 2], betanamelist, rnamelist, beta_list,
#                                   format, R2thresh, Rsigflag > 0)
#         textoutputs = {'regions': labeltext, 'beta': valuetext, 'R2': Rtext}
#         p, f = os.path.split(SAPMresultsname)
#         df = pd.DataFrame(textoutputs)
#         xlname = os.path.join(p, descriptor + '.xlsx')
#         df.to_excel(xlname)
#
#         descriptor = '{} Average Mconn values'.format(tag)
#         print('\n\n{}'.format(descriptor))
#         format = 'f'
#         pthresh = 0.05
#         labeltext, valuetext, Ttext, T, Tthresh = write_Mconn_values2(Mconn_avg, Mconn_sem, NP, betanamelist, rnamelist, beta_list, format, pthresh, Tsigflag > 0)
#         textoutputs = {'regions':labeltext, 'beta':valuetext, 'T':Ttext}
#
#         p, f = os.path.split(SAPMresultsname)
#         df = pd.DataFrame(textoutputs)
#         xlname = os.path.join(p, descriptor + '.xlsx')
#         df.to_excel(xlname)




#-------------------------------------------------------------
#---------------show all average values for each group--------
# def show_SAPM_average_beta_for_groups(covariatesfile, SAPMparametersname, SAPMresultsname, paradigm_centered, group='all',
#                                 windowoffset=0):
#     # settings = np.load(settingsfile, allow_pickle=True).flat[0]
#     # covariates1 = settings['GRPcharacteristicsvalues'][0]  # gender
#     # covariates2 = settings['GRPcharacteristicsvalues'][1].astype(float)  # painrating
#
#     covariatesdata = np.load(covariatesfile, allow_pickle=True).flat[0]
#     if 'gender' in covariatesdata['GRPcharacteristicslist']:
#         x = covariatesdata['GRPcharacteristicslist'].index('gender')
#         covariates1 = covariatesdata['GRPcharacteristicsvalues'][x]
#     else:
#         covariates1 = []
#     if 'painrating' in covariatesdata['GRPcharacteristicslist']:
#         x = covariatesdata['GRPcharacteristicslist'].index('painrating')
#         covariates2 = covariatesdata['GRPcharacteristicsvalues'][x].astype(float)
#     else:
#         covariates2 = []
#
#     painrating = covariates2
#
#     SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
#     network = SAPMparams['network']
#     beta_list = SAPMparams['beta_list']
#     betanamelist = SAPMparams['betanamelist']
#     nruns_per_person = SAPMparams['nruns_per_person']
#     rnamelist = SAPMparams['rnamelist']
#     fintrinsic_count = SAPMparams['fintrinsic_count']
#     fintrinsic_region = SAPMparams['fintrinsic_region']
#     vintrinsic_count = SAPMparams['vintrinsic_count']
#     nclusterlist = SAPMparams['nclusterlist']
#     tplist_full = SAPMparams['tplist_full']
#     tcdata_centered = SAPMparams['tcdata_centered']
#     ctarget = SAPMparams['ctarget']
#     csource = SAPMparams['csource']
#     tsize = SAPMparams['tsize']
#     timepoint = SAPMparams['timepoint']
#     epoch = SAPMparams['epoch']
#     Nintrinsic = fintrinsic_count + vintrinsic_count
#     # end of reloading parameters-------------------------------------------------------
#
#     # load the SAPM results
#     SAPMresults_load = np.load(SAPMresultsname, allow_pickle=True)
#
#     # for nperson in range(NP)
#     NP = len(SAPMresults_load)
#     resultscheck = np.zeros((NP, 4))
#     nbeta, tsize_full = np.shape(SAPMresults_load[0]['Sconn'])
#     ncon = nbeta - Nintrinsic
#
#     if epoch >= tsize:
#         et1 = 0
#         et2 = tsize
#     else:
#         et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
#         et2 = (timepoint + np.floor(epoch / 2)).astype(int)
#     ftemp = paradigm_centered[et1:et2]
#
#     #-------------compile all the results---------------------------
#     Mrecord = np.zeros((nbeta, nbeta, NP))
#     R2totalrecord = np.zeros(NP)
#     for nperson in range(NP):
#         Sinput = SAPMresults_load[nperson]['Sinput']
#         Sconn = SAPMresults_load[nperson]['Sconn']
#         Minput = SAPMresults_load[nperson]['Minput']
#         Mconn = SAPMresults_load[nperson]['Mconn']
#         beta_int1 = SAPMresults_load[nperson]['beta_int1']
#         R2total = SAPMresults_load[nperson]['R2total']
#         Meigv = SAPMresults_load[nperson]['Meigv']
#         betavals = SAPMresults_load[nperson]['betavals']
#         # fintrinsic1 = SAPMresults_load[nperson]['fintrinsic1']
#
#         nruns = nruns_per_person[nperson]
#         fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])
#
#         # ---------------------------------------------------
#         fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
#                                                                  vintrinsic_count, beta_int1, fintrinsic1)
#
#         nr, tsize_total = np.shape(Sinput)
#         tsize = (tsize_total / nruns).astype(int)
#         nbeta, tsize2 = np.shape(Sconn)
#
#         if nperson == 0:
#             Sinput_total = np.zeros((nr, tsize, NP))
#             Sconn_total = np.zeros((nbeta, tsize, NP))
#             fit_total = np.zeros((nr, tsize, NP))
#
#         tc = Sinput
#         tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
#         Sinput_total[:, :, nperson] = tc1
#
#         tc = Sconn
#         tc1 = np.mean(np.reshape(tc, (nbeta, nruns, tsize)), axis=1)
#         Sconn_total[:, :, nperson] = tc1
#
#         tc = fit
#         tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
#         fit_total[:, :, nperson] = tc1
#
#         Mrecord[:, :, nperson] = Mconn
#         R2totalrecord[nperson] = R2total
#
#     # separate by sex
#     g = list(range(NP))
#     if group == 'Female':
#         g = np.where(covariates1 == 'Female')[0]
#     if group == 'Male':
#         g = np.where(covariates1 == 'Male')[0]
#
#     cov = covariates2[g]
#
#     columns = [name[:3] + ' in' for name in betanamelist]
#     rows = [name[:3] for name in betanamelist]
#
#
#     # set the group
#     # g = list(range(NP))
#     # gtag = '_all'
#     # g2tag = '_Male'
#     # g1tag = '_Female'
#     Mdata = []
#
#     # for gnum in range(3):
#     #     if gnum == 0:  gg = g
#     #     if gnum == 1:  gg = g1
#     #     if gnum == 2:  gg = g2
#
#         # do this for each group---------------------------
#         # average Mconn values
#     Mconn_avg = np.mean(Mrecord[:, :, g], axis=2)
#     Mconn_sem = np.std(Mrecord[:, :, g], axis=2) / np.sqrt(len(g))
#     # rtext = write_Mconn_values(Mconn_avg, Mconn_sem, betanamelist, rnamelist, beta_list,
#     #                            format='f', minthresh=0.0001, maxthresh=0.0)
#     Tvals = Mconn_avg/(Mconn_sem + 1.0e-10)
#
#         # entry = {'Mconn_avg':Mconn_avg, 'Mconn_sem':Mconn_sem}
#         #
#         # Mdata.append(entry)
#
#     # correlation between beta and pain ratings
#     p = painrating[g] - np.mean(painrating[g])
#     Rvals = np.zeros(np.shape(Tvals))
#     Zvals = np.zeros(np.shape(Tvals))
#     for nn in range(len(csource)):
#         b = Mrecord[ctarget[nn],csource[nn],g]
#         R = np.corrcoef(b,p)
#         Rvals[ctarget[nn],csource[nn]] = R[0,1]
#         Zvals[ctarget[nn],csource[nn]] = np.arctanh(R[0,1])*np.sqrt(len(g)-3)
#
#
#     # for gnum in range(3):
#     #     if gnum == 0:  tag = gtag
#     #     if gnum == 1:  tag = g1tag
#     #     if gnum == 2:  tag = g2tag
#
#         # Mconn_avg = Mdata[gnum]['Mconn_avg']
#         # Mconn_sem = Mdata[gnum]['Mconn_sem']
#
#     descriptor = '{} All Average Mconn values'.format(group)
#     print('\n\n{}'.format(descriptor))
#     format = 'f'
#     pthresh = 0.05
#     sigflag = []
#     labeltext, valuetext, Ttext, T, Tthresh = write_Mconn_values2(Mconn_avg, Mconn_sem, NP, betanamelist, rnamelist, beta_list, format, pthresh, sigflag)
#     textoutputs = {'regions':labeltext, 'beta':valuetext, 'T':Ttext}
#
#     sigflag = []
#     labeltext, valuetext, Ztext, Rtext = write_Mcorr_values(Mconn_avg, Mconn_sem, Rvals, Zvals, NP, betanamelist, rnamelist, beta_list, format, pthresh, sigflag)
#     textoutputsR = {'regions':labeltext, 'beta':valuetext, 'Z':Ztext, 'R':Rtext}
#
#     p, f = os.path.split(SAPMresultsname)
#     df = pd.DataFrame(textoutputs)
#     dfR = pd.DataFrame(textoutputsR)
#     xlname = os.path.join(p, descriptor + '.xlsx')
#     xlnameR = os.path.join(p, descriptor + '_corr.xlsx')
#     print(xlname)
#
#     df.to_excel(xlname, sheet_name = 'average')
#     dfR.to_excel(xlnameR, sheet_name = 'correlation')
#
#     # write out info about R2 distribution
#     print('R2 average = {:.3f} {} {:.3f}'.format(np.mean(R2totalrecord),chr(177),np.std(R2totalrecord)))
#     print('R2 max/min = {:.3f} {:.3f}'.format(np.max(R2totalrecord),np.min(R2totalrecord)))
#--------------------------------------------------------------
#--------------------------------------------------------------

def plot_region_inputs_average(windownum, target, nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrange = []):
    Zthresh = stats.norm.ppf(1-np.array([1.0, 0.05,0.01,0.001]))
    symbollist = [' ','*', chr(8868),chr(8903)]

    if len(yrange) > 0:
        setylim = True
        ymin = yrange[0]
        ymax = yrange[1]
    else:
        setylim = False

    rtarget = rnamelist.index(target)
    m = Minput[rtarget, :]
    sources = np.where(m == 1)[0]
    rsources = [beta_list[ss]['pair'][0] for ss in sources]
    nsources = len(sources)
    nregions = len(rnamelist)
    checkdims = np.shape(Sinput_avg)
    if np.ndim(Sinput_avg) > 2:  nv = checkdims[2]
    tsize = checkdims[1]

    # get beta values from Mconn
    m = Mconn_avg[:,sources[0]]
    targets2ndlevel_list = np.where(m != 0.)[0]
    textlist = []
    for ss in sources:
        text = betanamelist[ss] + ': '
        beta = Mconn_avg[targets2ndlevel_list,ss]
        for ss2 in range(len(beta)):
            valtext = '{:.2f} '.format(beta[ss2])
            text1 = '{}{}'.format(valtext,betanamelist[targets2ndlevel_list[ss2]])
            text += text1 + ', '
        textlist += [text[:-1]]

    plt.close(windownum)
    fig1, axs = plt.subplots(nsources, 2, sharey=True, figsize=(12, 9), dpi=100, num=windownum)

    x = list(range(tsize))
    xx = x + x[::-1]
    tc1 = Sinput_avg[rtarget,:]
    tc1p = Sinput_sem[rtarget,:]
    tc1f = fit_avg[rtarget,:]
    tc1fp = fit_sem[rtarget,:]

    # Z1 = Sinput_reg[rtarget,:,3]
    # Z1f = fit_reg[rtarget,:,3]
    #
    # S = np.zeros(len(Z1)).astype(int)
    # for n in range(len(Z1)): c = np.where(Zthresh < Z1[n])[0];  S[n] = np.max(c)
    # Sf = np.zeros(len(Z1f)).astype(int)
    # for n in range(len(Z1f)): c = np.where(Zthresh < Z1f[n])[0];  Sf[n] = np.max(c)

    y1 = list(tc1f+tc1fp)
    y2 = list(tc1f-tc1fp)
    yy = y1 + y2[::-1]
    axs[1,1].plot(x, tc1, '-ob', linewidth=1, markersize=4)
    # axs[1,1].plot(tc1+tc1p, '-b')
    # axs[1,1].plot(tc1-tc1p, '--b')
    axs[1,1].plot(x, tc1f, '-xr', linewidth=1, markersize=4)
    axs[1,1].fill(xx,yy, facecolor=(1,0,0), edgecolor='None', alpha = 0.2)
    axs[1,1].plot(x, tc1f+tc1fp, color = (1,0,0), linestyle = '-', linewidth = 0.5)
    # axs[1,1].plot(x, tc1f-tc1fp, '--r')
    axs[1,1].set_title('target input {}'.format(rnamelist[rtarget]))

    # add marks for significant slope wrt pain
    ymax = np.max(np.abs(yy))
    # for n,s in enumerate(S):
    #     if s > 0: axs[1,1].annotate(symbollist[s], xy = (x[n]-0.25, ymax), fontsize=8)

    for ss in range(nsources):
        tc1 = Sconn_avg[sources[ss], :]
        tc1p = Sconn_sem[sources[ss], :]

        # Z1 = Sconn_reg[sources[ss], :, 3]
        # S = np.zeros(len(Z1)).astype(int)
        # for n in range(len(Z1)): c = np.where(Zthresh < Z1[n])[0];  S[n] = np.max(c)

        y1 = list(tc1 + tc1p)
        y2 = list(tc1 - tc1p)
        yy = y1 + y2[::-1]
        axs[ss,0].plot(x, tc1, '-xr')
        axs[ss,0].fill(xx,yy, facecolor=(1,0,0), edgecolor='None', alpha = 0.2)
        # axs[ss,0].plot(x, tc1+tc1p, '-r')
        # axs[ss,0].plot(x, tc1-tc1p, '--r')
        axs[ss,0].plot(x, tc1+tc1p, color = (1,0,0), linestyle = '-', linewidth = 0.5)
        if rsources[ss] >= nregions:
            axs[ss, 0].set_title('source output {} {}'.format(betanamelist[sources[ss]], 'int'))
        else:
            axs[ss,0].set_title('source output {} {}'.format(betanamelist[sources[ss]], rnamelist[rsources[ss]]))
        axs[ss,0].annotate(textlist[ss], xy=(.025, .025), xycoords='axes  fraction',
                horizontalalignment='left', verticalalignment='bottom', fontsize=10)

        # add marks for significant slope wrt pain
        # ymax = np.max(np.abs(yy))
        # for n, s in enumerate(S):
        #     if s > 0: axs[ss,0].annotate(symbollist[s], xy = (x[n]-0.25, ymax), fontsize=8)

        if setylim:
            axs[ss,0].set_ylim((ymin,ymax))
    # p, f = os.path.split(SAPMresultsname)
    svgname = os.path.join(outputdir, 'Avg_' + nametag1 + '.svg')
    plt.savefig(svgname)

    return svgname


def plot_region_inputs_regression(windownum, target, nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrange = []):
    Zthresh = stats.norm.ppf(1-np.array([1.0, 0.05,0.01,0.001]))
    symbollist = [' ','*', chr(8868),chr(8903)]

    if len(yrange) > 0:
        setylim = True
        ymin = yrange[0]
        ymax = yrange[1]
    else:
        setylim = False

    rtarget = rnamelist.index(target)
    m = Minput[rtarget, :]
    sources = np.where(m == 1)[0]
    rsources = [beta_list[ss]['pair'][0] for ss in sources]
    nsources = len(sources)
    nregions = len(rnamelist)
    checkdims = np.shape(Sinput_reg)
    if np.ndim(Sinput_reg) > 2:  nv = checkdims[2]
    tsize = checkdims[1]

    # get beta values from Mconn
    m = Mconn_avg[:,sources[0]]
    targets2ndlevel_list = np.where(m != 0.)[0]
    textlist = []
    for ss in sources:
        text = betanamelist[ss] + ': '
        beta = Mconn_avg[targets2ndlevel_list,ss]
        for ss2 in range(len(beta)):
            # signtext = 'none '
            # if beta[ss2] > 0:
            #     signtext = 'Excit '
            # if beta[ss2] < 0:
            #     signtext = 'Inhib '

            valtext = '{:.2f} '.format(beta[ss2])
            text1 = '{}{}'.format(valtext,betanamelist[targets2ndlevel_list[ss2]])
            text += text1 + ', '
        textlist += [text[:-1]]

    plt.close(windownum)
    fig1, axs = plt.subplots(nsources, 2, sharey=True, figsize=(12, 9), dpi=100, num=windownum)

    x = list(range(tsize))
    xx = x + x[::-1]
    tc1 = Sinput_reg[rtarget,:,0]
    tc1p = Sinput_reg[rtarget,:,1]
    tc1f = fit_reg[rtarget,:,0]
    tc1fp = fit_reg[rtarget,:,1]

    Z1 = Sinput_reg[rtarget,:,3]
    Z1f = fit_reg[rtarget,:,3]

    S = np.zeros(len(Z1)).astype(int)
    for n in range(len(Z1)): c = np.where(Zthresh < Z1[n])[0];  S[n] = np.max(c)
    Sf = np.zeros(len(Z1f)).astype(int)
    for n in range(len(Z1f)): c = np.where(Zthresh < Z1f[n])[0];  Sf[n] = np.max(c)

    y1 = list(tc1f+tc1fp)
    y2 = list(tc1f-tc1fp)
    yy = y1 + y2[::-1]
    axs[1,1].plot(x, tc1, '-ob', linewidth=1, markersize=4)
    # axs[1,1].plot(tc1+tc1p, '-b')
    # axs[1,1].plot(tc1-tc1p, '--b')
    axs[1,1].plot(x, tc1f, '-xr', linewidth=1, markersize=4)
    axs[1,1].fill(xx,yy, facecolor=(1,0,0), edgecolor='None', alpha = 0.2)
    axs[1,1].plot(x, tc1f+tc1fp, color = (1,0,0), linestyle = '-', linewidth = 0.5)
    # axs[1,1].plot(x, tc1f-tc1fp, '--r')
    axs[1,1].set_title('target input {}'.format(rnamelist[rtarget]))

    # add marks for significant slope wrt pain
    ymax = np.max(np.abs(yy))
    for n,s in enumerate(S):
        if s > 0: axs[1,1].annotate(symbollist[s], xy = (x[n]-0.25, ymax), fontsize=8)

    for ss in range(nsources):
        tc1 = Sconn_reg[sources[ss], :, 0]
        tc1p = Sconn_reg[sources[ss], :, 1]

        Z1 = Sconn_reg[sources[ss], :, 3]
        S = np.zeros(len(Z1)).astype(int)
        for n in range(len(Z1)): c = np.where(Zthresh < Z1[n])[0];  S[n] = np.max(c)

        y1 = list(tc1 + tc1p)
        y2 = list(tc1 - tc1p)
        yy = y1 + y2[::-1]
        axs[ss,0].plot(x, tc1, '-xr')
        axs[ss,0].fill(xx,yy, facecolor=(1,0,0), edgecolor='None', alpha = 0.2)
        # axs[ss,0].plot(x, tc1+tc1p, '-r')
        # axs[ss,0].plot(x, tc1-tc1p, '--r')
        axs[ss,0].plot(x, tc1+tc1p, color = (1,0,0), linestyle = '-', linewidth = 0.5)
        if rsources[ss] >= nregions:
            axs[ss, 0].set_title('source output {} {}'.format(betanamelist[sources[ss]], 'int'))
        else:
            axs[ss,0].set_title('source output {} {}'.format(betanamelist[sources[ss]], rnamelist[rsources[ss]]))
        axs[ss,0].annotate(textlist[ss], xy=(.025, .025), xycoords='axes  fraction',
                horizontalalignment='left', verticalalignment='bottom', fontsize=10)

        # add marks for significant slope wrt pain
        ymax = np.max(np.abs(yy))
        for n, s in enumerate(S):
            if s > 0: axs[ss,0].annotate(symbollist[s], xy = (x[n]-0.25, ymax), fontsize=8)

        if setylim:
            axs[ss,0].set_ylim((ymin,ymax))
    # p, f = os.path.split(SAPMresultsname)
    svgname = os.path.join(outputdir, 'Reg_' + nametag1 + '.svg')
    plt.savefig(svgname)

    return svgname


def plot_region_fits(window, regionlist, nametag, Sinput_avg, Sinput_sem, fit_avg, fit_sem, rnamelist, outputdir, yrange = []):
    if len(yrange) > 0:
        setylim = True
        ymin = yrange[0]
        ymax = yrange[1]
    else:
        setylim = False

    ndisplay = len(regionlist)

    # show regions (inputs real and fit)
    plt.close(window)
    fig2, axs = plt.subplots(ndisplay, sharey=False, figsize=(12, 6), dpi=100, num=window)

    Rtext_record = []
    Rval_record = []
    for nn in range(ndisplay):
        tc1 = Sinput_avg[regionlist[nn], :]
        tcf1 = fit_avg[regionlist[nn], :]
        t = np.array(range(len(tc1)))

        if len(Sinput_sem) > 0:
            tc1_sem = Sinput_sem[regionlist[nn], :]
            axs[nn].errorbar(t, tc1, tc1_sem, marker = 'o', markerfacecolor = 'b', markeredgecolor = 'b', linestyle = '-', color = 'b', linewidth=1, markersize=4)
        else:
            axs[nn].plot(t, tc1, '-ob', linewidth=1, markersize=4)

        if len(fit_sem) > 0:
            tcf1_sem = fit_sem[regionlist[nn], :]
            axs[nn].errorbar(t, tcf1, tcf1_sem, marker = 'o', markerfacecolor = 'r', markeredgecolor = 'r', linestyle = '-', color = 'r', linewidth=1, markersize=4)
        else:
            axs[nn].plot(t, tcf1, '-xr', linewidth=1, markersize=4)

        axs[nn].set_title('target {}'.format(rnamelist[regionlist[nn]]))
        if setylim:
            axs[nn].set_ylim((ymin,ymax))

        ssq = np.sum((tc1-np.mean(tc1))**2)
        dtc = tc1-tcf1
        ssqd = np.sum((dtc-np.mean(dtc))**2)
        R2fit = 1-ssqd/ssq

        R = np.corrcoef(tc1,tcf1)
        Rtext = 'target {}  Rfit = {:.2f}'.format(rnamelist[regionlist[nn]], R[0,1])
        print(Rtext)
        Rval = R[0,1]
        Rtext_record.append(Rtext)
        Rval_record.append([Rval])

    # p, f = os.path.split(SAPMresultsname)
    svgname = os.path.join(outputdir, 'Avg_' + nametag + '.svg')
    plt.savefig(svgname)

    return svgname, Rtext_record, Rval_record


# def write_Mconn_values(Mconn, Mconn_sem, betanamelist, rnamelist, beta_list, format = 'f', minthresh = 0.0, maxthresh = 0.0):
#     if maxthresh > minthresh:
#         maxlim = True
#     else:
#         maxlim = False
#     nregions = len(rnamelist)
#     nr1, nr2 = np.shape(Mconn)
#     if len(Mconn_sem) > 0:
#         write_sem = True
#     else:
#         write_sem = False
#     text_record = []
#     for n1 in range(nr1):
#         tname = betanamelist[n1]
#         tpair = beta_list[n1]['pair']
#         if tpair[0] >= nregions:
#             ts = 'int{}'.format(tpair[0]-nregions)
#         else:
#             ts = rnamelist[tpair[0]]
#             if len(ts) > 4:  ts = ts[:4]
#         tt = rnamelist[tpair[1]]
#         if len(tt) > 4:  tt = tt[:4]
#         text1 = '{}-{} input from '.format(ts,tt)
#         showval = False
#         for n2 in range(nr2):
#             if maxlim:
#                 check = np.abs(Mconn[n1,n2]) < maxthresh
#             else:
#                 check = np.abs(Mconn[n1,n2]) > minthresh
#             if check:
#                 showval = True
#                 sname = betanamelist[n2]
#                 spair = beta_list[n2]['pair']
#                 if spair[0] >= nregions:
#                     ss = 'int{}'.format(spair[0]-nregions)
#                 else:
#                     ss = rnamelist[spair[0]]
#                     if len(ss) > 4:  ss = ss[:4]
#                 st = rnamelist[spair[1]]
#                 if len(st) > 4:  st = st[:4]
#                 if format == 'f':
#                     if write_sem:
#                         texts = '{}-{} {:.3f} {} {:.3f} '.format(ss,st, Mconn[n1,n2],chr(177), Mconn_sem[n1,n2])
#                     else:
#                         texts = '{}-{} {:.3f}  '.format(ss,st, Mconn[n1,n2])
#                 else:
#                     if write_sem:
#                         texts = '{}-{} {:.3e} {} {:.3e} '.format(ss,st, Mconn[n1,n2],chr(177), Mconn_sem[n1,n2])
#                     else:
#                         texts = '{}-{} {:.3e}  '.format(ss,st, Mconn[n1,n2])
#                 text1 += texts
#         if showval:
#             print(text1)
#             text_record += [text1]
#     return text_record



def write_Mconn_values2(Mconn, Mconn_sem, NP, betanamelist, rnamelist, beta_list, format = 'f', pthresh = 0.05, sigflag = []):

    nregions = len(rnamelist)
    nr1, nr2 = np.shape(Mconn)

    if np.size(sigflag) == 0:
        sigflag = np.zeros(np.shape(Mconn))

    Tvals = Mconn/(Mconn_sem + 1.0e-20)
    Tthresh = stats.t.ppf(1 - pthresh, NP - 1)
    if np.isnan(Tthresh):  Tthresh = 0.0

    labeltext_record = []
    valuetext_record = []
    Ttext_record = []
    T_record = []
    for n1 in range(nr1):
        tname = betanamelist[n1]
        tpair = beta_list[n1]['pair']
        if tpair[0] >= nregions:
            ts = 'int{}'.format(tpair[0]-nregions)
        else:
            ts = rnamelist[tpair[0]]
            if len(ts) > 4:  ts = ts[:4]
        tt = rnamelist[tpair[1]]
        if len(tt) > 4:  tt = tt[:4]
        # text1 = '{}-{} input from '.format(ts,tt)
        showval = False
        for n2 in range(nr2):
            if (np.abs(Tvals[n1,n2]) > Tthresh)  or (sigflag[n1,n2]):
                showval = True
                sname = betanamelist[n2]
                spair = beta_list[n2]['pair']
                if spair[0] >= nregions:
                    ss = 'int{}'.format(spair[0]-nregions)
                else:
                    ss = rnamelist[spair[0]]
                    if len(ss) > 4:  ss = ss[:4]
                st = rnamelist[spair[1]]
                if len(st) > 4:  st = st[:4]

                labeltext = '{}-{}-{}'.format(ss, st, tt)
                T = Tvals[n1,n2]
                if format == 'f':
                    valuetext = '{:.3f} {} {:.3f} '.format(Mconn[n1, n2], chr(177), Mconn_sem[n1, n2])
                    Ttext = 'T = {:.2f} '.format(Tvals[n1,n2])
                else:
                    valuetext = '{:.3e} {} {:.3e} '.format(Mconn[n1, n2], chr(177), Mconn_sem[n1, n2])
                    Ttext = 'T = {:.2e} '.format(Tvals[n1,n2])

                labeltext_record += [labeltext]
                valuetext_record += [valuetext]
                Ttext_record += [Ttext]
                T_record += [T]
                if showval:
                    print(labeltext)
                    print(valuetext)
                    print(Ttext)
    return labeltext_record, valuetext_record, Ttext_record, T_record, Tthresh


# def write_Mcorr_values(Mconn, Mconn_sem, Rvals, Zvals, NP, betanamelist, rnamelist, beta_list, format = 'f', pthresh = 0.05, sigflag = []):
#
#     nregions = len(rnamelist)
#     nr1, nr2 = np.shape(Mconn)
#
#     if np.size(sigflag) == 0:
#         sigflag = np.zeros(np.shape(Mconn))
#
#     Zthresh = stats.norm.ppf(1 - pthresh)
#
#     labeltext_record = []
#     valuetext_record = []
#     Ztext_record = []
#     Rtext_record = []
#     for n1 in range(nr1):
#         tname = betanamelist[n1]
#         tpair = beta_list[n1]['pair']
#         if tpair[0] >= nregions:
#             ts = 'int{}'.format(tpair[0]-nregions)
#         else:
#             ts = rnamelist[tpair[0]]
#             if len(ts) > 4:  ts = ts[:4]
#         tt = rnamelist[tpair[1]]
#         if len(tt) > 4:  tt = tt[:4]
#         # text1 = '{}-{} input from '.format(ts,tt)
#         showval = False
#         for n2 in range(nr2):
#             if (np.abs(Zvals[n1,n2]) > Zthresh)  or (sigflag[n1,n2]):
#                 showval = True
#                 sname = betanamelist[n2]
#                 spair = beta_list[n2]['pair']
#                 if spair[0] >= nregions:
#                     ss = 'int{}'.format(spair[0]-nregions)
#                 else:
#                     ss = rnamelist[spair[0]]
#                     if len(ss) > 4:  ss = ss[:4]
#                 st = rnamelist[spair[1]]
#                 if len(st) > 4:  st = st[:4]
#
#                 labeltext = '{}-{}-{}'.format(ss, st, tt)
#                 if format == 'f':
#                     valuetext = '{:.3f} {} {:.3f} '.format(Mconn[n1, n2], chr(177), Mconn_sem[n1, n2])
#                     Ztext = 'Z = {:.2f} '.format(Zvals[n1,n2])
#                     Rtext = 'R = {:.3f} '.format(Rvals[n1,n2])
#                 else:
#                     valuetext = '{:.3e} {} {:.3e} '.format(Mconn[n1, n2], chr(177), Mconn_sem[n1, n2])
#                     Ztext = 'Z = {:.2e} '.format(Zvals[n1,n2])
#                     Rtext = 'R = {:.3e} '.format(Rvals[n1,n2])
#
#                 labeltext_record += [labeltext]
#                 valuetext_record += [valuetext]
#                 Ztext_record += [Ztext]
#                 Rtext_record += [Rtext]
#                 if showval:
#                     print(labeltext)
#                     print(valuetext)
#                     print(Ztext)
#                     print(Rtext)
#     return labeltext_record, valuetext_record, Ztext_record, Rtext_record


def write_Mreg_values(Mint, Mslope, R2, betanamelist, rnamelist, beta_list, format = 'f', R2thresh = 0.1, sigflag = []):

    nregions = len(rnamelist)
    nr1, nr2 = np.shape(Mslope)

    if np.size(sigflag) == 0:
        sigflag = np.zeros(np.shape(Mslope))

    labeltext_record = []
    inttext_record = []
    slopetext_record = []
    R2text_record = []
    R2_record = []
    for n1 in range(nr1):
        tname = betanamelist[n1]
        tpair = beta_list[n1]['pair']
        if tpair[0] >= nregions:
            ts = 'int{}'.format(tpair[0]-nregions)
        else:
            ts = rnamelist[tpair[0]]
            if len(ts) > 4:  ts = ts[:4]
        tt = rnamelist[tpair[1]]
        if len(tt) > 4:  tt = tt[:4]
        # text1 = '{}-{} input from '.format(ts,tt)
        showval = False
        for n2 in range(nr2):
            if (np.abs(R2[n1,n2]) > R2thresh) or sigflag[n1,n2]:
                showval = True
                sname = betanamelist[n2]
                spair = beta_list[n2]['pair']
                if spair[0] >= nregions:
                    ss = 'int{}'.format(spair[0]-nregions)
                else:
                    ss = rnamelist[spair[0]]
                    if len(ss) > 4:  ss = ss[:4]
                st = rnamelist[spair[1]]
                if len(st) > 4:  st = st[:4]
                labeltext = '{}-{}-{}'.format(ss, st, tt)
                if format == 'f':
                    inttext = '{:.3f}'.format(Mint[n1, n2])
                    slopetext = '{:.3f}'.format(Mslope[n1, n2])
                    R2text = 'R2 = {:.2f}'.format(R2[n1,n2])
                else:
                    inttext = '{:.3e}'.format(Mint[n1, n2])
                    slopetext = '{:.3e}'.format(Mslope[n1, n2])
                    R2text = 'R2 = {:.2e}'.format(R2[n1,n2])

                labeltext_record += [labeltext]
                inttext_record += [inttext]
                slopetext_record += [slopetext]
                R2text_record += [R2text]
                R2_record += [R2[n1,n2]]
                if showval:
                    print(labeltext)
                    print(inttext)
                    print(slopetext)
                    print(R2text)
    return labeltext_record, inttext_record, slopetext_record, R2text_record, R2_record, R2thresh


# def display_Mconn_properties(SAPMresultsname, rnamelist, betanamelist):
#     SAPMresults_load = np.load(SAPMresultsname, allow_pickle=True)
#     NP = len(SAPMresults_load)
#     Mconn = SAPMresults_load[0]['Mconn']
#     nr1, nr2 = np.shape(Mconn)
#     Mrecord = np.zeros((nr1,nr2,NP))
#     for nperson in range(NP):
#         Sinput = SAPMresults_load[nperson]['Sinput']
#         Sconn= SAPMresults_load[nperson]['Sconn']
#         Minput = SAPMresults_load[nperson]['Minput']
#         Mconn = SAPMresults_load[nperson]['Mconn']
#         Mrecord[:,:,nperson] = Mconn
#
#     Mpos = np.zeros(np.shape(Mrecord))
#     Mpos[Mrecord > 0] = 1
#     Mneg = np.zeros(np.shape(Mrecord))
#     Mneg[Mrecord < 0] = 1
#     Mposneg = np.sum(Mpos, axis = 2) - np.sum(Mneg, axis = 2)
#
#     columns = [name[:3] + ' in' for name in rnamelist]
#     columns += ['int1 in', 'int2 in']
#     rows = [name[:3] for name in rnamelist]
#     rows += ['int1', 'int2']
#
#     df = pd.DataFrame(Mposneg, columns=columns, index=rows)
#     pd.set_option('display.max_rows', None)
#     pd.set_option('display.max_columns', None)
#     pd.set_option('display.width', None)
#     pd.set_option('display.max_colwidth', None)
#
#     pd.options.display.float_format = '{:.0f}'.format
#     print(df)
#
#     p, f = os.path.split(SAPMresultsname)
#     xlname = os.path.join(p, 'Moutput_pos_neg_counts.xlsx')
#     df.to_excel(xlname)


# def display_SAPM_results_1person(nperson, Sinput, fit, regionlist, nruns, tsize, windowlist = [24]):
#     # show results
#     err = Sinput - fit
#     Smean = np.mean(Sinput)
#     errmean = np.mean(err)
#     R2total = 1 - np.sum((err-errmean)**2)/np.sum((Sinput-Smean)**2)
#     tsize_total = nruns*tsize
#
#     show_nregions = len(regionlist)
#     if len(windowlist) < show_nregions:
#         w = windowlist[0]
#         windowlist = [w+a for a in range(show_nregions)]
#
#     results_text_output = []
#     for rr in range(show_nregions):
#         regionnum = regionlist[rr]
#         windownum = windowlist[rr]
#
#         tc = Sinput[regionnum,:]
#         tc1 = np.mean(np.reshape(tc, (nruns, tsize)), axis=0)
#         tc = fit[regionnum,:]
#         tcf1 = np.mean(np.reshape(tc, (nruns, tsize)), axis=0)
#
#         plt.close(windownum)
#         fig = plt.figure(windownum, figsize=(12.5, 3.5), dpi=100)
#         plt.plot(range(tsize),tc1,'-ob')
#         plt.plot(range(tsize),tcf1,'-xr')
#
#         R1 = np.corrcoef(Sinput[regionnum, :], fit[regionnum, :])
#         Z1 = np.arctanh(R1[0, 1]) * np.sqrt(tsize_total - 3)
#         results_text = 'person {} region {}   R = {:.3f}  Z = {:.2f}'.format(nperson, regionnum, R1[0, 1], Z1)
#         print(results_text)
#
#         results_text_output += [results_text]
#
#     return results_text_output


def plot_correlated_results(SAPMresultsname, SAPMparametersname, connection_name, covariates, figurenumber = 1):
    outputdir = r'D:\threat_safety_python\individual_differences\fixed_C6RD0'
    # SAPMresultsname = os.path.join(outputdir, 'SEMphysio_model.npy')
    SAPMresults_load = np.load(SAPMresultsname, allow_pickle=True)

    # SAPMparametersname = os.path.join(outputdir, 'SEMparameters_model5.npy')
    SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
    ctarget = SAPMparams['ctarget']
    csource = SAPMparams['csource']
    rnamelist = SAPMparams['rnamelist']
    betanamelist = SAPMparams['betanamelist']
    beta_list = SAPMparams['beta_list']
    Mconn = SAPMparams['Mconn']

    # for nperson in range(NP)
    NP = len(SAPMresults_load)
    nconn, tsize_full = np.shape(SAPMresults_load[0]['Sconn'])
    nbeta = np.shape(SAPMresults_load[0]['betavals'])[0]
    beta_record = np.zeros((NP,nbeta))
    for nn in range(NP):
        beta_record[nn,:] = SAPMresults_load[nn]['betavals']

    labeltext_record, sources_per_target, intrinsic_flag = betavalue_labels(csource, ctarget, rnamelist, betanamelist, beta_list, Mconn)

    x = labeltext_record.index(connection_name)
    beta = beta_record[:,x]

    # prep regression lines
    b, fit, R2 = pydisplay.simple_GLMfit(covariates, beta)

    plt.close(figurenumber)
    fig = plt.figure(figurenumber)
    plt.plot(covariates, beta, color=(0, 0, 0), linestyle='None', marker='o', markerfacecolor=(0, 0, 0),
                    markersize=4)
    plt.plot(covariates, fit, color=(0, 0, 0), linestyle='solid', marker='None')
    textlabel = '{}'.format(connection_name)
    plt.title(textlabel)


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


def display_anatomical_cluster(clusterdataname, targetnum, targetcluster, orientation = 'axial', regioncolor = [0,1,1]):
    # get the voxel coordinates for the target region
    clusterdata = np.load(clusterdataname, allow_pickle=True).flat[0]

    IDX = clusterdata['cluster_properties'][targetnum]['IDX']
    idxx = np.where(IDX == targetcluster)
    cx = clusterdata['cluster_properties'][targetnum]['cx'][idxx]
    cy = clusterdata['cluster_properties'][targetnum]['cy'][idxx]
    cz = clusterdata['cluster_properties'][targetnum]['cz'][idxx]

    templatename = 'ccbs'
    outputimg = pydisplay.pydisplayvoxelregionslice(templatename, cx, cy, cz, orientation, displayslice = [], colorlist = regioncolor)
    return outputimg


# make labels for each betavalue
def betavalue_labels(csource, ctarget, rnamelist, betanamelist, beta_list, Mconn):

    labeltext_record = []
    nregions = len(rnamelist)
    nbeta = len(csource)
    sources_per_target = np.zeros(nbeta)
    intrinsic_flag = np.zeros(nbeta)
    for nn in range(nbeta):
        n1 = ctarget[nn]
        n2 = csource[nn]

        target_row = Mconn[n1,:]
        check = np.where(target_row > 0,1,0)
        nsources_for_target = np.sum(check)  # for this connection, how many sources contribute, in total?
        sources_per_target[nn] = nsources_for_target

        tname = betanamelist[n1]
        tpair = beta_list[n1]['pair']
        if tpair[0] >= nregions:
            ts = 'int{}'.format(tpair[0]-nregions)
            intrinsic_flag[nn] = 1
        else:
            ts = rnamelist[tpair[0]]
            if len(ts) > 4:  ts = ts[:4]
        tt = rnamelist[tpair[1]]
        if len(tt) > 4:  tt = tt[:4]

        sname = betanamelist[n2]
        spair = beta_list[n2]['pair']
        if spair[0] >= nregions:
            ss = 'int{}'.format(spair[0] - nregions)
            intrinsic_flag[nn] = 1
        else:
            ss = rnamelist[spair[0]]
            if len(ss) > 4:  ss = ss[:4]
        st = rnamelist[spair[1]]
        if len(st) > 4:  st = st[:4]
        labeltext = '{}-{}-{}'.format(ss, st, tt)

        labeltext_record += [labeltext]

    return labeltext_record, sources_per_target, intrinsic_flag



def show_SAPM_timecourse_results(covariatesfile, SAPMparametersname, SAPMresultsname, group='all', windowoffset = 0, yrange = [], yrange2 = []):
    # settings = np.load(settingsfile, allow_pickle=True).flat[0]
    # covariates1 = settings['GRPcharacteristicsvalues'][0]  # gender
    # covariates2 = settings['GRPcharacteristicsvalues'][1].astype(float)  # painrating

    covariatesdata = np.load(covariatesfile, allow_pickle=True).flat[0]
    if 'gender' in covariatesdata['GRPcharacteristicslist']:
        x = covariatesdata['GRPcharacteristicslist'].index('gender')
        covariates1 = covariatesdata['GRPcharacteristicsvalues'][x]
    else:
        covariates1 = []
    if 'painrating' in covariatesdata['GRPcharacteristicslist']:
        x = covariatesdata['GRPcharacteristicslist'].index('painrating')
        covariates2 = covariatesdata['GRPcharacteristicsvalues'][x].astype(float)
    else:
        covariates2 = []

    SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
    network = SAPMparams['network']
    beta_list = SAPMparams['beta_list']
    betanamelist = SAPMparams['betanamelist']
    nruns_per_person = SAPMparams['nruns_per_person']
    rnamelist = SAPMparams['rnamelist']
    fintrinsic_count = SAPMparams['fintrinsic_count']
    fintrinsic_region = SAPMparams['fintrinsic_region']
    vintrinsic_count = SAPMparams['vintrinsic_count']
    nclusterlist = SAPMparams['nclusterlist']
    tplist_full = SAPMparams['tplist_full']
    tcdata_centered = SAPMparams['tcdata_centered']
    ctarget = SAPMparams['ctarget']
    csource = SAPMparams['csource']
    tsize = SAPMparams['tsize']
    timepoint = SAPMparams['timepoint']
    epoch = SAPMparams['epoch']
    Nintrinsic = fintrinsic_count + vintrinsic_count
    # end of reloading parameters-------------------------------------------------------

    # load the SEM results
    SAPMresults_load = np.load(SAPMresultsname, allow_pickle=True)

    # for nperson in range(NP)
    NP = len(SAPMresults_load)
    resultscheck = np.zeros((NP, 4))
    nbeta, tsize_full = np.shape(SAPMresults_load[0]['Sconn'])
    ncon = nbeta - Nintrinsic

    if epoch >= tsize:
        et1 = 0
        et2 = tsize
    else:
        et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
        et2 = (timepoint + np.floor(epoch / 2)).astype(int)
    ftemp = paradigm_centered[et1:et2]

    Mrecord = np.zeros((nbeta, nbeta, NP))
    R2totalrecord = np.zeros(NP)
    for nperson in range(NP):
        Sinput = SAPMresults_load[nperson]['Sinput']
        Sconn = SAPMresults_load[nperson]['Sconn']
        Minput = SAPMresults_load[nperson]['Minput']
        Mconn = SAPMresults_load[nperson]['Mconn']
        beta_int1 = SAPMresults_load[nperson]['beta_int1']
        R2total = SAPMresults_load[nperson]['R2total']
        Meigv = SAPMresults_load[nperson]['Meigv']
        betavals = SAPMresults_load[nperson]['betavals']

        nruns = nruns_per_person[nperson]
        fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])

        # ---------------------------------------------------
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                                 vintrinsic_count, beta_int1, fintrinsic1)

        nr, tsize_total = np.shape(Sinput)
        tsize = (tsize_total / nruns).astype(int)
        nbeta,tsize2 = np.shape(Sconn)

        if nperson == 0:
            Sinput_total = np.zeros((nr,tsize, NP))
            Sconn_total = np.zeros((nbeta,tsize, NP))
            fit_total = np.zeros((nr,tsize, NP))

        tc = Sinput
        tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
        Sinput_total[:,:,nperson] = tc1

        tc = Sconn
        tc1 = np.mean(np.reshape(tc, (nbeta, nruns, tsize)), axis=1)
        Sconn_total[:,:,nperson] = tc1

        tc = fit
        tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
        fit_total[:,:,nperson] = tc1

        Mrecord[:, :, nperson] = Mconn
        R2totalrecord[nperson] = R2total

    if len(covariates1) > 1 and len(covariates2) > 1:
        # ancova sex x pain rating---------------------------------------
        # ANCOVA group vs pain rating
        statstype = 'ANCOVA'
        formula_key1 = 'C(Group)'
        formula_key2 = 'pain'
        formula_key3 = 'C(Group):' + 'pain'
        atype = 2

        # separate by sex
        g1 = np.where(covariates1 == 'Female')[0]
        g2 = np.where(covariates1 == 'Male')[0]

        cov1 = covariates2[g1]
        cov2 = covariates2[g2]
        ancova_p = np.ones((nbeta,nbeta,3))
        ttest_p = np.ones((nbeta,nbeta,2))
        for aa in range(ncon):
            for bb in range(ncon):
                m = Mrecord[aa, bb, :]
                if np.var(m) > 0:
                    b1 = m[g1]
                    b2 = m[g2]
                    anova_table, p_MeoG, p_MeoC, p_intGC = py2ndlevelanalysis.run_ANOVA_or_ANCOVA2(b1, b2, cov1, cov2, 'pain', formula_key1,
                                                                                formula_key2, formula_key3, atype)
                    ancova_p[aa,bb, :] = np.array([p_MeoG, p_MeoC, p_intGC])

                    if (np.var(b1) > 0) & (np.var(b2) > 0):
                        t, p = stats.ttest_ind(b1, b2, equal_var=False)
                        ttest_p[aa,bb,:] = np.array([p,t])

        columns = [name[:3] + ' in' for name in betanamelist]
        rows = [name[:3] for name in betanamelist]

        p, f = os.path.split(SEMresultsname)
        pd.options.display.float_format = '{:.2e}'.format
        df = pd.DataFrame(ancova_p[:,:,0], columns=columns, index=rows)
        xlname = os.path.join(p, 'Mancova_MeoG.xlsx')
        df.to_excel(xlname)
        df = pd.DataFrame(ancova_p[:,:,1], columns=columns, index=rows)
        xlname = os.path.join(p, 'Mancova_MeoP.xlsx')
        df.to_excel(xlname)
        df = pd.DataFrame(ancova_p[:,:,2], columns=columns, index=rows)
        xlname = os.path.join(p, 'Mancova_IntGP.xlsx')
        df.to_excel(xlname)
        df = pd.DataFrame(ttest_p[:,:,0], columns=columns, index=rows)
        xlname = os.path.join(p, 'Ttest_sexdiffs.xlsx')
        df.to_excel(xlname)


        print('\nMain effect of group:')
        text_MeoG = write_Mconn_values(ancova_p[:,:,0], [], betanamelist, rnamelist, beta_list, format = 'e', minthresh = 0.0, maxthresh = 0.05)
        print('\nMain effect of pain ratings:')
        text_MeoP = write_Mconn_values(ancova_p[:,:,1], [], betanamelist, rnamelist, beta_list, format = 'e', minthresh = 0.0, maxthresh = 0.05)
        print('\nInteraction group x pain:')
        text_IntGP = write_Mconn_values(ancova_p[:,:,2], [], betanamelist, rnamelist, beta_list, format = 'e', minthresh = 0.0, maxthresh = 0.05)
        print('\n\n')

        print('\nT-test sex differences:')
        text_T = write_Mconn_values(ttest_p[:,:,0], [], betanamelist, rnamelist, beta_list, format = 'e', minthresh = 0.0, maxthresh = 0.05)
        print('\n\n')


        # compare groups with T-tests

        # set the group
        g = list(range(NP))
        gtag = '_' + group

        if group.lower() == 'male':
            g = g2
            gtag = '_Male'

        if group.lower() == 'female':
            g = g1
            gtag = '_Female'


    # set the group
    g = list(range(NP))
    gtag = '_'+group
    Sinput_avg = np.mean(Sinput_total[:, :, g], axis=2)
    Sinput_sem = np.std(Sinput_total[:, :, g], axis=2) / np.sqrt(len(g))
    Sconn_avg = np.mean(Sconn_total[:, :, g], axis=2)
    Sconn_sem = np.std(Sconn_total[:, :, g], axis=2) / np.sqrt(len(g))
    fit_avg = np.mean(fit_total[:, :, g], axis=2)
    fit_sem = np.std(fit_total[:, :, g], axis=2) / np.sqrt(len(g))

    if len(covariates2) > 1:
        # regression based on pain ratings
        p = covariates2[np.newaxis, g]
        p -= np.mean(p)
        pmax = np.max(np.abs(p))
        p /= pmax
        G = np.concatenate((np.ones((1, len(g))),p), axis=0) # put the intercept term first
        Sinput_reg = np.zeros((nr,tsize,4))
        fit_reg = np.zeros((nr,tsize,4))
        Sconn_reg = np.zeros((nbeta,tsize,4))
        # Sinput_R2 = np.zeros((nr,tsize,2))
        # fit_R2 = np.zeros((nr,tsize,2))
        # Sconn_R2 = np.zeros((nbeta,tsize,2))
        for tt in range(tsize):
            for nn in range(nr):
                m = Sinput_total[nn,tt,g]
                b, fit, R2, total_var, res_var = pysem.general_glm(m, G)
                Sinput_reg[nn,tt,:2] = b
                Sinput_reg[nn,tt,-2:] = [R2, np.sign(R2)*np.arctanh(np.sqrt(np.abs(R2)))*np.sqrt(len(g)-3)]

                m = fit_total[nn,tt,g]
                b, fit, R2, total_var, res_var = pysem.general_glm(m, G)
                fit_reg[nn,tt,:2] = b
                fit_reg[nn,tt,-2:] = [R2, np.sign(R2)*np.arctanh(np.sqrt(np.abs(R2)))*np.sqrt(len(g)-3)]

            for nn in range(nbeta):
                m = Sconn_total[nn,tt,g]
                b, fit, R2, total_var, res_var = pysem.general_glm(m, G)
                Sconn_reg[nn,tt,:2] = b
                Sconn_reg[nn,tt,-2:] = [R2, np.sign(R2)*np.arctanh(np.sqrt(np.abs(R2)))*np.sqrt(len(g)-3)]

            # need to save Sinput_reg, Sinput_R2, etc., somewhere for later use....

            # regression of Mrecord with pain ratings
            # glm_fit
            Mregression = np.zeros((nbeta,nbeta,3))
            # Mregression1 = np.zeros((nbeta,nbeta,3))
            # Mregression2 = np.zeros((nbeta,nbeta,3))
            p = covariates2[np.newaxis, g]
            p -= np.mean(p)
            pmax = np.max(np.abs(p))
            p /= pmax
            G = np.concatenate((np.ones((1, len(g))), p), axis=0)  # put the intercept term first
            for aa in range(nbeta):
                for bb in range(nbeta):
                    m = Mrecord[aa,bb,g]
                    if np.var(m) > 0:
                        b, fit, R2, total_var, res_var = pysem.general_glm(m[np.newaxis,:], G)
                        Mregression[aa,bb,:] = [b[0,0],b[0,1],R2]

            print('\n\nMconn regression with pain ratings')
            # rtext = write_Mconn_values(Mregression[:,:,1], Mregression[:,:,2], betanamelist, rnamelist, beta_list, format='f', minthresh=0.0001, maxthresh=0.0)
            labeltext, inttext, slopetext, R2text, R2, R2thresh = write_Mreg_values(Mregression[:,:,1], Mregression[:,:,2], betanamelist, rnamelist, beta_list, format='f', Rthresh=0.1)

    # average Mconn values
    Mconn_avg = np.mean(Mrecord[:,:,g],axis = 2)
    Mconn_sem = np.std(Mrecord[:,:,g],axis = 2)/np.sqrt(len(g))
    # rtext = write_Mconn_values(Mconn_avg, Mconn_sem, betanamelist, rnamelist, beta_list,
    #                            format='f', minthresh=0.0001, maxthresh=0.0)

    pthresh = 0.05
    Tthresh = stats.t.ppf(1 - pthresh, NP - 1)

    print('\n\nAverage Mconn values')
    labeltext, valuetext, Ttext, T, Tthresh = write_Mconn_values2(Mconn_avg, Mconn_sem, NP, betanamelist, rnamelist, beta_list, format='f', pthresh=0.05)

    Rtextlist = [' ']*10
    Rvallist = [0]*10

    windownum_offset = windowoffset
    outputdir, f = os.path.split(SEMresultsname)
    # only show 3 regions in each plot for consistency in sizing
    # show some regions
    window2 = 25 + windownum_offset
    regionlist = [3,6,8]
    nametag = r'LC_NTS_PBN' + gtag
    svgname, Rtext, Rvals = plot_region_fits(window2, regionlist, nametag, Sinput_avg, Sinput_sem, fit_avg, fit_sem, rnamelist, outputdir, yrange)
    for n,x in enumerate(regionlist):
        print('Rtext = {}  Rvals = {}'.format(Rtext[n],Rvals[n]))
        Rtextlist[x] = Rtext[n]
        Rvallist[x] = Rvals[n]

    # show some regions
    window2 = 33 + windownum_offset
    regionlist = [0,5,1]
    nametag = r'cord_NRM_DRt' + gtag
    svgname, Rtext, Rvals = plot_region_fits(window2, regionlist, nametag, Sinput_avg, Sinput_sem, fit_avg, fit_sem, rnamelist, outputdir, yrange)
    for n,x in enumerate(regionlist):
        Rtextlist[x] = Rtext[n]
        Rvallist[x] = Rvals[n]

    window2b = 35 + windownum_offset
    regionlist = [0,5,4]
    nametag = r'cord_NRM_NGC' + gtag
    svgname, Rtext, Rvals = plot_region_fits(window2b, regionlist, nametag, Sinput_avg, Sinput_sem, fit_avg, fit_sem, rnamelist, outputdir, yrange)
    for n,x in enumerate(regionlist):
        Rtextlist[x] = Rtext[n]
        Rvallist[x] = Rvals[n]

    window2c = 36 + windownum_offset
    regionlist = [7,2,9]
    nametag = r'PAG_Hyp_Thal' + gtag
    svgname, Rtext, Rvals = plot_region_fits(window2c, regionlist, nametag, Sinput_avg, Sinput_sem, fit_avg, fit_sem, rnamelist, outputdir, yrange)
    for n,x in enumerate(regionlist):
        Rtextlist[x] = Rtext[n]
        Rvallist[x] = Rvals[n]


    # figure 0--------------------------------------------
    # inputs to C6RD
    window3 = 0 + windownum_offset
    target = 'C6RD'
    nametag1 = r'C6RDinput' + gtag
    if len(yrange2) > 0:
        ylim = yrange2[0]
        yrangethis = [-ylim,ylim]
    else:
        yrangethis = []
    plot_region_inputs_regression(window3, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    plot_region_inputs_average(window3+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    # yrange = [-1.5,1.5]
    # yrange = []
    # yrange2 = [-1.5,1.5]
    # yrange2 = []

    # show results
    # figure 1   -  inputs to NRM
    window1 = 1 + windownum_offset
    target = 'NRM'
    nametag1 = r'NRMinput' + gtag
    if len(yrange2) > 0:
        ylim = yrange2[1]
        yrangethis = [-ylim,ylim]
    else:
        yrangethis = []
    plot_region_inputs_regression(window1, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    plot_region_inputs_average(window1+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    # figure 2--------------------------------------------
    # inputs to NTS
    window4 = 2 + windownum_offset
    target = 'NTS'
    nametag1 = r'NTSinput' + gtag
    if len(yrange2) > 0:
        ylim = yrange2[2]
        yrangethis = [-ylim,ylim]
    else:
        yrangethis = []
    plot_region_inputs_regression(window4, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    plot_region_inputs_average(window4+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    # figure 3--------------------------------------------
    # inputs to NGC
    window7 = 3 + windownum_offset
    target = 'NGC'
    nametag1 = r'NGCinput' + gtag
    if len(yrange2) > 0:
        ylim = yrange2[3]
        yrangethis = [-ylim,ylim]
    else:
        yrangethis = []
    plot_region_inputs_regression(window7, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    plot_region_inputs_average(window7+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    # figure 4--------------------------------------------
    # inputs to PAG
    window5 = 4 + windownum_offset
    target = 'PAG'
    nametag1 = r'PAGinput' + gtag
    if len(yrange2) > 0:
        ylim = yrange2[4]
        yrangethis = [-ylim,ylim]
    else:
        yrangethis = []
    plot_region_inputs_regression(window5, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    plot_region_inputs_average(window5+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    # figure 5   -  inputs to PBN
    window11 = 5 + windownum_offset
    target = 'PBN'
    nametag1 = r'PBNinput' + gtag
    if len(yrange2) > 0:
        ylim = yrange2[5]
        yrangethis = [-ylim,ylim]
    else:
        yrangethis = []
    plot_region_inputs_regression(window11, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    plot_region_inputs_average(window11+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    # figure 6   -  inputs to LC
    window12 = 6 + windownum_offset
    target = 'LC'
    nametag1 = r'LCinput' + gtag
    if len(yrange2) > 0:
        ylim = yrange2[6]
        yrangethis = [-ylim,ylim]
    else:
        yrangethis = []
    plot_region_inputs_regression(window12, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    plot_region_inputs_average(window12+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    if len(covariates1) > 1 and len(covariates2) > 1:
        # plot a specific connection
        window6 = 29 + windownum_offset
        target = 'C6RD-PAG'
        source = 'NTS-C6RD'

        target = 'NTS-PBN'
        source = 'PBN-NTS'

        c = target.index('-')
        betaname_target = '{}_{}'.format(rnamelist.index(target[:c]),rnamelist.index(target[(c+1):]))
        c = source.index('-')
        betaname_source = '{}_{}'.format(rnamelist.index(source[:c]),rnamelist.index(source[(c+1):]))

        rt = betanamelist.index(betaname_target)
        rs = betanamelist.index(betaname_source)
        m = Mrecord[rt,rs,:]
        G = np.concatenate((covariates2[np.newaxis,g1],np.ones((1,len(g1)))), axis = 0)
        b1, fit1, R21, total_var, res_var = pysem.general_glm(m[g1], G)
        G = np.concatenate((covariates2[np.newaxis,g2],np.ones((1,len(g2)))), axis = 0)
        b2, fit2, R22, total_var, res_var = pysem.general_glm(m[g2], G)

        plt.close(window6)
        fig = plt.figure(window6)
        ax = fig.add_axes([0.1,0.1,0.8,0.8])
        plt.plot(covariates2[g1],m[g1], marker = 'o', linestyle = 'None', color = (0.1,0.9,0.1))
        plt.plot(covariates2[g1],fit1, marker = 'None', linestyle = '-', color = (0.1,0.9,0.1))
        plt.plot(covariates2[g2],m[g2], marker = 'o', linestyle = 'None', color = (1.0,0.5,0.))
        plt.plot(covariates2[g2],fit2, marker = 'None', linestyle = '-', color = (1.0,0.5,0.))

        ax.annotate('{} input to {}'.format(source,target), xy=(.025, .975), xycoords='axes  fraction', color = 'r',
                            horizontalalignment='left', verticalalignment='top', fontsize=10)
        ax.annotate('Female R2={:.3f}'.format(R21), xy=(.025, .025), xycoords='axes  fraction', color = (0.1,0.9,0.1),
                            horizontalalignment='left', verticalalignment='bottom', fontsize=10)
        ax.annotate('Male R2={:.3f}'.format(R22), xy=(.975, .025), xycoords='axes  fraction', color = (1.0,0.5,0.),
                            horizontalalignment='right', verticalalignment='bottom', fontsize=10)

    return Rtextlist, Rvallist



def display_SAPM_results(covariates, outputtype, outputdir, SAPMparametersname, SAPMresultsname, group, pthresh, descriptor, setylimits = []):
    # options of results to display:
    # 1) average input time-courses compared with model input
    # 2) modelled input signaling with corresponding source time-courses (outputs from source regions)
    # 3) t-test comparisons between groups, or w.r.t. zero (outputs to excel files)
    # 4) regression with continuous covariate

    # load SAPM parameters
    SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
    network = SAPMparams['network']
    beta_list = SAPMparams['beta_list']
    betanamelist = SAPMparams['betanamelist']
    nruns_per_person = SAPMparams['nruns_per_person']
    rnamelist = SAPMparams['rnamelist']
    fintrinsic_count = SAPMparams['fintrinsic_count']
    fintrinsic_region = SAPMparams['fintrinsic_region']
    vintrinsic_count = SAPMparams['vintrinsic_count']
    nclusterlist = SAPMparams['nclusterlist']
    tplist_full = SAPMparams['tplist_full']
    tcdata_centered = SAPMparams['tcdata_centered']
    ctarget = SAPMparams['ctarget']
    csource = SAPMparams['csource']
    tsize = SAPMparams['tsize']
    timepoint = SAPMparams['timepoint']
    epoch = SAPMparams['epoch']
    Nintrinsic = fintrinsic_count + vintrinsic_count
    # end of reloading parameters-------------------------------------------------------

    # load the SEM results
    SAPMresults_load = np.load(SAPMresultsname, allow_pickle=True)

    # for nperson in range(NP)
    NP = len(SAPMresults_load)
    resultscheck = np.zeros((NP, 4))
    nbeta, tsize_full = np.shape(SAPMresults_load[0]['Sconn'])
    ncon = nbeta - Nintrinsic
    paradigm_centered = SAPMresults_load[0]['fintrinsic_base']  # model paradigm used for fixed pattern latent inputs

    if epoch >= tsize:
        et1 = 0
        et2 = tsize
    else:
        et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
        et2 = (timepoint + np.floor(epoch / 2)).astype(int)
    ftemp = paradigm_centered[et1:et2]

    Mrecord = np.zeros((nbeta, nbeta, NP))
    R2totalrecord = np.zeros(NP)
    for nperson in range(NP):
        Sinput = SAPMresults_load[nperson]['Sinput']
        Sconn = SAPMresults_load[nperson]['Sconn']
        Minput = SAPMresults_load[nperson]['Minput']
        Mconn = SAPMresults_load[nperson]['Mconn']
        beta_int1 = SAPMresults_load[nperson]['beta_int1']
        R2total = SAPMresults_load[nperson]['R2total']
        Meigv = SAPMresults_load[nperson]['Meigv']
        betavals = SAPMresults_load[nperson]['betavals']

        nruns = nruns_per_person[nperson]
        fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])

        # ---------------------------------------------------
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                                 vintrinsic_count, beta_int1, fintrinsic1)
        nr, tsize_total = np.shape(Sinput)
        tsize = (tsize_total / nruns).astype(int)
        nbeta,tsize2 = np.shape(Sconn)

        if nperson == 0:
            Sinput_total = np.zeros((nr,tsize, NP))
            Sconn_total = np.zeros((nbeta,tsize, NP))
            fit_total = np.zeros((nr,tsize, NP))

        tc = Sinput
        tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
        Sinput_total[:,:,nperson] = tc1

        tc = Sconn
        tc1 = np.mean(np.reshape(tc, (nbeta, nruns, tsize)), axis=1)
        Sconn_total[:,:,nperson] = tc1

        tc = fit
        tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
        fit_total[:,:,nperson] = tc1

        Mrecord[:, :, nperson] = Mconn
        R2totalrecord[nperson] = R2total

    #----------------------------------------------------------------------------------
    # compare groups with T-tests------------------------------------------------------
    # or compare group average results to zero---------------------------------------
    # set the group
    # the input 'group' is a list of array indices for which data to use
    g = list(range(NP))
    if len(group) == NP:    # all values were selected for the group
        g1 = g
        g2 = []
    else:
        g1 = group
        g2 = [x for x in g if x not in g1]


    #-------------------------------------------------------------------------------
    #-------------prep for regression with continuous covariate------------------------------
    p = covariates[np.newaxis, g1]
    if len(np.unique(p)) > len(g1)/3:  # assume the values are continuous
        continuouscov = True
        p -= np.mean(p)
        G = np.concatenate((np.ones((1, len(g1))),p), axis=0) # put the intercept term first
    else:
        continuouscov = False

    #-------------------------------------------------------------------------------------
    # significance of average Mconn values -----------------------------------------------
    if outputtype == 'sig_Bvalues':
        Mconn_avg = np.mean(Mrecord[:,:,g1],axis = 2)
        Mconn_sem = np.std(Mrecord[:,:,g1],axis = 2)/np.sqrt(len(g1))
        # rtext = write_Mconn_values(Mconn_avg, Mconn_sem, betanamelist, rnamelist, beta_list,
        #                            format='f', minthresh=0.0001, maxthresh=0.0)
        # pthresh = 0.05
        # Tthresh = stats.t.ppf(1 - pthresh, NP - 1)
        print('\n\nAverage B values')
        labeltext, valuetext, Ttext, T, Tthresh = write_Mconn_values2(Mconn_avg, Mconn_sem, NP, betanamelist, rnamelist, beta_list, format='f', pthresh=pthresh)

        pthresh_list = ['{:.3e}'.format(pthresh)]*len(Ttext)
        Tthresh_list = ['{:.3f}'.format(Tthresh)]*len(Ttext)

        Rtextlist = [' ']*10
        Rvallist = [0]*10

        # sort output by magnitude of T
        si = np.argsort(np.abs(T))[::-1]
        T = np.array(T)[si]
        si2 = np.where(T < 1e3)  # dont write out values where B is always = 1
        textoutputs = {'regions': np.array(labeltext)[si[si2]], 'B': np.array(valuetext)[si[si2]], 'T': np.array(Ttext)[si[si2]],
                       'T thresh': np.array(Tthresh_list)[si[si2]], 'p thresh': np.array(pthresh_list)[si[si2]]}
        # p, f = os.path.split(SAPMresultsname)
        df = pd.DataFrame(textoutputs)
        xlname = os.path.join(outputdir, descriptor + '.xlsx')
        df.to_excel(xlname)


    #-------------------------------------------------------------------------------
    #-------------B-value regression with continuous covariate------------------------------
    # regression of Mrecord with continuous covariate
    # glm_fit
    if outputtype == 'Bvalue_regression':
        Mregression = np.zeros((nbeta,nbeta,3))
        for aa in range(nbeta):
            for bb in range(nbeta):
                m = Mrecord[aa,bb,g1]
                if np.var(m) > 0:
                    b, fit, R2, total_var, res_var = pysem.general_glm(m[np.newaxis,:], G)
                    Mregression[aa,bb,:] = [b[0,0],b[0,1],R2]

        Zthresh = stats.norm.ppf(1 - pthresh)
        # Z = arctanh(R)*sqrt(NP-1)
        Rthresh = np.tanh(Zthresh/np.sqrt(NP-1))
        R2thresh = Rthresh**2

        print('\n\nMconn regression with continuous covariate values')
        # rtext = write_Mconn_values(Mregression[:,:,1], Mregression[:,:,2], betanamelist, rnamelist, beta_list, format='f', minthresh=0.0001, maxthresh=0.0)
        labeltext, inttext, slopetext, R2text, R2, R2thresh = write_Mreg_values(Mregression[:,:,0], Mregression[:,:,1], Mregression[:,:,2], betanamelist, rnamelist, beta_list, format='f', R2thresh=R2thresh)

        if len(labeltext) > 0:
            pthresh_list = ['{:.3e}'.format(pthresh)] * len(inttext)
            R2thresh_list = ['{:.3f}'.format(R2thresh)] * len(inttext)

            # sort output by magnitude of R2
            si = np.argsort(np.abs(R2))[::-1]
            R2 = np.array(R2)[si]
            textoutputs = {'regions': np.array(labeltext)[si], 'int': np.array(inttext)[si],
                           'slope': np.array(slopetext)[si],'R2': np.array(R2text)[si],
                           'R2 thresh': np.array(R2thresh_list)[si], 'p thresh': np.array(pthresh_list)[si]}
            # p, f = os.path.split(SAPMresultsname)
            df = pd.DataFrame(textoutputs)
            xlname = os.path.join(outputdir, descriptor + '.xlsx')
            df.to_excel(xlname)
        else:
            print('Regression of B values with covariate ... no significant values found at p < {}'.format(pthresh))


    #-------------------------------------------------------------------------------
    # get the group averages etc-----------------------------------------------------
    Sinput_avg = np.mean(Sinput_total[:, :, g1], axis=2)
    Sinput_sem = np.std(Sinput_total[:, :, g1], axis=2) / np.sqrt(len(g1))
    Sconn_avg = np.mean(Sconn_total[:, :, g1], axis=2)
    Sconn_sem = np.std(Sconn_total[:, :, g1], axis=2) / np.sqrt(len(g1))
    fit_avg = np.mean(fit_total[:, :, g1], axis=2)
    fit_sem = np.std(fit_total[:, :, g1], axis=2) / np.sqrt(len(g1))

    #---------------------------------------------------------------------------
    # plot time-course averages and fits with continuous covariate--------------
    # need to specify which region to display ***
    # include options to display more than one region? ***
    # need to specify where to display it ***
    windownum_offset = windowoffset
    # show some regions
    window2 = 25 + windownum_offset
    regionlist = [3,6,8]
    nametag = r'LC_NTS_PBN' + gtag
    svgname, Rtext, Rvals = plot_region_fits(window2, regionlist, nametag, Sinput_avg, Sinput_sem, fit_avg, fit_sem, rnamelist, outputdir, yrange)

    for n,x in enumerate(regionlist):
        print('Rtext = {}  Rvals = {}'.format(Rtext[n],Rvals[n]))
        Rtextlist[x] = Rtext[n]
        Rvallist[x] = Rvals[n]


    #-------------------------------------------------------------------------------
    #-------------time-course regression with continuous covariate------------------------------
    # prepare time-course values to plot
    Sinput_reg = np.zeros((nr,tsize,4))
    fit_reg = np.zeros((nr,tsize,4))
    Sconn_reg = np.zeros((nbeta,tsize,4))
    for tt in range(tsize):
        for nn in range(nr):
            m = Sinput_total[nn,tt,g1]
            b, fit, R2, total_var, res_var = pysem.general_glm(m, G)
            Sinput_reg[nn,tt,:2] = b
            Sinput_reg[nn,tt,-2:] = [R2, np.sign(R2)*np.arctanh(np.sqrt(np.abs(R2)))*np.sqrt(len(g)-3)]

            m = fit_total[nn,tt,g1]
            b, fit, R2, total_var, res_var = pysem.general_glm(m, G)
            fit_reg[nn,tt,:2] = b
            fit_reg[nn,tt,-2:] = [R2, np.sign(R2)*np.arctanh(np.sqrt(np.abs(R2)))*np.sqrt(len(g1)-3)]

        for nn in range(nbeta):
            m = Sconn_total[nn,tt,g1]
            b, fit, R2, total_var, res_var = pysem.general_glm(m, G)
            Sconn_reg[nn,tt,:2] = b
            Sconn_reg[nn,tt,-2:] = [R2, np.sign(R2)*np.arctanh(np.sqrt(np.abs(R2)))*np.sqrt(len(g1)-3)]

        # need to save Sinput_reg, Sinput_R2, etc., somewhere for later use....



    #-------------------------------------------------------------------------------
    # plot region input time-courses averages and fits with continuous covariate----
    # inputs to C6RD
    window3 = 0 + windownum_offset
    target = 'C6RD'
    nametag1 = r'C6RDinput' + gtag
    if len(yrange2) > 0:
        ylim = yrange2[0]
        yrangethis = [-ylim,ylim]
    else:
        yrangethis = []
    plot_region_inputs_regression(window3, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    plot_region_inputs_average(window3+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    # yrange = [-1.5,1.5]
    # yrange = []
    # yrange2 = [-1.5,1.5]
    # yrange2 = []




    # show results
    # figure 1   -  inputs to NRM
    window1 = 1 + windownum_offset
    target = 'NRM'
    nametag1 = r'NRMinput' + gtag
    if len(yrange2) > 0:
        ylim = yrange2[1]
        yrangethis = [-ylim,ylim]
    else:
        yrangethis = []
    plot_region_inputs_regression(window1, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    plot_region_inputs_average(window1+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    # figure 2--------------------------------------------
    # inputs to NTS
    window4 = 2 + windownum_offset
    target = 'NTS'
    nametag1 = r'NTSinput' + gtag
    if len(yrange2) > 0:
        ylim = yrange2[2]
        yrangethis = [-ylim,ylim]
    else:
        yrangethis = []
    plot_region_inputs_regression(window4, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    plot_region_inputs_average(window4+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    # figure 3--------------------------------------------
    # inputs to NGC
    window7 = 3 + windownum_offset
    target = 'NGC'
    nametag1 = r'NGCinput' + gtag
    if len(yrange2) > 0:
        ylim = yrange2[3]
        yrangethis = [-ylim,ylim]
    else:
        yrangethis = []
    plot_region_inputs_regression(window7, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    plot_region_inputs_average(window7+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    # figure 4--------------------------------------------
    # inputs to PAG
    window5 = 4 + windownum_offset
    target = 'PAG'
    nametag1 = r'PAGinput' + gtag
    if len(yrange2) > 0:
        ylim = yrange2[4]
        yrangethis = [-ylim,ylim]
    else:
        yrangethis = []
    plot_region_inputs_regression(window5, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    plot_region_inputs_average(window5+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    # figure 5   -  inputs to PBN
    window11 = 5 + windownum_offset
    target = 'PBN'
    nametag1 = r'PBNinput' + gtag
    if len(yrange2) > 0:
        ylim = yrange2[5]
        yrangethis = [-ylim,ylim]
    else:
        yrangethis = []
    plot_region_inputs_regression(window11, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    plot_region_inputs_average(window11+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    # figure 6   -  inputs to LC
    window12 = 6 + windownum_offset
    target = 'LC'
    nametag1 = r'LCinput' + gtag
    if len(yrange2) > 0:
        ylim = yrange2[6]
        yrangethis = [-ylim,ylim]
    else:
        yrangethis = []
    plot_region_inputs_regression(window12, target,nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    plot_region_inputs_average(window12+100, target,nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis)

    if len(covariates1) > 1 and len(covariates2) > 1:
        # plot a specific connection
        window6 = 29 + windownum_offset
        target = 'C6RD-PAG'
        source = 'NTS-C6RD'

        target = 'NTS-PBN'
        source = 'PBN-NTS'

        c = target.index('-')
        betaname_target = '{}_{}'.format(rnamelist.index(target[:c]),rnamelist.index(target[(c+1):]))
        c = source.index('-')
        betaname_source = '{}_{}'.format(rnamelist.index(source[:c]),rnamelist.index(source[(c+1):]))

        rt = betanamelist.index(betaname_target)
        rs = betanamelist.index(betaname_source)
        m = Mrecord[rt,rs,:]
        G = np.concatenate((covariates2[np.newaxis,g1],np.ones((1,len(g1)))), axis = 0)
        b1, fit1, R21, total_var, res_var = pysem.general_glm(m[g1], G)
        G = np.concatenate((covariates2[np.newaxis,g2],np.ones((1,len(g2)))), axis = 0)
        b2, fit2, R22, total_var, res_var = pysem.general_glm(m[g2], G)

        plt.close(window6)
        fig = plt.figure(window6)
        ax = fig.add_axes([0.1,0.1,0.8,0.8])
        plt.plot(covariates2[g1],m[g1], marker = 'o', linestyle = 'None', color = (0.1,0.9,0.1))
        plt.plot(covariates2[g1],fit1, marker = 'None', linestyle = '-', color = (0.1,0.9,0.1))
        plt.plot(covariates2[g2],m[g2], marker = 'o', linestyle = 'None', color = (1.0,0.5,0.))
        plt.plot(covariates2[g2],fit2, marker = 'None', linestyle = '-', color = (1.0,0.5,0.))

        ax.annotate('{} input to {}'.format(source,target), xy=(.025, .975), xycoords='axes  fraction', color = 'r',
                            horizontalalignment='left', verticalalignment='top', fontsize=10)
        ax.annotate('Female R2={:.3f}'.format(R21), xy=(.025, .025), xycoords='axes  fraction', color = (0.1,0.9,0.1),
                            horizontalalignment='left', verticalalignment='bottom', fontsize=10)
        ax.annotate('Male R2={:.3f}'.format(R22), xy=(.975, .025), xycoords='axes  fraction', color = (1.0,0.5,0.),
                            horizontalalignment='right', verticalalignment='bottom', fontsize=10)

    return Rtextlist, Rvallist



#
#
# if __name__ == '__main__':
#     main()
#
