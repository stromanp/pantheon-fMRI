# functions to be called using multiprocessing
# these need to be defined in a separate function and then imported for
# the mutiprocessing to work

import numpy as np
import copy
import pysapm
import pysem
import multiprocessing as mp
import time

# import time
# import os
# import random
# import multiprocessing as mp
# import matplotlib.pyplot as plt
# import py2ndlevelanalysis
# import draw_sapm_diagram2 as dsd2
# from statsmodels.formula.api import ols
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import scipy.stats as stats
# import pandas as pd
# import statsmodels.api as sm
# import pyclustering
# import pydisplay
# from mpl_toolkits import mplot3d


#
# gradient descent method per person
def gradient_descent_per_person(data):
    # print('running gradient_descent_per_person (in parallel_functions.py)')
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
    # tsize = SEMparams['tsize']
    # tplist_full = SEMparams['tplist_full']
    # nruns_per_person = SEMparams['nruns_per_person']
    # nclusterlist = SEMparams['nclusterlist']
    # Minput = SEMparams['Minput']
    # fintrinsic_count = SEMparams['fintrinsic_count']
    # fintrinsic_region = SEMparams['fintrinsic_region']
    # vintrinsic_count = SEMparams['vintrinsic_count']
    # epoch = SEMparams['epoch']
    # timepoint = SEMparams['timepoint']
    # tcdata_centered = SEMparams['tcdata_centered']
    # ctarget = SEMparams['ctarget']
    # csource = SEMparams['csource']
    # latent_flag = SEMparams['latent_flag']
    # Mconn = SEMparams['Mconn']
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

    fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count,
                                                             beta_int1, fintrinsic1)
    # cost = np.sum(np.abs(betavals**2)) # L2 regularization
    cost = np.sum(np.abs(betavals))  # L1 regularization
    ssqd = err + Lweight * cost
    ssqd_starting = ssqd
    ssqd_record += [ssqd]

    # nitermax = 100
    # alpha_limit = 1.0e-4

    alphalist = initial_alpha * np.ones(len(betavals))
    alphabint = copy.deepcopy(initial_alpha)
    alphamax = copy.deepcopy(initial_alpha)
    iter = 0
    # vintrinsics_record = []
    converging = True
    dssq_record = np.ones(3)
    dssq_count = 0
    sequence_count = 0
    while alphamax > alpha_limit and iter < nitermax and converging:

        iter += 1
        # gradients in betavals and beta_int1
        Mconn[ctarget, csource] = betavals
        # fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
        #                                                          vintrinsic_count, beta_int1, fintrinsic1)
        # dssq_db, ssqd, dssq_dbeta1 = pysapm.gradients_for_betavals(Sinput, Minput, Mconn, betavals, ctarget, csource, dval,
        #                                                     fintrinsic_count, vintrinsic_count, beta_int1,
        #                                                     fintrinsic1, Lweight)

        betavals, beta_int1, fit, updatebflag, updatebintflag, dssq_db, dssq_dbeta1, ssqd, alphalist, alphabint = \
            pysapm.update_betavals_sequentially(Sinput, Minput, Mconn, betavals, ctarget, csource, dval,
                                         fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight,
                                         alphalist, alphabint)

        ssqd_record += [ssqd]

        err_total = Sinput - fit
        Smean = np.mean(Sinput)
        errmean = np.mean(err_total)
        # R2total = 1 - np.sum((err_total - errmean) ** 2) / np.sum((Sinput - Smean) ** 2)

        # R2list = [1-np.sum((Sinput[x,:]-fit[x,:])**2)/np.sum(Sinput[x,:]**2) for x in range(nregions)]
        R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
        R2avg = np.mean(R2list)
        R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)

        # Sinput_sim, Soutput_sim = network_sim(Sinput_full, Soutput_full, Minput, Moutput)
        results_record.append({'Sinput': Sinput, 'fit': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})
        atemp = np.append(alphalist, alphabint)
        alphamax = np.max(atemp)
        alphalist[alphalist < alpha_limit] = alpha_limit
        if alphabint < alpha_limit:  alphabint = copy.deepcopy(alpha_limit)

        ssqchange = ssqd - ssqd_old
        if np.abs(ssqchange) < 1e-5: converging = False


        # original method before change to sequentially updating beta values
        # ssqd_record += [ssqd]
        #
        # # fix some beta values at zero, if specified
        # if len(fixed_beta_vals) > 0:
        #     dssq_db[fixed_beta_vals] = 0
        #
        # # apply the changes
        # betavals -= alpha * dssq_db
        # beta_int1 -= alpha * dssq_dbeta1
        #
        # # limit the beta values related to intrinsic inputs to positive values
        # for aa in range(len(betavals)):
        #     if latent_flag[aa] > 0:
        #         # if betavals[aa] < 0:  betavals[aa] = 0.0
        #         betavals[aa] = 1.0
        #
        # # betavals[betavals >= betalimit] = betalimit
        # # betavals[betavals <= -betalimit] = -betalimit
        #
        # Mconn[ctarget, csource] = betavals
        # fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
        #                                                          vintrinsic_count, beta_int1, fintrinsic1)
        # # cost = np.sum(np.abs(betavals**2))  # L2 regularization
        # cost = np.sum(np.abs(betavals))  # L1 regularization
        # ssqd_new = err + Lweight * cost
        #
        # err_total = Sinput - fit
        # Smean = np.mean(Sinput)
        # errmean = np.mean(err_total)
        # R2total = 1 - np.sum((err_total - errmean) ** 2) / np.sum((Sinput - Smean) ** 2)
        # if R2total < 0: R2total = 0.0
        #
        # # Sinput_sim, Soutput_sim = network_sim(Sinput_full, Soutput_full, Minput, Moutput)
        # results_record.append({'Sinput': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})
        #
        # if ssqd_new >= ssqd:
        #     alpha *= 0.5
        #     # revert back to last good values
        #     betavals = copy.deepcopy(lastgood_betavals)
        #     beta_int1 = copy.deepcopy(lastgood_beta_int1)
        #     dssqd = ssqd - ssqd_new
        #     dssq_record = np.ones(3)  # reset the count
        #     dssq_count = 0
        #     sequence_count = 0
        #     if verbose: print('beta vals:  iter {} alpha {:.3e}  delta ssq > 0  - no update'.format(iter, alpha))
        # else:
        #     # save the good values
        #     lastgood_betavals = copy.deepcopy(betavals)
        #     lastgood_beta_int1 = copy.deepcopy(beta_int1)
        #
        #     dssqd = ssqd - ssqd_new
        #     ssqd = ssqd_new
        #
        #     sequence_count += 1
        #     if sequence_count > 5:
        #         alpha *= 1.5
        #         sequence_count = 0
        #
        #     dssq_count += 1
        #     dssq_count = np.mod(dssq_count, 3)
        #     # dssq_record[dssq_count] = 100.0 * dssqd / ssqd_starting
        #     dssq_record[dssq_count] = dssqd
        #     if np.max(dssq_record) < 0.1:  converging = False

        if verbose: print('beta vals:  iter {} alpha {:.3e}  delta ssq {:.4f}  relative: {:.1f} percent  '
                          'R2 {:.3f}'.format(iter, alpha, -dssqd, 100.0 * ssqd / ssqd_starting, R2total))
        # now repeat it ...

    # fit the results now to determine output signaling from each region
    Mconn[ctarget, csource] = betavals
    fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count,
                                                             beta_int1, fintrinsic1)
    Sconn = Meigv @ Mintrinsic  # signalling over each connection

    entry = {'Sinput': Sinput, 'Sconn': Sconn, 'beta_int1': beta_int1, 'Mconn': Mconn, 'Minput': Minput,
             'R2total': R2total, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv, 'betavals': betavals,
             'fintrinsic1': fintrinsic1, 'PCloadings': PCloadings, 'fintrinsic_base': fintrinsic_base}

    return entry


def main(input_data, nprocessors):
    startpool = time.time()
    pool = mp.Pool(nprocessors)
    print('runnning gradient_descent_per_person ... (with {} processors)'.format(nprocessors))
    SEMresults = pool.map(gradient_descent_per_person, input_data)
    pool.close()
    donepool = time.time()
    timeelapsed = donepool-startpool
    return SEMresults, timeelapsed

if __name__ == '__main__':
    main()