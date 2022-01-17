#-----------------------------------------------------------------------------------
#  apply structural equation modeling (SEM) using predefined clusters, region data
#  and neural network model
#
# 1) identify data sets to load
# 2) identify analysis type (1-source, 2-source, network)
# 3) identify network definition, if applicable
# 4) carry out sem analysis
# 5) display results

import pyclustering
import numpy as np
import copy
import time
import os

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


#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
def dataindices_from_sourceclusters(nclusterlist, sourcenums,index):
    vsize = nclusterlist[sourcenums]
    regionstart = np.array([np.sum(nclusterlist[:aa]) for aa in sourcenums])

    ndims = len(vsize)
    m = np.zeros(ndims)
    for nn in range(ndims):
        if nn == 0:
            m[0] = np.mod(index,vsize[0])
        else:
            m[nn] = np.mod( np.floor(index/np.prod(vsize[:nn])), vsize[nn])

    return m+regionstart


#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
def single_source_fit(y,X):
    # special case of y = b X for a single source
    CC = np.cov(y,X)
    b = CC[0,1]/CC[1,1]

    fit = b*X
    err = y - fit
    res_var = np.sum(err**2)
    total_var = np.sum(y**2)

    R2 = 1.0 - res_var / (total_var + 1.0e-20)
    if R2 > 1.0: R2 = 0.99999
    if R2 < -1.0: R2 = -0.99999

    return b, fit, R2, total_var, res_var


#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
def general_glm(y, X):
    '''
    separate from GLMfit functions to give consistent output forms with gradient_descent method
     y = b * X
    :param y:   dependent variable values [tsize]
    :param X:   independent varaible values [N x tsize]
    :return:  b0, fit, R2, total_var, res_var, iter, ssqd_record, gradnorm_record
    '''
    b = y @ X.T @ np.linalg.inv(X @ X.T)

    fit = b @ X
    err = y - fit
    res_var = np.sum((err-np.mean(err))**2)
    total_var = np.sum((y - np.mean(y))**2)

    R2 = 1.0 - res_var / (total_var + 1.0e-20)
    if R2 > 1.0: R2 = 0.99999
    if R2 < -1.0: R2 = -0.99999

    return b, fit, R2, total_var, res_var


#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
def gradient_descent_fullinfo(b0, y, X, Lweight, alpha, deltab, tol=1e-6, maxiter=1000):
    '''
     y = b * X with L1 regularization
    :param b0:  initial estimates of fit parameters [1 x N]
    :param y:   dependent variable values [tsize]
    :param X:   independent varaible values [N x tsize]
    :param Lweight:  weighting value for
    :param alpha:  the rate of updating values along the gradient descent
    :param deltab: the step size for updating b values
    :param tol: the tolerance criterion for stopping the gradient descent
    :param maxiter: the maximum number of iterations
    :return:  b0, fit, R2, total_var, res_var, iter, ssqd_record, gradnorm_record
    '''
    N = len(b0)
    fit0 = b0 @ X

    err0 = y - fit0
    ssqd_0 = np.sum(err0**2)

    if alpha == 0: alpha = 0.1/ssqd_0
    alpha0 = copy.deepcopy(alpha)
    alphamin = alpha0/1000.0
    if not hasattr(deltab, '__len__'):  deltab = deltab*np.ones(N)
    if (deltab == 0).any(): deltab = 0.1*np.ones(N)

    ssqd_record = np.zeros(maxiter)
    gradnorm_record = np.zeros(maxiter)

    iter = 0
    dssq_db = np.zeros(N)
    ssqd = copy.deepcopy(ssqd_0)

    while (alpha > alphamin) and (iter < maxiter):
        for aa in range(N):
            b = copy.deepcopy(b0)
            b[aa]= b[aa] + deltab[aa]
            fit = b @ X
            err = y - fit
            ssqdp = np.sum(err**2) + Lweight * np.sum(np.abs(b)) # L1 regularization
            dssq_db[aa] = (ssqdp - ssqd) / deltab[aa]

        b0 -= alpha * dssq_db
        fit = b0 @ X
        err = y - fit
        ssqd_old = copy.deepcopy(ssqd)
        ssqd = np.sum(err**2) + Lweight * np.sum(abs(b0)) # L1 regularization
        ssqd_record[iter] = copy.deepcopy(ssqd)
        gradnorm_record[iter] = np.linalg.norm(dssq_db)  # norm of the gradient used for each step
        if (ssqd_old - ssqd) < tol:
                alpha = alpha / 2.0
        iter = iter + 1

    fit = b0 @ X
    err = y - fit
    ssqd = np.sum(err**2)
    total_var = np.sum(y**2)
    res_var = copy.deepcopy(ssqd)
    R2 = 1.0 - res_var / (total_var + 1.0e-20)
    if R2 > 1.0: R2 = 0.99999
    if R2 < -1.0: R2 = -0.99999

    ssqd_record = ssqd_record[:iter]
    gradnorm_record = gradnorm_record[:iter]

    return b0, fit, R2, total_var, res_var, iter, ssqd_record, gradnorm_record



#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
def gradient_descent(b0, y, X, Lweight, alpha, deltab, tol=1e-6, maxiter=1000):
    '''
     y = b * X with L1 regularization
    :param b0:  initial estimates of fit parameters [1 x N]
    :param y:   dependent variable values [tsize]
    :param X:   independent varaible values [N x tsize]
    :param Lweight:  weighting value for
    :param alpha:  the rate of updating values along the gradient descent
    :param deltab: the step size for updating b values
    :param tol: the tolerance criterion for stopping the gradient descent
    :param maxiter: the maximum number of iterations
    :return:  b0, fit, R2, total_var, res_var, iter
    '''
    N = len(b0)
    fit0 = b0 @ X

    # fit0 = np.matmul(b0,X)
    # fit0 = b0 @ X

    err0 = y - fit0
    ssqd_0 = np.sum(err0**2)

    if alpha == 0: alpha = 0.1/ssqd_0
    alpha0 = copy.deepcopy(alpha)
    alphamin = alpha0/1000.0
    if not hasattr(deltab, '__len__'):  deltab = deltab*np.ones(N)
    if (deltab == 0).any(): deltab = 0.1*np.ones(N)

    iter = 0
    dssq_db = np.zeros(N)
    ssqd = copy.deepcopy(ssqd_0)

    while (alpha > alphamin) and (iter < maxiter):
        for aa in range(N):
            b = copy.deepcopy(b0)
            b[aa]= b[aa] + deltab[aa]
            fit = b @ X
            err = y - fit
            ssqdp = np.sum(err**2) + Lweight * np.sum(np.abs(b)) # L1 regularization
            dssq_db[aa] = (ssqdp - ssqd) / deltab[aa]

        b0 -= alpha * dssq_db
        fit = b0 @ X
        err = y - fit
        ssqd_old = copy.deepcopy(ssqd)
        ssqd = np.sum(err**2) + Lweight * np.sum(abs(b0)) # L1 regularization
        if (ssqd_old - ssqd) < tol:
                alpha = alpha / 2.0
        iter = iter + 1

    fit = b0 @ X
    err = y - fit
    ssqd = np.sum(err**2)
    total_var = np.sum(y**2)
    res_var = copy.deepcopy(ssqd)
    R2 = 1.0 - res_var / (total_var + 1.0e-20)
    if R2 > 1.0: R2 = 0.99999
    if R2 < -1.0: R2 = -0.99999

    return b0, fit, R2, total_var, res_var, iter



#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
def gradient_descent_single(b0, y, X, Lweight, alpha, deltab, tol=1e-6, maxiter=1000):
    '''
    special case of gradient descent with only 1 fit parameter (for comparisons)
     y = b * X with L1 regularization
    :param b0:  initial estimates of fit parameters [1 x N]
    :param y:   dependent variable values [tsize]
    :param X:   independent varaible values [N x tsize]
    :param Lweight:  weighting value for
    :param alpha:  the rate of updating values along the gradient descent
    :param deltab: the step size for updating b values
    :param tol: the tolerance criterion for stopping the gradient descent
    :param maxiter: the maximum number of iterations
    :return:  b0, fit, R2, total_var, res_var, iter, ssqd_record, gradnorm_record
    '''
    # N = len(b0)
    N = 1
    fit0 = b0*X

    err0 = y - fit0
    ssqd_0 = np.sum(err0**2)

    if alpha == 0: alpha = 0.1/ssqd_0
    alpha0 = copy.deepcopy(alpha)
    alphamin = alpha0/1000.0
    if not hasattr(deltab, '__len__'):  deltab = deltab*np.ones(N)
    if (deltab == 0).any(): deltab = 0.1*np.ones(N)

    ssqd_record = np.zeros(maxiter)
    gradnorm_record = np.zeros(maxiter)

    iter = 0
    dssq_db = np.zeros(N)
    ssqd = copy.deepcopy(ssqd_0)

    while (alpha > alphamin) and (iter < maxiter):
        for aa in range(N):
            b = copy.deepcopy(b0)
            b[aa]= b[aa] + deltab[aa]
            fit = b*X
            err = y - fit
            ssqdp = np.sum(err**2) + Lweight * np.sum(np.abs(b)) # L1 regularization
            dssq_db[aa] = (ssqdp - ssqd) / deltab[aa]

        b0 -= alpha * dssq_db
        fit = b0*X
        err = y - fit
        ssqd_old = copy.deepcopy(ssqd)
        ssqd = np.sum(err**2) + Lweight * np.sum(abs(b0)) # L1 regularization
        ssqd_record[iter] = copy.deepcopy(ssqd)
        gradnorm_record[iter] = np.linalg.norm(dssq_db)  # norm of the gradient used for each step
        if (ssqd_old - ssqd) < tol:
                alpha = alpha / 2.0
        iter = iter + 1

    fit = b0*X
    err = y - fit
    ssqd = np.sum(err**2)
    total_var = np.sum(y**2)
    res_var = copy.deepcopy(ssqd)
    R2 = 1.0 - res_var / (total_var + 1.0e-20)
    if R2 > 1.0: R2 = 0.99999
    if R2 < -1.0: R2 = -0.99999

    ssqd_record = ssqd_record[:iter]
    gradnorm_record = gradnorm_record[:iter]

    return b0, fit, R2, total_var, res_var, iter, ssqd_record, gradnorm_record


#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
# one-source and two-source sem
def pysem(cluster_properties, region_properties, timepoints = 0, epoch = 0):
    print('running pysem ...')

    # for each region, there is an entry in cluster_properties and in region_properties
    # clusterdef_entry = {'cx': cx, 'cy': cy, 'cz': cz, 'IDX': IDX, 'nclusters': nclusters, 'rname': rname,
    #                     'regionindex': regionindex, 'regionnum': regionnum}
    # regiondata_entry = {'tc': tc, 'tc_sem': tc_sem, 'nruns_per_person': nruns_per_person, 'tsize': tsize}
    nregions = len(cluster_properties)
    nclusterlist = [cluster_properties[i]['nclusters'] for i in range(nregions)]
    nclusterstotal = np.sum(nclusterlist)

    tsize = region_properties[0]['tsize']
    print('pysem:  tsize = {}'.format(tsize))
    nruns_per_person = region_properties[0]['nruns_per_person']
    NP = len(nruns_per_person)   # number of people in the data set

    tcdata = []
    for i in range(nregions):
        tc = region_properties[i]['tc']
        # nc = nclusterlist[i]
        if i == 0:
            tcdata = tc
        else:
            tcdata = np.append(tcdata,tc,axis=0)

    nclusters,tsize_full = tcdata.shape

    if epoch == 0:
        ntimepoints = 1  # use all the data (not dynamic)
    else:
        if not hasattr(timepoints,'__len__'):
            timepoints = [timepoints]
        ntimepoints = len(timepoints)

    # dynamic sem?
    # timepoints ?
    # epoch ?
    # create lists of the data timepoints that need to be extracted for each person/epoch
    # the result is a list of dictionaries, with size:  ntimepoints x NP
    if epoch == 0:  # use the full time-course, not dynamic
        tplist = []
        tplist1 = []
        for nn in range(NP):
            r1 = sum(nruns_per_person[:nn])
            r2 = sum(nruns_per_person[:(nn+1)])
            t1 = r1*tsize
            t2 = r2*tsize
            tpoints = list(range(t1,t2))
            tplist1.append({'tp':tpoints})
        tplist.append(tplist1)  # this is just for consistency when ntimepoints > 0
    else:
        tplist = []
        for ee in range(ntimepoints):
            et1 = (timepoints[ee] - np.floor(epoch/2)).astype(int)-1
            et2 = (timepoints[ee] + np.floor(epoch/2)).astype(int)
            tplist1 = []
            for nn in range(NP):
                r1 = sum(nruns_per_person[:nn])
                r2 = sum(nruns_per_person[:(nn+1)])
                tp = [] # initialize list
                tpoints = []
                for ee2 in range(r1,r2):
                    tp = list(range((ee2*tsize+et1),(ee2*tsize+et2)))
                    tpoints = tpoints + tp   # concatenate lists
                tplist1.append({'tp':tpoints})
            tplist.append(tplist1)

    tcdata_centered = copy.deepcopy(tcdata)
    for ttime in range(ntimepoints):
        for nn in range(NP):
            tp = tplist[ttime][nn]['tp']
            tcdata1 = tcdata[:, tp]
            for ee in range(nruns_per_person[nn]):
                tpe1 = ee * epoch
                tpe2 = (ee + 1) * epoch
                tcdata1_mean = np.mean(tcdata1[:, tpe1:tpe2], axis=1)
                # tcdata1_mean = np.repeat(tcdata1_mean[:,np.newaxis],epoch*nruns_per_person[nn],axis=1)
                tcdata1_mean = np.repeat(tcdata1_mean[:, np.newaxis], epoch, axis=1)
                tcdata1[:, tpe1:tpe2] = tcdata1[:, tpe1:tpe2] - tcdata1_mean  # need to set the mean for each epoch to zero
            tcdata_centered[:,tp] = tcdata1

    # change to setting data to mean = 0 when it is first loaded
    CCrecord = np.zeros((ntimepoints,NP,nclusters,nclusters))
    for ttime in range(ntimepoints):
        for nn in range(NP):
            tp = tplist[ttime][nn]['tp']

            print('time {}  person {}'.format(ttime+1,nn+1))
            print('tp includes {} points'.format(len(tp)))
            print('nruns is {} and tsize/run is {}'.format(nruns_per_person[nn],epoch))

            tcdata1 = tcdata_centered[:,tp]
            # variance/covariance grid
            covmatrix = np.cov(tcdata1)

            CCrecord[ttime,nn,:,:] = covmatrix

    print('pysem:  finished computing variance/covariance matrix ...')
    # covmatrix is the same as:
    # covmatrix2 = np.matmul(tcdata,tcdata.T)/(tsize_full-1)

    # get beta values for all combinations of targets and two sources, for each person
    Lweight = 1e-3
    alpha = 1e-3
    deltab = 0.01 * np.ones(2)
    tol = 1e-2
    maxiter = 250

    # initialize arrays for storing the results
    beta2 = np.zeros((nclusters, nclusters, nclusters, ntimepoints, NP, 2))
    beta1 = np.zeros((nclusters, nclusters, nclusters, ntimepoints, NP, 2))
    Zgrid2 = np.zeros((nclusters, nclusters, nclusters, ntimepoints, NP))
    Zgrid1_1 = np.zeros((nclusters, nclusters, nclusters, ntimepoints, NP))
    Zgrid1_2 = np.zeros((nclusters, nclusters, nclusters, ntimepoints, NP))

    for tregion in range(nregions):
        print('pysem: calculating for target region {} of {} regions ...{}'.format(tregion + 1, nregions, time.ctime()))
        t1 = np.sum(nclusterlist[:tregion]).astype(int)
        t2 = np.sum(nclusterlist[:(tregion+1)]).astype(int)
        tclusters = list(range(t1,t2))
        for tc in tclusters:
            targetdata = tcdata_centered[tc, :]
            # print('pysem: calculating for cluster {} of target {} s1 region {} s2 region {} ...{}'.format(tc+1,tregion,s1region,s2region,time.ctime(time.time())))

            s1regions = np.setdiff1d(list(range(nregions)), tregion)
            for s1region in s1regions:
                s11 = np.sum(nclusterlist[:s1region]).astype(int)
                s12 = np.sum(nclusterlist[:(s1region + 1)]).astype(int)
                s1clusters = list(range(s11, s12))
                for s1c in s1clusters:

                    s2regions = np.setdiff1d(list(range((s1region + 1), nregions)), tregion)
                    for s2region in s2regions:
                        # s2clusters = np.setdiff1d(list(range(nclusters)), [tclusters, s1clusters])
                        s21 = np.sum(nclusterlist[:s2region]).astype(int)
                        s22 = np.sum(nclusterlist[:(s2region + 1)]).astype(int)
                        s2clusters = list(range(s21, s22))
                        for s2c in s2clusters:

                            X = tcdata_centered[[s1c, s2c], :]
                            # do the SEM fit thing, per person:
                            for ttime in range(ntimepoints):
                                for nn in range(NP):
                                    tp = tplist[ttime][nn]['tp']

                                    targetdata1 = targetdata[tp]  # - np.mean(targetdata[tp])
                                    X1 = X[:,tp]
                                    X1mean = np.mean(X1,axis=1)
                                    X1 = X1 - np.repeat(X1mean[:,np.newaxis],len(tp),axis=1)

                                    b0 = np.zeros(2)
                                    b, fit, R2, total_var, res_var, iter = \
                                        gradient_descent(b0, targetdata1, X1, Lweight, alpha, deltab, tol, maxiter)

                                    b1, fit1, R21, total_var1, res_var1 = single_source_fit(targetdata1,X1[0,:])
                                    b2, fit2, R22, total_var2, res_var2 = single_source_fit(targetdata1,X1[1,:])

                                    # store the results ....
                                    Z = np.arctanh(np.sqrt(np.abs(R2))) * np.sqrt(nruns_per_person[nn] * epoch - 3)
                                    Z_1 = np.arctanh(np.sqrt(np.abs(R21))) * np.sqrt(nruns_per_person[nn] * epoch - 3)
                                    Z_2 = np.arctanh(np.sqrt(np.abs(R22))) * np.sqrt(nruns_per_person[nn] * epoch - 3)

                                    beta2[tc,s1c,s2c,ttime,nn,:] = b
                                    beta1[tc,s1c,s2c,ttime,nn,:] = [b1,b2]
                                    Zgrid2[tc,s1c,s2c,ttime,nn] = Z
                                    Zgrid1_1[tc,s1c,s2c,ttime,nn] = Z_1
                                    Zgrid1_2[tc,s1c,s2c,ttime,nn] = Z_2

    print('pysem:  finished calculating 2-source SEM values.  {}'.format(time.ctime()))

    return CCrecord, beta2, beta1, Zgrid2, Zgrid1_1, Zgrid1_2


#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
def pysem_network(cluster_properties, region_properties, networkmodel, timepoints, epoch, savedirectory, savenamelabel, resume_run = False):
    print('running pysem_network ...')
    network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkmodel)
    nclusterlist = np.array([ncluster_list[i]['nclusters'] for i in range(len(ncluster_list))])
    ntargets = len(network)
    nregions = len(cluster_properties)
    nclusterstotal = np.sum(nclusterlist)

    # for each region, there is an entry in cluster_properties and in region_properties
    # clusterdef_entry = {'cx': cx, 'cy': cy, 'cz': cz, 'IDX': IDX, 'nclusters': nclusters, 'rname': rname,
    #                     'regionindex': regionindex, 'regionnum': regionnum}
    # regiondata_entry = {'tc': tc, 'tc_sem': tc_sem, 'nruns_per_person': nruns_per_person, 'tsize': tsize}

    tsize = region_properties[0]['tsize']
    nruns_per_person = region_properties[0]['nruns_per_person']
    NP = len(nruns_per_person)   # number of people in the data set

    # pool the data into a convenient form
    tcdata = []
    for i in range(nregions):
        tc = region_properties[i]['tc']
        if i == 0:
            tcdata = tc
        else:
            tcdata = np.append(tcdata,tc,axis=0)

    nclusters,tsize_full = tcdata.shape

    # organize indices for extracting portions of the data based on people, epochs
    if epoch == 0:
        ntimepoints = 1  # use all the data (not dynamic)
    else:
        if not hasattr(timepoints,'__len__'):
            timepoints = [timepoints]
        ntimepoints = len(timepoints)

    # create lists of the data timepoints that need to be extracted for each person/epoch
    # the result is a list of dictionaries, with size:  ntimepoints x NP
    if epoch == 0:  # use the full time-course, not dynamic
        tplist = []
        tplist1 = []
        for nn in range(NP):
            r1 = np.sum(nruns_per_person[:nn])
            r2 = np.sum(nruns_per_person[:(nn+1)])
            t1 = r1*tsize
            t2 = r2*tsize
            tpoints = list(range(t1,t2))
            tplist1.append({'tp':tpoints})
        tplist.append(tplist1)  # this is just for consistency when ntimepoints > 0
    else:
        tplist = []
        for ee in range(ntimepoints):
            et1 = (timepoints[ee] - np.floor(epoch/2)).astype(int)-1
            et2 = (timepoints[ee] + np.floor(epoch/2)).astype(int)
            tplist1 = []
            for nn in range(NP):
                r1 = np.sum(nruns_per_person[:nn])
                r2 = np.sum(nruns_per_person[:(nn+1)])
                tp = [] # initialize list
                tpoints = []
                for ee2 in range(r1,r2):
                    tp = list(range((ee2*tsize+et1),(ee2*tsize+et2)))
                    tpoints = tpoints + tp   # concatenate lists
                tplist1.append({'tp':tpoints})
            tplist.append(tplist1)


    tcdata_centered = copy.deepcopy(tcdata)
    for ttime in range(ntimepoints):
        for nn in range(NP):
            tp = tplist[ttime][nn]['tp']
            tcdata1 = tcdata[:, tp]
            for ee in range(nruns_per_person[nn]):
                tpe1 = ee * epoch
                tpe2 = (ee + 1) * epoch
                tcdata1_mean = np.mean(tcdata1[:, tpe1:tpe2], axis=1)
                # tcdata1_mean = np.repeat(tcdata1_mean[:,np.newaxis],epoch*nruns_per_person[nn],axis=1)
                tcdata1_mean = np.repeat(tcdata1_mean[:, np.newaxis], epoch, axis=1)
                tcdata1[:, tpe1:tpe2] = tcdata1[:, tpe1:tpe2] - tcdata1_mean  # need to set the mean for each epoch to zero
            tcdata_centered[:,tp] = tcdata1

    # check if a previous failed run is being resumed
    targetstart = 0
    componentstart = 0
    sem_one_target_results_inprogress = []
    if resume_run:
        filecheck = []
        checknamelist = []
        for networkcomponent in range(ntargets):
            target = network[networkcomponent]['target']
            sources = network[networkcomponent]['sources']
            nametag = target
            for name in sources: nametag = nametag + '_' + name
            outputname_inprog = os.path.join(savedirectory, 'SEMresults_' + savenamelabel + '_' + nametag + '_inprogress.npy')
            checknamelist.append(outputname_inprog)
            check = os.path.isfile(outputname_inprog)
            filecheck.append(check)
        if any(filecheck):
            ii = np.where(filecheck)[0]
            outputname_inprog = checknamelist[ii[-1]]
            results_inprogress = np.load(outputname_inprog, allow_pickle=True).flat[0]
            sem_one_target_results_inprogress = results_inprogress['sem_one_target_results']
            lasttargetcomplete  = results_inprogress['sem_one_target_results'][-1]['targetcluster']
            lastcomponent = results_inprogress['sem_one_target_results'][-1]['networkcomponent']
            targetnum = network[lastcomponent]['targetnum']
            ntargetclusters = nclusterlist[targetnum]
            targetstart = lasttargetcomplete+1
            componentstart = lastcomponent
            if targetstart >= ntargetclusters:
                targetstart = 0
                componentstart += 1

    # run all combinations of the SEM
    outputnamelist = []
    outputnamecount = 0
    for networkcomponent in range(componentstart,ntargets):
        target = network[networkcomponent]['target']
        sources = network[networkcomponent]['sources']
        targetnum = network[networkcomponent]['targetnum']
        sourcenums = network[networkcomponent]['sourcenums']
        ncombinations = np.prod(nclusterlist[sourcenums])
        nsources = len(sourcenums)

        nametag = target
        for name in sources: nametag = nametag + '_' + name
        infolabel = ' '
        for name in sources: infolabel = infolabel + ' ' + name
        outputname = os.path.join(savedirectory,'SEMresults_' + savenamelabel + '_' + nametag + '.npy')
        outputname_inprog = os.path.join(savedirectory,'SEMresults_' + savenamelabel + '_' + nametag + '_inprogress.npy')
        if outputnamecount == 0:
            outputnamelist = [outputname]
            outputnamelist_inprog = [outputname_inprog]
        else:
            outputnamelist.append(outputname)
            outputnamelist_inprog.append(outputname_inprog)
        outputnamecount += 1

        print('pysem_network: calculating for target region {} of {} regions ...{}'.format(networkcomponent + 1, ntargets, time.ctime()))
        print('      target {}, sources {}'.format(target,infolabel))

        sem_one_target_results = sem_one_target_results_inprogress
        if networkcomponent > componentstart:
            targetstart = 0
            sem_one_target_results = []

        for targetcluster in range(targetstart,nclusterlist[targetnum]):
            print('     pysem_network: target cluster {} of {} ...{}'.format(targetcluster + 1, nclusterlist[targetnum], time.ctime()))
            targetdataindex = dataindices_from_sourceclusters(nclusterlist, [targetnum],targetcluster).astype(int)

            bresults = np.zeros((ncombinations,ntimepoints,NP,nsources))
            R2results = np.zeros((ncombinations,ntimepoints,NP))

            for ttime in range(ntimepoints):
                for nn in range(NP):
                    tp = tplist[ttime][nn]['tp']
                    targetdata = tcdata_centered[targetdataindex,tp]
                    targetdata1 = targetdata # - np.mean(targetdata)

                    for index in range(ncombinations):
                        sourceclusters = ind2sub_ndims(nclusterlist[sourcenums], index)   # the cluster numbers to use for each source region
                        sourcedataindices = dataindices_from_sourceclusters(nclusterlist, sourcenums,index).astype(int)

                        sourcedata = tcdata_centered[sourcedataindices,:]
                        sourcedata = sourcedata[:,tp]

                        # Smean = np.mean(sourcedata, axis=1)
                        sourcedata1 = sourcedata # - np.repeat(Smean[:, np.newaxis], len(tp), axis=1)

                        Lweight = 1e-2
                        alpha = 1e-2
                        deltab = 0.05 * np.ones(nsources)
                        tol = 1e-1
                        maxiter = 250

                        b0 = np.zeros(nsources)
                        b, fit, R2, total_var, res_var, iter = \
                            gradient_descent(b0, targetdata1, sourcedata1, Lweight, alpha, deltab, tol, maxiter)

                        bresults[index,ttime,nn,:] = b
                        R2results[index,ttime,nn] = R2

            entry = {'b':bresults, 'R2':R2results, 'networkcomponent':networkcomponent, 'targetcluster':targetcluster}
            sem_one_target_results.append(entry)
            print('appending entry to sem_one_target_results ... length now ',len(sem_one_target_results))
            # inprogressdata = {'sem_one_target_results':sem_one_target_results, 'targetcluster':targetcluster, 'networkcomponent':networkcomponent}
            inprog_results = {'sem_one_target_results':sem_one_target_results}
            np.save(outputname_inprog, inprog_results)

            # save for each network component, with a different file name

        print('saving sem_one_target_results ... length now ',len(sem_one_target_results))
        results = {'sem_one_target_results': sem_one_target_results}
        np.save(outputname, results)
        # delete in progress results - not needed anymore
        os.remove(outputname_inprog)

    print('completed SEM network calculations: ',time.ctime())
    return outputnamelist

    # now a function is needed to look at group characteristics etc of the results...
