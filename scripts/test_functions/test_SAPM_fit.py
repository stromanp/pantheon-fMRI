# sys.path.append(r'C:\Users\Stroman\PycharmProjects\pantheon\venv')
import numpy as np
import matplotlib.pyplot as plt
import pysapm
import os
import copy
import time
import pandas as pd
import pysem

studynumber = 3
nperson = 8
cnums = [0,3,1,0,0,1,3,2,1,3]
betascale = 0.01

datadir = r'E:\SAPMresults2_Oct2022\SAPM_NGCinput_test'
nametag = '_0310013213'
resultsbase = ['RSnostim','Sens', 'Low', 'Pain','High', 'RSstim']
covnamebase = ['RSnostim','Sens', 'Low', 'Pain2','High', 'RSstim2']
nresults = len(resultsbase)
resultsnames = [resultsbase[x]+nametag+'_results.npy' for x in range(nresults)]
paramsnames = [resultsbase[x]+nametag+'_params.npy' for x in range(nresults)]
covnames = [covnamebase[x]+'_covariates.npy' for x in range(nresults)]


DBname = r'E:\graded_pain_database_May2022.xlsx'
xls = pd.ExcelFile(DBname, engine='openpyxl')
df1 = pd.read_excel(xls, 'paradigm1_BOLD')
del df1['Unnamed: 0']  # get rid of the unwanted header column
fields = list(df1.keys())
paradigm = df1['paradigms_BOLD']
timevals = df1['time']
paradigm_centered = paradigm - np.mean(paradigm)
fintrinsic_base = copy.deepcopy(paradigm_centered)


# run SAPM fit for one person
resultsname = os.path.join(datadir,resultsnames[studynumber])
results = np.load(resultsname,allow_pickle=True)

# initialize gradient-descent parameters--------------------------------------------------------------
initial_alpha = 1e-3
initial_Lweight = 1e0
initial_dval = 0.01
nitermax = 300
alpha_limit = 1.0e-6
repeat_limit = 2
repeat_count = 0

paramsnamefull = os.path.join(datadir,paramsnames[studynumber])
SAPMparams = np.load(paramsnamefull, allow_pickle=True).flat[0]
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
reciprocal_flag = SAPMparams['reciprocal_flag']

tcdata_centered = np.array(tcdata_centered)
ntime, NP = np.shape(tplist_full)
Nintrinsics = vintrinsic_count + fintrinsic_count

full_rnum_base = [np.sum(nclusterlist[:x]) for x in range(len(nclusterlist))]
clusterlist = (np.array(cnums) + full_rnum_base).astype(int)

con_names = []
ncon = len(betanamelist)
for nn in range(ncon):
    pair = beta_list[nn]['pair']
    if pair[0] >= nregions:
        sname = 'int{}'.format(pair[0]-nregions)
    else:
        sname = rnamelist[pair[0]]
    tname = rnamelist[pair[1]]
    name = '{}-{}'.format(sname[:4],tname[:4])
    con_names += [name]

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

# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# repeat the process for each participant-----------------------------------------------------------------
betalimit = 3.0
epochnum = 0
SAPMresults = []
first_pass_results = []
second_pass_results = []
beta_init_record = []


person_results = []
nrepeats = 4
for repeatnum in range(nrepeats):
    print('starting person {} at {}'.format(nperson, time.ctime()))
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
            Sint = Sinput[fintrinsic_region, :]
            Sint = Sint - np.mean(Sint)
            # need to add constant to fit values
            G = np.concatenate((fintrinsic1[np.newaxis, :], np.ones((1, tsize_total))), axis=0)
            b, fit, R2, total_var, res_var = pysem.general_glm(Sint, G)
            beta_int1 = np.abs(b[0])
        else:
            beta_int1 = 0.0
    else:
        beta_int1 = 0.0
        fintrinsic1 = []

    lastgood_beta_int1 = copy.deepcopy(beta_int1)

    # initialize beta values-----------------------------------
    if isinstance(betascale, str):
        if betascale == 'shotgun':
            nbeta = len(csource)
            beta_initial = pysapm.betaval_init_shotgun(initial_Lweight, csource, ctarget, Sinput, Minput, Mconn,
                                                fintrinsic_count,
                                                vintrinsic_count, beta_int1, fintrinsic1, nreps=10000)
        else:
            # read saved beta_initial values
            b = np.load(betascale, allow_pickle=True).flat[0]
            beta_initial = b['beta_initial']
    else:
        beta_initial = betascale * np.random.randn(len(csource))

    beta_init_record.append({'beta_initial': beta_initial})

    # initalize Sconn
    betavals = copy.deepcopy(beta_initial)  # initialize beta values at zero
    lastgood_betavals = copy.deepcopy(betavals)

    results_record = []
    ssqd_record = []

    alpha = copy.deepcopy(initial_alpha)
    Lweight = copy.deepcopy(initial_Lweight)
    dval = copy.deepcopy(initial_dval)

    Mconn[ctarget, csource] = copy.deepcopy(betavals)

    # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
    # fit, Sconn_full = network_eigenvalue_method(Sconn_full, Minput, Mconn, ncon)

    fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count,
                                                             beta_int1, fintrinsic1)
    cost = np.mean(np.abs(betavals)) + np.mean(np.abs(Mintrinsic))  # L1 regularization
    ssqd = err + Lweight * cost
    ssqd_starting = copy.deepcopy(ssqd)
    ssqd_record += [ssqd]
    ssqd_starting = 1e20  # start big

    iter = 0
    # vintrinsics_record = []
    converging = True
    dssq_record = np.ones(3)
    dssq_count = 0
    sequence_count = 0
    repeat_count = 0
    while alpha > alpha_limit and repeat_count < 1 and iter < nitermax and converging:
        iter += 1
        # gradients in betavals and beta_int1
        Mconn[ctarget, csource] = copy.deepcopy(betavals)
        fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                                 vintrinsic_count, beta_int1, fintrinsic1)
        dssq_db, ssqd, dssq_dbeta1 = pysapm.gradients_for_betavals(Sinput, Minput, Mconn, betavals, ctarget, csource, dval,
                                                            fintrinsic_count, vintrinsic_count, beta_int1,
                                                            fintrinsic1, Lweight)
        ssqd_record += [ssqd]

        # fix some beta values at zero, if specified
        fixed_beta_vals = []
        if len(fixed_beta_vals) > 0:
            dssq_db[fixed_beta_vals] = 0

        # apply the changes
        # limit the betaval changes
        dsmax = 0.1 / alpha
        dssq_db[dssq_db < -dsmax] = -dsmax
        dssq_db[dssq_db > dsmax] = dsmax

        betavals -= alpha * dssq_db
        beta_int1 -= alpha * dssq_dbeta1
        beta_int1 = np.abs(beta_int1)  # limit beta_int1

        Mconn[ctarget, csource] = copy.deepcopy(betavals)
        fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                                 vintrinsic_count, beta_int1, fintrinsic1)

        cost = np.mean(np.abs(betavals)) + np.mean(np.abs(Mintrinsic))  # L1 regularization
        ssqd_new = err + Lweight * cost

        err_total = Sinput - fit
        Smean = np.mean(Sinput)
        errmean = np.mean(err_total)
        R2total = 1 - np.sum((err_total - errmean) ** 2) / np.sum((Sinput - Smean) ** 2)

        # Sinput_sim, Soutput_sim = network_sim(Sinput_full, Soutput_full, Minput, Moutput)
        results_record.append({'Sinput': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})

        if ssqd_new > ssqd:
            alpha *= 0.5
            # revert back to last good values
            betavals = copy.deepcopy(lastgood_betavals)
            beta_int1 = copy.deepcopy(lastgood_beta_int1)
            dssqd = ssqd - ssqd_new
            dssq_record = np.ones(3)  # reset the count
            dssq_count = 0
            sequence_count = 0

            print(
                '{} beta vals:  iter {} alpha {:.3e}  repeat_count {}  delta ssq > 0  - no update'.format(nperson, iter,alpha, repeatnum))
        else:
            # save the good values
            lastgood_betavals = copy.deepcopy(betavals)
            lastgood_beta_int1 = copy.deepcopy(beta_int1)

            dssqd = ssqd - ssqd_new
            ssqd = copy.deepcopy(ssqd_new)

            dssq_count += 1
            dssq_count = np.mod(dssq_count, 3)
            # dssq_record[dssq_count] = 100.0 * dssqd / ssqd_starting
            dssq_record[dssq_count] = dssqd
            if np.max(dssq_record) < 0.1:  converging = False

        print('{} beta vals:  iter {} alpha {:.3e}  repeat_count {}  delta ssq {:.4f}  relative: {:.1f} percent  '
              'R2 {:.3f}'.format(nperson, iter, alpha, repeatnum, -dssqd, 100.0 * ssqd / ssqd_starting, R2total))
        # now repeat it ...

    # fit the results now to determine output signaling from each region
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count,
                                                             beta_int1, fintrinsic1)
    Sconn = Meigv @ Mintrinsic  # signalling over each connection

    entry = {'Sinput': Sinput, 'Sconn': Sconn, 'beta_int1': beta_int1, 'Mconn': Mconn, 'Minput': Minput,
             'R2total': R2total, 'Mintrinsic': Mintrinsic, 'fintrinsic_count': fintrinsic_count,
             'vintrinsic_count': vintrinsic_count,
             'Meigv': Meigv, 'betavals': betavals, 'fintrinsic1': fintrinsic1, 'clusterlist': clusterlist,
             'fintrinsic_base': fintrinsic_base}

    person_results.append(entry)


# show results
windownum = 55
plt.close(windownum)
fig = plt.figure(windownum)
nregions,tsizefull = np.shape(fit)
for nn in range(nregions):
    ax = fig.add_subplot(3,4,nn+1)
    plt.plot(range(tsizefull),Sinput[nn,:],'-xr')
    plt.plot(range(tsizefull),fit[nn,:],'-g')


# original results---------------------
r = results[nperson][0]
Sinput0 = r['Sinput']
Minput0 = r['Minput']
Mconn0 = r['Mconn']
beta_int10 = r['beta_int1']
fintrinsic10 = r['fintrinsic1']
betavals0 = r['betavals']
fit0, Mintrinsic0, Meigv0, err0 = pysapm.network_eigenvector_method(Sinput0, Minput0, Mconn0,
                                        fintrinsic_count, vintrinsic_count, beta_int10, fintrinsic10)

plt.close(windownum+1)
fig = plt.figure(windownum+1)
for nn in range(nregions):
    ax = fig.add_subplot(3,4,nn+1)
    nreps = int(tsizefull/tsize)
    tc = np.mean(np.reshape(Sinput0[nn,:],(tsize,nreps)),axis=1)
    plt.plot(range(tsize),tc,'-xr')
    tc = np.mean(np.reshape(fit0[nn,:],(tsize,nreps)),axis=1)
    plt.plot(range(tsize),tc,'-g')

plt.close(windownum+2)
fig = plt.figure(windownum+2)
plt.plot(range(len(betavals0)),betavals0,'og')
for repeatnum in range(nrepeats):
    plt.plot(range(len(betavals)),person_results[repeatnum]['betavals'],'o')



# are the results equally good with the latents flipped?
flipcheckresults = []
ncombos = 2 ** vintrinsic_count
search_size = 2 * np.ones(vintrinsic_count)
scalefactors = np.zeros((ncombos, vintrinsic_count))
for nn in range(ncombos):
    scalefactor = 1.0 - 2.0 * pysapm.ind2sub_ndims(search_size, nn)
    scalefactors[nn, :] = scalefactor

betavals0 = copy.deepcopy(betavals)
Mconn[ctarget, csource] = copy.deepcopy(betavals0)
fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                                vintrinsic_count, beta_int1, fintrinsic1)
flipcheckresults.append({'fit':fit,'Mintrinsic':Mintrinsic, 'Meigv':Meigv, 'err':err})

for nn in range(ncombos):
    betavalsworking = copy.deepcopy(betavals0)
    for vv in range(vintrinsic_count):
        c = np.where(latent_flag == (vv+fintrinsic_count+1))[0]
        betavalsworking[c] *= scalefactors[nn,vv]

    Mconn[ctarget, csource] = copy.deepcopy(betavalsworking)
    fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count,
                                                             beta_int1, fintrinsic1)

    flipcheckresults.append({'fit': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv, 'err': err})


plt.close(windownum+3)
fig = plt.figure(windownum+3)
for nn in range(nregions):
    ax = fig.add_subplot(3,4,nn+1)
    nreps = int(tsizefull/tsize)
    tc = np.mean(np.reshape(flipcheckresults[1]['fit'][nn,:],(tsize,nreps)),axis=1)
    plt.plot(range(tsize),tc,'-xr')
    tc = np.mean(np.reshape(flipcheckresults[2]['fit'][nn,:],(tsize,nreps)),axis=1)
    plt.plot(range(tsize),tc,'-g')
    tc = np.mean(np.reshape(flipcheckresults[3]['fit'][nn,:],(tsize,nreps)),axis=1)
    plt.plot(range(tsize),tc,'-b')
