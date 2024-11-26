
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

SAPMresults = []
first_pass_results = []
second_pass_results = []
nperson = 3
# repeat for each person

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

search_record = []

betascale = 0.1
beta_initial = betascale * np.random.randn(len(csource))  # initialize beta values at zero
# apply gradient descent method


# copy from pysapm.py --------------------------------------
initial_alpha = 1e-3
initial_Lweight = 1e0
initial_dval = 0.01
nitermax = 200
alpha_limit = 1.0e-6
repeat_limit = 2
repeat_count = 0

# initalize Sconn
betavals = copy.deepcopy(beta_initial) # initialize beta values at zero
lastgood_betavals = copy.deepcopy(betavals)

results_record = []
ssqd_record = []

alpha = initial_alpha
Lweight = initial_Lweight
dval = initial_dval

Mconn[ctarget,csource] = copy.deepcopy(betavals)

fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
# cost = np.sum(np.abs(betavals**2)) # L2 regularization
# cost = np.sum(np.abs(betavals))  # L1 regularization, original
cost = np.mean(np.abs(betavals)) + np.mean(np.abs(Mintrinsic))  # L1 regularization
# cost = np.mean(np.abs(betavals))  # L1 regularization
ssqd = err + Lweight * cost
ssqd_starting = ssqd
ssqd_record += [ssqd]


iter = 0
# vintrinsics_record = []
converging = True
dssq_record = np.ones(3)
dssq_count = 0
sequence_count = 0
repeat_count = 0
while alpha > alpha_limit and repeat_count < repeat_limit and iter < nitermax:
    iter += 1
    # gradients in betavals and beta_int1
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                             vintrinsic_count, beta_int1, fintrinsic1)
    dssq_db, ssqd, dssq_dbeta1 = pysapm.gradients_for_betavals(Sinput, Minput, Mconn, betavals, ctarget, csource, dval,
                                                        fintrinsic_count, vintrinsic_count, beta_int1,
                                                        fintrinsic1, Lweight)
    ssqd_record += [ssqd]

    # apply the changes
    dsmax = 0.1/alpha
    dssq_db[dssq_db < -dsmax] = -dsmax
    dssq_db[dssq_db > dsmax] = dsmax

    betavals -= alpha * dssq_db
    beta_int1 -= alpha * dssq_dbeta1

    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                             vintrinsic_count, beta_int1, fintrinsic1)
    # cost = np.sum(np.abs(betavals**2))  # L2 regularization
    # cost = np.sum(np.abs(betavals))  # L1 regularization, original
    cost = np.mean(np.abs(betavals)) + np.mean(np.abs(Mintrinsic))  # L1 regularization
    # cost = np.mean(np.abs(betavals))  # L1 regularization
    ssqd_new = err + Lweight * cost

    err_total = Sinput - fit
    Smean = np.mean(Sinput)
    errmean = np.mean(err_total)
    R2total = 1 - np.sum((err_total - errmean) ** 2) / np.sum((Sinput - Smean) ** 2)

    # Sinput_sim, Soutput_sim = network_sim(Sinput_full, Soutput_full, Minput, Moutput)
    interim_results = {'Sinput': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv, 'betavals':betavals, 'R2':R2total, 'dssq_db':dssq_db, 'beta_int1':beta_int1}
    results_record.append(copy.deepcopy(interim_results))

    if ssqd_new >= ssqd:
        alpha *= 0.5
        # revert back to last good values
        betavals = copy.deepcopy(lastgood_betavals)
        beta_int1 = copy.deepcopy(lastgood_beta_int1)
        dssqd = ssqd - ssqd_new
        dssq_record = np.ones(3)  # reset the count
        dssq_count = 0
        sequence_count = 0

        if alpha < alpha_limit and repeat_count < repeat_limit:
            repeat_count += 1
            alpha = initial_alpha

        print('beta vals:  iter {} alpha {:.3e}   repeat_count {}  delta ssq > 0  - no update'.format(iter, alpha, repeat_count))
    else:
        # save the good values
        lastgood_betavals = copy.deepcopy(betavals)
        lastgood_beta_int1 = copy.deepcopy(beta_int1)

        dssqd = ssqd - ssqd_new
        ssqd = ssqd_new

        # sequence_count += 1
        # if sequence_count > 5:
        #     alpha *= 1.3
        #     sequence_count = 0

        dssq_count += 1
        dssq_count = np.mod(dssq_count, 3)
        # dssq_record[dssq_count] = 100.0 * dssqd / ssqd_starting
        dssq_record[dssq_count] = dssqd
        if np.max(dssq_record) < 1e-6:  converging = False

    print('beta vals:  iter {} alpha {:.3e}  repeat_count {}  delta ssq {:.4f}  relative: {:.1f} percent  '
          'R2 {:.3f}'.format(iter, alpha, repeat_count, -dssqd, 100.0 * ssqd / ssqd_starting, R2total))
    # now repeat it ...

    if (alpha < alpha_limit or iter >= nitermax) and repeat_count == 0:
        # fit the results now to determine output signaling from each region
        Mconn[ctarget, csource] = copy.deepcopy(betavals)
        fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                                 vintrinsic_count, beta_int1, fintrinsic1)
        Sconn = Meigv @ Mintrinsic  # signalling over each connection

        first_pass = {'Sinput': Sinput, 'Sconn': Sconn, 'beta_int1': beta_int1, 'Mconn': Mconn,
                      'Minput': Minput,
                      'R2total': R2total, 'Mintrinsic': Mintrinsic,
                      'Meigv': Meigv, 'betavals': betavals, 'fintrinsic1': fintrinsic1,
                      'clusterlist': clusterlist,
                      'fintrinsic_base': fintrinsic_base}

        betavals *= -1.0
        lastgood_betavals = copy.deepcopy(betavals)
        iter = 0
        alpha = copy.deepcopy(initial_alpha)
        Lweight = copy.deepcopy(initial_Lweight)
        dval = copy.deepcopy(initial_dval)
        repeat_count = 1
        print('flipping the betavals ...')


# fit the results now to determine output signaling from each region
Mconn[ctarget, csource] = copy.deepcopy(betavals)
fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
Sconn = Meigv @ Mintrinsic    # signalling over each connection

entry = {'Sinput':Sinput, 'Sconn':Sconn, 'beta_int1':beta_int1, 'Mconn':Mconn, 'Minput':Minput,
         'R2total':R2total, 'Mintrinsic':Mintrinsic,
         'Meigv':Meigv, 'betavals':betavals, 'fintrinsic1':fintrinsic1, 'clusterlist':clusterlist,
         'fintrinsic_base':fintrinsic_base}

first_pass_results.append(copy.deepcopy(first_pass))
second_pass_results.append(copy.deepcopy(entry))

if first_pass['R2total'] >= entry['R2total']:
    entry = copy.deepcopy(first_pass)
    print('best fit was first pass...')
else:
    print('\n\nbest fit was second pass...\n\n')

SAPMresults.append(copy.deepcopy(entry))



# plot some results
nrep = len(results_record)
betavals_all = np.array([results_record[x]['betavals'] for x in range(nrep)])
R2 = np.array([results_record[x]['R2'] for x in range(nrep)])
dssq_db = np.array([results_record[x]['dssq_db'] for x in range(nrep)])
bint1 = np.array([results_record[x]['beta_int1'] for x in range(nrep)])
var_betavals_all = np.var(betavals_all,axis = 0)
x = np.argmax(var_betavals_all)

connections = list(range(10))
w1 = 12
plt.close(w1)
fig = plt.figure(w1)
for n in connections:
    b = betavals_all[:,n]
    plt.plot(range(nrep),b)

plt.close(w1+1)
fig = plt.figure(w1+1)
plt.plot(range(nrep),R2)

    #
    #
    #
    #     #--------------------old part--------------------------------------
    #     nreps = 10000
    #     for rr in range(nreps):
    #         # initialize beta values at random values-----------------------------------
    #         betavals = betascale*np.random.randn(len(csource)) # initialize beta values at zero
    #         Mconn[ctarget,csource] = betavals
    #         fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
    #         cost = np.mean(np.abs(betavals))  # L1 regularization
    #         ssqd = err + Lweight * cost
    #         R2 = 1 - np.sum((Sinput-fit)**2)/np.sum(Sinput**2)
    #         search_record.append({'betavals':betavals, 'ssqd':ssqd, 'R2':R2})
    #
    #     ssqd_list = np.array([search_record[x]['ssqd'] for x in range(nreps)])
    #     R2_list = np.array([search_record[x]['R2'] for x in range(nreps)])
    #     b_list = np.array([search_record[x]['betavals'] for x in range(nreps)])
    #     windownum1 = 10
    #     plt.close(windownum1)
    #     fig = plt.figure(windownum1)
    #     plt.hist(ssqd_list, 500)
    #
    #     # check values
    #     x = ssqd_list.argmin()
    #     best_betavals = b_list[x,:]
    #     best_betaval_record.append({'best_betavals':best_betavals})
    #     print('{} best betavals gives ssqd = {:.2f}  R2 = {:.2f}'.format(nperson, ssqd_list[x], R2_list[x]))
    #
    #     # compare with other values
    #     init_file1 = os.path.join(SAPMresultsdir, namelist[gn]+'_init1.npy')
    #     init_file2 = os.path.join(SAPMresultsdir, namelist[gn]+'_init2.npy')
    #
    #     b = np.load(init_file1, allow_pickle=True).flat[0]
    #     binit1 = b['beta_initial']
    #     b = np.load(init_file2, allow_pickle=True).flat[0]
    #     binit2 = b['beta_initial']
    #
    #     dist1 = [np.sqrt(np.sum( (binit1 - b_list[x,:])**20)) for x in range(nreps)]
    #     x = np.argmin(dist1)
    #     print('{} closest match to binit1 gives ssqd = {:.2f}'.format(nperson, ssqd_list[x]))
    #
    #     dist2 = [np.sqrt(np.sum( (binit2 - b_list[x,:])**20)) for x in range(nreps)]
    #     x = np.argmin(dist2)
    #     print('{} closest match to binit2 gives ssqd = {:.2f}'.format(nperson, ssqd_list[x]))
    #
    #
    # best_betavals = np.array([best_betaval_record[x]['best_betavals'] for x in range(NP)])
    # bestb_avg = np.mean(best_betavals,axis=0)
    # bestb_std = np.std(best_betavals,axis=0)
    #
    # plt.close(windownum)
    # fig = plt.figure(windownum)
    # fig.subplots_adjust(bottom=0.3)
    # ax = fig.add_subplot()
    # ind = np.arange(len(bestb_avg))
    # plt.errorbar(connection_names,bestb_avg, bestb_std,linestyle='none',marker = 'o')
    # plt.pause(0.1)
    # locs, labels = plt.xticks()
    # plt.xticks(locs,labels,rotation=90,fontsize=8)
    #
    # repeat_betaval_record.append({'best_betavals':best_betavals})