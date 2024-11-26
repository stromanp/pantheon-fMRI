# view and compare SAPM results from multiple study groups
# custom program for specific results - not for sharing

# # results
# entry = {'Sinput': Sinput, 'Sconn': Sconn, 'beta_int1': beta_int1, 'Mconn': Mconn, 'Minput': Minput,
#          'R2total': R2total, 'Mintrinsic': Mintrinsic,
#          'Meigv': Meigv, 'betavals': betavals, 'fintrinsic1': fintrinsic1, 'clusterlist': clusterlist,
#          'fintrinsic_base': fintrinsic_base}

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import HuberRegressor, LinearRegression
import pysem

resultsdir = r'E:\SAPMresults2_Oct2022\'
nametags = ['Sens', 'Low', 'RSnostim', 'Pain', 'High', 'RSstim']   # , 'Allpain'
refnames = ['nulldata_10000']
covtags = ['Sens', 'Low', 'RSnostim', 'Pain2', 'High', 'RSstim2']   # , 'allpain'
namesuffix = '_0310013210_all'

SAPMparamsnames = r'E:\FM2021data\FMstim_individualguided_test_params.npy'
SAPMresultsnames = r'E:\FM2021data\FMstim_individualguided_test_results_corr.npy'
covnames = r'E:\FM2021data\covariates.npy'
covtag = 'CPM'

params = np.load(SAPMparamsnames, allow_pickle=True).flat[0]
params_keys = params.keys()
results = np.load(SAPMresultsname, allow_pickle=True)
results_keys = results[0].keys()
NP = len(results)
covariates = np.load(covnames, allow_pickle=True).flat[0]


# create labels for each connection
rnamelist = params['rnamelist']
Mconn = results[0]['Mconn']
Minput = results[0]['Minput']
betanamelist = params['betanamelist']
beta_list = params['beta_list']

nregions = len(rnamelist)
nr1, nr2 = np.shape(Mconn)
labeltext_record = []
Mconn_index_record = []
for n1 in range(nr1):
    tname = betanamelist[n1]
    tpair = beta_list[n1]['pair']
    if tpair[0] >= nregions:
        ts = 'int{}'.format(tpair[0] - nregions)
    else:
        ts = rnamelist[tpair[0]]
        if len(ts) > 4:  ts = ts[:4]
    tt = rnamelist[tpair[1]]
    if len(tt) > 4:  tt = tt[:4]

    for n2 in range(nr2):
        if np.abs(Mconn[n1,n2]) > 0:
            sname = betanamelist[n2]
            spair = beta_list[n2]['pair']
            if spair[0] >= nregions:
                ss = 'int{}'.format(spair[0] - nregions)
            else:
                ss = rnamelist[spair[0]]
                if len(ss) > 4:  ss = ss[:4]
            st = rnamelist[spair[1]]
            if len(st) > 4:  st = st[:4]

            labeltext = '{}-{}-{}'.format(ss, st, tt)
            labeltext_record += [labeltext]
            Mconn_index_record.append({'i':[n1,n2]})


# ------pick a connection to plot-----------------
connection_name_list = ['PAG-NRM']

for count,connection_name in enumerate(connection_name_list):
    x = labeltext_record.index(connection_name)
    n1,n2 = Mconn_index_record[x]['i']

    Mdata = []

    params = np.load(SAPMparamsnames, allow_pickle=True).flat[0]
    results = np.load(SAPMresultsnames, allow_pickle=True)
    NP = len(results)
    try:
        covariates = np.load(covnames, allow_pickle=True).flat[0]
        cx = covariates['GRPcharacteristicslist'].index(covtag)
        covvalues = covariates['GRPcharacteristicsvalues'][cx,:].astype(float)
    except:
        covvalues = np.zeros(NP)

    M = [results[n]['Mconn'][n1,n2] for n in range(NP)]
    entry = {'name':nametags, 'M':M, 'cov':covvalues}
    Mdata.append(entry)

    # plot the data ...
    plotdata = Mdata['M']
    covdata = Mdata['cov']

    windownum = 10+count
    plt.close(windownum)
    fig = plt.figure(windownum)
    plt.plot(covdata,plotdata,'og')