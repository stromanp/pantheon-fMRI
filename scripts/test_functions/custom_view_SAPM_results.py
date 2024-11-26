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

resultsdir = r'E:\SAPMresults2_Oct2022'
nametags = ['Sens', 'Low', 'RSnostim', 'Pain', 'High', 'RSstim']   # , 'Allpain'
refnames = ['nulldata_10000']
covtags = ['Sens', 'Low', 'RSnostim', 'Pain2', 'High', 'RSstim2']   # , 'allpain'
namesuffix = '_0310013210_all'

SAPMparamsnames = [os.path.join(resultsdir,nt+namesuffix+'_params.npy') for nt in nametags]
SAPMresultsnames = [os.path.join(resultsdir,nt+namesuffix+'_results.npy') for nt in nametags]
covnames = [os.path.join(resultsdir,nt+'_covariates.npy') for nt in covtags]
covtag = 'painrating'

ref_paramsnames = [os.path.join(resultsdir,nt+'_params.npy') for nt in refnames]
ref_resultsnames = [os.path.join(resultsdir,nt+'_results.npy') for nt in refnames]

ng = 0
params = np.load(SAPMparamsnames[ng], allow_pickle=True).flat[0]
params_keys = params.keys()
results = np.load(SAPMresultsnames[ng], allow_pickle=True)
results_keys = results[0].keys()
NP = len(results)
covariates = np.load(covnames[ng], allow_pickle=True).flat[0]


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
connection_name_list = ['Hypo-LC-C6RD','NRM-LC-C6RD','PAG-PBN-NTS', 'Hypo-PAG-PBN', 'Thal-PAG-NTS', 'Thal-PAG-NGC', 'PAG-NRM-LC']

# connection_name_list = ['LC-C6RD-DRt', 'NRM-LC-Thal']
for count,connection_name in enumerate(connection_name_list):
    x = labeltext_record.index(connection_name)
    n1,n2 = Mconn_index_record[x]['i']

    Mdata = []
    for ng in range(len(nametags)):
        params = np.load(SAPMparamsnames[ng], allow_pickle=True).flat[0]
        results = np.load(SAPMresultsnames[ng], allow_pickle=True)
        NP = len(results)
        try:
            covariates = np.load(covnames[ng], allow_pickle=True).flat[0]
            cx = covariates['GRPcharacteristicslist'].index(covtag)
            covvalues = covariates['GRPcharacteristicsvalues'][cx,:].astype(float)
        except:
            covvalues = np.zeros(NP)

        M = [results[n]['Mconn'][n1,n2] for n in range(NP)]
        entry = {'name':nametags[ng], 'M':M, 'cov':covvalues}
        Mdata.append(entry)

    for ng in range(len(refnames)):
        params = np.load(ref_paramsnames[ng], allow_pickle=True).flat[0]
        results = np.load(ref_resultsnames[ng], allow_pickle=True)
        NP = len(results)
        covvalues = np.zeros(NP)

        M = [results[n]['Mconn'][n1,n2] for n in range(NP)]
        entry = {'name':nametags[ng], 'M':M, 'cov':covvalues}
        Mdata.append(entry)

    # plot the data ...
    plotdata = [Mdata[n]['M'] for n in range(len(nametags)+len(refnames))]
    covdata = [Mdata[n]['cov'] for n in range(len(nametags)+len(refnames))]

    windownum = 10+count
    plt.close(windownum)
    fig = plt.figure(windownum)
    # ax = fig.add_axes([0,0,1,1])
    # ax.suptitle(connection_name)
    # plt.boxplot(plotdata, labels = nametags, notch = True, showfliers = False, showmeans = True)
    plt.violinplot(plotdata, showmeans = True, showmedians = False, showextrema = False)
    plt.xticks(list(range(1,len(nametags)+len(refnames)+1)), nametags+refnames)
    plt.xlabel("Group", size=14)
    plt.ylabel("B values", size=14)
    plt.title(connection_name, size=16)
    # plt.savefig("Violinplot_testplot.png", format='png', dpi=150)

    windownum = 100+count
    plt.close(windownum)
    fig = plt.figure(windownum)
    cols = np.array([[1,0,0],[1,0.5,0],[0.5,0.5,0],[0.5,1,0],[0,1,0],[0,1,0.5],[0,0.5,0.5],[0,0,1],[0.5,0,1],[1,0,1]])

    cols = np.array([[1,0,0],[0,0.5,0.5],[0.5,0,0],[0,1,0],[0,1,1],[0,0,1], [0,0,0]])
    covlimits = [0,30,0,30,30,30,30]   # lower limits for reliable covariates

    print('\n{}'.format(connection_name))
    for ng in range(len(nametags)):
        plt.plot(covdata[ng],plotdata[ng],marker = 'o', linestyle = 'none', color = cols[ng,:])
        plt.xlabel("Group", size=14)
        plt.ylabel("B values", size=14)
        plt.title(connection_name, size=16)

        x = covdata[ng][:,np.newaxis]
        y = np.array(plotdata[ng])[:,np.newaxis]
        if covlimits[ng] > 0:
            c = np.where(x[:,0] > covlimits[ng])
        else:
            c = range(len(covdata[ng]))
        reg = LinearRegression().fit(x[c],y[c])
        D = reg.predict(x)
        # plt.plot(x,D, color = cols[ng,:])
        R2 = reg.score(x[c],y[c])
        m = reg.coef_
        b = reg.intercept_
        print('{} slope = {:.2e}  int = {:.2e}  R2 = {:.2f}'.format(nametags[ng],m.flat[0],b.flat[0],R2))

        huber = HuberRegressor().fit(x[c],y[c].flat)
        huberD = huber.predict(x)
        huberR2 = huber.score(x[c],y[c].flat)
        huberm = huber.coef_
        huberb = huber.intercept_
        print('{} huber slope = {:.2e}  int = {:.2e}  R2 = {:.2f}'.format(nametags[ng],huberm.flat[0],huberb.flat[0],huberR2))


        # # alternate fitting method
        # x = np.array(covdata[ng])
        # if covlimits[ng] > 0:
        #     c = np.where(x > covlimits[ng])
        # else:
        #     c = range(len(covdata[ng]))
        # x = x[c]
        # x -= np.mean(x)
        # y = np.array(plotdata[ng])[c][np.newaxis,:]
        # try:
        #     G = np.concatenate((np.ones((1, len(x))),x[np.newaxis,:]), axis=0) # put the intercept term first
        #     b, fit, R2, total_var, res_var = pysem.general_glm(y, G)
        #     print('{} slope = {:.2e}  int = {:.2e}  R2 = {:.2f}'.format(nametags[ng],b[0,1],b[0,0],R2))
        #     plt.plot(x[np.newaxis,:], fit, color=cols[ng, :])
        # except:
        #     print('{}  cannot fit data'.format(nametags[ng]))

    for nn in range(len(nametags)):
        m = np.mean(covdata[nn])
        s = np.std(covdata[nn])
        print('{}  pain ratings:  {:.2f} {} {:.2f}'.format(nametags[nn],m,chr(177),s))
