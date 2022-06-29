# sys.path.append(r'C:\Users\Stroman\PycharmProjects\pantheon\venv')

import numpy as np
import matplotlib.pyplot as plt
import py2ndlevelanalysis
import copy
import pyclustering
import pydisplay
import time
import pysem
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
import scipy.stats as stats
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
from mpl_toolkits import mplot3d
import random
import draw_sapm_diagram2 as dsd2
import copy

from hmmlearn import hmm



# main program
def analyze_sapm_latents(cord_cluster, type):

    # studyname:   allthreat, RS1nostim, Low, Sens, all_condition
    # use these settings for testing consistency etc...
    type = 'fixed'
    cord_cluster = 4
    studyname = 'all_condition'

    # setup inputs for this particular set of studies
    if type == 'fixed':
        if cord_cluster == 0:
            cnums = [0, 3, 3, 1, 4, 1, 3, 3, 4, 1]  # fixed 0
        if cord_cluster == 1:
            cnums = [1, 3, 3, 1, 3, 1, 3, 3, 2, 1]  # fixed 1
        if cord_cluster == 2:
            cnums = [2, 3, 3, 1, 1, 1, 3, 3, 2, 0]  # fixed 2
        if cord_cluster == 3:
            cnums = [3, 3, 2, 1, 0, 1, 2, 3, 4, 1]  # fixed 3
        if cord_cluster == 4:
            cnums = [4, 3, 3, 1, 0, 1, 2, 3, 4, 3]  # fixed 4
    else:
        if cord_cluster == 0:
            cnums = [0, 4, 4, 2, 2, 3, 3, 2, 3, 1]  # random 0
        if cord_cluster == 1:
            cnums = [1, 4, 2, 3, 2, 1, 1, 2, 3, 0]  # random 1
        if cord_cluster == 2:
            cnums = [2, 2, 2, 0, 0, 2, 0, 3, 1, 3]  # random 2
        if cord_cluster == 3:
            cnums = [3, 3, 1, 4, 4, 1, 3, 3, 1, 0]  # random 3
        if cord_cluster == 4:
            cnums = [4, 4, 2, 1, 0, 3, 3, 3, 2, 0]  # random 4

    # from gradient descent search for best clusters...
    # based on pain data
    # best cluster set is: [3. 4. 4. 0. 4. 2. 4. 0. 4. 4.]

    basedir = r'E:\beta_distribution'
    outputdir = r'E:\beta_distribution\{}_{}_C6RD{}'.format(studyname,type,cord_cluster)
    if not os.path.exists(outputdir): os.mkdir(outputdir)

    if studyname == 'all_condition':
        covariatesfile = r'E:\all_condition_covariates.npy'
        regiondataname = r'E:\all_condition_region_data.npy'
        clusterdataname = r'E:\threat_safety_clusterdata.npy'

    if studyname == 'allthreat':
        covariatesfile = r'E:\allthreat_covariates.npy'
        regiondataname = r'E:\threat_safety_regiondata_allthreat55.npy'
        clusterdataname = r'E:\threat_safety_clusterdata.npy'

    if studyname == 'RS1nostim':
        covariatesfile = r'E:\RS1nostim_covariates.npy'
        regiondataname = r'E:\RS1nostim_region_data.npy'
        clusterdataname = r'E:\threat_safety_clusterdata.npy'

    if studyname == 'Low':
        covariatesfile = r'E:\Low_covariates.npy'
        regiondataname = r'E:\Low_region_data.npy'
        clusterdataname = r'E:\threat_safety_clusterdata.npy'

    if studyname == 'Sens':
        covariatesfile = r'E:\Sens_covariates.npy'
        regiondataname = r'E:\Sens_region_data.npy'
        clusterdataname = r'E:\threat_safety_clusterdata.npy'

    SEMresultsname = os.path.join(outputdir, 'SEMphysio_model.npy')
    SEMparametersname = os.path.join(outputdir, 'SEMparameters_model5.npy')
    networkfile = r'E:\network_model_5cluster_v5_w_3intrinsics.xlsx'

    cov = np.load(covariatesfile, allow_pickle=True).flat[0]
    GRPcharacteristicslist = cov['GRPcharacteristicslist']
    GRPcharacteristicsvalues = cov['GRPcharacteristicsvalues']


    # load paradigm data--------------------------------------------------------------------
    DBname = r'E:\graded_pain_database_May2022.xlsx'
    xls = pd.ExcelFile(DBname, engine='openpyxl')
    df1 = pd.read_excel(xls, 'paradigm1_BOLD')
    del df1['Unnamed: 0']  # get rid of the unwanted header column
    fields = list(df1.keys())
    paradigm = df1['paradigms_BOLD']
    timevals = df1['time']
    paradigm_centered = paradigm - np.mean(paradigm)
    dparadigm = np.zeros(len(paradigm))
    dparadigm[1:] = np.diff(paradigm_centered)


    # rnamelist = ['C6RD',  'DRt', 'Hypothalamus','LC', 'NGC',
    #                'NRM', 'NTS', 'PAG', 'PBN', 'Thalamus']
    full_rnum_base =  np.array([0,5,10,15,20,25,30,35,40,45])

    namelist = ['C6RD',  'DRt', 'Hypothalamus','LC', 'NGC', 'NRM', 'NTS', 'PAG', 'PBN', 'Thalamus',
            'Rtotal', 'R C6RD',  'R DRt', 'R Hyp','R LC', 'R NGC', 'R NRM', 'R NTS', 'R PAG',
            'R PBN', 'R Thal']

    #----------------------------------------------------------------
    # load the results now
    #----------------------------------------------------------------
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
    timepoint = SEMparams['timepoint']
    epoch = SEMparams['epoch']

    SEMresults = np.load(SEMresultsname, allow_pickle=True)

    NP = len(SEMresults)
    for nn in range(NP):
        Sinput = SEMresults[nn]['Sinput']
        Sconn = SEMresults[nn]['Sconn']
        beta_int1 = SEMresults[nn]['beta_int1']
        Mconn = SEMresults[nn]['Mconn']
        Minput = SEMresults[nn]['Minput']
        R2total = SEMresults[nn]['R2total']
        Mintrinsic = SEMresults[nn]['Mintrinsic']
        Meigv = SEMresults[nn]['Meigv']
        betavals = SEMresults[nn]['betavals']
        fintrinsic1 = SEMresults[nn]['fintrinsic1']
        clusterlist = SEMresults[nn]['clusterlist']

        nl, tsize_full = np.shape(Mintrinsic)
        nruns = np.floor(tsize_full / tsize).astype(int)
        latent_short = np.zeros((nl, tsize))
        for nnl in range(nl):
            temp = np.reshape(Mintrinsic[nnl,:],(nruns,tsize))
            latent_short[nnl,:] = np.mean(temp,axis=0)

        nr, tsize_full = np.shape(Sinput)
        Sinput_short = np.zeros((nr, tsize))
        for nnr in range(nr):
            temp = np.reshape(Sinput[nnr,:],(nruns,tsize))
            Sinput_short[nnr,:] = np.mean(temp,axis=0)

        nc, tsize_full = np.shape(Sconn)
        Sconn_short = np.zeros((nc, tsize))
        for nnc in range(nc):
            temp = np.reshape(Sconn[nnc,:],(nruns,tsize))
            Sconn_short[nnc,:] = np.mean(temp,axis=0)

        valueset = np.concatenate((latent_short,Sinput_short,Sconn_short),axis=0)
        # valueset = copy.deepcopy(Sinput_short)

        # for timepoint in range(tsize):
        # vals = valueset[:,timepoint]

        if nn == 0:
            nvals = np.shape(valueset)[0]
            person_state = np.zeros((NP,tsize,nvals))
            nbetavals = len(betavals)
            betaval_record = np.zeros((NP,nbetavals))
            R2_record = np.zeros(NP)

        # t1 = nn*tsize
        # t2 = (nn+1)*tsize
        person_state[nn,:,:] = valueset.T
        betaval_record[nn,:] = betavals
        R2_record[nn] = R2total


    # show the R2 distribution
    plt.close(204)
    fig = plt.figure(204, figsize = (8,6))
    b = np.linspace(0.3, 0.8, 12)
    plt.hist(R2_record, bins=b)
    plt.xlim(0.2, 0.9)

    print('R2 range from {:.3f} to {:.3f}'.format(R2_record.min(), R2_record.max()))


    betanamelist2 = []
    for nn in range(len(beta_list)):
        sn,tn = beta_list[nn]['pair']
        tname = rnamelist[tn][:4]
        if sn >= nregions:
            sname = 'lat{}'.format(sn-nregions)
        else:
            sname = rnamelist[sn][:4]
        betanamelist2 += ['{}-{}'.format(sname,tname)]

    rnamelistshort = [x[:4] for x in rnamelist]
    valuenames = ['lat0','lat1','lat2'] + rnamelistshort + betanamelist2
    # valuenames = copy.deepcopy(rnamelistshort)

    connection_name_list = []
    ncon = len(csource)
    for nn in range(ncon):
        spair = beta_list[csource[nn]]['pair']
        tpair = beta_list[ctarget[nn]]['pair']
        if spair[0] < nregions:
            sname0 = rnamelist[spair[0]]
        else:
            sname0 = 'latent{}'.format(spair[0]-nregions)
        # latent regions are never target
        sname1 = rnamelist[spair[1]]
        tname1 = rnamelist[tpair[1]]

        name = '{}-{}-{}'.format(sname0,sname1,tname1)
        connection_name_list.append(name)

    # look for common "states" during different periods of the paradigm and in different conditions

    # use PCA to look for common states
    # select data to use for PCA
    ss = 5
    if ss == 0: pnums = np.array(list(range(23)) + list(range(24,57)))  # pain condition, exclude 23
    if ss == 1: pnums = np.array(list(range(57,74)))  # RS1 condition
    if ss == 2: pnums = np.array(list(range(74,94)))  # Low condition
    if ss == 3: pnums = np.array([94, 95] + list(range(97,114)))  # Sens condition
    if ss == 4: pnums = np.array(list(range(114)))  # all conditions
    if ss == 5: pnums = np.array(list(range(23)) + list(range(24,57)) + list(range(74,114)))  # all stim conditions
    NPsub = len(pnums)

    timesample = list(range(3,30))
    ntimesample = len(timesample)

    nstates = 10   # the number to look at
    pca = PCA(n_components = nstates)
    person_stater = np.reshape(person_state,(NP*tsize,nvals))
    person_stater_sub = np.reshape(person_state[pnums,:,:][:,timesample,:],(NPsub*ntimesample,nvals))
    pca.fit(person_stater_sub)

    S_pca_ = pca.fit(person_stater_sub).transform(person_stater_sub)

    # components_   is [ncomponents x nfeatures]
    #  scores from pca.transform(x) is   [nsamples x ncomponents]
    # input data X is  [nsamples x nfeatures]

    print(pca.explained_variance_ratio_)
    components = pca.components_
    singular_values = pca.singular_values_
    ncomponents = pca.n_components_
    explained_variance = pca.explained_variance_

    fit_person_state = (pca.fit_transform(person_stater_sub))
    params = pca.get_params()

    # components is [nterms x nvalues]
    # person_state.T is [ndatapoints x nvalues]
    # person_state.T = loadings @ components   --> fitting the original data to the principcal components
    #  therefore loadings is [ndatapoints x nterms]

    # get loadings for all data, not just ones used for PCA
    mu = np.mean(person_stater, axis=0)
    mu = np.repeat(mu[np.newaxis, :], NP*tsize, axis=0)

    loadings = (person_stater-mu) @ components.T @ np.linalg.inv(components @ components.T)
    fit_check = (loadings @ components) + mu
    fit_check2 = loadings[:,:2] @ components[:2,:] + mu

    loadings2 = pca.transform(person_stater)

    loadingsr = np.reshape(loadings,(NP,tsize,ncomponents))
    fit_checkr = np.reshape(fit_check,(NP,tsize,nvals))


    # compare with pain ratings
    ss = 6
    if ss == 0: pnums = np.array(list(range(23)) + list(range(24,57)))  # pain condition, exclude 23
    if ss == 1: pnums = np.array(list(range(57,74)))  # RS1 condition
    if ss == 2: pnums = np.array(list(range(74,94)))  # Low condition
    if ss == 3: pnums = np.array([94, 95] + list(range(97,114)))  # Sens condition
    if ss == 4: pnums = np.array(list(range(114)))  # all conditions
    if ss == 5: pnums = np.array(list(range(57)) + list(range(74,114)))  # all stim conditions
    if ss == 6: pnums = np.array(list(range(57,74)) + list(range(74,114)) + [94, 95] + list(range(97,114)) ) # all low conditions

    painratings = np.array(GRPcharacteristicsvalues[1,:]).astype(float)   # size is NP

    # now show results .....
    windownum = 54
    plt.close(windownum)
    fig = plt.figure(windownum)
    nrows = np.floor(np.sqrt(nvals)).astype(int)
    ncols = nvals // nrows + 1
    for aa in range(nvals):
        ax = fig.add_subplot(nrows,ncols,aa+1)
        plt.plot(person_stater[:, aa], fit_check[:, aa], 'ob')
        # add a label for each plot
        ax.title.set_text(valuenames[aa])


    # show the different components ...
    ncomponents_to_show = 3
    windownum = 62
    plt.close(windownum)
    fig = plt.figure(windownum)
    ax = fig.add_subplot()
    colorvals = np.array([[1, 0, 0], [1, 0.5, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]])
    colorvals = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    state_definition = []
    bwidth = 1.0 / (ncomponents_to_show + 1)
    for nn in range(ncomponents_to_show):
        x = np.array(range(nvals)) + nn * bwidth
        state = components[nn,:]
        statelabel = 'state{}'.format(nn)
        ax.barh(x, np.array(state), bwidth, color=colorvals[nn, :], label=statelabel)

    ax.set_xlabel('Values')
    ax.set_title('Signal values for each state')
    y = x = np.array(range(nvals)) + (ncomponents_to_show / 2) * bwidth
    ax.set_yticks(y)
    ax.set_yticklabels(valuenames)
    ax.legend()


    # are there patterns in the amounts of each component, and pain ratings?
    #   but ... at which time point?
    p = painratings[pnums]
    c = np.where(p > 20)[0]

    cluster_record = []
    timepoints = range(8,18)   # anticipation period
    timepoints = range(18,23)  # stim period
    transitiontime = 2
    for tp in timepoints:
        L = loadingsr[pnums[c],tp,:6]
        data = np.concatenate((painratings[pnums[c],np.newaxis], L), axis = 1)
        # try clustering to look for patterns
        nstates = 6
        kmeans = KMeans(n_clusters=nstates, random_state=1)
        kmeans.fit(data)
        cv = kmeans.cluster_centers_
        labels = kmeans.labels_
        # sort by pain rating
        # x = np.argsort(cv[:,0])
        # for aa in range(nstates): print(cv[x[aa],:])

        cluster_record.append({'cv':cv, 'labels':labels})

    # plot clusters
    windownum = 8
    plt.close(windownum)
    fig = plt.figure(windownum)
    fig.suptitle('6 clusters')
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:,0],data[:,1],data[:,2],'ob')
    ax.scatter(cv[:,0],cv[:,1],cv[:,2],'or')


    # plot cluster centers at each time point, using the cluster labels at one time point
    tref = 0
    cluster_record2 = []
    labels1 = cluster_record[tref]['labels']
    for tp in timepoints:
        L = loadingsr[pnums[c], tp, :6]
        data = np.concatenate((painratings[pnums[c], np.newaxis], L), axis=1)
        cv2 = np.zeros((nstates,7))
        for aa in range(nstates):
            x = np.where(labels1 == aa)[0]
            datasub = data[x,:]
            cv2[aa,:] = np.mean(datasub,axis=0)
            print('state {}  {} members  pain rating {:.1f}'.format(aa,len(x),cv2[aa,0]))
        cluster_record2.append({'cv': cv2, 'labels': labels1})

    # plot temporal progression of clusters
    ntime = len(cluster_record2)
    dynamic_cv_data = np.zeros((ntime,nstates,7))
    for aa in range(nstates):
        for tt in range(ntime):
            dynamic_cv_data[tt,:,:] = cluster_record2[tt]['cv']


    clusterpains = dynamic_cv_data[0,:,0]
    x = np.argsort(clusterpains)
    windownum = 10
    plt.close(windownum)
    fig = plt.figure(windownum)
    fig.suptitle('temporal progress of clusters')
    statelist = list(range(nstates))
    # statelist.remove(8)
    xmax = dynamic_cv_data[:,:,1].max()
    xmin = dynamic_cv_data[:,:,1].min()
    ymax = dynamic_cv_data[:,:,2].max()
    ymin = dynamic_cv_data[:,:,2].min()
    zmax = dynamic_cv_data[:,:,3].max()
    zmin = dynamic_cv_data[:,:,3].min()
    nrows = np.floor(np.sqrt(nstates)).astype(int)
    ncols = np.ceil(nstates/nrows).astype(int)
    for aa in range(len(statelist)):
        ax = fig.add_subplot(ncols,nrows,aa+1, projection='3d')
        cv = dynamic_cv_data[:,statelist[x[aa]],:]
        clusterpain = cv[0,0]
        ax.plot(cv[:,1],cv[:,2],cv[:,3],'-ok')
        ax.plot(cv[transitiontime:,1],cv[transitiontime:,2],cv[transitiontime:,3],'-b')
        ax.plot(cv[0,1],cv[0,2],cv[0,3],'og')
        ax.plot(cv[transitiontime,1],cv[transitiontime,2],cv[transitiontime,3],'or')
        ax.plot(cv[-1,1],cv[-1,2],cv[-1,3],'oy')
        paintext = 'pain = {:.1f}'.format(clusterpain)
        ax.title.set_text(paintext)

        plt.xlim(xmin, xmax)
        plt.ylim(ymin,ymax)   # component 2
        # if statelist[x[aa]] != 8:
        #     # plt.ylim(-0.7,1.5)   # component 1
        #     plt.ylim(-1.5,1.5)   # component 2

    windownum = 29
    plt.close(windownum)
    fig = plt.figure(windownum)
    fig.suptitle('temporal progress of clusters')
    ax = fig.add_subplot(projection='3d')
    statelist = list(range(nstates))
    # statelist.remove(8)
    flagpoint = 2
    for aa in statelist:
        # ax.scatter(data[:,0],data[:,1],data[:,2],'ob')
        cv = dynamic_cv_data[:,aa,:]
        ax.plot3D(cv[:,0],cv[:,1],cv[:,2],'-og')
    ax.scatter3D(dynamic_cv_data[flagpoint,statelist,0], dynamic_cv_data[flagpoint,statelist,1], dynamic_cv_data[flagpoint,statelist,2], color = [1,0,0])
    plt.ylim(-5, 5)


    for state in range(6):
        comp = 3
        p = dynamic_cv_data[flagpoint,state,0]
        title = 'pain {:.1f} component {} vs pain'.format(p,comp)
        y = dynamic_cv_data[:,state,comp]
        x = np.array(list(range(5)))
        XYplot(x, y, 30+state, title, [0, 1, 0], 'o', True)
        plt.ylim(-5,3)




    # change methods now ....

    # compare with pain ratings
    ss = 6
    if ss == 0: pnums = np.array(list(range(23)) + list(range(24,57)))  # pain condition
    if ss == 1: pnums = np.array(list(range(57,74)))  # RS1 condition
    if ss == 2: pnums = np.array(list(range(74,94)))  # Low condition
    if ss == 3: pnums = np.array([94, 95] + list(range(97,114)))  # Sens condition
    if ss == 4: pnums = np.array(list(range(23)) + list(range(24,114)))  # all conditions
    if ss == 5: pnums = np.array(list(range(23)) + list(range(24,57)) + list(range(74,114)))  # all stim conditions
    if ss == 6: pnums = np.array(list(range(23)) + list(range(24,57)) + list(range(74,94)))  # all pain conditions

    painratings = np.array(GRPcharacteristicsvalues[1,:]).astype(float)   # size is NP
    loadings_corr = np.zeros((tsize,nstates))
    for aa in range(tsize):
        for bb in range(nstates):
            p = painratings[pnums]
            y = loadingsr[pnums,aa,bb]
            c = np.where(p > 20)[0]
            R = np.corrcoef(p[c],y[c])
            loadings_corr[aa,bb] = R[0,1]

    # correlations with original values
    person_state_corr = np.zeros((tsize,nvals))
    for aa in range(tsize):
        for bb in range(nvals):
            p = painratings[pnums]
            y = person_state[pnums,aa,bb]
            c = np.where(p > 20)[0]
            R = np.corrcoef(p[c],y[c])
            person_state_corr[aa,bb] = R[0,1]

    # correlations with beta values
    betaval_corr = np.zeros(nbetavals)
    for aa in range(nbetavals):
        if np.std(betaval_record[pnums, aa]) > 0:
            p = painratings[pnums]
            y = betaval_record[pnums, aa]
            c = np.where(p > 20)[0]
            R = np.corrcoef(p[c],y[c])
            betaval_corr[aa] = R[0,1]

    x = np.argsort(np.abs(betaval_corr))
    x = x[::-1]
    aa = 1
    title = 'beta vals {}'.format(connection_name_list[x[aa]])
    p = painratings[pnums]
    c = np.where(p > 20)[0]
    XYplot(painratings[pnums[c]], betaval_record[pnums[c], x[aa]], 21, title, [0, 1, 0], 'o', True)


    # look at top results in loadings_corr
    ia,ib = np.unravel_index(np.argsort(np.abs(loadings_corr), axis=None), loadings_corr.shape)
    ia = ia[::-1]
    ib = ib[::-1]
    aa=1
    title = 'loadings, time {} {}'.format(ia[aa], connection_name_list[ib[aa]])
    p = painratings[pnums]
    c = np.where(p > 20)[0]
    XYplot(painratings[pnums[c]], loadingsr[pnums[c], ia[aa],ib[aa]], 22, title, [0, 1, 0], 'o', True)


    # look at top results in person_state_corr
    ia,ib = np.unravel_index(np.argsort(np.abs(person_state_corr), axis=None), person_state_corr.shape)
    ia = ia[::-1]
    ib = ib[::-1]
    aa = 3
    title = 'person state, time {}, {}'.format(ia[aa],connection_name_list[ib[aa]])
    p = painratings[pnums]
    c = np.where(p > 20)[0]
    XYplot(painratings[pnums[c]], person_state[pnums[c],ia[aa],ib[aa]], 23, title, [0, 1, 0], 'o', True)



    # # sequence of changes over time course of paradigm
    # bb = 33
    # bb = 5
    # ymax = np.max(person_state[pnums,:,bb])
    # ymin = np.min(person_state[pnums,:,bb])
    # windownum = 10
    # plt.close(windownum)
    # fig = plt.figure(windownum)
    # fig.suptitle(valuenames[bb])
    # aa = 0
    # hh, = plt.plot(painratings[pnums],person_state[pnums,aa,bb],'ob')
    # plt.ylim(ymin,ymax)
    # label = 'time {}'.format(aa)
    # ahh = plt.annotate(label, (xa,ya))
    # xa = np.min(painratings[pnums])
    # ya = ymin
    #
    # for aa in range(tsize):
    #     # plt.plot(painratings[pnums],person_state[pnums,aa,bb],'ob')
    #     hh.set_ydata(person_state[pnums,aa,bb])
    #     plt.draw()
    #     label = 'time {}'.format(aa)
    #     ahh.remove()
    #     ahh = plt.annotate(label, (xa,ya))
    #     plt.show()
    #     plt.pause(0.5)




    # use kmeans to look for common states
    windownum = 33
    plt.close(windownum)
    fig = plt.figure(windownum)

    ss = 5
    if ss == 0: pnums = np.array(list(range(57)))  # pain condition
    if ss == 1: pnums = np.array(list(range(57,74)))  # RS1 condition
    if ss == 2: pnums = np.array(list(range(74,94)))  # Low condition
    if ss == 3: pnums = np.array(list(range(94,114)))  # Sens condition
    if ss == 4: pnums = np.array(list(range(114)))  # all conditions
    if ss == 5: pnums = np.array(list(range(57)) + list(range(74,94)))  # all pain conditions

    tperiod = np.array(list(range(6,17)))   # before stim
    tperiod = np.array(list(range(25,36)))   # after stim
    tperiod = np.array(list(range(40)))    #  all time points
    tperiod = np.array(list(range(18,23)))   # during stim

    valuelist = np.array(list(range(3,13)))   # subset of values
    valuelist = np.array(list(range(46)))   # all values
    npnums = len(pnums)
    epoch = len(tperiod)
    subnvals = len(valuelist)

    vals = np.reshape(person_state[pnums,:,:][:,tperiod,:][:,:,valuelist],(npnums*epoch, subnvals))

    # try clustering to look for states
    nstates = 6
    kmeans = KMeans(n_clusters=nstates, random_state=1)
    kmeans.fit(vals)
    cv = kmeans.cluster_centers_

    # use these as initial estimates for gradient-descent fitting method
    initial_states = copy.deepcopy(cv)
    working_states, ssqd_record, wout, fit = gradient_descent_states(vals, initial_states, 250, 1e-4, 1e-6, 0.05)

    # now show results .....
    windownum = 49
    plt.close(windownum)
    fig = plt.figure(windownum)
    nrows = np.floor(np.sqrt(subnvals)).astype(int)
    ncols = subnvals // nrows + 1
    for aa in range(subnvals):
        ax = fig.add_subplot(nrows,ncols,aa+1)
        plt.plot(vals[:, aa], fit[:, aa], 'ob')
        # add a label for each plot
        ax.title.set_text(valuenames[valuelist[aa]])

    meanvals = np.repeat(np.mean(vals,axis=0)[np.newaxis,:],npnums*epoch,axis=0)
    R2 = 1 - np.sum((vals-fit)**2)/np.sum((vals-meanvals)**2)
    print('R2 value of fit is {:.3f}'.format(R2))

    # identify when the states occur during the stimulation paradigm
    statenum = np.argmax(np.abs(wout),axis=1)
    statenumr = np.reshape(statenum,(npnums,epoch))
    woutr = np.reshape(wout,(npnums,epoch,nstates))
    stateflag = copy.deepcopy(woutr)
    stateflag[np.abs(stateflag)>0] = 1
    stateprob = np.sum(stateflag,axis=0)/NP

    windownum=34
    plt.close(windownum)
    fig = plt.figure(windownum)
    nrows = np.floor(np.sqrt(nstates)).astype(int)
    ncols = nstates // nrows + 1
    colorvals = np.array([[1, 0, 0], [1, 0.5, 0], [1, 1, 0], [0.5, 1, 0],[0,1,0], [0, 1, 0.5], [0, 1, 1], [0, 0.5, 1], [0, 0, 1]])
    for nn in range(nstates):
        ax = fig.add_subplot(nstates,1,nn+1)
        plt.plot(range(epoch),stateprob[:,nn],'-o',color = colorvals[nn,:])


    # compare with covariates
    # GRPcharacteristicslist = cov['GRPcharacteristicslist']
    # GRPcharacteristicsvalues = cov['GRPcharacteristicsvalues']
    painratings = np.array(GRPcharacteristicsvalues[1,:]).astype(float)   # size is NP
    woutr   # size is NP x tsize x nstates
    person_state   # size is NP x tsize x nvals
    stateflag = copy.deepcopy(woutr)
    stateflag[np.abs(stateflag)>0] = 1

    wcorr = np.zeros((tsize,nstates))
    state_corr = np.zeros((tsize,nstates))
    person_state_corr = np.zeros((NP,nvals))
    for aa in range(tsize):
        for bb in range(nstates):
            R = np.corrcoef(painratings,woutr[:,aa,bb])
            wcorr[aa,bb] = R[0,1]
            R = np.corrcoef(painratings,stateflag[:,aa,bb])
            state_corr[aa,bb] = R[0,1]

    for aa in range(tsize):
        for bb in range(nvals):
            R = np.corrcoef(painratings,person_state[:,aa,bb])
            person_state_corr[aa,bb] = R[0,1]


    look_at_groups = False
    if look_at_groups:
        # look at groups
        for ss in range(5):
            if ss == 0: pnums2 = np.array(list(range(57)))  # pain condition
            if ss == 1: pnums2 = np.array(list(range(57,74)))  # RS1 condition
            if ss == 2: pnums2 = np.array(list(range(74,94)))  # Low condition
            if ss == 3: pnums2 = np.array(list(range(94,114)))  # Sens condition
            if ss == 4: pnums2 = np.array(list(range(114)))  # all conditions

            person_state2 = person_state[pnums2,:,:]
            npnums2 = len(pnums2)
            vals2 = np.reshape(person_state2,(npnums2*epoch, subnvals))
            woutr2 = woutr[pnums2,:,:]
            wout2 = np.reshape(woutr2, (npnums2*epoch, nstates))
            fit2 = wout2 @ working_states

            meanvals2 = np.repeat(np.mean(vals2,axis=0)[np.newaxis,:],npnums2*epoch,axis=0)
            R2 = 1 - np.sum((vals2-fit2)**2)/np.sum((vals2-meanvals2)**2)
            print('R2 value of fit is {:.3f}'.format(R2))

            # identify when the states occur during the stimulation paradigm
            statenum2 = np.argmax(np.abs(wout2),axis=1)
            statenumr2 = np.reshape(statenum2,(npnums2,epoch))
            stateflag2 = copy.deepcopy(woutr2)
            stateflag2[np.abs(stateflag2)>0] = 1
            stateprob2 = np.sum(stateflag2,axis=0)/npnums2

            windownum=40+ss
            plt.close(windownum)
            fig = plt.figure(windownum)
            nrows = np.floor(np.sqrt(nstates)).astype(int)
            ncols = nstates // nrows + 1
            colorvals = np.array([[1, 0, 0], [1, 0.5, 0], [1, 1, 0], [0.5, 1, 0],[0,1,0], [0, 1, 0.5], [0, 1, 1], [0, 0.5, 1], [0, 0, 1]])
            for nn in range(nstates):
                ax = fig.add_subplot(nstates,1,nn+1)
                plt.plot(range(epoch),stateprob2[:,nn],'-o',color = colorvals[nn,:])



    #
    # ax = fig.add_subplot()
    # colorvals = np.array([[1, 0, 0], [1, 0.5, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]])
    # state_definition = []
    # bwidth = 1.0 / (ncomponents + 1)
    # for nn in range(ncomponents):
    #     x = np.array(range(nvals)) + nn * bwidth
    #     state = cv[nn,:]
    #     statelabel = 'state{}'.format(nn)
    #     ax.barh(x, np.array(state), bwidth, color=colorvals[nn, :], label=statelabel)
    #
    # ax.set_xlabel('Values')
    # ax.set_title('Signal values for each state')
    # y = x = np.array(range(nvals)) + (ncomponents / 2) * bwidth
    # ax.set_yticks(y)
    # ax.set_yticklabels(valuenames)
    # ax.legend()
    #



def identify_states(data, initial_states):
    ndata, nvals = np.shape(data)
    nstates, nvals2 = np.shape(initial_states)
    assert nvals == nvals2, print('identify_states:   data and initial_states do not have consistent sizes')

    working_states = copy.deepcopy(initial_states)

    ssq = cost_function(data,working_states)   # initial ssq
    ssq_total = np.sum(ssq)


def cost_function(data,working_states,Lweight):
    ndata, nvals = np.shape(data)
    nstates, nvals2 = np.shape(working_states)
    ssq_record = np.zeros(ndata)
    wout_record = np.zeros((ndata,nstates))

    working_states_mean = np.mean(working_states,axis=1)
    working_states_zm = working_states - np.repeat(working_states_mean[:,np.newaxis],nvals2,axis=1)
    statecov = np.diag(working_states_zm @ working_states_zm.T)

    for nn in range(ndata):
        d = data[nn,:]
        d0 = d - np.mean(d)
        w = (d0 @ working_states_zm.T)/(statecov + 1.0e-10)
        fit = np.diag(w) @ working_states_zm
        d2 = np.repeat(d0[np.newaxis,:],nstates,axis=0)
        ssq = np.sum((fit-d2)**2,axis=1)
        x = np.argmin(ssq)
        # wout = np.zeros(nstates)
        # wout[x] = w[x]
        # wout_record[nn,:] = wout
        ssq_record[nn] = ssq[x]

    err = np.sum(ssq_record)
    cost = np.sum(np.abs(working_states))  # L1 regularization
    totalcost = err + Lweight * cost
    return totalcost, ssq_record


def fit_function(data,working_states):
    ndata, nvals = np.shape(data)
    nstates, nvals2 = np.shape(working_states)
    fit_record = np.zeros(np.shape(data))
    wout_record = np.zeros((ndata,nstates))

    working_states_mean = np.mean(working_states,axis=1)
    working_states_zm = working_states - np.repeat(working_states_mean[:,np.newaxis],nvals2,axis=1)
    statecov = np.diag(working_states_zm @ working_states_zm.T)

    for nn in range(ndata):
        d = data[nn,:]
        d0 = d - np.mean(d)
        w = (d0 @ working_states_zm.T)/(statecov + 1.0e-10)
        fit = np.diag(w) @ working_states_zm    # + np.repeat(working_states_mean[:,np.newaxis],nvals2,axis=1)
        d2 = np.repeat(d0[np.newaxis,:],nstates,axis=0)
        ssq = np.sum((fit-d2)**2,axis=1)
        x = np.argmin(ssq)
        wout = np.zeros(nstates)
        wout[x] = w[x]
        wout_record[nn,:] = wout
        fit_record[nn,:] = fit[x,:] + np.mean(d)
        # ssq_record[nn] = ssq[x]

    return wout_record, fit_record


def gradients_for_statevalues(data, working_states, dval, Lweight):
    starttime = time.time()
    nstates,nvals = np.shape(working_states)
    totalcost, ssq_record = cost_function(data,working_states,Lweight)
    dstate_db = np.zeros((nstates,nvals))
    # for nn in range(nstates*nvals):
    for aa in range(nstates):
        for bb in range(nvals):
    #     aa = nn // nvals
    #     bb = nn % nvals
            temp_states = copy.deepcopy(working_states)
            temp_states[aa,bb] += dval
            tempcost, ssq_record = cost_function(data, temp_states, Lweight)
            dstate_db[aa,bb] = (tempcost - totalcost)/dval
    endtime = time.time()
    print('gradients_for_statevalues:  time elapsed = {:.1f} sec'.format(endtime-starttime))
    return dstate_db


def gradient_descent_states(data, initial_states, itermax = 20, initial_alpha = 0.01, initial_Lweight = 1e-8, initial_dval = 0.1):
    working_states = copy.deepcopy(initial_states)
    lastgood_working_states = copy.deepcopy(initial_states)
    ndata, nvals = np.shape(data)
    nstates, nvals2 = np.shape(working_states)

    results_record = []
    ssqd_record = []

    alpha = initial_alpha
    Lweight = initial_Lweight
    dval = initial_dval

    totalcost, ssq = cost_function(data,working_states,Lweight)

    ssqd_starting = totalcost
    ssqd_record += [totalcost]

    alpha_limit = 1.0e-6
    iter = 0
    while alpha > alpha_limit and iter < itermax:
        iter += 1
        # gradients in state values
        dstate_db = gradients_for_statevalues(data,working_states,dval,Lweight)

        # apply the changes
        working_states -= alpha * dstate_db
        totalcost_new, ssq_new = cost_function(data, working_states, Lweight)

        if totalcost_new >= totalcost:
            alpha *= 0.1
            # revert back to last good values
            working_states = copy.deepcopy(lastgood_working_states)
            dssqd = totalcost - totalcost_new
            print('state vals:  iter {} alpha {:.3e}  delta ssq > 0  - no update'.format(iter, alpha))
        else:
            # save the good values
            lastgood_working_states = copy.deepcopy(working_states)

            dssqd = totalcost - totalcost_new
            totalcost = totalcost_new
            ssqd_record += [totalcost]

        wout_temp, fit = fit_function(data, lastgood_working_states)
        wcount = np.sum((np.abs(wout_temp) > 0), axis=0)
        count_text = '     distribution:'
        for cc in range(nstates):  count_text += '  {:.1f} %'.format(100.0*wcount[cc]/ndata)

        meanvals = np.repeat(np.mean(data, axis=0)[np.newaxis, :], ndata, axis=0)
        R2 = 1 - np.sum((data - fit) ** 2) / np.sum((data - meanvals) ** 2)

        print('state vals:  iter {} alpha {:.3e} R2 {:.3f}  delta ssq {:.4f}  relative: {:.1f} percent'.format(iter, alpha,R2, -dssqd, 100.0 * totalcost / ssqd_starting))
        print(count_text)

    # check quality of result
    working_states = copy.deepcopy(lastgood_working_states)
    wout, fit = fit_function(data, working_states)

    return working_states, ssqd_record, wout, fit



def XYplot(x,y, windownum = 1, title = '', color = [0,0,1], symbol = 'o', regressionline = False):
    plt.close(windownum)
    fig = plt.figure(windownum)
    plt.plot(x,y,marker = symbol, linestyle = '', color = color)
    if len(title) > 0:
        fig.suptitle(title)

    if regressionline:
        nx = len(x)
        G = np.concatenate((x[np.newaxis,:],np.ones((1,nx))),axis=0)
        b = y @ G.T @ np.linalg.inv(G @ G.T)
        fit = b @ G
        plt.plot(x, fit, marker='', color=color)

        R2 = 1 - np.sum((y-fit)**2) / (np.sum((y-np.mean(y))**2) + 1.0e-20)
        R2text = 'R2 = {:.3f}'.format(R2)

        scale = np.max(np.abs([np.log10(np.abs(b))]))
        if scale > 3:
            linetext = 'y = {:.3e} x + {:.3e}  {}'.format(b[0],b[1],R2text)
        else:
            linetext = 'y = {:.3f} x + {:.3f}  {}'.format(b[0],b[1],R2text)

        xa = np.min(x)
        if b[0] > 0:
            ya = np.max(y)
        else:
            ya = np.min(y)
        plt.annotate(linetext, (xa,ya))
