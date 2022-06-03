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

from hmmlearn import hmm



# main program
def analyze_sapm_latents(studyname, cord_cluster, type):

    # studyname:   allthreat, RS1nostim, Low, Sens, all_condition

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

        # t1 = nn*tsize
        # t2 = (nn+1)*tsize
        person_state[nn,:,:] = valueset.T

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


    # find dominant states in the data set
    pca = PCA(n_components = 6)
    person_stater = np.reshape(person_state,(NP*tsize,nvals))
    pca.fit(person_stater)

    print(pca.explained_variance_ratio_)
    components = pca.components_
    singular_values = pca.singular_values_
    ncomponents = pca.n_components_
    explained_variance = pca.explained_variance_

    fit_person_state = (pca.fit_transform(person_stater))
    params = pca.get_params()


    # components is [nterms x nvalues]
    # person_state.T is [ndatapoints x nvalues]
    # person_state.T = loadings @ components   --> fitting the original data to the principcal components
    #  therefore loadings is [ndatapoints x nterms]
    loadings = person_stater @ components.T @ np.linalg.inv(components @ components.T)
    fit_check = (loadings @ components)

    loadingsr = np.reshape(loadings,(NP,tsize,ncomponents))
    fit_checkr = np.reshape(fit_check,(NP,tsize,nvals))

    # look for common "states" during different periods of the paradigm and in different conditions

    windownum = 10
    plt.close(windownum)
    fig = plt.figure(windownum)
    for ss in range(5):
        if ss == 0: pnums = np.array(list(range(55)))  # pain condition
        if ss == 1: pnums = np.array(list(range(55,74)))  # RS1 condition
        if ss == 2: pnums = np.array(list(range(74,94)))  # Low condition
        if ss == 3: pnums = np.array(list(range(94,114)))  # Sens condition
        if ss == 4: pnums = np.array(list(range(114)))  # all conditions
        tperiod = np.array(list(range(6,17)))   # before stim
        npnums = len(pnums)
        epoch = len(tperiod)
        vals = np.reshape(loadingsr[pnums,:,:][:,tperiod,:],(npnums*epoch, ncomponents))

        fig.add_subplot(2,3,ss+1)

        # try clustering to look for states
        kmeans = KMeans(n_clusters=4, random_state=1)
        kmeans.fit(vals)
        cv = kmeans.cluster_centers_
        plt.plot(vals[:,0],vals[:,1],'xb')
        plt.plot(cv[:,0],cv[:,1],'or')
        plt.xlim(-15,15)
        plt.ylim(-10,10)

        # for vv in range(6):
        #     plt.hist(vals[:,vv],bins = 31)


    mean_loadingsr = np.mean(loadingsr,axis=0)

    windownum = 27
    plt.close(windownum)
    fig = plt.figure(windownum)
    runnum = 70
    tpoint = 10
    plt.plot(person_state[runnum,tpoint,:],fit_checkr[runnum,tpoint,:],'ob')


    windownum = 31
    plt.close(windownum)
    fig = plt.figure(windownum)
    runnum = 70
    termnum = 20
    componentnum = 1
    plt.plot(range(tsize),person_state[runnum,:,termnum],'-ob')
    plt.plot(range(tsize),fit_checkr[runnum,:,termnum],'-og')
    plt.plot(range(tsize),loadingsr[runnum,:,componentnum],'-xr')



    windownum = 71
    plt.close(windownum)
    fig = plt.figure(windownum)
    componentnum = 6
    plt.plot(range(tsize),mean_loadingsr[:,componentnum],'-ob')



    windownum = 28
    plt.close(windownum)
    fig = plt.figure(windownum)
    termnum = 0
    ntoshow = 114
    for person in range(ntoshow):
        plt.plot(range(40), loadingsr[person,:, termnum], 'ob')

        if person == 0:
            totalload = np.array(loadingsr[person,:, termnum])
        else:
            totalload += np.array(loadingsr[person,:, termnum])
    totalload /= ntoshow
    plt.plot(range(40),totalload,'-or')


    # hidden markov model---------------------------------------------
    nruns,tsize,nvals = np.shape(person_state)

    training_ratio = 0.7
    train_sample = np.sort(np.random.choice(list(range(nruns)),np.floor(training_ratio*nruns).astype(int), replace=False))
    validate_sample = np.array([x for x in list(range(nruns)) if x not in train_sample])

    ntrain = len(train_sample)
    nvalidate = len(validate_sample)

    X = np.reshape(person_state,(nruns*tsize,nvals))
    X_train = np.reshape(person_state[train_sample,:,:],(ntrain*tsize,nvals))
    X_validate = np.reshape(person_state[validate_sample,:,:],(nvalidate*tsize,nvals))


    best_score_record = []
    for ncomponents in [4]:
        model = hmm.GaussianHMM(n_components=ncomponents, covariance_type="full", init_params = 'stmc')
        # assume 4 possible states:   outward focussed, inward focused, focusing on pain, ignoring pain
        # model.startprob_ = np.array([0.4, 0.4, 0.1, 0.1])
        # model.transmat_ = np.array([[0.5, 0.3, 0.1, 0.1],
        #                             [0.3, 0.5, 0.1, 0.1],
        #                             [0.1, 0.1, 0.5, 0.3],
        #                             [0.1, 0.1, 0.3, 0.5]])


        best_score = best_model = None
        n_fits = 250
        np.random.seed(13)
        for idx in range(n_fits):
            model = hmm.GaussianHMM(n_components=ncomponents, covariance_type="full",
                                    random_state=idx, init_params='stmc')
            model.fit(X_train)
            score = model.score(X_validate)
            print(f'Model #{idx}\tScore: {score}')
            if best_score is None or score > best_score:
                best_model = model
                best_score = score

        print(f'Best score:      {best_score}')
        best_score_record.append({'ncomponents':ncomponents, 'best_score':best_score})

        # use the Viterbi algorithm to predict the most likely sequence of states
        # given the model
        states = best_model.predict(X_validate)

        # save the model---------------------------------------------
        model_output_name = r'E:\{}_markov_model_{}.pkl'.format(studyname,ncomponents)
        reload_the_model = False
        save_the_model = False
        if save_the_model:
            import pickle
            with open(model_output_name, "wb") as file:
                pickle.dump(best_model, file)
        if reload_the_model:
            with open(model_output_name, "rb") as file:
                pickle.load(file)
        # ----------------------------------------------------

        # look at the results
        predicted_states = best_model.predict(X)

        state_prob = best_model.predict_proba(X)
        state_probr = np.reshape(state_prob,(nruns,tsize,ncomponents))
        mean_state_probr = np.mean(state_probr,axis=0)

        windownum = 37
        plt.close(windownum)
        fig = plt.figure(windownum)
        for nn in range(ncomponents):
            ax1 = fig.add_subplot(ncomponents,1,nn+1)
            plt.plot(range(40), mean_state_probr[:, nn], '-or')


    # model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
    # model.covars_ = np.tile(np.identity(2), (3, 1, 1))
    # X, Z = model.sample(100)

    #---------------------

    # print(f'Emission Matrix Generated:\n{best_model.emissionprob_.round(3)}\n\n')

    plt.close(121)
    fig = plt.figure(121)
    ax = fig.add_subplot()
    colorvals = np.array([[1,0,0],[1,0.5,0],[0,1,0],[0,1,1],[0,0,1]])
    state_definition = []
    bwidth = 1.0/(ncomponents+1)
    for nn in range(ncomponents):
        x = np.array(range(nvals))+nn*bwidth
        state = best_model.sample(currstate=nn)[0]
        state_definition.append({'state':state})
        statelabel = 'state{}'.format(nn)
        ax.barh(x,np.array(state[0]),bwidth,color = colorvals[nn,:],label = statelabel)

    ax.set_xlabel('Values')
    ax.set_title('Signal values for each state')
    y = x = np.array(range(nvals))+(ncomponents/2)*bwidth
    ax.set_yticks(y)
    ax.set_yticklabels(valuenames)
    ax.legend()


    # look at the states in a particular run
    runnum = 20
    run_state_prob = state_probr[runnum,:,:]
    valueset = person_state[runnum,:,:]

    valuenum = 7
    val_actual = np.zeros(tsize)
    val_predicted = np.zeros(tsize)
    for tt in range(tsize):
        p = run_state_prob[tt,:]
        pc = np.argmax(p)
        val_actual[tt] = valueset[tt,valuenum]
        val_predicted[tt] = state_definition[pc]['state'][0][valuenum]

    plt.close(99)
    fig = plt.figure(99)
    plt.plot(range(tsize),val_actual,'-ob')
    plt.plot(range(tsize),val_predicted,'-xr')