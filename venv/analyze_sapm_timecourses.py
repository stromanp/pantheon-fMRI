
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



# main program
# def analyze_sapm_latents(studyname, cord_cluster, type):
#
#     # studyname:   allthreat, RS1nostim, Low, Sens, all_condition
#
#     # setup inputs for this particular set of studies
#     if type == 'fixed':
#         if cord_cluster == 0:
#             cnums = [0, 3, 3, 1, 4, 1, 3, 3, 4, 1]  # fixed 0
#         if cord_cluster == 1:
#             cnums = [1, 3, 3, 1, 3, 1, 3, 3, 2, 1]  # fixed 1
#         if cord_cluster == 2:
#             cnums = [2, 3, 3, 1, 1, 1, 3, 3, 2, 0]  # fixed 2
#         if cord_cluster == 3:
#             cnums = [3, 3, 2, 1, 0, 1, 2, 3, 4, 1]  # fixed 3
#         if cord_cluster == 4:
#             cnums = [4, 3, 3, 1, 0, 1, 2, 3, 4, 3]  # fixed 4
#     else:
#         if cord_cluster == 0:
#             cnums = [0, 4, 4, 2, 2, 3, 3, 2, 3, 1]  # random 0
#         if cord_cluster == 1:
#             cnums = [1, 4, 2, 3, 2, 1, 1, 2, 3, 0]  # random 1
#         if cord_cluster == 2:
#             cnums = [2, 2, 2, 0, 0, 2, 0, 3, 1, 3]  # random 2
#         if cord_cluster == 3:
#             cnums = [3, 3, 1, 4, 4, 1, 3, 3, 1, 0]  # random 3
#         if cord_cluster == 4:
#             cnums = [4, 4, 2, 1, 0, 3, 3, 3, 2, 0]  # random 4
#
#     basedir = r'E:\beta_distribution'
#     outputdir = r'E:\beta_distribution\{}_{}_C6RD{}'.format(studyname,type,cord_cluster)
#     if not os.path.exists(outputdir): os.mkdir(outputdir)
#
#     if studyname == 'all_condition':
#         covariatesfile = r'E:\all_condition_covariates.npy'
#         regiondataname = r'E:\all_condition_region_data.npy'
#         clusterdataname = r'E:\threat_safety_clusterdata.npy'
#
#     if studyname == 'allthreat':
#         covariatesfile = r'E:\allthreat_covariates.npy'
#         regiondataname = r'E:\threat_safety_regiondata_allthreat55.npy'
#         clusterdataname = r'E:\threat_safety_clusterdata.npy'
#
#     if studyname == 'RS1nostim':
#         covariatesfile = r'E:\RS1nostim_covariates.npy'
#         regiondataname = r'E:\RS1nostim_region_data.npy'
#         clusterdataname = r'E:\threat_safety_clusterdata.npy'
#
#     if studyname == 'Low':
#         covariatesfile = r'E:\Low_covariates.npy'
#         regiondataname = r'E:\Low_region_data.npy'
#         clusterdataname = r'E:\threat_safety_clusterdata.npy'
#
#     if studyname == 'Sens':
#         covariatesfile = r'E:\Sens_covariates.npy'
#         regiondataname = r'E:\Sens_region_data.npy'
#         clusterdataname = r'E:\threat_safety_clusterdata.npy'
#
#     SEMresultsname = os.path.join(outputdir, 'SEMphysio_model.npy')
#     SEMparametersname = os.path.join(outputdir, 'SEMparameters_model5.npy')
#     networkfile = r'E:\network_model_5cluster_v5_w_3intrinsics.xlsx'
#
#     cov = np.load(covariatesfile, allow_pickle=True).flat[0]
#     GRPcharacteristicslist = cov['GRPcharacteristicslist']
#     GRPcharacteristicsvalues = cov['GRPcharacteristicsvalues']
#
#
#     # load paradigm data--------------------------------------------------------------------
#     DBname = r'E:\graded_pain_database_May2022.xlsx'
#     xls = pd.ExcelFile(DBname, engine='openpyxl')
#     df1 = pd.read_excel(xls, 'paradigm1_BOLD')
#     del df1['Unnamed: 0']  # get rid of the unwanted header column
#     fields = list(df1.keys())
#     paradigm = df1['paradigms_BOLD']
#     timevals = df1['time']
#     paradigm_centered = paradigm - np.mean(paradigm)
#     dparadigm = np.zeros(len(paradigm))
#     dparadigm[1:] = np.diff(paradigm_centered)
#
#
#     # rnamelist = ['C6RD',  'DRt', 'Hypothalamus','LC', 'NGC',
#     #                'NRM', 'NTS', 'PAG', 'PBN', 'Thalamus']
#     full_rnum_base =  np.array([0,5,10,15,20,25,30,35,40,45])
#
#     namelist = ['C6RD',  'DRt', 'Hypothalamus','LC', 'NGC', 'NRM', 'NTS', 'PAG', 'PBN', 'Thalamus',
#             'Rtotal', 'R C6RD',  'R DRt', 'R Hyp','R LC', 'R NGC', 'R NRM', 'R NTS', 'R PAG',
#             'R PBN', 'R Thal']
#
#     #----------------------------------------------------------------
#     # load the results now
#     #----------------------------------------------------------------
#     SEMparams = np.load(SEMparametersname, allow_pickle=True).flat[0]
#     # load the data values
#     betanamelist = SEMparams['betanamelist']
#     beta_list = SEMparams['beta_list']
#     nruns_per_person = SEMparams['nruns_per_person']
#     nclusterstotal = SEMparams['nclusterstotal']
#     rnamelist = SEMparams['rnamelist']
#     nregions = SEMparams['nregions']
#     cluster_properties = SEMparams['cluster_properties']
#     cluster_data = SEMparams['cluster_data']
#     network = SEMparams['network']
#     fintrinsic_count = SEMparams['fintrinsic_count']
#     vintrinsic_count = SEMparams['vintrinsic_count']
#     sem_region_list = SEMparams['sem_region_list']
#     nclusterlist = SEMparams['nclusterlist']
#     tsize = SEMparams['tsize']
#     tplist_full = SEMparams['tplist_full']
#     tcdata_centered = SEMparams['tcdata_centered']
#     ctarget = SEMparams['ctarget']
#     csource = SEMparams['csource']
#     fintrinsic_region = SEMparams['fintrinsic_region']
#     Mconn = SEMparams['Mconn']
#     Minput = SEMparams['Minput']
#     timepoint = SEMparams['timepoint']
#     epoch = SEMparams['epoch']
#
#     SEMresults = np.load(SEMresultsname, allow_pickle=True)
#
#     for timepoint in range(tsize):
#         NP = len(SEMresults)
#         for nn in range(NP):
#             Sinput = SEMresults[nn]['Sinput']
#             Sconn = SEMresults[nn]['Sconn']
#             beta_int1 = SEMresults[nn]['beta_int1']
#             Mconn = SEMresults[nn]['Mconn']
#             Minput = SEMresults[nn]['Minput']
#             R2total = SEMresults[nn]['R2total']
#             Mintrinsic = SEMresults[nn]['Mintrinsic']
#             Meigv = SEMresults[nn]['Meigv']
#             betavals = SEMresults[nn]['betavals']
#             fintrinsic1 = SEMresults[nn]['fintrinsic1']
#             clusterlist = SEMresults[nn]['clusterlist']
#
#             nl, tsize_full = np.shape(Mintrinsic)
#             nruns = np.floor(tsize_full / tsize).astype(int)
#             latent_short = np.zeros((nl, tsize))
#             for nnl in range(nl):
#                 temp = np.reshape(Mintrinsic[nnl,:],(nruns,tsize))
#                 latent_short[nnl,:] = np.mean(temp,axis=0)
#
#             nr, tsize_full = np.shape(Sinput)
#             Sinput_short = np.zeros((nr, tsize))
#             for nnr in range(nr):
#                 temp = np.reshape(Sinput[nnr,:],(nruns,tsize))
#                 Sinput_short[nnr,:] = np.mean(temp,axis=0)
#
#             nc, tsize_full = np.shape(Sconn)
#             Sconn_short = np.zeros((nc, tsize))
#             for nnc in range(nc):
#                 temp = np.reshape(Sconn[nnc,:],(nruns,tsize))
#                 Sconn_short[nnc,:] = np.mean(temp,axis=0)
#
#             valueset = np.concatenate((latent_short,Sinput_short,Sconn_short),axis=0)
#             vals = valueset[:,timepoint]
#
#             if nn == 0:
#                 nvals = len(vals)
#                 person_state = np.zeros((nvals,NP))
#
#             person_state[:,nn] = vals
#
#         # find dominant states at the selected timepoint
#         pca = PCA(n_components = 3)
#         pca.fit(person_state.T)
#
#         print(pca.explained_variance_ratio_)
#         components = pca.components_
#
#         if timepoint == 0:
#             main_components = components[:,:,np.newaxis]
#         else:
#             main_components = np.concatenate((main_components,components[:,:,np.newaxis]),axis=2)
#
#     # rnamelist = ['C6RD',  'DRt', 'Hypothalamus','LC', 'NGC',
#     #                'NRM', 'NTS', 'PAG', 'PBN', 'Thalamus']
#
#     # plot results
#     latents_main = main_components[0,:3,:]
#     Sinput_main = main_components[0,3:13,:]
#     Sconn_main = main_components[0,-33:,:]
#
#     color = np.eye(3)
#     color = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
#
#     # latents
#     wnum = 100
#     plt.close(wnum)
#     fig = plt.figure(wnum)
#     for nn in range(3):
#         plt.plot(range(tsize), latents_main[nn, :], '-x',color = color[nn,:])
#         name = 'latent{}'.format(nn)
#
#     # Sinput
#     wnum = 110
#     for nn in range(9):
#         if nn%3 == 0:
#             plt.close(wnum)
#             fig = plt.figure(wnum)
#             wnum += 1
#
#         nn2 = nn%3
#         plt.plot(range(tsize), Sinput_main[nn, :], '-x',color = color[nn2,:])
#         name = '{}'.format(rnamelist[nn])
#
#
#     # Sconn
#     nsources = [3,3,3,3,4,4,2,3,3,2]
#     nlimit = np.array([0] + list(np.cumsum(nsources)))
#     wnum = 120
#     for nn in range(30):
#         c = np.where(nn >= nlimit)[0]
#         ss = nn-nlimit[c[-1]]
#         if ss == 0:
#             plt.close(wnum)
#             fig = plt.figure(wnum)
#             wnum += 1
#
#         plt.plot(range(tsize), Sconn_main[nn, :], '-x',color = color[ss,:])
#
#         rn1 = beta_list[nn]['pair'][0]
#         rn2 = beta_list[nn]['pair'][1]
#         if rn1 >= nregions:
#             name1 = 'latent{}'.format(rn1-nregions)
#         else:
#             name1 = rnamelist[rn1]
#         name2 = rnamelist[rn2]
#         name = '{}-{}'.format(name1,name2)
#         print(name)
#
#
#
#
#     # look at the results
#     #  Mintrinsic   3 x 200
#     #  fintrinsic1   200,
#
#     nl, tsize_full = np.shape(Mintrinsic)
#     nruns = np.floor(tsize_full/tsize).astype(int)
#     latent_short = np.zeros((nl,tsize))
#     for nn in range(nl):
#         temp = np.reshape(Mintrinsic[nn,:],(nruns,tsize))
#         temp2 = np.mean(temp,axis=0)
#         latent_short[nn,:] = temp2
#
#
#     plt.close(1)
#     fig = plt.figure(1)
#     plt.plot(range(tsize_full),Mintrinsic[0,:],'-xr')
#     plt.plot(range(tsize_full),Mintrinsic[1,:],'-xg')
#
#     plt.close(2)
#     fig = plt.figure(2)
#     plt.plot(range(tsize_full),Mintrinsic[0,:],'-xr')
#     plt.plot(range(tsize_full),Mintrinsic[2,:],'-xb')
#
#     plt.close(3)
#     fig = plt.figure(3)
#     plt.plot(range(tsize_full),Mintrinsic[1,:],'-xg')
#     plt.plot(range(tsize_full),Mintrinsic[2,:],'-xb')
#
#
#     plt.close(4)
#     fig = plt.figure(4)
#     plt.plot(range(tsize),latent_short[0,:],'-xr')
#     plt.plot(range(tsize),latent_short[1,:],'-xg')
#
#     plt.close(5)
#     fig = plt.figure(5)
#     plt.plot(range(tsize),latent_short[0,:],'-xr')
#     plt.plot(range(tsize),latent_short[2,:],'-xb')
#
#     plt.close(6)
#     fig = plt.figure(6)
#     plt.plot(range(tsize),latent_short[1,:],'-xg')
#     plt.plot(range(tsize),latent_short[2,:],'-xb')
#
