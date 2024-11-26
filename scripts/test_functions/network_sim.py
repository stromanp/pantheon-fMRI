# test_CCrecord_display
# sys.path.append(r'C:\Users\Stroman\PycharmProjects\pyspinalfmri3\venv')

# test_connectivity_with_derivatives.py

import numpy as np
import matplotlib.pyplot as plt
import py2ndlevelanalysis
import copy
import pyclustering
import pydisplay
import time
import pysem
from sklearn.cluster import KMeans
import scipy.stats as stats
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
from mpl_toolkits import mplot3d

starttime = time.ctime()
# main function
outputdir = r'D:/threat_safety_python/SEMresults'
SEMresultsname = os.path.join(outputdir,'SEMresults_newmethod_1.npy')
networkfile = r'D:/threat_safety_python/network_model_with_intrinsics.xlsx'
network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkfile)

# load data
DBname = r'D:/threat_safety_python/threat_safety_database.xlsx'
xls = pd.ExcelFile(DBname, engine='openpyxl')
df1 = pd.read_excel(xls, 'paradigm1_BOLD')
del df1['Unnamed: 0']  # get rid of the unwanted header column
fields = list(df1.keys())
paradigm = df1['paradigms_BOLD']
timevals = df1['time']
paradigm_centered = paradigm - np.mean(paradigm)
dparadigm = np.zeros(len(paradigm))
dparadigm[1:] = np.diff(paradigm_centered)

region_data_name1 = r'D:/threat_safety_python/SEMresults/threat_safety_regiondata_allthreat55.npy'
clustername = r'D:/threat_safety_python/SEMresults/threat_safety_clusterdata.npy'

region_data1 = np.load(region_data_name1, allow_pickle=True).flat[0]
region_properties = region_data1['region_properties']

cluster_data = np.load(clustername, allow_pickle=True).flat[0]
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
tplist_full = []
dtplist_full = []
et1 = 0
et2 = tsize
dtsize = tsize-1  # for using deriviation of tc wrt time
tplist1 = []
dtplist1 = []
nclusterstotal,tsizetotal = np.shape(tcdata)
tcdata_centered = copy.deepcopy(tcdata)
dtcdata_centered = np.zeros((nclusterstotal,nruns_total*tsize))
for nn in range(NP):
    r1 = sum(nruns_per_person[:nn])
    r2 = sum(nruns_per_person[:(nn + 1)])
    tp = []  # initialize list
    dtp = []  # initialize list
    tpoints = []
    dtpoints = []
    for ee2 in range(r1, r2):
        tp = list(range((ee2 * tsize), (ee2 * tsize + tsize)))
        # dtp = list(range((ee2 * dtsize), (ee2 * dtsize + dtsize)))
        tpoints = tpoints + tp  # concatenate lists
        # dtpoints = dtpoints + dtp  # concatenate lists
        temp = np.mean(tcdata[:, tp],axis=1)
        temp_mean = np.repeat(temp[:, np.newaxis], tsize, axis=1)
        tcdata_centered[:, tp] = tcdata[:, tp] - temp_mean   # center each epoch, in each person
        dtcdata_centered[:, tp[1:]] = np.diff(tcdata[:, tp])   # 1st dervitive of timecourse wrt time (estimated)
    tplist1.append({'tp': tpoints})
    # dtplist1.append({'tp': dtpoints})
tplist_full.append(tplist1)
# dtplist_full.append(dtplist1)


# setup matrices for solving network equation
Nintrinsic = 2
nregions = len(rnamelist)
m = Nintrinsic + nregions
Minput = np.zeros((m,m))
Moutput = np.zeros((m,m))
for nn in range(nregions):
    a = network[nn]['targetnum']
    b = network[nn]['sourcenums']
    Minput[a,b] = 1
    Moutput[a,b] = 1

for nn in range(Nintrinsic):
    Minput[nn+nregions,nn+nregions] = 1
    Moutput[nn+nregions,nn+nregions] = 1

# keep a record of matrix indices that need to be estimated---------------
ctarget0,csource0 = np.where(Minput > 0)
exclude = np.where( (ctarget0 >= nregions) | (csource0 >= nregions))[0]   # don't scale intrinsics at the output
keep = np.setdiff1d(list(range(len(ctarget0))),exclude)
ctarget = ctarget0[keep]
csource = csource0[keep]

nbeta = len(ctarget)  # the number of beta values to be estimated

timepoint = 0
SEMresults = []
nperson = 0
# for nperson in range(NP):    # select one person (for testing)
#     print('starting person {} at {}'.format(nperson,time.ctime()))

tp = tplist_full[timepoint][nperson]['tp']
tsize_total = len(tp)

# get tc data for each region/cluster
clusterlist = [4,9,11,15,23,26,31,36,43,45]
Sinput = []
for cval in clusterlist:
    tc1 = tcdata_centered[cval, tp]
    Sinput.append(tc1)
# Sinput is size:  nregions x tsize_total

beta_int1 = 0.1    # start the magnitude of intrinsic1 at a small value
intrinsic1 = np.array(list(paradigm_centered) * nruns_per_person[nperson])
intrinsic2 = np.random.randn(tsize_total)    # initialize unknown intrinsic with random values

#  Sinput = Minput @ Moutput @ Soutput    --> solve for Soutput (including intrinsic2)
Soutput = np.zeros((nregions,tsize_total))   # initialize Soutput with small random values
betavals = 0.2*np.random.randn(nbeta)  # initialize beta values with random values

# Soutput_full = Moutput @ Soutput_full    # Moutput is a fixed matrix desribing the network
# Sinput_full = Minput @ Soutput_full

# perturb the system
scale1 = 1.0
scale2 = 1.0
Sinput_full = np.concatenate((Sinput, scale1 * intrinsic1[np.newaxis, :], scale2*intrinsic2[np.newaxis, :]), axis=0)
Soutput_full = np.concatenate((Soutput, scale1 * intrinsic1[np.newaxis, :], scale2*intrinsic2[np.newaxis, :]), axis=0)
Moutput[ctarget, csource] = betavals

Soutput_working = copy.deepcopy(Soutput_full)
plotnum = 30
plotcount = 0
plt.close(plotnum)
fig = plt.figure(plotnum)
for aa in range(10):
    Soutput_working = Moutput @ Soutput_working
    if np.mod(aa,2) == 0:
        plotcount += 1
        plt.subplot(3,2,plotcount)
        plt.plot(Soutput_working[0,:],'-xb')
        plt.plot(Soutput_working[5,:],'-r')
        plt.plot(Soutput_working[1,:],'-g')


# simulate inputs and outputs of each region, for given values of intrinsic1 and intrinsic2
# use an iterative approach to a limit
#
#
#     columns = [name[:3] +' in' for name in rnamelist]
#     columns += ['int1 in', 'int2 in']
#     rows = [name[:3] for name in rnamelist]
#     rows += ['int1', 'int2']
#
#     df = pd.DataFrame(Moutput,columns = columns, index = rows)
#     pd.set_option('display.max_rows', None)
#     pd.set_option('display.max_columns', None)
#     pd.set_option('display.width', None)
#     pd.set_option('display.max_colwidth', None)
#
#
#     pd.options.display.float_format = '{:.2f}'.format
#     print(df)
