import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import copy
import pysapm
import multiprocessing as mp
import time
import os

outputdir = r'E:\FM2021data'
SAPMresultsname = r'E:\FM2021data\HCstim_2022332124_V5_results.npy'
SAPMparametersname = r'E:\FM2021data\HCstim_2022332124_V5_params.npy'
networkfile = r'E:\FM2021data\network_model_June2023_SAPM.xlsx'
regiondataname = r'E:\FM2021data\HCstim_equal_region_data_Jan23_2024_V5.npy'
clusterdataname = r'E:/FM2021data\allstim_equal_cluster_def_Jan22_2024_V3.npy'
samplesplit = [1,0]
samplestart=0
initial_clusters=[]
timepoint='all'
epoch='all'
betascale=0.1
Lweight = 1.0


#-----------------start running part of function---------------------------
overall_start_time_text = time.ctime()
overall_start_time = time.time()

if not os.path.exists(outputdir): os.mkdir(outputdir)

# load some data, setup some parameters...
network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = pysapm.load_network_model_w_intrinsics(
    networkfile)
ncluster_list = np.array([nclusterlist[x]['nclusters'] for x in range(len(nclusterlist))])
cluster_name = [nclusterlist[x]['name'] for x in range(len(nclusterlist))]
not_latent = [x for x in range(len(cluster_name)) if 'intrinsic' not in cluster_name[x]]
ncluster_list = ncluster_list[not_latent]
full_rnum_base = [np.sum(ncluster_list[:x]) for x in range(len(ncluster_list))]
namelist = [cluster_name[x] for x in not_latent]
namelist += ['Rtotal']
namelist += ['R ' + cluster_name[x] for x in not_latent]

nregions = len(ncluster_list)

print('best cluster search:  preparing data ...')
pysapm.prep_data_sem_physio_model_SO_V2(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch,
                                 fullgroup=False, normalizevar=True, filter_tcdata=False)

print('best cluster search:  loading parameters ...')
SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
tcdata = SAPMparams['tcdata_centered']  # data for all regions/clusters concatenated along time dimension for all runs
# need to get principal components for each region to model the clusters as a continuum

nclusters_total, tsize_total = np.shape(tcdata)

maxiter = 50
subsample = [samplesplit, samplestart]  # [2,0] use every 2nd data set, starting with samplestart

full_rnum_base = np.array([np.sum(ncluster_list[:x]) for x in range(len(ncluster_list))]).astype(int)
initial_clusters = np.array(initial_clusters)
if (initial_clusters < 0).any():
    fixed_clusters = np.where(initial_clusters >= 0)[0]
else:
    fixed_clusters = []

if (len(initial_clusters) != nregions):
    temp_clusters = -1 * np.ones(nregions)
    temp_clusters[:len(initial_clusters)] = copy.deepcopy(initial_clusters)  # pad list with -1
    initial_clusters = copy.deepcopy(temp_clusters)

cluster_numbers = np.zeros(nregions)
for nn in range(nregions):
    if initial_clusters[nn] < 0:
        cnum = np.random.choice(range(ncluster_list[nn]))
        cluster_numbers[nn] = copy.deepcopy(cnum)
    else:
        cluster_numbers[nn] = copy.deepcopy(initial_clusters[nn])
cluster_numbers = np.array(cluster_numbers).astype(int)

print('starting clusters: {}'.format(cluster_numbers))

lastgood_clusters = copy.deepcopy(cluster_numbers)

# gradient descent to find best cluster combination
iter = 0
costrecord = []
print('starting step descent search of clusters at {}'.format(time.ctime()))
converging = True

nitermax = 100
nitermax_stage1 = 30
nitermax_stage2 = 50
nsteps_stage1 = 30

random_region_order = list(range(nregions))
np.random.shuffle(random_region_order)
nnn = random_region_order[0]

cost_values = np.zeros(ncluster_list[nnn])
print('testing region {}'.format(nnn))

parameters = []
for ccc in range(ncluster_list[nnn]):
    test_clusters = copy.deepcopy(cluster_numbers)
    if test_clusters[nnn] != ccc:  # no change in cluster number from last run
        test_clusters[nnn] = ccc

        params = {'cluster_numbers': test_clusters + full_rnum_base,
                  'fintrinsic_base': fintrinsic_base, 'SAPMresultsname': SAPMresultsname,
                  'SAPMparametersname': SAPMparametersname, 'fixed_beta_vals': [],
                  'betascale': betascale, 'Lweight': Lweight, 'normalizevar': False,
                  'nitermax_stage3': nitermax, 'nitermax_stage2': nitermax_stage2,
                  'nitermax_stage1': nitermax_stage1, 'nsteps_stage1': nsteps_stage1,
                  'nsteps_stage2': 2, 'converging_slope_limit': [1e-2, 1e-4, 1e-6]}
        parameters.append(params)

nprocessors = ncluster_list[nnn] - 1
pool = mp.Pool(nprocessors)
searchresults = pool.map(pysapm.SAPM_parallel_runs, parameters)
pool.close()

results_record = []
for nn in range(len(searchresults)):
    R2list = searchresults[nn]['R2list']
    R2list2 = searchresults[nn]['R2list2']
    cost_value = searchresults[nn]['cost_value']

    entry = {'R2list': R2list, 'R2list2': R2list2, 'region': nnn, 'cluster': ccc}
    results_record.append(entry)
    print('  using cluster {}  total of (1-R2 avg) for the group is {:.3f}'.format(ccc, cost_values[ccc]))





