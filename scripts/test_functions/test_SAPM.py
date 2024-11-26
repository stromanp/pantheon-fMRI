# sys.path.append(r'C:\Users\Stroman\PycharmProjects\pantheon\venv')

# test_SAPM with one person's data
import numpy as np
import load_templates
import os
import pandas as pd
import pysapm
import pysem
import time
import copy
import matplotlib.pyplot as plt

import importlib
importlib.reload(pysapm)


window_number = 104
nperson = 7   # worst results of set
# nperson = 40   # another example
betascale = 0.01

settingsfile = r'C:\Users\Stroman\PycharmProjects\pantheon\venv\base_settings_file.npy'

settings = np.load(settingsfile, allow_pickle=True).flat[0]
DBname = r'E:/graded_pain_database_May2022.xlsx'
DBnum = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,
         29,30,31,32,33,34,35,36,37,38,0,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,
         55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,85,
         86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,
         109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,
         129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,
         149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,
         169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,
         188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,209,210,211,212,
         213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,
         232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,
         252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,
         271,272,273,274,275,276,277]
# SAPMprefix = settings['SAPMprefix']
networkfile = r'E:/SAPMresults_Dec2022/network_model_v2_Dec2022_w_3intrinsics.xlsx'
clusterdataname = r'E:/SAPMresults_Dec2022\\Pain_cluster_def.npy'
regiondataname = r'E:/SAPMresults_Dec2022\allpainconditions_regiondata2.npy'
SAPMresultsname = r'E:/SAPMresults_Dec2022\High_0203023213_results.npy'
SAPMparametersname = r'E:/SAPMresults_Dec2022\High_0203023213_params.npy'
SAPMresultsdir = r'E:/SAPMresults_Dec2022'
covariates_file = r'E:/allpain_covariates.npy'
timepoint = 'all'
epoch = 'all'
cnums = [0, 2, 0, 3, 0, 2, 3, 2, 1, 3]

beta_initial_record = r'E:/SAPMresults_Dec2022\\beta_initial.npy'
use_saved_beta_initial_values = False

SAPMresultsname = os.path.join(SAPMresultsdir, SAPMresultsname)
xls = pd.ExcelFile(DBname, engine='openpyxl')
df1 = pd.read_excel(xls, 'datarecord')

normtemplatename = df1.loc[DBnum[0], 'normtemplatename']
resolution = 1
template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_map = \
    load_templates.load_template_and_masks(normtemplatename, resolution)

region_data = np.load(regiondataname, allow_pickle=True).flat[0]
region_properties = region_data['region_properties']

cluster_data = np.load(clusterdataname, allow_pickle=True).flat[0]
cluster_properties = cluster_data['cluster_properties']

# excelfilename = os.path.join(SAPMresultsdir, SAPMsavetag + '_results.xlsx')

print('running SAPM with selected clusters ...')

SAPMresultsname = os.path.join(SAPMresultsdir, SAPMresultsname)
SAPMparametersname = os.path.join(SAPMresultsdir, SAPMparametersname)

# pysapm.SAPMrun(cnums, regiondataname, clusterdataname,
#                SAPMresultsname, SAPMparametersname, networkfile, DBname, timepoint, epoch,
#                reload_existing=False)

# from pysapm:
# def SAPMrun(cnums, regiondataname, clusterdataname, SAPMresultsname, SAPMparametersname, networkfile, DBname, timepoint, epoch, reload_existing = False):
    # load paradigm data--------------------------------------------------------------------
xls = pd.ExcelFile(DBname, engine='openpyxl')
df1 = pd.read_excel(xls, 'paradigm1_BOLD')
del df1['Unnamed: 0']  # get rid of the unwanted header column
fields = list(df1.keys())
paradigm = df1['paradigms_BOLD']
timevals = df1['time']
paradigm_centered = paradigm - np.mean(paradigm)
dparadigm = np.zeros(len(paradigm))
dparadigm[1:] = np.diff(paradigm_centered)

# load some data, setup some parameters...
network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count = pysapm.load_network_model_w_intrinsics(networkfile)
ncluster_list = np.array([nclusterlist[x]['nclusters'] for x in range(len(nclusterlist))])
cluster_name = [nclusterlist[x]['name'] for x in range(len(nclusterlist))]
not_latent = [x for x in range(len(cluster_name)) if 'intrinsic' not in cluster_name[x]]
ncluster_list = ncluster_list[not_latent]
full_rnum_base = [np.sum(ncluster_list[:x]) for x in range(len(ncluster_list))]
namelist = [cluster_name[x] for x in not_latent]
namelist += ['Rtotal']
namelist += ['R ' + cluster_name[x] for x in not_latent]

# starting values
cnums_original = copy.deepcopy(cnums)
excelsheetname = 'clusters'

# run the analysis with SAPM
clusterlist = np.array(cnums) + full_rnum_base
# prep_data_sem_physio_model(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch)



# prep_data_sem_physio_model---------------------------------------------------------------------------------
# def prep_data_sem_physio_model(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint = 'all', epoch = 'all', fullgroup = False):
timepoint = 'all'
epoch = 'all'
fullgroup = False

outputdir, f = os.path.split(SAPMparametersname)
network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count = pysapm.load_network_model_w_intrinsics(networkfile)

fintrinsic_region = []
if fintrinsic_count > 0:  # find which regions have fixed intrinsic input
    for nn in range(len(network)):
        sources = network[nn]['sources']
        if 'fintrinsic1' in sources:
            fintrinsic_region = network[nn]['targetnum']  # only one region should have this input

region_data1 = np.load(regiondataname, allow_pickle=True).flat[0]
region_properties = region_data1['region_properties']

cluster_data = np.load(clusterdataname, allow_pickle=True).flat[0]
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
if timepoint == 'all':
    epoch = tsize
    timepoint = np.floor(tsize/2)

tplist_full = []
if epoch >= tsize:
    et1 = 0
    et2 = tsize
else:
    if np.floor(epoch/2).astype(int) == np.ceil(epoch/2).astype(int):   # even numbered epoch
        et1 = (timepoint - np.floor(epoch / 2)).astype(int)
        et2 = (timepoint + np.floor(epoch / 2)).astype(int)
    else:
        et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
        et2 = (timepoint + np.floor(epoch / 2)).astype(int)
if et1 < 0: et1 = 0
if et2 > tsize: et2 = tsize
epoch = et2-et1

dtsize = tsize - 1  # for using deriviation of tc wrt time
tplist1 = []
nclusterstotal, tsizetotal = np.shape(tcdata)
tcdata_centered = copy.deepcopy(tcdata)
for nn in range(NP):
    r1 = sum(nruns_per_person[:nn])
    r2 = sum(nruns_per_person[:(nn + 1)])
    tp = []  # initialize list
    tpoints = []
    for ee2 in range(r1, r2):
        # tp = list(range((ee2 * tsize), (ee2 * tsize + tsize)))
        tp = list(range((ee2*tsize+et1),(ee2*tsize+et2)))
        tpoints += tp  # concatenate lists
        temp = np.mean(tcdata[:, tp], axis=1)
        temp_mean = np.repeat(temp[:, np.newaxis], epoch, axis=1)
        tcdata_centered[:, tp] = tcdata[:, tp] - temp_mean  # center each epoch, in each person
    tplist1.append({'tp': tpoints})
tplist_full.append(tplist1)

if fullgroup:
    # special case to fit the full group together
    # treat the whole group like one person
    tpgroup_full = []
    tpgroup = []
    tp = []
    for nn in range(NP):
        tp += tplist_full[0][nn]['tp']   # concatenate timepoint lists
    tpgroup.append({'tp': tp})
    tpgroup_full.append(tpgroup)
    tplist_full = copy.deepcopy(tpgroup_full)
    nruns_per_person = [np.sum(nruns_per_person)]


Nintrinsic = fintrinsic_count + vintrinsic_count
nregions = len(rnamelist)

beta_list = []
nbeta = 0
targetnumlist = []
beta_id = []
sourcelist = []
for nn in range(len(network)):
    target = network[nn]['targetnum']
    sources = network[nn]['sourcenums']
    targetnumlist += [target]
    for mm in range(len(sources)):
        source = sources[mm]
        sourcelist += [source]
        betaname = '{}_{}'.format(source, target)
        entry = {'name': betaname, 'number': nbeta, 'pair': [source, target]}
        beta_list.append(entry)
        beta_id += [1000 * source + target]
        nbeta += 1

ncon = nbeta - Nintrinsic

# reorder to put intrinsic inputs at the end-------------
beta_list2 = []
beta_id2 = []
x = np.where(np.array(sourcelist) < nregions)[0]
for xx in x:
    beta_list2.append(beta_list[xx])
    beta_id2 += [beta_id[xx]]
for sn in range(nregions, nregions + Nintrinsic):
    x = np.where(np.array(sourcelist) == sn)[0]
    for xx in x:
        beta_list2.append(beta_list[xx])
        beta_id2 += [beta_id[xx]]

for nn in range(len(beta_list2)):
    beta_list2[nn]['number'] = nn

beta_list = beta_list2
beta_id = beta_id2

beta_pair = []
Mconn = np.zeros((nbeta, nbeta))
count = 0
for nn in range(len(network)):
    target = network[nn]['targetnum']
    sources = network[nn]['sourcenums']
    for mm in range(len(sources)):
        source = sources[mm]
        conn1 = beta_id.index(source * 1000 + target)
        if source >= nregions:  # intrinsic input
            conn2 = conn1
            Mconn[conn1, conn2] = 1  # set the intrinsic beta values
        else:
            x = targetnumlist.index(source)
            source_sources = network[x]['sourcenums']
            for nn in range(len(source_sources)):
                ss1 = source_sources[nn]
                conn2 = beta_id.index(ss1 * 1000 + source)
                beta_pair.append([conn1, conn2])
                count += 1
                Mconn[conn1, conn2] = count

# prep to index Mconn for updating beta values
beta_pair = np.array(beta_pair)
ctarget = beta_pair[:, 0]
csource = beta_pair[:, 1]

latent_flag = np.zeros(len(ctarget))
found_latent_list = []
for nn in range(len(ctarget)):
    if csource[nn] >= ncon:
        if not csource[nn] in found_latent_list:
            latent_flag[nn] = 1
            found_latent_list += [csource[nn]]

# setup Minput matrix--------------------------------------------------------------
# Sconn = Mconn @ Sconn    # propagate the intrinsic inputs through the network
# Sinput = Minput @ Mconn
Minput = np.zeros((nregions, nbeta))  # mixing of connections to model the inputs to each region
betanamelist = [beta_list[a]['name'] for a in range(nbeta)]
for nn in range(len(network)):
    target = network[nn]['targetnum']
    sources = network[nn]['sourcenums']
    for mm in range(len(sources)):
        source = sources[mm]
        betaname = '{}_{}'.format(source, target)
        x = betanamelist.index(betaname)
        Minput[target, x] = 1

# save parameters for looking at results later
SAPMparams = {'betanamelist': betanamelist, 'beta_list': beta_list, 'nruns_per_person': nruns_per_person,
             'nclusterstotal': nclusterstotal, 'rnamelist': rnamelist, 'nregions': nregions,
             'cluster_properties': cluster_properties, 'cluster_data': cluster_data,
             'network': network, 'fintrinsic_count': fintrinsic_count, 'vintrinsic_count': vintrinsic_count,
             'fintrinsic_region':fintrinsic_region, 'sem_region_list': sem_region_list,
             'nclusterlist': nclusterlist, 'tsize': tsize, 'tplist_full': tplist_full,
             'tcdata_centered': tcdata_centered, 'ctarget':ctarget ,'csource':csource,
             'Mconn':Mconn, 'Minput':Minput, 'timepoint':timepoint, 'epoch':epoch, 'latent_flag':latent_flag}
# print('saving SAPM parameters to file: {}'.format(SAPMparametersname))
# np.save(SAPMparametersname, SAPMparams)
# end of  prep_data_sem_physio_model -------------------------------------------------------------------------------------


SAPMresults = []
#-----------for one person ----------------------------

# output = sem_physio_model(clusterlist, paradigm_centered, SAPMresultsname, SAPMparametersname)
# sem_physio_model -------------------------------------------------------------------------
# def sem_physio_model(clusterlist, fintrinsic_base, SAPMresultsname, SAPMparametersname, fixed_beta_vals = [], verbose = True):
fintrinsic_base = copy.deepcopy(paradigm_centered)
fixed_beta_vals = []
verbose = True

starttime = time.ctime()

# initialize gradient-descent parameters--------------------------------------------------------------
initial_alpha = 0.01
initial_Lweight = 0.0
initial_dval = 0.01
# betascale = 0.1

# SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
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
# repeat the process for each participant-----------------------------------------------------------------
betalimit = 3.0
epochnum = 0
beta_init_record = []
# for nperson in range(NP):
# run for person number "nperson"

print('starting person {} at {}'.format(nperson,time.ctime()))
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
# Sinput_scalefactor = np.zeros(len(clusterlist))
for nc, cval in enumerate(clusterlist):
    tc1 = tcdata_centered[cval, tp]
    # Sinput_scalefactor[nc] = np.std(tc1)
    # tc1 /= np.std(tc1)
    Sinput.append(tc1)
Sinput = np.array(Sinput)
# Sinput is size:  nregions x tsize_total

# Mscale = np.zeros((len(clusterlist),len(clusterlist)))
# for xx in range(10): Mscale[xx, xx] = Sinput_scalefactor[xx]

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


#-------------------------------------------------------------
generate_synthetic_data = False
if generate_synthetic_data:
    # generate synthetic data-------------------------------------
    nr,tsizefull = np.shape(Sinput)
    betavals_synth = 0.5*np.random.randn(len(csource))
    beta_int1_synth = 0.1
    Nintrinsic = fintrinsic_count + vintrinsic_count
    Mintrinsic_synth = np.zeros((Nintrinsic,tsizefull))
    if fintrinsic_count > 0:
        Mintrinsic_synth[0,:] = beta_int1_synth*fintrinsic1
        Mintrinsic_synth[1:,:] = 0.5*np.random.randn(vintrinsic_count,tsizefull)
    else:
        Mintrinsic_synth = 0.5*np.random.randn(vintrinsic_count,tsizefull)

    nruns = np.floor(tsizefull/tsize).astype(int)
    for nn in range(nruns):
        t1 = nn*tsize
        t2 = (nn+1)*tsize
        for aa in range(Nintrinsic):
            Mintrinsic_synth[aa,t1:t2] -= np.mean(Mintrinsic_synth[aa,t1:t2])

    Mconn_synth = np.zeros(np.shape(Mconn))
    Mconn_synth[ctarget,csource] = copy.deepcopy(betavals_synth)
    for aa in range(Nintrinsic):
        Mconn_synth[-aa-1,-aa-1] = 1.0

    e, v = np.linalg.eig(Mconn_synth)  # Mconn is nbeta x nbeta   where nbeta = ncon + Nintrinsic
    Meigv = np.real(v[:, -Nintrinsic:])
    # scale to make the term corresponding to each intrinsic = 1
    for aa in range(Nintrinsic):
        Meigv[:, aa] = Meigv[:, aa] / Meigv[(-Nintrinsic + aa), aa]

    Sconn_synth = Meigv @ Mintrinsic_synth
    Sinput_synth = Minput @ Sconn_synth
    Meigv_synth = copy.deepcopy(Meigv)

    # initialize values for testing
    Mconn = copy.deepcopy(Mconn_synth)
    Sconn = copy.deepcopy(Sconn_synth)
    Sinput = copy.deepcopy(Sinput_synth)

# for nn in range(nr):
#     plt.close(10+nn)
#     fig = plt.figure(10+nn)
#     plt.plot(range(tsizefull),Sinput_synth[nn,:])
#-------------------------------------------------------------
#-------------------------------------------------------------

lastgood_beta_int1 = copy.deepcopy(beta_int1)

# initialize beta values-----------------------------------
beta_initial = np.zeros(len(csource))
# beta_initial = betascale*np.random.randn(len(csource))
beta_initial = betascale*np.ones(len(csource))
beta_initial = betascale*np.random.randn(len(csource))

if use_saved_beta_initial_values == True:
    beta_initial_saved_values = np.load(beta_initial_record,allow_pickle=True).flat[0]
    beta_initial =beta_initial_saved_values['beta_initial']

# limit the beta values related to intrinsic inputs to positive values
# for aa in range(len(beta_initial)):
#     if latent_flag[aa] > 0:
#         beta_initial[aa] = 1.0

beta_init_record.append({'beta_initial':beta_initial})

# initalize Sconn
betavals = copy.deepcopy(beta_initial) # initialize beta values at zero
lastgood_betavals = copy.deepcopy(betavals)

results_record = []
ssqd_record = []

alpha = initial_alpha
alphalist = initial_alpha*np.ones(len(betavals))
alphabint = copy.deepcopy(initial_alpha)
alphamax = copy.deepcopy(initial_alpha)
Lweight = initial_Lweight
dval = initial_dval
nitermax = 200
alpha_limit = 1.0e-4

Mconn[ctarget,csource] = copy.deepcopy(betavals)

# # starting point for optimizing intrinsics with given betavals----------------------------------------------------
# fit, Sconn_full = network_eigenvalue_method(Sconn_full, Minput, Mconn, ncon)

fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
ssqd = pysapm.sapm_error_function(Sinput, fit, Lweight, betavals, beta_int1, Mintrinsic)
ssqd_starting = copy.deepcopy(ssqd)
ssqd_record += [ssqd]
ssqd_old = copy.deepcopy(ssqd)

iter = 0
# vintrinsics_record = []
converging = True
dssq_record = np.ones(3)
dssq_count = 0
sequence_count = 0
betaval_record = betavals[:,np.newaxis]
while alphamax > alpha_limit and iter < nitermax and converging:
    iter += 1

    betavals, beta_int1, fit, updatebflag, updatebintflag, dssq_db, dssq_dbeta1, ssqd, alphalist, alphabint = \
        pysapm.update_betavals_sequentially(Sinput, Minput, Mconn, betavals, ctarget, csource, dval,
                                     fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight,
                                     alphalist, alphabint)

    ssqd_record += [ssqd]

    err_total = Sinput - fit
    Smean = np.mean(Sinput)
    errmean = np.mean(err_total)
    # R2total = 1 - np.sum((err_total - errmean) ** 2) / np.sum((Sinput - Smean) ** 2)

    # R2list = [1-np.sum((Sinput[x,:]-fit[x,:])**2)/np.sum(Sinput[x,:]**2) for x in range(nregions)]
    R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
    R2avg = np.mean(R2list)
    R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)

    # Sinput_sim, Soutput_sim = network_sim(Sinput_full, Soutput_full, Minput, Moutput)
    results_record.append({'Sinput': Sinput, 'fit': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})
    atemp = np.append(alphalist, alphabint)
    alphamax = np.max(atemp)
    alphalist[alphalist < alpha_limit] = alpha_limit
    if alphabint < alpha_limit:  alphabint = copy.deepcopy(alpha_limit)

    ssqchange = ssqd - ssqd_old
    if np.abs(ssqchange) < 1e-5: converging = False

    print('SAPM2  {} beta vals:  iter {} alpha {:.3e}  ssqd {:.2f} change {:.3f}  percent {:.1f}  R2 avg {:.3f}  R2 total {:.3f}'.format(nperson,
            iter, np.mean(alphalist), ssqd, ssqchange, 100. * ssqd / ssqd_starting, R2avg, R2total))
    ssqd_old = copy.deepcopy(ssqd)
    # now repeat it ...

# fit the results now to determine output signaling from each region
Mconn[ctarget, csource] = betavals
fit, Mintrinsic, Meigv, err = pysapm.network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)

for nn in range(nr):
    plt.close(10+nn)
    fig = plt.figure(10+nn)
    plt.plot(range(tsizefull),Sinput_synth[nn,:],'-x')
    plt.plot(range(tsizefull),Sinput[nn,:],'-')


# check on Sinput_scalefactor

# Sconn = Meigv @ Mintrinsic    # signalling over each connection
#
# entry = {'Sinput':Sinput, 'Sconn':Sconn, 'beta_int1':beta_int1, 'Mconn':Mconn, 'Minput':Minput,
#          'R2total':R2total, 'Mintrinsic':Mintrinsic,
#          'Meigv':Meigv, 'betavals':betavals, 'fintrinsic1':fintrinsic1, 'clusterlist':clusterlist,
#          'fintrinsic_base':fintrinsic_base}
# sim_result.append(copy.deepcopy(entry))

stoptime = time.ctime()

# np.save(SAPMresultsname, SAPMresults)
print('finished SAPM at {}'.format(time.ctime()))
print('     started at {}'.format(starttime))

# end of sem_physio_model-----------------------------------------------
print('R2 = {:.3f}'.format(R2total))
#
# # display results
# n = 9
# plt.close(50+n)
# fig = plt.figure(50+n)
# for tt in range(len(results_record)):
#     fig.clear()
#     fit = results_record[tt]['fit']
#     Sinput = results_record[tt]['Sinput']
#     rr,tsizefull = np.shape(Sinput)
#     plt.plot(range(tsizefull),Sinput[n,:],'-x')
#     plt.plot(range(tsizefull),fit[n,:],'-')
#     plt.show()
#     plt.pause(0.3)
#
#
#
