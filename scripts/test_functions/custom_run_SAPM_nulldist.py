# run SAPM for a large number of simulated null data sets, using the most recent SAPM parameters

import numpy as np
import load_templates
import pysapm
import os
import pandas as pd
import copy

nsamples = 10000

basedir = r'C:\Users\Stroman\PycharmProjects\pantheon\venv'
settingsfile = os.path.join(basedir, '../base_settings_file.npy')

# define the clusters and load the data
settings = np.load(settingsfile, allow_pickle=True).flat[0]
DBname = settings['DBname']
DBnum = settings['DBnum']
# SAPMprefix = settings['SAPMprefix']
networkmodel = settings['networkmodel']
SAPMclustername = settings['SAPMclustername']
SAPMregionname = settings['SAPMregionname']
SAPMparamsname = settings['SAPMparamsname']
SAPMresultsname = settings['SAPMresultsname']
SAPMresultsdir = settings['SAPMresultsdir']
# SAPMsavetag = settings['SAPMsavetag']
SAPMtimepoint = settings['SAPMtimepoint']
SAPMepoch = settings['SAPMepoch']
SAPMcnums = settings['SAPMcnums']
# SEMresumerun = settings['SEMresumerun']
# SAPMkeyinfo1.config(text=' ', fg='gray')

nullresultsname = 'biasdata_{}_results.npy'.format(nsamples)
nullparamsname = 'biasdata_{}_params.npy'.format(nsamples)
SAPMresultsname = os.path.join(SAPMresultsdir, nullresultsname)
SAPMparamsname = os.path.join(SAPMresultsdir, nullparamsname)
xls = pd.ExcelFile(DBname, engine='openpyxl')
df1 = pd.read_excel(xls, 'datarecord')

normtemplatename = df1.loc[DBnum[0], 'normtemplatename']
resolution = 1
template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_map = \
    load_templates.load_template_and_masks(normtemplatename, resolution)

region_data = np.load(SAPMregionname, allow_pickle=True).flat[0]
region_properties = region_data['region_properties']

cluster_data = np.load(SAPMclustername, allow_pickle=True).flat[0]
cluster_properties = cluster_data['cluster_properties']

# excelfilename = os.path.join(SAPMresultsdir, SAPMsavetag + '_results.xlsx')

print('running SAPM with null data ...')

SAPMresultsname = os.path.join(SAPMresultsdir, SAPMresultsname)
SAPMparamsname = os.path.join(SAPMresultsdir, SAPMparamsname)

# pysapm.SAPMrun(SAPMcnums, SAPMregionname, SAPMclustername,
#                SAPMresultsname, SAPMparamsname, networkmodel, DBname, SAPMtimepoint, SAPMepoch, reload_existing=False)
#


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
network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count = pysapm.load_network_model_w_intrinsics(networkmodel)
ncluster_list = np.array([nclusterlist[x]['nclusters'] for x in range(len(nclusterlist))])
cluster_name = [nclusterlist[x]['name'] for x in range(len(nclusterlist))]
not_latent = [x for x in range(len(cluster_name)) if 'intrinsic' not in cluster_name[x]]
ncluster_list = ncluster_list[not_latent]
full_rnum_base = [np.sum(ncluster_list[:x]) for x in range(len(ncluster_list))]
namelist = [cluster_name[x] for x in not_latent]
namelist += ['Rtotal']
namelist += ['R ' + cluster_name[x] for x in not_latent]

# full_rnum_base =  np.array([0,5,10,15,20,25,30,35,40,45])
#
# namelist = ['C6RD',  'DRt', 'Hypothalamus','LC', 'NGC', 'NRM', 'NTS', 'PAG', 'PBN', 'Thalamus',
#         'Rtotal', 'R C6RD',  'R DRt', 'R Hyp','R LC', 'R NGC', 'R NRM', 'R NTS', 'R PAG',
#         'R PBN', 'R Thal']

# starting values
cnums_original = copy.deepcopy(SAPMcnums)
excelsheetname = 'clusters'

# run the analysis with SAPM
clusterlist = np.array(SAPMcnums) + full_rnum_base
pysapm.prep_null_data_sem_physio_model(nsamples, networkmodel, SAPMregionname, SAPMclustername, SAPMparamsname, timepoint = 'all', epoch = 'all', fullgroup = False, addglobalbias = True)
output = pysapm.sem_physio_model(clusterlist, paradigm_centered, SAPMresultsname, SAPMparamsname)

SAPMresults = np.load(output, allow_pickle=True)
NP = len(SAPMresults)
R2list =np.zeros(len(SAPMresults))
for nperson in range(NP):
    R2list[nperson] = SAPMresults[nperson]['R2total']
print('SAPM parameters computed for {} data sets'.format(NP))
print('R2 values averaged {:.3f} {} {:.3f}'.format(np.mean(R2list),chr(177),np.std(R2list)))
print('R2 values ranged from {:.3f} to {:.3f}'.format(np.min(R2list),np.max(R2list)))

