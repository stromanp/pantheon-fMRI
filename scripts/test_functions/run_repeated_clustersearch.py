# sys.path.extend([r'C:\Users\Stroman\PycharmProjects\pantheon\.venv'])

import os
import numpy as np
import pandas as pd
import load_templates
import matplotlib
import pysapm
import copy

workingdir = r'E:\'

matplotlib.use('TkAgg')   # explicitly set this - it might help with displaying figures in different environments

trial_number = 0


SAPMrandomclusterstart = 1
SAPMcnums = []
DBname = r'E:\graded_pain_database_May2022.xlsx'
listname = r'E:\allpain_list.npy'
listdata = np.load(listname, allow_pickle = True).flat[0]
DBnum = copy.deepcopy(listdata['dbnum'])

networkmodel = os.path.join(workingdir, 'network_model_June2023_SAPM.xlsx')
SAPMclustername = os.path.join(workingdir, 'threat_safety_clusterdata.npy')
SAPMregionname = os.path.join(workingdir, 'threat_safety_regiondata_allthreat55.npy')
SAPMparamsname = os.path.join(workingdir, 'search_trial{:03d}_g_params.npy'.format(trial_number))
SAPMresultsname = os.path.join(workingdir, 'search_trial{:03d}_g_results.npy'.format(trial_number))
SAPMresultsdir = copy.deepcopy(workingdir)
SAPMbetascale = 0.1
SAPMLweight = 0.01
SAPMgrouplevel = False

levelthreshold = [1e-4, 1e-5, 1e-6]
leveliter = [20, 4, 1]
leveltrials = [100, 200, 500]
run_whole_group = True


# do a gradient-descent search for the best clusters
# settings = np.load(settingsfile, allow_pickle=True).flat[0]
# self.SAPMcnums = settings['SAPMcnums']
# self.DBname = settings['DBname']
# self.DBnum = settings['DBnum']
# self.networkmodel = settings['networkmodel']
# self.SAPMclustername = settings['SAPMclustername']
# self.SAPMregionname = settings['SAPMregionname']
# self.SAPMparamsname = settings['SAPMparamsname']
# self.SAPMresultsname = settings['SAPMresultsname']
# self.SAPMresultsdir = settings['SAPMresultsdir']
# self.SAPMbetascale = settings['SAPMbetascale']
# self.SAPMLweight = settings['SAPMLweight']
# self.SAPMgrouplevel= settings['SAPMgrouplevel']
# self.SAPMsavetag = settings['SAPMsavetag']

pysapm.SAPMupdate_network_info()

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

print('running search for best clusters to use with SAPM ...')

SAPMresultsname = os.path.join(SAPMresultsdir, SAPMresultsname)
SAPMparamsname = os.path.join(SAPMresultsdir, SAPMparamsname)

search_data_file = os.path.join(SAPMresultsdir ,'SAPM_search_parameters.npy')

if SAPMrandomclusterstart > 0:
    clusterstart = []
else:
    clusterstart = SAPMcnums
print('SAPMrandomclusterstart = {}'.format(SAPMrandomclusterstart))
print('clusterstart set to {}'.format(clusterstart))
np.save(search_data_file, {'SAPMresultsdir' :SAPMresultsdir, 'SAPMresultsname' :SAPMresultsname, 'SAPMparamsname' :SAPMparamsname,
                           'networkmodel' :networkmodel, 'DBname' :DBname, 'SAPMregionname' :SAPMregionname,
                           'SAPMclustername' :SAPMclustername, 'initial_clusters' :clusterstart, 'betascale' :SAPMbetascale})

nprocessors = 1
samplesplit = 1
samplestart = 0
best_clusters = pysapm.SAPM_cluster_stepsearch(SAPMresultsdir, SAPMresultsname, SAPMparamsname, networkmodel,
                                               SAPMregionname, SAPMclustername, samplesplit, samplestart, initial_clusters=clusterstart,
                                               timepoint='all', epoch='all', betascale=SAPMbetascale, Lweight = SAPMLweight,
                                               levelthreshold= levelthreshold, leveliter=leveliter, leveltrials=leveltrials,
                                               run_whole_group = SAPMgrouplevel)

# save the cluster search results
np.save(search_data_file, {'SAPMresultsdir' :SAPMresultsdir, 'SAPMresultsname' :SAPMresultsname, 'SAPMparamsname' :SAPMparamsname,
                           'networkmodel' :networkmodel, 'DBname' :DBname, 'SAPMregionname' :SAPMregionname,
                           'SAPMclustername' :SAPMclustername, 'initial_clusters' :clusterstart, 'betascale' :SAPMbetascale,
                           'bestclusters' :best_clusters})

# now run the result
timepoint = 'all'
epoch = 'all'
SAPMparamsname = os.path.join(workingdir, 'search_trial{:03d}_params.npy'.format(trial_number))
SAPMresultsname = os.path.join(workingdir, 'search_trial{:03d}_results.npy'.format(trial_number))
pysapm.SAPMrun_V2(best_clusters, SAPMregionname, SAPMclustername, SAPMresultsname, SAPMparamsname, networkmodel, timepoint,
            epoch, betascale = 0.1, Lweight = 0.01, alphascale = 0.01, leveltrials = [30, 4, 1], leveliter = [100, 250, 1200],
               levelthreshold = [1e-6, 1e-6, 1e-7], reload_existing = False, multiple_output = False, run_whole_group = False,
               verbose = True, silentrunning = False)