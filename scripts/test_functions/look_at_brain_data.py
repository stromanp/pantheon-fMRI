import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pydatabase
import pyclustering

data_type = 'brain'
data_type = 'ccbs'
if data_type == 'brain':
	networkmodel = r'D:\Howie_FM2_Brain_Data\network_model_brain_Dec2023_SAPM_V2.xlsx'
	DBname = r'D:\Howie_FM2_Brain_Data\Howie_FMS2_brain_fMRI_database_JAN2020.xlsx'
	DBnum = [9,10,12]   # FM get the data for the first person in the set
	DBnum = [4,6,7]   # HC get the data for the first person in the set
	clusterdefname = r'D:\Howie_FM2_Brain_Data\allstim_cluster_def_brain_Jan2024.npy'
	windownum = 101
else:
	networkmodel = r'E:\FM2021data\network_model_June2023_SAPM.xlsx'
	DBname = r'E:\FM2021data\FMS2_database_July27_2022b.xlsx'
	DBnum = [8,9,11,14,15]   # FM get the data for the first person in the set
	DBnum = [10, 12, 13, 16, 17]   # HC get the data for the first person in the set
	DBnum = [39,40,42]   # HC get the data for the first person in the set
	clusterdefname = r'E:\FM2021data\allstim_cluster_def_Jan16_2024.npy'
	windownum = 102


region = 'Thalamus'

# get info about data
prefix = 'ptc'
filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix, mode='list')
nruns_per_person = np.zeros(NP).astype(int)
for nn in range(NP):
	nruns_per_person[nn] = len(filename_list[nn])
nruns_total = np.sum(nruns_per_person)

clusterdef = np.load(clusterdefname, allow_pickle=True).flat[0]
cluster_properties = clusterdef['cluster_properties']
template_img = clusterdef['template_img']
regionmap_img = clusterdef['regionmap_img']
rnamelist = [cluster_properties[xx]['rname'] for xx in range(len(cluster_properties))]
index = rnamelist.index(region)
cx,cy,cz = cluster_properties[index]['cx'], cluster_properties[index]['cy'], cluster_properties[index]['cz']
IDX = cluster_properties[index]['IDX']
nvox = len(cx)

# reload data for one person
nvolmask = 0
prefix = 'xptc'
region_properties_test = pyclustering.load_cluster_data(cluster_properties, DBname, DBnum, prefix, nvolmask, networkmodel)
nregions0 = len(region_properties_test)
rnamelist0 = [region_properties_test[xx]['rname'] for xx in range(nregions0)]
index0 = rnamelist0.index(region_name)
tc0 = region_properties_test[index0]['tc']
nclusters,tsize_full = np.shape(tc0)
tsize = (tsize_full/nruns_total).astype(int)

tc0r = np.reshape(tc0,(nclusters,nruns_total,tsize))


# reload data for one person
nvolmask = 0
prefix = 'ptc'
region_properties_raw = pyclustering.load_cluster_data(cluster_properties, DBname, DBnum, prefix, nvolmask, networkmodel)
nregions0b = len(region_properties_raw)
rnamelist0b = [region_properties_raw[xx]['rname'] for xx in range(nregions0b)]
index0b = rnamelist0.index(region_name)
tc0b = region_properties_raw[index0b]['tc']

plt.close(windownum)
nc,nt = np.shape(tc0b)
fig = plt.figure(windownum)
plt.plot(range(nt),tc0b[0,:],'-b')
