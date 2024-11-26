import numpy as np
import pysapm
import os
import copy
casenumber = 1

if casenumber == 1:
	networkfile = r'E:\SAPMresults_Dec2022\network_model_April2023_SAPM_V2.xlsx'
if casenumber == 2:
	networkfile = r'E:\SAPMre
sults_Dec2022\network_model_April2023_SAPM_2L.xlsx'

clusterdataname = r'E:/SAPMresults_Dec2022\Pain_equalsize_cluster_def.npy'
SAPMparametersname = r'E:/SAPMresults_Dec2022\AllPain_2Lt_3242423012_params.npy'

network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = pysapm.load_network_model_w_intrinsics(networkfile)

fintrinsic_region = []
if fintrinsic_count > 0:  # find which regions have fixed intrinsic input
	for nn in range(len(network)):
		sources = network[nn]['sources']
		if 'fintrinsic1' in sources:
			fintrinsic_region = network[nn]['targetnum']  # only one region should have this input

# region_data1 = np.load(regiondataname, allow_pickle=True).flat[0]
# region_properties = region_data1['region_properties']
#
cluster_data = np.load(clusterdataname, allow_pickle=True).flat[0]
cluster_properties = pysapm.load_filtered_cluster_properties(clusterdataname, networkfile)

nregions = len(cluster_properties)
nclusterlist = [cluster_properties[i]['nclusters'] for i in range(nregions)]
rnamelist = [cluster_properties[i]['rname'] for i in range(nregions)]
nclusterstotal = np.sum(nclusterlist)

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
# Mconn = np.zeros((nbeta, nbeta))
Mconn = np.zeros((nregions + Nintrinsic, nregions + Nintrinsic))
count = 0
for nn in range(len(network)):
	target = network[nn]['targetnum']
	sources = network[nn]['sourcenums']
	for mm in range(len(sources)):
		source = sources[mm]
		conn1 = beta_id.index(source * 1000 + target)

		count += 1
		beta_pair.append([target, source])
		Mconn[target, source] = count

		if source >= nregions:  # intrinsic input
			# conn2 = conn1
			# Mconn[conn1, conn2] = 1  # set the intrinsic beta values
			Mconn[source, source] = 1  # set the intrinsic beta values

# prep to index Mconn for updating beta values
beta_pair = np.array(beta_pair)
ctarget = beta_pair[:, 0]
csource = beta_pair[:, 1]

latent_flag = np.zeros(len(ctarget))
found_latent_list = []
for nn in range(len(ctarget)):
	# if csource[nn] >= ncon  and ctarget[nn] < ncon:
	if csource[nn] >= nregions and ctarget[nn] < nregions:
		found_latent_list += [csource[nn]]
		occurence = np.count_nonzero(found_latent_list == csource[nn])
		latent_flag[nn] = csource[nn] - nregions + 1

reciprocal_flag = np.zeros(len(ctarget))
for nn in range(len(ctarget)):
	spair = beta_list[csource[nn]]['pair']
	tpair = beta_list[ctarget[nn]]['pair']
	if spair[0] == tpair[1]:
		reciprocal_flag[nn] = 1

# setup Minput matrix--------------------------------------------------------------
# Sconn = Mconn @ Sconn    # propagate the intrinsic inputs through the network
# Sinput = Minput @ Mconn
# Minput = np.zeros((nregions, nbeta))  # mixing of connections to model the inputs to each region
Minput = np.zeros((nregions, nregions + Nintrinsic))  # mixing of connections to model the inputs to each region
betanamelist = [beta_list[a]['name'] for a in range(nbeta)]
for nn in range(len(network)):
	target = network[nn]['targetnum']
	sources = network[nn]['sourcenums']
	for mm in range(len(sources)):
		source = sources[mm]
		betaname = '{}_{}'.format(source, target)
		x = betanamelist.index(betaname)
		# Minput[target, x] = 1
		Minput[target, source] = 1

Mconn2 = copy.deepcopy(Mconn)
Mconn2[ctarget,csource] = np.random.randn(len(ctarget))
e, v = np.linalg.eig(Mconn2)

if casenumber == 1:
	Mconn2_V2 = copy.deepcopy(Mconn2)
	Minput_V2 = copy.deepcopy(Minput)
	eV2, vV2 = np.linalg.eig(Mconn2_V2)

if casenumber == 2:
	Mconn2_2L = copy.deepcopy(Mconn2)
	Minput_2L = copy.deepcopy(Minput)
	e2L, v2L = np.linalg.eig(Mconn2_2L)