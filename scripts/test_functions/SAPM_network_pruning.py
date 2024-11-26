#
import numpy as np
import matplotlib.pyplot as plt
import pysapm
import os
import copy
import shutil
import pydisplay
import openpyxl

studynumber = 3
cnums = [0,3,1,0,0,1,3,2,1,3]

datadir = r'E:\SAPMresults2_Oct2022\SAPM_network_test'
nametag = '_0310013213'
resultsbase = ['RSnostim','Sens', 'Low', 'Pain','High', 'RSstim']
covnamebase = ['RSnostim','Sens', 'Low', 'Pain2','High', 'RSstim2']

SAPMresultsname = os.path.join(datadir,resultsbase[studynumber]+nametag+'_results.npy')
SAPMparametersname = os.path.join(datadir,resultsbase[studynumber]+nametag+'_params.npy')
covnames = os.path.join(datadir,covnamebase[studynumber]+'_covariates.npy')
regiondataname = os.path.join(datadir,resultsbase[studynumber]+'_regiondata2.npy')
clusterdataname = r'E:/SAPMresults2_Oct2022\Pain_cluster_def.npy'
networkfile = r'E:\SAPMresults2_Oct2022\SAPM_network_test\network_model_verybig_3latents_v1.xlsx'
DBname = r'E:\graded_pain_database_May2022.xlsx'
timepoint = 'all'
epoch = 'all'
betascale=0.01

network_pruning_data = []

# run it
pysapm.SAPMrun(cnums, regiondataname, clusterdataname, SAPMresultsname, SAPMparametersname, networkfile, DBname, timepoint,
			   epoch, betascale=betascale, reload_existing=False)
results = np.load(SAPMresultsname,allow_pickle=True)
R2record = [results[x]['R2total'] for x in range(len(results))]
R2mean = np.mean(R2record)
R2std = np.std(R2record)
network_pruning_data.append({'resultsname':SAPMresultsname, 'R2record':R2record, 'R2mean':R2mean, 'R2std':R2std})

newnetworkfile, nconnectionstotal = prune_network_model(networkfile, 0)
# edit the network file and take out one connection
for connection_to_cut in range(nconnectionstotal):
	print('\nremoved connection {} of {}\n'.format(connection_to_cut,nconnectionstotal))
	newnetworkfile, nconnectionstotal = prune_network_model(networkfile, connection_to_cut)
	newnametag = '_0310013213_pruned'
	SAPMresultsname = os.path.join(datadir,resultsbase[studynumber]+newnametag+'_results.npy')
	SAPMparametersname = os.path.join(datadir,resultsbase[studynumber]+newnametag+'_params.npy')
	pysapm.SAPMrun(cnums, regiondataname, clusterdataname, SAPMresultsname, SAPMparametersname, newnetworkfile, DBname, timepoint,
					   epoch, betascale=betascale, reload_existing=False)
	results = np.load(SAPMresultsname,allow_pickle=True)
	R2record = [results[x]['R2total'] for x in range(len(results))]
	R2mean = np.mean(R2record)
	R2std = np.std(R2record)
	network_pruning_data.append({'resultsname':SAPMresultsname, 'R2record':R2record, 'R2mean':R2mean, 'R2std':R2std})

	result = {'network_pruning_data':network_pruning_data, 'networkfile':networkfile}
	outputname = r'E:\SAPMresults2_Oct2022\SAPM_network_test\pruning_test2.npy'
	np.save(outputname,result)

# for nn in range(len(network_pruning_data)):
# 	R2record = network_pruning_data[nn]['R2record']
# 	R2mean = np.mean(R2record)
# 	R2std = np.std(R2record)
# 	network_pruning_data[nn]['R2mean'] = R2mean
# 	network_pruning_data[nn]['R2std'] = R2std

print('finished testing pruned networks')
for nn in range(len(network_pruning_data)):
	if nn == 0:
		print('original:  R2 average = {:.3f} {} {:.3f}'.format(network_pruning_data[nn]['R2mean'],chr(177),network_pruning_data[nn]['R2std']))
	else:
		print('cut {}:  R2 average = {:.3f} {} {:.3f}'.format(nn-1,network_pruning_data[nn]['R2mean'],chr(177),network_pruning_data[nn]['R2std']))


# now generate results figures----------------------------------------------------------------
print('generating figures to save as svg files ...')
# outputoptions = ['B_Significance', 'B_Regression', 'Plot_BOLDModel', 'Plot_SourceModel', 'DrawSAPMdiagram',
#                  'DrawAnatomy_axial', 'DrawAnatomy_sagittal']


SRoptionvalue = 'Plot_SourceModel'
SRresultsdir = copy.deepcopy(datadir)
target_region_list = ['C6RD','NGC','NRM']

for nn in range(nresults):
		SRnametag = resultsbase[nn]
		covnamefull = os.path.join(datadir,covnames[nn])
		cov = np.load(covnamefull, allow_pickle=True).flat[0]
		charlist = cov['GRPcharacteristicslist']
		x = charlist.index('painrating')
		covvals = cov['GRPcharacteristicsvalues'][x].astype(float)

		SRparamsname = os.path.join(datadir,paramsnames[nn])
		SRresultsname = os.path.join(datadir,resultsnames[nn])
		SRvariant = 0  # get this from the order information

		c = list(range(len(covvals)))
		SRgroup = c

		SRpvalue = 0.05

		for regionname in target_region_list:
				SRtargetregion = copy.deepcopy(regionname)
				outputname = pysapm.display_SAPM_results(124, SRnametag, covvals, SRoptionvalue,
														 SRresultsdir, SRparamsname, SRresultsname, SRvariant,
														 SRgroup, SRtargetregion, SRpvalue, [], 'none', False)
				print('created figure {}'.format(outputname))




def prune_network_model(networkmodel, connection_number_to_cut):
		network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count = pysapm.load_network_model_w_intrinsics(networkmodel)
		nregions = len(sem_region_list) - fintrinsic_count - vintrinsic_count
		ntargets = len(network)
		sourcecount = np.zeros(ntargets)
		latentcount = np.zeros(ntargets)
		for nn in range(ntargets):
				sourcenums = np.array(network[nn]['sourcenums'])
				c = np.where(sourcenums < nregions)[0]
				sourcecount[nn] = len(c)
				c = np.where(sourcenums >= nregions)[0]
				latentcount[nn] = len(c)

		nconnections_total = int(np.sum(sourcecount))

		runningcount = np.array([0] + list(np.cumsum(sourcecount)))
		c = np.where(connection_number_to_cut >= runningcount)[0]
		cuttargetnum = c[-1]
		cutsourcenum = (connection_number_to_cut - runningcount[cuttargetnum]).astype(int)

		# prune the network
		newnetwork = copy.deepcopy(network)

		sources = newnetwork[cuttargetnum]['sources']
		sourcenums = newnetwork[cuttargetnum]['sourcenums']
		nsources = len(sourcenums)

		sources.remove(sources[cutsourcenum])
		sourcenums.remove(sourcenums[cutsourcenum])

		newnetwork[cuttargetnum]['sources'] = sources
		newnetwork[cuttargetnum]['sourcenums'] = sourcenums
		sourcecount[cuttargetnum] -= 1

		# write out the new network file--------------------------------
		maxsources = np.max(sourcecount + latentcount).astype(int)

		# create the new network file based on the original
		p,f = os.path.split(networkmodel)
		newnetworkname = os.path.join(p,'pruned_'+f)
		shutil.copyfile(networkmodel,newnetworkname)
		workbook = openpyxl.load_workbook(newnetworkname)
		del workbook['connections']
		workbook.save(newnetworkname)

		output = []
		header = ['target']
		for nn in range(maxsources): header += ['source{}'.format(nn+1)]
		for nn in range(ntargets):
				data = [newnetwork[nn]['target']]
				for ss in newnetwork[nn]['sources']:
						data += [ss]
				while len(data) < (maxsources+1): data += ['']

				entry = dict(zip(header, data))
				output.append(entry)

		excelsheetname = 'connections'
		print('writing results to {}, sheet {}'.format(newnetworkname, excelsheetname))
		pydisplay.pywriteexcel(output, newnetworkname, excelsheetname, 'append')
		print('finished writing results to ', newnetworkname)

		return newnetworkname, nconnections_total