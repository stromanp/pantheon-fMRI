# functions to load and analyze the results of voxel-by-voxel GLM analysis
#
# The structure of GLM results:
# results = {'type' :'GLM' ,'B' :B ,'sem' :sem ,'T' :T, 'template' :template_img, 'regionmap' :regionmap_img,
#              'roi_map' :roi_map, 'Tthresh' :Tthresh, 'normtemplatename' :normtemplatename, 'DBname' :self.DBname,
#              'DBnum' :self.DBnum}
#
#  GLMoptions = ['group_average', 'group_concatenate', 'group_concatenate_by_avg_person', 'per_person_avg',
#                         'per_person_concatenate_runs']
# Regardless of how the GLM analysis was done, the data type needs to be set to 'per_person_avg' in order to plot the
# BOLD time-series data for each person

import numpy as np
import pydatabase
import copy
import GLMfit
from skimage import measure
import load_templates
import os
import GLMfit
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
import pandas as pd
import pydisplay

# Steps needed before running this...
# 1) Run GLM in the way that is wanted for identifying active voxels
# 2) compile basis sets in 'per_person_avg' or 'per_person_concatenate_runs' mode
# 3) compile data in the same 'per_person_avg' or 'per_person_concatenate_runs' mode


def CA_displayregion(regionselection, clusters, template, windownum):
	# display a region for testing-----------------------------
	for nn in range(len(clusters)):
		if clusters[nn]['region'] == regionselection:
			x = clusters[nn]['x']
			y = clusters[nn]['y']
			z = clusters[nn]['z']
	x0 = np.round(np.mean(x)).astype(int)
	y0 = np.round(np.mean(y)).astype(int)
	z0 = np.round(np.mean(z)).astype(int)
	rmap = template/np.max(template)
	redmap = copy.deepcopy(rmap)
	greenmap = copy.deepcopy(rmap)
	bluemap = copy.deepcopy(rmap)
	redmap[x,y,z,np.newaxis] = 1.0
	greenmap[x,y,z,np.newaxis] = 1.0
	bluemap[x,y,z,np.newaxis] = 0.0
	colmap = np.stack((redmap,greenmap,bluemap),axis=3)
	aximg = colmap[:,:,z0,:]
	corimg = colmap[:,y0,:,:]
	sagimg = colmap[x0,:,:]

	plt.close(windownum)
	fig = plt.figure(windownum)
	ax1 = plt.subplot(1,2,1)
	plt.imshow(aximg)
	ax1 = plt.subplot(1,2,2)
	plt.imshow(sagimg)


def CA_loadGLMresults(GLMresultsname):
	# load the GLM results
	results = np.load(GLMresultsname, allow_pickle=True).flat[0]
	B = results['B']
	sem = results['sem']
	T = results['T']
	template = results['template']
	regionmap = results['regionmap']
	roi_map = results['roi_map']
	Tthresh_original = results['Tthresh']
	normtemplatename = results['normtemplatename']
	DBname = results['DBname']
	DBnum = results['DBnum']

	return results, B, sem, T, template, regionmap, roi_map, Tthresh_original, normtemplatename, DBname, DBnum


def CA_defineclusters(clusterdataname, clustermethod, clustersizelimit, anatnumbers, anatnames, regionmap,
					  sigmap):

	clustermethod_list = ['region', 'clusters_in_regions', 'clusters']
	if clustermethod not in clustermethod_list:
		print('invalid clustermethod chosen ...')

	if clustermethod == 'region':
		clusters = []
		for nn, aa in enumerate(anatnumbers):
			x, y, z = np.where(sigmap * (regionmap == aa))
			if len(x) > 0:
				entry = {'anatnum': aa, 'voxelcount': len(x), 'x': x, 'y': y, 'z': z, 'region': anatnames[nn]}
				clusters.append(entry)

	if clustermethod == 'clusters_in_regions':
		clusters = []
		labels = measure.label(sigmap, connectivity=3)  # identify clusters of significant voxels
		ulabels, ucounts = np.unique(labels, return_counts=True)
		for nn, aa in enumerate(ulabels):
			if aa != 0:
				x, y, z = np.where(sigmap * (labels == aa))
				if ucounts[nn] >= clustersizelimit:
					# find the anatomical region where these voxels are
					rnums = regionmap[x, y, z]
					rr, rc = np.unique(rnums, return_counts=True)
					if len(rr) > 1:
						rrr = np.argmax(rc)
						rr = rr[rrr]  # use the most common region number
					print('region number {}'.format(int(rr)))
					if rr in anatnumbers:
						print('region {} is in the list of anatomical region numbers'.format(int(rr)))
						rn = anatnumbers.index(rr)
						entry = {'anatnum': rr, 'voxelcount': len(x), 'x': x, 'y': y, 'z': z,
								 'region': anatnames[rn]}
						clusters.append(entry)
					else:
						print('{} is not in the list of anatomical regions'.format(int(rr)))

	if clustermethod == 'clusters':
		clusters = []
		labels = measure.label(sigmap, connectivity=3)  # identify clusters of significant voxels
		ulabels, ucounts = np.unique(labels, return_counts=True)
		for nn, aa in enumerate(ulabels):
			if aa != 0:
				x, y, z = np.where(sigmap * (labels == aa))
				if ucounts[nn] >= clustersizelimit:
					entry = {'voxelcount': len(x), 'x': x, 'y': y, 'z': z}
					clusters.append(entry)

	return clusters


def CA_definesigmap(T, Tthresh):
	# identify clusters based on GLM results
	#  1) identify voxels based on regions?
	#  2) identify clusters of contiguous significant voxels?

	# check the structure of the T-value map
	if np.ndim(T) < 4:
		T = T[:, :, :, np.newaxis]
		Tmapmode = 'group'
	else:
		Tmapmode = 'perperson'
	NP = np.shape(T)[3]

	if Tmapmode == 'group':
		# look for contiguous signficant voxels
		sigmap = np.abs(T[:, :, :, 0]) > Tthresh

	if Tmapmode == 'perperson':
		# need a consensus on which voxels to include
		sigmap_person = np.abs(T) > Tthresh
		sigmap_count = np.sum(sigmap_person, axis=3)
		sigmap = sigmap_count > 0.333 * NP

	return sigmap


def CA_extact_cluster_data(clusterdataname, GroupDataname, clusters):
	# load the group data
	group_data = np.load(GroupDataname, allow_pickle=True)
	print('Image data set has been loaded, with size {}'.format(np.shape(group_data)))
	xs, ys, zs, tsize, NP = np.shape(group_data)

	nc = len(clusters)
	for nn in range(nc):
		x = clusters[nn]['x']
		y = clusters[nn]['y']
		z = clusters[nn]['z']
		cdata = group_data[x, y, z, :, :]
		mdata = np.mean(cdata, axis=0)
		sdata = np.std(cdata, axis=0)
		clusters[nn]['mdata'] = copy.deepcopy(mdata)
		clusters[nn]['sdata'] = copy.deepcopy(sdata)

	np.save(clusterdataname, clusters)
	print('cluster data saved in {}'.format(clusterdataname))

	return clusters


def CA_runGLMclusters(BasisSetname, contrast, clusters):
	b = np.load(BasisSetname, allow_pickle=True).flat[0]
	basisset = b['basisset']
	pardigm_names = b['paradigm_names']

	nb, tsize, NP = np.shape(basisset)

	# set the contrast if not already defined
	# if contrast is empty ...
	if (contrast == 'None') | (not contrast):
		contrast = np.zeros(nb)
		contrast[0] = 1
	else:
		if len(contrast) < nb:
			temp = [0] * nb
			temp[:len(contrast)] = contrast
			contrast = copy.deepcopy(np.array(temp))

	nclusters = len(clusters)

	T = np.zeros((nclusters, NP))
	sem_contrast = np.zeros((nclusters, NP))
	B_contrast = np.zeros((nclusters, NP))

	for cc in range(nclusters):
		mdata = clusters[cc]['mdata']
		sdata = clusters[cc]['sdata']

		for pp in range(NP):
			G = basisset[:, :, pp]  # nb x tsize
			S = mdata[:, pp]  # tsize

			# B = S*G*inv(G*G')
			iGG = np.linalg.inv(np.dot(G, G.T))

			B = np.dot(np.dot(S, G.T), iGG)
			fit = np.dot(B, G)
			err2 = np.sum(((S - fit) ** 2) / tsize)

			scale_check = abs(np.dot(np.dot(contrast, iGG), contrast.T))
			sem_contrast[cc, pp] = np.sqrt(scale_check * err2)
			B_contrast[cc, pp] = contrast @ B
			T[cc, pp] = B_contrast[cc, pp] / (sem_contrast[cc, pp] + 1.0e-20)

	return B_contrast, sem_contrast, T


def CA_load_covariates(covariatesfile, covariatesfile2, covariatename):
	if os.path.isfile(covariatesfile):
		covdata = np.load(covariatesfile, allow_pickle=True).flat[0]
		covnamelist = covdata['GRPcharacteristicslist']
		covvalslist = covdata['GRPcharacteristicsvalues']
		if covariatename in covnamelist:
			x = covnamelist.index(covariatename)
			try:
				covvalues = (covvalslist[x, :]).astype(float)
			except:
				covvalues = copy.deepcopy(covvalslist[x, :])
	else:
		covvalues = []

	if os.path.isfile(covariatesfile2):
		covdata2 = np.load(covariatesfile2, allow_pickle=True).flat[0]
		covnamelist2 = covdata2['GRPcharacteristicslist']
		covvalslist2 = covdata2['GRPcharacteristicsvalues']
		if covariatename in covnamelist2:
			x = covnamelist2.index(covariatename)
			try:
				covvalues2 = (covvalslist2[x, :]).astype(float)
			except:
				covvalues2 = copy.deepcopy(covvalslist2[x, :])
	else:
		covvalues2 = []

	return covvalues, covvalues2


def run_ANOVA_or_ANCOVA(beta1, beta2, cov1, cov2, covname, formula_key1, formula_key2, formula_key3, atype):
    # make up test values
    NP1 = len(beta1)
    NP2 = len(beta2)

    g1 = ['group1']
    g2 = ['group2']
    group = g1 * NP1 + g2 * NP2
    beta = list(beta1) + list(beta2)
    cov = list(cov1) + list(cov2)

    d = {'beta': beta, 'Group': group, covname:cov}
    # print('size of beta is {}'.format(np.shape(beta)))
    # print('size of group is {}'.format(np.shape(group)))
    # print('size of cov is {}'.format(np.shape(cov)))
    # print('d = {}'.format(d))

    df = pd.DataFrame(data=d)

    formula = 'beta ~ ' + formula_key1 + ' + ' + formula_key2 + ' + ' + formula_key3

    try:
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=atype)

        p_MeoG = anova_table['PR(>F)'][formula_key1]
        p_MeoC = anova_table['PR(>F)'][formula_key2]
        p_intGC = anova_table['PR(>F)'][formula_key3]
    except:
        anova_table = []
        p_MeoG = 1.0
        p_MeoC = 1.0
        p_intGC = 1.0

    return anova_table, p_MeoG, p_MeoC, p_intGC


def CA_group_analysis(B_contrast, B_contrast2, covvalues, covvalues2, covariatename, Group_Analysis_Method, group_Tthreshold,
					  Zthreshold, group_pthreshold, TwoGroup, comparison_type):
	if Group_Analysis_Method[:3] == 'sig':
		print('identifying clusters with beta values that are signficantly different than zero ....')
		nc = np.shape(B_contrast)[0]
		if TwoGroup:
			print('identify significant differences between groups')
			if comparison_type[:3] == 'unp':
				print('...unpaired differences')
				NP1 = np.shape(B_contrast)[1]
				NP2 = np.shape(B_contrast2)[1]
				B1mean = np.mean(B_contrast, axis=1)
				B2mean = np.mean(B_contrast2, axis=1)
				B1std = np.std(B_contrast, axis=1)
				B2std = np.std(B_contrast2, axis=1)
				Sp = np.sqrt(((NP1 - 1) * B1std ** 2 + (NP2 - 1) * B2std ** 2) / (NP1 + NP2 - 2))
				T = (B1mean - B2mean) / (Sp * np.sqrt(1 / NP1 + 1 / NP2))
			else:
				print('...paired differences')
				dB = B_contrast - B_contrast2
				NP1 = np.shape(dB)[1]
				dBmean = np.mean(dB, axis=1)
				dBstd = np.std(dB, axis=1)
				T = dBmean / (dBstd / np.sqrt(NP1) + 1.0e-20)
		else:
			print('identify significant clusters')
			NP1 = np.shape(B_contrast)[1]
			B1mean = np.mean(B_contrast, axis=1)
			B1std = np.std(B_contrast, axis=1)
			T = B1mean / (B1std / np.sqrt(NP1) + 1.0e-20)

		group_sig = np.abs(T) > group_Tthreshold
		stats_details = copy.deepcopy(T)

	if Group_Analysis_Method[:3] == 'cor':
		print('identifying clusters in group 1 with beta values that are signficantly correlated with pain ratings....')
		nc = np.shape(B_contrast)[0]
		NP = np.shape(B_contrast)[1]
		Rresults = np.zeros((nc, 2))
		for cc in range(nc):
			b = B_contrast[cc, :]
			r = np.corrcoef(b, covvalues)[0, 1]
			Z = np.arctanh(r) * np.sqrt(NP - 3)
			Rresults[cc, :] = np.array([r, Z])

		group_sig = np.abs(Rresults[:, 1]) > Zthreshold
		stats_details = copy.deepcopy(Rresults)

	if Group_Analysis_Method[:3] == 'ANC':
		print('identifying clusters with beta values that are signficantly dependent on the group and/or pain ratings....')

		statstype = 'ANCOVA'
		formula_key1 = 'C(Group)'
		formula_key2 = covariatename
		formula_key3 = 'C(Group):' + covariatename
		atype = 2

		nc = np.shape(B_contrast)[0]
		ancova_p = np.zeros((nc, 3))

		NP1 = np.shape(B_contrast)[1]
		NP2 = np.shape(B_contrast2)[1]
		for cc in range(nc):
			b1 = B_contrast[cc, :]
			b2 = B_contrast2[cc, :]
			if np.var(b1) > 0 and np.var(b2) > 0:
				anova_table, p_MeoG, p_MeoC, p_intGC = run_ANOVA_or_ANCOVA(b1, b2, covvalues, covvalues2,
											covariatename, formula_key1, formula_key2, formula_key3, atype)
				ancova_p[cc, :] = np.array([p_MeoG, p_MeoC, p_intGC])

		group_sig = np.abs(ancova_p) < group_pthreshold
		stats_details = copy.deepcopy(ancova_p)

	return group_sig, stats_details


# setup color scales for displays
def custom_colormap(nvalues):
	n = np.array(range(nvalues))
	midval = nvalues/2.0

	# red scale
	red = (midval-n)/midval
	red[red<0] = 0

	# green scale
	green = 1.0-np.abs(midval-n)/midval

	# blue scale
	blue = (n-midval)/midval
	blue[blue<0] = 0

	return red, green, blue


def main():   # main function that calls all of the other sub-functions
	# YOU NEED TO SET ALL OF THESE VALUES UP TO THE DOUBLE DASHED LINES
	#
	# names for saving group data and basis sets, or for re-using data sets that have already been compiled
	GLMresultsname = r'D:\RSstim_GLMresults.npy'   # the GLM results used to identify voxels of interest
	pthreshold = 0.05   # signficance for identifying voxels of interest from prior GLM analysis

	GroupDataname = r'D:\RSstim_data_sets.npy'   # this does not have to be compiled in the same mode used for the GLM results
	BasisSetname = r'D:\RSstim_basis_sets.npy'  # this must be compiled in the same mode used for the data sets in 'GroupDataname'
	clusterdataname = r'D:\RSstim_cluster_data.npy'
	covariatesfile = r'D:\RSstim2_covariates.npy'
	covariatename = 'painrating'
	contrast = [1,0]   # this is the contrast for the GLM analysis of the cluster data
					 # the contrast will be padded with zeros to the correct length, if it is too short

	# if there are two groups...
	TwoGroup = False    # if the analysis is for only one group, then set this to False
	# GLMresultsname2 = r'D:\RSnostim_GLMresults.npy'
	GroupDataname2 = r'D:\RSnostim_group_data.npy'   # this does not have to be compiled in the same mode used for the GLM results
	BasisSetname2 = r'D:\RSnostim_basis_sets.npy'  # this must be compiled in the same mode used for the data sets in 'GroupDataname'
	clusterdataname2 = r'D:\RSnostim_cluster_data.npy'
	covariatesfile2 = r'D:\RS1nostim_covariates.npy'

	reload_existing_clusterdata = False   # if these two (reload_existing...) are False then the clusters are defined based on the GLM results
	reload_existing_clusterdata2 = False  # if these are True then the previously saved data are reloaded and used (faster)

	prefix = 'ptc'   # specify howt the data used for the clusters have been pre-processed
	NP = 16   # estimate number of participants for estimating T value thresholds
	clustersizelimit = 10    # make this an input parameter
	# options for clustermethod are 'region', 'clusters_in_regions', 'clusters'
	clustermethod = 'clusters_in_regions'

	group_pthreshold = 0.1   # significance for group comparisons of clusters across participants

	Group_Analysis_Method = 'significance'  # options are 'significance', 'correlation', 'ANCOVA'
	comparison_type = 'unpaired'   # only applies to 'signficance'

	results_file_name = r'D:\RSstim_nostim_compare.xlsx'

	#----------------------------------------------------------------------------------------
	#----------------------------------------------------------------------------------------
	Tthreshold = stats.t.ppf(1 - pthreshold, NP - 1)
	group_Tthreshold = stats.t.ppf(1 - group_pthreshold, NP - 1)
	Zthreshold = stats.norm.ppf(1 - group_pthreshold)   # significance threshold for correlations

	#----------------------------------------------------------------------------------------
	#----------------------------------------------------------------------------------------
	if TwoGroup:
		print('Running GLM on clusters, with data sets\n{}, and\n{}'.format(GroupDataname, GroupDataname2))
	else:
		print('Running GLM on clusters, with data set\n{}'.format(GroupDataname))


	#-----------------------------------load the GLM results info-----------------------------
	#-----------------------------------------------------------------------------------------
	results, B, sem, T, template, regionmap, roi_map, Tthresh_original, normtemplatename, \
			DBname, DBnum = CA_loadGLMresults(GLMresultsname)

	# use the input T value threshold is one was defined,
	# otherwise use the value loaded from the GLM results
	try:
		Tthresh = copy.deepcopy(Tthreshold)
	except:
		Tthresh = copy.deepcopy(Tthresh_original)

	print('loaded the GLM results ...')

	# -------------------get the anatomical template information------------------------------
	#-----------------------------------------------------------------------------------------
	if normtemplatename == 'brain':
		resolution = 2
	else:
		resolution = 1
	template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_map = load_templates.load_template_and_masks(normtemplatename, resolution)
	anatnumbers = list(anatlabels.get('numbers'))
	anatnames = list(anatlabels.get('names'))

	print('loaded anatomical info ...')

	# --------------------prepare to organize/load the data-----------------------------------
	#-----------------------------------------------------------------------------------------
	# dataname_list, dbnum_list_per_person, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix)

	# --------------------------define significance map---------------------------------------
	#-----------------------------------------------------------------------------------------
	sigmap = CA_definesigmap(T, Tthresh)

	print('defined the significance map...')

	#-----------identify clusters- depending on the chosen method-----------------------------
	#-----------------------------------------------------------------------------------------
	if os.path.isfile(clusterdataname) and reload_existing_clusterdata:
		print('Using existing cluster data ...')
		clusters = np.load(clusterdataname, allow_pickle=True)
	else:
		clusters = CA_defineclusters(clusterdataname, clustermethod, clustersizelimit, anatnumbers, anatnames, regionmap, sigmap)
		#----------------extract the data for each region/cluster------------------------------
		# more data get appended to 'clusters'
		clusters = CA_extact_cluster_data(clusterdataname, GroupDataname, clusters)

	if TwoGroup:
		# lod the data for the 2nd group as well
		if os.path.isfile(clusterdataname2) and reload_existing_clusterdata2:
			print('Using existing cluster data ...')
			clusters2 = np.load(clusterdataname2, allow_pickle=True)
		else:
			#----------------extract the data for each region/cluster------------------------------
			# use the cluster definitions for group1, with the data for group2
			clusters2 = CA_extact_cluster_data(clusterdataname2, GroupDataname2, clusters)

	print('defined the clusters and loaded the data ...')

	#-----------------run the GLM----------------------------------------------------------
	#--------------------------------------------------------------------------------------
	B_contrast, sem_contrast, T = CA_runGLMclusters(BasisSetname, contrast, clusters)
	if TwoGroup:
		B_contrast2, sem_contrast2, T2 = CA_runGLMclusters(BasisSetname2, contrast, clusters2)
	else:
		B_contrast2 = []

	print('ran the GLM on the cluster data ...')

	# look for signfiicant clusters
	#   - clusters with significant averages
	#   - clusters with beta values correlated with pain ratings etc.
	#   - ANCOVA across groups, pain ratings, etc.
	# load covariates data, if provided
	covvalues, covvalues2 = CA_load_covariates(covariatesfile, covariatesfile2, covariatename)

	group_sig, stats_details = CA_group_analysis(B_contrast, B_contrast2, covvalues, covvalues2, covariatename,
											  Group_Analysis_Method, group_Tthreshold, Zthreshold, group_pthreshold,
											  TwoGroup, comparison_type)

	#-----------------save and display the results-----------------------------------------
	#--------------------------------------------------------------------------------------
	# 1) write out summary of cluster results to Excel file
	# 2) create image of significant clusters
	nc = len(clusters)
	r,g,b = custom_colormap(nc)
	background = template_img/np.max(template_img)
	clustermap = np.zeros(np.shape(background))
	# background= np.repeat(background[:, :, :, np.newaxis], 3, axis=3)

	if Group_Analysis_Method[:3] == 'ANC':
		outputG = []
		outputC = []
		outputI = []
		clustermapG = copy.deepcopy(clustermap)
		clustermapC = copy.deepcopy(clustermap)
		clustermapI = copy.deepcopy(clustermap)
	else:
		output = []

	for cc in range(nc):
		x = clusters[cc]['x']
		y = clusters[cc]['y']
		z = clusters[cc]['z']
		x0 = np.mean(x)
		y0 = np.mean(y)
		z0 = np.mean(z)
		region = clusters[cc]['region']
		voxelcount = clusters[cc]['voxelcount']
		anatnum = clusters[cc]['anatnum']

		if Group_Analysis_Method[:3] == 'sig':
			if group_sig[cc]:
				T = stats_details[cc]
				entry = {'T':T, 'region':region, 'voxelcount':voxelcount, 'x':x0, 'y':y0, 'z':z0}
				output.append(entry)

				clustermap[x,y,z] = cc+1

		if Group_Analysis_Method[:3] == 'cor':
			if group_sig[cc]:
				R = stats_details[cc,0]
				Z = stats_details[cc,1]
				entry = {'Z':Z, 'R':R, 'region':region, 'voxelcount':voxelcount, 'x':x0, 'y':y0, 'z':z0}
				output.append(entry)

				clustermap[x,y,z] = cc+1

		if Group_Analysis_Method[:3] == 'ANC':
			if group_sig[cc,0]:  # MeoG
				p = stats_details[cc,:]   # MeoG, MeoC, Interation
				entryG = {'MeoG':p[0], 'MeoC':p[1], 'Interaction':p[2], 'region':region, 'voxelcount':voxelcount, 'x':x0, 'y':y0, 'z':z0}
				outputG.append(entryG)

				clustermapG[x,y,z] = cc+1

			if group_sig[cc,1]:  # MeoC
				p = stats_details[cc,:]   # MeoG, MeoC, Interation
				entryC = {'MeoG':p[0], 'MeoC':p[1], 'Interaction':p[2], 'region':region, 'voxelcount':voxelcount, 'x':x0, 'y':y0, 'z':z0}
				outputC.append(entryC)

				clustermapC[x,y,z] = cc+1

			if group_sig[cc,2]:  # Interaction
				p = stats_details[cc,:]   # MeoG, MeoC, Interation
				entryI = {'MeoG':p[0], 'MeoC':p[1], 'Interaction':p[2], 'region':region, 'voxelcount':voxelcount, 'x':x0, 'y':y0, 'z':z0}
				outputI.append(entryI)

				clustermapI[x,y,z] = cc+1

	if Group_Analysis_Method[:3] == 'ANC':
		mask = np.ones(np.shape(template_img))
		results_imgG = pydisplay.pydisplaystatmap(clustermapG, 0.5, template_img, mask, normtemplatename)
		results_imgC = pydisplay.pydisplaystatmap(clustermapC, 0.5, template_img, mask, normtemplatename)
		results_imgI = pydisplay.pydisplaystatmap(clustermapI, 0.5, template_img, mask, normtemplatename)

		pydisplay.pywriteexcel(outputG, results_file_name, 'MeoG', 'append')
		pydisplay.pywriteexcel(outputC, results_file_name, 'MeoC', 'append')
		pydisplay.pywriteexcel(outputI, results_file_name, 'Interaction', 'append')

		pname,fname = os.path.split(results_file_name)
		fname, ext = os.path.splitext(fname)
		outputimagename = os.path.join(pname, fname + '_ANCOVA_MeoG.png')
		matplotlib.image.imsave(outputimagename, results_imgG)
		outputimagename = os.path.join(pname, fname + '_ANCOVA_MeoC.png')
		matplotlib.image.imsave(outputimagename, results_imgC)
		outputimagename = os.path.join(pname, fname + '_ANCOVA_Interaction.png')
		matplotlib.image.imsave(outputimagename, results_imgI)

	else:
		mask = np.ones(np.shape(template_img))
		results_img = pydisplay.pydisplaystatmap(clustermap, 0.5, template_img, mask,normtemplatename)
		pydisplay.pywriteexcel(output, results_file_name, Group_Analysis_Method, 'append')

		pname,fname = os.path.split(results_file_name)
		fname, ext = os.path.splitext(fname)
		outputimagename = os.path.join(pname, fname + '_' + Group_Analysis_Method + '.png')
		matplotlib.image.imsave(outputimagename, results_img)

	print('finished writing results to ', results_file_name)

if __name__ == '__main__':
    main()