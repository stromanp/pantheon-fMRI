# functions to load previous GLM results and apply masks in various ways
#
# options:
#  1) restrict analysis to specific regions, like small-volume correction in SPM
#  2) generate mask based on significant voxels from both template-wide and small-volume analyses
#  3) extract mean beta values from signfiicant clusters
#  ... then display the results

import numpy as np
import copy
from scipy import stats
import load_templates
from skimage.measure import label
import pydatabase
import pydisplay
import matplotlib.pyplot as plt
import matplotlib
import py_fmristats

#--------------------------------------------------------------------------
#  -- INPUT PARAMETERS-----------------------------------------------------
resultsname = r'E:\Spinalfmri_Sample_for_DrStroman\GLMtest_results.npy'
pvalue_unc = 1.0e-3
region_names = ['PBN', 'NRM', 'NGC', 'PAG', 'C6RD']
cluster_min_size = 1
outputname = r'E:\Spinalfmri_Sample_for_DrStroman\GLMtest_results_clustered.npy'

# mode is 'run' or 'estimate p-threshold'
mode = 'run'
data_prefix = 'xptc'   # needed for 'estimate p-threshold'

if mode != 'run':
	estimate_GRF_pthreshold = True
else:
	estimate_GRF_pthreshold = False


#  -- END OF INPUT PARAMETERS----------------------------------------------
#--------------------------------------------------------------------------

# identify clusters
# identify regions
# identify clusters in regions

# results = {'type': 'GLM', 'B': B, 'sem': sem, 'T': T, 'template': template_img, 'regionmap': regionmap_img,
# 		   'roi_map': roi_map, 'Tthresh': Tthresh, 'normtemplatename': normtemplatename, 'DBname': self.DBname,
# 		   'DBnum': self.DBnum}

results = np.load(resultsname, allow_pickle=True).flat[0]
B = copy.deepcopy(results['B'])
sem = copy.deepcopy(results['sem'])
T = copy.deepcopy(results['T'])
template = copy.deepcopy(results['template'])
template = copy.deepcopy(results['template'])
regionmap = copy.deepcopy(results['regionmap'])
roi_map = copy.deepcopy(results['roi_map'])
# Tthresh = copy.deepcopy(results['Tthresh'])
normtemplatename = copy.deepcopy(results['normtemplatename'])
DBname = copy.deepcopy(results['DBname'])
DBnum = copy.deepcopy(results['DBnum'])

dataname_list, dbnum_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, '')
degrees_of_freedom = NP - 1
Tthreshold = stats.t.ppf(1 - pvalue_unc, degrees_of_freedom)

if normtemplatename == 'brain':
	resolution = 2
else:
	resolution = 1
template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_map = \
	load_templates.load_template_and_masks(normtemplatename, resolution)

anatnamelist = []
for name in anatlabels['names']:
	anatnamelist.append(name)


# GRF estimate
estimate_GRF_pthreshold = True
data_prefix = 'xptc'
if estimate_GRF_pthreshold:
	filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, data_prefix, mode='list')
	input_img = nib.load(filename_list[0])   # estimate the smoothness from one data set
	input_data = input_img.get_data()
	xs, ys, zs, ts = input_data.shape

	p_corr = 0.05
	p_GRF_unc, FWHM, R = py_fmristats.py_GRFcorrected_pthreshold(p_corr, input_data, roi_map, df=0)
	GRF_scale = p_corr/p_GRF_unc

#-------------------------------------------------------
# identify regions to include, and voxels in the regions
anat_number_list = []
voxel_coord_list = []
total_nvox = 0
if region_names[0].lower() != 'all':
	nregions = len(region_names)
	for nn in range(nregions):
		name = region_names[nn]
		c = np.where(anatlabels['names'] == name)[0]
		c_region = [anatlabels['numbers'][cc] for cc in c]
		anat_number_list += [c_region]

		if len(c) > 0:
			for n1 in range(len(c_region)):   # allow that some regions might have more than one number in the regionmap (i.e. left/right)
				c1 = c_region[n1]
				x,y,z = np.where(regionmap_img == c1)
				if n1 == 0:
					cx = copy.deepcopy(x)
					cy = copy.deepcopy(y)
					cz = copy.deepcopy(z)
				else:
					cx = np.concatenate((cx,x),axis=0)
					cy = np.concatenate((cy,y),axis=0)
					cz = np.concatenate((cz,z),axis=0)
		else:
			cx = []
			cy = []
			cz = []
		voxel_coord_list.append({'cx':cx, 'cy':cy, 'cz':cz, 'nvox':len(cx), 'name':name})
		total_nvox += len(cx)
else:
	cx,cy,cz = np.where(roi_map > 0)
	anat_number_list = [0]
	voxel_coord_list.append({'cx':cx, 'cy':cy, 'cz':cz, 'nvox':len(cx), 'name':name})
	total_nvox += len(cx)


if mode != 'run':
	print('-----------------------------------------------------------')
	print('-----------------------------------------------------------')
	print('number of voxels to be analyzed = {}'.format(total_nvox))
	print('Bonferroni correction - divide p by {}'.format(total_nvox))
	print('GRF correction - divide p by {:.1f}'.format(GRF_scale))
	print('\n  for corrected p-value of 0.5:')
	print('              Bonferroni p-thresh =    {:.3e}'.format(pvalue_unc/total_nvox))
	print('              GRF corrected p-thresh = {:.3e}'.format(pvalue_unc/GRF_scale))
	print('-----------------------------------------------------------')
	print('-----------------------------------------------------------')

else:
	print('Total of {} voxels to be analyzed...'.format(total_nvox))
	#-------------------------------------------------------
	# cluster the selected voxels based on GLM results
	results = []
	xs, ys, zs, Nbvals = np.shape(B)   # allow for multiple contrasts etc.
	for bb in range(Nbvals):  # for each contrast, identify clusters of significant voxels within each region
		cluster_data_list = []
		output_data_list = []
		for nn in range(len(voxel_coord_list)):
			cx = voxel_coord_list[nn]['cx']
			cy = voxel_coord_list[nn]['cy']
			cz = voxel_coord_list[nn]['cz']
			name = voxel_coord_list[nn]['name']

			mask = np.zeros((xs,ys,zs))
			mask[cx,cy,cz] = 1.0
			Tmask = np.abs(T[:,:,:,bb]) > Tthreshold
			cluster_mask = Tmask*mask   # keep only the B values in the regions of interest
			cluster_labels = label(cluster_mask, connectivity = 2)
			clusternums = np.unique(cluster_labels)
			for cc in clusternums:
				if cc > 0:
					ccx,ccy,ccz = np.where(cluster_labels == cc)
					nvox = len(ccx)
					if nvox > cluster_min_size:
						meanB = np.mean(B[ccx,ccy,ccz,bb])
						semB = np.std(B[ccx,ccy,ccz,bb])/np.sqrt(nvox)
						cluster_data_list.append({'cx':ccx, 'cy':ccy, 'cz':ccz, 'nvox':nvox, 'Bmean':meanB, 'Bsem':semB})
						xtext = '{:.3f} {} {:.3f}'.format(np.mean(ccx),chr(177),np.std(ccx))
						ytext = '{:.3f} {} {:.3f}'.format(np.mean(ccy),chr(177),np.std(ccy))
						ztext = '{:.3f} {} {:.3f}'.format(np.mean(ccz),chr(177),np.std(ccz))
						Btext = '{:.3f} {} {:.3f}'.format(meanB,chr(177),semB)
						output_data_list.append({'name':name, 'nvox':nvox, 'B':Btext, 'x':xtext, 'y':ytext, 'z':ztext})
		# create a figure showing the clusters
		nclusters_total = len(cluster_data_list)
		results_img = np.zeros((xs,ys,zs))
		for nn in range(nclusters_total):
			cx = copy.deepcopy(cluster_data_list[nn]['cx'])
			cy = copy.deepcopy(cluster_data_list[nn]['cy'])
			cz = copy.deepcopy(cluster_data_list[nn]['cz'])
			results_img[cx,cy,cz] = nn+1
		# create image ...
		outputimg = pydisplay.pydisplaystatmap(results_img, 0.5, template_img, roi_map, normtemplatename)
		pname, fname = os.path.split(outputname)
		fname, ext = os.path.splitext(fname)
		outputimagename = os.path.join(pname, fname + '_contrast{}.png'.format(bb))
		matplotlib.image.imsave(outputimagename, outputimg)

		# write results to excel...
		excelsheetname = 'contrast{}'.format(bb)
		excelfilename = os.path.join(pname, fname + '.xlsx')
		print('writing results to {}, sheet {}'.format(excelfilename, excelsheetname))
		pydisplay.pywriteexcel(output_data_list, excelfilename, excelsheetname, 'append')

		results.append({'cluster_data':cluster_data_list})
		np.save(outputname, results)