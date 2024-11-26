import numpy as np
import pyclustering
import load_templates
import copy
import image_operations_3D as i3d
import os
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import pydisplay

DBname = r'D:\Howie_FM2_Brain_Data\Howie_FMS2_brain_fMRI_database_JAN2020.xlsx'
DBnumref = 4
prefix = 'xptc'

clusterdef_ccbs_name = r'E:\FM2021data\allstim_equal_cluster_def_Jan22_2024_V3.npy'
clusterdef_brain_name = r'D:\Howie_FM2_Brain_Data\allstim_cluster_def_brain_Jan28_2024_V3.npy'

# mode = 'copy_brainclusters_to_ccbs'
# region_name = 'Thalamus'

mode = 'copy_ccbsclusters_to_brain'
# region_name = 'LC'
# region_name = 'Hypothalamus'
# region_name = 'PAG'
region_name = 'PBN'

networkmodel_ccbs = r'E:\FM2021data\network_model_June2023_SAPM.xlsx'
networkmodel_brain = r'D:\Howie_FM2_Brain_Data\network_model_brain_Dec2023_SAPM_V2.xlsx'

#---------------------------------------

clusterdef_brain = np.load(clusterdef_brain_name, allow_pickle=True).flat[0]
clusterdef_ccbs = np.load(clusterdef_ccbs_name, allow_pickle=True).flat[0]

brain_rname_list = [clusterdef_brain['cluster_properties'][xx]['rname'] for xx in range(len(clusterdef_brain['cluster_properties']))]
brain_rname_index = brain_rname_list.index(region_name)

ccbs_rname_list = [clusterdef_ccbs['cluster_properties'][xx]['rname'] for xx in range(len(clusterdef_ccbs['cluster_properties']))]
ccbs_rname_index = ccbs_rname_list.index(region_name)

template_img_ccbs = copy.deepcopy(clusterdef_ccbs['template_img'])
template_img_brain = copy.deepcopy(clusterdef_brain['template_img'])


IDX_map_ccbs = np.zeros(np.shape(template_img_ccbs))
cx = copy.deepcopy(clusterdef_ccbs['cluster_properties'][ccbs_rname_index]['cx'])
cy = copy.deepcopy(clusterdef_ccbs['cluster_properties'][ccbs_rname_index]['cy'])
cz = copy.deepcopy(clusterdef_ccbs['cluster_properties'][ccbs_rname_index]['cz'])
IDX = copy.deepcopy(clusterdef_ccbs['cluster_properties'][ccbs_rname_index]['IDX'])
template_img_ccbs = copy.deepcopy(clusterdef_ccbs['template_img'])
regionmap_img_ccbs = copy.deepcopy(clusterdef_ccbs['regionmap_img'])
pos_ccbs = [np.mean(cx), np.mean(cy), np.mean(cz)]

IDX_map_brain = np.zeros(np.shape(template_img_brain))
cx2 = copy.deepcopy(clusterdef_brain['cluster_properties'][brain_rname_index]['cx'])
cy2 = copy.deepcopy(clusterdef_brain['cluster_properties'][brain_rname_index]['cy'])
cz2 = copy.deepcopy(clusterdef_brain['cluster_properties'][brain_rname_index]['cz'])
IDX2 = copy.deepcopy(clusterdef_brain['cluster_properties'][brain_rname_index]['IDX'])
template_img_brain = copy.deepcopy(clusterdef_brain['template_img'])
regionmap_img_brain = copy.deepcopy(clusterdef_brain['regionmap_img'])
pos_brain = [np.mean(cx2), np.mean(cy2), np.mean(cz2)]

res1 = 1   # ccbs resolution
res2 = 2   # brain resolution

if mode[:10] == 'copy_brain':
	# coordinates of ccbs voxels in brain template space
	cx_b = (cx - pos_ccbs[0]) * (res1 / res2) + pos_brain[0]
	cy_b = (cy - pos_ccbs[1]) * (res1 / res2) + pos_brain[1]
	cz_b = (cz - pos_ccbs[2]) * (res1 / res2) + pos_brain[2]

	ni = len(IDX)
	IDX_compare = np.zeros(ni)
	for nn in range(ni):
		p = np.array([cx_b[nn],cy_b[nn],cz_b[nn]])
		dist = np.sqrt( (cx_b[nn] - cx2)**2 + (cy_b[nn] - cy2)**2 + (cz_b[nn] - cz2)**2)
		cc = np.argmin(dist)
		IDX_compare[nn] = copy.deepcopy(IDX2[cc])

if mode[:9] == 'copy_ccbs':
	# coordinates of ccbs voxels in brain template space
	cx_c = (cx2 - pos_brain[0]) * (res2 / res1) + pos_ccbs[0]
	cy_c = (cy2 - pos_brain[1]) * (res2 / res1) + pos_ccbs[1]
	cz_c = (cz2 - pos_brain[2]) * (res2 / res1) + pos_ccbs[2]

	ni = len(IDX2)
	IDX_compare = np.zeros(ni)
	for nn in range(ni):
		p = np.array([cx_c[nn], cy_c[nn], cz_c[nn]])
		dist = np.sqrt((cx_c[nn] - cx) ** 2 + (cy_c[nn] - cy) ** 2 + (cz_c[nn] - cz) ** 2)
		cc = np.argmin(dist)
		IDX_compare[nn] = copy.deepcopy(IDX[cc])

# compare IDX and IDX2 and find most frequent relationships
nclusters = 5
for nn in range(nclusters):
	c = np.where(IDX2 == nn)[0]
	other_idx_vals = IDX_compare[c]
	iv = np.unique(other_idx_vals)
	print(' ccbs cluster {}:   '.format(nn))
	for mm in range(len(iv)):
		vv = iv[mm]
		cc = np.where(other_idx_vals == vv)[0]
		nv = len(cc)
		print('        contains {} of brain cluster {}'.format(nv,vv))

# brain cluster timecourse data
# tc = region_properties_brain[brain_rname_index]['tc']
# tsize = region_properties_brain[brain_rname_index]['tsize']
# nruns_per_person = region_properties_brain[brain_rname_index]['nruns_per_person']
# nruns_total = np.sum(nruns_per_person)
# nclusters,tsizetotal = np.shape(tc)
# tc_brain_avg = np.mean(np.reshape(tc, (nclusters, nruns_total, tsize)), axis=1)

# # ccbs cluster timecourse data
# tc = region_properties_ccbs[ccbs_rname_index]['tc']
# tsize = region_properties_ccbs[ccbs_rname_index]['tsize']
# nruns_per_person = region_properties_ccbs[ccbs_rname_index]['nruns_per_person']
# nruns_total = np.sum(nruns_per_person)
# nclusters,tsizetotal = np.shape(tc)
# tc_ccbs_avg = np.mean(np.reshape(tc, (nclusters, nruns_total, tsize)), axis=1)


for xx in range(5):

	if mode[:10] == 'copy_brain':
		c2 = np.where(IDX2 == xx)[0]
		xm2, ym2, zm2 = np.mean(cx2[c2]), np.mean(cy2[c2]), np.mean(cz2[c2])
		c = np.where(IDX_compare == xx)[0]
		xm, ym, zm = np.mean(cx[c]), np.mean(cy[c]), np.mean(cz[c])
		print('cluster {}:  brain {:.1f} {:.1f} {:.1f}  ccbs {:.1f} {:.1f} {:.1f}'.format(xx,xm2,ym2,zm2,xm,ym,zm))

		outputimg = pydisplay.pydisplayvoxelregionslice('brain', template_img_brain, cx2[c2], cy2[c2], cz2[c2], 'axial')
		plt.close(20+xx)
		fig = plt.figure(20+xx)
		plt.imshow(outputimg)

		outputimg = pydisplay.pydisplayvoxelregionslice('ccbs', template_img_ccbs, cx[c], cy[c], cz[c], 'axial')
		plt.close(30+xx)
		fig = plt.figure(30+xx)
		plt.imshow(outputimg)

	if mode[:9] == 'copy_ccbs':
		c2 = np.where(IDX_compare == xx)[0]
		xm2, ym2, zm2 = np.mean(cx2[c2]), np.mean(cy2[c2]), np.mean(cz2[c2])
		c = np.where(IDX == xx)[0]
		xm, ym, zm = np.mean(cx[c]), np.mean(cy[c]), np.mean(cz[c])
		print('cluster {}:  brain {:.1f} {:.1f} {:.1f}  ccbs {:.1f} {:.1f} {:.1f}'.format(xx, xm2, ym2, zm2, xm, ym, zm))

		outputimg = pydisplay.pydisplayvoxelregionslice('brain', template_img_brain, cx2[c2], cy2[c2], cz2[c2], 'axial')
		plt.close(20+xx)
		fig = plt.figure(20 + xx)
		plt.imshow(outputimg)

		outputimg = pydisplay.pydisplayvoxelregionslice('ccbs', template_img_ccbs, cx[c], cy[c], cz[c], 'axial')
		plt.close(30+xx)
		fig = plt.figure(30 + xx)
		plt.imshow(outputimg)

if mode[:10] == 'copy_brain':
	cluster_properties = copy.deepcopy(clusterdef_ccbs['cluster_properties'])
	cluster_properties[ccbs_rname_index]['IDX'] = copy.deepcopy(IDX_compare)

if mode[:9] == 'copy_ccbs':
	cluster_properties = copy.deepcopy(clusterdef_brain['cluster_properties'])
	cluster_properties[brain_rname_index]['IDX'] = copy.deepcopy(IDX_compare)

write_new_data = True

if write_new_data:

	if mode[:10] == 'copy_brain':
		# replace cluster_properties and region_properties in saved files
		clusterdef_ccbs['cluster_properties'] = copy.deepcopy(cluster_properties)
		np.save(clusterdef_ccbs_name, clusterdef_ccbs)

	if mode[:9] == 'copy_ccbs':
		# replace cluster_properties and region_properties in saved files
		clusterdef_brain['cluster_properties'] = copy.deepcopy(cluster_properties)
		np.save(clusterdef_brain_name, clusterdef_brain)

