# sys.path.extend([r'C:\Users\Stroman\PycharmProjects\pantheon\venv'])
# sys.path.extend([r'C:\Users\Stroman\PycharmProjects\pantheon\venv\test_functions'])

import numpy as np
import matplotlib.pyplot as plt
import pydisplay

rname = r'D:\Howie_FM2_Brain_Data\Brain_equalsize_cluster_def.npy'
region_name = 'AC'

cdata = np.load(rname, allow_pickle=True).flat[0]
cluster_properties = cdata['cluster_properties']
template_img = cdata['template_img']
regionmap_img = cdata['regionmap_img']
nregions = len(cluster_properties)

rname_list = [cluster_properties[xx]['rname'] for xx in range(nregions)]
rindex = rname_list.index(region_name)

cx = cluster_properties[rindex]['cx']
cy = cluster_properties[rindex]['cy']
cz = cluster_properties[rindex]['cz']
IDX = cluster_properties[rindex]['IDX']
nclusters = cluster_properties[rindex]['nclusters']

xs,ys,zs = np.shape(template_img)

for cc in range(nclusters):
	c = np.where(IDX == cc)[0]
	x,y,z = cx[c],cy[c],cz[c]
	xmean, ymean, zmean = np.mean(x), np.mean(y), np.mean(z)
	xmn, ymn, zmn = np.min(x), np.min(y), np.min(z)
	xmx, ymx, zmx = np.max(x), np.max(y), np.max(z)

	print('cluster {}   center: {:.1f} {:.1f} {:.1f}   range: x {}-{} y {}-{} z {}-{}'.format(cc,xmean,ymean,zmean,xmn,xmx,ymn,ymx,zmn,zmx))

	outputimg = pydisplay.pydisplayvoxelregionslice('brain', template_img, x, y, z, 'axial')
	fig = plt.figure(20+cc)
	plt.imshow(outputimg)

	outputimg2= pydisplay.pydisplayvoxelregionslice('brain', template_img, x, y, z, 'sagittal')
	fig = plt.figure(30+cc)
	plt.imshow(outputimg2)


