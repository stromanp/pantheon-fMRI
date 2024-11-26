import pydisplay
import numpy as np

import pandas as pd
import os
import dicom2nifti
import pydicom
import shutil
import nibabel as nib
import py_mirt3D as mirt
import copy
import matplotlib.pyplot as plt
import image_operations_3D as i3d
import pynormalization as pnorm
import scipy

import trimesh
import time
import io
from PIL import Image

import load_templates


def pointdist(cx,cy,cz, point, maxdist = 0, mindist = 0):
    dist = np.sqrt( (cx - point[0])**2 +  (cy - point[1])**2 +  (cz - point[2])**2)
    outputval = copy.deepcopy(dist)

    if maxdist > 0:
        c = np.where((dist <= maxdist) & (dist > mindist))[0]
        outputval = copy.deepcopy(c)

    return outputval


def interpolate_3d_point(point,volumedata):
    # linear interpolation in 3D
    v = volumedata

    x0 = point[0]
    y0 = point[1]
    z0 = point[2]

    dx = x0-np.floor(x0)
    dy = y0-np.floor(y0)
    dz = z0-np.floor(z0)

    nh = [[x0,y0,z0],[x0+1,y0,z0],[x0-1,y0,z0],
          [x0, y0+1, z0], [x0+1, y0+1, z0], [x0-1, y0+1, z0],
          [x0, y0-1, z0], [x0+1, y0-1, z0], [x0-1, y0-1, z0],
          [x0, y0, z0+1], [x0 + 1, y0, z0+1], [x0 - 1, y0, z0+1],
          [x0, y0 + 1, z0+1], [x0 + 1, y0 + 1, z0+1], [x0 - 1, y0 + 1, z0+1],
          [x0, y0 - 1, z0+1], [x0 + 1, y0 - 1, z0+1], [x0 - 1, y0 - 1, z0+1],
          [x0, y0, z0-1], [x0 + 1, y0, z0-1], [x0 - 1, y0, z0-1],
          [x0, y0 + 1, z0-1], [x0 + 1, y0 + 1, z0-1], [x0 - 1, y0 + 1, z0-1],
          [x0, y0 - 1, z0-1], [x0 + 1, y0 - 1, z0-1], [x0 - 1, y0 - 1, z0-1]
          ]
    nh = np.array(np.floor(nh)).astype(int)  # dropped to nearest actual coordinates
    distlist = pointdist(nh[:,0], nh[:,1], nh[:,2], point)
    weight = 1.0 - distlist
    weight[weight < 0] = 0.0

    value = np.sum(weight*v[nh[:,0], nh[:,1], nh[:,2]])/np.sum(weight)
    return value


def define_triangles_from_cloud(cx,cy,cz,volsize,max_sidelength, min_sidelength = 0.):
    npoints = len(cx)
    allpoints = np.concatenate((cx[:,np.newaxis],cy[:,np.newaxis],cz[:,np.newaxis]),axis=1)

    # identify "surface" points
    method = 'method1'

    if method == 'method1':
        testvol = np.zeros(volsize)
        testvol[cx,cy,cz] = 1.0
        kernel = np.ones((3,3,3))

        testvol2 = scipy.ndimage.convolve(testvol, kernel, mode='constant', cval=0.0)
        maxval = np.max(testvol2)
        cxi,cyi,czi = np.where(testvol2 >= 0.75*maxval)
        testvol[cxi,cyi,czi] = 2.0   # 1 for surface points, 2 for inside points

        cxo,cyo,czo = np.where(testvol == 1.0)

    if method == 'method2':
        testvol = np.zeros(volsize)
        testvol[cx,cy,cz] = 2.0
        xycode = 1000*cx + cy
        for nn in range(len(cx)):
            cc = np.where(xycode == xycode[nn])[0]
            zmin = np.min(cz[cc])
            zmax = np.max(cz[cc])
            textvol[cx[nn],cy[nn],zmin] = 1.0
            textvol[cx[nn],cy[nn],zmax] = 1.0

        cxo,cyo,czo = np.where(testvol == 1.0)

    # create triangles based on points within the distance given by "sidelength"
    triangles = []
    maxlength = copy.deepcopy(max_sidelength)
    minlength = copy.deepcopy(min_sidelength)
    npointso = len(cxo)
    allpointso = np.concatenate((cxo[:,np.newaxis],cyo[:,np.newaxis],czo[:,np.newaxis]),axis=1)
    for nn in range(npointso):
        point = [cxo[nn], cyo[nn], czo[nn]]
        cpoints = pointdist(cxo,cyo,czo,point, maxlength, minlength)
        # triangles based on these points
        nv = len(cpoints)
        for n1 in range(nv):
            for n2 in range(n1+1,nv):
                singletri = [nn, cpoints[n1], cpoints[n2]]
                triangles += [singletri]

    # find redundant triangles
    triangles = np.array(triangles)
    ntri, n3 = np.shape(triangles)
    print('identified {} triangles ... removing redundant ones ...'.format(ntri))
    temp_tri = copy.deepcopy(triangles)
    temp_tri = np.sort(temp_tri,axis=1)
    tri_score = temp_tri[:,0]*1e12 + temp_tri[:,1]*1e6 + temp_tri[:,2]
    u_score, ui = np.unique(tri_score, return_index = True)
    triangles = triangles[ui,:]
    ntri, n3 = np.shape(triangles)
    print('identified {} unique triangles ...'.format(ntri))

    # find normals pointing out of volume
    triangles = np.array(triangles)
    ntri, n3 = np.shape(triangles)
    print('identified {} triangles ... determining normal vectors ...'.format(ntri))
    normvectors = np.zeros(np.shape(triangles))
    normcheck = np.zeros((ntri,2))
    mag_record = np.zeros(ntri)
    pmid_record = np.zeros((ntri,3))
    for nn in range(ntri):
        t = triangles[nn,:]
        p0 = np.array([cxo[t[0]], cyo[t[0]], czo[t[0]]])
        p1 = np.array([cxo[t[1]], cyo[t[1]], czo[t[1]]])
        p2 = np.array([cxo[t[2]], cyo[t[2]], czo[t[2]]])
        normv = np.cross(p1-p0,p2-p0).astype(float)
        norm_mag = np.sqrt(np.sum(normv**2))
        mag_record[nn] = norm_mag
        normv /= (norm_mag + 1.0e-10)

        # pointing into or out of the cloud?
        pmid = (p0+p1+p2)/3.
        pmid_record[nn,:] = pmid
        # interpolate values at pmid + 0.5*normv and pmid - 0.5*normv
        pp = pmid + normv
        pm = pmid - normv

        vp = interpolate_3d_point(pp,testvol2)
        vm = interpolate_3d_point(pm,testvol2)
        if (vp+vm) > 0:
            pout = vp/(vp+vm)
            pin = vm/(vp+vm)
        else:
            pout = 0.
            pin = 0.
        normcheck[nn,:] = np.array([pout,pin])

        if pout > pin:  # reverse the triangle direction
            triangles[nn, :] = np.array([triangles[nn,0],triangles[nn,2],triangles[nn,1]])
            normv *= -1.0
            normcheck[nn,:] = np.array([pin,pout])
        normvectors[nn,:] = normv

    print('Determined normal vectors ... checking for triangles within the cloud ...')
    print('identified {} unique triangles ...'.format(ntri))

    # discard triangles that are likely not near a surface
    includevecs = np.where(normcheck[:,1] > 0.6)[0]
    triangles_filtered = triangles[includevecs,:]
    normvectors_filtered = normvectors[includevecs,:]
    normcheck_filtered = normcheck[includevecs,:]

    ntri,n3 = np.shape(triangles_filtered)
    print('identified {} triangles.'.format(ntri))

    return triangles_filtered, normvectors_filtered, normcheck_filtered, allpointso



def display_anatomical_cluster(clusterdataname, targetnum, targetcluster, orientation = 'axial', regioncolor = [0,1,1], templatename = 'ccbs', write_output = False):
	# get the voxel coordinates for the target region
	clusterdata = np.load(clusterdataname, allow_pickle=True).flat[0]
	cluster_properties = clusterdata['cluster_properties']
	nregions = len(cluster_properties)
	nclusterlist = [cluster_properties[i]['nclusters'] for i in range(nregions)]
	rnamelist = [cluster_properties[i]['rname'] for i in range(nregions)]
	nclusterstotal = np.sum(nclusterlist)

	datadir, f = os.path.split(clusterdataname)

	if type(targetnum) == int:
		r = targetnum
	else:
		# assume "targetnum" input is a region name
		r = rnamelist.index(targetnum)

	IDX = clusterdata['cluster_properties'][r]['IDX']
	# get points for all clusters
	cx0 = []
	cy0 = []
	cz0 = []
	for tt in range(nclusterlist[r]):
		idxx = np.where(IDX == tt)
		cx0 += [clusterdata['cluster_properties'][r]['cx'][idxx]]
		cy0 += [clusterdata['cluster_properties'][r]['cy'][idxx]]
		cz0 += [clusterdata['cluster_properties'][r]['cz'][idxx]]

	cx0 = np.array(cx0)
	cy0 = np.array(cy0)
	cz0 = np.array(cz0)

	# get points for cluster of interest
	idxx = np.where(IDX == targetcluster)
	cx = clusterdata['cluster_properties'][r]['cx'][idxx]
	cy = clusterdata['cluster_properties'][r]['cy'][idxx]
	cz = clusterdata['cluster_properties'][r]['cz'][idxx]

	# load template
	if templatename.lower() == 'brain':
		resolution = 2
	else:
		resolution = 1
	template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_img = \
			load_templates.load_template_and_masks(templatename, resolution)

	# render in 3D
	print(time.ctime())
	max_sidelength = 3.0
	min_sidelength = 0.9
	volsize = np.shape(template_img) * np.array([1, 1, 1])
	check = np.where(cx0 < 13.0)[0]   # keep only the right side
	cx0 = cx0[check]
	cy0 = cy0[check]
	cz0 = cz0[check]
	triangles, normvectors, normcheck, reduced_points = define_triangles_from_cloud(cx0, cy0, cz0, volsize, max_sidelength,
																					min_sidelength)
	print(time.ctime())

	triangles2, normvectors2, normcheck2, reduced_points2 = define_triangles_from_cloud(cx, cy, cz, volsize,
																						max_sidelength, min_sidelength)
	print(time.ctime())

	ntri, n3 = np.shape(triangles)
	facecolors = np.zeros((ntri, 3))
	facecolors[:, 0] = 255
	mesh = trimesh.Trimesh(vertices=reduced_points, faces=triangles, normals=normvectors, face_colors=facecolors, transparency = 0.1)
	# mesh.show()
	meshS = trimesh.smoothing.filter_humphrey(mesh, alpha=0.1, beta=0.5)

	ntri2, n3 = np.shape(triangles2)
	facecolors2 = np.zeros((ntri2, 3))
	facecolors2[:, 2] = 255
	mesh2 = trimesh.Trimesh(vertices=reduced_points2, faces=triangles2, normals=normvectors2, face_colors=facecolors2)
	# mesh.show()
	mesh2S = trimesh.smoothing.filter_humphrey(mesh2, alpha=0.1, beta=0.5)

	mesh_record = {'mesh1': meshS, 'mesh2': mesh2S,
						'reduced_points1': reduced_points, 'triangles1': triangles, 'normvectors1': normvectors,
						'normcheck1': normcheck,
						'reduced_points2': reduced_points2, 'triangles2': triangles2, 'normvectors2': normvectors2,
						'normcheck2': normcheck2}

	# outputname = os.path.join(datadir, 'mesh_data2.npy')
	# np.save(outputname, mesh_record)
	# print('saved results to {}'.format(outputname))
	print(time.ctime())

	camera1 = trimesh.scene.Camera(fov=[200., 200.], resolution=[4000, 4000])
	# display the results...
	orientation1 = [0.75 * np.pi, 0., 0.2 * np.pi]
	orientation2 = [np.pi, -0.25 * np.pi, 0.]

	mesh1 = mesh_record['mesh1']
	mesh2 = mesh_record['mesh2']
	scene = trimesh.scene.scene.Scene(mesh1)
	scene.add_geometry(mesh2)
	scene.set_camera(angles=orientation2)
	imgbytes = scene.save_image()
	pngname = os.path.join(datadir, 'render2_{}{}.png'.format(targetnum,targetcluster))
	image = np.array(Image.open(io.BytesIO(imgbytes)))
	im = Image.fromarray(image)
	im.save(pngname)

	scene.show()


