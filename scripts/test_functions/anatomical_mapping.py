# sys.path.extend([r'C:\Users\Stroman\PycharmProjects\pantheon\venv'])
# sys.path.extend([r'C:\Users\Stroman\PycharmProjects\pantheon\venv\test_functions'])

import numpy as np
import nibabel as nib
import image_operations_3D as i3d
import math
import matplotlib.pyplot as plt
from scipy import interpolate
import copy
import os
import py_mirt3D as mirt
from PIL import Image, ImageTk
import tkinter as tk
import load_templates
import pynormalization
import pydatabase
from sklearn.cluster import KMeans
import scipy
from skimage import measure
import pandas as pd
import scipy.ndimage as nd
import py_mirt3D as mirt
import scipy.stats as stats


def check_anatomy():

	# specify which data set to read, based on a database file---------------------
	DBname = r'E:\FM2021data\FMS2_database_July27_2022b.xlsx'
	dbnum = 8  # random choice, deal with multiple runs later

	# listname1 = r'E:\FM2021data\FMstim_list.npy'
	# listname2 = r'E:\FM2021data\HCstim_list.npy'
	xls = pd.ExcelFile(DBname, engine='openpyxl')
	df1 = pd.read_excel(xls, 'datarecord')

	normtemplatename = df1.loc[dbnum, 'normtemplatename']
	resolution = 1
	template_img, regionmap_img, template_affine, anatlabels = load_templates.load_template(normtemplatename, resolution)
	# pid = 'HW2018_006_FMS_Stim'
	# pid = 'HW2018_011_FMS_Stim'


	dbhome = df1.loc[dbnum, 'datadir']
	fname = df1.loc[dbnum, 'niftiname']
	seriesnumber = df1.loc[dbnum, 'seriesnumber']
	niiname = os.path.join(dbhome, fname)
	fullpath, filename = os.path.split(niiname)
	prefix = ''
	prefix_niiname = os.path.join(fullpath, prefix + filename)
	nametag = '_s{}'.format(seriesnumber)


	# list1 = np.load(listname1, allow_pickle=True).flat[0]
	# dbnumlist1 = list1['dbnumlist']
	# list2 = np.load(listname2, allow_pickle=True).flat[0]
	# dbnumlist2 = list2['dbnumlist']

	# prefix = ''
	# dataname_list1, dbnum_list1, NP1 = pydatabase.get_datanames_by_person(DBname, dbnumlist1, prefix)
	# dataname_list2, dbnum_list2, NP2 = pydatabase.get_datanames_by_person(DBname, dbnumlist2, prefix)
	# dbnum = dbnum_list1[pid][0]

	normtemplatename = df1.loc[dbnum, 'normtemplatename']
	normdataname = df1.loc[dbnum, 'normdataname']
	normdataname_full = os.path.join(dbhome, normdataname)
	normdata = np.load(normdataname_full, allow_pickle=True).flat[0]

	# load data and apply normalization
	input_image, affine = i3d.load_and_scale_nifti(niiname)  # this takes care of resizing the data to 1 mm voxels
	avg_img = np.mean(input_image[:,:,:,4:], axis = 3)
	std_img = np.std(input_image[:,:,:,4:], axis = 3)


	# get normalization data from the associated pre-processing data ---------------------
	T = normdata['T']
	Tfine = normdata['Tfine']
	template_affine = normdata['template_affine']

	reverse_map_image = normdata['reverse_map_image']
	norm_image_fine = normdata['norm_image_fine']
	xt, yt, zt = np.shape(T['Xs'])  # get the dimensions of the resulting normalized images
	x, y, z = np.shape(T['Xt'])  # get the dimensions of the original input images

	# get list of z positions spanned by each section
	Nnormsections = len(normdata['warpdata'])
	zposlist = []
	for aa in range(Nnormsections):
		Z = copy.deepcopy(normdata['warpdata'][aa]['Z'])
		zmin = np.min(Z)
		zmax = np.max(Z)
		zposlist += [[zmin,zmax]]
	zposlist = np.array(zposlist)

	#--------------------------------------------------------------------------------------
	# ------------choose a position along the head-foot direction to look at---------------
	z0 = 50   # random choice
	#--------------------------------------------------------------------------------------
	#--------------------------------------------------------------------------------------

	# find normalization reference section closest to this position
	check = [ (z0 < zposlist[aa,1]) & (z0 >= zposlist[aa,0]) for aa in range(Nnormsections)]   # in a section
	check2 = [ (z0 < zposlist[aa+1,0]) & (z0 >= zposlist[aa,1]) for aa in range(Nnormsections-1)] # between sections
	if any(check):
		nw = np.where(check)[0]
		zf = np.zeros(len(nw))
		for aa in range(len(nw)):
			p = zposlist[nw[aa],:]
			zf[aa] = np.min([(z0-p[0])/(p[1]-p[0]), (p[1]-z0)/(p[1]-p[0])])
		nw = nw[np.argmax(zf)]
	else:
		if any(check2):
			nw2 = np.where(check2)[0][0]   # check if nw is empty
			if (zposlist[nw2+1,0] - z0) < (z0 - zposlist[nw2,0] - z0):  # find the closest section
				nw = nw2 + 1
			else:
				nw = nw2
		else:
			nw = Nnormsections-1

	# find where sections of the template mapped to the original image
	X = copy.deepcopy(normdata['warpdata'][nw]['X'])
	Y = copy.deepcopy(normdata['warpdata'][nw]['Y'])
	Z = copy.deepcopy(normdata['warpdata'][nw]['Z'])
	Xt = copy.deepcopy(normdata['warpdata'][nw]['Xt'])
	Yt = copy.deepcopy(normdata['warpdata'][nw]['Yt'])
	Zt = copy.deepcopy(normdata['warpdata'][nw]['Zt'])

	# ---------check mapping of section from template to original----------
	# make sure coordinates are within limits
	# use these limits because values will be rounded when used as coordinates
	Xt = np.where(Xt <= -0.5, 0, Xt)
	Xt = np.where(Xt >= xt - 0.5, xt - 1, Xt)
	Yt = np.where(Yt <= -0.5, 0, Yt)
	Yt = np.where(Yt >= yt - 0.5, yt - 1, Yt)
	Zt = np.where(Zt <= -0.5, 0, Zt)
	Zt = np.where(Zt >= zt - 0.5, zt - 1, Zt)

	X = np.where(X <= -0.5, 0, X)
	X = np.where(X >= x - 0.5, x - 1, X)
	Y = np.where(Y <= -0.5, 0, Y)
	Y = np.where(Y >= y - 0.5, y - 1, Y)
	Z = np.where(Z <= -0.5, 0, Z)
	Z = np.where(Z >= z - 0.5, z - 1, Z)

	template_section = template_img[np.round(Xt).astype('int'), np.round(Yt).astype('int'), np.round(Zt).astype('int')]
	regionmap_section = regionmap_img[np.round(Xt).astype('int'), np.round(Yt).astype('int'), np.round(Zt).astype('int')]
	avg_section = avg_img[np.round(X).astype('int'), np.round(Y).astype('int'), np.round(Z).astype('int')]
	std_section = std_img[np.round(X).astype('int'), np.round(Y).astype('int'), np.round(Z).astype('int')]

	xts,yts,zts = np.shape(template_section)
	xs,ys,zs = np.shape(avg_section)

	cx,cy,cz = np.where(np.round(Z).astype('int') == z0)
	template_slice = np.zeros((xts,yts))
	regionmap_slice = np.zeros((xts,yts))
	avg_slice = np.zeros((xs,ys))
	std_slice = np.zeros((xs,ys))
	for aa in range(len(cx)):
		template_slice[cx[aa],cy[aa]] = template_img[np.round(Xt[cx[aa],cy[aa],cz[aa]]).astype('int'), np.round(Yt[cx[aa],cy[aa],cz[aa]]).astype('int'), np.round(Zt[cx[aa],cy[aa],cz[aa]]).astype('int')]
		regionmap_slice[cx[aa],cy[aa]] = regionmap_img[np.round(Xt[cx[aa],cy[aa],cz[aa]]).astype('int'), np.round(Yt[cx[aa],cy[aa],cz[aa]]).astype('int'), np.round(Zt[cx[aa],cy[aa],cz[aa]]).astype('int')]
		avg_slice[cx[aa],cy[aa]] = avg_img[np.round(X[cx[aa],cy[aa],cz[aa]]).astype('int'), np.round(Y[cx[aa],cy[aa],cz[aa]]).astype('int'), np.round(Z[cx[aa],cy[aa],cz[aa]]).astype('int')]
		std_slice[cx[aa],cy[aa]] = std_img[np.round(X[cx[aa],cy[aa],cz[aa]]).astype('int'), np.round(Y[cx[aa],cy[aa],cz[aa]]).astype('int'), np.round(Z[cx[aa],cy[aa],cz[aa]]).astype('int')]

	# display the original slice and the template slice----------------------------------
	shifted_template, Xt, Yt, Zt = rigid_alignment(template_slice, avg_slice)
	shifted_regionslice = apply_rigid_alignment(regionmap_slice, Xt, Yt, Zt)

	# now fine-tune labeling of cord or brainstem in slice, based on voxel properties
	slice_mask = shifted_regionslice > 0.4*np.max(shifted_regionslice)
	notslice_mask = shifted_regionslice <= 0.4*np.max(shifted_regionslice)
	xc,yc = np.where(slice_mask)
	xcn,ycn = np.where(notslice_mask)

	cordpos = np.round([np.mean(xc), np.mean(yc)]).astype(int)

	# intensity
	cord_int_avg = np.mean(avg_slice[xc,yc])
	cord_int_std = np.std(avg_slice[xc,yc])
	cord_std_avg = np.mean(std_slice[xc,yc])
	cord_std_std = np.std(std_slice[xc,yc])
	notcord_avg = np.mean(avg_slice[xcn,ycn])
	notcord_std = np.mean(std_slice[xcn,ycn])


	# distance map
	cordpos = [np.mean(xc), np.mean(yc)]
	xo, yo = np.mgrid[0:(xs - 1):xs * 1j, 0:(ys - 1):ys * 1j]
	distmap = np.sqrt( (xo - cordpos[0])**2 + (yo - cordpos[1])**2 )
	distxmap = np.abs(xo - cordpos[0])
	distymap = np.abs(yo - cordpos[1])

	# # clustering
	# voxeldata = np.concatenate((avg_slice[:,:,np.newaxis], std_slice[:,:,np.newaxis], distmap[:,:,np.newaxis]), axis = 2)
	# voxeldata2 = np.reshape(voxeldata, (xs*ys, 3))
	# # k-means clustering
	# kmeans = KMeans(n_clusters=3).fit(voxeldata2)
	# IDX2 = kmeans.labels_
	# cluster_tc = kmeans.cluster_centers_
	# IDX = np.reshape(IDX2, (xs,ys))

	# shape mapping
	nangles = 8
	distrecord = np.zeros((nangles, 2))
	radarmask = np.zeros((xs,ys))
	for nn in range(nangles):
		theta = (2.0*np.pi)*(nn/nangles)  # angle in radians
		distmax = np.sqrt(xs**2 + ys**2)
		keepgoing = True
		dd = 0
		lim = 1.5
		cordimax = cord_int_avg + lim*cord_int_std
		cordimin = cord_int_avg - lim*cord_int_std
		cordsmax = cord_std_avg + lim*cord_std_std
		cordsmin = cord_std_avg - lim*cord_std_std
		while keepgoing:
			px = np.round(cordpos[0] + dd*np.cos(theta)).astype(int)
			py = np.round(cordpos[1] + dd*np.sin(theta)).astype(int)
			if (px >= 0) & (px < xs) & (py >= 0) & (py < ys):
				vi = avg_slice[px,py]
				vs = std_slice[px,py]
				if (vi < cordimax) & (vi > cordimin) &  (vs < cordsmax) & (vs > cordsmin):
					keepgoing = True
					dd += 0.5
					radarmask[px,py] = 1.0
				else:
					keepgoing = False
			else:
				keepgoing = False

			distrecord[nn,:] = np.array([theta,dd])

		plt.close(28)
		fig = plt.figure(28)
		plt.imshow(radarmask)







	# find a smooth contiguous region that matches the cord properties
	drop_mask = np.zeros((xs,ys))
	drop_mask[cordpos[0],cordpos[1]] = 1.0
	drop_mask1 = copy.deepcopy(drop_mask)
	drop_mask2 = copy.deepcopy(drop_mask)

	drop_mask = copy.deepcopy(drop_mask2)
	drop_mask2 = scipy.ndimage.binary_dilation(drop_mask)
	dx,dy = np.where(drop_mask2 > 0.5)
	dx2, dy2 = [],[]
	maxi = cord_int_avg + cord_int_std
	mini = cord_int_avg - cord_int_std
	for aa in range(len(dx)):
		ii = avg_slice[dx[aa],dy[aa]]
		if (ii < maxi) & (ii >= mini):
			dx2 += [dx[aa]]
			dy2 += [dy[aa]]
	drop_mask2 = np.zeros((xs,ys))
	drop_mask2[dx2,dy2] = 1.0


	windownum = 15
	plt.close(windownum)
	fig = plt.figure(windownum)
	ax1 = fig.add_subplot(231)
	plt.imshow(avg_slice)
	ax2 = fig.add_subplot(232)
	plt.imshow(shifted_regionslice)
	ax3 = fig.add_subplot(233)
	plt.imshow(slice_mask)
	ax4 = fig.add_subplot(234)
	plt.imshow(drop_mask1)
	ax4 = fig.add_subplot(235)
	plt.imshow(drop_mask2)




	# display image information
	windownum = 16
	plt.close(windownum)
	fig = plt.figure(windownum)
	plt.plot(avg_slice[xc,yc].flatten(), std_slice[xc,yc].flatten(), 'og')
	plt.plot(avg_slice[xcn,ycn].flatten(), std_slice[xcn,ycn].flatten(), 'or')





	#distance
	cordpos = [np.mean(xc), np.mean(yc)]
	maxx = np.max(xc)
	minx = np.min(xc)
	maxy = np.max(yc)
	miny = np.min(yc)

	corddist = np.sqrt( (xc - cordpos[0])**2  + (yc - cordpos[1])**2)
	cordxdist = np.abs(xc - cordpos[0])
	cordydist = np.abs(yc - cordpos[1])
	notcorddist = np.sqrt( (xcn - cordpos[0])**2  + (ycn - cordpos[1])**2)
	cordxdist_avg = np.mean(cordxdist)
	cordxdist_std = np.std(cordxdist)
	cordydist_avg = np.mean(cordydist)
	cordydist_std = np.std(cordydist)
	notcorddist_avg = np.mean(notcorddist)
	notcorddist_std = np.std(notcorddist)


	# distance map
	xo, yo = np.mgrid[0:(xs - 1):xs * 1j, 0:(ys - 1):ys * 1j]
	distxmap = np.abs(xo - cordpos[0])
	distymap = np.abs(yo - cordpos[1])

	print('intensity:  cord {:.1f} {} {:.1f}    not cord  {:.1f} {} {:.1f}'.format(cord_avg, chr(177), cord_std, notcord_avg, chr(177), notcord_std))
	print('distance:   cord {:.1f} {} {:.1f}    not cord  {:.1f} {} {:.1f}'.format(corddist_avg, chr(177), corddist_std, notcorddist_avg, chr(177), notcorddist_std))

	# calculate probabilities
	# ival = cord_int_avg - cord_int_std  # sample test
	testval = np.abs(avg_slice - cord_int_avg) + cord_int_avg
	pint = 2.0 - 2.0*stats.norm(cord_int_avg, cord_int_std).cdf(testval)   # probability of an intensity value being within the distribution of cord voxels
	# checkp = [(2.0 - 2.0*stats.norm(cord_int_avg, cord_int_std).cdf(np.abs(ival - cord_int_avg) + cord_int_avg)) for ival in range(300, 900)]

	pint_mask = pint > 0.2
	# now erode/dilate
	pint_mask2 = scipy.ndimage.binary_erosion(pint_mask)
	pint_mask2 = scipy.ndimage.binary_dilation(pint_mask2)


	# sval = cord_std_avg - cord_std_std  # sample test
	testval = np.abs(std_slice - cord_std_avg) + cord_std_avg
	pstd = 2.0 - 2.0*stats.norm(cord_std_avg, cord_std_std).cdf(testval)   # probability of a standard deviation value being within the distribution of cord voxels

	# dval = 0.0 # sample test   # this is the distance from the centre of the cord region mask
	pxdist = 1.0 - stats.norm(cordxdist_avg, 2.*cordxdist_std).cdf(distxmap)
	pydist = 1.0 - stats.norm(cordydist_avg, 2.*cordydist_std).cdf(distymap)

	distmask = np.zeros((xs,ys))
	distmask[minx:maxx, miny:maxy] = 1.0
	# checkp = [(1.0 - stats.norm(corddist_avg, corddist_std).cdf(dval)) for dval in range(0, 10)]

	pcord = pint * pstd * pxdist * pydist
	fig = plt.figure(30)
	plt.imshow(pcord)

	windownum = 15
	plt.close(windownum)
	fig = plt.figure(windownum)
	ax1 = fig.add_subplot(221)
	plt.imshow(pint)
	ax2 = fig.add_subplot(222)
	plt.imshow(pint_mask)
	ax3 = fig.add_subplot(223)
	plt.imshow(shifted_regionslice)
	ax4 = fig.add_subplot(224)
	plt.imshow(avg_slice)



	plt.close(windownum+1)
	fig = plt.figure(windownum+1)
	plt.hist(avg_section.flatten(), bins = 25)

	# false-color images
	red = avg_slice/np.max(avg_slice)
	green = std_slice/np.max(std_slice)
	zzz = np.zeros(np.shape(avg_slice))
	cmap = np.concatenate((red[:,:,np.newaxis],green[:,:,np.newaxis],zzz[:,:,np.newaxis]), axis = 2)
	amap = np.concatenate((red[:,:,np.newaxis],zzz[:,:,np.newaxis],zzz[:,:,np.newaxis]), axis = 2)
	smap = np.concatenate((zzz[:,:,np.newaxis],green[:,:,np.newaxis],zzz[:,:,np.newaxis]), axis = 2)

	plt.close(windownum+2)
	fig = plt.figure(windownum+2)
	ax1 = fig.add_subplot(221)
	plt.imshow(amap)
	ax2 = fig.add_subplot(222)
	plt.imshow(smap)
	ax2 = fig.add_subplot(224)
	plt.imshow(cmap)






# def align_slices(movingimg, fixedimg):
# 	pynormalization.py_norm_fine_tuning(input_image, template, T, input_type='normalized'):
#

def align_slices(movingimg, fixedimg):
	# adapted from pynormalization fine-tuning method
	fixedimg = np.where(np.isfinite(fixedimg), fixedimg, 0.)
	xs,ys = np.shape(fixedimg)

	# set default main settings for MIRT coregistration
	main_init = {'similarity': 'cc',  # similarity measure, e.g. SSD, CC, SAD, RC, CD2, MS, MI
				 'subdivide': 1,  # use 1 hierarchical level
				 'okno': 4,  # mesh window size
				 'lambda': 0.,  # transformation regularization weight, 0 for none
				 'single': 1}

	# Optimization settings
	optim_init = {'maxsteps': 1000,  # maximum number of iterations at each hierarchical level
				  'fundif': 1e-6,  # tolerance (stopping criterion)
				  'gamma': 0.1,  # initial optimization step size
				  'anneal': 0.7}  # annealing rate on the optimization step


	movingimg = movingimg/np.max(movingimg)
	fixedimg = fixedimg/np.max(fixedimg)

	optim = copy.deepcopy(optim_init)
	main = copy.deepcopy(main_init)

	fixedimg3 = np.repeat(fixedimg[:, :, np.newaxis], 3, axis=2)
	movingimg3 = np.repeat(movingimg[:, :, np.newaxis], 3, axis=2)

	res, norm_img_fine = mirt.py_mirt3D_register(fixedimg3, movingimg3, main, optim)
	print('completed fine-tune mapping with py_norm_fine_tuning ...')


	F = mirt.py_mirt3D_F(res['okno']); # Precompute the matrix B - spline basis functions
	Xx, Xy, Xz = mirt.py_mirt3D_nodes2grid(res['X'], F, res['okno']); # obtain the position of all image voxels (Xx, Xy, Xz)
															# from the positions of B-spline control points (res['X']

	xs, ys = np.shape(movingimg)
	X, Y = np.mgrid[range(xs), range(ys)]

	# fine-tuning deviation from the original positions
	dX = Xx[:xs,:ys,1]-X
	dY = Xy[:xs,:ys,1]-Y

	Tfine = {'dX':dX, 'dY':dY}

	plt.close(10)
	fig10 = plt.figure(10)
	ax1 = fig10.add_subplot(221)
	plt.imshow(fixedimg, 'gray')
	ax2 = fig10.add_subplot(222)
	plt.imshow(movingimg, 'gray')

	ax3 = fig10.add_subplot(223)
	plt.imshow(fixedimg, 'gray')
	ax4 = fig10.add_subplot(224)
	plt.imshow(norm_img_fine[:,:,1], 'gray')


	return Tfine, norm_img_fine[:,:,1]



def rigid_alignment(movingimg, fixedimg):
	# input images need to be the same size
	# movingimg is warped non-linearly to match the fixedimg
	maxm = np.max(movingimg)
	minm = np.min(movingimg)
	movingimgS = (movingimg - minm)/(maxm-minm)
	maxf = np.max(fixedimg)
	minf = np.min(fixedimg)
	fixedimgS = (fixedimg - minf)/(maxf-minf)

	xs,ys = np.shape(fixedimg)
	# iteratively work at finer and finer resolution
	xo, yo, zo = np.mgrid[0:(xs - 1):xs * 1j, 0:(ys - 1):ys * 1j, 0:0:1 * 1j]
	refd = np.sqrt( (xs/2)**2 + (ys/2)**2)
	weight = (refd - np.sqrt((xo[:,:,0]-xs/2)**2 + (yo[:,:,0]-ys/2)**2))/refd

	cc = i3d.normxcorr3(movingimgS[:,:,np.newaxis], fixedimgS[:,:,np.newaxis], shape='same')[:,:,0]*weight
	xp, yp = np.where(cc == np.max(cc))
	basecenter = np.array([xp[0], yp[0]])
	baseshift = basecenter - np.array([xs/2, ys/2])

	warpmatrixX = np.zeros((xs,ys,1))
	warpmatrixY = np.zeros((xs,ys,1))
	warpmatrixX[:,:] = baseshift[0]
	warpmatrixY[:,:] = baseshift[1]

	X = xo+warpmatrixX
	Y = yo+warpmatrixY
	Z = zo
	movingimg2 = i3d.warp_image_fast(movingimg[:,:,np.newaxis], X, Y, Z)[:,:,0]

	# plt.close(10)
	# fig10 = plt.figure(10)
	# ax1 = fig10.add_subplot(211)
	# plt.imshow(fixedimg, 'gray')
	# ax2 = fig10.add_subplot(212)
	# plt.imshow(movingimg2, 'gray')

	return movingimg2, X, Y, Z


def apply_rigid_alignment(movingimg, X,Y,Z):
	movingimg2 = i3d.warp_image_fast(movingimg[:,:,np.newaxis], X, Y, Z)[:,:,0]

	return movingimg2



def animal2d(movingimg, fixedimg):
	# input images need to be the same size
	# movingimg is warped non-linearly to match the fixedimg
	maxm = np.max(movingimg)
	minm = np.min(movingimg)
	movingimg = (movingimg - minm)/(maxm-minm)
	maxf = np.max(fixedimg)
	minf = np.min(fixedimg)
	fixedimg = (fixedimg - minf)/(maxf-minf)

	xs,ys = np.shape(fixedimg)
	# iteratively work at finer and finer resolution

	cc = i3d.normxcorr3(movingimg[:,:,np.newaxis], fixedimg[:,:,np.newaxis], shape='same')
	xp, yp, zp = np.where(cc == np.max(cc))
	basecenter = np.array([xp[0], yp[0]])
	baseshift = basecenter - np.array([xs/2, ys/2])

	warpmatrixX = np.zeros((xs,ys,1))
	warpmatrixY = np.zeros((xs,ys,1))
	warpmatrixX[:,:] = baseshift[0]
	warpmatrixY[:,:] = baseshift[1]

	xo, yo, zo = np.mgrid[0:(xs-1):xs * 1j, 0:(ys-1):ys * 1j, 0:0:1 * 1j]

	movingimg2 = i3d.warp_image_fast(movingimg[:,:,np.newaxis], xo+warpmatrixX, yo+warpmatrixY, zo)[:,:,0]

	plt.close(10)
	fig10 = plt.figure(10)
	ax1 = fig10.add_subplot(211)
	plt.imshow(fixedimg, 'gray')
	ax2 = fig10.add_subplot(212)
	plt.imshow(movingimg2, 'gray')

	plt.close(11)
	fig11 = plt.figure(11)
	ax3 = fig11.add_subplot(211)
	plt.imshow(fixedimg, 'gray')
	ax4 = fig11.add_subplot(212)
	plt.imshow(movingimg2, 'gray')


	# repeat
	alpha = 0.1  # make small adjustments
	dx = 0.2
	dy = 0.2
	maxiter = 1000

	scale = 10
	while scale > 0.5:
		converging = True
		smoothval = np.array([scale,scale])
		smooth_fixed = nd.gaussian_filter(fixedimg, smoothval)
		movingimg2 = i3d.warp_image_fast(movingimg[:,:,np.newaxis], xo+warpmatrixX, yo+warpmatrixY, zo)[:,:,0]
		total_error = np.sum(np.abs(movingimg2 - fixedimg))
		smooth_moving = nd.gaussian_filter(movingimg2, smoothval)
		iter = 0

		while converging and (iter < maxiter):
			iter += 1
			print('iter {} scale = {:.2f}  total error = {:.3f}'.format(iter, scale,total_error))
			# adjust warpmatrices
			moving_dxp = i3d.linear_translation(smooth_moving[:,:,np.newaxis], [dx,0,0])[:,:,0]
			moving_dxm = i3d.linear_translation(smooth_moving[:,:,np.newaxis], [-dx,0,0])[:,:,0]
			moving_dyp = i3d.linear_translation(smooth_moving[:,:,np.newaxis], [0,dy,0])[:,:,0]
			moving_dym = i3d.linear_translation(smooth_moving[:,:,np.newaxis], [0,-dy,0])[:,:,0]

			error = np.abs(smooth_moving - smooth_fixed)
			dxerrorp = np.abs(moving_dxp - smooth_fixed)
			dxerrorm = np.abs(moving_dxm - smooth_fixed)
			dyerrorp = np.abs(moving_dyp - smooth_fixed)
			dyerrorm = np.abs(moving_dym - smooth_fixed)
			xshiftp = (dxerrorp - error)/dx
			xshiftm = (dxerrorm - error)/dx
			yshiftp = (dyerrorp - error)/dy
			yshiftm = (dyerrorm - error)/dy

			# xshift = i3d.linear_translation(((xshiftp + xshiftm)/2.)[:,:,np.newaxis], [3,3,0])[:,:,0]
			# yshift = i3d.linear_translation(((yshiftp + yshiftm)/2.)[:,:,np.newaxis], [3,3,0])[:,:,0]

			xshift = (xshiftp + xshiftm)/2.
			yshift = (yshiftp + yshiftm)/2.

			new_warpmatrixX = warpmatrixX - alpha*xshift[:,:,np.newaxis]
			new_warpmatrixY = warpmatrixY - alpha*yshift[:,:,np.newaxis]

			new_movingimg2 = i3d.warp_image_fast(movingimg[:,:,np.newaxis], xo+new_warpmatrixX, yo+new_warpmatrixY, zo)[:,:,0]
			new_total_error = np.sum(np.abs(new_movingimg2 - fixedimg))

			if new_total_error < total_error:
				converging = True
				warpmatrixX = copy.deepcopy(new_warpmatrixX)
				warpmatrixY = copy.deepcopy(new_warpmatrixY)
				total_error = copy.deepcopy(new_total_error)
			else:
				print('iter {}  scale = {:.2f}   total error = {:.3f}    stopped converging'.format(iter, scale,total_error))
				movingimg2 = i3d.warp_image_fast(movingimg[:, :, np.newaxis], xo + warpmatrixX, yo + warpmatrixY, zo)[:, :, 0]

				ax3.clear()
				ax3.imshow(fixedimg, 'gray')
				ax4.clear()
				ax4.imshow(movingimg2, 'gray')

				scale *= 0.5
				alpha *= 0.9
				converging = False






	# # make sure reverse mapping is monotonically increasing across the A/P dimension
	# #
	# Yt = copy.deepcopy(T['Yt'])
	# for xx in range(xt):
	# 	for zz in range(zt):
	# 		ymap = T['Yt'][xx,:,zz]
	# 		dy = ymap[1:] - ymap[:-1]
	# 		dy = np.append(dy,dy[-1])
	# 		ymap[dy < 0] = -1
	# 		T['Yt'][xx, :, zz] = ymap
	#
	# mapped_template = pynormalization.py_apply_normalization(template_img, T, Tfine = 'none', map_to_normalized_space = False)
	# mapped_regionmap = pynormalization.py_apply_normalization(regionmap_img, T, Tfine = 'none', map_to_normalized_space = False)
	#
	# windownum = 21
	# plt.close(windownum)
	# fig = plt.figure(windownum)
	# plt.imshow(avg_img[9,:,:], 'gray')
	#
	# plt.close(windownum+1)
	# fig = plt.figure(windownum+1)
	# plt.imshow(mapped_template[9,:,:], 'gray')
	#
	# plt.close(windownum+2)
	# fig = plt.figure(windownum+2)
	# plt.imshow(mapped_regionmap[9,:,:])
	#
	# bg = copy.deepcopy(avg_img)
	# bg *= 1.0/np.max(bg)
	#
	# map = copy.deepcopy(mapped_regionmap)
	# map *= 1.0/np.max(map)
	#
	# red = copy.deepcopy(bg)
	# green = copy.deepcopy(bg)
	# blue = copy.deepcopy(bg)
	#
	# map_threshold = 0.4
	# x,y,z = np.where(map > map_threshold)
	# red[x,y,z] = map[x,y,z]
	# green[x,y,z] = 1.0
	# blue[x,y,z] = 0
	#
	# cmap = np.concatenate((red[:,:,:,np.newaxis], green[:,:,:,np.newaxis], blue[:,:,:,np.newaxis]), axis=3)
	# plt.close(windownum+3)
	# fig = plt.figure(windownum+3)
	# plt.imshow(cmap[9,:,:,:])
	#
	# plt.close(windownum+4)
	# fig = plt.figure(windownum+4)
	# plt.imshow(cmap[:,:,100,:])
	#
	# plt.close(windownum+5)
	# fig = plt.figure(windownum+5)
	# plt.imshow(avg_img[:,:,100], 'gray')
	#
	#
	# complete_mask_image = np.zeros(np.shape(avg_img))
	# xs,ys,zs = np.shape(avg_img)
	#
	# # create probability map for a single axial slice
	# for zz in range(zs):
	# 	img_slice = avg_img[:,:,zz]
	# 	map_slice = map[:,:,zz]
	# 	x,y = np.where(map_slice > map_threshold)
	#
	# 	# plt.close(windownum+6)
	# 	# fig = plt.figure(windownum+6)
	# 	# plt.hist(img_slice[x,y], bins = 20)
	#
	# 	# find the values in the region
	# 	if len(x) > 10:
	# 		nclusters = 3
	# 		kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(img_slice[x,y].reshape(-1,1))
	# 		IDX = kmeans.labels_
	# 		clusters = kmeans.cluster_centers_[:,0]
	#
	# 		cc = np.argsort(clusters)
	# 		imean = np.zeros(nclusters)
	# 		isd = np.zeros(nclusters)
	# 		map_slice2 = np.zeros(np.shape(map_slice))
	# 		map_this_slice = []
	# 		for nn in range(nclusters):
	# 			c = np.where(IDX == cc[nn])[0]
	# 			map_slice2[x[c],y[c]] = (nn+1)/nclusters
	#
	# 			imean[nn] = np.mean(img_slice[x[c],y[c]])
	# 			isd[nn] = np.std(img_slice[x[c],y[c]])
	#
	# 		# erode and dilate the entire region and look for voxels to add to the existing clusters
	# 		xx, yy = np.where(map_slice2 > 0.)
	# 		npasses = 5
	# 		for nnn in range(npasses):
	# 			xx,yy = np.where(map_slice2 > 0.)
	# 			mask_temp = np.zeros(np.shape(map_slice2))
	# 			mask_temp[xx,yy] = 1
	#
	# 			# dilate-----------------------------------------
	# 			dmask = scipy.ndimage.binary_dilation(mask_temp)
	# 			xcd, ycd = np.where(dmask == 1)
	# 			# compare original and dilated mask
	# 			dx, dy = np.where((dmask - mask_temp) == 1)
	# 			iid = img_slice[dx, dy]
	#
	# 			for nn in range(nclusters):
	# 				ccc = np.where((iid > (imean[nn] - isd[nn])) & (iid < (imean[nn] + isd[nn])))[0]
	# 				# add to map
	# 				map_slice2[dx[ccc], dy[ccc]] = (nn + 1) / nclusters
	# 				# update values
	# 				xc2, yc2 = np.where(map_slice2 == (nn + 1) / nclusters)
	# 				icheck2 = img_slice[xc2, yc2]
	# 				imean[nn] = np.mean(icheck2)
	# 				isd[nn] = np.std(icheck2)
	#
	# 		x,y = np.where(map_slice > map_threshold)
	# 		for nn in range(nclusters):
	# 			c = np.where(IDX == cc[nn])[0]  # original clustering
	# 			xx, yy = np.where(map_slice2 == (nn + 1) / nclusters)   # updated clustering
	# 			map_this_slice.append({'x':x[c],'y':y[c], 'xx':xx, 'yy':yy, 'imean':imean[nn], 'isd':isd[nn]})
	#
	# 		# identify the anatomy cross-section
	# 		# lowest intensity
	# 		nn = 0
	# 		xx0, yy0 = np.where( (map_slice2 == 1/nclusters) | (map_slice2 == 2/nclusters))
	# 		mask_temp = np.zeros(np.shape(map_slice2))
	# 		mask_temp[xx0, yy0] = 1
	# 		# center of mass of original region
	# 		x0 = np.mean(map_this_slice[0]['x']).astype(int)
	# 		y0 = np.mean(map_this_slice[0]['y']).astype(int)
	#
	# 		labeledImage = measure.label(mask_temp, connectivity=2)
	# 		label = labeledImage[x0,y0]
	# 		dx,dy = np.where(labeledImage == label)
	#
	# 		map_slice_final = np.zeros(np.shape(map_slice))
	# 		map_slice_final[dx,dy] = 1.
	#
	#
	# 		#
	# 		# # dilate-----------------------------------------
	# 		# dmask = scipy.ndimage.binary_dilation(mask_temp)
	# 		# xc0, yc0 = np.where(dmask == 0)
	# 		# map_slice_check = copy.deepcopy(map_slice2)
	# 		# map_slice_check[xc0,yc0] = 0.
	# 		#
	# 		# dx, dy = np.where((map_slice_check == (1/nclusters)) | (map_slice_check == (2/nclusters)))
	# 		# # include neighboring points in next cluster
	# 		# map_slice_final = np.zeros(np.shape(map_slice))
	# 		# map_slice_final[dx,dy] = 1.
	#
	# 		complete_mask_image[:,:,zz] = map_slice_final
	#
	#
	# plt.close(windownum+7)
	# fig = plt.figure(windownum+7)
	# plt.imshow(complete_mask_image[9,:,:], 'gray')
	#
