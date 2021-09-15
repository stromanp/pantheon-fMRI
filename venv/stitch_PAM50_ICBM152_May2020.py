# stitch_PAM50_ICBM152_May2020.py
import os
import numpy as np
import nibabel as nib
#import dicom2nifti
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#from scipy.interpolate import interpn, griddata
#import scipy.io
#from warp_image import warp_image
import image_operations_3D as i3d
import copy
import pandas as pd
#from rotate_image_around_point import rotate3D
import extract_downloaded_templates as edt

verbose = False

original_data_path = r'C:\stroman\MNI_PAM50_templates_2020'
save_data_path = r'C:\stroman\MNI_PAM50_templates_2020\checking_results_2021'

if not os.path.isdir(save_data_path):
    os.mkdir(save_data_path)

create_template = True
create_regionmap = True

#if create_template:

# load templates from PAM50, conn15e, and spinalfmri8
# T2-weighted image
type = 'T2'
save_filename = os.path.join(save_data_path, 'stitched_PAM50_icbm152_May2020_' + type + '.nii')
save_filename_small = os.path.join(save_data_path, 'stitched_PAM50_icbm152_May2020_' + type + '_1mm.nii')
save_filename_smallwm = os.path.join(save_data_path, 'stitched_PAM50_icbm152_May2020_wm_1mm.nii')
# cord images
PAM_data_path = os.path.join(original_data_path, 'PAM50_template')
#cord_data_path = r'C:\stroman\spinalcordtoolbox-master\spinalcordtoolbox-master\20180410_PAM50\PAM50\template'
filenamep = os.path.join(PAM_data_path, 'PAM50_' + type + '.nii.gz')
#brain atlas ICBM152 non-linear symmmetric (2009a)

cord_data_path = 'C:\\stroman\\spinalfmri_analysis8\\anatomicals'
filenamec = os.path.join(cord_data_path, 'CCBS_template_QU300_1440_May2020_V2.nii')

brain_data_path = os.path.join(original_data_path, 'mni_icbm152_nlin_sym_09a')
filenameb = os.path.join(brain_data_path, 'mni_icbm152_' + type + '_tal_nlin_sym_09a.nii')
filenamebwm = os.path.join(brain_data_path, 'mni_icbm152_wm_tal_nlin_sym_09a.nii')

pam_img = nib.load(filenamep)
pam_data = pam_img.get_data()
pam_hdr = pam_img.header
pam_size = np.shape(pam_data)
pam_affine = pam_img.affine

# WM template:
filename_wm = os.path.join(PAM_data_path, 'PAM50_wm.nii.gz')
pam_wm_img = nib.load(filename_wm)
pam_wm_data = pam_wm_img.get_data()

cord_img = nib.load(filenamec)
cord_data = cord_img.get_data()
cord_hdr = cord_img.header
cord_size = np.shape(cord_data)
cord_affine = cord_img.affine

brain_img = nib.load(filenameb)
brain_data = brain_img.get_data()
brain_hdr = brain_img.header
brain_size = np.shape(brain_data)
brain_affine = brain_img.affine

brainwm_img = nib.load(filenamebwm)
brainwm_data = brainwm_img.get_data()

# match the resolution of the brain template to the pam template
vsp = np.array([pam_affine[0,0],pam_affine[1,1],pam_affine[2,2]])
vsb = np.array([brain_affine[0,0],brain_affine[1,1],brain_affine[2,2]])
xi, yi, zi = np.mgrid[range(brain_size[0]), range(brain_size[1]), range(brain_size[2])]
new_brain_size = brain_size*abs(vsb)/abs(vsp)
new_brain_size = np.floor(new_brain_size)
new_brain_size = new_brain_size.astype('int')

brain_linear = i3d.resize_3D(brain_data, new_brain_size)
brain_size2 = np.shape(brain_linear)
brainwm_linear = i3d.resize_3D(brainwm_data, new_brain_size)

brain2_affine = np.zeros((4,4))
brain2_affine[0:3,0:3] = np.abs(pam_affine[0:3,0:3])*np.sign(brain_affine[0:3,0:3])
brain2_affine[0:3,3] = brain_affine[0:3,3]
brain2_affine[3,3] = 1

# match the resolution of the BS/CC template to the pam template
vsp = np.array([pam_affine[0,0],pam_affine[1,1],pam_affine[2,2]])
vsc = np.array([cord_affine[0,0],cord_affine[1,1],cord_affine[2,2]])
xi, yi, zi = np.mgrid[range(cord_size[0]), range(cord_size[1]), range(cord_size[2])]
new_cord_size = cord_size*abs(vsc)/abs(vsp)
new_cord_size = np.floor(new_cord_size)
new_cord_size = new_cord_size.astype('int')

cord_linear = i3d.resize_3D(cord_data, new_cord_size)
cord_size2 = np.shape(cord_linear)
# rotate cord template 180 degrees
cord_linear = i3d.rotate3D(cord_linear, 180, np.multiply(cord_size2,0.5), 2)

if verbose:
    fig = plt.figure(1), plt.imshow(pam_data[np.floor(pam_size[0]/2).astype('int'),:,:], 'gray')
    fig = plt.figure(2), plt.imshow(cord_linear[np.floor(cord_size2[0]/2).astype('int'),:,:], 'gray')
    fig = plt.figure(3), plt.imshow(brain_linear[np.floor(brain_size2[0]/2).astype('int'),:,:], 'gray')


# these are points in each template that should align in the combined template
#if abs(vsc[0]) == 0.5:
    # join point, jpb for brain, jpc for cord
jpb = [197, 168, 23]
jpp = [71 ,60, 1006]
jpc = [24, 51, 262]
# intensity reference points
bref = [197,173,7]   
pref = [71, 66, 969]
cref = [24, 57, 232]

# coords of join point in stitched image
x0 = max([jpb[0],jpp[0]])
y0 = max([jpb[1],jpp[1]])
z0 = max([jpb[2],jpp[2]])

# max dimensions of stitched image
x1 = x0 + max([brain_size2[0]-jpb[0], pam_size[0]-jpp[0]])
y1 = y0 + max([brain_size2[1]-jpb[1], pam_size[1]-jpp[1]])
z1 = z0 + max([brain_size2[2]-jpb[2], pam_size[2]-jpp[2]])

# brain, cord image, corner coordinates in stitched image
xb = [x0-jpb[0], x0-jpb[0]+brain_size2[0]]
yb = [y0-jpb[1], y0-jpb[1]+brain_size2[1]]
zb = [z0-jpb[2], z0-jpb[2]+brain_size2[2]]

xp = [x0-jpp[0], x0-jpp[0]+pam_size[0]]
yp = [y0-jpp[1], y0-jpp[1]+pam_size[1]]
zp = [z0-jpp[2], z0-jpp[2]+pam_size[2]]

xc = [x0-jpc[0], x0-jpc[0]+cord_size2[0]]
yc = [y0-jpc[1], y0-jpc[1]+cord_size2[1]]
zc = [z0-jpc[2], z0-jpc[2]+cord_size2[2]]

# also normalize the intensity values so the image sections have the same approximate intensity scaling
pval = pam_data[pref[0],pref[1],pref[2]]/np.amax(pam_data)
pvalwm = pam_wm_data[pref[0],pref[1],pref[2]]/np.amax(pam_wm_data)
cval = cord_linear[cref[0],cref[1],cref[2]]/np.amax(cord_linear)
bval = brain_linear[bref[0],bref[1],bref[2]]/np.amax(brain_linear)
bvalwm = brainwm_linear[bref[0],bref[1],bref[2]]/np.amax(brainwm_linear)
if (cval == 0.0) or (bval == 0.0) or (pval == 0.0):
    cval = np.amax(cord_data)
    pval = np.amax(pam_data)
    bval = np.amax(brain_data)
if (bvalwm == 0.0) or (pvalwm == 0.0):
    bvalwm = np.amax(brainwm_data)
    pvalwm = np.amax(pam_wm_data)

stitched_image = np.zeros([x1,y1,z1])
stitched_imagewm = np.zeros([x1,y1,z1])
stitched_image[xp[0]:xp[1],yp[0]:yp[1],zp[0]:zp[1]] = pam_data*(bval/pval)/np.amax(pam_data)
stitched_imagewm[xp[0]:xp[1],yp[0]:yp[1],zp[0]:zp[1]] = pam_wm_data*(bvalwm/pvalwm)/np.amax(pam_wm_data)
stitched_image[xb[0]:xb[1],yb[0]:yb[1],(zb[0]+2):zb[1]] = brain_linear[:,:,2:]/np.amax(brain_linear)
stitched_imagewm[xb[0]:xb[1],yb[0]:yb[1],(zb[0]+2):zb[1]] = brainwm_linear[:,:,2:]/np.amax(brainwm_linear)
# the CC/BS region template overlaps the other two templates and is not needed in the actual template
#stitched_image[xc[0]:xc[1],yc[0]:yc[1],zc[0]:zc[1]] = cord_linear*(bval/cval)/np.amax(cord_linear)

# determine affine matrix for result
# also determine affine matrices for brain and CC/BS sections in the same space
# images are expected to be R/L x P/A x F/H, but the brain template (MNI) is flipped R/L into a stupid orientation
# physical coordinates increase towards the right, towards anterior, and towards the head
# the affine matrix must be consistent with this orientation

# origin in brain data
origin_brain = np.matmul(np.linalg.inv(brain_affine),np.array([0,0,0,1]))
origin_brain = origin_brain[0:3]*(abs(vsb)/abs(vsp))   # after scaling to match PAM50 template

corner = np.zeros(3)
corner[0] = brain_affine[0,3] - abs(vsp[0])*(xb[0]-x1)
corner[1] = brain_affine[1,3] - abs(vsp[1])*(yb[0])
corner[2] = brain_affine[2,3] - abs(vsp[2])*(zb[0])

# for the stitched volume
corner = np.zeros(3)
corner = -vsp*(origin_brain + np.array([xb[0],yb[0],zb[0]]) )
new_affine = np.array([[-abs(vsp[0]), 0.0, 0.0, corner[0]],[0.0, abs(vsp[0]), 0.0, corner[1]],[0.0, 0.0, abs(vsp[2]), corner[2]],[0., 0., 0., 1.]])

# for the brain volume 
corner_brain = np.zeros(3)
corner_brain = -vsp*origin_brain
new_affine_brain = np.array([[-abs(vsp[0]), 0.0, 0.0, corner_brain[0]],[0.0, abs(vsp[0]), 0.0, corner_brain[1]],[0.0, 0.0, abs(vsp[2]), corner_brain[2]],[0., 0., 0., 1.]])

# for the CC/BS volume 
corner_ccbs = np.zeros(3)
corner_ccbs = -vsp*(origin_brain + np.array([xc[0],yc[0],zc[0]]) )
new_affine_ccbs = np.array([[-abs(vsp[0]), 0.0, 0.0, corner_ccbs[0]],[0.0, abs(vsp[0]), 0.0, corner_ccbs[1]],[0.0, 0.0, abs(vsp[2]), corner_ccbs[2]],[0., 0., 0., 1.]])

# for the PAM50 volume 
corner_pam = np.zeros(3)
corner_pam = -vsp*(origin_brain + np.array([xp[0],yp[0],zp[0]]) )
new_affine_pam = np.array([[-abs(vsp[0]), 0.0, 0.0, corner_pam[0]],[0.0, abs(vsp[0]), 0.0, corner_pam[1]],[0.0, 0.0, abs(vsp[2]), corner_pam[2]],[0., 0., 0., 1.]])


# create NIfTI format output of the stitched together template
resulting_img = nib.Nifti1Image(stitched_image, new_affine)
nib.save(resulting_img, save_filename)

# save other templates as well, for consistency, and convenience
# brain
save_filename2 = os.path.join(save_data_path, 'brain_template_aligned_with_stitched_PAM50_icbm152_' + type + '.nii')
resulting_img2 = nib.Nifti1Image(brain_linear/np.amax(brain_linear), new_affine_brain)
nib.save(resulting_img2, save_filename2)

save_filename2 = os.path.join(save_data_path, 'brain_template_aligned_with_stitched_PAM50_icbm152_wm.nii')
resulting_img2 = nib.Nifti1Image(brainwm_linear/np.amax(brainwm_linear), new_affine_brain)
nib.save(resulting_img2, save_filename2)

# CCBS
save_filename2 = os.path.join(save_data_path, 'CCBS_template_aligned_with_stitched_PAM50_icbm152.nii')
resulting_img2 = nib.Nifti1Image(cord_linear/np.amax(cord_linear), new_affine_ccbs)
nib.save(resulting_img2, save_filename2)

# PAM50
save_filename2 = os.path.join(save_data_path, 'PAM50_template_aligned_with_stitched_PAM50_icbm152.nii')
resulting_img2 = nib.Nifti1Image(pam_data/np.amax(pam_data), new_affine_pam)
nib.save(resulting_img2, save_filename2)

# the templates so far are all 0.5 mm isotropic resolution - convert to 1 mm resolution as well
# whole CNS
vs = np.array([new_affine[0,0],new_affine[1,1],new_affine[2,2]])
stitched_size = np.shape(stitched_image)
new_size_1mm = np.floor(stitched_size*abs(vs)).astype('int')
new_affine_1mm = np.array([[np.sign(new_affine[0,0]), 0.0, 0.0, new_affine[0,3]],[0.0, np.sign(new_affine[1,1]), 0.0, new_affine[1,3]],[0.0, 0.0, np.sign(new_affine[2,2]), new_affine[2,3]],[0., 0., 0., 1.]])
stitched_image_small = i3d.resize_3D(stitched_image, new_size_1mm)
resulting_img_small = nib.Nifti1Image(stitched_image_small/np.amax(stitched_image_small), new_affine_1mm)
nib.save(resulting_img_small, save_filename_small)
stitched_image_smallwm = i3d.resize_3D(stitched_imagewm, new_size_1mm)
resulting_img_smallwm = nib.Nifti1Image(stitched_image_smallwm/np.amax(stitched_image_smallwm), new_affine_1mm)
nib.save(resulting_img_smallwm, save_filename_smallwm)

# brain
temp_affine = new_affine_brain
temp_data = brain_linear
temp_filename = os.path.join(save_data_path, 'brain_template_aligned_with_stitched_PAM50_icbm152_' + type + '_1mm.nii')
vs = np.array([temp_affine[0,0],temp_affine[1,1],temp_affine[2,2]])
small_size = np.shape(temp_data)
new_size_1mm = np.floor(small_size*abs(vs)).astype('int')
new_affine_1mm = np.array([[np.sign(temp_affine[0,0]), 0.0, 0.0, temp_affine[0,3]],[0.0, np.sign(temp_affine[1,1]), 0.0, temp_affine[1,3]],[0.0, 0.0, np.sign(temp_affine[2,2]), temp_affine[2,3]],[0., 0., 0., 1.]])
temp_data_small = i3d.resize_3D(temp_data, new_size_1mm)
resulting_img_small = nib.Nifti1Image(temp_data_small/np.amax(temp_data_small), new_affine_1mm)
nib.save(resulting_img_small, temp_filename)

# brain wm
temp_affine = new_affine_brain
temp_data = brainwm_linear
temp_filename = os.path.join(save_data_path, 'brain_template_aligned_with_stitched_PAM50_icbm152_wm_1mm.nii')
vs = np.array([temp_affine[0,0],temp_affine[1,1],temp_affine[2,2]])
small_size = np.shape(temp_data)
new_size_1mm = np.floor(small_size*abs(vs)).astype('int')
new_affine_1mm = np.array([[np.sign(temp_affine[0,0]), 0.0, 0.0, temp_affine[0,3]],[0.0, np.sign(temp_affine[1,1]), 0.0, temp_affine[1,3]],[0.0, 0.0, np.sign(temp_affine[2,2]), temp_affine[2,3]],[0., 0., 0., 1.]])
temp_data_small = i3d.resize_3D(temp_data, new_size_1mm)
resulting_img_small = nib.Nifti1Image(temp_data_small/np.amax(temp_data_small), new_affine_1mm)
nib.save(resulting_img_small, temp_filename)
brainwm_linear = []
brainwm_data = []


# CC/BS
temp_affine = new_affine_ccbs
temp_data = cord_linear
temp_filename = os.path.join(save_data_path, 'CCBS_template_aligned_with_stitched_PAM50_icbm152_1mm.nii')
vs = np.array([temp_affine[0,0],temp_affine[1,1],temp_affine[2,2]])
small_size = np.shape(temp_data)
new_size_1mm = np.floor(small_size*abs(vs)).astype('int')
new_affine_1mm = np.array([[np.sign(temp_affine[0,0]), 0.0, 0.0, temp_affine[0,3]],[0.0, np.sign(temp_affine[1,1]), 0.0, temp_affine[1,3]],[0.0, 0.0, np.sign(temp_affine[2,2]), temp_affine[2,3]],[0., 0., 0., 1.]])
temp_data_small = i3d.resize_3D(temp_data, new_size_1mm)
resulting_img_small = nib.Nifti1Image(temp_data_small/np.amax(temp_data_small), new_affine_1mm)
nib.save(resulting_img_small, temp_filename)

# PAM data
temp_affine = new_affine_pam
temp_data = pam_data
temp_filename = os.path.join(save_data_path, 'PAM50_template_aligned_with_stitched_PAM50_icbm152_1mm.nii')
vs = np.array([temp_affine[0,0],temp_affine[1,1],temp_affine[2,2]])
small_size = np.shape(temp_data)
new_size_1mm = np.floor(small_size*abs(vs)).astype('int')
new_affine_1mm = np.array([[np.sign(temp_affine[0,0]), 0.0, 0.0, temp_affine[0,3]],[0.0, np.sign(temp_affine[1,1]), 0.0, temp_affine[1,3]],[0.0, 0.0, np.sign(temp_affine[2,2]), temp_affine[2,3]],[0., 0., 0., 1.]])
temp_data_small = i3d.resize_3D(temp_data, new_size_1mm)
resulting_img_small = nib.Nifti1Image(temp_data_small/np.amax(temp_data_small), new_affine_1mm)
nib.save(resulting_img_small, temp_filename)


#-----------------display to check on the results---------------------------------
# stitched display to see how it worked
sag_slice = int(x1/2)
ax_slices = [800, 1070]
cor_slice = 179
fig1s = plt.figure(4)
ax1s = fig1s.add_subplot(211)
##plt.subplots(nrows=1,ncols=3)
##plt.subplot(131)
img1s = ax1s.imshow(stitched_image[sag_slice,:,:], 'gray')
ax2s = fig1s.add_subplot(223)
img2s = ax2s.imshow(stitched_image[:,cor_slice,:], 'gray')
ax3s = fig1s.add_subplot(224)
img3s = ax3s.imshow(stitched_image[:,:,ax_slices[1]], 'gray')

# check
test_img = nib.load(save_filename)
sag_slice = int(x1/2)
ax_slices = [800, 1070]
cor_slice = 179
fig1x = plt.figure(5)
ax1x = fig1x.add_subplot(211)
##plt.subplots(nrows=1,ncols=3)
##plt.subplot(131)
img1x = ax1x.imshow(stitched_image[sag_slice,:,:], 'gray')
ax2x = fig1x.add_subplot(223)
img2x = ax2x.imshow(stitched_image[:,cor_slice,:], 'gray')
ax3x = fig1x.add_subplot(224)
img3x = ax3x.imshow(stitched_image[:,:,ax_slices[1]], 'gray')


#----------------------------------------------------------------------------------------------
# Now, do the same for the anatomical region maps----------------------------------------------
#----------------------------------------------------------------------------------------------
# region maps are different for the cord though
# maps were done in a different orientation than PAM50 and need to be rotated 180 degrees 
# around S/I axis

#if create_regionmap:
# "atlas" refers to the brain atlas regions, i.e. region map
atlas_dir = os.path.join(original_data_path, 'conn15e_rois')
#atlas_dir = r'C:\stroman\spinalfmri_analysis8\conn15e\conn\rois'

atlasname = os.path.join(atlas_dir, 'atlas.nii')
atlasimg = nib.load(atlasname)
atlashdr = atlasimg.header
atlas_affine = atlasimg.affine
atlas_data = atlasimg.get_data()
atlas_size = np.shape(atlas_data)

# first, align the atlas with the brain image data
# match the resolution of the brain atlas to the cord template
vsp = np.array([pam_affine[0,0],pam_affine[1,1],pam_affine[2,2]])
vsb = np.array([atlas_affine[0,0],atlas_affine[1,1],atlas_affine[2,2]])
new_atlas_size = atlas_size*abs(vsb)/abs(vsp)
new_atlas_size = np.floor(new_atlas_size)
new_atlas_size = new_atlas_size.astype('int')
atlas_linear = i3d.resize_3D_nearest(atlas_data, new_atlas_size)  # do nearest-neighbor resizing to retain region label values
atlas_size2 = np.shape(atlas_linear)
atlas2_affine = np.zeros((4,4))
atlas2_affine[0:3,0:3] = pam_affine[0:3,0:3]
atlas2_affine[0:3,3] = atlas_affine[0:3,3]
atlas2_affine[3,3] = 1

# expect that atlas and brain template were originally centered at the same coords, but the 
# volumes were cropped to different sizes (generated with different programs)
#ats = ((np.array(brain_size2) - np.array(atlas_size2))/2).astype('int')   # atlas template shift
#atlas2 = np.zeros(brain_size2)
#atlas2[ats[0]:ats[0]+atlas_size2[0], ats[1]:ats[1]+atlas_size2[1], ats[2]:ats[2]+atlas_size2[2]] = atlas_linear
#atlas_size2 = np.shape(atlas2)

# instead - use this method:
#brain2_affine instead of new_affine_brain?
atlas2 = i3d.convert_affine_matrices(atlas_linear, atlas2_affine, new_affine_brain, new_brain_size)
atlas2 = np.round(atlas2)
atlas_size2 = np.shape(atlas2)
atlas_linear = []  # free up memory


# see if atlas lines up with the brain
# overlay region maps on anatomical
if verbose:
    background = brain_linear.astype(float)/brain_linear.max()
    red = copy.deepcopy(background)
    green = copy.deepcopy(background)
    blue = copy.deepcopy(background)

    a = np.where(atlas2 > 0)
    red[a] = atlas2[a]/atlas2.max()
    #fig = plt.figure(20), plt.imshow(brain_linear[np.floor(brain_size2[0]/2).astype('int'),:,:], 'gray')

    sag_slice = np.floor(brain_size2[0]/2).astype('int')
    tcimg = np.dstack((red[sag_slice,:,:],green[sag_slice,:,:],blue[sag_slice,:,:]))
    fig = plt.figure(21), plt.imshow(tcimg)

    ax_slice = np.floor(brain_size2[2]/2).astype('int')
    tcimg = np.dstack((red[:,:,ax_slice],green[:,:,ax_slice],blue[:,:,ax_slice]))
    fig = plt.figure(22), plt.imshow(tcimg)


#----------------------------------------------------------------------
# now line up the CC/BS region maps with the CC/BS template
#----------------------------------------------------------------------

# BS/SC region maps
# these were drawn in ITKsnap over the CC/BS template called cord_data above
# most of the regions are drawn on the RHS only, and need to be mirrored for the other side
# the mid-line of the volume, with 1 mm isotropic resolution, is at x = 15
ccbs_dir = os.path.join(original_data_path, 'stromanlab_definitions')
ccname1 = os.path.join(ccbs_dir, 'NRM_rNGC_May2020.nii')
ccname2 = os.path.join(ccbs_dir, 'PAG_rHypothalamus_May2020.nii')
ccname3 = os.path.join(ccbs_dir, 'PBN_NTS_LC_DRt_PRF_May2020.nii')
ccname4 = os.path.join(ccbs_dir, 'thalamus_BSmask.nii')

namelist = [ccname4, ccname1, ccname2, ccname3]
symmetry_type = [2, 1, 1, 1]   # indicate how to handle symmetry

combined_cord_atlas = np.zeros(np.shape(cord_linear))
for num, name in enumerate(namelist, start=0):
    ccimg = nib.load(name)
    cchdr = ccimg.header
    cc_affine = ccimg.affine
    cc_data = ccimg.get_data()
    cc_size = np.shape(cc_data)
    
    if symmetry_type[num] == 1:
        # need to reflect RHS regions onto LHS for CC/BS anat maps - midline is x = 15 in ITKsnap
        # original x-size of this map is 25
        cc_data[14:,:,:] = cc_data[14:3:-1,:,:]
        
    if symmetry_type[num] == 2:
        # need to flip both sides and take the average - midline is x = 15 in ITKsnap
        # original x-size of this map is 25
        v = np.unique(cc_data)
        tmp = copy.deepcopy(cc_data)
        tmp[3:24,:,:] = tmp[24:3:-1,:,:]
        cc_data2 = np.round((cc_data+tmp)/2)
        cc_data = np.zeros(np.shape(cc_data2))
        for v2 in v:
            a = np.where(cc_data2 == v2)
            cc_data[a] = v2
    
    # rotate cord atlas 180 degrees
    cc_data = i3d.rotate3D(cc_data, 180, np.multiply(cc_size,0.5), 2)
    
    # first, align the atlas with the brain image data
    # match the resolution of the brain atlas to the cord template
    vsp = np.array([pam_affine[0,0],pam_affine[1,1],pam_affine[2,2]])
    vsc = np.array([cc_affine[0,0],cc_affine[1,1],cc_affine[2,2]])
    new_cc_size = cc_size*abs(vsc)/abs(vsp)
    new_cc_size = (np.floor(new_cc_size)).astype('int')
    cc_atlas1 = i3d.resize_3D_nearest(cc_data, new_cc_size)  # do nearest-neighbor resizing to retain region label values
    cc_atlas1 = np.round(cc_atlas1)
    cc_size2 = np.shape(cc_atlas1)
    
    a = np.where(cc_atlas1 > 0)
    combined_cord_atlas[a] = cc_atlas1[a]
    
#--------display to check on the results------------------------------------
# see if atlas lines up with the CC/BS
# overlay region maps on anatomical
if verbose:
    background = cord_linear.astype(float)/cord_linear.max()
    red = copy.deepcopy(background)
    green = copy.deepcopy(background)
    blue = copy.deepcopy(background)

    a = np.where(combined_cord_atlas > 0)
    red[a] = combined_cord_atlas[a]/combined_cord_atlas.max()
    #fig = plt.figure(20), plt.imshow(brain_linear[np.floor(brain_size2[0]/2).astype('int'),:,:], 'gray')

    sag_slice = np.floor(cord_size2[0]/2).astype('int')
    tcimg = np.dstack((red[sag_slice,:,:],green[sag_slice,:,:],blue[sag_slice,:,:]))
    fig = plt.figure(31), plt.imshow(tcimg)

    ax_slice = 280
    tcimg = np.dstack((red[:,:,ax_slice],green[:,:,ax_slice],blue[:,:,ax_slice]))
    fig = plt.figure(32), plt.imshow(tcimg)

    ax_slice = 390
    tcimg = np.dstack((red[:,:,ax_slice],green[:,:,ax_slice],blue[:,:,ax_slice]))
    fig = plt.figure(33), plt.imshow(tcimg)

    cor_slice = 82
    tcimg = np.dstack((red[:,cor_slice,:],green[:,cor_slice,:],blue[:,cor_slice,:]))
    fig = plt.figure(34), plt.imshow(tcimg)


# PAM50 atlas over PAM50 template:
# GM template:
filename_gm = os.path.join(PAM_data_path, 'PAM50_gm.nii.gz')
pam_gm_img = nib.load(filename_gm)
pam_gm_data = pam_gm_img.get_data()
#pam_gm_hdr = pam_gm_img.header
#pam_gm_size = shape(pam_gm_data)
#pam_gm_affine = pam_gm_img.affine

# WM template:
filename_wm = os.path.join(PAM_data_path, 'PAM50_wm.nii.gz')
pam_wm_img = nib.load(filename_wm)
pam_wm_data = pam_wm_img.get_data()

# CSF template:
# filename_csf = os.path.join(PAM_data_path, 'PAM50_csf.nii.gz')
# pam_csf_img = nib.load(filename_csf)
# pam_csf_data = pam_csf_img.get_data()


# see if atlas lines up with the template
# overlay region maps on anatomical
if verbose:
    background = pam_data.astype(float)/pam_data.max()
    red = copy.deepcopy(background)
    green = copy.deepcopy(background)
    blue = copy.deepcopy(background)

    a = np.where(pam_gm_data > 0)
    red[a] = pam_gm_data[a]/pam_gm_data.max()
    a = np.where(pam_wm_data > 0)
    green[a] = pam_wm_data[a]/pam_wm_data.max()
    # a = np.where(pam_csf_data > 0)
    # blue[a] = pam_csf_data[a]/pam_csf_data.max()
    #fig = plt.figure(20), plt.imshow(brain_linear[np.floor(brain_size2[0]/2).astype('int'),:,:], 'gray')

    sag_slice = np.floor(pam_size[0]/2).astype('int')
    tcimg = np.dstack((red[sag_slice,:,:],green[sag_slice,:,:],blue[sag_slice,:,:]))
    fig = plt.figure(41), plt.imshow(tcimg)

    ax_slice = 988
    tcimg = np.dstack((red[:,:,ax_slice],green[:,:,ax_slice],blue[:,:,ax_slice]))
    fig = plt.figure(42), plt.imshow(tcimg)


#-----------------------------------------------------------------------------------------------
# now piece together the anatomical region maps, the same way the template pieces were stitched
# also save some of the smaller section separately
# brain_linear lines up with atlas2, for brain regions defined in conn15e
# cord_linear lines up with combined_cord_atlas, for cervical cord/brainstem regions
# as used in spinalfmri8 (matlab)
# pam_data lines up with pam_gm_data, pam_wm_data, and pam_csf_data

# atlas2 with conn15e regions
# combined_cord_atlas with CC/BS regions
# pam_gm_data, pam_wm_data, pam_csf_data  with PAM50 regions

# first, make full size atlases for each separate set of regions
stitched_atlas1a = np.zeros([x1,y1,z1])
stitched_atlas1a[xp[0]:xp[1],yp[0]:yp[1],zp[0]:zp[1]] = pam_gm_data
stitched_atlas1b = np.zeros([x1,y1,z1])
stitched_atlas1b[xp[0]:xp[1],yp[0]:yp[1],zp[0]:zp[1]] = pam_wm_data
# stitched_atlas1c = np.zeros([x1,y1,z1])
# stitched_atlas1c[xp[0]:xp[1],yp[0]:yp[1],zp[0]:zp[1]] = pam_csf_data
stitched_atlas2 = np.zeros([x1,y1,z1])
stitched_atlas2[xb[0]:xb[1],yb[0]:yb[1],zb[0]:zb[1]] = atlas2
stitched_atlas3 = np.zeros([x1,y1,z1])
stitched_atlas3[xc[0]:xc[1],yc[0]:yc[1],zc[0]:zc[1]] = combined_cord_atlas
atlas2 = [] # free up memory
pam_gm_data = []
pam_wm_data = []
pam_csf_data = []

# now combine, and keep track of region names/labels
# load the region label definitions
regionname_file = os.path.join(original_data_path, 'maps_region_numbers_definitions.xlsx')
xls = pd.ExcelFile(regionname_file)
df1 = pd.read_excel(xls)

# put all the regions in one atlas
region_count = 0
stitched_atlas = np.zeros([x1,y1,z1])
# CONN15E regions
exclude_list = ['Brain-Stem']
for num, region_number in enumerate(df1['Cnumber']):   # regions in CONN15E template
    region_abbrev = df1['CONN15E_abbrev'][num]
    region_fullname = df1['CONN15E template'][num]
    if not(region_abbrev in exclude_list):
        a = np.where(stitched_atlas2 == region_number)
        # print('size of a is ', np.size(a))
        count = np.size(a)
        # count = (stitched_atlas2 == region_number).sum()
        if count > 0:
            region_count += 1
            print(region_count, ' ', region_abbrev)
            stitched_atlas[a] = region_count
            if region_count == 1:
                stitched_region_labels = {'number':[region_count], 'name':[region_fullname], 'abbreviation':[region_abbrev], 'voxcount':[count]}
            else:
                stitched_region_labels['number'].append(region_count)
                stitched_region_labels['name'].append(region_fullname)
                stitched_region_labels['abbreviation'].append(region_abbrev)
                stitched_region_labels['voxcount'].append(count)
                
            print('num: ',num,'  region_number: ',region_count,' abbrev: ',region_abbrev,'  full: ',region_fullname)
            
# CC/BS regions
for num, region_number in enumerate(df1['CCBSnumber']):   # regions in CC/BS template
    region_abbrev = df1['CCBS template'][num]
    region_fullname = df1['CCBS template'][num]
    if not(region_abbrev in exclude_list):
        a = np.where(stitched_atlas3 == region_number)
        # print('size of a is ', np.size(a))
        count = np.size(a)
        # count = (stitched_atlas3 == region_number).sum()
        if count > 0:
            region_count += 1
            print(region_count, ' ', region_abbrev)
            stitched_atlas[a] = region_count
            if region_count == 1:
                stitched_region_labels = {'number':[region_count], 'name':[region_fullname], 'abbreviation':[region_abbrev], 'voxcount':[count]}
            else:
                stitched_region_labels['number'].append(region_count)
                stitched_region_labels['name'].append(region_fullname)
                stitched_region_labels['abbreviation'].append(region_abbrev)
                stitched_region_labels['voxcount'].append(count)
                
            print('num: ',num,'  region_number: ',region_count,' abbrev: ',region_abbrev,'  full: ',region_fullname, ' count = ',count)
            

# free up some memory here
stitched_atlas2 = []
stitched_atlas3 = []

# ---------------------------------------------------------------------------------
# specially defined regions - from atlases from other sources
# defined in extract_downloaded_templates.py
#----------------------------------------------------------------------------------
#regions defined are: 'NAC','Amygdala','Substantia Nigra', 'PBP', 'VTA','Hypothalamus', 'PAG','VTA','LC', 'LC'
# 'ow' means over-write
# 'combine' means use the combination of both
# 'ignore' means keep the existing definition
handle_method = ['ow', 'ow', 'ow', 'ow', 'ow', 'ow', 'combine', 'ow', 'ignore', 'combine']

srn_list = [0,1,2,3,4,5,6,7,9]
for special_region_number in srn_list:
    special_stitched_atlas = np.zeros([x1,y1,z1])
    region_map, region_name = edt.load_anatomical_template(special_region_number, new_affine_brain, new_brain_size)
    special_stitched_atlas[xb[0]:xb[1],yb[0]:yb[1],zb[0]:zb[1]] = region_map
    region_map = []
    a = np.where(special_stitched_atlas > 0)
                
    # combine with existing regions, depending on the region
    rna = stitched_region_labels['abbreviation']
    if region_name in rna:
        b = rna.index(region_name)
        existing_region_number = stitched_region_labels['number'][b]
    else:
        existing_region_number = -1
        
    # see if atlas lines up with the brain
    # overlay region maps on anatomical
    verbose = False
    if verbose:
        background = stitched_image.astype(float)/stitched_image.max()
        red = copy.deepcopy(background)
        green = copy.deepcopy(background)
        blue = copy.deepcopy(background)

        sagslice = np.round(np.average(a[0][:])).astype('int')
        corslice = np.round(np.average(a[1][:])).astype('int')
        axslice = np.round(np.average(a[2][:])).astype('int')
        #red[a] = 1.0
    
    if existing_region_number >= 0:
        b = np.where(stitched_atlas == existing_region_number)
        if handle_method[special_region_number] == 'ow':   # overwrite
            stitched_atlas[b] = 0    # removing the existing region definition
            stitched_atlas[a] = existing_region_number
        if handle_method[special_region_number] == 'combine':   # merge the two regions
            stitched_atlas[a] = existing_region_number
            
        # update the number of voxels in the region labels list
        b2 = np.where(stitched_atlas == existing_region_number)
        loc = stitched_region_labels['number'].index(existing_region_number)
        stitched_region_labels['voxcount'][loc] = np.shape(b2)[1]
        
        print('region_number: ',existing_region_number,' abbrev: ',stitched_region_labels['abbreviation'][loc],'  full: ',stitched_region_labels['name'][loc], ' count = ',stitched_region_labels['voxcount'][loc])
        
    else:   # region is not already in the list
        region_count += 1
        print(region_count, ' ', region_name)
        stitched_atlas[a] = region_count
        stitched_region_labels['number'].append(region_count)
        stitched_region_labels['name'].append(region_name)
        stitched_region_labels['abbreviation'].append(region_name)
        stitched_region_labels['voxcount'].append(np.shape(a)[1])
        print('region_number: ',region_count,' abbrev: ',region_name,'  full: ',region_name, ' count = ',np.shape(a)[1])

        # for display only:
    #    blue[b] = 1.0
    #    green[b] = 0.0

    special_stitched_atlas = []
    
    #sagslice -= 10
    #tcimg = np.dstack((red[sagslice,:,:],green[sagslice,:,:],blue[sagslice,:,:]))
    #fig, axs = plt.figure(51), plt.imshow(tcimg)
    #fig.suptitle(region_name)
    #
    #tcimg = np.dstack((red[:,:,axslice],green[:,:,axslice],blue[:,:,axslice]))
    #fig, axs = plt.figure(52), plt.imshow(tcimg)
    #fig.suptitle(region_name)
    #
    #tcimg = np.dstack((red[:,corslice,:],green[:,corslice,:],blue[:,corslice,:]))
    #fig, axs = plt.figure(53), plt.imshow(tcimg)
    #fig.suptitle(region_name)
    
# ---------------------------------------------------------------------------------
# end of specially defined regions
#----------------------------------------------------------------------------------


# PAM50 regions
# first make sure regions already identified are not over-written
# want to make two sets of templates
#  1) showing gm/wm/csf
#  2) showing cord segments/quadrants
# a = np.where(stitched_atlas > 0)
# stitched_atlas1a[a] = 0
# stitched_atlas1b[a] = 0
# stitched_atlas1c[a] = 0
stitched_atlas1a = np.where(stitched_atlas > 0, 0, stitched_atlas1a)
stitched_atlas1b = np.where(stitched_atlas > 0, 0, stitched_atlas1b)

a = np.where(stitched_atlas1a > 0)   # gm
region_count += 1
stitched_atlas[a] = region_count
stitched_region_labels['number'].append(region_count)
stitched_region_labels['name'].append('PAM50 gm')
stitched_region_labels['abbreviation'].append('cord gm')
stitched_region_labels['voxcount'].append(np.shape(a)[1])
region_number_gm = region_count
stitched_atlas1a = [] # free up memory

a = np.where(stitched_atlas1b > 0)   # wm
region_count += 1
stitched_atlas[a] = region_count
stitched_region_labels['number'].append(region_count)
stitched_region_labels['name'].append('PAM50 wm')
stitched_region_labels['abbreviation'].append('cord wm')
stitched_region_labels['voxcount'].append(np.shape(a)[1])
region_number_wm = region_count
stitched_atlas1b = [] # free up memory

#a = np.where(stitched_atlas1c > 0)   # csf
#region_count += 1
#stitched_atlas[a] = region_count
#stitched_region_labels['number'].append(region_count)
#stitched_region_labels['name'].append('PAM50 csf')
#stitched_region_labels['abbreviation'].append('cord csf')

# make a cord ROI template
stitched_cord_roi = np.zeros([x1,y1,z1])
a = np.where( (stitched_atlas == region_number_gm) | (stitched_atlas == region_number_wm))
stitched_cord_roi[a] = 1

save_wm_template = True
if save_wm_template:
    # save a version of the template with cord gm/wm shown
    #------------write out the new list of region numbers and names
    # df = pd.DataFrame(data = stitched_region_labels)
    # output_excel_name = os.path.join(save_data_path, 'wholeCNS_region_definitions_cordwmgm.xlsx')
    # df.to_excel(output_excel_name)
    
    # create NIfTI format output of the stitched together template
    template_filename = os.path.join(save_data_path, 'wholeCNS_wm_map.nii')
    resulting_img = nib.Nifti1Image(stitched_imagewm, new_affine)
    nib.save(resulting_img, template_filename)
    
    # extract sections of the region maps corresponding to the CCBS template
    # from above:
    # the CC/BS region template overlaps the other two templates and is not needed in the actual template
    #stitched_image[xc[0]:xc[1],yc[0]:yc[1],zc[0]:zc[1]] = cord_linear*(bval/cval)/np.amax(cord_linear)
    # stitched_atlasCCBS = stitched_atlas[xc[0]:xc[1],yc[0]:yc[1],zc[0]:zc[1]]
    # # CCBS
    # save_filename_ccbs = os.path.join(save_data_path, 'CCBS_region_map_cordwmgm.nii')
    # resulting_img_ccbs = nib.Nifti1Image(stitched_atlasCCBS, new_affine_ccbs)
    # nib.save(resulting_img_ccbs, save_filename_ccbs)

    # the region maps so far are all 0.5 mm isotropic resolution - convert to 1 mm resolution as well
    # whole CNS
    temp_affine = new_affine
    temp_data = stitched_imagewm
    temp_filename = os.path.join(save_data_path, 'wholeCNS_wm_map_1mm.nii')
    vs = np.array([temp_affine[0,0],temp_affine[1,1],temp_affine[2,2]])
    small_size = np.shape(temp_data)
    new_size_1mm = np.floor(small_size*abs(vs)).astype('int')
    new_affine_1mm = np.array([[np.sign(temp_affine[0,0]), 0.0, 0.0, temp_affine[0,3]],[0.0, np.sign(temp_affine[1,1]), 0.0, temp_affine[1,3]],[0.0, 0.0, np.sign(temp_affine[2,2]), temp_affine[2,3]],[0., 0., 0., 1.]])
    temp_data_small = i3d.resize_3D_nearest(temp_data, new_size_1mm)
    resulting_img_small = nib.Nifti1Image(temp_data_small, new_affine_1mm)
    nib.save(resulting_img_small, temp_filename)
    
    # # CCBS
    # stitched_atlasCCBS = stitched_atlas[xc[0]:xc[1], yc[0]:yc[1], zc[0]:zc[1]]
    temp_affine = new_affine_ccbs
    temp_data = stitched_imagewm[xc[0]:xc[1], yc[0]:yc[1], zc[0]:zc[1]]
    temp_filename = os.path.join(save_data_path, 'CCBS_wm_map.nii')
    vs = np.array([temp_affine[0,0],temp_affine[1,1],temp_affine[2,2]])
    small_size = np.shape(temp_data)
    resulting_img_small = nib.Nifti1Image(temp_data, temp_affine)
    nib.save(resulting_img_small, temp_filename)

    temp_filename = os.path.join(save_data_path, 'CCBS_wm_map_1mm.nii')
    new_size_1mm = np.floor(small_size*abs(vs)).astype('int')
    new_affine_1mm = np.array([[np.sign(temp_affine[0,0]), 0.0, 0.0, temp_affine[0,3]],[0.0, np.sign(temp_affine[1,1]), 0.0, temp_affine[1,3]],[0.0, 0.0, np.sign(temp_affine[2,2]), temp_affine[2,3]],[0., 0., 0., 1.]])
    temp_data_small = i3d.resize_3D_nearest(temp_data, new_size_1mm)
    resulting_img_small = nib.Nifti1Image(temp_data_small, new_affine_1mm)
    nib.save(resulting_img_small, temp_filename)


# top of C1 starts at z = 990, in full CNS template with 0.5 mm resolution
# this is approximately 21 mm from the PMJ, and is the rostral end of the PAM50 template
# the center of the cord is at about y = 175 in 0.5 mm isovoxel template
# the PMJ is at about z = 1010, y = 182

voxpos = np.array([x1/2., 182, 1010, 1])
abspos = new_affine@voxpos
position_record = {'name':['PMJ'], 'x':[abspos[0]], 'y':[abspos[1]], 'z':[abspos[2]]}

# from the German paper: (Lang J, Bartram CT. [Fila radicularia of the ventral and dorsal radices 
# of the human spinal cord]. Gegenbaurs Morphol Jahrb 1982;128(4):417-462)
# Also look at leijsne 2016 about whether or not the cord is actually segmented
# D = [    0     #  PMJ - top of Medulla
#    21.0000     #  C1
#    29.0000     #  C2
#    41.5000     #  C3
#    51.9000     #  C4
#    63.4000     #  C5
#    78.4000     #  C6
#    92.4000     #  C7
#   104.8000     #  C8
#   117.8000];   #  T1
d = np.array([21, 29, 41.5, 51.9, 63.4, 78.4, 92.4, 104.8, 117.8])   # these distances/positions are in mm
cervical_segment_tops = 990 + (21-d)/0.5   # need to account for the voxel size!!!

# 5 lumbar enlargement appears to start around z = 190, and end around z = 130, in full CNS template with 0.5 mm resolution
# 5 Sacral segments appear to be about z = 130 to z = 80
# 12 thoracic segments, starting at 796 (which is 990+ (21-117.8)/0.5), ending at z = 130
thoracic_segment_tops = np.array(np.linspace(796,190,13))
lumbar_segment_tops = np.array(np.linspace(190,130,6))
sacral_segment_tops = np.array(np.linspace(130,80,6))
DVlimit = 180   # A/P dividing line between dorsal and ventral
RLlimit = 197   # R/L dividing line between right and left sides

Lmap = np.zeros([x1,y1,z1])
Lmap[RLlimit:,:,:] = 1   # locations on the left side
Vmap = np.zeros([x1,y1,z1])
Vmap[:,DVlimit:,:] = 1   # locations on the ventral side

# overwrite the gm/wm defintions
region_count -= 2
stitched_region_labels['number'] = stitched_region_labels['number'][:-2]
stitched_region_labels['name'] = stitched_region_labels['name'][:-2]
stitched_region_labels['abbreviation'] = stitched_region_labels['abbreviation'][:-2]
stitched_region_labels['voxcount'] = stitched_region_labels['voxcount'][:-2]

#cord_quadrant_map = np.zeros([x1,y1,z1])
region_defs = {'C':cervical_segment_tops, 'T':thoracic_segment_tops, 'L':lumbar_segment_tops, 'S':sacral_segment_tops}
region_names = ['C','T','L','S']

segmap = np.zeros([x1,y1,z1])
for region in region_names:
    segment_tops = region_defs[region]
    for zz in range(len(segment_tops)-1):
        zz1 = np.round(segment_tops[zz]).astype('int')
        zz2 = np.round(segment_tops[zz+1]).astype('int')
        
        voxpos = np.array([x1/2., 175, segment_tops[zz], 1])
        abspos = new_affine@voxpos
        segname = f'{region}{zz+1}'
        position_record['name'].append(segname)
        position_record['x'].append(abspos[0])
        position_record['y'].append(abspos[1])
        position_record['z'].append(abspos[2])

        segmap[:,:,:] = 0   # refresh this map
        segmap[:,:,zz2:zz1] = 1  # map of segment S/I positions,  z indices move rostral to caudal (decreasing)
        
        cordsegmap = (segmap == 1)*(stitched_cord_roi == 1)
        # right dorsal
        a = np.where( (Lmap == 0)*(Vmap == 0)*cordsegmap)
        region_count += 1
        quadname = f'{region}{zz+1} right dorsal'
        quadabb = f'{region}{zz+1}RD'
        stitched_atlas[a] = region_count
        stitched_region_labels['number'].append(region_count)
        stitched_region_labels['name'].append(quadname)
        stitched_region_labels['abbreviation'].append(quadabb)
        stitched_region_labels['voxcount'].append(np.shape(a)[1])
        print('region_number: ',region_count,' abbrev: ',quadabb,'  full: ',quadname, ' count = ',np.shape(a)[1])
        
        # left dorsal
        a = np.where( (Lmap == 1)*(Vmap == 0)*cordsegmap)
        region_count += 1
        quadname = f'{region}{zz+1} left dorsal'
        quadabb = f'{region}{zz+1}LD'
        stitched_atlas[a] = region_count
        stitched_region_labels['number'].append(region_count)
        stitched_region_labels['name'].append(quadname)
        stitched_region_labels['abbreviation'].append(quadabb)
        stitched_region_labels['voxcount'].append(np.shape(a)[1])
        print('region_number: ',region_count,' abbrev: ',quadabb,'  full: ',quadname, ' count = ',np.shape(a)[1])
        
        # right ventral
        a = np.where( (Lmap == 0)*(Vmap == 1)*cordsegmap)
        region_count += 1
        quadname = f'{region}{zz+1} right ventral'
        quadabb = f'{region}{zz+1}RV'
        stitched_atlas[a] = region_count
        stitched_region_labels['number'].append(region_count)
        stitched_region_labels['name'].append(quadname)
        stitched_region_labels['abbreviation'].append(quadabb)
        stitched_region_labels['voxcount'].append(np.shape(a)[1])
        print('region_number: ',region_count,' abbrev: ',quadabb,'  full: ',quadname, ' count = ',np.shape(a)[1])
        
        # left ventral
        a = np.where( (Lmap == 1)*(Vmap == 1)*cordsegmap)
        region_count += 1
        quadname = f'{region}{zz+1} left ventral'
        quadabb = f'{region}{zz+1}LV'
        stitched_atlas[a] = region_count
        stitched_region_labels['number'].append(region_count)
        stitched_region_labels['name'].append(quadname)
        stitched_region_labels['abbreviation'].append(quadabb)
        stitched_region_labels['voxcount'].append(np.shape(a)[1])
        print('region_number: ',region_count,' abbrev: ',quadabb,'  full: ',quadname, ' count = ',np.shape(a)[1])

Vmap = []
Lmap = []
cordsegmap = []
segmap = []
stitched_cord_roi = []

# save a version of the template with cord quadrants shown
#------------write out the new list of region numbers and names
df = pd.DataFrame(data = stitched_region_labels)
output_excel_name = os.path.join(save_data_path, 'wholeCNS_region_definitions_cordsegments.xlsx')
df.to_excel(output_excel_name)

df2 = pd.DataFrame(data = position_record)
output_excel_name2 = os.path.join(save_data_path, 'cord_region_positions.xlsx')
df2.to_excel(output_excel_name2, sheet_name = 'reference_positions')

# create NIfTI format output of the stitched together template
template_filename = os.path.join(save_data_path, 'wholeCNS_region_map_cordsegments.nii')
resulting_img = nib.Nifti1Image(stitched_atlas, new_affine)
nib.save(resulting_img, template_filename)
resulting_img = []

# extract sections of the region maps corresponding to the CCBS template
# from above:
# the CC/BS region template overlaps the other two templates and is not needed in the actual template
#stitched_image[xc[0]:xc[1],yc[0]:yc[1],zc[0]:zc[1]] = cord_linear*(bval/cval)/np.amax(cord_linear)
stitched_atlasCCBS = stitched_atlas[xc[0]:xc[1],yc[0]:yc[1],zc[0]:zc[1]]
# CCBS
save_filename_ccbs = os.path.join(save_data_path, 'CCBS_region_map_cordsegments.nii')
resulting_img_ccbs = nib.Nifti1Image(stitched_atlasCCBS, new_affine_ccbs)
nib.save(resulting_img_ccbs, save_filename_ccbs)
resulting_img_ccbs = []

# the region maps so far are all 0.5 mm isotropic resolution - convert to 1 mm resolution as well
# whole CNS
temp_affine = new_affine
# temp_data = stitched_atlas
temp_filename = os.path.join(save_data_path, 'wholeCNS_region_map_cordsegments_1mm.nii')
vs = np.array([temp_affine[0,0],temp_affine[1,1],temp_affine[2,2]])
small_size = np.shape(stitched_atlas)
new_size_1mm = np.floor(small_size*abs(vs)).astype('int')
new_affine_1mm = np.array([[np.sign(temp_affine[0,0]), 0.0, 0.0, temp_affine[0,3]],[0.0, np.sign(temp_affine[1,1]), 0.0, temp_affine[1,3]],[0.0, 0.0, np.sign(temp_affine[2,2]), temp_affine[2,3]],[0., 0., 0., 1.]])
temp_data_small = i3d.resize_3D_nearest(stitched_atlas, new_size_1mm)
resulting_img_small = nib.Nifti1Image(temp_data_small, new_affine_1mm)
nib.save(resulting_img_small, temp_filename)

# CCBS
temp_affine = new_affine_ccbs
# temp_data = stitched_atlasCCBS
temp_filename = os.path.join(save_data_path, 'CCBS_region_map_cordsegments_1mm.nii')
vs = np.array([temp_affine[0,0],temp_affine[1,1],temp_affine[2,2]])
small_size = np.shape(stitched_atlasCCBS)
new_size_1mm = np.floor(small_size*abs(vs)).astype('int')
new_affine_1mm = np.array([[np.sign(temp_affine[0,0]), 0.0, 0.0, temp_affine[0,3]],[0.0, np.sign(temp_affine[1,1]), 0.0, temp_affine[1,3]],[0.0, 0.0, np.sign(temp_affine[2,2]), temp_affine[2,3]],[0., 0., 0., 1.]])
temp_data_small = i3d.resize_3D_nearest(stitched_atlasCCBS, new_size_1mm)
resulting_img_small = nib.Nifti1Image(temp_data_small, new_affine_1mm)
nib.save(resulting_img_small, temp_filename)



