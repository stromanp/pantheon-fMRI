"""
=========================
Affine Registration in 3D
=========================

This example explains how to compute an affine transformation to register two
3D volumes by maximization of their Mutual Information [Mattes03]_. The
optimization strategy is similar to that implemented in ANTS [Avants11]_.

We will do this twice. The first part of this tutorial will walk through the
details of the process with the object-oriented interface implemented in
the ``dipy.align`` module. The second part will use a simplified functional
interface.
"""

# from os.path import join as pjoin
import numpy as np
# from dipy.viz import regtools
from dipy.align import affine_registration
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)

import image_operations_3D as i3d
import time
import matplotlib.pyplot as plt
import nibabel as nib
import os
import copy
import pandas as pd

def test_dipy_brain_registration():

    # set the reference image
    img_template = r'C:\stroman\spm12\spm12\canonical\avg152T2.nii'
    # ref_data, ref_affine = i3d.load_and_scale_nifti(img_template)
    input_img = nib.load(img_template)
    ref_affine = input_img.affine
    ref_hdr = input_img.header
    ref_data = input_img.get_fdata()
    ref_data = ref_data / np.max(ref_data)

    # set the moving image
    input_name = r'C:\fMRI-EEG_shared_data_X1\sub-xp101\func\sub-xp101_task-motorloc_bold.nii.gz'
    # img_data, img_affine = i3d.load_and_scale_nifti(input_image)
    input_img = nib.load(input_name)
    img_affine = input_img.affine
    img_hdr = input_img.header
    img_data = input_img.get_fdata()
    img_data = img_data / np.max(img_data)

    affine_data_filename = r'C:\fMRI-EEG_shared_data_X1\sub-xp101\func\sub-xp101_task-motorloc_bold_affine.npy'

    # coregistration-----------------------------
    input_name = input_image
    coreg_prefix = 'c'
    coreged_images, affine_record = dipy_brain_coregistration(img_data, img_affine, ref_volume=3, verbose = True)

    # save the coregistered nifti images ...
    resulting_img = nib.Nifti1Image(coreged_images, img_affine)
    p,f_full = os.path.split(input_name)
    f,e = os.path.splitext(f_full)
    ext = '.nii'
    if e == '.gz':
        f, e2 = os.path.splitext(f)  # split again
        ext = '.nii.gz'
    output_niiname = os.path.join(p,coreg_prefix+f+ext)
    nib.save(resulting_img, output_niiname)

    display_slices(coreged_images[:,:,:,3], coreged_images[:,:,:,-1], axis=0, image_number = 5)
    display_slices(coreged_images[:,:,:,3], coreged_images[:,:,:,-1], axis=1, image_number = 6)
    display_slices(coreged_images[:,:,:,3], coreged_images[:,:,:,-1], axis=2, image_number = 7)

    #---reload data if needed---------------
    reload_coreg_data = True
    if reload_coreg_data:
        p,f_full = os.path.split(input_name)
        f,e = os.path.splitext(f_full)
        ext = '.nii'
        if e == '.gz':
            f, e2 = os.path.splitext(f) # split again
            ext = '.nii.gz'
        coreg_prefix = 'c'
        niiname = os.path.join(p,coreg_prefix+f+ext)

        input_img = nib.load(niiname)
        coreged_images = input_img.get_fdata()
        img_affine = input_img.affine
        coreged_hdr = input_img.header
    else:
        niiname = output_niiname

    # compute normalization-----------------------------------------
    input_name = niiname
    img_data = copy.deepcopy(coreged_images)

    img_data_norm = img_data[:, :, :, 3]
    img_data_norm = img_data_norm / np.max(img_data_norm)

    print('starting normalization calculation ....')
    norm_brain_img, norm_brain_affine = dipy_compute_brain_normalization(img_data_norm, img_affine, ref_data, ref_affine)
    print('finished normalization calculation ....')
    # save norm_brain_affine for later use...
    np.save(affine_data_filename,{'norm_affine_transformation':norm_brain_affine})

    # apply normalization---------------------------------------------
    norm_prefix = 'p'
    print('starting applying normalization ....')
    norm_brain_data = dipy_apply_brain_normalization(img_data, norm_brain_affine, verbose = True)
    print('finished applying normalization ....')
    # save the normalized nifti images ...

    resulting_img = nib.Nifti1Image(norm_brain_data, norm_brain_affine.affine)
    p,f_full = os.path.split(input_name)
    f,e = os.path.splitext(f_full)
    ext = '.nii'
    if e == '.gz':
        f, e2 = os.path.splitext(f) # split again
        ext = '.nii.gz'
    output_niiname = os.path.join(p,norm_prefix+f+ext)
    # need to write the image data out in a smaller format to save disk space ...
    nib.save(resulting_img, output_niiname)

    display_slices(ref_data, norm_brain_data[:,:,:,10], axis=0, image_number = 11)
    display_slices(ref_data, norm_brain_data[:,:,:,10], axis=1, image_number = 12)
    display_slices(ref_data, norm_brain_data[:,:,:,10], axis=2, image_number = 13)


def dipy_compute_brain_normalization(img_data, img_affine, ref_data, ref_affine, level_iters = [10000, 1000, 100], sigmas = [3.0, 1.0, 0.0], factors = [4,2,1], nbins=32):
    # apply registration/normalization steps in a sequence
    # use this method instead of the dipy pipeline options in order to keep track of the
    # final transformation that needs to be saved and later applied to all volumes of the time-series
    #
    # # To avoid getting stuck at local optima, and to accelerate convergence, dipy uses
    # # a multi-resolution strategy (similar to ANTS [Avants11]_) by building a Gaussian
    # # Pyramid. First, specify how many resolutions to use. This is indirectly specified
    # # by providing a list of the number of iterations to perform at each resolution.
    # # Default is to specify 3 resolutions and 10000 iterations at the coarsest resolution,
    # # 1000 iterations at medium resolution and 100 at the finest.
    # level_iters = [10000, 1000, 100]
    #
    # # To compute the Gaussian pyramid, the original image is first smoothed at each
    # # level of the pyramid using a Gaussian kernel with the requested sigma. A good
    # # initial choice is [3.0, 1.0, 0.0]
    # sigmas = [3.0, 1.0, 0.0]
    #
    # # specify the sub-sampling factors. A good configuration is [4, 2, 1],
    # # this means the shape of the coarsest image will be divded by 4, the shape in
    # # the middle resolution is divided by 2, etc.
    # factors = [4, 2, 1]
    #

    print('dipy_compute_brain_normalization:   ....')
    print('   input image data type:  {}'.format(type(img_data)))
    print('   template image data type:  {}'.format(type(ref_data)))

    # rough normalization based on center of mass
    print('initial rough normalization based on center of mass ...{}'.format(time.ctime()))
    c_of_mass = transform_centers_of_mass(ref_data, ref_affine,img_data, img_affine)
    transformed = c_of_mass.transform(img_data)

    # refine with affine transformation
    print('affine transformation ...{}'.format(time.ctime()))
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    # default settings
    # level_iters = [10000, 1000, 100]
    # sigmas = [3.0, 1.0, 0.0]
    # factors = [4, 2, 1]

    affreg = AffineRegistration(metric=metric,level_iters=level_iters,sigmas=sigmas,factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(ref_data, img_data, transform, params0,
                                  ref_affine, img_affine, starting_affine=starting_affine)

    # look at result
    # transformed = translation.transform(img_data)

    # refine with rigid transformation
    print('refine with a rigid transformation ...{}'.format(time.ctime()))
    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(ref_data, img_data, transform, params0,
                            ref_affine, img_affine, starting_affine=starting_affine)

    # transformed = rigid.transform(img_data)

    # refine with full affine transform
    print('refine again with an affine transformation ...{}'.format(time.ctime()))
    transform = AffineTransform3D()
    params0 = None
    starting_affine = rigid.affine
    affine = affreg.optimize(ref_data, img_data, transform, params0,
                             ref_affine, img_affine, starting_affine=starting_affine)

    # look at the final result
    transformed = affine.transform(img_data)
    print('finished computing normalization parameters ...{}'.format(time.ctime()))

    print('dipy_compute_brain_normalization:   ....')
    print('   output image data type:  {}'.format(type(transformed)))

    return transformed, affine


def dipy_apply_brain_normalization(input_data, norm_affine, verbose = False):
    xs,ys,zs,ts = np.shape(input_data)

    # define a list of tranformation steps
    pipeline = ["center_of_mass", "translation"]

    for tt in range(ts):
        if verbose: print('applying normalization to volume {} of {}'.format(tt+1,ts))
        img_data = input_data[:,:,:,tt]
        resampled = norm_affine.transform(img_data)
        if tt == 0:
            xs,ys,zs = np.shape(resampled)
            output_data = np.zeros((xs,ys,zs,ts))
        output_data[:,:,:,tt] = resampled

    return output_data


def dipy_brain_coregistration(input_data, input_affine, ref_volume = 3, verbose = False):
    # these options are only for testing coregistration methods
    apply_affine = False
    apply_rigid = False

    xs,ys,zs,ts = np.shape(input_data)
    ref_data = input_data[:,:,:,ref_volume]
    ref_affine = input_affine
    img_affine = input_affine
    output_data = np.zeros(np.shape(input_data))

    # using the dipy toolbox apply the transformations much like for normalization,
    # except that data are registered to one volume in the time-series with fewer
    # steps that normalization because the images are expected to match very well

    affine_record = []
    for tt in range(ts):
        if verbose: print('coregistering volume {} of {}   {}'.format(tt+1,ts,time.ctime()))
        img_data = input_data[:,:,:,tt]

        # rough normalization based on center of mass
        current_affine = transform_centers_of_mass(ref_data, ref_affine, img_data, img_affine)
        transformed = current_affine.transform(img_data)

        if apply_affine:
            # refine with affine transformation
            print('affine transformation ...{}'.format(time.ctime()))
            sampling_prop = None
            metric = MutualInformationMetric(nbins, sampling_prop)

            # set these values to be inputs?
            level_iters = [10000, 1000, 100]
            sigmas = [3.0, 1.0, 0.0]
            factors = [4, 2, 1]

            affreg = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors)

            transform = TranslationTransform3D()
            params0 = None
            starting_affine = current_affine.affine
            current_affine = affreg.optimize(ref_data, img_data, transform, params0,
                                          ref_affine, img_affine, starting_affine=starting_affine)

        if apply_rigid:
            # refine with rigid transformation
            print('refine with a rigid transformation ...{}'.format(time.ctime()))
            transform = RigidTransform3D()
            params0 = None
            starting_affine = current_affine.affine
            current_affine = affreg.optimize(ref_data, img_data, transform, params0,
                                    ref_affine, img_affine, starting_affine=starting_affine)

        # look at the final result
        transformed = current_affine.transform(img_data)

        output_data[:,:,:,tt] = transformed
        affine_record.append(current_affine)

    return output_data, affine_record



def dipy_brain_coregister_onevolume(img_data, ref_data, input_affine, verbose = False):
    # these options are only for testing coregistration methods
    apply_affine = False
    apply_rigid = False

    # xs,ys,zs,ts = np.shape(input_data)
    # ref_data = input_data[:,:,:,ref_volume]
    # ref_affine = input_affine
    # img_affine = input_affine
    # output_data = np.zeros(np.shape(input_data))

    # using the dipy toolbox apply the transformations much like for normalization,
    # except that data are registered to one volume in the time-series with fewer
    # steps that normalization because the images are expected to match very well

    # rough normalization based on center of mass
    current_affine = transform_centers_of_mass(ref_data, input_affine, img_data, input_affine)
    # transformed = current_affine.transform(img_data)

    if apply_affine:
        # refine with affine transformation
        print('affine transformation ...{}'.format(time.ctime()))
        sampling_prop = None
        metric = MutualInformationMetric(nbins, sampling_prop)

        # set these values to be inputs?
        level_iters = [10000, 1000, 100]
        sigmas = [3.0, 1.0, 0.0]
        factors = [4, 2, 1]

        affreg = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors)

        transform = TranslationTransform3D()
        params0 = None
        starting_affine = current_affine.affine
        current_affine = affreg.optimize(ref_data, img_data, transform, params0,
                                      input_affine, input_affine, starting_affine=starting_affine)

    if apply_rigid:
        # refine with rigid transformation
        print('refine with a rigid transformation ...{}'.format(time.ctime()))
        transform = RigidTransform3D()
        params0 = None
        starting_affine = current_affine.affine
        current_affine = affreg.optimize(ref_data, img_data, transform, params0,
                                input_affine, input_affine, starting_affine=starting_affine)

    # look at the final result
    transformed = current_affine.transform(img_data)

    return transformed, current_affine


def brain_coregistration(niiname, nametag, coregistered_prefix = 'c'):
    input_img = nib.load(niiname)
    img_affine = input_img.affine
    input_hdr = input_img.header
    img_data = input_img.get_fdata()
    # input_data = input_img.dataobj[..., volume]

    xs,ys,zs,ts = np.shape(img_data)
    ref_volume = 3
    ref_data = img_data[:,:,:,ref_volume]
    # ref_data = input_img.dataobj[..., ref_volume]
    # coreged_images, affine_record = dipy_brain_coregistration(img_data, img_affine, ref_volume=ref_volume, verbose = True)

    affine_record = []
    for tt in range(ts):
        print('registering volume {} of {}'.format(tt+1,ts))
        transformed, current_affine = dipy_brain_coregister_onevolume(img_data[:,:,:,tt], ref_data, img_affine, verbose=False)
        img_data[:,:,:,tt] = transformed
        affine_record.append(current_affine)

    # save the coregistered nifti images ...
    resulting_img = nib.Nifti1Image(img_data, img_affine)
    p,f_full = os.path.split(niiname)
    f,e = os.path.splitext(f_full)
    ext = '.nii'
    if e == '.gz':
        f, e2 = os.path.splitext(f) # split again
        ext = '.nii.gz'
    output_niiname = os.path.join(p,coregistered_prefix+f+ext)
    nib.save(resulting_img, output_niiname)

    Qcheck = np.zeros(ts)
    for tt in range(ts):
        # check quality of result
        test_img = img_data[:,:,:,tt]
        R = np.corrcoef(ref_data.flatten(), test_img.flatten())
        Qcheck[tt] = R[0, 1]

    # get the motion parameters from the affine record while we are here
    motion_record = np.zeros((3,ts))
    for tt in range(ts):
        affine1 = affine_record[tt].affine
        motion_record[:,tt] = affine1[:3, 3]

    motion_parameters = {'dx':motion_record[0,:]-motion_record[0,0],
                         'dy': motion_record[1, :] - motion_record[1, 0],
                         'dz': motion_record[2, :] - motion_record[2, 0]}
    motiondata = pd.DataFrame(data = motion_parameters)
    output_motiondata_xlname = os.path.join(p, 'motiondata'+nametag+'.xlsx')
    motiondata.to_excel(output_motiondata_xlname, sheet_name='motion_data')   # write it out to excel

    # write result as new NIfTI format image set
    pname, fname = os.path.split(niiname)
    fnameroot, ext = os.path.splitext(fname)
    if e == '.gz':
        fnameroot, e2 = os.path.splitext(fnameroot) # split again
    # define names for outputs
    coregdata_name = os.path.join(pname, 'coregdata'+nametag+'.npy')
    np.save(coregdata_name, {'affine_record':affine_record,'motion_record':motion_record})

    return output_niiname, Qcheck


# display brain slices in comparison
def display_slices(volume1, volume2, axis=0, image_number = 10):
    xs1,ys1,zs1 = np.shape(volume1)
    xs2,ys2,zs2 = np.shape(volume2)

    x01,y01,z01 = np.floor(np.array([xs1,ys1,zs1])/2).astype(int)
    x02,y02,z02 = np.floor(np.array([xs2,ys2,zs2])/2).astype(int)

    if axis == 0:
        img1 = volume1[x01,:,:]
        img2 = volume2[x02,:,:]
    if axis == 1:
        img1 = volume1[:,y01,:]
        img2 = volume2[:,y02,:]
    if axis == 2:
        img1 = volume1[:,:,z01]
        img2 = volume2[:,:,z02]

    img1 = img1/np.max(img1)
    img2 = img2/np.max(img2)
    xs,ys = np.shape(img1)
    display1 = np.concatenate((img1[:,:,np.newaxis],img1[:,:,np.newaxis],img1[:,:,np.newaxis]),axis = 2)
    display2 = np.concatenate((img2[:,:,np.newaxis],img2[:,:,np.newaxis],img2[:,:,np.newaxis]),axis = 2)
    overlay = np.concatenate((img1[:,:,np.newaxis],img2[:,:,np.newaxis],np.zeros((xs,ys,1))),axis = 2)

    fig = plt.figure(image_number)
    ax1 = plt.subplot(1,3,1)
    ax1.imshow(display1)

    ax2= plt.subplot(1,3,2)
    ax2.imshow(overlay)

    ax3 = plt.subplot(1,3,3)
    ax3.imshow(display2)
