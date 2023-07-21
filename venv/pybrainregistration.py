"""
pybrainregistration.py

This set of functions is for spatial normalization of brain MRI data for the purposes of
fMRI analysis.  The methods used for normalizing brain data are different from those used
for the brainstem and spinal cord regions.

The methods here are adapted from various 3D affine registration methods, all of which were
based on methods that are similar to the normalization used in ANTS.

The methods in this file are not original and were not developed by P. Stroman.
This was done in an effort to not reinvent the wheel.

"""

# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# "Pantheon" is a python software repository for complete analysis of functional
# magnetic resonance imaging data at all level of the central nervous system,
# including the brain, brainstem, and spinal cord.
#
# The software in this repository was written by P. Stroman, and the bulk of the methods in this
# package have been developed by P. W. Stroman, Queen's University at Kingston, Ontario, Canada.
#
# Some of the methods have been adapted from other freely available packages
# as noted in the documentation.
#
# This software is for research purposes only, and no guarantees are given that it is
# free of bugs or errors.
#
# Use this software as needed, with the condition that you reference it in any
# published works or presentations, with the following citations:
#
# Proof-of-concept of a novel structural equation modelling approach for the analysis of
# functional MRI data applied to investigate individual differences in human pain responses
# P. W. Stroman, J. M. Powers, G. Ioachim
# Human Brain Mapping, 44:2523–2542 (2023). https://doi.org/10.1002/hbm.26228
#
#  Ten key insights into the use of spinal cord fMRI
#  J. M Powers, G. Ioachim, P. W. Stroman
#  Brain Sciences 8(9), (DOI: 10.3390/brainsci8090173 ) 2018.
#
#  Validation of structural equation modeling (SEM) methods for functional MRI data acquired in the human brainstem and spinal cord
#  P. W. Stroman
#  Critical Reviews in Biomedical Engineering 44(4): 227-241 (2016).
#
#  Assessment of data acquisition parameters, and analysis techniques for noise
#  reduction in spinal cord fMRI data
#  R.L. Bosma & P.W. Stroman
#  Magnetic Resonance Imaging, 2014 (10.1016/j.mri.2014.01.007).
#
# also see https://www.queensu.ca/academia/stromanlab/
#
# Patrick W. Stroman, Queen's University, Centre for Neuroscience Studies
# stromanp@queensu.ca
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

import numpy as np
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


def dipy_compute_brain_normalization(img_data, img_affine, ref_data, ref_affine, level_iters = [10000, 1000, 100], sigmas = [3.0, 1.0, 0.0], factors = [4,2,1], nbins=32):
    # apply registration/normalization steps in a sequence
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
