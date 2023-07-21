# -*- coding: utf-8 -*-
"""
load_templates.py

This file contains a range of functions for loading anatomical templates, region maps, etc. from
the set of templates etc that are included in the Pantheon repository.

"""
# load_templates


#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------


import os
import image_operations_3D as i3d
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time


# function to find the indices of a list that match a value
def tagfind(tag, rnamelist):
    return [i for i, name in enumerate(rnamelist) if tag in name]


def numindices(vals, vallist):
    indexlist = np.zeros(np.size(vals))
    for i,v in enumerate(vals):
        check = np.where(vallist == v)[0]
        if np.size(check) > 0:
            indexlist[i] = check[0]
        else:
            indexlist[i] = np.nan
    return indexlist.astype('int')


#--------------load_template----------------------------------
#-------------------------------------------------------------
def load_template(region_name, resolution, verbose = False):
    '''
    template_img, regionmap_img, template_affine, anatlabels = ...
        load_template_and_masks(region_name, resolution, verbose)
    :param region_name: choices are 'brain', 'ccbs' or  a range of cord segments separated by the word ''to'' such
            as ''T1toT12''
    :param resolution: choice of template resolution, options are 0.5 and 1 (default)
    :param verbose:  ''True'' or ''False'' (default), for added output displays
    :return: template_img, regionmap_img, template_affine, anatlabels
            these are the template image, a map of regions, the affine matrix for the template, a list of
            region labels corresponding to voxel values in the regionmap_img data

    See also: load_template_and_masks
    '''

    start_time = time.time()
    # templates were created/compiled using stitch_PAM50_ICBM152_May2020.py
    
    # load templates for indicated regions of the CNS
    # include anatomical template, region map, and list of region labels
    # region_name can be brain, ccbs, or any pair of cord segments separated by the word "to", such as "T1toT12"
    
    workingdir = os.path.dirname(os.path.realpath(__file__))
    template_folder = os.path.join(workingdir,'templates')
    print('Loading template from ',template_folder)

    if 'to' in region_name:  # a range of cord segments is specified
        a = region_name.find('to')
        region_name2 = region_name[a + 2:]
        region_name = region_name[:a]
    else:
        region_name2 = ''

    if region_name.lower() == 'thoracic':
        region_name = 'T1'
        region_name2 = 'T12'
    
    regiondef_file = 'wholeCNS_region_definitions_cordsegments.xlsx'
    regionname_file = os.path.join(template_folder, regiondef_file)
    xls = pd.ExcelFile(regionname_file, engine = 'openpyxl')
    df1 = pd.read_excel(xls)
    
    crop = False
    match_affine = False
    
    if region_name.lower() in ['brain','ccbs']:
        
        if region_name.lower() == 'brain':
            if resolution == 0.5:
                template_file = 'brain_template_aligned_with_stitched_PAM50_icbm152_T2.nii.gz'
                regionmap_file = 'wholeCNS_region_map_cordsegments.nii.gz'
                match_affine = False
                crop = False
            else:
                template_file = 'brain_template_aligned_with_stitched_PAM50_icbm152_T2_1mm.nii.gz'
                regionmap_file = 'wholeCNS_region_map_cordsegments_1mm.nii.gz'
                match_affine = True
                crop = False
                
        if region_name.lower() == 'ccbs':
            if resolution == 0.5:
                template_file = 'CCBS_template_aligned_with_stitched_PAM50_icbm152.nii.gz'
                regionmap_file = 'CCBS_region_map_cordsegments.nii.gz'
                match_affine = False
                crop = False
            else:
                template_file = 'CCBS_template_aligned_with_stitched_PAM50_icbm152_1mm.nii.gz'
                regionmap_file = 'CCBS_region_map_cordsegments_1mm.nii.gz'
                match_affine = False
                crop = False
                
    else:
        # allow for any range of cord segments
        if resolution == 0.5:
            template_file = 'stitched_PAM50_icbm152_May2020_T2.nii.gz'
            regionmap_file = 'wholeCNS_region_map_cordsegments.nii.gz'
            match_affine = False
            crop = True
        else:
            template_file = 'stitched_PAM50_icbm152_May2020_T2_1mm.nii.gz'
            regionmap_file = 'wholeCNS_region_map_cordsegments_1mm.nii.gz'
            match_affine = False
            crop = True
            
            
    # load the images and region maps
    template_file = os.path.join(template_folder, template_file)
    regionmap_file = os.path.join(template_folder, regionmap_file)
    
    template_data = nib.load(template_file)
    template_img = template_data.get_fdata()
    template_size = np.shape(template_img)
    template_affine = template_data.affine
    
    regionmap_data = nib.load(regionmap_file)
    regionmap_img = regionmap_data.get_fdata()
#    regionmap_size = np.shape(regionmap_img)
    regionmap_affine = regionmap_data.affine

    namelist = df1['abbreviation']
    numberlist = df1['number']
    
    if match_affine:
        # adjust the regionmap to match the affine matrix of the template
        regionmap_img = i3d.convert_affine_matrices_nearest(regionmap_img, regionmap_affine, template_affine, template_size)
        regionmap_img = np.round(regionmap_img).astype('int')
        regionmap_affine = template_affine
        
    if crop:
        # crop the regionmap and template to match the size/shape of the coordinates provided
        # calculate the new affine matrix to go with the cropped image
        check1 = tagfind(region_name, namelist)
        check2 = tagfind(region_name2, namelist)
        # get the full range of region numbers that is spanned by the region names
        listn1 = np.min([check1,check2])
        listn2 = np.max([check1,check2])
        rn1 = df1['number'][listn1]
        rn2 = df1['number'][listn2]
        a = np.where( (regionmap_img >= rn1) & (regionmap_img <= rn2))
        # now get the range of z coordinates:
        z1 = np.min(a[2])
        z2 = np.max(a[2])
        
        if resolution == 0.5:
            APcenter = 175
            RLrange = 30
            APrange = 30
        else:
            APcenter = 88
            RLrange = 15
            APrange = 15
        
        xs,ys,zs = np.shape(regionmap_img)
        x0 = np.round(xs/2).astype('int')
        # define the range for cropping:
        cc = [x0-RLrange, x0+RLrange, APcenter-APrange, APcenter+APrange, z1, z2]
        # and crop the maps:
        regionmap_img = regionmap_img[cc[0]:cc[1],cc[2]:cc[3],cc[4]:cc[5]]
        template_img = template_img[cc[0]:cc[1],cc[2]:cc[3],cc[4]:cc[5]]
        corner_coords = [cc[0],cc[2],cc[4],1]
        corner_position = template_affine@corner_coords
        template_affine[0:3,3] = corner_position[0:3]
        regionmap_affine = template_affine
    
    v = np.unique(regionmap_img)
    print('found {num} unique values in region map'.format(num = np.size(v)))
#    print('values are ',v)
    
    if verbose:
    # see if atlas lines up with the brain
    # overlay region maps on anatomical
        template_size = np.shape(template_img)
        background = template_img.astype(float)/template_img.max()
        red = copy.deepcopy(background)
        green = copy.deepcopy(background)
        blue = copy.deepcopy(background)
        
        # define colors for each value of v
        nvals = np.size(v)
        redcol = np.linspace(0,1,nvals)
        greencol = redcol[::-1]
        nvals2 = np.floor(nvals/2).astype('int')
        bluecol = np.concatenate((np.linspace(0,1,nvals2), np.linspace(1,0,nvals-nvals2)), axis = 0)
        
        a = np.where(regionmap_img > 0)
        vals = regionmap_img[a]
        ii = numindices(vals, v)
        red[a] = redcol[ii]
        green[a] = greencol[ii]
        blue[a] = bluecol[ii]
        
        sag_slice = np.floor(template_size[0]/2).astype('int')
        tcimg = np.dstack((red[sag_slice,:,:],green[sag_slice,:,:],blue[sag_slice,:,:]))
        fig = plt.figure(21), plt.imshow(tcimg)
        
        ax_slice = np.floor(template_size[2]/2).astype('int')
        tcimg = np.dstack((red[:,:,ax_slice],green[:,:,ax_slice],blue[:,:,ax_slice]))
        fig = plt.figure(22), plt.imshow(tcimg)
        
        cor_slice = np.floor(template_size[1]/2).astype('int')
        tcimg = np.dstack((red[:,cor_slice,:],green[:,cor_slice,:],blue[:,cor_slice,:]))
        fig = plt.figure(23), plt.imshow(tcimg)
    
    
    stop_time = time.time()
    run_time = stop_time-start_time
    # print('loaded template in {h} hours {m} minutes {s} seconds'.format(h = np.floor(run_time/3600), m = np.floor(np.mod(run_time/60, 60)), s = np.round(np.mod(run_time,60))))

    anatlabels = {'names':namelist, 'numbers':numberlist}

    return template_img, regionmap_img, template_affine, anatlabels




#--------------load_template_and_masks----------------------------------
#-------------------------------------------------------------
def load_template_and_masks(region_name, resolution, verbose=False):
    '''
    template_img, regionmap_img, template_affine, anatlabels, wmmap_img, roi_map, gmwm_img = ...
        load_template_and_masks(region_name, resolution, verbose)
    :param region_name: choices are 'brain', 'ccbs' or  a range of cord segments separated by the word ''to'' such
            as ''T1toT12''
    :param resolution: choice of template resolution, options are 0.5 and 1 (default)
    :param verbose:  ''True'' or ''False'' (default), for added output displays
    :return: template_img, regionmap_img, template_affine, anatlabels, wmmap_img, roi_map
            these are the template image, a map of regions, the affine matrix for the template, a list of
            region labels corresponding to voxel values in the regionmap_img data, a map of white matter
            regions, and a region mask for excluding outlying voxels
    '''
    start_time = time.time()
    # templates were created/compiled using stitch_PAM50_ICBM152_May2020.py

    # load templates for indicated regions of the CNS
    # include anatomical template, region map, and list of region labels
    # region_name can be brain, ccbs, or any pair of cord segments separated by the word "to", such as "T1toT12"

    workingdir = os.path.dirname(os.path.realpath(__file__))
    template_folder = os.path.join(workingdir, 'templates')
    print('Loading template from ', template_folder)

    a = region_name.find('to')
    if a > 0:  # a range of cord segments is specified
        region_name2 = region_name[a + 2:]
        region_name = region_name[:a]
    else:
        region_name2 = ''

    if region_name.lower() == 'thoracic':
        region_name = 'T1'
        region_name2 = 'T12'

    regiondef_file = 'wholeCNS_region_definitions_cordsegments.xlsx'
    regionname_file = os.path.join(template_folder, regiondef_file)
    xls = pd.ExcelFile(regionname_file, engine = 'openpyxl')
    df1 = pd.read_excel(xls)

    crop = False
    match_affine = False

    if region_name.lower() in ['brain', 'ccbs']:

        if region_name.lower() == 'brain':
            if resolution == 0.5:
                template_file = 'brain_template_aligned_with_stitched_PAM50_icbm152_T2.nii.gz'
                regionmap_file = 'wholeCNS_region_map_cordsegments.nii.gz'
                wmmap_file = 'wholeCNS_wm_map.nii.gz'
                gm_wm_mask_file = 'none'
                match_affine = True
                crop = False
            else:
                template_file = 'brain_template_aligned_with_stitched_PAM50_icbm152_T2_1mm.nii.gz'
                regionmap_file = 'wholeCNS_region_map_cordsegments_1mm.nii.gz'
                wmmap_file = 'wholeCNS_wm_map_1mm.nii.gz'
                gm_wm_mask_file = 'none'
                match_affine = True
                crop = False

        if region_name.lower() == 'ccbs':
            if resolution == 0.5:
                template_file = 'CCBS_template_aligned_with_stitched_PAM50_icbm152.nii.gz'
                regionmap_file = 'CCBS_region_map_cordsegments.nii.gz'
                wmmap_file = 'CCBS_wm_map.nii.gz'
                gm_wm_mask_file = 'CCBS_cord_wm_gm_mask.nii.gz'
                match_affine = False
                crop = False
            else:
                template_file = 'CCBS_template_aligned_with_stitched_PAM50_icbm152_1mm.nii.gz'
                regionmap_file = 'CCBS_region_map_cordsegments_1mm.nii.gz'
                wmmap_file = 'CCBS_wm_map_1mm.nii.gz'
                gm_wm_mask_file = 'CCBS_cord_wm_gm_mask_1mm.nii.gz'
                match_affine = False
                crop = False

    else:
        # allow for any range of cord segments
        if resolution == 0.5:
            template_file = 'stitched_PAM50_icbm152_May2020_T2.nii.gz'
            regionmap_file = 'wholeCNS_region_map_cordsegments.nii.gz'
            wmmap_file = 'wholeCNS_wm_map.nii.gz'
            gm_wm_mask_file = 'CNS_cord_wm_gm_mask.nii.gz'
            match_affine = False
            crop = True
        else:
            template_file = 'stitched_PAM50_icbm152_May2020_T2_1mm.nii.gz'
            regionmap_file = 'wholeCNS_region_map_cordsegments_1mm.nii.gz'
            wmmap_file = 'wholeCNS_wm_map_1mm.nii.gz'
            gm_wm_mask_file = 'CNS_cord_wm_gm_mask_1mm.nii.gz'
            match_affine = False
            match_affine = True
            crop = True

    # load the images and region maps
    template_file = os.path.join(template_folder, template_file)
    regionmap_file = os.path.join(template_folder, regionmap_file)
    wmmap_file = os.path.join(template_folder, wmmap_file)

    template_data = nib.load(template_file)
    template_img = template_data.get_fdata()
    template_size = np.shape(template_img)
    template_affine = template_data.affine

    regionmap_data = nib.load(regionmap_file)
    regionmap_img = regionmap_data.get_fdata()
    regionmap_affine = regionmap_data.affine

    wmmap_data = nib.load(wmmap_file)
    wmmap_img = wmmap_data.get_fdata()
    wmmap_affine = wmmap_data.affine

    if region_name.lower() != 'brain':
        gm_wm_mask_file = os.path.join(template_folder, gm_wm_mask_file)
        gmwm_data = nib.load(gm_wm_mask_file)
        gmwm_img = gmwm_data.get_fdata()
        gmwm_size = np.shape(gmwm_img)
        gmwm_affine = gmwm_data.affine
    else:
        gmwm_img = []
        gmwm_affine = []

    namelist = df1['abbreviation']
    numberlist = df1['number']

    # print('load_templates:   GMWM mask file is: ',gm_wm_mask_file)

    if match_affine:
        # adjust the regionmap to match the affine matrix of the template
        regionmap_img = i3d.convert_affine_matrices_nearest(regionmap_img, regionmap_affine, template_affine,
                                                            template_size)
        regionmap_img = np.round(regionmap_img).astype('int')
        regionmap_affine = template_affine

        wmmap_img = i3d.convert_affine_matrices_nearest(wmmap_img, wmmap_affine, template_affine,
                                                            template_size)
        wmmap_img = np.round(wmmap_img).astype('int')
        wmmap_affine = template_affine

        if region_name.lower() != 'brain':
            gmwm_img = i3d.convert_affine_matrices_nearest(gmwm_img, gmwm_affine, template_affine,
                                                                template_size)
            gmwm_img = np.round(gmwm_img).astype('int')
            gmwm_affine = template_affine


    if crop:
        # cropping images to correct region sizes
        # crop the regionmap and template to match the size/shape of the coordinates provided
        # calculate the new affine matrix to go with the cropped image
        check1 = tagfind(region_name, namelist)
        check2 = tagfind(region_name2, namelist)
        # get the full range of region numbers that is spanned by the region names
        listn1 = np.min([np.min(check1), np.min(check2)])
        listn2 = np.max([np.max(check1), np.max(check2)])
        rn1 = df1['number'][listn1]
        rn2 = df1['number'][listn2]

        print('rn1 = {}   rn2 = {}'.format(rn1,rn2))
        print('size of regionmap_img is ',np.shape(regionmap_img))

        a = np.where((regionmap_img >= rn1) & (regionmap_img <= rn2))
        # now get the range of z coordinates:
        z1 = np.min(a[2])
        z2 = np.max(a[2])

        if resolution == 0.5:
            APcenter = 175
            RLrange = 30
            APrange = 30
        else:
            APcenter = 88
            RLrange = 15
            APrange = 15

        xs, ys, zs = np.shape(regionmap_img)
        x0 = np.round(xs / 2).astype('int')
        # define the range for cropping:
        cc = [x0 - RLrange, x0 + RLrange, APcenter - APrange, APcenter + APrange, z1, z2]

        regionmap_img = regionmap_img[cc[0]:cc[1], cc[2]:cc[3], cc[4]:cc[5]]
        template_img = template_img[cc[0]:cc[1], cc[2]:cc[3], cc[4]:cc[5]]
        wmmap_img = wmmap_img[cc[0]:cc[1], cc[2]:cc[3], cc[4]:cc[5]]
        corner_coords = [cc[0], cc[2], cc[4], 1]
        corner_position = template_affine @ corner_coords
        template_affine[0:3, 3] = corner_position[0:3]
        regionmap_affine = template_affine
        wmmap_affine = template_affine
        gmwm_img = gmwm_img[cc[0]:cc[1], cc[2]:cc[3], cc[4]:cc[5]]
        gmwm_affine = template_affine

    # mask gray matter regions in the cord
    # if gm and wm regions are defined for cord regions, then apply the gm mask to the regions
    if np.ndim(gmwm_img) == 3:
        xt,yt,zt = np.shape(gmwm_img)
        for z in range(zt):
            tslice = regionmap_img[:,:,z]
            slice = gmwm_img[:,:,z]
            v = list(np.unique(slice))
            if v == [0,1,2]:  # check that this slice has gm and wm regions labeled properly
                ax,ay = np.where(slice != 2)   # find non-gm voxels
                tslice[ax,ay] = 0
                regionmap_img[:, :, z] = tslice


    v = np.unique(regionmap_img)
    if verbose:
        # see if atlas lines up with the brain
        # overlay region maps on anatomical
        template_size = np.shape(template_img)
        background = template_img.astype(float) / template_img.max()
        red = copy.deepcopy(background)
        green = copy.deepcopy(background)
        blue = copy.deepcopy(background)

        # define colors for each value of v
        nvals = np.size(v)
        redcol = np.linspace(0, 1, nvals)
        greencol = redcol[::-1]
        nvals2 = np.floor(nvals / 2).astype('int')
        bluecol = np.concatenate((np.linspace(0, 1, nvals2), np.linspace(1, 0, nvals - nvals2)), axis=0)

        a = np.where(regionmap_img > 0)
        vals = regionmap_img[a]
        ii = numindices(vals, v)
        red[a] = redcol[ii]
        green[a] = greencol[ii]
        blue[a] = bluecol[ii]

        sag_slice = np.floor(template_size[0] / 2).astype('int')
        tcimg = np.dstack((red[sag_slice, :, :], green[sag_slice, :, :], blue[sag_slice, :, :]))
        fig = plt.figure(21), plt.imshow(tcimg)

        ax_slice = np.floor(template_size[2] / 2).astype('int')
        tcimg = np.dstack((red[:, :, ax_slice], green[:, :, ax_slice], blue[:, :, ax_slice]))
        fig = plt.figure(22), plt.imshow(tcimg)

        cor_slice = np.floor(template_size[1] / 2).astype('int')
        tcimg = np.dstack((red[:, cor_slice, :], green[:, cor_slice, :], blue[:, cor_slice, :]))
        fig = plt.figure(23), plt.imshow(tcimg)

    stop_time = time.time()
    run_time = stop_time - start_time
    # print('loaded template in {h} hours {m} minutes {s} seconds'.format(h=np.floor(run_time / 3600),
    #                                                                     m=np.floor(np.mod(run_time / 60, 60)),
    #                                                                     s=np.round(np.mod(run_time, 60))))

    anatlabels = {'names': namelist, 'numbers': numberlist}
    roi_map = (regionmap_img > 0) | (wmmap_img > 0)
    return template_img, regionmap_img, template_affine, anatlabels, wmmap_img, roi_map, gmwm_img




#-----------load gm/wm maps--------(special case)----------------------
def load_wm_maps(region_name, resolution, verbose=False):
    start_time = time.time()
    # templates were created/compiled using stitch_PAM50_ICBM152_May2020.py

    # load gm/wm maps for indicated regions of the CNS
    # region_name can be brain, ccbs, or any pair of cord segments separated by the word "to", such as "T1toT12"

    workingdir = os.path.dirname(os.path.realpath(__file__))
    template_folder = os.path.join(workingdir, 'templates')
    print('Loading template from ', template_folder)

    if 'to' in region_name:  # a range of cord segments is specified
        a = region_name.find('to')
        region_name2 = region_name[a + 2:]
        region_name = region_name[:a]
    else:
        region_name2 = ''

    if region_name.lower() == 'thoracic':
        region_name = 'T1'
        region_name2 = 'T12'

    regiondef_file = 'wholeCNS_region_definitions_cordsegments.xlsx'
    regionname_file = os.path.join(template_folder, regiondef_file)
    xls = pd.ExcelFile(regionname_file, engine = 'openpyxl')
    df1 = pd.read_excel(xls)

    crop = False
    match_affine = False

    if region_name.lower() in ['brain', 'ccbs']:

        if region_name.lower() == 'brain':
            if resolution == 0.5:
                template_file = 'brain_template_aligned_with_stitched_PAM50_icbm152_T2.nii.gz'
                regionmap_file = 'wholeCNS_region_map_cordsegments.nii.gz'
                wmmap_file = 'wholeCNS_wm_map.nii.gz'
                gm_wm_mask_file = 'none'
                match_affine = True
                crop = False
            else:
                template_file = 'brain_template_aligned_with_stitched_PAM50_icbm152_T2_1mm.nii.gz'
                regionmap_file = 'wholeCNS_region_map_cordsegments.nii.gz'
                wmmap_file = 'wholeCNS_wm_map_1mm.nii.gz'
                gm_wm_mask_file = 'none'
                match_affine = True
                crop = False

        if region_name.lower() == 'ccbs':
            if resolution == 0.5:
                template_file = 'CCBS_template_aligned_with_stitched_PAM50_icbm152.nii.gz'
                regionmap_file = 'CCBS_region_map_cordsegments.nii.gz'
                wmmap_file = 'CCBS_wm_map.nii.gz'
                gm_wm_mask_file = 'CCBS_cord_wm_gm_mask.nii.gz'
                match_affine = False
                crop = False
            else:
                template_file = 'CCBS_template_aligned_with_stitched_PAM50_icbm152_1mm.nii.gz'
                regionmap_file = 'CCBS_region_map_cordsegments_1mm.nii.gz'
                wmmap_file = 'CCBS_wm_map_1mm.nii.gz'
                gm_wm_mask_file = 'CCBS_cord_wm_gm_mask_1mm.nii.gz'
                match_affine = False
                crop = False

    else:
        # allow for any range of cord segments
        if resolution == 0.5:
            template_file = 'stitched_PAM50_icbm152_May2020_T2.nii.gz'
            regionmap_file = 'wholeCNS_region_map_cordsegments.nii.gz'
            wmmap_file = 'wholeCNS_wm_map.nii.gz'
            gm_wm_mask_file = 'CNS_cord_wm_gm_mask.nii.gz'
            match_affine = False
            crop = True
        else:
            template_file = 'stitched_PAM50_icbm152_May2020_T2_1mm.nii.gz'
            regionmap_file = 'wholeCNS_region_map_cordsegments_1mm.nii.gz'
            wmmap_file = 'wholeCNS_wm_map_1mm.nii.gz'
            gm_wm_mask_file = 'CNS_cord_wm_gm_mask_1mm.nii.gz'
            match_affine = False
            crop = True

    # load the images and region maps
    template_file = os.path.join(template_folder, template_file)
    regionmap_file = os.path.join(template_folder, regionmap_file)
    wmmap_file = os.path.join(template_folder, wmmap_file)

    template_data = nib.load(template_file)
    template_img = template_data.get_fdata()
    template_size = np.shape(template_img)
    template_affine = template_data.affine

    regionmap_data = nib.load(regionmap_file)
    regionmap_img = regionmap_data.get_fdata()
    regionmap_affine = regionmap_data.affine

    wmmap_data = nib.load(wmmap_file)
    wmmap_img = wmmap_data.get_fdata()
    wmmap_affine = wmmap_data.affine

    if region_name.lower() != 'brain':
        gm_wm_mask_file = os.path.join(template_folder, gm_wm_mask_file)
        gmwm_data = nib.load(gm_wm_mask_file)
        gmwm_img = gmwm_data.get_fdata()
        gmwm_size = np.shape(gmwm_img)
        gmwm_affine = gmwm_data.affine
    else:
        gmwm_img = []
        gmwm_affine = []

    namelist = df1['abbreviation']
    numberlist = df1['number']

    print('load_templates:   GMWM mask file is: ',gm_wm_mask_file)

    if match_affine:
        # adjust the regionmap to match the affine matrix of the template
        regionmap_img = i3d.convert_affine_matrices_nearest(regionmap_img, regionmap_affine, template_affine,
                                                            template_size)
        regionmap_img = np.round(regionmap_img).astype('int')
        regionmap_affine = template_affine

        wmmap_img = i3d.convert_affine_matrices_nearest(wmmap_img, wmmap_affine, template_affine,
                                                            template_size)
        wmmap_img = np.round(wmmap_img).astype('int')
        wmmap_affine = template_affine

        if region_name.lower() != 'brain':
            gmwm_img = i3d.convert_affine_matrices_nearest(gmwm_img, gmwm_affine, template_affine,
                                                                template_size)
            gmwm_img = np.round(gmwm_img).astype('int')
            gmwm_affine = template_affine

    if crop:
        # crop the regionmap and template to match the size/shape of the coordinates provided
        # calculate the new affine matrix to go with the cropped image
        check1 = tagfind(region_name, namelist)
        check2 = tagfind(region_name2, namelist)

        # get the full range of region numbers that is spanned by the region names
        listn1 = np.min(np.array(check1+check2))
        listn2 = np.max(np.array(check1+check2))
        rn1 = df1['number'][listn1]
        rn2 = df1['number'][listn2]
        a = np.where((regionmap_img >= rn1) & (regionmap_img <= rn2))
        # now get the range of z coordinates:
        z1 = np.min(a[2])
        z2 = np.max(a[2])

        if resolution == 0.5:
            APcenter = 175
            RLrange = 30
            APrange = 30
        else:
            APcenter = 88
            RLrange = 15
            APrange = 15

        xs, ys, zs = np.shape(regionmap_img)
        x0 = np.round(xs / 2).astype('int')
        # define the range for cropping:
        cc = [x0 - RLrange, x0 + RLrange, APcenter - APrange, APcenter + APrange, z1, z2]

        # and crop the remaining maps if needed:
        regionmap_img = regionmap_img[cc[0]:cc[1], cc[2]:cc[3], cc[4]:cc[5]]
        template_img = template_img[cc[0]:cc[1], cc[2]:cc[3], cc[4]:cc[5]]
        wmmap_img = wmmap_img[cc[0]:cc[1], cc[2]:cc[3], cc[4]:cc[5]]
        corner_coords = [cc[0], cc[2], cc[4], 1]
        corner_position = template_affine @ corner_coords
        template_affine[0:3, 3] = corner_position[0:3]
        regionmap_affine = template_affine
        wmmap_affine = template_affine
        gmwm_img = gmwm_img[cc[0]:cc[1], cc[2]:cc[3], cc[4]:cc[5]]
        gmwm_affine = template_affine


    stop_time = time.time()
    run_time = stop_time - start_time
    print('loaded wm map in {h} hours {m} minutes {s} seconds'.format(h=np.floor(run_time / 3600),
                            m=np.floor(np.mod(run_time / 60, 60)), s=np.round(np.mod(run_time, 60))))

    roi_map = (regionmap_img > 0) | (wmmap_img > 0)
    return wmmap_img, template_img, template_affine, roi_map, gmwm_img


def load_brain_template(templatefilename):
    workingdir = os.path.dirname(os.path.realpath(__file__))
    brain_templates_folder = os.path.join(workingdir, 'braintemplates')
    templatename = os.path.join(brain_templates_folder, templatefilename)
    input_img = nib.load(templatename)
    template_affine = input_img.affine
    template_img = input_img.get_fdata()
    template_img = template_img / np.max(template_img)

    roi_map = template_img > 0.1   # threshold the image at some base level

    return template_img, template_affine, roi_map