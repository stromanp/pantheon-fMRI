# -*- coding: utf-8 -*-
"""
Created on Sat May 30 13:29:16 2020

@author: stroman
"""

import numpy as np
import os
import nibabel as nib
import image_operations_3D as i3d


def load_anatomical_template(region_number, affine_matrix, output_size):
    #    save_data_path_brain = r'C:\stroman\MNI_PAM50_templates_2020\mni_icbm152_nlin_sym_09a'

    save_data_path = r'C:\stroman\MNI_PAM50_templates_2020\CITtemplates'
    namelist = ['CIT168toMNI152_prob_atlas_bilat_1mm__(volume 3).nii.gz', \
                'CIT168toMNI152_prob_atlas_bilat_1mm__(volume 4).nii.gz',
                'CIT168toMNI152_prob_atlas_bilat_1mm__(volume 7).nii.gz',
                'CIT168toMNI152_prob_atlas_bilat_1mm__(volume 10).nii.gz',
                'CIT168toMNI152_prob_atlas_bilat_1mm__(volume 11).nii.gz',
                'CIT168toMNI152_prob_atlas_bilat_1mm__(volume 14).nii.gz']

    regionlabels = ['NAC', 'Amygdala', 'Substantia Nigra', 'PBN', 'VTA', 'Hypothalamus']

    save_data_path2 = r'C:\stroman\MNI_PAM50_templates_2020\AAN_MNI152_1mm_v1p0'
    namelist2 = ['AAN_PAG_MNI152_1mm_v1p0_20150630.nii', \
                 'AAN_VTA_MNI152_1mm_v1p0_20150630.nii',
                 'AAN_LC_MNI152_1mm_v1p0_20150630.nii']
    regionlabels2 = ['PAG', 'VTA', 'LC']

    save_data_path3 = r'C:\stroman\MNI_PAM50_templates_2020\Eckert_templates'
    namelist3 = ['LC_2SD_BINARY_TEMPLATE.nii']
    regionlabels3 = ['LC']

    # 6 regions in first set, 3 regions in second set, 1 region in third set
    if region_number < 6:
        rnum = region_number
        setnumber = 1
    else:
        if region_number < 9:
            rnum = region_number - 6
            setnumber = 2
        else:
            if region_number < 10:
                rnum = region_number - 9
                setnumber = 3
            else:
                rnum = 0
                setnumber = 3

    # set the number and name of list to read
    if setnumber == 1:
        filename = os.path.join(save_data_path, namelist[rnum])
        regionname = regionlabels[rnum]
    if setnumber == 2:
        filename = os.path.join(save_data_path2, namelist2[rnum])
        regionname = regionlabels2[rnum]
    if setnumber == 3:
        print('namelist3 = ',namelist3)
        print('rnum = ',rnum)
        filename = os.path.join(save_data_path3, namelist3[rnum])
        regionname = regionlabels3[rnum]

    template_img = nib.load(filename)
    template_data = template_img.get_data()
    #    template_hdr = template_img.header1
    template_affine = template_img.affine
    #    template_size = np.shape(template_data)

    # line up with MNI brain template
    # filenameb = os.path.join(save_data_path_brain, 'mni_icbm152_T2_tal_nlin_sym_09a.nii')
    # brain_img = nib.load(filenameb)
    # brain_data = brain_img.get_data()
    # brain_hdr = brain_img.header
    # brain_size = np.shape(brain_data)
    # brain_affine = brain_img.affine
    # brain_affine matrix is off R/L
    # brain_affine[0,3] += -10

    # match the anat map with the brain template affine matrix
    # anatmap = i3d.convert_affine_matrices(template_data, template_affine, brain_affine, brain_size)
    anatmap = i3d.convert_affine_matrices(template_data, template_affine, affine_matrix, output_size)
    anatmap = np.round(anatmap)

    return anatmap, regionname