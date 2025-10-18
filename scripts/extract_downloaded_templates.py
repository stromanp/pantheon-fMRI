# sys.path.extend([r'C:\Users\Stroman\PycharmProjects\pantheon\scripts'])

# -*- coding: utf-8 -*-
"""
Created on Sat May 30 13:29:16 2020

@author: stroman
"""

import numpy as np
import os
import nibabel as nib
import scipy.ndimage as nd
import image_operations_3D as i3d
import matplotlib.pyplot as plt
import matplotlib.image as img
import copy


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



# convert templates to region maps - Coulomb paper and You_Park paper Oct 2025
def copy_template_bits():
    template_img_name = r'Y:\BigAnatomicalAnalysis\CCBS_region_map_cordsegments_Aug2025.nii.gz'
    template_data = nib.load(template_img_name)
    template_img = template_data.get_fdata()
    template_size = np.shape(template_img)
    template_affine = template_data.affine

    # for saving templates later...
    # resulting_img = nib.Nifti1Image(output_images, new_affine)
    # nib.save(resulting_img, outputname)

    source_directory = r'Y:\BigAnatomicalAnalysis\brainstem_template'

    count = 0
    for dirName, subdirList, fileList in os.walk(source_directory):
        for filename in fileList:
            f,e = os.path.splitext(filename)
            if '_V2' in filename:
                print('already converted {}'.format(filename))
            else:
                if e == '.png':
                    if 'compiling_brainstem' in filename:
                        count += 1
                        postag = f[20:]
                        signtag = postag[:3]
                        valtag = postag[3:]
                        valtag = valtag.replace('p','.')
                        valtag = valtag.replace('mm', '')
                        posval = float(valtag)
                        if signtag == 'neg':
                            posval *= -1.0
                        zcoord = np.round(340. + 2.336*(posval+13)).astype(int)
                        print('{:.2f}  z = {:d}   file: {}'.format(posval, zcoord, filename))

                        template_slice = template_img[:,:,zcoord]
                        template_slice_big = resize_2D(template_slice, [10*template_size[0], 10*template_size[1]])
                        # rotate 90 degrees
                        template_slice_bigR = (template_slice_big.T)[::-1,:]
                        template_slice_bigR /= np.max(template_slice_bigR)  # scale values
                        xt,yt = np.shape(template_slice_bigR)
                        template_slice_big_col = np.concatenate((template_slice_bigR[:,:,np.newaxis],template_slice_bigR[:,:,np.newaxis],template_slice_bigR[:,:,np.newaxis]), axis = 2)

                        wholename = os.path.join(source_directory, filename)
                        png_img = img.imread(wholename)

                        # add the template image to the PNG image
                        png_img[:xt,:yt,:] = copy.deepcopy(template_slice_big_col)

                        f,e = os.path.splitext(filename)
                        newwholename = os.path.join(source_directory, f + '_V2' + e)
                        plt.imsave(newwholename, png_img)


def compile_temp_from_bits():
    template_img_name = r'Y:\BigAnatomicalAnalysis\CCBS_region_map_cordsegments_Aug2025.nii.gz'
    outputname = r'Y:\BigAnatomicalAnalysis\CCBS_region_map_sections_Oct2025.nii.gz'

    template_data = nib.load(template_img_name)
    template_img = template_data.get_fdata()
    template_size = np.shape(template_img)
    template_affine = template_data.affine

    new_template = np.zeros(template_size)
    xt = 10*template_size[0]
    yt = 10*template_size[1]

    new_region_names = ['NRM', 'NGC', 'LC', 'NTS', 'PAG', 'PBN']
    color_list = [[255,0,0],[0,255,0],[255,255,0],[0,0,255],[0,255,255],[255,128,0]]
    nregions = len(new_region_names)

    # for saving templates later...
    # resulting_img = nib.Nifti1Image(output_images, new_affine)
    # nib.save(resulting_img, outputname)

    source_directory = r'Y:\BigAnatomicalAnalysis\brainstem_template'

    count = 0
    for dirName, subdirList, fileList in os.walk(source_directory):
        for filename in fileList:
            f,e = os.path.splitext(filename)

            if e == '.png':
                if ('compiling_brainstem' in filename) and ('_V3' in filename):
                    count += 1
                    postag = f[20:]
                    signtag = postag[:3]
                    valtag = postag[3:]
                    valtag = valtag.replace('p','.')
                    valtag = valtag.replace('mm', '')
                    valtag = valtag.replace('_V3', '')
                    posval = float(valtag)
                    if signtag == 'neg':
                        posval *= -1.0
                    zcoord = np.round(340. + 2.336*(posval+13)).astype(int)
                    print('{:.2f}  z = {:d}   file: {}'.format(posval, zcoord, filename))

                    # read the png image
                    wholename = os.path.join(source_directory, filename)
                    png_img = img.imread(wholename)
                    template_slice_ref = png_img[:yt,:xt,:]

                    template_slice = np.zeros((yt,xt))

                    # find the regions in the slice
                    dd = 0.05
                    for rr in range(nregions):
                        col = np.array(color_list[rr])/255.
                        redcheck = (template_slice_ref[:,:,0] > col[0]-dd) & (template_slice_ref[:,:,0] < col[0]+dd)
                        greencheck = (template_slice_ref[:,:,1] > col[1]-dd) & (template_slice_ref[:,:,1] < col[1]+dd)
                        bluecheck = (template_slice_ref[:,:,2] > col[2]-dd) & (template_slice_ref[:,:,2] < col[2]+dd)
                        cc = np.where(redcheck & greencheck & bluecheck)
                        template_slice[cc] = rr+1
                    # resize template_slice
                    # template_slice_small = resize_2D(template_slice,[template_size[0],template_size[1]])

                    # rotate 90 degrees
                    template_slicer = (template_slice[::-1,:]).T
                    template_slice_small = resize_3D_nearest(template_slicer[:,:,np.newaxis], [template_size[0],template_size[1],1])

                    new_template[:,:,zcoord] = copy.deepcopy(template_slice_small[:,:,0])


    # save result as NIfTI file
    resulting_img = nib.Nifti1Image(new_template, template_affine)
    nib.save(resulting_img, outputname)



def resize_2D(input_data, newsize_input, mode = 'constant'):
    original_size = np.shape(input_data)
    if np.ndim(newsize_input) == 0:
        newsize = np.floor(np.array(original_size)*newsize_input).astype(int)
    else:
        newsize = newsize_input

    matrix = [ [original_size[0]/newsize[0], 0],
              [0, original_size[1]/newsize[1]] ]
    output_image = nd.affine_transform(input_data, matrix, offset=0.0,
            output_shape=newsize, order=1, mode=mode, cval=0.0, prefilter=True)
    return output_image


def warp_image_nearest(input_image, mapX, mapY, mapZ):
    # warp images, but return the nearest-neighbor result
    output_size = mapX.shape
    # interpolate points in output_image

    data = input_image

    data = data.astype(float)
    xs, ys, zs = data.shape

    mapXr = np.reshape(mapX, np.prod(mapX.shape))
    mapYr = np.reshape(mapY, np.prod(mapY.shape))
    mapZr = np.reshape(mapZ, np.prod(mapZ.shape))
    mapXr0 = np.floor(mapXr)
    mapYr0 = np.floor(mapYr)
    mapZr0 = np.floor(mapZr)

    check1 = mapXr0 >= 0
    check2 = mapYr0 >= 0
    check3 = mapZr0 >= 0
    check4 = mapXr0 < xs
    check5 = mapYr0 < ys
    check6 = mapZr0 < zs
    checkall = check1 * check2 * check3 * check4 * check5 * check6

    ii = zs * ys * mapXr0 + zs * mapYr0 + mapZr0
    ii = ii.astype(int)
    ii = ii * checkall
    datar = np.reshape(data, np.prod(data.shape))

    output_image_nearest = np.reshape(datar[ii], output_size)

    return output_image_nearest


def resize_3D_nearest(input_data, newsize):
    # resize 3D images, but return the nearest-neighbor result
    xs, ys, zs = np.shape(input_data)
    xo, yo, zo = np.mgrid[0:xs:newsize[0]*1j, 0:ys:newsize[1]*1j, 0:zs:newsize[2]*1j]
    mask_array = np.zeros((xs,ys,zs))
    a = np.where(np.isnan(input_data))
    mask_array[a] = 1   # create a mask for keeping track of the nan locations
    input_data[a] = 0   # remove the nans from the input data, before warping
    # do the same warping to the input and the mask
    output_data = warp_image_nearest(input_data, xo, yo, zo)
    warped_mask = warp_image_nearest(mask_array, xo, yo, zo)
    a = np.where(warped_mask > 0.5)
    output_data[a] = np.nan   # put back the nans
    return output_data

