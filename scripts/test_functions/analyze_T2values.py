# sys.path.extend([r'C:\Users\Stroman\PycharmProjects\pantheon\venv'])
# function to track a time-series of anatomical changes
import pandas as pd
import os
import dicom2nifti
import pydicom
import numpy as np
import shutil
import nibabel as nib
import py_mirt3D as mirt
import copy
import matplotlib.pyplot as plt
import image_operations_3D as i3d
import pynormalization
import load_templates
import scipy

import trimesh
import time
import io
from PIL import Image

def find(lst, a):
    return [i for i, x in enumerate(lst) if x==a]


def read_dicom_directory(dicom_directory, stop_before_pixels=False):
    """
    List all files in a given directory (stop before pixels)
    If they are organized correctly, they should form a complete single dicom image series
    """
    file_list = []
    for root, _, files in os.walk(dicom_directory):
        for dicom_file in files:
            file_path = os.path.join(root, dicom_file)
            file_list.append(file_path)
    return file_list


def get_series_numbers(pname):
    # find all of the series in a dataset directory
    # allow for sorted, or unsorted, data sets

    DICOMlistfull = []  # create an empty list
    seriesnumberlist = []
    DICOMextension = '.ima'

    for filename in os.listdir(pname):
        fullpath = os.path.join(pname,filename)
        check1 = os.path.isdir(fullpath)
        if check1:
            for filename2 in os.listdir(fullpath):
                if DICOMextension in filename2.lower():  # check whether the file is DICOM
                    fullname = os.path.join(fullpath, filename2)
                    DICOMlistfull.append(fullname)
                    ds = pydicom.dcmread(fullname)
                    seriesnumberlist.append(ds.SeriesNumber)

        check2 = DICOMextension in filename.lower()
        if check2:
            fullname = os.path.join(pname, filename)
            DICOMlistfull.append(fullname)
            ds = pydicom.dcmread(fullname)
            seriesnumberlist.append(ds.SeriesNumber)

    x = np.array(seriesnumberlist)
    serieslist = np.unique(x)
    print('serieslist = ', serieslist)
    return serieslist


def organize_dicom_files(pname):
    # find all of the files in a dataset directory

    DICOMlistfull = []  # create an empty list
    seriesnumberlist = []
    DICOMextension = '.ima'

    for filename in os.listdir(pname):
        if DICOMextension in filename.lower():  # check whether the file is DICOM
            fullname = os.path.join(pname, filename)
            DICOMlistfull.append(fullname)
            ds = pydicom.dcmread(fullname)
            seriesnumberlist.append(ds.SeriesNumber)

    x = np.array(seriesnumberlist)
    serieslist = np.unique(x)
    print('serieslist = ', serieslist)

    if len(serieslist) > 1:  # don't do anything if the directory contains a single series
        # find all entries in the database matching pname
        # and list the series numbers for these entries

        # to write an excel file:    df.to_excel(outputname)
        for snum in serieslist:
            # find which database entries match the seriesnumber = snum
            # and then replace the pname with the new folder name

            # find all of the files in a series
            ii = find(seriesnumberlist, snum)
            temp_array = np.array(DICOMlistfull)
            list_of_dicom_files = temp_array[ii]

            # check if the subfolder needs to be created
            subfolder = 'Series{number}'.format(number=snum)
            check = pname.find(subfolder)
            if check == -1:
                subfolderpath = os.path.join(pname, subfolder)
            else:
                subfolderpath = subfolder  # don't add another layer if the subfolder already exists

            # create the new sub-folder
            # move the dicom files to the new folder
            if not os.path.isdir(subfolderpath):
                os.mkdir(subfolderpath)

            for dicomname in list_of_dicom_files:
                nameparts = os.path.split(dicomname)
                newdicomname = os.path.join(nameparts[0], subfolder, nameparts[1])
                # move the dicom file to the new location
                shutil.move(dicomname, newdicomname)

    return serieslist


def convert_dicom_folder(pname, seriesnumber):
    basename = 'Series'
    niiname = '{base}{number}.nii'.format(base=basename, number=seriesnumber)
    output_file = os.path.join(pname, niiname)

    print('output_file = ', output_file)

    # still need to check the orientation for both BS/SC data and brain data
    dicom2nifti.dicom_series_to_nifti(pname, output_file, reorient_nifti=True)
    # this will put images in the very stupid but "standard" LAS orientation, which is left-handed

    return output_file




def process_anatomy(run_number, convert_dicom_to_nifti = True):
    datadir = r'C:\anatomical_study2014'
    datafoldernames = ['MAY22_2024', 'MAY30_2024A', 'MAY30_2024B']
    dataseriesnums = np.array([[4,5], [4,5], [4,5]])
    zpos = [146, 146, 180]
    pos = np.array([[10,77,81], [10,71,80], [10,81, 117]])
    windownumlist = [10, 20, 30]
    calcnorm = [False, True, True]
    TE = np.array([0.034, 0.094])  # echo time in sec

    studynum = copy.deepcopy(run_number)
    seriesnumbers = dataseriesnums[studynum,:]

    pname = os.path.join(datadir,datafoldernames[studynum])

# define the data set and convert dicom to nifti if not already done
    if convert_dicom_to_nifti:
        serieslist = organize_dicom_files(pname)
        if len(serieslist) == 0:  # data were probably already organized
            serieslist = get_series_numbers(pname)

        for ss in serieslist:
            pname2 = os.path.join(pname,'Series{}'.format(ss))
            try:
                filename = convert_dicom_folder(pname2, ss)
            except:
                print('was not able to convert Series {} in {}'.format(ss,pname2))
    else:
        serieslist = get_series_numbers(pname)


    imgdata = []
    datadetails = []
    # for ss in seriesnumbers[0]:
    ss = seriesnumbers[0]
    basename = 'Series{}'.format(ss)
    fullpath = os.path.join(pname,basename)
    niiname = os.path.join(fullpath, 'Series{}.nii'.format(ss))
    input_img = nib.load(niiname)
    input_data = input_img.get_fdata()
    affine = input_img.affine
    input_hdr = input_img.header
    input_size = np.shape(input_data)
    pixdim = input_hdr['pixdim']
    dim = input_hdr['dim']
    FOV = dim[1:4]*pixdim[1:4]
    voxvol = np.prod(pixdim[1:4])
    affine2 = copy.deepcopy(affine)
    affine2[1,:3] *= -1
    affine2[1,3] += (dim[2]-1)*pixdim[2]

    # calculate normalization?
    calculate_normalization = calcnorm[studynum]
    normdataname_full, T, Tfine, warpdata, reverse_map_image, norm_image_fine, template_affine, result, FOV_original, dim_original, pixdim_original = calculate_normalization_parameters(niiname, calculate_normalization = calculate_normalization)


    img1 = input_data[:,:,:,0]
    img2 = input_data[:,:,:,1]
    img1[img1 < 1.0e-10] = 1.0e-10
    img2[img2 < 1.0e-10] = 1.0e-10
    logint1 = np.log(img1)
    logint2 = np.log(img2)

    # S = S0 exp(-TE/T2)
    # log(S) = -TE/T2 + log(S0)

    T2max = 1.0
    T2min = 0.001

    slope = (logint2 - logint1) / (TE[1] - TE[0])
    T2 = 1.0/(-slope + 1.0e-20)
    T2[T2 > T2max] = T2max
    T2[T2 < T2min] = T2min



    # sx = logint1 + logint2
    # sy = TE[1] + TE[0]
    # sx2 = logint1**2 + logint2**2
    # sxy = logint1*TE[0] + logint2*TE[1]
    #
    # a = (sy*sx2 - sx*sxy)/(2*sx2 - sx**2 + 1.0e-10)
    # b = (2*sxy - sx*sy)/(2*sx2 - sx**2 + 1.0e-10)
    #
    # T2 = 1.0/(-b + 1.0e-20)
    # S0 = np.exp(a)
    #
    # T2[T2 > 3.0] = 3.0

    # apply normalization
    # first need to scale to 1 mm cubic voxels
    [xd, yd, zd] = np.shape(T2)
    newsize = np.floor(FOV_original).astype('int')
    T2r = i3d.resize_3D(T2, newsize)


    T2norm = pynormalization.py_apply_normalization(T2r, T, Tfine, map_to_normalized_space=True)

    # py_apply_normalization(input_image, T, Tfine='none', map_to_normalized_space=True)

    xpos = 10
    windownum = windownumlist[studynum]
    plt.close(windownum)
    fig = plt.figure(windownum)
    plt.imshow(T2[xpos,:,:])

    plt.close(windownum+1)
    fig = plt.figure(windownum+1)
    plt.imshow(T2[:,:,zpos[studynum]])

    plt.close(windownum+2)
    fig = plt.figure(windownum+2)
    plt.imshow(T2norm[13,:,:])

    plt.close(windownum+3)
    fig = plt.figure(windownum+3)
    plt.imshow(T2norm[:,15:45,100])

    plt.close(windownum+4)
    fig = plt.figure(windownum+4)
    plt.imshow(norm_image_fine[13,:,:])




    # plt.close(12)
    # fig = plt.figure(12)
    # ax1 = plt.subplot(121)
    # plt.imshow(imgTE1,'gray')
    # ax2 = plt.subplot(122)
    # plt.imshow(imgTE2,'gray')



def calculate_normalization_parameters(niiname, calculate_normalization = True):
    # calculate normalization?
    p,f1 = os.path.split(niiname)
    f,e = os.path.splitext(f1)
    normdataname_full = os.path.join(p,f + '_normdata.npy')

    input_img = nib.load(niiname)
    input_data = input_img.get_fdata()
    affine = input_img.affine
    input_hdr = input_img.header
    input_size = np.shape(input_data)
    pixdim = input_hdr['pixdim']
    dim = input_hdr['dim']
    FOV = dim[1:4] * pixdim[1:4]
    voxvol = np.prod(pixdim[1:4])

    if calculate_normalization:
        resolution = 1
        normtemplatename = 'ccbs'
        template_img, regionmap_img, template_affine, anatlabels, wmmap_img, roi_map, gmwm_img = load_templates.load_template_and_masks(
            normtemplatename, resolution)
        fit_parameters = [50, 50, 5, 6, -10, 20, -10, 10]
        T, warpdata, reverse_map_image, displayrecord, imagerecord, resultsplot, result = pynormalization.run_rough_normalization_calculations(
            niiname, normtemplatename, template_img, fit_parameters, reftime = 1)

        Tfine, norm_image_fine = pynormalization.py_norm_fine_tuning(reverse_map_image, template_img, T,
                                                                     input_type='normalized')

        normdata = {'T': T, 'Tfine': Tfine, 'warpdata': warpdata, 'reverse_map_image': reverse_map_image,
                    'norm_image_fine': norm_image_fine, 'template_affine': template_affine, 'imagerecord': imagerecord,
                    'result': result, 'FOV_original':FOV, 'dim_original':dim, 'pixdim_original':pixdim}
        np.save(normdataname_full, normdata)

    else:
        normdata = np.load(normdataname_full, allow_pickle=True).flat[0]
        T = normdata['T']
        Tfine = normdata['Tfine']
        warpdata = normdata['warpdata']
        reverse_map_image = normdata['reverse_map_image']
        norm_image_fine = normdata['norm_image_fine']
        template_affine = normdata['template_affine']
        result = normdata['result']
        FOV = normdata['FOV_original']
        dim = normdata['dim_original']
        pixdim = normdata['pixdim_original']

    return normdataname_full, T, Tfine, warpdata, reverse_map_image, norm_image_fine, template_affine, result, FOV, dim, pixdim