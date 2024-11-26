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
import pynormalization as pnorm
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



def define_triangles_from_cloud(cx,cy,cz,volsize,max_sidelength, min_sidelength = 0.):
    # import open3d as o3d
    # return triangles
    # midpoint of cloud
    npoints = len(cx)

    # x0 = np.mean(cx)
    # y0 = np.mean(cy)
    # z0 = np.mean(cz)

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

    # # find redundant triangles
    # triangles = np.array(triangles)
    # ntri, n3 = np.shape(triangles)
    # temp_tri = copy.deepcopy(triangles)
    # finishhed_looking = False
    # nn = 0
    # while not finished_looking:
    #     t1= triangles[nn,:]
    #     mag1 = mag_record[nn]
    #     pmid1 = pmid_record[nn,:]
    #     norm1 = normvectors[nn,:]
    #     # compare triangles, remove the smaller one, if it is further inside the cloud
    #
    #     ntri, n3 = np.shape(triangles)
    #     for mm in range(nn+1,ntri):
    #         t2= triangles[mm,:]
    #         mag2 = mag_record[mm]
    #         pmid2 = pmid_record[mm,:]
    #         norm2 = normvectors[mm,:]


    print('identified {} unique triangles ...'.format(ntri))

    # discard triangles that are likely not near a surface
    includevecs = np.where(normcheck[:,1] > 0.6)[0]
    triangles_filtered = triangles[includevecs,:]
    normvectors_filtered = normvectors[includevecs,:]
    normcheck_filtered = normcheck[includevecs,:]

    ntri,n3 = np.shape(triangles_filtered)
    print('identified {} triangles.'.format(ntri))

    return triangles_filtered, normvectors_filtered, normcheck_filtered, allpointso




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


def check_mask(mask, labels, ref_data):
    # check on the mask here and see if it needs to be adjusted
    for nn in range(len(labels)):
        xc, yc, zc = np.where(mask == labels[nn])
        iicheck = ref_data[xc,yc,zc]
        iimean = np.mean(iicheck)
        iisd = np.std(iicheck)

        mask_temp = np.zeros(np.shape(mask))
        mask_temp[xc,yc,zc] = 1

        # erode-------------------------------------
        emask = scipy.ndimage.binary_erosion(mask_temp)
        xce, yce, zce = np.where(emask == 1)
        # compare original and eroded mask
        dx,dy,dz = np.where((mask_temp-emask) == 1)
        iie = ref_data[dx,dy,dz]
        cc = np.where( (iie < (iimean-2.0*iisd)) | (iie > (iimean+2.0*iisd)) )[0]
        mask_temp[dx[cc],dy[cc],dz[cc]] = 0

        # update values
        xc2, yc2, zc2 = np.where(mask_temp == 1)
        iicheck = ref_data[xc2,yc2,zc2]
        iimean = np.mean(iicheck)
        iisd = np.std(iicheck)

        # dilate-----------------------------------------
        dmask = scipy.ndimage.binary_dilation(mask_temp)
        xcd, ycd, zcd = np.where(dmask == 1)
        # compare original and dilated mask
        dx,dy,dz = np.where((dmask-mask_temp) == 1)

        iid = ref_data[dx,dy,dz]
        cc = np.where( (iid > (iimean-iisd)) & (iid < (iimean+iisd)) )[0]
        mask_temp[dx[cc],dy[cc],dz[cc]] = 1

        xc3, yc3, zc3 = np.where(mask_temp == 1)
        mask[xc3,yc3,zc3] = labels[nn]

    return mask


def process_anatomy(run_number, show_the_3Dscene = True):
    # run_number = 1
    labels = [7., 8.]

    datadir = r'D:\cMRI_data'
    labeldir = r'D:\cMRI_data\labeled_images'

    dataset_list = [2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,19,20,21]
    data_set_number = dataset_list[run_number]

    # dataset_list = [2,3,4,5,7,8,9,11,12,14,15,16]

    # missing data
    if data_set_number == 2:
        studyname = 'JUN19_2018'
        reffilename = r'June 19- CP2017-004_CMRI2018-002.MR.DR_PUKALL_FMRI.0010.0001.2018.06.19.nii'
        maskfilename = r'June 19- CP2017-004_CMRI2018-002.MR.DR_PUKALL_FMRI.0010.0001.2018.06.19.nii.gz'

    if data_set_number == 3:
        studyname = 'JUN27_2018'
        maskfilename = r'June 27- Seg-CP2017-004_CMRI2018-003.MR.DR_PUKALL_FMRI.0004.0001.2018.06.27.nii.gz'
        maskfilename = r'June 27- Seg_new-CP2017-004_CMRI2018-003.MR.DR_PUKALL_FMRI.0004.0001.2018.06.27.nii.gz'
        reffilename = r'June 27- CP2017-004_CMRI2018-003.MR.DR_PUKALL_FMRI.0004.0001.2018.06.27.nii'

    if data_set_number == 4:
        studyname = 'OCT18_2018'
        maskfilename = r'Oct 18- Seg-CP2017-004_CMRI2018-004.MR.DR_PUKALL_FMRI.0004.0001.2018.10.18.nii.gz'
        reffilename = r'Oct 18- CP2017-004_CMRI2018-004.MR.DR_PUKALL_FMRI.0004.0001.2018.10.18.nii'

    if data_set_number == 5:
        studyname = 'OCT04_2018'
        maskfilename = r'Oct 4- Seg-CP2017-004_CMRI2018-005.MR.DR_PUKALL_FMRI.0004.0001.2018.10.04.nii.gz'
        maskfilename = r'Oct 4- Seg_new-CP2017-004_CMRI2018-005.MR.DR_PUKALL_FMRI.0004.0001.2018.10.04.nii.gz'
        reffilename = r'Oct 4- CP2017-004_CMRI2018-005.MR.DR_PUKALL_FMRI.0004.0001.2018.10.04.nii'

    # missing data
    if data_set_number == 6:
        studyname = 'OCT09_2018'
        maskfilename = r'Oct 9- Seg-CP2017-004_CMRI2018-006.MR.DR_PUKALL_FMRI.0005.0001.2018.10.09.nii.gz'
        reffilename = r'Oct 9- Seg-CP2017-004_CMRI2018-006.MR.DR_PUKALL_FMRI.0005.0001.2018.10.09.nii'

    if data_set_number == 7:
        studyname = 'OCT16_2018'
        maskfilename = r'Oct 16- Seg-CP2017-004_CMRI2018-007.MR.DR_PUKALL_FMRI.0002.0035.2018.10.16.nii.gz'
        reffilename = r'Oct 16- CP2017-004_CMRI2018-007.MR.DR_PUKALL_FMRI.0004.0001.2018.10.16.15.nii'

    if data_set_number == 8:
        studyname = 'NOV06_2018'
        maskfilename = r'Nov 6- Seg-CP2017-004_CMRI2018_-008.MR.DR_PUKALL_FMRI.0008.0001.2018.11.06.nii.gz'
        reffilename = r'Nov 6- CP2017-004_CMRI2018_-008.MR.DR_PUKALL_FMRI.0008.0001.2018.11.06.nii'

    if data_set_number == 9:
        studyname = 'OCT31_2018'
        maskfilename = r'Seg-CP2017-004_CMRI2018-009.MR.DR_PUKALL_FMRI.0004.0001.2018.10.31.nii.gz'
        maskfilename = r'Oct 31- Seg_new-CP2017-004_CMRI2018-009.MR.DR_PUKALL_FMRI.0004.0001.2018.10.31.nii.gz'
        reffilename = r'CP2017-004_CMRI2018-009.MR.DR_PUKALL_FMRI.0004.0001.2018.10.31.nii'

    if data_set_number == 11:
        studyname = 'NOV09_2018'
        maskfilename = r'Nov 9- Seg-CP2017-004_CMRI2018-011.MR.DR_PUKALL_FMRI.0004.0001.2018.11.09.nii.gz'
        maskfilename = r'Nov 9- Seg_new-CP2017-004_CMRI2018-011.MR.DR_PUKALL_FMRI.0004.0001.2018.11.09.nii.gz'
        reffilename = r'Nov 9- Seg-CP2017-004_CMRI2018-011.MR.DR_PUKALL_FMRI.0004.0001.2018.11.09.nii'
        # reffilename = r'Nov 9 - CP2017-004_CMRI2018-011_Series4.nii'

    if data_set_number == 12:
        # labels are 1 and 3 ?
        studyname = 'NOV27_2018'
        maskfilename = r'Nov 27- Seg-CP2017-004_CMRI-012.MR.DR_PUKALL_FMRI.0004.0001.2018.11.27.nii.gz'
        maskfilename = r'Nov 27- Seg_new-CP2017-004_CMRI-012.MR.DR_PUKALL_FMRI.0004.0001.2018.11.27.nii.gz'
        # maskfilename = r'Nov 27- CMRI2018-0001-series2.nii.gz'
        reffilename = r'Nov 27- CP2017-004_CMRI-012.MR.DR_PUKALL_FMRI.0004.0001.2018.11.27.nii'

    if data_set_number == 13:
        studyname = 'JAN21_2019'
        maskfilename = r'Jan 21- Seg-CP2017_004_CMRI2018-013.MR.DR_PUKALL_FMRI.0004.0001.2019.01.21.nii.gz'
        reffilename = r'Jan 21- CP2017_004_CMRI2018-013.MR.DR_PUKALL_FMRI.0004.0001.2019.01.21.nii'

    # problem with labeling
    if data_set_number == 14:
        studyname = 'JAN11_2019'
        maskfilename = r'Jan 11- Seg-CP2017_004_CMRI2018-014.MR.DR_PUKALL_FMRI.0004.0001.2019.01.11.nii.gz'
        reffilename = r'Jan 11- CP2017_004_CMRI2018-014.MR.DR_PUKALL_FMRI.0004.0001.2019.01.11.nii'

    if data_set_number == 15:
        studyname = 'JAN17_2019'
        maskfilename = r'Jan 17- Seg-CP2017_004_CMRI2018-015.MR.DR_PUKALL_FMRI.0004.0001.2019.01.17.nii.gz'
        reffilename = r'Jan 17- CP2017_004_CMRI2018-015.MR.DR_PUKALL_FMRI.0004.0001.2019.01.17.nii'

    if data_set_number == 16:
        studyname = 'JAN14_2019'
        maskfilename = r'Jan 14- Seg-CP2017_004_CMRI2018-016.MR.DR_PUKALL_FMRI.0010.0001.2019.01.14.nii.gz'
        reffilename = r'Jan 14- CP2017_004_CMRI2018-016.MR.DR_PUKALL_FMRI.0010.0001.2019.01.14.nii'

    # data missing
    if data_set_number == 17:
        studyname = 'FEB04_2019'
        # CMRI2018-017
        maskfilename = r'Feb 4- Seg-CP2017_004_CMRI2018-017.MR.DR_PUKALL_FMRI.0005.0009.2019.02.04.nii.gz'
        reffilename = r'Feb 4- CP2017_004_CMRI2018-017.MR.DR_PUKALL_FMRI.0005.0009.2019.02.04.nii'

    # data missing
    if data_set_number == 19:
        studyname = 'FEB14_2019'
        maskfilename = r'Feb 14- Seg-CP2017_004_CMRI2018-019.MR.DR_PUKALL_FMRI.0004.0001.2019.02.14.nii.gz'
        reffilename = r'Feb 14- CP2017_004_CMRI2018-019.MR.DR_PUKALL_FMRI.0004.0001.2019.02.14.nii'

    # data missing
    if data_set_number == 20:
        studyname = 'FEB13_2019'
        maskfilename = r'Feb 13- Seg-CP2017_004_CMRI2018-020.MR.DR_PUKALL_FMRI.0004.0001.2019.02.13.nii.gz'
        reffilename = r'Feb 13- CP2017_004_CMRI2018-020.MR.DR_PUKALL_FMRI.0004.0001.2019.02.13.nii'

    # data missing
    if data_set_number == 21:
        studyname = 'FEB08_2019'
        maskfilename = r'Feb 8- Seg-CP2017_004_CMRI2018-021.MR.DR_PUKALL_FMRI.0003.0002.2019.02.08.nii.gz'
        reffilename = r'Feb 8- CP2017_004_CMRI2018-021.MR.DR_PUKALL_FMRI.0003.0002.2019.02.08.nii'

    if data_set_number == 999:
        studyname = 'MAR01_2019'
        # CMRI2018-023

    pname = os.path.join(datadir,studyname)
    maskfilename = os.path.join(labeldir, maskfilename)
    reffilename = os.path.join(labeldir, reffilename)

# define the data set and convert dicom to nifti if not already done
    seriesnumbers = []
    convert_dicom_to_nifti = True
    find_data_series = True

    pname = os.path.join(datadir,studyname)
    outputname = os.path.join(pname,r'mapping_results.npy')

    # load the mask data and get the mask affine matrix
    maskdata = nib.load(maskfilename)
    mask = maskdata.get_fdata()
    mask_affine = maskdata.affine
    mask_hdr = maskdata.header
    mask_size = np.shape(mask)

    # load the ref data and get the ref affine matrix
    refimg = nib.load(reffilename)
    ref_data = refimg.get_fdata()
    ref_affine = refimg.affine
    ref_hdr = refimg.header
    ref_size = np.shape(ref_data)

    # check on the mask here and see if it needs to be adjusted
    mask = check_mask(mask, labels, ref_data)

    # organize the data and convert to nifti
    pname = os.path.join(datadir,studyname)
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

    # find the series numbers that have image sets with the same size
    # and could describe a time-series of changes
    if find_data_series:
        size_record = []
        for ss in serieslist:
            basename = 'Series{}'.format(ss)
            fullpath = os.path.join(pname,basename)
            niiname = os.path.join(fullpath,'Series{}.nii'.format(ss))

            input_img = nib.load(niiname)
            input_data = input_img.get_fdata()
            affine = input_img.affine
            input_hdr = input_img.header
            dims = np.shape(input_data)

            size_record += [dims]

        size_record = np.array(size_record)
        usize,index,counts = np.unique(size_record,axis=0,return_counts = True,return_inverse = True)
        x = np.argmax(counts)
        c = np.where(index == x)[0]
        seriesnumbers  = serieslist[c]

    # seriesnumbers = [3, 4, 5, 6, 7, 8, 9]

    # load data sets
    # ss = seriesnumbers[0]
    # basename = 'Series{}'.format(ss)
    # fullpath = os.path.join(pname, basename)
    # niiname = os.path.join(fullpath, 'Series{}.nii'.format(ss))
    # input_img = nib.load(niiname)
    # ref_affine = input_img.affine
    # ref_data = input_img.get_fdata()
    # refsize = np.shape(ref_data)

    imgdata = []
    datadetails = []
    for ss in seriesnumbers:
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

        # account for weird reorientation in ITK-SNAP
        # conversion_matrix = np.eye(4)
        # conversion_matrix[0,0] = -1
        # conversion_matrix[1,1] = -1
        affine2 = copy.deepcopy(affine)
        affine2[1,:3] *= -1
        affine2[1,3] += (dim[2]-1)*pixdim[2]

        # affine2 = conversion_matrix @ affine2

        input_data2 = i3d.convert_affine_matrices_nearest(input_data, affine, affine2,
                                                    mask_size)

        input_data2 = i3d.convert_affine_matrices_nearest(input_data2, affine2, mask_affine,
                                                    mask_size)

        # input_data2 = copy.deepcopy(input_data)

        imgdata += [input_data2]
        # datadetails.append({'img':input_data2, 'hdr':input_hdr, 'affine':ref_affine, 'original_img':input_data})
        datadetails.append({'img':input_data2, 'hdr':input_hdr, 'affine':affine2, 'original_img':input_data})
    imgdata = np.array(imgdata)


    # mask2 = i3d.convert_affine_matrices_nearest(mask, mask_affine, datadetails[0]['affine'], np.shape(datadetails[0]['img']))
    mask2 = copy.deepcopy(mask)

    # display one slice with mask overlaid
    # crop the image data to focus on the region of interest
    xc, yc, zc = np.where(mask2 > 0)
    buffer = 10
    x0 = xc.min()-buffer
    x1 = xc.max() + buffer
    y0 = yc.min()-buffer
    y1 = yc.max() + buffer
    z0 = zc.min()-buffer
    z1 = zc.max() + buffer

    # x1,y1,z1 = np.shape(mask2)
    # x0,y0,z0 = 0,0,0

    cropped_mask = copy.deepcopy(mask2[x0:x1,y0:y1,z0:z1])

    cols = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
                     [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
                     [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]])
    nvals = np.unique(mask2)
    img1 = copy.deepcopy(imgdata[0,x0:x1,y0:y1,z0:z1])

    img1 = copy.deepcopy(datadetails[0]['img'][x0:x1,y0:y1,z0:z1])
    ref_data_cropped = copy.deepcopy(ref_data[x0:x1,y0:y1,z0:z1])

    xc, yc, zc = np.where(cropped_mask > 0)

    bg = img1/img1.max()
    red = copy.deepcopy(bg)
    green = copy.deepcopy(bg)
    blue = copy.deepcopy(bg)
    for count,nn in enumerate(nvals):
        if nn > 0:
            xc,yc,zc = np.where(cropped_mask == nn)
            red[xc,yc,zc] = cols[count,0]
            green[xc,yc,zc] = cols[count,1]
            blue[xc,yc,zc] = cols[count,2]

    xs,ys,zs = np.shape(red)
    # zs1 = np.floor(zs/2).astype(int)
    zs1 = np.floor(np.mean(zc)).astype(int)

    displayimg = np.zeros((xs,ys,3))
    displayimg[:,:,0] = red[:,:,zs1]
    displayimg[:,:,1] = green[:,:,zs1]
    displayimg[:,:,2] = blue[:,:,zs1]

    plt.close(10)
    fig = plt.figure(10)
    plt.imshow(displayimg)

    plt.close(11)
    fig = plt.figure(11)
    plt.imshow(img1[:,:,zs1],'gray')

    plt.close(12)
    fig = plt.figure(12)
    plt.imshow(ref_data_cropped[:,:,zs1],'gray')



    #----------------------------------------------------------------------------------------------
    # map each image volume to the first one in the series-----------------------------------------
    # set default main settings for MIRT coregistration
    main_init = {'similarity': 'ssd',  # similarity measure, e.g. SSD, CC, SAD, RC, CD2, MS, MI
                 'subdivide': 1,  # use 1 hierarchical level
                 'okno': 4,  # mesh window size
                 'lambda': 0,  # transformation regularization weight, 0 for none
                 'single': 1}

    # Optimization settings
    optim_init = {'maxsteps': 200,  # maximum number of iterations at each hierarchical level
                  'fundif': 1e-6,  # tolerance (stopping criterion)
                  'gamma': 0.1,  # initial optimization step size
                  'anneal': 0.7}  # annealing rate on the optimization step

    nvols = np.shape(imgdata)[0]
    refvol = 0
    norm_record = []
    for checkvol in range(1,nvols):
        refvol = checkvol - 1
        print('normalizing volume {} with reference volume {}'.format(checkvol,refvol))
        img1 = imgdata[checkvol,x0:x1,y0:y1,z0:z1]/np.max(imgdata[checkvol,x0:x1,y0:y1,z0:z1])

        if refvol == 0:
            imgref = imgdata[refvol,x0:x1,y0:y1,z0:z1]/np.max(imgdata[refvol,x0:x1,y0:y1,z0:z1])
            working_mask = copy.deepcopy(cropped_mask)
        else:
            imgref = copy.deepcopy(normimg)  # use results of previous iteration
            working_mask = copy.deepcopy(mapped_mask)

        optim = copy.deepcopy(optim_init)
        main = copy.deepcopy(main_init)

        # res, normimg = mirt.py_mirt3D_register(imgref, img1, main, optim)
        # map the reference image to where the moving image is instead, to get the reverse normalization
        # see how things have changed
        res, normimg = mirt.py_mirt3D_register(img1, imgref, main, optim)  # this is the normal way

        # # now apply the mapping to the mask
        # F = mirt.py_mirt3D_F(res['okno']);  # Precompute the matrix B - spline basis functions
        # Xx, Xy, Xz = mirt.py_mirt3D_nodes2grid(res['X'], F,
        #                                        res['okno']);  # obtain the position of all image voxels (Xx, Xy, Xz)
        # # from the positions of B-spline control points (res['X']
        # xs, ys, zs = np.shape(normimg)
        # X, Y, Z = np.mgrid[range(xs), range(ys), range(zs)]
        #
        # T = {'Xs': X, 'Ys': Y, 'Zs': Z, 'Xt': Xx, 'Yt': Xy, 'Zt': Xz}
        # applying normalization ...
        mapped_img = mirt.py_mirt3D_transform(imgref, res)
        mapped_mask = mirt.py_mirt3D_transform_nearest(working_mask, res)

        normimg[normimg < 0] = 0.
        normimg[normimg > 1] = 1.
        norm_record.append({'normimg':normimg, 'res':res, 'imgref':imgref, 'img1':img1, 'mapped_mask':mapped_mask})

    np.save(outputname, {'norm_record':norm_record, 'cropped_mask':cropped_mask, 'FOV':FOV, 'voxvol':voxvol, 'pixdim':pixdim, 'imgdata':imgdata})

    # display images-------------------------------------------------
    xc, yc, zc = np.where(cropped_mask > 0.5)
    xp = np.round(np.mean(xc)).astype(int)
    yp = np.round(np.mean(yc)).astype(int)
    zp = np.round(np.mean(zc)).astype(int)

    plt.close(41)
    fig = plt.figure(41)
    fig.add_subplot(3, 6, 1)
    img1 = imgdata[0, x0:x1, y0:y1, z0:z1] / np.max(imgdata[0, x0:x1, y0:y1, z0:z1])
    plt.imshow(img1[:, :, zp], 'gray')
    for vv in range(5):
        fig.add_subplot(3, 6, vv+2)
        plt.imshow(norm_record[vv]['img1'][:, :, zp], 'gray')

    svgname = os.path.join(pname, 'original_images.svg')
    plt.savefig(svgname, format='svg')

    fig.add_subplot(3, 6, 7)
    plt.imshow(cropped_mask[:, :, zp])
    for vv in range(5):
        fig.add_subplot(3, 6, vv+8)
        plt.imshow(norm_record[vv]['mapped_mask'][:, :, zp])

    svgname = os.path.join(pname, 'mapped_masks.svg')
    plt.savefig(svgname, format='svg')


    # plt.close(42)
    # fig = plt.figure(42)
    fig.add_subplot(3, 6, 13)
    imageFOV = np.round(FOV[[1,2]]).astype(int)
    temp = cropped_mask[xp, :, :]
    temp = i3d.resize_2D(temp, imageFOV)
    plt.imshow(temp)
    for vv in range(5):
        fig.add_subplot(3, 6, vv+14)
        temp = norm_record[vv]['mapped_mask'][xp, :, :]
        temp = i3d.resize_2D(temp, imageFOV)
        plt.imshow(temp)

    svgname = os.path.join(pname, 'mapped_masks2.svg')
    plt.savefig(svgname, format='svg')

    # plt.close(43)
    # fig = plt.figure(43)


    #-----------get quantitative measures-------------------------------------------

    labelvals = np.unique(cropped_mask)
    region1 = labelvals[1]
    region2 = labelvals[2]

    measure_record = []

    checkvol = 0
    img = copy.deepcopy(imgdata[checkvol, x0:x1, y0:y1, z0:z1])
    xc, yc, zc = np.where(cropped_mask == region1)
    nvox = len(xc)
    region1vol = nvox*voxvol
    intensity1 = np.mean(img[xc,yc,zc])
    intensity1_sd = np.std(img[xc,yc,zc])
    xc, yc, zc = np.where(cropped_mask == region2)
    nvox = len(xc)
    region2vol = nvox*voxvol
    intensity2 = np.mean(img[xc,yc,zc])
    intensity2_sd = np.std(img[xc,yc,zc])
    entry = {'volume':0, 'vol1':region1vol, 'intensity1':intensity1, 'intensity1_sd':intensity1_sd, 'vol2':region2vol, 'intensity2':intensity2, 'intensity2_sd':intensity2_sd}
    measure_record.append(entry)

    np.save(outputname, {'norm_record':norm_record, 'cropped_mask':cropped_mask, 'FOV':FOV, 'voxvol':voxvol,
                         'pixdim':pixdim, 'measure_record':measure_record, 'imgdata':imgdata})

    reload_results = False
    if reload_results:
        results = np.load(outputname, allow_pickle=True).flat[0]
        norm_record = results['norm_record']
        cropped_mask = results['cropped_mask']
        FOV = results['FOV']
        voxvol = results['voxvol']
        pixdim = results['pixdim']
        measure_record = results['measure_record']

    for vv in range(len(norm_record)):
        img = copy.deepcopy(imgdata[vv+1, x0:x1, y0:y1, z0:z1])
        temp_mask = np.round(norm_record[vv]['mapped_mask'])
        xc, yc, zc = np.where(temp_mask == region1)
        nvox = len(xc)
        region1vol = nvox*voxvol
        intensity1 = np.mean(img[xc,yc,zc])
        intensity1_sd = np.std(img[xc,yc,zc])
        xc, yc, zc = np.where(temp_mask == region2)
        nvox = len(xc)
        region2vol = nvox*voxvol
        intensity2 = np.mean(img[xc,yc,zc])
        intensity2_sd = np.std(img[xc,yc,zc])
        entry = {'volume':vv+1, 'vol1':region1vol, 'intensity1':intensity1, 'intensity1_sd':intensity1_sd, 'vol2':region2vol, 'intensity2':intensity2, 'intensity2_sd':intensity2_sd}
        measure_record.append(entry)

    np.save(outputname, {'norm_record':norm_record, 'cropped_mask':cropped_mask, 'FOV':FOV, 'voxvol':voxvol,
                         'pixdim':pixdim, 'measure_record':measure_record, 'imgdata':imgdata})

    nmeasures = len(measure_record)
    vol1 = [measure_record[x]['vol1'] for x in range(nmeasures)]
    intensity1 = [measure_record[x]['intensity1'] for x in range(nmeasures)]
    intensity1_sd = [measure_record[x]['intensity1_sd'] for x in range(nmeasures)]
    vol2 = [measure_record[x]['vol2'] for x in range(nmeasures)]
    intensity2 = [measure_record[x]['intensity2'] for x in range(nmeasures)]
    intensity2_sd = [measure_record[x]['intensity2_sd'] for x in range(nmeasures)]

    winnum = 10
    plt.close(winnum)
    fig = plt.figure(winnum)
    fig.add_subplot(1, 2, 1)
    plt.errorbar(range(nmeasures),intensity1,yerr=intensity1_sd,linestyle='-',color = 'g', marker = 'o')
    plt.errorbar(range(nmeasures),intensity2,yerr=intensity2_sd,linestyle='-',color = 'y', marker = 'o')
    plt.xlabel('image volume number')
    plt.ylabel('image intensity (avg +/- sd) (arb. units)')
    fig.add_subplot(1, 2, 2)
    plt.plot(range(nmeasures),np.array(vol1)/1000.,linestyle='-',color = 'g', marker = 'o')
    plt.plot(range(nmeasures),np.array(vol2)/1000.,linestyle='-',color = 'y', marker = 'o')
    plt.xlabel('image volume number')
    plt.ylabel('region volume (cc)')

    svgname = os.path.join(pname, 'volume_and_intensity_measures.svg')
    plt.savefig(svgname, format='svg')

    # triangulate in the underlying parametrization
    # import matplotlib.tri as mtri
    # plot 3D surface
    vs = pixdim[1:4]
    vv=0
    mesh_record = []
    for vv in range(6):
        print('rendering volume {} of {}'.format(vv+1,6))
        if vv == 0:
            temp_mask = cropped_mask
        else:
            temp_mask = np.round(norm_record[vv-1]['mapped_mask'])
        xs,ys,zs = np.shape(temp_mask)
        cx,cy,cz = np.where((temp_mask > 6.5) & (temp_mask < 7.5))
        cx2,cy2,cz2 = np.where((temp_mask > 7.5) & (temp_mask < 8.5))

        # allpoints = np.concatenate((cx[:,np.newaxis],cy[:,np.newaxis],cz[:,np.newaxis]),axis=1)
        # cloud = trimesh.points.PointCloud(allpoints)
        # mesh0 = trimesh.convex.convex_hull(cloud)

        print(time.ctime())
        max_sidelength = 2.5
        min_sidelength = 0.9
        volsize = np.shape(cropped_mask)*np.array([1,1,2])
        triangles, normvectors, normcheck, reduced_points = define_triangles_from_cloud(cx, cy, 2*cz, volsize, max_sidelength, min_sidelength)
        print(time.ctime())

        triangles2, normvectors2, normcheck2, reduced_points2 = define_triangles_from_cloud(cx2, cy2, 2*cz2, volsize, max_sidelength, min_sidelength)
        print(time.ctime())

        ntri,n3 = np.shape(triangles)
        facecolors = np.zeros((ntri,3))
        facecolors[:,0] = 255
        mesh = trimesh.Trimesh(vertices = reduced_points, faces = triangles, normals = normvectors, face_colors = facecolors)
        # mesh.show()
        meshS = trimesh.smoothing.filter_humphrey(mesh, alpha=0.1, beta=0.5)

        ntri2,n3 = np.shape(triangles2)
        facecolors2 = np.zeros((ntri2,3))
        facecolors2[:,2] = 255
        mesh2 = trimesh.Trimesh(vertices = reduced_points2, faces = triangles2, normals = normvectors2, face_colors = facecolors2)
        # mesh.show()
        mesh2S = trimesh.smoothing.filter_humphrey(mesh2, alpha=0.1, beta=0.5)

        mesh_record.append({'mesh1':meshS, 'mesh2':mesh2S,
                            'reduced_points1':reduced_points, 'triangles1':triangles, 'normvectors1':normvectors, 'normcheck1':normcheck,
                            'reduced_points2':reduced_points2, 'triangles2':triangles2, 'normvectors2':normvectors2, 'normcheck2':normcheck2 })

    pname = os.path.join(datadir, studyname)
    outputname = os.path.join(pname,'mesh_data2.npy')
    np.save(outputname, mesh_record)
    print('saved results to {}'.format(outputname))
    print(time.ctime())

    camera1 = trimesh.scene.Camera(fov = [60., 45.], resolution=[1800, 1350])
    # camera1 = trimesh.scene.Camera(fov = [60., 60.], resolution=[1800, 1800])
    # display the results...
    orientation1 = [0.75*np.pi, 0., 0.2*np.pi]
    orientation2 = [np.pi, -0.25*np.pi, 0.]
    pname = os.path.join(datadir, studyname)
    for vv in range(6):
        mesh1 = mesh_record[vv]['mesh1']
        mesh2 = mesh_record[vv]['mesh2']
        scene = trimesh.scene.scene.Scene(mesh1)
        scene.add_geometry(mesh2)
        scene.set_camera(angles = orientation2)
        imgbytes = scene.save_image()
        pngname = os.path.join(pname, 'render2_vol{}.png'.format(vv))
        image = np.array(Image.open(io.BytesIO(imgbytes)))
        im = Image.fromarray(image)
        im.save(pngname)

    if show_the_3Dscene:
        scene.show()



        # point_cloud = np.concatenate((cx[:,np.newaxis],cy[:,np.newaxis],cz[:,np.newaxis]),axis=1)

        # temp = (cy - np.min(cy))/(np.max(cy)-np.min(cy))
        # red = 1.0-temp
        # green = np.zeros(len(temp))
        # blue = temp
        # cloud_colors = np.concatenate((red[:,np.newaxis],green[:,np.newaxis],blue[:,np.newaxis]),axis=1)





def display_process_anatomy_results(run_number, show_the_3Dscene = True):
    # run_number = 1
    labels = [7., 8.]

    datadir = r'D:\cMRI_data'
    labeldir = r'D:\cMRI_data\labeled_images'

    dataset_list = [2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,19,20,21]
    data_set_number = dataset_list[run_number]

    # missing data
    if data_set_number == 2:
        studyname = 'JUN19_2018'
        reffilename = r'June 19- CP2017-004_CMRI2018-002.MR.DR_PUKALL_FMRI.0010.0001.2018.06.19.nii'
        maskfilename = r'June 19- CP2017-004_CMRI2018-002.MR.DR_PUKALL_FMRI.0010.0001.2018.06.19.nii.gz'

    if data_set_number == 3:
        studyname = 'JUN27_2018'
        maskfilename = r'June 27- Seg-CP2017-004_CMRI2018-003.MR.DR_PUKALL_FMRI.0004.0001.2018.06.27.nii.gz'
        maskfilename = r'June 27- Seg_new-CP2017-004_CMRI2018-003.MR.DR_PUKALL_FMRI.0004.0001.2018.06.27.nii.gz'
        reffilename = r'June 27- CP2017-004_CMRI2018-003.MR.DR_PUKALL_FMRI.0004.0001.2018.06.27.nii'

    if data_set_number == 4:
        studyname = 'OCT18_2018'
        maskfilename = r'Oct 18- Seg-CP2017-004_CMRI2018-004.MR.DR_PUKALL_FMRI.0004.0001.2018.10.18.nii.gz'
        reffilename = r'Oct 18- CP2017-004_CMRI2018-004.MR.DR_PUKALL_FMRI.0004.0001.2018.10.18.nii'

    if data_set_number == 5:
        studyname = 'OCT04_2018'
        maskfilename = r'Oct 4- Seg-CP2017-004_CMRI2018-005.MR.DR_PUKALL_FMRI.0004.0001.2018.10.04.nii.gz'
        maskfilename = r'Oct 4- Seg_new-CP2017-004_CMRI2018-005.MR.DR_PUKALL_FMRI.0004.0001.2018.10.04.nii.gz'
        reffilename = r'Oct 4- CP2017-004_CMRI2018-005.MR.DR_PUKALL_FMRI.0004.0001.2018.10.04.nii'

    # missing data
    if data_set_number == 6:
        studyname = 'OCT09_2018'
        maskfilename = r'Oct 9- Seg-CP2017-004_CMRI2018-006.MR.DR_PUKALL_FMRI.0005.0001.2018.10.09.nii.gz'
        reffilename = r'Oct 9- Seg-CP2017-004_CMRI2018-006.MR.DR_PUKALL_FMRI.0005.0001.2018.10.09.nii'

    if data_set_number == 7:
        studyname = 'OCT16_2018'
        maskfilename = r'Oct 16- Seg-CP2017-004_CMRI2018-007.MR.DR_PUKALL_FMRI.0002.0035.2018.10.16.nii.gz'
        reffilename = r'Oct 16- CP2017-004_CMRI2018-007.MR.DR_PUKALL_FMRI.0004.0001.2018.10.16.15.nii'

    if data_set_number == 8:
        studyname = 'NOV06_2018'
        maskfilename = r'Nov 6- Seg-CP2017-004_CMRI2018_-008.MR.DR_PUKALL_FMRI.0008.0001.2018.11.06.nii.gz'
        reffilename = r'Nov 6- CP2017-004_CMRI2018_-008.MR.DR_PUKALL_FMRI.0008.0001.2018.11.06.nii'

    if data_set_number == 9:
        studyname = 'OCT31_2018'
        maskfilename = r'Seg-CP2017-004_CMRI2018-009.MR.DR_PUKALL_FMRI.0004.0001.2018.10.31.nii.gz'
        maskfilename = r'Oct 31- Seg_new-CP2017-004_CMRI2018-009.MR.DR_PUKALL_FMRI.0004.0001.2018.10.31.nii.gz'
        reffilename = r'CP2017-004_CMRI2018-009.MR.DR_PUKALL_FMRI.0004.0001.2018.10.31.nii'

    if data_set_number == 11:
        studyname = 'NOV09_2018'
        maskfilename = r'Nov 9- Seg-CP2017-004_CMRI2018-011.MR.DR_PUKALL_FMRI.0004.0001.2018.11.09.nii.gz'
        maskfilename = r'Nov 9- Seg_new-CP2017-004_CMRI2018-011.MR.DR_PUKALL_FMRI.0004.0001.2018.11.09.nii.gz'
        reffilename = r'Nov 9- Seg-CP2017-004_CMRI2018-011.MR.DR_PUKALL_FMRI.0004.0001.2018.11.09.nii'
        # reffilename = r'Nov 9 - CP2017-004_CMRI2018-011_Series4.nii'

    if data_set_number == 12:
        # labels are 1 and 3 ?
        studyname = 'NOV27_2018'
        maskfilename = r'Nov 27- Seg-CP2017-004_CMRI-012.MR.DR_PUKALL_FMRI.0004.0001.2018.11.27.nii.gz'
        maskfilename = r'Nov 27- Seg_new-CP2017-004_CMRI-012.MR.DR_PUKALL_FMRI.0004.0001.2018.11.27.nii.gz'
        # maskfilename = r'Nov 27- CMRI2018-0001-series2.nii.gz'
        reffilename = r'Nov 27- CP2017-004_CMRI-012.MR.DR_PUKALL_FMRI.0004.0001.2018.11.27.nii'

    if data_set_number == 13:
        studyname = 'JAN21_2019'
        maskfilename = r'Jan 21- Seg-CP2017_004_CMRI2018-013.MR.DR_PUKALL_FMRI.0004.0001.2019.01.21.nii.gz'
        reffilename = r'Jan 21- CP2017_004_CMRI2018-013.MR.DR_PUKALL_FMRI.0004.0001.2019.01.21.nii'

    # problem with labeling
    if data_set_number == 14:
        studyname = 'JAN11_2019'
        maskfilename = r'Jan 11- Seg-CP2017_004_CMRI2018-014.MR.DR_PUKALL_FMRI.0004.0001.2019.01.11.nii.gz'
        reffilename = r'Jan 11- CP2017_004_CMRI2018-014.MR.DR_PUKALL_FMRI.0004.0001.2019.01.11.nii'

    if data_set_number == 15:
        studyname = 'JAN17_2019'
        maskfilename = r'Jan 17- Seg-CP2017_004_CMRI2018-015.MR.DR_PUKALL_FMRI.0004.0001.2019.01.17.nii.gz'
        reffilename = r'Jan 17- CP2017_004_CMRI2018-015.MR.DR_PUKALL_FMRI.0004.0001.2019.01.17.nii'

    if data_set_number == 16:
        studyname = 'JAN14_2019'
        maskfilename = r'Jan 14- Seg-CP2017_004_CMRI2018-016.MR.DR_PUKALL_FMRI.0010.0001.2019.01.14.nii.gz'
        reffilename = r'Jan 14- CP2017_004_CMRI2018-016.MR.DR_PUKALL_FMRI.0010.0001.2019.01.14.nii'

    # data missing
    if data_set_number == 17:
        studyname = 'FEB04_2019'
        # CMRI2018-017
        maskfilename = r'Feb 4- Seg-CP2017_004_CMRI2018-017.MR.DR_PUKALL_FMRI.0005.0009.2019.02.04.nii.gz'
        reffilename = r'Feb 4- CP2017_004_CMRI2018-017.MR.DR_PUKALL_FMRI.0005.0009.2019.02.04.nii'

    # data missing
    if data_set_number == 19:
        studyname = 'FEB14_2019'
        maskfilename = r'Feb 14- Seg-CP2017_004_CMRI2018-019.MR.DR_PUKALL_FMRI.0004.0001.2019.02.14.nii.gz'
        reffilename = r'Feb 14- CP2017_004_CMRI2018-019.MR.DR_PUKALL_FMRI.0004.0001.2019.02.14.nii'

    # data missing
    if data_set_number == 20:
        studyname = 'FEB13_2019'
        maskfilename = r'Feb 13- Seg-CP2017_004_CMRI2018-020.MR.DR_PUKALL_FMRI.0004.0001.2019.02.13.nii.gz'
        reffilename = r'Feb 13- CP2017_004_CMRI2018-020.MR.DR_PUKALL_FMRI.0004.0001.2019.02.13.nii'

    # data missing
    if data_set_number == 21:
        studyname = 'FEB08_2019'
        maskfilename = r'Feb 8- Seg-CP2017_004_CMRI2018-021.MR.DR_PUKALL_FMRI.0003.0002.2019.02.08.nii.gz'
        reffilename = r'Feb 8- CP2017_004_CMRI2018-021.MR.DR_PUKALL_FMRI.0003.0002.2019.02.08.nii'

    if data_set_number == 999:
        studyname = 'MAR01_2019'
        # CMRI2018-023

    pname = os.path.join(datadir,studyname)
    maskfilename = os.path.join(labeldir, maskfilename)
    reffilename = os.path.join(labeldir, reffilename)



    pname = os.path.join(datadir,studyname)
    outputname = os.path.join(pname,r'mapping_results.npy')


    # np.save(outputname, {'norm_record':norm_record, 'cropped_mask':cropped_mask, 'FOV':FOV, 'voxvol':voxvol, 'pixdim':pixdim, 'imgdata':imgdata})
    mappingdata = np.load(outputname, allow_pickle = True).flat[0]
    # fields are: 'norm_record', 'cropped_mask', 'FOV', 'voxvol', 'pixdim', 'imgdata'
    norm_record = copy.deepcopy(mappingdata['norm_record'])
    cropped_mask = copy.deepcopy(mappingdata['cropped_mask'])
    FOV = copy.deepcopy(mappingdata['FOV'])
    voxvol = copy.deepcopy(mappingdata['voxvol'])
    pixdim = copy.deepcopy(mappingdata['pixdim'])
    # imgdata = copy.deepcopy(mappingdata['imgdata'])
    measure_record = copy.deepcopy(mappingdata['measure_record'])


    # display images-------------------------------------------------
    xc, yc, zc = np.where(cropped_mask > 0.5)
    xp = np.round(np.mean(xc)).astype(int)
    yp = np.round(np.mean(yc)).astype(int)
    zp = np.round(np.mean(zc)).astype(int)

    plt.close(41)
    fig = plt.figure(41)
    fig.add_subplot(3, 6, 1)
    # img1 = imgdata[0, x0:x1, y0:y1, z0:z1] / np.max(imgdata[0, x0:x1, y0:y1, z0:z1])
    # plt.imshow(img1[:, :, zp], 'gray')
    for vv in range(5):
        fig.add_subplot(3, 6, vv+2)
        plt.imshow(norm_record[vv]['img1'][:, :, zp], 'gray')

    svgname = os.path.join(pname, 'original_images.svg')
    # plt.savefig(svgname, format='svg')

    fig.add_subplot(3, 6, 7)
    plt.imshow(cropped_mask[:, :, zp])
    for vv in range(5):
        fig.add_subplot(3, 6, vv+8)
        plt.imshow(norm_record[vv]['mapped_mask'][:, :, zp])

    svgname = os.path.join(pname, 'mapped_masks.svg')
    # plt.savefig(svgname, format='svg')


    # plt.close(42)
    # fig = plt.figure(42)
    fig.add_subplot(3, 6, 13)
    imageFOV = np.round(FOV[[1,2]]).astype(int)
    temp = cropped_mask[xp, :, :]
    temp = i3d.resize_2D(temp, imageFOV)
    plt.imshow(temp)
    for vv in range(5):
        fig.add_subplot(3, 6, vv+14)
        temp = norm_record[vv]['mapped_mask'][xp, :, :]
        temp = i3d.resize_2D(temp, imageFOV)
        plt.imshow(temp)

    svgname = os.path.join(pname, 'mapped_masks2.svg')
    # plt.savefig(svgname, format='svg')

    # plt.close(43)
    # fig = plt.figure(43)


    #-----------get quantitative measures-------------------------------------------

    labelvals = np.unique(cropped_mask)
    region1 = labelvals[1]
    region2 = labelvals[2]

    nmeasures = len(measure_record)
    vol1 = [measure_record[x]['vol1'] for x in range(nmeasures)]
    intensity1 = [measure_record[x]['intensity1'] for x in range(nmeasures)]
    intensity1_sd = [measure_record[x]['intensity1_sd'] for x in range(nmeasures)]
    vol2 = [measure_record[x]['vol2'] for x in range(nmeasures)]
    intensity2 = [measure_record[x]['intensity2'] for x in range(nmeasures)]
    intensity2_sd = [measure_record[x]['intensity2_sd'] for x in range(nmeasures)]

    winnum = 10
    plt.close(winnum)
    fig = plt.figure(winnum)
    fig.add_subplot(1, 2, 1)
    plt.errorbar(range(nmeasures),intensity1,yerr=intensity1_sd,linestyle='-',color = 'g', marker = 'o')
    plt.errorbar(range(nmeasures),intensity2,yerr=intensity2_sd,linestyle='-',color = 'y', marker = 'o')
    plt.xlabel('image volume number')
    plt.ylabel('image intensity (avg +/- sd) (arb. units)')
    fig.add_subplot(1, 2, 2)
    plt.plot(range(nmeasures),np.array(vol1)/1000.,linestyle='-',color = 'g', marker = 'o')
    plt.plot(range(nmeasures),np.array(vol2)/1000.,linestyle='-',color = 'y', marker = 'o')
    plt.xlabel('image volume number')
    plt.ylabel('region volume (cc)')

    svgname = os.path.join(pname, 'volume_and_intensity_measures.svg')
    # plt.savefig(svgname, format='svg')



    pname = os.path.join(datadir, studyname)
    outputname = os.path.join(pname,'mesh_data2.npy')
    # np.save(outputname, mesh_record)
    mesh_record = np.load(outputname, allow_pickle=True)


    # print('saved results to {}'.format(outputname))
    # print(time.ctime())
    winnum = 20
    plt.close(winnum)
    fig = plt.figure(winnum)

    camera1 = trimesh.scene.Camera(fov = [60., 45.], resolution=[1800, 1350])
    # camera1 = trimesh.scene.Camera(fov = [60., 60.], resolution=[1800, 1800])
    # display the results...
    orientation1 = [0.75*np.pi, 0., 0.2*np.pi]
    orientation2 = [np.pi, -0.25*np.pi, 0.]
    pname = os.path.join(datadir, studyname)
    for vv in range(6):
        mesh1 = mesh_record[vv]['mesh1']
        mesh2 = mesh_record[vv]['mesh2']
        scene = trimesh.scene.scene.Scene(mesh1)
        scene.add_geometry(mesh2)
        scene.set_camera(angles = orientation2)
        imgbytes = scene.save_image()
        pngname = os.path.join(pname, 'render2_vol{}.png'.format(vv))
        image = np.array(Image.open(io.BytesIO(imgbytes)))
        im = Image.fromarray(image)
        # im.save(pngname)

    if show_the_3Dscene:
        scene.show()




def display_processed_3Danatomy(run_number, generate_image_frames = True):
    # run_number = 1
    labels = [7., 8.]

    datadir = r'D:\cMRI_data'
    labeldir = r'D:\cMRI_data\labeled_images'

    dataset_list = [2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,19,20,21]
    data_set_number = dataset_list[run_number]

    # missing data
    if data_set_number == 2:
        studyname = 'JUN19_2018'
        reffilename = r'June 19- CP2017-004_CMRI2018-002.MR.DR_PUKALL_FMRI.0010.0001.2018.06.19.nii'
        maskfilename = r'June 19- CP2017-004_CMRI2018-002.MR.DR_PUKALL_FMRI.0010.0001.2018.06.19.nii.gz'

    if data_set_number == 3:
        studyname = 'JUN27_2018'
        maskfilename = r'June 27- Seg-CP2017-004_CMRI2018-003.MR.DR_PUKALL_FMRI.0004.0001.2018.06.27.nii.gz'
        maskfilename = r'June 27- Seg_new-CP2017-004_CMRI2018-003.MR.DR_PUKALL_FMRI.0004.0001.2018.06.27.nii.gz'
        reffilename = r'June 27- CP2017-004_CMRI2018-003.MR.DR_PUKALL_FMRI.0004.0001.2018.06.27.nii'

    if data_set_number == 4:
        studyname = 'OCT18_2018'
        maskfilename = r'Oct 18- Seg-CP2017-004_CMRI2018-004.MR.DR_PUKALL_FMRI.0004.0001.2018.10.18.nii.gz'
        reffilename = r'Oct 18- CP2017-004_CMRI2018-004.MR.DR_PUKALL_FMRI.0004.0001.2018.10.18.nii'

    if data_set_number == 5:
        studyname = 'OCT04_2018'
        maskfilename = r'Oct 4- Seg-CP2017-004_CMRI2018-005.MR.DR_PUKALL_FMRI.0004.0001.2018.10.04.nii.gz'
        maskfilename = r'Oct 4- Seg_new-CP2017-004_CMRI2018-005.MR.DR_PUKALL_FMRI.0004.0001.2018.10.04.nii.gz'
        reffilename = r'Oct 4- CP2017-004_CMRI2018-005.MR.DR_PUKALL_FMRI.0004.0001.2018.10.04.nii'

    # missing data
    if data_set_number == 6:
        studyname = 'OCT09_2018'
        maskfilename = r'Oct 9- Seg-CP2017-004_CMRI2018-006.MR.DR_PUKALL_FMRI.0005.0001.2018.10.09.nii.gz'
        reffilename = r'Oct 9- Seg-CP2017-004_CMRI2018-006.MR.DR_PUKALL_FMRI.0005.0001.2018.10.09.nii'

    if data_set_number == 7:
        studyname = 'OCT16_2018'
        maskfilename = r'Oct 16- Seg-CP2017-004_CMRI2018-007.MR.DR_PUKALL_FMRI.0002.0035.2018.10.16.nii.gz'
        reffilename = r'Oct 16- CP2017-004_CMRI2018-007.MR.DR_PUKALL_FMRI.0004.0001.2018.10.16.15.nii'

    if data_set_number == 8:
        studyname = 'NOV06_2018'
        maskfilename = r'Nov 6- Seg-CP2017-004_CMRI2018_-008.MR.DR_PUKALL_FMRI.0008.0001.2018.11.06.nii.gz'
        reffilename = r'Nov 6- CP2017-004_CMRI2018_-008.MR.DR_PUKALL_FMRI.0008.0001.2018.11.06.nii'

    if data_set_number == 9:
        studyname = 'OCT31_2018'
        maskfilename = r'Seg-CP2017-004_CMRI2018-009.MR.DR_PUKALL_FMRI.0004.0001.2018.10.31.nii.gz'
        maskfilename = r'Oct 31- Seg_new-CP2017-004_CMRI2018-009.MR.DR_PUKALL_FMRI.0004.0001.2018.10.31.nii.gz'
        reffilename = r'CP2017-004_CMRI2018-009.MR.DR_PUKALL_FMRI.0004.0001.2018.10.31.nii'

    if data_set_number == 11:
        studyname = 'NOV09_2018'
        maskfilename = r'Nov 9- Seg-CP2017-004_CMRI2018-011.MR.DR_PUKALL_FMRI.0004.0001.2018.11.09.nii.gz'
        maskfilename = r'Nov 9- Seg_new-CP2017-004_CMRI2018-011.MR.DR_PUKALL_FMRI.0004.0001.2018.11.09.nii.gz'
        reffilename = r'Nov 9- Seg-CP2017-004_CMRI2018-011.MR.DR_PUKALL_FMRI.0004.0001.2018.11.09.nii'
        # reffilename = r'Nov 9 - CP2017-004_CMRI2018-011_Series4.nii'

    if data_set_number == 12:
        # labels are 1 and 3 ?
        studyname = 'NOV27_2018'
        maskfilename = r'Nov 27- Seg-CP2017-004_CMRI-012.MR.DR_PUKALL_FMRI.0004.0001.2018.11.27.nii.gz'
        maskfilename = r'Nov 27- Seg_new-CP2017-004_CMRI-012.MR.DR_PUKALL_FMRI.0004.0001.2018.11.27.nii.gz'
        # maskfilename = r'Nov 27- CMRI2018-0001-series2.nii.gz'
        reffilename = r'Nov 27- CP2017-004_CMRI-012.MR.DR_PUKALL_FMRI.0004.0001.2018.11.27.nii'

    if data_set_number == 13:
        studyname = 'JAN21_2019'
        maskfilename = r'Jan 21- Seg-CP2017_004_CMRI2018-013.MR.DR_PUKALL_FMRI.0004.0001.2019.01.21.nii.gz'
        reffilename = r'Jan 21- CP2017_004_CMRI2018-013.MR.DR_PUKALL_FMRI.0004.0001.2019.01.21.nii'

    # problem with labeling
    if data_set_number == 14:
        studyname = 'JAN11_2019'
        maskfilename = r'Jan 11- Seg-CP2017_004_CMRI2018-014.MR.DR_PUKALL_FMRI.0004.0001.2019.01.11.nii.gz'
        reffilename = r'Jan 11- CP2017_004_CMRI2018-014.MR.DR_PUKALL_FMRI.0004.0001.2019.01.11.nii'

    if data_set_number == 15:
        studyname = 'JAN17_2019'
        maskfilename = r'Jan 17- Seg-CP2017_004_CMRI2018-015.MR.DR_PUKALL_FMRI.0004.0001.2019.01.17.nii.gz'
        reffilename = r'Jan 17- CP2017_004_CMRI2018-015.MR.DR_PUKALL_FMRI.0004.0001.2019.01.17.nii'

    if data_set_number == 16:
        studyname = 'JAN14_2019'
        maskfilename = r'Jan 14- Seg-CP2017_004_CMRI2018-016.MR.DR_PUKALL_FMRI.0010.0001.2019.01.14.nii.gz'
        reffilename = r'Jan 14- CP2017_004_CMRI2018-016.MR.DR_PUKALL_FMRI.0010.0001.2019.01.14.nii'

    # data missing
    if data_set_number == 17:
        studyname = 'FEB04_2019'
        # CMRI2018-017
        maskfilename = r'Feb 4- Seg-CP2017_004_CMRI2018-017.MR.DR_PUKALL_FMRI.0005.0009.2019.02.04.nii.gz'
        reffilename = r'Feb 4- CP2017_004_CMRI2018-017.MR.DR_PUKALL_FMRI.0005.0009.2019.02.04.nii'

    # data missing
    if data_set_number == 19:
        studyname = 'FEB14_2019'
        maskfilename = r'Feb 14- Seg-CP2017_004_CMRI2018-019.MR.DR_PUKALL_FMRI.0004.0001.2019.02.14.nii.gz'
        reffilename = r'Feb 14- CP2017_004_CMRI2018-019.MR.DR_PUKALL_FMRI.0004.0001.2019.02.14.nii'

    # data missing
    if data_set_number == 20:
        studyname = 'FEB13_2019'
        maskfilename = r'Feb 13- Seg-CP2017_004_CMRI2018-020.MR.DR_PUKALL_FMRI.0004.0001.2019.02.13.nii.gz'
        reffilename = r'Feb 13- CP2017_004_CMRI2018-020.MR.DR_PUKALL_FMRI.0004.0001.2019.02.13.nii'

    # data missing
    if data_set_number == 21:
        studyname = 'FEB08_2019'
        maskfilename = r'Feb 8- Seg-CP2017_004_CMRI2018-021.MR.DR_PUKALL_FMRI.0003.0002.2019.02.08.nii.gz'
        reffilename = r'Feb 8- CP2017_004_CMRI2018-021.MR.DR_PUKALL_FMRI.0003.0002.2019.02.08.nii'

    if data_set_number == 999:
        studyname = 'MAR01_2019'
        # CMRI2018-023

    pname = os.path.join(datadir,studyname)
    maskfilename = os.path.join(labeldir, maskfilename)
    reffilename = os.path.join(labeldir, reffilename)

    pname = os.path.join(datadir, studyname)
    outputname = os.path.join(pname,'mesh_data2.npy')
    # np.save(outputname, mesh_record)
    mesh_record = np.load(outputname, allow_pickle=True)


    # print('saved results to {}'.format(outputname))
    # print(time.ctime())

    camera1 = trimesh.scene.Camera(fov = [60., 45.], resolution=[1800, 1350])
    # camera1 = trimesh.scene.Camera(fov = [60., 60.], resolution=[1800, 1800])
    # display the results...
    orientation1 = [0.75*np.pi, 0., 0.2*np.pi]
    orientation2 = [np.pi, -0.25*np.pi, 0.]
    pname = os.path.join(datadir, studyname)

    ntransition = 4
    nframes = 6*ntransition + 40

    if generate_image_frames:
        for ff in range(nframes):
            vv = np.floor(ff/ntransition).astype(int)
            if vv > 5: vv = 5
            mesh1 = mesh_record[vv]['mesh1']
            mesh2 = mesh_record[vv]['mesh2']
            scene = trimesh.scene.scene.Scene(mesh1)
            scene.add_geometry(mesh2)

            if ff > 23:
                orientation2 = [np.pi, (-0.35 - 2.0*(ff-23)/(nframes-24)) * np.pi, 0.]
            else:
                orientation2 = [np.pi, -0.35 * np.pi, 0.]

            print('frame {} orientation {}'.format(ff,orientation2))
            try:
                scene.set_camera(angles = orientation2)
                imgbytes = scene.save_image()
                pngname = os.path.join(pname, 'video_vol{}.png'.format(ff))
                image = np.array(Image.open(io.BytesIO(imgbytes)))
                im = Image.fromarray(image)
                im.save(pngname)
            except:
                print('  frame {}     cound not display this one'.format(ff))


        # plt.imshow(image)
        # plt.show()
        # plt.pause(framerate)


        # scene.show()
        # plt.pause(framerate)

    frame_record = []
    for ff in range(nframes):
        try:
            pngname = os.path.join(pname, 'video_vol{}.png'.format(ff))
            img = Image.open(pngname)
            frame_record.append({'frame':ff, 'img':img})
        except:
            print('frame {} not found'.format(ff))

    winnum = 20
    plt.close(winnum)
    fig = plt.figure(winnum, figsize=(12, 12))
    ax = fig.add_subplot(111, aspect = 'equal', frame_on = False)

    framerate = 0.001
    nframes2 = len(frame_record)
    for ff in range(nframes2):
        img = frame_record[ff]['img']
        plt.cla()
        plt.imshow(img)
        plt.show()
        ax.axis('off')
        if ff == 0:
            plt.pause(3.0)
        else:
            if ff < 23:
                plt.pause(0.2)
            else:
                plt.pause(framerate)



def run_process_anatomy():
    # in process_anatomy:
    # dataset_list = [2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,19,20,21]
    #  good data sets:
    # dataset_list = [ 3,4,5,  7,8,9,11,12,   14,15,16]
    # run numbers to use are  [1,2,3,5,6,7,8,9,11,12,13]

    datadir = r'D:\cMRI_data'
    labeldir = r'D:\cMRI_data\labeled_images'

    run_numbers = [1,2,3,5,6,7,8,9,11,12,13]

    # runs with updated labels
    run_numbers = [1,3,7,8,9]
    for nn in run_numbers:
        try:
            process_anatomy(nn, False)
        except:
            print('processing anatomy did not work for run {}'.format(nn))




def group_analysis():

    datadir = r'D:\cMRI_data'
    labeldir = r'D:\cMRI_data\labeled_images'

    # dataset_list = [3,4,5,7,8,9,12,15,16]
    run_numbers = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13]
    dataset_list = [3,4,5,7,8,9,11,12,14,15,16]

    # dataset_list = [2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,19,20,21]

    big_record = []
    for run_number in range(len(dataset_list)):
        data_set_number = dataset_list[run_number]
        # missing data
        if data_set_number == 2:
            studyname = 'JUN19_2018'
            reffilename = r'June 19- CP2017-004_CMRI2018-002.MR.DR_PUKALL_FMRI.0010.0001.2018.06.19.nii'
            maskfilename = r'June 19- CP2017-004_CMRI2018-002.MR.DR_PUKALL_FMRI.0010.0001.2018.06.19.nii.gz'

        if data_set_number == 3:
            studyname = 'JUN27_2018'
            maskfilename = r'June 27- Seg-CP2017-004_CMRI2018-003.MR.DR_PUKALL_FMRI.0004.0001.2018.06.27.nii.gz'
            maskfilename = r'June 27- Seg_new-CP2017-004_CMRI2018-003.MR.DR_PUKALL_FMRI.0004.0001.2018.06.27.nii.gz'
            reffilename = r'June 27- CP2017-004_CMRI2018-003.MR.DR_PUKALL_FMRI.0004.0001.2018.06.27.nii'

        if data_set_number == 4:
            studyname = 'OCT18_2018'
            maskfilename = r'Oct 18- Seg-CP2017-004_CMRI2018-004.MR.DR_PUKALL_FMRI.0004.0001.2018.10.18.nii.gz'
            reffilename = r'Oct 18- CP2017-004_CMRI2018-004.MR.DR_PUKALL_FMRI.0004.0001.2018.10.18.nii'

        if data_set_number == 5:
            studyname = 'OCT04_2018'
            maskfilename = r'Oct 4- Seg-CP2017-004_CMRI2018-005.MR.DR_PUKALL_FMRI.0004.0001.2018.10.04.nii.gz'
            maskfilename = r'Oct 4- Seg_new-CP2017-004_CMRI2018-005.MR.DR_PUKALL_FMRI.0004.0001.2018.10.04.nii.gz'
            reffilename = r'Oct 4- CP2017-004_CMRI2018-005.MR.DR_PUKALL_FMRI.0004.0001.2018.10.04.nii'

        # missing data
        if data_set_number == 6:
            studyname = 'OCT09_2018'
            maskfilename = r'Oct 9- Seg-CP2017-004_CMRI2018-006.MR.DR_PUKALL_FMRI.0005.0001.2018.10.09.nii.gz'
            reffilename = r'Oct 9- Seg-CP2017-004_CMRI2018-006.MR.DR_PUKALL_FMRI.0005.0001.2018.10.09.nii'

        if data_set_number == 7:
            studyname = 'OCT16_2018'
            maskfilename = r'Oct 16- Seg-CP2017-004_CMRI2018-007.MR.DR_PUKALL_FMRI.0002.0035.2018.10.16.nii.gz'
            reffilename = r'Oct 16- CP2017-004_CMRI2018-007.MR.DR_PUKALL_FMRI.0004.0001.2018.10.16.15.nii'

        if data_set_number == 8:
            studyname = 'NOV06_2018'
            maskfilename = r'Nov 6- Seg-CP2017-004_CMRI2018_-008.MR.DR_PUKALL_FMRI.0008.0001.2018.11.06.nii.gz'
            reffilename = r'Nov 6- CP2017-004_CMRI2018_-008.MR.DR_PUKALL_FMRI.0008.0001.2018.11.06.nii'

        if data_set_number == 9:
            studyname = 'OCT31_2018'
            maskfilename = r'Seg-CP2017-004_CMRI2018-009.MR.DR_PUKALL_FMRI.0004.0001.2018.10.31.nii.gz'
            maskfilename = r'Oct 31- Seg_new-CP2017-004_CMRI2018-009.MR.DR_PUKALL_FMRI.0004.0001.2018.10.31.nii.gz'
            reffilename = r'CP2017-004_CMRI2018-009.MR.DR_PUKALL_FMRI.0004.0001.2018.10.31.nii'

        # something is wrong
        if data_set_number == 11:
            studyname = 'NOV09_2018'
            maskfilename = r'Nov 9- Seg-CP2017-004_CMRI2018-011.MR.DR_PUKALL_FMRI.0004.0001.2018.11.09.nii.gz'
            maskfilename = r'Nov 9- Seg_new-CP2017-004_CMRI2018-011.MR.DR_PUKALL_FMRI.0004.0001.2018.11.09.nii.gz'
            reffilename = r'Nov 9- Seg-CP2017-004_CMRI2018-011.MR.DR_PUKALL_FMRI.0004.0001.2018.11.09.nii'
            # reffilename = r'Nov 9 - CP2017-004_CMRI2018-011_Series4.nii'

        if data_set_number == 12:
            # labels are 1 and 3 ?
            studyname = 'NOV27_2018'
            maskfilename = r'Nov 27- Seg-CP2017-004_CMRI-012.MR.DR_PUKALL_FMRI.0004.0001.2018.11.27.nii.gz'
            maskfilename = r'Nov 27- Seg_new-CP2017-004_CMRI-012.MR.DR_PUKALL_FMRI.0004.0001.2018.11.27.nii.gz'
            # maskfilename = r'Nov 27- CMRI2018-0001-series2.nii.gz'
            reffilename = r'Nov 27- CP2017-004_CMRI-012.MR.DR_PUKALL_FMRI.0004.0001.2018.11.27.nii'


        if data_set_number == 13:
            studyname = 'JAN21_2019'
            maskfilename = r'Jan 21- Seg-CP2017_004_CMRI2018-013.MR.DR_PUKALL_FMRI.0004.0001.2019.01.21.nii.gz'
            reffilename = r'Jan 21- CP2017_004_CMRI2018-013.MR.DR_PUKALL_FMRI.0004.0001.2019.01.21.nii'

        # problem with labeling
        if data_set_number == 14:
            studyname = 'JAN11_2019'
            maskfilename = r'Jan 11- Seg-CP2017_004_CMRI2018-014.MR.DR_PUKALL_FMRI.0004.0001.2019.01.11.nii.gz'
            reffilename = r'Jan 11- CP2017_004_CMRI2018-014.MR.DR_PUKALL_FMRI.0004.0001.2019.01.11.nii'

        if data_set_number == 15:
            studyname = 'JAN17_2019'
            maskfilename = r'Jan 17- Seg-CP2017_004_CMRI2018-015.MR.DR_PUKALL_FMRI.0004.0001.2019.01.17.nii.gz'
            reffilename = r'Jan 17- CP2017_004_CMRI2018-015.MR.DR_PUKALL_FMRI.0004.0001.2019.01.17.nii'

        if data_set_number == 16:
            studyname = 'JAN14_2019'
            maskfilename = r'Jan 14- Seg-CP2017_004_CMRI2018-016.MR.DR_PUKALL_FMRI.0010.0001.2019.01.14.nii.gz'
            reffilename = r'Jan 14- CP2017_004_CMRI2018-016.MR.DR_PUKALL_FMRI.0010.0001.2019.01.14.nii'

        # data missing
        if data_set_number == 17:
            studyname = 'FEB04_2019'
            # CMRI2018-017
            maskfilename = r'Feb 4- Seg-CP2017_004_CMRI2018-017.MR.DR_PUKALL_FMRI.0005.0009.2019.02.04.nii.gz'
            reffilename = r'Feb 4- CP2017_004_CMRI2018-017.MR.DR_PUKALL_FMRI.0005.0009.2019.02.04.nii'

        # data missing
        if data_set_number == 19:
            studyname = 'FEB14_2019'
            maskfilename = r'Feb 14- Seg-CP2017_004_CMRI2018-019.MR.DR_PUKALL_FMRI.0004.0001.2019.02.14.nii.gz'
            reffilename = r'Feb 14- CP2017_004_CMRI2018-019.MR.DR_PUKALL_FMRI.0004.0001.2019.02.14.nii'

        # data missing
        if data_set_number == 20:
            studyname = 'FEB13_2019'
            maskfilename = r'Feb 13- Seg-CP2017_004_CMRI2018-020.MR.DR_PUKALL_FMRI.0004.0001.2019.02.13.nii.gz'
            reffilename = r'Feb 13- CP2017_004_CMRI2018-020.MR.DR_PUKALL_FMRI.0004.0001.2019.02.13.nii'

        # data missing
        if data_set_number == 21:
            studyname = 'FEB08_2019'
            maskfilename = r'Feb 8- Seg-CP2017_004_CMRI2018-021.MR.DR_PUKALL_FMRI.0003.0002.2019.02.08.nii.gz'
            reffilename = r'Feb 8- CP2017_004_CMRI2018-021.MR.DR_PUKALL_FMRI.0003.0002.2019.02.08.nii'

        print('studyname = {}'.format(studyname))

        if data_set_number == 999:
            studyname = 'MAR01_2019'
            # CMRI2018-023

        pname = os.path.join(datadir,studyname)
        maskfilename = os.path.join(labeldir, maskfilename)
        reffilename = os.path.join(labeldir, reffilename)

        # define the data set and convert dicom to nifti if not already done
        seriesnumbers = []
        convert_dicom_to_nifti = False
        find_data_series = True

        pname = os.path.join(datadir,studyname)
        outputname = os.path.join(pname,r'mapping_results.npy')

        # load the mask data and get the mask affine matrix
        maskdata = nib.load(maskfilename)
        mask = maskdata.get_fdata()
        mask_affine = maskdata.affine
        mask_hdr = maskdata.header
        mask_size = np.shape(mask)

        # load the ref data and get the ref affine matrix
        refimg = nib.load(reffilename)
        ref_data = refimg.get_fdata()
        ref_affine = refimg.affine
        ref_hdr = refimg.header
        ref_size = np.shape(ref_data)

        labels = [7., 8.]
        mask = check_mask(mask,labels, ref_data)

        # organize the data and convert to nifti
        pname = os.path.join(datadir,studyname)
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

        # find the series numbers that have image sets with the same size
        # and could describe a time-series of changes
        if find_data_series:
            size_record = []
            for ss in serieslist:
                basename = 'Series{}'.format(ss)
                fullpath = os.path.join(pname,basename)
                niiname = os.path.join(fullpath,'Series{}.nii'.format(ss))

                input_img = nib.load(niiname)
                input_data = input_img.get_fdata()
                affine = input_img.affine
                input_hdr = input_img.header
                dims = np.shape(input_data)

                size_record += [dims]

            size_record = np.array(size_record)
            usize,index,counts = np.unique(size_record,axis=0,return_counts = True,return_inverse = True)
            x = np.argmax(counts)
            c = np.where(index == x)[0]
            seriesnumbers  = serieslist[c]

        imgdata = []
        datadetails = []
        for ss in seriesnumbers:
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

            input_data2 = i3d.convert_affine_matrices_nearest(input_data, affine, affine2,
                                                        mask_size)

            input_data2 = i3d.convert_affine_matrices_nearest(input_data2, affine2, mask_affine,
                                                        mask_size)

            # input_data2 = copy.deepcopy(input_data)

            imgdata += [input_data2]
            # datadetails.append({'img':input_data2, 'hdr':input_hdr, 'affine':ref_affine, 'original_img':input_data})
            datadetails.append({'img':input_data2, 'hdr':input_hdr, 'affine':affine2, 'original_img':input_data})
        imgdata = np.array(imgdata)

        # mask2 = i3d.convert_affine_matrices_nearest(mask, mask_affine, datadetails[0]['affine'], np.shape(datadetails[0]['img']))
        mask2 = copy.deepcopy(mask)

        # display one slice with mask overlaid
        # crop the image data to focus on the region of interest
        xc, yc, zc = np.where(mask2 > 0)
        buffer = 10
        x0 = xc.min()-buffer
        x1 = xc.max() + buffer
        y0 = yc.min()-buffer
        y1 = yc.max() + buffer
        z0 = zc.min()-buffer
        z1 = zc.max() + buffer

        cropped_mask = copy.deepcopy(mask2[x0:x1,y0:y1,z0:z1])
        xc, yc, zc = np.where(cropped_mask > 0)


        cols = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
                         [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
                         [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]])
        nvals = np.unique(mask2)
        img1 = copy.deepcopy(imgdata[0,x0:x1,y0:y1,z0:z1])

        img1 = copy.deepcopy(datadetails[0]['img'][x0:x1,y0:y1,z0:z1])
        ref_data_cropped = copy.deepcopy(ref_data[x0:x1,y0:y1,z0:z1])

        # load the processed data
        results = np.load(outputname, allow_pickle=True).flat[0]
        big_record.append(results)

    data_record = []
    for nn in range(len(big_record)):
        results = big_record[nn]
        norm_record = big_record[nn]['norm_record']
        cropped_mask = big_record[nn]['cropped_mask']
        FOV = big_record[nn]['FOV']
        voxvol = big_record[nn]['voxvol']
        pixdim = big_record[nn]['pixdim']
        measure_record = big_record[nn]['measure_record']

        nmeasures = len(measure_record)
        vol1 = [measure_record[x]['vol1'] for x in range(nmeasures)]
        intensity1 = [measure_record[x]['intensity1'] for x in range(nmeasures)]
        intensity1_sd = [measure_record[x]['intensity1_sd'] for x in range(nmeasures)]
        vol2 = [measure_record[x]['vol2'] for x in range(nmeasures)]
        intensity2 = [measure_record[x]['intensity2'] for x in range(nmeasures)]
        intensity2_sd = [measure_record[x]['intensity2_sd'] for x in range(nmeasures)]

        data_record.append({'nmeasures':nmeasures, 'intensity1':intensity1, 'intensity1_sd':intensity1_sd,
                            'intensity2':intensity2, 'intensity2_sd':intensity2_sd, 'vol1':vol1, 'vol2':vol2})

        if nn == 0:
            intensity1_record = np.zeros((len(big_record),nmeasures))
            intensity2_record = np.zeros((len(big_record),nmeasures))
            vol1_record = np.zeros((len(big_record),nmeasures))
            vol2_record = np.zeros((len(big_record),nmeasures))

        intensity1_record[nn,:] = copy.deepcopy(intensity1)
        intensity2_record[nn,:] = copy.deepcopy(intensity2)
        vol1_record[nn,:] = copy.deepcopy(vol1)
        vol2_record[nn,:] = copy.deepcopy(vol2)


    # calculate relative intensity changes
    rel_intensity1_record = copy.deepcopy(intensity1_record)
    rel_intensity2_record = copy.deepcopy(intensity2_record)
    for nn in range(len(big_record)):
        rel_intensity1_record[nn,:] = 100.0*intensity1_record[nn,:]/intensity1_record[nn,0]
        rel_intensity2_record[nn,:] = 100.0*intensity2_record[nn,:]/intensity2_record[nn,0]

    avg_intensity1 = np.mean(intensity1_record, axis = 0)
    avg_intensity2 = np.mean(intensity2_record, axis = 0)
    sem_intensity1 = np.std(intensity1_record, axis = 0)   # /np.sqrt(len(big_record))
    sem_intensity2 = np.std(intensity2_record, axis = 0)   # /np.sqrt(len(big_record))

    avg_rel_intensity1 = np.mean(rel_intensity1_record, axis = 0)
    avg_rel_intensity2 = np.mean(rel_intensity2_record, axis = 0)
    sem_rel_intensity1 = np.std(rel_intensity1_record, axis = 0)   # /np.sqrt(len(big_record))
    sem_rel_intensity2 = np.std(rel_intensity2_record, axis = 0)   # /np.sqrt(len(big_record))

    # calculate relative volume changes
    rel_vol1_record = copy.deepcopy(vol1_record)
    rel_vol2_record = copy.deepcopy(vol2_record)
    for nn in range(len(big_record)):
        rel_vol1_record[nn,:] = 100.0*rel_vol1_record[nn,:]/rel_vol1_record[nn,0]
        rel_vol2_record[nn,:] = 100.0*rel_vol2_record[nn,:]/rel_vol2_record[nn,0]

    avg_vol1 = np.mean(vol1_record, axis = 0)
    avg_vol2 = np.mean(vol2_record, axis = 0)
    sem_vol1 = np.std(vol1_record, axis = 0)   # /np.sqrt(len(big_record))
    sem_vol2 = np.std(vol2_record, axis = 0)   # /np.sqrt(len(big_record))

    avg_rel_vol1 = np.mean(rel_vol1_record, axis = 0)
    avg_rel_vol2 = np.mean(rel_vol2_record, axis = 0)
    sem_rel_vol1 = np.std(rel_vol1_record, axis = 0)   # /np.sqrt(len(big_record))
    sem_rel_vol2 = np.std(rel_vol2_record, axis = 0)   # /np.sqrt(len(big_record))

    # absolute measures
    winnum = 10
    plt.close(winnum)
    fig = plt.figure(winnum)
    fig.add_subplot(1, 2, 1)
    nmeasures = 6
    for nn in range(len(data_record)):
        intensity1 = data_record[nn]['intensity1']
        intensity1_sd = data_record[nn]['intensity1_sd']
        intensity2 = data_record[nn]['intensity2']
        intensity2_sd = data_record[nn]['intensity2_sd']
        plt.errorbar(range(nmeasures),intensity1,yerr=intensity1_sd,linestyle='-',color = 'g', marker = 'o')
        plt.errorbar(range(nmeasures),intensity2,yerr=intensity2_sd,linestyle='-',color = 'y', marker = 'o')

    plt.xlabel('image volume number')
    plt.ylabel('image intensity (avg +/- sd) (arb. units)')

    fig.add_subplot(1, 2, 2)
    nmeasures = 6
    for nn in range(len(data_record)):
        vol1 = vol1_record[nn,:]
        vol2 = vol2_record[nn,:]
        plt.plot(range(nmeasures),np.array(vol1)/1000.0,linestyle='-',color = 'g', marker = 'o')
        plt.plot(range(nmeasures),np.array(vol2)/1000.0,linestyle='-',color = 'y', marker = 'o')

    plt.xlabel('image volume number')
    plt.ylabel('region volume (CC)')

    svgname = os.path.join(datadir, 'individual_volume_and_intensity_measures_absolute.svg')
    plt.savefig(svgname, format='svg')


    winnum = 11
    plt.close(winnum)
    fig = plt.figure(winnum)
    fig.add_subplot(1, 2, 1)
    plt.errorbar(range(nmeasures), avg_intensity1, yerr=sem_intensity1, linestyle='-', color='g', marker='o')
    plt.errorbar(range(nmeasures), avg_intensity2, yerr=sem_intensity2, linestyle='-', color='y', marker='o')

    plt.xlabel('image volume number')
    plt.ylabel('image intensity (avg +/- sd) (arb. units)')

    fig.add_subplot(1, 2, 2)
    plt.errorbar(range(nmeasures), avg_vol1/1000.0, yerr=sem_vol1/1000.0, linestyle='-', color='g', marker='o')
    plt.errorbar(range(nmeasures), avg_vol2/1000.0, yerr=sem_vol2/1000.0, linestyle='-', color='y', marker='o')

    plt.xlabel('image volume number')
    plt.ylabel('region volume (CC)')

    svgname = os.path.join(datadir, 'group_volume_and_intensity_measures_absolute.svg')
    plt.savefig(svgname, format='svg')



    # relative measures
    winnum = 12
    plt.close(winnum)
    fig = plt.figure(winnum)
    fig.add_subplot(1, 2, 1)
    nmeasures = 6
    for nn in range(len(data_record)):
        intensity1 = rel_intensity1_record[nn,:]
        # intensity1_sd = data_record[nn]['intensity1_sd']
        intensity2 = rel_intensity2_record[nn,:]
        # intensity2_sd = data_record[nn]['intensity2_sd']
        plt.plot(range(nmeasures),intensity1,linestyle='-',color = 'g', marker = 'o')
        plt.plot(range(nmeasures),intensity2,linestyle='-',color = 'y', marker = 'o')

    plt.xlabel('image volume number')
    plt.ylabel('rel. image intensity (%)')

    fig.add_subplot(1, 2, 2)
    nmeasures = 6
    for nn in range(len(data_record)):
        vol1 = rel_vol1_record[nn,:]
        vol2 = rel_vol2_record[nn,:]
        plt.plot(range(nmeasures),np.array(vol1),linestyle='-',color = 'g', marker = 'o')
        plt.plot(range(nmeasures),np.array(vol2),linestyle='-',color = 'y', marker = 'o')

    plt.xlabel('image volume number')
    plt.ylabel('relative region volume (%)')

    svgname = os.path.join(datadir, 'individual_volume_and_intensity_measures_relative.svg')
    plt.savefig(svgname, format='svg')


    winnum = 13
    plt.close(winnum)
    fig = plt.figure(winnum)
    fig.add_subplot(1, 2, 1)
    plt.errorbar(range(nmeasures), avg_rel_intensity1, yerr=sem_rel_intensity1, linestyle='-', color='g', marker='o')
    plt.errorbar(range(nmeasures), avg_rel_intensity2, yerr=sem_rel_intensity2, linestyle='-', color='y', marker='o')

    plt.xlabel('image volume number')
    plt.ylabel('rel. image intensity (%)')

    fig.add_subplot(1, 2, 2)
    plt.errorbar(range(nmeasures), avg_rel_vol1, yerr=sem_rel_vol1, linestyle='-', color='g', marker='o')
    plt.errorbar(range(nmeasures), avg_rel_vol2, yerr=sem_rel_vol2, linestyle='-', color='y', marker='o')

    plt.xlabel('image volume number')
    plt.ylabel('relative region volume (%)')

    svgname = os.path.join(datadir, 'group_volume_and_intensity_measures_relative.svg')
    plt.savefig(svgname, format='svg')
