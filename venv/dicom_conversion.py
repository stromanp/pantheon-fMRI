# -*- coding: utf-8 -*-
"""
This module includes functions to organize dicom format files,
convert dicom to NIfTI, and update the database file

Created on Sun Apr 26 15:38:31 2020

@author: stroman
"""

import os
import pydicom
import numpy
import pandas as pd
import shutil  
import dicom2nifti
import openpyxl

# move files into subfolders if wanted
# update database file if files moved, if wanted
# read a data set
# convert a data set to NIfTI

# function to find the indices of a list that match a value
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


def get_database_numbers(databasename, dbhome, pname, seriesnumber = 0):
    # BASEdir = os.path.dirname(databasename)
    pname_sub = pname.replace(dbhome,'')
    pname_sub = pname_sub.lstrip(os.sep)   # remove the leading separator that is left behind
    
    xls = pd.ExcelFile(databasename, engine = 'openpyxl')
    df1 = pd.read_excel(xls, 'datarecord')
    
    if seriesnumber > 0:
        location = df1.loc[(df1['pname']==pname_sub)  &  (df1['seriesnumber']==seriesnumber)].index.values
    else:
        location = df1.loc[df1['pname']==pname_sub].index.values
        
    return location


def move_files_and_update_database(databasename, dbhome, pname):
    # find all of the files in a dataset directory

    # BASEdir = os.path.dirname(databasename)
    pname_sub = pname.replace(dbhome,'')
    pname_sub = pname_sub.lstrip(os.sep)   # remove the leading separator that is left behind
    
    DICOMlistfull = []  # create an empty list
    seriesnumberlist = []
    DICOMextension = '.ima'

    # original method - not good if folder has already been organized
    # for dirName, subdirList, fileList in os.walk(pname):
    #     for filename in fileList:
    #         if DICOMextension in filename.lower():  # check whether the file is DICOM
    #             singlefile = os.path.join(dirName,filename)
    #             DICOMlistfull.append(singlefile)
    #             ds = pydicom.dcmread(singlefile)
    #             seriesnumberlist.append(ds.SeriesNumber)

    for filename in os.listdir(pname):
        if DICOMextension in filename.lower():  # check whether the file is DICOM
            fullname = os.path.join(pname,filename)
            DICOMlistfull.append(fullname)
            ds = pydicom.dcmread(fullname)
            seriesnumberlist.append(ds.SeriesNumber)

    x = numpy.array(seriesnumberlist)
    serieslist = numpy.unique(x)
    print('serieslist = ', serieslist)
    
    if len(serieslist) > 1:   # don't do anything if the directory contains a single series
        # find all entries in the database matching pname
        # and list the series numbers for these entries
        xls = pd.ExcelFile(databasename, engine = 'openpyxl')
        df1 = pd.read_excel(xls, 'datarecord')
        keylist = df1.keys()
        for kname in keylist:
            if 'Unnamed' in kname:
                df1.pop(kname)   # remove this blank field from the beginning
        
        # to write an excel file:    df.to_excel(outputname)
        for snum in serieslist:
            # find which database entries match the seriesnumber = snum
            # and then replace the pname with the new folder name
            dbindex = get_database_numbers(databasename, dbhome, pname, seriesnumber = snum)
            
            # find all of the files in a series
            ii = find(seriesnumberlist, snum)
            temp_array = numpy.array(DICOMlistfull)
            list_of_dicom_files = temp_array[ii]
            
            # check if the subfolder needs to be created
            subfolder = 'Series{number}'.format(number=snum)
            check = pname_sub.find(subfolder)
            if check == -1:
                subfolderpath = os.path.join(pname_sub,subfolder)
            else:
                subfolderpath = subfolder  # don't add another layer if the subfolder already exists
            
            # create the new sub-folder
            # move the dicom files to the new folder
            subfolderabspath = os.path.join(dbhome,subfolderpath)
            if not os.path.isdir(subfolderabspath):
                os.mkdir(subfolderabspath)
            
            # update the database entries, if they match the directory and series number
#            # niftiname  needs to be changed first, if it has been set already
            # there should only be one matching series number for a given directory
            if len(dbindex) > 0:
                niftiname = df1.loc[dbindex[0], 'niftiname']
                nameparts = os.path.split(niftiname)
                nameparts2 = os.path.splitext(nameparts[1])
                niiext = '.nii'
                if nameparts2[1] == niiext:   # if a niftiname has already been specified, then deal with it, otherwise do nothing
                    newniftiname = os.path.join(nameparts[0], subfolder, nameparts[1])
                    niftinamefull = os.path.join(dbhome,niftiname)
                    newniftinamefull = os.path.join(dbhome,newniftiname)
                    # if a file called niftinamefull exists, need to move it to the new location
                    if os.path.isfile(niftinamefull):
                        shutil.move(niftinamefull, newniftinamefull)
                    df1.loc[dbindex[0], 'niftiname'] = newniftiname
                    print('{}  updating niftiname to {}'.format(dbindex[0],newniftiname))
                    
                # normdataname  needs to be changed next, if it has been set already
                normname = df1.loc[dbindex[0], 'normdataname']
                nameparts = os.path.split(normname)
                nameparts2 = os.path.splitext(nameparts[1])
                normext = '.npy'  # this will probably need to be updated-----------------------------------------------------------
                if nameparts2[1] == normext:   # if a normdataname has already been specified, then deal with it, otherwise do nothing
                    newnormname = os.path.join(nameparts[0], subfolder, nameparts[1])
                    normnamefull = os.path.join(dbhome,normname)
                    newnormnamefull = os.path.join(dbhome,newnormname)
                    # if a file called niftinamefull exists, need to move it to the new location
                    if os.path.isfile(normnamefull):
                        shutil.move(normnamefull, newnormnamefull)
                    df1.loc[dbindex[0], 'normdataname'] = newnormname
                    print('{}  updating normdataname to {}'.format(dbindex[0],newnormname))
                      
                # now replace the pname for the data, with the new name
                df1.loc[dbindex[0], 'pname'] = subfolderpath
                print('{}  updating pname to {}'.format(dbindex[0],subfolderpath))
            
            for dicomname in list_of_dicom_files:
                nameparts = os.path.split(dicomname)
                newdicomname = os.path.join(nameparts[0], subfolder, nameparts[1])
                # move the dicom file to the new location
                shutil.move(dicomname, newdicomname)

        # need to delete the existing database sheet before writing the new one
        workbook = openpyxl.load_workbook(databasename)
        std = workbook.get_sheet_by_name('datarecord')
        workbook.remove_sheet(std)
        workbook.save(databasename)

        # write it to the database by appending a sheet to the excel file
        with pd.ExcelWriter(databasename, engine="openpyxl", mode='a') as writer:
            df1.to_excel(writer, sheet_name='datarecord')


def convert_dicom_folder(databasename, databasenumber, basename = 'Series'):
    # BASEdir = os.path.dirname(databasename)
    xls = pd.ExcelFile(databasename, engine = 'openpyxl')
    df1 = pd.read_excel(xls, 'datarecord')
    df1.pop('Unnamed: 0')   # remove this blank field from the beginning
    dbhome = df1.loc[databasenumber, 'datadir']
    dicom_directory = df1.loc[databasenumber, 'pname']
    seriesnumber = df1.loc[databasenumber, 'seriesnumber']
    niiname = '{base}{number}.nii'.format(base = basename, number=seriesnumber)
    
    output_file = os.path.join(dbhome, dicom_directory, niiname)
    dicom_directory_full = os.path.join(dbhome, dicom_directory)

    print('output_file = ',output_file)
    print('dicom_directory_full = ',dicom_directory_full)
    
    # still need to check the orientation for both BS/SC data and brain data
    dicom2nifti.dicom_series_to_nifti(dicom_directory_full, output_file, reorient_nifti=True)
    # this will put images in the very stupid but "standard" LAS orientation, which is left-handed
    
    # now update the database with the new niftiname
    df1.loc[databasenumber, 'niftiname'] = os.path.join(dicom_directory, niiname)
    # df1.to_excel(databasename, sheet_name='datarecord')

    # need to delete the existing database sheet before writing the new one
    workbook = openpyxl.load_workbook(databasename)
    std = workbook.get_sheet_by_name('datarecord')
    workbook.remove_sheet(std)
    workbook.save(databasename)

    # write it to the database by appending a sheet to the excel file
    with pd.ExcelWriter(databasename, engine="openpyxl", mode='a') as writer:
        df1.to_excel(writer, sheet_name='datarecord')

    return output_file
    

# functions to load or create templates?