# move data to new location, based on a database file
import numpy as np
import pandas as pd
import openpyxl
import os
import shutil

folder_to_backup = r"D:\threat_safety_python\backup_copy_of_original_data"
new_folder = r"D:\threat_safety_python\dicom_backup"
if not os.path.isdir(new_folder):
    os.mkdir(new_folder)

DICOMlistfull = []  # create an empty list
seriesnumberlist = []
DICOMextension = '.ima'
for dirName, subdirList, fileList in os.walk(folder_to_backup):
    newdirName = dirName.replace(folder_to_backup,new_folder)
    if not os.path.isdir(newdirName):
        os.mkdir(newdirName)
    for filename in fileList:
        if DICOMextension in filename.lower():  # check whether the file is DICOM
            singlefile = os.path.join(dirName,filename)
            new_singlefile = os.path.join(newdirName,filename)

            print('move {} to {}'.format(singlefile,new_singlefile))
            shutil.move(singlefile, new_singlefile)
