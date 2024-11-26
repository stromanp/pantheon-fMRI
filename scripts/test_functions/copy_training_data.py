# copy data for testing ML method
import pandas as pd
import os
import shutil

DBname = r'D:\threat_safety_python\threat_safety_database.xlsx'
DBnums = list(range(278))

destdir = r'Z:\stromanlab\training_data'

xls = pd.ExcelFile(DBname, engine='openpyxl')
df1 = pd.read_excel(xls, 'datarecord')
df1.pop('Unnamed: 0')   # remove this blank field from the beginning
basename = 'normdata_s'
for dbnum in DBnums:
    dbhome = df1.loc[dbnum, 'datadir']
    pname = df1.loc[dbnum, 'pname']
    dicom_directory = df1.loc[dbnum, 'pname']
    seriesnumber = df1.loc[dbnum, 'seriesnumber']
    niftiname = df1.loc[dbnum, 'niftiname']
    niiname = os.path.join(dbhome, niftiname)
    normdatanamebase = '{base}{number}.npy'.format(base = basename, number=seriesnumber)
    normdataname = os.path.join(dbhome, pname, normdatanamebase)

    niitarget = os.path.join(destdir, niftiname)
    normdatanametarget = os.path.join(destdir, pname, normdatanamebase)
    targetdir = os.path.join(destdir, pname)

    if not os.path.isdir(targetdir):
        os.makedirs(targetdir)

    try:
        shutil.copy(niiname, niitarget)
        shutil.copy(normdataname, normdatanametarget)
        print('{} copied  nifti file {} to {}'.format(dbnum,niiname,niitarget))
        print('{}         norm data  {} to {}'.format(dbnum,normdataname,normdatanametarget))
    except:
        print('database number {}  -  some problem with copying files'.format(dbnum))