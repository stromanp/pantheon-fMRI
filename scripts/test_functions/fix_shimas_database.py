
import numpy as np
import pandas as pd
import openpyxl
import os
import shutil
import scipy.io
import copy


DBname = r'E:\FMstudy2023\PS2023_database.xlsx'
new_DBname = r'E:\FMstudy2023\PS2023_database_corrected.xlsx'
xls = pd.ExcelFile(DBname, engine='openpyxl')
xls_sheets = xls.sheet_names

df1 = pd.read_excel(xls, 'datarecord')
keylist = df1.keys()
for kname in keylist:
    if 'Unnamed' in kname: df1.pop(kname)  # remove blank fields from the database

num = len(df1)   # number of entries
basedir = 'FM2023'
for nn in range(num):
    pname = copy.deepcopy(df1.loc[nn, 'pname'])
    # try:
    #     pname_small = pname.replace(basedir, '')[1:]
    # except:
    #     pname_small = pname

    try:
        path_parts = os.path.normpath(pname).split(os.path.sep)
        new_pname = path_parts[1].upper()
        df1.loc[nn, 'pname'] = new_pname
    except:
        print('pname = {}'.format(pname))

    patientgroup = copy.deepcopy(df1.loc[nn, 'patientgroup'])
    runtype = copy.deepcopy(df1.loc[nn, 'runtype'])
    try:
        new_patientgroup = patientgroup + runtype[:4]
    except:
        new_patientgroup = patientgroup

    df1.loc[nn, 'patientgroup'] = new_patientgroup


# write database data
with pd.ExcelWriter(new_DBname) as writer:
    df1.to_excel(writer, sheet_name='datarecord')
