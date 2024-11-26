# move data to new location, based on a database file
import numpy as np
import pandas as pd
import openpyxl
import os
import shutil

DBname = r"D:\threat_safety_python\threat_safety_database.xlsx"
xls = pd.ExcelFile(DBname, engine='openpyxl')
df1 = pd.read_excel(xls, 'datarecord')
nentries = len(df1)

new_data_dir = r'D:\threat_safety_python'
sourcelist = []
destinationlist = []
elena_data_folder = 'TouchPain'
for nn in range(nentries):
    pname = df1.loc[nn, 'pname']
    datadir = df1.loc[nn, 'datadir']
    niftiname = df1.loc[nn, 'niftiname']
    normdataname = df1.loc[nn, 'normdataname']
    pname = pname[:-1]
    datadir = datadir[:-1]

    sourcedir = os.path.join(datadir,pname)

    if datadir == r"Z:\stromanlab\Elena\Data":
        new_pname = os.path.join(elena_data_folder,pname)
        new_normdataname = os.path.join(elena_data_folder,normdataname)
        new_niftiname = os.path.join(elena_data_folder,niftiname)

        df1.loc[nn, 'pname'] = new_pname
        df1.loc[nn, 'normdataname'] = new_normdataname
        df1.loc[nn, 'niftiname'] = new_niftiname
    else:
        df1.loc[nn, 'pname'] = pname

# write out new database values
# need to delete the existing sheet before writing the new version
existing_sheet_names = xls.sheet_names
if 'datarecord' in existing_sheet_names:
    # delete sheet - need to use openpyxl
    workbook = openpyxl.load_workbook(DBname)
    # std = workbook.get_sheet_by_name('datarecord')
    # workbook.remove_sheet(std)
    del workbook['datarecord']
    workbook.save(DBname)

# write it to the database by appending a sheet to the excel file
# remove old version of datarecord first
with pd.ExcelWriter(DBname) as writer:
    df1.to_excel(writer, sheet_name='datarecord')

