
import numpy as np
import pandas as pd
import openpyxl
import os
import shutil
import scipy.io
import copy

# load data set and create a copy with only NIfTI format images, without any pre-processing applied
# create a corresponding database file as well

# PVD brain data
# DBname = r'F:\PVD_study\PVD_brain_database_April2024.xlsx'
# new_DBname = r'F:\PVD_study_shared\PVD_brain_database_shared.xlsx'
# participant_type_list = ['PVD', 'CON']
# sheets_to_copy = ['paradigm1']  # datarecord is already implied
# columns_to_replace = ['normdataname', 'pulsefilename', 'norm_quality', 'coreg_quality']

# FM brain data
# DBname = r'D:\Howie_FM2_Brain_Data\Howie_FMS2_brain_fMRI_database_JAN2020_V2.xlsx'
# new_DBname = r'F:\FMS2_brain_shared\FM_brain_database_shared.xlsx'
# participant_type_list = ['FM', 'HC']
# sheets_to_copy = ['paradigm1']  # datarecord is already implied
# columns_to_replace = ['normdataname', 'pulsefilename', 'norm_quality', 'coreg_quality']

# FM BSSC data
# DBname = r'E:\FM2021data\FMS2_database_July27_2022b.xlsx'
# new_DBname = r'F:\FMS2_bssc_shared\FM_bssc_database_shared.xlsx'
# participant_type_list = ['FM', 'HC']
# sheets_to_copy = ['paradigm1']  # datarecord is already implied
# columns_to_replace = ['normdataname', 'pulsefilename', 'norm_quality', 'coreg_quality']

# graded pain BSSC data - 3 different studies
DBname = r'E:\graded_pain_database_May2022.xlsx'
new_DBname = r'F:\multiple_conditions_bssc_shared\multipain_bssc_database_shared.xlsx'
participant_type_list = ['notused']
sheets_to_copy = ['paradigm1']  # datarecord is already implied
columns_to_replace = ['normdataname', 'pulsefilename', 'norm_quality', 'coreg_quality']

p,f = os.path.split(new_DBname)
new_datadir = copy.deepcopy(p)

if not os.path.exists(new_datadir):
    os.makedirs(new_datadir)

xls = pd.ExcelFile(DBname, engine='openpyxl')
xls_sheets = xls.sheet_names

# read datarecord sheet
df1 = pd.read_excel(xls, 'datarecord')
keylist = df1.keys()
for kname in keylist:
    if 'Unnamed' in kname: df1.pop(kname)  # remove blank fields from the database

num = len(df1)   # number of entries
# create new names for each participant
idlist = []
basepnamelist = []
typelist = []
for nn in range(num):
    patientid = copy.deepcopy(df1.loc[nn, 'patientid'])
    pname = copy.deepcopy(df1.loc[nn, 'pname'])
    studygroup = copy.deepcopy(df1.loc[nn, 'studygroup'])
    pname_parts = os.path.normpath(pname).split(os.path.sep)
    idlist += [patientid]
    basepnamelist += [pname_parts[0]]

    pt = np.array([participant_type_list[aa] in studygroup for aa in range(len(participant_type_list))])
    c = np.where(pt)[0]
    if len(c) < 1:
        # participant_type = 'undefined'
        participant_type = 'HC'
    else:
        participant_type = participant_type_list[c[0]]
    typelist += [participant_type]


unique_id_list, uii = np.unique(idlist, return_index = True)
new_pid = []
new_pname = []
unique_types = np.unique(typelist)
ntypes = len(unique_types)
count_for_each_type = np.zeros(ntypes)
ids_counted = []
id_conversion = []
for nn in range(num):
    # c = np.where(unique_id_list == idlist[nn])[0]
    c = np.where(typelist[nn] == unique_types)[0]
    if idlist[nn] in ids_counted:
        check = [id_conversion[xx]['id'] == idlist[nn] for xx in range(len(id_conversion))]
        c = np.where(check)[0]
        pid = id_conversion[c[0]]['new_id']
    else:
        count_for_each_type[c[0]] += 1
        thiscount = int(count_for_each_type[c[0]])
        ids_counted += [idlist[nn]]
        pid = '{}subject{:02d}'.format(typelist[nn], thiscount)
        id_conversion.append({'id':idlist[nn],'new_id':pid})

    # thiscount = int(c[0]+1)

    new_pid += [pid]
    new_pname += [pid]


# now replace values in datarecord
for nn in range(num):
    df1.loc[nn, 'patientid'] = new_pid[nn]
    df1.loc[nn, 'patientid'] = new_pid[nn]
    seriesnumber = int(float(df1.loc[nn, 'seriesnumber']))
    temp_pname = os.path.join(new_pname[nn], 'Series{}'.format(seriesnumber))
    df1.loc[nn, 'pname'] = temp_pname
    temp_niftiname = 'Series{}.nii'.format(seriesnumber)
    temp_dirname_root = os.path.join(new_datadir,new_pname[nn])
    temp_dirname = os.path.join(new_datadir,temp_pname)
    new_datafile_name = os.path.join(temp_dirname,temp_niftiname)

    datadir = copy.deepcopy(df1.loc[nn, 'datadir'])
    niftiname = copy.deepcopy(df1.loc[nn, 'niftiname'])
    old_datafile_name = os.path.join(datadir,niftiname)

    # create data folder if it does not exist
    if not os.path.exists(temp_dirname_root):
        os.makedirs(temp_dirname_root)
    if not os.path.exists(temp_dirname):
        os.makedirs(temp_dirname)

    # copyfile
    if os.path.exists(new_datafile_name):
        print('file already exists: {}'.format(new_datafile_name))
    else:
        print('copying {} to {}'.format(old_datafile_name, new_datafile_name))
        shutil.copyfile(old_datafile_name, new_datafile_name)

    new_niftiname = os.path.join(temp_pname,temp_niftiname)
    df1.loc[nn, 'niftiname'] = new_niftiname

    # overwrite unwanted values last
    df1.loc[nn, 'datadir'] = 'enter_base_directory_here'
    for colname in columns_to_replace:
        df1.loc[nn, colname] = 'notdefined'


# copy other sheets also
extra_df = []
for aa, sheetname in enumerate(sheets_to_copy):
    extra_df += [pd.read_excel(xls, sheetname)]

# write database data
with pd.ExcelWriter(new_DBname) as writer:
    df1.to_excel(writer, sheet_name='datarecord')
    for aa, sheetname in enumerate(sheets_to_copy):
        extra_df[aa].to_excel(writer, sheet_name=sheetname)
