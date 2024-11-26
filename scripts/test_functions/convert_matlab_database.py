# move data to new location, based on a database file
import numpy as np
import pandas as pd
import openpyxl
import os
import shutil
import scipy.io
import copy

# read mat file
matDBname = r'Y:\PVDstudy_data\LYessick_braindata.mat'
output_DBname = r'Y:\PVDstudy_data\PVD_brain_database_April2024.xlsx'
matdata = scipy.io.loadmat(matDBname)
datarecord = matdata['datarecord']

input_fields = ['patientid', 'datadir', 'studygroup1', 'seriesnumber', 'studygroup2', 'firstpainrating',
                'lastpainrating', 'temperature', 'niftinames']

root_dir = r'Y:\PVDstudy_data\PVD'

#setup to write out data to excel file in the correct format

required_db_sheets = ['datarecord', 'paradigm']
required_db_fields = ['datadir', 'patientid', 'studygroup', 'pname', 'seriesnumber', 'niftiname', 'TR', 'normdataname',
                      'normtemplatename', 'paradigms']
optional_db_fields = ['pulsefilename', 'sex', 'age', 'firstpain', 'lastpain', 'temperature']  # examples that can be changed
db_fields = required_db_fields + optional_db_fields

TR = 3.0
niftiname = 'NA'
normdataname = 'NA'
normtemplatename = 'brain'
paradigms = 'paradigm1'
pulsefilename = 'NA'
sex = 'F'
age = 0

# special case------------------------------------------
extradatafile = r'Y:\PVDstudy_data\LY2018_Protocol_BrainData.xlsx'
xls = pd.ExcelFile(extradatafile, engine='openpyxl')
xls_sheets = xls.sheet_names
#-------------------------------------------------------


old_base_directory = str(r'F:\fMRI CCB and PVD\FMRI data\MRI data\data')
new_base_directory = str(r'Y:\PVDstudy_data')
destination_directory = r'Y:\PVDstudy_data\PVD2'

NP = len(datarecord[0][:])
data = []
already_copied_list = []
for nn in range(NP):
    patientid = copy.deepcopy(datarecord[0][nn][0][0])
    pname_full = str(datarecord[0][nn][1][0])
    seriesnumber = datarecord[0][nn][2][0][0]
    studygroup = datarecord[0][nn][3][0]
    firstpain = datarecord[0][nn][4][0][0]
    lastpain = datarecord[0][nn][5][0][0]
    temperature = datarecord[0][nn][6][0][0]

    pname_full_actual = pname_full.replace(old_base_directory, new_base_directory)

    # special case------------------------------------------
    df1 = pd.read_excel(xls, patientid)
    age = df1.keys()[2]
    # ------------------------------------------------------

    path_parts = os.path.normpath(pname_full_actual).split(os.path.sep)
    pname = path_parts[-1]
    new_directory = os.path.join(destination_directory, pname)

    if os.path.exists(new_directory):
        print('\n\n{} already exists ...'.format(new_directory))
    else:
        os.mkdir(new_directory)
        print('\n\ncreated folder: {}'.format(new_directory))

    if not patientid in already_copied_list:
        # copy all the files for this patientid
        for file_name in os.listdir(pname_full_actual):
            f,e = os.path.splitext(file_name)
            if e == '.IMA':
                # construct full file path
                source = os.path.join(pname_full_actual, file_name)
                destination = os.path.join(new_directory, file_name)
                # copy only files
                if os.path.isfile(source):
                    if not os.path.isfile(destination):
                        shutil.copy(source, destination)
                        print('copied', file_name)
        already_copied_list.append(patientid)

    # required_db_fields = ['datadir', 'patientid', 'studygroup', 'pname', 'seriesnumber', 'niftiname', 'TR',
    #                       'normdataname', 'normtemplatename', 'paradigms']
    # optional_db_fields = ['pulsefilename', 'sex', 'age', 'firstpain', 'lastpain',
    #                       'temperature']  # examples that can be changed

    entries = [new_directory, patientid, studygroup, pname, seriesnumber, niftiname, TR, normdataname, normtemplatename,
               paradigms, pulsefilename, sex, age, firstpain, lastpain, temperature]
    entry = dict(zip(db_fields, entries))
    data.append(entry)

df1 = pd.DataFrame(data=data)

# DBname = settings['DBname']
# write it to the database by appending a sheet to the excel file
with pd.ExcelWriter(output_DBname) as writer:
    df1.to_excel(writer, sheet_name='datarecord')

# initialize the paradigm sheet
required_pd_sheets = ['dt', 'paradigm']
sample_paradigm = list(zip(5 * np.ones(12), [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]))
df2 = pd.DataFrame(columns=required_pd_sheets, data=sample_paradigm)
with pd.ExcelWriter(output_DBname, engine="openpyxl", mode='a') as writer:
    df2.to_excel(writer, sheet_name='paradigm')

