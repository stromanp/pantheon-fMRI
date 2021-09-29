# set of database manipulation functions for pyspinalfmri

import os
import pandas as pd
import numpy as np
import scipy.io

#----------------get_datanames_by_person------------------------------------------
# function to extract nifti format data file names, according to the participant id
# this function allows for multiple data sets from each person
def get_datanames_by_person(DBname, dbnumlist, prefix, mode = 'dict'):
    # output mode can be 'dict' for outputs in dictionary form,
    # or the output can be as a list
    # BASEdir = os.path.dirname(DBname)
    xls = pd.ExcelFile(DBname, engine = 'openpyxl')
    df1 = pd.read_excel(xls, 'datarecord')
    filename_list = []
    patientid_list = []
    studygroup_list = []

    # step through all of the values in dbnumlist before sorting/grouping
    for nn, dbnum in enumerate(dbnumlist):
        dbhome = df1.loc[dbnum, 'datadir']
        fname = df1.loc[dbnum, 'niftiname']
        patientid = df1.loc[dbnum, 'patientid']
        studygroup = df1.loc[dbnum, 'studygroup']

        niiname = os.path.join(dbhome, fname)
        fullpath, filename = os.path.split(niiname)
        prefix_niiname = os.path.join(fullpath, prefix + filename)

        filename_list.append(prefix_niiname)
        patientid_list.append(patientid)
        studygroup_list.append(studygroup)

    # get the unique patient id's
    unique_pid, unique_index, original_index = unique_in_list(patientid_list)
    NP = np.size(unique_pid)
    # get all the information that belongs with each patient (i.e. person)
    if mode == 'dict':
        dataname_list = {}
        dbnum_list = {}
    else:
        dataname_list = []
        dbnum_list = []
    for num, pid in enumerate(unique_pid):
        # need to list the filenames for every entry where original_index equals num
        person_filename_list = [filename_list[x] for x,value in enumerate(original_index) if value == num]
        person_studygroup_list = [studygroup_list[x] for x,value in enumerate(original_index) if value == num]
        person_dbnum_list = [studygroup_list[x] for x,value in enumerate(original_index) if value == num]
        dbnums  = [dbnumlist[x] for x,value in enumerate(original_index) if value == num]
        if mode == 'dict':
            dataname_list[pid] = person_filename_list
            dbnum_list[pid] = dbnums
        else:
            dataname_list.append(person_filename_list)
            dbnum_list.append(dbnums)

    return dataname_list, dbnum_list, NP



#----------------get_dbnumlists_by_keyword------------------------------------------
# function to find database numbers based on database entry keyword/value pairs
def get_dbnumlists_by_keyword(DBname, keywordlist):
    # keyword list is a dicitionary matching database entry values
    # the resulting list will match each of the keyword/value pairs in the database
    # BASEdir = os.path.dirname(DBname)
    xls = pd.ExcelFile(DBname, engine = 'openpyxl')
    datarecord = pd.read_excel(xls, 'datarecord')
    del datarecord['Unnamed: 0']   # get rid of the unwanted header column

    # fieldnames = datarecord.keys()   # get the list of keys in the database
    searchnames = keywordlist.keys()   # get the list of keys we are looking for

    NP,ndbfields = np.shape(datarecord)
    nsearch = len(searchnames)
    searchresult = np.zeros((NP,nsearch),dtype='bool')

    for bb in range(NP):
        for aa, searchfield in enumerate(searchnames):
            searchresult[bb,aa] = (datarecord.loc[bb,searchfield] == keywordlist[searchfield])

    dbnumlist = [num for num, value in enumerate(np.all(searchresult,axis =1)) if value == True]

    return dbnumlist



#----------------unique_in_list------------------------------------------
# function to return the unique values from a list, as well as the indices where
# the unique values were found, and the indices to reconstruct the original list
# from the unique data
def unique_in_list(input_list):
    # first find the unique entries
    unique_list = []
    unique_index = []
    for num, val in enumerate(input_list):
        if val not in unique_list:
            unique_list.append(val)
            unique_index.append(num)

    # now get the reverse mapping - which entries match each unique val?
    original_index = []
    for num, val in enumerate(input_list):
        a = unique_list.index(val)
        original_index.append(a)

    return unique_list, unique_index, original_index


def convert_matlab_database(matlabfilename, database_inputfields, database_outputfields, DBdir):
    # function to read a matlab ".mat" file and convert the database to an excel file that cna be read in python
    # "inputfields" must list the database fields to read, and convert to excel
    # "outputfields" must list the corresponding fields, in matching order to inputfields
    # "DBdir" is the full path name of where the database will be saved

    # database_inputfields = ['dbdir', 'patientid','studygroup','pname','seriesnumber','niftiname','TR','normdataname','normtemplatename','paradigms',
    #             'painrating','unpleasantness','temperature','age','gender']
    #
    # paradigm_inputfields = ['paradigms', 'dt']
    #
    # outputfields = ['datadir', 'patientid', 'studygroup', 'pname', 'seriesnumber', 'niftiname', 'TR', 'normdataname',
    #                       'normtemplatename', 'paradigms', 'painrating','unpleasantness','temperature','age','gender']

    outputfields = database_outputfields

    p,f = os.path.split(matlabfilename)
    f2,e = os.path.splitext(f)
    outputname = os.path.join(DBdir, f2 + '.xlsx')

    databaseinfo = scipy.io.loadmat(matlabfilename)
    input_datarecord = databaseinfo['datarecord']
    nentries = len(input_datarecord[database_inputfields[0]][0])

    # required_db_sheets = ['datarecord', 'paradigm']
    # required_db_fields = ['datadir', 'patientid', 'studygroup', 'pname', 'seriesnumber', 'niftiname', 'TR', 'normdataname',
    #                       'normtemplatename', 'paradigms']

    # datarecord sheet
    output_datarecord = []
    for ii in range(nentries):
        for nn in range(len(outputfields)):
            if outputfields[nn] == 'paradigms':
                value = 'paradigm1'
            else:
                try:
                    value = (input_datarecord[database_inputfields[nn]][0][ii].flatten())[0]
                except:
                    value = (input_datarecord[database_inputfields[nn]][0][ii].flatten())
            if nn == 1:
                entry = {outputfields[nn]:value}
            else:
                entry[outputfields[nn]] = value
        output_datarecord.append(entry)

    # write it to the database by appending a sheet to the excel file
    dataf = pd.DataFrame(output_datarecord)
    with pd.ExcelWriter(outputname, mode='w') as writer:
        dataf.to_excel(writer, sheet_name='datarecord')

    # df1 = pd.DataFrame(columns=outputfields, data=[])
    # for aa, colname in enumerate(outputfields):
    #     df1.loc[0, colname] = db_entries[aa]

    # paradigm1 sheet
    pdata = []
    paradigm = (input_datarecord['paradigms'][0][ii].flatten())
    dt = (input_datarecord['dt'][0][0].flatten())[0]
    np = len(paradigm)
    for ii in range(np):
        entry = {'dt':dt, 'paradigms':paradigm[ii]}
        pdata.append(entry)

    # write it to the database by appending a sheet to the excel file
    datap = pd.DataFrame(pdata)
    with pd.ExcelWriter(outputname, mode='a') as writer:
        datap.to_excel(writer, sheet_name='paradigm1')
