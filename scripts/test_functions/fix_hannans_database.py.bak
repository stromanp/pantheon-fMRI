
import numpy as np
import pandas as pd
import openpyxl
import os
import shutil
import scipy.io
import copy

DBname = r'C:\HA2023\HA2023_database.xlsx'
new_DBname = r'C:\HA2023\HA2023_database_database_corrected.xlsx'
xls = pd.ExcelFile(DBname, engine='openpyxl')
xls_sheets = xls.sheet_names

df1 = pd.read_excel(xls, 'datarecord')
keylist = df1.keys()
for kname in keylist:
    if 'Unnamed' in kname: df1.pop(kname)  # remove blank fields from the database

num = len(df1)   # number of entries
for nn in range(num):
    group = copy.deepcopy(df1.loc[nn, 'group'])
    pattern = copy.deepcopy(df1.loc[nn, 'pattern'])

    if pattern[:3].lower() = 'neg':
        stimname = 'Negative'
    else:
        stimname = 'Neutral'

    sgname = group + stimname
    df1.loc[nn, 'studygroup'] = copy.deepcopy(sgname)

# write database data
with pd.ExcelWriter(new_DBname) as writer:
    df1.to_excel(writer, sheet_name='datarecord')
