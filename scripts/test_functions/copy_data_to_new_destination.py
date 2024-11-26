# move data to new location, based on a database file
import numpy as np
import pandas as pd
import openpyxl
import os
import shutil

DBname = r"Z:\stromanlab\threat_safety_study\threat_safety_database.xlsx"
xls = pd.ExcelFile(DBname, engine='openpyxl')
df1 = pd.read_excel(xls, 'datarecord')
nentries = len(df1)

new_data_dir = r'D:\threat_safety_python'
sourcelist = []
destinationlist = []
for nn in range(nentries):
    pname = df1.loc[nn, 'pname']
    datadir = df1.loc[nn, 'datadir']
    pname = pname[:-1]
    datadir = datadir[:-1]

    sourcedir = os.path.join(datadir,pname)

    if datadir == r"Z:\stromanlab\Elena\Data":
        destinationdir = os.path.join(new_data_dir,'TouchPain',pname)
    else:
        destinationdir = os.path.join(new_data_dir,pname)

    print('{} copying data from {} to {}'.format(nn, sourcedir, destinationdir))
    sourcelist.append(sourcedir)
    destinationlist.append(destinationdir)

# find unique folder names so they are not copied more than once
usource, sindexlist = np.unique(sourcelist, return_index = True)
udest, dindexlist = np.unique(destinationlist, return_index = True)

check = (sindexlist == dindexlist).all()
print('\n\n')
if check:
    for nn in range(20, len(sindexlist)):
        ss = sourcelist[sindexlist[nn]]
        dd = destinationlist[sindexlist[nn]]
        print('{} copying data from {} to {}'.format(nn, ss, dd))
        shutil.copytree(ss,dd)
else:
    print('something is wrong - source and dest folders do not match up')