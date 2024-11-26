
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy.stats as stats
import pandas as pd
import os
import openpyxl

nregions = 4
nintrinsics = 2
ntotal = nregions + nintrinsics

Moutput = np.zeros((ntotal,ntotal))
Moutput_upper = np.random.randn(nregions,nregions)
# throw in some zeros
x = list(range(nregions))
Moutput_upper[x,x] = 0
Moutput_upper[2,0] = 0
Moutput_upper[1,3] = 0

Moutput_lower = np.eye(nintrinsics)

Moutput[:nregions,:nregions] = Moutput_upper
Moutput[-nintrinsics:,-nintrinsics:] = Moutput_lower
# mixing
Moutput[0,nregions] = 1
Moutput[2,nregions+1] = 1

# eigenvalues, eigenvectors
e,v = np.linalg.eig(Moutput)


namelist = ['region1','region2','region3','region4']
#
# write out Moutput
columns = [name[:3] + ' in' for name in namelist]
columns += ['int1 in', 'int2 in']
rows = [name[:3] for name in namelist]
rows += ['int1', 'int2']

df = pd.DataFrame(Moutput, columns=columns, index=rows)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

pd.options.display.float_format = '{:.2f}'.format
print(df)

savename = r'D:\threat_safety_python\SEMresults\Moutput_saved.xlsx'
df.to_excel(savename)

#
# write out eigvenvectors and eigenvalues
columns = [('v{}'.format(a)) for a in range(nregions+nintrinsics)]
rows = [name[:3] for name in namelist]
rows += ['int1', 'int2']

df = pd.DataFrame(v, columns=columns, index=rows)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

pd.options.display.float_format = '{:.2f}'.format
print(df)

savename = r'D:\threat_safety_python\SEMresults\Moutput_eigenvectors_saved.xlsx'

with pd.ExcelWriter(savename, engine = 'openpyxl') as writer:
    df.to_excel(writer, sheet_name = 'eigenvectors')


df = pd.DataFrame(e)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

pd.options.display.float_format = '{:.2f}'.format
print(df)

with pd.ExcelWriter(savename, mode='a', engine = 'openpyxl') as writer:
    df.to_excel(writer, sheet_name = 'eigenvalues')

