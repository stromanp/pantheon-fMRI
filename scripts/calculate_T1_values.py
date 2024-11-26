# sys.path.extend([r'C:\Users\Stroman\PycharmProjects\pantheon\venv'])
# sys.path.extend([r'C:\Users\Stroman\PycharmProjects\pantheon\venv\test_functions'])

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyclustering
import load_templates
import copy
import image_operations_3D as i3d
import os
import pandas as pd
import nibabel as nib
from scipy import interpolate
import py2ndlevelanalysis
import pydatabase

datatype = 'brain'
grouptype = 'FM'


region_name = 'FOrb'
region_name = 'IC'
region_name = 'Hippocampus'
region_name = 'Hypothalamus'
region_name = 'LC'
region_name = 'IC'
region_name = 'Thalamus'
region_name = 'PAG'
windownum = 123

grouptype1 = 'FM'
grouptype2 = 'HC'
stimtype1 = 'stim'

DBname = r'D:\Howie_FM2_Brain_Data\Howie_FMS2_brain_fMRI_database_JAN2020_V2.xlsx'
listname1 = r'D:\Howie_FM2_Brain_Data\FMstim_brain_list19.npy'
listname2 = r'D:\Howie_FM2_Brain_Data\HCstim_brain_list.npy'

list1 = np.load(listname1, allow_pickle=True).flat[0]
dbnumlist1 = list1['dbnumlist']
list2 = np.load(listname2, allow_pickle=True).flat[0]
dbnumlist2 = list2['dbnumlist']

prefix = ''
TR = 2.0
alpha = 84.0  # flip angle in degrees
dtor = np.pi/180.

dataname_list1, dbnum_list1, NP1 = pydatabase.get_datanames_by_person(DBname, dbnumlist1, prefix)
dataname_list2, dbnum_list2, NP2 = pydatabase.get_datanames_by_person(DBname, dbnumlist2, prefix)


dnamelist = copy.deepcopy(dataname_list1)
keylist1 = dnamelist.keys()
groupdata = []
for nn, keyval in enumerate(keylist1):
	namelist = dnamelist[keyval]
	for aa in range(len(namelist)):
		print('person {}   file:  {}'.format(nn,namelist[aa]))

		input_img = nib.load(namelist[aa])
		input_data = input_img.get_fdata()
		affine = input_img.affine

		if aa == 0:
			persondata = input_data[:,:,:,:5][:,:,:,:,np.newaxis]
		else:
			persondata = np.concatenate((persondata,input_data[:,:,:,:5][:,:,:,:,np.newaxis]),axis=4)

	S2 = persondata[:,:,:,1,:]/(persondata[:,:,:,0,:] + 1.0e-20)
	count = np.sum(S2 < 1.0, axis = 3)
	S2[S2 >= 1.0] = 0.0
	meanS2 = np.sum(S2,axis=3)/(count + 1.0e-20)

	groupdata.append({'persondata':persondata, 'S2':S2, 'meanS2':meanS2})





def fitfunc(x,T1,S0,TR,alpha):
	if x < 0:
		S = 0.
	else:
		S = 1.0
		if x > 0:
			S += -np.exp(-TR/T1) + np.cos(alpha)*np.exp(-TR/T1)
		if x > 1:
			S += -np.cos(alpha)*np.exp(-2.*TR/T1) + np.cos(alpha)*np.cos(alpha)*np.exp(-2.*TR/T1)
		if x > 2:
			S += -np.cos(alpha)*np.cos(alpha)*np.exp(-3.*TR/T1) + ((np.cos(alpha))**3)*np.exp(-3.*TR/T1)
		if x > 3:
			S += -((np.cos(alpha))**3)*np.exp(-4.*TR/T1) + ((np.cos(alpha))**4)*np.exp(-4.*TR/T1)

	S *= S0*np.sin(alpha)
	return S