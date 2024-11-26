import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import os

# specify a folder
pname = r'D:\Howie_FM2_Brain_Data\SEP27_2018\Series16'

# find all dicom files in the folder
filenamelist = []
for file in os.listdir(pname):
	if file.endswith(".IMA"):
		filenamelist += [file]
        # print(os.path.join("/mydir", file))

dicomfilename = r'D:\Howie_FM2_Brain_Data\SEP27_2018\Series16\PWS2018_004_HW2018_001.MR.DR_STROMAN_BRAIN_FMRI.0016.0116.2018.09.27.13.23.23.593750.20680113.IMA'

acquisitiontime = np.zeros(len(filenamelist))
for nn, fname in enumerate(filenamelist):
	dicomfilename = os.path.join(pname, fname)
	filedata = pydicom.dcmread(dicomfilename)
	element = filedata['0x00080032']
	acquisitiontime[nn] = float(element.value)

