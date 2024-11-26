
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pydatabase
import nibabel as nib

# you need to set the values on lines 15, 16, 18, and only one of 20 or 21 is set to True
#  if you set the value on line 23 equal to True, then you need to set line 25
#  or
#  if you set the value on line 29 equal to True, then you need to set line 31


DBname = r'Y:\Copy_of_PS2023\PS2023_database_corrected.xlsx'   # override this
prefix = 'ptc'

windownum = 20

show_first_run_per_person = True
show_all_runs_per_person = False

use_list_name = False
if use_list_name:
	listname = r'Y:\Copy_of_PS2023\HCfast_list.npy'
	listdata = np.load(listname, allow_pickle=True).flat[0]
	DBnum = listdata['dbnumlist']

use_regiondata_name = True
if use_regiondata_name:
	regiondatafile = r'Y:\PS2023_analysis\FMfast_paindef_regiondata.npy'
	regiondata = np.load(regiondatafile, allow_pickle=True).flat[0]
	DBnum = regiondata['DBnum']
	# region_properties = regiondata['region_properties']

datanamelist, dbnumlist, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix)
namelist = [name for name in datanamelist.keys()]


plt.close(windownum)
fig = plt.figure(windownum)
ax = fig.add_axes([0, 0, 1, 1])

for name in namelist:
	namelist = datanamelist[name]
	dblist = dbnumlist[name]
	if show_first_run_per_person:
		fname = namelist[0]
		db = dblist[0]
		input_img = nib.load(fname)
		input_data = input_img.get_fdata()
		ax.clear()
		plt.imshow(input_data[13,:,:,5], cmap = 'gray')

	for run in range(1,len(namelist)):
		if show_all_runs_per_person:
			fname = namelist[run]
			db = dblist[run]
			input_img = nib.load(fname)
			input_data = input_img.get_fdata()
			ax.clear()
			plt.imshow(input_data[13,:,:,5], cmap = 'gray')

	textlabel = 'DBnum = {}'.format(db)
	ax.annotate(textlabel, xy=(2, 1), xytext=(0, 0), fontsize = 18, color = [0,0,0])
	plt.pause(1)
