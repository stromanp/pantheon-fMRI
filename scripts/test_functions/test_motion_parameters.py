

import numpy as np
import nibabel as nib

DBname = r'D:\Howie_FM2_Brain_Data\Howie_FMS2_brain_fMRI_database_JAN2020.xlsx'
dbnumlistname = [r'D:\Howie_FM2_Brain_Data\FMstim_brain_list19.npy', r'D:\Howie_FM2_Brain_Data\HCstim_brain_list.npy']
listnum = 1

dbnumlistdata = np.load(dbnumlistname[listnum], allow_pickle=True).flat[0]
DBnumlist = dbnumlistdata['dbnumlist']

xls = pd.ExcelFile(DBname, engine='openpyxl')
df1 = pd.read_excel(xls, 'datarecord')

tsize = 135
ns = len(DBnumlist)
motion_data = np.zeros((ns,3,tsize))

for nn, dbnum in enumerate(DBnumlist):
	print('motion check:  checking motion for databasenum ', dbnum)
	dbhome = df1.loc[dbnum, 'datadir']
	fname = df1.loc[dbnum, 'niftiname']
	seriesnumber = int(df1.loc[dbnum, 'seriesnumber'])
	niiname = os.path.join(dbhome, fname)
	fullpath, filename = os.path.split(niiname)
	# prefix_niiname = os.path.join(fullpath, prefix + filename)
	pname, fname = os.path.split(niiname)
	# fnameroot, ext = os.path.splitext(fname)
	# fnameroot2 = fnameroot.replace(prefix, '')

	nametag = '_s{}'.format(seriesnumber)
	motiondata_xlname = os.path.join(pname, 'motiondata' + nametag + '.xlsx')

	xls2 = pd.ExcelFile(motiondata_xlname, engine='openpyxl')
	df2 = pd.read_excel(xls2, 'motion_data')

	# check motion data format
	if ('dx' in df2.keys()) and ('dy' in df2.keys()) and ('dz' in df2.keys()):
		templatetype = 'brain'
		namelist = ['dx', 'dy', 'dz']

		for num, name in enumerate(namelist):
			dp = df2.loc[:, name]
			if num == 0:
				dpvals = [dp]
			else:
				dpvals.append(dp)
		dpvals = np.array(dpvals)
		motion_data[nn,:,:] = dpvals

	else:
		print('wrong motion data format ....')


	motion_avg = np.mean(motion_data, axis=0)
	motion_std = np.std(motion_data,axis=0)

	windownum = 111
	plt.close(windownum)
	fig = plt.figure(windownum)
	plt.errorbar(range(tsize), motion_avg[0,:], yerr = motion_std[0,:], marker = 'o', color = 'r')
	plt.errorbar(range(tsize), motion_avg[1,:], yerr = motion_std[1,:], marker = 'o', color = 'g')
	plt.errorbar(range(tsize), motion_avg[2,:], yerr = motion_std[2,:], marker = 'o', color = 'b')

