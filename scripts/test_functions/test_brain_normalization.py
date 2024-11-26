import numpy as np
import pandas as pd
import os
import nibabel as nib
import time
import image_operations_3D as i3d
import pybrainregistration


settingsfile = r'C:\Users\Stroman\PycharmProjects\pantheon\venv\base_settings_file.npy'

# first get the necessary input data
settings = np.load(settingsfile, allow_pickle=True).flat[0]
NCdatabasename = settings['DBname']
NCdatabasenum = settings['DBnum']
# BASEdir = os.path.dirname(NCdatabasename)
xls = pd.ExcelFile(NCdatabasename, engine='openpyxl')
df1 = pd.read_excel(xls, 'datarecord')

normdatasavename = settings['NCsavename']  # default prefix value
fitparameters = settings['NCBparameters']  # [(10000, 1000, 100),(3.0, 1.0, 0.0),(4, 2, 1)]  # default prefix value
iters = fitparameters[0]
sigmas = fitparameters[1]
factors = fitparameters[2]

braintemplatename = settings['braintemplate']

# load the brain template
workingdir = r'C:\Users\Stroman\PycharmProjects\pantheon\venv'

brain_template_folder = os.path.join(workingdir, 'braintemplates')
template_filename = os.path.join(brain_template_folder, braintemplatename)
# ref_data, ref_affine = i3d.load_and_scale_nifti(template_filename)   # also scales to 1 mm cubic
input_ref = nib.load(template_filename)
ref_affine = input_ref.affine
ref_hdr = input_ref.header
ref_data = input_ref.get_fdata()
ref_data = ref_data / np.max(ref_data)

# display original image for first dbnum entry-------------------
dbnum = NCdatabasenum[0]
dbhome = df1.loc[dbnum, 'datadir']
fname = df1.loc[dbnum, 'niftiname']
seriesnumber = df1.loc[dbnum, 'seriesnumber']
niiname = os.path.join(dbhome, fname)

# input_data, new_affine = i3d.load_and_scale_nifti(niiname)  # this also scales to 1 mm cubic voxels
input_img = nib.load(niiname)
new_affine = input_img.affine
input_hdr = input_img.header
input_data = input_img.get_fdata()
input_data = input_data / np.max(input_data)

print('shape of input_data is ', np.shape(input_data))
print('niiname = ', niiname)
if np.ndim(input_data) == 4:
	xs, ys, zs, ts = np.shape(input_data)
	zmid = np.round(zs / 2).astype(int)
	img = input_data[:, :, zmid, 0]
	img = (255. * img / np.max(img)).astype(np.uint8)
	# image_tk = ImageTk.PhotoImage(Image.fromarray(img))
else:
	xs, ys, zs = np.shape(input_data)
	zmid = np.round(zs / 2).astype(int)
	img = input_data[:, :, zmid]
	img = (255. * img / np.max(img)).astype(np.uint8)
	# image_tk = ImageTk.PhotoImage(Image.fromarray(img))

# controller.img1d = image_tk  # keep a copy so it persists
# window1.configure(width=image_tk.width(), height=image_tk.height())
# windowdisplay1 = window1.create_image(0, 0, image=image_tk, anchor=tk.NW)

# inprogressfile = os.path.join(basedir, 'underconstruction.gif')
# image_tk = tk.PhotoImage('photo', file=inprogressfile)
# image_tk = image_tk.subsample(2)
# controller.img2d = image_tk  # keep a copy so it persists
# window2.configure(width=image_tk.width(), height=image_tk.height())
# windowdisplay2 = window2.create_image(0, 0, image=image_tk, anchor=tk.NW)

# inprogressfile = os.path.join(basedir, 'underconstruction.gif')
# image_tk = tk.PhotoImage('photo', file=inprogressfile)
# image_tk = image_tk.subsample(2)
# controller.img3d = image_tk  # keep a copy so it persists
# window3.configure(width=image_tk.width(), height=image_tk.height())
# windowdisplay3 = window3.create_image(0, 0, image=image_tk, anchor=tk.NW)

# time.sleep(0.1)
# -----------end of display--------------------------------

print('Normalization: databasename ', NCdatabasename)
print('Normalization: started organizing at ', time.ctime(time.time()))

# assume that all the data sets being normalized in a group are from the same region
# and have the same template and anatomical region - no need to load these for each dbnum

for nn, dbnum in enumerate(NCdatabasenum):
	print('NCrunclick: databasenum ', dbnum)
	dbhome = df1.loc[dbnum, 'datadir']
	fname = df1.loc[dbnum, 'niftiname']
	seriesnumber = df1.loc[dbnum, 'seriesnumber']
	normtemplatename = df1.loc[dbnum, 'normtemplatename']
	niiname = os.path.join(dbhome, fname)
	fullpath, filename = os.path.split(niiname)
	# prefix_niiname = os.path.join(fullpath,prefix+filename)
	tag = '_s' + str(seriesnumber)
	normdataname_full = os.path.join(fullpath, normdatasavename + tag + '.npy')

	# load the nifti data
	# input_datar, affiner = i3d.load_and_scale_nifti(niiname)
	input_img = nib.load(niiname)
	affiner = input_img.affine
	input_hdr = input_img.header
	input_datar = input_img.get_fdata()
	input_datar = input_datar / np.max(input_datar)

	if np.ndim(input_datar) > 3:
		x, y, z, t = np.shape(input_datar)
		if t > 3:
			t0 = 3
		else:
			t0 = 0
		input_image = input_datar[:, :, :, t0]
	else:
		x, y, z = np.shape(input_datar)
		input_image = input_datar
	input_datar = []  # clear it from memory


	# load the intermediate reference scan if it was specified, and it exists
	try:
		intermediate_norm_dbref = df1.loc[dbnum, 'norm_bridge_ref']
		if intermediate_norm_dbref >= 0:
			print('...normalizing to a reference first, then to the brain template.')
			dbhome = df1.loc[intermediate_norm_dbref, 'datadir']
			fname = df1.loc[intermediate_norm_dbref, 'niftiname']
			seriesnumber = df1.loc[intermediate_norm_dbref, 'seriesnumber']
			norm_ref_name = os.path.join(dbhome, fname)

			normref_data = nib.load(norm_ref_name)
			normref_affine = normref_data.affine
			normref_hdr = normref_data.header
			normref_pixdim = normref_hdr['pixdim'][1:4]
			normref_dim = normref_hdr['dim'][1:4]
			normref_img = normref_data.get_fdata()
			normref_img = normref_img / np.max(normref_img)

			# resize to brain template resolution
			brainres = [2., 2., 2.]
			newsize = np.round(normref_dim*normref_pixdim/brainres)
			normref_img = i3d.resize_3D_nearest(normref_img, newsize)
			brainres_affine_scale = np.array([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1]])
			normref_affine = normref_affine @ brainres_affine_scale
		else:
			intermediate_norm_dbref = -1
	except:
		intermediate_norm_dbref = -1

	# run the normalization
	print('starting normalization calculation ....')
	# set the cursor to reflect being busy ...
	# controller.master.config(cursor="wait")
	# controller.master.update()

	if intermediate_norm_dbref >= 0:
		norm_brain_img_1_2, norm_brain_affine_1_2, norm_brain_img_2_norm, norm_brain_affine_2_norm = pybrainregistration.dipy_compute_twostage_brain_normalization(input_image, affiner,
												normref_img, normref_affine, ref_data, ref_affine, iters, sigmas, factors, nbins = 32)

		check_resampled = norm_brain_affine_1_2.transform(input_image)
		check_resampled = norm_brain_affine_2_norm.transform(check_resampled)

		a12 = norm_brain_affine_1_2.get_affine()
		a2n = norm_brain_affine_2_norm.get_affine()

		a1n = a12 @ a2n

		new_affine = copy.deepcopy(norm_brain_affine_1_2)
		new_affine.set_affine(a1n)

		affine_map = AffineMap(np.eye(4), new_shape, new_affine, old_shape,
							   img_in.affine)

		norm_brain_affine = dipy.align._public.AffineMap(a1n, np.shape(ref_data), ref_affine, np.shape(input_image), affiner)
		norm_brain_img = norm_brain_affine.transform(input_image)

	else:
		norm_brain_img, norm_brain_affine = pybrainregistration.dipy_compute_brain_normalization(input_image, affiner, ref_data, ref_affine, iters, sigmas, factors, nbins=32)
	# controller.master.config(cursor="")
	# controller.master.update()
	print('finished normalization calculation ....')

