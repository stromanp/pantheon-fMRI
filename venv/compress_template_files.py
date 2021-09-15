# program to load existing template files and save them in a more compressed form

import numpy as np
import os
import nibabel as nib

filepath_source = r'C:\stroman\MNI_PAM50_templates_2020\checking_results_2021'
filepath_results = r'C:\stroman\MNI_PAM50_templates_2020\compressed_versions'

searchlist = []
typelist = []
files = os.listdir(filepath_source)
for ff in files:
    f1,e = os.path.splitext(ff)
    if e == '.nii':
        if f1.find('region_map') > 0:
            imagetype = 'map'
        else:
            imagetype = 'anatomical'
        searchlist += [ff]
        typelist += [imagetype]


for nn, fname in enumerate(searchlist):
    imagetype = typelist[nn]

    fname1,ext = os.path.splitext(fname)

    filename_in = os.path.join(filepath_source,fname)
    filename_out = os.path.join(filepath_results,fname1+'.nii.gz')

    img = nib.load(filename_in)
    img_affine = img.affine
    imgdata = img.get_data()

    if imagetype == 'anatomical':
        imgdata = (255.0*imgdata/np.max(imgdata)).astype('int')
    else:
        imgdata = imgdata.astype('int')

    resulting_img = nib.Nifti1Image(imgdata, img_affine)
    nib.save(resulting_img, filename_out)




filepath_source = r'C:\Users\Stroman\PycharmProjects\pyspinalfmri2\venv\templates\images_hidden_to_save_space'
filepath_results = r'C:\Users\Stroman\PycharmProjects\pyspinalfmri2\venv\templates\images_hidden_to_save_space'
fname = r'save_space_image.nii'
imagetype = 'anatomical'
fname1, ext = os.path.splitext(fname)

filename_in = os.path.join(filepath_source, fname)
filename_out = os.path.join(filepath_results, fname1 + '.nii.gz')

img = nib.load(filename_in)
img_affine = img.affine
imgdata = img.get_data()

if imagetype == 'anatomical':
    imgdata = (255.0 * imgdata / np.max(imgdata)).astype('int')
else:
    imgdata = imgdata.astype('int')

resulting_img = nib.Nifti1Image(imgdata, img_affine)
nib.save(resulting_img, filename_out)