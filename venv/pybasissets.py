# -*- coding: utf-8 -*-
"""
Created on Fri May  8 09:08:15 2020

@author: stroman
"""
import nibabel as nib
import numpy as np
import scipy.ndimage as nd
import scipy.stats as sps
import os
import pandas as pd
from scipy.interpolate import interp1d
import load_templates
import py_mirt3D as mirt
import image_operations_3D as i3d
import openpyxl
from sklearn.decomposition import PCA

#---------createparadigmfile----------------------------------------
#-------------------------------------------------------------------
# def createparadigmfile(DBname):
#     # this is an example of how a paradigm file might be created
#     # the paradigms are written to the database entry file (excel)
#     # the paradigm name is the sheet name
#     paradigmname = 'paradigm1'
#
#     # BASEdir = os.path.dirname(DBname)
#     # xls = pd.ExcelFile(DBname, engine = 'openpyxl')
#     # df1 = pd.read_excel(xls, 'datarecord')
#
#     dt = 0.5    # seconds per point
#     rest1 = 57.0   # initial rest period in seconds
#     isi = 1.5  # interstimulus interval
#     stim = 1.5  # contact stim duration
#     rest2 = 68.0  # final rest period in seconds
#
#     nrest1 = np.zeros(np.round(rest1/dt).astype('int'))
#     nstim = np.ones(np.round(stim/dt).astype('int'))
#     nisi = np.zeros(np.round(isi/dt).astype('int'))
#     nrest2 = np.zeros(np.round(rest2/dt).astype('int'))
#
#     nnostim = np.zeros(np.round(30.0/dt).astype('int'))
#     nafterstim = np.ones(np.round(15.0/dt).astype('int'))
#     nremaining = np.zeros(np.round(53.0/dt).astype('int'))
#
#     paradigm = np.concatenate((nrest1,   # intial rest period
#                 nstim, nisi,   # one contact followed by isi
#                 nstim, nisi,   # next contact ...
#                 nstim, nisi,   # next contact ...
#                 nstim, nisi,   # next contact ...
#                 nstim, nisi,   # next contact ...
#                 nstim, nisi,   # next contact ...
#                 nstim, nisi,   # next contact ...
#                 nstim, nisi,   # next contact ...
#                 nstim, nisi,   # next contact ...
#                 nstim, nisi,   # next contact ...
#                 nrest2)  )  # final rest period
#
#     aftersensations = np.concatenate( (nrest1, nnostim, nafterstim, nremaining))   # another thing for testing
#
#     # it is possible to have more than one paradigm - give them different names
#     paradigmdef = {'dt':dt,'paradigm':paradigm, 'aftersensations':aftersensations}
#
#     # create a dataframe
#     paradigmdata = pd.DataFrame(data = paradigmdef)
#     # write it to the database by appending a sheet to the excel file
#     # want to overwrite sheet of the same name if it already exists, or create a new one if not
#     # writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
#
#     # first check and see if the sheet already exists - and delete it
#     xls = pd.ExcelFile(DBname, engine = 'openpyxl')
#     existing_sheet_names = xls.sheet_names
#     if paradigmname in existing_sheet_names:
#         # delete sheet - need to use openpyxl
#         workbook = openpyxl.load_workbook(DBname)
#         std = workbook.get_sheet_by_name(paradigmname)
#         workbook.remove_sheet(std)
#         workbook.save(DBname)
#
#     # now write the new sheet
#     with pd.ExcelWriter(DBname, mode='a') as writer:
#         paradigmdata.to_excel(writer, sheet_name=paradigmname)



#---------hemodynamic response function definition------------------
#-------------------------------------------------------------------
def HRF(TR=2, length=32, peak_delay=6, under_delay=16,
                  peak_disp=1, under_disp=1, p_u_ratio=6,  normalize=True):
    # the hemodynamic response function is the sum of two gamma functions, as defined in SPM

    t = np.linspace(0, length, np.ceil(length/TR).astype('int'))
    if len([v for v in [peak_delay, peak_disp, under_delay, under_disp]
            if v <= 0]):
        raise ValueError("delays and dispersions must be > 0")
    # gamma.pdf only defined for t > 0
    hrf = np.zeros(t.shape, dtype=float)
    pos_t = t[t > 0]
    peak = sps.gamma.pdf(pos_t,
                         peak_delay / peak_disp,
                         loc=0,
                         scale=peak_disp)
    undershoot = sps.gamma.pdf(pos_t,
                               under_delay / under_disp,
                               loc=0,
                               scale=under_disp)
    hrf[t > 0] = peak - undershoot / p_u_ratio
    if not normalize:
        return hrf
    return hrf / np.max(hrf)


#---------coreg_to_motionparams-------------------------------------
#-------------------------------------------------------------------
def coreg_to_motionparams(niiname, normdataname, normtemplatename, nametag):
    pname, fname = os.path.split(niiname)
    fnameroot, ext = os.path.splitext(fname)

    # define names for coreg data saved in pycoregistration
    coregdata_name = os.path.join(pname, 'coregdata'+nametag+'.npy')
    print('coregdata_name = ', coregdata_name)
    output_motiondata_name = os.path.join(pname, 'motiondata'+nametag+'.npy')
    coreg_data = np.load(coregdata_name, allow_pickle=True)

    # data saved from normalization step
    normdata = np.load(normdataname, allow_pickle=True).flat[0]
    T = normdata['T']
    Tfine = normdata['Tfine']
    xt,yt,zt = np.shape(normdata['reverse_map_image'])

    # load the image data
    input_img = nib.load(niiname)
    input_data = input_img.get_fdata()
    affine = input_img.affine
    input_hdr = input_img.header
    FOV = input_hdr['pixdim'][1:4]*input_hdr['dim'][1:4]
    xs,ys,zs,ts = np.shape(input_data)

    print('coreg_to_motionparams:  check on process: ')
    print('  niiname =  ', niiname)
    print('  FOV =  ', FOV)
    print('  xs,ys,zs,ts = ',xs,' ',ys,' ', zs,' ',ts)

    # get the number of volumes from the coregdata
    ts = np.size(coreg_data)
    print('coreg_to_motionparams: number of volumes is ',ts)

    # T was called PTform in matlab
    # check that this is consistent with coreg program
    Xs = T['Xs'] + Tfine['dXs'];
    Ys = T['Ys'] + Tfine['dYs'];
    Zs = T['Zs'] + Tfine['dZs'];

    # xs,ys,zs = np.shape(Xs)
    # print('  xs,ys,zs,ts = ',xs,' ',ys,' ', zs,' ',ts)

    dx = np.zeros(ts)
    dy = np.zeros(ts)
    dz = np.zeros(ts)

    dx2 = np.zeros((3,ts))
    dy2 = np.zeros((3,ts))
    dz2 = np.zeros((3,ts))

    # use this data to estimate motion parameters
    foundtemplate = False
    if normtemplatename.lower() == 'thoracic':
        refpoints = [[13, 72, 60],
                     [13, 72, 107],
                     [13, 57, 143]]
        motiontestpoint = [5, 79, 100]
        foundtemplate = True

    if 'to' in normtemplatename.lower():
        refpoints = [[15, 15, (np.round(zt*0.25)).astype(int)],
                     [15, 15, (np.round(zt*0.50)).astype(int)],
                     [15, 15, (np.round(zt*0.75)).astype(int)]]
        motiontestpoint = [9, 18, (np.round(zt*0.50)).astype(int)]
        foundtemplate = True

    if not foundtemplate:   # only setup for ccbs, thoracic cord, and "to" ranges, for now
        refpoints = [[13, 72, 60],
                     [13, 72, 107],
                     [13, 72, 145]]
        motiontestpoint = [5, 79, 100]

    motion_parameters = {}

    print('coreg_to_motionparams: compiling motion data ...')
    for tt in range(ts):
        res = coreg_data[tt]

        F = mirt.py_mirt3D_F(res['okno'])  # Precompute the matrix B-spline basis functions
        X, Y, Z = mirt.py_mirt3D_nodes2grid(res['X'], F, res['okno'])   # get the position of all image voxels

        X = X[:xs,:ys,:zs]
        Y = Y[:xs,:ys,:zs]
        Z = Z[:xs,:ys,:zs]

        temp = i3d.resize_3D(X,FOV.astype('int'))   # interpolate to 1 mm voxels
        XX = i3d.warp_image(temp, Xs, Ys, Zs) # apply normalization to coregistration data (to get regions of interest)
        temp = i3d.resize_3D(Y,FOV.astype('int'))   # interpolate to 1 mm voxels
        YY = i3d.warp_image(temp, Xs, Ys, Zs)
        temp = i3d.resize_3D(Z,FOV.astype('int'))   # interpolate to 1 mm voxels
        ZZ = i3d.warp_image(temp, Xs, Ys, Zs)

        XX = np.where(np.isfinite(XX), XX, 0)
        YY = np.where(np.isfinite(YY), YY, 0)
        ZZ = np.where(np.isfinite(ZZ), ZZ, 0)

        # check for motion in normalized data space
        mtp = motiontestpoint
        dx[tt] = XX[mtp[0], mtp[1], mtp[2]]- mtp[0]
        dy[tt] = YY[mtp[0], mtp[1], mtp[2]] - mtp[1]
        dz[tt] = ZZ[mtp[0], mtp[1], mtp[2]] - mtp[2]

        for ss in range(3):
            dx2[ss, tt] = XX[refpoints[ss][0], refpoints[ss][1], refpoints[ss][2]]
            dy2[ss, tt] = YY[refpoints[ss][0], refpoints[ss][1], refpoints[ss][2]]
            dz2[ss, tt] = ZZ[refpoints[ss][0], refpoints[ss][1], refpoints[ss][2]]

    print('coreg_to_motionparams: saving results ...')
    # adjust to the first volume
    for ss in range(3):
        dx2[ss, :] = dx2[ss,:] - dx2[ss, 0]
        dy2[ss, :] = dy2[ss,:] - dy2[ss, 0]
        dz2[ss, :] = dz2[ss,:] - dz2[ss, 0]

    motion_parameters = {'dx1':dx2[0,:], 'dy1':dy2[0,:], 'dz1':dz2[0,:],
                         'dx2':dx2[1,:], 'dy2':dy2[1,:], 'dz2':dz2[1,:],
                         'dx3':dx2[2,:], 'dy3':dy2[2,:], 'dz3':dz2[2,:]}

    # is it better to write the motion parameters to an excel file?
    # other methods could be used to create the motion parameters file, and have it written to excel for convenience
    np.save(output_motiondata_name, motion_parameters)

    # put the data in a format for writing to excel
    # this file goes with the nifti format data, because there is one for each fMRI run
    motiondata = pd.DataFrame(data = motion_parameters)
    output_motiondata_xlname = os.path.join(pname, 'motiondata'+nametag+'.xlsx')
    motiondata.to_excel(output_motiondata_xlname, sheet_name='motion_data')   # write it out to excel

    return output_motiondata_xlname


#---------guided_coreg_to_motionparams-------------------------------------
#-------------------------------------------------------------------
def guided_coreg_to_motionparams(normdataname, normtemplatename, nametag):
    pname, fname = os.path.split(normdataname)
    fnameroot, ext = os.path.splitext(fname)

    # define names for coreg data saved in pycoregistration
    coregdata_name = os.path.join(pname, 'coregdata'+nametag+'.npy')
    print('coregdata_name = ', coregdata_name)
    output_motiondata_name = os.path.join(pname, 'motiondata'+nametag+'.npy')
    coreg_data = np.load(coregdata_name, allow_pickle=True)

    # get the number of volumes from the coregdata
    # coreg_data is an array of {'T': T2, 'map_step': map_step}

    ts = np.size(coreg_data)
    nsections = len(coreg_data[0]['map_step'])
    print('guided_coreg_to_motionparams: number of volumes is {} for {} sections'.format(ts,nsections))

    # use this data to estimate motion parameters
    if normtemplatename.lower() == 'thoracic':
        refsections = [0,np.floor(nsections/2).astype(int),nsections]
    else:   # only setup for ccbs and thoracic cord for now
        last_section = np.min([nsections, 7])
        refsections = [2,0, last_section]

    dx2 = np.zeros((3,ts))
    dy2 = np.zeros((3,ts))
    dz2 = np.zeros((3,ts))

    motion_parameters = {}

    print('coreg_to_motionparams: compiling motion data ...')
    for tt in range(ts):
        coords = coreg_data[tt]['map_step']['coords']

        for ss in range(3):
            dx2[ss, tt] = coords[refsections[ss][0]]
            dy2[ss, tt] = coords[refsections[ss][1]]
            dz2[ss, tt] = coords[refsections[ss][2]]

    print('coreg_to_motionparams: saving results ...')
    # adjust to the first volume
    for ss in range(3):
        dx2[ss, :] = dx2[ss,:] - dx2[ss, 0]
        dy2[ss, :] = dy2[ss,:] - dy2[ss, 0]
        dz2[ss, :] = dz2[ss,:] - dz2[ss, 0]

    motion_parameters = {'dx1':dx2[0,:], 'dy1':dy2[0,:], 'dz1':dz2[0,:],
                         'dx2':dx2[1,:], 'dy2':dy2[1,:], 'dz2':dz2[1,:],
                         'dx3':dx2[2,:], 'dy3':dy2[2,:], 'dz3':dz2[2,:]}

    # is it better to write the motion parameters to an excel file?
    # other methods could be used to create the motion parameters file, and have it written to excel for convenience
    np.save(output_motiondata_name, motion_parameters)

    # put the data in a format for writing to excel
    # this file goes with the nifti format data, because there is one for each fMRI run
    motiondata = pd.DataFrame(data = motion_parameters)
    output_motiondata_xlname = os.path.join(pname, 'motiondata'+nametag+'.xlsx')
    motiondata.to_excel(output_motiondata_xlname, sheet_name='motion_data')   # write it out to excel

    return output_motiondata_xlname


#---------get_whitematter_noise-------------------------------------
#-------------------------------------------------------------------
def get_whitematter_noise(niiname, normtemplatename, nametag):
    # the niiname needs to refer to spatially normalized data
    # load templates to find the white matter regions
    resolution = 1
    template_img, regionmap_img, template_affine, anatlabels = load_templates.load_template(normtemplatename, resolution)
    wmmap, template_img, template_affine, roi_map, gmwm_img = load_templates.load_wm_maps(normtemplatename, resolution)

    # load the data to get the white matter time-courses
    input_img = nib.load(niiname)
    input_data = input_img.get_fdata()
    affine = input_img.affine
    input_hdr = input_img.header
    FOV = input_hdr['pixdim'][1:4]*input_hdr['dim'][1:4]
    xs,ys,zs,ts = np.shape(input_data)

    # selected wm regions and extract time-courses
    pos = [0.25, 0.5, 0.75]
    wmtc = np.zeros((3,ts))
    # step through the regions
    for num,p in enumerate(pos):
        mask = np.zeros((xs,ys,zs))
        z0 = np.round(p * zs).astype('int')
        mask[:, :, range((z0-1),(z0+2))] = 1.   # take all the wm voxels in a selected slice
        ax, ay, az = np.where((mask > 0) & (wmmap > 50))   # wmmap is scaled 0-255 to span the probability range 0-1
        nvox = np.size(ax)
        tcdata = np.zeros((nvox,ts))
        for tt in range(ts):
            temp = input_data[:,:,:,tt]
            tcdata[:,tt] = temp[ax,ay,az].flatten()
        singletc = np.mean(tcdata, axis = 0)
        avgtc = np.mean(singletc)
        # convert to percent signal change from average
        singletc = 100.0*(singletc-avgtc)/avgtc
        # wmtc is the list of timecourses in the 3 selected regions of white matter
        wmtc[num,:] = singletc

    # write the results to an excel file
    # set the output name
    pname, fname = os.path.split(niiname)
    fnameroot, ext = os.path.splitext(fname)
    xlname = os.path.join(pname, 'wmnoise'+nametag+'.xlsx')

    # put the data in a format for writing to excel
    # this file goes with the nifti format data, because there is one for each fMRI run
    output_noise = {'region1':wmtc[0,:], 'region2':wmtc[1,:], 'region3':wmtc[2,:]}
    noisedata = pd.DataFrame(data = output_noise)
    output_motiondata_xlname = os.path.join(pname, 'wmdata'+nametag+'.xlsx')
    noisedata.to_excel(xlname, sheet_name='wmnoise')   # write it out to excel

    return wmtc, xlname


#---------get_brain_whitematter_noise-------------------------------------
#-------------------------------------------------------------------
def get_brain_whitematter_noise(niiname, WMmapname, nametag):
    # the niiname needs to refer to spatially normalized data
    # load templates to find the white matter regions

    # load the specified white matter map
    input_img = nib.load(WMmapname)
    wm_data = input_img.get_fdata()
    wm_affine = input_img.affine
    wm_hdr = input_img.header

    # load the data to get the white matter time-courses
    input_img = nib.load(niiname)
    input_data = input_img.get_fdata()
    affine = input_img.affine
    input_hdr = input_img.header
    FOV = input_hdr['pixdim'][1:4]*input_hdr['dim'][1:4]
    xs,ys,zs,ts = np.shape(input_data)

    # selected wm regions and extract time-courses
    x,y,z = np.where(wm_data == 1)   # white matter mask for brain regions
    nvox = len(x)
    tcdata = np.zeros((nvox,ts))
    # step through the time points
    for tt in range(ts):
        temp = input_data[:,:,:,tt]
        tcdata[:,tt] = temp[x,y,z].flatten()

    # make sure the first 2 volumes are blanked out
    tcdata[:,0] = tcdata[:,2]
    tcdata[:,1] = tcdata[:,2]
    # take the top 3 principal components

    pca = PCA(n_components=3)
    pca.fit(tcdata)
    wmtc = pca.components_

    # write the results to an excel file
    # set the output name
    pname, fname = os.path.split(niiname)
    fnameroot, ext = os.path.splitext(fname)
    xlname = os.path.join(pname, 'wmnoise'+nametag+'.xlsx')

    # put the data in a format for writing to excel
    # this file goes with the nifti format data, because there is one for each fMRI run
    output_noise = {'region1':wmtc[0,:], 'region2':wmtc[1,:], 'region3':wmtc[2,:]}
    noisedata = pd.DataFrame(data = output_noise)
    output_motiondata_xlname = os.path.join(pname, 'wmdata'+nametag+'.xlsx')
    noisedata.to_excel(xlname, sheet_name='wmnoise')   # write it out to excel

    return wmtc, xlname


#---------maineffects-----------------------------------------------
#-------------------------------------------------------------------
def calculate_maineffects(DBname, dbnum, TR, nvols):
    # the stimulation paradigm needs to be defined in an excel file
    # and must include values for each time point and the time period between points (dt)
    # TR is the repetition time used for sampling the data
    # nvols is the number of volumes in the fMRI data set (and the resulting length of the basis set)

    xls = pd.ExcelFile(DBname, engine = 'openpyxl')
    df1 = pd.read_excel(xls, 'datarecord')
    paradigmname = df1.loc[dbnum, 'paradigms']
    df2 = pd.read_excel(xls, paradigmname)
    del df2['Unnamed: 0']   # get rid of the unwanted header column

    # get the names of the columns in this sheet ....
    colnames = df2.keys()

    dt = df2.loc[:, 'dt'][0]
    hrf = HRF(dt)   # get ready with the hemodynamic response function
    vol_times = (np.array(range(nvols)) + 0.5)*TR   # define the times at the middle of each volume
    output = {'time':vol_times}

    print('defining basis sets...')
    print('  computing paradigm for {} volumes'.format(nvols))
    print('  paradigm definition sheet {}'.format(paradigmname))
    print('  data in paradigm definition sheet are:  {}'.format(colnames))
    for basisname in colnames:
        if basisname != 'dt':
            paradigmdef = df2.loc[:, basisname]
            # convolve with hemodynamic response function (HRF) and
            # then resample at the time each volume was acquired
            paradigm_hrf = np.convolve(paradigmdef, hrf, mode = 'same')
            # resample at the TR
            numpoints = np.size(paradigmdef)
            paradigm_times = np.array(range(numpoints))*dt
            # interpolate
            f = interp1d(paradigm_times, paradigm_hrf, kind='cubic')
            print('range of paradigm_times is {} to {}'.format(np.min(paradigm_times),np.max(paradigm_times)))
            print('range of vol_times is {} to {}'.format(np.min(vol_times),np.max(vol_times)))
            BOLD = f(vol_times)
            BOLD = BOLD - np.mean(BOLD)
            BOLDname = basisname+'_BOLD'
            output[BOLDname] = BOLD/np.max(BOLD)  # scale to max = 1, and save

    # create a dataframe
    BOLDdata = pd.DataFrame(data = output)
    # write it to the database by appending a sheet to the excel file
    sheetname = paradigmname + '_BOLD'
    existing_sheet_names = xls.sheet_names
    if sheetname in existing_sheet_names:
        # delete sheet - need to use openpyxl
        workbook = openpyxl.load_workbook(DBname)
        std = workbook.get_sheet_by_name(sheetname)
        workbook.remove_sheet(std)
        workbook.save(DBname)

    with pd.ExcelWriter(DBname, mode='a', engine='openpyxl') as writer:
        BOLDdata.to_excel(writer, sheet_name=sheetname)

    return sheetname



#---------read_maineffects-----------------------------------------------
#-------------------------------------------------------------------
def read_maineffects(DBname, dbnum):
    # the stimulation paradigm needs to be defined in an excel file
    # and must include values for each time point and the time period between points (dt)
    # TR is the repetition time used for sampling the data
    # nvols is the number of volumes in the fMRI data set (and the resulting length of the basis set)

    xls = pd.ExcelFile(DBname, engine = 'openpyxl')
    df1 = pd.read_excel(xls, 'datarecord')
    paradigmname = df1.loc[dbnum, 'paradigms']
    maineffects_sheetname = paradigmname + '_BOLD'

    df2 = pd.read_excel(xls, maineffects_sheetname)
    del df2['Unnamed: 0']   # get rid of the unwanted header column

    # get the names of the columns in this sheet ....
    colnames = df2.keys()

    time = df2.loc[:, 'time']
    paradigm_names = []

    count = 0
    for num, basisname in enumerate(colnames):
        if basisname != 'time':
            count += 1
            paradigm_names.append(basisname)
            if count == 1:
                paradigmdef = np.array(df2.loc[:, basisname])
                paradigmdef = paradigmdef[np.newaxis,:]
            else:
                nextparadigm = df2.loc[:, basisname]
                paradigmdef = np.concatenate((paradigmdef,nextparadigm[np.newaxis,:]),axis = 0)

    return paradigmdef, paradigm_names

