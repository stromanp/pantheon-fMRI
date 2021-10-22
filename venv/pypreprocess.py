# -*- coding: utf-8 -*-
"""
Created on Fri May  8 09:08:15 2020

@author: stroman
"""
import nibabel as nib
import py_mirt3D as mirt
import numpy as np
import scipy.ndimage as nd
import copy
import os
import pandas as pd
import time
import pybasissets
import GLMfit
import image_operations_3D as i3d
import pynormalization

# -----coregister------------------------------------------------------------------
# this function assumes the input is given as a time-series, and the images are all coregistered
# with the 3rd volume in the series, or the 1st volume if only two volumes are provided
def coregister(filename, nametag, coregistered_prefix = 'c'):
    starttime = time.time()
    #set default main settings for MIRT coregistration
    # use 'ssd', or 'cc' which was used in the previous matlab version of this function
    main_init = {'similarity':'ssd',   # similarity measure, e.g. SSD, CC, SAD, RC, CD2, MS, MI
            'subdivide':1,       # use 1 hierarchical level
            'okno':4,           # mesh window size
            'lambda':0.5,     # transformation regularization weight, 0 for none
            'single':1}
    
    #Optimization settings
    optim_init = {'maxsteps':100,    # maximum number of iterations at each hierarchical level
             'fundif':1e-4,     # tolerance (stopping criterion)
             'gamma':1.0,         # initial optimization step size
             'anneal':0.1}        # annealing rate on the optimization step

    # original values
    # optim_init = {'maxsteps':100,    # maximum number of iterations at each hierarchical level
    #          'fundif':1e-6,     # tolerance (stopping criterion)
    #          'gamma':1.0,         # initial optimization step size
    #          'anneal':0.7}        # annealing rate on the optimization step

    
    smooth_refimage = False
    smooth_regimage = False
    
    input_img = nib.load(filename)
    input_data = input_img.get_data()
    affine = input_img.affine
    
    xs,ys,zs,ts = np.shape(input_data)
    
    # coregister to the 3rd volume in the time-series
    if ts >= 2:
        refimage = input_data[:,:,:,2]
    else:
        refimage = input_data[:,:,:,0]
            
    smoothval = (1.1, 1.1, 1.1)
    if smooth_refimage:
        refimage = nd.gaussian_filter(refimage, smoothval)
        
    refimage = refimage/np.max(refimage)
    
    result = np.zeros((xs,ys,zs,ts))
    print('Running coregistration step ...')
    for tt in range(ts):
        print('Volume {vnum} of {vtotal}'.format(vnum = tt+1, vtotal = ts))
        regimage = input_data[:,:,:,tt]
        
        if smooth_regimage:
            regimages = nd.gaussian_filter(regimage, smoothval)
        else:
            regimages = regimage
                
        regimages = regimages/np.max(regimages)
        
        optim = copy.deepcopy(optim_init)
        main = copy.deepcopy(main_init)

        print('gamma = {}'.format(optim['gamma']))
        res, new_img = mirt.py_mirt3D_register(refimage, regimages, main, optim)
        m = np.max(regimage)
        new_img2 = m*mirt.py_mirt3D_transform(regimage/m,res)
        result[:,:,:,tt] = new_img2

        if tt == 0:
            coreg_data = [res]
        else:
            coreg_data.append(res)
        
    # write result as new NIfTI format image set
    pname, fname = os.path.split(filename)
    fnameroot, ext = os.path.splitext(fname)
    # define names for outputs
    coregdata_name = os.path.join(pname, 'coregdata'+nametag+'.npy')
    np.save(coregdata_name, coreg_data)

    niiname = os.path.join(pname, coregistered_prefix+fname)
    resulting_img = nib.Nifti1Image(result, affine)
    nib.save(resulting_img, niiname)

    endtime = time.time()
    print('coregistration of volume took {} seconds'.format(np.round(endtime-starttime)))

    return niiname

#------------slice timing from slice order ------------------------------------------------
#------------------------------------------------------------------------------------------
def get_slice_times(sliceorder, nslices):
    # input must be 4D fMRI data
    # slice_times indicates the onset time for acquisition of each slice, relative to the TR period
    # refslice is the timing of the slice to correct all slices to
    # for example, if 10 slices are acquired in interleaved order, odd slices first, evenly spaced
    # over the TR period, then slice_order = np.concatenate((np.arange(0,10,2), np.arange(1,10,2)))
    # time_per_slice = np.arange(nslices)/nslices, sort_order = np.argsort(slice_order)
    # therefore slice_times = time_per_slice[sort_order]
    # orderoptions = {'Inc,Alt,Odd', 'Inc,Alt,Even', 'Dec,Alt,N', 'Dec,Alt,N-1', 'Inc,Seq.', 'Dec,Seq.'}

    if sliceorder == 'Inc,Alt,Odd':
        slice_order = np.concatenate((np.arange(1, nslices+1, 2), np.arange(2, nslices+1, 2)))

    if sliceorder == 'Inc,Alt,Even':
        # slice index 1 is even because it is slice number 2
        slice_order = np.concatenate((np.arange(2, nslices+1, 2), np.arange(1, nslices+1, 2)))

    if sliceorder == 'Dec,Alt,N':
        slice_order = np.concatenate((np.arange(nslices, 0, -2), np.arange(nslices-1, 0, -2)))

    if sliceorder == 'Dec,Alt,N-1':
        slice_order = np.concatenate((np.arange(nslices-1, 0, -2), np.arange(nslices, 0, -2)))

    if sliceorder == 'Inc,Seq.':
        slice_order = np.arange(1,nslices+1,1)

    if sliceorder == 'Dec,Seq.':
        slice_order = np.arange(nslices,0,-1)

    time_per_slice = np.arange(nslices)/nslices
    sort_order = np.argsort(slice_order)
    slice_times = time_per_slice[sort_order]

    return slice_times


#-----------slice-timing correction-----------------------------------------
#---------------------------------------------------------------------------

# -----slice timing correction for fMRI data------------------------------------------------------------------
def slicetimecorrection(filename, sliceorder, refslice, slice_axis = 2, corrected_prefix = 't'):
    # input must be 4D fMRI data
    # slice_times indicates the onset time for acquisition of each slice, relative to the TR period
    # refslice is the timing of the slice to correct all slices to
    # for example, if 10 slices are acquired in interleaved order, odd slices first, evenly spaced
    # over the TR period, then slice_order = np.concatenate((np.arange(0,10,2), np.arange(1,10,2)))
    # time_per_slice = np.arange(nslices)/nslices, sort_order = np.argsort(slice_order)
    # therefore slice_times = time_per_slice[sort_order]
    #
    input_img = nib.load(filename)
    input_data = input_img.get_data()
    affine = input_img.affine

    [xs,ys,zs,ts] = np.shape(input_data)
    corrected_image = np.zeros((xs,ys,zs,ts))

    if slice_axis == 0:
        nslices = xs
        slice_times = get_slice_times(sliceorder, nslices)

        for xx in range(xs):
            # thisslicetimes = np.arange(ts) + slice_times[xx]
            timeshift = slice_times[xx] - slice_times[refslice]
            slices = np.squeeze(input_data[xx,:,:,:])
            # interpolate to the interptimes
            corrected_image[xx,:,:,:] = i3d.linear_translation(slices,[0, 0, timeshift])

    if slice_axis == 1:
        nslices = ys
        slice_times = get_slice_times(sliceorder, nslices)

        for yy in range(ys):
            # thisslicetimes = np.arange(ts) + slice_times[xx]
            timeshift = slice_times[yy] - slice_times[refslice]
            slices = np.squeeze(input_data[:,yy,:,:])
            # interpolate to the interptimes
            corrected_image[:,yy,:,:] = i3d.linear_translation(slices,[0, 0, timeshift])

    if slice_axis == 2:
        nslices = zs
        slice_times = get_slice_times(sliceorder, nslices)

        for zz in range(zs):
            # thisslicetimes = np.arange(ts) + slice_times[xx]
            timeshift = slice_times[zz] - slice_times[refslice]
            slices = np.squeeze(input_data[:,:,zz,:])
            # interpolate to the interptimes
            corrected_image[:,:,zz,:] = i3d.linear_translation(slices,[0, 0, timeshift])

    # write result as new NIfTI format image set
    pname, fname = os.path.split(filename)
    niiname = os.path.join(pname, corrected_prefix + fname)
    resulting_img = nib.Nifti1Image(corrected_image, affine)
    nib.save(resulting_img, niiname)

    return niiname


#----------image smoothing----------------------------------------------------------
#-----------------------------------------------------------------------------------
def imagesmooth(filename, smoothing_width, smoothed_prefix = 's'):

    input_img = nib.load(filename)
    input_data = input_img.get_data()
    affine = input_img.affine

    # get the size of the image images, could 2, 3, or 4 dimensional
    dims = np.shape(input_data)
    ndims = len(dims)
    if ndims > 4:
        print('pysmoothing:  maximum number of dimensions allowed is 4:  input has ',ndims,' dimensions')
        return

    nspatialdims = np.min([3,ndims])

    if np.size(smoothing_width) < nspatialdims:
        smoothval = smoothing_width*np.ones(nspatialdims)

    smoothed_image = np.zeros(np.shape(input_data))
    if ndims == 4:
        for tt in range(dims[3]):
            smoothed_image[:,:,:,tt] = nd.gaussian_filter(input_data[:,:,:,tt], smoothval)
    else:
        smoothed_image = nd.gaussian_filter(input_data, smoothval)


    # write result as new NIfTI format image set
    pname, fname = os.path.split(filename)

    niiname = os.path.join(pname, smoothed_prefix + fname)

    resulting_img = nib.Nifti1Image(smoothed_image, affine)
    nib.save(resulting_img, niiname)

    return niiname


#----------clean the data----------------------------------------------------------
#-----------------------------------------------------------------------------------
def cleandata(niiname, prefix,nametag, clean_prefix = 'x'):
    # cleaning the data ....
    # load the nifti image data that has already been coregistered etc.
    # load the confounds, noise, etc., basis sets
    # do a GLM fit, and subtract the noise from the data
    # scale the data to percent signal change from the mean
    # save copies of 1) clean data, variations from the mean
    # and 2) average intensity data
    #
    # this function is very specific to the file structure etc. that
    # has been adopted for this software package, because of the need to
    # predict the names of files containing information about noise, confounds, etc.

    pname, fname = os.path.split(niiname)
    fnameroot, ext = os.path.splitext(fname)
    input_img = nib.load(niiname)
    input_data = input_img.get_data()
    affine = input_img.affine

    # name of white matter confounds excel file ...
    wmdata_xlname = os.path.join(pname, 'wmnoise'+nametag+'.xlsx')
    xls = pd.ExcelFile(wmdata_xlname, engine = 'openpyxl')
    df1 = pd.read_excel(xls, 'wmnoise')
    wm1 = np.array(df1.loc[:, 'region1'])
    wm2 = np.array(df1.loc[:, 'region2'])
    wm3 = np.array(df1.loc[:, 'region3'])

    # name of motion confounds excel file ...
    # need the base name with no prefixes for the motiondata (calculated before other pre-processing steps are done)
    fnameroot2 = fnameroot.replace(prefix,'')
    motiondata_xlname = os.path.join(pname, 'motiondata'+nametag+'.xlsx')
    xls = pd.ExcelFile(motiondata_xlname, engine = 'openpyxl')
    df1 = pd.read_excel(xls, 'motion_data')

    namelist = ['dx1','dy1','dz1','dx2','dy2','dz2','dx3','dy3','dz3']
    for num, name in enumerate(namelist):
        dp = df1.loc[:, name]
        if num == 0:
            dpvals = [dp]
        else:
            dpvals.append(dp)

    # put the wmdata and motiondata into a basis set
    # do a GLM
    # subtract off the fit --> get the clean data
    # write out the data to a new nifti file

    ndrop = 2  # make this an input
    basis_set = np.concatenate((wm1[np.newaxis,:],wm2[np.newaxis,:],wm3[np.newaxis,:],np.array(dpvals)),axis=0)
    varcheck = np.var(basis_set[:,ndrop:], axis = 1)
    a = np.argwhere(varcheck > 1.0e-3)   # set some tolerance on the variance before it is considered essentially zero
    basis_set = basis_set[a.flatten(),:]   # keep only the basis elements with non-zero variance (i.e. not constant)
    print('cleandata:  shape of basis_set is ', np.shape(basis_set))

    residual, meanvalue = GLMfit.GLMfit_subtract_and_separate(input_data, basis_set, add_constant = True, ndrop = ndrop)

    cleaned_data = 100.*residual/(meanvalue + 1.0e-20)   # express the clean data as percent signal change from the average

    pname, fname = os.path.split(niiname)
    niiname_out = os.path.join(pname, clean_prefix + fname)
    resulting_img = nib.Nifti1Image(cleaned_data, affine)
    nib.save(resulting_img, niiname_out)

    niiname_avg = os.path.join(pname, clean_prefix+fnameroot+'_avg.nii')
    avg_img = nib.Nifti1Image(meanvalue[:,:,:,0], affine)
    nib.save(avg_img, niiname_avg)

    return niiname_out


#-----------set the prefix list------------------------------------
# do this in a separate function so it can be used elsewhere for information
def setprefixlist(settingsfile):
    settings = np.load(settingsfile, allow_pickle=True).flat[0]

    coreg_choice = settings['coreg_choice']
    slicetime_choice = settings['slicetime_choice']
    norm_choice = settings['norm_choice']
    smooth_choice = settings['smooth_choice']
    define_choice = settings['define_choice']
    clean_choice = settings['clean_choice']

    # identify pre-processing steps to be completed, and data prefixes for inputs for each step
    prefix = ''
    prefix_list = [prefix]
    if (coreg_choice == 'Yes.') | (coreg_choice == 'Done'):
        prefix = 'c' + prefix
    prefix_list.append(prefix)
    if (slicetime_choice == 'Yes.') | (slicetime_choice == 'Done'):
        prefix = 't' + prefix
    prefix_list.append(prefix)
    if (norm_choice == 'Yes.') | (norm_choice == 'Done'):
        prefix = 'p' + prefix
    prefix_list.append(prefix)
    if (smooth_choice == 'Yes.') | (smooth_choice == 'Done'):
        prefix = 's' + prefix
    prefix_list.append(prefix)
    # defining the basis sets does not add a prefix
    prefix_list.append(prefix)
    if (clean_choice == 'Yes.') | (clean_choice == 'Done'):
        prefix = 'x' + prefix
    prefix_list.append(prefix)

    return prefix_list

# -----------run the pre-processing steps in the necessary order -------------------
#-----------------------------------------------------------------------------------
def run_preprocessing(settingsfile):
    # inputs could be: DBname, DBnum, coreg_choice, slicetime_choice, norm_choice, smooth_choice, define_choice, clean_choice

    # first get the necessary input data
    settings = np.load(settingsfile, allow_pickle=True).flat[0]
    print('Pre-processing:  settings file = ', settingsfile)
    PPdatabasename = settings['DBname']
    PPdatabasenum = settings['DBnum']
    # BASEdir = os.path.dirname(PPdatabasename)
    xls = pd.ExcelFile(PPdatabasename, engine = 'openpyxl')
    df1 = pd.read_excel(xls, 'datarecord')

    coreg_choice = settings['coreg_choice']
    slicetime_choice = settings['slicetime_choice']
    norm_choice = settings['norm_choice']
    smooth_choice = settings['smooth_choice']
    define_choice = settings['define_choice']
    clean_choice = settings['clean_choice']

    # identify pre-processing steps to be completed, and data prefixes for inputs for each step
    prefix_list = setprefixlist(settingsfile)

    # prefix = ''
    # prefix_list = [prefix]
    # if (coreg_choice == 'Yes.') | (coreg_choice == 'Done'):
    #     prefix = 'c' + prefix
    # prefix_list.append(prefix)
    # if (slicetime_choice == 'Yes.') | (slicetime_choice == 'Done'):
    #     prefix = 't' + prefix
    # prefix_list.append(prefix)
    # if (norm_choice == 'Yes.') | (norm_choice == 'Done'):
    #     prefix = 'p' + prefix
    # prefix_list.append(prefix)
    # if (smooth_choice == 'Yes.') | (smooth_choice == 'Done'):
    #     prefix = 's' + prefix
    # prefix_list.append(prefix)
    # # defining the basis sets does not add a prefix
    # prefix_list.append(prefix)
    # if (clean_choice == 'Yes.') | (clean_choice == 'Done'):
    #     prefix = 'x' + prefix
    # prefix_list.append(prefix)

    print('Pre-processing: started organizing at ', time.ctime(time.time()))
    # print('prefix_list = ',prefix_list)

    # if running co-registration ....
    print('coreg_choice = ',coreg_choice)
    if (coreg_choice == 'Yes.'):
        print('running co-registration ...')
        prefix = prefix_list[0]   # first step of pre-processing
        for nn, dbnum in enumerate(PPdatabasenum):
            print('pre-processing:  coregistration: databasenum ', dbnum)
            dbhome = df1.loc[dbnum, 'datadir']
            fname = df1.loc[dbnum, 'niftiname']
            seriesnumber = df1.loc[dbnum, 'seriesnumber']
            niiname = os.path.join(dbhome, fname)
            fullpath, filename = os.path.split(niiname)
            prefix_niiname = os.path.join(fullpath, prefix + filename)
            # run the coregistration ...
            nametag = '_s{}'.format(seriesnumber)
            niiname = coregister(prefix_niiname, nametag)
            print('Coregistration: output name is ',niiname)

        print('Coregistration finished ...', time.ctime(time.time()))


    # if applying slice-timing correction ...
    print('slicetime_choice =',slicetime_choice,'...')
    if (slicetime_choice == 'Yes.'):
        print('running slice time correction ...')
        prefix = prefix_list[1]   # second step of pre-processing
        sliceaxis = settings['sliceaxis']
        refslice = settings['refslice']
        sliceorder = settings['sliceorder']
        # calculate slice times from sliceorder ...
        print('Pre-processing:  need to calculate slice times from sliceorder ...')
        print('         sliceorder = ',sliceorder)

        for nn, dbnum in enumerate(PPdatabasenum):
            print('pre-processing:  slice-timing correction: databasenum ', dbnum)
            dbhome = df1.loc[dbnum, 'datadir']
            fname = df1.loc[dbnum, 'niftiname']
            niiname = os.path.join(dbhome, fname)
            fullpath, filename = os.path.split(niiname)
            prefix_niiname = os.path.join(fullpath, prefix + filename)
            # run the slice-timing correction ...
            niiname = slicetimecorrection(prefix_niiname, sliceorder, refslice, sliceaxis)
            print('Slice-timing correction: output name is ',niiname)

        print('Slice-timing correction finished ...', time.ctime(time.time()))


    # if applying normalization ...
    print('norm_choice =',norm_choice,'...')
    if (norm_choice == 'Yes.'):
        print('applying normalization to data ...')
        prefix = prefix_list[2]   # 3rd step of pre-processing

        for nn, dbnum in enumerate(PPdatabasenum):
            print('pre-processing:  applying normalization: databasenum ', dbnum)
            dbhome = df1.loc[dbnum, 'datadir']
            fname = df1.loc[dbnum, 'niftiname']
            niiname = os.path.join(dbhome, fname)
            fullpath, filename = os.path.split(niiname)
            prefix_niiname = os.path.join(fullpath, prefix + filename)

            normdataname = df1.loc[dbnum.astype('int'), 'normdataname']
            normdataname_full = os.path.join(dbhome, normdataname)
            normdata = np.load(normdataname_full, allow_pickle=True).flat[0]
            T = normdata['T']
            Tfine = normdata['Tfine']
            template_affine = normdata['template_affine']

            # applying normalization to each data set ...
            niiname = pynormalization.apply_normalization_to_nifti(prefix_niiname, T, Tfine, template_affine)
            print('Applying normalization: output name is ',niiname)

        print('Finished applying normalization ...', time.ctime(time.time()))


    # if applying smoothing ...
    print('smooth_choice =',smooth_choice,'...')
    if (smooth_choice == 'Yes.'):
        print('applying smoothing to data ...')
        prefix = prefix_list[3]   # 4th step of pre-processing

        for nn, dbnum in enumerate(PPdatabasenum):
            print('pre-processing:  applying smoothing: databasenum ', dbnum)
            dbhome = df1.loc[dbnum, 'datadir']
            fname = df1.loc[dbnum, 'niftiname']
            niiname = os.path.join(dbhome, fname)
            fullpath, filename = os.path.split(niiname)
            prefix_niiname = os.path.join(fullpath, prefix + filename)

            smoothwidth = settings['smoothwidth']

            # applying smoothing to each data set ...
            niiname = imagesmooth(prefix_niiname, smoothwidth)
            print('Applying smoothing: output name is ',niiname)

        print('Finished applying smoothing ...', time.ctime(time.time()))


    # if defining basis sets ...
    print('define_choice =',define_choice,'...')
    if (define_choice == 'Yes.'):
        print('defining basis sets for main effects and confounds ...')
        prefix = prefix_list[4]   # 5th step of pre-processing

        for nn, dbnum in enumerate(PPdatabasenum):
            print('pre-processing:  defining basis sets: databasenum ', dbnum)
            dbhome = df1.loc[dbnum, 'datadir']
            fname = df1.loc[dbnum, 'niftiname']
            niiname = os.path.join(dbhome, fname)
            fullpath, filename = os.path.split(niiname)
            prefix_niiname = os.path.join(fullpath, prefix + filename)

            input_img = nib.load(prefix_niiname)
            input_data = input_img.get_data()
            nvols = np.shape(input_data)[3]

            TR = df1.loc[dbnum, 'TR']
            normtemplatename = df1.loc[dbnum, 'normtemplatename']
            normdataname = df1.loc[dbnum, 'normdataname']
            seriesnumber = df1.loc[dbnum, 'seriesnumber']
            normdataname_full = os.path.join(dbhome, normdataname)

            # generate main effects - saved as sheet in database file
            sheetname = pybasissets.calculate_maineffects(PPdatabasename, dbnum, TR, nvols)
            # generate white matter confounds - saved as excel file with nifti data, named ..._motiondata.xlsx
            nametag = '_s{}'.format(seriesnumber)
            wmtc, xlname = pybasissets.get_whitematter_noise(prefix_niiname, normtemplatename, nametag)
            # generate motion confounds - saved as excel file with nifti data
            motion_xlname = pybasissets.coreg_to_motionparams(niiname, normdataname_full, normtemplatename, nametag)

            # compile these into one basis set file? - see what is easiest for GLM when it is ready
            print('Finished defining basis sets ...', time.ctime(time.time()))

    # if cleaning the data ...
    print('clean_choice =',clean_choice,'...')
    if (clean_choice == 'Yes.'):
        print('cleaning the data (i.e. removing noise and confounds ...')
        prefix = prefix_list[5]   # 6th step of pre-processing

        for nn, dbnum in enumerate(PPdatabasenum):
            print('pre-processing:  cleaning the data: databasenum ', dbnum)
            dbhome = df1.loc[dbnum, 'datadir']
            fname = df1.loc[dbnum, 'niftiname']
            seriesnumber = df1.loc[dbnum, 'seriesnumber']
            niiname = os.path.join(dbhome, fname)
            fullpath, filename = os.path.split(niiname)
            prefix_niiname = os.path.join(fullpath, prefix + filename)

            nametag = '_s{}'.format(seriesnumber)
            niiname = cleandata(prefix_niiname, prefix, nametag)

        print('Finished cleaning the data ...', time.ctime(time.time()))

    successflag = 1

    return successflag

