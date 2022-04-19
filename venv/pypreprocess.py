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
import openpyxl
import pybrainregistration

# -----coregister------------------------------------------------------------------
# this function assumes the input is given as a time-series, and the images are all coregistered
# with the 3rd volume in the series, or the 1st volume if only two volumes are provided
def coregister(filename, nametag, coregistered_prefix = 'c'):
    starttime = time.time()
    #set default main settings for MIRT coregistration
    # use 'ssd', or 'cc' which was used in the previous matlab version of this function
    main_init = {'similarity':'cc',   # similarity measure, e.g. SSD, CC, SAD, RC, CD2, MS, MI
            'subdivide':1,       # use 1 hierarchical level
            'okno':4,           # mesh window size
            'lambda':0.1,     # transformation regularization weight, 0 for none
            'single':1}
    
    #Optimization settings
    optim_init = {'maxsteps':500,    # maximum number of iterations at each hierarchical level
             'fundif':1e-4,     # tolerance (stopping criterion)
             'gamma':0.1,         # initial optimization step size
             'anneal':0.7}        # annealing rate on the optimization step

    # original values
    # optim_init = {'maxsteps':100,    # maximum number of iterations at each hierarchical level
    #          'fundif':1e-6,     # tolerance (stopping criterion)
    #          'gamma':1.0,         # initial optimization step size
    #          'anneal':0.7}        # annealing rate on the optimization step

    
    smooth_refimage = True
    smooth_regimage = True
    
    input_img = nib.load(filename)
    input_data = input_img.get_fdata()
    affine = input_img.affine
    
    xs,ys,zs,ts = np.shape(input_data)

    #--------------------------------------------------
    # for registration want spatial dimensions no smaller than 24
    original_input_data = copy.deepcopy(input_data)
    dimen = np.array([xs, ys, zs])
    dimen[dimen < 24] = 24
    padsize = np.floor((dimen - np.array([xs, ys, zs])) / 2.).astype(int)
    padding_applied = (padsize > 0).any()
    if padding_applied:
        original_input_data = copy.deepcopy(input_data)
        input_data = np.zeros((dimen[0], dimen[1], dimen[2], ts))
        x1 = padsize[0]
        x2 = x1 + xs
        y1 = padsize[1]
        y2 = y1 + ys
        z1 = padsize[2]
        z2 = z1 + zs
        for tt in range(ts):
            input_data[x1:x2, y1:y2, z1:z2, tt] = original_input_data[:, :, :, tt]
        xs, ys, zs, ts = np.shape(input_data)

    #--------------------------------------------------
    
    # coregister to the 3rd volume in the time-series
    # if ts >= 2:
    #     refimage = input_data[:,:,:,2]
    # else:
    #     refimage = input_data[:,:,:,0]
    refimage = np.mean(input_data, axis =3)
            
    smoothval = (1.2, 1.2, 1.2)
    if smooth_refimage:
        refimages = nd.gaussian_filter(refimage, smoothval)
    else:
        refimages = refimage

    refimages = refimages/np.max(refimages)
    
    result = np.zeros((xs,ys,zs,ts))
    print('Running coregistration step ...')
    Qcheck_initial = np.zeros(ts)
    Qcheck = np.zeros(ts)
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
        res, new_img = mirt.py_mirt3D_register(refimages, regimages, main, optim)
        m = np.max(regimage)
        new_img2 = m*mirt.py_mirt3D_transform(regimage/m,res)

        R = np.corrcoef(refimage.flatten(),new_img2.flatten())
        Qcheck[tt] = R[0,1]
        R = np.corrcoef(refimage.flatten(),regimage.flatten())
        Qcheck_initial[tt] = R[0,1]

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

    if padding_applied:
        xs,ys,zs,ts = np.shape(original_input_data)
        cropped_result = np.zeros((xs,ys,zs,ts))
        x1 = padsize[0]
        x2 = x1 + xs
        y1 = padsize[1]
        y2 = y1 + ys
        z1 = padsize[2]
        z2 = z1 + zs
        for tt in range(ts):
            cropped_result[:, :, :, tt] = result[x1:x2, y1:y2, z1:z2, tt]
        result = cropped_result

    niiname = os.path.join(pname, coregistered_prefix+fname)
    resulting_img = nib.Nifti1Image(result, affine)
    nib.save(resulting_img, niiname)

    minQ = np.min(Qcheck)
    minQinitial = np.min(Qcheck_initial)

    endtime = time.time()
    print('coregistration of volume took {} seconds,  min correlation is {:.2f}, was originally {:.2f}'.format(np.round(endtime-starttime), minQ, minQinitial))

    Qcheck = np.concatenate((Qcheck[:,np.newaxis],Qcheck_initial[:,np.newaxis]),axis=1)
    return niiname, Qcheck


#-------------coregistration guided by rough normalization results-------------------------
#------------------------------------------------------------------------------------------
def guided_coregistration(filename, nametag, normdataname, normtemplatename, coregistered_prefix = 'c'):
    # use rough normalization section positions to check for larger movements than can be
    # corrected by MIRT coregistration
    print('coregistration guided by rough normalization results ...')

    starttime = time.time()
    # set default main settings for MIRT coregistration
    # use 'ssd', or 'cc' which was used in the previous matlab version of this function
    main_init = {'similarity': 'cc',  # similarity measure, e.g. SSD, CC, SAD, RC, CD2, MS, MI
                 'subdivide': 1,  # use 1 hierarchical level
                 'okno': 4,  # mesh window size
                 'lambda': 0.5,  # transformation regularization weight, 0 for none
                 'single': 1}

    # Optimization settings
    optim_init = {'maxsteps': 500,  # maximum number of iterations at each hierarchical level
                  'fundif': 1e-4,  # tolerance (stopping criterion)
                  'gamma': 0.05,  # initial optimization step size
                  'anneal': 0.5}  # annealing rate on the optimization step

    # input_img = nib.load(filename)
    # input_data = input_img.get_data()
    # affine = input_img.affine

    input_data, affine = i3d.load_and_scale_nifti(filename)   # this function also scales the image to 1 mm cubic voxels

    xs, ys, zs, ts = np.shape(input_data)

    # coregister to the 3rd volume in the time-series
    if ts >= 2:
        tref = 2
    else:
        tref = 0
    refimage = input_data[:, :, :, tref]

    refimage = refimage/np.max(refimage)

    # check if sections need to be adjusted
    # load normalization results
    normdata = np.load(normdataname, allow_pickle=True).flat[0]
    result = normdata['result']
    new_result = copy.deepcopy(result)
    nsections = len(result)

    # setup
    pw = 1e-3   #  position_stiffness
    ddx, ddy, ddz = np.mgrid[range(xs), range(ys), range(zs)]

    # do this for each volume in the time-series
    output_img = np.zeros(np.shape(input_data))
    transformation_record = []
    coreg_data = []
    for tt in range(ts):
        if tt == tref:
            new_img2 = input_data[:, :, :, tt]
        else:
            starttime = time.time()
            print('volume {} of {}    {}'.format(tt + 1, ts, time.ctime()))
            img = input_data[:, :, :, tt]
            img = img / np.max(img)
            warpdata = []
            map_step = []
            for nn in range(nsections):
                angle = result[nn]['angle']
                angley = result[nn]['angley']
                coords = result[nn]['coords']  # reference position for regions of interest
                original_section = result[nn]['original_section']
                template_section = result[nn]['template_section']
                # section_mapping_coords = result[nn]['section_mapping_coords']

                template = original_section

                # check position - keep rotation angles the same
                imgR = i3d.rotate3D(img, angle, coords, 0)
                imgRR = i3d.rotate3D(imgR, angley, coords, 1)
                cc = i3d.normxcorr3(imgRR, template, shape='same')

                # find the combination of correlation and proximity to the expected location
                dist = np.sqrt((ddx - coords[0]) ** 2 + (ddy - coords[1]) ** 2 + (ddz - coords[2]) ** 2)
                # dist[dist < 5] = 0
                pos_weight = 1 / (pw * dist + 1)

                cc_temp = cc * pos_weight
                m = np.max(cc_temp)
                xp, yp, zp = np.where(cc_temp == m)
                coordsR = np.array([xp[0], yp[0], zp[0]])

                # rotate coords back
                v = coordsR - coords  # vector from rotation point to the best match position
                Mx = pynormalization.rotation_matrix(-angle, axis=0)
                My = pynormalization.rotation_matrix(-angley, axis=1)
                Mtotal = np.dot(My, Mx)
                bestpos = np.dot(v, Mtotal) + coords
                new_result[nn]['coords'] = bestpos

                map_step.append({'coords': bestpos})

                dx, dy, dz = np.floor(np.array(np.shape(result[nn]['template_section'])) / 2).astype(int)
                # get mapping coordinates from the current volume to the original
                Xt, Yt, Zt = np.mgrid[(coords[0] - dx):(coords[0] + dx):(2 * dx + 1) * 1j,
                             (coords[1] - dy):(coords[1] + dy):(2 * dy + 1) * 1j,
                             (coords[2] - dz):(coords[2] + dz):(2 * dz + 1) * 1j]
                # X etc are image coordinates in the rotated image, which was matched to the fixed template
                X, Y, Z = np.mgrid[(bestpos[0] - dx):(bestpos[0] + dx):(2 * dx + 1) * 1j,
                          (bestpos[1] - dy):(bestpos[1] + dy):(2 * dy + 1) * 1j,
                          (bestpos[2] - dz):(bestpos[2] + dz):(2 * dz + 1) * 1j]

                # organize the outputs
                section_mapping_coords = {'X': X, 'Y': Y, 'Z': Z, 'Xt': Xt, 'Yt': Yt, 'Zt': Zt}
                new_result[nn]['section_mapping_coords'] = section_mapping_coords
                new_result[nn]['angle'] = 0.
                new_result[nn]['angley'] = 0.
                warpdata.append(section_mapping_coords)

            time1 = time.time()
            # show the results
            # Xlist = np.zeros(nsections)
            # Ylist = np.zeros(nsections)
            # Xlist2 = np.zeros(nsections)
            # Ylist2 = np.zeros(nsections)
            # for nn in range(nsections):
            #     Xlist[nn] = result[nn]['coords'][2]
            #     Ylist[nn] = result[nn]['coords'][1]
            #     Xlist2[nn] = map_step[nn]['coords'][2]
            #     Ylist2[nn] = map_step[nn]['coords'][1]
            # fig = plt.figure(21), plt.imshow(refimage[10, :, :], 'gray')
            # plt.plot(Xlist, Ylist, color="red", linewidth=2)
            #
            # fig = plt.figure(22), plt.imshow(img[10, :, :], 'gray')
            # plt.plot(Xlist2, Ylist2, color="red", linewidth=2)

            # make mapped sections consistent
            adjusted_sections = []
            normtemplatename = 'ccbs'
            # new_result2 = pynormalization.align_override_sections(new_result, adjusted_sections, filename,
            #                                                       normtemplatename)
            new_result2 = new_result    # override this step temporarily


            time2 = time.time()
            new_warpdata = []
            for nn in range(len(new_result2)):
                new_warpdata.append(new_result2[nn]['section_mapping_coords'])

            # combine the warp fields from each section into one map
            fit_order = [1,1,1]
            T, reverse_map_image, forward_map_image, inv_Rcheck = \
                    pynormalization.py_combine_warp_fields(new_warpdata,img,refimage,fit_order)

            img1 = i3d.warp_image(img, T['Xs'], T['Ys'], T['Zs'])
            time3 = time.time()

            do_fine_tuning = False
            if do_fine_tuning:
                # now do the fine-tuning with MIRT------------------------------------------
                optim = copy.deepcopy(optim_init)
                main = copy.deepcopy(main_init)

                res, norm_img_fine = mirt.py_mirt3D_register(refimage / np.max(refimage), img1 / np.max(img1), main, optim)
                print('completed fine-tune mapping with py_norm_fine_tuning ...')
                time4 = time.time()

                print('guided_coregistration: mapping sections {:.1f} seconds, aligning {:.1f} seconds, combining {:.1f} seconds, fine-tuning {:.1f} seconds'.format(time1-starttime,time2-time1,time3-time2,time4-time3))

                F = mirt.py_mirt3D_F(res['okno']);  # Precompute the matrix B - spline basis functions
                Xx, Xy, Xz = mirt.py_mirt3D_nodes2grid(res['X'], F, res['okno']);  # obtain the position of all image voxels (Xx, Xy, Xz)
                # from the positions of B-spline control points (res['X']

                xs, ys, zs = np.shape(img)
                X, Y, Z = np.mgrid[range(xs), range(ys), range(zs)]

                # fine-tuning deviation from the original positions
                dX = Xx[:xs, :ys, :zs] - X
                dY = Xy[:xs, :ys, :zs] - Y
                dZ = Xz[:xs, :ys, :zs] - Z

                T2 = T
                T2['Xs'] += dX
                T2['Ys'] += dY
                T2['Zs'] += dZ
            else:
                T2 = T
                print('guided_coregistration: mapping sections {:.1f} seconds, aligning {:.1f} seconds, combining {:.1f} seconds'.format(time1-starttime,time2-time1,time3-time2))


            img3 = i3d.warp_image(img1, T2['Xs'], T2['Ys'], T2['Zs'])

        output_img[:, :, :, tt] = img3

        coreg_data.append({'T':T2, 'map_step':map_step})

    # write result as new NIfTI format image set
    pname, fname = os.path.split(filename)
    fnameroot, ext = os.path.splitext(fname)
    if ext == '.gz':
        fnameroot, e2 = os.path.splitext(fnameroot) # split again
    # define names for outputs
    coregdata_name = os.path.join(pname, 'coregdata' + nametag + '.npy')
    np.save(coregdata_name, coreg_data)

    niiname = os.path.join(pname, coregistered_prefix + fname)
    resulting_img = nib.Nifti1Image(output_img, affine)
    nib.save(resulting_img, niiname)

    endtime = time.time()
    print('coregistration of volume took {} seconds'.format(np.round(endtime - starttime)))

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
    if ext == '.gz':
        fnameroot, e2 = os.path.splitext(fnameroot) # split again
    input_img = nib.load(niiname)
    input_data = input_img.get_fdata()
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
    a = np.argwhere(varcheck > 1.0e-6)   # set some tolerance on the variance before it is considered essentially zero
    basis_set = basis_set[a.flatten(),:]   # keep only the basis elements with non-zero variance (i.e. not constant)
    print('cleandata:  shape of basis_set is ', np.shape(basis_set))

    residual, meanvalue = GLMfit.GLMfit_subtract_and_separate(input_data, basis_set, add_constant = True, ndrop = ndrop)

    cleaned_data = 100. + 100.*residual/(meanvalue + 1.0e-20)   # express the clean data as percent signal change from the average

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

            # new guided coregistration method
            normtemplatename = df1.loc[dbnum, 'normtemplatename']
            # normdataname = df1.loc[dbnum, 'normdataname']
            # normdataname_full = os.path.join(dbhome, normdataname)
            # niiname = guided_coregistration(prefix_niiname, nametag, normdataname_full, normtemplatename)
            if normtemplatename == 'brain':
                niiname, Qcheck = pybrainregistration.brain_coregistration(niiname, nametag)
            else:
                # original coregistration method
                niiname, Qcheck = coregister(prefix_niiname, nametag)

            # now write the new database values
            keylist = df1.keys()
            for kname in keylist:
                if 'Unnamed' in kname: df1.pop(kname)  # remove blank fields from the database
            # add coregistration quality to database
            if 'coreg_quality' not in keylist:
                df1['coreg_quality'] = 0
            df1.loc[dbnum, 'coreg_quality'] = np.min(Qcheck[:,0])

            # need to delete the existing sheet before writing the new version
            # delete sheet - need to use openpyxl
            workbook = openpyxl.load_workbook(PPdatabasename)
            del workbook['datarecord']
            workbook.save(PPdatabasename)

            # write it to the database by appending a sheet to the excel file
            # remove old version of datarecord first
            with pd.ExcelWriter(PPdatabasename, engine="openpyxl", mode='a') as writer:
                df1.to_excel(writer, sheet_name='datarecord')

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

            normtemplatename = df1.loc[dbnum, 'normtemplatename']
            normdataname = df1.loc[dbnum, 'normdataname']
            normdataname_full = os.path.join(dbhome, normdataname)
            normdata = np.load(normdataname_full, allow_pickle=True).flat[0]

            if normtemplatename == 'brain':
                norm_brain_affine = normdata['norm_affine_transformation']
                # img_data, img_affine = i3d.load_and_scale_nifti(niiname)   # this function also scales the images to 1mm cubic voxels

                input_img = nib.load(niiname)
                img_affine = input_img.affine
                img_hdr = input_img.header
                img_data = input_img.get_fdata()

                norm_prefix = 'p'
                print('starting applying normalization ....')
                norm_brain_data = pybrainregistration.dipy_apply_brain_normalization(img_data, norm_brain_affine, verbose=True)
                print('finished applying normalization ....')
                # save the normalized nifti images ...

                resulting_img = nib.Nifti1Image(norm_brain_data, norm_brain_affine.affine)
                p, f_full = os.path.split(niiname)
                f, e = os.path.splitext(f_full)
                ext = '.nii'
                if e == '.gz':
                    f, e2 = os.path.splitext(f) # split again
                    ext = '.nii.gz'
                output_niiname = os.path.join(p, norm_prefix + f + ext)
                # need to write the image data out in a smaller format to save disk space ...
                nib.save(resulting_img, output_niiname)

            else:
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
            input_data = input_img.get_fdata()
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

            # original motion paramters method
            motion_xlname = pybasissets.coreg_to_motionparams(niiname, normdataname_full, normtemplatename, nametag)

            # new method based on guided coregistration
            # motion_xlname = pybasissets.guided_coreg_to_motionparams(normdataname_full, normtemplatename, nametag)

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

