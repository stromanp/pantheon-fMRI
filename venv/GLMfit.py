# general linear model fitting
# variations for fMRI analysis
import numpy as np
import os
import pandas as pd
import nibabel as nib
import pybasissets
import pydatabase
import copy


#----------------GLMfit------------------------------------------
# function to do a general GLM fit to image data, and return stats values
def GLMfit(image_data, basis_set, contrast='None', add_constant = False, ndrop = 0):
    # ignore initial ndrop points, if needed
    xs,ys,zs,ts = np.shape(image_data[:,:,:,ndrop:])
    nvox = xs*ys*zs
    S = np.reshape(image_data[:,:,:,ndrop:], (nvox,ts))
    # S = BG (matrix multiplication)
    # so B = S*G'*inv(G*G')
    # S is nvox X ts,  so the basis set needs to be nbasis x ts
    G = basis_set[:,ndrop:ts]

    print('size of image_data is {}'.format(np.shape(S)))
    print('size of basis_set is {}'.format(np.shape(G)))

    if add_constant:
        # first check if there is already a constant value
        varcheck = np.var(G,axis=1)
        c = np.where(varcheck < 1.0e-6)
        if np.size(c) == 0:
            # add the constant to the basis set
            constant = np.ones((1,ts))
            G = np.concatenate((G, constant), axis=0)
            constant_index = np.shape(G)[0]
        else:
            # there is already one or more constants in the basis set
            # remove them first, and add the constant at the end
            c2 = np.where(varcheck >= 1.0e-6)
            Gtemp = G[c2[0],:]
            for v in c2[1:]:
                Gtemp = np.vstack((Gtemp,G[v,:]))
            constant = np.ones((1,ts))
            G = np.concatenate((Gtemp, constant), axis=0)
            constant_index = np.shape(G)[0]

    nb = np.shape(G)[0]   # number of basis elements
    print('nb = ', nb)

    # B = S*G*inv(G*G')
    iGG = np.linalg.inv(np.dot(G,G.T))

    B = np.dot(np.dot(S,G.T),iGG)
    fit = np.dot(B,G)
    err2 = np.sum(((S - fit)**2)/ts, axis = 1)

    # if contrast is empty ...
    if (contrast == 'None') | (not contrast):
        T = np.zeros((nvox,nb))
        sem = np.zeros((nvox,nb))
        for nn in range(nb):
            contrast = np.zeros((1,nb))
            contrast[0,nn] = 1
            scale_check = abs(np.dot(np.dot(contrast,iGG),contrast.T))
            sem[:,nn] = np.sqrt(scale_check*err2)
            T[:,nn] = B[:,nn]/(sem[:,nn] + 1.0e-20)
        T = np.reshape(T,(xs,ys,zs,nb))
        B = np.reshape(B, (xs, ys, zs, nb))
        sem = np.reshape(sem, (xs, ys, zs, nb))
    else:
        if len(contrast) < nb:
            temp = [0]*nb
            temp[:len(contrast)] = contrast
            contrast = temp
        contrast = np.array(contrast)
        print('GLMfit:  size of contrast is ',np.shape(contrast))
        T = np.zeros((nvox,1))
        sem_contrast = np.zeros((nvox,1))
        B_contrast = np.zeros((nvox,1))
        scale_check = abs(np.dot(np.dot(contrast,iGG),contrast.T))
        sem_contrast = np.sqrt(scale_check*err2)
        Bcontrast = np.dot(B,contrast)
        T = Bcontrast/(sem_contrast + 1.0e-20)

        T = np.reshape(T,(xs,ys,zs))
        B = np.reshape(Bcontrast, (xs, ys, zs))
        sem = np.reshape(sem_contrast, (xs, ys, zs))

    return B, sem, T


#----------------GLMfit_and_subtract------------------------------------------
# function to fit a basis set to data, then subtract the result from the data
def GLMfit_and_subtract(image_data, basis_set, add_constant = False, ndrop = 2):
    # this function fits the image data to the basis set, and
    # subtracts the fit from the data to give only the residual
    # but the output needs to be the same shape as the input, so pad to replace the dropped values
    xs,ys,zs,ts = np.shape(image_data[:,:,:,ndrop:])
    nvox = xs*ys*zs
    S = np.reshape(image_data[:,:,:,ndrop:], (nvox,ts))
    # S = BG (matrix multiplication)
    # so B = S*G'*inv(G*G')
    # S is nvox X ts,  so the basis set needs to be nbasis x ts
    G = basis_set[:,ndrop:]
    if add_constant:
        # first check if there is already a constant value
        varcheck = np.var(G,axis=1)
        c = np.where(varcheck < 1.0e-6)
        if np.size(c) == 0:
            # add the constant to the basis set
            constant = np.ones((1,ts))
            G = np.concatenate((G, constant), axis=0)
            constant_index = np.shape(G)[0]
        else:
            # there is already a constant in the basis set
            # constant_index = c[0][0]

            # there is already one or more constants in the basis set
            # remove them first, and add the constant at the end
            c2 = np.where(varcheck >= 1.0e-6)
            Gtemp = G[c2,:]
            constant = np.ones((1,ts))
            G = np.concatenate((Gtemp, constant), axis=0)
            constant_index = np.shape(G)[0]


    nb = np.shape(G)[0]   # number of basis elements
    # B = S*G*inv(G*G')
    iGG = np.linalg.inv(np.dot(G,G.T))
    # calculate the fit paraamters
    B = np.dot(np.dot(S,G.T),iGG)

    # calculate the resulting fit values, using all data points
    xs,ys,zs,ts = np.shape(image_data)
    nvox = xs*ys*zs
    S = np.reshape(image_data, (nvox,ts))
    G = basis_set
    if add_constant:
        if np.size(c) == 0:
            # add the constant to the basis set again
            constant = np.ones((1,ts))
            G = np.concatenate((G, constant), axis=0)

    fit = np.dot(B,G)
    # subtract fit from the original
    residual = S-fit
    # put it back to original shape
    residual = np.reshape(residual,(xs,ys,zs,ts))

    # pad to replace the dropped values - replace with the time-series average
    npad = ((0,0),(0,0),(0,0),(ndrop,0))
    residual2 = np.pad(residual, npad, mode = 'mean')

    return residual2


#----------------GLMfit_subtract_and_separate------------------------------------------
# function to fit a basis set to data, then subtract the result from the data, and
# return the cleaned data as well as the average intensity data
def GLMfit_subtract_and_separate(image_data, basis_set, add_constant = False, ndrop = 2):
    # this function separates the fit values and the constant values
    # the output includes the residual after the fit is subtracted, as well as the constant value
    # ndrop is the number of initial points to ignore
    xs,ys,zs,ts = np.shape(image_data[:,:,:,ndrop:])
    nvox = xs*ys*zs
    S = np.reshape(image_data[:,:,:,ndrop:], (nvox,ts))
    # S = BG (matrix multiplication)
    # so B = S*G'*inv(G*G')
    # S is nvox X ts,  so the basis set needs to be nbasis x ts
    G = basis_set[:,ndrop:]
    varcheck = np.var(G,axis=1)
    c = np.where(varcheck < 1.0e-6)
    if add_constant:
        # first check if there is already a constant value
        if np.size(c) == 0:
            # add the constant to the basis set
            constant = np.ones((1,ts))
            G = np.concatenate((G, constant), axis=0)
            constant_index = np.shape(G)[0]-1
        else:
            # there is already a constant in the basis set
            # constant_index = c[0][0]-1

            # there is already one or more constants in the basis set
            # remove them first, and add the constant at the end
            c2 = np.where(varcheck >= 1.0e-6)
            Gtemp = G[c2,:]
            constant = np.ones((1,ts))
            G = np.concatenate((Gtemp, constant), axis=0)
            constant_index = np.shape(G)[0]-1
    else:
        constant_index = c[0][0]-1

    print('constant_index = ', constant_index)

    nb = np.shape(G)[0]   # number of basis elements
    # B = S*G*inv(G*G')
    iGG = np.linalg.inv(np.dot(G,G.T))
    # calculate the fit paraamters
    B = np.dot(np.dot(S,G.T),iGG)

    # calculate the resulting fit values, using all data points
    fit = np.dot(B,G)
    # subtract fit from the original
    residual = S-fit
    # put it back to original shape
    residual = np.reshape(residual,(xs,ys,zs,ts))

    # need to add the np.newaxis ref when slicing the arrays to deal with the dimensions with size=1 being dropped
    meanval =  np.dot(B[:,constant_index,np.newaxis],G[np.newaxis,constant_index,:])
    meanval = np.reshape(meanval,(xs,ys,zs,ts))

    # want to keep the result the same size as the original - pad the results
    # pad to replace the dropped values - replace with the time-series average
    npad = ((0,0),(0,0),(0,0),(ndrop,0))
    residual2 = np.pad(residual, npad, mode = 'mean')
    meanval2 = np.pad(meanval, npad, mode = 'mean')

    return residual2, meanval2


#----------------compile_data_sets------------------------------------------
# function to load nifti format data and organize it as needed for analysis
def compile_data_sets(DBname, dbnumlist, prefix, mode = 'group_concatenate', nvolmask = 2):
    # options are:
    #   with input data being size xs, ys, zs, ts   and N database numbers, from Npeople participants, in dbnumlist:
    #  1) group_concatenate:   concatenate across the entire group  - tsize  = N x ts
    #  2) group_concatenate_by_avg_person:  concatenate averaged data for each person  - tsize = Npeople x ts
    #  3) per_person_avg:  average data for each person  - tsize = ts, plus 5th dimension of size Npeople
    #  4) per_person_concatenate_runs:  concatenate data for each person  - tsize = Nruns per person x ts, plus 5th dimension of size Npeople
    #  5) group_average:  average all data - tsize = ts
    # the input option "nvolmask" is similar to dropping initial volumes, but instead they are masked by
    # replacing the initial time point values with replications of the one at timepoint = nvolmask, to
    # eliminate the influence of the initial volumes even though they are still represented

    # options_list = ['concatenate_group', 'group_concatenate_by_person_avg', 'avg_by_person', 'concatenate_by_person', 'group_average']
    options_list = ['group_average', 'group_concatenate', 'group_concatenate_by_avg_person', 'per_person_avg',
                  'per_person_concatenate_runs']
    # person_avg_options = ['group_concatenate_by_person_avg', 'avg_by_person', 'group_average']
    person_avg_options = ['group_concatenate_by_avg_person', 'per_person_avg', 'group_average']

    if mode in options_list:
        print('compile_data_sets:  DBname = ',DBname)
        filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, dbnumlist, prefix, mode = 'list')
        # group level
        group_divisor = 0
        for pnum in range(NP):
            print('compile_data_sets:  reading participant data ',pnum+1,' of ',NP)
            # per_person_level
            list1 = filename_list[pnum]
            divisor = 0
            for runnum, name in enumerate(list1):
                # read in the name, and do something with the data ...
                # if mode is group_concatenate_by_avg_person, per_person_avg, or group_average, then sum the data for now
                # otherwise, concatenate the data
                input_img = nib.load(name)
                input_data = input_img.get_data()
                xs,ys,zs,ts = input_data.shape
                # mask out the initial volumes, if wanted
                if nvolmask > 0:
                    for tt in range(nvolmask): input_data[:,:,:,tt] = input_data[:,:,:,nvolmask]

                if runnum == 0 and pnum == 0:
                    fixed_ts = ts    # set the number of volumes to be constant
                else:
                    if ts < fixed_ts:
                        mean_data = np.mean(input_data, axis=3)
                        temp_data = np.repeat(mean_data[:,:,:,np.newaxis],fixed_ts,axis=3)
                        temp_data[:,:,:,:ts] = input_data    # pad the missing data points with the
                                                            # time-series mean value for each voxel
                        input_data = copy.deepcopy(temp_data)
                    if ts > fixed_ts:
                        input_data = input_data[:,:,:,:fixed_ts]
                        ts = fixed_ts

                # # convert to signal change from the average----------------
                # if data have been cleaned they are already percent signal changes
                mean_data = np.mean(input_data, axis=3)
                mean_data = np.repeat(mean_data[:,:,:,np.newaxis],ts,axis=3)
                if prefix[0] == 'x':
                    input_data = input_data - mean_data
                else:
                    input_data = 100.0*(input_data - mean_data)/(mean_data + 1.0e-20)

                if runnum == 0:
                    person_data = copy.deepcopy(input_data)
                    divisor += 1
                else:
                    if mode in person_avg_options:
                        # average across the person
                        person_data += input_data
                        divisor += 1
                    else:
                        # concatenate across the person
                        person_data = np.concatenate((person_data,input_data), axis = 3)
            person_data = person_data/divisor # average

            # check that concatenated runs all have the same number of volumes
            if pnum == 0:
                big_fixed_ts = np.shape(person_data)[3]  # set the number of volumes to be constant
            else:
                ts = np.shape(person_data)[3]
                if ts < big_fixed_ts:
                    mean_data = np.mean(person_data, axis=3)
                    temp_data = np.repeat(mean_data[:, :, :, np.newaxis], big_fixed_ts, axis=3)
                    temp_data[:, :, :, :ts] = person_data  # pad the missing data points with the
                    # time-series mean value for each voxel
                    person_data = copy.deepcopy(temp_data)
                if ts > fixed_ts:
                    person_data = person_data[:, :, :, :big_fixed_ts]

            # group level
            options_list = ['group_average', 'group_concatenate', 'group_concatenate_by_avg_person', 'per_person_avg',
             'per_person_concatenate_runs']

            if pnum == 0:
                group_data = copy.deepcopy(person_data)
                group_divisor += 1
            else:
                if mode == 'group_average':
                    group_data += person_data
                    group_divisor += 1
                if mode == 'group_concatenate_by_avg_person':
                    group_data = np.concatenate((group_data,person_data),axis = 3)
                if mode == 'group_concatenate':
                    group_data = np.concatenate((group_data,person_data),axis = 3)
                if mode == 'per_person_avg':
                    if np.ndim(group_data) == 4:
                        group_data = np.concatenate((group_data[:,:,:,:,np.newaxis],person_data[:,:,:,:,np.newaxis]),axis = 4)
                    else:
                        group_data = np.concatenate((group_data,person_data[:,:,:,:,np.newaxis]),axis = 4)
                if mode == 'per_person_concatenate_runs':
                    # need to pad every person out to the same length  --> taken care of earlier now
                    # tsize_group = np.shape(group_data)[3]
                    # xs,ys,zs,tsize_person = np.shape(person_data)
                    # if tsize_person > tsize_group:
                    #     new_group_data = np.zeros((xs,ys,zs,tsize_person))
                    #     new_group_data[:,:,:,:tsize_group] = group_data
                    #     group_data = new_group_data
                    # if tsize_group > tsize_person:
                    #     new_person_data = np.zeros((xs,ys,zs,tsize_group))
                    #     new_person_data[:,:,:,:tsize_person] = person_data
                    #     person_data = new_person_data
                    if np.ndim(group_data) == 4:
                        group_data = np.concatenate((group_data[:,:,:,:,np.newaxis],person_data[:,:,:,:,np.newaxis]),axis = 4)
                    else:
                        group_data = np.concatenate((group_data,person_data[:,:,:,:,np.newaxis]),axis = 4)

        group_data = group_data/group_divisor
    else:
        print('compile_data_sets:  invalid mode selected: ', mode)
        group_data = 'None'

    return group_data



#----------------compile_basis_sets------------------------------------------
# function to load nifti format data and organize it as needed for analysis
def compile_basis_sets(DBname, dbnumlist, prefix, mode = 'group_concatenate', nvolmask = 2):
    # options are:
    #   with input data being size xs, ys, zs, ts   and N database numbers, from Npeople participants, in dbnumlist:
    #  1) group_concatenate:   concatenate across the entire group  - tsize  = N x ts
    #  2) group_concatenate_by_avg_person:  concatenate averaged data for each person  - tsize = Npeople x ts
    #  3) per_person_avg:  average data for each person  - tsize = ts, plus 5th dimension of size Npeople
    #  4) per_person_concatenate_runs:  concatenate data for each person  - tsize = Nruns per person x ts, plus 5th dimension of size Npeople
    #  5) group_average:  average all data - tsize = ts
    # the input option "nvolmask" is similar to dropping initial volumes, but instead they are masked by
    # replacing the initial time point values with replications of the one at timepoint = nvolmask, to
    # eliminate the influence of the initial volumes even though they are still represented
    xls = pd.ExcelFile(DBname)
    datarecord = pd.read_excel(xls, 'datarecord')

    # options_list = ['concatenate_group', 'group_concatenate_by_person_avg', 'avg_by_person', 'concatenate_by_person', 'group_average']
    options_list = ['group_average', 'group_concatenate', 'group_concatenate_by_avg_person', 'per_person_avg',
                  'per_person_concatenate_runs']
    # person_avg_options = ['group_concatenate_by_person_avg', 'avg_by_person', 'group_average']
    person_avg_options = ['group_concatenate_by_avg_person', 'per_person_avg', 'group_average']
    if mode in options_list:
        filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, dbnumlist, prefix, mode = 'list')
        # group level
        group_divisor = 0
        for pnum in range(NP):
            print('compile_basis_sets:  reading participant data ',pnum+1,' of ',NP)
            # per_person_level
            list1 = filename_list[pnum]
            dbnums = dbnum_person_list[pnum]
            divisor = 0
            for runnum, name in enumerate(list1):
                # read in the name, and do something with the data ...
                # if mode is group_concatenate_by_avg_person, per_person_avg, or group_average, then sum the data for now
                # otherwise, concatenate the data
                # input_img = nib.load(name)
                # input_data = input_img.get_data()

                # get the basis sets for each run - based on the nifti name
                pname, fname = os.path.split(name)
                fnameroot, ext = os.path.splitext(fname)
                # input_img = nib.load(niiname)
                # input_data = input_img.get_data()
                # affine = input_img.affine

                # load the maineffects basis sets
                paradigmdef, paradigm_names = pybasissets.read_maineffects(DBname, dbnums[runnum])

                # name of white matter confounds excel file ...
                seriesnumber = datarecord.loc[dbnums[runnum], 'seriesnumber']
                nametag = '_s{}'.format(seriesnumber)
                wmdata_xlname = os.path.join(pname, 'wmnoise'+nametag+'.xlsx')
                xls = pd.ExcelFile(wmdata_xlname)
                df1 = pd.read_excel(xls, 'wmnoise')
                wm1 = np.array(df1.loc[:, 'region1'])
                wm2 = np.array(df1.loc[:, 'region2'])
                wm3 = np.array(df1.loc[:, 'region3'])

                # name of motion confounds excel file ...
                # need the base name with no prefixes for the motiondata (calculated before other pre-processing steps are done)
                fnameroot2 = fnameroot.replace(prefix, '')
                motiondata_xlname = os.path.join(pname, 'motiondata'+nametag+'.xlsx')
                xls = pd.ExcelFile(motiondata_xlname)
                df1 = pd.read_excel(xls, 'motion_data')

                # check motion data format
                if ('dx' in df1.keys()) and ('dy' in df1.keys()) and ('dz' in df1.keys()):
                    templatetype = 'brain'
                    namelist = ['dx', 'dy', 'dz']
                else:
                    templatetype = 'notbrain'
                    namelist = ['dx1', 'dy1', 'dz1', 'dx2', 'dy2', 'dz2', 'dx3', 'dy3', 'dz3']

                for num, name in enumerate(namelist):
                    dp = df1.loc[:, name]
                    if num == 0:
                        dpvals = [dp]
                    else:
                        dpvals.append(dp)

                print('size of dpvals is {}'.format(np.shape(dpvals)))

                if np.ndim(paradigmdef) == 1:
                    paradigmdef = paradigmdef[np.newaxis,:]

                nc,ts = np.shape(paradigmdef)

                basis_set = np.concatenate( (paradigmdef,wm1[np.newaxis, :ts], wm2[np.newaxis, :ts], wm3[np.newaxis, :ts],
                                             np.array(dpvals)[:,:ts]), axis=0)

                # mask out the initial volumes, if wanted
                if nvolmask > 0:
                    for tt in range(nvolmask): basis_set[:,tt] = basis_set[:,nvolmask]

                if runnum == 0:
                    person_data = basis_set
                    divisor += 1
                else:
                    if mode in person_avg_options:
                        # average across the person
                        person_data += basis_set
                        divisor += 1
                    else:
                        # concatenate across the person
                        person_data = np.concatenate((person_data,basis_set), axis = 1)
            person_data = person_data/divisor # average

            # group level
            options_list = ['group_concatenate', 'group_concatenate_by_avg_person', 'per_person_avg',
                'per_person_concatenate_runs', 'group_average']

            if pnum == 0:
                group_basis_set = person_data
                group_divisor += 1
            else:
                if mode == 'group_average':
                    group_basis_set += person_data
                    group_divisor += 1
                if mode == 'group_concatenate_by_avg_person':
                    group_basis_set = np.concatenate((group_basis_set,person_data),axis = 1)
                if mode == 'group_concatenate':
                    group_basis_set = np.concatenate((group_basis_set,person_data),axis = 1)
                if mode == 'per_person_avg':
                    if np.ndim(group_basis_set) == 2:
                        group_basis_set = np.concatenate((group_basis_set[:,:,np.newaxis],person_data[:,:,np.newaxis]),axis = 2)
                    else:
                        group_basis_set = np.concatenate((group_basis_set,person_data[:,:,np.newaxis]),axis = 2)
                if mode == 'per_person_concatenate_runs':
                    # need to pad every person out to the same length
                    tsize_group = np.shape(group_basis_set)[1]
                    nd,tsize_person = np.shape(person_data)
                    if tsize_person > tsize_group:
                        new_group_data = np.zeros((nd,tsize_person))
                        new_group_data[:,:tsize_group] = group_basis_set
                        group_basis_set = new_group_data
                    if tsize_group > tsize_person:
                        new_person_data = np.zeros((nd,tsize_group))
                        new_person_data[:,:tsize_person] = person_data
                        person_data = new_person_data
                    if np.ndim(group_basis_set) == 2:
                        group_basis_set = np.concatenate((group_basis_set[:,:,np.newaxis],person_data[:,:,np.newaxis]),axis = 2)
                    else:
                        group_basis_set = np.concatenate((group_basis_set,person_data[:,:,np.newaxis]),axis = 2)

        group_basis_set = group_basis_set/group_divisor
    else:
        print('compile_basis_sets:  invalid mode selected: ', mode)
        group_basis_set = 'None'

    return group_basis_set, paradigm_names

