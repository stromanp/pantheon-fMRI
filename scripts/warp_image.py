# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 08:07:50 2018

@author: Stroman
"""
def warp_image(input_image, mapX, mapY, mapZ):
    import os
    import numpy as np
    import nibabel as nib
    import dicom2nifti
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from numpy import linspace, zeros, array

    output_size = mapX.shape
    output_image = zeros(output_size)
    # interpolate points in output_image
    
#    data = input_image.get_data()
    data = input_image
#    hdr = input_image.header
    # need to get the input image position parameters, and return the output image position parameters
#    input_image.get_data_dtype()
#    hdr.get_xyzt_units()
#    input_image.affine
#    print(hdr)
#    hdr['pixdim']
    
    #
    data = data.astype(float)
    xs,ys,zs = data.shape
    dx = np.roll(data,-1,axis=0)-data
    dy = np.roll(data,-1,axis=1)-data
    dz = np.roll(data,-1,axis=2)-data
    dxr = np.reshape(dx,np.prod(dx.shape))
    dyr = np.reshape(dy,np.prod(dy.shape))
    dzr = np.reshape(dz,np.prod(dz.shape))
                     
    mapXr = np.reshape(mapX, np.prod(mapX.shape))
    mapYr = np.reshape(mapY, np.prod(mapY.shape))
    mapZr = np.reshape(mapZ, np.prod(mapZ.shape))
    mapXr0 = np.floor(mapXr)
    mapYr0 = np.floor(mapYr)
    mapZr0 = np.floor(mapZr)
    x1 = mapXr-mapXr0
    y1 = mapYr-mapYr0
    z1 = mapZr-mapZr0

    check1 = mapXr0 >= 0
    check2 = mapYr0 >= 0
    check3 = mapZr0 >= 0
    check4 = mapXr0 < xs
    check5 = mapYr0 < ys
    check6 = mapZr0 < zs
    checkall = check1*check2*check3*check4*check5*check6

    ii = zs*ys*mapXr0 + zs*mapYr0 + mapZr0
    ii = ii.astype(int)
    ii = ii*checkall
    datar = np.reshape(data,np.prod(data.shape))
    imager = datar[ii] + x1*dxr[ii] + y1*dyr[ii] + z1*dzr[ii]

    output_image_nearest = np.reshape(datar[ii],output_size)
    output_image_linear = np.reshape(imager,output_size)
                    
    return output_image_nearest, output_image_linear