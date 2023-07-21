# -*- coding: utf-8 -*-
"""
Created on Tue May  5 19:28:21 2020

@author: stroman
"""
import numpy as np
import scipy.ndimage as nd
from scipy import signal
import math
import nibabel as nib
import copy


def resize_3D(input_data, newsize, mode = 'nearest'):
    original_size = np.shape(input_data)
    matrix = [ [original_size[0]/newsize[0], 0, 0], 
              [0, original_size[1]/newsize[1], 0], 
              [0, 0, original_size[2]/newsize[2]] ]
    output_image = nd.affine_transform(input_data, matrix, offset=0.0,
            output_shape=newsize, order=1, mode=mode, cval=0.0, prefilter=True)
    return output_image


def resize_2D(input_data, newsize_input, mode = 'nearest'):
    original_size = np.shape(input_data)
    if np.ndim(newsize_input) == 0:
        newsize = np.floor(np.array(original_size)*newsize_input).astype(int)
    else:
        newsize = newsize_input

    matrix = [ [original_size[0]/newsize[0], 0],
              [0, original_size[1]/newsize[1]] ]
    output_image = nd.affine_transform(input_data, matrix, offset=0.0,
            output_shape=newsize, order=1, mode=mode, cval=0.0, prefilter=True)
    return output_image


def linear_translation(input_data, offsetvalue):
    original_size = np.shape(input_data)
    matrix = np.eye(3)
    output_image = nd.affine_transform(input_data, matrix, offset=offsetvalue,
            output_shape=original_size, order=1, mode='nearest', cval=0.0, prefilter=True)
    return output_image


def resize_3D_nearest(input_data, newsize):
    # resize 3D images, but return the nearest-neighbor result
    xs, ys, zs = np.shape(input_data)
    xo, yo, zo = np.mgrid[0:xs:newsize[0]*1j, 0:ys:newsize[1]*1j, 0:zs:newsize[2]*1j]
    mask_array = np.zeros((xs,ys,zs))
    a = np.where(np.isnan(input_data))
    mask_array[a] = 1   # create a mask for keeping track of the nan locations
    input_data[a] = 0   # remove the nans from the input data, before warping
    # do the same warping to the input and the mask
    output_data = warp_image_nearest(input_data, xo, yo, zo)
    warped_mask = warp_image_nearest(mask_array, xo, yo, zo)
    a = np.where(warped_mask > 0.5)
    output_data[a] = np.nan   # put back the nans
    return output_data



def rotate3D(input_image, input_angle, refpoint, axis):
    angle = input_angle*np.pi/180.
    
    xs, ys, zs = np.shape(input_image)
    xi, yi, zi = np.mgrid[range(xs), range(ys), range(zs)]   # coordinates of input points
        
    if axis == 0:  # rotate around 1st axis
        yo = (yi-refpoint[1])*math.cos(angle) - (zi-refpoint[2])*math.sin(angle) + refpoint[1]
        zo = (yi-refpoint[1])*math.sin(angle) + (zi-refpoint[2])*math.cos(angle) + refpoint[2]
        xo = xi
        
    if axis == 1:  # rotate around 2nd axis
        xo = (xi-refpoint[0])*math.cos(angle) - (zi-refpoint[2])*math.sin(angle) + refpoint[0]
        zo = (xi-refpoint[0])*math.sin(angle) + (zi-refpoint[2])*math.cos(angle) + refpoint[2]
        yo = yi
        
    if axis == 2:  # rotate around 3rd axis
        xo = (xi-refpoint[0])*math.cos(angle) - (yi-refpoint[1])*math.sin(angle) + refpoint[0]
        yo = (xi-refpoint[0])*math.sin(angle) + (yi-refpoint[1])*math.cos(angle) + refpoint[1]
        zo = zi
        
#    rotimage_nearest, rotimage_linear = warp_image(input_image, xo, yo, zo)
    rotimage = nd.map_coordinates(input_image, [xo, yo, zo])
    
    return rotimage


def warp_image(input_image, mapX, mapY, mapZ):
    # mask = np.zeros(np.shape(input_image))   # create a mask of nan values
    mask = np.where(np.isnan(input_image), 1, 0)
    input_image = np.where(np.isnan(input_image), 0, input_image)
    # a = np.where(np.isnan(input_image))
    # mask[a] = 1
    # input_image[a] = 0  # mask out the nans
    # do the same warping on the mask and the input image
    output_image = nd.map_coordinates(input_image, [mapX, mapY, mapZ], order = 3, prefilter = True)
    mapped_mask = nd.map_coordinates(mask, [mapX, mapY, mapZ], order = 3, prefilter = True)
    # a = np.where(mapped_mask > 0.5)
    # output_image[a] = np.nan   # put back the nans
    output_image = np.where(mapped_mask > 0.5, np.nan, output_image)   # put back the nans
    return output_image


def warp_image_fast(input_image, mapX, mapY, mapZ):
    output_image = nd.map_coordinates(input_image, [mapX, mapY, mapZ], order = 3, prefilter = True)
    return output_image


def warp_image_nearest(input_image, mapX, mapY, mapZ):
    # warp images, but return the nearest-neighbor result
    output_size = mapX.shape
    # interpolate points in output_image
    
    data = input_image
    
    data = data.astype(float)
    xs,ys,zs = data.shape
                     
    mapXr = np.reshape(mapX, np.prod(mapX.shape))
    mapYr = np.reshape(mapY, np.prod(mapY.shape))
    mapZr = np.reshape(mapZ, np.prod(mapZ.shape))
    mapXr0 = np.floor(mapXr)
    mapYr0 = np.floor(mapYr)
    mapZr0 = np.floor(mapZr)

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

    output_image_nearest = np.reshape(datar[ii],output_size)
                    
    return output_image_nearest



def convert_affine_matrices(input_image, original_affine, new_affine = np.eye(4), output_size = 0):
    
    if np.size(output_size) != np.ndim(input_image):  output_size = np.shape(input_image)
    xs, ys, zs = output_size[0], output_size[1], output_size[2]
    
    xi, yi, zi = np.mgrid[range(xs), range(ys), range(zs)]   # pixel coordinates of input points, over the range of output size
    coords = np.concatenate( (np.reshape(xi, (1,np.size(xi))), np.reshape(yi, (1,np.size(yi))), np.reshape(zi, (1,np.size(zi))), np.ones((1,np.size(xi))) ), axis = 0)
    
    # calcuate mapX, mapY, mapZ relative coordinates for input to map_coordinates to transform the input
    # like this:
    #output_image = nd.map_coordinates(input_image, [mapX, mapY, mapZ], order = 3, prefilter = True)
    
    # where is the first pixel, in the original image space?
#    P = original_affine@old_coords    - "old_coords" are the coords of where the new points were, in the original image
#    Pnew = new_affine@coords
#    original_affine@old_coords = new_affine@coords
    
    old_coords = np.linalg.inv(original_affine)@new_affine@coords
    mapX = np.reshape(old_coords[0,:], (xs,ys,zs))
    mapY = np.reshape(old_coords[1,:], (xs,ys,zs))
    mapZ = np.reshape(old_coords[2,:], (xs,ys,zs))
    
    output_image = nd.map_coordinates(input_image, [mapX, mapY, mapZ], order = 3, prefilter = True)
    return output_image


def convert_affine_matrices_nearest(input_image, original_affine, new_affine = np.eye(4), output_size = 0):
    
    if np.size(output_size) != np.ndim(input_image):  output_size = np.shape(input_image)
    xs, ys, zs = output_size[0], output_size[1], output_size[2]
    
    xi, yi, zi = np.mgrid[range(xs), range(ys), range(zs)]   # pixel coordinates of input points, over the range of output size
    coords = np.concatenate( (np.reshape(xi, (1,np.size(xi))), np.reshape(yi, (1,np.size(yi))), np.reshape(zi, (1,np.size(zi))), np.ones((1,np.size(xi))) ), axis = 0)
    
    # calcuate mapX, mapY, mapZ relative coordinates for input to map_coordinates to transform the input
    # like this:
    #output_image = nd.map_coordinates(input_image, [mapX, mapY, mapZ], order = 3, prefilter = True)
    
    # where is the first pixel, in the original image space?
#    P = original_affine@old_coords    - "old_coords" are the coords of where the new points were, in the original image
#    Pnew = new_affine@coords
#    original_affine@old_coords = new_affine@coords
    
    old_coords = np.linalg.inv(original_affine)@new_affine@coords
    mapX = np.reshape(old_coords[0,:], (xs,ys,zs))
    mapY = np.reshape(old_coords[1,:], (xs,ys,zs))
    mapZ = np.reshape(old_coords[2,:], (xs,ys,zs))
    
#    output_image = nd.map_coordinates(input_image, [mapX, mapY, mapZ], order = 3, prefilter = True)
    output_image = warp_image_nearest(input_image, mapX, mapY, mapZ)
    
    return output_image


# fft with origin shifted to the center
def fft(array):
    fft = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(array)))
    return fft

def ifft(array):
    ifft = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(array)))
    return ifft




# normxcorr3 from matlab-------------------------------------
def normxcorr3(image, kernel, shape = 'full'):
    # replicate the matlab version of normxcorr3

    # image = image - np.mean(image)
    # kernel = kernel - np.mean(kernel)

    szK = np.shape(kernel)
    szI = np.shape(image)

    if np.any(szK > szI):
        print('template must be smaller than image')

    pSzK = np.product(szK)

    # make the running - sum / integral - images of A and A ^ 2, which are
    # used to speed up the computation of the NCC denominator

    intImgI = integralImage(image, szK)
    intImgI2 = integralImage(image*image, szK)

    szOut = np.shape(intImgI)

    # compute the numerator of the NCC
    # emulate 3D correlation by rotating templates dimensions
    # in 3D frequency - domain correlation is MUCH faster than the spatial - domain
    # variety

    rotK = np.flip(np.flip(np.flip(kernel, axis=0), axis=1), axis=2) # this is rot90 in 3d
    # need to pad rotK with zeros, to make it size szOut
    rotKpadded = np.zeros(szOut)
    rotKpadded[:szK[0],:szK[1],:szK[2]] = rotK   # should the padding be symmetric?  Does this matter? (only changes the phase of the FT)
    fftRotK = np.fft.fftn(rotKpadded)   # need equivalent of matlab fftn with output size option --> pad the input first

    # input image is supposed to be have size szOut - make sure this is the case, or pad/crop it
    imagepadded = np.zeros(szOut)
    imagepadded[:szI[0],:szI[1],:szI[2]] = image
    fftI = np.fft.fftn(imagepadded)

    # corrKI = (ifft(fftI*fftRotK)).real
    corrKI = (np.fft.ifftn(fftI*fftRotK)).real
    num = (corrKI - intImgI*np.sum(kernel) / pSzK ) / (pSzK - 1)

    # compute the denominator of the NCC
    denom2 = (intImgI2 - (intImgI*intImgI) / pSzK) / (pSzK - 1)
    denom2 = np.where(denom2 < 0, 0, denom2)
    denomI = np.sqrt(denom2)
    denomK = np.std(kernel, ddof = 1)
    # denom = denomT * denomA   # in matlab this is matrix multiplication
    denom = np.dot(denomK,denomI)

    # compute the NCC
    tol = 1.0e-10
    C = num/(denom + tol)

    # replace the NaN( if any) with 0's
    zeroInd = np.where(denomI < tol)
    C[zeroInd] = 0

    if shape.lower() == 'same':
        szKp = np.floor((np.array(szK)-1)/2).astype('int')
        C = C[szKp[0]:szKp[0]+szI[0], szKp[1]:szKp[1]+szI[1], szKp[2]:szKp[2]+szI[2]]

    if shape.lower() == 'valid':
        C = C[szK[0]:-szK[0], szK[1]:-szK[1], szK[2]:-szK[2]]

    return C




def integralImage(A, szT):
    # return integralImageA
    # this is adapted from Matlab's normxcorr2
    szA = np.shape(A)

    B = np.zeros((np.array(szA) + 2*np.array(szT) - 1))   # larger version of A, padded with zeros around all sides

    # B(szT(1) + 1: szT(1) + szA(1), szT(2) + 1: szT(2) + szA(2), szT(3) + 1: szT(3) + szA(3) ) = A;
    B[szT[0]:(szT[0]+szA[0]), szT[1]:(szT[1]+szA[1]), szT[2]:(szT[2]+szA[2])] = A

    s = np.cumsum(B, axis=0)
    c = s[szT[0]:,:,:]-s[:-szT[0],:,:]
    s = np.cumsum(c, axis=1)
    c = s[:,szT[1]:,:]-s[:,:-szT[1],:]
    s = np.cumsum(c, axis=2)
    integralImageA = s[:,:, szT[2]:]-s[:,:,:-szT[2]]

    return integralImageA


#------------load_and_scale_nifti------------------------------------------------------------
# function to read in nifti image data and resize to 1 mm voxels because this comes up a lot
def load_and_scale_nifti(niiname):
    input_img = nib.load(niiname)
    input_data = input_img.get_fdata()
    affine = input_img.affine
    input_hdr = input_img.header

    if np.ndim(input_data) > 3:
        FOV = input_hdr['pixdim'][1:4] * input_hdr['dim'][1:4]
        # images must first be interpolated to 1 mm cubic voxels
        [xd, yd, zd, td] = np.shape(input_data)
        newsize = np.floor(FOV).astype('int')
        input_datar = np.zeros((newsize[0],newsize[1], newsize[2],td))
        for tt in range(td):
            input_datar[:,:,:,tt] = resize_3D(input_data[:, :, :, tt], newsize)
    else:
        FOV = input_hdr['pixdim'][1:4] * input_hdr['dim'][1:4]
        # images must first be interpolated to 1 mm cubic voxels
        [xd, yd, zd] = np.shape(input_data)
        newsize = np.floor(FOV).astype('int')
        input_datar = np.zeros((newsize[0],newsize[1], newsize[2]))
        input_datar[:,:,:] = resize_3D(input_data, newsize)
        td = 1

    # update the affine matrix
    voxel_sizes = np.linalg.norm(affine,axis=0)[:3]
    new_affine = affine
    new_affine[:,0] = affine[:,0]/voxel_sizes[0]
    new_affine[:,1] = affine[:,1]/voxel_sizes[1]
    new_affine[:,2] = affine[:,2]/voxel_sizes[2]
    return input_datar, new_affine

