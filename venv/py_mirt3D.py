# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:04:28 2020

@author: stroman
"""
#
#% MIRT3D_REGISTER The main function for non-rigid registration 
#% of a single pair of 3D images using cubic B-spline
#% based transformation parametrization. 
#%
#%   Input
#%   ------------------ 
#%   refim       Reference (or fixed) 2D image with intensities [0..1]
#%   im          Source (or float) 2D image to be deformed with intensities [0..1]
#%               If you want to exlude certain image parts from the
#%               computation of the similarity measure, set those parts as
#%               NaNs (masking).
#%
#%   main        a structure of main MIRT options
#%
#%       .similarity=['SSD','CC','SAD','RC', 'CD2', 'MS','MI'] (default SSD) 
#%                    SSD - Sum of squared differences 
#%                    CC  - Correlation Coefficient 
#%                    SAD - Sum of absolute differences
#%                    RC -  Residual Complexity (monomodal, nonstationary slow-varying intensity distortions)
#%                    CD2, MS - Ultrasound similarity measures
#%                    MI - Mutual Inormation (multimodal)
#%               Similarity measure. All current similarity measures are
#%               listed in mirt3D_similarity. You can easily add your own
#%               one there. If you do please email me and I'll include it in MIRT
#%               with full acknowledgment.
#%       .okno (default 16) mesh window size between the B-spline
#%               control points. The mesh cell is square. The smaller the
#%               window the more complex deformations are possible, but
#%               also more regularization (main.lambda) is required. 
#%       .subdivide (default 3) - a number of hierarchical levels. 
#%               E.g. for main.subdivide=3 the registration is carried sequentially
#%               at image size 2^(3-1)=4 times smaller, 2^(2-1)=2 times smaller and the 
#%               original size. The mesh window size remain the same for all levels.
#%       .lambda (default 0.01) - a regularization weight. A regularization
#%               is defined as Laplacian (or curvature) penalization of the
#%               displacements of B-spline control points. Set main.single=1 to see the mesh,
#%               if your mesh is getting too much deformed or even unnaturally folded,
#%               set lambda to higher values.(try [0..0.4]). See my thesis 5.7
#%       .alpha  (default 0.1) - a parameter of the similarity measure, for
#%               e.g. alpha value of the Residual Complexity (try [0.01..1]) or scaling (parameter D) in CD2 and MS.
#%       .single (default 0) show mesh deformation at each iteration
#%               0 - do not show, 1 or anything else - show.
#%       .ro ([0..0.99]) a correlation parameter of MS similarity measure
#%       .MIbins (default 64)  Number of bins to use for the MI similarity measure
#%
#%   optim       a structure of optimization options
#%
#%       .maxsteps (default 300) Maximum number of iterations. If at the
#%               final iterations, the similarity measure is still getting
#%               significantly decreased, you may need to set more maximum
#%               iterations.
#%       .fundif (default 1e-6) Tolerance, stopping criterion. 
#%       .gamma (default 1) Important parameter: Initial optimization step size.
#%               During optimization the optimizer is taking
#%               steps starting from (optim.gamma) to decrease
#%               the similarity measure. If the step is too big
#%               it will be adjusted to smaller as gamma=gamma*anneal (see
#%               below). During the registration look at the
#%               value of gamma, if it stays constant at an
#%               initial level for too long, it means your
#%               initial step is too small, and the optimization
#%               will take longer. Ideally gamma is constant
#%               during first several iterations and then slowly
#%               decreasing.
#%       .anneal (default 0.9) The multiplicative constant to update the
#%               step size
#%
#%   Output
#%   ------------------ 
#%   res         a structure of the resulting parameters:
#%
#%       .X      a 4D matrix of final B-spline control points positions
#%               The 4th dimension size is 3, it indicates the coordinate
#%               x,y and z.
#%       .okno   window size/spacing between the control points (equal to
#%               the initial main.okno).
#%   im_int      Deformed float image (result)
#%
#%           
#%
#%   Examples
#%   --------
#%
#%   See many detailed examples in the 'examples' folder.
#%
#%   See also mirt3D_registration, mirt2D_register
#
#% Copyright (C) 2007-2010 Andriy Myronenko (myron@csee.ogi.edu)
#% also see http://www.bme.ogi.edu/~myron/matlab/MIRT/
#%
#%     This file is part of the Medical Image Registration Toolbox (MIRT).
#%
#%     The source code is provided under the terms of the GNU General Public License as published by
#%     the Free Software Foundation version 2 of the License.

#function [res, im_int]=mirt3D_register(refim, im, main, optim)

import math
import copy
import numpy as np
import image_operations_3D as i3d
import time
from scipy.fftpack import dct as scipy_dct
from scipy.fftpack import idct as scipy_idct
import scipy.ndimage as nd


def py_dct2D(x):
    xt = scipy_dct(scipy_dct(x, axis = 0, norm="ortho"), axis = 1, norm="ortho")
    return xt

def py_idct2D(x):
    xt = scipy_idct(scipy_idct(x, axis = 1, norm="ortho"), axis = 0, norm="ortho")
    return xt

def py_dct3D(x):
    xt = scipy_dct(scipy_dct(scipy_dct(x, axis = 0, norm="ortho"), axis = 1, norm="ortho"), axis = 2, norm="ortho")
    return xt

def py_idct3D(x):
    xt = scipy_idct(scipy_idct(scipy_idct(x, axis = 2, norm="ortho"), axis = 1, norm="ortho"), axis = 0, norm="ortho")
    return xt


#function [main,optim]=mirt_check(main,optim,n)
def py_mirt_check(main,optim,n):
    # return main,optim
    
    #% Default parameters
    defmain = {'similarity':'ssd',   # similarity measure (SSD,SAD,CC,RC,MS,CD2,MI)
               'okno':16,           #  mesh window size
               'subdivide':3,     #  number of hierarchical levels
               'lambda':0.01,     # regularization weight, 0 for none
               'single':0,          # show the mesh deformation at each iteration
               'alpha':0.1,         # similarity measure parameter (e.g. for RC, MS)
               'ro':0.9,            # a parameter of MS similarity measure (the assumed correlation)
               'MIbins':64}         # Number of bins for the MI similarity measure
    
    defoptim = {'maxsteps':300,   # Maximum number of iterations
                'fundif':1e-6,    # Function tolerance stopping condition
                'gamma':1,        # Initial step size
                'anneal':0.9}       # Annealing rate
    
    # Check the input options and set the defaults
    if n<3:
        main['similarity']=defmain['similarity']  # similarity measure
        main['okno']=defmain['okno']           # mesh window size
        main['subdivide'] = defmain['subdivide']     # number of hierarchical levels
        main['lambda'] = defmain['lambda']     # regularization weight, 0 for none
        main['single']=defmain['single']          # show the mesh deformation at each iteration
        main['alpha']=defmain['alpha']        # similarity measure parameter
        main['ro']=defmain['ro']            # a parameter of MS similarity measure (the assumed correlation)
        main['MIbins']=defmain['MIbins']
    
    if n<4:
        optim['maxsteps'] = defoptim['maxsteps']
        optim['fundif'] = defoptim['fundif']
        optim['gamma'] = defoptim['gamma']
        optim['anneal']=defoptim['anneal']

    if not('similarity' in main.keys()) or not(main['similarity']): main['similarity'] = defmain['similarity']
    if not('okno' in main.keys()) or not(main['okno']): main['okno'] = defmain['okno']
    if not('subdivide' in main.keys()) or not(main['subdivide']): main['subdivide'] = defmain['subdivide']
    if not('lambda' in main.keys()) or not(main['lambda']): main['lambda'] = defmain['lambda']
    if not('single' in main.keys()) or not(main['single']): main['single'] = defmain['single']
    if not('alpha' in main.keys()) or not(main['alpha']): main['alpha'] = defmain['alpha']
    if not('ro' in main.keys()) or not(main['ro']): main['ro'] = defmain['ro']
    if not('MIbins' in main.keys()) or not(main['MIbins']): main['MIbins'] = defmain['MIbins']
    
    if not('maxsteps' in optim.keys()) or not(optim['maxsteps']): optim['maxsteps'] = defoptim['maxsteps']
    if not('fundif' in optim.keys()) or not(optim['fundif']): optim['fundif'] = defoptim['fundif']
    if not('gamma' in optim.keys()) or not(optim['gamma']): optim['gamma'] = defoptim['gamma']
    if not('anneal' in optim.keys()) or not(optim['anneal']): optim['anneal'] = defoptim['anneal']
    
    
    # some groupwise options
    if not('group' in main.keys()) or not(main['group']): main['group'] = 1
    if not('fundif' in optim.keys()) or not(optim['fundif']): optim['fundif'] = 0.1
    if not('maxcycle' in optim.keys()) or not(optim['maxcycle']): optim['maxcycle'] = 40
    
    main['cycle']=0
    main['volume']=0 
    return main,optim




#function Fi=mirt3D_F(okno)
def py_mirt3D_F(okno):
    # create the matrix of weights
    B=np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 0, 3, 0], [1, 4, 1, 0]])/6
#    % create 'okno' points in between of control points
    u=np.linspace(0,1,okno+1)
    u=u[:-1]
#    % create the polynomial matrix [4 x 4]
    T = np.vstack((u**3,u**2,u,np.ones(okno)))
#    % compute the precomputed matrix of B-spline basis functions in 1D
#    B=T*B;
    B=np.matmul(T.T,B)  # the T matrix is transposed to make the result the same as in matlab, but this might be a mistake
#    % do kronneker produce to create a 2D matrix
    Fi=np.kron(B,B)
#    % one more time to create a 3D precomputed matrix
    Fi=np.kron(B,Fi)
    return Fi




#function K=mirt3D_initK(siz)
def py_mirt3D_initK(siz):
    # return K
    
    #% extract the projected size
    M=siz[0]
    N=siz[1]
    L=siz[2]
    
    #% create the 1D arrays of eigenvalues in x,y,z directions
    #% create the 1D arrays of eigenvalues in x,y,z directions
    
    v1 = np.zeros(M)
    for m in range(M):
        v1[m] = ( math.cos(m*np.pi/M) )
    v2 = np.zeros(N)
    for n in range(N):
        v2[n] = ( math.cos(n*np.pi/N) )
    v3 = np.zeros(L)
    for l in range(L):
        v3[l] = ( math.cos(l*np.pi/L) )
        
    #% compute the 3D array with eigenvalues of the 3D Laplacian
    #K=repmat(v1,[1 N L])+repmat(v2,[M 1 L])+repmat(v3,[M N 1]);
    #K=2*(3-K);
    v1c = v1[:,np.newaxis,np.newaxis]
    v2c = v2[np.newaxis,:,np.newaxis]
    v3c = v3[np.newaxis,np.newaxis,:]
    K=np.tile(v1c,(1,N,L))+np.tile(v2c,(M,1,L))+np.tile(v3c,(M,N,1))
    K=2*(3-K)
    
    #% square is because we use Laplacian regularization
    #% If we use gradient-based regularization, the squaring is not required
    K=K**2
    return K


def py_mirt3D_subdivide2(X):
    # this is very much like py_mirt3D_subdivide, without the extra dimension
    # return Y
    mg, ng, kg, tmp = np.shape(X)
    
    x=X[:,:,:,0]
    y=X[:,:,:,1]
    z=X[:,:,:,2]
    
    xnew=np.zeros((mg, 2*ng-2, kg))
    ynew=np.zeros((mg, 2*ng-2, kg))
    znew=np.zeros((mg, 2*ng-2, kg))
    
    xfill=(x[:,:-1,:]+x[:,1:,:])/2
    yfill=(y[:,:-1,:]+y[:,1:,:])/2
    zfill=(z[:,:-1,:]+z[:,1:,:])/2

    print('py_mirt3D_subdivide2:  size of x:  {}'.format(np.shape(x)))
    print('py_mirt3D_subdivide2:  size of xfill:  {}'.format(np.shape(xfill)))
    
    for i in range(0,ng-1):
        # xnew[:,2*i:2*i+2,:] = np.stack((x[:,i,:], xfill[:,i,:]), axis = 1)
        # ynew[:,2*i:2*i+2,:] = np.stack((y[:,i,:], yfill[:,i,:]), axis = 1)
        # znew[:,2*i:2*i+2,:] = np.stack((z[:,i,:], zfill[:,i,:]), axis = 1)

        xnew[:,2*i,:] = x[:,i,:]
        xnew[:,2*i+1,:] = xfill[:,i,:]
        ynew[:,2*i,:] = y[:,i,:]
        ynew[:,2*i+1,:] = yfill[:,i,:]
        znew[:,2*i,:] = z[:,i,:]
        znew[:,2*i+1,:] = zfill[:,i,:]

    
    x=xnew[:,1:,:]
    y=ynew[:,1:,:] 
    z=znew[:,1:,:] 
    
    xnew=np.zeros((2*mg-2, 2*ng-3, kg))
    ynew=np.zeros((2*mg-2, 2*ng-3, kg))
    znew=np.zeros((2*mg-2, 2*ng-3, kg))
    
    xfill=(x[:-1,:,:]+x[1:,:,:])/2
    yfill=(y[:-1,:,:]+y[1:,:,:])/2
    zfill=(z[:-1,:,:]+z[1:,:,:])/2
    
#    for i=1:mg-1
    for i in range(0,mg-1):
        # xnew[2*i:2*i+2,:,:]=np.stack( (x[i,:,:], xfill[i,:,:]), axis = 0)
        # ynew[2*i:2*i+2,:,:]=np.stack( (y[i,:,:], yfill[i,:,:]), axis = 0)
        # znew[2*i:2*i+2,:,:]=np.stack( (z[i,:,:], zfill[i,:,:]), axis = 0)

        xnew[2*i,:,:] = x[i,:,:]
        xnew[2*i+1,:,:] = xfill[i,:,:]
        ynew[2*i,:,:] = y[i,:,:]
        ynew[2*i+1,:,:] = yfill[i,:,:]
        znew[2*i,:,:] = z[i,:,:]
        znew[2*i+1,:,:] = zfill[i,:,:]


    x=xnew[1:,:,:]
    y=ynew[1:,:,:]
    z=znew[1:,:,:]
    
    xnew=np.zeros((2*mg-3, 2*ng-3, 2*kg-2))
    ynew=np.zeros((2*mg-3, 2*ng-3, 2*kg-2))
    znew=np.zeros((2*mg-3, 2*ng-3, 2*kg-2))
    
    #    xfill=(x(:,:,1:end-1,:)+x(:,:,2:end,:))/2;
    xfill=(x[:,:,:-1]+x[:,:,1:])/2
    yfill=(y[:,:,:-1]+y[:,:,1:])/2
    zfill=(z[:,:,:-1]+z[:,:,1:])/2

    for i in range(0,kg-1):
        # xnew[:,:,2*i:2*i+2]=np.stack((x[:,:,i], xfill[:,:,i]), axis = 2)
        # ynew[:,:,2*i:2*i+2]=np.stack((y[:,:,i], yfill[:,:,i]), axis = 2)
        # znew[:,:,2*i:2*i+2]=np.stack((z[:,:,i], zfill[:,:,i]), axis = 2)

        xnew[:,:,2*i] = x[:,:,i]
        xnew[:,:,2*i+1] = xfill[:,:,i]
        ynew[:,:,2*i] = y[:,:,i]
        ynew[:,:,2*i+1] = yfill[:,:,i]
        znew[:,:,2*i] = z[:,:,i]
        znew[:,:,2*i+1] = zfill[:,:,i]
    
    x=xnew[:,:,1:]
    y=ynew[:,:,1:]
    z=znew[:,:,1:]
    
    Y=np.stack((x, y, z),axis=3)
    
    Y=2*Y-1

    print('py_mirt3D_subdivide2:  size of input X: {}'.format(np.shape(X)))
    print('py_mirt3D_subdivide2:  size of output Y: {}'.format(np.shape(Y)))

    return Y


#% MIRT3D_NODES2GRID  computes the dense transformation (positions of all image voxels)
#% from the current positions of B-spline control points
#%
#% Input
#% X - 5D array of B-spline control point positions. The first 3 dimensions
#% include the coordinates of B-spline control points in a particular
#% volume. The 4th dimension is of size 3, it indicates wheather it is the
#% X, Y or Z coordinate. The 5th dimension is time (volume number).
#%
#% F - is a precomputed matrix of B-spline basis function coefficients, see.
#% mirt3D_F.m file
#%
#% okno - is a spacing width between the B-spline control points
#%
#% Output
#% Xx,Xy,Xz - 3D matrices of positions of all image voxels computed from the
#% corresponding positions of B-spline control points
#
#% Copyright (C) 2007-2010 Andriy Myronenko (myron@csee.ogi.edu)
#% also see http://www.bme.ogi.edu/~myron/matlab/MIRT/

#function [Xx,Xy,Xz]=mirt3D_nodes2grid(X, F, okno)

def py_mirt3D_nodes2grid(X, F, okno):
    # return Xx,Xy,Xz

    #[mg,ng,kg, tmp]=size(X);
    mg,ng,kg,tmp=np.shape(X)

    Xx = np.zeros(( (mg-3)*okno, (ng-3)*okno, (kg-3)*okno))
    Xy = np.zeros(( (mg-3)*okno, (ng-3)*okno, (kg-3)*okno))
    Xz = np.zeros(( (mg-3)*okno, (ng-3)*okno, (kg-3)*okno))
    for i in range(mg-3):
        for j in range(ng-3):
            for k in range(kg-3):
#                % define the indices of the voxels corresponding
#                % to the given 4x4x4 patch
            
#                % take the X coordinates of the current 4x4x4 patch of B-spline
#                % control points, rearrange in vector and multiply by the matrix
#                % of B-spline basis functions (F) to get the dense coordinates of 
#                % the voxels within the given patch
                tmp=X[i:i+4,j:j+4,k:k+4,0]

                tmp2 = np.matmul(F,np.reshape(tmp, (64,1), order = 'F'))
                Xx[i*okno:(i+1)*okno, j*okno:(j+1)*okno, k*okno:(k+1)*okno] = np.reshape(tmp2,(okno, okno, okno), order = 'F')  # , order = 'F'   
                
                # same for other axes:
                tmp=X[i:i+4,j:j+4,k:k+4,1]
                tmp2 = np.matmul(F,np.reshape(tmp, (64,1), order = 'F'))
                Xy[i*okno:(i+1)*okno, j*okno:(j+1)*okno, k*okno:(k+1)*okno]=np.reshape(tmp2,(okno, okno, okno), order = 'F')  # , order = 'F'
                
                tmp=X[i:i+4,j:j+4,k:k+4,2]
                tmp2 = np.matmul(F,np.reshape(tmp, (64,1), order = 'F'))
                Xz[i*okno:(i+1)*okno, j*okno:(j+1)*okno, k*okno:(k+1)*okno]=np.reshape(tmp2,(okno, okno, okno), order = 'F')  # , order = 'F'
                
    return Xx,Xy,Xz
            

#% MIRT3D_GRID2NODES  transforms the dense gradient to the
#% gradient at each node (B-spline control point).
#%
#% Input 
#% ddx, ddy, ddz - the gradient of similarity measure at each image voxel
#% in x, y and z direction respectively
#%
#% F - is a precomputed matrix of B-spline basis function coefficients, see.
#% mirt3D_F.m file
#%
#% okno - is a spacing width between the B-spline control points
#% mg, ng, kg - size of B-spline control points in x,y and z directions
#%
#% Output
#% Gr - a 4D array of the gradient of the similarity measure at B-spline
#% control point positions. The first 3 dimenstion (size = mg x ng x kg) is the organized control
#% points. The 4th dimesion (size 3) is the index of x,y or z component.
#
#% Copyright (C) 2007-2010 Andriy Myronenko (myron@csee.ogi.edu)
#% also see http://www.bme.ogi.edu/~myron/matlab/MIRT/
#% This file is a part of Medical Image Registration Toolbox (MIRT)

def py_mirt3D_grid2nodes(ddx, ddy, ddz, F, okno, mg, ng, kg):
    #% greate the 3D matrices wich includes
    #% the gradient of the similarity measure at
    #% each of the B-spline control points
    
    grx=np.zeros((mg,ng,kg))
    gry=np.zeros((mg,ng,kg))
    grz=np.zeros((mg,ng,kg))

    for i in range(mg-3):
        for j in range(ng-3):
            for k in range(kg-3):
#                % define the indices of the voxels corresponding
#                % to the given 4x4x4 patch
                
#                % extract the voxel-wise gradient (in x direction) of the similarity measure
#                % that correspond to the given 4x4x4 patch of B-spline cotrol
#                % points. 

#                % multiply the voxel-wise gradient by the transpose matrix of
#                % precomputed B-spline basis functions and accumulate into node-wise (B-spline)
#                % gradient. Accumulation is because some B-spline control
#                % points are shared between the patches
#                grx(i:i+3,j:j+3,k:k+3,1)=grx(i:i+3,j:j+3,k:k+3,1)+reshape(F'*tmp(:),[4 4 4]);
           
                tmp=ddx[i*okno:(i+1)*okno, j*okno:(j+1)*okno, k*okno:(k+1)*okno]
                tmp2 = np.matmul(F.T,np.reshape(tmp, (okno**3,1))) 
                grx[i:i+4,j:j+4,k:k+4] = grx[i:i+4,j:j+4,k:k+4] + np.reshape(tmp2,(4, 4, 4)) 
                
#               % do the same thing for y and z coordinates
                tmp=ddy[i*okno:(i+1)*okno, j*okno:(j+1)*okno, k*okno:(k+1)*okno]
                tmp2 = np.matmul(F.T,np.reshape(tmp, (okno**3,1)))
                gry[i:i+4,j:j+4,k:k+4] = gry[i:i+4,j:j+4,k:k+4] + np.reshape(tmp2,(4, 4, 4)) 
                
                tmp=ddz[i*okno:(i+1)*okno, j*okno:(j+1)*okno, k*okno:(k+1)*okno]
                tmp2 = np.matmul(F.T,np.reshape(tmp, (okno**3,1)))
                grz[i:i+4,j:j+4,k:k+4] = grz[i:i+4,j:j+4,k:k+4] + np.reshape(tmp2,(4, 4, 4)) 
                
#    % concatinate into a single 4D array
    Gr=np.stack([grx,gry,grz],axis=3)
    return Gr



#% mirt3D_similarity  The function computes the current similarity measure
#% value and its dense (voxel-wise) gradients
#
#% Copyright (C) 2007-2010 Andriy Myronenko (myron@csee.ogi.edu)
#% also see http://www.bme.ogi.edu/~myron/matlab/MIRT/
#%
#% This file is part of the Medical Image Registration Toolbox (MIRT).
#function [f,ddx,ddy, ddz,imsmall]=mirt3D_similarity(main, Xx, Xy, Xz)
def py_mirt3D_similarity(main, Xx, Xy, Xz):
    #% interpolate image and its gradients simultaneously
    #% main.imsmall is a 4D array where
    #% main.imsmall(:,:,:,1) - is the float image to be deformed
    #% main.imsmall(:,:,:,2:4) - its x,y,z gradient
    
    # im_int = i3d.warp_image(main['imsmall'][:,:,:,0], Xx, Xy, Xz)
    im_int = i3d.warp_image_ignorenan(main['imsmall'][:,:,:,0], Xx, Xy, Xz)

#    nanvals = np.isnan(im_int).sum()
#    print('py_mirt3D_similarity   im_int has ',nanvals,' nan vals and ',np.size(im_int),' vals in total')

    # instead of warping the gradient fields, recalculate them
    gx_int, gy_int, gz_int=np.gradient(im_int)

#    % isolate the interpolated 3D image
#    imsmall=im_int(:,:,:,1); 
    imsmall=copy.deepcopy(im_int) 

#% Compute the similarity function value (f) and its gradient (dd)
#switch lower(main.similarity)
    condition_found = False
#   % sum of squared differences
    if main['similarity'] == 'ssd':
        dd=imsmall-main['refimsmall']
        f=np.nansum(dd**2)/2
        condition_found = True
        
#   % sum of absolute differences       
#   case 'sad' 
    if main['similarity'] == 'sad':
        dd=imsmall-main['refimsmall']
        f=np.nansum(np.sqrt(dd**2 + 1e-10))
        dd=dd/np.sqrt(dd**2+1e-10)    
        condition_found = True 
    
#   % correlation coefficient     
#   case 'cc' 
    if main['similarity'] == 'cc':
       
        finitevals = np.isfinite(main['refimsmall']).sum()
        SJ=main['refimsmall']-np.nansum(main['refimsmall'])/finitevals
        finitevals = np.isfinite(imsmall).sum()
        SI=imsmall-np.nansum(imsmall)/finitevals
        
        a = np.nansum(imsmall*SJ)/np.nansum(imsmall*SI)
        f=-a*np.nansum(imsmall*SJ)
        dd=-2*(a*SJ-(a**2)*SI)     
        condition_found = True

#   % Residual Complexity: A. Myronenko, X. Song: "Image Registration by
#   % Minimization of Residual Complexity.", CVPR'09
    if main['similarity'] == 'rc':
        rbig=imsmall-main['refimsmall']
#        [y,x]=find_imagebox(rbig); r=rbig(y,x);
#        r(isnan(r))=nanmean(r(:));
        x0,x1,y0,y1 = py_find_imagebox2d(rbig)
        r=rbig[x0:x1,y0:y1]
        r[np.where(np.isnan(r))]=np.nanmean(r)
        
#        Qr=mirt_dctn(r);
       # implement 2D DCT
        Qr = py_dct2D(r) 
        Li=Qr**2+main['alpha']

        f=0.5*sum(math.log(Li/main['alpha']))

#        r=mirt_idctn(Qr./Li);
        r = py_idct2D(Qr/Li)
        
        dd=np.zeros(np.shape(rbig)) 
        dd[x0:x1,y0:y1]=r
     
#   % CD2 similarity measure: Cohen, B., Dinstein, I.: New maximum likelihood motion estimation schemes for
#   % noisy ultrasound images. Pattern Recognition 35(2),2002
#        f=(imsmall-main.refimsmall)/main.alpha;
#        dd=2*tanh(f);
#        f=2*nansum(log(cosh(f(:))));
        
        f=(imsmall-main['refimsmall'])/main['alpha']
        dd=2*math.tanh(f)
        f=2*np.nansum(math.log(math.cosh(f)))
        condition_found = True
        
#   % MS similarity measure: Myronenko A., Song X., Sahn, D. J. "Maximum Likelihood Motion Estimation
#   % in 3D Echocardiography through Non-rigid Registration in Spherical Coordinates.", FIMH 2009
    if main['similarity'] == 'ms':
        
        f=(imsmall-main['refimsmall'])/main['alpha']
        coshd2=math.cosh(f)**2
        dd=math.tanh(f)*(2*coshd2+main['ro'])/(coshd2-main['ro'])
        f=np.nansum(1.5*math.log(coshd2-main['ro'])-0.5*math.log(coshd2)) 
        condition_found = True
        
        
#  % (minus) Mutual Information: Paul A. Viola "Alignment by Maximization of Mutual Information"   
#    if main['similarity'] == 'mi':
##        % MI computation is somewhat more involved, so let's compute it in separate function
##        [f, dd]=mirt_MI(main.refimsmall,imsmall,main.MIbins);
#        
#        % MI computation is somewhat more involved, so let's compute it in separate function
#        [f, dd]=mirt_MI(main.refimsmall,imsmall,main.MIbins);

    if not(condition_found):
        print('Similarity measure is wrong. Supported values are ssd, sad, cc, rc, ms')
#    % Multiply by interpolated image gradients
    
#    % Multiply by interpolated image gradients
    ddx=dd*gx_int
    ddx[np.isnan(ddx)]=0
    ddy=dd*gy_int
    ddy[np.isnan(ddy)]=0
    ddz=dd*gz_int
    ddz[np.isnan(ddz)]=0
    
    
#    print('py_mirt3D_similarity   min/max of ddx are: ',np.min(ddx), np.max(ddx))
#    print('py_mirt3D_similarity   min/max of ddy are: ',np.min(ddy), np.max(ddy))
#    print('py_mirt3D_similarity   min/max of ddz are: ',np.min(ddz), np.max(ddz))
    
    
    return f,ddx,ddy,ddz,imsmall
    
    

def py_find_imagebox3d(im):
#    ind=find(~isnan(im));
#    [i,j,k]=ind2sub(size(im),ind);
    i,j,k=np.where(np.isfinite(im))
    
    n=0 #% extra border size to cut
#    y=min(i)+n:max(i)-n;
#    x=min(j)+n:max(j)-n;
#    z=min(k)+n:max(k)-n;
    
#    x = range(np.min(i)+n, np.max(i)-n)
#    y = range(np.min(j)+n, np.max(j)-n)
#    z = range(np.min(k)+n, np.max(k)-n)
    
    x0 = np.min(i)+n
    x1 = np.max(i)-n
    y0 = np.min(j)+n
    y1 = np.max(j)-n
    z0 = np.min(k)+n
    z1 = np.max(k)-n
    return x0,x1,y0,y1,z0,z1


def py_find_imagebox2d(im):
#    ind=find(~isnan(im));
#    [i,j,k]=ind2sub(size(im),ind);
    i,j=np.where(np.isfinite(im))
    
    n=0 #% extra border size to cut
#    y=min(i)+n:max(i)-n;
#    x=min(j)+n:max(j)-n;
#    z=min(k)+n:max(k)-n;
#    x = range(np.min(i)+n, np.max(i)-n)
#    y = range(np.min(j)+n, np.max(j)-n)
    
    x0 = np.min(i)+n
    x1 = np.max(i)-n
    y0 = np.min(j)+n
    y1 = np.max(j)-n
    
    return x0,x1,y0,y1



#% MIRT3D_grad computes the value of the similarity measure and its gradient
#
#% Copyright (C) 2007-2010 Andriy Myronenko (myron@csee.ogi.edu)
#% also see http://www.bme.ogi.edu/~myron/matlab/MIRT/
#%
#% This file is part of the Medical Image Registration Toolbox (MIRT).

def py_mirt3D_grad(X,  main):

#% find the dense transformation for a given position of B-spline control
#% points (X).
#[Xx,Xy,Xz]=mirt3D_nodes2grid(X, main.F, main.okno)
    Xx,Xy,Xz = py_mirt3D_nodes2grid(X, main['F'], main['okno'])

#% Compute the similarity function value (f) and its gradient (dd) at Xx, Xy
#% (densely)
#[f,ddx,ddy, ddz, imsmall]=mirt3D_similarity(main, Xx, Xy,Xz);
    f,ddx,ddy,ddz,imsmall = py_mirt3D_similarity(main, Xx, Xy,Xz)

#% Find the gradient at each B-spline control point
#Gr=mirt3D_grid2nodes(ddx, ddy, ddz, main.F, main.okno, main.mg, main.ng, main.kg);
    Gr = py_mirt3D_grid2nodes(ddx, ddy, ddz, main['F'], main['okno'], main['mg'], main['ng'], main['kg'])

    return f, Gr, imsmall
        


#
#% MIRT3D_REGSOLVE  The function updates the position of B-spline control
#% points based on the given gradient.
#%
#% Input
#% X - a 4D array of positions of B-spline control points in a single 3D image
#%     The 4th dimension size is 3, it indicates the coordinate x, y or z
#% T - a 4D array with gradient of the similarity measure at the B-spline
#%     control points
#%
#% Output
#% X - new coordinates of B-spline control points
#% f - the value of the regularization term
#% mode - a switch indicator, mode=1 if we simply want the value of the
#%        regularization term without computation of the new positions
#%
#% Copyright (C) 2007-2010 Andriy Myronenko (myron@csee.ogi.edu)
#% also see http://www.bme.ogi.edu/~myron/matlab/MIRT/
#%
#% This file is part of the Medical Image Registration Toolbox (MIRT).

def py_mirt3D_regsolve(X,T,main,optim,mode=0):
#    return [X,f]

#    % find the displacements, as the regularization is defined over the
#    % displacements of the B-spline control points
#    X=X-main.Xgrid;
    X=X-main['Xgrid'] 
    
    if mode > 0:       #% if we want simply compute the value of the regularization term
#        X=X-main['Xgrid']   
#        dux=mirt_dctn(X(:,:,:,1));
#        duy=mirt_dctn(X(:,:,:,2));
#        duz=mirt_dctn(X(:,:,:,3));
        
        dux=py_dct3D(X[:,:,:,0])
        duy=py_dct3D(X[:,:,:,1])
        duz=py_dct3D(X[:,:,:,2])
        
        f=0.5*main['lambda']*np.sum(main['K']*(dux**2+duy**2+duz**2))
#        X=X+main['Xgrid']
        
    else:
#        %% Update the node positions
#        X=X-main['Xgrid']
#        
#        % make step in the direction of the gradient
        X=X-T*optim['gamma']
#        
#        % solve the Laplace equation in 3D,
#        % through DCT. Here main.K is the precomputed
#        % matrix of the squared Laplacian eigenvalues
#        diax=main.lambda*optim.gamma*main.K+1;
#        dux=mirt_dctn(X(:,:,:,1))./diax;
#        duy=mirt_dctn(X(:,:,:,2))./diax;
#        duz=mirt_dctn(X(:,:,:,3))./diax;
        
        diax=main['lambda']*optim['gamma']*main['K']+1
        dux=py_dct3D(X[:,:,:,0])/diax
        duy=py_dct3D(X[:,:,:,1])/diax
        duz=py_dct3D(X[:,:,:,2])/diax
        
#        f=0.5*main.lambda*sum(main.K(:).*(dux(:).^2+duy(:).^2+duz(:).^2)); % compute the value of the regularization term
#        dux=mirt_idctn(dux);
#        duy=mirt_idctn(duy);
#        duz=mirt_idctn(duz);
        
        f=0.5*main['lambda']*np.sum(main['K']*(dux**2+duy**2+duz**2))  # % compute the value of the regularization term
        dux=py_idct3D(dux)
        duy=py_idct3D(duy)
        duz=py_idct3D(duz)
        
#        % concatinate the new displacement poistions in 4D array and add
#        % the offset (grid).
#        X=cat(4,dux,duy,duz)+main.Xgrid;
        X=np.stack([dux,duy,duz],axis=3)+main['Xgrid']
        
    return X,f



#function [X, im]=mirt3D_registration(X, main, optim)
def py_mirt3D_registration(X, main, optim):
# normalize the initial optimization step size
#% compute the gradient of the similarity measure
#[Xx,Xy,Xz]=mirt3D_nodes2grid(X, main.F, main.okno);
#[f, ddx, ddy,ddz]=mirt3D_similarity(main, Xx, Xy,Xz);
    
    Xx,Xy,Xz = py_mirt3D_nodes2grid(X, main['F'], main['okno'])
    f,ddx,ddy,ddz, imsmall = py_mirt3D_similarity(main, Xx, Xy, Xz)
    
    #% divide the initial optimization step size by the std of the gradient
    #% this somewhat normalizes the initial step size for different possible
    #% similarity measures used
    #optim.gamma=optim.gamma/std([ddx(:); ddy(:); ddz(:)],0);
    #clear ddx ddy ddz Xx Xy Xz;
    
#    print('line 933: np.std(np.stack([ddx, ddy, ddz],axis=3), ddof = 1) = ',np.std(np.stack([ddx, ddy, ddz],axis=3), ddof = 1) )

    # need to test if gamma value should be scaled to be in a reasonable range
    # scaling based on dd values seems to be inconsistent
    # optim['gamma']=optim['gamma']/np.std(np.stack([ddx, ddy, ddz],axis=3), ddof = 1)   # check this is right
    
    #% Start the main registration
    #% compute the objective function and its gradient
    #[fe, T, im]=mirt3D_grad(X,  main);              % compute the similarity measure and its gradient
    #[Xp, fr]=mirt3D_regsolve(X,T,main, optim, 1);   % compute the regularization term and update the transformation
    #f=fe+fr;                                        % compute the value of the total objective function (similarity + regularization)
    
    [fe, T, im] = py_mirt3D_grad(X,  main)
    Xp, fr = py_mirt3D_regsolve(X,T,main,optim,1)
    f=fe+fr

    #    fchange=optim.fundif+1; % some big number
    #    iter=0;
    fchange=optim['fundif']+1 #% some big number
    iter=0
    
    #    % do while the relative function difference is below the threshold and
    #    % the meximum number of iterations has not been reached

    while (abs(fchange)>optim['fundif']) & (iter<optim['maxsteps']):
                
    #            % find the new positions of B-spline control points,
    #            % given their currect positions (X) and gradient in (T)
    #            [Xp, fr]=mirt3D_regsolve(X,T,main, optim);
        Xp, fr = py_mirt3D_regsolve(X,T,main, optim)
        #            % compute new function value and new gradient
        #            [fe, Tp, imb]=mirt3D_grad(Xp,  main);
        fe, Tp, imb = py_mirt3D_grad(Xp,  main)
        fp=fe+fr
        
        #            % compute the relative objective function change
        fchange=(fp-f)/f
        
    #            % check if the step size is appropriate
        if ((fp-f)>0):
    #                % if the new objective function value does not decrease,
    #                % then reduce the optimization step size and
    #                % slightly increase the value of the objective function 
    #                % (this is an optimization heuristic to avoid some local minima)
            optim['gamma']*=optim['anneal']
            f += 0.001*abs(f)
            # print("    no reduction: {fval:5.3e} dif {fchangeval:5.3e} iter {iterval}  gamma = {gam:5.3e}".format(fval = f, fchangeval = fchange, iterval = iter, gam = optim['gamma']))
        else:
    #                % if the new objective function value decreased
    #                % then accept all the variable as true and show the progress
            X=copy.deepcopy(Xp)
            f=copy.deepcopy(fp)
            T=copy.deepcopy(Tp)
            im=copy.deepcopy(imb)
            
    #               % mesh_epsplot(X(:,:,round(end/2),1),X(:,:,round(end/2),2)); drawnow;
    #                % show progress
    #               disp([upper(main.similarity) ' ' num2str(f) ' dif ' num2str(fchange) ' sub ' num2str(main.level) ' cyc ' num2str(main.cycle) ' vol ' num2str(main.volume) ' iter ' num2str(iter) ' gamma = ' num2str(optim.gamma)]);
            print("{sim} {fval:5.3e} dif {fchangeval:5.3e} sub {mlevel} cyc {mcycle} vol {mvol} iter {iterval}  gamma = {gam:5.3e}".format(sim = main['similarity'], fval = f, fchangeval = fchange, mlevel = main['level'], mcycle = main['cycle'], mvol = main['volume'], iterval = iter, gam = optim['gamma']))
        iter+=1
    
    return X, im



def py_mirt3D_register(refim, im, main, optim):
    # return [res, im_int]
    verbose = False
    #% Check the input options and set the defaults
    #if nargin<2, error('mirt3D_register error! Not enough input parameters.'); end;
    #% Check the proper parameter initilaization
    #[main,optim]=mirt_check(main,optim,nargin);
    
    # print('py_mirt3D_register   min/max of im are: ',np.min(im), np.max(im))
    
    main,optim = py_mirt_check(main,optim,4)   # require 4 inputs to py_mirt3D_register, different than in matlab
    
    #% checking for the possible errors
    #if numel(size(im))~=numel(size(refim)), error('The dimensions of images are not the same.'); end;
    #if size(im,1)~=size(refim,1), error('The images must be of the same size.'); end;
    #if size(im,2)~=size(refim,2), error('The images must be of the same size.'); end;
    #if size(im,3)~=size(refim,3), error('The images must be of the same size.'); end;
    
    if verbose:
        start_time = time.time()
        print('MIRT: Starting 3D registration...')
        print('Using the following parameters:')
        print(main)
        print(optim)
    
    #% Original image size 
    dimen=np.shape(refim)
    #% Size at the smallest hierarchical level, when we resize to smaller
    M=np.ceil(np.array(dimen)/(2**(main['subdivide']-1)))
    M = M.astype('int')
    
#    nanvals = np.isnan(refim).sum()
#    print('py_mirt3D_register   refim has ',nanvals,' nan vals and ',np.size(refim),' vals in total')
    
    
    #% Generate B-splines mesh of control points equally spaced with main.okno spacing
    #% at the smallest hierarchical level (the smallest image size)
    #[x, y, z]=meshgrid(1-main.okno:main.okno:M(2)+2*main.okno, 1-main.okno:main.okno:M(1)+2*main.okno, 1-main.okno:main.okno:M(3)+2*main.okno);

    nsteps = np.ceil(M/main['okno'])+3
    topstep = main['okno']*(nsteps-2)
    x,y,z = np.mgrid[-main['okno']:topstep[0]:nsteps[0]*1j, -main['okno']:topstep[1]:nsteps[1]*1j, -main['okno']:topstep[2]:nsteps[2]*1j]

    #% the size of the B-spline mesh is in (main.mg, main.ng, main.kg) at the smallest
    #% hierarchival level.
    #[main.mg, main.ng, main.kg]=size(x); 
    main['mg'], main['ng'], main['kg'] = np.shape(x)
    
    #% new image size at the smallest hierarchical level
    #% this image size is equal or bigger than the original resized image at the
    #% smallest level. This size includes the image and the border of nans when the image size can not be
    #% exactly divided by integer number of control points.
    #main.siz=[(main.mg-3)*main.okno (main.ng-3)*main.okno (main.kg-3)*main.okno];
    main['siz']=[(main['mg']-3)*main['okno'], (main['ng']-3)*main['okno'], (main['kg']-3)*main['okno']]

    main['X']=np.stack([x,y,z],axis=3)  #% Put x, y and z control point positions into a single mg x ng x kg x 3  4Dmatrix
    main['Xgrid'] = copy.deepcopy(main['X'])    #% save the regular grid (used for regularization)

    #main.F=mirt3D_F(main.okno); % Init B-spline coefficients
    main['F']=py_mirt3D_F(main['okno'])   #% Init B-spline coefficients
    
    #% Leave only the image described by control points.
    #% At the original (large) image size calculate
    #% the size described my B-spline control points,
    #% which will be larger or equal to the original image size
    #% and initialize with NaNs. Then patch with the original images,
    #% so that the original images (refim, im) now include the border of NaNs.
    #% NaNs here signalizes the values to be ignored during the registration

    tmp = np.empty(np.multiply(2**(main['subdivide']-1),main['siz']))
    tmp[:] = np.nan
    
    #    print('main[siz] =', main['siz'])
    #    print('main[subdivide] =', main['subdivide'])
    #    
    #    print('dimen =', dimen)
    #    print('shape of refim is ',np.shape(refim))
    #    print('shape of tmp is ',np.shape(tmp))
    
    tmp[:dimen[0],:dimen[1],:dimen[2]]=copy.deepcopy(refim)
    refim = copy.deepcopy(tmp)
    tmp[:dimen[0],:dimen[1],:dimen[2]]=copy.deepcopy(im)
    im = copy.deepcopy(tmp)

    #% Go across sublevels
    #for level=1:main.subdivide
    for level in range(1, main['subdivide']+1):
    #    % update the size of B-spline mesh to twice bigger 
    #    % only do it for the 2nd or higher levels 
        if level>1:
            main['mg']=2*main['mg']-3 #% compute the new mesh size
            main['ng']=2*main['ng']-3
            main['kg']=2*main['kg']-3
        main['level']= copy.deepcopy(level)
        
        #    % current image size
        #    main.siz=[(main.mg-3)*main.okno (main.ng-3)*main.okno (main.kg-3)*main.okno];
        #    main.K=mirt3D_initK([main.mg main.ng main.kg]); % Init Laplacian eigenvalues (used for regularization)
        main['siz']=[(main['mg']-3)*main['okno'], (main['ng']-3)*main['okno'], (main['kg']-3)*main['okno']]
        main['K']=py_mirt3D_initK([main['mg'], main['ng'], main['kg']])  #% Init Laplacian eigenvalues (used for regularization)
        
        main['refimsmall'] = i3d.resize_3D(refim, main['siz'])
#        print('py_mirt3D_register   min/max of main[refimsmall]  are: ',np.min(main['refimsmall']), np.max(main['refimsmall'] ))

        b = np.where(np.isnan(main['refimsmall']))
        main['refimsmall'][b] = 0
        a = np.where(main['refimsmall'] < 0)
        main['refimsmall'][a] = 0
        a = np.where(main['refimsmall'] > 1)
        main['refimsmall'][a] = 1
        main['refimsmall'][b] = np.nan

        imsmall = i3d.resize_3D(im,main['siz'])
#        print('py_mirt3D_register   min/max of imsmall are: ',np.min(imsmall), np.max(imsmall))
        
        b = np.where(np.isnan(imsmall))
        imsmall[b] = 0
        a = np.where(imsmall < 0)
        imsmall[a] = 0
        a = np.where(imsmall > 1)
        imsmall[a] = 1
        imsmall[b] = np.nan

        # the definition of gradx and grady are flipped compared to the matlab version
        # because of the row-order vs column-order difference
        gradx, grady, gradz=np.gradient(imsmall)
        
#        nanvals = np.isnan(imsmall).sum()
#        print('py_mirt3D_register   imsmall has ',nanvals,' nan vals and ',np.size(imsmall),' vals in total')
#        nanvals = np.isnan(gradx).sum()
#        print('py_mirt3D_register   gradx has ',nanvals,' nan vals and ',np.size(gradx),' vals in total')
        
        #    main.imsmall=cat(4,imsmall, gradx,grady, gradz); % concatenate the image with its gradients
        main['imsmall']= np.stack([imsmall, gradx, grady, gradz],axis=3)
        
        #    % main.imsmall is a 4D matrix - a concatenation of the image and its x,
        #    % y and z gradients at the given hierarchical level. The reason to
        #    % concatinate them together is to save time later on image interpolation
        #    % using mirt3D_mexinterp. The image and its gradients have to be
        #    % interpolated at the same coordinates, which can be done more
        #    % efficiently then interpolating each of them individially.

    #    % a single level 3D non-rigid image registration
    #    [main.X, result]=mirt3D_registration(main.X, main, optim);
    
#        nanvals = np.isnan(main['X']).sum()
#        print('py_mirt3D_register  main[X] has ',nanvals,' nan vals and ',np.size(main['X']),' vals in total')
    
        main['X'], result = py_mirt3D_registration(main['X'], main, optim)
        
#        % if the sublevel is not last prepair for the next level
        if level<main['subdivide']:
            main['X']=py_mirt3D_subdivide2(main['X']);
            main['Xgrid']=py_mirt3D_subdivide2(main['Xgrid'])
            
            
#    % Prepare the output
    res = {'X': main['X'], 'okno': main['okno']}
#    res['X']=copy.deepcopy(main['X'])
#    res['okno']=copy.deepcopy(main['okno'])
    
#    % because we have appended the initial images with the border of NaNs during the
#    % registration, now we want to remove that border and get the result of the
#    % initial image size
#    im_int=zeros(dimen); [M,N,K]=size(result);
#    im_int(1:min(dimen(1),M),1:min(dimen(2),N),1:min(dimen(3),K))=result(1:min(dimen(1),M),1:min(dimen(2),N),1:min(dimen(3),K));
    
    im_int=np.zeros((dimen))
    M,N,K = np.shape(result)
    m = np.min([dimen[0], M])
    n = np.min([dimen[1], N])
    k = np.min([dimen[2], K])
    im_int[:m, :n, :k] = result[:m, :n, :k]
    
    if verbose:
        print('MIRT: 3D non-rigid registration is succesfully completed.')
        stop_time = time.time()
        run_time = stop_time-start_time
        print('{h} hours {m} minutes {s} seconds'.format(h = np.floor(run_time/3600), m = np.floor(np.mod(run_time/60, 60)), s = np.round(np.mod(run_time,60))))
    
    return res, im_int
    



# im=mirt3D_transform(refim, res)  applies the transformation
# stored in res structure (output of mirt3D_register) to the 3D image refim

# for the MATLAB version:
# Copyright (C) 2007-2010 Andriy Myronenko (myron@csee.ogi.edu)
# also see http://www.bme.ogi.edu/~myron/matlab/MIRT/
#
# This file is part of the Medical Image Registration Toolbox (MIRT).

#function im=mirt3D_transform(refim, res)
def py_mirt3D_transform(refim, res):
    # return im

    dimen=np.shape(refim)
    
    # Precompute the matrix B-spline basis functions
    F = py_mirt3D_F(res['okno'])
    
    # obtaine the position of all image voxels (Xx,Xy,Xz) from the positions
    # of B-spline control points (res.X)
    Xx,Xy,Xz = py_mirt3D_nodes2grid(res['X'], F, res['okno'])

    # interpolate the image refim at Xx, Xy, Xz
#    newim=mirt3D_mexinterp(refim, Xx, Xy,Xz); newim(isnan(newim))=0;
#    newim = nd.map_coordinates(refim, [Xx, Xy,Xz])   # original method
    
    newim = i3d.warp_image(refim, Xx, Xy, Xz)   # use this to be consistent with other functions

# cut the interpolated image size to the original size
# the image produced by B-splines has an additional black border,
# these lines remove that border.
    im=np.zeros(dimen)
    M,N,K=np.shape(newim)
    im[:np.min([dimen[0],M]),:np.min([dimen[1],N]),:np.min([dimen[2],K])] = newim[:np.min([dimen[0],M]), :np.min([dimen[1],N]), :np.min([dimen[2],K])]
    
    return im


