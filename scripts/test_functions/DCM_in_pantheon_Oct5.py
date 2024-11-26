#  attempt to replicate DCM as done in SPM

import numpy as np
import os
import matplotlib.pyplot as plt
import time
import nibabel as nib
import pandas as pd
import scipy

# setup DCM settings structure like in SPM

TR = 3.0
ROInames = ['C6RD', 'NRM', 'NGC', 'PAG']   # pick something for setup
xY = []
nR = len(ROInames)
for rr in range(nR):
	# get voxel coordinates for each region - do this later, for now this is just simulation
	coords = [0,0,0]
	xY.append({'coords':coords, })

tsize = 200
X0 = 0  # no idea what this is, something about the voxel data
tcdata = np.random.randn(nR,tsize)

paradigm_base = np.random.randn(1,tsize)   # stimulus function
stim_names = ['paradigm1']

Q = 0 # autocorrelation - figure this out later

Y = {'dt':TR, 'X0':X0, 'y':tcdata, 'name':ROInames, 'Q':Q}
U = {'dt':TR,'u':paradigm_base, 'name':stim_names, 'idx':[1,0]}   # guessing at idx
delays = np.tile(TR,(nR,1))/2  # equivalent to matlab function
Q = np.ones((1,nR))*tsize

TE = 0.030

options = {'nonlinear':0, 'two_state':0, 'stochastic':0, 'centre':0, 'nograph':1, 'maxnode':8, 'maxit':128, 'hidden':[], 'induced':0}

# connectivity matrices
a = [[1,1,1,0],[1,1,0,1],[1,0,1,1],[0,1,1,1]]
b0 = np.zeros((nR,nR))   # this example has two experimental factors that could modulate connectivity
b1 = np.array([[1,0,1,0], [0,1,0,1],[1,0,1,0],[0,1,0,1]])
b = np.concatenate((b0[:,:,np.newaxis],b1[:,:,np.newaxis]),axis=2)
c = np.array([[1,0],[1,0],[1,0],[1,0]])
d = np.zeros((nR,nR,0))

DCM = {'xY':xY, 'n':nR, 'v':tsize, 'U':U, 'Y':Y, 'delays':delays, 'TE':TE, 'options':options,'a':a,'b':b,'c':c,'d':d}


#----------------------------------------------------------------
# spm_dcm_estimate(DCM)    - need to replicate this function

U = DCM['U']	# exogenous inputs
Y = DCM['Y']	# responses
n = DCM['n']	# number of regions
v = DCM['v']	# number of scans (time points)

# data in Y are scaled to a maximum change of 4% for some reason
# data are already expressed as percent signal change when loaded
# data are detrended in SPM -- do this before input


if 'X0' not in Y.keys():    # X0 are models of confounds
	Y['X0'] = np.ones((tsize,1))
if np.ndim(Y['X0']) != 2:
	Y['X0'] = np.ones((tsize,1))

if 'hE' not in options.keys():
	options['hE'] = 6
if 'hC' not in options.keys():
	options['hC'] = 1/128


# if d has 3 dimensions then a nonlinear dcm is specified
# slice timing delays
#  IS = 'spm_int' means this is not setting up a nonlinear DCM
M = {'delays':np.ones((nR,1)),'TE':TE, 'IS':'spm_int'}


# priors and initial states
# spm_dcm_fmri_priors
pE, C, x, pC = spm_dcm_fmri_priors(a,b,c,d,options)

# % hyperpriors over precision - expectation and covariance
# %--------------------------------------------------------------------------
# hE      = scipy.sparse.csr_matrix((n,1), dtype=float).toarray() +options['hE']
hE      = np.zeros((n,1)) +options['hE']
hC      = np.eye(n)  * options['hC']
ii       = options['hidden']
hE[ii]   = -4
hC[ii,ii] = np.exp(-16)



# % complete model specification
# %--------------------------------------------------------------------------
if np.ndim(U['u']) < 2:
	m = 0
else:
	m = np.shape(U['u'])[1]

addM = {'f':'spm_fx_fmri', 'g':'spm_gx_fmri', 'x':x, 'pE':pE, 'pC':pC, 'hE':hE, 'hC':hC,
	 'm':m, 'n':np.prod(x.shape), 'l':np.shape(x)[0], 'N':64, 'dt':0.5, 'ns':v}
M.update(addM)



# % nonlinear system identification (nlsi)
# %==========================================================================
if not options['stochastic']:
    # % nonlinear system identification (Variational EM) - deterministic DCM
    # %----------------------------------------------------------------------
    # [Ep,Cp,Eh,F] = spm_nlsi_GN(M,U,Y);
    #
    # % predicted responses (y) and residuals (R)
    # %----------------------------------------------------------------------
    # y      = feval(M.IS,Ep,M,U);
    # R      = Y.y - y;
    # R      = R - Y.X0*spm_inv(Y.X0'*Y.X0)*(Y.X0'*R);
    # Ce     = exp(-Eh);

	Ep, Cp, Eh, F = spm_nlsi_GN(M, U, Y)


def spm_vec(X):
	vtype = type(X)

	# type can be int, float, tuple, list, dict, str, np.ndarray
	if isinstance(X,int) | isinstance(X,float) | isinstance(X,np.ndarray):
		vX = np.array(X).flatten()
	else:
		if isinstance(X,bool):
			vX = np.array(X).flatten()

	if isinstance(X, dict):
		vX = []
		f = X.keys()
		for nn, fname in enumerate(f):
			if nn == 0:
				vX = X[fname].flatten()
			else:
				vX = np.concatenate((vX, X[fname].flatten()), axis=0)

	return vX




def spm_dcm_fmri_priors(A,B,C,D,options):
	# converted from spm
	# % Returns the priors for a two-state DCM for fMRI.
	# % FORMAT:[pE,pC,x,vC] = spm_dcm_fmri_priors(A,B,C,D,options)
	# %
	# %   options.two_state:  (0 or 1) one or two states per region
	# %   options.stochastic: (0 or 1) exogenous or endogenous fluctuations
	# %   options.precision:           log precision on connection rates
	# %
	# % INPUT:
	# %    A,B,C,D - constraints on connections (1 - present, 0 - absent)
	# %
	# % OUTPUT:
	# %    pE     - prior expectations (connections and hemodynamic)
	# %    pC     - prior covariances  (connections and hemodynamic)
	# %    x      - prior (initial) states
	# %    vC     - prior variances    (in struct form)
	# %__________________________________________________________________________
	n = np.shape(A)[0]

	if options['two_state']:
		# x = scipy.sparse.csr_matrix((n,6), dtype=float).toarray()
		x = np.zeros((n,6))
		tempA = A - np.diag(np.diag(A))
		A = (tempA > 0)

		try:
			pA = np.exp(options['precision'])
		except:
			pA = 16

		# prior expectations and covariances
		peA = (A + np.eye(n))*32 -32
		peB = B*0
		peC = C*0
		peD = D*0

		# prior covariances
		if np.ndim(A) > 2:
			for ii in range(np.shape(A)[2]):
				pcA[:,:,ii] = A[:,:,ii]/pA + np.eye(n)/pA
		else:
			pcA = A/pA + np.eye(n)/pA

		pcB = B/4
		pcC = C*4
		pcD = D/4

		# excitatory proportion
		if options['backwards']:
			peA[:,:,1] = A*0
			pcA[:,:,1] = A/pA

	else:
		# one hidden state per node
		# x = scipy.sparse.csr_matrix((n,5), dtype=float).toarray()
		x = np.zeros((n,5))

		# precision of connections
		try:
			pA = np.exp(options['precision'])
		except:
			pA = 64

		try:
			dA = np.exp(options['decay'])
		except:
			dA = 1

		# prior expectations
		if np.ndim(A) == 1:
			A = (A > 0)
			peA = (A-1)*dA
		else:
			tempA = A - np.diag(np.diag(A))
			A = (tempA > 0)
			peA = A/128

		peB = B*0
		peC = C*0
		peD = D*0

		# prior covariances
		if np.ndim(A) == 1:
			pcA = copy.deepcopy(A)
		else:
			if np.ndim(A) > 2:
				for ii in range(np.shape(A)[2]):
					pcA[:,:,ii] = A[:,:,ii]/pA + np.eye(n)/pA
			else:
				pcA = A/pA + np.eye(n)/pA
		pcB = B
		pcC = C
		pcD = D

	pE = {'A':peA, 'B':peB, 'C':peC, 'D':peD}
	pC = {'A':pcA, 'B':pcB, 'C':pcC, 'D':pcD}

	# add hemodynamic priors
	pE['transit'] = np.zeros((n,1))
	pC['transit'] = np.zeros((n,1)) + 1/256

	pE['decay'] = np.zeros((1,1))
	pC['decay'] = np.zeros((1,1)) + 1/256

	pE['epsilon'] = np.zeros((1,1))
	pC['epsilon'] = np.zeros((1,1))+ 1/256

	if options['induced']:
		pE['A'] = np.zeros((2,1))
		pC['A'] = np.zeros((2,1)) + 1/64

		pE['B'] = np.zeros((2,1))
		pC['B'] = np.zeros((2,1)) + 1/64

		pE['C'] = np.zeros((1,n))
		pC['C'] = np.zeros((1,n)) + 1/64


	# prior covariance matrix
	# C = np.diag(spm_vec(pC))
	# vectorize this here instead ...
	# vX = []
	# f = pC.keys()
	# for nn, fname in enumerate(f):
	# 	if nn == 0:
	# 		vX = pC[fname].flatten()
	# 	else:
	# 		vX = np.concatenate((vX,pC[fname].flatten()),axis=0)

	vX = spm_vec(pC)
	C = np.diag(vX)

	return pE, C, x, pC




def spm_nlsi_GN(M, U, Y):
	# function to call is named in M['FS']
	# this version is reduced from the matlab version

	M['IS'] = 'spm_int'
	y = Y['y']
	# convert 'IS', 'f', 'g', and 'h' to function calls
	ns = np.shape(y)[0]
	ny = np.prod(np.shape(y))
	nr = ny/ns
	M['ns'] = ns

	dt = Y['dt']
	Q = Y['Q']
	try:
		nh = len(Q)
	except:
		nh = 1
	nq = ny/nh

	pC = M['pC']
	pE = M['pE']

	# confounds if specified
	try:
		nb = np.shape(Y['X0'])[0]
		nx = ny/nb
		dfdu = np.kron(np.eye(nx),Y['X0'])
	except:
		dfdu = np.zeros((ny,0))


	# hyperpriors - expectation
	try:
		hE = M['hE']
		if len(hE) != nh:
			hE += np.zeros((nh,1))
	except:
		hE = np.zeros((nh,1)) - np.log(np.var(spm_vec(y))) + 4
	h = hE

	# hyperpriors - covariance
	try:
		ihC = np.linalg.inv(M['hC'])
		if np.shape(ihC)[0] != nh:
			ihC = ihC * np.eye(nh)
	except:
		ihC = np.eye(nh) * np.exp(4)


	# unpack covariance
	pC = np.diag(spm_vec(pC))

	# dimension reduction of parameter space
	Us,Ss,Vs = np.linalg.svd(pC)
	V = copy.deepcopy(Us)
	nu = np.shape(dfdu)[1]
	np_var = np.shape(V)[1]
	ip = np.array(range(np_var))
	iu = np.array(range(nu)) + np_var

	# second-order moments (in reduced space)
	pC = V.T @ pC @ V
	uC = np.eye(nu)/1e-8
	# ipC = np.linalg.inv(np.concatenate((pC, np.diag(uC)), axis = 0))
	ipC = np.linalg.inv(pC)   # this wil probably work in most cases

# % initialize conditional density
# %--------------------------------------------------------------------------
	Eu  = np.linalg.inv(dfdu)*spm_vec(y)
	p   = np.concatenate((V.T @ (spm_vec(M['P']) - spm_vec(M['pE'])), Eu),axis=0)
	Ep   = spm_unvec(spm_vec(pE) + V @ p[ip], pE)


	return Ep, Cp, Eh, F
