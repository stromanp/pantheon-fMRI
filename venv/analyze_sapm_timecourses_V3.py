# sys.path.append(r'C:\Users\Stroman\PycharmProjects\pantheon\venv')

import numpy as np
import matplotlib.pyplot as plt
import py2ndlevelanalysis
import copy
import pyclustering
import pydisplay
import time
import pysem
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
import scipy.stats as stats
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
from mpl_toolkits import mplot3d
import random
import draw_sapm_diagram2 as dsd2
import copy

from hmmlearn import hmm


def identify_states(data, initial_states):
    ndata, nvals = np.shape(data)
    nstates, nvals2 = np.shape(initial_states)
    assert nvals == nvals2, print('identify_states:   data and initial_states do not have consistent sizes')

    working_states = copy.deepcopy(initial_states)

    ssq = cost_function(data,working_states)   # initial ssq
    ssq_total = np.sum(ssq)


def cost_function(data,working_states,Lweight):
    ndata, nvals = np.shape(data)
    nstates, nvals2 = np.shape(working_states)
    ssq_record = np.zeros(ndata)
    wout_record = np.zeros((ndata,nstates))

    working_states_mean = np.mean(working_states,axis=1)
    working_states_zm = working_states - np.repeat(working_states_mean[:,np.newaxis],nvals2,axis=1)
    statecov = np.diag(working_states_zm @ working_states_zm.T)

    for nn in range(ndata):
        d = data[nn,:]
        d0 = d - np.mean(d)
        w = (d0 @ working_states_zm.T)/(statecov + 1.0e-10)
        fit = np.diag(w) @ working_states_zm
        d2 = np.repeat(d0[np.newaxis,:],nstates,axis=0)
        ssq = np.sum((fit-d2)**2,axis=1)
        x = np.argmin(ssq)
        # wout = np.zeros(nstates)
        # wout[x] = w[x]
        # wout_record[nn,:] = wout
        ssq_record[nn] = ssq[x]

    err = np.sum(ssq_record)
    cost = np.sum(np.abs(working_states))  # L1 regularization
    totalcost = err + Lweight * cost
    return totalcost, ssq_record



def gradients_for_statevalues(data, working_states, dval, Lweight):
    starttime = time.time()
    nstates,nvals = np.shape(working_states)
    totalcost, ssq_record = cost_function(data,working_states,Lweight)
    dstate_db = np.zeros((nstates,nvals))
    # for nn in range(nstates*nvals):
    for aa in range(nstates):
        for bb in range(nvals):
    #     aa = nn // nvals
    #     bb = nn % nvals
            temp_states = copy.deepcopy(working_states)
            temp_states[aa,bb] += dval
            tempcost, ssq_record = cost_function(data, temp_states, Lweight)
            dstate_db[aa,bb] = (tempcost - totalcost)/dval
    endtime = time.time()
    print('gradients_for_statevalues:  time elapsed = {:.1f} sec'.format(endtime-starttime))
    return dstate_db

