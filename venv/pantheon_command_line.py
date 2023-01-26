
# import necessary modules
import numpy as np
import pysapm
#
# import tkinter as tk
# import tkinter.filedialog as tkf
# from tkinter import ttk
# import os
# import re
# import pandas as pd
# import dicom_conversion
# import time
# import pynormalization
# from PIL import Image, ImageTk
# import load_templates
# import nibabel as nib
# import matplotlib.pyplot as plt
# import matplotlib
# import image_operations_3D as i3d
# import pypreprocess
# import openpyxl
# import pydatabase
# import GLMfit
# import py_fmristats
# import configparser as cp
# import pyclustering
# from scipy import stats
# from scipy import ndimage
# import pydisplay
# import pysem
# import py2ndlevelanalysis
# import copy
# import math
# import pybrainregistration
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# matplotlib.use('TkAgg')   # explicitly set this - it might help with displaying figures in different environments

def SAPM_cluster_search_commandline(search_data_file, nprocessors, samplesplit,samplestart, timepoint = 'all', epoch = 'all'):
    # this is meant to be run from the command line - to make use of parallel processing
    print('Running SAPM_cluster_search with parallel processing')
    search_data = np.load(search_data_file, allow_pickle=True).flat[0]

    outputdir = search_data['SAPMresultsdir']
    SEMresultsname = search_data['SAPMresultsname']
    SEMparametersname = search_data['SAPMparamsname']
    networkfile = search_data['networkmodel']
    DBname = search_data['DBname']
    regiondataname = search_data['SAPMregionname']
    clusterdataname = search_data['SAPMclustername']
    initial_clusters = search_data['initial_clusters']

    best_clusters = pysapm.SAPM_cluster_search(search_data['SAPMresultsdir'], search_data['SAPMresultsname'],
                                               search_data['SAPMparamsname'],
                                               search_data['networkmodel'], search_data['DBname'],
                                               search_data['SAPMregionname'],
                                               search_data['SAPMclustername'], nprocessors, samplesplit,samplestart,
                                               search_data['initial_clusters'], timepoint, epoch)
    print('best clusters appear to be: {}'.format(best_clusters))



def SAPM_cluster_search_commandline2(search_data_file, nprocessors, samplesplit,samplestart, timepoint = 'all', epoch = 'all'):
    # this is meant to be run from the command line - to make use of parallel processing
    # search_data_file = r'E:/SAPMresults_Dec2022\SAPM_search_parameters.npy'
    # nprocessors = 1
    # samplesplit = 1
    # samplestart = 0
    # timepoint = 'all'
    # epoch = 'all'

    print('Running SAPM_cluster_search with parallel processing')
    search_data = np.load(search_data_file, allow_pickle=True).flat[0]

    outputdir = search_data['SAPMresultsdir']
    SEMresultsname = search_data['SAPMresultsname']
    SEMparametersname = search_data['SAPMparamsname']
    networkfile = search_data['networkmodel']
    DBname = search_data['DBname']
    regiondataname = search_data['SAPMregionname']
    clusterdataname = search_data['SAPMclustername']
    initial_clusters = search_data['initial_clusters']

    best_clusters = pysapm.SAPM_cluster_search2b(search_data['SAPMresultsdir'], search_data['SAPMresultsname'],
                                               search_data['SAPMparamsname'],
                                               search_data['networkmodel'], search_data['DBname'],
                                               search_data['SAPMregionname'],
                                               search_data['SAPMclustername'], nprocessors, samplesplit,samplestart,
                                               search_data['initial_clusters'], timepoint, epoch)

    # outputdir = search_data['SAPMresultsdir']
    # SAPMresultsname = search_data['SAPMresultsname']
    # SAPMparametersname = search_data['SAPMparamsname']
    # networkfile = search_data['networkmodel']
    # DBname = search_data['DBname']
    # regiondataname = search_data['SAPMregionname']
    # clusterdataname = search_data['SAPMclustername']
    # nprocessors = 1
    # samplesplit = 1
    # samplestart=0
    # initial_clusters=[]
    # timepoint='all'
    # epoch='all'
    # betascale=0.0



    print('best clusters appear to be: {}'.format(best_clusters))
