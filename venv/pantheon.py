"""
pantheon.py
version 1.0.0

Created on Tue Apr 21 16:16:09 2020
Last edited  10:33 am Saturday, April 25, 2020

@author: Patrick W. Stroman, Queen's University, Kingston

This module organizes inputs to functions to load, pre-process, and analyze 
functional MRI data from the brainstem and spinal cord

Jobs to do:
1) Specify database file (done, excepting for creating new database file, and functions to read the database)
2) Specify database entry numbers to work on (done)
3) Conversion from DICOM to NIfTI format (done)
4) Calculation of normalization parameters (done)
5) Pre-processing of data (done)
6) Model-driven fit of predicted BOLD responses to voxel data (GLM)) (done, except for creating results figure)
7) Definition of region clusters, and extraction of cluster data
8) Data-driven connectivity analysis, using structural equation modeling (SEM), and advances soon to come ...)
9) Regression analysis of cluster data
10) Visualization of results

"""
# sys.path.append(r'C:\Users\Stroman\PycharmProjects\pantheon\venv')
# sys.path.append('/Users/brieanakeast/PycharmProjects/pantheon-fMRI/venv')

# import necessary modules 
import tkinter as tk
import tkinter.filedialog as tkf
from tkinter import ttk
import os
import numpy as np
import re
import pandas as pd
import dicom_conversion
import time
import pynormalization
from PIL import Image, ImageTk
import load_templates
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
import image_operations_3D as i3d
import pypreprocess
import openpyxl
import pydatabase
import GLMfit
import py_fmristats
import configparser as cp
import pyclustering
from scipy import stats
from scipy import ndimage
import pydisplay
import pysem
import pysapm
import py2ndlevelanalysis
import copy
import math
import pybrainregistration
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.use('TkAgg')   # explicitly set this - it might help with displaying figures in different environments
enable_sapm = True

# save some colors for consistent layout, and make them easy to change
# colours for Windows
if os.name == 'nt':
    fgcol1 = 'navy'
    fgcol2 = 'gold2'
    fgcol3 = 'firebrick4'
    bgcol = 'grey94'
    fgletter1 = 'white'
    fgletter2 = 'black'
    fgletter3 = 'white'
    widgetfont = "none 9 bold"
    widgetfont2 = "none 9 bold"
    labelfont = "none 9 bold"
    radiofont = "none 9"
    infofont = "none 8"
    widgetbg = 'SystemButtonFace'
    mainbg = 'SystemButtonFace'
else:
    # colours for Mac/Linux
    fgcol1 = 'red'
    fgcol2 = 'red'
    fgcol3 = 'red'
    bgcol = '#808080'
    fgletter1 = '#FFFFFF'
    fgletter2 = '#FFFFFF'
    fgletter3 = '#FFFFFF'
    widgetfont = "none 9 bold"
    widgetfont2 = "none 9 bold"
    labelfont = "none 9 bold"
    radiofont = "none 9"
    infofont = "none 9"
    widgetbg = '#808080'
    mainbg = '#808080'

bigbigbuttonsize = 21
bigbuttonsize = 14
smallbuttonsize = 9

# define a single place for saving setup parameters for ease of retrieving, updating, etc.
basedir = os.getcwd()
settingsfile = os.path.join(basedir,'base_settings_file.npy')

if os.path.isfile(settingsfile):
    print('name of the settings file is : ', settingsfile)
    settings = np.load(settingsfile, allow_pickle = True).flat[0]
    # set some defaults
    settings['GLMpvalue_unc'] = settings['GLMpvalue']
    settings['GRPanalysistype'] = 'Sig1'
else:
    settings = {'DBname':'none',
            'DBnum':'none',
            'DBname2': 'none',
            'DBnum2': 'none',
            'DBnumstring':'none',
            'NIbasename':'Series',
            'CRprefix':'',
            'NCparameters':[50,50,5,6,-10,20,-10,10],
            'NCsavename':'normdata',
            'coreg_choice':'Yes.',
            'slicetime_choice':'Yes.',
            'norm_choice':'Yes.',
            'smooth_choice':'Yes.',
            'define_choice':'Yes.',
            'clean_choice':'Yes.',
            'sliceorder':'Inc,Alt,Odd',
            'sliceaxis':0,
            'refslice':1,
            'smoothwidth':3,
            'GLM1_option':'group_average',
            'GLMprefix':'ptc',
            'GLMndrop':2,
            'GLMcontrast':[1,0],
            'GLMresultsdir':basedir,
            'GLMpvalue':0.05,
            'GLMpvalue_unc':0.05,
            'GLMvoxvolume':1.0,
            'networkmodel':'none',
            'CLprefix':'xptc',
            'CLclustername':'notdefined',
            'CLregionname':'notdefined',
            'CLresultsdir':basedir,
            'last_folder':basedir,
            'SEMprefix':'xptc',
            'SEMclustername':'notdefined',
            'SEMregionname':'notdefined',
            'SEMresultsdir':basedir,
            'SEMsavetag':'base',
            'SEMtimepoints':[11,18],
            'SEMepoch':7,
            'SEMresumerun':False,
            'SRoptionvalue':'unknown',
            'SRcovname':'unknown',
            'SRpvalue':0.05,
            'SRgroup':'unknown',
            'SRtargetregion':'unknown',
            'SRnametag':'unknown',
            'SRdrawfile':'unknown',
            'SRthresholdtext':'unknown',
            'SAPMBfile':'unknown',
            'GRPresultsname':'notdefined',
            'GRPresultsname2':'notdefined',
            'GRPcharacteristicscount':0,
            'GRPcharacteristicslist':[],
            'GRPcharacteristicsvalues':[],
            'GRPcharacteristicsvalues2':[],
            'GRPanalysistype':'undefined',
            'GRPdatafiletype1':0,
            'GRPdatafiletype2':0,
            'GRPpvalue':0.05,
            'NCBparameters':[(10000, 1000, 100),(3.0, 1.0, 0.0),(4, 2, 1)],
            'braintemplate':'avg152T2.nii',
            'SAPMcnums': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'SAPMresultsdir': '',
            'SAPMresultsname': '',
            'SAPMparamsname': '',
            'networkmodel': '',
            'SAPMclustername': '',
            'SAPMregionname': '',
            'SAPMtimepoint': 'all',
            'SAPMepoch': 'all',
            'SRoptionvalue':1,
            'SRcovname':'none',
            'SRpvalue':0.05,
            'SRgroup':'',
            'SRtargetregion':'none',
            'SRnametag':'',
            'SRdrawfile':'',
            'SRthresholdtext':'',
            'SAPMBfile':'',
            'SAPMsavetag':'',
            'SRresultsdir':'',
            'SRresultsname':'',
            'SRparamsname':'',
            'SAPMBsheet':'',
            'SAPMBcolumn':'',
            'SRcovvalue':'',
            'GLMbasisset':'',
            'GLMparadigmnames':'',
            'GLMdataname':'',
            'SAPMbetascale':'',
            'SRvariant':0,
            'DISPconndefnamefull':''}
np.save(settingsfile,settings)

# ------Create the Base Window that will hold everything, widgets, etc.---------------------
class mainspinalfmri_window:
    # defines the main window, and other windows for functions are defined in separate classes

    def __init__(self, master):
        # tk.Frame.__init__(self, master)
        self.master = master

        # need to initialize these here and use them later to save some display items
        self.photo1 = []
        self.photo2 = []

        # first create the frames that do not move
        self.frames = {}  # this is to keep a record of the frame names, so that one can be moved to the top when wanted
        # create an initial fixed frame with labels etc
        # The frame with pictures, labels, etc, is Fbase and is on the top row
        Fbase = tk.Frame(self.master, relief='raised', bd=5, highlightcolor=fgcol2, bg = mainbg)
        Fbase.grid(row=0, column=1)
        BaseFrame(Fbase, self)
        page_name = BaseFrame.__name__
        self.frames[page_name] = Fbase

        # The Optionsbase frome contains the buttons with the choice of which sections to show
        # and is on the left side of the base window, spanning two rows
        Optionsbase = tk.Frame(self.master, relief='raised', bd=5, highlightcolor=fgcol1)
        Optionsbase.grid(row=0, column=0, rowspan=2)
        OptionsFrame(Optionsbase, self)
        page_name = OptionsFrame.__name__
        self.frames[page_name] = Optionsbase

        # IMGbase = tk.Frame(self.master, relief='raised', bd=5, highlightcolor=fgcol1)
        # IMGbase.grid(row=1, column=1, sticky="nsew")
        # IMGFrame = IMGDispFrame(IMGbase, self)
        # page_name = IMGDispFrame.__name__
        # self.frames[page_name] = IMGbase
        # self.imgwindow = IMGFrame

        # The remaining frames are on the bottom of the main window, and are on top of each other
        # All of these frames are defined at the same time, but only the top one is visible
        DBbase = tk.Frame(self.master, relief='raised', bd=5, highlightcolor=fgcol1)
        DBbase.grid(row=1, column=1, sticky="nsew")
        self.DBhandle = DBFrame(DBbase, self)
        page_name = DBFrame.__name__
        self.frames[page_name] = DBbase

        NIbase = tk.Frame(self.master, relief='raised', bd=5, highlightcolor=fgcol1)
        NIbase.grid(row=1, column=1, sticky="nsew")
        NIFrame(NIbase, self)
        page_name = NIFrame.__name__
        self.frames[page_name] = NIbase

        PPbase = tk.Frame(self.master, relief='raised', bd=5, highlightcolor=fgcol1)
        PPbase.grid(row=1, column=1, sticky="nsew")
        PPFrame(PPbase, self)
        page_name = PPFrame.__name__
        self.frames[page_name] = PPbase

        NCbase = tk.Frame(self.master, relief='raised', bd=5, highlightcolor=fgcol1)
        NCbase.grid(row=1, column=1, sticky="nsew")
        NCFrame(NCbase, self)
        page_name = NCFrame.__name__
        self.frames[page_name] = NCbase

        NCbrainbase = tk.Frame(self.master, relief='raised', bd=5, highlightcolor=fgcol1)
        NCbrainbase.grid(row=1, column=1, sticky="nsew")
        NCbrainFrame(NCbrainbase, self)
        page_name = NCbrainFrame.__name__
        self.frames[page_name] = NCbrainbase

        GLMbase = tk.Frame(self.master, relief='raised', bd=5, highlightcolor=fgcol1)
        GLMbase.grid(row=1, column=1, sticky="nsew")
        GLMFrame(GLMbase, self)
        page_name = GLMFrame.__name__
        self.frames[page_name] = GLMbase

        CLbase = tk.Frame(self.master, relief='raised', bd=5, highlightcolor=fgcol1)
        CLbase.grid(row=1, column=1, sticky="nsew")
        CLFrame(CLbase, self)
        page_name = CLFrame.__name__
        self.frames[page_name] = CLbase
        self.CLbase = CLbase

        SEMbase = tk.Frame(self.master, relief='raised', bd=5, highlightcolor=fgcol1)
        SEMbase.grid(row=1, column=1, sticky="nsew")
        SEMFrame(SEMbase, self)
        page_name = SEMFrame.__name__
        self.frames[page_name] = SEMbase
        self.SEMbase = SEMbase

        SAPMbase = tk.Frame(self.master, relief='raised', bd=5, highlightcolor=fgcol1)
        SAPMbase.grid(row=1, column=1, sticky="nsew")
        SAPMFrame(SAPMbase, self)
        page_name = SAPMFrame.__name__
        self.frames[page_name] = SAPMbase
        self.SAPMbase = SAPMbase

        SAPMResultsbase = tk.Frame(self.master, relief='raised', bd=5, highlightcolor=fgcol1)
        SAPMResultsbase.grid(row=1, column=1, sticky="nsew")
        # SAPMResultsFrame(SAPMResultsbase, self)
        framerefSR = SAPMResultsFrame(SAPMResultsbase, self)
        page_name = SAPMResultsFrame.__name__
        self.frames[page_name] = SAPMResultsbase

        self.SRPlotFigure = framerefSR.SRPlotFigure
        self.SRPlotAx = framerefSR.SRPlotAx
        self.SRCanvas = framerefSR.SRCanvas

        GRPbase = tk.Frame(self.master, relief='raised', bd=5, highlightcolor=fgcol1)
        GRPbase.grid(row=1, column=1, sticky="nsew")
        GRPFrame(GRPbase, self)
        page_name = GRPFrame.__name__
        self.frames[page_name] = GRPbase

        DISPbase = tk.Frame(self.master, relief='raised', bd=5, highlightcolor=fgcol1)
        DISPbase.grid(row=1, column=1, sticky="nsew")
        DisplayFrame(DISPbase, self)
        page_name = DisplayFrame.__name__
        self.frames[page_name] = DISPbase

        DISPbase2 = tk.Frame(self.master, relief='raised', bd=5, highlightcolor=fgcol1)
        DISPbase2.grid(row=1, column=1, sticky="nsew")
        frameref = DisplayFrame2(DISPbase2, self)
        page_name = DisplayFrame2.__name__
        self.frames[page_name] = DISPbase2

        self.PlotFigure3 = frameref.PlotFigure3
        self.PlotAx3 = frameref.PlotAx3
        self.Canvas3 = frameref.Canvas3

        self.PlotFigure4 = frameref.PlotFigure4
        self.PlotAx4 = frameref.PlotAx4
        self.Canvas4 = frameref.Canvas4

        # start with the Database information frame on top
        self.show_frame('DBFrame')


    # Definition of the function to choose which frame is on top
    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        keylist = self.frames.keys()
        if page_name in keylist:
            frame = self.frames[page_name]
            frame.tkraise()

#--------------------BASE FRAME---------------------------------------------------------------
# Definition of the frame that holds pictures, labels, etc.
class BaseFrame:
    # initialize the values, keeping track of the frame this definition works on (parent), and 
    # also the main window containing that frame (controller)
    def __init__(self, parent, controller):
        parent.configure(relief='raised', bd = 5, highlightcolor = fgcol2)  # just defining some visual features
        self.parent = parent
        self.controller = controller
        
        # make objects in the base frame
        # load in a picture, for no good reason, and display it in the window to look nice :)
        photo1 = tk.PhotoImage(file = os.path.join(basedir,'queens_flag_small.gif')) # 'queens_flag2.gif'
        controller.photo1 = photo1   # need to keep a copy so it is not cleared from memory
        # put this figure, in the 1st row, 1st column, of a grid layout for the window
        # and make the background black
        self.P1=tk.Label(self.parent, image = photo1, bg='grey94', font = labelfont).grid(row=0, column=0, sticky = 'W')

        # load in another picture, because if one picture is good, two is better
        photo2 = tk.PhotoImage(file = os.path.join(basedir,'pantheon_logo_small.gif'))  # 'pantheon_logo2.gif'
        controller.photo2 = photo2   # need to keep a copy so it is not cleared from memory
        # put in another figure, for pure artistic value, in the 1st row, 2nd column, of a grid layout for the window
        # and make the background black
        self.P2=tk.Label(self.parent, image = photo2, bg='grey94').grid(row=0, column=1, sticky = 'W')

        # load in another picture, because if one picture is good, two is better, etc.
        photo3 = tk.PhotoImage(file = os.path.join(basedir,'lablogo_small.gif'))  # 'lablogo.gif'
        controller.photo3 = photo3   # need to keep a copy so it is not cleared from memory
        # put in another figure, for pure artistic value, in the 1st row, 2nd column, of a grid layout for the window
        # and make the background black
        self.P3=tk.Label(self.parent, image = photo3, bg='grey94').grid(row=0, column=2, sticky = 'W')

        # create a label under the pictures (row 2), spanning two columns, to tell the user what they are running
        # specify a black background and white letters, with 12 point bold font
        self.L0 = tk.Label(self.parent, text = "PANTHEON: whole CNS fMRI Analysis", bg = bgcol, fg = fgcol1, font = "none 12 bold")
        self.L0.grid(row=1, column = 0, columnspan = 3, sticky = 'W')


# --------------------BASE FRAME2---------------------------------------------------------------
# Definition of the frame that holds pictures, labels, etc.
# class BaseFrame2:
#     # initialize the values, keeping track of the frame this definition works on (parent), and
#     # also the main window containing that frame (controller)
#     def __init__(self, parent, controller):
#         parent.configure(relief='raised', bd=5, highlightcolor=fgcol2)  # just defining some visual features
#         self.parent = parent
#         self.controller = controller
#
#         # make objects in the base frame
#         # load in a picture, for no good reason, and display it in the window to look nice :)
#         photo1 = tk.PhotoImage(file=os.path.join(basedir, 'queens_flag2.gif'))
#         controller.photob1 = photo1  # need to keep a copy so it is not cleared from memory
#         # put this figure, in the 1st row, 1st column, of a grid layout for the window
#         # and make the background black
#         self.B1 = tk.Label(self.parent, image=photo1, bg='grey94').grid(row=0, column=0, sticky='W')
#
#         # load in another picture, because if one picture is good, two is better
#         photo2 = tk.PhotoImage(file=os.path.join(basedir, 'lablogo.gif'))
#         controller.photob2 = photo2  # need to keep a copy so it is not cleared from memory
#         # put in another figure, for pure artistic value, in the 1st row, 2nd column, of a grid layout for the window
#         # and make the background black
#         self.B2 = tk.Label(self.parent, image=photo2, bg='grey94').grid(row=0, column=1, sticky='W')
#
#         # create a label under the pictures (row 2), spanning two columns, to tell the user what they are running
#         # specify a black background and white letters, with 12 point bold font
#         self.L1 = tk.Label(self.parent, text="SC/BS fMRI Analysis", bg=bgcol, fg=fgcol1, font="none 16 bold")
#         self.L1.grid(row=1, column=0, columnspan=2, sticky='W')


#--------------------OPTIONS FRAME---------------------------------------------------------------
# Definition of the frame that holds the buttons for choosing which frame to have visible
class OptionsFrame:
    # initialize the values, keeping track of the frame this definition works on (parent), and
    # also the main window containing that frame (controller)

    # select frame function
    def options_show_frame(self,page_name, widgetname):
        buttonlist = self.buttons.keys()
        for button in buttonlist:
            buttonname = self.buttons[button]
            buttonname.configure(fg=fgletter2)
        if widgetname in buttonlist:
            self.buttons[widgetname].configure(fg=fgletter1)
        self.controller.show_frame(page_name)


    def __init__(self, parent, controller):
        parent.configure(relief='raised', bd = 5, highlightcolor = fgcol1)
        self.parent = parent
        self.controller = controller
        self.buttons = {}  # to keep track of button handles

        # button for running the conversion to NIfTI format step
        self.setdb = tk.Button(self.parent, text = 'Database', width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = lambda: self.options_show_frame('DBFrame', 'setdb'), relief='raised', bd = 5, highlightbackground = widgetbg)
        self.setdb.grid(row = 0, column = 0)
        self.buttons['setdb'] = self.setdb

        colors = {attr: self.setdb.cget(attr) for attr in (
            "background", "foreground", "disabledforeground",
            "highlightbackground", "highlightcolor")}
        print('button colors are {}'.format(colors))

        # button for running the conversion to NIfTI format step
        self.writenifti = tk.Button(self.parent, text = 'Write NIfTI', width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = lambda: self.options_show_frame('NIFrame', 'writenifti'), relief='raised', bd = 5, highlightbackground = widgetbg)
        self.writenifti.grid(row = 1, column = 0)
        self.buttons['writenifti'] = self.writenifti

        # button for calculating the normalization parameters for each data set
        self.normalizationcalc = tk.Button(self.parent, text = 'Norm. Calc.', width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.choosenormframe, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.normalizationcalc.grid(row = 2, column = 0)
        self.buttons['normalizationcalc'] = self.normalizationcalc

        # button for selecting and running the pre-processing steps
        self.preprocess = tk.Button(self.parent, text = 'Pre-Process', width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = lambda: self.options_show_frame('PPFrame', 'preprocess'), relief='raised', bd = 5, highlightbackground = widgetbg)
        self.preprocess.grid(row = 3, column = 0)
        self.buttons['preprocess'] = self.preprocess

        # button for selecting and running the GLM analysis steps
        self.glmanalysis = tk.Button(self.parent, text = 'GLM fit', width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = lambda: self.options_show_frame('GLMFrame', 'glmanalysis'), relief='raised', bd = 5, highlightbackground = widgetbg)
        self.glmanalysis.grid(row = 4, column = 0)
        self.buttons['glmanalysis'] = self.glmanalysis

        # button for selecting and running the data clustering steps
        self.clustering = tk.Button(self.parent, text = 'Cluster', width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = lambda: self.options_show_frame('CLFrame', 'clustering'), relief='raised', bd = 5, highlightbackground = widgetbg)
        self.clustering.grid(row = 5, column = 0)
        self.buttons['clustering'] = self.clustering

        # button for selecting and running the SEM analysis steps
        self.sem = tk.Button(self.parent, text = 'SEM', width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = lambda: self.options_show_frame('SEMFrame', 'sem'), relief='raised', bd = 5, highlightbackground = widgetbg)
        self.sem.grid(row = 6, column = 0)
        self.buttons['sem'] = self.sem

        # button for selecting and running the group-level analysis steps
        self.groupanalysis = tk.Button(self.parent, text = 'Group-level', width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont2, command = lambda: self.options_show_frame('GRPFrame', 'groupanalysis'), relief='raised', bd = 5, highlightbackground = widgetbg)
        self.groupanalysis.grid(row = 7, column = 0)
        self.buttons['groupanalysis'] = self.groupanalysis

        # button for selecting and running the group-level analysis steps
        self.display = tk.Button(self.parent, text = 'View GL', width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont2, command = lambda: self.options_show_frame('DisplayFrame', 'display'), relief='raised', bd = 5, highlightbackground = widgetbg)
        self.display.grid(row = 8, column = 0)
        self.buttons['display'] = self.display

        # button for selecting and running the group-level analysis steps
        self.display2 = tk.Button(self.parent, text = 'View GL 2', width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont2, command = lambda: self.options_show_frame('DisplayFrame2', 'display2'), relief='raised', bd = 5, highlightbackground = widgetbg)
        self.display2.grid(row = 9, column = 0)
        self.buttons['display2'] = self.display2

        if enable_sapm:
            # button for selecting and running the SAPM analysis steps
            self.sapm = tk.Button(self.parent, text = 'SAPM', width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = lambda: self.options_show_frame('SAPMFrame', 'sapm'), relief='raised', bd = 5, highlightbackground = widgetbg)
            self.sapm.grid(row = 10, column = 0)
            self.buttons['sapm'] = self.sapm

            # button for selecting and running the step to view/display SAPM results
            self.sapmresults = tk.Button(self.parent, text = 'SAPM Results', width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = lambda: self.options_show_frame('SAPMResultsFrame', 'sapmresults'), relief='raised', bd = 5, highlightbackground = widgetbg)
            self.sapmresults.grid(row = 11, column = 0)
            self.buttons['sapmresults'] = self.sapmresults

        # define a button to exit the GUI
        # also, define the function for what to do when this button is pressed
        self.exit_button = tk.Button(self.parent, text = 'Exit', width = smallbuttonsize, bg = 'grey80', fg = fgletter2, font = widgetfont, command = self.close_window, relief='sunken', bd = 5, highlightbackground = widgetbg)
        self.exit_button.grid(row = 12, column = 0)


    def choosenormframe(self):
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        DBname = settings['DBname']
        DBnum = settings['DBnum']
        if os.path.isfile(DBname):
            xls = pd.ExcelFile(DBname, engine = 'openpyxl')
            df1 = pd.read_excel(xls, 'datarecord')
            normtemplatename = df1.loc[DBnum[0], 'normtemplatename']
        else:
            normtemplatename = 'notdefined'

        if normtemplatename.lower() == 'brain':
            self.options_show_frame('NCbrainFrame', 'normalizationcalc')
        else:
            self.options_show_frame('NCFrame', 'normalizationcalc')

        # exit function
    def close_window(self):
        self.controller.master.destroy()


#--------------------DATABASE FRAME---------------------------------------------------------------
# Definition of the frame that has inputs for the database name, and entry numbers to use
class DBFrame:
    # inputs to search database, and create/save dbnum lists
    def get_DB_fields(self):
        if os.path.isfile(self.DBname):
            xls = pd.ExcelFile(self.DBname, engine = 'openpyxl')
            df1 = pd.read_excel(xls, 'datarecord')
            del df1['Unnamed: 0']  # get rid of the unwanted header column
            fields = list(df1.keys())
        else:
            fields = 'empty'
        return fields

    # inputs to search database, and create/save dbnum lists
    def get_DB_field_values(self):
        if os.path.isfile(self.DBname):
            xls = pd.ExcelFile(self.DBname, engine = 'openpyxl')
            df1 = pd.read_excel(xls, 'datarecord')
            del df1['Unnamed: 0']  # get rid of the unwanted header column
            fieldvalues = df1.loc[:,self.field_var.get()]
        else:
            fieldvalues = 'empty'

        ufieldvalues = []
        for value in fieldvalues:
            if value not in ufieldvalues:
                ufieldvalues.append(value)
        return ufieldvalues


    # initialize the values, keeping track of the frame this definition works on (parent), and
    # also the main window containing that frame (controller)
    def __init__(self, parent, controller):
        parent.configure(relief='raised', bd = 5, highlightcolor = fgcol3)
        self.parent = parent
        self.controller = controller
        self.searchkeys = {}
        self.DBname = settings['DBname']

        # create an entry box so that the user can specify the database file to use
        # first make a title for the box, in row 3, column 1 of the grid for the main window
        self.DBL1 = tk.Label(self.parent, text = "Database name:", font = labelfont)
        self.DBL1.grid(row=0,column=0, sticky='W')

        # make a label to show the current setting of the database name
        self.DBnametext = tk.StringVar()
        self.DBnametext.set(self.DBname)
        self.DBnamelabel2 = tk.Label(self.parent, textvariable = self.DBnametext, bg = bgcol, fg = fgletter2, font = "none 10", wraplength = 200, justify = 'left')
        self.DBnamelabel2.grid(row=0, column = 1, sticky = 'W')

        # define a button to browse and select an existing database file, and write out the selected name
        # also, define the function for what to do when this button is pressed
        self.DBsubmit = tk.Button(self.parent, text = 'Browse', width = smallbuttonsize, bg = fgcol1, fg = fgletter1, command = self.DBbrowseclick, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.DBsubmit.grid(row = 0, column = 2)

        # define a button to select a new database file
        # also, define the function for what to do when this button is pressed
        self.DBnew = tk.Button(self.parent, text = 'New', width = smallbuttonsize, bg = fgcol2, fg = fgletter2, command = self.DBnewclick, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.DBnew.grid(row = 0, column = 3)

        # now define an Entry box so that the user can enter database numbers
        # give it a label first
        self.DBL2 = tk.Label(self.parent, text = "Database numbers:", font = labelfont)
        self.DBL2.grid(row=1,column=0, sticky='W')

        # create the Entry box
        self.DBnumenter = tk.Entry(self.parent, width = 20, bg="white", highlightbackground = widgetbg)
        self.DBnumenter.grid(row=1, column = 1, sticky = "W")
        self.DBnumenter.insert(0,settings['DBnumstring'])

        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.DBnumsubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol1, fg = fgletter1, command = self.DBnumsubmitclick, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.DBnumsubmit.grid(row = 1, column = 2)

        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.DBnumload = tk.Button(self.parent, text = "Load", width = smallbuttonsize, bg = fgcol1, fg = fgletter1, command = self.DBnumlistload, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.DBnumload.grid(row = 1, column = 3)

        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.DBnumsavebutton = tk.Button(self.parent, text = "Save", width = smallbuttonsize, bg = fgcol1, fg = fgletter1, command = self.DBnumlistsave, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.DBnumsavebutton.grid(row = 1, column = 4)

        # add a button to clear the entered values
        self.DBnumclearbutton = tk.Button(self.parent, text = "Clear", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, command = self.DBnumclear, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.DBnumclearbutton.grid(row = 1, column = 5)

        # put a separator for readability
        ttk.Separator(self.parent).grid(row=2, column=1, columnspan=6, sticky="nswe", padx=2, pady=5)

        # add label, and pull-down menu for selected database values for searching
        self.DBL3 = tk.Label(self.parent, text = "Database search:", font = labelfont, highlightbackground = widgetbg)
        self.DBL3.grid(row=3,column=0,columnspan = 2, sticky='W')

        # add a button to clear the search values
        self.DBsearchclear = tk.Button(self.parent, text = "Clear Search", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, command = self.DBsearchclear, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.DBsearchclear.grid(row = 3, column = 3)

        # add a button to run the search
        self.DBrunsearch = tk.Button(self.parent, text = "Run Search", width = smallbuttonsize, bg = fgcol1, fg = fgletter1, command = self.DBrunsearch, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.DBrunsearch.grid(row = 3, column = 2)

        self.DBL5 = tk.Label(self.parent, text = "Value:", font = labelfont, highlightbackground = widgetbg)
        self.DBL5.grid(row=4,column=2, sticky='W')
        # fieldvalues = DBFrame.get_DB_field_values(self)
        fields = self.get_DB_fields()
        self.field_var = tk.StringVar()
        if len(fields) > 0:
            self.field_var.set(fields[0])
        else:
            self.field_var.set('empty')
        fieldvalues = self.get_DB_field_values()
        self.fieldvalue_var = tk.StringVar()
        if len(fieldvalues)>0:
            self.fieldvalue_var.set(fieldvalues[0])
        else:
            self.fieldvalue_var.set('empty')
        fieldvalue_menu = tk.OptionMenu(self.parent, self.fieldvalue_var, *fieldvalues, command = self.DBfieldvaluechoice)
        fieldvalue_menu.config(bg=bgcol)
        fieldvalue_menu.grid(row=4, column=3, sticky='EW')
        self.fieldvaluesearch_opt = fieldvalue_menu   # save this way so that values are not cleared

        self.DBL4 = tk.Label(self.parent, text = "Keyword:", font = labelfont, highlightbackground = widgetbg)
        self.DBL4.grid(row=4,column=0, sticky='W')
        # fields = DBFrame.get_DB_fields(self)
        field_menu = tk.OptionMenu(self.parent, self.field_var, *fields, command = self.DBfieldchoice)
        field_menu.config(bg=bgcol)
        field_menu.grid(row=4, column=1, sticky='EW')
        self.fieldsearch_opt = field_menu   # save this way so that values are not cleared

        # add information to show the current search terms
        self.searchterm_text = tk.StringVar()
        self.searchterm_text.set('Database search keys:  empty')
        self.DBsearchtext = tk.Label(self.parent, text = self.searchterm_text.get(), font = labelfont, highlightbackground = widgetbg)
        self.DBsearchtext.grid(row=5,column=1,columnspan = 2, sticky='W')

        # add information to show the current search terms
        self.searchresult_text = tk.StringVar()
        self.searchresult_text.set('Database search results:  empty')
        self.DBsearchresult = tk.Label(self.parent, text = self.searchresult_text.get(), font = labelfont, highlightbackground = widgetbg)
        self.DBsearchresult.grid(row=5,column=3,columnspan = 3, sticky='W')


        # add a button to convert database entries for different operating systems
        self.DBrunconvertosbutton = tk.Button(self.parent, text = "Convert DB for OS", width = smallbuttonsize, bg = fgcol1, fg = fgletter1, command = self.DBrunconvertOS, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.DBrunconvertosbutton.grid(row = 7, column = 2)


    # define functions before they are used in the database frame------------------------------------------
    # action when the button to browse for a DB fie is pressed
    def DBbrowseclick(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        # use a dialog box to prompt the user to select an existing file, the default being .xlsx type
        filename =  tkf.askopenfilename(title = "Select file",filetypes = (("excel files","*.xlsx"),("all files","*.*")))
        print('filename = ',filename)
        # save the selected file name in the settings
        settings['DBname'] = filename
        self.DBname = filename
        # write the result to the label box for display
        self.DBnametext.set(settings['DBname'])

        # update search menus
        fields = self.get_DB_fields()
        self.field_var = tk.StringVar()
        self.field_var.set(fields[0])
        fieldvalues = self.get_DB_field_values()
        self.fieldvalue_var = tk.StringVar()
        self.fieldvalue_var.set(fieldvalues[0])

        fieldvalue_menu = tk.OptionMenu(self.parent, self.fieldvalue_var, *fieldvalues, command = self.DBfieldvaluechoice)
        fieldvalue_menu.config(bg=bgcol)
        fieldvalue_menu.grid(row=4, column=3, sticky='EW')
        self.fieldvaluesearch_opt = fieldvalue_menu   # save this way so that values are not cleared

        field_menu = tk.OptionMenu(self.parent, self.field_var, *fields, command = self.DBfieldchoice)
        field_menu.config(bg=bgcol)
        field_menu.grid(row=4, column=1, sticky='EW')
        self.fieldsearch_opt = field_menu   # save this way so that values are not cleared

        # change the current working directory to the folder where the database file is, for future convenience
        pathsections = os.path.split(filename)
        # os.chdir(pathsections[0])   # on second thought, don't do it
        # save the updated settings file again
        np.save(settingsfile,settings)


    # action when the button to select a new DB file
    def DBnewclick(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        # use a dialog box to prompt the user to select an existing file, the default being .xlsx type
        filechoice =  tkf.asksaveasfilename(title = "Select file",filetypes = (("excel files","*.xlsx"),("all files","*.*")))
        # save the selected file name in the settings
        settings['DBname'] = filechoice
        self.DBname = filechoice
        # write the result to the label box for display
        self.DBnametext.set(settings['DBname'])
        # change the current working directory to the folder where the database file is, for future convenience
        pathsections = os.path.split(filechoice)
        # os.chdir(pathsections[0])   # don't do it
        # save the updated settings file again
        np.save(settingsfile,settings)

        #
        #-----------initialize a database file in excel----------------------------------------
        # ---------the user still needs to fill in the information in the database--------------
        required_db_sheets = ['datarecord','paradigm']
        required_db_fields = ['datadir','patientid','studygroup','pname','seriesnumber','niftiname','TR','normdataname','normtemplatename','paradigms']
        optional_db_fields = ['pulsefilename','sex','age','painrating','temperature']  # examples that can be changed
        db_fields = required_db_fields + optional_db_fields

        # initialize the database sheet
        initial_entries = ['root directory', 'person name', 'condition name', 'directory under root', 'series no.', 'img file name', 'repetition time', 'leave blank', 'leave blank', 'name of excel sheet']
        optional_entries = ['blank' for aa in range(len(optional_db_fields))]
        db_entries = np.array(initial_entries + optional_entries)

        df1 = pd.DataFrame(columns=db_fields,data = [])
        for aa,colname in enumerate(db_fields):
            df1.loc[0,colname] = db_entries[aa]

        # DBname = settings['DBname']
        # write it to the database by appending a sheet to the excel file
        with pd.ExcelWriter(settings['DBname']) as writer:
            df1.to_excel(writer, sheet_name='datarecord')

        # initialize the paradigm sheet
        required_pd_sheets = ['dt','paradigm']
        sample_paradigm = list(zip(5*np.ones(12),[0,0,0,1,1,1,0,0,0,1,1,1]))
        df2 = pd.DataFrame(columns=required_pd_sheets, data=sample_paradigm)
        with pd.ExcelWriter(settings['DBname'], engine="openpyxl", mode='a') as writer:
            df2.to_excel(writer, sheet_name='paradigm')


    def DBnumlistload(self):
        # prompt for a name for loading the list
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        last_folder = settings['last_folder']

        file_path = tkf.askopenfilename(title = "Select database list file", initialdir = last_folder, filetypes = (("npy","*.npy"),("all files","*.*")))
        print('loading list from ',file_path)
        dblist = np.load(file_path, allow_pickle = True).flat[0]
        dbnumlist = dblist['dbnumlist']
        print('database number list: ',dbnumlist)

        # convert list to text
        entered_text = ''
        if len(dbnumlist) > 0:
            entered_text = str(dbnumlist[0])
            for value in dbnumlist[1:]:
                termtext = ',{}'.format(value)
                entered_text += termtext
        self.DBnumsave_text = entered_text

        # copy the text to the dbnum entry box as well
        self.DBnumenter.delete(0, 'end')
        self.DBnumenter.insert(0, entered_text)

        settings['DBnum'] = dbnumlist
        settings['DBnumstring'] = self.DBnumsave_text

        # save the updated settings file again
        save_folder = os.path.dirname(file_path)
        settings['last_folder'] = save_folder
        np.save(settingsfile,settings)


    def DBnumlistsave(self):
        # prompt for a name for loading the list
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        last_folder = settings['last_folder']
        self.DBname = settings['DBname']

        # check database file and see how many entries exist
        if os.path.isfile(self.DBname):
            xls = pd.ExcelFile(self.DBname, engine='openpyxl')
            df1 = pd.read_excel(xls, 'datarecord')
            del df1['Unnamed: 0']  # get rid of the unwanted header column
            nentries, nfields = np.shape(df1)
            dbnum_max = nentries - 1
        else:
            dbnum_max = 0

        entered_text = self.DBnumenter.get()  # collect the text from the text entry box
        # allow for "all" to be entered
        if entered_text == 'all': entered_text = str(0) + ':' + str(dbnum_max)

        # parse the entered text into values
        entered_values = self.DBparsenumlist(entered_text, dbnum_max)
        print(entered_values)

        # convert back to shorter string for display
        value_list_for_display = self.DBdisplaynumlist(entered_values)
        self.DBnumsave_text = value_list_for_display

        settings['DBnum'] = entered_values
        settings['DBnumstring'] = self.DBnumsave_text
        self.DBnumenter.delete(0, 'end')
        self.DBnumenter.insert(0, settings['DBnumstring'])

        # now save the list
        # prompt for a name for saving the list
        filechoice = tkf.asksaveasfilename(title = "Select file", initialdir = last_folder, filetypes = (("npy","*.npy"),("all files","*.*")))
        list = {'dbnumlist':entered_values}
        print('list = ',list)
        print('saving database list saved to ',filechoice)
        np.save(filechoice,list)
        print('database number list saved to ',filechoice)

        # save a record for convenience, for next time
        save_folder = os.path.dirname(filechoice)
        settings['last_folder'] = save_folder
        np.save(settingsfile,settings)

    def DBdisplaynumlist(self, entered_values):
        delta = np.concatenate(([0],np.diff(entered_values)))
        dv = np.where(delta != 1)[0]
        textv = ''
        for nn, ndelta in enumerate(delta):
            if ndelta != 1:
                textv += str(entered_values[nn])
                if nn != (len(delta)-1):
                    if delta[nn+1] == 1:
                        textv += ':'
                    else:
                        textv += ','
            else:
                if nn != (len(delta)-1):
                    if delta[nn+1] != 1:
                        textv += str(entered_values[nn]) + ','
                else:
                    textv += str(entered_values[nn])
        if textv[-1] == ':': textv = textv[:-1]

        return textv


    def DBparsenumlist(self, entered_text, maxvalue):
        # need to make sure we are working with numbers, not text
        # first, replace any double spaces with single spaces, and then replace spaces with commas
        entered_text = re.sub('\ +', ',', entered_text)
        entered_text = re.sub('\,\,+', ',', entered_text)
        # remove any leading or trailing commas
        if entered_text[0] == ',': entered_text = entered_text[1:]
        if entered_text[-1] == ',': entered_text = entered_text[:-1]

        # fix up the text bit, allow the user to specify ranges of numbers with a colon
        # need to change the text to write out the range
        # first see if there are any colons included in the text
        # this part is complicated - need to find any pairs of numbers separated by a colon, to indicate a range
        m = re.search(r'\d*:\d*', entered_text)
        while m:
            # m[0] is the string that matches what we are looking for - two numbers separated by a colon
            # in m[0] we find where there is the colon, with m[0].find(':')
            # so, within m[0] everything in the range of :m[0].find(':') is the first number
            # and everything in the range of (m[0].find(':')+1): is the second number
            num1 = int(m[0][:m[0].find(':')])
            num2 = int(m[0][(m[0].find(':') + 1):])
            # now create the long string that we need to replace the short form indicated with the colon
            numbers = np.arange(num1, num2 + 1)
            numtext = ''
            for n in numbers:
                numtext1 = '{:d},'.format(n)
                numtext += numtext1
            # now insert this where the colon separated pair of number came out
            new_text = entered_text[:m.start()] + numtext + entered_text[(m.end() + 1):]
            entered_text = new_text
            m = re.search(r'\d*:\d*', entered_text)

        entered_values = np.fromstring(entered_text, dtype=int, sep=',')

        # check upper limit
        entered_values = entered_values[entered_values <= maxvalue]

        return entered_values


        # action when the button is pressed to submit the DB entry number list
    def DBnumsubmitclick(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.DBname = settings['DBname']

        # check database file and see how many entries exist
        if os.path.isfile(self.DBname):
            xls = pd.ExcelFile(self.DBname, engine = 'openpyxl')
            df1 = pd.read_excel(xls, 'datarecord')
            del df1['Unnamed: 0']  # get rid of the unwanted header column
            nentries,nfields = np.shape(df1)
            dbnum_max = nentries-1
        else:
            dbnum_max = 0

        entered_text = self.DBnumenter.get()  # collect the text from the text entry box
        # allow for "all" to be entered
        if entered_text == 'all': entered_text = str(0) + ':' + str(dbnum_max)

        # parse the entered text into values
        entered_values = self.DBparsenumlist(entered_text, dbnum_max)
        print(entered_values)

        # convert back to shorter string for display
        value_list_for_display = self.DBdisplaynumlist(entered_values)
        self.DBnumsave_text = value_list_for_display

        settings['DBnum'] = entered_values
        settings['DBnumstring'] = self.DBnumsave_text
        self.DBnumenter.delete(0,'end')
        self.DBnumenter.insert(0,settings['DBnumstring'])
        # save the updated settings file again
        np.save(settingsfile,settings)


    # action when the button is pressed to clear the DB entry number list
    def DBnumclear(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        settings['DBnum'] = 'none'
        settings['DBnumstring'] = 'none'
        self.DBnumenter.delete(0,'end')
        self.DBnumenter.insert(0,settings['DBnumstring'])
        # save the updated settings file again
        np.save(settingsfile,settings)


    def DBsearchclear(self):
        self.searchkeys = {}
        print('search values have been cleared ...')
        searchtext = ''
        self.searchterm_text.set(searchtext)
        self.DBsearchtext.configure(text = self.searchterm_text.get())
        return self


    def DBrunsearch(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        last_folder  = settings['last_folder']

        self.searchkeys
        print('search values are: ',self.searchkeys)
        dbnumlist = pydatabase.get_dbnumlists_by_keyword(self.DBname, self.searchkeys)
        self.dbnumlist = dbnumlist

        # write out the dbnumlist ....
        print('dbnumlist = ',dbnumlist)

        dbnumtext = ''
        if len(dbnumlist) > 0:
            dbnumtext = str(dbnumlist[0])
            for value in dbnumlist[1:]:
                termtext = ',{}'.format(value)
                dbnumtext += termtext
        print(dbnumtext)
        dbnumtextshort = '{} values found'.format(len(dbnumlist))
        self.searchresult_text.set(dbnumtextshort)
        self.DBsearchresult.configure(text=self.searchresult_text.get())

        # prompt for a name for saving the list
        filechoice = tkf.asksaveasfilename(title = "Select file", initialdir = last_folder, filetypes = (("npy","*.npy"),("all files","*.*")))
        list = {'dbnumlist':dbnumlist}
        print('list = ',list)
        print('saving database list saved to ',filechoice)
        np.save(filechoice,list)
        print('database number list saved to ',filechoice)

        # save a record for convenience, for next time
        save_folder = os.path.dirname(filechoice)
        settings['last_folder'] = save_folder
        np.save(settingsfile,settings)

        return self

    def DBfieldchoice(self,value):
        # get the field value choices for the selected field
        self.field_var.set(value)
        fieldvalues = DBFrame.get_DB_field_values(self)

        # destroy the old pulldown menu and create a new one with the new choices
        self.fieldvaluesearch_opt.destroy()  # remove it
        fieldvalue_menu = tk.OptionMenu(self.parent, self.fieldvalue_var, *fieldvalues, command = self.DBfieldvaluechoice)
        fieldvalue_menu.config(bg=bgcol)
        fieldvalue_menu.grid(row=4, column=3, sticky='EW')
        self.fieldvaluesearch_opt = fieldvalue_menu   # save this way so that values are not cleared

        return self


    def DBfieldvaluechoice(self,value):
        self.fieldvalue_var.set(value)
        self.searchkeys[self.field_var.get()] = self.fieldvalue_var.get()

        print('Search keys are: ', self.searchkeys)
        searchtext = ''
        for keyname in self.searchkeys:
            termtext = '{}:{}\n'.format(keyname,self.searchkeys[keyname])
            searchtext += termtext
        print(searchtext)
        self.searchterm_text.set(searchtext)
        self.DBsearchtext.configure(text = self.searchterm_text.get())

        return self

    def DBrunconvertOS(self):
        # determine the operating system and change database entries to use the correct file separators
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.DBname = settings['DBname']
        newdbname = pydatabase.convert_database_OS(self.DBname)
        print('converted database file {}'.format(newdbname))


#--------------------NIFTI conversion FRAME---------------------------------------------------------------
# Definition of the frame that will have inputs and options for converting DICOM images to NIfTI format
class NIFrame:
    # initialize the values, keeping track of the frame this definition works on (parent), and
    # also the main window containing that frame (controller)
    def __init__(self, parent, controller):
        parent.configure(relief='raised', bd = 5, highlightcolor = fgcol3)
        self.parent = parent
        self.controller = controller

        # initialize some values
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.NIdatabasename = settings['DBname']
        self.NIdatabasenum = settings['DBnum']
        self.NIpname = 'not yet defined'
        self.NIbasename = settings['NIbasename']  # 'Series'   # default base name

        # put some text as a place-holder
        self.NIlabel1 = tk.Label(self.parent, text = "1) Organize data into\none series per folder", fg = 'gray')
        self.NIlabel1.grid(row=0,column=0, sticky='W')

        self.NIlabel1 = tk.Label(self.parent, text = "2) Convert each series\ninto one NIfTI file", fg = 'gray')
        self.NIlabel1.grid(row=1,column=0, sticky='W')

                # now define an Entry box so that the user can enter database numbers
        # give it a label first
        self.NIinfo1 = tk.Label(self.parent, text = "Base name:", font = labelfont)
        self.NIinfo1.grid(row=0,column=1, sticky='E')

        # create the Entry box, and put it next to the label, 4th row, 2nd column
        self.NInameenter = tk.Entry(self.parent, width = 20, bg="white")
        self.NInameenter.grid(row=0, column = 2, sticky = "W")
        self.NInameenter.insert(0,settings['NIbasename'])

        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.NInamesubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, command = self.NInamesubmitclick, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.NInamesubmit.grid(row = 0, column = 3)

        # for now, just put a button that will eventually call the NIfTI conversion program
        self.NIorganizedata = tk.Button(self.parent, text = 'Organize Data', width = bigbuttonsize, bg = fgcol1, fg = fgletter1, command = self.NIorganizeclick, font = widgetfont, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.NIorganizedata.grid(row = 1, column = 1, columnspan = 2)

        # butotn to call the NIfTI conversion program
        self.NIrunconvert = tk.Button(self.parent, text = 'Convert', width = bigbuttonsize, bg = fgcol1, fg = fgletter1, command = self.NIconversionclick, font = widgetfont, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.NIrunconvert.grid(row = 2, column = 1, columnspan = 2)


    # action when the button is pressed to submit the DB entry number list
    def NInamesubmitclick(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]

        entered_text = self.NInameenter.get()  # collect the text from the text entry box
        # remove any spaces
        entered_text = re.sub('\ +','',entered_text)
        print(entered_text)

        # update the text in the box, in case it has changed
        settings['NIbasename'] = entered_text
        self.NInameenter.delete(0,'end')
        self.NInameenter.insert(0,settings['NIbasename'])
        # save the updated settings file again
        np.save(settingsfile,settings)


    # action when the button is pressed to organize dicom data into folders based on series numbers
    def NIorganizeclick(self):
        # first get necessary the input data
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.NIdatabasename = settings['DBname']
        self.NIdatabasenum = settings['DBnum']
        # BASEdir = os.path.dirname(self.NIdatabasename)
        xls = pd.ExcelFile(self.NIdatabasename, engine = 'openpyxl')
        df1 = pd.read_excel(xls, 'datarecord')

        print('File organization: databasename ',self.NIdatabasename)
        print('File organization: started organizing at ', time.ctime(time.time()))

        for nn, dbnum in enumerate(self.NIdatabasenum):
            print('NIorganizeclick: databasenum ',dbnum)
            pname = df1.loc[dbnum, 'pname']
            dbhome = df1.loc[dbnum, 'datadir']
            self.NIpname = os.path.join(dbhome, pname)
            dicom_conversion.move_files_and_update_database(self.NIdatabasename, dbhome, self.NIpname)

        print('File organization: finished organizing data ...', time.ctime(time.time()))


    # action when the button is pressed to convert dicom data into NIfTI format
    def NIconversionclick(self):
        # first get necessary the input data
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.NIdatabasename = settings['DBname']
        self.NIdatabasenum = settings['DBnum']
        self.NIbasename = settings['NIbasename']
        print('NIfTI conversion: databasename ',self.NIdatabasename)
        print('NIfTI conversion: started organizing at ', time.ctime(time.time()))

        for nn, dbnum in enumerate(self.NIdatabasenum):
            niiname = dicom_conversion.convert_dicom_folder(self.NIdatabasename, dbnum, self.NIbasename)
            print('NIfTI conversion: converted ',dbnum,' : ',niiname)

        print('NIfTI conversion: finished converting data to NIfTI ...', time.ctime(time.time()))



# --------------------Calculate Normalization Parameters FRAME---------------------------------------------------------------
# Definition of the frame that will have inputs and options for normalizing NIfTI format data
class NCFrame:
    # initialize the values, keeping track of the frame this definition works on (parent), and
    # also the main window containing that frame (controller)
    def __init__(self, parent, controller):
        parent.configure(relief='raised', bd=5, highlightcolor=fgcol3)
        self.parent = parent
        self.controller = controller
        self.NCmanomode = 'OFF'
        self.overrideactive = False
        self.overrideangle = False
        self.overridepos = False
        self.overridesection = 0
        self.NCresult = []
        self.NCresult_copy = []

        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.normdatasavename = settings['NCsavename']  # default prefix value
        self.fitparameters = settings['NCparameters'] #  [50,50,5,6,-10,20,-10,10]  # default prefix value
        self.fitp0 = self.fitparameters[0]
        self.fitp1 = self.fitparameters[1]
        self.fitp2 = self.fitparameters[2]
        self.fitp3 = self.fitparameters[3]
        self.fitp45 = '{},{}'.format(self.fitparameters[4],self.fitparameters[5])
        self.fitp67 = '{},{}'.format(self.fitparameters[6],self.fitparameters[7])
        self.roughnorm = 0
        self.finetune = 0
        self.copydbnumber = 0

        # initialize some values
        self.NCdatabasename = settings['DBname']
        self.NCdbnum = settings['DBnum']

        if os.path.isfile(self.NCdatabasename):
            xls = pd.ExcelFile(self.NCdatabasename, engine = 'openpyxl')
            df1 = pd.read_excel(xls, 'datarecord')
            self.normtemplatename = df1.loc[self.NCdbnum[0], 'normtemplatename']
        else:
            self.normtemplatename = 'notdefined'

        self.NCtemplatelabel = tk.Label(self.parent, text = 'Normalizing region: '+self.normtemplatename, fg = 'gray', font = labelfont)
        self.NCtemplatelabel.grid(row=0,column=2, sticky='W')

        # put some text as a place-holder
        self.NClabel1 = tk.Label(self.parent, text = "1) Calculate normalization\nparameters", fg = 'gray')
        self.NClabel1.grid(row=0,column=0, sticky='W')
        self.NClabel2 = tk.Label(self.parent, text = "2) Save for next steps", fg = 'gray')
        self.NClabel2.grid(row=1,column=0, sticky='W')

        # now define an Entry box so that the user can indicate the prefix name of the data to normalize
        # give it a label first
        self.NCinfo1 = tk.Label(self.parent, text = "Save name base:", font = labelfont)
        self.NCinfo1.grid(row=1,column=1, sticky='E')

        # create the Entry box, and put it next to the label
        self.NCsavename = tk.Entry(self.parent, width = 20, bg="white")
        self.NCsavename.grid(row=1, column = 2, sticky = "W")
        self.NCsavename.insert(1,self.normdatasavename)

        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.NCsavenamesubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, command = self.NCsavenamesubmit, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.NCsavenamesubmit.grid(row = 1, column = 3)

        # need entry boxes for "fit_parameters"
        # fit_parameters: 0 position stiffness upper  1 position stiffness lower  2 angle_stiffness upper 3 angle stiffness lower
        # 4 anglestart, 5 anglestop 6 anglestart lower 7 anglestop lower
        # these parameters are scaled by [1e-4, 1e-4, 1e-4, 1, 1, 1, 1, 1] before use  so the inputs are more intuitive values

        # label the columns
        self.NCupper_label = tk.Label(self.parent, text = "Brainstem Regions", font = labelfont)
        self.NCupper_label.grid(row=2,column=2, sticky='W')
        self.NClower_label = tk.Label(self.parent, text = "Cord Regions", font = labelfont)
        self.NClower_label.grid(row=2,column=3, sticky='W')

        # define an Entry box for each fit_parameters value
        # give it a label -- position stiffness
        self.NCfitp0_label = tk.Label(self.parent, text = "Position Stiffness (0-100):", font = labelfont)
        self.NCfitp0_label.grid(row=3,column=1, sticky='E')
        # create the Entry box, for position stiffness upper
        self.NCfitp0 = tk.Entry(self.parent, width = 8, bg="white")
        self.NCfitp0.grid(row=3, column = 2, sticky = "W")
        self.NCfitp0.insert(0,self.fitp0)
        # create the Entry box, for position stiffness lower
        self.NCfitp1 = tk.Entry(self.parent, width = 8, bg="white")
        self.NCfitp1.grid(row=3, column = 3, sticky = "W")
        self.NCfitp1.insert(0,self.fitp1)

        # define an Entry box for each fit_parameters value
        # give it a label -- angle stiffness
        self.NCfitp2_label = tk.Label(self.parent, text="Angle Stiffness (0-100):", font = labelfont)
        self.NCfitp2_label.grid(row=4, column=1, sticky='E')
        # create the Entry box, for position stiffness upper
        self.NCfitp2 = tk.Entry(self.parent, width=8, bg="white")
        self.NCfitp2.grid(row=4, column=2, sticky="W")
        self.NCfitp2.insert(0, self.fitp2)
        # create the Entry box, for position stiffness lower
        self.NCfitp3 = tk.Entry(self.parent, width=8, bg="white")
        self.NCfitp3.grid(row=4, column=3, sticky="W")
        self.NCfitp3.insert(0, self.fitp3)

        # define an Entry box for each fit_parameters value
        # give it a label -- angle stiffness
        self.NCfitp4_label = tk.Label(self.parent, text="Angle Start,Stop (degrees):", font = labelfont)
        self.NCfitp4_label.grid(row=5, column=1, sticky='E')
        # create the Entry box, for position stiffness upper
        self.NCfitp45 = tk.Entry(self.parent, width=8, bg="white")
        self.NCfitp45.grid(row=5, column=2, sticky="W")
        self.NCfitp45.insert(0, self.fitp45)
        # create the Entry box, for position stiffness lower
        self.NCfitp67 = tk.Entry(self.parent, width=8, bg="white")
        self.NCfitp67.grid(row=5, column=3, sticky="W")
        self.NCfitp67.insert(0, self.fitp67)

        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.NCfitparamsubmit = tk.Button(self.parent, text="Submit", width=smallbuttonsize, bg=fgcol2, fg = fgletter2, font = widgetfont,
                                          command=self.NCfitparamsubmit, relief='raised', bd=5, highlightbackground = widgetbg)
        self.NCfitparamsubmit.grid(row=3, column=4, rowspan=3)

        # checkboxes to indicate 1) do rough normalization, 2) do fine-tuning normalization
        self.var1 = tk.IntVar()
        self.NCroughnorm = tk.Checkbutton(self.parent, text = 'Rough Norm.', width = smallbuttonsize, fg = fgletter2,
                                          command = self.NCcheckboxes, variable = self.var1)
        self.NCroughnorm.grid(row = 4, column = 0, sticky="E")
        self.var2 = tk.IntVar()
        self.NCfinetune = tk.Checkbutton(self.parent, text = 'Fine-Tune', width = smallbuttonsize, fg = fgletter2,
                                          command = self.NCcheckboxes, variable = self.var2)
        self.NCfinetune.grid(row = 5, column = 0, sticky="E")

        # button to call the normalization program
        self.NCrun = tk.Button(self.parent, text = 'Calculate Normalization', width = bigbigbuttonsize, bg = fgcol1, fg = fgletter1, command = self.NCrunclick, font = widgetfont, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.NCrun.grid(row = 6, column = 1)

        # button to run program to manually adjust rough normalization sections
        self.NCmano = tk.Button(self.parent, text = 'Manual Over-ride', width = bigbuttonsize, bg = fgcol3, fg = fgletter3, command = self.NCmanoclick, font = widgetfont, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.NCmano.grid(row = 6, column = 2)

        # button to recalculate normalization after manual over-ride
        self.NCrun = tk.Button(self.parent, text = 'Recalculate', width = bigbigbuttonsize, bg = fgcol3, fg = fgletter3, command = self.NCrecalculate_after_override, font = widgetfont, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.NCrun.grid(row = 7, column = 4)


        # entry box and button to copy rough normalization from another database number
        self.NCcopy_label = tk.Label(self.parent, text="Copy rough normalization from another data set:", font = labelfont)
        self.NCcopy_label.grid(row=8, column=4, columnspan = 3, sticky='SE')
        self.NCcopy_label2 = tk.Label(self.parent, text="Datbase number:", font = labelfont)
        self.NCcopy_label2.grid(row=9, column=4, sticky='E')
        # create the Entry box, for position stiffness upper
        self.NCcopyentry = tk.Entry(self.parent, width=8, bg="white")
        self.NCcopyentry.grid(row=9, column=5, sticky="W")
        self.NCcopyentry.insert(0, self.copydbnumber)
        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.NCcopysubmit = tk.Button(self.parent, text = "Copy", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.NCcopyroughnorm, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.NCcopysubmit.grid(row = 9, column = 6)

        img1 = tk.PhotoImage(file=os.path.join(basedir, 'smily.gif'))
        controller.img1d = img1  # need to keep a copy so it is not cleared from memory
        self.window1 = tk.Canvas(master = self.parent, width=img1.width(), height=img1.height(), bg='black')
        self.window1.grid(row=7, column=0,rowspan = 3, columnspan = 2, sticky='NW')
        self.windowdisplay1 = self.window1.create_image(0, 0, image=img1, anchor=tk.NW)

        img2 = tk.PhotoImage(file=os.path.join(basedir, 'smily.gif'))
        img2 = img2.subsample(2)
        controller.img2d = img2  # need to keep a copy so it is not cleared from memory
        self.window2 = tk.Canvas(master = self.parent, width=img2.width(), height=img2.height(), bg='black')
        self.window2.grid(row=7, column=2,rowspan = 2, columnspan = 2, sticky='NW')
        self.windowdisplay2 = self.window2.create_image(0, 0, image=img2, anchor=tk.NW)

        img3 = tk.PhotoImage(file=os.path.join(basedir, 'smily.gif'))
        img3 = img3.subsample(2)
        controller.img3d = img3  # need to keep a copy so it is not cleared from memory
        self.window3 = tk.Canvas(master = self.parent, width=img3.width(), height=img3.height(), bg='black')
        self.window3.grid(row=9, column=2,rowspan = 2, columnspan = 2, sticky='NW')
        self.windowdisplay3 = self.window3.create_image(0, 0, image=img3, anchor=tk.NW)


    def outline_section(self, sectionnumber):
        nf = sectionnumber
        coords = self.NCresult_copy[nf]['coords']
        angle = self.NCresult_copy[nf]['angle']
        sectionsize = np.shape(self.NCresult_copy[nf]['template_section'])
        smallestside = np.min(sectionsize[1:])
        sa = (np.pi / 180) * angle
        hv = (sectionsize[2] / 2) * np.array([math.cos(sa), -math.sin(sa)])
        vv = (sectionsize[1] / 2) * np.array([math.sin(sa), math.cos(sa)])
        p0 = coords[[2, 1]] - hv - vv
        p1 = coords[[2, 1]] - hv + vv
        p2 = coords[[2, 1]] + hv + vv
        p3 = coords[[2, 1]] + hv - vv

        return p0,p1,p2,p3,coords,angle,sectionsize,smallestside


    def findclosestsection(self, x,y):
        P = np.array([x,y])
        nfordisplay = len(self.NCresult_copy)
        mindist_record = np.zeros(nfordisplay)
        inside_record = [False for i in range(nfordisplay)]
        nearedge_record = [False for i in range(nfordisplay)]

        for nf in range(nfordisplay):
            p0, p1, p2, p3, coords, angle, sectionsize, smallestside = self.outline_section(nf)
            points = [p0,p1,p2,p3,p0]

            # distance to closest point on line passing through points p0 and p1
            mindist = np.zeros(4)
            for aa in range(4):
                p1 = points[aa+1]
                p0 = points[aa]
                v = p1-p0
                dline = np.linalg.norm( (p1[0]-p0[0])*(p0[1]-P[1]) - (p0[0]-P[0])*(p1[1]-p0[1]) )/np.linalg.norm(v)
                vp0 = P - p0
                dp0 = np.linalg.norm(vp0)
                vp1 = P - p1
                dp1 = np.linalg.norm(vp1)
                mindist[aa] = np.min(np.array([dline, dp0, dp1]))
            dist_to_edge = np.min(mindist)
            nearedge = dist_to_edge < 0.3*smallestside
            # is the point, P, inside the rectangular area?
            # since the corners are all defined in clockwise order - use the cross-product
            z = np.zeros(4)
            for aa in range(4):
                a = points[aa]
                b = points[aa+1]
                v0 = P - a
                v1 = P - b
                z[aa] = v1[0]*v0[1] - v1[1]*v0[0]   # positive or negative?
            inside_rect = (z >= 0).all()

            mindist_record[nf] = dist_to_edge
            inside_record[nf] = inside_rect
            nearedge_record[nf] = nearedge

        a = np.where(inside_record)[0]
        if len(a) == 0:
            region_selected = False
            closest_section = -1
            nearedge = False
            # print('region selected = ',region_selected)
        else:
            region_selected = True
            if len(a) > 1:
                b = [mindist_record[a[ii]] for ii in range(len(a))]
                ii = np.argmax(b)
                a = a[ii]
            else:
                a = a[0]
            # print('region selected = ',region_selected)
            # print('closest section is number ',a)
            closest_section = a
            nearedge = nearedge_record[a]

        return region_selected, closest_section, nearedge


    def mouseleftclick(self,event):
        if self.NCmanomode == 'ON':
            # print('image window (x,y) = ({},{})'.format(event.x,event.y))
            region_selected, closest_section, nearedge = self.findclosestsection(event.x,event.y)

            # if rough normalization results are shown, then indicate which region has been selected
            # first, find which section is closest to selected point ...
            if region_selected:
                self.overrideactive = True
                self.overridesection = closest_section
                nfordisplay = len(self.NCresult_copy)
                if nfordisplay > 0:
                    image_tk = self.controller.img1d
                    # self.window1.configure(width=image_tk.width(), height=image_tk.height())
                    self.windowdisplay1 = self.window1.create_image(0, 0, image=image_tk, anchor=tk.NW)

                    for nf in range(nfordisplay):
                        if nf == closest_section:
                            if nearedge:
                                fillcolor = 'red'  # indicates to change rotation angle
                                self.overrideangle = True
                                self.overridepos = False
                            else:
                                fillcolor = 'blue' # indicates to change position
                                self.overrideangle = False
                                self.overridepos = True
                        else:
                            # fillcolor = 'yellow'
                            colorval1 = np.floor(255.*(nf+1)/nfordisplay).astype(int)
                            colorval2 = 255-np.floor(127.*(nf+1)/nfordisplay).astype(int)
                            fillcolor = "#%02x%02x%02x" % (colorval1, colorval2, 0)
                        # draw the rectangular regions for each section
                        p0, p1, p2, p3, coords, angle, sectionsize, smallestside = self.outline_section(nf)
                        self.window1.create_line(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p0[0], p0[1],
                                                 fill=fillcolor, width=1)
        else:
            print('manual over-ride mode is OFF')
        return event.x,event.y


    def mouserightclick(self,event):
        # self.NCresult_copy = copy.deepcopy(self.NCresult)
        if self.NCmanomode == 'ON':
            # print('right-click in image window (x,y) = ({},{})'.format(event.x,event.y))
            # determine if translation or rotation indicated
            if self.overrideactive:
                if self.overridepos:
                    # get position of active section
                    coords = self.NCresult_copy[self.overridesection]['coords']
                    new_coords = coords
                    new_coords[1] = event.y
                    new_coords[2] = event.x
                    self.NCresult_copy[self.overridesection]['coords'] = new_coords

                if self.overrideangle:
                    angle = self.NCresult_copy[self.overridesection]['angle']
                    coords = self.NCresult_copy[self.overridesection]['coords']
                    new_angle = angle
                    deltay = event.y-coords[1]
                    deltaz = event.x-coords[2]
                    delta_angle = 0.1*deltaz
                    new_angle -= delta_angle
                    self.NCresult_copy[self.overridesection]['angle'] = new_angle

                nfordisplay = len(self.NCresult_copy)
                if nfordisplay > 0:
                    image_tk = self.controller.img1d
                    # self.window1.configure(width=image_tk.width(), height=image_tk.height())
                    self.windowdisplay1 = self.window1.create_image(0, 0, image=image_tk, anchor=tk.NW)

                    for nf in range(nfordisplay):
                        if nf == self.overridesection:
                            if self.overrideangle:
                                fillcolor = 'red'  # indicates to change rotation angle
                            else:
                                fillcolor = 'blue' # indicates to change position
                        else:
                            # fillcolor = 'yellow'
                            colorval1 = np.floor(255.*(nf+1)/nfordisplay).astype(int)
                            colorval2 = 255-np.floor(127.*(nf+1)/nfordisplay).astype(int)
                            fillcolor = "#%02x%02x%02x" % (colorval1, colorval2, 0)
                        # draw the rectangular regions for each section
                        p0, p1, p2, p3, coords, angle, sectionsize, smallestside = self.outline_section(nf)
                        self.window1.create_line(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p0[0], p0[1],
                                                 fill=fillcolor, width=1)
        else:
            print('manual over-ride mode is OFF')
        return event.x,event.y


    def NCcopyroughnorm(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        copydbnum = int(self.NCcopyentry.get())  # collect the db num from the entry box
        self.NCdatabasename = settings['DBname']
        self.NCdatabasenum = settings['DBnum']
        # BASEdir = os.path.dirname(self.NCdatabasename)
        xls = pd.ExcelFile(self.NCdatabasename, engine = 'openpyxl')
        df1 = pd.read_excel(xls, 'datarecord')

        # get current normdataname, etc.
        dbnum = self.NCdatabasenum[0]
        dbhome = df1.loc[dbnum, 'datadir']
        fname = df1.loc[dbnum, 'niftiname']
        seriesnumber = df1.loc[dbnum, 'seriesnumber']
        normtemplatename = df1.loc[dbnum, 'normtemplatename']
        niiname = os.path.join(dbhome, fname)

        normname = df1.loc[dbnum, 'normdataname']
        normdataname_full = os.path.join(dbhome, normname)

        # get normdataname to copy from
        dbhomeC = df1.loc[copydbnum, 'datadir']
        normnameC = df1.loc[copydbnum, 'normdataname']
        normdataname_fullC = os.path.join(dbhomeC, normnameC)

        # load normdata to be copied
        print('loading normdata from {}'.format(normdataname_fullC))
        normdata = np.load(normdataname_fullC, allow_pickle=True).flat[0]
        result = normdata['result']
        template_affine = normdata['template_affine']
        normdata['Tfine'] = 'none'
        normdata['norm_image_fine'] = 'none'   # do not include the fine-tuned data from the copy
        self.NCresult = copy.deepcopy(result)
        self.NCresult_copy = copy.deepcopy(result)

        T, warpdata, reverse_map_image, forward_map_image, new_result, imagerecord, displayrecord = pynormalization.py_load_modified_normalization(
            niiname, normtemplatename, result)

        Tfine = 'none'
        norm_image_fine = 'none'
        normdata = {'T': T, 'Tfine': Tfine, 'warpdata': warpdata, 'reverse_map_image': reverse_map_image,
                    'norm_image_fine': norm_image_fine, 'template_affine': template_affine, 'imagerecord': imagerecord,
                    'result': result}
        np.save(normdataname_full, normdata)

        print('saving normdata to {}'.format(normdataname_full))
        np.save(normdataname_full, normdata)   # overwrite the norm data with the copy

        # display results-----------------------------------------------------
        nfordisplay = len(imagerecord)
        for nf in range(nfordisplay):
            img1 = imagerecord[nf]['img']
            img1 = (255. * img1 / np.max(img1)).astype(np.uint8)
            image_tk = ImageTk.PhotoImage(Image.fromarray(img1))
            self.controller.img1d = image_tk
            self.window1.configure(width=image_tk.width(), height=image_tk.height())
            self.windowdisplay1 = self.window1.create_image(0, 0, image=image_tk, anchor=tk.NW)
            time.sleep(1)

        nfordisplay = len(result)
        for nf in range(nfordisplay):
            # draw the rectangular regions for each section
            p0, p1, p2, p3, coords, angle, sectionsize, smallestside = self.outline_section(nf)
            self.window1.create_line(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p0[0], p0[1],
                                     fill='yellow', width=2)

        print('finished copying rough normalization data and saving results ...')


    # action when the button is pressed to submit the DB entry number list
    def NCsavenamesubmit(self):
        entered_text = self.NCsavename.get()  # collect the text from the text entry box
        # remove any spaces
        entered_text = re.sub('\ +', '', entered_text)
        print(entered_text)

        # update the text in the box, in case it has changed
        self.normdatasavename = entered_text

        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        settings['NCsavename'] = entered_text
        np.save(settingsfile,settings)

        return self


    # action when the button is pressed to submit the fit_parameters
    def NCfitparamsubmit(self):
        text0 = self.NCfitp0.get()  # collect the text from the text entry box
        print(text0)
        p0 = float(text0)

        # # remove any spaces
        # entered_text0 = re.sub('\ +','',entered_text0)

        text1 = self.NCfitp1.get()  # collect the text from the text entry box
        p1 = float(text1)

        text2 = self.NCfitp2.get()  # collect the text from the text entry box
        p2 = float(text2)

        text3 = self.NCfitp3.get()  # collect the text from the text entry box
        p3 = float(text3)

        text45 = self.NCfitp45.get()  # collect the text from the text entry box
        c = text45.find(',')
        p4 = float(text45[:c])
        p5 = float(text45[c+1:])

        text67 = self.NCfitp67.get()  # collect the text from the text entry box
        c = text67.find(',')
        p6 = float(text67[:c])
        p7 = float(text67[c+1:])

        # update the text in the box, in case it has changed
        self.fitparameters = [p0,p1,p2,p3,p4,p5,p6,p7]
        print('fit parameters: ',self.fitparameters)

        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        settings['NCparameters'] = self.fitparameters
        np.save(settingsfile,settings)

        return self


    # action when checkboxes are selected/deselected
    def NCcheckboxes(self):
        self.roughnorm = self.var1.get()
        self.finetune = self.var2.get()
        return self


    # action when the button is pressed to organize dicom data into folders based on series numbers
    def NCrunclick(self):
        # first get the necessary input data
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.NCdatabasename = settings['DBname']
        self.NCdatabasenum = settings['DBnum']
        # BASEdir = os.path.dirname(self.NCdatabasename)
        xls = pd.ExcelFile(self.NCdatabasename, engine = 'openpyxl')
        df1 = pd.read_excel(xls, 'datarecord')
        fit_parameters = self.fitparameters
        normdatasavename = self.normdatasavename

        # display original image for first dbnum entry-------------------
        dbnum = self.NCdatabasenum[0]
        dbhome = df1.loc[dbnum, 'datadir']
        fname = df1.loc[dbnum, 'niftiname']
        seriesnumber = df1.loc[dbnum, 'seriesnumber']
        niiname = os.path.join(dbhome, fname)
        input_data, new_affine = i3d.load_and_scale_nifti(niiname)
        print('shape of input_data is ',np.shape(input_data))
        print('niiname = ', niiname)
        if np.ndim(input_data) == 4:
            xs,ys,zs,ts = np.shape(input_data)
            xmid = np.round(xs/2).astype(int)
            img = input_data[xmid,:,:,0]
            img = (255.*img/np.max(img)).astype(np.uint8)
            image_tk = ImageTk.PhotoImage(Image.fromarray(img))
        else:
            xs,ys,zs = np.shape(input_data)
            xmid = np.round(xs/2).astype(int)
            img = input_data[xmid,:,:]
            img = (255.*img/np.max(img)).astype(np.uint8)
            image_tk = ImageTk.PhotoImage(Image.fromarray(img))
        verticalsize = ys
        self.controller.img1d = image_tk  # keep a copy so it persists
        self.window1.configure(width=image_tk.width(), height=image_tk.height())
        self.windowdisplay1 = self.window1.create_image(0, 0, image=image_tk, anchor=tk.NW)

        inprogressfile = os.path.join(basedir, 'underconstruction.gif')
        image_tk = tk.PhotoImage('photo', file=inprogressfile)
        image_tk = image_tk.subsample(2)
        self.controller.img2d = image_tk  # keep a copy so it persists
        self.window2.configure(width=image_tk.width(), height=image_tk.height())
        self.windowdisplay2 = self.window2.create_image(0, 0, image=image_tk, anchor=tk.NW)

        inprogressfile = os.path.join(basedir, 'underconstruction.gif')
        image_tk = tk.PhotoImage('photo', file=inprogressfile)
        image_tk = image_tk.subsample(2)
        self.controller.img3d = image_tk  # keep a copy so it persists
        self.window3.configure(width=image_tk.width(), height=image_tk.height())
        self.windowdisplay3 = self.window3.create_image(0, 0, image=image_tk, anchor=tk.NW)

        time.sleep(0.1)
        #-----------end of display--------------------------------

        print('Normalization: databasename ',self.NCdatabasename)
        print('Normalization: started organizing at ', time.ctime(time.time()))

        # assume that all the data sets being normalized in a group are from the same region
        # and have the same template and anatomical region - no need to load these for each dbnum

        for nn, dbnum in enumerate(self.NCdatabasenum):
            print('NCrunclick: databasenum ',dbnum)
            dbhome = df1.loc[dbnum, 'datadir']
            fname = df1.loc[dbnum, 'niftiname']
            seriesnumber = df1.loc[dbnum, 'seriesnumber']
            normtemplatename = df1.loc[dbnum, 'normtemplatename']
            niiname = os.path.join(dbhome, fname)
            fullpath, filename = os.path.split(niiname)
            # prefix_niiname = os.path.join(fullpath,self.prefix+filename)
            tag = '_s'+str(seriesnumber)
            normdataname_full  = os.path.join(fullpath,normdatasavename+tag+'.npy')

            resolution = 1
            # template_img, regionmap_img, template_affine, anatlabels = load_templates.load_template(normtemplatename, resolution)
            template_img, regionmap_img, template_affine, anatlabels, wmmap_img, roi_map, gmwm_img = load_templates.load_template_and_masks(normtemplatename, resolution)
            # still need to write the resulting normdata file name to the database excel file

            # run the rough normalization
            print('self.roughnorm = ', self.roughnorm)
            if self.roughnorm == 1:
                # set the cursor to reflect being busy ...
                self.controller.master.config(cursor = "wait")
                self.controller.master.update()
                T, warpdata, reverse_map_image, displayrecord, imagerecord, resultsplot, result = pynormalization.run_rough_normalization_calculations(niiname, normtemplatename,
                                    template_img, fit_parameters)  # , display_window1, image_in_W1, display_window2, image_in_W2
                self.NCresult = result  # need this for manual over-ride etc.
                self.NCresult_copy = copy.deepcopy(self.NCresult)  # need this for manual over-ride etc.
                Tfine = 'none'
                norm_image_fine = 'none'
                self.controller.master.config(cursor = "")
                self.controller.master.update()

                # display results-----------------------------------------------------
                nfordisplay = len(imagerecord)
                for nf in range(nfordisplay):
                    img1 = imagerecord[nf]['img']
                    img1 = (255. * img1 / np.max(img1)).astype(np.uint8)
                    image_tk = ImageTk.PhotoImage(Image.fromarray(img1))
                    self.controller.img1d = image_tk
                    self.window1.configure(width=image_tk.width(), height=image_tk.height())
                    self.windowdisplay1 = self.window1.create_image(0, 0, image=image_tk, anchor=tk.NW)
                    time.sleep(1)

                nfordisplay = len(result)
                for nf in range(nfordisplay):
                    # draw the rectangular regions for each section
                    p0, p1, p2, p3, coords, angle, sectionsize, smallestside = self.outline_section(nf)
                    self.window1.create_line(p0[0],p0[1],p1[0],p1[1],p2[0],p2[1],p3[0],p3[1],p0[0],p0[1], fill = 'yellow', width = 2)

                self.window1.create_text(np.round(image_tk.width()/2),image_tk.height()-5,text = 'template sections mapped onto image', fill = 'white')

                display_image = imagerecord[0]['img']
                display_image = (255. * display_image / np.max(display_image)).astype(np.uint8)
                image_tk = ImageTk.PhotoImage(Image.fromarray(display_image))
                # show normalization result instead
                xs,ys,zs = np.shape(reverse_map_image)
                xmid = np.round(xs/2).astype(int)
                display_image = reverse_map_image[xmid,:,:]
                display_image = (255. * display_image / np.max(display_image)).astype(np.uint8)
                vscale = 0.5*(verticalsize/ys)
                display_imager = i3d.resize_2D(display_image, vscale)
                # if ys > 128:
                #     display_imager = i3d.resize_2D(display_image, 0.5)
                # else:
                #     display_imager = copy.deepcopy(display_image)
                image_tk = ImageTk.PhotoImage(Image.fromarray(display_imager))

                self.controller.img2d = image_tk   # keep a copy so it persists
                self.window2.configure(width=image_tk.width(), height=image_tk.height())
                self.windowdisplay2 = self.window2.create_image(0, 0, image=image_tk, anchor=tk.NW)

                self.window2.create_text(np.round(image_tk.width()/2),image_tk.height()-5,text = 'normalization result', fill = 'white')

                # show template
                xs,ys,zs = np.shape(template_img)
                xmid = np.round(xs/2).astype(int)
                display_image = template_img[xmid,:,:]
                display_image = (255. * display_image / np.max(display_image)).astype(np.uint8)
                vscale = 0.5*(verticalsize/ys)
                display_imager = i3d.resize_2D(display_image, vscale)
                # if ys > 128:
                #     display_imager = i3d.resize_2D(display_image, 0.5)
                # else:
                #     display_imager = copy.deepcopy(display_image)
                image_tk = ImageTk.PhotoImage(Image.fromarray(display_imager))
                self.controller.img3d = image_tk   # keep a copy so it persists
                self.window3.configure(width=image_tk.width(), height=image_tk.height())
                self.windowdisplay3 = self.window3.create_image(0, 0, image=image_tk, anchor=tk.NW)
                self.window3.create_text(np.round(image_tk.width()/2),image_tk.height()-5,text = 'reference template', fill = 'white')

            else:
                # if rough norm is not being run, then assume that it has already been done and the results need to be loaded
                normdata = np.load(normdataname_full, allow_pickle=True).flat[0]
                T = normdata['T']
                warpdata = normdata['warpdata']
                reverse_map_image = normdata['reverse_map_image']
                Tfine = normdata['Tfine']
                norm_image_fine = normdata['norm_image_fine']
                imagerecord = normdata['imagerecord']
                result = normdata['result']
                self.NCresult = result
                self.NCresult_copy = copy.deepcopy(self.NCresult)

                xs,ys,zs = np.shape(reverse_map_image)
                xmid = np.round(xs/2).astype(int)
                img2 = reverse_map_image[xmid,:,:]
                img2 = (255. * img2 / np.max(img2)).astype(np.uint8)
                vscale = 0.5*(verticalsize/ys)
                img2r = i3d.resize_2D(img2, vscale)
                # img2r = i3d.resize_2D(img2, 0.5)
                image_tk = ImageTk.PhotoImage(Image.fromarray(img2r))
                self.controller.img2d = image_tk   # keep a copy so it persists
                self.window2.configure(width=image_tk.width(), height=image_tk.height())
                self.windowdisplay2 = self.window2.create_image(0, 0, image=image_tk, anchor=tk.NW)

                xs,ys,zs = np.shape(template_img)
                xmid = np.round(xs/2).astype(int)
                img3 = template_img[xmid,:,:]
                img3 = (255. * img3 / np.max(img3)).astype(np.uint8)
                vscale = 0.5*(verticalsize/ys)
                img3r = i3d.resize_2D(img3, vscale)
                # img3r = i3d.resize_2D(img3, 0.5)
                image_tk = ImageTk.PhotoImage(Image.fromarray(img3r))
                self.controller.img3d = image_tk   # keep a copy so it persists
                self.window3.configure(width=image_tk.width(), height=image_tk.height())
                self.windowdisplay3 = self.window3.create_image(0, 0, image=image_tk, anchor=tk.NW)

                image_tk = self.controller.img1d
                self.window1.configure(width=image_tk.width(), height=image_tk.height())
                self.windowdisplay1 = self.window1.create_image(0, 0, image=image_tk, anchor=tk.NW)
                time.sleep(0.1)

                nfordisplay = len(result)
                for nf in range(nfordisplay):
                    # draw the rectangular regions for each section
                    p0, p1, p2, p3, coords, angle, sectionsize, smallestside = self.outline_section(nf)
                    self.window1.create_line(p0[0],p0[1],p1[0],p1[1],p2[0],p2[1],p3[0],p3[1],p0[0],p0[1], fill = 'yellow', width = 2)
                # arial6 = tkFont.Font(family='Arial', size=6)
                self.window1.create_text(np.round(image_tk.width()/2),image_tk.height()-5,text = 'template sections mapped onto image', fill = 'white')

            # manual over-ride?

            # run the normalization fine-tuning
            print('self.finetune = ', self.finetune)
            if self.finetune == 1:
                inprogressfile = os.path.join(basedir, 'underconstruction.gif')
                image_tk = tk.PhotoImage('photo', file=inprogressfile)
                image_tk = image_tk.subsample(2)
                self.controller.img2d = image_tk  # keep a copy so it persists
                self.window2.configure(width=image_tk.width(), height=image_tk.height())
                self.windowdisplay2 = self.window2.create_image(0, 0, image=image_tk, anchor=tk.NW)
                Tfine, norm_image_fine = pynormalization.py_norm_fine_tuning(reverse_map_image, template_img, T, input_type = 'normalized')

            # check the quality of the resulting normalization
            if np.ndim(norm_image_fine) >= 3:
                norm_result_image = norm_image_fine
            else:
                norm_result_image = reverse_map_image
            norm_result_image[np.isnan(norm_result_image)] = 0.0
            norm_result_image[np.isinf(norm_result_image)] = 0.0
            # dilate the roi_map
            dstruct = ndimage.generate_binary_structure(3, 3)
            roi_map2 = ndimage.binary_dilation(roi_map, structure=dstruct).astype(roi_map.dtype)
            cx,cy,cz = np.where(roi_map2)
            vimg = norm_result_image[cx,cy,cz]
            vtemp = template_img[cx,cy,cz]
            Q = np.corrcoef(vimg,vtemp)
            print('normalization quality (correlation with template) = {}'.format(Q[0,1]))

            # now write the new database values
            xls = pd.ExcelFile(self.NCdatabasename, engine = 'openpyxl')
            df1 = pd.read_excel(xls, 'datarecord')
            keylist = df1.keys()
            for kname in keylist:
                if 'Unnamed' in kname: df1.pop(kname)  # remove blank fields from the database
            # df1.pop('Unnamed: 0')
            normdataname_small = normdataname_full.replace(dbhome, '')  # need to strip off dbhome before writing the name
            df1.loc[dbnum.astype('int'), 'normdataname'] = normdataname_small[1:]
            # df1.loc[dbnum.astype('int'), 'normdataname'] = normdataname_small

            # add normalization quality to database
            if 'norm_quality' not in keylist:
                df1['norm_quality'] = 0
            df1.loc[dbnum.astype('int'), 'norm_quality'] = Q[0,1]

            # need to delete the existing sheet before writing the new version
            existing_sheet_names = xls.sheet_names
            if 'datarecord' in existing_sheet_names:
                # delete sheet - need to use openpyxl
                workbook = openpyxl.load_workbook(self.NCdatabasename)
                # std = workbook.get_sheet_by_name('datarecord')
                # workbook.remove_sheet(std)
                del workbook['datarecord']
                workbook.save(self.NCdatabasename)

            # write it to the database by appending a sheet to the excel file
            # remove old version of datarecord first
            with pd.ExcelWriter(self.NCdatabasename, engine="openpyxl", mode='a') as writer:
                df1.to_excel(writer, sheet_name='datarecord')

            normdata = {'T':T, 'Tfine':Tfine, 'warpdata':warpdata, 'reverse_map_image':reverse_map_image, 'norm_image_fine':norm_image_fine, 'template_affine':template_affine, 'imagerecord':imagerecord, 'result':result}
            np.save(normdataname_full, normdata)
            print('normalization data saved in ',normdataname_full)

            # display the resulting normalized image
            xs, ys, zs = np.shape(norm_result_image)
            xmid = np.round(xs / 2).astype(int)
            img2 = norm_result_image[xmid, :, :]
            img2 = (255. * img2 / np.max(img2)).astype(np.uint8)
            vscale = 0.5 * (verticalsize / ys)
            img2r = i3d.resize_2D(img2, vscale)
            # img2r = i3d.resize_2D(img2, 0.5)
            image_tk = ImageTk.PhotoImage(Image.fromarray(img2r))
            self.controller.img2d = image_tk  # keep a copy so it persists
            self.window2.configure(width=image_tk.width(), height=image_tk.height())
            self.windowdisplay2 = self.window2.create_image(0, 0, image=image_tk, anchor=tk.NW)

            # display the template
            xs, ys, zs = np.shape(template_img)
            xmid = np.round(xs / 2).astype(int)
            img3 = template_img[xmid, :, :]
            img3 = (255. * img3 / np.max(img3)).astype(np.uint8)
            image_tk = ImageTk.PhotoImage(Image.fromarray(img3))
            self.controller.img3d = image_tk  # keep a copy so it persists
            self.window3.configure(width=image_tk.width(), height=image_tk.height())
            self.windowdisplay3 = self.window3.create_image(0, 0, image=image_tk, anchor=tk.NW)

        print('Normalization: finished processing data ...', time.ctime(time.time()))


    # options for manual correction of rough normalization?
    # action when the button is pressed to run the manual over-ride function
    def NCmanoclick(self):
        # first get the necessary input data
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.NCdatabasename = settings['DBname']
        self.NCdatabasenum = settings['DBnum']
        # BASEdir = os.path.dirname(self.NCdatabasename)
        xls = pd.ExcelFile(self.NCdatabasename, engine = 'openpyxl')
        df1 = pd.read_excel(xls, 'datarecord')
        fit_parameters = self.fitparameters
        normdatasavename = self.normdatasavename

        # get the button click position and print it out, then quit
        if self.NCmanomode == 'OFF':
            self.NCmanomode = 'ON'
            self.button_funcid = self.window1.bind("<Button-1>",self.mouseleftclick,"+")
            self.button_funcidL = self.window1.bind("<Button-3>",self.mouserightclick,"+")
        else:
            self.NCmanomode = 'OFF'
            self.window1.unbind("<Button-1>", self.button_funcid)
            self.window1.unbind("<Button-3>", self.button_funcidL)
            self.overrideactive = False
            self.overrideangle = False
            self.overridepos = False
            self.NCresult_copy = copy.deepcopy(self.NCresult)   # refresh the copy

            # refresh the display
            nfordisplay = len(self.NCresult_copy)
            if nfordisplay > 0:
                image_tk = self.controller.img1d
                # self.window1.configure(width=image_tk.width(), height=image_tk.height())
                self.windowdisplay1 = self.window1.create_image(0, 0, image=image_tk, anchor=tk.NW)

                for nf in range(nfordisplay):
                    fillcolor = 'blue'

                    # fillcolor = 'yellow'
                    colorval1 = np.floor(255. * (nf + 1) / nfordisplay).astype(int)
                    colorval2 = 255 - np.floor(255. * (nf + 1) / nfordisplay).astype(int)
                    fillcolor = "#%02x%02x%02x" % (colorval1, colorval2, 0)

                    # draw the rectangular regions for each section
                    p0, p1, p2, p3, coords, angle, sectionsize, smallestside = self.outline_section(nf)
                    self.window1.create_line(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p0[0], p0[1],
                                             fill=fillcolor, width=1)


    def NCrecalculate_after_override(self):
        # calculate new normalization based on sections with manual over-ride applied
        # first get the necessary input data
        # use the values saved in self because these need to be up to date from the previous steps,
        # if not, then over-ride is not ready to be used

        # shut off the manual over-ride mode
        self.NCmanomode = 'OFF'
        self.window1.unbind("<Button-1>", self.button_funcid)
        self.window1.unbind("<Button-3>", self.button_funcidL)
        self.overrideactive = False
        self.overrideangle = False
        self.overridepos = False

        xls = pd.ExcelFile(self.NCdatabasename, engine = 'openpyxl')
        df1 = pd.read_excel(xls, 'datarecord')
        normdatasavename = self.normdatasavename

        # display original image for first dbnum entry-------------------
        dbnum = self.NCdatabasenum[0]
        dbhome = df1.loc[dbnum, 'datadir']
        fname = df1.loc[dbnum, 'niftiname']
        seriesnumber = df1.loc[dbnum, 'seriesnumber']
        normtemplatename = df1.loc[dbnum, 'normtemplatename']
        niiname = os.path.join(dbhome, fname)
        fullpath, filename = os.path.split(niiname)
        # prefix_niiname = os.path.join(fullpath,self.prefix+filename)
        tag = '_s' + str(seriesnumber)
        normdataname_full = os.path.join(fullpath, normdatasavename + tag + '.npy')

        input_data, new_affine = i3d.load_and_scale_nifti(niiname)
        print('shape of input_data is ',np.shape(input_data))
        print('niiname = ', niiname)
        if np.ndim(input_data) == 4:
            xs,ys,zs,ts = np.shape(input_data)
            xmid = np.round(xs/2).astype(int)
            img = input_data[xmid,:,:,0]
            img = (255.*img/np.max(img)).astype(np.uint8)
            image_tk = ImageTk.PhotoImage(Image.fromarray(img))
            input_image = input_data[:,:,:,3]
        else:
            xs,ys,zs = np.shape(input_data)
            xmid = np.round(xs/2).astype(int)
            img = input_data[xmid,:,:]
            img = (255.*img/np.max(img)).astype(np.uint8)
            image_tk = ImageTk.PhotoImage(Image.fromarray(img))
            input_image = input_data
        self.controller.img1d = image_tk  # keep a copy so it persists
        self.window1.configure(width=image_tk.width(), height=image_tk.height())
        self.windowdisplay1 = self.window1.create_image(0, 0, image=image_tk, anchor=tk.NW)
        time.sleep(0.1)
        #-----------end of display--------------------------------

        # tweak the normalization results for consistency

        nsections = len(self.NCresult_copy)
        adjusted_sections = []
        for nn in range(nsections):
            angle1 = self.NCresult_copy[nn]['angle']
            coords1 = self.NCresult_copy[nn]['coords']
            angle2 = self.NCresult[nn]['angle']
            coords2 = self.NCresult[nn]['coords']
            v1 = angle1+coords1
            v2 = angle2+coords2
            changecheck = (np.abs(v1-v2) > 0.01).any()  # if any values have changed, mark these as sections to keep where they are
            if changecheck:
                adjusted_sections.append(nn)
        adjusted_sections = np.array(adjusted_sections)

        result = copy.deepcopy(self.NCresult_copy)
        new_result = pynormalization.align_override_sections(result, adjusted_sections, niiname, normtemplatename)
        self.NCresult_copy = new_result

        #-------get the modified normalization information -------------------------------------
        self.NCresult = copy.deepcopy(self.NCresult_copy)  # lock in the changes
        result = copy.deepcopy(self.NCresult)

        T, warpdata, reverse_map_image, forward_map_image, new_result, imagerecord, displayrecord = pynormalization.py_load_modified_normalization(niiname, normtemplatename, result)
        self.NCresult = new_result  # keep a copy

        # over-write existing normalization data
        normdata_original = np.load(normdataname_full, allow_pickle=True).flat[0]
        Tfine = normdata_original['Tfine']
        norm_image_fine = normdata_original['norm_image_fine']
        # imagerecord = normdata_original['imagerecord']
        template_affine = normdata_original['template_affine']

        normdata = {'T': T, 'Tfine': Tfine, 'warpdata': warpdata, 'reverse_map_image': reverse_map_image,
                    'norm_image_fine': norm_image_fine, 'template_affine': template_affine, 'imagerecord': imagerecord,
                    'result': result}
        np.save(normdataname_full, normdata)
        print('normalization data saved in ', normdataname_full)

        # display results-----------------------------------------------------
        nfordisplay = len(imagerecord)
        for nf in range(nfordisplay):
            img1 = imagerecord[nf]['img']
            img1 = (255. * img1 / np.max(img1)).astype(np.uint8)
            image_tk = ImageTk.PhotoImage(Image.fromarray(img1))
            self.controller.img1d = image_tk
            self.window1.configure(width=image_tk.width(), height=image_tk.height())
            self.windowdisplay1 = self.window1.create_image(0, 0, image=image_tk, anchor=tk.NW)
            time.sleep(1)

        nfordisplay = len(result)
        for nf in range(nfordisplay):
            # draw the rectangular regions for each section
            p0, p1, p2, p3, coords, angle, sectionsize, smallestside = self.outline_section(nf)
            self.window1.create_line(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p0[0], p0[1],
                                     fill='yellow', width=2)

        self.window1.create_text(np.round(image_tk.width() / 2), image_tk.height() - 5,
                                 text='template sections mapped onto image', fill='white')

        display_image = imagerecord[0]['img']
        display_image = (255. * display_image / np.max(display_image)).astype(np.uint8)
        image_tk = ImageTk.PhotoImage(Image.fromarray(display_image))
        # show normalization result instead
        xs, ys, zs = np.shape(reverse_map_image)
        xmid = np.round(xs / 2).astype(int)
        display_image = reverse_map_image[xmid, :, :]
        display_image = (255. * display_image / np.max(display_image)).astype(np.uint8)
        vscale = 0.5 * (verticalsize / ys)
        display_imager = i3d.resize_2D(display_image, vscale)
        # if ys > 128:
        #     display_imager = i3d.resize_2D(display_image, 0.5)
        # else:
        #     display_imager = copy.deepcopy(display_image)
        image_tk = ImageTk.PhotoImage(Image.fromarray(display_imager))

        self.controller.img2d = image_tk  # keep a copy so it persists
        self.window2.configure(width=image_tk.width(), height=image_tk.height())
        self.windowdisplay2 = self.window2.create_image(0, 0, image=image_tk, anchor=tk.NW)
        self.window2.create_text(np.round(image_tk.width() / 2), image_tk.height() - 5, text='normalization result',
                                 fill='white')

        # show template image
        resolution = 1
        template_img, regionmap_img, template_affine, anatlabels, wmmap_img, roi_map, gmwm_img = load_templates.load_template_and_masks(normtemplatename, resolution)
        xs, ys, zs = np.shape(template_img)
        xmid = np.round(xs / 2).astype(int)
        display_image = template_img[xmid, :, :]
        display_image = (255. * display_image / np.max(display_image)).astype(np.uint8)
        vscale = 0.5 * (verticalsize / ys)
        display_imager = i3d.resize_2D(display_image, vscale)
        # if ys > 128:
        #     display_imager = i3d.resize_2D(display_image, 0.5)
        # else:
        #     display_imager = copy.deepcopy(display_image)
        image_tk = ImageTk.PhotoImage(Image.fromarray(display_imager))

        self.controller.img3d = image_tk  # keep a copy so it persists
        self.window3.configure(width=image_tk.width(), height=image_tk.height())
        self.windowdisplay3 = self.window3.create_image(0, 0, image=image_tk, anchor=tk.NW)
        self.window3.create_text(np.round(image_tk.width() / 2), image_tk.height() - 5, text='template',
                                 fill='white')


# --------------------Calculate Normalization Parameters for Brain Data FRAME---------------------------------------------------------------
# Definition of the frame that will have inputs and options for normalizing NIfTI format data
class NCbrainFrame:
    # initialize the values, keeping track of the frame this definition works on (parent), and
    # also the main window containing that frame (controller)
    def __init__(self, parent, controller):
        parent.configure(relief='raised', bd=5, highlightcolor=fgcol3)
        self.parent = parent
        self.controller = controller
        self.NCresult = []

        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        self.normdatasavename = settings['NCsavename']  # default prefix value
        self.fitparameters = settings['NCBparameters']  # [(10000, 1000, 100),(3.0, 1.0, 0.0),(4, 2, 1)]  # default prefix value
        self.iters = self.fitparameters[0]
        self.sigmas = self.fitparameters[1]
        self.factors = self.fitparameters[2]

        self.braintemplatename = settings['braintemplate']

        # initialize some values
        self.NCdatabasename = settings['DBname']
        self.NCdbnum = settings['DBnum']

        if os.path.isfile(self.NCdatabasename):
            xls = pd.ExcelFile(self.NCdatabasename, engine='openpyxl')
            df1 = pd.read_excel(xls, 'datarecord')
            self.normtemplatename = df1.loc[self.NCdbnum[0], 'normtemplatename']
        else:
            self.normtemplatename = 'notdefined'

        # check we are in the correct mode
        if self.normtemplatename.lower() != 'brain':
            print('ERROR:  expecting brain data for normalization method ...')
            print('database number {} in database {}'.format(self.NCdbnum[0],self.NCdatabasename))
            print('normalization template is indicated as: {}'.format(self.normtemplatename))

        self.NCtemplatelabel = tk.Label(self.parent, text='Normalizing region: ' + self.normtemplatename, fg='gray', font = labelfont)
        self.NCtemplatelabel.grid(row=0, column=2, sticky='W')

        # put some text as a place-holder
        self.NClabel1 = tk.Label(self.parent, text="1) Calculate normalization\nparameters", fg='gray')
        self.NClabel1.grid(row=0, column=0, sticky='W')
        self.NClabel2 = tk.Label(self.parent, text="2) Save for next steps", fg='gray')
        self.NClabel2.grid(row=1, column=0, sticky='W')

        # now define an Entry box so that the user can indicate the prefix name of the data to normalize
        # give it a label first
        self.NCinfo1 = tk.Label(self.parent, text="Save name base:", font = labelfont)
        self.NCinfo1.grid(row=1, column=1, sticky='E')

        # create the Entry box, and put it next to the label
        self.NCsavename = tk.Entry(self.parent, width=20, bg="white")
        self.NCsavename.grid(row=1, column=2, sticky="W")
        self.NCsavename.insert(1, self.normdatasavename)

        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.NCsavenamesubmit = tk.Button(self.parent, text="Submit", width=smallbuttonsize, bg=fgcol2, fg='black', command=self.NCsavenamesubmit, relief='raised', bd=5, font = widgetfont, highlightbackground = widgetbg)
        self.NCsavenamesubmit.grid(row=1, column=3)

        # for brain normalization, need parameters ...
        #  template name
        self.NCinfo2 = tk.Label(self.parent, text="Brain template:", font = labelfont)
        self.NCinfo2.grid(row=2, column=1, sticky='E')

        # create the Entry box, and put it next to the label
        self.NCBtemplate = tk.Entry(self.parent, width=20, bg="white")
        self.NCBtemplate.grid(row=2, column=2, sticky="W")
        self.NCBtemplate.insert(1, self.braintemplatename)

        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.NCBtemplatename = tk.Button(self.parent, text="Submit", width=smallbuttonsize, bg=fgcol2, fg=fgletter2, font = widgetfont, command=self.NCBtemplatenamesubmit, relief='raised', bd=5, highlightbackground = widgetbg)
        self.NCBtemplatename.grid(row=2, column=3)

        # the entry box needs a "browse" button
        self.NCBtemplatebrowse = tk.Button(self.parent, text="Browse", width=smallbuttonsize, bg=fgcol2, fg=fgletter2, font = widgetfont, command=self.NCBtemplatenamebrowse, relief='raised', bd=5, highlightbackground = widgetbg)
        self.NCBtemplatebrowse.grid(row=2, column=4)

        # for brain normalization, need parameters ...
        #  level_iters = [10000, 1000, 100]
        #  sigmas = [3.0, 1.0, 0.0]
        #  factors = [4, 2, 1]

        self.NCinfo3 = tk.Label(self.parent, text="iterations per level:", font = labelfont)
        self.NCinfo3.grid(row=3, column=1, sticky='E')
        # create the Entry box, and put it next to the label
        self.NCBiters = tk.Entry(self.parent, width=20, bg="white")
        self.NCBiters.grid(row=3, column=2, sticky="W")
        self.NCBiters.insert(1, self.iters)
        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.NCBitersubmitbutton = tk.Button(self.parent, text="Submit", width=smallbuttonsize, bg=fgcol2, fg=fgletter2, font = widgetfont, command=self.NCBitersubmit, relief='raised', bd=5, highlightbackground = widgetbg)
        self.NCBitersubmitbutton.grid(row=3, column=3)
        # need an option to restore default values
        self.NCBiterdefaults = tk.Button(self.parent, text="Restore", width=smallbuttonsize, bg=fgcol2, fg=fgletter2, font = widgetfont, command=self.NCBiterrestoredefault, relief='raised', bd=5, highlightbackground = widgetbg)
        self.NCBiterdefaults.grid(row=3, column=4)


        self.NCinfo4 = tk.Label(self.parent, text="smoothing per level:", font = labelfont)
        self.NCinfo4.grid(row=4, column=1, sticky='E')
        # create the Entry box, and put it next to the label
        self.NCBsigmas = tk.Entry(self.parent, width=20, bg="white")
        self.NCBsigmas.grid(row=4, column=2, sticky="W")
        self.NCBsigmas.insert(1, self.sigmas)
        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.NCBsigmasubmitbutton = tk.Button(self.parent, text="Submit", width=smallbuttonsize, bg=fgcol2, fg=fgletter2, font = widgetfont, command=self.NCBsigmassubmit, relief='raised', bd=5, highlightbackground = widgetbg)
        self.NCBsigmasubmitbutton.grid(row=4, column=3)
        # need an option to restore default values
        self.NCBsigmadefaults = tk.Button(self.parent, text="Restore", width=smallbuttonsize, bg=fgcol2, fg=fgletter2, font = widgetfont, command=self.NCBsigmarestoredefault, relief='raised', bd=5, highlightbackground = widgetbg)
        self.NCBsigmadefaults.grid(row=4, column=4)

        self.NCinfo5 = tk.Label(self.parent, text="divide per level:", font = labelfont)
        self.NCinfo5.grid(row=5, column=1, sticky='E')
        # create the Entry box, and put it next to the label
        self.NCBfactors = tk.Entry(self.parent, width=20, bg="white")
        self.NCBfactors.grid(row=5, column=2, sticky="W")
        self.NCBfactors.insert(1, self.factors)
        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.NCBfactorssubmitbutton = tk.Button(self.parent, text="Submit", width=smallbuttonsize, bg=fgcol2, fg=fgletter2, font = widgetfont, command=self.NCBfactorssubmit, relief='raised', bd=5, highlightbackground = widgetbg)
        self.NCBfactorssubmitbutton.grid(row=5, column=3)
        # need an option to restore default values
        self.NCBfactordefaults = tk.Button(self.parent, text="Restore", width=smallbuttonsize, bg=fgcol2, fg=fgletter2, font = widgetfont, command=self.NCBfactorrestoredefault, relief='raised', bd=5, highlightbackground = widgetbg)
        self.NCBfactordefaults.grid(row=5, column=4)

        # button to call the normalization program
        self.NCBrun = tk.Button(self.parent, text='Calculate Normalization', width=bigbigbuttonsize, bg=fgcol1, fg=fgletter1, font = widgetfont, command=self.NCBrunclick, relief='raised', bd=5, highlightbackground = widgetbg)
        self.NCBrun.grid(row=6, column=1)

        img1 = tk.PhotoImage(file=os.path.join(basedir, 'smily.gif'))
        controller.img1d = img1  # need to keep a copy so it is not cleared from memory
        self.window1 = tk.Canvas(master=self.parent, width=img1.width(), height=img1.height(), bg='black')
        self.window1.grid(row=7, column=0, rowspan=3, columnspan=2, sticky='NW')
        self.windowdisplay1 = self.window1.create_image(0, 0, image=img1, anchor=tk.NW)

        img2 = tk.PhotoImage(file=os.path.join(basedir, 'smily.gif'))
        img2 = img2.subsample(2)
        controller.img2d = img2  # need to keep a copy so it is not cleared from memory
        self.window2 = tk.Canvas(master=self.parent, width=img2.width(), height=img2.height(), bg='black')
        self.window2.grid(row=7, column=2, rowspan=2, columnspan=2, sticky='NW')
        self.windowdisplay2 = self.window2.create_image(0, 0, image=img2, anchor=tk.NW)

        img3 = tk.PhotoImage(file=os.path.join(basedir, 'smily.gif'))
        img3 = img3.subsample(2)
        controller.img3d = img3  # need to keep a copy so it is not cleared from memory
        self.window3 = tk.Canvas(master=self.parent, width=img3.width(), height=img3.height(), bg='black')
        self.window3.grid(row=9, column=2, rowspan=2, columnspan=2, sticky='NW')
        self.windowdisplay3 = self.window3.create_image(0, 0, image=img3, anchor=tk.NW)


    # action when the button is pressed to submit the DB entry number list
    def NCsavenamesubmit(self):
        entered_text = self.NCsavename.get()  # collect the text from the text entry box
        # remove any spaces
        entered_text = re.sub('\ +', '', entered_text)
        print(entered_text)

        # update the text in the box, in case it has changed
        self.normdatasavename = entered_text

        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        settings['NCsavename'] = entered_text
        np.save(settingsfile, settings)

        return self

    # define functions for defining the brain template name for normalization
    # the template files are all stored in the "braintemplates" folder
    def NCBtemplatenamebrowse(self):
        workingdir = os.path.dirname(os.path.realpath(__file__))
        brain_templates_folder = os.path.join(workingdir, 'braintemplates')
        basedir = os.getcwd()
        os.chdir(brain_templates_folder)

        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        templatename_full =  tkf.askopenfilename(title = "Select template file",
                        filetypes=(("nii files", "*.nii","*.gz"), ("all files", "*.*")))
        os.chdir(basedir)  # put it back
        p, templatename = os.path.split(templatename_full)
        print('templatename = ',templatename)
        # save the selected file name in the settings
        settings['braintemplate'] = templatename
        self.braintemplatename = templatename
        # write the result to the label box for display
        self.NCBtemplate.delete(0,'end')
        self.NCBtemplate.insert(1, self.braintemplatename)

        # save the updated settings file again
        np.save(settingsfile,settings)

    # action when the button is pressed to submit the slice information
    def NCBtemplatenamesubmit(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        templatename = self.NCBtemplate.get()
        settings['braintemplate'] = templatename
        np.save(settingsfile, settings)
        # update the text in the box, in case it has changed
        self.braintemplatename = templatename
        return self


    def NCBparsenumlist(self, entered_text):
        # need to make sure we are working with numbers, not text
        # first, replace any double spaces with single spaces, and then replace spaces with commas
        entered_text = re.sub('\ +', ',', entered_text)
        entered_text = re.sub('\,\,+', ',', entered_text)
        # remove any leading or trailing commas
        if entered_text[0] == ',': entered_text = entered_text[1:]
        if entered_text[-1] == ',': entered_text = entered_text[:-1]
        entered_values = list(np.fromstring(entered_text, dtype=int, sep=','))
        return entered_values


    def NCBitersubmit(self):
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        entered_text = self.NCBiters.get()  # collect the text from the text entry box

        # parse the entered text
        entered_vals = self.NCBparsenumlist(entered_text)
        print(entered_vals)
        self.iters = entered_vals

        # update the values in the box, in case they have changed
        self.fitparameters[0] = self.iters
        settings['NCBparameters'] = self.fitparameters
        np.save(settingsfile, settings)
        return self

    def NCBsigmassubmit(self):
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        entered_text = self.NCBsigmas.get()  # collect the text from the text entry box

        # parse the entered text
        entered_vals = self.NCBparsenumlist(entered_text)
        print(entered_vals)

        # update the values in the box, in case they have changed
        self.sigmas = entered_vals

        self.fitparameters[1] = self.sigmas
        settings['NCBparameters'] = self.fitparameters
        np.save(settingsfile, settings)
        return self

    def NCBfactorssubmit(self):
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        entered_text = self.NCBfactors.get()  # collect the text from the text entry box
        # parse the entered text
        entered_vals = self.NCBparsenumlist(entered_text)
        print(entered_vals)
        # update the values in the box, in case they have changed
        self.factors = entered_vals
        self.fitparameters[2] = self.factors
        settings['NCBparameters'] = self.fitparameters
        np.save(settingsfile, settings)
        return self


    def NCBiterrestoredefault(self):
        # for brain normalization, need parameters ...
        #  level_iters = [10000, 1000, 100]
        #  sigmas = [3.0, 1.0, 0.0]
        #  factors = [4, 2, 1]
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        self.iters = (10000, 1000, 100)
        self.fitparameters[0] = self.iters
        self.NCBiters.delete(0, 'end')
        self.NCBiters.insert(1, self.iters)
        settings['NCBparameters'] = self.fitparameters
        np.save(settingsfile, settings)
        return self

    def NCBsigmarestoredefault(self):
        # for brain normalization, need parameters ...
        #  level_iters = [10000, 1000, 100]
        #  sigmas = [3.0, 1.0, 0.0]
        #  factors = [4, 2, 1]
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        self.sigmas = (3.0, 1.0, 0.0)
        self.fitparameters[1] = self.sigmas
        self.NCBsigmas.delete(0, 'end')
        self.NCBsigmas.insert(1, self.sigmas)
        settings['NCBparameters'] = self.fitparameters
        np.save(settingsfile, settings)
        return self

    def NCBfactorrestoredefault(self):
        # for brain normalization, need parameters ...
        #  level_iters = [10000, 1000, 100]
        #  sigmas = [3.0, 1.0, 0.0]
        #  factors = [4, 2, 1]
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        self.factors = (4, 2, 1)
        self.fitparameters[2] = self.factors
        self.NCBfactors.delete(0, 'end')
        self.NCBfactors.insert(1, self.factors)
        settings['NCBparameters'] = self.fitparameters
        np.save(settingsfile, settings)
        return self


    def NCBrunclick(self):
        # first get the necessary input data
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        self.NCdatabasename = settings['DBname']
        self.NCdatabasenum = settings['DBnum']
        # BASEdir = os.path.dirname(self.NCdatabasename)
        xls = pd.ExcelFile(self.NCdatabasename, engine='openpyxl')
        df1 = pd.read_excel(xls, 'datarecord')

        normdatasavename = self.normdatasavename
        iters = self.iters
        sigmas = self.sigmas
        factors = self.factors
        braintemplatename = self.braintemplatename

        # load the brain template
        workingdir = os.path.dirname(os.path.realpath(__file__))
        brain_template_folder = os.path.join(workingdir, 'braintemplates')
        template_filename = os.path.join(brain_template_folder, braintemplatename)
        # ref_data, ref_affine = i3d.load_and_scale_nifti(template_filename)   # also scales to 1 mm cubic
        input_ref = nib.load(template_filename)
        ref_affine = input_ref.affine
        ref_hdr = input_ref.header
        ref_data = input_ref.get_fdata()
        ref_data = ref_data / np.max(ref_data)

        # display original image for first dbnum entry-------------------
        dbnum = self.NCdatabasenum[0]
        dbhome = df1.loc[dbnum, 'datadir']
        fname = df1.loc[dbnum, 'niftiname']
        seriesnumber = df1.loc[dbnum, 'seriesnumber']
        niiname = os.path.join(dbhome, fname)

        # input_data, new_affine = i3d.load_and_scale_nifti(niiname)  # this also scales to 1 mm cubic voxels
        input_img = nib.load(niiname)
        new_affine = input_img.affine
        input_hdr = input_img.header
        input_data = input_img.get_fdata()
        input_data = input_data / np.max(input_data)

        print('shape of input_data is ', np.shape(input_data))
        print('niiname = ', niiname)
        if np.ndim(input_data) == 4:
            xs, ys, zs, ts = np.shape(input_data)
            zmid = np.round(zs / 2).astype(int)
            img = input_data[:, :, zmid, 0]
            img = (255. * img / np.max(img)).astype(np.uint8)
            image_tk = ImageTk.PhotoImage(Image.fromarray(img))
        else:
            xs, ys, zs = np.shape(input_data)
            zmid = np.round(zs / 2).astype(int)
            img = input_data[:, :, zmid]
            img = (255. * img / np.max(img)).astype(np.uint8)
            image_tk = ImageTk.PhotoImage(Image.fromarray(img))
        self.controller.img1d = image_tk  # keep a copy so it persists
        self.window1.configure(width=image_tk.width(), height=image_tk.height())
        self.windowdisplay1 = self.window1.create_image(0, 0, image=image_tk, anchor=tk.NW)

        inprogressfile = os.path.join(basedir, 'underconstruction.gif')
        image_tk = tk.PhotoImage('photo', file=inprogressfile)
        image_tk = image_tk.subsample(2)
        self.controller.img2d = image_tk  # keep a copy so it persists
        self.window2.configure(width=image_tk.width(), height=image_tk.height())
        self.windowdisplay2 = self.window2.create_image(0, 0, image=image_tk, anchor=tk.NW)

        inprogressfile = os.path.join(basedir, 'underconstruction.gif')
        image_tk = tk.PhotoImage('photo', file=inprogressfile)
        image_tk = image_tk.subsample(2)
        self.controller.img3d = image_tk  # keep a copy so it persists
        self.window3.configure(width=image_tk.width(), height=image_tk.height())
        self.windowdisplay3 = self.window3.create_image(0, 0, image=image_tk, anchor=tk.NW)

        time.sleep(0.1)
        # -----------end of display--------------------------------

        print('Normalization: databasename ', self.NCdatabasename)
        print('Normalization: started organizing at ', time.ctime(time.time()))

        # assume that all the data sets being normalized in a group are from the same region
        # and have the same template and anatomical region - no need to load these for each dbnum

        for nn, dbnum in enumerate(self.NCdatabasenum):
            print('NCrunclick: databasenum ', dbnum)
            dbhome = df1.loc[dbnum, 'datadir']
            fname = df1.loc[dbnum, 'niftiname']
            seriesnumber = df1.loc[dbnum, 'seriesnumber']
            normtemplatename = df1.loc[dbnum, 'normtemplatename']
            niiname = os.path.join(dbhome, fname)
            fullpath, filename = os.path.split(niiname)
            # prefix_niiname = os.path.join(fullpath,self.prefix+filename)
            tag = '_s' + str(seriesnumber)
            normdataname_full = os.path.join(fullpath, normdatasavename + tag + '.npy')

            # load the nifti data
            # input_datar, affiner = i3d.load_and_scale_nifti(niiname)
            input_img = nib.load(niiname)
            affiner = input_img.affine
            input_hdr = input_img.header
            input_datar = input_img.get_fdata()
            input_datar = input_datar / np.max(input_datar)

            if np.ndim(input_datar) > 3:
                x, y, z, t = np.shape(input_datar)
                if t > 3:
                    t0 = 3
                else:
                    t0 = 0
                input_image = input_datar[:, :, :, t0]
            else:
                x, y, z = np.shape(input_datar)
                input_image = input_datar
            input_datar = []  # clear it from memory

            # run the normalization
            print('starting normalization calculation ....')
            # set the cursor to reflect being busy ...
            self.controller.master.config(cursor="wait")
            self.controller.master.update()
            norm_brain_img, norm_brain_affine = pybrainregistration.dipy_compute_brain_normalization(input_image, affiner, ref_data, ref_affine, iters, sigmas, factors, nbins=32)
            self.controller.master.config(cursor="")
            self.controller.master.update()
            print('finished normalization calculation ....')
            # save norm_brain_affine for later use...
            np.save(normdataname_full, {'norm_affine_transformation': norm_brain_affine, 'output_affine':ref_affine})
            self.NCresult = norm_brain_img

            # display results-----------------------------------------------------
            xs,ys,zs = np.shape(input_image)
            zmid = np.floor(zs/2).astype(int)
            img1 = input_image[:,:,zmid]
            img1 = (255. * img1 / np.max(img1)).astype(np.uint8)
            image_tk = ImageTk.PhotoImage(Image.fromarray(img1))
            self.controller.img1d = image_tk
            self.window1.configure(width=image_tk.width(), height=image_tk.height())
            self.windowdisplay1 = self.window1.create_image(0, 0, image=image_tk, anchor=tk.NW)
            time.sleep(1)
            self.window1.create_text(np.round(image_tk.width() / 2), image_tk.height() - 5, text='original image', fill='white')

            xs,ys,zs = np.shape(norm_brain_img)
            zmid = np.floor(zs/2).astype(int)
            img2 = norm_brain_img[:,:,zmid]
            img2= (255. * img2 / np.max(img2)).astype(np.uint8)
            vscale = 0.5 * (verticalsize / ys)
            img2r = i3d.resize_2D(img2, vscale)
            # img2r = i3d.resize_2D(img2, 0.5)
            image_tk = ImageTk.PhotoImage(Image.fromarray(img2r))
            self.controller.img2d = image_tk
            self.window2.configure(width=image_tk.width(), height=image_tk.height())
            self.windowdisplay2= self.window2.create_image(0, 0, image=image_tk, anchor=tk.NW)
            time.sleep(1)
            self.window2.create_text(np.round(image_tk.width() / 2), image_tk.height() - 5, text='normalization result', fill='white')

            xs,ys,zs = np.shape(ref_data)
            zmid = np.floor(zs/2).astype(int)
            img3 = ref_data[:,:,zmid]
            img3 = (255. * img3 / np.max(img3)).astype(np.uint8)
            vscale = 0.5 * (verticalsize / ys)
            img3r = i3d.resize_2D(img3, vscale)
            # img3r = i3d.resize_2D(img3, 0.5)
            image_tk = ImageTk.PhotoImage(Image.fromarray(img3r))
            self.controller.img3d = image_tk
            self.window3.configure(width=image_tk.width(), height=image_tk.height())
            self.windowdisplay3 = self.window3.create_image(0, 0, image=image_tk, anchor=tk.NW)
            time.sleep(1)
            self.window3.create_text(np.round(image_tk.width() / 2), image_tk.height() - 5, text='normalization template', fill='white')

            # check the quality of the resulting normalization
            norm_result_image = copy.deepcopy(norm_brain_img)
            norm_result_image[np.isnan(norm_result_image)] = 0.0
            norm_result_image[np.isinf(norm_result_image)] = 0.0

            Q = np.corrcoef(np.ndarray.flatten(norm_result_image), np.ndarray.flatten(ref_data))
            print('normalization quality (correlation with template) = {}'.format(Q[0, 1]))

            # now write the new database values
            xls = pd.ExcelFile(self.NCdatabasename, engine='openpyxl')
            df1 = pd.read_excel(xls, 'datarecord')
            keylist = df1.keys()
            for kname in keylist:
                if 'Unnamed' in kname: df1.pop(kname)  # remove blank fields from the database
            normdataname_small = normdataname_full.replace(dbhome, '')  # need to strip off dbhome before writing the name
            df1.loc[dbnum, 'normdataname'] = normdataname_small[1:]
            # df1.loc[dbnum, 'normdataname'] = normdataname_small

            # add normalization quality to database
            if 'norm_quality' not in keylist:
                df1['norm_quality'] = 0
            df1.loc[dbnum, 'norm_quality'] = Q[0, 1]

            # need to delete the existing sheet before writing the new version
            existing_sheet_names = xls.sheet_names
            if 'datarecord' in existing_sheet_names:
                # delete sheet - need to use openpyxl
                workbook = openpyxl.load_workbook(self.NCdatabasename)
                # std = workbook.get_sheet_by_name('datarecord')
                # workbook.remove_sheet(std)
                del workbook['datarecord']
                workbook.save(self.NCdatabasename)

            # write it to the database by appending a sheet to the excel file
            # remove old version of datarecord first
            with pd.ExcelWriter(self.NCdatabasename, engine="openpyxl", mode='a') as writer:
                df1.to_excel(writer, sheet_name='datarecord')

            print('normalization data saved in ', normdataname_full)

        print('Normalization: finished processing data ...', time.ctime(time.time()))


#--------------------Image Pre-Processing FRAME---------------------------------------------------------------
# Definition of the frame that will have inputs and options for co-registering NIfTI format data
# this should become the preprocessing frame, and include coregistration, applying normalization, slice-timing correction
# smoothing, basis set definitions for GLM fit and noise modeling, data cleaning, and cluster definition
#
class PPFrame:
    # initialize the values, keeping track of the frame this definition works on (parent), and
    # also the main window containing that frame (controller)
    def __init__(self, parent, controller):
        parent.configure(relief='raised', bd = 5, highlightcolor = fgcol3)
        self.parent = parent
        self.controller = controller

        # initialize some values
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.PPdatabasename = settings['DBname']
        self.prefix = settings['CRprefix']   # default prefix value
        self.coreg_choice = settings['coreg_choice']
        self.slicetime_choice = settings['slicetime_choice']
        self.norm_choice = settings['norm_choice']
        self.smooth_choice = settings['smooth_choice']
        self.define_choice = settings['define_choice']
        self.clean_choice = settings['clean_choice']

        self.sliceorder = settings['sliceorder']
        self.refslice = settings['refslice']
        self.sliceaxis = settings['sliceaxis']
        self.smoothwidth = settings['smoothwidth']

        # put some text as a place-holder
        self.PPlabel1 = tk.Label(self.parent, text = "1) Select pre-processing options", fg = 'gray')
        self.PPlabel1.grid(row=0,column=0, sticky='W')
        self.PPlabel2 = tk.Label(self.parent, text = "2) Data name prefixes indicate\nthe applied processing", fg = 'gray')
        self.PPlabel2.grid(row=1,column=0, sticky='W')

        # now define entry boxes so that the user can indicate the slice timing parameters
        # give the group of boxes a label first
        srow = 0;  scol = 1;
        self.PPinfo1 = tk.Label(self.parent, text = "Slice timing information:", font = labelfont)
        self.PPinfo1.grid(row=srow,column=scol, columnspan = 4, sticky='NSEW')

        orderoptions = {'Inc,Alt,Odd','Inc,Alt,Even','Dec,Alt,N','Dec,Alt,N-1','Inc,Seq.','Dec,Seq.'}

        self.PPslicelabel1 = tk.Label(self.parent, text="Slice Order:", font = labelfont).grid(row=1, column=1, sticky='E')
        self.sliceorder_var = tk.StringVar()
        self.sliceorder_var.set(self.sliceorder)
        self.sliceorder_opt = tk.OptionMenu(self.parent, self.sliceorder_var, *orderoptions, command = self.PPsliceorderchoice)
        self.sliceorder_opt.config(bg=bgcol)
        self.sliceorder_opt.grid(row=srow+1, column=scol+1, sticky='EW')
        # tk.CreateToolTip(self.sliceorder_opt, text='Inc,Alt,Odd = 1,3,5...2,4,6...\n'
        #                                            'Inc,Alt,Even = 2,4,6...1,3,5...\n'
        #                                            'Dec,Alt,N = N,N-2,...N-1,N-3,...\n'
        #                                            'Dec,Alt,N-1 = N-1,N-3,...N,N-2,...\n'
        #                                            'Inc,Seq. = 1,2,3,...\n'
        #                                            'Dec,Seq. = N,N-1,N-2...\n')

        self.PPslicelabel2 = tk.Label(self.parent, text = "Ref. Slice:", font = labelfont)
        self.PPslicelabel2.grid(row=srow+1, column=scol+2, sticky='E')
        # create entry box, and put it next to the label
        self.PPrefslice = tk.Entry(self.parent, width = 8, bg="white")
        self.PPrefslice.grid(row=srow+1, column=scol+3, sticky = "W")
        self.PPrefslice.insert(0,self.refslice)

        self.PPslicelabel3 = tk.Label(self.parent, text = "Slice Axis:", font = labelfont)
        self.PPslicelabel3.grid(row=srow+1, column=scol+4, sticky='E')
        # create entry box, and put it next to the label
        self.PPsliceaxis = tk.Entry(self.parent, width = 8, bg="white")
        self.PPsliceaxis.grid(row=srow+1, column=scol+5, sticky = "W")
        self.PPsliceaxis.insert(0,self.sliceaxis)

        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.PPslicesubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.PPslicesubmit, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.PPslicesubmit.grid(row=srow+1, column=scol+6, columnspan = 4)
        ttk.Separator(self.parent).grid(row=srow+2, column=scol, columnspan=6, sticky="nswe", padx=2, pady=5)


        # now define entry boxes so that the user can indicate the smoothing parameters
        # give the group of boxes a label first
        srow = 3; scol = 1;
        self.PPinfo2 = tk.Label(self.parent, text = "Smoothing Width:", font = labelfont)
        self.PPinfo2.grid(row=srow, column=scol+1, sticky='NSEW')
        self.PPsmoothing = tk.Entry(self.parent, width = 8, bg="white")
        self.PPsmoothing.grid(row=srow, column=scol+2, sticky = "W")
        self.PPsmoothing.insert(0,self.smoothwidth)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.PPsmoothsubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.PPsmoothsubmit, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.PPsmoothsubmit.grid(row=srow, column=scol+3)
        ttk.Separator(self.parent).grid(row=srow+1, column=scol, columnspan=6, sticky="nswe", padx=2, pady=5)

        # setup the choices for which pre-processing steps to run
        # create pull-down lists
        options = {'Yes.', '.No.', 'Done'}
        srow = 5;  scol = 1;   # positions to begin this set of checkboxes (to make it easy to move later)
        # coregistration
        self.coreg_label = tk.Label(self.parent, text="Coregister:", font = labelfont).grid(row=srow, column=scol, sticky='E')
        self.coreg_var = tk.StringVar()
        self.coreg_var.set(self.coreg_choice)
        self.coreg_opt = tk.OptionMenu(self.parent, self.coreg_var, *options, command = self.PPcoregchoice)
        self.coreg_opt.config(bg=bgcol)
        self.coreg_opt.grid(row=srow, column=scol+1, sticky='EW')
        # slice-time correction
        self.slicetime_label = tk.Label(self.parent, text="Slice-timing:", font = labelfont).grid(row=srow, column=scol+2, sticky='E')
        self.slicetime_var = tk.StringVar()
        self.slicetime_var.set(self.slicetime_choice)
        self.slicetime_opt = tk.OptionMenu(self.parent, self.slicetime_var, *options, command = self.PPslicetimechoice)
        self.slicetime_opt.config(bg=bgcol)
        self.slicetime_opt.grid(row=srow, column=scol+3, sticky='EW')
        # apply normalization
        self.norm_label = tk.Label(self.parent, text="Normalize:", font = labelfont).grid(row=srow, column=scol+4, sticky='E')
        self.norm_var = tk.StringVar()
        self.norm_var.set(self.norm_choice)
        self.norm_opt = tk.OptionMenu(self.parent, self.norm_var, *options, command = self.PPnormchoice)
        self.norm_opt.config(bg=bgcol)
        self.norm_opt.grid(row=srow, column=scol+5, sticky='EW')
        # apply smoothing
        self.smooth_label = tk.Label(self.parent, text="Smooth:", font = labelfont).grid(row=srow+1, column=scol, sticky='E')
        self.smooth_var = tk.StringVar()
        self.smooth_var.set(self.smooth_choice)
        self.norm_opt = tk.OptionMenu(self.parent, self.smooth_var, *options, command=self.PPsmoothchoice)
        self.norm_opt.config(bg=bgcol)
        self.norm_opt.grid(row=srow+1, column=scol + 1, sticky='EW')
        # define basis sets
        self.define_label = tk.Label(self.parent, text="Define basis sets:", font = labelfont).grid(row=srow+1, column=scol+2, sticky='E')
        self.define_var = tk.StringVar()
        self.define_var.set(self.define_choice)
        self.define_opt = tk.OptionMenu(self.parent, self.define_var, *options, command=self.PPdefinechoice)
        self.define_opt.config(bg=bgcol)
        self.define_opt.grid(row=srow+1, column=scol+3, sticky='EW')
        # clean the data
        self.clean_label = tk.Label(self.parent, text="Clean data:", font = labelfont).grid(row=srow+1, column=scol+4, sticky='E')
        self.clean_var = tk.StringVar()
        self.clean_var.set(self.clean_choice)
        self.clean_opt = tk.OptionMenu(self.parent, self.clean_var, *options, command=self.PPcleanchoice)
        self.clean_opt.config(bg=bgcol)
        self.clean_opt.grid(row=srow+1, column=scol+5, sticky='EW')

        # button to run the selected-preprocessing
        self.PPrun = tk.Button(self.parent, text = 'Process Data', width = bigbuttonsize, bg = fgcol1, fg = fgletter1 , command = self.PPrunclick, font = widgetfont, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.PPrun.grid(row = srow+2, column = 1, columnspan = 2)

        # text to show the resulting prefix for the data that will be created
        self.predictedprefixtext = tk.StringVar()
        prefix_list = pypreprocess.setprefixlist(settingsfile)
        self.predictedprefixtext.set(prefix_list[-1])
        self.PPprefixdisplaytitle = tk.Label(self.parent, text = "output nifti prefix: ", fg = 'gray', font = labelfont)
        self.PPprefixdisplaytitle.grid(row=srow+2, column=3, sticky='E')
        self.PPprefixdisplay = tk.Label(self.parent, textvariable = self.predictedprefixtext, fg = 'gray', font = labelfont)
        self.PPprefixdisplay.grid(row=srow+2, column=4, sticky='W')

        # self.DBnametext = tk.StringVar()
        # self.DBnametext.set(self.DBname)
        # self.DBnamelabel2 = tk.Label(self.parent, textvariable=self.DBnametext, bg=bgcol, fg="black", font="none 10",
        #                              wraplength=200, justify='left')


    def PPsliceorderchoice(self,value):
        self.sliceorder = value
        print('Slice order set to ',self.sliceorder)
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        settings['sliceorder'] = self.sliceorder
        np.save(settingsfile,settings)
        # update output prefix information
        prefix_list = pypreprocess.setprefixlist(settingsfile)
        self.predictedprefixtext.set(prefix_list[-1])
        return self

    def PPcoregchoice(self, value):
        # action when checkboxes are selected/deselected
        self.coreg_choice = value
        print('Co-registration set to ',self.coreg_choice)
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        settings['coreg_choice'] = self.coreg_choice
        np.save(settingsfile,settings)
        # update output prefix information
        prefix_list = pypreprocess.setprefixlist(settingsfile)
        self.predictedprefixtext.set(prefix_list[-1])
        return self

    def PPslicetimechoice(self, value):
        # action when checkboxes are selected/deselected
        self.slicetime_choice = value
        print('Slice time correction set to ',self.slicetime_choice)
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        settings['slicetime_choice'] = self.slicetime_choice
        np.save(settingsfile,settings)
        # update output prefix information
        prefix_list = pypreprocess.setprefixlist(settingsfile)
        self.predictedprefixtext.set(prefix_list[-1])
        return self

    def PPnormchoice(self, value):
        # action when checkboxes are selected/deselected
        self.norm_choice = value
        print('Apply normalization set to ',self.norm_choice)
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        settings['norm_choice'] = self.norm_choice
        np.save(settingsfile,settings)
        # update output prefix information
        prefix_list = pypreprocess.setprefixlist(settingsfile)
        self.predictedprefixtext.set(prefix_list[-1])
        return self

    def PPsmoothchoice(self, value):
        # action when checkboxes are selected/deselected
        self.smooth_choice = value
        print('Apply smoothing set to ',self.smooth_choice)
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        settings['smooth_choice'] = self.smooth_choice
        np.save(settingsfile,settings)
        # update output prefix information
        prefix_list = pypreprocess.setprefixlist(settingsfile)
        self.predictedprefixtext.set(prefix_list[-1])
        return self

    def PPdefinechoice(self, value):
        # action when checkboxes are selected/deselected
        self.define_choice = value
        print('Define basis sets set to ',self.define_choice)
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        settings['define_choice'] = self.define_choice
        np.save(settingsfile,settings)
        # update output prefix information
        prefix_list = pypreprocess.setprefixlist(settingsfile)
        self.predictedprefixtext.set(prefix_list[-1])
        return self

    def PPcleanchoice(self, value):
        # action when checkboxes are selected/deselected
        self.clean_choice = value
        print('Clean the data set to ',self.clean_choice)
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        settings['clean_choice'] = self.clean_choice
        np.save(settingsfile,settings)
        # update output prefix information
        prefix_list = pypreprocess.setprefixlist(settingsfile)
        self.predictedprefixtext.set(prefix_list[-1])
        return self

    # action when the button is pressed to submit the slice information
    def PPslicesubmit(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        entered_text = self.PPrefslice.get()
        refslice = int(entered_text)
        entered_text = self.PPsliceaxis.get()
        sliceaxis = int(entered_text)
        settings['refslice'] = refslice
        settings['sliceaxis'] = sliceaxis
        np.save(settingsfile,settings)
        # update the text in the box, in case it has changed
        self.refslice = refslice
        self.sliceaxis = sliceaxis
        return self

    # action when the button is pressed to submit the slice information
    def PPsmoothsubmit(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        entered_text = self.PPsmoothing.get()
        smoothwidth = float(entered_text)
        settings['smoothwidth'] = smoothwidth
        np.save(settingsfile,settings)
        # update the text in the box, in case it has changed
        self.smoothwidth = smoothwidth
        return self


    # action when the button is pressed to organize dicom data into folders based on series numbers
    def PPrunclick(self):
        sucessflag = pypreprocess.run_preprocessing(settingsfile)



#-----------GLMfit FRAME--------------------------------------------------
# Definition of the frame that has inputs for the database name, and entry numbers to use
class GLMFrame:
    # initialize the values, keeping track of the frame this definition works on (parent), and
    # also the main window containing that frame (controller)
    def __init__(self, parent, controller):
        parent.configure(relief='raised', bd=5, highlightcolor=fgcol3)
        self.parent = parent
        self.controller = controller

        # initialize some values
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.DBname = settings['DBname']
        self.DBnum = settings['DBnum']
        self.GLM1_option = settings['GLM1_option']
        self.GLMprefix = settings['GLMprefix']
        self.GLMndrop = settings['GLMndrop']
        self.GLMcontrast = settings['GLMcontrast']
        self.GLMresultsdir = settings['GLMresultsdir']
        self.GLMpvalue = settings['GLMpvalue']
        self.GLMpvalue_unc = settings['GLMpvalue_unc']
        self.GLMvoxvolume = settings['GLMvoxvolume']

        # put some text as a place-holder
        self.GLMlabel1 = tk.Label(self.parent, text = "1) Select GLM analysis options", fg = 'gray')
        self.GLMlabel1.grid(row=0,column=0, sticky='W')
        self.GLMlabel1 = tk.Label(self.parent, text = "2) Specify database numbers,\ncontrast, and analysis mode", fg = 'gray')
        self.GLMlabel1.grid(row=1,column=0, sticky='W')

        GLMoptions = {'concatenate_group', 'group_concatenate_by_person_avg', 'avg_by_person',
                        'concatenate_by_person', 'group_average'}

        self.GLMoptionlabel1 = tk.Label(self.parent, text="GLM Method:", font = labelfont).grid(row=0, column=1, sticky='E')
        self.GLMoption_var = tk.StringVar()
        self.GLMoption_var.set(self.GLM1_option)
        self.GLMoption_opt = tk.OptionMenu(self.parent, self.GLMoption_var, *GLMoptions, command = self.GLMoptionchoice)
        self.GLMoption_opt.config(bg=bgcol)
        self.GLMoption_opt.grid(row=0, column=2, sticky='EW')

        # set the data prefix for analysis
        self.GLMlabel2 = tk.Label(self.parent, text = 'Data prefix:', font = labelfont).grid(row=1, column=1, sticky='NSEW')
        self.GLMprefixbox = tk.Entry(self.parent, width = 8, bg="white")
        self.GLMprefixbox.grid(row=1, column=2, sticky="W")
        self.GLMprefixbox.insert(0,self.GLMprefix)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.GLMprefixsubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.GLMprefixsubmit, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.GLMprefixsubmit.grid(row=1, column=3)


        # set the number of initial volumes to drop, or mask, for each data set
        self.GLMlabel3 = tk.Label(self.parent, text = 'Initial volumes to drop/mask:', font = labelfont).grid(row=2, column=1, sticky='NSEW')
        self.GLMndropbox = tk.Entry(self.parent, width = 8, bg="white")
        self.GLMndropbox.grid(row=2, column=2, sticky = "W")
        self.GLMndropbox.insert(0,self.GLMndrop)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.GLMndropsubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.GLMndropsubmit, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.GLMndropsubmit.grid(row=2, column=3)

        # indicate the contrast to use for analysis
        self.GLMlabel4 = tk.Label(self.parent, text = 'Contrast between basis elements:', font = labelfont).grid(row=3, column=1, sticky='NSEW')
        self.GLMcontrastbox = tk.Entry(self.parent, width = 8, bg="white")
        self.GLMcontrastbox.grid(row=3, column=2, sticky = "W")
        self.GLMcontrastbox.insert(0,self.GLMcontrast)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.GLMcontrastsubmitbut = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.GLMcontrastsubmit, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.GLMcontrastsubmitbut.grid(row=3, column=3)

        # put in choices for statistical threshold
        self.GLMlabel5 = tk.Label(self.parent, text = 'p-value threhold:', font = labelfont).grid(row=4, column=1, sticky='NSEW')
        self.GLMpvaluebox = tk.Entry(self.parent, width = 8, bg="white")
        self.GLMpvaluebox.grid(row=4, column=2, sticky = "W")
        self.GLMpvaluebox.insert(0,self.GLMpvalue)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.GLMpvaluesubmitbut = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.GLMpvaluesubmit, relief='raised', bd = 5, state = tk.DISABLED, highlightbackground = widgetbg)
        self.GLMpvaluesubmitbut.grid(row=4, column=3)

        # put in choices for voxel volume
        self.GLMlabel6 = tk.Label(self.parent, text = 'voxel volume (mm3):', font = labelfont).grid(row=5, column=1, sticky='NSEW')
        self.GLMvvolumebox = tk.Entry(self.parent, width = 8, bg="white")
        self.GLMvvolumebox.grid(row=5, column=2, sticky = "W")
        self.GLMvvolumebox.insert(0,self.GLMvoxvolume)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.GLMvvolumesubmitbut = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.GLMvvolumesubmit, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.GLMvvolumesubmitbut.grid(row=5, column=3)


        # radioboxes to indicate stats correction choices
        self.GLML7 = tk.Label(self.parent, text="Correction for multiple comparisons:", font = labelfont)
        self.GLML7.grid(row=6, column=1, sticky='W')
        self.GLMpcorrectionmethod = tk.IntVar(None,1)
        self.GLMcorr1 = tk.Radiobutton(self.parent, text = 'no correction', width = smallbuttonsize, fg = fgletter2, font = radiofont,
                                          command = self.GLMsetcorrtype, variable = self.GLMpcorrectionmethod, value = 1, state = tk.DISABLED)
        self.GLMcorr1.grid(row = 6, column = 2, sticky="E")
        self.GLMcorr2 = tk.Radiobutton(self.parent, text = 'GRF', width = smallbuttonsize, fg = fgletter2, font = radiofont,
                                          command = self.GLMsetcorrtype, variable = self.GLMpcorrectionmethod, value = 2, state = tk.DISABLED)
        self.GLMcorr2.grid(row = 6, column = 3, sticky="E")
        self.GLMcorr3 = tk.Radiobutton(self.parent, text = 'Bonferroni', width = smallbuttonsize, fg = fgletter2, font = radiofont,
                                          command = self.GLMsetcorrtype, variable = self.GLMpcorrectionmethod, value = 3, state = tk.DISABLED)
        self.GLMcorr3.grid(row = 6, column = 4, sticky="E")


        ttk.Separator(self.parent).grid(row=7, column=1, columnspan=6, sticky="nswe", padx=2, pady=5)

        # entry box for specifying folder for saving results
        # create an entry box so that the user can specify the results folder to use
        self.GLML2 = tk.Label(self.parent, text="Results folder:", font = labelfont)
        self.GLML2.grid(row=8, column=1, sticky='W')

        # make a label to show the current setting of the results folder name
        self.GLMresultstext = tk.StringVar()
        self.GLMresultstext.set(self.GLMresultsdir)
        self.GLMresultsinfo = tk.Label(self.parent, textvariable=self.GLMresultstext, bg=bgcol, fg=fgletter2, font = labelfont,
                                     wraplength=150, justify='left')
        self.GLMresultsinfo.grid(row=8, column=2, sticky='W')

        # define a button to browse and select a location for saving the GLM results, and write out the selected name
        # also, define the function for what to do when this button is pressed
        self.GLMfolderbrowse = tk.Button(self.parent, text='Browse', width=smallbuttonsize, bg=fgcol2, fg = fgletter2, font = widgetfont,
                                  command=self.GLMresultsbrowseclick, relief='raised', bd=5, highlightbackground = widgetbg)
        self.GLMfolderbrowse.grid(row=8, column=3)

        # label, button, and information box for compiling the basis sets
        self.GLMbasisbutton = tk.Button(self.parent, text="Compile Basis Sets", width=bigbigbuttonsize, bg=fgcol1, fg = fgletter1, font = widgetfont,
                                        command=self.GLMbasissetup, relief='raised', bd=5, highlightbackground = widgetbg)
        self.GLMbasisbutton.grid(row=9, column=1)
        self.GLMbasisbutton2 = tk.Button(self.parent, text="Load Basis Sets", width=bigbigbuttonsize, bg=fgcol2, fg = fgletter2, font = widgetfont,
                                        command=self.GLMbasisload, relief='raised', bd=5, highlightbackground = widgetbg)
        self.GLMbasisbutton2.grid(row=9, column=2)
        self.GLMbasisinfo= tk.Label(self.parent, text='basis set information ...', font = labelfont)
        self.GLMbasisinfo.grid(row=9, column=3, columnspan = 4, sticky='E')


        # label, button, and information box for compiling the data sets
        self.GLMdatabutton = tk.Button(self.parent, text="Compile Data Sets", width=bigbigbuttonsize, bg=fgcol1, fg = fgletter1, font = widgetfont,
                                        command=self.GLMdatasetup, relief='raised', bd=5, highlightbackground = widgetbg)
        self.GLMdatabutton.grid(row=10, column=1)
        self.GLMdatabutton2 = tk.Button(self.parent, text="Load Data Sets", width=bigbigbuttonsize, bg=fgcol2, fg = fgletter2, font = widgetfont,
                                        command=self.GLMdataload, relief='raised', bd=5, highlightbackground = widgetbg)
        self.GLMdatabutton2.grid(row=10, column=2)
        self.GLMdatainfo = tk.Label(self.parent, text='data set information ...', font = labelfont)
        self.GLMdatainfo.grid(row=10, column=3, columnspan = 4, sticky='E')

        # setup input for contrast
        # label, button, and information box for running the GLM analysis
        self.GLMrunbutton = tk.Button(self.parent, text="Run GLM", width=bigbigbuttonsize, bg=fgcol1, fg = fgletter1, font = widgetfont,
                                        command=self.GLMrun1, relief='raised', bd=5, highlightbackground = widgetbg)
        self.GLMrunbutton.grid(row=11, column=1)
        self.GLMruninfo= tk.Label(self.parent, text='GLM results information ...', font = labelfont)
        self.GLMruninfo.grid(row=11, column=2, columnspan = 4, sticky='E')


    def GLMpvaluesubmit(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        GLMpvalue = float(self.GLMpvaluebox.get())
        settings['GLMpvalue'] = GLMpvalue
        np.save(settingsfile,settings)
        # update the text in the box, in case it has changed
        self.GLMpvalue = GLMpvalue
        print('p-value for GLM analysis set to ',self.GLMpvalue)

        # update uncorrected p-value as well
        # value = self.GLMpcorrectionmethod.get()
        self.GLMsetcorrtype()

        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.GLMpvalue_unc = settings['GLMpvalue_unc']
        print('uncorrected p-value for GLM analysis set to {:0.3e}'.format(self.GLMpvalue_unc))

        return self

    def GLMvvolumesubmit(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        GLMvoxvolume = float(self.GLMvvolumebox.get())
        settings['GLMvoxvolume'] = GLMvoxvolume
        np.save(settingsfile,settings)
        # update the text in the box, in case it has changed
        self.GLMvoxvolume = GLMvoxvolume
        print('voxel volume for estimating family-wise error correction for GLM analysis set to ',self.GLMvoxvolume)

        # update uncorrected p-value as well
        # value = self.GLMpcorrectionmethod.get()
        self.GLMsetcorrtype()
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.GLMpvalue_unc = settings['GLMpvalue_unc']
        print('uncorrected p-value for GLM analysis set to {:0.3e}'.format(self.GLMpvalue_unc))

        return self


    def GLMsetcorrtype(self):
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        DBname = settings['DBname']
        DBnum = settings['DBnum']
        GLMpvalue = settings['GLMpvalue']
        GLMvoxvolume = settings['GLMvoxvolume']

        self.GLMpvalue = settings['GLMpvalue']
        print('GLMsetcorrtype: corrected p-value set to ',self.GLMpvalue)

        value = self.GLMpcorrectionmethod.get()
        if value == 1:
            self.GLMcorrmethod = 'none'
            p_unc = self.GLMpvalue

        if value == 2:
            self.GLMcorrmethod = 'GRF'
            xls = pd.ExcelFile(DBname, engine='openpyxl')
            df1 = pd.read_excel(xls, 'datarecord')
            normtemplatename = df1.loc[DBnum[0], 'normtemplatename']

            if normtemplatename.lower() == 'brain':
                braintemplate = settings['braintemplate']
                template_img, template_affine, roi_map = load_templates.load_brain_template(braintemplate)
            else:
                resolution = 1
                template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_map = load_templates.load_template_and_masks(
                    normtemplatename, resolution)

            # correct using gaussian random field theory
            search_mask = roi_map
            if np.ndim(self.dataset) > 4:
                residual_data = self.dataset[:, :, :, :, 0]  # take data from one person, as an example
                xs,ys,zs,ts = np.shape(self.dataset)
                degrees_of_freedom = ts - 1
            else:
                residual_data = self.dataset
                degrees_of_freedom = 0
            p_unc, FWHM, R = py_fmristats.py_GRFcorrected_pthreshold(self.GLMpvalue, residual_data, roi_map, degrees_of_freedom)

        if value == 3:
            self.GLMcorrmethod = 'Bonferroni'
            xls = pd.ExcelFile(DBname, engine='openpyxl')
            df1 = pd.read_excel(xls, 'datarecord')
            normtemplatename = df1.loc[DBnum[0], 'normtemplatename']
            if normtemplatename.lower() == 'brain':
                braintemplate = settings['braintemplate']
                template_img, template_affine, roi_map = load_templates.load_brain_template(braintemplate)
            else:
                resolution = 1
                template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_map = load_templates.load_template_and_masks(
                    normtemplatename, resolution)
            p_unc, nvox = py_fmristats.py_Bonferonni_corrected_pthreshold(self.GLMpvalue, roi_map, GLMvoxvolume)

        settings['GLMpvalue_unc'] = p_unc
        self.GLMpvalue_unc = p_unc
        # convert p-threshol (p_unc) to T-threshold
        # Tthresh = stats.t.ppf(1 - p_unc, degrees_of_freedom)

        print('Family-wise error correction method set: ',self.GLMcorrmethod)
        print('  corrected p-value: {:0.3e}    uncorrected p-value: {:0.3e}'.format(self.GLMpvalue,self.GLMpvalue_unc))

        np.save(settingsfile,settings)

        return self


    def GLMoptionchoice(self,value):
        # select the method for GLM analysis
        self.GLM1_option = value
        print('GLM method set to ',self.GLM1_option)
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        settings['GLM1_option'] = self.GLM1_option
        np.save(settingsfile,settings)
        return self

    def GLMprefixsubmit(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        GLMprefix = self.GLMprefixbox.get()
        settings['GLMprefix'] = GLMprefix
        np.save(settingsfile,settings)
        # update the text in the box, in case it has changed
        self.GLMprefix = GLMprefix
        print('prefix for GLM analysis set to ',self.GLMprefix)
        return self

    def GLMndropsubmit(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        GLMndrop = int(self.GLMndropbox.get())
        settings['GLMndrop'] = GLMndrop
        np.save(settingsfile,settings)
        # update the text in the box, in case it has changed
        self.GLMndrop = GLMndrop
        print('no. of volumes to drop for GLM analysis set to ',self.GLMndrop)
        return self

    def GLMcontrastsubmit(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        entered_text = self.GLMcontrastbox.get()
        # parse the string into a list of integers
        # first,  replace commas with spaces, and then replace any double spaces with single spaces
        entered_text = re.sub('\,+',' ',entered_text)
        entered_text = re.sub('\ \ +',' ',entered_text)
        entered_text = entered_text.split()
        GLMcontrast = list(map(int, entered_text))

        settings['GLMcontrast'] = GLMcontrast
        np.save(settingsfile,settings)
        # update the text in the box, in case it has changed
        self.GLMcontrast = GLMcontrast
        self.GLMcontrastbox.delete(0, 'end')
        self.GLMcontrastbox.insert(0, self.GLMcontrast)

        print('contrast for GLM analysis set to ',self.GLMcontrast)
        return self


    def GLMbasissetup(self):
        # select the method for GLM analysis
        self.GLMbasisinfo.configure(text = 'Compiling basis set ...')

        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.DBname = settings['DBname']
        self.dbnumlist = settings['DBnum']
        self.prefix = settings['GLMprefix']
        self.ndrop = settings['GLMndrop']

        xls = pd.ExcelFile(self.DBname, engine='openpyxl')
        df1 = pd.read_excel(xls, 'datarecord')
        datadir = df1.loc[self.dbnumlist[0], 'datadir']
        pname = df1.loc[self.dbnumlist[0], 'pname']
        basedir = os.path.join(datadir,pname)
        os.chdir(basedir)  # move to the most likely place to save the paradigm file

        basisset, paradigm_names = GLMfit.compile_basis_sets(self.DBname, self.dbnumlist, self.prefix, mode=self.GLM1_option, nvolmask=self.ndrop)
        self.basisset = basisset
        self.paradigm_names = paradigm_names

        settings['GLMbasisset'] = basisset
        settings['GLMparadigmnames'] = paradigm_names
        np.save(settingsfile,settings)

        bshape = np.shape(basisset)
        basisinfo = 'basis set, size {}'.format(bshape[0])
        for val in bshape[1:]:
            infotext = ' x {}'.format(val)
            basisinfo += infotext
        nameline = '\nParadigms: {}'.format(paradigm_names[0])
        basisinfo += nameline
        linelength = len(nameline)
        for val in paradigm_names[1:]:
            if linelength > 25:
                infotext = ',\n{}'.format(val)
                linelength = len(infotext)
            else:
                infotext = ', {}'.format(val)
                linelength += len(infotext)
            basisinfo += infotext
        self.GLMbasisinfo.configure(text = basisinfo)

        # prompt for a filename for saving the basis set
        filechoice = tkf.asksaveasfilename(title="Save basis set as:",
                        filetypes=(("npy files", "*.npy"), ("all files", "*.*")))
        basisdata = {'basisset':basisset, 'paradigm_names':paradigm_names}
        np.save(filechoice, basisdata)
        print('Finished saving  basis set to ',filechoice)

        return self


    def GLMbasisload(self):
        # prompt for a filename for loading the basis set
        self.GLMbasisinfo.configure(text='Loading basis set...')

        filename = tkf.askopenfilename(title="Load basis set from:",
                        filetypes=(("npy files", "*.npy"), ("all files", "*.*")))
        basisdata = np.load(filename,allow_pickle = True).flat[0]
        basisset = basisdata['basisset']
        paradigm_names = basisdata['paradigm_names']

        self.basisset = basisset
        self.paradigm_names = paradigm_names

        bshape = np.shape(basisset)
        basisinfo = 'basis set, size {}'.format(bshape[0])
        for val in bshape[1:]:
            infotext = ' x {}'.format(val)
            basisinfo += infotext
        nameline = '\nParadigms: {}'.format(paradigm_names[0])
        basisinfo += nameline
        linelength = len(nameline)
        for val in paradigm_names[1:]:
            if linelength > 25:
                infotext = ',\n{}'.format(val)
                linelength = len(infotext)
            else:
                infotext = ', {}'.format(val)
                linelength += len(infotext)
            basisinfo += infotext
        self.GLMbasisinfo.configure(text = basisinfo)
        print('Finished loading basis set from ',filename)

        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        settings['GLMbasisset'] = basisset
        settings['GLMparadigmnames'] = paradigm_names
        np.save(settingsfile,settings)

        return self



    # define functions before they are used in the database frame------------------------------------------
    # action when the button to browse for a DB fie is pressed
    def GLMresultsbrowseclick(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        # use a dialog box to prompt the user to select an existing file, the default being .xlsx type
        dirname =  tkf.askdirectory(title = "Select folder")
        print('dirname = ',dirname)
        # save the selected file name in the settings
        settings['GLMresultsdir'] = dirname
        self.GLMresultsdir = dirname
        # write the result to the label box for display
        self.GLMresultstext.set(settings['GLMresultsdir'])

        # save the updated settings file again
        np.save(settingsfile,settings)


    def GLMdatasetup(self):
        self.GLMdatainfo.configure(text = 'Compiling data ...')

        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        # select the method for GLM analysis
        self.DBname = settings['DBname']
        self.DBnum = settings['DBnum']
        self.prefix = settings['GLMprefix']
        self.ndrop = settings['GLMndrop']

        dataset = GLMfit.compile_data_sets(self.DBname, self.DBnum, self.prefix, mode=self.GLM1_option, nvolmask=self.ndrop)
        self.dataset = dataset

        bshape = np.shape(dataset)
        datainfo = 'data set, size {}'.format(bshape[0])
        for val in bshape[1:]:
            infotext = ' x {}'.format(val)
            datainfo += infotext

        self.GLMdatainfo.configure(text = datainfo)

        # prompt for a filename for saving the data
        filechoice = tkf.asksaveasfilename(title="Save data as:",
                        filetypes=(("npy files", "*.npy"), ("all files", "*.*")))
        np.save(filechoice,dataset)
        print('Finished saving  data to ',filechoice)

        # settings['GLMdata'] = dataset
        settings['GLMdataname'] = filechoice
        np.save(settingsfile,settings)

        self.GLMpvaluesubmitbut['state'] = tk.NORMAL
        self.GLMcorr1['state'] = tk.NORMAL
        self.GLMcorr2['state'] = tk.NORMAL
        self.GLMcorr3['state'] = tk.NORMAL

        return self


    def GLMdataload(self):
        # this function is useless except for checking the data shape/size
        # it makes the base_setup_file very large
        # it will be replaced in future versions
        self.GLMdatainfo.configure(text = 'Loading data ...')
        # select the method for GLM analysis
        # prompt for a filename for loading the basis set
        filename = tkf.askopenfilename(title="Load data set from:",
                                                 filetypes=(("npy files", "*.npy"), ("all files", "*.*")))
        dataset = np.load(filename, allow_pickle=True)
        self.dataset = dataset

        print('size of dataset = ',np.shape(dataset))

        bshape = np.shape(dataset)
        datainfo = 'data set, size {}'.format(bshape[0])
        for val in bshape[1:]:
            infotext = ' x {}'.format(val)
            datainfo += infotext

        self.GLMdatainfo.configure(text=datainfo)
        print('Finished loading  data from ',filename)

        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        # settings['GLMdata'] = dataset
        settings['GLMdataname'] = filename
        np.save(settingsfile,settings)

        self.GLMpvaluesubmitbut['state'] = tk.NORMAL
        self.GLMcorr1['state'] = tk.NORMAL
        self.GLMcorr2['state'] = tk.NORMAL
        self.GLMcorr3['state'] = tk.NORMAL

        return self


    def GLMrun1(self):
        # select the method for GLM analysis
        # GLMoptions = {'concatenate_group', 'group_concatenate_by_person_avg', 'avg_by_person',
        #               'concatenate_by_person', 'group_average'}
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        DBname = settings['DBname']
        DBnum = settings['DBnum']
        # dataset = settings['GLMdata']
        filename = settings['GLMdataname']
        basisset = settings['GLMbasisset']
        GLMpvalue = settings['GLMpvalue']
        GLMpvalue_unc = settings['GLMpvalue_unc']

        # actually load the data here
        dataset = np.load(filename, allow_pickle=True)
        self.dataset = dataset
        bshape = np.shape(dataset)
        datainfo = 'data set, size {}'.format(bshape[0])
        for val in bshape[1:]:
            infotext = ' x {}'.format(val)
            datainfo += infotext
        self.GLMdatainfo.configure(text=datainfo)
        print('Finished loading  data from ',filename)

        # for consistency with other program components - update these
        self.DBname = DBname
        self.DBnum = DBnum
        self.dataset = dataset
        self.basisset = basisset

        per_person_modes = ['avg_by_person', 'concatenate_by_person']
        runmode = settings['GLM1_option']
        contrast = settings['GLMcontrast']

        # some run modes need to be run person-by-person
        if runmode in per_person_modes:
            if np.ndim(dataset) == 3:  dataset = dataset[:,:,:,np.newaxis]
            if np.ndim(dataset) == 4:  dataset = dataset[:,:,:,:,np.newaxis]
            xs, ys, zs, ts, NP = np.shape(dataset)
            if NP == 1: basisset = basisset[:,:,np.newaxis]
            for nn in range(NP):
                persondata = dataset[:,:,:,:,nn]
                personbasis = basisset[:,:,nn]
                print('GLM mode: ',runmode)
                print('person ',nn,' size of dataset ',np.shape(persondata),' size of basisset ',np.shape(personbasis), 'contrast = ',contrast)
                Bperson, semperson, Tperson = GLMfit.GLMfit(persondata, personbasis, contrast, add_constant=True, ndrop=0)

                # compile results together across people
                if nn == 0:
                    B = Bperson[:,:,:,np.newaxis]
                    sem = semperson[:,:,:,np.newaxis]
                    T = Tperson[:,:,:,np.newaxis]
                else:
                    B = np.concatenate((B,Bperson[:,:,:,np.newaxis]),axis = 3)
                    sem = np.concatenate((sem,semperson[:,:,:,np.newaxis]),axis = 3)
                    T = np.concatenate((T,Tperson[:,:,:,np.newaxis]),axis = 3)

        else:
            xs, ys, zs, ts = np.shape(dataset)
            print('GLM mode: ',runmode)
            print(' size of dataset ',np.shape(dataset),' size of basisset ',np.shape(basisset), 'contrast = ',contrast)
            B, sem, T = GLMfit.GLMfit(dataset, basisset, contrast, add_constant=True, ndrop=0)


        # write out information about the results
        # bshape = np.shape(B)
        bshape = np.shape(B)
        GLMinfo = 'GLM results: dimensions of beta map are {}'.format(bshape[0])
        for v in bshape[1:]:
            addtext = ' x {}'.format(v)
            GLMinfo += addtext

        self.GLMruninfo.configure(text=GLMinfo)

        #-----------create outputs------------------------------------
        # choose a statistical threshold and create a results figure
        xls = pd.ExcelFile(DBname, engine = 'openpyxl')
        df1 = pd.read_excel(xls, 'datarecord')
        normtemplatename = df1.loc[DBnum[0], 'normtemplatename']

        if normtemplatename == 'brain':
            braintemplate = settings['braintemplate']
            template_img, template_affine, roi_map = load_templates.load_brain_template(braintemplate)
            regionmap_img = []
        else:
            resolution = 1
            template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_map = load_templates.load_template_and_masks(normtemplatename, resolution)
        print('normtemplatename = {}'.format(normtemplatename))
        print('size of template_img is {}'.format(np.shape(template_img)))


        #
        # p_corr = 0.05   # corrected p-value threshold
        # # correct using gaussian random field theory
        # search_mask = roi_map
        # if np.ndim(dataset) > 4:
        #     residual_data = dataset[:,:,:,:,0]  # take data from one person, as an example
        # else:
        #     residual_data = dataset
        # p_unc, FWHM, R = py_fmristats.py_GRFcorrected_pthreshold(p_corr, residual_data, search_mask, df=0)

        # convert p-threshol (p_unc) to T-threshold
        degrees_of_freedom = ts-1
        Tthresh = stats.t.ppf(1-GLMpvalue_unc,degrees_of_freedom)

        if runmode in per_person_modes:
            Tthresh_per_person = stats.t.ppf(1-GLMpvalue_unc,degrees_of_freedom*NP)

        print('ready for displaying GLM results, with p_unc = ',GLMpvalue_unc,', T-treshold = ',Tthresh)

        # find all the values in T that exceed the threshold, and create a figure with these voxels
        # overlying template_img
        outputimg = pydisplay.pydisplaystatmap(T,Tthresh,template_img,roi_map,normtemplatename)
        # fig = plt.figure(77), plt.imshow(outputimg)

        # save the image for later use
        filechoice = tkf.asksaveasfilename(title="Save results as:",
                        filetypes=(("npy files", "*.npy"), ("all files", "*.*")))
        pname,fname = os.path.split(filechoice)
        fname,ext = os.path.splitext(fname)
        outputname = os.path.join(pname,fname+'.png')

        print('size of output image: ',outputimg.shape)
        print('image output name: ',outputname)

        # scipy.misc.imsave(outputname, outputimg)   # alternative way of writing image to file
        matplotlib.image.imsave(outputname, outputimg)

        # now save the results B, sem, and T, and information to go with them
        # prompt for a filename for saving the data
        results = {'type':'GLM','B':B,'sem':sem,'T':T, 'template':template_img, 'regionmap':regionmap_img, 'roi_map':roi_map, 'Tthresh':Tthresh, 'normtemplatename':normtemplatename, 'DBname':self.DBname, 'DBnum':self.DBnum}
        np.save(filechoice,results)

        return self



#-----------Clustering FRAME--------------------------------------------------
# Definition of the frame that has inputs for the database name, and entry numbers to use
class CLFrame:

    # define functions before they are used in the database frame------------------------------------------
    # action when the button to browse for a DB fie is pressed
    def CLnetbrowseclick(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        # use a dialog box to prompt the user to select an existing file, the default being .xlsx type
        filename =  tkf.askopenfilename(title = "Select file",filetypes = (("excel files","*.xlsx"),("all files","*.*")))
        print('filename = ',filename)
        # save the selected file name in the settings
        settings['networkmodel'] = filename
        self.networkmodel = filename

        # write the result to the label box for display
        npname, nfname = os.path.split(self.networkmodel)
        self.CLnetnametext.set(nfname)
        self.CLnetdirtext.set(npname)
        np.save(settingsfile,settings)

    def CLprefixsubmitaction(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        CLprefix = self.CLprefixbox.get()
        settings['CLprefix'] = CLprefix
        np.save(settingsfile,settings)
        # update the text in the box, in case it has changed
        self.CLprefix = CLprefix
        print('prefix for clustering analysis set to ',self.CLprefix)
        self.CLupdate_network_info()
        return self


    # define functions before they are used in the database frame------------------------------------------
    # action when the button to browse for a DB fie is pressed
    def CLclusternamebrowseaction(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        # use a dialog box to prompt the user to select an existing file, the default being .xlsx type
        filechoice =  tkf.asksaveasfilename(title = "Select file",filetypes = (("npy files","*.npy"),("all files","*.*")))
        print('cluster definition name = ',filechoice)
        # save the selected file name in the settings
        CLclustername = filechoice

        npname, nfname = os.path.split(CLclustername)
        nfname,ext = os.path.splitext(nfname)
        settings['CLresultsdir'] = npname
        self.CLresultsdir = npname

        CLclustername = os.path.join(npname,nfname+'.npy')

        settings['CLclustername'] = CLclustername
        self.CLclustername = CLclustername

        # write the result to the label box for display
        self.CLclusternamebox.delete(0, 'end')
        self.CLclusternamebox.insert(0,CLclustername)

        np.save(settingsfile,settings)
        self.CLupdate_network_info()


    def CLclusternamesubmitaction(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        CLclustername = self.CLclusternamebox.get()

        # check if chosen name includes the full directory path, or the extension
        npname, nfname = os.path.split(CLclustername)
        nfname, ext = os.path.splitext(nfname)

        if os.path.isdir(npname):
            settings['CLresultsdir'] = npname
            self.CLresultsdir = npname
        else:
            # select a directory
            npname = settings['CLresultsdir']

        # join up the name parts
        CLclustername = os.path.join(npname,nfname+'.npy')

        settings['CLclustername'] = CLclustername
        self.CLclustername = CLclustername

        # write the result to the label box for display
        self.CLclusternamebox.delete(0, 'end')
        self.CLclusternamebox.insert(0, self.CLclustername)

        np.save(settingsfile, settings)
        self.CLupdate_network_info()


    # define functions before they are used in the database frame------------------------------------------
    # action when the button to browse for a DB fie is pressed
    def CLregionnamebrowseaction(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        # use a dialog box to prompt the user to select an existing file, the default being .xlsx type
        filechoice =  tkf.asksaveasfilename(title = "Select file",filetypes = (("npy files","*.npy"),("all files","*.*")))
        print('region/cluster data name = ',filechoice)
        CLregionname = filechoice

        npname, nfname = os.path.split(CLregionname)
        fname, ext = os.path.splitext(nfname)
        settings['CLresultsdir'] = npname
        self.CLresultsdir = npname

        CLregionname = os.path.join(npname,fname+'.npy')

        # write the result to the label box for display
        self.CLregionnamebox.delete(0, 'end')
        self.CLregionnamebox.insert(0,CLregionname)

        # save the selected file name in the settings
        settings['CLregionname'] = CLregionname
        self.CLregionname = CLregionname

        np.save(settingsfile,settings)
        self.CLupdate_network_info()

    def CLregionnamesubmitaction(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        CLregionname = self.CLregionnamebox.get()

        # check if chosen name includes the full directory path, or the extension
        npname, nfname = os.path.split(CLregionname)
        nfname, ext = os.path.splitext(nfname)

        if os.path.isdir(npname):
            settings['CLresultsdir'] = npname
            self.CLresultsdir = npname
        else:
            # select a directory
            npname = settings['CLresultsdir']

        # join up the name parts
        CLregionname = os.path.join(npname,nfname+'.npy')

        settings['CLregionname'] = CLregionname
        self.CLregionname = CLregionname

        # write the result to the label box for display
        self.CLregionnamebox.delete(0, 'end')
        self.CLregionnamebox.insert(0, self.CLregionname)

        np.save(settingsfile, settings)
        self.CLupdate_network_info()


    def CLdefineandload(self):
        # define the clusters and load the data
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.DBname = settings['DBname']
        self.DBnum = settings['DBnum']
        self.CLprefix = settings['CLprefix']
        self.networkmodel = settings['networkmodel']
        self.CLclustername = settings['CLclustername']
        self.CLregionname = settings['CLregionname']

        xls = pd.ExcelFile(self.DBname, engine = 'openpyxl')
        df1 = pd.read_excel(xls, 'datarecord')

        normtemplatename = df1.loc[self.DBnum[0], 'normtemplatename']
        resolution = 1
        template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_map = \
            load_templates.load_template_and_masks(normtemplatename, resolution)

        if normtemplatename.lower() == 'brain':
            # for brain data, need to match the template, region map, etc., to the data size/position
            dbhome = df1.loc[self.DBnum[0], 'datadir']
            fname = df1.loc[self.DBnum[0], 'niftiname']
            niiname = os.path.join(dbhome, fname)
            fullpath, filename = os.path.split(niiname)
            prefix_niiname = os.path.join(fullpath, self.CLprefix + filename)
            temp_data = nib.load(prefix_niiname)
            img_data_affine = temp_data.affine
            hdr = temp_data.header
            template_img = i3d.convert_affine_matrices_nearest(template_img, template_affine, img_data_affine, hdr['dim'][1:4])
            regionmap_img = i3d.convert_affine_matrices_nearest(regionmap_img, template_affine, img_data_affine, hdr['dim'][1:4])

        cluster_properties, region_properties = \
            pyclustering.define_clusters_and_load_data(self.DBname, self.DBnum, self.CLprefix, self.networkmodel, regionmap_img, anatlabels)

        cluster_definition = {'cluster_properties':cluster_properties, 'template_img':template_img,'regionmap_img':regionmap_img}
        region_data = {'region_properties':region_properties, 'DBname':self.DBname, 'DBnum':self.DBnum}

        # save the results
        np.save(self.CLclustername,cluster_definition)
        np.save(self.CLregionname,region_data)
        messagetext = 'defining clusters and loading data \ncompleted: ' + time.ctime(time.time())
        self.CLdefinebuttontext.set(messagetext)
        print(messagetext)
        self.CLupdate_network_info()


    def CLload(self):
        # load the data using previous cluster definitions
        # need a better way of saving/loading data
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.CLclustername = settings['CLclustername']
        self.DBname = settings['DBname']
        self.DBnum = settings['DBnum']
        self.CLprefix = settings['CLprefix']
        self.networkmodel = settings['networkmodel']

        cluster_data = np.load(self.CLclustername, allow_pickle=True).flat[0]
        cluster_properties = cluster_data['cluster_properties']

        print('CLload:  DBname = ', self.DBname)
        region_properties = pyclustering.load_cluster_data(cluster_properties, self.DBname, self.DBnum, self.CLprefix, self.networkmodel)
        region_data = {'region_properties':region_properties, 'DBname':self.DBname, 'DBnum':self.DBnum}

        np.save(self.CLregionname,region_data)
        messagetext = 'loading cluster data completed: \n' + time.ctime(time.time())
        self.CLloadbuttontext.set(messagetext)
        print(messagetext)

        self.CLupdate_network_info()


    def CLupdate_network_info(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.networkmodel = settings['networkmodel']
        npname, nfname = os.path.split(self.networkmodel)
        self.CLnetnametext.set(nfname)
        self.CLnetdirtext.set(npname)

    # initialize the values, keeping track of the frame this definition works on (parent), and
    # also the main window containing that frame (controller)
    def __init__(self, parent, controller):
        parent.configure(relief='raised', bd=5, highlightcolor=fgcol3)
        self.parent = parent
        self.controller = controller

        # initialize some values
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.DBname = settings['DBname']
        self.DBnum = settings['DBnum']
        self.CLprefix = settings['CLprefix']
        self.networkmodel = settings['networkmodel']
        self.CLclustername = settings['CLclustername']
        self.CLregionname = settings['CLregionname']

        # put some text as a place-holder
        self.CLabel1 = tk.Label(self.parent, text = "1) Select clustering options", fg = 'gray', justify = 'left')
        self.CLabel1.grid(row=0,column=0, sticky='W')
        self.CLabel2 = tk.Label(self.parent, text = "   Specify the model definition which\nincludes which regions to load\nand set names for new or existing\ncluster and region data files", fg = 'gray', justify = 'left')
        self.CLabel2.grid(row=1,column=0, sticky='W')
        self.CLabel3 = tk.Label(self.parent, text = "2) Define clusters and load data, or\nonly load data using existing\ncluster definition", fg = 'gray', justify = 'left')
        self.CLabel3.grid(row=2,column=0, sticky='W')

        # create an entry box so that the user can specify the network file to use
        # first make a title for the box, in row 3, column 1 of the grid for the main window
        self.CLL1 = tk.Label(self.parent, text="Network Model:", font = labelfont)
        self.CLL1.grid(row=0, column=1, sticky='SW')

        # make a label to show the current setting of the network definition file name
        npname, nfname = os.path.split(self.networkmodel)
        self.CLnetnametext = tk.StringVar()
        self.CLnetnametext.set(nfname)
        self.CLfnamelabel = tk.Label(self.parent, textvariable=self.CLnetnametext, bg=bgcol, fg="#4B4B4B", font = labelfont,
                                     wraplength=300, justify='left')
        self.CLfnamelabel.grid(row=0, column=2, sticky='S')

        # make a label to show the current setting of the network definition file directory name
        self.CLnetdirtext = tk.StringVar()
        self.CLnetdirtext.set(npname)
        self.CLdnamelabel = tk.Label(self.parent, textvariable=self.CLnetdirtext, bg=bgcol, fg="#4B4B4B", font = labelfont,
                                     wraplength=300, justify='left')
        self.CLdnamelabel.grid(row=1, column=2, sticky='N')

        # define a button to browse and select an existing network definition file, and write out the selected name
        # also, define the function for what to do when this button is pressed
        self.CLnetworkbrowse = tk.Button(self.parent, text='Browse', width=smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont,
                                  command=self.CLnetbrowseclick, relief='raised', bd=5, highlightbackground = widgetbg)
        self.CLnetworkbrowse.grid(row=0, column=3)

        # create entry box for the nifti data name prefix (indicates which preprocessing steps were done)
        self.CLpreflabel = tk.Label(self.parent, text = 'Data name prefix:', font = labelfont)
        self.CLpreflabel.grid(row=2, column=1, sticky='N')
        self.CLprefixbox = tk.Entry(self.parent, width = 8, bg="white")
        self.CLprefixbox.grid(row=2, column=2, sticky='N')
        self.CLprefixbox.insert(0,self.CLprefix)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.CLprefixsubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.CLprefixsubmitaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.CLprefixsubmit.grid(row=2, column=3, sticky='N')

        # need an input for the cluster definition name - save to it, or read from it
        self.CLclusternamelabel = tk.Label(self.parent, text = 'Cluster definition name:', font = labelfont)
        self.CLclusternamelabel.grid(row=3, column=1, sticky='N')
        self.CLclusternamebox = tk.Entry(self.parent, width = 30, bg="white",justify = 'right')
        self.CLclusternamebox.grid(row=3, column=2, sticky='N')
        self.CLclusternamebox.insert(0,self.CLclustername)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.CLclusternamesubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.CLclusternamesubmitaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.CLclusternamesubmit.grid(row=3, column=3, sticky='N')
        # the entry boxes need a "browse" button to allow selection of existing cluster definition file
        self.CLclusternamebrowse = tk.Button(self.parent, text = "Browse", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.CLclusternamebrowseaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.CLclusternamebrowse.grid(row=3, column=4, sticky='N')

        # box etc for entering the name for saving the region data
        self.CLregionnamelabel = tk.Label(self.parent, text = 'Region/cluster data name:', font = labelfont)
        self.CLregionnamelabel.grid(row=4, column=1, sticky='N')
        self.CLregionnamebox = tk.Entry(self.parent, width = 30, bg="white",justify = 'right')
        self.CLregionnamebox.grid(row=4, column=2, sticky='N')
        self.CLregionnamebox.insert(0,self.CLregionname)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.CLregionnamesubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.CLregionnamesubmitaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.CLregionnamesubmit.grid(row=4, column=3, sticky='N')
        # the entry boxes need a "browse" button to allow selection of existing cluster definition file
        self.CLregionnamebrowse = tk.Button(self.parent, text = "Browse", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.CLregionnamebrowseaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.CLregionnamebrowse.grid(row=4, column=4, sticky='N')


        # label, button, for running the definition of clusters, and loading data
        self.CLdefineandloadbutton = tk.Button(self.parent, text="Define Clusters", width=bigbigbuttonsize, bg=fgcol1, fg = fgletter1, font = widgetfont,
                                        command=self.CLdefineandload, relief='raised', bd=5, highlightbackground = widgetbg)
        self.CLdefineandloadbutton.grid(row=5, column=2)

        self.CLdefinebuttontext = tk.StringVar()
        self.CLdefinebuttonlabel = tk.Label(self.parent, textvariable=self.CLdefinebuttontext, bg=bgcol, fg="#4B4B4B", font = labelfont,
                                     wraplength=200, justify='left')
        self.CLdefinebuttonlabel.grid(row=5, column=3, sticky='N')

        # label, button, for running the definition of clusters, and loading data
        self.CLloadbutton = tk.Button(self.parent, text="Load Data", width=bigbigbuttonsize, bg=fgcol1, fg = fgletter1, font = widgetfont,
                                        command=self.CLload, relief='raised', bd=5, highlightbackground = widgetbg)
        self.CLloadbutton.grid(row=6, column=2)

        self.CLloadbuttontext = tk.StringVar()
        self.CLloadbuttonlabel = tk.Label(self.parent, textvariable=self.CLloadbuttontext, bg=bgcol, fg="#4B4B4B", font = labelfont,
                                     wraplength=200, justify='left')
        self.CLloadbuttonlabel.grid(row=6, column=3, sticky='N')


# inputs needed:   network model definition, existing cluster definition (if needed), run buttons



#-----------SEM FRAME--------------------------------------------------
# Definition of the frame that has inputs for the database name, and entry numbers to use
class SEMFrame:

    # define functions before they are used in the database frame------------------------------------------
    # action when the button to browse for a DB fie is pressed
    def SEMnetbrowseclick(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        # use a dialog box to prompt the user to select an existing file, the default being .xlsx type
        filename =  tkf.askopenfilename(title = "Select file",filetypes = (("excel files","*.xlsx"),("all files","*.*")))
        print('filename = ',filename)
        # save the selected file name in the settings
        settings['networkmodel'] = filename
        self.networkmodel = filename

        # write the result to the label box for display
        npname, nfname = os.path.split(self.networkmodel)
        self.SEMnetnametext.set(nfname)
        self.SEMnetdirtext.set(npname)
        np.save(settingsfile,settings)
        self.SEMupdate_network_info()

    def SEMprefixsubmitaction(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        SEMprefix = self.SEMprefixbox.get()
        settings['SEMprefix'] = SEMprefix
        np.save(settingsfile,settings)
        # update the text in the box, in case it has changed
        self.SEMprefix = SEMprefix
        print('prefix for SEM analysis set to ',self.SEMprefix)
        self.SEMupdate_network_info()
        return self


    # define functions before they are used in the database frame------------------------------------------
    # action when the button to browse for a DB fie is pressed
    def SEMclusternamebrowseaction(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        # use a dialog box to prompt the user to select an existing file, the default being .xlsx type
        filechoice =  tkf.askopenfilename(title = "Select file",filetypes = (("npy files","*.npy"),("all files","*.*")))
        print('cluster definition name = ',filechoice)
        # save the selected file name in the settings
        SEMclustername = filechoice

        npname, nfname = os.path.split(SEMclustername)
        nfname,ext = os.path.splitext(nfname)
        settings['SEMresultsdir'] = npname
        self.SEMresultsdir = npname
        # write the result to the label box for display
        self.SEMresultsdirtext.set(settings['SEMresultsdir'])

        SEMclustername = os.path.join(npname,nfname+'.npy')

        settings['SEMclustername'] = SEMclustername
        self.SEMclustername = SEMclustername

        # write the result to the label box for display
        self.SEMclusternamebox.delete(0, 'end')
        self.SEMclusternamebox.insert(0,SEMclustername)

        np.save(settingsfile,settings)
        self.SEMupdate_network_info()


    def SEMclusternamesubmitaction(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        SEMclustername = self.SEMclusternamebox.get()

        # check if chosen name includes the full directory path, or the extension
        npname, nfname = os.path.split(SEMclustername)
        nfname, ext = os.path.splitext(nfname)

        if os.path.isdir(npname):
            settings['SEMresultsdir'] = npname
            self.SEMresultsdir = npname
            # write the result to the label box for display
            self.SEMresultsdirtext.set(settings['SEMresultsdir'])
        else:
            # select a directory
            npname = settings['SEMresultsdir']

        # join up the name parts
        SEMclustername = os.path.join(npname,nfname+'.npy')

        settings['SEMclustername'] = SEMclustername
        self.SEMclustername = SEMclustername

        # write the result to the label box for display
        self.SEMclusternamebox.delete(0, 'end')
        self.SEMclusternamebox.insert(0, self.SEMclustername)

        np.save(settingsfile, settings)
        self.SEMupdate_network_info()


    def SEMsavetagsubmitaction(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        SEMsavetag = self.SEMsavetagbox.get()

        settings['SEMsavetag'] = SEMsavetag
        self.SEMsavetag = SEMsavetag

        # write the result to the label box for display
        self.SEMsavetagbox.delete(0, 'end')
        self.SEMsavetagbox.insert(0, SEMsavetag)

        np.save(settingsfile, settings)
        self.SEMupdate_network_info()


    # define functions before they are used in the database frame------------------------------------------
    # action when the button to browse for a DB fie is pressed
    def SEMregionnamebrowseaction(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        # use a dialog box to prompt the user to select an existing file, the default being .xlsx type
        filechoice =  tkf.askopenfilename(title = "Select file",filetypes = (("npy files","*.npy"),("all files","*.*")))
        print('region/cluster data name = ',filechoice)
        SEMregionname = filechoice

        npname, nfname = os.path.split(SEMregionname)
        fname, ext = os.path.splitext(nfname)
        settings['SEMresultsdir'] = npname
        self.SEMresultsdir = npname
        # write the result to the label box for display
        self.SEMresultsdirtext.set(settings['SEMresultsdir'])

        SEMregionname = os.path.join(npname,fname+'.npy')

        # write the result to the label box for display
        self.SEMregionnamebox.delete(0, 'end')
        self.SEMregionnamebox.insert(0,SEMregionname)

        # save the selected file name in the settings
        settings['SEMregionname'] = SEMregionname
        self.SEMregionname = SEMregionname

        np.save(settingsfile,settings)
        self.SEMupdate_network_info()


    def SEMregionnamesubmitaction(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        SEMregionname = self.SEMregionnamebox.get()

        # check if chosen name includes the full directory path, or the extension
        npname, nfname = os.path.split(SEMregionname)
        nfname, ext = os.path.splitext(nfname)

        if os.path.isdir(npname):
            settings['SEMresultsdir'] = npname
            self.SEMresultsdir = npname
            # write the result to the label box for display
            self.SEMresultsdirtext.set(settings['SEMresultsdir'])
        else:
            # select a directory
            npname = settings['SEMresultsdir']

        # join up the name parts
        SEMregionname = os.path.join(npname,nfname+'.npy')

        settings['SEMregionname'] = SEMregionname
        self.SEMregionname = SEMregionname

        # write the result to the label box for display
        self.SEMregionnamebox.delete(0, 'end')
        self.SEMregionnamebox.insert(0, self.SEMregionname)

        np.save(settingsfile, settings)
        self.SEMupdate_network_info()


    def SEMtimesubmitclick(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        SEMtimepoints = settings['SEMtimepoints']

        entered_text = self.SEMtimeenter.get()  # collect the text from the text entry box
        # first, replace any double spaces with single spaces, and then replace spaces with commas
        entered_text = re.sub('\ +', ',', entered_text)
        entered_text = re.sub('\,\,+', ',', entered_text)
        SEMtimepoints = np.fromstring(entered_text, dtype=int, sep=',')

        print(SEMtimepoints)

        timetext = ''
        for val in SEMtimepoints: timetext += (str(val) + ',')
        timetext = timetext[:-1]
        self.SEMtimetext = timetext

        settings['SEMtimepoints'] = SEMtimepoints
        self.SEMtimeenter.delete(0, 'end')
        self.SEMtimeenter.insert(0, self.SEMtimetext)
        # save the updated settings file again
        np.save(settingsfile, settings)
        self.SEMupdate_network_info()


    def SEMepochsubmitclick(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        SEMepoch = settings['SEMepoch']

        entered_text = self.SEMepochenter.get()  # collect the text from the text entry box
        SEMepoch = int(entered_text)

        print('Epoch set at ',SEMepoch)

        epochtext = str(SEMepoch)
        settings['SEMepoch'] = SEMepoch
        self.SEMepochenter.delete(0, 'end')
        self.SEMepochenter.insert(0, epochtext)
        # save the updated settings file again
        np.save(settingsfile, settings)
        self.SEMupdate_network_info()


    def SEMresultsdirbrowseaction(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.SEMresultsdir = settings['SEMresultsdir']
        # browse for new name
        dirname =  tkf.askdirectory(title = "Select folder")
        print('SEM results save directory name = ',dirname)
        # save the selected file name in the settings
        settings['SEMresultsdir'] = dirname
        self.SEMresultsdir = dirname
        # write the result to the label box for display
        self.SEMresultsdirtext.set(settings['SEMresultsdir'])

        # save the updated settings file again
        np.save(settingsfile,settings)
        self.SEMupdate_network_info()


    def SEMonesource(self):
        # define the clusters and load the data
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.DBname = settings['DBname']
        self.DBnum = settings['DBnum']
        self.SEMprefix = settings['SEMprefix']
        self.networkmodel = settings['networkmodel']
        self.SEMclustername = settings['SEMclustername']
        self.SEMregionname = settings['SEMregionname']
        self.SEMresultsdir = settings['SEMresultsdir']
        self.SEMsavetag = settings['SEMsavetag']
        self.SEMtimepoints = settings['SEMtimepoints']
        self.SEMepoch = settings['SEMepoch']

        self.SEMupdate_network_info()

        xls = pd.ExcelFile(self.DBname, engine = 'openpyxl')
        df1 = pd.read_excel(xls, 'datarecord')

        normtemplatename = df1.loc[self.DBnum[0], 'normtemplatename']
        resolution = 1
        template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_map = \
            load_templates.load_template_and_masks(normtemplatename, resolution)

        region_data = np.load(self.SEMregionname, allow_pickle=True).flat[0]
        region_properties = region_data['region_properties']

        cluster_data = np.load(self.SEMclustername, allow_pickle=True).flat[0]
        cluster_properties = cluster_data['cluster_properties']

        CCrecord, CRrecord = pysem.pysem_onesource(cluster_properties, region_properties, self.SEMtimepoints, self.SEMepoch)

        DBname = region_data['DBname']   # keep the values that were saved with the data
        DBnum = region_data['DBnum']
        # save the results somehow
        results = {'type':'1source','CCrecord':CCrecord, 'Correcord':CRrecord, 'DBname':DBname, 'DBnum':DBnum, 'cluster_properties':cluster_properties}
        resultsrecordname = os.path.join(self.SEMresultsdir, 'SEMresults_1source_record_' + self.SEMsavetag + '.npy')
        np.save(resultsrecordname, results)


    def SEMtwosource(self):
        # define the clusters and load the data
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.DBname = settings['DBname']
        self.DBnum = settings['DBnum']
        self.SEMprefix = settings['SEMprefix']
        self.networkmodel = settings['networkmodel']
        self.SEMclustername = settings['SEMclustername']
        self.SEMregionname = settings['SEMregionname']
        self.SEMresultsdir = settings['SEMresultsdir']
        self.SEMsavetag = settings['SEMsavetag']
        self.SEMtimepoints = settings['SEMtimepoints']
        self.SEMepoch = settings['SEMepoch']

        self.SEMupdate_network_info()

        xls = pd.ExcelFile(self.DBname, engine = 'openpyxl')
        df1 = pd.read_excel(xls, 'datarecord')

        normtemplatename = df1.loc[self.DBnum[0], 'normtemplatename']
        resolution = 1
        template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_map = \
            load_templates.load_template_and_masks(normtemplatename, resolution)

        region_data = np.load(self.SEMregionname, allow_pickle=True).flat[0]
        region_properties = region_data['region_properties']

        cluster_data = np.load(self.SEMclustername, allow_pickle=True).flat[0]
        cluster_properties = cluster_data['cluster_properties']

        CCrecord, beta2, beta1, Zgrid2, Zgrid1_1, Zgrid1_2 = pysem.pysem(cluster_properties, region_properties, self.SEMtimepoints, self.SEMepoch)

        DBname = region_data['DBname']   # keep the values that were saved with the data
        DBnum = region_data['DBnum']
        # save the results somehow
        results = {'type':'2source','CCrecord':CCrecord, 'beta2':beta2, 'beta1':beta1, 'Zgrid2':Zgrid2, 'Zgrid1_1':Zgrid1_1,'Zgrid1_2':Zgrid1_2, 'DBname':DBname, 'DBnum':DBnum, 'cluster_properties':cluster_properties}
        resultsrecordname = os.path.join(self.SEMresultsdir, 'SEMresults_2source_record_' + self.SEMsavetag + '.npy')
        np.save(resultsrecordname, results)


    def SEMrunnetwork(self):
        # define the clusters and load the data
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        self.DBname = settings['DBname']
        self.DBnum = settings['DBnum']
        self.SEMprefix = settings['SEMprefix']
        self.networkmodel = settings['networkmodel']
        self.SEMclustername = settings['SEMclustername']
        self.SEMregionname = settings['SEMregionname']
        self.SEMresultsdir = settings['SEMresultsdir']
        self.SEMsavetag = settings['SEMsavetag']
        self.SEMtimepoints = settings['SEMtimepoints']
        self.SEMepoch = settings['SEMepoch']
        self.SEMresumerun = settings['SEMresumerun']

        self.SEMupdate_network_info()

        xls = pd.ExcelFile(self.DBname, engine='openpyxl')
        df1 = pd.read_excel(xls, 'datarecord')

        normtemplatename = df1.loc[self.DBnum[0], 'normtemplatename']
        resolution = 1
        template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_map = \
            load_templates.load_template_and_masks(normtemplatename, resolution)

        region_data = np.load(self.SEMregionname, allow_pickle=True).flat[0]
        region_properties = region_data['region_properties']

        cluster_data = np.load(self.SEMclustername, allow_pickle=True).flat[0]
        cluster_properties = cluster_data['cluster_properties']

        outputnamelist = pysem.pysem_network(cluster_properties, region_properties, self.networkmodel, self.SEMtimepoints, self.SEMepoch, self.SEMresultsdir, self.SEMsavetag, self.SEMresumerun)

        DBname = region_data['DBname']   # keep the values that were saved with the data
        DBnum = region_data['DBnum']
        # save the results somehow
        results = {'type':'network','resultsnames':outputnamelist, 'network':self.networkmodel, 'regionname':self.SEMregionname, 'clustername':self.SEMclustername, 'DBname':DBname, 'DBnum':DBnum}
        resultsrecordname = os.path.join(self.SEMresultsdir, 'SEMresults_network_record_' + self.SEMsavetag + '.npy')
        np.save(resultsrecordname, results)


    # action when checkboxes are selected/deselected
    def SEMresumecheck(self):
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        self.SEMresumerun = self.var1.get() == 1
        settings['SEMresumerun'] = self.SEMresumerun
        np.save(settingsfile, settings)
        if self.SEMresumerun:
            print('choice to resume previous run set to ',self.SEMresumerun)
            print('     previous in-progress data will be reloaded and run will be resumed from closest possible point')
        else:
            print('choice to resume previous run set to ',self.SEMresumerun)
        return self


    def SEMupdate_network_info(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.networkmodel = settings['networkmodel']
        npname, nfname = os.path.split(self.networkmodel)
        self.SEMnetnametext.set(nfname)
        self.SEMnetdirtext.set(npname)


    # initialize the values, keeping track of the frame this definition works on (parent), and
    # also the main window containing that frame (controller)
    def __init__(self, parent, controller):
        parent.configure(relief='raised', bd=5, highlightcolor=fgcol3)
        self.parent = parent
        self.controller = controller

        # initialize some values
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.DBname = settings['DBname']
        self.DBnum = settings['DBnum']
        self.SEMprefix = settings['SEMprefix']
        self.networkmodel = settings['networkmodel']
        self.SEMclustername = settings['SEMclustername']
        self.SEMregionname = settings['SEMregionname']
        self.SEMresultsdir = settings['SEMresultsdir']
        self.SEMsavetag = settings['SEMsavetag']
        self.SEMtimepoints = settings['SEMtimepoints']
        self.SEMepoch = settings['SEMepoch']

        timetext = ''
        for val in self.SEMtimepoints: timetext += (str(val) + ',')
        timetext = timetext[:-1]
        self.SEMtimetext = timetext


        # put some text as a place-holder
        self.SEMLabel1 = tk.Label(self.parent, text = "1) Select SEM options\nChoices are: 1- and 2-source SEM,\nor SEM based on a network", fg = 'gray', justify = 'left')
        self.SEMLabel1.grid(row=0,column=0, sticky='W')
        self.SEMLabel3 = tk.Label(self.parent, text = "2) Run selected SEM", fg = 'gray', justify = 'left')
        self.SEMLabel3.grid(row=2,column=0, sticky='W')

        # create an entry box so that the user can specify the network file to use
        # first make a title for the box, in row 3, column 1 of the grid for the main window
        self.SEML1 = tk.Label(self.parent, text="Network Model:", font = labelfont)
        self.SEML1.grid(row=0, column=1, sticky='SW')

        # make a label to show the current setting of the network definition file name
        npname, nfname = os.path.split(self.networkmodel)
        self.SEMnetnametext = tk.StringVar()
        self.SEMnetnametext.set(nfname)
        self.SEMfnamelabel = tk.Label(self.parent, textvariable=self.SEMnetnametext, bg=bgcol, fg="#4B4B4B", font = labelfont,
                                     wraplength=300, justify='left')
        self.SEMfnamelabel.grid(row=0, column=2, sticky='S')

        # make a label to show the current setting of the network definition file directory name
        self.SEMnetdirtext = tk.StringVar()
        self.SEMnetdirtext.set(npname)
        self.SEMdnamelabel = tk.Label(self.parent, textvariable=self.SEMnetdirtext, bg=bgcol, fg="#4B4B4B", font = labelfont,
                                     wraplength=300, justify='left')
        self.SEMdnamelabel.grid(row=1, column=2, sticky='N')

        # define a button to browse and select an existing network definition file, and write out the selected name
        # also, define the function for what to do when this button is pressed
        self.SEMnetworkbrowse = tk.Button(self.parent, text='Browse', width=smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont,
                                  command=self.SEMnetbrowseclick, relief='raised', bd=5, highlightbackground = widgetbg)
        self.SEMnetworkbrowse.grid(row=0, column=3)

        # create entry box for the nifti data name prefix (indicates which preprocessing steps were done)
        self.SEMpreflabel = tk.Label(self.parent, text = 'Data name prefix:', font = labelfont)
        self.SEMpreflabel.grid(row=2, column=1, sticky='N')
        self.SEMprefixbox = tk.Entry(self.parent, width = 8, bg="white")
        self.SEMprefixbox.grid(row=2, column=2, sticky='N')
        self.SEMprefixbox.insert(0,self.SEMprefix)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.SEMprefixsubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SEMprefixsubmitaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SEMprefixsubmit.grid(row=2, column=3, sticky='N')

        # need an input for the cluster definition name - save to it, or read from it
        self.SEMclusternamelabel = tk.Label(self.parent, text = 'Cluster definition name:', font = labelfont)
        self.SEMclusternamelabel.grid(row=3, column=1, sticky='N')
        self.SEMclusternamebox = tk.Entry(self.parent, width = 30, bg="white",justify = 'right')
        self.SEMclusternamebox.grid(row=3, column=2, sticky='N')
        self.SEMclusternamebox.insert(0,self.SEMclustername)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.SEMclusternamesubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SEMclusternamesubmitaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SEMclusternamesubmit.grid(row=3, column=3, sticky='N')
        # the entry boxes need a "browse" button to allow selection of existing cluster definition file
        self.SEMclusternamebrowse = tk.Button(self.parent, text = "Browse", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SEMclusternamebrowseaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SEMclusternamebrowse.grid(row=3, column=4, sticky='N')

        # box etc for entering the name for saving the region data
        self.SEMregionnamelabel = tk.Label(self.parent, text = 'Region/cluster data name:', font = labelfont)
        self.SEMregionnamelabel.grid(row=4, column=1, sticky='N')
        self.SEMregionnamebox = tk.Entry(self.parent, width = 30, bg="white",justify = 'right')
        self.SEMregionnamebox.grid(row=4, column=2, sticky='N')
        self.SEMregionnamebox.insert(0,self.SEMregionname)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.SEMregionnamesubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SEMregionnamesubmitaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SEMregionnamesubmit.grid(row=4, column=3, sticky='N')
        # the entry boxes need a "browse" button to allow selection of existing cluster definition file
        self.SEMregionnamebrowse = tk.Button(self.parent, text = "Browse", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SEMregionnamebrowseaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SEMregionnamebrowse.grid(row=4, column=4, sticky='N')


        # create the SEM timepoints entry box
        self.SEMtimelabel = tk.Label(self.parent, text = 'Epoch center times:', font = labelfont)
        self.SEMtimelabel.grid(row=5, column=1, sticky='N')
        self.SEMtimeenter = tk.Entry(self.parent, width=20, bg="white")
        self.SEMtimeenter.grid(row=5, column=2, sticky="W")
        self.SEMtimeenter.insert(0, self.SEMtimetext)
        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.SEMtimesubmit = tk.Button(self.parent, text="Submit", width=smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont,
                                     command=self.SEMtimesubmitclick, relief='raised', bd=5, highlightbackground = widgetbg)
        self.SEMtimesubmit.grid(row=5, column=3)


        # create the SEM epoch entry box
        self.SEMepochlabel = tk.Label(self.parent, text = 'Epoch length:', font = labelfont)
        self.SEMepochlabel.grid(row=6, column=1, sticky='N')
        self.SEMepochenter = tk.Entry(self.parent, width=20, bg="white")
        self.SEMepochenter.grid(row=6, column=2, sticky="W")
        self.SEMepochenter.insert(0, self.SEMepoch)
        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.SEMepochsubmit = tk.Button(self.parent, text="Submit", width=smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont,
                                     command=self.SEMepochsubmitclick, relief='raised', bd=5, highlightbackground = widgetbg)
        self.SEMepochsubmit.grid(row=6, column=3)


        # box etc for entering the name of the directory for saving the results
        # make a label to show the current setting of the network definition file directory name
        self.SEMresultsdirlabel = tk.Label(self.parent, text = 'Results save folder:', font = labelfont)
        self.SEMresultsdirlabel.grid(row=7, column=1, sticky='N')
        self.SEMresultsdirtext = tk.StringVar()
        self.SEMresultsdirtext.set(self.SEMresultsdir)
        self.SEMresultsdirdisplay = tk.Label(self.parent, textvariable=self.SEMresultsdirtext, bg=bgcol, fg="#4B4B4B", font = labelfont,
                                     wraplength=300, justify='left')
        self.SEMresultsdirdisplay.grid(row=7, column=2, sticky='N')
        # the entry boxes need a "browse" button to allow selection of existing cluster definition file
        self.SEMresultsdirbrowse = tk.Button(self.parent, text = "Browse", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SEMresultsdirbrowseaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SEMresultsdirbrowse.grid(row=7, column=3, sticky='N')


        # box etc for entering the name used in labeling the results files
        self.SEMsavetaglabel = tk.Label(self.parent, text = 'tag for results names:', font = labelfont)
        self.SEMsavetaglabel.grid(row=8, column=1, sticky='N')
        self.SEMsavetagbox = tk.Entry(self.parent, width = 30, bg="white",justify = 'right')
        self.SEMsavetagbox.grid(row=8, column=2, sticky='N')
        self.SEMsavetagbox.insert(0,self.SEMsavetag)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.SEMsavetagsubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SEMsavetagsubmitaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SEMsavetagsubmit.grid(row=8, column=3, sticky='N')


        # label, button, for running the definition of clusters, and loading data
        self.SEMrun1sourcebutton = tk.Button(self.parent, text="Var/Cov", width=bigbigbuttonsize, bg=fgcol1, fg = fgletter1, font = widgetfont,
                                        command=self.SEMonesource, relief='raised', bd=5, highlightbackground = widgetbg)
        self.SEMrun1sourcebutton.grid(row=9, column=2)

        # label, button, for running the definition of clusters, and loading data
        self.SEMrun2sourcebutton = tk.Button(self.parent, text="2-source SEM", width=bigbigbuttonsize, bg=fgcol1, fg = fgletter1, font = widgetfont,
                                        command=self.SEMtwosource, relief='raised', bd=5, highlightbackground = widgetbg)
        self.SEMrun2sourcebutton.grid(row=10, column=2)

        # check to indicate to resume a failed run
        self.var1 = tk.IntVar()
        self.SEMresumebox = tk.Checkbutton(self.parent, text = 'Resume previous', width = bigbigbuttonsize, fg = fgletter2,
                                          command = self.SEMresumecheck, variable = self.var1, highlightbackground = widgetbg)
        self.SEMresumebox.grid(row = 11, column = 1, sticky="E")

        # label, button, for running the definition of clusters, and loading data
        self.SEMrunnetworkbutton = tk.Button(self.parent, text="Network SEM", width=bigbuttonsize, bg=fgcol1, fg = fgletter1, font = widgetfont,
                                        command=self.SEMrunnetwork, relief='raised', bd=5, highlightbackground = widgetbg)
        self.SEMrunnetworkbutton.grid(row=11, column=2)


#-----------Group-level analysis FRAME--------------------------------------------------
# Definition of the frame that has inputs for the database name, and entry numbers to use
class GRPFrame:

    def GRPcharacteristicslistclear(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]

        self.GRPcharacteristicscount = 0
        self.GRPcharacteristicslist = []
        self.GRPcharacteristicstext.set('empty')
        self.GRPcharacteristicsvalues = []
        self.GRPcharacteristicsvalues2 = []

        # in case the database has been updated
        # destroy the old pulldown menu and create a new one with the new choices
        print('Clearing the data base field choice list ....')
        self.fields = self.get_DB_fields()

        print('  new fields are:  ', self.fields)

        self.GRPfield_menu.destroy()
        self.fieldsearch_opt.destroy()  # remove it

        self.field_var = tk.StringVar()
        if len(self.fields) > 0:
            self.field_var.set(self.fields[0])
        else:
            self.field_var.set('empty')

        self.GRPfield_menu = tk.OptionMenu(self.parent, self.field_var, *self.fields, command=self.DBfieldchoice)
        self.GRPfield_menu.config(bg=bgcol)
        self.GRPfield_menu.grid(row=8, column=2, sticky='EW')
        self.fieldsearch_opt = self.GRPfield_menu  # save this way so that values are not cleared

        settings['GRPcharacteristicscount'] = self.GRPcharacteristicscount
        settings['GRPcharacteristicslist'] = self.GRPcharacteristicslist
        settings['GRPcharacteristicsvalues'] = self.GRPcharacteristicsvalues
        settings['GRPcharacteristicsvalues2'] = self.GRPcharacteristicsvalues2
        np.save(settingsfile,settings)

        return self


    def DBfieldchoice(self, value):
        # get the field value choices for the selected field
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.DBname = settings['DBname']
        self.DBnum = settings['DBnum']
        self.DBname2 = settings['DBname2']
        self.DBnum2 = settings['DBnum2']
        self.GRPcharacteristicscount = settings['GRPcharacteristicscount']

        print('GRPcharacteristicslist = ', self.GRPcharacteristicslist)
        fvalue = self.field_var.get()

        print('fvalue = ', fvalue)

        self.GRPcharacteristicscount += 1

        if self.GRPcharacteristicscount == 1:
            self.GRPcharacteristicslist =  [fvalue]
            fieldvalues, fieldvalues2 = GRPFrame.get_DB_field_values(self)
            print('size of fieldvalues is ',np.shape(fieldvalues))
            self.GRPcharacteristicsvalues = np.array(fieldvalues)[np.newaxis,:]
            print('size of GRPcharacteristicsvalues is ',np.shape(self.GRPcharacteristicsvalues))

            if os.path.isfile(self.DBname2):
                print('size of fieldvalues2 is ',np.shape(fieldvalues2))
                self.GRPcharacteristicsvalues2 = np.array(fieldvalues2)[np.newaxis,:]
                print('size of GRPcharacteristicsvalues2 is ',np.shape(self.GRPcharacteristicsvalues2))
        else:
            self.GRPcharacteristicslist.append(value)
            fieldvalues, fieldvalues2 = GRPFrame.get_DB_field_values(self)
            print('size of fieldvalues is ',np.shape(fieldvalues))
            print('size of GRPcharacteristicsvalues is ',np.shape(self.GRPcharacteristicsvalues))
            self.GRPcharacteristicsvalues = np.concatenate((self.GRPcharacteristicsvalues,np.array(fieldvalues)[np.newaxis,:]),axis=0)
            print('size of GRPcharacteristicsvalues is ',np.shape(self.GRPcharacteristicsvalues))

            if os.path.isfile(self.DBname2):
                print('size of fieldvalues2 is ',np.shape(fieldvalues2))
                print('size of GRPcharacteristicsvalues2 is ',np.shape(self.GRPcharacteristicsvalues2))
                self.GRPcharacteristicsvalues2 = np.concatenate((self.GRPcharacteristicsvalues2,np.array(fieldvalues2)[np.newaxis,:]),axis=0)
                print('size of GRPcharacteristicsvalues2 is ',np.shape(self.GRPcharacteristicsvalues2))

        print('GRPcharacteristicslist = ', self.GRPcharacteristicslist)

        chartext = ''
        for names in self.GRPcharacteristicslist:
            chartext += names + ','
        chartext = chartext[:-1]
        print('text for group characteristics list is: ',chartext)
        self.GRPcharacteristicstext.set(chartext)

        settings['GRPcharacteristicscount'] = self.GRPcharacteristicscount
        settings['GRPcharacteristicslist'] = self.GRPcharacteristicslist
        settings['GRPcharacteristicsvalues'] = self.GRPcharacteristicsvalues
        settings['GRPcharacteristicsvalues2'] = self.GRPcharacteristicsvalues2
        np.save(settingsfile,settings)

        # save a copy of the covariates list for other analyses
        p, f = os.path.split(self.DBname)
        covsavename = os.path.join(p, 'copy_of_covariates.npy')
        np.save(covsavename, {'GRPcharacteristicsvalues': self.GRPcharacteristicsvalues,
                              'GRPcharacteristicslist': self.GRPcharacteristicslist,
                              'GRPcharacteristicsvalues2': self.GRPcharacteristicsvalues2,
                              'GRPcharacteristicscount': self.GRPcharacteristicscount})

        return self


    # inputs to search database, and create/save dbnum lists
    def get_DB_fields(self):
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        self.DBname = settings['DBname']

        print('reading fields from ', self.DBname)
        if os.path.isfile(self.DBname):
            xls = pd.ExcelFile(self.DBname, engine = 'openpyxl')
            df1 = pd.read_excel(xls, 'datarecord')
            del df1['Unnamed: 0']  # get rid of the unwanted header column
            fields = list(df1.keys())
        else:
            fields = 'empty'
        # print('fields are: ',fields)
        return fields


    # inputs to search database, and create/save dbnum lists
    def get_DB_field_values(self, mode = 'average_per_person'):
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        DBname = settings['DBname']
        DBnum = settings['DBnum']
        prefix = settings['CLprefix']

        if os.path.isfile(DBname):
            xls = pd.ExcelFile(DBname, engine = 'openpyxl')
            df1 = pd.read_excel(xls, 'datarecord')
            del df1['Unnamed: 0']  # get rid of the unwanted header column

            print('GRPcharacteristicslist = ',self.GRPcharacteristicslist)
            fieldname = self.GRPcharacteristicslist[-1]
            print('fieldname = ',fieldname)

            if mode == 'average_per_person':  # average values over entries for the same person
                filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix, mode='list')
                fieldvalues = []
                for nn in range(NP):
                    DBnum_person = dbnum_person_list[nn]
                    fv1 = list(df1.loc[DBnum_person,fieldname])
                    if type(fv1[0]) == str:
                        # print('characteristic is {} value type ... using the first listed value'.format(type(fv1[0])))
                        fieldvalues += [fv1[0]]
                    else:
                        # print('characteristic is {} value type ... using the average value for each participant'.format(type(fv1[0])))
                        fieldvalues += [np.mean(fv1)]
            else:
                fieldvalues = list(df1.loc[DBnum,fieldname])
        else:
            fieldvalues = 'empty'
        # print('get_DB_field_values: fieldvalues = ',fieldvalues)

        #------------------------------------------------------------
        # in case there are two sets of data to be loaded/compared
        DBname2 = settings['DBname2']
        DBnum2 = settings['DBnum2']
        prefix = settings['CLprefix']

        if os.path.isfile(DBname2):
            xls = pd.ExcelFile(DBname2, engine = 'openpyxl')
            df1 = pd.read_excel(xls, 'datarecord')
            del df1['Unnamed: 0']  # get rid of the unwanted header column

            print('GRPcharacteristicslist2 = ',self.GRPcharacteristicslist)
            fieldname = self.GRPcharacteristicslist[-1]
            print('fieldname2 = ',fieldname)

            if mode == 'average_per_person':  # average values over entries for the same person
                filename_list2, dbnum_person_list2, NP2 = pydatabase.get_datanames_by_person(DBname2, DBnum2, prefix, mode='list')
                fieldvalues2 = []
                for nn in range(NP2):
                    DBnum_person2 = dbnum_person_list2[nn]
                    fv1 = list(df1.loc[DBnum_person2,fieldname])
                    if type(fv1[0]) == str:
                        # print('characteristic is {} value type ... using the first listed value'.format(type(fv1[0])))
                        fieldvalues2 += [fv1[0]]
                    else:
                        # print('characteristic is {} value type ... using the average value for each participant'.format(type(fv1[0])))
                        fieldvalues2 += [np.mean(fv1)]
            else:
                fieldvalues2 = list(df1.loc[DBnum2,fieldname])
        else:
            fieldvalues2 = 'empty'
        # print('get_DB_field_values: fieldvalues = ',fieldvalues)

        return fieldvalues, fieldvalues2


    def GRPresultsbrowseaction(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        # use a dialog box to prompt the user to select an existing file, the default being .xlsx type
        filechoice =  tkf.askopenfilename(title = "Select results file",filetypes = (("npy files","*.npy"),("all files","*.*")))
        print('results data file name = ',filechoice)
        self.GRPresultsname = filechoice
        settings['GRPresultsname'] = filechoice

        npname, nfname = os.path.split(self.GRPresultsname)
        # write the result to the label box for display
        self.GRPresultsnametext.set(nfname)
        self.GRPresultsdirtext.set(npname)

        # check on the results
        try:
            data = np.load(filechoice, allow_pickle = True).flat[0]
            keylist = data.keys()
            datafiletype = 0
            if 'type' in keylist:
                if data['type'] == 'GLM':
                    datafiletype = 3
                else:
                    datafiletype = 1
            if 'region_properties' in keylist: datafiletype = 2
        except:
            print('Error reading selected data file - unexpected contents or format')
            return

        if datafiletype == 0:  print('selected data file does not have the required format')
        if datafiletype == 1:  print('found SEM results: for group comparisons be sure to select 2 results files')
        if datafiletype == 2:  print('found time-course data: select correlation as the analysis type in order to do Bayesian regression')
        if datafiletype == 3:  print('found GLM results')

        settings['GRPdatafiletype1'] = datafiletype
        settings['DBname'] = data['DBname']
        settings['DBnum'] = data['DBnum']
        self.DBname = settings['DBname']
        self.DBnum = settings['DBnum']
        np.save(settingsfile, settings)

        # update Database information
        # DBFrame.DBhandle.DBnametext.set(self.DBname)
        # value_list_for_display = DBFrame.DBdisplaynumlist(self.DBnum)
        # DBFrame.DBhandle.DBnumsave_text = value_list_for_display
        # settings['DBnum'] = entered_values
        # settings['DBnumstring'] = DBFrame.DBhandle.DBnumsave_text
        # DBFrame.DBhandle.DBnumenter.insert(0,settings['DBnumstring'])

        # clear the characteristics list
        self.GRPcharacteristicslistclear()

        self.GRPresultsnamebrowse2['state'] = tk.NORMAL
        self.GRPresultsnameclear2['state'] = tk.NORMAL
        self.GRPresultsnameswap['state'] = tk.NORMAL

        return self


    def GRPresultsclearaction(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.GRPresultsname = 'notdefined'
        settings['GRPresultsname'] = 'notdefined'
        # write the result to the label box for display
        self.GRPresultsnametext.set('notdefined')
        self.GRPresultsdirtext.set('')

        self.GRPresultsname2 = 'notdefined'
        settings['GRPresultsname2'] = 'notdefined'
        # write the result to the label box for display
        self.GRPresultsnametext2.set('notdefined')
        self.GRPresultsdirtext2.set('')

        settings['GRPdatafiletype1'] = 0
        settings['GRPdatafiletype2'] = 0
        settings['DBname2'] = 'none'
        settings['DBnum2'] = 'none'
        self.DBname2 = 'none'
        self.DBnum2 = 'none'
        np.save(settingsfile, settings)

        self.GRPresultsnamebrowse2['state'] = tk.DISABLED
        self.GRPresultsnameclear2['state'] = tk.DISABLED

        return self

    def GRPresultsbrowseaction2(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        # use a dialog box to prompt the user to select an existing file, the default being .xlsx type
        filechoice = tkf.askopenfilename(title="Select results file 2",
                                         filetypes=(("npy files", "*.npy"), ("all files", "*.*")))
        print('results data file name 2 = ', filechoice)
        self.GRPresultsname2 = filechoice
        settings['GRPresultsname2'] = filechoice

        npname, nfname = os.path.split(self.GRPresultsname2)
        # write the result to the label box for display
        self.GRPresultsnametext2.set(nfname)
        self.GRPresultsdirtext2.set(npname)

        # check on the results
        try:
            data = np.load(filechoice, allow_pickle = True).flat[0]
            keylist = data.keys()
            datafiletype = 0
            if 'type' in keylist:
                if data['type'] == 'GLM':
                    datafiletype = 3
                else:
                    datafiletype = 1
            if 'region_properties' in keylist: datafiletype = 2
        except:
            print('Error reading selected data file - unexpected contents or format')
            return

        if datafiletype == 0:  print('selected data file does not have the required format')
        if datafiletype == 1:  print('found SEM results: for group comparisons be sure to select 2 results files')
        if datafiletype == 2:  print('found time-course data: select correlation as the analysis type in order to do Bayesian regression')
        if datafiletype == 3:  print('found GLM results')

        settings['GRPdatafiletype2'] = datafiletype
        settings['DBname2'] = data['DBname']
        settings['DBnum2'] = data['DBnum']
        self.DBname2 = settings['DBname2']
        self.DBnum2 = settings['DBnum2']
        np.save(settingsfile,settings)  # get_DB_field_values will use the values in the settings file

        # load characteristics values if fields have already been selected-------------------
        GRPcharacteristicslist_copy = copy.deepcopy(self.GRPcharacteristicslist)
        for num, fvalue in enumerate(GRPcharacteristicslist_copy):
            self.GRPcharacteristicslist = [fvalue]
            print('loading characteristics for field: {}'.format(fvalue))
            fieldvalues, fieldvalues2 = GRPFrame.get_DB_field_values(self)
            if num == 0:
                self.GRPcharacteristicsvalues2 = np.array(fieldvalues2)[np.newaxis,:]
            else:
                self.GRPcharacteristicsvalues2 = np.concatenate((self.GRPcharacteristicsvalues2,np.array(fieldvalues2)[np.newaxis,:]),axis=0)

        print('size of GRPcharacteristicsvalues2 is ', np.shape(self.GRPcharacteristicsvalues2))

        self.GRPcharacteristicslist = GRPcharacteristicslist_copy
        settings['GRPcharacteristicsvalues2'] = self.GRPcharacteristicsvalues2
        np.save(settingsfile,settings)


    def GRPresultsclearaction2(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.GRPresultsname2 = 'notdefined'
        settings['GRPresultsname2'] = 'notdefined'

        # write the result to the label box for display
        self.GRPresultsnametext2.set('notdefined')
        self.GRPresultsdirtext2.set('')

        settings['GRPdatafiletype2'] = 0
        settings['DBname2'] = 'none'
        settings['DBnum2'] = 'none'
        self.DBname2 = 'none'
        self.DBnum2 = 'none'
        np.save(settingsfile, settings)

        return self


    def GRPresultsswapaction(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]

        filename1 = settings['GRPresultsname']
        filename2 = settings['GRPresultsname2']
        datafiletype1 = settings['GRPdatafiletype1']
        datafiletype2 = settings['GRPdatafiletype2']
        DBname1 = settings['DBname']
        DBname2 = settings['DBname2']
        DBnum1 = settings['DBnum']
        DBnum2 = settings['DBnum2']

        # swap all the settings
        newfname1 = filename2
        newfname2 = filename1

        newtype1 = datafiletype2
        newtype2 = datafiletype1

        newDBnum1 = DBnum2
        newDBnum2 = DBnum1

        newDBname1 = DBname2
        newDBname2 = DBname1

        self.GRPresultsname = newfname1
        settings['GRPresultsname'] = newfname1
        settings['GRPdatafiletype1'] = newtype1
        settings['DBname'] = newDBname1
        settings['DBnum'] = newDBnum1
        self.DBname = newDBname1
        self.DBnum = newDBnum1

        # need to update dbnumlist in the DBframe
        value_list_for_display = self.controller.DBhandle.DBdisplaynumlist(newDBnum1)
        settings['DBnumstring'] = value_list_for_display
        self.controller.DBhandle.DBnumenter.delete(0, 'end')
        self.controller.DBhandle.DBnumenter.insert(0, settings['DBnumstring'])

        self.GRPresultsname2 = newfname2
        settings['GRPresultsname2'] = newfname2
        settings['GRPdatafiletype2'] = newtype2
        settings['DBname2'] = newDBname2
        settings['DBnum2'] = newDBnum2
        self.DBname2 = newDBname2
        self.DBnum2 = newDBnum2

        # write the result to the label box for display
        npname, nfname = os.path.split(self.GRPresultsname)
        self.GRPresultsnametext.set(nfname)
        self.GRPresultsdirtext.set(npname)

        # write the result to the label box for display
        npname, nfname = os.path.split(self.GRPresultsname2)
        self.GRPresultsnametext2.set(nfname)
        self.GRPresultsdirtext2.set(npname)

        np.save(settingsfile, settings)   # need to save during intermediate steps because
                                    # loading the characteristics values checks the settings file
        # update the group characteristics lists if they have been chosen already------
        GRPcharacteristicslist_copy = copy.deepcopy(self.GRPcharacteristicslist)
        for num, fvalue in enumerate(GRPcharacteristicslist_copy):
            self.GRPcharacteristicslist = [fvalue]
            print('loading characteristics for field: {}'.format(fvalue))
            fieldvalues, fieldvalues2 = GRPFrame.get_DB_field_values(self)
            if num == 0:
                self.GRPcharacteristicsvalues = np.array(fieldvalues)[np.newaxis,:]
                self.GRPcharacteristicsvalues2 = np.array(fieldvalues2)[np.newaxis,:]
            else:
                self.GRPcharacteristicsvalues = np.concatenate((self.GRPcharacteristicsvalues,np.array(fieldvalues)[np.newaxis,:]),axis=0)
                self.GRPcharacteristicsvalues2 = np.concatenate((self.GRPcharacteristicsvalues2,np.array(fieldvalues2)[np.newaxis,:]),axis=0)

        print('size of GRPcharacteristicsvalues is ', np.shape(self.GRPcharacteristicsvalues))
        print('size of GRPcharacteristicsvalues2 is ', np.shape(self.GRPcharacteristicsvalues2))

        self.GRPcharacteristicslist = GRPcharacteristicslist_copy
        settings['GRPcharacteristicsvalues'] = self.GRPcharacteristicsvalues
        settings['GRPcharacteristicsvalues2'] = self.GRPcharacteristicsvalues2

        self.GRPresultsnamebrowse2['state'] = tk.NORMAL
        self.GRPresultsnameclear2['state'] = tk.NORMAL
        self.GRPresultsnameswap['state'] = tk.NORMAL
        np.save(settingsfile, settings)

        return self


    def GRPpvaluesubmit(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        GRPpvalue = float(self.GRPpvaluebox.get())
        settings['GRPpvalue'] = GRPpvalue
        np.save(settingsfile,settings)
        # update the text in the box, in case it has changed
        self.GRPpvalue = GRPpvalue
        print('p-value for GRP analysis set to ',self.GRPpvalue)

        return self


    # action when checkboxes are selected/deselected
    def GRPselecttype(self):
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        value = self.GRPanalysistypevalue.get()
        if value == 1: self.GRPanalysistype = 'Sig1'
        if value == 2: self.GRPanalysistype = 'Sig2'
        if value == 3: self.GRPanalysistype = 'Sig2paired'
        if value == 4: self.GRPanalysistype = 'Correlation'
        if value == 5: self.GRPanalysistype = 'Regression'
        if value == 6: self.GRPanalysistype = 'ANOVA'
        if value == 7: self.GRPanalysistype = 'ANCOVA'
        if value == 8: self.GRPanalysistype = 'Time'

        print('Group analysis type set to: ',self.GRPanalysistype)

        if value == 2:
            print('Sig2:  be sure to indicate two sets of results to compare')
        if value == 8:
            print('Time:  only indicate one set of results (first set will be used)')
        if value == 3:
            print('Sig2paired:  be sure to indicate two sets of results to compare')
        if value == 4:
            print('Correlation:  only indicate one set of results (first set will be used)')
        if value == 5:
            print('Regression:  only indicate one set of results (first set will be used)')
        if value == 6:
            print('ANOVA:  if two sets of results are indicated, they will be considered to be separate groups')
            print('         only indicate one additional discrete value for the personal characteristic')
            print('ANOVA:  if one set of results is indicated, then two discrete values need to chosen, for the personal characteristics')
        if value == 7:
            print('ANCOVA:  if two sets of results are indicated, they will be considered to be separate groups')
            print('         only indicate one additional continuous value for the personal characteristic')
            print('ANCOVA:  if one set of results is indicated, then one discrete value needs to chosen, and a continuous value chosen 2nd, for the personal characteristics')

        settings['GRPanalysistype'] = self.GRPanalysistype
        np.save(settingsfile,settings)

        return self


    def GRPrunanalysis(self):
        # run the selected analyses
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        GRPanalysistype = settings['GRPanalysistype']
        # Sig1, Sig2, Sig2paired, Correlation, ANOVA, or ANCOVA

        # check on the data to be used
        datafile1 = settings['GRPresultsname']
        datafile2 = settings['GRPresultsname2']
        datafiletype1 = settings['GRPdatafiletype1']
        datafiletype2 = settings['GRPdatafiletype2']
        GRPcharacteristicslist = settings['GRPcharacteristicslist']
        GRPcharacteristicsvalues = settings['GRPcharacteristicsvalues']
        GRPcharacteristicsvalues2 = settings['GRPcharacteristicsvalues2']
        GRPpvalue = settings['GRPpvalue']

        print('GRPrunanalysis: type: ',GRPanalysistype, '  with data file type1: ',datafiletype1, '  file type2: ',datafiletype2)

        # for ANOVA/ANCOVA - could have two data files (Sig2) as a discrete category,
        # or one data file with two characteristics select

        if GRPanalysistype == 'Sig1':
            pthreshold = GRPpvalue
            outputfilename = py2ndlevelanalysis.group_significance(datafile1, pthreshold, 'average', covariates='none')

            # if datafiletype1 == 1:  # SEM data
            #     # look for significant group-average beta-value differences from zero
            #     # sem_results = np.load(datafile1, allow_pickle=True).flat[0]
            #     pthreshold = GRPpvalue
            #     outputfilename = py2ndlevelanalysis.group_significance(datafile1, pthreshold, statstype='average', covariates='none')
            #
            # if datafiletype1 == 2:  # BOLD data
            #     # look for significant group-average BOLD response differences from zero

        if GRPanalysistype == 'Sig2':
            pthreshold = GRPpvalue
            outputfilename = py2ndlevelanalysis.group_difference_significance(datafile1, datafile2, pthreshold,'unpaired','average', covariates='none')

            # if datafiletype1 == 1:  # SEM data
            #     # look for significant group-average beta-value differences between two groups
            #     if datafiletype2 == 1:
            #         # go ahead and do the comparison, otherwise quit
            #         outputfilename = py2ndlevelanalysis.group_difference_significance(datafile1, datafile2, pthreshold,'unpaired','average', covariates='none')
            #
            # if datafiletype1 == 2:  # BOLD data
            #     # look for significant group-average BOLD response differences between two groups
            #     if datafiletype2 == 2:
            #         # go ahead and do the comparison, otherwise quit
            #         outputfilename = py2ndlevelanalysis.group_difference_significance(datafile1, datafile2, pthreshold,'paired','average', covariates='none')

        if GRPanalysistype == 'Sig2paired':
            pthreshold = GRPpvalue
            outputfilename = py2ndlevelanalysis.group_difference_significance(datafile1, datafile2, pthreshold,'paired','average', covariates='none')

            # if datafiletype1 == 1:  # SEM data
            #     # look for significant group-average beta-value paired differences between two groups
            #     if datafiletype2 == 1:
            #         # go ahead and do the comparison, otherwise quit
            #         print('hold this')
            #
            # if datafiletype1 == 2:  # BOLD data
            #     # look for significant group-average BOLD response paired differences between two groups
            #     if datafiletype2 == 2:
            #         # go ahead and do the comparison, otherwise quit
            #         print('hold this')

        if GRPanalysistype == 'Correlation':
            pthreshold = GRPpvalue
            covariates = GRPcharacteristicsvalues
            covariatesnames = GRPcharacteristicslist
            outputfilename = py2ndlevelanalysis.group_significance(datafile1, pthreshold, statstype='correlation', covariates=covariates, covnames=covariatesnames)

            # if datafiletype1 == 1:
            #     # look for significant correlations between beta-values and the first personal characteristic in the list
            #     pthreshold = GRPpvalue
            #     covariates = GRPcharacteristicsvalues
            #     outputfilename = py2ndlevelanalysis.group_significance(datafile1, pthreshold, statstype='correlation', covariates=covariates)
            #
            # if datafiletype1 == 2:  # BOLD data
            #     # look for significant correlations between BOLD values and the first personal characteristic in the list
            #     pthreshold = GRPpvalue
            #     covariates = GRPcharacteristicsvalues
            #     outputfilename = py2ndlevelanalysis.group_significance(datafile1, pthreshold, statstype='correlation', covariates=covariates)

        if GRPanalysistype == 'Regression':
            pthreshold = GRPpvalue
            covariates = GRPcharacteristicsvalues
            covariatesnames = GRPcharacteristicslist
            outputfilename = py2ndlevelanalysis.group_significance(datafile1, pthreshold, statstype='regression', covariates=covariates,covnames=covariatesnames)

            # if datafiletype1 == 1:
            #     # look for significant correlations between beta-values and the first personal characteristic in the list
            #     pthreshold = GRPpvalue
            #     covariates = GRPcharacteristicsvalues
            #     outputfilename = py2ndlevelanalysis.group_significance(datafile1, pthreshold, statstype='regression', covariates=covariates)
            #
            # if datafiletype1 == 2:  # BOLD data
            #     # look for significant correlations between BOLD values and the first personal characteristic in the list
            #     pthreshold = GRPpvalue
            #     covariates = GRPcharacteristicsvalues
            #     outputfilename = py2ndlevelanalysis.group_significance(datafile1, pthreshold, statstype='regression', covariates=covariates)
            #

        if GRPanalysistype == 'Time':
            pthreshold = GRPpvalue
            outputfilename = py2ndlevelanalysis.group_significance(datafile1, pthreshold, 'time', covariates='none')


        if GRPanalysistype == 'ANOVA':
            if datafiletype1 == 1:  # SEM data
                if datafiletype2 == 1:
                    # apply ANOVA analysis to beta-values, with the two files indicating two discrete groups,
                    # and the first personal characteristic used as a second discrete variable
                    pthreshold = GRPpvalue
                    covariates1 = GRPcharacteristicsvalues[0,:]
                    covariates2 = GRPcharacteristicsvalues2[0,:]
                    covname = GRPcharacteristicslist[0]
                    outputfilename = py2ndlevelanalysis.group_comparison_ANOVA(datafile1, datafile2, covariates1, covariates2, pthreshold, mode = 'ANOVA', covariate_name = covname)
                else:
                    # apply ANOVA analysis to beta-values in the first data file named,
                    # with the first two personal characteristics used as discrete variables
                    pthreshold = GRPpvalue
                    covariates1 = GRPcharacteristicsvalues[:2,:]
                    covname = GRPcharacteristicslist[:2]
                    # outputfilename = py2ndlevelanalysis.group_comparison_ANOVA(datafile1, 'none', covariates1, 'none', pthreshold, mode = 'ANOVA', covariate_name = covname)
                    outputfilename = py2ndlevelanalysis.single_group_ANOVA(datafile1, covariates1, pthreshold, mode='ANOVA', covariate_names=covname)

            if datafiletype1 == 2:  # BOLD data
                if datafiletype2 == 2:
                    # apply ANOVA analysis to beta-values, with the two files indicating two discrete groups,
                    # and the first personal characteristic used as a second discrete variable
                    pthreshold = GRPpvalue
                    covariates1 = GRPcharacteristicsvalues[0,:]
                    covariates2 = GRPcharacteristicsvalues2[0,:]
                    covname = GRPcharacteristicslist[0]
                    outputfilename = py2ndlevelanalysis.group_comparison_ANOVA(datafile1, datafile2, covariates1, covariates2, pthreshold, mode = 'ANOVA', covariate_name = covname)
                else:
                    # apply ANOVA analysis to beta-values in the first data file named,
                    # with the first two personal characteristics used as discrete variables
                    pthreshold = GRPpvalue
                    covariates1 = GRPcharacteristicsvalues[:2,:]
                    covname = GRPcharacteristicslist[:2]
                    # outputfilename = py2ndlevelanalysis.group_comparison_ANOVA(datafile1, 'none', covariates1, 'none', pthreshold, mode = 'ANOVA', covariate_name = covname)
                    outputfilename = py2ndlevelanalysis.single_group_ANOVA(datafile1, covariates1, pthreshold, mode='ANOVA', covariate_names=covname)


        if GRPanalysistype == 'ANCOVA':
            if datafiletype1 == 1:  # SEM data
                if datafiletype2 == 1:
                    # apply ANOVA analysis to beta-values, with the two files indicating two discrete groups,
                    # and the first personal characteristic used as a continuous variable
                    pthreshold = GRPpvalue
                    covariates1 = GRPcharacteristicsvalues[0,:]
                    covariates2 = GRPcharacteristicsvalues2[0,:]
                    covname = GRPcharacteristicslist[0]
                    outputfilename = py2ndlevelanalysis.group_comparison_ANOVA(datafile1, datafile2, covariates1, covariates2, pthreshold, mode = 'ANCOVA', covariate_name = covname)
                else:
                    # apply ANOVA analysis to beta-values in the first data file named,
                    # with the first two personal characteristics used as one discrete and one continuous variable
                    pthreshold = GRPpvalue
                    covariates1 = GRPcharacteristicsvalues[:2,:]
                    covname = GRPcharacteristicslist[:2]
                    # outputfilename = py2ndlevelanalysis.group_comparison_ANOVA(datafile1, 'none', covariates1, 'none', pthreshold, mode = 'ANCOVA', covariate_name = covname)
                    outputfilename = py2ndlevelanalysis.single_group_ANOVA(datafile1, covariates1, pthreshold, mode='ANCOVA', covariate_names=covname)


            if datafiletype1 == 2:  # BOLD data
                if datafiletype2 == 2:
                    # apply ANOVA analysis to beta-values, with the two files indicating two discrete groups,
                    # and the first personal characteristic used as a continuous variable
                    pthreshold = GRPpvalue
                    covariates1 = GRPcharacteristicsvalues[0,:]
                    covariates2 = GRPcharacteristicsvalues2[0,:]
                    covname = GRPcharacteristicslist[0]
                    outputfilename = py2ndlevelanalysis.group_comparison_ANOVA(datafile1, datafile2, covariates1, covariates2, pthreshold, mode = 'ANCOVA', covariate_name = covname)
                else:
                    # apply ANOVA analysis to beta-values in the first data file named,
                    # with the first two personal characteristics used as one discrete and one continuous variable
                    pthreshold = GRPpvalue
                    covariates1 = GRPcharacteristicsvalues[:2,:]
                    covname = GRPcharacteristicslist[:2]
                    # outputfilename = py2ndlevelanalysis.group_comparison_ANOVA(datafile1, 'none', covariates1, 'none', pthreshold, mode = 'ANCOVA', covariate_name = covname)
                    outputfilename = py2ndlevelanalysis.single_group_ANOVA(datafile1, covariates1, pthreshold, mode='ANCOVA', covariate_names=covname)


    def GRPmake2groups(self):
        # split data file into two groups
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        GRPanalysistype = settings['GRPanalysistype']
        # Sig1, Sig2, Sig2paired, Correlation, ANOVA, or ANCOVA

        # check on the data to be used
        datafile1 = settings['GRPresultsname']
        datafile2 = settings['GRPresultsname2']
        datafiletype1 = settings['GRPdatafiletype1']
        datafiletype2 = settings['GRPdatafiletype2']
        GRPcharacteristicslist = settings['GRPcharacteristicslist']
        GRPcharacteristicsvalues = settings['GRPcharacteristicsvalues']
        GRPcharacteristicsvalues2 = settings['GRPcharacteristicsvalues2']
        GRPpvalue = settings['GRPpvalue']

        if len(GRPcharacteristicslist) == 0:
            print('GRPmake2groups:  no characteristics selected ...')
            print('A categorical group characteristic must be selected that defines two distinct groups, for splitting a data file')
            return 0
        else:
            print('GRPmake2groups: with data file type: ',datafiletype1, '  into groups based on: ',GRPcharacteristicslist[0])

        # get the group names and indices------------------------------------------------
        if np.ndim(GRPcharacteristicsvalues) > 1:
            groups, ii = np.unique(GRPcharacteristicsvalues[0, :], return_inverse=True)
        else:
            groups, ii = np.unique(GRPcharacteristicsvalues, return_inverse=True)
        g1 = np.where(ii == 0)[0]
        g2 = np.where(ii == 1)[0]

        if len(groups) > 2:
            print('GRPmake2groups: more than two groups indicated by selected characteristic - cannot proceed. ')
            return 0

        # load the data----------------------------------------------------------------
        data = np.load(datafile1, allow_pickle=True).flat[0]

        filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(data['DBname'], data['DBnum'], '', mode='list')
        DBnumlist1 = []
        for x in g1: DBnumlist1 += dbnum_person_list[x][:]
        DBnumlist2 = []
        for x in g2: DBnumlist2 += dbnum_person_list[x][:]

        if datafiletype1 == 0:  print('selected data file does not have the required format')

        if datafiletype1 == 1:
            print('SEM results selected')
            if data['type'] == '2source':
                # dict_keys(['type', 'CCrecord', 'beta2', 'beta1', 'Zgrid2', 'Zgrid1_1', 'Zgrid1_2', 'DBname', 'DBnum',
                #            'cluster_properties'])
                # beta1  [nr x nr x nr x ntime x NP x 2]
                # beta2  [nr x nr x nr x ntime x NP x 2]
                # CCrecord  [ntime x NP x nr x nr]
                # Zgrid1_1  [nr x nr x nr x ntime x NP]
                # Zgrid1_2  [nr x nr x nr x ntime x NP]
                # Zgrid2  [nr x nr x nr x ntime x NP]

                data1 = {'type':data['type'], 'beta1':data['beta1'][:,:,:,:,g1,:], 'beta2':data['beta2'][:,:,:,:,g1,:],
                         'CCrecord': data['CCrecord'][:, g1, :, :], 'Zgrid1_1':data['Zgrid1_1'][:,:,:,:,g1],
                         'Zgrid1_2':data['Zgrid1_2'][:,:,:,:,g1], 'Zgrid2':data['Zgrid2'][:,:,:,:,g1],
                         'DBname':data['DBname'],'DBnum':DBnumlist1, 'cluster_properties':data['cluster_properties']}

                data2 = {'type':data['type'], 'beta1':data['beta1'][:,:,:,:,g2,:], 'beta2':data['beta2'][:,:,:,:,g2,:],
                         'CCrecord': data['CCrecord'][:, g2, :, :], 'Zgrid1_1':data['Zgrid1_1'][:,:,:,:,g2],
                         'Zgrid1_2':data['Zgrid1_2'][:,:,:,:,g2], 'Zgrid2':data['Zgrid2'][:,:,:,:,g2],
                         'DBname':data['DBname'],'DBnum':DBnumlist2, 'cluster_properties':data['cluster_properties']}

                # setup new names
                [p,f] = os.path.split(datafile1)
                [f1,e] = os.path.splitext(f)
                newdatafile1 = os.path.join(p, '{}_{}{}'.format(f1,groups[0],e))
                newdatafile2 = os.path.join(p, '{}_{}{}'.format(f1,groups[1],e))

                # write the new data files
                np.save(newdatafile1,data1)
                print('GRPmake2groups: writing results to {}'.format(newdatafile1))
                np.save(newdatafile2,data2)
                print('GRPmake2groups: writing results to {}'.format(newdatafile2))

            if data['type'] == 'network':
                # dict_keys(['type', 'resultsnames', 'network', 'regionname', 'clustername', 'DBname', 'DBnum'])
                # data in regionname
                # dict_keys(['region_properties', 'DBname', 'DBnum'])
                # array of region properties
                # dict_keys(['tc', 'tc_sem', 'nruns_per_person', 'tsize', 'rname'])
                # tc [nclusters x tsize*nruns]
                # tc_sem [nclusters x tsize*nruns]
                # nruns_per_person [NP]

                # data in regionname
                rdatafile1 = data['regionname']
                rdata1 = np.load(rdatafile1, allow_pickle=True).flat[0]

                nregions = len(rdata1['region_properties'])
                nruns_per_person = rdata1['region_properties'][0]['nruns_per_person']
                tsize = rdata1['region_properties'][0]['tsize']

                nruns_per_person1 = nruns_per_person[g1]
                nruns_per_person2 = nruns_per_person[g2]

                # create list of timepoints for data in group1---------------------
                tp_list1 = []
                for gg in g1:
                    r1 = np.sum(nruns_per_person[:gg])*tsize
                    r2 = np.sum(nruns_per_person[:(gg+1)])*tsize
                    tp_person = list(range(r1,r2))
                    tp_list1 += tp_person
                # create list of timepoints for data in group2---------------------
                tp_list2 = []
                for gg in g2:
                    r1 = np.sum(nruns_per_person[:gg])*tsize
                    r2 = np.sum(nruns_per_person[:(gg+1)])*tsize
                    tp_person = list(range(r1,r2))
                    tp_list2 += tp_person

                region_properties1 = copy.deepcopy(rdata1['region_properties'])
                region_properties2 = copy.deepcopy(rdata1['region_properties'])
                for nn in range(nregions):
                    region_properties1[nn]['tc'] = region_properties1[nn]['tc'][:,tp_list1]
                    region_properties1[nn]['tc_sem'] = region_properties1[nn]['tc_sem'][:,tp_list1]
                    region_properties1[nn]['nruns_per_person'] = nruns_per_person1

                    region_properties2[nn]['tc'] = region_properties2[nn]['tc'][:,tp_list2]
                    region_properties2[nn]['tc_sem'] = region_properties2[nn]['tc_sem'][:,tp_list2]
                    region_properties2[nn]['nruns_per_person'] = nruns_per_person2

                # dict_keys(['region_properties', 'DBname', 'DBnum'])
                new_regiondata1 = {'region_properties':region_properties1,'DBname':rdata1['DBname'],'DBnum':DBnumlist1}
                new_regiondata2 = {'region_properties':region_properties2,'DBname':rdata1['DBname'],'DBnum':DBnumlist2}

                # new region name
                [p,f] = os.path.split(rdatafile1)
                [f1,e] = os.path.splitext(f)
                new_regionname1 = os.path.join(p, '{}_{}{}'.format(f1,groups[0],e))
                new_regionname2 = os.path.join(p, '{}_{}{}'.format(f1,groups[1],e))

                # write the new region data files
                np.save(new_regionname1,new_regiondata1)
                print('GRPmake2groups: writing results to {}'.format(new_regionname1))
                np.save(new_regionname2,new_regiondata2)
                print('GRPmake2groups: writing results to {}'.format(new_regionname2))

                # data in clustername is the same for every group

                networkname = data['network']
                network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkname)

                # data in resultsnames
                resultsnames = data['resultsnames']
                new_resultsnames1 = copy.deepcopy(resultsnames)
                new_resultsnames2 = copy.deepcopy(resultsnames)
                ncomponents = len(resultsnames)
                for nn in range(ncomponents):
                    # array of resultsfields
                    results = np.load(resultsnames[nn], allow_pickle=True).flat[0]
                    results_g1 = copy.deepcopy(results)
                    results_g2 = copy.deepcopy(results)
                    ntargetclusters = len(results['sem_one_target_results'])
                    for nt in range(ntargetclusters):
                        # b  [ncombo x ntime x NP x 6]
                        # R2 [ncombo x ntime x NP]
                        results_g1['sem_one_target_results'][nt]['b'] = results['sem_one_target_results'][nt]['b'][:,:,g1,:]
                        results_g1['sem_one_target_results'][nt]['R2'] = results['sem_one_target_results'][nt]['R2'][:,:,g1]

                        results_g2['sem_one_target_results'][nt]['b'] = results['sem_one_target_results'][nt]['b'][:,:,g2,:]
                        results_g2['sem_one_target_results'][nt]['R2'] = results['sem_one_target_results'][nt]['R2'][:,:,g2]

                    # new names
                    # find the old name tag and add to it
                    target = network[nn]['target']
                    spos = resultsnames[nn].find(target)
                    rn1 = resultsnames[nn][:spos] + groups[0] + '_' + resultsnames[nn][spos:]
                    rn2 = resultsnames[nn][:spos] + groups[1] + '_' + resultsnames[nn][spos:]

                    new_resultsnames1[nn] = rn1
                    new_resultsnames2[nn] = rn2

                    # write the new region data files
                    np.save(rn1, results_g1)
                    print('GRPmake2groups: writing results to {}'.format(rn1))
                    np.save(rn2, results_g2)
                    print('GRPmake2groups: writing results to {}'.format(rn2))

                # write out new
                new_data1 = {'type': data['type'], 'resultsnames':new_resultsnames1, 'network':data['network'],
                         'regionname':new_regionname1, 'clustername':data['clustername'],
                         'DBname':data['DBname'], 'DBnum':DBnumlist1}

                new_data2 = {'type': data['type'], 'resultsnames':new_resultsnames2, 'network':data['network'],
                         'regionname':new_regionname2, 'clustername':data['clustername'],
                         'DBname':data['DBname'], 'DBnum':DBnumlist2}

                # setup new names
                [p,f] = os.path.split(datafile1)
                [f1,e] = os.path.splitext(f)
                newdatafile1 = os.path.join(p, '{}_{}{}'.format(f1,groups[0],e))
                newdatafile2 = os.path.join(p, '{}_{}{}'.format(f1,groups[1],e))

                # write the new data files
                np.save(newdatafile1, new_data1)
                print('GRPmake2groups: writing results to {}'.format(newdatafile1))
                np.save(newdatafile2, new_data2)
                print('GRPmake2groups: writing results to {}'.format(newdatafile2))

        if datafiletype1 == 2:
            print('time-course data selected .... ')

            # dict_keys(['region_properties', 'DBname', 'DBnum'])
            region_properties = data['region_properties']
            DBname = data['DBname']
            nregions = len(region_properties)
            nruns_per_person = region_properties[0]['nruns_per_person']
            tsize = region_properties[0]['tsize']

            nruns_per_person1 = nruns_per_person[g1]
            nruns_per_person2 = nruns_per_person[g2]

            # create list of timepoints for data in group1---------------------
            tp_list1 = []
            for gg in g1:
                r1 = np.sum(nruns_per_person[:gg]) * tsize
                r2 = np.sum(nruns_per_person[:(gg + 1)]) * tsize
                tp_person = list(range(r1, r2))
                tp_list1 += tp_person
            # create list of timepoints for data in group2---------------------
            tp_list2 = []
            for gg in g2:
                r1 = np.sum(nruns_per_person[:gg]) * tsize
                r2 = np.sum(nruns_per_person[:(gg + 1)]) * tsize
                tp_person = list(range(r1, r2))
                tp_list2 += tp_person

            region_properties1 = copy.deepcopy(region_properties)
            region_properties2 = copy.deepcopy(region_properties)
            for nn in range(nregions):
                region_properties1[nn]['tc'] = region_properties1[nn]['tc'][:, tp_list1]
                region_properties1[nn]['tc_sem'] = region_properties1[nn]['tc_sem'][:, tp_list1]
                region_properties1[nn]['nruns_per_person'] = nruns_per_person1

                region_properties2[nn]['tc'] = region_properties2[nn]['tc'][:, tp_list2]
                region_properties2[nn]['tc_sem'] = region_properties2[nn]['tc_sem'][:, tp_list2]
                region_properties2[nn]['nruns_per_person'] = nruns_per_person2

            # dict_keys(['region_properties', 'DBname', 'DBnum'])
            new_regiondata1 = {'region_properties': region_properties1, 'DBname':DBname, 'DBnum': DBnumlist1}
            new_regiondata2 = {'region_properties': region_properties2, 'DBname':DBname, 'DBnum': DBnumlist2}

            # new region name
            [p, f] = os.path.split(datafile1)
            [f1, e] = os.path.splitext(f)
            new_regionname1 = os.path.join(p, '{}_{}{}'.format(f1, groups[0], e))
            new_regionname2 = os.path.join(p, '{}_{}{}'.format(f1, groups[1], e))

            # write the new region data files
            np.save(new_regionname1, new_regiondata1)
            print('GRPmake2groups: writing results to {}'.format(new_regionname1))
            np.save(new_regionname2, new_regiondata2)
            print('GRPmake2groups: writing results to {}'.format(new_regionname2))

        print('Finished splitting results files into two groups.')
        return self

    def __init__(self, parent, controller):
        parent.configure(relief='raised', bd=5, highlightcolor=fgcol3)
        self.parent = parent
        self.controller = controller

        # initialize some values
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.DBname = settings['DBname']
        self.DBnum = settings['DBnum']
        self.DBname2 = settings['DBname2']   # for 2nd set of comparison results
        self.DBnum2 = settings['DBnum2']
        self.networkmodel = settings['networkmodel']
        self.GRPresultsname = settings['GRPresultsname']
        self.GRPresultsname2 = settings['GRPresultsname2']
        self.GRPpvalue = settings['GRPpvalue']

        settings['GRPcharacteristicscount'] = 0
        settings['GRPcharacteristicslist'] = []
        settings['GRPcharacteristicsvalues'] = []
        settings['GRPcharacteristicsvalues2'] = []
        np.save(settingsfile,settings)

        self.GRPcharacteristicscount = settings['GRPcharacteristicscount']
        self.GRPcharacteristicslist = settings['GRPcharacteristicslist']
        self.GRPcharacteristicsvalues = settings['GRPcharacteristicsvalues']
        self.GRPcharacteristicsvalues2 = settings['GRPcharacteristicsvalues2']

        # put some text as a place-holder
        self.GRPLabel1 = tk.Label(self.parent, text = "1) Choose data/results files;  \nSEMresults_network..., SEMresults_2source...,\nor region_properties... files", fg = 'gray', justify = 'left')
        self.GRPLabel1.grid(row=0,column=0,rowspan=2, sticky='W')
        self.GRPLabel1 = tk.Label(self.parent, text = "2) Select group-level analysis options\nChoices are: Bayesian regression of BOLD\nresponses, analyses of SEM results w.r.t.\npersonal characteristics", fg = 'gray', justify = 'left')
        self.GRPLabel1.grid(row=2,column=0,rowspan=2, sticky='W')
        self.GRPLabel2 = tk.Label(self.parent, text = "3) For Bayesian regression select correlation\n for the analysis type, and for group\ncomparisons choose 2 results files", fg = 'gray', justify = 'left')
        self.GRPLabel2.grid(row=4,column=0,rowspan=2, sticky='W')
        self.GRPLabel3 = tk.Label(self.parent, text = "4) Run selected group-level analysis", fg = 'gray', justify = 'left')
        self.GRPLabel3.grid(row=6,column=0, sticky='W')

        # make a label to show the current setting of the network definition file directory name
        # file1 (in case there are two sets of results to be compared)-------------------------------
        pname, fname = os.path.split(self.GRPresultsname)
        self.GRPresultsdirtext = tk.StringVar()
        self.GRPresultsdirtext.set(pname)
        self.GRPresultsnametext = tk.StringVar()
        self.GRPresultsnametext.set(fname)

        self.GRPlabel1 = tk.Label(self.parent, text = 'Results file1:', font = labelfont)
        self.GRPlabel1.grid(row=0, column=1, sticky='N')
        self.GRPresultsnamelabel = tk.Label(self.parent, textvariable=self.GRPresultsnametext, bg=bgcol, fg="#4B4B4B", font = labelfont,
                                      wraplength=300, justify='left')
        self.GRPresultsnamelabel.grid(row=0, column=2, columnspan=2, sticky='S')
        self.GRPresultsdirlabel = tk.Label(self.parent, textvariable=self.GRPresultsdirtext, bg=bgcol, fg="#4B4B4B", font = labelfont,
                                      wraplength=300, justify='left')
        self.GRPresultsdirlabel.grid(row=1, column=2, columnspan=2, sticky='N')
        # define a button to browse and select an existing network definition file, and write out the selected name
        # also, define the function for what to do when this button is pressed
        self.GRPresultsnamebrowse = tk.Button(self.parent, text='Browse', width=smallbuttonsize, bg=fgcol2, fg = fgletter2, font = widgetfont,
                                          command=self.GRPresultsbrowseaction, relief='raised', bd=5, highlightbackground = widgetbg)
        self.GRPresultsnamebrowse.grid(row=0, column=4)
        self.GRPresultsnameclear = tk.Button(self.parent, text='Clear', width=smallbuttonsize, bg=fgcol2, fg = fgletter2, font = widgetfont,
                                          command=self.GRPresultsclearaction, relief='raised', bd=5, highlightbackground = widgetbg)
        self.GRPresultsnameclear.grid(row=0, column=5)


        # file2 (in case there are two sets of results to be compared)-------------------------------
        if settings['GRPresultsname'] == 'notdefined':
            initial_state = tk.DISABLED
        else:
            initial_state = tk.NORMAL

        pname, fname = os.path.split(self.GRPresultsname2)
        self.GRPresultsdirtext2 = tk.StringVar()
        self.GRPresultsdirtext2.set(pname)
        self.GRPresultsnametext2 = tk.StringVar()
        self.GRPresultsnametext2.set(fname)

        self.GRPlabel2 = tk.Label(self.parent, text = 'Results file2:', font = labelfont)
        self.GRPlabel2.grid(row=2, column=1, sticky='N')
        self.GRPresultsnamelabel2 = tk.Label(self.parent, textvariable=self.GRPresultsnametext2, bg=bgcol, fg="#4B4B4B", font = labelfont,
                                      wraplength=300, justify='left')
        self.GRPresultsnamelabel2.grid(row=2, column=2, columnspan=2, sticky='S')
        self.GRPresultsdirlabel2 = tk.Label(self.parent, textvariable=self.GRPresultsdirtext2, bg=bgcol, fg="#4B4B4B", font = labelfont,
                                      wraplength=300, justify='left')
        self.GRPresultsdirlabel2.grid(row=3, column=2, columnspan=2, sticky='N')

        self.GRPresultsnamebrowse2 = tk.Button(self.parent, text='Browse', width=smallbuttonsize, bg=fgcol2, fg = fgletter2, font = widgetfont,
                                          command=self.GRPresultsbrowseaction2, relief='raised', bd=5, state = initial_state, highlightbackground = widgetbg)
        self.GRPresultsnamebrowse2.grid(row=2, column=4)
        self.GRPresultsnameclear2 = tk.Button(self.parent, text='Clear', width=smallbuttonsize, bg=fgcol2, fg = fgletter2, font = widgetfont,
                                          command=self.GRPresultsclearaction2, relief='raised', bd=5, state = initial_state, highlightbackground = widgetbg)
        self.GRPresultsnameclear2.grid(row=2, column=5)

        self.GRPresultsnameswap = tk.Button(self.parent, text='Swap 1-2', width=smallbuttonsize, bg=fgcol2, fg = fgletter2, font = widgetfont,
                                          command=self.GRPresultsswapaction, relief='raised', bd=5, state = initial_state, highlightbackground = widgetbg)
        self.GRPresultsnameswap.grid(row=4, column=4)


        # label, button, for splitting data sets based on a selected characteristic
        self.GRPsplitgroupbutton = tk.Button(self.parent, text="Split Results", width=smallbuttonsize, bg=fgcol2, fg = fgletter2, font = widgetfont,
                                        command=self.GRPmake2groups, relief='raised', bd=5, highlightbackground = widgetbg)
        self.GRPsplitgroupbutton.grid(row=4, column=2, sticky = 'N')


        # ---------radio buttons to indicate type of analysis to do----------------
        # checkboxes to indicate 1) signficiance from zero, 2) group differences, 3) correlation
        self.GRPanalysistypevalue = tk.IntVar(None,1)
        self.GRPsig1 = tk.Radiobutton(self.parent, text = 'Sign. non-zero', width = bigbuttonsize, fg = fgletter2, font = radiofont,
                                          command = self.GRPselecttype, variable = self.GRPanalysistypevalue, value = 1, highlightbackground = widgetbg)
        self.GRPsig1.grid(row = 5, column = 1, sticky="W")

        self.var2 = tk.IntVar()
        self.GRPsig2 = tk.Radiobutton(self.parent, text = 'Avg. Group Diff.', width = bigbuttonsize, fg = fgletter2, font = radiofont,
                                          command = self.GRPselecttype, variable = self.GRPanalysistypevalue, value = 2, highlightbackground = widgetbg)
        self.GRPsig2.grid(row = 6, column = 1, sticky="W")

        self.var3 = tk.IntVar()
        self.GRPsig2p = tk.Radiobutton(self.parent, text = 'Paired Group Diff.', width = bigbuttonsize, fg = fgletter2, font = radiofont,
                                          command = self.GRPselecttype, variable = self.GRPanalysistypevalue, value = 3, highlightbackground = widgetbg)
        self.GRPsig2p.grid(row = 7, column = 1, sticky="W")

        self.var4 = tk.IntVar()
        self.GRPcorr = tk.Radiobutton(self.parent, text = 'Correlation', width = bigbuttonsize, fg = fgletter2, font = radiofont,
                                          command = self.GRPselecttype, variable = self.GRPanalysistypevalue, value = 4, highlightbackground = widgetbg)
        self.GRPcorr.grid(row = 5, column = 2, sticky="W")

        self.var5 = tk.IntVar()
        self.GRPcorr = tk.Radiobutton(self.parent, text = 'Regression', width = bigbuttonsize, fg = fgletter2, font = radiofont,
                                          command = self.GRPselecttype, variable = self.GRPanalysistypevalue, value = 5, highlightbackground = widgetbg)
        self.GRPcorr.grid(row = 6, column = 2, sticky="W")

        self.var8 = tk.IntVar()
        self.GRPtime = tk.Radiobutton(self.parent, text = 'Time Paired Diff.', width = bigbuttonsize, fg = fgletter2, font = radiofont,
                                          command = self.GRPselecttype, variable = self.GRPanalysistypevalue, value = 8, highlightbackground = widgetbg)
        self.GRPtime.grid(row = 7, column = 2, sticky="W")


        self.var6 = tk.IntVar()
        self.GRPcorr = tk.Radiobutton(self.parent, text = 'ANOVA', width = bigbuttonsize, fg = fgletter2, font = radiofont,
                                          command = self.GRPselecttype, variable = self.GRPanalysistypevalue, value = 6, highlightbackground = widgetbg)
        self.GRPcorr.grid(row = 5, column = 3, sticky="W")

        self.var7 = tk.IntVar()
        self.GRPcorr = tk.Radiobutton(self.parent, text = 'ANCOVA', width = bigbuttonsize, fg = fgletter2, font = radiofont,
                                          command = self.GRPselecttype, variable = self.GRPanalysistypevalue, value = 7, highlightbackground = widgetbg)
        self.GRPcorr.grid(row = 6, column = 3, sticky="W")


        # indicate which "personal characteristics" to use - selected from database
        self.GRPlabel2 = tk.Label(self.parent, text = "Select characteristic:", font = labelfont, highlightbackground = widgetbg)
        self.GRPlabel2.grid(row=8,column=1, sticky='W')
        # fieldvalues = DBFrame.get_DB_field_values(self)
        self.fields = self.get_DB_fields()
        self.field_var = tk.StringVar()
        if len(self.fields) > 0:
            self.field_var.set(self.fields[0])
        else:
            self.field_var.set('empty')

        self.GRPfield_menu = tk.OptionMenu(self.parent, self.field_var, *self.fields, command=self.DBfieldchoice)
        self.GRPfield_menu.config(bg=bgcol)
        self.GRPfield_menu.grid(row=8, column=2, columnspan = 2, sticky='EW')
        self.fieldsearch_opt = self.GRPfield_menu  # save this way so that values are not cleared

        # label, button, for running the definition of clusters, and loading data
        self.GRPcharclearbutton = tk.Button(self.parent, text="Clear/Update", width=smallbuttonsize, bg=fgcol1, fg = fgletter1, font = widgetfont,
                                        command=self.GRPcharacteristicslistclear, relief='raised', bd=5, highlightbackground = widgetbg)
        self.GRPcharclearbutton.grid(row=8, column=4)

        self.GRPlabel3 = tk.Label(self.parent, text = 'Characteristics list:', font = labelfont, highlightbackground = widgetbg)
        self.GRPlabel3.grid(row=9, column=1, sticky='N')

        self.GRPcharacteristicscount = 0
        self.GRPcharacteristicslist = []
        self.GRPcharacteristicstext = tk.StringVar()
        self.GRPcharacteristicstext.set('not defined yet')
        self.GRPcharacteristicsdisplay = tk.Label(self.parent, textvariable=self.GRPcharacteristicstext, bg=bgcol, fg="#4B4B4B", font = labelfont,
                                      wraplength=300, justify='left')
        self.GRPcharacteristicsdisplay.grid(row=9, column=2, columnspan=2, sticky='N')

        # put in choices for statistical threshold
        self.GRPlabel5 = tk.Label(self.parent, text = 'p-value threhold:', font = labelfont, highlightbackground = widgetbg).grid(row=10, column=1, sticky='NSEW')
        self.GRPpvaluebox = tk.Entry(self.parent, width = 8, bg="white")
        self.GRPpvaluebox.grid(row=10, column=2, sticky = "W")
        self.GRPpvaluebox.insert(0,self.GRPpvalue)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.GRPpvaluesubmitbut = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.GRPpvaluesubmit, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.GRPpvaluesubmitbut.grid(row=10, column=3)

        # label, button, for running the definition of clusters, and loading data
        self.GRPrunbutton = tk.Button(self.parent, text="Run Group Analysis", width=bigbigbuttonsize, bg=fgcol1, fg = fgletter1, font = widgetfont,
                                        command=self.GRPrunanalysis, relief='raised', bd=5, highlightbackground = widgetbg)
        self.GRPrunbutton.grid(row=11, column=1, columnspan = 2)


# --------------------DISPLAY FRAME---------------------------------------------------------------
# Definition of the frame that holds image display windows
class DisplayFrame:

    def get_data_fields(self):
        fields = 'empty'
        if os.path.isfile(self.DISPresultsname):
            data = np.load(self.DISPresultsname, allow_pickle=True).flat[0]
            keylist = data.keys()
            if 'type' in keylist:
                self.DISPdatatype = data['type']
                if data['type'] == '2source':
                    fields = ['beta1','beta2','CCrecord','Zgrid1_1','Zgrid1_2','Zgrid2']
                if data['type'] == 'network':
                    fields = ['b','R2']
                if data['type'] == 'GLM':
                    fields = ['B','sem','T']
                    # ['B', 'sem', 'T', 'template','regionmap', 'roi_map', 'Tthresh','normtemplatename', 'DBname', 'DBnum']
            else:
                self.DISPdatatype = 'unknown'
                fields = 'empty'
        else:
            self.DISPdatatype = 'unknown'
            fields = 'empty'
        return fields

    def DISPfieldchoice(self,value):
        # get the field value choices for the selected field
        fvalue = self.field_var.get()
        self.field_var.set(value)
        print('data field selected is: {}'.format(value))
        self.DISP_get_connectiondata_fields()

        return self

    def DISP_get_connectiondata_fields(self):
        # 3) networkcomponent, tt, combo, timepoint, ss for network results "b" or "R2"
        #                   or  t, s1, s2, timepoint, nb for 2source results "beta1" or "beta2"
        #                   or timepoint, t, s for "CCrecord"
        #                   or t, s1, s2, timepoint for "Zgrid2" or "Zgrid1_1" or "Zgrid1_2"
        datafield = self.field_var.get()
        print('DISP_get_connectiondata_fields:   datafield = {}'.format(datafield))
        self.connectiondata = ['not defined']*5
        self.connectiondata_names = ['not defined']*5

        if self.DISPdatatype == '2source':
            if datafield == 'beta1' or datafield == 'beta2':
                self.connectiondata = ['t', 's1', 's2', 'tt', 'nb']
                self.connectiondata_names = ['target', 'source1', 'source2', 'timepoint', 'conn. num.']
            if datafield == 'CCrecord':
                self.connectiondata = ['timepoint', 't', 's']
                self.connectiondata_names = ['timepoint', 'target', 'source']
            if datafield == 'Zgrid2' or datafield == 'Zgrid1_1' or datafield == 'Zgrid1_2':
                self.connectiondata = ['t', 's1', 's2', 'timepoint']
                self.connectiondata_names = ['target', 'source1', 'source2', 'timepoint']
        if self.DISPdatatype == 'network':
            if datafield == 'b' or datafield == 'R2':
                self.connectiondata = ['networkcomponent', 'tt', 'combo', 'timepoint', 'ss']
                self.connectiondata_names = ['networkcomponent', 'target', 'source comb.', 'timepoint', 'source num.']
        return self


    def DISPupdateaction(self):
        # initialize some values
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.DBname = settings['DBname']
        self.DBnum = settings['DBnum']
        self.DBname2 = settings['DBname2']  # for 2nd set of comparison results
        self.DBnum2 = settings['DBnum2']
        self.networkmodel = settings['networkmodel']
        self.DISPresultsname = settings['GRPresultsname']
        self.DISPresultsname2 = settings['GRPresultsname2']
        self.DISPcharacteristicscount = settings['GRPcharacteristicscount']
        self.DISPcharacteristicslist = settings['GRPcharacteristicslist']
        self.DISPcharacteristicsvalues = settings['GRPcharacteristicsvalues']
        self.DISPcharacteristicsvalues2 = settings['GRPcharacteristicsvalues2']

        pname, fname = os.path.split(self.DISPresultsname)
        self.DISPresultsdirtext.set(pname)
        self.DISPresultsnametext.set(fname)
        pname, fname = os.path.split(self.DISPresultsname2)
        self.DISPresultsdirtext2.set(pname)
        self.DISPresultsnametext2.set(fname)

        # covariate names
        chartext = ''
        for names in self.DISPcharacteristicslist:
            chartext += names + ','
        chartext = chartext[:-1]
        self.DISPcovtext.set(chartext)

        # update field choices
        fields = self.get_data_fields()
        if len(fields) > 0:
            self.field_var.set(fields[0])
        else:
            self.field_var.set('empty')
        self.DISPfields = fields
        self.fieldchoice_opt.destroy()
        field_menu = tk.OptionMenu(self.parent, self.field_var, *fields, command=self.DISPfieldchoice)
        field_menu.config(bg=bgcol)
        field_menu.grid(row=5, column=2, sticky='EW')
        self.fieldchoice_opt = field_menu  # save this way so that values are not cleared

        # update connection data fields----------------------------------------------
        self.DISP_get_connectiondata_fields()   # update self.connectiondata and self.connectiondata_names
        # print('DISPfields set to {}'.format(self.DISPfields))
        # print('connectiondata_names set to {}'.format(self.connectiondata_names))

        nvalues = len(self.connectiondata_names)
        self.DISPboxname1.set(self.connectiondata_names[0])
        self.DISPboxname2.set(self.connectiondata_names[1])
        self.DISPboxname3.set(self.connectiondata_names[2])
        if nvalues > 3:
            self.DISPboxname4.set(self.connectiondata_names[3])
        else:
            self.DISPboxname4.set('not needed')
        if nvalues > 4:
            self.DISPboxname5.set(self.connectiondata_names[4])
        else:
            self.DISPboxname5.set('not needed')

        # reset entry box values
        self.DISPboxenter1.delete(0, 'end')  # collect the text from the text entry box
        self.DISPboxenter1.insert(0, 'not set')  # collect the text from the text entry box
        self.DISPboxnumtext1.set('not set')
        self.DISPboxenter2.delete(0, 'end')  # collect the text from the text entry box
        self.DISPboxenter2.insert(0, 'not set')  # collect the text from the text entry box
        self.DISPboxnumtext2.set('not set')
        self.DISPboxenter3.delete(0, 'end')  # collect the text from the text entry box
        self.DISPboxenter3.insert(0, 'not set')  # collect the text from the text entry box
        self.DISPboxnumtext3.set('not set')
        self.DISPboxenter4.delete(0, 'end')  # collect the text from the text entry box
        self.DISPboxenter4.insert(0, 'not set')  # collect the text from the text entry box
        self.DISPboxnumtext4.set('not set')
        self.DISPboxenter5.delete(0, 'end')  # collect the text from the text entry box
        self.DISPboxenter5.insert(0, 'not set')  # collect the text from the text entry box
        self.DISPboxnumtext5.set('not set')

        # reset excel file name information etc.
        self.DISPexcelnameinput = 'not defined'
        self.DISPexcelsheetnamelist = ['not defined']
        self.DISPexcelentrynums = 'not defined'

        self.DISPexcelnametext.set(self.DISPexcelnameinput)

        # update the pulldown menu with the sheet names
        self.excelsheetchoice_opt.destroy()
        self.sheetname_var.set('not defined')
        self.excelsheet_menu = tk.OptionMenu(self.parent, self.sheetname_var, *self.DISPexcelsheetnamelist, command=self.DISPexcelsheetchoice)
        self.excelsheet_menu.config(bg=bgcol)
        self.excelsheet_menu.grid(row=15, column=2, sticky='EW')
        self.excelsheetchoice_opt = self.excelsheet_menu  # save this way so that values are not cleared
        self.DISPexcelsheetinput = self.DISPexcelsheetnamelist[0]

        print('DISPLAY function:  files and values have been updated.  {}'.format(time.ctime()))


    def DISPsetmethodtype(self):
        value = self.DISPplotmethod.get()
        if value == 1:
            self.DISPmethod = 'boxplot'
        if value == 2:
            self.DISPmethod = 'lineplot'
        if value == 3:
            self.DISPmethod = 'connectionplot'
        print('Plot method set to: ',self.DISPmethod)
        return self


    def DISPsetanattype(self):
        value = self.DISPshowanat.get()
        if value == 1:
            self.DISPanatorientation = 'axial'
        if value == 2:
            self.DISPanatorientation = 'sagittal'
        if value == 3:
            self.DISPanatorientation = 'coronal'
        print('Anatomical orientation set to: ',self.DISPanatorientation)
        return self



    def DISPboxsubmitclick(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        # get inputs from DISPbox entries (values of which connections to display)
        # how many values are we looking for?
        nvalues = len(self.connectiondata)   # nvalues is always between 3 and 5

        self.connectiondata_values = {}
        # box1
        entered_text = self.DISPboxenter1.get()  # collect the text from the text entry box
        entered_values = np.fromstring(entered_text, dtype=int, sep=',')
        fieldname = self.connectiondata[0]
        self.connectiondata_values[fieldname] = entered_values
        if len(entered_values) == 1:
            numtext = '1 value'
        else:
            numtext = '{} values'.format(len(entered_values))
        self.DISPboxnumtext1.set(numtext)

        # box2
        entered_text = self.DISPboxenter2.get()  # collect the text from the text entry box
        entered_values = np.fromstring(entered_text, dtype=int, sep=',')
        fieldname = self.connectiondata[1]
        self.connectiondata_values[fieldname] = entered_values
        if len(entered_values) == 1:
            numtext = '1 value'
        else:
            numtext = '{} values'.format(len(entered_values))
        self.DISPboxnumtext2.set(numtext)

        # box3
        entered_text = self.DISPboxenter3.get()  # collect the text from the text entry box
        entered_values = np.fromstring(entered_text, dtype=int, sep=',')
        fieldname = self.connectiondata[2]
        self.connectiondata_values[fieldname] = entered_values
        if len(entered_values) == 1:
            numtext = '1 value'
        else:
            numtext = '{} values'.format(len(entered_values))
        self.DISPboxnumtext3.set(numtext)

        # box4
        if nvalues > 3:
            entered_text = self.DISPboxenter4.get()  # collect the text from the text entry box
            entered_values = np.fromstring(entered_text, dtype=int, sep=',')
            fieldname = self.connectiondata[3]
            self.connectiondata_values[fieldname] = entered_values
            if len(entered_values) == 1:
                numtext = '1 value'
            else:
                numtext = '{} values'.format(len(entered_values))
            self.DISPboxnumtext4.set(numtext)
        else:
            self.DISPboxenter4.delete(0,'end')
            self.DISPboxenter4.insert(0,'no values needed')
            self.DISPboxnumtext4.set('no values')

        # box5
        if nvalues > 4:
            entered_text = self.DISPboxenter5.get()  # collect the text from the text entry box
            entered_values = np.fromstring(entered_text, dtype=int, sep=',')
            fieldname = self.connectiondata[4]
            self.connectiondata_values[fieldname] = entered_values
            if len(entered_values) == 1:
                numtext = '1 value'
            else:
                numtext = '{} values'.format(len(entered_values))
            self.DISPboxnumtext5.set(numtext)
        else:
            self.DISPboxenter5.delete(0,'end')
            self.DISPboxenter5.insert(0,'no values needed')
            self.DISPboxnumtext5.set('no values')

        return self


    def DISPexcelnamebrowseclick(self):
        # first load the settings file so that values can be used later
        # settings = np.load(settingsfile, allow_pickle=True).flat[0]
        # use a dialog box to prompt the user to select an existing file, the default being .xlsx type
        filechoice = tkf.askopenfilename(title="Select results excel file",
                                         filetypes=(("excel files", "*.xlsx"), ("all files", "*.*")))
        print('DISP excel file = ', filechoice)
        self.DISPexcelnameinput = filechoice
        self.DISPexcelnametext.set(filechoice)
        # get the list of sheet names
        xls = pd.ExcelFile(filechoice, engine='openpyxl')
        self.DISPexcelsheetnamelist = xls.sheet_names

        # update the pulldown menu with the sheet names
        # update field choices
        self.excelsheetchoice_opt.destroy()
        if len(self.DISPexcelsheetnamelist) > 0:
            self.sheetname_var.set(self.DISPexcelsheetnamelist[0])
        else:
            self.sheetname_var.set('empty')
        excelsheet_menu = tk.OptionMenu(self.parent, self.sheetname_var, *self.DISPexcelsheetnamelist, command=self.DISPexcelsheetchoice)
        excelsheet_menu.config(bg=bgcol)
        excelsheet_menu.grid(row=16, column=2, sticky='EW')
        self.excelsheetchoice_opt = excelsheet_menu  # save this way so that values are not cleared
        self.DISPexcelsheetinput = self.DISPexcelsheetnamelist[0]

        # print('DISPLAY:  excel field name set to {}'.format(self.DISPexcelnameinput))
        # print('DISPLAY:  excel sheet name set to {}'.format(self.DISPexcelsheetinput))

        return self

    def DISPexcelsheetchoice(self,value):
        self.DISPexcelsheetinput = value
        # print('DISPLAY:  excel sheet name set to {}'.format(self.DISPexcelsheetinput))
        return self


    def DISPentrynumsubmitclick(self):
        entered_text = self.DISPentrynumenter.get()  # collect the text from the text entry box
        # get the excel file name that these row numbers refer to
        excelfilename = self.DISPexcelnameinput
        excelsheetname = self.DISPexcelsheetinput

        xls = pd.ExcelFile(excelfilename, engine='openpyxl')
        df1 = pd.read_excel(xls, excelsheetname)
        nvalues = len(df1)

        # allow for "all" to be entered
        if entered_text == 'all': entered_text = str(0) + ':' + str(nvalues-1)
        # parse the entered text into values
        entered_values = DBFrame.DBparsenumlist(self, entered_text, nvalues-1)
        self.DISPexcelentrynums = entered_values
        print(entered_values)

        # convert back to shorter string for display
        value_list_for_display = DBFrame.DBdisplaynumlist(self, entered_values)
        self.DISPentrynumenter.delete(0, 'end')
        self.DISPentrynumenter.insert(0, value_list_for_display)

        # update the values of the connection data, based on the excel file choices

        fields = self.connectiondata
        rownums = self.DISPexcelentrynums
        # print('fields = {}'.format(fields))
        # print('rownums = {}'.format(rownums))

        self.connectiondata_values = {}
        for nn in range(len(fields)):
            fieldname = fields[nn]
            values = df1.loc[rownums,fieldname]
            value_text = ''
            value_num = []
            for v in values:
                value_text += str(v)+', '
                value_num += [int(v)]
            value_text = value_text[:-2]
            self.connectiondata_values[fieldname] = value_num
            if len(values) == 1:
                numtext = '1 value'
            else:
                numtext = '{} values'.format(len(values))

            if nn == 0:
                self.DISPboxenter1.delete(0,'end')  # collect the text from the text entry box
                self.DISPboxenter1.insert(0,value_text)  # collect the text from the text entry box
                self.DISPboxnumtext1.set(numtext)
            if nn == 1:
                self.DISPboxenter2.delete(0,'end')  # collect the text from the text entry box
                self.DISPboxenter2.insert(0,value_text)  # collect the text from the text entry box
                self.DISPboxnumtext2.set(numtext)
            if nn == 2:
                self.DISPboxenter3.delete(0,'end')  # collect the text from the text entry box
                self.DISPboxenter3.insert(0,value_text)  # collect the text from the text entry box
                self.DISPboxnumtext3.set(numtext)
            if nn == 3:
                self.DISPboxenter4.delete(0,'end')  # collect the text from the text entry box
                self.DISPboxenter4.insert(0,value_text)  # collect the text from the text entry box
                self.DISPboxnumtext4.set(numtext)
            if nn == 4:
                self.DISPboxenter5.delete(0,'end')  # collect the text from the text entry box
                self.DISPboxenter5.insert(0,value_text)  # collect the text from the text entry box
                self.DISPboxnumtext5.set(numtext)

        return self


    def DISPgeneratefigs(self):
        # use the input data and generate the figures
        filename1 = self.DISPresultsname
        filename2 = self.DISPresultsname2
        covariates1 = self.DISPcharacteristicsvalues
        covariates2 = self.DISPcharacteristicsvalues2

        connectiondata = self.connectiondata_values
        field_to_plot = self.field_var.get()

        if self.DISPmethod == 'boxplot':
            # generate box plot
            # pydisplay.display_whisker_plots(filename1, filename2, connectiondata, field_to_plot, self.Canvas1, self.PlotAx1)
            pydisplay.display_whisker_plots(filename1, filename2, connectiondata, field_to_plot, self.controller.Canvas3, self.controller.PlotAx3)
            k = list(connectiondata.keys())
            nvals = len(connectiondata[k[0]])
            if nvals > 1:
                self.controller.PlotAx4.clear()
            else:
                xls = pd.ExcelFile(self.DBname, engine='openpyxl')
                df1 = pd.read_excel(xls, 'datarecord')
                normtemplatename = df1.loc[self.DBnum[0], 'normtemplatename']
                # pydisplay.display_anatomical_figure(filename1, connectiondata, normtemplatename, [1, 0, 0], self.DISPanatorientation, self.Canvas2, self.PlotAx2)
                pydisplay.display_anatomical_figure(filename1, connectiondata, normtemplatename, [1, 0, 0], self.DISPanatorientation, self.controller.Canvas4, self.controller.PlotAx4)

        if self.DISPmethod == 'lineplot':
            # generate line plot
            # pydisplay.display_correlation_plots(filename1, filename2, connectiondata, field_to_plot, covariates1[0,:], covariates2[0,:], 'none', self.Canvas1, self.PlotAx1)
            if np.ndim(covariates1) > 1:
                cov1 = covariates1[0,:]
            else:
                cov1 = []
            if np.ndim(covariates2) > 1:
                cov2 = covariates2[0,:]
            else:
                cov2 = []
            pydisplay.display_correlation_plots(filename1, filename2, connectiondata, field_to_plot, cov1, cov2, 'none', self.controller.Canvas3, self.controller.PlotAx3)
            xls = pd.ExcelFile(self.DBname, engine='openpyxl')
            df1 = pd.read_excel(xls, 'datarecord')
            normtemplatename = df1.loc[self.DBnum[0], 'normtemplatename']
            # pydisplay.display_anatomical_figure(filename1, connectiondata, normtemplatename, [1,0,0], self.DISPanatorientation, self.Canvas2, self.PlotAx2)
            pydisplay.display_anatomical_figure(filename1, connectiondata, normtemplatename, [1,0,0], self.DISPanatorientation, self.controller.Canvas4, self.controller.PlotAx4)

        if self.DISPmethod == 'connectionplot':
            # generate SEM style connection plot
            # needed inputs are excel file with results to show, excel file with network drawing parameters
            print('reading SEM data from {},  sheet {}'.format(self.DISPexcelnameinput, self.DISPexcelsheetinput))
            if len(self.DISPexcelentrynums) > 10:
                print('plotting data in rows from {} to {}'.format(np.min(self.DISPexcelentrynums),np.max(self.DISPexcelentrynums)))
            else:
                print('plotting data in rows: {}'.format(self.DISPexcelentrynums))
            print('plotting with definition in {}'.format(self.DISPconndefnamefull))
            statname = 'T value'
            scalefactor = 'auto'
            results_file = self.DISPexcelnameinput
            sheetname = self.DISPexcelsheetinput
            drawregionsfile = self.DISPconndefnamefull
            rownumbers = self.DISPexcelentrynums
            svgname = pydisplay.draw_sem_plot(results_file, sheetname, rownumbers, drawregionsfile, statname, scalefactor, thresholdtext='abs>0',
                          writefigure=True)

    # def DISPconndatabrowse(self):
    #     # use the input data and generate the figures
    #     filechoice = tkf.askopenfilename(title="Select SEM results excel file",
    #                                      filetypes=(("excel files", "*.xlsx"), ("all files", "*.*")))
    #     print('SEM results file = ', filechoice)
    #     self.DISPconnplotnamefull = filechoice
    #     p,f = os.path.split(filechoice)
    #     self.DISPconnplotname.set(f)

    def DISPconndefbrowse(self):
        # use the input data and generate the figures
        filechoice = tkf.askopenfilename(title="Select SEM plot definition excel file",
                                         filetypes=(("excel files", "*.xlsx"), ("all files", "*.*")))
        print('SEM plot defintion file = ', filechoice)
        self.DISPconndefnamefull = filechoice
        p,f = os.path.split(filechoice)
        self.DISPconndefname.set(f[:15])

        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        settings['DISPconndefnamefull'] = self.DISPconndefnamefull
        np.save(settingsfile, settings)


    # initialize the values, keeping track of the frame this definition works on (parent), and
    # also the main window containing that frame (controller)
    def __init__(self, parent, controller):
        parent.configure(relief='raised', bd=5, highlightcolor=fgcol3)
        self.parent = parent
        self.controller = controller
        self.DISPmethod = 'boxplot'
        self.DISPanatorientation = 'sagittal'

        # initialize some values
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.DBname = settings['DBname']
        self.DBnum = settings['DBnum']
        self.DBname2 = settings['DBname2']  # for 2nd set of comparison results
        self.DBnum2 = settings['DBnum2']
        self.networkmodel = settings['networkmodel']
        self.DISPresultsname = settings['GRPresultsname']
        self.DISPresultsname2 = settings['GRPresultsname2']
        self.DISPcharacteristicscount = settings['GRPcharacteristicscount']
        self.DISPcharacteristicslist = settings['GRPcharacteristicslist']
        self.DISPcharacteristicsvalues = settings['GRPcharacteristicsvalues']
        self.DISPcharacteristicsvalues2 = settings['GRPcharacteristicsvalues2']
        self.DISPdatatype = 'unknown'
        self.DISPfields = self.get_data_fields()
        self.DISPexcelnameinput = 'not defined'
        self.DISPexcelsheetnamelist = ['not defined']
        self.DISPexcelentrynums = 'not defined'
        self.connectiondata_names = ['not defined']*5
        self.connectiondata = ['not defined']*5
        self.connectiondata_values = []

        if 'DISPconndefnamefull' in settings.keys():
            self.DISPconndefnamefull = settings['DISPconndefnamefull']
        else:
            self.DISPconndefnamefull = ''

        # put some text as a place-holder
        self.DISPLabel1 = tk.Label(self.parent,
                                   text="1) The results to be displayed are\nlinked to the Group Analysis tab",
                                   fg='gray', justify='left')
        self.DISPLabel1.grid(row=0, column=0, rowspan=2, sticky='W')
        self.DISPLabel1 = tk.Label(self.parent,
                                   text="2) Select the 2source or Network analysis\nresults in the Group tab", fg='gray',
                                   justify='left')
        self.DISPLabel1.grid(row=2, column=0, rowspan=2, sticky='W')
        self.DISPLabel2 = tk.Label(self.parent,
                                   text="3) Indicate which type of results, and\nwhich specific values to show",
                                   fg='gray', justify='left')
        self.DISPLabel2.grid(row=4, column=0, rowspan=2, sticky='W')
        self.DISPLabel3 = tk.Label(self.parent, text="4) Generate the output figures\nwith the Run button", fg='gray',
                                   justify='left')
        self.DISPLabel3.grid(row=6, column=0, sticky='W')

        # make a label to show the current setting of the network definition file directory name
        # file1 (in case there are two sets of results to be compared)-------------------------------
        pname, fname = os.path.split(self.DISPresultsname)
        self.DISPresultsdirtext = tk.StringVar()
        self.DISPresultsdirtext.set(pname)
        self.DISPresultsnametext = tk.StringVar()
        self.DISPresultsnametext.set(fname)

        self.DISPlabel1 = tk.Label(self.parent, text='Results file1:', font = labelfont)
        self.DISPlabel1.grid(row=0, column=1, sticky='N')
        self.DISPresultsnamelabel = tk.Label(self.parent, textvariable=self.DISPresultsnametext, bg=bgcol, fg="#4B4B4B",
                                             font = labelfont,
                                             wraplength=250, justify='left')
        self.DISPresultsnamelabel.grid(row=0, column=2, columnspan=2, sticky='S')
        self.DISPresultsdirlabel = tk.Label(self.parent, textvariable=self.DISPresultsdirtext, bg=bgcol, fg="#4B4B4B",
                                            font = labelfont,
                                            wraplength=250, justify='left')
        self.DISPresultsdirlabel.grid(row=1, column=2, columnspan=2, sticky='N')
        # define a button to browse and select an existing network definition file, and write out the selected name
        # also, define the function for what to do when this button is pressed

        # file2 (in case there are two sets of results to be compared)-------------------------------
        pname, fname = os.path.split(self.DISPresultsname2)
        self.DISPresultsdirtext2 = tk.StringVar()
        self.DISPresultsdirtext2.set(pname)
        self.DISPresultsnametext2 = tk.StringVar()
        self.DISPresultsnametext2.set(fname)

        self.DISPlabel2 = tk.Label(self.parent, text='Results file2:', font = labelfont)
        self.DISPlabel2.grid(row=2, column=1, sticky='N')
        self.DISPresultsnamelabel2 = tk.Label(self.parent, textvariable=self.DISPresultsnametext2, bg=bgcol,
                                              fg="#4B4B4B", font = labelfont,
                                              wraplength=250, justify='left')
        self.DISPresultsnamelabel2.grid(row=2, column=2, columnspan=2, sticky='S')
        self.DISPresultsdirlabel2 = tk.Label(self.parent, textvariable=self.DISPresultsdirtext2, bg=bgcol, fg="#4B4B4B",
                                             font = labelfont,
                                             wraplength=250, justify='left')
        self.DISPresultsdirlabel2.grid(row=3, column=2, columnspan=2, sticky='N')

        # covariate names
        chartext = ''
        for names in self.DISPcharacteristicslist:
            chartext += names + ','
        chartext = chartext[:-1]
        self.DISPcovtext = tk.StringVar()
        self.DISPcovtext.set(chartext)

        self.DISPchartextlabel = tk.Label(self.parent, text='Covariates: ', font = labelfont)
        self.DISPchartextlabel.grid(row=4, column=1, sticky='N')
        self.DISPchartextlabel2 = tk.Label(self.parent, textvariable=self.DISPcovtext, bg=bgcol, fg="#4B4B4B", font = labelfont,
                                             wraplength=250, justify='left')
        self.DISPchartextlabel2.grid(row=4, column=2, columnspan=2, sticky='N')

        # button to update values if they have been changed in the Group frame
        self.DISPupdatebutton = tk.Button(self.parent, text='Refresh', width=smallbuttonsize, bg=fgcol2, fg = fgletter2, font = widgetfont,
                                          command=self.DISPupdateaction, relief='raised', bd=5, highlightbackground = widgetbg)
        self.DISPupdatebutton.grid(row=0, column=4)

        # create entry boxes to indicate which results to plot
        # need:
        # 1) field_to_plot  -  which entry in the data to show?
        # 2) boxplot or correlation plot?
        # 3) networkcomponent, tt, combo, timepoint, ss for network results or
        #     t, s2, s2, timepoint, nb for 2source results
        # 4) optional excel file for indicating which result to show
        #       - which excel entries to show?  seletions or all?
        # 5) option for entering/changing values instead of only from excel

        # 1) field_to_plot  -  which entry in the data to show?-----------------------------
        # create pull-down menu with list of fields that can be selected from the data
        # fieldvalues = DBFrame.get_DB_field_values(self)
        self.DISPlabel3 = tk.Label(self.parent, text = "Select data field:", font = labelfont)
        self.DISPlabel3.grid(row=5,column=1, sticky='W')

        self.field_var = tk.StringVar()
        if len(self.DISPfields) > 0:
            self.field_var.set(self.DISPfields[0])
        else:
            self.field_var.set('empty')
        self.DISP_get_connectiondata_fields()   # update self.connectiondata and self.connectiondata_names
        # print('DISPfields set to {}'.format(self.DISPfields))
        # print('connectiondata_names set to {}'.format(self.connectiondata_names))

        field_menu = tk.OptionMenu(self.parent, self.field_var, *self.DISPfields, command = self.DISPfieldchoice)
        field_menu.config(bg=bgcol)
        field_menu.grid(row=5, column=2, columnspan=2, sticky='EW')
        self.fieldchoice_opt = field_menu   # save this way so that values are not cleared

        # 2) boxplot or correlation plot?--------------------------------------------------
        self.DISPlabel4 = tk.Label(self.parent, text="Plot Method:", font = labelfont)
        self.DISPlabel4.grid(row=6, column=1, sticky='E')
        self.DISPplotmethod = tk.IntVar(None,1)
        self.DISPmethod1 = tk.Radiobutton(self.parent, text = 'Box Plot', width = smallbuttonsize, fg = fgletter2, font = radiofont,
                                          command = self.DISPsetmethodtype, variable = self.DISPplotmethod, value = 1)
        self.DISPmethod1.grid(row = 6, column = 2, sticky="W")

        self.DISPmethod2 = tk.Radiobutton(self.parent, text = 'Scatter Plot', width = smallbuttonsize, fg = fgletter2, font = radiofont,
                                          command = self.DISPsetmethodtype, variable = self.DISPplotmethod, value = 2)
        self.DISPmethod2.grid(row = 6, column = 3, sticky="W")

        # SEM connection plot
        self.DISPmethod3 = tk.Radiobutton(self.parent, text = 'Connection', width = smallbuttonsize, fg = fgletter2, font = radiofont,
                                          command = self.DISPsetmethodtype, variable = self.DISPplotmethod, value = 3)
        self.DISPmethod3.grid(row = 7, column = 2, sticky="W")
        # self.DISPconnplotname = tk.StringVar(self.parent, 'SEM data file')
        # self.DISPconnplottext = tk.Label(self.parent, textvariable=self.DISPconnplotname, font = infofont)
        # self.DISPconnplottext.grid(row=7, column=3, sticky='N')
        # self.DISPconndatabutton = tk.Button(self.parent, text='Browse', width=smallbuttonsize, bg=fgcol2, fg = fgletter2, font = widgetfont,
        #                                   command=self.DISPconndatabrowse, relief='raised', bd=5, highlightbackground = widgetbg)
        # self.DISPconndatabutton.grid(row=7, column=4)
        f,p = os.path.split(self.DISPconndefnamefull)
        self.DISPconndefname = tk.StringVar(self.parent, f[:15])
        self.DISPconndeftext = tk.Label(self.parent, textvariable=self.DISPconndefname, font = infofont)
        self.DISPconndeftext.grid(row=7, column=3, sticky='N')
        self.DISPconndefbutton = tk.Button(self.parent, text='Browse', width=smallbuttonsize, bg=fgcol2, fg = fgletter2, font = widgetfont,
                                          command=self.DISPconndefbrowse, relief='raised', bd=5, highlightbackground = widgetbg)
        self.DISPconndefbutton.grid(row=7, column=4)


        # 2b) show anatomical images (if appropriate)?------------------------------------
        self.DISPlabel5 = tk.Label(self.parent, text="Show Anat.:", font = labelfont)
        self.DISPlabel5.grid(row=6, column=4, sticky='E')

        self.DISPshowanat = tk.IntVar(None,2)
        self.DISPanat1 = tk.Radiobutton(self.parent, text = 'Axial', width = smallbuttonsize, fg = fgletter2, font = radiofont,
                                          command = self.DISPsetanattype, variable = self.DISPshowanat, value = 1)
        self.DISPanat1.grid(row = 6, column = 5, sticky="W")

        self.DISPanat2 = tk.Radiobutton(self.parent, text = 'Sagittal', width = smallbuttonsize, fg = fgletter2, font = radiofont,
                                          command = self.DISPsetanattype, variable = self.DISPshowanat, value = 2)
        self.DISPanat2.grid(row = 6, column = 6, sticky="W")

        self.DISPanat3 = tk.Radiobutton(self.parent, text = 'Coronal', width = smallbuttonsize, fg = fgletter2, font = radiofont,
                                          command = self.DISPsetanattype, variable = self.DISPshowanat, value = 3)
        self.DISPanat3.grid(row = 6, column = 7, sticky="W")

        #--------------------------------------------------------------------------------------------------------
        # 3) networkcomponent, tt, combo, timepoint, ss for network results "b" or "R2"
        #                   or  t, s1, s2, timepoint, nb for 2source results "beta1" or "beta2"
        #                   or timepoint, t, s for "CCrecord"
        #                   or t, s1, s2, timepoint for "Zgrid2" or "Zgrid1_1" or "Zgrid1_2"
        # 4) optional excel file for indicating which result to show
        #       - which excel entries to show?  seletions or all?
        # 5) option for entering/changing values instead of only from excel

        # provide 5 text boxes for inputing values,  one for each entry in self.connectiondata, and self.connectiondata_names

        self.DISPsectionlabel1 = tk.Label(self.parent, text='Specify data to display:', font = labelfont)
        self.DISPsectionlabel1.grid(row=8, column=1, columnspan = 2, sticky='W')
        nvalues = len(self.connectiondata_names)  # for initializing values
        # box1
        self.DISPboxname1 = tk.StringVar(self.parent, self.connectiondata_names[0])
        # self.DISPboxname1.set(self.connectiondata_names[0])
        self.DISPboxlabel1 = tk.Label(self.parent, textvariable=self.DISPboxname1, font = labelfont)
        self.DISPboxlabel1.grid(row=9, column=1, sticky='W')
        # create entry box1
        self.DISPboxenter1 = tk.Entry(self.parent, width=20, bg="white")
        self.DISPboxenter1.grid(row=9, column=2, sticky="W")
        self.DISPboxenter1.insert(0, 'not set')
        self.DISPboxnumtext1 = tk.StringVar()
        self.DISPboxnumtext1.set('no values set')
        self.DISPboxnum1 = tk.Label(self.parent, textvariable=self.DISPboxnumtext1, font = labelfont)
        self.DISPboxnum1.grid(row=9, column=3, sticky='W')

        # box2
        self.DISPboxname2 = tk.StringVar(self.parent, self.connectiondata_names[1])
        # self.DISPboxname2.set(self.connectiondata_names[1])
        self.DISPboxlabel2 = tk.Label(self.parent, textvariable=self.DISPboxname2, font = labelfont)
        self.DISPboxlabel2.grid(row=10, column=1, sticky='W')
        # create entry box2
        self.DISPboxenter2 = tk.Entry(self.parent, width=20, bg="white")
        self.DISPboxenter2.grid(row=10, column=2, sticky="W")
        self.DISPboxenter2.insert(0, 'not set')
        self.DISPboxnumtext2 = tk.StringVar()
        self.DISPboxnumtext2.set('no values set')
        self.DISPboxnum2 = tk.Label(self.parent, textvariable=self.DISPboxnumtext2, font = labelfont)
        self.DISPboxnum2.grid(row=10, column=3, sticky='W')

        # box3
        self.DISPboxname3 = tk.StringVar(self.parent, self.connectiondata_names[2])
        self.DISPboxlabel3 = tk.Label(self.parent, textvariable=self.DISPboxname3, font = labelfont)
        self.DISPboxlabel3.grid(row=11, column=1, sticky='W')
        # create entry box3
        self.DISPboxenter3 = tk.Entry(self.parent, width=20, bg="white")
        self.DISPboxenter3.grid(row=11, column=2, sticky="W")
        self.DISPboxenter3.insert(0, 'not set')
        self.DISPboxnumtext3 = tk.StringVar(self.parent, 'no values set')
        self.DISPboxnum3 = tk.Label(self.parent, textvariable=self.DISPboxnumtext3, font = labelfont)
        self.DISPboxnum3.grid(row=11, column=3, sticky='W')

        # box4
        if nvalues > 3:
            nameval = self.connectiondata_names[3]
        else:
            nameval = 'not needed'
        self.DISPboxname4 = tk.StringVar(self.parent, nameval)
        self.DISPboxlabel4 = tk.Label(self.parent, textvariable=self.DISPboxname4, font = labelfont)
        self.DISPboxlabel4.grid(row=12, column=1, sticky='W')
        # create entry box4
        self.DISPboxenter4 = tk.Entry(self.parent, width=20, bg="white")
        self.DISPboxenter4.grid(row=12, column=2, sticky="W")
        self.DISPboxenter4.insert(0, 'not set')
        self.DISPboxnumtext4 = tk.StringVar(self.parent, 'no values set')
        self.DISPboxnum4 = tk.Label(self.parent, textvariable=self.DISPboxnumtext4, font = labelfont)
        self.DISPboxnum4.grid(row=12, column=3, sticky='W')

        # box5
        if nvalues > 4:
            nameval = self.connectiondata_names[4]
        else:
            nameval = 'not needed'
        self.DISPboxname5 = tk.StringVar(self.parent, nameval)
        self.DISPboxlabel5 = tk.Label(self.parent, textvariable=self.DISPboxname5, font = labelfont)
        self.DISPboxlabel5.grid(row=13, column=1, sticky='W')
        # create entry box5
        self.DISPboxenter5 = tk.Entry(self.parent, width=20, bg="white")
        self.DISPboxenter5.grid(row=13, column=2, sticky="W")
        self.DISPboxenter5.insert(0, 'not set')
        self.DISPboxnumtext5 = tk.StringVar(self.parent, 'no values set')
        self.DISPboxnum5 = tk.Label(self.parent, textvariable=self.DISPboxnumtext5, font = labelfont)
        self.DISPboxnum5.grid(row=13, column=3, sticky='W')

        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.DISPboxsubmit = tk.Button(self.parent, text="Submit Conn. Details", width=bigbigbuttonsize, bg=fgcol2, fg = fgletter2, font = widgetfont,
                                     command=self.DISPboxsubmitclick, relief='raised', bd=5, highlightbackground = widgetbg)
        self.DISPboxsubmit.grid(row=14, column=1, columnspan=2, sticky='W')

        # option to select an excel file instead-----------------------------------------------------
        # make a label to show the current setting of the database name
        self.DISPexcelnamelabel = tk.Label(self.parent, text='Excel file name:', font = labelfont)
        self.DISPexcelnamelabel.grid(row=15, column=1, sticky='W')
        self.DISPexcelnametext = tk.StringVar()
        self.DISPexcelnametext.set(self.DISPexcelnameinput)
        self.DISPexcelnametextbox = tk.Label(self.parent, textvariable=self.DISPexcelnametext, bg=bgcol, fg="black", font = labelfont,
                                     wraplength=250, justify='left')
        self.DISPexcelnametextbox.grid(row=15, column=2, columnspan=2, sticky='W')
        # define a browse button
        self.DISPexcelnamebrowse = tk.Button(self.parent, text='Browse', width=smallbuttonsize, bg=fgcol2, fg = fgletter2, font = widgetfont,
                                  command=self.DISPexcelnamebrowseclick, relief='raised', bd=5, highlightbackground = widgetbg)
        self.DISPexcelnamebrowse.grid(row=15, column = 4)

        # create pulldown menu for the sheet name
        self.DISPexcelsheetnamelabel = tk.Label(self.parent, text='Excel sheet name:', font = labelfont)
        self.DISPexcelsheetnamelabel.grid(row=16, column=1, sticky='W')
        self.sheetname_var = tk.StringVar()
        if len(self.DISPexcelsheetnamelist) > 0:
            self.sheetname_var.set(self.DISPexcelsheetnamelist[0])
        else:
            self.sheetname_var.set('empty')
        self.excelsheet_menu = tk.OptionMenu(self.parent, self.sheetname_var, *self.DISPexcelsheetnamelist, command=self.DISPexcelsheetchoice)
        self.excelsheet_menu.config(bg=bgcol)
        self.excelsheet_menu.grid(row=16, column=2, columnspan=2, sticky='W')
        self.excelsheetchoice_opt = self.excelsheet_menu  # save this way so that values are not cleared

        # box for entering values of which excel file rows to read
        self.DISPexcelentrynumlabel = tk.Label(self.parent, text='Excel rows:', font = labelfont)
        self.DISPexcelentrynumlabel.grid(row=17, column=1, sticky='W')
        # create entry box
        self.DISPentrynumenter = tk.Entry(self.parent, width=20, bg="white")
        self.DISPentrynumenter.grid(row=17, column=2, sticky="W")
        self.DISPentrynumenter.insert(0, self.DISPexcelentrynums)
        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.DISPentrynumsubmit = tk.Button(self.parent, text="Submit", width=smallbuttonsize, bg=fgcol2, fg = fgletter2, font = widgetfont,
                                     command=self.DISPentrynumsubmitclick, relief='raised', bd=5, highlightbackground = widgetbg)
        self.DISPentrynumsubmit.grid(row=17, column=4)

        # button to generate the plot
        # for now, just put a button that will eventually call the NIfTI conversion program
        self.DISPrunbutton = tk.Button(self.parent, text = 'Generate Figures', width = bigbigbuttonsize, bg = fgcol1, fg = fgletter1, font = widgetfont, command = self.DISPgeneratefigs, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.DISPrunbutton.grid(row = 18, column = 1, columnspan = 2)


class DisplayFrame2:
    def DISP2saveclick1(self):
        # save the figure in PlotFigure3, Canvas3
        filechoice = tkf.asksaveasfilename(title="Select file", filetypes=(("eps files", "*.eps"), ("svg files", "*.svg"), ("all files", "*.*")))
        [f,e] = os.path.splitext(filechoice)
        if e == '.eps':
            filechoice = f+'.eps'
            plt.figure(93)
            plt.savefig(filechoice, format = 'eps')
            print('saved plot as {}'.format(filechoice))
        if e == '.svg':
            filechoice = f+'.svg'
            plt.figure(93)
            plt.savefig(filechoice, format = 'svg')
            print('saved plot as {}'.format(filechoice))
        if not e in ['.eps', '.svg']:
            print('image not saved - bad type chosen')


    def DISP2saveclick2(self):
        # save the figure in PlotFigure4, Canvas4
        filechoice = tkf.asksaveasfilename(title="Select file",
                                           filetypes=(("eps files", "*.eps"), ("svg files", "*.svg"), ("all files", "*.*")))
        [f, e] = os.path.splitext(filechoice)
        if e == '.eps':
            filechoice = f+'.eps'
            plt.figure(94)
            plt.savefig(filechoice, format = 'eps')
        if e == '.svg':
            filechoice = f+'.svg'
            plt.figure(94)
            plt.savefig(filechoice, format = 'svg')
            print('saved anat image as {}'.format(filechoice))
        if not e in ['.eps', '.svg']:
            print('image not saved - bad type chosen')


    def __init__(self, parent, controller):
        parent.configure(relief='raised', bd=5, highlightcolor=fgcol3)
        self.parent = parent
        self.controller = controller
        self.DISPmethod = 'boxplot'

        # initialize some values
        settings = np.load(settingsfile, allow_pickle=True).flat[0]

        # -----------------------------------------------------------------------------------------------
        # create the plot figures and axes--------------------------------------------------------------
        self.PlotFigure3 = plt.figure(93, figsize=(4, 4), dpi=100)
        self.PlotAx3 = self.PlotFigure3.add_subplot(111)
        self.Canvas3 = FigureCanvasTkAgg(self.PlotFigure3, self.parent)
        self.Canvas3.get_tk_widget().grid(row=0, column=0, sticky='W')
        self.PlotAx3.set_title('data to be plotted here')

        self.PlotFigure4 = plt.figure(94, figsize=(4, 4), dpi=100)
        self.PlotAx4 = self.PlotFigure4.add_subplot(111)
        self.Canvas4 = FigureCanvasTkAgg(self.PlotFigure4, self.parent)
        self.Canvas4.get_tk_widget().grid(row=0, column=1, sticky='W')
        self.PlotAx4.set_title('anatomical images to be displayed here')

        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.DISP2figsavebutton1 = tk.Button(self.parent, text="Save plot", width=bigbigbuttonsize, bg=fgcol2,
                                       fg = fgletter2, font = widgetfont, command=self.DISP2saveclick1, relief='raised', bd=5, highlightbackground = widgetbg)
        self.DISP2figsavebutton1.grid(row=1, column=0, sticky='S')

        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.DISP2figsavebutton2 = tk.Button(self.parent, text="Save image", width=bigbigbuttonsize, bg=fgcol2,
                                       fg = fgletter2, font = widgetfont, command=self.DISP2saveclick2, relief='raised', bd=5, highlightbackground = widgetbg)
        self.DISP2figsavebutton2.grid(row=1, column=1, sticky='S')




#-----------SAPM FRAME--------------------------------------------------
# Definition of the frame that has inputs for the database name, and entry numbers to use
class SAPMFrame:

    def SAPMcnumparse(self, entered_text):
        # need to make sure we are working with numbers, not text
        # first, replace any double spaces with single spaces, and then replace spaces with commas
        entered_text = re.sub('\ +', ',', entered_text)
        entered_text = re.sub('\,\,+', ',', entered_text)
        # replace x's with -1's
        entered_text = re.sub('x+', '-1', entered_text)
        # remove any leading or trailing commas
        if entered_text[0] == ',': entered_text = entered_text[1:]
        if entered_text[-1] == ',': entered_text = entered_text[:-1]

        entered_values = list(np.fromstring(entered_text, dtype=int, sep=','))

        return entered_values


    # define functions before they are used in the database frame------------------------------------------
    # action when the button to browse for a DB fie is pressed
    def SAPMnetbrowseclick(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        # use a dialog box to prompt the user to select an existing file, the default being .xlsx type
        filename =  tkf.askopenfilename(title = "Select file",filetypes = (("excel files","*.xlsx"),("all files","*.*")))
        print('filename = ',filename)
        # save the selected file name in the settings
        settings['networkmodel'] = filename   # use the same network for SEM and SAPM, no need to specify two
        self.networkmodel = filename

        # write the result to the label box for display
        npname, nfname = os.path.split(self.networkmodel)
        self.SAPMnetnametext.set(nfname)
        self.SAPMnetdirtext.set(npname)

        network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count = pysapm.load_network_model_w_intrinsics(self.networkmodel)
        # update cnums to have the correct number of entries for the selected network
        nregions = len(network)
        if 'SAPMcnums' in settings.keys():
            SAPMcnums = settings['SAPMcnums']
        else:
            SAPMcnums = [0]

        if len(SAPMcnums) < nregions:
            temp = list(np.zeros(nregions).astype(int))
            temp[:len(SAPMcnums)] = SAPMcnums
            SAPMcnums = copy.deepcopy(temp)
        if len(SAPMcnums) > nregions:
            SAPMcnums = SAPMcnums[:nregions]

        self.SAPMcnums = copy.deepcopy(SAPMcnums)
        self.SAPMcnumsbox.delete(0, 'end')
        self.SAPMcnumsbox.insert(0, self.SAPMcnums)
        settings['SAPMcnums'] = self.SAPMcnums

        self.SAPMkeyinfo1.config(text=' ', fg='gray')
        np.save(settingsfile,settings)


    def SAPMprefixsubmitaction(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        SAPMprefix = self.SAPMprefixbox.get()
        settings['SAPMprefix'] = SAPMprefix
        np.save(settingsfile,settings)
        # update the text in the box, in case it has changed
        self.SAPMprefix = SAPMprefix
        self.SAPMkeyinfo1.config(text=' ', fg='gray')
        print('prefix for SAPM analysis set to ',self.SAPMprefix)
        self.SAPMupdate_network_info()
        return self

    # define functions before they are used in the database frame------------------------------------------
    # action when the button to browse for a DB fie is pressed
    def SAPMclusternamebrowseaction(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        # use a dialog box to prompt the user to select an existing file, the default being .xlsx type
        filechoice =  tkf.askopenfilename(title = "Select file",filetypes = (("npy files","*.npy"),("all files","*.*")))
        print('cluster definition name = ',filechoice)
        # save the selected file name in the settings
        SAPMclustername = filechoice

        npname, nfname = os.path.split(SAPMclustername)
        nfname,ext = os.path.splitext(nfname)
        settings['SAPMresultsdir'] = npname
        self.SAPMresultsdir = npname
        # write the result to the label box for display
        self.SAPMresultsdirtext.set(settings['SAPMresultsdir'])

        SAPMclustername = os.path.join(npname,nfname+'.npy')

        settings['SAPMclustername'] = SAPMclustername
        self.SAPMclustername = SAPMclustername

        # write the result to the label box for display
        self.SAPMclusternamebox.delete(0, 'end')
        self.SAPMclusternamebox.insert(0,SAPMclustername)

        self.SAPMkeyinfo1.config(text=' ', fg='gray')
        np.save(settingsfile,settings)

        self.SAPMupdate_network_info()


    def SAPMclusternamesubmitaction(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        SAPMclustername = self.SAPMclusternamebox.get()

        # check if chosen name includes the full directory path, or the extension
        npname, nfname = os.path.split(SAPMclustername)
        nfname, ext = os.path.splitext(nfname)

        if os.path.isdir(npname):
            settings['SAPMresultsdir'] = npname
            self.SAPMresultsdir = npname
            # write the result to the label box for display
            self.SAPMresultsdirtext.set(settings['SAPMresultsdir'])
        else:
            # select a directory
            npname = settings['SAPMresultsdir']

        # join up the name parts
        SAPMclustername = os.path.join(npname,nfname+'.npy')

        settings['SAPMclustername'] = SAPMclustername
        self.SAPMclustername = SAPMclustername

        # write the result to the label box for display
        self.SAPMclusternamebox.delete(0, 'end')
        self.SAPMclusternamebox.insert(0, self.SAPMclustername)

        self.SAPMkeyinfo1.config(text=' ', fg='gray')
        np.save(settingsfile, settings)

        self.SAPMupdate_network_info()


    def SAPMbetascalebrowseaction(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        # use a dialog box to prompt the user to select an existing file, the default being .xlsx type
        filechoice =  tkf.askopenfilename(title = "Select file",filetypes = (("npy files","*.npy"),("all files","*.*")))
        print('betascale definition name = ',filechoice)
        # save the selected file name in the settings
        SAPMbetascale = filechoice

        settings['SAPMbetascale'] = SAPMbetascale
        self.SAPMbetascale = SAPMbetascale

        # write the result to the label box for display
        self.SAPMbetascalebox.delete(0, 'end')
        self.SAPMbetascalebox.insert(0,SAPMbetascale)

        np.save(settingsfile,settings)


    def SAPMbetascalesubmitaction(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        SAPMbetascale = self.SAPMbetascalebox.get()

        # check if SAPMbetascale is a file name, or a floating-point number
        try:
            SAPMbetascale = float(SAPMbetascale)
        except:
            # check if SAPMbetascale is a file name
            check = os.path.isfile(SAPMbetascale)
            check2 = (SAPMbetascale.lower() == 'shotgun')
            if check:
                print('Reading initial beta values from {}'.format(SAPMbetascale))
            else:
                if check2:
                    print('initial beta values will be estimated using a random sampling')
                else:
                    print('initial beta entry is neither a floating-point number nor a file name')
                    print('reverting to default initial beta values of 0.0')
                    SAPMbetascale = 0.0

        settings['SAPMbetascale'] = SAPMbetascale
        self.SAPMbetascale = SAPMbetascale

        print('SAPMbetascale = {}'.format(SAPMbetascale))

        # write the result to the label box for display
        self.SAPMbetascalebox.delete(0, 'end')
        self.SAPMbetascalebox.insert(0, self.SAPMbetascale)

        np.save(settingsfile, settings)


    def SAPMbetainitcheckboxes(self):
        self.SAPMsavebetainit = self.varS2.get()
        print('SAPMsavebetainit = {}'.format(self.SAPMsavebetainit))
        return self



    def SAPMresultsnamesubmitaction(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        SAPMresultsname = self.SAPMresultsnamebox.get()

        # remove directory information if included, and make sure extension is .npy
        p, f_full = os.path.split(SAPMresultsname)
        f,e = os.path.splitext(f_full)
        SAPMresultsname = f + '.npy'

        settings['SAPMresultsname'] = SAPMresultsname
        self.SAPMresultsname = SAPMresultsname

        # write the result to the label box for display
        self.SAPMresultsnamebox.delete(0, 'end')
        self.SAPMresultsnamebox.insert(0, self.SAPMresultsname)

        self.SAPMkeyinfo1.config(text=' ', fg='gray')
        np.save(settingsfile, settings)

        self.SAPMupdate_network_info()


    def SAPMparamsnamesubmitaction(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        SAPMparamsname = self.SAPMparamsnamebox.get()

        # remove directory information if included, and make sure extension is .npy
        p, f_full = os.path.split(SAPMparamsname)
        f,e = os.path.splitext(f_full)
        SAPMparamsname = f + '.npy'

        settings['SAPMparamsname'] = SAPMparamsname
        self.SAPMparamsname = SAPMparamsname

        # write the result to the label box for display
        self.SAPMparamsnamebox.delete(0, 'end')
        self.SAPMparamsnamebox.insert(0, SAPMparamsname)

        self.SAPMkeyinfo1.config(text=' ', fg='gray')
        np.save(settingsfile, settings)

        self.SAPMupdate_network_info()

    # define functions before they are used in the database frame------------------------------------------
    # action when the button to browse for a DB fie is pressed
    def SAPMregionnamebrowseaction(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        # use a dialog box to prompt the user to select an existing file, the default being .xlsx type
        filechoice =  tkf.askopenfilename(title = "Select file",filetypes = (("npy files","*.npy"),("all files","*.*")))
        print('region/cluster data name = ',filechoice)
        SAPMregionname = filechoice

        npname, nfname = os.path.split(SAPMregionname)
        fname, ext = os.path.splitext(nfname)
        settings['SAPMresultsdir'] = npname
        self.SAPMresultsdir = npname
        # write the result to the label box for display
        self.SAPMresultsdirtext.set(settings['SAPMresultsdir'])

        SAPMregionname = os.path.join(npname,fname+'.npy')

        # write the result to the label box for display
        self.SAPMregionnamebox.delete(0, 'end')
        self.SAPMregionnamebox.insert(0,SAPMregionname)

        # save the selected file name in the settings
        settings['SAPMregionname'] = SAPMregionname
        self.SAPMregionname = SAPMregionname

        self.SAPMkeyinfo1.config(text=' ', fg='gray')
        np.save(settingsfile,settings)

        self.SAPMupdate_network_info()


    def SAPMregionnamesubmitaction(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        SAPMregionname = self.SAPMregionnamebox.get()

        # check if chosen name includes the full directory path, or the extension
        npname, nfname = os.path.split(SAPMregionname)
        nfname, ext = os.path.splitext(nfname)

        if os.path.isdir(npname):
            settings['SAPMresultsdir'] = npname
            self.SAPMresultsdir = npname
            # write the result to the label box for display
            self.SAPMresultsdirtext.set(settings['SAPMresultsdir'])
        else:
            # select a directory
            npname = settings['SAPMresultsdir']

        # join up the name parts
        SAPMregionname = os.path.join(npname,nfname+'.npy')

        settings['SAPMregionname'] = SAPMregionname
        self.SAPMregionname = SAPMregionname

        # write the result to the label box for display
        self.SAPMregionnamebox.delete(0, 'end')
        self.SAPMregionnamebox.insert(0, self.SAPMregionname)

        self.SAPMkeyinfo1.config(text=' ', fg='gray')
        np.save(settingsfile, settings)

        self.SAPMupdate_network_info()


    def SAPMcnumssubmitaction(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        SAPMcnums = self.SAPMcnumsbox.get()

        # parse the entered values into a list
        entered_values = self.SAPMcnumparse(SAPMcnums)
        self.SAPMcnums = copy.deepcopy(list(entered_values))

        network = []
        try:
            network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count = pysapm.load_network_model_w_intrinsics(self.networkmodel)
            # update cnums to have the correct number of entries for the selected network
            nregions = len(network)
            if len(self.SAPMcnums) < nregions:
                temp = list(np.zeros(nregions).astype(int))
                temp[:len(self.SAPMcnums)] = self.SAPMcnums
                self.SAPMcnums = copy.deepcopy(temp)
            if len(self.SAPMcnums ) > nregions:
                self.SAPMcnums = self.SAPMcnums[:nregions]

            self.SAPMcnumsbox.delete(0, 'end')
            self.SAPMcnumsbox.insert(0, self.SAPMcnums)

        except:
            print('valid network file not yet defined - cannot check if number of clusters entered matches the network')

        settings['SAPMcnums'] = self.SAPMcnums

        # write the result to the label box for display
        self.SAPMcnumsbox.delete(0, 'end')
        self.SAPMcnumsbox.insert(0, self.SAPMcnums)

        self.SAPMkeyinfo1.config(text=' ', fg='gray')
        np.save(settingsfile, settings)

        self.SAPMupdate_network_info()


    def SAPMtimesubmitclick(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        # SAPMtimepoint = settings['SAPMtimepoint']
        # SAPMepoch = settings['SAPMepoch']

        entered_text = self.SAPMtimeenter.get()  # collect the text from the text entry box

        if entered_text[:3] == 'all':
            # set time and epoch to span all data
            self.SAPMtimetext = 'all'
            settings['SAPMtimepoint'] = 'all'

            self.SAPMtimeenter.delete(0, 'end')
            self.SAPMtimeenter.insert(0, self.SAPMtimetext)

            settings['SAPMepoch'] = 'all'
            self.SAPMtimepoint = 'all'
            self.SAPMepoch = 'all'
            # self.SAPMepochenter.delete(0, 'end')
            # self.SAPMepochenter.insert(0, 'all')
        else:
            # first, replace any double spaces with single spaces, and then replace spaces with commas
            entered_text = re.sub('\ +', ',', entered_text)
            entered_text = re.sub('\,\,+', ',', entered_text)
            timevals = np.fromstring(entered_text, dtype=int, sep=',')
            SAPMtimepoint = timevals[0]
            SAPMepoch = timevals[1]

            print('timepoint {} and epoch {}'.format(SAPMtimepoint, SAPMepoch))

            timetext = '{},{}'.format(SAPMtimepoint, SAPMepoch)
            self.SAPMtimetext = timetext
            self.SAPMtimepoint = SAPMtimepoint
            self.SAPMepoch = SAPMepoch

            settings['SAPMtimepoint'] = SAPMtimepoint
            settings['SAPMepoch'] = SAPMepoch

            self.SAPMtimeenter.delete(0, 'end')
            self.SAPMtimeenter.insert(0, self.SAPMtimetext)


        self.SAPMupdate_network_info()
        # save the updated settings file again
        np.save(settingsfile, settings)
        return self


    def SAPMresultsdirbrowseaction(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.SAPMresultsdir = settings['SAPMresultsdir']
        # browse for new name
        dirname =  tkf.askdirectory(title = "Select folder")
        print('SAPM results save directory name = ',dirname)
        # save the selected file name in the settings
        settings['SAPMresultsdir'] = dirname
        self.SAPMresultsdir = dirname
        # write the result to the label box for display
        self.SAPMresultsdirtext.set(settings['SAPMresultsdir'])

        self.SAPMkeyinfo1.config(text=' ', fg='gray')
        # save the updated settings file again
        np.save(settingsfile,settings)

        self.SAPMupdate_network_info()


    # action when checkboxes are selected/deselected
    def SAPMcheckboxes(self):
        self.SAPMrandomclusterstart = self.varS1.get()
        print('SAPMrandomclusterstart = {}'.format(self.SAPMrandomclusterstart))
        self.SAPMkeyinfo1.config(text=' ', fg='gray')
        return self

    def SAPMbestclusters(self):
        # do a gradient-descent search for the best clusters
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        self.SAPMcnums = settings['SAPMcnums']
        self.DBname = settings['DBname']
        self.DBnum = settings['DBnum']
        self.networkmodel = settings['networkmodel']
        self.SAPMclustername = settings['SAPMclustername']
        self.SAPMregionname = settings['SAPMregionname']
        self.SAPMparamsname = settings['SAPMparamsname']
        self.SAPMresultsname = settings['SAPMresultsname']
        self.SAPMresultsdir = settings['SAPMresultsdir']
        self.SAPMbetascale = settings['SAPMbetascale']
        # self.SAPMsavetag = settings['SAPMsavetag']

        self.SAPMupdate_network_info()

        xls = pd.ExcelFile(self.DBname, engine='openpyxl')
        df1 = pd.read_excel(xls, 'datarecord')

        normtemplatename = df1.loc[self.DBnum[0], 'normtemplatename']
        resolution = 1
        template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_map = \
            load_templates.load_template_and_masks(normtemplatename, resolution)

        region_data = np.load(self.SAPMregionname, allow_pickle=True).flat[0]
        region_properties = region_data['region_properties']

        cluster_data = np.load(self.SAPMclustername, allow_pickle=True).flat[0]
        cluster_properties = cluster_data['cluster_properties']

        # excelfilename = os.path.join(self.SAPMresultsdir, self.SAPMsavetag + '_results.xlsx')

        print('running search for best clusters to use with SAPM ...')

        SAPMresultsname = os.path.join(self.SAPMresultsdir, self.SAPMresultsname)
        SAPMparamsname = os.path.join(self.SAPMresultsdir, self.SAPMparamsname)

        search_data_file = os.path.join(self.SAPMresultsdir,'SAPM_search_parameters.npy')

        if self.SAPMrandomclusterstart > 0:
            clusterstart = []
        else:
            clusterstart = self.SAPMcnums
        print('SAPMrandomclusterstart = {}'.format(self.SAPMrandomclusterstart))
        print('clusterstart set to {}'.format(clusterstart))
        np.save(search_data_file, {'SAPMresultsdir':self.SAPMresultsdir, 'SAPMresultsname':SAPMresultsname, 'SAPMparamsname':SAPMparamsname,
                                   'networkmodel':self.networkmodel, 'DBname':self.DBname, 'SAPMregionname':self.SAPMregionname,
                                    'SAPMclustername':self.SAPMclustername, 'initial_clusters':clusterstart, 'betascale':self.SAPMbetascale})

        use_parallel_processing = False
        if use_parallel_processing:
            message_text = 'Run the cluster search method from the \ncommand line. Follow the instructions \nwritten to the command window.'
            self.SAPMkeyinfo1.config(text = message_text, fg = 'red')

            # print out command line to use parallel processing ...
            print('\n\nIt is much faster to run the cluster search from the python command line\nusing the following commands...')
            print('  Note: if you get an error saying \'pantheon_command_line\' cannot be found, you need to add the folder containing '
                  'panthon_command_line.py to your path by entering the command sys.path.append( ...complete folder name... ) at the command line\n')
            print('Enter the following commands at the command line. Change the values of nprocessors, samplesplit, and samplestart if needed:  ')
            print('\nimport pantheon_command_line as pp')
            print('import multiprocessing as mp')
            print('max_processors = mp.cpu_count()')
            print('print(\'maximum number of processors is {}\'.format(max_processors))')
            print('nprocessors = 8  # ...choose the number to use, using the max available is not always the fastest')
            print('samplesplit = 1  # ...choose how to divide the sample, 1 for all data, 2 for 1/2, 3 for 1/3 etc...')
            print('samplestart = 0  # ...if the sample is split, which number to start with, i.e. for 1/2 use 0,2,4... or 1,3,5...')
            print('# ...optional inputs that can be added after "samplestart" are: ,timepoint, epoch')
            print('#    timepoint is the volume number at the center of the range of data you want to use (default is "all")')
            print('#    epoch is the volume span of the range of data you want to use (default is "all")')
            print('pp.SAPM_cluster_search_commandline(r\'{}\',nprocessors, samplesplit,samplestart)'.format(search_data_file))
        else:
            nprocessors = 1
            samplesplit = 1
            samplestart = 0
            # best_clusters = pysapm.SAPM_cluster_search(self.SAPMresultsdir, SAPMresultsname, SAPMparamsname, self.networkmodel, self.DBname, self.SAPMregionname,
            #                     self.SAPMclustername, nprocessors, samplesplit, samplestart, initial_clusters=clusterstart)

            best_clusters = pysapm.SAPM_cluster_stepsearch(self.SAPMresultsdir, SAPMresultsname, SAPMparamsname, self.networkmodel, self.DBname, self.SAPMregionname,
                        self.SAPMclustername, samplesplit, samplestart, initial_clusters=clusterstart)

            self.SAPMcnums = copy.deepcopy(list(best_clusters))
            self.SAPMcnumsbox.delete(0, 'end')
            self.SAPMcnumsbox.insert(0, self.SAPMcnums)
            message_text = 'Best clusters appear to be\n{}\nstarting clusters were\n{}'.format(self.SAPMcnums,clusterstart)
            self.SAPMkeyinfo1.config(text = message_text, fg = 'red')
            settings['SAPMcnums'] = self.SAPMcnums
            np.save(settingsfile,settings)

    def SAPMrunnetwork(self):
        # define the clusters and load the data
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        self.DBname = settings['DBname']
        self.DBnum = settings['DBnum']
        # self.SAPMprefix = settings['SAPMprefix']
        self.networkmodel = settings['networkmodel']
        self.SAPMclustername = settings['SAPMclustername']
        self.SAPMregionname = settings['SAPMregionname']
        self.SAPMparamsname = settings['SAPMparamsname']
        self.SAPMresultsname = settings['SAPMresultsname']
        self.SAPMresultsdir = settings['SAPMresultsdir']
        # self.SAPMsavetag = settings['SAPMsavetag']
        self.SAPMtimepoint = settings['SAPMtimepoint']
        self.SAPMepoch = settings['SAPMepoch']
        self.SAPMcnums = settings['SAPMcnums']
        self.SAPMbetascale = settings['SAPMbetascale']
        # self.SEMresumerun = settings['SEMresumerun']
        self.SAPMkeyinfo1.config(text=' ', fg='gray')

        self.SAPMupdate_network_info()

        SAPMresultsname = os.path.join(self.SAPMresultsdir, )
        xls = pd.ExcelFile(self.DBname, engine='openpyxl')
        df1 = pd.read_excel(xls, 'datarecord')

        normtemplatename = df1.loc[self.DBnum[0], 'normtemplatename']
        resolution = 1
        template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_map = \
            load_templates.load_template_and_masks(normtemplatename, resolution)

        region_data = np.load(self.SAPMregionname, allow_pickle=True).flat[0]
        region_properties = region_data['region_properties']

        cluster_data = np.load(self.SAPMclustername, allow_pickle=True).flat[0]
        cluster_properties = cluster_data['cluster_properties']

        # excelfilename = os.path.join(self.SAPMresultsdir, self.SAPMsavetag + '_results.xlsx')

        print('running SAPM with selected clusters ...')

        SAPMresultsname = os.path.join(self.SAPMresultsdir, self.SAPMresultsname)
        SAPMparamsname = os.path.join(self.SAPMresultsdir, self.SAPMparamsname)

        multiple_output = False
        pysapm.SAPMrun(self.SAPMcnums, self.SAPMregionname, self.SAPMclustername,
                       SAPMresultsname, SAPMparamsname, self.networkmodel, self.DBname, self.SAPMtimepoint,
                       self.SAPMepoch, self.SAPMbetascale, reload_existing=False, multiple_output = multiple_output)

        # get beta values and save for future betascale (initializing beta values)
        if self.SAPMsavebetainit:
            # settings['SAPMbetascale'] = self.SAPMbetascale
            # settings['SAPMsavebetainit'] = self.SAPMsavebetainit
            results = np.load(SAPMresultsname, allow_pickle=True)
            betavals = np.array([results[x][0]['betavals'] for x in range(len(results))])
            beta_initial = np.mean(betavals,axis=0)
            p,f = os.path.split(SAPMresultsname)
            betascale_name = os.path.join(p,'beta_initial_values.npy')
            np.save(betascale_name, {'beta_initial':beta_initial})



    def SAPMupdate_network_info(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.networkmodel = settings['networkmodel']
        npname, nfname = os.path.split(self.networkmodel)
        self.SAPMnetnametext.set(nfname)
        self.SAPMnetdirtext.set(npname)


    # initialize the values, keeping track of the frame this definition works on (parent), and
    # also the main window containing that frame (controller)
    def __init__(self, parent, controller):
        parent.configure(relief='raised', bd=5, highlightcolor=fgcol3)
        self.parent = parent
        self.controller = controller

        # initialize some values
        self.SAPMrandomclusterstart = 0
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        keylist = settings.keys()
        # if 'SAPMcnums' in keylist:
        self.SAPMcnums = settings['SAPMcnums']
        self.SAPMresultsdir = settings['SAPMresultsdir']
        self.SAPMresultsname = settings['SAPMresultsname']
        self.SAPMparamsname = settings['SAPMparamsname']
        self.DBname = settings['DBname']
        self.DBnum = settings['DBnum']
        self.networkmodel = settings['networkmodel']
        self.SAPMclustername = settings['SAPMclustername']
        self.SAPMregionname = settings['SAPMregionname']
        self.SAPMbetascale = settings['SAPMbetascale']
        self.SAPMtimepoint = settings['SAPMtimepoint']
        self.SAPMepoch = settings['SAPMepoch']

        self.SAPMtimepoint = settings['SAPMtimepoint']
        self.SAPMepoch = settings['SAPMepoch']
        # settings['SAPMtimepoint'] = self.SAPMtimepoint
        # settings['SAPMepoch'] = self.SAPMepoch
        self.SAPMtimetext = str(self.SAPMtimepoint)

        if 'SAPMsavebetainit' not in list(settings.keys()): settings['SAPMsavebetainit'] = False
        self.SAPMsavebetainit = settings['SAPMsavebetainit']
        # settings['SAPMsavebetainit'] = self.SAPMsavebetainit


        # self.SAPMsavetag = settings['SAPMsavetag']
        # else:
        #     self.DBname = settings['DBname']
        #     self.DBnum = settings['DBnum']
        #     self.SAPMcnums = [0,0,0,0,0,0,0,0,0,0]   # need a function to initialize this
        #     self.SAPMresultsdir = ''
        #     self.SAPMresultsname = ''
        #     self.SAPMparamsname = ''
        #     self.networkmodel = ''
        #     self.SAPMclustername = ''
        #     self.SAPMregionname = ''
        #     self.SAPMbetascale = 0.0
        #     self.SAPMtimepoint = 'all'
        #     self.SAPMepoch = 'all'
            # self.SAPMsavetag = ''

        # self.SAPMtimepoint = 'all'
        # self.SAPMepoch = 'all'
        # settings['SAPMtimepoint'] = self.SAPMtimepoint
        # settings['SAPMepoch'] = self.SAPMepoch
        # self.SAPMtimetext = str(self.SAPMtimepoint)

        self.SAPMsavebetainit = False
        settings['SAPMsavebetainit'] = self.SAPMsavebetainit

        # put some text as a place-holder
        self.SAPMLabel1 = tk.Label(self.parent, text = "1) Select SAPM options...\n   network definition, cluster\n   definitions, region data ...", fg = 'gray', justify = 'left')
        self.SAPMLabel1.grid(row=0,column=0, sticky='W',rowspan=2)
        self.SAPMLabel2 = tk.Label(self.parent, text = "2) The time span of data to\nuse can be specified as \'all\'\nor as the volume at the \ncenter of the time period\n(epoch), and the span of\nthe time period, separated\nby a comma", fg = 'gray', justify = 'left')
        self.SAPMLabel2.grid(row=2,column=0, sticky='W',rowspan=4)
        self.SAPMLabel3 = tk.Label(self.parent, text = "3) Clusters with the best\nfit to the data can be\nsearched, either starting\nfrom random values,\nspecified clusters, or\nwith some clusters fixed\nby specifying some clusters\nand indicating others with\n\'x\' or -1", fg = 'gray', justify = 'left')
        self.SAPMLabel3.grid(row=6,column=0, sticky='W',rowspan=4)
        self.SAPMLabel4 = tk.Label(self.parent, text = "4) Once parameters are selected:\nRun selected SAPM", fg = 'gray', justify = 'left')
        self.SAPMLabel4.grid(row=10,column=0, sticky='W',rowspan=3)

        # network file--------------------------------------------------
        # create an entry box so that the user can specify the network file to use-----------------
        # first make a title for the box, in row 3, column 1 of the grid for the main window
        self.SAPML1 = tk.Label(self.parent, text="Network Model:", font = labelfont)
        self.SAPML1.grid(row=0, column=1, sticky='SW')

        # make a label to show the current setting of the network definition file name
        npname, nfname = os.path.split(self.networkmodel)
        self.SAPMnetnametext = tk.StringVar()
        self.SAPMnetnametext.set(nfname)
        self.SAPMfnamelabel = tk.Label(self.parent, textvariable=self.SAPMnetnametext, bg=bgcol, fg="#4B4B4B", font = labelfont,
                                     wraplength=300, justify='left')
        self.SAPMfnamelabel.grid(row=0, column=2, sticky='S')

        # make a label to show the current setting of the network definition file directory name
        self.SAPMnetdirtext = tk.StringVar()
        self.SAPMnetdirtext.set(npname)
        self.SAPMdnamelabel = tk.Label(self.parent, textvariable=self.SAPMnetdirtext, bg=bgcol, fg="#4B4B4B", font = labelfont,
                                     wraplength=300, justify='left')
        self.SAPMdnamelabel.grid(row=1, column=2, sticky='N')

        # define a button to browse and select an existing network definition file, and write out the selected name
        # also, define the function for what to do when this button is pressed
        self.SAPMnetworkbrowse = tk.Button(self.parent, text='Browse', width=smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont,
                                  command=self.SAPMnetbrowseclick, relief='raised', bd=5, highlightbackground = widgetbg)
        self.SAPMnetworkbrowse.grid(row=0, column=3)

        # cluster definition name ---------------------------------------------
        # need an input for the cluster definition name - save to it, or read from it
        self.SAPMclusternamelabel = tk.Label(self.parent, text = 'Cluster definition name:', font = labelfont)
        self.SAPMclusternamelabel.grid(row=3, column=1, sticky='N')
        self.SAPMclusternamebox = tk.Entry(self.parent, width = 30, bg="white",justify = 'right')
        self.SAPMclusternamebox.grid(row=3, column=2, sticky='N')
        self.SAPMclusternamebox.insert(0,self.SAPMclustername)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.SAPMclusternamesubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SAPMclusternamesubmitaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SAPMclusternamesubmit.grid(row=3, column=3, sticky='N')
        # the entry boxes need a "browse" button to allow selection of existing cluster definition file
        self.SAPMclusternamebrowse = tk.Button(self.parent, text = "Browse", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SAPMclusternamebrowseaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SAPMclusternamebrowse.grid(row=3, column=4, sticky='N')

        # region data name ---------------------------------------------
        # box etc for entering the name for saving the region data
        self.SAPMregionnamelabel = tk.Label(self.parent, text = 'Region/cluster data name:', font = labelfont)
        self.SAPMregionnamelabel.grid(row=4, column=1, sticky='N')
        self.SAPMregionnamebox = tk.Entry(self.parent, width = 30, bg="white",justify = 'right')
        self.SAPMregionnamebox.grid(row=4, column=2, sticky='N')
        self.SAPMregionnamebox.insert(0,self.SAPMregionname)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.SAPMregionnamesubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SAPMregionnamesubmitaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SAPMregionnamesubmit.grid(row=4, column=3, sticky='N')
        # the entry boxes need a "browse" button to allow selection of existing cluster definition file
        self.SAPMregionnamebrowse = tk.Button(self.parent, text = "Browse", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SAPMregionnamebrowseaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SAPMregionnamebrowse.grid(row=4, column=4, sticky='N')

        # outputdir ----------------------------------------------------------------------
        # box etc for entering the name of the directory for saving the results
        # make a label to show the current setting of the network definition file directory name
        rownum = 6
        self.SAPMresultsdirlabel = tk.Label(self.parent, text = 'Results save folder:', font = labelfont)
        self.SAPMresultsdirlabel.grid(row=rownum, column=1, sticky='N')
        self.SAPMresultsdirtext = tk.StringVar()
        self.SAPMresultsdirtext.set(self.SAPMresultsdir)
        self.SAPMresultsdirdisplay = tk.Label(self.parent, textvariable=self.SAPMresultsdirtext, bg=bgcol, fg="#4B4B4B", font = labelfont,
                                     wraplength=300, justify='left')
        self.SAPMresultsdirdisplay.grid(row=rownum, column=2, sticky='N')
        # the entry boxes need a "browse" button to allow selection of existing cluster definition file
        self.SAPMresultsdirbrowse = tk.Button(self.parent, text = "Browse", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SAPMresultsdirbrowseaction, relief='raised', bd = 5)
        self.SAPMresultsdirbrowse.grid(row=rownum, column=3, sticky='N')

        rownum = 7
        # cluster numbers ------------------------------------------------------------------------
        self.SAPMcnumslabel = tk.Label(self.parent, text = 'cluster numbers:', font = labelfont)
        self.SAPMcnumslabel.grid(row=rownum, column=1, sticky='N')
        self.SAPMcnumsbox = tk.Entry(self.parent, width = 30, bg="white",justify = 'right')
        self.SAPMcnumsbox.grid(row=rownum, column=2, sticky='N')
        self.SAPMcnumsbox.insert(0,self.SAPMcnums)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.SAPMcnumssubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SAPMcnumssubmitaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SAPMcnumssubmit.grid(row=rownum, column=3, sticky='N')

        rownum = 8
        # create the SAPM timepoint and epoch entry box
        self.SAPMtimelabel = tk.Label(self.parent, text = 'Epoch center,span:', font = labelfont)
        self.SAPMtimelabel.grid(row=rownum, column=1, sticky='N')
        self.SAPMtimeenter = tk.Entry(self.parent, width=20, bg="white")
        self.SAPMtimeenter.grid(row=rownum, column=2, sticky="W")
        self.SAPMtimeenter.insert(0, self.SAPMtimetext)

        # # create the SEM epoch entry box
        # self.SAPMepochenter = tk.Entry(self.parent, width=20, bg="white")
        # self.SAPMepochenter.grid(row=rownum, column=3, sticky="W")
        # self.SAPMepochenter.insert(0, self.SAPMepoch)
        # # the entry box needs a "submit" button so that the program knows when to take the entered values

        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.SAPMtimesubmit = tk.Button(self.parent, text="Submit", width=smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont,
                                     command=self.SAPMtimesubmitclick, relief='raised', bd=5, highlightbackground = widgetbg)
        self.SAPMtimesubmit.grid(row=rownum, column=3)



        # SAPM results name --------------------------------------------------------------
        rownum = 9
        # box etc for entering the name used in labeling the results files
        self.SAPMresultsnamelabel = tk.Label(self.parent, text = 'name for SAPM results file:', font = labelfont)
        self.SAPMresultsnamelabel.grid(row=rownum, column=1, sticky='N')
        self.SAPMresultsnamebox = tk.Entry(self.parent, width = 30, bg="white",justify = 'right')
        self.SAPMresultsnamebox.grid(row=rownum, column=2, sticky='N')
        self.SAPMresultsnamebox.insert(0,self.SAPMresultsname)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.SAPMresultsnamesubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SAPMresultsnamesubmitaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SAPMresultsnamesubmit.grid(row=rownum, column=3, sticky='N')

        # SAPM parameters name --------------------------------------------------------------
        rownum = 10
        # box etc for entering the name used in labeling the results files
        self.SAPMparamsnamelabel = tk.Label(self.parent, text='name for SAPM parameters file:', font=labelfont)
        self.SAPMparamsnamelabel.grid(row=rownum, column=1, sticky='N')
        self.SAPMparamsnamebox = tk.Entry(self.parent, width=30, bg="white", justify='right')
        self.SAPMparamsnamebox.grid(row=rownum, column=2, sticky='N')
        self.SAPMparamsnamebox.insert(0, self.SAPMparamsname)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.SAPMparamsnamesubmit = tk.Button(self.parent, text="Submit", width=smallbuttonsize, bg=fgcol2,
                                              fg=fgletter2, font=widgetfont, command=self.SAPMparamsnamesubmitaction,
                                              relief='raised', bd=5, highlightbackground = widgetbg)
        self.SAPMparamsnamesubmit.grid(row=rownum, column=3, sticky='N')

        #
        # betascale definition ---------------------------------------------
        # this can be a number, or a file name that contains stored values
        rownum = 11
        self.SAPMbetascalelabel = tk.Label(self.parent, text = 'Initial beta:', font = labelfont)
        self.SAPMbetascalelabel.grid(row=rownum, column=1, sticky='N')
        self.SAPMbetascalebox = tk.Entry(self.parent, width = 30, bg="white",justify = 'right')
        self.SAPMbetascalebox.grid(row=rownum, column=2, sticky='N')
        self.SAPMbetascalebox.insert(0,self.SAPMbetascale)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.SAPMbetascalesubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SAPMbetascalesubmitaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SAPMbetascalesubmit.grid(row=rownum, column=3, sticky='N')
        # the entry boxes need a "browse" button to allow selection of existing cluster definition file
        self.SAPMbetascalebrowse = tk.Button(self.parent, text = "Browse", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SAPMbetascalebrowseaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SAPMbetascalebrowse.grid(row=rownum, column=4, sticky='N')


        rownum = 12
        self.varS1 = tk.IntVar()
        self.SAPMrandomcluster = tk.Checkbutton(self.parent, text = 'Random search start?', width = bigbigbuttonsize, fg = fgletter2,
                                          command = self.SAPMcheckboxes, variable = self.varS1)
        self.SAPMrandomcluster.grid(row=rownum, column=1, columnspan = 1, sticky='E')

        # label, button, for running the definition of clusters, and loading data
        self.SAPMrunsearchbutton = tk.Button(self.parent, text="Best clusters?", width=bigbigbuttonsize, bg=fgcol2, fg = fgletter2, font = widgetfont,
                                        command=self.SAPMbestclusters, relief='raised', bd=5, highlightbackground = widgetbg)
        self.SAPMrunsearchbutton.grid(row=rownum, column=2, columnspan = 2, sticky='W')

        self.SAPMkeyinfo1 = tk.Label(self.parent, text = " ", fg = 'gray', justify = 'left')
        self.SAPMkeyinfo1.grid(row=rownum,column=3, sticky='W')


        rownum = 13
        self.varS2 = tk.IntVar()
        self.SAPMsavebetascale = tk.Checkbutton(self.parent, text = 'Save beta init?', width = bigbigbuttonsize, fg = fgletter2,
                                          command = self.SAPMbetainitcheckboxes, variable = self.varS2)
        self.SAPMsavebetascale.grid(row=rownum, column=1, columnspan = 1, sticky='E')

        # label, button, for running the definition of clusters, and loading data
        self.SAPMrunnetworkbutton = tk.Button(self.parent, text="Run SAPM", width=bigbigbuttonsize, bg=fgcol1, fg = fgletter1, font = widgetfont,
                                        command=self.SAPMrunnetwork, relief='raised', bd=5, highlightbackground = widgetbg)
        self.SAPMrunnetworkbutton.grid(row=rownum, column=2, columnspan = 2, sticky='W')



#-----------SAPM Results FRAME--------------------------------------------------
# Definition of the frame that has inputs for viewing the results of SAPM analysis
class SAPMResultsFrame:

    # define functions before they are used in the database frame------------------------------------------
    # action when the button to browse for a DB fie is pressed
    def SRupdatevaluesclick(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        # load required values from previous pages: networkmodel, SAPMresultsname, SAPMparamsname, DBname
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.SAPMcnums = settings['SAPMcnums']
        self.SAPMresultsdir = copy.deepcopy(settings['SAPMresultsdir'])
        self.SAPMresultsname = copy.deepcopy(settings['SAPMresultsname'])
        self.SAPMparamsname = copy.deepcopy(settings['SAPMparamsname'])
        self.DBname = copy.deepcopy(settings['DBname'])
        self.networkmodel = settings['networkmodel']

        # np.save(settingsfile,settings)
        self.SRnetnametext.set(self.networkmodel)
        self.SRcnumtext.set(self.SAPMcnums)
        self.SRresultsdirtext.set(self.SRresultsdir)
        self.SRresultsnametext.set(self.SRresultsname)
        self.SRparamsnametext.set(self.SRparamsname)

        # update network
        self.SRtargetregion_opt.destroy()  # remove it
        try:
            network, nclusterlist, sapm_region_list, fintrinsic_count, vintrinsic_count \
                = pysapm.load_network_model_w_intrinsics(self.networkmodel)
            region_list = [x for x in sapm_region_list if 'intrinsic' not in x]
        except:
            region_list = 'not defined'
        self.SRtargetregion_var = tk.StringVar()
        self.SRtargetregion_var.set('empty')
        SRtargetregion_menu = tk.OptionMenu(self.parent, self.SRtargetregion_var, *region_list, command = self.SRtargetregionvalue_choice)
        SRtargetregion_menu.config(bg=bgcol)
        SRtargetregion_menu.grid(row=8, column=4, sticky='EW')
        self.SRtargetregion_opt = SRtargetregion_menu   # save this way so that values are not cleared

        return self


    def SRresultsdirbrowseaction(self):
        # browse for new name
        dirname = tkf.askdirectory(title="Select folder")
        print('SAPM results output directory = ',dirname)
        self.SRresultsdir = dirname
        self.SRresultsdirtext.set(self.SRresultsdir)
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        settings['SRresultsdir'] = self.SRresultsdir
        np.save(settingsfile, settings)
        return self


    def SRresultsnamebrowseaction(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.SRresultsname = copy.deepcopy(settings['SAPMresultsname'])
        # browse for new name
        filechoice =  tkf.askopenfilename(title = "Select file",filetypes = (("npy files","*.npy"),("all files","*.*")))
        print('SAPM results name = ',filechoice)
        self.SRresultsname = filechoice
        self.SRresultsnametext.set(self.SRresultsname)
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        settings['SRresultsname'] = self.SRresultsname
        np.save(settingsfile, settings)
        return self

    def SRparamsnamebrowseaction(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        self.SRparamsname = copy.deepcopy(settings['SAPMparamsname'])
        # browse for new name
        filechoice =  tkf.askopenfilename(title = "Select file",filetypes = (("npy files","*.npy"),("all files","*.*")))
        print('SAPM parameters name = ',filechoice)
        self.SRparamsname = filechoice
        self.SRparamsnametext.set(self.SRparamsname)
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        settings['SRparamsname'] = self.SRparamsname
        np.save(settingsfile, settings)
        return self


    def SRcovnamebrowseaction(self):
        # browse for new name
        filechoice =  tkf.askopenfilename(title = "Select file",filetypes = (("npy files","*.npy"),("all files","*.*")))
        print('covariates data name = ',filechoice)
        self.SRcovname = filechoice
        self.SRcovnametext.set(self.SRcovname)

        covariatesfile = self.SRcovnametext.get()
        covariatesdata = np.load(covariatesfile, allow_pickle=True).flat[0]
        self.covnamelist = covariatesdata['GRPcharacteristicslist']
        self.allcovariatesvalues = covariatesdata['GRPcharacteristicsvalues']

        # destroy the old pulldown menu and create a new one with the new choices
        fieldvalues = copy.deepcopy(self.covnamelist)
        self.SRcovfield_var = tk.StringVar()
        self.SRcovfield_var.set('empty')
        self.SRcovfieldvaluesearch_opt.destroy()  # remove it
        SRcovfieldvalue_menu = tk.OptionMenu(self.parent, self.SRcovfield_var, *fieldvalues,
                                        command=self.SRcovfieldvaluechoice)
        SRcovfieldvalue_menu.config(bg=bgcol)
        SRcovfieldvalue_menu.grid(row=7, column=2, sticky='EW')
        self.SRcovfieldvaluesearch_opt = SRcovfieldvalue_menu  # save this way so that values are not cleared

        return self


    def SRcovfieldvaluechoice(self,value):
        # get the field value choices for the selected field
        self.SRcovfield = self.SRcovfield_var.get()
        print('Selected covariate: {}'.format(self.SRcovfield))

        # self.covnamelist = covariatesdata['GRPcharacteristicslist']
        # self.allcovariatesvalues = covariatesdata['GRPcharacteristicsvalues']
        c = np.where(np.array(self.covnamelist) == self.SRcovfield)
        self.covariatesvalues = self.allcovariatesvalues[c[0],:][0]
        uniquevals = np.unique(self.covariatesvalues)
        uniquecheck = len(uniquevals)
        if uniquecheck < len(self.covariatesvalues)/4:
            self.SRcovcategorical = True
            print('Selected covariate appears to be a categorical variable')
            print('    values are:  {}'.format(uniquevals))
        else:
            self.SRcovcategorical = False
            # expect continuous values to be numeric
            self.covariatesvalues = self.covariatesvalues.astype(float)
            print('Selected covariate appears to be a continuous variable')
            print('    average value {} std dev:  {:.2f} {} {:.2f}'.format(chr(177),np.mean(self.covariatesvalues),chr(177),np.std(self.covariatesvalues)))
            print('    median value {} std dev:  {:.2f} {} {:.2f}'.format(chr(177),np.median(self.covariatesvalues),chr(177),np.std(self.covariatesvalues)))
            print('    value range  {:.2f} to {:.2f}'.format(np.min(self.covariatesvalues),np.max(self.covariatesvalues)))

        # update value selection pull-down menu

        # destroy the old pulldown menu and create a new one with the new choices
        if self.SRcovcategorical:
            fieldvalues = ['all'] + list(np.unique(self.covariatesvalues))
        else:
            avgcovvalue = np.median(self.covariatesvalues)
            if np.abs(avgcovvalue) > 0.1:
                fieldvalues = ['all', '> {:.2f}'.format(avgcovvalue), '<= {:.2f}'.format(avgcovvalue)]
            else:
                fieldvalues = ['all', '> {:.2e}'.format(avgcovvalue), '<= {:.2e}'.format(avgcovvalue)]

        self.SRcovvalue_var = tk.StringVar()
        self.SRcovvalue_var.set('empty')
        self.SRcovvaluesearch_opt.destroy()  # remove it
        SRcovvalue_menu = tk.OptionMenu(self.parent, self.SRcovvalue_var, *fieldvalues,
                                        command=self.SRcovvaluechoice)
        SRcovvalue_menu.config(bg=bgcol)
        SRcovvalue_menu.grid(row=7, column=3, sticky='EW')
        self.SRcovvaluesearch_opt = SRcovvalue_menu  # save this way so that values are not cleared

        return self


    def SRcovvaluechoice(self,value):
        # get the field value choices for the selected field
        self.SRcovvalue = self.SRcovvalue_var.get()
        print('Selected covariate value: {}'.format(self.SRcovvalue))

        # self.covnamelist = covariatesdata['GRPcharacteristicslist']
        # self.allcovariatesvalues = covariatesdata['GRPcharacteristicsvalues']
        if self.SRcovcategorical:
            if self.SRcovvalue == 'all':
                c = list(range(len(self.covariatesvalues)))
            else:
                c = np.where(np.array(self.covariatesvalues) == self.SRcovvalue)[0]
            self.SRgroup = c
        else:
            # continuous covariate
            if self.SRcovvalue == 'all':
                c = list(range(len(self.covariatesvalues)))
            else:
                numericvalue = float(self.SRcovvalue[2:])
                print('selecting group ... numeric value threshold is {:.2f}'.format(numericvalue))
                if self.SRcovvalue[0] == '>':
                    c = np.where(np.array(self.covariatesvalues) > numericvalue)[0]
                else:
                    c = np.where(np.array(self.covariatesvalues) <= numericvalue)[0]
            self.SRgroup = c

        print('selected {} sets of data in group {}'.format(len(self.SRgroup),self.SRcovvalue))
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        settings['SRgroup'] = self.SRgroup
        settings['SRcovvalue'] = self.SRcovvalue
        np.save(settingsfile, settings)
        return self


    def SRoptionvalue_choice(self,value):
        # get the field value choices for the selected field
        self.SRoptionvalue = self.SRoption_var.get()
        print('Selected output type: {}'.format(self.SRoptionvalue))
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        settings['SRoptionvalue'] = self.SRoptionvalue
        np.save(settingsfile, settings)

        if self.SRoptionvalue == 'DrawSAPMdiagram':
            self.SRdrawfilebox['state'] = tk.NORMAL
            self.SAPMBfilebox['state'] = tk.NORMAL
            self.SAPMBsheetfield_search_opt.config(state = tk.NORMAL)
            self.SAPMBcolumnfield_search_opt.config(state = tk.NORMAL)
            self.SRthresholdbox['state'] = tk.NORMAL
        else:
            self.SRdrawfilebox['state'] = tk.DISABLED
            self.SAPMBfilebox['state'] = tk.DISABLED
            self.SAPMBsheetfield_search_opt.config(state = tk.DISABLED)
            self.SAPMBcolumnfield_search_opt.config(state = tk.DISABLED)
            self.SRthresholdbox['state'] = tk.DISABLED

        return self

    def SRtargetregionvalue_choice(self,value):
        # get the field value choices for the selected field
        self.SRtargetregion = self.SRtargetregion_var.get()
        print('Selected target region: {}'.format(self.SRtargetregion))
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        settings['SRtargetregion'] = self.SRtargetregion
        np.save(settingsfile, settings)
        return self

    def SRpvaluesubmit(self):
        self.SRpvalue = float(self.SRpvaluebox.get())
        print('p-value for SAPM results display is set to ',self.SRpvalue)
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        settings['SRpvalue'] = self.SRpvalue
        np.save(settingsfile, settings)
        return self

    def SRvariantvaluesubmit(self):
        self.SRvariant = int(self.SRvariantvaluebox.get())
        print('p-value for SAPM results display is set to ',self.SRvariant)
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        settings['SRvariant'] = self.SRvariant
        np.save(settingsfile, settings)
        return self


    def SRnametagsubmit(self):
        self.SRnametag = self.SRnametagbox.get()
        print('output name base for saving SAPM results ',self.SRnametag)
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        settings['SRnametag'] = self.SRnametag
        np.save(settingsfile, settings)
        return self


    def SRgenerateoutput(self):
        print('covariates:  {} values'.format(len(self.covariatesvalues)))
        print('option:  {}'.format(self.SRoptionvalue))
        print('SAPMresultsdir:  {}'.format(self.SRresultsdir))
        print('SAPMparamsname:  {}'.format(self.SRparamsname))
        print('SAPMresultsname:  {}'.format(self.SRresultsname))
        print('group:  {} values'.format(len(self.SRgroup)))
        print('SRtargetregion:  {}'.format(self.SRtargetregion))
        print('SRpvalue:  {}'.format(self.SRpvalue))
        print('SRnametag:  {}'.format(self.SRnametag))
        print('SRvariant:  {}'.format(self.SRvariant))

        multiple_output = False

        # print('running pysapm.display_SAPM_results ...')
        # print('SRCanvas = {}'.format(self.SRCanvas))
        # print('  SRCanvas is a string:  {}'.format(isinstance(self.SRCanvas,str)))
        # print('  SRCanvas type:  {}'.format(type(self.SRCanvas)))

        # self.SRPlotFigure = 123
        outputname = pysapm.display_SAPM_results(123, self.SRnametag, self.covariatesvalues, self.SRoptionvalue,
                            self.SRresultsdir, self.SRparamsname, self.SRresultsname, self.SRvariant,
                            self.SRgroup, self.SRtargetregion, self.SRpvalue, [], self.SRCanvas, True, multiple_output = multiple_output)

        if ('Plot' in self.SRoptionvalue):     # or ('Draw' in self.SRoptionvalue)
            print('generating figures to save as svg files ...')
            outputname = pysapm.display_SAPM_results(124, self.SRnametag, self.covariatesvalues, self.SRoptionvalue,
                                self.SRresultsdir, self.SRparamsname, self.SRresultsname, self.SRvariant,
                                self.SRgroup, self.SRtargetregion, self.SRpvalue, [], 'none', False, multiple_output = multiple_output)

        if self.SRoptionvalue == 'DrawSAPMdiagram':
            # need:
            #  drawregionsfile ==> file defining how to draw network region
            # resultsfile ==> file with Mconn values (.xlsx) for example - SAPM_Bfile
            # sheetname ==> sheet in resultsfile with values to plot
            # regionname ==> column in sheetname, containing names of connections:  regionnames = 'regions'
            # statname ==> another column in sheetname, containing values to plot:  statname = 'beta'
            # scalefactor ==> how to scale statvalue to a linewidth in the display
            # threshold ==> lower threshold of value to plot, could be 0.0 - submitted as threshold_text
            # cnums ==> list of cluster numbers for regions to display
            # outputname ==> name of svg file to write for output
            # writefigure ==> write the result to disk, or not:  writefigure = True
            # figurenumber ==> value number for figure to write to ...

            drawregionsfile = self.SRdrawfile
            results_file = self.SAPMBfile
            sheetname = self.SAPMBsheet
            regionnames = 'regions'
            statname = self.SAPMBcolumn
            figurenumber = 200
            scalefactor = 'auto'
            cnums = self.SAPMcnums
            threshold_text = self.SRthresholdtext
            writefigure = True

            regions = pysapm.define_drawing_regions_from_file(drawregionsfile)
            multiple_output = False
            if multiple_output:
                outputname = pysapm.draw_sapm_plot(results_file, sheetname, regionnames, regions, statname,
                                                   figurenumber, scalefactor, cnums, threshold_text, writefigure)
            else:
                outputname = pysapm.draw_sapm_plot_SO(results_file, sheetname, regionnames, regions, statname,
                                                   figurenumber, scalefactor, cnums, threshold_text, writefigure)


        if  'DrawAnatomy' in self.SRoptionvalue:
            clusterdataname = self.SAPMclustername
            targetcluster_list = self.SAPMcnums
            network, nclusterlist, region_list, fintrinsic_count, vintrinsic_count = pysapm.load_network_model_w_intrinsics(self.networkmodel)
            rnamelist = [r for r in region_list if 'intrinsic' not in r]

            xls = pd.ExcelFile(self.DBname, engine='openpyxl')
            df1 = pd.read_excel(xls, 'datarecord')
            normtemplatename = df1.loc[self.DBnum[0], 'normtemplatename']

            if self.SRoptionvalue == 'DrawAnatomy_axial':
                for nn, targetname in enumerate(rnamelist):
                    outputimg, outputname = pysapm.display_anatomical_cluster(clusterdataname, targetname, targetcluster_list[nn], orientation = 'axial',
                                                      regioncolor = [0,1,1], templatename = normtemplatename, write_output = True)

            if self.SRoptionvalue == 'DrawAnatomy_sagittal':
                for nn, targetname in enumerate(rnamelist):
                    outputimg, outputname = pysapm.display_anatomical_cluster(clusterdataname, targetname, targetcluster_list[nn], orientation = 'sagittal',
                                                      regioncolor = [0,1,1], templatename = normtemplatename, write_output = True)


        print('output results to {}'.format(outputname))


    def SRdrawfilebrowseaction(self):
        # browse for new name
        filechoice =  tkf.askopenfilename(title = "Select file",filetypes = (("xlsx files","*.xlsx"),("all files","*.*")))
        print('file containing information about plotting SAPM results = ',filechoice)
        self.SRdrawfile = filechoice
        self.SRdrawfiletext.set(self.SRdrawfile[-30:])
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        settings['SRdrawfile'] = self.SRdrawfile
        np.save(settingsfile, settings)
        return self

    def SAPMBfilebrowseaction(self):
        # browse for new name
        filechoice =  tkf.askopenfilename(title = "Select file",filetypes = (("xlsx files","*.xlsx"),("all files","*.*")))
        print('file containing B values etc from SAPM results = ',filechoice)
        self.SAPMBfile = filechoice
        self.SAPMBfiletext.set(self.SAPMBfile[-30:])
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        settings['SAPMBfile'] = self.SAPMBfile
        np.save(settingsfile, settings)

        xls = pd.ExcelFile(self.SAPMBfile, engine='openpyxl') # get the sheetnames
        self.SRsheetnames = xls.sheet_names

        # destroy the old pulldown menu and create a new one with the new choices
        rownum = 12
        columnum = 4
        fieldvalues = copy.deepcopy(self.SRsheetnames)
        self.SAPMBsheetfield_var = tk.StringVar()
        self.SAPMBsheetfield_var.set(fieldvalues[0])
        self.SAPMBsheetfield_search_opt.destroy()  # remove it
        SAPMBsheet_menu = tk.OptionMenu(self.parent, self.SAPMBsheetfield_var, *fieldvalues,
                                        command=self.SAPMBsheetfieldvaluechoice)
        SAPMBsheet_menu.config(bg=bgcol)
        SAPMBsheet_menu.grid(row=rownum, column=columnum, sticky='EW')
        self.SAPMBsheetfield_search_opt = SAPMBsheet_menu  # save this way so that values are not cleared

        return self

    def SRthresholdsubmit(self):
        self.SRthresholdtext = self.SRthresholdbox.get()
        print('threshold for plotting SAPM values ',self.SRthresholdtext)
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        settings['SRthresholdtext'] = self.SRthresholdtext
        np.save(settingsfile, settings)
        print('Threshold text set to {}'.format(self.SRthresholdtext))
        return self


    def SAPMBsheetfieldvaluechoice(self,value):
        # get the field value choices for the selected field
        self.SAPMBsheet = self.SAPMBsheetfield_var.get()
        print('Selected sheet: {}'.format(self.SAPMBsheet))
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        settings['SAPMBsheet'] = self.SAPMBsheet
        np.save(settingsfile, settings)

        xls = pd.ExcelFile(self.SAPMBfile, engine='openpyxl') # get the sheetnames
        df1 = pd.read_excel(xls, self.SAPMBsheet)
        del df1['Unnamed: 0']  # get rid of the unwanted header column
        columnnames = list(df1.keys())
        self.SRcolumnnames = [x for x in columnnames if 'Unnamed' not in x]

        # destroy the old pulldown menu and create a new one with the new choices
        rownum = 12
        columnum = 5
        fieldvalues = copy.deepcopy(self.SRcolumnnames)
        self.SAPMBcolumnfield_var = tk.StringVar()
        self.SAPMBcolumnfield_var.set(fieldvalues[0])
        self.SAPMBcolumnfield_search_opt.destroy()  # remove it
        SAPMBcolumn_menu = tk.OptionMenu(self.parent, self.SAPMBcolumnfield_var, *fieldvalues,
                                        command=self.SAPMBcolumnfieldvaluechoice)
        SAPMBcolumn_menu.config(bg=bgcol)
        SAPMBcolumn_menu.grid(row=rownum, column=columnum, sticky='EW')
        self.SAPMBcolumnfield_search_opt = SAPMBcolumn_menu  # save this way so that values are not cleared
        return self


    def SAPMBcolumnfieldvaluechoice(self, value):
        # get the field value choices for the selected field
        self.SAPMBcolumn = self.SAPMBcolumnfield_var.get()
        print('Selected column: {}'.format(self.SAPMBcolumn))
        settings = np.load(settingsfile, allow_pickle=True).flat[0]
        settings['SAPMBcolumn'] = self.SAPMBcolumn
        np.save(settingsfile, settings)
        return self


    # initialize the values, keeping track of the frame this definition works on (parent), and
    # also the main window containing that frame (controller)
    def __init__(self, parent, controller):
        parent.configure(relief='raised', bd=5, highlightcolor=fgcol3)
        self.parent = parent
        self.controller = controller

        # self.SRwindownum = 123
        # self.SRwindownum2 = 124

        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        keylist = settings.keys()
        self.SAPMcnums = settings['SAPMcnums']
        self.SAPMresultsdir = settings['SAPMresultsdir']
        self.SAPMresultsname = settings['SAPMresultsname']
        self.SAPMparamsname = settings['SAPMparamsname']
        self.DBname = settings['DBname']
        self.DBnum = settings['DBnum']
        self.networkmodel = settings['networkmodel']
        self.SAPMclustername = settings['SAPMclustername']
        self.SAPMregionname = settings['SAPMregionname']
        self.SRvariant = 0
        self.SRoptionvalue = settings['SRoptionvalue']
        self.SRcovname = settings['SRcovname']
        self.SRpvalue = settings['SRpvalue']
        self.SRgroup = settings['SRgroup']
        self.SRtargetregion = settings['SRtargetregion']
        self.SRnametag = settings['SRnametag']
        self.SRdrawfile = settings['SRdrawfile']
        self.SRthresholdtext = settings['SRthresholdtext']
        self.SAPMBfile = settings['SAPMBfile']

        # initialize some values
        # if 'SRoptionvalue' in keylist: # expect that if one value for this section is
        #                                 # in settings, then they all are
        #     self.SRoptionvalue = settings['SRoptionvalue']
        #     self.SRcovname = settings['SRcovname']
        #     self.SRpvalue = settings['SRpvalue']
        #     self.SRgroup = settings['SRgroup']
        #     self.SRtargetregion = settings['SRtargetregion']
        #     self.SRnametag = settings['SRnametag']
        #     self.SRdrawfile = settings['SRdrawfile']
        #     self.SRthresholdtext = settings['SRthresholdtext']
        #     self.SAPMBfile = settings['SAPMBfile']
        # else:
        #     self.SRoptionvalue = 'not defined'
        #     self.SRcovname = 'not defined'
        #     self.SRpvalue = 0.05
        #     self.SRgroup = []
        #     self.SRtargetregion = []
        #     self.SRnametag = ''
        #     self.SRdrawfile = ''
        #     self.SRthresholdtext = 'abs>0'
        #     self.SAPMBfile = ''
        #     self.SAPMBsheet = 'sheet'
        #     self.SRcolumnnames = 'stat'
        #
        #     settings['SRoptionvalue'] = self.SRoptionvalue
        #     settings['SRcovname'] = self.SRcovname
        #     settings['SRpvalue'] = self.SRpvalue
        #     settings['SRgroup'] = self.SRgroup
        #     settings['SRtargetregion'] = self.SRtargetregion
        #     settings['SRnametag'] = self.SRnametag
        #     settings['SRvariant'] = self.SRvariant

        # if 'SRdrawfile' in settings.keys():
        #     self.SRdrawfile = settings['SRdrawfile']
        # else:
        #     self.SRdrawfile = ''

        # if 'SRthresholdtext' in settings.keys():
        #     self.SRthresholdtext = settings['SRthresholdtext']
        # else:
        #     self.SRthresholdtext = '>0'

        # initialize sheet and column names, if possible
        try:
            xls = pd.ExcelFile(self.SAPMBfile, engine='openpyxl')  # get the sheetnames
            self.SRsheetnames = xls.sheet_names
            self.SAPMBsheet = self.SRsheetnames[0]
            df1 = pd.read_excel(xls, self.SAPMBsheet)
            columnnames = list(df1.keys())
            self.SRcolumnnames = [x for x in columnnames if 'Unnamed' not in x]
            self.SAPMBcolumn = self.SRcolumnnames[0]
        except:
            self.SAPMBfile = ''
            self.SAPMBsheet = 'sheet'
            self.SRcolumnnames = 'stat'
            self.SRsheetnames = ['not set']


        self.SRresultsdir = copy.deepcopy(self.SAPMresultsdir)
        self.SRresultsname = os.path.join(self.SAPMresultsdir, self.SAPMresultsname)
        self.SRparamsname = os.path.join(self.SAPMresultsdir, self.SAPMparamsname)
        settings['SRresultsdir'] = self.SRresultsdir
        settings['SRresultsname'] = self.SRresultsname
        settings['SRparamsname'] = self.SRparamsname
        settings['SRdrawfile'] = self.SRdrawfile
        np.save(settingsfile, settings)

        # put some text as a place-holder
        self.SRLabel1 = tk.Label(self.parent, text = "1) Select SAPM results files... or use the Update \nbutton to load from previous pages", fg = 'gray', justify = 'left', font = infofont)
        self.SRLabel1.grid(row=0,column=0, sticky='W')
        self.SRLabel3 = tk.Label(self.parent, text = "2) Select covariates file if outputs are to show \ncorrelations etc with covariates", fg = 'gray', justify = 'left', font = infofont)
        self.SRLabel3.grid(row=1,column=0, sticky='W')
        self.SRLabel3 = tk.Label(self.parent, text = "3) For DrawSAPMdiagram you must choose xlsx files \ndefining plot parameters and results to plot", fg = 'gray', justify = 'left', font = infofont)
        self.SRLabel3.grid(row=2,column=0, sticky='W')

        rownum = 0
        # define a button to browse and select an existing network definition file, and write out the selected name
        # also, define the function for what to do when this button is pressed
        self.SRupdatevalues = tk.Button(self.parent, text='Update', width=smallbuttonsize, bg = fgcol1, fg = fgletter1, font = widgetfont,
                                  command=self.SRupdatevaluesclick, relief='raised', bd=5, highlightbackground = widgetbg)
        self.SRupdatevalues.grid(row = rownum, column=4, rowspan = 2)


        # cnums--------------------------------------------------
        # display the cluster numbers from the SAPM analysis page
        self.SRL0 = tk.Label(self.parent, text="Cluster numbers:", font = labelfont)
        self.SRL0.grid(row=rownum, column=1, sticky='E')
        # make a label to show the current setting of the network definition file name
        self.SRcnumtext = tk.StringVar()
        self.SRcnumtext.set(self.SAPMcnums)
        self.SRcnumtextbox = tk.Label(self.parent, textvariable=self.SRcnumtext, bg=bgcol, fg="#4B4B4B", font = infofont,
                                     wraplength=300, justify='left')
        self.SRcnumtextbox.grid(row=rownum, column=2, columnspan=2, sticky='W')


        rownum = 1
        # network file--------------------------------------------------
        # display the network file from the SAPM analysis page
        self.SRL1 = tk.Label(self.parent, text="Network Model:", font = labelfont)
        self.SRL1.grid(row=rownum, column=1, sticky='E')
        # make a label to show the current setting of the network definition file name
        npname, nfname = os.path.split(self.networkmodel)
        self.SRnetnametext = tk.StringVar()
        self.SRnetnametext.set(nfname)
        self.SRnetnametextbox = tk.Label(self.parent, textvariable=self.SRnetnametext, bg=bgcol, fg="#4B4B4B", font = infofont,
                                     wraplength=300, justify='left')
        self.SRnetnametextbox.grid(row=rownum, column=2, columnspan=2, sticky='W')

        # outputdir ----------------------------------------------------------------------
        # box etc for entering the name of the directory for saving the results
        # make a label to show the current setting of the network definition file directory name
        rownum = 2
        self.SRresultsdirlabel = tk.Label(self.parent, text = 'Results save folder:', font = labelfont)
        self.SRresultsdirlabel.grid(row=rownum, column=1, sticky='E')
        self.SRresultsdirtext = tk.StringVar()
        self.SRresultsdirtext.set(self.SRresultsdir)
        self.SRresultsdirbox = tk.Label(self.parent, textvariable=self.SRresultsdirtext, bg=bgcol, fg="#4B4B4B", font = infofont,
                                     wraplength=300, justify='left')
        self.SRresultsdirbox.grid(row=rownum, column=2, columnspan=2, sticky='W')

        # the entry boxes need a "browse" button to allow selection of existing cluster definition file
        self.SRresultsdirbrowse = tk.Button(self.parent, text = "Browse", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SRresultsdirbrowseaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SRresultsdirbrowse.grid(row=rownum, column=4, sticky='N')

        rownum = 3
        # put in base name for output files
        self.SRL9 = tk.Label(self.parent, text = 'Output name base:', font = labelfont).grid(row=rownum, column=1, sticky='NSEW')
        self.SRnametagbox = tk.Entry(self.parent, width = 30, bg="white")
        self.SRnametagbox.grid(row=rownum, column=2, columnspan = 2, sticky = "W")
        self.SRnametagbox.insert(0,self.SRnametag)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.SRnamesubmitbutton = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2,
                                font = widgetfont, command = self.SRnametagsubmit, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SRnamesubmitbutton.grid(row=rownum, column=4)


        # SAPM results name --------------------------------------------------------------
        rownum = 4
        # box etc for entering the name used in labeling the results files
        self.SRresultsnamelabel = tk.Label(self.parent, text = 'SAPM results file:', font = labelfont)
        self.SRresultsnamelabel.grid(row=rownum, column=1, sticky='E')
        self.SRresultsnametext = tk.StringVar()
        self.SRresultsnametext.set(self.SRresultsname)
        self.SRresultsnamebox = tk.Label(self.parent, textvariable=self.SRresultsnametext, bg=bgcol, fg="#4B4B4B", font = infofont,
                                     wraplength=300, justify='left')
        self.SRresultsnamebox.grid(row=rownum, column=2, columnspan=2, sticky='W')
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.SRresultsnamesubmit = tk.Button(self.parent, text = "Browse", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SRresultsnamebrowseaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SRresultsnamesubmit.grid(row=rownum, column=4, sticky='N')


        # SAPM parameters name --------------------------------------------------------------
        rownum = 5
        # box etc for entering the name used in labeling the parameters file
        self.SRparamsnamelabel = tk.Label(self.parent, text = 'SAPM parameters file:', font = labelfont)
        self.SRparamsnamelabel.grid(row=rownum, column=1, sticky='E')
        self.SRparamsnametext = tk.StringVar()
        self.SRparamsnametext.set(self.SRparamsname)
        self.SRparamsnamebox = tk.Label(self.parent, textvariable=self.SRparamsnametext, bg=bgcol, fg="#4B4B4B", font = infofont,
                                     wraplength=300, justify='left')
        self.SRparamsnamebox.grid(row=rownum, column=2, columnspan=2, sticky='W')
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.SRparamsnamebrowse = tk.Button(self.parent, text = "Browse", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SRparamsnamebrowseaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SRparamsnamebrowse.grid(row=rownum, column=4, sticky='N')

        # covariates file name --------------------------------------------------------------
        rownum = 6
        # box etc for entering the file name that contains information about covariates ...
        self.SRcovlabel = tk.Label(self.parent, text = 'Covariates file:', font = labelfont)
        self.SRcovlabel.grid(row=rownum, column=1, sticky='E')
        self.SRcovnametext = tk.StringVar()
        self.SRcovnametext.set(self.SRcovname)
        self.SRcovnamebox = tk.Label(self.parent, textvariable=self.SRcovnametext, bg=bgcol, fg="#4B4B4B", font = infofont,
                                     wraplength=300, justify='left')
        self.SRcovnamebox.grid(row=rownum, column=2, columnspan=2, sticky='W')
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.SRcovnamebrowse = tk.Button(self.parent, text = "Browse", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SRcovnamebrowseaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SRcovnamebrowse.grid(row=rownum, column=4, sticky='N')

        rownum = 7
        # add label, and pull-down menu for selected covariate values for searching
        self.SRL3 = tk.Label(self.parent, text = "Covariate term:", font = labelfont)
        self.SRL3.grid(row=rownum,column=1, sticky='W')

        # covariate pull-down menu
        # fieldvalues = DBFrame.get_DB_field_values(self)
        self.covnamelist = ['empty']
        self.covariatesvalues = []
        self.SRcovfield_var = tk.StringVar()
        self.SRcovfield_var.set('empty')
        fieldvalues = 'empty'
        SRcovfieldvalue_menu = tk.OptionMenu(self.parent, self.SRcovfield_var, *fieldvalues, command = self.SRcovfieldvaluechoice)
        SRcovfieldvalue_menu.config(bg=bgcol)
        SRcovfieldvalue_menu.grid(row=7, column=2, sticky='EW')
        self.SRcovfieldvaluesearch_opt = SRcovfieldvalue_menu   # save this way so that values are not cleared

        # covariate pull-down menu
        # fieldvalues = DBFrame.get_DB_field_values(self)
        self.SRcovvalue_var = tk.StringVar()
        self.SRcovvalue_var.set('empty')
        fieldvalues = 'empty'
        SRcovvalue_menu = tk.OptionMenu(self.parent, self.SRcovvalue_var, *fieldvalues, command = self.SRcovvaluechoice)
        SRcovvalue_menu.config(bg=bgcol)
        SRcovvalue_menu.grid(row=rownum, column=3, sticky='EW')
        self.SRcovvaluesearch_opt = SRcovvalue_menu   # save this way so that values are not cleared

        # specific which outputs to generate...
        rownum = 8
        outputoptions = ['B_Significance', 'B_Regression', 'Plot_BOLDModel','Plot_SourceModel', 'DrawSAPMdiagram', 'DrawAnatomy_axial', 'DrawAnatomy_sagittal' ]
        # add label, and pull-down menu for selected covariate values for searching
        self.SRL5 = tk.Label(self.parent, text = "Output: ", font = labelfont)
        self.SRL5.grid(row=rownum,column=1, sticky='W')
        # option pull-down menu
        self.SRoption_var = tk.StringVar()
        self.SRoption_var.set('empty')
        SRoptionvalue_menu = tk.OptionMenu(self.parent, self.SRoption_var, *outputoptions, command = self.SRoptionvalue_choice)
        SRoptionvalue_menu.config(bg=bgcol)
        SRoptionvalue_menu.grid(row=rownum, column=2, sticky='EW')
        self.SRoptionvalue_opt = SRoptionvalue_menu   # save this way so that values are not cleared

        # specify target region for generating outputs (if applicable)
        rownum = 8
        try:
            network, nclusterlist, sapm_region_list, fintrinsic_count, vintrinsic_count \
                = pysapm.load_network_model_w_intrinsics(self.networkmodel)
            region_list = [x for x in sapm_region_list if 'intrinsic' not in x]
        except:
            region_list = 'not defined'

        # add label, and pull-down menu for selected covariate values for searching
        self.SRL6 = tk.Label(self.parent, text = "Target Region: ", font = labelfont)
        self.SRL6.grid(row=rownum,column=3, sticky='W')
        # option pull-down menu
        self.SRtargetregion_var = tk.StringVar()
        self.SRtargetregion_var.set('empty')
        SRtargetregion_menu = tk.OptionMenu(self.parent, self.SRtargetregion_var, *region_list, command = self.SRtargetregionvalue_choice)
        SRtargetregion_menu.config(bg=bgcol)
        SRtargetregion_menu.grid(row=rownum, column=4, sticky='EW')
        self.SRtargetregion_opt = SRtargetregion_menu   # save this way so that values are not cleared

        rownum = 9
        # put in choices for statistical threshold
        self.SRL7 = tk.Label(self.parent, text = 'p-value:', font = labelfont).grid(row=rownum, column=1, sticky='NSEW')
        self.SRpvaluebox = tk.Entry(self.parent, width = 8, bg="white")
        self.SRpvaluebox.grid(row=rownum, column=2, sticky = "W")
        self.SRpvaluebox.insert(0,self.SRpvalue)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.SRpvaluesubmitbut = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2,
                                font = widgetfont, command = self.SRpvaluesubmit, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SRpvaluesubmitbut.grid(row=rownum, column=3)


        # choose run variation number (latent inputs being +/-)
        self.SRL8 = tk.Label(self.parent, text = 'result variant:', font = labelfont).grid(row=rownum, column=4, sticky='NSEW')
        self.SRvariantvaluebox = tk.Entry(self.parent, width = 8, bg="white")
        self.SRvariantvaluebox.grid(row=rownum, column=5, sticky = "W")
        self.SRvariantvaluebox.insert(0,self.SRvariant)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.SRvariantvaluesubmitbut = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2,
                                font = widgetfont, command = self.SRvariantvaluesubmit, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SRvariantvaluesubmitbut.grid(row=rownum, column=6)

        rownum = 10
        # button to launch the generation of outputs with the selected options/values
        self.SRrunoutputbutton = tk.Button(self.parent, text = "GO", width = smallbuttonsize, bg = fgcol1, fg = fgletter1, font = widgetfont, command = self.SRgenerateoutput, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SRrunoutputbutton.grid(row=rownum, column=0, sticky='N')

        self.SRPlotFigure = plt.figure(123, figsize=(3, 1.5), dpi=100)
        self.SRPlotAx = self.SRPlotFigure.add_subplot(111)
        self.SRCanvas = FigureCanvasTkAgg(self.SRPlotFigure, self.parent)
        self.SRCanvas.get_tk_widget().grid(row=rownum, column=1,rowspan = 6, columnspan = 4, sticky='NW')
        self.SRPlotAx.set_title('SAPM results to be shown here')

        #
        # inputs and info for displaying the SAPM network diagram
        # outputdir ----------------------------------------------------------------------
        # box etc for entering the name of the directory for saving the results
        # make a label to show the current setting of the network definition file directory name

        rownum = 10
        columnum = 4
        self.SRdrawfilelabel = tk.Label(self.parent, text = 'Draw params:', font = labelfont)
        self.SRdrawfilelabel.grid(row=rownum, column=columnum, sticky='E')
        self.SRdrawfiletext = tk.StringVar()
        self.SRdrawfiletext.set(self.SRdrawfile[-30:])
        self.SRdrawfilebox = tk.Label(self.parent, textvariable=self.SRdrawfiletext, bg=bgcol, fg="#4B4B4B", font = infofont,
                                     wraplength=150, justify='left', state = tk.DISABLED)
        self.SRdrawfilebox.grid(row=rownum, column=columnum+1, sticky='W')
        # the entry boxes need a "browse" button to allow selection of existing cluster definition file
        self.SRdrawfilebrowse = tk.Button(self.parent, text = "Browse", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SRdrawfilebrowseaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SRdrawfilebrowse.grid(row=rownum, column=columnum+2, sticky='W')

        rownum = 11
        # SAPMBfile
        columnum = 4
        self.SAPMBfilelabel = tk.Label(self.parent, text = 'SAPM B file:', font = labelfont)
        self.SAPMBfilelabel.grid(row=rownum, column=columnum, sticky='E')
        self.SAPMBfiletext = tk.StringVar()
        self.SAPMBfiletext.set(self.SAPMBfile[-30:])
        self.SAPMBfilebox = tk.Label(self.parent, textvariable=self.SAPMBfiletext, bg=bgcol, fg="#4B4B4B", font = infofont,
                                     wraplength=150, justify='left', state = tk.DISABLED)
        self.SAPMBfilebox.grid(row=rownum, column=columnum+1, sticky='W')
        # the entry boxes need a "browse" button to allow selection of existing cluster definition file
        self.SAPMBfilebrowse = tk.Button(self.parent, text = "Browse", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, font = widgetfont, command = self.SAPMBfilebrowseaction, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SAPMBfilebrowse.grid(row=rownum, column=columnum+2, sticky='W')

        rownum = 12
        columnum = 4
        # option pull-down menu
        # initialchoice = 'sheet'
        self.SAPMBsheetfield_var = tk.StringVar()
        self.SAPMBsheetfield_var.set('sheet')
        SAPMBsheet_menu = tk.OptionMenu(self.parent, self.SAPMBsheetfield_var, *self.SRsheetnames, command = self.SAPMBsheetfieldvaluechoice)
        SAPMBsheet_menu.config(bg=bgcol)
        SAPMBsheet_menu.grid(row=rownum, column=columnum, sticky='EW')
        SAPMBsheet_menu.config(state = tk.DISABLED)
        self.SAPMBsheetfield_search_opt = SAPMBsheet_menu   # save this way so that values are not cleared

        rownum = 12
        columnum = 5
        # option pull-down menu
        # initialchoice = 'stat'
        self.SAPMBcolumnfield_var = tk.StringVar()
        self.SAPMBcolumnfield_var.set('stat')
        SAPMBcolumn_menu = tk.OptionMenu(self.parent, self.SAPMBcolumnfield_var, *self.SRcolumnnames, command = self.SAPMBcolumnfieldvaluechoice)
        SAPMBcolumn_menu.config(bg=bgcol)
        SAPMBcolumn_menu.grid(row=rownum, column=columnum, sticky='EW')
        SAPMBcolumn_menu.config(state = tk.DISABLED)
        self.SAPMBcolumnfield_search_opt = SAPMBcolumn_menu   # save this way so that values are not cleared

        rownum = 13
        columnum = 4
        # put in base name for output files
        self.SRL11 = tk.Label(self.parent, text = 'Threshold:', font = labelfont).grid(row=rownum, column=columnum, sticky='NSEW')
        self.SRthresholdbox = tk.Entry(self.parent, width = 20, bg="white", state = tk.DISABLED)
        self.SRthresholdbox.grid(row=rownum, column=columnum+1, columnspan = 1, sticky = "W")
        self.SRthresholdbox.insert(0,self.SRthresholdtext)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.SRthresholdsubmitbutton = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2,
                                font = widgetfont, command = self.SRthresholdsubmit, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.SRthresholdsubmitbutton.grid(row=rownum, column=columnum+2)

#----------MAIN calling function----------------------------------------------------
# the main function that starts everything running
def main():
    root = tk.Tk()
    root.title('FMRI Analysis')

    # original that only works on PCs
    # tk.Tk.iconbitmap(root, default='lablogoicon.ico')

    logofile = os.path.join(basedir, 'lablogo.gif')
    img = tk.PhotoImage('photo', file=logofile)
    # root.iconphoto(True, img) # you may also want to try this.
    root.tk.call('wm', 'iconphoto', root._w, img)

    app = mainspinalfmri_window(root)

    root.mainloop()

if __name__ == '__main__':
    main()
    
    
