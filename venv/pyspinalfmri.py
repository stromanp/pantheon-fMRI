"""
pyspinalfmri.py
version 0.0.0

Created on Tue Apr 21 16:16:09 2020
Last edited  10:33 am Saturday, April 25, 2020

@author: Patrick W. Stroman, Queen's University, Kingston

This module organizes inputs to functions to load, pre-process, and analyze 
functional MRI data from the brainstem and spinal cord

Jobs to do:  (not entirely complete yet)
1) Specify database file (done, excepting for creating new database file, and functions to read the database)
2) Specify database entry numbers to work on (done)
3) Conversion from DICOM to NIfTI format (done)
4) Calculation of normalization parameters (done)
5) Pre-processing of data (done)
6) Model-driven fit of predicted BOLD responses to voxel data (GLM)) (done, except for creating results figure)
7) Definition of region clusters, and extraction of cluster data
8) Data-driven connectivity analysis, using structural equation modeling (SEM))
9) Bayesian regression analysis of cluster data
10) Visualization of results

"""

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
# import scipy
import py2ndlevelanalysis
import copy
import math

matplotlib.use('TkAgg')   # explicitly set this - it might help with displaying figures in different environments

# save some colors for consistent layout, and make them easy to change
fgcol1 = 'navy'
fgcol2 = 'gold2'
fgcol3 = 'firebrick4'
bgcol = 'grey94'

bigbigbuttonsize = 21
bigbuttonsize = 14
smallbuttonsize = 9

# define a single place for saving setup parameters for ease of retrieving, updating, etc.
basedir = os.getcwd()
settingsfile = os.path.join(basedir,'base_settings_file.npy')

#-----load the preferences file and information in it --------------------
# preferencesfile = os.path.join(basedir,'preferences.ini')
# preferences = cp.ConfigParser()
# preferences.read(preferencesfile)
# fields = preferences.options('OPTIONAL_FIELDS')
# optional_db_fields = [preferences['OPTIONAL_FIELDS'][fields[aa]] for aa in range(len(fields))]
# optional_db_fields
#-------------------------------------------------------------------------

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
            'GRPresultsname':'notdefined',
            'GRPresultsname2':'notdefined',
            'GRPcharacteristicscount':0,
            'GRPcharacteristicslist':[],
            'GRPcharacteristicsvalues':[],
            'GRPcharacteristicsvalues2':[],
            'GRPanalysistype':'undefined',
            'GRPdatafiletype1':0,
            'GRPdatafiletype2':0,
            'GRPpvalue':0.05}
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
        Fbase = tk.Frame(self.master, relief='raised', bd=5, highlightcolor=fgcol2)
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

        IMGbase = tk.Frame(self.master, relief='raised', bd=5, highlightcolor=fgcol1)
        IMGbase.grid(row=1, column=1, sticky="nsew")
        IMGFrame = IMGDispFrame(IMGbase, self)
        page_name = IMGDispFrame.__name__
        self.frames[page_name] = IMGbase
        self.imgwindow = IMGFrame

        # The remaining frames are on the bottom of the main window, and are on top of each other
        # All of these frames are defined at the same time, but only the top one is visible
        DBbase = tk.Frame(self.master, relief='raised', bd=5, highlightcolor=fgcol1)
        DBbase.grid(row=1, column=1, sticky="nsew")
        DBFrame(DBbase, self)
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

        SEMbase = tk.Frame(self.master, relief='raised', bd=5, highlightcolor=fgcol1)
        SEMbase.grid(row=1, column=1, sticky="nsew")
        SEMFrame(SEMbase, self)
        page_name = SEMFrame.__name__
        self.frames[page_name] = SEMbase

        GRPbase = tk.Frame(self.master, relief='raised', bd=5, highlightcolor=fgcol1)
        GRPbase.grid(row=1, column=1, sticky="nsew")
        GRPFrame(GRPbase, self)
        page_name = GRPFrame.__name__
        self.frames[page_name] = GRPbase


        # start with the Database information frame on top
        self.show_frame('DBFrame')


    # Definition of the function to choose which frame is on top
    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        keylist = self.frames.keys()
        if page_name in keylist:
            frame = self.frames[page_name]
            frame.tkraise()

    def get_display_window(self, windownumber):
        # access the display windows
        window, image_in_window = self.imgwindow.get_window(windownumber)
        return window, image_in_window



# ------Create the Window for displaying images, results, etc.---------------------
class IMGDispFrame:
    # defines the image display window, separate from the main window so the main window is not too bulky
    def __init__(self, parent, controller):
        # tk.Frame.__init__(self, master)
        self.parent = parent
        self.displaycontroller = controller

        # need to initialize these here and use them later to save some display items
        self.photo1 = []
        self.photo2 = []

        # first create the frames that do not move
        self.frames = {}  # this is to keep a record of the frame names, so that one can be moved to the top when wanted
        # create an initial fixed frame with labels etc
        # The frame with pictures, labels, etc, is Fbase and is on the top row
        Fbasew = tk.Frame(self.parent, relief='raised', bd=5, highlightcolor=fgcol2)
        Fbasew.grid(row=0, column=0, columnspan=2)
        BaseFrame2(Fbasew, self)
        page_name = BaseFrame2.__name__

        # create the display windows - 4 of them, for displaying images, results, etc...
        # make objects in the display frame - for testing as place holders
        # load in a picture, for no good reason, and display it in the window to look nice :)
        photo1 = tk.PhotoImage(file=os.path.join(basedir, 'queens_flag2.gif'))
        controller.photod1 = photo1  # need to keep a copy so it is not cleared from memory
        # put this figure, in the 1st row, 1st column, of a grid layout for the window
        # and make the background black
        self.W1 = tk.Canvas(master = self.parent, width=photo1.width(), height=photo1.height(), bg='black')
        self.W1.grid(row=1, column=0, sticky='W')
        self.image_in_W1 = self.W1.create_image(0, 0, image=photo1, anchor = tk.NW)
        # self.W1 = W1
        self.frames['window1'] = self.W1
        self.frames['image_in_window1'] = self.image_in_W1

        # load in another picture, because if one picture is good, two is better
        photo2 = tk.PhotoImage(file=os.path.join(basedir, 'lablogo.gif'))
        controller.photod2 = photo2  # need to keep a copy so it is not cleared from memory
        # put in another figure, for pure artistic value, in the 1st row, 2nd column, of a grid layout for the window
        # and make the background black
        self.W2 = tk.Canvas(master=self.parent, width=photo2.width(), height=photo2.height(), bg='black')
        self.W2.grid(row=1, column=1, sticky='W')
        self.image_in_W2 = self.W2.create_image(0, 0, image=photo2, anchor=tk.NW)
        self.frames['window2'] = self.W2
        self.frames['image_in_window2'] = self.image_in_W2

        photo3 = tk.PhotoImage(file=os.path.join(basedir, 'queens_flag2.gif'))
        controller.photod3 = photo3  # need to keep a copy so it is not cleared from memory
        self.W3 = tk.Canvas(master=self.parent, width=photo3.width(), height=photo3.height(), bg='black')
        self.W3.grid(row=2, column=0, sticky='W')
        self.image_in_W3 = self.W3.create_image(0, 0, image=photo3, anchor=tk.NW)
        self.frames['window3'] = self.W3
        self.frames['image_in_window3'] = self.image_in_W3

        photo4 = tk.PhotoImage(file=os.path.join(basedir, 'lablogo.gif'))
        controller.photod4 = photo4  # need to keep a copy so it is not cleared from memory
        self.W4 = tk.Canvas(master=self.parent, width=photo4.width(), height=photo4.height(), bg='black')
        self.W4.grid(row=2, column=1, sticky='W')
        self.image_in_W4 = self.W4.create_image(0, 0, image=photo4, anchor = tk.NW)
        self.frames['window4'] = self.W4
        self.frames['image_in_window4'] = self.image_in_W4

        controller.W1 = self.W1  # pass upward for access
        controller.W2 = self.W2  # pass upward for access
        controller.W3 = self.W3  # pass upward for access
        controller.W4 = self.W4  # pass upward for access


    def get_window(self, window_number):
        if window_number == 1:
            window = self.frames['window1']
            image_in_window = self.frames['image_in_window1']
            print('returning window number 1')
        if window_number == 2:
            window = self.frames['window2']
            image_in_window = self.frames['image_in_window2']
            print('returning window number 2')
        if window_number == 3:
            window = self.frames['window3']
            image_in_window = self.frames['image_in_window3']
            print('returning window number 3')
        if window_number == 4:
            window = self.frames['window4']
            image_in_window = self.frames['image_in_window4']
            print('returning window number 4')
        return window, image_in_window


# --------------------DISPLAY FRAME---------------------------------------------------------------
# Definition of the frame that holds image display windows
class DisplayFrame:
    # initialize the values, keeping track of the frame this definition works on (parent), and
    # also the main window containing that frame (controller)
    def __init__(self, parent, controller):
        parent.configure(relief='raised', bd=5, highlightcolor=fgcol2)  # just defining some visual features
        self.parent = parent
        self.controller = controller

        # make objects in the display frame - for testing as place holders
        # load in a picture, for no good reason, and display it in the window to look nice :)
        photo1 = tk.PhotoImage(file=os.path.join(basedir, 'queens_flag2.gif'))
        controller.photod1 = photo1  # need to keep a copy so it is not cleared from memory
        # put this figure, in the 1st row, 1st column, of a grid layout for the window
        # and make the background black
        self.W1 = tk.Label(self.parent, image=photo1, bg='grey94').grid(row=0, column=0, sticky='W')

        # load in another picture, because if one picture is good, two is better
        photo2 = tk.PhotoImage(file=os.path.join(basedir, 'lablogo.gif'))
        controller.photod2 = photo2  # need to keep a copy so it is not cleared from memory
        # put in another figure, for pure artistic value, in the 1st row, 2nd column, of a grid layout for the window
        # and make the background black
        self.W2 = tk.Label(self.parent, image=photo2, bg='grey94').grid(row=0, column=1, sticky='W')

        photo3 = tk.PhotoImage(file=os.path.join(basedir, 'queens_flag2.gif'))
        controller.photod3 = photo3  # need to keep a copy so it is not cleared from memory
        self.W3 = tk.Label(self.parent, image=photo3, bg='grey94').grid(row=1, column=0, sticky='W')

        photo4 = tk.PhotoImage(file=os.path.join(basedir, 'lablogo.gif'))
        controller.photod4 = photo4  # need to keep a copy so it is not cleared from memory
        self.W4 = tk.Label(self.parent, image=photo2, bg='grey94').grid(row=1, column=1, sticky='W')


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
        photo1 = tk.PhotoImage(file = os.path.join(basedir,'queens_flag2.gif'))
        controller.photo1 = photo1   # need to keep a copy so it is not cleared from memory
        # put this figure, in the 1st row, 1st column, of a grid layout for the window
        # and make the background black
        self.P1=tk.Label(self.parent, image = photo1, bg='grey94').grid(row=0, column=0, sticky = 'W')
        
        # load in another picture, because if one picture is good, two is better
        photo2 = tk.PhotoImage(file = os.path.join(basedir,'lablogo.gif'))
        controller.photo2 = photo2   # need to keep a copy so it is not cleared from memory
        # put in another figure, for pure artistic value, in the 1st row, 2nd column, of a grid layout for the window
        # and make the background black
        self.P2=tk.Label(self.parent, image = photo2, bg='grey94').grid(row=0, column=1, sticky = 'W')

        # create a label under the pictures (row 2), spanning two columns, to tell the user what they are running
        # specify a black background and white letters, with 12 point bold font
        self.L0 = tk.Label(self.parent, text = "Whole CNS fMRI Analysis", bg = bgcol, fg = fgcol1, font = "none 16 bold")
        self.L0.grid(row=1, column = 0, columnspan = 2, sticky = 'W')


# --------------------BASE FRAME2---------------------------------------------------------------
# Definition of the frame that holds pictures, labels, etc.
class BaseFrame2:
    # initialize the values, keeping track of the frame this definition works on (parent), and
    # also the main window containing that frame (controller)
    def __init__(self, parent, controller):
        parent.configure(relief='raised', bd=5, highlightcolor=fgcol2)  # just defining some visual features
        self.parent = parent
        self.controller = controller

        # make objects in the base frame
        # load in a picture, for no good reason, and display it in the window to look nice :)
        photo1 = tk.PhotoImage(file=os.path.join(basedir, 'queens_flag2.gif'))
        controller.photob1 = photo1  # need to keep a copy so it is not cleared from memory
        # put this figure, in the 1st row, 1st column, of a grid layout for the window
        # and make the background black
        self.B1 = tk.Label(self.parent, image=photo1, bg='grey94').grid(row=0, column=0, sticky='W')

        # load in another picture, because if one picture is good, two is better
        photo2 = tk.PhotoImage(file=os.path.join(basedir, 'lablogo.gif'))
        controller.photob2 = photo2  # need to keep a copy so it is not cleared from memory
        # put in another figure, for pure artistic value, in the 1st row, 2nd column, of a grid layout for the window
        # and make the background black
        self.B2 = tk.Label(self.parent, image=photo2, bg='grey94').grid(row=0, column=1, sticky='W')

        # create a label under the pictures (row 2), spanning two columns, to tell the user what they are running
        # specify a black background and white letters, with 12 point bold font
        self.L1 = tk.Label(self.parent, text="SC/BS fMRI Analysis", bg=bgcol, fg=fgcol1, font="none 16 bold")
        self.L1.grid(row=1, column=0, columnspan=2, sticky='W')

        
        
        
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
            buttonname.configure(fg='black')
        if widgetname in buttonlist:
            self.buttons[widgetname].configure(fg='white')
        self.controller.show_frame(page_name)


    def __init__(self, parent, controller):
        parent.configure(relief='raised', bd = 5, highlightcolor = fgcol1)
        self.parent = parent
        self.controller = controller
        self.buttons = {}  # to keep track of button handles

        # button for running the conversion to NIfTI format step
        self.setdb = tk.Button(self.parent, text = 'Database', width = smallbuttonsize, bg = fgcol2, fg = 'white', font = "none 9 bold", command = lambda: self.options_show_frame('DBFrame', 'setdb'), relief='raised', bd = 5)
        self.setdb.grid(row = 0, column = 0)
        self.buttons['setdb'] = self.setdb
        
        # button for running the conversion to NIfTI format step
        self.writenifti = tk.Button(self.parent, text = 'Write NIfTI', width = smallbuttonsize, bg = fgcol2, fg = 'black', font = "none 9 bold", command = lambda: self.options_show_frame('NIFrame', 'writenifti'), relief='raised', bd = 5)
        self.writenifti.grid(row = 1, column = 0)
        self.buttons['writenifti'] = self.writenifti

        # button for calculating the normalization parameters for each data set
        self.normalizationcalc = tk.Button(self.parent, text = 'Norm. Calc.', width = smallbuttonsize, bg = fgcol2, fg = 'black', font = "none 9 bold", command = lambda: self.options_show_frame('NCFrame', 'normalizationcalc'), relief='raised', bd = 5)
        self.normalizationcalc.grid(row = 2, column = 0)
        self.buttons['normalizationcalc'] = self.normalizationcalc

        # button for selecting and running the pre-processing steps
        self.preprocess = tk.Button(self.parent, text = 'Pre-Process', width = smallbuttonsize, bg = fgcol2, fg = 'black', font = "none 9 bold", command = lambda: self.options_show_frame('PPFrame', 'preprocess'), relief='raised', bd = 5)
        self.preprocess.grid(row = 3, column = 0)
        self.buttons['preprocess'] = self.preprocess

        # button for selecting and running the GLM analysis steps
        self.glmanalysis = tk.Button(self.parent, text = 'GLM fit', width = smallbuttonsize, bg = fgcol2, fg = 'black', font = "none 9 bold", command = lambda: self.options_show_frame('GLMFrame', 'glmanalysis'), relief='raised', bd = 5)
        self.glmanalysis.grid(row = 4, column = 0)
        self.buttons['glmanalysis'] = self.glmanalysis

        # button for selecting and running the data clustering steps
        self.clustering = tk.Button(self.parent, text = 'Cluster', width = smallbuttonsize, bg = fgcol2, fg = 'black', font = "none 9 bold", command = lambda: self.options_show_frame('CLFrame', 'clustering'), relief='raised', bd = 5)
        self.clustering.grid(row = 5, column = 0)
        self.buttons['clustering'] = self.clustering

        # button for selecting and running the SEM analysis steps
        self.sem = tk.Button(self.parent, text = 'SEM', width = smallbuttonsize, bg = fgcol2, fg = 'black', font = "none 9 bold", command = lambda: self.options_show_frame('SEMFrame', 'sem'), relief='raised', bd = 5)
        self.sem.grid(row = 6, column = 0)
        self.buttons['sem'] = self.sem

        # button for selecting and running the group-level analysis steps
        self.groupanalysis = tk.Button(self.parent, text = 'Group-level', width = smallbuttonsize, bg = fgcol2, fg = 'black', font = "none 9 bold", command = lambda: self.options_show_frame('GRPFrame', 'groupanalysis'), relief='raised', bd = 5)
        self.groupanalysis.grid(row = 7, column = 0)
        self.buttons['groupanalysis'] = self.groupanalysis

        # button for selecting and running the group-level analysis steps
        self.imgdisplay = tk.Button(self.parent, text = 'Display', width = smallbuttonsize, bg = fgcol1, fg = 'white', font = "none 9 bold", command = lambda: self.options_show_frame('IMGDispFrame', 'imgdisplay'), relief='raised', bd = 5)
        self.imgdisplay.grid(row = 8, column = 0)
        self.buttons['imgdisplay'] = self.imgdisplay
        
        # define a button to exit the GUI
        # also, define the function for what to do when this button is pressed
        self.exit_button = tk.Button(self.parent, text = 'Exit', width = smallbuttonsize, bg = 'grey80', fg = 'black', font = "none 9 bold", command = self.close_window, relief='sunken', bd = 5)
        self.exit_button.grid(row = 9, column = 0)

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
        self.DBL1 = tk.Label(self.parent, text = "Database name:")
        self.DBL1.grid(row=0,column=0, sticky='W')
        
        # make a label to show the current setting of the database name
        self.DBnametext = tk.StringVar()
        self.DBnametext.set(self.DBname)
        self.DBnamelabel2 = tk.Label(self.parent, textvariable = self.DBnametext, bg = bgcol, fg = "black", font = "none 10", wraplength = 200, justify = 'left')
        self.DBnamelabel2.grid(row=0, column = 1, sticky = 'W')
        
        # define a button to browse and select an existing database file, and write out the selected name
        # also, define the function for what to do when this button is pressed
        self.DBsubmit = tk.Button(self.parent, text = 'Browse', width = smallbuttonsize, bg = fgcol1, fg = 'white', command = self.DBbrowseclick, relief='raised', bd = 5)
        self.DBsubmit.grid(row = 0, column = 2)
        
        # define a button to select a new database file
        # also, define the function for what to do when this button is pressed
        self.DBnew = tk.Button(self.parent, text = 'New', width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.DBnewclick, relief='raised', bd = 5)
        self.DBnew.grid(row = 0, column = 3)
        
        # now define an Entry box so that the user can enter database numbers
        # give it a label first
        self.DBL2 = tk.Label(self.parent, text = "Database numbers:")
        self.DBL2.grid(row=1,column=0, sticky='W')
        
        # create the Entry box
        self.DBnumenter = tk.Entry(self.parent, width = 20, bg="white")
        self.DBnumenter.grid(row=1, column = 1, sticky = "W")
        self.DBnumenter.insert(0,settings['DBnumstring'])
        
        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.DBnumsubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol1, fg = 'white', command = self.DBnumsubmitclick, relief='raised', bd = 5)
        self.DBnumsubmit.grid(row = 1, column = 2)

        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.DBnumload = tk.Button(self.parent, text = "Load", width = smallbuttonsize, bg = fgcol1, fg = 'white', command = self.DBnumlistload, relief='raised', bd = 5)
        self.DBnumload.grid(row = 1, column = 3)

        # add a button to clear the entered values
        self.DBnumsubmit = tk.Button(self.parent, text = "Clear", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.DBnumclear, relief='raised', bd = 5)
        self.DBnumsubmit.grid(row = 1, column = 4)

        # put a separator for readability
        ttk.Separator(self.parent).grid(row=2, column=1, columnspan=6, sticky="nswe", padx=2, pady=5)

        # add label, and pull-down menu for selected database values for searching
        self.DBL3 = tk.Label(self.parent, text = "Database search:")
        self.DBL3.grid(row=3,column=0,columnspan = 2, sticky='W')

        # add a button to clear the search values
        self.DBsearchclear = tk.Button(self.parent, text = "Clear Search", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.DBsearchclear, relief='raised', bd = 5)
        self.DBsearchclear.grid(row = 3, column = 3)

        # add a button to run the search
        self.DBrunsearch = tk.Button(self.parent, text = "Run Search", width = smallbuttonsize, bg = fgcol1, fg = 'white', command = self.DBrunsearch, relief='raised', bd = 5)
        self.DBrunsearch.grid(row = 3, column = 2)

        self.DBL5 = tk.Label(self.parent, text = "Value:")
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
        fieldvalue_menu.grid(row=4, column=3, sticky='EW')
        self.fieldvaluesearch_opt = fieldvalue_menu   # save this way so that values are not cleared

        self.DBL4 = tk.Label(self.parent, text = "Keyword:")
        self.DBL4.grid(row=4,column=0, sticky='W')
        # fields = DBFrame.get_DB_fields(self)
        field_menu = tk.OptionMenu(self.parent, self.field_var, *fields, command = self.DBfieldchoice)
        field_menu.grid(row=4, column=1, sticky='EW')
        self.fieldsearch_opt = field_menu   # save this way so that values are not cleared

        # add information to show the current search terms
        self.searchterm_text = tk.StringVar()
        self.searchterm_text.set('Database search keys:  empty')
        self.DBsearchtext = tk.Label(self.parent, text = self.searchterm_text.get())
        self.DBsearchtext.grid(row=5,column=1,columnspan = 2, sticky='W')

        # add information to show the current search terms
        self.searchresult_text = tk.StringVar()
        self.searchresult_text.set('Database search results:  empty')
        self.DBsearchresult = tk.Label(self.parent, text = self.searchresult_text.get())
        self.DBsearchresult.grid(row=5,column=3,columnspan = 3, sticky='W')


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
        fieldvalue_menu.grid(row=4, column=3, sticky='EW')
        self.fieldvaluesearch_opt = fieldvalue_menu   # save this way so that values are not cleared

        field_menu = tk.OptionMenu(self.parent, self.field_var, *fields, command = self.DBfieldchoice)
        field_menu.grid(row=4, column=1, sticky='EW')
        self.fieldsearch_opt = field_menu   # save this way so that values are not cleared

        # change the current working directory to the folder where the database file is, for future convenience
        pathsections = os.path.split(filename)
        os.chdir(pathsections[0])
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
        os.chdir(pathsections[0])
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

        # need to make sure we are working with numbers, not text
        # first, replace any double spaces with single spaces, and then replace spaces with commas
        entered_text = re.sub('\ +',',',entered_text)
        entered_text = re.sub('\,\,+',',',entered_text)
        # remove any leading or trailing commas
        if entered_text[0] == ',': entered_text = entered_text[1:]
        if entered_text[-1] == ',': entered_text = entered_text[:-1]
        
        self.DBnumsave_text = entered_text
        
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
            num2 = int(m[0][(m[0].find(':')+1):])
            # now create the long string that we need to replace the short form indicated with the colon
            numbers = np.arange(num1,num2+1)
            numtext = ''
            for n in numbers:
                numtext1 = '{:d},'.format(n)
                numtext += numtext1
            # now insert this where the colon separated pair of number came out
            new_text = entered_text[:m.start()] + numtext + entered_text[(m.end()+1):]
            entered_text = new_text
            m = re.search(r'\d*:\d*', entered_text)
            
        entered_values = np.fromstring( entered_text, dtype=int, sep=',')

        # check upper limit
        entered_values = entered_values[entered_values <= dbnum_max]
        # parse the entered text into values
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
        self.NIinfo1 = tk.Label(self.parent, text = "Base name:")
        self.NIinfo1.grid(row=0,column=1, sticky='E')
        
        # create the Entry box, and put it next to the label, 4th row, 2nd column
        self.NInameenter = tk.Entry(self.parent, width = 20, bg="white")
        self.NInameenter.grid(row=0, column = 2, sticky = "W")
        self.NInameenter.insert(0,settings['NIbasename'])
        
        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.NInamesubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.NInamesubmitclick, relief='raised', bd = 5)
        self.NInamesubmit.grid(row = 0, column = 3)
        
        # for now, just put a button that will eventually call the NIfTI conversion program
        self.NIorganizedata = tk.Button(self.parent, text = 'Organize Data', width = bigbuttonsize, bg = fgcol1, fg = 'white', command = self.NIorganizeclick, font = "none 9 bold", relief='raised', bd = 5)
        self.NIorganizedata.grid(row = 1, column = 1, columnspan = 2)

        # for now, just put a button that will eventually call the NIfTI conversion program
        self.NIrunconvert = tk.Button(self.parent, text = 'Convert', width = bigbuttonsize, bg = fgcol1, fg = 'white', command = self.NIconversionclick, font = "none 9 bold", relief='raised', bd = 5)
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

        self.NCtemplatelabel = tk.Label(self.parent, text = 'Normalizing region: '+self.normtemplatename, fg = 'gray')
        self.NCtemplatelabel.grid(row=0,column=2, sticky='W')

        # put some text as a place-holder
        self.NClabel1 = tk.Label(self.parent, text = "1) Calculate normalization\nparameters", fg = 'gray')
        self.NClabel1.grid(row=0,column=0, sticky='W')
        self.NClabel2 = tk.Label(self.parent, text = "2) Save for next steps", fg = 'gray')
        self.NClabel2.grid(row=1,column=0, sticky='W')

        # now define an Entry box so that the user can indicate the prefix name of the data to normalize
        # give it a label first
        self.NCinfo1 = tk.Label(self.parent, text = "Save name base:")
        self.NCinfo1.grid(row=1,column=1, sticky='E')

        # create the Entry box, and put it next to the label
        self.NCsavename = tk.Entry(self.parent, width = 20, bg="white")
        self.NCsavename.grid(row=1, column = 2, sticky = "W")
        self.NCsavename.insert(1,self.normdatasavename)

        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.NCsavenamesubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.NCsavenamesubmit, relief='raised', bd = 5)
        self.NCsavenamesubmit.grid(row = 1, column = 3)

        # need entry boxes for "fit_parameters"
        # fit_parameters: 0 position stiffness upper  1 position stiffness lower  2 angle_stiffness upper 3 angle stiffness lower
        # 4 anglestart, 5 anglestop 6 anglestart lower 7 anglestop lower
        # these parameters are scaled by [1e-4, 1e-4, 1e-4, 1, 1, 1, 1, 1] before use  so the inputs are more intuitive values

        # label the columns
        self.NCupper_label = tk.Label(self.parent, text = "Brainstem Regions")
        self.NCupper_label.grid(row=2,column=2, sticky='W')
        self.NClower_label = tk.Label(self.parent, text = "Cord Regions")
        self.NClower_label.grid(row=2,column=3, sticky='W')

        # define an Entry box for each fit_parameters value
        # give it a label -- position stiffness
        self.NCfitp0_label = tk.Label(self.parent, text = "Position Stiffness (0-100):")
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
        self.NCfitp2_label = tk.Label(self.parent, text="Angle Stiffness (0-100):")
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
        self.NCfitp4_label = tk.Label(self.parent, text="Angle Start,Stop (degrees):")
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
        self.NCfitparamsubmit = tk.Button(self.parent, text="Submit", width=smallbuttonsize, bg=fgcol2, fg='black',
                                          command=self.NCfitparamsubmit, relief='raised', bd=5)
        self.NCfitparamsubmit.grid(row=3, column=4, rowspan=3)

        # checkboxes to indicate 1) do rough normalization, 2) do fine-tuning normalization
        self.var1 = tk.IntVar()
        self.NCroughnorm = tk.Checkbutton(self.parent, text = 'Rough Norm.', width = smallbuttonsize, fg = 'black',
                                          command = self.NCcheckboxes, variable = self.var1)
        self.NCroughnorm.grid(row = 4, column = 0, sticky="E")
        self.var2 = tk.IntVar()
        self.NCfinetune = tk.Checkbutton(self.parent, text = 'Fine-Tune', width = smallbuttonsize, fg = 'black',
                                          command = self.NCcheckboxes, variable = self.var2)
        self.NCfinetune.grid(row = 5, column = 0, sticky="E")

        # button to call the normalization program
        self.NCrun = tk.Button(self.parent, text = 'Calculate Normalization', width = bigbigbuttonsize, bg = fgcol1, fg = 'white', command = self.NCrunclick, font = "none 9 bold", relief='raised', bd = 5)
        self.NCrun.grid(row = 6, column = 1)

        # button to run program to manually adjust rough normalization sections
        self.NCmano = tk.Button(self.parent, text = 'Manual Over-ride', width = bigbuttonsize, bg = fgcol3, fg = 'white', command = self.NCmanoclick, font = "none 9 bold", relief='raised', bd = 5)
        self.NCmano.grid(row = 6, column = 2)

        # button to recalculate normalization after manual over-ride
        self.NCrun = tk.Button(self.parent, text = 'Recalculate', width = bigbigbuttonsize, bg = fgcol3, fg = 'white', command = self.NCrecalculate_after_override, font = "none 9 bold", relief='raised', bd = 5)
        self.NCrun.grid(row = 7, column = 5)


        # entry box and button to copy rough normalization from another database number
        self.NCcopy_label = tk.Label(self.parent, text="Copy rough normalization from another data set:")
        self.NCcopy_label.grid(row=8, column=5, columnspan = 2, sticky='E')
        self.NCcopy_label = tk.Label(self.parent, text="Datbase number:")
        self.NCcopy_label.grid(row=9, column=5, sticky='E')
        # create the Entry box, for position stiffness upper
        self.NCcopyentry = tk.Entry(self.parent, width=8, bg="white")
        self.NCcopyentry.grid(row=9, column=6, sticky="W")
        self.NCcopyentry.insert(0, self.copydbnumber)
        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.NCcopysubmit = tk.Button(self.parent, text = "Copy", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.NCcopyroughnorm, relief='raised', bd = 5)
        self.NCcopysubmit.grid(row = 9, column = 7)


        img1 = tk.PhotoImage(file=os.path.join(basedir, 'smily.gif'))
        controller.img1d = img1  # need to keep a copy so it is not cleared from memory
        self.window1 = tk.Canvas(master = self.parent, width=img1.width(), height=img1.height(), bg='black')
        self.window1.grid(row=7, column=0,rowspan = 2, columnspan = 2, sticky='NW')
        self.windowdisplay1 = self.window1.create_image(0, 0, image=img1, anchor=tk.NW)

        img2 = tk.PhotoImage(file=os.path.join(basedir, 'smily.gif'))
        controller.img2d = img2  # need to keep a copy so it is not cleared from memory
        self.window2 = tk.Canvas(master = self.parent, width=img2.width(), height=img2.height(), bg='black')
        self.window2.grid(row=7, column=2,rowspan = 2, columnspan = 2, sticky='NW')
        self.windowdisplay2 = self.window2.create_image(0, 0, image=img2, anchor=tk.NW)


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
                            fillcolor = 'yellow'
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
                            fillcolor = 'yellow'
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

        # fullpath, filename = os.path.split(niiname)
        # tag = '_s' + str(seriesnumber)
        # normdataname_full = os.path.join(fullpath, normdatasavename + tag + '.npy')

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


        # input_datar = i3d.load_and_scale_nifti(niiname)
        # if np.ndim(input_datar) > 3:
        #     x, y, z, t = np.shape(input_datar)
        #     if t > 3:
        #         t0 = 3
        #     else:
        #         t0 = 0
        #     input_image = input_datar[:, :, :, t0]
        # else:
        #     x, y, z = np.shape(input_datar)
        #     input_image = input_datar
        #
        # background2 = input_image
        # xs,ys,zs = np.shape(input_image)
        # xmid = np.round(xs/2).astype(int)
        # img = input_image[xmid,:,:]
        # img = (255.*img/np.max(img)).astype(int)
        # image_tk = ImageTk.PhotoImage(Image.fromarray(img))
        # self.controller.img1d = image_tk  # keep a copy so it persists
        # imagerecord = []
        # imagerecord.append({'img':img})
        #
        # resolution = 1
        # template_img, regionmap_img, template_affine, anatlabels, wmmap_img, roi_map, gmwm_img = load_templates.load_template_and_masks(
        #     normtemplatename, resolution)
        # normdata['imagerecord'] = imagerecord

        print('saving normdata to {}'.format(normdataname_full))
        np.save(normdataname_full, normdata)   # overwrite the norm data with the copy

        # display results-----------------------------------------------------
        nfordisplay = len(imagerecord)
        for nf in range(nfordisplay):
            img1 = imagerecord[nf]['img']
            img1 = (255. * img1 / np.max(img1)).astype(int)
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
        input_data = i3d.load_and_scale_nifti(niiname)
        print('shape of input_data is ',np.shape(input_data))
        print('niiname = ', niiname)
        if np.ndim(input_data) == 4:
            xs,ys,zs,ts = np.shape(input_data)
            xmid = np.round(xs/2).astype(int)
            img = input_data[xmid,:,:,0]
            img = (255.*img/np.max(img)).astype(int)
            image_tk = ImageTk.PhotoImage(Image.fromarray(img))
        else:
            xs,ys,zs = np.shape(input_data)
            xmid = np.round(xs/2).astype(int)
            img = input_data[xmid,:,:]
            img = (255.*img/np.max(img)).astype(int)
            image_tk = ImageTk.PhotoImage(Image.fromarray(img))
        self.controller.img1d = image_tk  # keep a copy so it persists
        self.window1.configure(width=image_tk.width(), height=image_tk.height())
        self.windowdisplay1 = self.window1.create_image(0, 0, image=image_tk, anchor=tk.NW)

        inprogressfile = os.path.join(basedir, 'underconstruction.gif')
        image_tk = tk.Image('photo', file=inprogressfile)
        self.controller.img2d = image_tk  # keep a copy so it persists
        self.window2.configure(width=image_tk.width(), height=image_tk.height())
        self.windowdisplay2 = self.window2.create_image(0, 0, image=image_tk, anchor=tk.NW)

        time.sleep(0.5)
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
                    img1 = (255. * img1 / np.max(img1)).astype(int)
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
                display_image = (255. * display_image / np.max(display_image)).astype(int)
                image_tk = ImageTk.PhotoImage(Image.fromarray(display_image))
                # show normalization result instead
                xs,ys,zs = np.shape(reverse_map_image)
                xmid = np.round(xs/2).astype(int)
                display_image = reverse_map_image[xmid,:,:]
                display_image = (255. * display_image / np.max(display_image)).astype(int)
                image_tk = ImageTk.PhotoImage(Image.fromarray(display_image))

                self.controller.img2d = image_tk   # keep a copy so it persists
                self.window2.configure(width=image_tk.width(), height=image_tk.height())
                self.windowdisplay2 = self.window2.create_image(0, 0, image=image_tk, anchor=tk.NW)

                self.window2.create_text(np.round(image_tk.width()/2),image_tk.height()-5,text = 'normalization result', fill = 'white')
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
                img2 = (255. * img2 / np.max(img2)).astype(int)
                image_tk = ImageTk.PhotoImage(Image.fromarray(img2))
                self.controller.img2d = image_tk   # keep a copy so it persists
                self.window2.configure(width=image_tk.width(), height=image_tk.height())
                self.windowdisplay2 = self.window2.create_image(0, 0, image=image_tk, anchor=tk.NW)

                image_tk = self.controller.img1d
                self.window1.configure(width=image_tk.width(), height=image_tk.height())
                self.windowdisplay1 = self.window1.create_image(0, 0, image=image_tk, anchor=tk.NW)
                time.sleep(1)

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
                image_tk = tk.Image('photo', file=inprogressfile)
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
            img2 = (255. * img2 / np.max(img2)).astype(int)
            image_tk = ImageTk.PhotoImage(Image.fromarray(img2))
            self.controller.img2d = image_tk  # keep a copy so it persists
            self.window2.configure(width=image_tk.width(), height=image_tk.height())
            self.windowdisplay2 = self.window2.create_image(0, 0, image=image_tk, anchor=tk.NW)

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

        input_data = i3d.load_and_scale_nifti(niiname)
        print('shape of input_data is ',np.shape(input_data))
        print('niiname = ', niiname)
        if np.ndim(input_data) == 4:
            xs,ys,zs,ts = np.shape(input_data)
            xmid = np.round(xs/2).astype(int)
            img = input_data[xmid,:,:,0]
            img = (255.*img/np.max(img)).astype(int)
            image_tk = ImageTk.PhotoImage(Image.fromarray(img))
            input_image = input_data[:,:,:,3]
        else:
            xs,ys,zs = np.shape(input_data)
            xmid = np.round(xs/2).astype(int)
            img = input_data[xmid,:,:]
            img = (255.*img/np.max(img)).astype(int)
            image_tk = ImageTk.PhotoImage(Image.fromarray(img))
            input_image = input_data
        self.controller.img1d = image_tk  # keep a copy so it persists
        self.window1.configure(width=image_tk.width(), height=image_tk.height())
        self.windowdisplay1 = self.window1.create_image(0, 0, image=image_tk, anchor=tk.NW)
        time.sleep(0.5)
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
            img1 = (255. * img1 / np.max(img1)).astype(int)
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
        display_image = (255. * display_image / np.max(display_image)).astype(int)
        image_tk = ImageTk.PhotoImage(Image.fromarray(display_image))
        # show normalization result instead
        xs, ys, zs = np.shape(reverse_map_image)
        xmid = np.round(xs / 2).astype(int)
        display_image = reverse_map_image[xmid, :, :]
        display_image = (255. * display_image / np.max(display_image)).astype(int)
        image_tk = ImageTk.PhotoImage(Image.fromarray(display_image))

        self.controller.img2d = image_tk  # keep a copy so it persists
        self.window2.configure(width=image_tk.width(), height=image_tk.height())
        self.windowdisplay2 = self.window2.create_image(0, 0, image=image_tk, anchor=tk.NW)

        self.window2.create_text(np.round(image_tk.width() / 2), image_tk.height() - 5, text='normalization result',
                                 fill='white')

        
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
        self.PPinfo1 = tk.Label(self.parent, text = "Slice timing information:")
        self.PPinfo1.grid(row=srow,column=scol, columnspan = 4, sticky='NSEW')

        orderoptions = {'Inc,Alt,Odd','Inc,Alt,Even','Dec,Alt,N','Dec,Alt,N-1','Inc,Seq.','Dec,Seq.'}

        self.PPslicelabel1 = tk.Label(self.parent, text="Slice Order:").grid(row=1, column=1, sticky='E')
        self.sliceorder_var = tk.StringVar()
        self.sliceorder_var.set(self.sliceorder)
        self.sliceorder_opt = tk.OptionMenu(self.parent, self.sliceorder_var, *orderoptions, command = self.PPsliceorderchoice).grid(
            row=srow+1, column=scol+1, sticky='EW')
        # tk.CreateToolTip(self.sliceorder_opt, text='Inc,Alt,Odd = 1,3,5...2,4,6...\n'
        #                                            'Inc,Alt,Even = 2,4,6...1,3,5...\n'
        #                                            'Dec,Alt,N = N,N-2,...N-1,N-3,...\n'
        #                                            'Dec,Alt,N-1 = N-1,N-3,...N,N-2,...\n'
        #                                            'Inc,Seq. = 1,2,3,...\n'
        #                                            'Dec,Seq. = N,N-1,N-2...\n')

        self.PPslicelabel2 = tk.Label(self.parent, text = "Ref. Slice:")
        self.PPslicelabel2.grid(row=srow+1, column=scol+2, sticky='E')
        # create entry box, and put it next to the label
        self.PPrefslice = tk.Entry(self.parent, width = 8, bg="white")
        self.PPrefslice.grid(row=srow+1, column=scol+3, sticky = "W")
        self.PPrefslice.insert(0,self.refslice)

        self.PPslicelabel3 = tk.Label(self.parent, text = "Slice Axis:")
        self.PPslicelabel3.grid(row=srow+1, column=scol+4, sticky='E')
        # create entry box, and put it next to the label
        self.PPsliceaxis = tk.Entry(self.parent, width = 8, bg="white")
        self.PPsliceaxis.grid(row=srow+1, column=scol+5, sticky = "W")
        self.PPsliceaxis.insert(0,self.sliceaxis)

        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.PPslicesubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.PPslicesubmit, relief='raised', bd = 5)
        self.PPslicesubmit.grid(row=srow+1, column=scol+6, columnspan = 4)
        ttk.Separator(self.parent).grid(row=srow+2, column=scol, columnspan=6, sticky="nswe", padx=2, pady=5)


        # now define entry boxes so that the user can indicate the smoothing parameters
        # give the group of boxes a label first
        srow = 3; scol = 1;
        self.PPinfo2 = tk.Label(self.parent, text = "Smoothing Width:")
        self.PPinfo2.grid(row=srow, column=scol+1, sticky='NSEW')
        self.PPsmoothing = tk.Entry(self.parent, width = 8, bg="white")
        self.PPsmoothing.grid(row=srow, column=scol+2, sticky = "W")
        self.PPsmoothing.insert(0,self.smoothwidth)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.PPsmoothsubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.PPsmoothsubmit, relief='raised', bd = 5)
        self.PPsmoothsubmit.grid(row=srow, column=scol+3)
        ttk.Separator(self.parent).grid(row=srow+1, column=scol, columnspan=6, sticky="nswe", padx=2, pady=5)

        # setup the choices for which pre-processing steps to run
        # create pull-down lists
        options = {'Yes.', '.No.', 'Done'}
        srow = 5;  scol = 1;   # positions to begin this set of checkboxes (to make it easy to move later)
        # coregistration
        self.coreg_label = tk.Label(self.parent, text="Coregister:").grid(row=srow, column=scol, sticky='E')
        self.coreg_var = tk.StringVar()
        self.coreg_var.set(self.coreg_choice)
        self.coreg_opt = tk.OptionMenu(self.parent, self.coreg_var, *options, command = self.PPcoregchoice).grid(
            row=srow, column=scol+1, sticky='EW')
        # slice-time correction
        self.slicetime_label = tk.Label(self.parent, text="Slice-timing:").grid(row=srow, column=scol+2, sticky='E')
        self.slicetime_var = tk.StringVar()
        self.slicetime_var.set(self.slicetime_choice)
        self.slicetime_opt = tk.OptionMenu(self.parent, self.slicetime_var, *options, command = self.PPslicetimechoice).grid(
            row=srow, column=scol+3, sticky='EW')
        # apply normalization
        self.norm_label = tk.Label(self.parent, text="Normalize:").grid(row=srow, column=scol+4, sticky='E')
        self.norm_var = tk.StringVar()
        self.norm_var.set(self.norm_choice)
        self.norm_opt = tk.OptionMenu(self.parent, self.norm_var, *options, command = self.PPnormchoice).grid(
            row=srow, column=scol+5, sticky='EW')
        # apply smoothing
        self.smooth_label = tk.Label(self.parent, text="Smooth:").grid(row=srow+1, column=scol, sticky='E')
        self.smooth_var = tk.StringVar()
        self.smooth_var.set(self.smooth_choice)
        self.norm_opt = tk.OptionMenu(self.parent, self.smooth_var, *options, command=self.PPsmoothchoice).grid(
            row=srow+1, column=scol + 1, sticky='EW')
        # define basis sets
        self.define_label = tk.Label(self.parent, text="Define basis sets:").grid(row=srow+1, column=scol+2, sticky='E')
        self.define_var = tk.StringVar()
        self.define_var.set(self.define_choice)
        self.define_opt = tk.OptionMenu(self.parent, self.define_var, *options, command=self.PPdefinechoice).grid(
            row=srow+1, column=scol+3, sticky='EW')
        # clean the data
        self.clean_label = tk.Label(self.parent, text="Clean data:").grid(row=srow+1, column=scol+4, sticky='E')
        self.clean_var = tk.StringVar()
        self.clean_var.set(self.clean_choice)
        self.clean_opt = tk.OptionMenu(self.parent, self.clean_var, *options, command=self.PPcleanchoice).grid(
            row=srow+1, column=scol+5, sticky='EW')

        # button to run the selected-preprocessing
        self.PPrun = tk.Button(self.parent, text = 'Process Data', width = bigbuttonsize, bg = fgcol1, fg = 'white', command = self.PPrunclick, font = "none 9 bold", relief='raised', bd = 5)
        self.PPrun.grid(row = srow+2, column = 1, columnspan = 2)

        # text to show the resulting prefix for the data that will be created
        self.predictedprefixtext = tk.StringVar()
        prefix_list = pypreprocess.setprefixlist(settingsfile)
        self.predictedprefixtext.set(prefix_list[-1])
        self.PPprefixdisplaytitle = tk.Label(self.parent, text = "output nifti prefix: ", fg = 'gray')
        self.PPprefixdisplaytitle.grid(row=srow+2, column=3, sticky='E')
        self.PPprefixdisplay = tk.Label(self.parent, textvariable = self.predictedprefixtext, fg = 'gray')
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

        self.GLMoptionlabel1 = tk.Label(self.parent, text="GLM Method:").grid(row=0, column=1, sticky='E')
        self.GLMoption_var = tk.StringVar()
        self.GLMoption_var.set(self.GLM1_option)
        self.GLMoption_opt = tk.OptionMenu(self.parent, self.GLMoption_var, *GLMoptions, command = self.GLMoptionchoice).grid(
            row=0, column=2, sticky='EW')

        # set the data prefix for analysis
        self.GLMlabel2 = tk.Label(self.parent, text = 'Data prefix:').grid(row=1, column=1, sticky='NSEW')
        self.GLMprefixbox = tk.Entry(self.parent, width = 8, bg="white")
        self.GLMprefixbox.grid(row=1, column=2, sticky="W")
        self.GLMprefixbox.insert(0,self.GLMprefix)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.GLMprefixsubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.GLMprefixsubmit, relief='raised', bd = 5)
        self.GLMprefixsubmit.grid(row=1, column=3)


        # set the number of initial volumes to drop, or mask, for each data set
        self.GLMlabel3 = tk.Label(self.parent, text = 'Initial volumes to drop/mask:').grid(row=2, column=1, sticky='NSEW')
        self.GLMndropbox = tk.Entry(self.parent, width = 8, bg="white")
        self.GLMndropbox.grid(row=2, column=2, sticky = "W")
        self.GLMndropbox.insert(0,self.GLMndrop)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.GLMndropsubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.GLMndropsubmit, relief='raised', bd = 5)
        self.GLMndropsubmit.grid(row=2, column=3)

        # indicate the contrast to use for analysis
        self.GLMlabel4 = tk.Label(self.parent, text = 'Contrast between basis elements:').grid(row=3, column=1, sticky='NSEW')
        self.GLMcontrastbox = tk.Entry(self.parent, width = 8, bg="white")
        self.GLMcontrastbox.grid(row=3, column=2, sticky = "W")
        self.GLMcontrastbox.insert(0,self.GLMcontrast)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.GLMcontrastsubmitbut = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.GLMcontrastsubmit, relief='raised', bd = 5)
        self.GLMcontrastsubmitbut.grid(row=3, column=3)

        # put in choices for statistical threshold
        self.GLMlabel5 = tk.Label(self.parent, text = 'p-value threhold:').grid(row=4, column=1, sticky='NSEW')
        self.GLMpvaluebox = tk.Entry(self.parent, width = 8, bg="white")
        self.GLMpvaluebox.grid(row=4, column=2, sticky = "W")
        self.GLMpvaluebox.insert(0,self.GLMpvalue)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.GLMpvaluesubmitbut = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.GLMpvaluesubmit, relief='raised', bd = 5, state = tk.DISABLED)
        self.GLMpvaluesubmitbut.grid(row=4, column=3)

        # put in choices for voxel volume
        self.GLMlabel6 = tk.Label(self.parent, text = 'voxel volume (mm3):').grid(row=5, column=1, sticky='NSEW')
        self.GLMvvolumebox = tk.Entry(self.parent, width = 8, bg="white")
        self.GLMvvolumebox.grid(row=5, column=2, sticky = "W")
        self.GLMvvolumebox.insert(0,self.GLMvoxvolume)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.GLMvvolumesubmitbut = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.GLMvvolumesubmit, relief='raised', bd = 5)
        self.GLMvvolumesubmitbut.grid(row=5, column=3)


        # radioboxes to indicate stats correction choices
        self.GLML7 = tk.Label(self.parent, text="Correction for multiple comparisons:")
        self.GLML7.grid(row=6, column=1, sticky='W')
        self.GLMpcorrectionmethod = tk.IntVar(None,1)
        self.GLMcorr1 = tk.Radiobutton(self.parent, text = 'no correction', width = smallbuttonsize, fg = 'black',
                                          command = self.GLMsetcorrtype, variable = self.GLMpcorrectionmethod, value = 1, state = tk.DISABLED)
        self.GLMcorr1.grid(row = 6, column = 2, sticky="E")
        self.GLMcorr2 = tk.Radiobutton(self.parent, text = 'GRF', width = smallbuttonsize, fg = 'black',
                                          command = self.GLMsetcorrtype, variable = self.GLMpcorrectionmethod, value = 2, state = tk.DISABLED)
        self.GLMcorr2.grid(row = 6, column = 3, sticky="E")
        self.GLMcorr3 = tk.Radiobutton(self.parent, text = 'Bonferroni', width = smallbuttonsize, fg = 'black',
                                          command = self.GLMsetcorrtype, variable = self.GLMpcorrectionmethod, value = 3, state = tk.DISABLED)
        self.GLMcorr3.grid(row = 6, column = 4, sticky="E")


        ttk.Separator(self.parent).grid(row=7, column=1, columnspan=6, sticky="nswe", padx=2, pady=5)

        # entry box for specifying folder for saving results
        # create an entry box so that the user can specify the results folder to use
        self.GLML2 = tk.Label(self.parent, text="Results folder:")
        self.GLML2.grid(row=8, column=1, sticky='W')

        # make a label to show the current setting of the results folder name
        self.GLMresultstext = tk.StringVar()
        self.GLMresultstext.set(self.GLMresultsdir)
        self.GLMresultsinfo = tk.Label(self.parent, textvariable=self.GLMresultstext, bg=bgcol, fg="black", font="none 10",
                                     wraplength=150, justify='left')
        self.GLMresultsinfo.grid(row=8, column=2, sticky='W')

        # define a button to browse and select a location for saving the GLM results, and write out the selected name
        # also, define the function for what to do when this button is pressed
        self.GLMfolderbrowse = tk.Button(self.parent, text='Browse', width=smallbuttonsize, bg=fgcol2, fg='black',
                                  command=self.GLMresultsbrowseclick, relief='raised', bd=5)
        self.GLMfolderbrowse.grid(row=8, column=3)

        # label, button, and information box for compiling the basis sets
        self.GLMbasisbutton = tk.Button(self.parent, text="Compile Basis Sets", width=bigbigbuttonsize, bg=fgcol1, fg='white',
                                        command=self.GLMbasissetup, relief='raised', bd=5)
        self.GLMbasisbutton.grid(row=9, column=1)
        self.GLMbasisbutton2 = tk.Button(self.parent, text="Load Basis Sets", width=bigbigbuttonsize, bg=fgcol2, fg='black',
                                        command=self.GLMbasisload, relief='raised', bd=5)
        self.GLMbasisbutton2.grid(row=9, column=2)
        self.GLMbasisinfo= tk.Label(self.parent, text='basis set information ...')
        self.GLMbasisinfo.grid(row=9, column=3, columnspan = 4, sticky='E')


        # label, button, and information box for compiling the data sets
        self.GLMdatabutton = tk.Button(self.parent, text="Compile Data Sets", width=bigbigbuttonsize, bg=fgcol1, fg='white',
                                        command=self.GLMdatasetup, relief='raised', bd=5)
        self.GLMdatabutton.grid(row=10, column=1)
        self.GLMdatabutton2 = tk.Button(self.parent, text="Load Data Sets", width=bigbigbuttonsize, bg=fgcol2, fg='black',
                                        command=self.GLMdataload, relief='raised', bd=5)
        self.GLMdatabutton2.grid(row=10, column=2)
        self.GLMdatainfo = tk.Label(self.parent, text='data set information ...')
        self.GLMdatainfo.grid(row=10, column=3, columnspan = 4, sticky='E')

        # setup input for contrast
        # label, button, and information box for running the GLM analysis
        self.GLMrunbutton = tk.Button(self.parent, text="Run GLM", width=bigbigbuttonsize, bg=fgcol1, fg='white',
                                        command=self.GLMrun1, relief='raised', bd=5)
        self.GLMrunbutton.grid(row=11, column=1)
        self.GLMruninfo= tk.Label(self.parent, text='GLM results information ...')
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

        basisset, paradigm_names = GLMfit.compile_basis_sets(self.DBname, self.dbnumlist, self.prefix, mode=self.GLM1_option, nvolmask=ndrop)
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

        dataset = GLMfit.compile_data_sets(self.DBname, self.DBnum, self.prefix, mode=self.GLM1_option, nvolmask=ndrop)
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

        # dataset = self.dataset
        # basisset = self.basisset

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
            xs, ys, zs, ts, NP = np.shape(dataset)
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

        # now do something with B, sem, and T
        # prompt for a filename for saving the data
        results = {'B':B,'sem':sem,'T':T}
        filechoice = tkf.asksaveasfilename(title="Save results as:",
                        filetypes=(("npy files", "*.npy"), ("all files", "*.*")))
        np.save(filechoice,results)

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
        # xls = pd.ExcelFile(DBname, engine = 'openpyxl')
        # df1 = pd.read_excel(xls, 'datarecord')
        # normtemplatename = df1.loc[DBnum[0], 'normtemplatename']
        # resolution = 1
        # template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_map = load_templates.load_template_and_masks(normtemplatename, resolution)
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
        pname,fname = os.path.split(filechoice)
        fname,ext = os.path.splitext(fname)
        outputname = os.path.join(pname,fname+'.png')

        print('size of output image: ',outputimg.shape)
        print('image output name: ',outputname)

        # im = Image.fromarray(outputimg)
        # im.save(outputname)

        # scipy.misc.imsave(outputname, outputimg)   # alternative way of writing image to file

        matplotlib.image.imsave(outputname, outputimg)

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

        cluster_properties, region_properties = \
            pyclustering.define_clusters_and_load_data(self.DBname, self.DBnum, self.CLprefix, self.networkmodel, regionmap_img, anatlabels)

        cluster_definition = {'cluster_properties':cluster_properties}
        region_data = {'region_properties':region_properties, 'DBname':self.DBname, 'DBnum':self.DBnum}

        # save the results
        np.save(self.CLclustername,cluster_definition)
        np.save(self.CLregionname,region_data)
        messagetext = 'defining clusters and loading data \ncompleted: ' + time.ctime(time.time())
        self.CLdefinebuttontext.set(messagetext)
        print(messagetext)


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

    # initialize the values, keeping track of the frame this definition works on (parent), and
    # also the main window containing that frame (controller)
    def __init__(self, parent, controller):
        parent.configure(relief='raised', bd=5, highlightcolor=fgcol3)
        self.parent = parent
        self.controller = controller
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
        self.CLL1 = tk.Label(self.parent, text="Network Model:")
        self.CLL1.grid(row=0, column=1, sticky='SW')

        # make a label to show the current setting of the network definition file name
        npname, nfname = os.path.split(self.networkmodel)
        self.CLnetnametext = tk.StringVar()
        self.CLnetnametext.set(nfname)
        self.CLfnamelabel = tk.Label(self.parent, textvariable=self.CLnetnametext, bg=bgcol, fg="#4B4B4B", font="none 10",
                                     wraplength=300, justify='left')
        self.CLfnamelabel.grid(row=0, column=2, sticky='S')

        # make a label to show the current setting of the network definition file directory name
        self.CLnetdirtext = tk.StringVar()
        self.CLnetdirtext.set(npname)
        self.CLdnamelabel = tk.Label(self.parent, textvariable=self.CLnetdirtext, bg=bgcol, fg="#4B4B4B", font="none 8",
                                     wraplength=300, justify='left')
        self.CLdnamelabel.grid(row=1, column=2, sticky='N')

        # define a button to browse and select an existing network definition file, and write out the selected name
        # also, define the function for what to do when this button is pressed
        self.CLnetworkbrowse = tk.Button(self.parent, text='Browse', width=smallbuttonsize, bg = fgcol2, fg = 'black',
                                  command=self.CLnetbrowseclick, relief='raised', bd=5)
        self.CLnetworkbrowse.grid(row=0, column=3)

        # create entry box for the nifti data name prefix (indicates which preprocessing steps were done)
        self.CLpreflabel = tk.Label(self.parent, text = 'Data name prefix:')
        self.CLpreflabel.grid(row=2, column=1, sticky='N')
        self.CLprefixbox = tk.Entry(self.parent, width = 8, bg="white")
        self.CLprefixbox.grid(row=2, column=2, sticky='N')
        self.CLprefixbox.insert(0,self.CLprefix)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.CLprefixsubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.CLprefixsubmitaction, relief='raised', bd = 5)
        self.CLprefixsubmit.grid(row=2, column=3, sticky='N')

        # need an input for the cluster definition name - save to it, or read from it
        self.CLclusternamelabel = tk.Label(self.parent, text = 'Cluster definition name:')
        self.CLclusternamelabel.grid(row=3, column=1, sticky='N')
        self.CLclusternamebox = tk.Entry(self.parent, width = 30, bg="white",justify = 'right')
        self.CLclusternamebox.grid(row=3, column=2, sticky='N')
        self.CLclusternamebox.insert(0,self.CLclustername)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.CLclusternamesubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.CLclusternamesubmitaction, relief='raised', bd = 5)
        self.CLclusternamesubmit.grid(row=3, column=3, sticky='N')
        # the entry boxes need a "browse" button to allow selection of existing cluster definition file
        self.CLclusternamebrowse = tk.Button(self.parent, text = "Browse", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.CLclusternamebrowseaction, relief='raised', bd = 5)
        self.CLclusternamebrowse.grid(row=3, column=4, sticky='N')

        # box etc for entering the name for saving the region data
        self.CLregionnamelabel = tk.Label(self.parent, text = 'Region/cluster data name:')
        self.CLregionnamelabel.grid(row=4, column=1, sticky='N')
        self.CLregionnamebox = tk.Entry(self.parent, width = 30, bg="white",justify = 'right')
        self.CLregionnamebox.grid(row=4, column=2, sticky='N')
        self.CLregionnamebox.insert(0,self.CLregionname)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.CLregionnamesubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.CLregionnamesubmitaction, relief='raised', bd = 5)
        self.CLregionnamesubmit.grid(row=4, column=3, sticky='N')
        # the entry boxes need a "browse" button to allow selection of existing cluster definition file
        self.CLregionnamebrowse = tk.Button(self.parent, text = "Browse", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.CLregionnamebrowseaction, relief='raised', bd = 5)
        self.CLregionnamebrowse.grid(row=4, column=4, sticky='N')


        # label, button, for running the definition of clusters, and loading data
        self.CLdefineandloadbutton = tk.Button(self.parent, text="Define Clusters", width=bigbigbuttonsize, bg=fgcol1, fg='white',
                                        command=self.CLdefineandload, relief='raised', bd=5)
        self.CLdefineandloadbutton.grid(row=5, column=2)

        self.CLdefinebuttontext = tk.StringVar()
        self.CLdefinebuttonlabel = tk.Label(self.parent, textvariable=self.CLdefinebuttontext, bg=bgcol, fg="#4B4B4B", font="none 10",
                                     wraplength=200, justify='left')
        self.CLdefinebuttonlabel.grid(row=5, column=3, sticky='N')

        # label, button, for running the definition of clusters, and loading data
        self.CLloadbutton = tk.Button(self.parent, text="Load Data", width=bigbigbuttonsize, bg=fgcol1, fg='white',
                                        command=self.CLload, relief='raised', bd=5)
        self.CLloadbutton.grid(row=6, column=2)

        self.CLloadbuttontext = tk.StringVar()
        self.CLloadbuttonlabel = tk.Label(self.parent, textvariable=self.CLloadbuttontext, bg=bgcol, fg="#4B4B4B", font="none 10",
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

    def SEMprefixsubmitaction(self):
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        SEMprefix = self.SEMprefixbox.get()
        settings['SEMprefix'] = SEMprefix
        np.save(settingsfile,settings)
        # update the text in the box, in case it has changed
        self.SEMprefix = SEMprefix
        print('prefix for SEM analysis set to ',self.SEMprefix)
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


    def SEMtimesubmitclick(self):
        # first load the settings file so that values can be used later
        settings = np.load(settingsfile, allow_pickle = True).flat[0]
        SEMtimepoints = settings['SEMtimepoints']

        entered_text = self.SEMtimeenter.get()  # collect the text from the text entry box
        # first, replace any double spaces with single spaces, and then replace spaces with commas
        entered_text = re.sub('\ +', ',', entered_text)
        entered_text = re.sub('\,\,+', ',', entered_text)
        SEMtimepoints = np.fromstring(entered_text, dtype=np.int, sep=',')

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

        # save the results somehow
        results = {'type':'2source','CCrecord':CCrecord, 'beta2':beta2, 'beta1':beta1, 'Zgrid2':Zgrid2, 'Zgrid1_1':Zgrid1_1,'Zgrid1_2':Zgrid1_2, 'DBname':self.DBname, 'DBnum':self.DBnum, 'cluster_properties':cluster_properties}
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

        # save the results somehow
        results = {'type':'network','resultsnames':outputnamelist, 'network':self.networkmodel, 'regionname':self.SEMregionname, 'clustername':self.SEMclustername, 'DBname':self.DBname, 'DBnum':self.DBnum}
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


    # initialize the values, keeping track of the frame this definition works on (parent), and
    # also the main window containing that frame (controller)
    def __init__(self, parent, controller):
        parent.configure(relief='raised', bd=5, highlightcolor=fgcol3)
        self.parent = parent
        self.controller = controller
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
        self.SEML1 = tk.Label(self.parent, text="Network Model:")
        self.SEML1.grid(row=0, column=1, sticky='SW')

        # make a label to show the current setting of the network definition file name
        npname, nfname = os.path.split(self.networkmodel)
        self.SEMnetnametext = tk.StringVar()
        self.SEMnetnametext.set(nfname)
        self.SEMfnamelabel = tk.Label(self.parent, textvariable=self.SEMnetnametext, bg=bgcol, fg="#4B4B4B", font="none 10",
                                     wraplength=300, justify='left')
        self.SEMfnamelabel.grid(row=0, column=2, sticky='S')

        # make a label to show the current setting of the network definition file directory name
        self.SEMnetdirtext = tk.StringVar()
        self.SEMnetdirtext.set(npname)
        self.SEMdnamelabel = tk.Label(self.parent, textvariable=self.SEMnetdirtext, bg=bgcol, fg="#4B4B4B", font="none 8",
                                     wraplength=300, justify='left')
        self.SEMdnamelabel.grid(row=1, column=2, sticky='N')

        # define a button to browse and select an existing network definition file, and write out the selected name
        # also, define the function for what to do when this button is pressed
        self.SEMnetworkbrowse = tk.Button(self.parent, text='Browse', width=smallbuttonsize, bg = fgcol2, fg = 'black',
                                  command=self.SEMnetbrowseclick, relief='raised', bd=5)
        self.SEMnetworkbrowse.grid(row=0, column=3)

        # create entry box for the nifti data name prefix (indicates which preprocessing steps were done)
        self.SEMpreflabel = tk.Label(self.parent, text = 'Data name prefix:')
        self.SEMpreflabel.grid(row=2, column=1, sticky='N')
        self.SEMprefixbox = tk.Entry(self.parent, width = 8, bg="white")
        self.SEMprefixbox.grid(row=2, column=2, sticky='N')
        self.SEMprefixbox.insert(0,self.SEMprefix)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.SEMprefixsubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.SEMprefixsubmitaction, relief='raised', bd = 5)
        self.SEMprefixsubmit.grid(row=2, column=3, sticky='N')

        # need an input for the cluster definition name - save to it, or read from it
        self.SEMclusternamelabel = tk.Label(self.parent, text = 'Cluster definition name:')
        self.SEMclusternamelabel.grid(row=3, column=1, sticky='N')
        self.SEMclusternamebox = tk.Entry(self.parent, width = 30, bg="white",justify = 'right')
        self.SEMclusternamebox.grid(row=3, column=2, sticky='N')
        self.SEMclusternamebox.insert(0,self.SEMclustername)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.SEMclusternamesubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.SEMclusternamesubmitaction, relief='raised', bd = 5)
        self.SEMclusternamesubmit.grid(row=3, column=3, sticky='N')
        # the entry boxes need a "browse" button to allow selection of existing cluster definition file
        self.SEMclusternamebrowse = tk.Button(self.parent, text = "Browse", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.SEMclusternamebrowseaction, relief='raised', bd = 5)
        self.SEMclusternamebrowse.grid(row=3, column=4, sticky='N')

        # box etc for entering the name for saving the region data
        self.SEMregionnamelabel = tk.Label(self.parent, text = 'Region/cluster data name:')
        self.SEMregionnamelabel.grid(row=4, column=1, sticky='N')
        self.SEMregionnamebox = tk.Entry(self.parent, width = 30, bg="white",justify = 'right')
        self.SEMregionnamebox.grid(row=4, column=2, sticky='N')
        self.SEMregionnamebox.insert(0,self.SEMregionname)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.SEMregionnamesubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.SEMregionnamesubmitaction, relief='raised', bd = 5)
        self.SEMregionnamesubmit.grid(row=4, column=3, sticky='N')
        # the entry boxes need a "browse" button to allow selection of existing cluster definition file
        self.SEMregionnamebrowse = tk.Button(self.parent, text = "Browse", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.SEMregionnamebrowseaction, relief='raised', bd = 5)
        self.SEMregionnamebrowse.grid(row=4, column=4, sticky='N')


        # create the SEM timepoints entry box
        self.SEMtimelabel = tk.Label(self.parent, text = 'Epoch center times:')
        self.SEMtimelabel.grid(row=5, column=1, sticky='N')
        self.SEMtimeenter = tk.Entry(self.parent, width=20, bg="white")
        self.SEMtimeenter.grid(row=5, column=2, sticky="W")
        self.SEMtimeenter.insert(0, self.SEMtimetext)
        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.SEMtimesubmit = tk.Button(self.parent, text="Submit", width=smallbuttonsize, bg = fgcol2, fg = 'black',
                                     command=self.SEMtimesubmitclick, relief='raised', bd=5)
        self.SEMtimesubmit.grid(row=5, column=3)


        # create the SEM epoch entry box
        self.SEMepochlabel = tk.Label(self.parent, text = 'Epoch length:')
        self.SEMepochlabel.grid(row=6, column=1, sticky='N')
        self.SEMepochenter = tk.Entry(self.parent, width=20, bg="white")
        self.SEMepochenter.grid(row=6, column=2, sticky="W")
        self.SEMepochenter.insert(0, self.SEMepoch)
        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.SEMepochsubmit = tk.Button(self.parent, text="Submit", width=smallbuttonsize, bg = fgcol2, fg = 'black',
                                     command=self.SEMepochsubmitclick, relief='raised', bd=5)
        self.SEMepochsubmit.grid(row=6, column=3)


        # box etc for entering the name of the directory for saving the results
        # make a label to show the current setting of the network definition file directory name
        self.SEMresultsdirlabel = tk.Label(self.parent, text = 'Results save folder:')
        self.SEMresultsdirlabel.grid(row=7, column=1, sticky='N')
        self.SEMresultsdirtext = tk.StringVar()
        self.SEMresultsdirtext.set(self.SEMresultsdir)
        self.SEMresultsdirdisplay = tk.Label(self.parent, textvariable=self.SEMresultsdirtext, bg=bgcol, fg="#4B4B4B", font="none 10",
                                     wraplength=300, justify='left')
        self.SEMresultsdirdisplay.grid(row=7, column=2, sticky='N')
        # the entry boxes need a "browse" button to allow selection of existing cluster definition file
        self.SEMresultsdirbrowse = tk.Button(self.parent, text = "Browse", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.SEMresultsdirbrowseaction, relief='raised', bd = 5)
        self.SEMresultsdirbrowse.grid(row=7, column=3, sticky='N')


        # box etc for entering the name used in labeling the results files
        self.SEMsavetaglabel = tk.Label(self.parent, text = 'tag for results names:')
        self.SEMsavetaglabel.grid(row=8, column=1, sticky='N')
        self.SEMsavetagbox = tk.Entry(self.parent, width = 30, bg="white",justify = 'right')
        self.SEMsavetagbox.grid(row=8, column=2, sticky='N')
        self.SEMsavetagbox.insert(0,self.SEMsavetag)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.SEMsavetagsubmit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.SEMsavetagsubmitaction, relief='raised', bd = 5)
        self.SEMsavetagsubmit.grid(row=8, column=3, sticky='N')


        # label, button, for running the definition of clusters, and loading data
        self.SEMrun2sourcebutton = tk.Button(self.parent, text="2-source SEM", width=bigbigbuttonsize, bg=fgcol1, fg='white',
                                        command=self.SEMtwosource, relief='raised', bd=5)
        self.SEMrun2sourcebutton.grid(row=9, column=2)

        # check to indicate to resume a failed run
        self.var1 = tk.IntVar()
        self.SEMresumebox = tk.Checkbutton(self.parent, text = 'Resume previous', width = bigbigbuttonsize, fg = 'black',
                                          command = self.SEMresumecheck, variable = self.var1)
        self.SEMresumebox.grid(row = 10, column = 1, sticky="E")

        # label, button, for running the definition of clusters, and loading data
        self.SEMrunnetworkbutton = tk.Button(self.parent, text="Network SEM", width=bigbuttonsize, bg=fgcol1, fg='white',
                                        command=self.SEMrunnetwork, relief='raised', bd=5)
        self.SEMrunnetworkbutton.grid(row=10, column=2)




#-----------Group-level analysis FRAME--------------------------------------------------
# Definition of the frame that has inputs for the database name, and entry numbers to use
class GRPFrame:

    def GRPcharacteristicslistclear(self):
        self.GRPcharacteristicscount = 0
        self.GRPcharacteristicslist = []
        self.GRPcharacteristicstext.set('empty')

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
        self.GRPfield_menu.grid(row=7, column=2, sticky='EW')
        self.fieldsearch_opt = self.GRPfield_menu  # save this way so that values are not cleared

        settings = np.load(settingsfile, allow_pickle = True).flat[0]
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
                fieldvalues = np.zeros(NP)
                for nn in range(NP):
                    DBnum_person = dbnum_person_list[nn]
                    fv1 = list(df1.loc[DBnum_person,fieldname])
                    fieldvalues[nn] = np.mean(fv1)
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
                fieldvalues2 = np.zeros(NP2)
                for nn in range(NP2):
                    DBnum_person2 = dbnum_person_list2[nn]
                    fv1 = list(df1.loc[DBnum_person2,fieldname])
                    fieldvalues2[nn] = np.mean(fv1)
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
            if 'type' in keylist: datafiletype = 1
            if 'region_properties' in keylist: datafiletype = 2
        except:
            print('Error reading selected data file - unexpected contents or format')
            return

        if datafiletype == 0:  print('selected data file does not have the required format')
        if datafiletype == 1:  print('found SEM results: for group comparisons be sure to select 2 results files')
        if datafiletype == 2:  print('found time-course data: select correlation as the analysis type in order to do Bayesian regression')

        settings['GRPdatafiletype1'] = datafiletype
        settings['DBname'] = data['DBname']
        settings['DBnum'] = data['DBnum']
        self.DBname = settings['DBname']
        self.DBnum = settings['DBnum']
        np.save(settingsfile, settings)

        # clear the characteristics list
        self.GRPcharacteristicslistclear()

        self.GRPresultsnamebrowse2['state'] = tk.NORMAL
        self.GRPresultsnameclear2['state'] = tk.NORMAL

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
            if 'type' in keylist: datafiletype = 1
            if 'region_properties' in keylist: datafiletype = 2
        except:
            print('Error reading selected data file - unexpected contents or format')
            return

        if datafiletype == 0:  print('selected data file does not have the required format')
        if datafiletype == 1:  print('found SEM results: for group comparisons be sure to select 2 results files')
        if datafiletype == 2:  print('found time-course data: select correlation as the analysis type in order to do Bayesian regression')

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

        print('Group analysis type set to: ',self.GRPanalysistype)

        if value == 2:
            print('Sig2:  be sure to indicate two sets of results to compare')
        if value == 3:
            print('Sig2paired:  be sure to indicate two sets of results to compare')
        if value == 4:
            print('Correlation:  only indicate one set of results (first set will be used)')
        if value == 5:
            print('Regression:  only indicate one set of results (first set will be used)')
        if value == 6:
            print('ANOVA:  select only discrete or categorical values for the personal characteristics')
        if value == 7:
            print('ANCOVA:  select a discrete value first, and a continuous value second, for the personal characteristics')

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
            outputfilename = py2ndlevelanalysis.group_significance(datafile1, pthreshold, statstype='correlation', covariates=covariates)

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
            outputfilename = py2ndlevelanalysis.group_significance(datafile1, pthreshold, statstype='regression', covariates=covariates)

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
                    covariates1 = GRPcharacteristicsvalues[:1,:]
                    covname = GRPcharacteristicslist[:1]
                    outputfilename = py2ndlevelanalysis.group_comparison_ANOVA(datafile1, 'none', covariates1, 'none', pthreshold, mode = 'ANOVA', covariate_name = covname)


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
                    covariates1 = GRPcharacteristicsvalues[:1,:]
                    covname = GRPcharacteristicslist[:1]
                    outputfilename = py2ndlevelanalysis.group_comparison_ANOVA(datafile1, 'none', covariates1, 'none', pthreshold, mode = 'ANOVA', covariate_name = covname)


        if GRPanalysistype == 'ANCOVA':
            if datafiletype1 == 1:  # SEM data
                if datafiletype2 == 1:
                    # apply ANOVA analysis to beta-values, with the two files indicating two discrete groups,
                    # and the first personal characteristic used as a continuous variable
                    pthreshold = GRPpvalue
                    covariates1 = GRPcharacteristicsvalues[0,:]
                    covariates2 = GRPcharacteristicsvalues2[0,:]
                    covname = GRPcharacteristicslist[0]
                    outputfilename = py2ndlevelanalysis.group_comparison_ANOVA(datafile1, datafile2, covariates1, covariates2, pthreshold, mode = 'ANOVA', covariate_name = covname)
                else:
                    # apply ANOVA analysis to beta-values in the first data file named,
                    # with the first two personal characteristics used as one discrete and one continuous variable
                    pthreshold = GRPpvalue
                    covariates1 = GRPcharacteristicsvalues[:1,:]
                    covname = GRPcharacteristicslist[:1]
                    outputfilename = py2ndlevelanalysis.group_comparison_ANOVA(datafile1, 'none', covariates1, 'none', pthreshold, mode = 'ANOVA', covariate_name = covname)


            if datafiletype1 == 2:  # BOLD data
                if datafiletype2 == 2:
                    # apply ANOVA analysis to beta-values, with the two files indicating two discrete groups,
                    # and the first personal characteristic used as a continuous variable
                    pthreshold = GRPpvalue
                    covariates1 = GRPcharacteristicsvalues[0,:]
                    covariates2 = GRPcharacteristicsvalues2[0,:]
                    covname = GRPcharacteristicslist[0]
                    outputfilename = py2ndlevelanalysis.group_comparison_ANOVA(datafile1, datafile2, covariates1, covariates2, pthreshold, mode = 'ANOVA', covariate_name = covname)
                else:
                    # apply ANOVA analysis to beta-values in the first data file named,
                    # with the first two personal characteristics used as one discrete and one continuous variable
                    pthreshold = GRPpvalue
                    covariates1 = GRPcharacteristicsvalues[:1,:]
                    covname = GRPcharacteristicslist[:1]
                    outputfilename = py2ndlevelanalysis.group_comparison_ANOVA(datafile1, 'none', covariates1, 'none', pthreshold, mode = 'ANOVA', covariate_name = covname)



    def __init__(self, parent, controller):
        parent.configure(relief='raised', bd=5, highlightcolor=fgcol3)
        self.parent = parent
        self.controller = controller
        self.DBname = settings['DBname']
        self.DBnum = settings['DBnum']
        self.DBname2 = settings['DBname2']   # for 2nd set of comparison results
        self.DBnum2 = settings['DBnum2']
        self.networkmodel = settings['networkmodel']
        self.GRPresultsname = settings['GRPresultsname']
        self.GRPresultsname2 = settings['GRPresultsname2']
        self.GRPcharacteristicscount = settings['GRPcharacteristicscount']
        self.GRPcharacteristicslist = settings['GRPcharacteristicslist']
        self.GRPcharacteristicsvalues = settings['GRPcharacteristicsvalues']
        self.GRPcharacteristicsvalues2 = settings['GRPcharacteristicsvalues2']
        self.GRPpvalue = settings['GRPpvalue']

        # put some text as a place-holder
        self.GRPLabel1 = tk.Label(self.parent, text = "1) Choose data/results files;  \nSEMresults_network..., SEMresults_2source...,\nor region_properties... files", fg = 'gray', justify = 'left')
        self.GRPLabel1.grid(row=0,column=0,rowspan=2, sticky='W')
        self.GRPLabel1 = tk.Label(self.parent, text = "2) Select group-level analysis options\nChoices are: Bayesian regression of BOLD\nresponses, analyses of SEM results w.r.t.\npersonal characteristics", fg = 'gray', justify = 'left')
        self.GRPLabel1.grid(row=2,column=0,rowspan=2, sticky='W')
        self.GRPLabel2 = tk.Label(self.parent, text = "3) For Bayesian regression select correlation\n for the analysis type, and for group comparisons choose\n2 results files", fg = 'gray', justify = 'left')
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

        self.GRPlabel1 = tk.Label(self.parent, text = 'Results file:')
        self.GRPlabel1.grid(row=0, column=1, sticky='N')
        self.GRPresultsnamelabel = tk.Label(self.parent, textvariable=self.GRPresultsnametext, bg=bgcol, fg="#4B4B4B", font="none 10",
                                      wraplength=300, justify='left')
        self.GRPresultsnamelabel.grid(row=0, column=2, sticky='S')
        self.GRPresultsdirlabel = tk.Label(self.parent, textvariable=self.GRPresultsdirtext, bg=bgcol, fg="#4B4B4B", font="none 8",
                                      wraplength=300, justify='left')
        self.GRPresultsdirlabel.grid(row=1, column=2, sticky='N')
        # define a button to browse and select an existing network definition file, and write out the selected name
        # also, define the function for what to do when this button is pressed
        self.GRPresultsnamebrowse = tk.Button(self.parent, text='Browse', width=smallbuttonsize, bg=fgcol2, fg='black',
                                          command=self.GRPresultsbrowseaction, relief='raised', bd=5)
        self.GRPresultsnamebrowse.grid(row=0, column=3)
        self.GRPresultsnameclear = tk.Button(self.parent, text='Clear', width=smallbuttonsize, bg=fgcol2, fg='black',
                                          command=self.GRPresultsclearaction, relief='raised', bd=5)
        self.GRPresultsnameclear.grid(row=0, column=4)


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

        self.GRPlabel2 = tk.Label(self.parent, text = 'Results file2:')
        self.GRPlabel2.grid(row=2, column=1, sticky='N')
        self.GRPresultsnamelabel2 = tk.Label(self.parent, textvariable=self.GRPresultsnametext2, bg=bgcol, fg="#4B4B4B", font="none 10",
                                      wraplength=300, justify='left')
        self.GRPresultsnamelabel2.grid(row=2, column=2, sticky='S')
        self.GRPresultsdirlabel2 = tk.Label(self.parent, textvariable=self.GRPresultsdirtext2, bg=bgcol, fg="#4B4B4B", font="none 8",
                                      wraplength=300, justify='left')
        self.GRPresultsdirlabel2.grid(row=3, column=2, sticky='N')

        self.GRPresultsnamebrowse2 = tk.Button(self.parent, text='Browse', width=smallbuttonsize, bg=fgcol2, fg='black',
                                          command=self.GRPresultsbrowseaction2, relief='raised', bd=5, state = initial_state)
        self.GRPresultsnamebrowse2.grid(row=2, column=3)
        self.GRPresultsnameclear2 = tk.Button(self.parent, text='Clear', width=smallbuttonsize, bg=fgcol2, fg='black',
                                          command=self.GRPresultsclearaction2, relief='raised', bd=5, state = initial_state)
        self.GRPresultsnameclear2.grid(row=2, column=4)


        # ---------radio buttons to indicate type of analysis to do----------------
        # checkboxes to indicate 1) signficiance from zero, 2) group differences, 3) correlation
        self.GRPanalysistypevalue = tk.IntVar(None,1)
        self.GRPsig1 = tk.Radiobutton(self.parent, text = 'Sign. non-zero', width = bigbuttonsize, fg = 'black',
                                          command = self.GRPselecttype, variable = self.GRPanalysistypevalue, value = 1)
        self.GRPsig1.grid(row = 4, column = 1, sticky="W")

        self.var2 = tk.IntVar()
        self.GRPsig2 = tk.Radiobutton(self.parent, text = 'Avg. Group Diff.', width = bigbuttonsize, fg = 'black',
                                          command = self.GRPselecttype, variable = self.GRPanalysistypevalue, value = 2)
        self.GRPsig2.grid(row = 5, column = 1, sticky="W")

        self.var3 = tk.IntVar()
        self.GRPsig2p = tk.Radiobutton(self.parent, text = 'Paired Group Diff.', width = bigbuttonsize, fg = 'black',
                                          command = self.GRPselecttype, variable = self.GRPanalysistypevalue, value = 3)
        self.GRPsig2p.grid(row = 6, column = 1, sticky="W")

        self.var4 = tk.IntVar()
        self.GRPcorr = tk.Radiobutton(self.parent, text = 'Correlation', width = bigbuttonsize, fg = 'black',
                                          command = self.GRPselecttype, variable = self.GRPanalysistypevalue, value = 4)
        self.GRPcorr.grid(row = 4, column = 2, sticky="W")

        self.var5 = tk.IntVar()
        self.GRPcorr = tk.Radiobutton(self.parent, text = 'Regression', width = bigbuttonsize, fg = 'black',
                                          command = self.GRPselecttype, variable = self.GRPanalysistypevalue, value = 5)
        self.GRPcorr.grid(row = 5, column = 2, sticky="W")


        self.var6 = tk.IntVar()
        self.GRPcorr = tk.Radiobutton(self.parent, text = 'ANOVA', width = bigbuttonsize, fg = 'black',
                                          command = self.GRPselecttype, variable = self.GRPanalysistypevalue, value = 6)
        self.GRPcorr.grid(row = 4, column = 3, sticky="W")

        self.var7 = tk.IntVar()
        self.GRPcorr = tk.Radiobutton(self.parent, text = 'ANCOVA', width = bigbuttonsize, fg = 'black',
                                          command = self.GRPselecttype, variable = self.GRPanalysistypevalue, value = 7)
        self.GRPcorr.grid(row = 5, column = 3, sticky="W")

        # indicate which "personal characteristics" to use - selected from database
        self.GRPlabel2 = tk.Label(self.parent, text = "Select characteristic:")
        self.GRPlabel2.grid(row=7,column=1, sticky='W')
        # fieldvalues = DBFrame.get_DB_field_values(self)
        self.fields = self.get_DB_fields()
        self.field_var = tk.StringVar()
        if len(self.fields) > 0:
            self.field_var.set(self.fields[0])
        else:
            self.field_var.set('empty')

        self.GRPfield_menu = tk.OptionMenu(self.parent, self.field_var, *self.fields, command=self.DBfieldchoice)
        self.GRPfield_menu.grid(row=7, column=2, sticky='EW')
        self.fieldsearch_opt = self.GRPfield_menu  # save this way so that values are not cleared

        # label, button, for running the definition of clusters, and loading data
        self.GRPcharclearbutton = tk.Button(self.parent, text="Clear/Update", width=bigbigbuttonsize, bg=fgcol1, fg='white',
                                        command=self.GRPcharacteristicslistclear, relief='raised', bd=5)
        self.GRPcharclearbutton.grid(row=7, column=3)

        self.GRPlabel3 = tk.Label(self.parent, text = 'Characteristics list:')
        self.GRPlabel3.grid(row=8, column=1, sticky='N')

        self.GRPcharacteristicscount = 0
        self.GRPcharacteristicslist = []
        self.GRPcharacteristicstext = tk.StringVar()
        self.GRPcharacteristicstext.set('not defined yet')
        self.GRPcharacteristicsdisplay = tk.Label(self.parent, textvariable=self.GRPcharacteristicstext, bg=bgcol, fg="#4B4B4B", font="none 10",
                                      wraplength=300, justify='left')
        self.GRPcharacteristicsdisplay.grid(row=8, column=2, sticky='N')

        # put in choices for statistical threshold
        self.GRPlabel5 = tk.Label(self.parent, text = 'p-value threhold:').grid(row=9, column=1, sticky='NSEW')
        self.GRPpvaluebox = tk.Entry(self.parent, width = 8, bg="white")
        self.GRPpvaluebox.grid(row=9, column=2, sticky = "W")
        self.GRPpvaluebox.insert(0,self.GRPpvalue)
        # the entry boxes need a "submit" button so that the program knows when to take the entered values
        self.GRPpvaluesubmitbut = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = 'black', command = self.GRPpvaluesubmit, relief='raised', bd = 5)
        self.GRPpvaluesubmitbut.grid(row=9, column=3)

        # label, button, for running the definition of clusters, and loading data
        self.GRPrunbutton = tk.Button(self.parent, text="Run Group Analysis", width=bigbigbuttonsize, bg=fgcol1, fg='white',
                                        command=self.GRPrunanalysis, relief='raised', bd=5)
        self.GRPrunbutton.grid(row=10, column=1, columnspan = 2)



#----------MAIN calling function----------------------------------------------------
# the main function that starts everything running
def main():
    root = tk.Tk()
    root.title('FMRI Analysis')

    # original that only works on PCs
    # tk.Tk.iconbitmap(root, default='lablogoicon.ico')

    logofile = os.path.join(basedir, 'lablogo.gif')
    img = tk.Image('photo', file=logofile)
    # root.iconphoto(True, img) # you may also want to try this.
    root.tk.call('wm', 'iconphoto', root._w, img)

    app = mainspinalfmri_window(root)

    root.mainloop()

if __name__ == '__main__':
    main()
    
    
