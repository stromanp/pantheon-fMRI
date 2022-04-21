# pantheon-fMRI
fMRI analysis at any level of the CNS - brain, brainstem, spinal cord

Preparing python for Patheon and installing the package
 
Pantheon is a set of programs for analyzing functional MRI data from any level of the central nervous system (CNS). The user must 
provide a file in Excel format to define the data locations, some information about the data, the region imaged, etc., and this 
file is used to guide automated processing and analysis. The processing includes conversion from DICOM to NIfTI format, co-registration 
(i.e. motion correction), slice-timing correction, spatial normalization, smoothing, definition of stimulation paradigms to predict 
BOLD responses, estimation of physiological noise based on bulk motion (from the co-registration step) and bulk noise estimated from
white matter regions. The user can also create a “cleaned” version of the data with the noise/confounds fit and removed from the data, 
and the time-series response in each voxel represented as the percent signal change from the time-series average.
Analysis steps currently include fitting of predicted BOLD responses to the data with a general linear model (GLM), as well as 
connectivity analyses based on clustered data within specified anatomical regions-of-interest.  The connectivity analyses include 
single region-region correlations, two-region fits to each target region, and structural equation modeling using a predefined anatomical 
network model.

The data must be acquired in a way to enable some analysis steps to function, such as spatial normalization.  The data must include 
enough distinct anatomical features to identify the region, and so studies of the thoracic spinal cord, for example, can be difficult 
to normalize without additional position information input by the user. For most studies of the brain, brainstem/cervical cord, or 
lumbar cord regions, the normalization appears to work well.

Development of this software is on-going, and no guarantees are given or implied about the accuracy of the analysis results.  This software 
is for research only.

The current version was developed in Windows, and has been tested somewhat on Mac environments.  It has not been tested on Unix systems.


Initial setup and installation

STEP 1:

Use Python 3 by installing Anaconda from https://www.anaconda.com/distribution/

STEP 2:
Install Pycharm.  This is the programming environment program:

https://www.jetbrains.com/help/pycharm/installation-guide.html

STEP 3:  
Configure Pycharm

Create a new project
	…from the options available when setting up the project:
	- choose new environment using Virtualenv
	- base interpreter  - choose python3.6, or 3.8 also seems to work, from wherever this was installed with Anaconda
	- do NOT choose to inherit global site packages (will configure these)

STEP 4:
Get pyspinalfmri3 from GitHub
- copy the python files and “template” directory into the “venv” folder that was created for the project


STEP 5
Setup the environment in PyCharm for pyspinalfmri3
	- go into “Settings” for the project
	-install packages:
		dicom2nifti
		numpy
		matplotlib (choose version 3.2.2)
		nibabel
		openpyxl
		pandas
		Pillow
		pyinstaller
		sklearn
		xlrd (choose version 1.2.0)
		dipy


STEP 6
run pyspinalfmri.py in Pycharm

