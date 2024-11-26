# move data to new location, excluding DICOM format, if wanted
import numpy as np
import pandas as pd
import openpyxl
import os
import shutil
import copy


sourcedir = r"Y:\Copy_of_PS2023"
destdir = r"F:\FMstudy2023_processed"

exclusionlist = ['.IMA', '.SR']
overwrite = False

record = []
for root, dirs, files in os.walk(sourcedir):
    for dir in dirs:
        fullsourcedir = os.path.join(root, dir)
        fulldestdir = fullsourcedir.replace(sourcedir,destdir)
        if not os.path.isdir(fulldestdir):
            os.mkdir(fulldestdir)
            print('creating directory {}'.format(fulldestdir))

    for file in files:
        f,e = os.path.splitext(file)
        if e not in exclusionlist:
            record.append({'root':root, 'dirs':dirs, 'file':file})

            fullsourcepath = os.path.join(root, file)
            fulldestpath = fullsourcepath.replace(sourcedir,destdir)

            if overwrite:
                print('copy {} to {}'.format(fullsourcepath,fulldestpath))
                shutil.copyfile(fullsourcepath,fulldestpath)
            else:
                if os.path.isfile(fulldestpath):
                    print('already exists: {}'.format(fulldestpath))
                else:
                    print('copy {} to {}'.format(fullsourcepath,fulldestpath))
                    shutil.copyfile(fullsourcepath,fulldestpath)

