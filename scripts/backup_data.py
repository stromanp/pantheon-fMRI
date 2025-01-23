# move data to new location, based on a database file
import numpy as np
import os
import shutil

def backup_data(source_directory, target_directory, exclude = ['.IMA']):
    # makes a copy of the entire directory tree, excluding DICOM format files by default
    # change which files are excluded, if any, by changing the "exclude" input
    # i.e. exclude = [] to copy all files, or exclude = ['.docx','.pdf'] to exclude text docs etc.

    excludelist = [exclude[x].lower() for x in range(len(exclude))]

    for dirName, subdirList, fileList in os.walk(source_directory):
        newdirName = dirName.replace(source_directory,target_directory)

        if not os.path.isdir(newdirName):
            os.makedirs(newdirName, exist_ok=True)
            print('create directory {}'.format(newdirName))

        for filename in fileList:
            extensioncheck = np.array([excludelist[x] in filename.lower() for x in range(len(excludelist))]).any()
            if not extensioncheck:  # check whether the file is an excluded type
                singlefile = os.path.join(dirName,filename)
                new_singlefile = os.path.join(newdirName,filename)

                print('copy {} to {}'.format(singlefile,new_singlefile))
                shutil.copyfile(singlefile, new_singlefile)
