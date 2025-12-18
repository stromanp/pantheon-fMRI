# move data to new location, based on a database file
import numpy as np
import os
import shutil
import copy
import pandas as pd


def run_steps(stepnumber = 1):
    # these are just examples of how to run the functions in this file

    #-------------------------------------------------------------------
    # to backup files to the NAS, do this:------------------------------
    if stepnumber == 1:


        source_directory = r'E:\JM2024_processed'    # folder where you want the copy on the NAS
        target_directory = r'Y:\JM2024_processed'    # folder where you want the copy on the NAS

        source_directory = r'E:\BK2024_Processed'    # folder where you want the copy on the NAS
        target_directory = r'Y:\BK2024_Processed'    # folder where you want the copy on the NAS

        source_directory = r'E:\FMstudy2023_processed'    # folder where you want the copy on the NAS
        target_directory = r'Y:\FMstudy2023_processed'    # folder where you want the copy on the NAS


        # make additional backups
        source_directory = r'E:\FMstudy2023_processed'    # folder where you want the copy on the NAS
        target_directory = r'F:\Shima_backup\fMRI_data'    # folder where you want the copy on the NAS

        source_directory = r'E:\JM2024_processed'    # original data folder
        target_directory = r'F:\Jessica_backup\JM_copy2\JM2024'    # folder with backup copy

        source_directory = r'E:\BK2024_Processed'    # original data folder
        target_directory = r'F:\Brieana_backup\CPM_Data_Copy'    # folder with backup copy

        source_directory = r'E:\FMstudy2023_processed'    # new data on NAS
        target_directory = r'F:\pain_data_sets\FMstudy2023_processed'    # backup folder

        source_directory = r'Y:\HA2024_data'    # new data on NAS
        target_directory = r'F:\Hannan_backup\HA2024_data'    # backup folder

        source_directory = r'Y:\HA2024_data'    # new data on NAS
        # target_directory = r'E:\HA2024_processed'    # backup folder
        target_directory = r'F:\pain_data_sets\HA2024_processed'    # backup folder

        copymode = 'update'   # options are 'overwrite', 'update', or 'noreplacement'
                              #   overwrite:   copy all files even if they are already in the target folder
                              #   update:  copy any newer/updated files, and any that dont exist yet in the target folder
                              #   noreplacement:  only copy files if they do not exist in the target folder
        backup_data(source_directory, target_directory, copymode=copymode)   # run the backup step


    #----------------------------------------------------------------------------------------------------
    # to copy all edf files to one folder for conversion to ascii, do this:------------------------------
    if stepnumber == 2:
        source_directory = r'C:\mystudy\my_main_data_folder'   # folder containing all study data on local computer
        copy_edf_files(source_directory)    # this will create a folder called EDF_eyetrackingfiles
                            # within the source_directory folder
                            # all of the edf files within subfolders of the source_directory will then be
                            # copied to the EDF_eyetrackingfiles folder



def backup_data(source_directory, target_directory, exclude = ['.IMA', '.SR'], copymode = 'noreplacement'):
    # makes a copy of the entire directory tree, excluding DICOM format files by default
    # change which files are excluded, if any, by changing the "exclude" input
    # i.e. exclude = [] to copy all files, or exclude = ['.docx','.pdf'] to exclude text docs etc.

    if copymode.lower() in ['overwrite', 'update', 'noreplacement']:
        if copymode.lower() == 'overwrite':
            overwriteexisting = True
            overwriteolder = True

        if copymode.lower() == 'update':
            overwriteexisting = False
            overwriteolder = True

        if copymode.lower() == 'noreplacement':
            overwriteexisting = False
            overwriteolder = False
    else:
        print('backup_data:   invalid copymode parameter entered.')
        print('               copymode must one of: noreplacement, update, or overwrite')
        return

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
                singlefile_mtime = os.path.getmtime(singlefile)
                if os.path.isfile(new_singlefile):
                    new_singlefile_mtime = os.path.getmtime(new_singlefile)
                    DSTcheckdif = np.abs(singlefile_mtime - new_singlefile_mtime) == 3600.  # check if difference could be because of clock change
                    if overwriteexisting:
                        print('overwrite {} to {}'.format(singlefile,new_singlefile))
                        shutil.copyfile(singlefile, new_singlefile)
                    else:
                        if overwriteolder:
                            dtime = new_singlefile_mtime - singlefile_mtime
                            if (dtime < 0) and not DSTcheckdif:   # new_singlefile was created earlier
                                shutil.copyfile(singlefile, new_singlefile)
                                print('newer:  overwrite {} to {}'.format(singlefile,new_singlefile))
                        else:
                            print('current file exists - not overwriting: {}'.format(new_singlefile))
                else:
                    print('copy {} to {}'.format(singlefile,new_singlefile))
                    shutil.copyfile(singlefile, new_singlefile)



def copy_edf_files(source_directory):
    # makes a copy of the entire directory tree, excluding DICOM format files by default
    # change which files are excluded, if any, by changing the "exclude" input
    # i.e. exclude = [] to copy all files, or exclude = ['.docx','.pdf'] to exclude text docs etc.
    target_directory = os.path.join(source_directory, 'EDF_eyetrackingfiles')
    if not os.path.isdir(target_directory):
        os.makedirs(target_directory, exist_ok=True)
        print('created directory {}'.format(target_directory))

    for dirName, subdirList, fileList in os.walk(source_directory):
        for filename in fileList:
            f,e = os.path.splitext(filename)
            if e == '.edf':
                originalname = os.path.join(dirName, filename)
                newname = os.path.join(target_directory, filename)

                if os.path.isfile(newname):
                    print('file already exists, not copying: {}'.format(newname))
                else:
                    shutil.copyfile(originalname, newname)
                    print('copied {} to {}'.format(originalname, newname))



# ----------MAIN calling function----------------------------------------------------
# the main function that starts everything running
def main():
    stepnumber = 1
    run_steps(stepnumber)

if __name__ == "__main__":
    main()




def save_data_for_conversion_to_new_format():
    # run this on system with old version of Pandas

    fname = r'Y:\VeronicaVoth\Pain_equalsize_cluster_def.npy'
    data = np.load(fname, allow_pickle=True).flat[0]
    cluster_properties = data['cluster_properties']
    template_img = data['template_img']
    regionmap_img = data['regionmap_img']

    newfname1 = r'Y:\VeronicaVoth\PainCD_part1.npy'
    newfname2 = r'Y:\VeronicaVoth\PainCD_part2.npy'
    newfname3 = r'Y:\VeronicaVoth\PainCD_part3.pickle'

    np.save(newfname1, {'template_img':template_img})
    np.save(newfname2, {'regionmap_img':regionmap_img})

    df = pd.DataFrame.from_dict(data['cluster_properties'])
    df.to_pickle(newfname3)



def load_data_for_conversion_to_new_format():
    # run this on system with new version of Pandas
    outputname = r'Y:\VeronicaVoth\Pain_equalsize_ClusterDef_Oct2025.npy'

    newfname1 = r'Y:\VeronicaVoth\PainCD_part1.npy'
    newfname2 = r'Y:\VeronicaVoth\PainCD_part2.npy'
    newfname3 = r'Y:\VeronicaVoth\PainCD_part3.pickle'

    data1 = np.load(newfname1, allow_pickle=True).flat[0]
    data2 = np.load(newfname2, allow_pickle=True).flat[0]

    template_img = copy.deepcopy(data1['template_img'])
    regionmap_img = copy.deepcopy(data2['regionmap_img'])


    df1 = pd.read_pickle(newfname3)
    nvals = len(df1)
    keylist = df1.keys()

    cluster_properties = []
    for nn in range(nvals):
        datavals = []
        for keyname in keylist:
            datavals += [df1[keyname][nn]]
        entry = dict(zip(keylist,datavals))
        cluster_properties.append(entry)

    np.save(outputname, {'cluster_properties':cluster_properties, 'template_img':template_img, 'regionmap_img':regionmap_img})



# def temp_function():
#
#     fname = r'Y:\Veronica\HCfast_F_regiondata_PainCD_Sept12_2025_VV.npy'
#     newfname1 = r'Y:\Veronica\HCfast_F_part1.npy'
#     newfname2 = r'Y:\Veronica\HCfast_F_part2.npy'
#     newfname3 = r'Y:\Veronica\HCfast_F_part3.npy'
#     data = np.load(fname, allow_pickle=True).flat[0]
#     new_data1 = []
#     new_data2 = []
#     new_data3 = []
#     for nn in range(10):
#         newentry1 = {'tc':data['region_properties'][nn]['tc'], 'tc_sem':data['region_properties'][nn]['tc_sem']}
#         newentry2 = {'tc_original':data['region_properties'][nn]['tc_original'], 'tc_sem_original':data['region_properties'][nn]['tc_sem_original']}
#         newentry3 = {'nruns_per_person':data['region_properties'][nn]['nruns_per_person'], 'tsize':data['region_properties'][nn]['tsize'],
#                      'rname':data['region_properties'][nn]['rname'], 'DBname':data['region_properties'][nn]['DBname'],
#                      'DBnum':data['region_properties'][nn]['DBnum'] , 'prefix':data['region_properties'][nn]['prefix'] ,
#                      'occurrence':data['region_properties'][nn]['occurrence']}
#         new_data1.append(newentry1)
#         new_data2.append(newentry2)
#         new_data3.append(newentry3)
#
#     np.save(newfname1, new_data1)
#     np.save(newfname2, new_data2)
#     np.save(newfname3, new_data3)


def compare_region_properities():
    cdef1 = r'Y:\BigAnatomicalAnalysis\Pain_cluster_def_Oct2025_exp.npy'
    cdef2 = r'Y:\BigAnatomicalAnalysis\Pain_equalsize_ClusterDef_Oct2025.npy'

    cdef_data1 = np.load(cdef1, allow_pickle=True).flat[0]
    cdef_data2 = np.load(cdef2, allow_pickle=True).flat[0]

    cp1 = copy.deepcopy(cdef_data1['cluster_properties'])
    cp2 = copy.deepcopy(cdef_data2['cluster_properties'])

    nregions = len(cp1)
    for rr in range(nregions):
        nclusters = cp1[rr]['nclusters']

        cx1 = cp1[rr]['cx']
        cy1 = cp1[rr]['cy']
        cz1 = cp1[rr]['cz']

        cx2 = cp2[rr]['cx']
        cy2 = cp2[rr]['cy']
        cz2 = cp2[rr]['cz']

        IDX1 = cp1[rr]['IDX']
        IDX2 = cp2[rr]['IDX']

        clustermatch = np.zeros((nclusters, nclusters))
        nvox1_list = np.zeros(nclusters)
        for cc1 in range(nclusters):
            clusterdiv1 = np.where(IDX1 == cc1)[0]
            nvox1_list[cc1] = len(clusterdiv1)
            for cc2 in range(nclusters):
                clusterdiv2 = np.where(IDX2 == cc2)[0]

                posval1 = np.array(cx1[clusterdiv1]) + 1000.*np.array(cy1[clusterdiv1]) + 1e6 * np.array(cz1[clusterdiv1])
                posval2 = np.array(cx2[clusterdiv2]) + 1000.*np.array(cy2[clusterdiv2]) + 1e6 * np.array(cz2[clusterdiv2])

                match = [posval1[x] for x in range(len(posval1))  if posval1[x] in posval2]
                clustermatch[cc1,cc2] = len(match)

        print('\nregion {}  {}'.format(rr, cp1[rr]['rname']))
        for cc1 in range(nclusters):
            for cc2 in range(nclusters):
                if cc2 == (nclusters-1):
                    print('  {:5.1f} % '.format(100.0*clustermatch[cc1,cc2]/nvox1_list[cc1]))
                else:
                    print('  {:5.1f} % '.format(100.0*clustermatch[cc1,cc2]/nvox1_list[cc1]), end = '')


