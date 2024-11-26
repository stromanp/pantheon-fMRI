
import numpy as np
import pandas as pd
import openpyxl
import os
import shutil
import scipy.io
import copy
import matplotlib.pyplot as plt
import time
from scipy import interpolate

# temp
def load_group():
    outputname = r'Y:\FMstudy2023\eyetrackingdata.npy'
    data = np.load(outputname, allow_pickle=True)
    group_eye_data = []
    groupcount = 0

    runtype = 3

    if runtype == 0:
        group = 'FMfast'
        maxtimelimit = 270000.
        show_plot = True
        windownum = 23
        pltcolor = [1,0,0]

    if runtype == 1:
        group = 'HCfast'
        maxtimelimit = 270000.
        show_plot = True
        windownum = 24
        pltcolor = [0,0,1]

    if runtype == 2:
        group = 'FMslow'
        maxtimelimit = 270000.
        show_plot = True
        windownum = 23
        pltcolor = [1,0.5,0]

    if runtype == 3:
        group = 'HCslow'
        maxtimelimit = 270000.
        show_plot = True
        windownum = 24
        pltcolor = [0,0.5,1]


    npointslist = []
    nnlist = list(range(len(data)))
    nnlist.remove(296)  # problem entry for FMfast
    for nn in nnlist:
        sg = data[nn]['studygroup']
        if np.ndim(data[nn]['eyedata']) == 3:
            eyedata = np.array(data[nn]['eyedata'])[0,:,:]
        else:
            eyedata = np.array(data[nn]['eyedata'])
        rate = np.array(data[nn]['rate'])
        if (sg == group) & (np.shape(eyedata)[0] > 0):
            if groupcount == 0:
                npoints1, nd1 = np.shape(eyedata)

                maxtimestamp = np.max(eyedata[:, 0])
                maxtimeindex = np.argmax(eyedata[:, 0])
                timestamp = eyedata[:maxtimeindex, 0]
                pupilsize = eyedata[:maxtimeindex, 3]
                pupilsize[pupilsize < 0] = 0.

                maxinterptime = np.min([maxtimestamp, maxtimelimit]).astype(int)
                newtime = np.arange(0, maxinterptime, 2)
                f = interpolate.interp1d(timestamp, pupilsize, fill_value = 'extrapolate')
                pupilinterp = f(newtime)

                # if maxinterptime < maxtimelimit:
                #     pupilsize = np.zeros(int(maxtimelimit/2))
                #     pupilsize[:len(pupilinterp)] = pupilinterp
                #     datacount = np.zeros(int(maxtimelimit/2))
                #     datacount[:len(pupilinterp)] = 1
                # else:
                #     datacount = np.ones(int(maxtimelimit/2))
                #     pupilsize = pupilinterp


                if maxinterptime == maxtimelimit:
                    datacount = np.ones(int(maxtimelimit/2))
                    pupilsize = pupilinterp

                    group_eye_data = pupilsize[:,np.newaxis]
                    avgdatacount = datacount
                    npointslist = [npoints1]
                    ratelist = [rate]
                    groupcount += 1

            else:
                npoints,nd = np.shape(group_eye_data)
                npoints1, nd1 = np.shape(eyedata)
                npointslist += [npoints1]

                # ratelist += [rate]
                # if npoints1 > npoints:
                #     temp = np.zeros((npoints1,nd))
                #     temp[:npoints,:] = group_eye_data
                #     group_eye_data = copy.deepcopy(temp)
                # if npoints > npoints1:
                #     temp = np.zeros((npoints,4))
                #     temp[:npoints1,:] = eyedata
                #     eyedata = copy.deepcopy(temp)

                maxtimestamp = np.max(eyedata[:,0])
                maxtimeindex = np.argmax(eyedata[:,0])
                timestamp = eyedata[:maxtimeindex,0]
                pupilsize = eyedata[:maxtimeindex,3]
                pupilsize[pupilsize < 0] = 0.

                maxinterptime = np.min([maxtimestamp, maxtimelimit]).astype(int)
                newtime = np.arange(0,maxinterptime,2)
                f = interpolate.interp1d(timestamp,pupilsize, fill_value = 'extrapolate')
                pupilinterp = f(newtime)

                # if maxinterptime < maxtimelimit:
                #     pupilsize = np.zeros(int(maxtimelimit/2))
                #     pupilsize[:len(pupilinterp)] = pupilinterp
                #     datacount = np.zeros(int(maxtimelimit/2))
                #     datacount[:len(pupilinterp)] = 1
                # else:
                #     datacount = np.ones(int(maxtimelimit/2))
                #     pupilsize = copy.deepcopy(pupilinterp)


                if maxinterptime == maxtimelimit:
                    if (pupilinterp > 0).all():
                        datacount = np.ones(int(maxtimelimit/2))
                        pupilsize = copy.deepcopy(pupilinterp)
                        avgdatacount += datacount
                        group_eye_data = np.concatenate((group_eye_data, pupilsize[:,np.newaxis]), axis = 1)

                        groupcount += 1

    # check the results
    # checkvals = np.ones(groupcount)
    # for aa in range(groupcount):
    #     check = (group_eye_data[:,aa] < 0).any()
    #     if check:
    #         checkvals[aa] = 0
    # gv = np.where(checkvals == 1)[0]

    if np.max(avgdatacount) > 1:
        avgdata = np.sum(group_eye_data,axis=1)/avgdatacount
        avgdata2 = np.repeat(avgdata[:,np.newaxis],groupcount,axis=1)
        sddata = np.sqrt(np.sum( (group_eye_data - avgdata2)**2, axis = 1)/(avgdatacount-1))
        semdata = np.sqrt(np.sum( (group_eye_data - avgdata2)**2, axis = 1))/(avgdatacount-1)
    else:
        avgdata = np.sum(group_eye_data,axis=1)/avgdatacount
        avgdata2 = np.repeat(avgdata[:,np.newaxis],groupcount,axis=1)
        sddata = np.sqrt(np.sum( (group_eye_data - avgdata2)**2, axis = 1)/(avgdatacount + 1.0e-20))
        semdata = np.sqrt(np.sum( (group_eye_data - avgdata2)**2, axis = 1))/(avgdatacount + 1.0e-20)

    print('found {} data sets'.format(np.max(avgdatacount)))

    if show_plot:
        newtime = np.arange(0,maxtimelimit,2)
        fig = plt.figure(windownum)
        plt.plot(newtime,avgdata,linestyle = '-', linewidth = 2, marker = '', color = pltcolor)
        yps = avgdata + semdata
        yms = avgdata - semdata
        xx = np.concatenate((newtime, newtime[::-1]))
        yy = np.concatenate((yps, yms[::-1]))
        plt.fill(xx,yy,color = pltcolor,alpha = 0.1)




def load_all_eyetracking():
    beginloadtime = time.time()
    outputname = r'Y:\FMstudy2023\eyetrackingdata.npy'
    DBname = r'Y:\FMstudy2023\PS2023_database_corrected.xlsx'
    xls = pd.ExcelFile(DBname, engine='openpyxl')
    xls_sheets = xls.sheet_names

    df1 = pd.read_excel(xls, 'datarecord')
    keylist = df1.keys()
    for kname in keylist:
        if 'Unnamed' in kname: df1.pop(kname)  # remove blank fields from the database

    num = len(df1)   # number of entries
    data = []
    loadcount = 0
    for nn in range(num):
        datadir = copy.deepcopy(df1.loc[nn, 'datadir'])
        pname = copy.deepcopy(df1.loc[nn, 'pname'])
        patientgroup = copy.deepcopy(df1.loc[nn, 'patientgroup'])
        patientid = copy.deepcopy(df1.loc[nn, 'patientid'])
        seriesnumber = copy.deepcopy(df1.loc[nn, 'seriesnumber']).astype(int)

        try:
            etdir_subdir = copy.deepcopy(pname)
            etdir_subdir = 'ET_' + (etdir_subdir.lower()).capitalize()
        except:
            etdir_subdir = 'notdefined'
        tagdir = r'ASCII'

        try:
            filename = 'S23' + patientid[-2:] + 's' + '{:02d}'.format(seriesnumber) + '.asc'
            filename_alt = 'S23' + patientid[-2:] + 's' + '{}'.format(seriesnumber) + '.asc'
        except:
            filename = 'notdefined'
            filename_alt = 'notdefined'

        try:
            etname = os.path.join(datadir, pname, etdir_subdir, tagdir, filename)
            etname_alt = os.path.join(datadir, pname, etdir_subdir, tagdir, filename_alt)
        except:
            etname = 'notdefined'
            etname_alt = 'notdefined'

        p, f1 = os.path.split(etname)
        f, e = os.path.splitext(f1)
        npname = os.path.join(p, f + '.npy')

        # check for correct file naming
        if not os.path.isfile(etname):
            if os.path.isfile(etname_alt):
                print('copying from {} to {}'.format(etname_alt, etname))
                shutil.copy(etname_alt, etname)
                dataexists = True
            else:
                dataexists = False
                print('ERROR:  data does not exist: {}'.format(etname))
        else:
            dataexists = True

        if os.path.isfile(npname):
            npfileexists = True
        else:
            npfileexists = False

        # read the ascii file
        if dataexists:
            if npfileexists:
                entry = np.load(npname, allow_pickle=True).flat[0]
                print('loaded {}'.format(npname))
                loadsuccess = True
            else:
                eyedata, starttime, rate = read_eyedata_ascii(etname)
                if starttime < 0:  # error
                    loadsuccess = False
                    print('ERROR:  could not load {}'.format(etname))
                else:
                    loadsuccess = True
                    timestamp = eyedata[:,0]
                    ct = np.where(timestamp >= 0)[0]
                    eyedata = eyedata[ct,:]
                    print('finished loading data set {} of {}     {}'.format(nn,num,time.ctime()))
                    entry = {'eyedata':eyedata, 'starttime':starttime, 'rate':rate, 'dbnum':nn, 'studygroup':patientgroup, 'patientid':patientid, 'seriesnumber':seriesnumber}

                    np.save(npname, entry)
                    print('saved {}'.format(npname))

            if loadsuccess:
                data.append(entry)
                loadcount += 1
                TL = time.time()-beginloadtime
                h = np.floor(TL / 3600.).astype(int)
                m = np.floor((TL % 3600) / 60.).astype(int)
                s = np.floor(TL % 60).astype(int)
                print('time lapsed to load {} data sets:  {} hours {} minutes {} seconds'.format(loadcount, h,m,s))

    np.save(outputname, data)



def show_eyetracking(personnum, windownum = 10):
    DBname = r'Y:\FMstudy2023\PS2023_database_corrected.xlsx'
    xls = pd.ExcelFile(DBname, engine='openpyxl')
    xls_sheets = xls.sheet_names

    df1 = pd.read_excel(xls, 'datarecord')
    keylist = df1.keys()
    for kname in keylist:
        if 'Unnamed' in kname: df1.pop(kname)  # remove blank fields from the database

    num = len(df1)   # number of entries
    # for nn in range(num):
    nn = personnum  # pick one

    datadir = copy.deepcopy(df1.loc[nn, 'datadir'])
    pname = copy.deepcopy(df1.loc[nn, 'pname'])
    patientgroup = copy.deepcopy(df1.loc[nn, 'patientgroup'])
    patientid = copy.deepcopy(df1.loc[nn, 'patientid'])
    seriesnumber = copy.deepcopy(df1.loc[nn, 'seriesnumber']).astype(int)

    etdir_subdir = copy.deepcopy(pname)
    etdir_subdir = 'ET_' + (etdir_subdir.lower()).capitalize()
    tagdir = r'ASCII'
    filename = 'S' + pname[-2:] + patientid[-2:] + 's' + '{:02d}'.format(seriesnumber) + '.asc'

    etname = os.path.join(datadir, pname, etdir_subdir, tagdir, filename)
    p, f1 = os.path.split(etname)
    f, e = os.path.splitext(f1)
    npname = os.path.join(p, f + '.npy')

    # read the ascii file
    if os.path.isfile(npname):
        entry = np.load(npname, allow_pickle=True).flat[0]
        eyedata = entry['eyedata']
        starttime = entry['starttime']
        rate = entry['rate']
    else:
        eyedata, starttime, rate = read_eyedata_ascii(etname)

    tt = eyedata[:, 0] - eyedata[0, 0]
    # windownum = 10
    plt.close(windownum)
    plt.figure(windownum)
    plt.plot(tt, eyedata[:, 3])

    plt.close(windownum + 1)
    plt.figure(windownum + 1)
    plt.plot(tt, eyedata[:, 1], '-r')
    plt.plot(tt, eyedata[:, 2], '-b')

    print('start time = {}   rate = {}'.format(starttime, rate))



# example
# E:\FMstudy2023\APR17_2023\ET_Apr17_2023\ASCII\S2301s05.asc

# files to read and analyze eye-tracking data in ascii format

# for testing....
# fname = r'E:\FMstudy2023\NOV23_2023\ET_Nov23_2023\ASCII\S2337s04.asc'


def read_eyedata_ascii(fname):
    f = open(fname,'r')
    data = f.read()

    # find the flags to guide parsing the ascii data
    flags = ['START\t', 'INPUT\t', 'END\t']
    xlist = []
    for n in range(len(flags)):
        if n == 0:
            startpoint = 0
        else:
            startpoint = xlist[n-1]
        l = len(flags[n])
        try:
            x = data[startpoint:].index(flags[n]) + startpoint
            xlist += [x]
            print('{}  {}'.format(x,data[x:x+l]))
        except:
            print('warning:  flag {}  not found in {}'.format(flags[n], fname))
            if n == 0:
                xlist += [0]
            if n == 1:
                # cannot read data
                return [], -1, -1
            if n == 2:
                xlist += [len(data)]

    # parse the data based on the flags
    # before START find the calibration information
    tpos = 0
    keepsearching = True
    foundcoords = False
    foundcal0 = False
    foundcal1 = False
    foundcal2 = False

    display_coords = [0,0,0,0]
    pos0 = [0,0]
    pos1 = [0,0]
    pos2 = [0,0]
    while (tpos < xlist[0]) & keepsearching:
        p = data[tpos:].index('\n') + tpos
        text = data[tpos:p]
        splittext = split_text_by_delimiter(text, '\t')

        if text[:3] == 'MSG':
            if not foundcoords:
                searchphrase = 'DISPLAY_COORDS'
                if searchphrase in text:
                    foundcoords = True
                    p2 = text.index(searchphrase)
                    splittext2 = split_text_by_delimiter(text[p2:], ' ')
                    if len(splittext2) > 4:
                        display_coords = np.array(splittext2[1:]).astype(float)

            if not foundcal0:
                searchphrase = 'VALIDATE R 4POINT 0 RIGHT'
                if searchphrase in text:
                    foundcal0 = True
                    p2 = text.index(searchphrase)
                    splittext2 = split_text_by_delimiter(text[p2:], ' ')
                    if 'at' in splittext2:
                        c = np.where(splittext2 == 'at')[0]
                        pos = splittext2[c+1]
                        splittext3 = split_text_by_delimiter(pos[0], ',')
                        pos0 = splittext3.astype(float)

            if not foundcal1:
                searchphrase = 'VALIDATE R 4POINT 1 RIGHT'
                if searchphrase in text:
                    foundcal1 = True
                    p2 = text.index(searchphrase)
                    splittext2 = split_text_by_delimiter(text[p2:], ' ')
                    if 'at' in splittext2:
                        c = np.where(splittext2 == 'at')[0]
                        pos = splittext2[c+1]
                        splittext3 = split_text_by_delimiter(pos[0], ',')
                        pos1 = splittext3.astype(float)

            if not foundcal2:
                searchphrase = 'VALIDATE R 4POINT 2 RIGHT'
                if searchphrase in text:
                    foundcal2 = True
                    p2 = text.index(searchphrase)
                    splittext2 = split_text_by_delimiter(text[p2:], ' ')
                    if 'at' in splittext2:
                        c = np.where(splittext2 == 'at')[0]
                        pos = splittext2[c+1]
                        splittext3 = split_text_by_delimiter(pos[0], ',')
                        pos2 = splittext3.astype(float)

            if foundcoords & foundcal0 & foundcal1 & foundcal2: keepsearching = False

        tpos = p + 1
    # finished reading calibration points
    if keepsearching:
        foundcalibration = False
    else:
        foundcalibration = True



    # starting point
    tpos = xlist[1]
    p = data[tpos:].index('\n') + tpos
    text = data[tpos:p]
    splittext = split_text_by_delimiter(text, '\t')
    starttime = int(splittext[1])
    rate = int(splittext[2])
    tpos = p+1

    messagelist = ['MSG','EFIX','SFIX','SSACC','ESACC']
    eyedata = []
    message_record = []
    progress_message = np.zeros(10)
    progress_count = 0
    foundstarttime = False
    runstarttime = 0
    while tpos < xlist[-1]:
        p = data[tpos:].index('\n') + tpos
        text = data[tpos:p]
        splittext = split_text_by_delimiter(text, '\t')

        try:
            timestamp = int(splittext[0])
            xpos = float(splittext[1])
            ypos = float(splittext[2])
            pupilsize = float(splittext[3])
            datapoint = [timestamp, xpos, ypos, pupilsize]
            eyedata += [datapoint]
        except:
            # expect a flag
            for mm, msg in enumerate(messagelist):
                ml = len(msg)
                check = splittext[0][:ml] == msg
                if check:
                    if msg == 'MSG':
                        messagebody = splittext[1]
                        splitmessage = split_text_by_delimiter(messagebody, ' ')
                        timestamp = splitmessage[0]
                        if len(splitmessage) > 2:
                            messagecode = splitmessage[1]
                            message = splitmessage[2]
                        else:
                            message = splitmessage[1]
                        text = copy.deepcopy(message)
                        if text == 'WAIT_FOR_TRIGGER':
                            runstarttime = copy.deepcopy(float(timestamp))
                            foundstarttime = True
                        # WAIT_FOR_TRIGGER
                        # END_MESSAGE
                    else:
                        timestamp = -1
                    entry = {'msg':msg, 'text':text, 'index':tpos, 'timestamp':timestamp}
                    message_record.append(entry)
        tpos = p+1

        progress = 100.0*(tpos-xlist[1])/(xlist[-1]-xlist[1])
        if progress > 10.0*progress_count:
            print('{:.1f} percent done ...'.format(progress))
            progress_count += 1

    print('    done.')
    eyedata = np.array(eyedata)

    try:
        npoints,nd= np.shape(eyedata)
        eyedata[:,0] -= runstarttime*np.ones(npoints)
    except:
        # problem reading the data
        return [], -1, -1

    return eyedata, starttime, rate


def split_text_by_delimiter(text, delimiter):
	parsedtext = []
	keeplooking = True
	dd = len(delimiter)
	while keeplooking:
		try:
			c = text.index(delimiter)
			oneword = text[:c]
			parsedtext += [oneword]
			text = text[(c+dd):]
		except:
			keeplooking = False
			parsedtext += [text]
	return np.array(parsedtext)
