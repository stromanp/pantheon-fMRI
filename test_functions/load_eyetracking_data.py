
import numpy as np
import pandas as pd
import os
import shutil
import copy
import matplotlib.pyplot as plt
import time
from scipy import interpolate


def test_load_gazeposition(eyedataname, entrynum):
    alldata = np.load(eyedataname, allow_pickle=True)
    reffig = r'Y:\JM2024_processed\PainIntensity_and_UnpleasantnessScale_11pt_isoL.png'
    refdisplay = plt.imread(reffig)
    dx, dy, dc = np.shape(refdisplay)

    screenh = 1920
    screenv = 1080
    hcenter = 1920/2.0
    vcenter = 1080/2.0
    hrange = 0.5
    vrange = 0.5

    entrynum = 8
    windownum = 30
    tt = copy.deepcopy(alldata[entrynum]['eyedata'][:, 0])
    xx = copy.deepcopy(alldata[entrynum]['eyedata'][:, 1])
    yy = copy.deepcopy(alldata[entrynum]['eyedata'][:, 2])
    ps = copy.deepcopy(alldata[entrynum]['eyedata'][:, 3])

    xx[xx < 0] = 0.0
    xx[xx > screenh] = screenh
    yy[yy < 0] = 0.0
    yy[yy > screenv] = screenv

    t1 = [0, 60000, 120000, 150000]
    t2= [60000, 120000, 150000, 270000]

    for nn in range(len(t1)):
        c = np.where( (tt >= t1[nn]) &  (tt <= t2[nn]) )[0]
        plt.close(windownum+nn)
        fig = plt.figure(windownum+nn)
        plt.imshow(refdisplay)
        plt.plot(xx[c]*dy/screenh, yy[c]*dx/screenv, '-r')



def run_setup_steps(stepnumber):
    # this section just gives examples of how to do each step in the process
    datadir = r'Y:\Julia_Baldassarre'
    eyedata_outputname = os.path.join(datadir, 'eyetrackingdata_July2025.npy')
    BOLDdata_list = [os.path.join(datadir, '38C_regiondata_PainCD_Sept2025.npy'),
                     os.path.join(datadir, '46C_regiondata_PainCD_Sept2025.npy'),
                     os.path.join(datadir, '51C_regiondata_PainCD_Sept2025.npy')]
    BOLD_temperatures = [38, 46, 51]
    BOLD_TR = 6.75   # sampling rate of fMRI data (sec)
    eye_TR = 0.002   # sampling rate of eye data (sec)
    database_number_offset = 892   # difference in starting number between the two databases

    # specify name for saving the combined data
    alldata_outputname = os.path.join(datadir, 'eyetracking_BOLD_combined_Sept2025.npy')

    if stepnumber == 1:
        #--------------------------------------------
        # load the eye-tracking data, correspondiong covariate data, and save it in a more convenient form
        DBname = r'Y:\Julia_Baldassarre\database_Merletti11.xlsx'
        datafields = ['temperature', 'painintensity', 'painunpleasantness']
        datanametag = 'J24'
        load_all_eyetracking(DBname, datafields, datanametag, eyedata_outputname)
        #-----------------------------------------------

    if stepnumber == 2:
        #--------------------------------------------
        # load the eye data
        eyedata = np.load(eyedata_outputname, allow_pickle=True)
        nruns_e = len(eyedata)
        # dict_keys(['eyedata', 'starttime', 'rate', 'dbnum', 'studygroup', 'patientid', 'seriesnumber', 'temperature',
        #            'painintensity', 'painunpleasantness', 'sex'])
        num_eye_runs = len(eyedata)

        # load the BOLD region data, combined it with the eye data, and save it in a more convenient for
        all_combined_data = []

        for listnumber in range(len(BOLDdata_list)):
            bdataname = BOLDdata_list[listnumber]
            bdata = np.load(bdataname, allow_pickle=True).flat[0]
            rp = copy.deepcopy(bdata['region_properties'])   # rp has length equal to number of regions
            nregions = len(rp)
            ncluster_list = [np.shape(rp[xx]['tc'])[0] for xx in range(nregions)]
            nclusters_total = np.sum(ncluster_list)
            bold_DBnum_list = copy.deepcopy(bdata['DBnum'])
            nruns_per_person = copy.deepcopy(rp[0]['nruns_per_person'])
            nruns_total = np.sum(nruns_per_person)
            tsize = copy.deepcopy(rp[0]['tsize'])

            for eye_run in range(num_eye_runs):
                this_run_eyedata = copy.deepcopy(eyedata[eye_run]['eyedata'])
                starttime = copy.deepcopy(eyedata[eye_run]['starttime'])
                rate = copy.deepcopy(eyedata[eye_run]['rate'])
                eye_dbnum = copy.deepcopy(eyedata[eye_run]['dbnum'])
                temperature = copy.deepcopy(eyedata[eye_run]['temperature'])
                studygroup = copy.deepcopy(eyedata[eye_run]['studygroup'])
                sex = copy.deepcopy(eyedata[eye_run]['sex'])
                seriesnumber = copy.deepcopy(eyedata[eye_run]['seriesnumber'])
                painintensity = copy.deepcopy(eyedata[eye_run]['painintensity'])
                painunpleasantness = copy.deepcopy(eyedata[eye_run]['painunpleasantness'])
                bold_dbnum = eye_dbnum + database_number_offset

                # now match up the data sets
                if bold_dbnum in bold_DBnum_list:
                    print('loading DBnum {}'.format(bold_dbnum))
                    run_number = np.where(bold_DBnum_list == bold_dbnum)[0][0]
                    t1 = run_number*tsize
                    t2 = (run_number+1)*tsize
                    db1 = copy.deepcopy(bold_DBum[run_number])

                    this_run_bold = np.zeros((nclusters_total,tsize))
                    for nn in range(nregions):
                        for cc in range(ncluster_list[nn]):
                            clusternumber = np.sum(ncluster_list[:nn]).astype(int) + cc
                            this_run_bold[clusternumber,:] = copy.deepcopy(rp[nn]['tc'][cc,t1:t2])

                    entry = {'eye_data':this_run_eyedata, 'bold':this_run_bold, 'bold_dbnum':bold_DBnum, 'eye_dbnum':eye_dbnum, 'sex':sex,
                             'temperature':temperature, 'intensity':painintensity, 'unpleasantness':painunpleasantness, 'starttime':starttime,
                             'rate':rate}
                    all_combined_data.append(entry)
                else:
                    print('DBnum {} not found in current BOLD data list'.format(bold_dbnum))

        # save it
        np.save(alldata_outputname, all_combined_data)


    if stepnumber == 3:
        reffig = os.path.join(datadir, 'PainIntensity_and_UnpleasantnessScale_11pt_isoL.png')
        refdisplay = plt.imread(reffig)
        dx, dy, dc = np.shape(refdisplay)

        # load reference images for areas of interest
        maskfig = os.path.join(datadir, 'PainIntensity_and_UnpleasantnessScale_mask.png')
        maskdisplay = plt.imread(maskfig)
        mask = maskdisplay[:,:,0]
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0
        zonefig = os.path.join(datadir, 'PainIntensity_and_UnpleasantnessScale_zones.png')
        zonedisplay = plt.imread(zonefig)
        zones = zonedisplay[:,:,0]
        zones[zones > 0.5] = 1.0
        zones[zones <= 0.5] = 0.0

        screenh = 1920
        screenv = 1080

        xlimits = [120,3340]   # vertical
        ylimits = [600, 3010]  # horizontal


        reload_all_data = True
        if reload_all_data:
            all_combined_data = np.load(alldata_outputname, allow_pickle=True)

        # run some stats etc on the data to compare things
        run_number = 1

        eyedata1 = copy.deepcopy(all_combined_data[run_number]['eye_data'])
        bolddata1 = copy.deepcopy(all_combined_data[run_number]['bold'])
        intensity1 = copy.deepcopy(all_combined_data[run_number]['intensity'])
        unpleasantness1 = copy.deepcopy(all_combined_data[run_number]['unpleasantness'])
        nclusters_total, tsize = np.shape(bolddata1)

        # resample eye data to match sampling rate of BOLD data
        eye_tsize, ss = np.shape(eyedata1)
        p_engaged = np.zeros(tsize)
        for tt in range(tsize):
            t1 = np.floor(tt*BOLD_TR/eye_TR).astype(int)
            t2 = np.floor((tt+1)*BOLD_TR/eye_TR).astype(int)

            # eyedata_resampled[tt,:] = np.mean(eyedata1[t1:t2,:],axis=0)

            xpos = copy.deepcopy(eyedata1[t1:t2,1]) * dx / screenv   # vertical
            ypos = copy.deepcopy(eyedata1[t1:t2,2]) * dy / screenh   # horizontal

            # windownum = 102
            # plt.close(windownum)
            # fig = plt.figure(windownum)
            # plt.imshow(refdisplay)
            # plt.plot(xpos, ypos, '-r')

            npoints = t2-t1

            # proportion of time within the screen
            c_onscreen = np.where((xpos > xlimits[0]) & (xpos < xlimits[1]) & (ypos > ylimits[0]) & (ypos < ylimits[1]))[0]
            p_onscreen = len(c_onscreen)/npoints

            # proportion of time moving
            mx = xpos[1:] - xpos[:-1]
            my = ypos[1:] - ypos[:-1]
            v = np.sqrt(mx**2 + my**2)
            c_moving = np.where(v > 2)[0]
            p_moving = len(c_moving)/(npoints-1)

            # proportion of time eyes moving over a focussed area
            x = np.round(xpos).astype(int)  # vertical
            y = np.round(ypos).astype(int)  # horizontal
            x[x < 0] = 0
            x[x >= dx] = dx-1
            y[y < 0] = 0
            y[y >= dy] = dy-1
            zoneval = zones[x,y]
            c_inzone = np.where(zoneval > 0.5)[0]
            p_inzone = len(c_inzone)/(npoints-1)

            c_movinginzone = np.where((zoneval[1:] > 0.5) & (v > 2))[0]
            p_movinginzone = len(c_movinginzone)/(npoints-1)

            p_engaged[tt] = copy.deepcopy(p_movinginzone)

        correlation_vals = np.zeros(nclusters_total)
        for nn in range(nclusters_total):
            R = np.corrcoef(p_engaged, bolddata1[nn,:])
            correlation_vals[nn] = R[0,1]
        ss = np.argsort(np.abs(correlation_vals))[::-1]

        print('strongest correlation between engagement and BOLD response is {:.3f} in region {}'.format(correlation_vals[ss[0]], ss[0]))

        windownum = 100
        plt.close(windownum)
        fig = plt.figure(windownum)
        b = copy.deepcopy(bolddata1[ss[0],:])
        b /= np.max(np.abs(b))
        plt.plot(list(range(tsize)), b, '-r')
        plt.plot(list(range(tsize)), p_engaged, '-xb')








    # display pupil size variations averaged over groups
    eyedataname = os.path.join(datadir, 'eyetrackingdata_July2025.npy')
    grouplist = ['38C', '46C', '51C']
    groupcolor = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    excludelist = [449]
    marktimes = [[60,70],[120,150]]
    markcolor = [[1, 1, 0], [0, 1, 0]]

    load_group_pupilsize_fast(eyedataname, grouplist, groupcolor, excludelist, plotmode = 'multiple', maxtimelimit=270000.,
                              windownum=77, marktimes = marktimes, markcolor = markcolor)


    # datafolder = r'Y:\JM2024\EDF_eyetrackingfiles'
    # for root, dirs, files in os.walk(datafolder):
    #     for filename in files:
    #         fullname = os.path.join(root,filename)
    #         f,e = os.path.splitext(filename)
    #         if e == '.npy':
    #             data = np.load(fullname, allow_pickle=True).flat[0]
    #             data['studygroup'] = copy.deepcopy(data['temperature'])
    #             np.save(fullname, data)
    #             print('updated {}'.format(fullname))



def analyze_pupil_size():
    windownum = 20
    resultsdir = r'Y:\FMstudy2023_analysis'

    grouplist = ['FMfast', 'FMslow', 'HCfast', 'HCslow']
    colorlist = [[1, 0, 0], [0.4, 0, 0], [0, 0, 1], [0, 0, 0.4]]

    scalefactor = 1000./2.   # time in milliseconds, with 2 msec per point
    baseline1 = [10., 120.] # time span in seconds
    baseline2 = [160., 260.] # time span in seconds
    stimperiod = [120., 150.] # time span in seconds

    pointspan1 = list((np.array(baseline1)*scalefactor).astype(int))
    pointspan2 = list((np.array(baseline2)*scalefactor).astype(int))
    stimspan = list((np.array(stimperiod)*scalefactor).astype(int))

    for gggg in range(len(grouplist)):
        group = copy.deepcopy(grouplist[gggg])
        groupdataname = os.path.join(resultsdir, group + '_eyetracking_groupdata.npy')

        groupdata = np.load(groupdataname, allow_pickle=True).flat[0]

        group_eye_data = copy.deepcopy(groupdata['group_eye_data'])
        intensity = copy.deepcopy(groupdata['intensity'])
        unpleasantness = copy.deepcopy(groupdata['unpleasantness'])
        nruns = np.shape(group_eye_data)[1]

        baseline_values = np.zeros(nruns)
        stim_values = np.zeros(nruns)
        for nn in range(nruns):
            b = np.mean(group_eye_data[pointspan1+pointspan2, nn])
            s = np.mean(group_eye_data[stimspan, nn])
            baseline_values[nn] = copy.deepcopy(b)
            stim_values[nn] = copy.deepcopy(s)

        baseline_values = np.array(baseline_values)
        stim_values = np.array(stim_values)
        stimdilation = stim_values - baseline_values

        stimdilation_corr = np.corrcoef(stimdilation, intensity)
        baseline_corr = np.corrcoef(baseline_values, intensity)

        window1 = windownum + 2*gggg
        plt.close(window1)
        fig = plt.figure(window1, figsize = (8,4))
        plt.plot(intensity, baseline_values, marker = 'o', linestyle = 'none', color = colorlist[gggg])
        x1 = np.min(intensity)
        x2 = np.max(intensity)
        y1 = np.min(baseline_values)
        y2 = np.max(baseline_values)
        R = np.corrcoef(intensity, baseline_values)
        plt.annotate('{}  baseline vs pain intensity'.format(group), xy = (x1,y2))
        plt.annotate('R = {:.4f}'.format(R[0,1]), xy = (x1,y2-300))
        plt.annotate('intensity = {:.3f} {} {:.3f}'.format(np.mean(intensity),chr(177),np.std(intensity)), xy = (x1,y2-600))
        plt.annotate('baseline = {:.1f} {} {:.1f}'.format(np.mean(baseline_values),chr(177),np.std(baseline_values)), xy = (x1,y2-900))
        m, b = np.polyfit(intensity, baseline_values, 1)
        plt.plot([x1,x2],[m*x1+b, m*x2+b], '-k')
        figsavename = os.path.join(resultsdir, '{}_baseline_vs_intensity.svg'.format(group))
        plt.savefig(figsavename)

        window1 = windownum + 2*gggg + 1
        plt.close(window1)
        fig = plt.figure(window1, figsize = (8,4))
        plt.plot(intensity, stimdilation, marker = 'o', linestyle = 'none', color = colorlist[gggg])
        x1 = np.min(intensity)
        x2 = np.max(intensity)
        y1 = np.min(stimdilation)
        y2 = np.max(stimdilation)
        R = np.corrcoef(intensity, stimdilation)
        plt.annotate('{}  stimdilation vs pain intensity'.format(group), xy = (x1,y2))
        plt.annotate('R = {:.4f}'.format(R[0,1]), xy = (x1,y2-300))
        plt.annotate('intensity = {:.3f} {} {:.3f}'.format(np.mean(intensity),chr(177),np.std(intensity)), xy = (x1,y2-600))
        plt.annotate('stimdilation = {:.1f} {} {:.1f}'.format(np.mean(stimdilation),chr(177),np.std(stimdilation)), xy = (x1,y2-900))
        m, b = np.polyfit(intensity, stimdilation, 1)
        plt.plot([x1,x2],[m*x1+b, m*x2+b], '-k')
        figsavename = os.path.join(resultsdir, '{}_stimdilation_vs_intensity.svg'.format(group))
        plt.savefig(figsavename)



def load_group_pupilsize_fast(eyedataname, grouplist, groupcolor, excludelist, plotmode = 'multiple', maxtimelimit = 270000.,
                              windownum = 77, marktimes = [], markcolor = []):
    alldata = np.load(eyedataname, allow_pickle=True)
    p,f = os.path.split(eyedataname)

    # setup plot window
    plt.close(windownum)
    Ngroups = len(grouplist)
    if plotmode == 'single':
        fig,ax = plt.subplots(1, 1, sharex = 'all', sharey = 'all', num = windownum, figsize = (8,4))
    else:
        fig,ax = plt.subplots(Ngroups, 1, sharex = 'all', sharey = 'all', num = windownum, figsize = (8,10))

    for gg in range(len(grouplist)):
        groupname = copy.deepcopy(grouplist[gg])
        output_group_dataname = os.path.join(p,'pupildata_{}.npy'.format(groupname))
        groupcount = 0
        avgdatacount = 0
        for nn in range(len(alldata)):
            groupcheck = alldata[nn]['studygroup'] == groupname
            dbnumcheck = alldata[nn]['dbnum'] in excludelist
            eyedata = copy.deepcopy(alldata[nn]['eyedata'])
            rate = copy.deepcopy(alldata[nn]['rate'])

            dbnum = copy.deepcopy(alldata[nn]['dbnum'])
            studygroup = copy.deepcopy(alldata[nn]['studygroup'])
            patientid = copy.deepcopy(alldata[nn]['patientid'])
            seriesnumber = copy.deepcopy(alldata[nn]['seriesnumber'])

            try:
                mincheck = np.min(eyedata)
            except:
                mincheck = -1.0e10

            if (groupcheck == True) and (dbnumcheck == False) and (mincheck > -5000):
                pupildata = copy.deepcopy(eyedata[:,3])

                if groupcount == 0:
                    npoints1, nd1 = np.shape(eyedata)

                    maxtimestamp = np.max(eyedata[:, 0])
                    maxtimeindex = np.argmax(eyedata[:, 0])
                    timestamp = eyedata[:maxtimeindex, 0]
                    pupilsize = eyedata[:maxtimeindex, 3]
                    pupilsize[pupilsize < 0] = 0.

                    maxinterptime = np.min([maxtimestamp, maxtimelimit]).astype(int)
                    newtime = np.arange(0, maxinterptime, 2)
                    f = interpolate.interp1d(timestamp, pupilsize, fill_value='extrapolate')
                    pupilinterp = f(newtime)

                    if maxinterptime == maxtimelimit:
                        datacount = np.ones(int(maxtimelimit / 2))
                        pupilsize = pupilinterp

                        group_eye_data = pupilsize[:, np.newaxis]
                        avgdatacount = datacount
                        npointslist = [npoints1]
                        ratelist = [rate]

                        dbnum_list = [dbnum]
                        studygroup_list = [studygroup]
                        patientid_list = [patientid]
                        seriesnumber_list = [seriesnumber]

                        groupcount += 1

                else:
                    npoints, nd = np.shape(group_eye_data)
                    npoints1, nd1 = np.shape(eyedata)
                    npointslist += [npoints1]

                    maxtimestamp = np.max(eyedata[:, 0])
                    maxtimeindex = np.argmax(eyedata[:, 0])
                    timestamp = eyedata[:maxtimeindex, 0]
                    pupilsize = eyedata[:maxtimeindex, 3]
                    pupilsize[pupilsize < 0] = 0.

                    maxinterptime = np.min([maxtimestamp, maxtimelimit]).astype(int)
                    newtime = np.arange(0, maxinterptime, 2)
                    f = interpolate.interp1d(timestamp, pupilsize, fill_value='extrapolate')
                    pupilinterp = f(newtime)

                    if maxinterptime == maxtimelimit:
                        datacount = np.ones(int(maxtimelimit / 2))
                        pupilsize = copy.deepcopy(pupilinterp)
                        avgdatacount += datacount
                        group_eye_data = np.concatenate((group_eye_data, pupilsize[:, np.newaxis]), axis=1)

                        dbnum_list += [dbnum]
                        studygroup_list += [studygroup]
                        patientid_list += [patientid]
                        seriesnumber_list += [seriesnumber]

                        groupcount += 1

        if np.max(avgdatacount) > 1:
            avgdata = np.sum(group_eye_data, axis=1) / avgdatacount
            avgdata2 = np.repeat(avgdata[:, np.newaxis], groupcount, axis=1)
            sddata = np.sqrt(np.sum((group_eye_data - avgdata2) ** 2, axis=1) / (avgdatacount - 1))
            semdata = np.sqrt(np.sum((group_eye_data - avgdata2) ** 2, axis=1)) / (avgdatacount - 1)
        else:
            avgdata = np.sum(group_eye_data, axis=1) / avgdatacount
            avgdata2 = np.repeat(avgdata[:, np.newaxis], groupcount, axis=1)
            sddata = np.sqrt(np.sum((group_eye_data - avgdata2) ** 2, axis=1) / (avgdatacount + 1.0e-20))
            semdata = np.sqrt(np.sum((group_eye_data - avgdata2) ** 2, axis=1)) / (avgdatacount + 1.0e-20)

        print('found {} data sets'.format(np.max(avgdatacount).astype(int)))
        all_group_data = {'group_eye_data': group_eye_data, 'avgdata': avgdata, 'sddata': sddata,
                          'semdata': semdata, 'dbnum': dbnum_list,
                          'studygroup': studygroup_list, 'patientid': patientid_list,
                          'seriesnumber': seriesnumber_list}

        np.save(output_group_dataname, all_group_data)

        newtime = np.arange(0, maxtimelimit, 2)

        if plotmode == 'single':
            ax.plot(newtime, avgdata, marker=None, linestyle='-', linewidth=1, color=groupcolor[gg])
        else:
            ax[gg].plot(newtime, avgdata, marker=None, linestyle='-', linewidth=1, color=groupcolor[gg])
        yps = avgdata + semdata
        yms = avgdata - semdata
        xx = np.concatenate((newtime, newtime[::-1]))
        yy = np.concatenate((yps, yms[::-1]))
        # ymax = np.max([ymax, np.max(yy)])
        if plotmode == 'single':
            ax.fill(xx, yy, color=groupcolor[gg], alpha=0.1)
        else:
            ax[gg].fill(xx, yy, color=groupcolor[gg], alpha=0.1)

        if len(marktimes) > 0:
            ymax = np.max(yps)
            for mm in range(len(marktimes)):
                x1 = marktimes[mm][0]*1000
                x2 = marktimes[mm][1]*1000
                xx = [x1,x2,x2,x1]
                yy = [0,0,ymax,ymax]
                if plotmode == 'single':
                    ax.fill(xx, yy, color=markcolor[mm], alpha=0.1)
                else:
                    ax[gg].fill(xx, yy, color=markcolor[mm], alpha=0.1)

        if plotmode == 'single':
            ax.set_ylim([0, None])
            ax.annotate('{}'.format(groupname), (20000 + gg*3000,0))
        else:
            ax[gg].set_ylim([0, None])
            ax[gg].annotate('{}'.format(groupname), (20000,0))


    plotsavename = 'pupilsize_'
    for name in grouplist:
        plotsavename += (name + '_')
    if plotmode == 'single':
        plotsavename = plotsavename[:-1] + '_single.svg'
    else:
        plotsavename = plotsavename[:-1] + '.svg'
    plotsavename = os.path.join(p,plotsavename)

    plt.savefig(plotsavename)



def load_group_pupilsize():
    outputname = r'Y:\FMstudy2023_analysis\eyetrackingdata_Feb2025.npy'
    resultsdir = r'Y:\FMstudy2023_analysis'
    DBname = r'Y:\FMstudy2023\PS2023_database_corrected_Jan2025B.xlsx'
    grouplist = ['FMslow', 'FMfast']
    colorlist = [[0.4, 0, 0], [1, 0, 0]]

    # print('finished loading data set {} of {}     {}'.format(nn, num, time.ctime()))
    # entry1 = {'eyedata': eyedata, 'starttime': starttime, 'rate': rate, 'dbnum': nn, 'studygroup': studygroup,
    #           'patientid': patientid, 'seriesnumber': seriesnumber}

    entry2 = dict(zip(datafields, datavalues))
    entry = dict(entry1, **entry2)

    excludelist = [278, 377, 521, 522, 523] + list(range(479, 489)) + list(
        range(513, 523))  # PS2023_58 and PS2023_64 are bad data sets

    data = np.load(outputname, allow_pickle=True)
    print('finished loading data ...')

    xls = pd.ExcelFile(DBname, engine='openpyxl')
    xls_sheets = xls.sheet_names
    df1 = pd.read_excel(xls, 'datarecord')

    # grouplist = ['FMfast', 'FMslow', ]
    # colorlist = [[1, 0, 0], [0.4, 0, 0]]
    #
    # grouplist = ['HCfast', 'HCslow']
    # colorlist = [[0, 0, 1], [0, 0, 0.4]]
    #
    # grouplist = ['HCslow', 'HCfast']
    # colorlist = [[0, 0, 0.4], [0, 0, 1]]
    #
    # grouplist = ['HCfast', 'FMfast']
    # colorlist = [[0, 0, 1], [1, 0, 0]]

    # grouplist = ['HCslow', 'FMslow']
    # colorlist = [[0, 0, 0.4], [0.4, 0, 0]]

    plotsavename = '{}_{}_comparison_pupilarea.svg'.format(grouplist[0], grouplist[1])
    plotsavename = os.path.join(resultsdir, plotsavename)

    windownum = 33
    show_plot = True
    save_groupdata = False

    if show_plot:
        plt.close(windownum)
        fig = plt.figure(windownum, figsize=(8, 4))

    ymax = 0.
    for gggg in range(len(grouplist)):
        group = copy.deepcopy(grouplist[gggg])
        groupdataname = os.path.join(resultsdir, group + '_eyetracking_groupdata.npy')
        maxtimelimit = 270000.
        pltcolor = copy.deepcopy(colorlist[gggg])

        group_eye_data = []
        groupcount = 0

        npointslist = []
        n1 = 0
        n2 = np.floor(0.50 * len(data)).astype(int)
        n3 = np.floor(0.75 * len(data)).astype(int)
        n4 = len(data)
        nnlist = list(range(n1, n4))

        for bb in excludelist:
            try:
                nnlist.remove(bb)  # problem entry for FMfast
                print('{} removed'.format(bb))
            except:
                print('{} not in list'.format(bb))

        for nn in nnlist:
            sg = data[nn]['studygroup']
            dbnum = data[nn]['dbnum']

            studygroup = copy.deepcopy(df1.loc[dbnum, 'studygroup'])
            patientid = copy.deepcopy(df1.loc[dbnum, 'patientid'])
            seriesnumber = copy.deepcopy(df1.loc[dbnum, 'seriesnumber']).astype(int)

            # painrating = copy.deepcopy(df1.loc[dbnum, 'intensitypainrating']).astype(float)
            # temperature = copy.deepcopy(df1.loc[dbnum, 'temperature']).astype(float)
            # intensity = copy.deepcopy(df1.loc[dbnum, 'intensitypainrating']).astype(float)
            # unpleasantness = copy.deepcopy(df1.loc[dbnum, 'unpleasantnesspainrating']).astype(float)
            # BDI = copy.deepcopy(df1.loc[dbnum, 'BDI']).astype(float)
            # STAIS = copy.deepcopy(df1.loc[dbnum, 'STAIS']).astype(float)
            # STAIT = copy.deepcopy(df1.loc[dbnum, 'STAIT']).astype(float)
            # PCS = copy.deepcopy(df1.loc[dbnum, 'PCS']).astype(float)

            datacheck1 = (patientid == data[nn]['patientid'])
            datacheck2 = (seriesnumber == data[nn]['seriesnumber'])
            if not datacheck1:
                print('ERROR!   patientid is not consistent!   nn = {}   dbnum = {}'.format(nn, dbnum))
            if not datacheck2:
                print('ERROR!   seriesnumber is not consistent!   nn = {}   dbnum = {}'.format(nn, dbnum))

            if np.ndim(data[nn]['eyedata']) == 3:
                eyedata = np.array(data[nn]['eyedata'])[0, :, :]
            else:
                eyedata = np.array(data[nn]['eyedata'])
            rate = np.array(data[nn]['rate'])
            try:
                mincheck = np.min(eyedata)
            except:
                mincheck = -1.0e10
            # print('{}  minval = {}'.format(nn,mincheck))
            if (sg == group) & (np.shape(eyedata)[0] > 0) & (mincheck > -5000):
                if groupcount == 0:
                    npoints1, nd1 = np.shape(eyedata)

                    maxtimestamp = np.max(eyedata[:, 0])
                    maxtimeindex = np.argmax(eyedata[:, 0])
                    timestamp = eyedata[:maxtimeindex, 0]
                    pupilsize = eyedata[:maxtimeindex, 3]
                    pupilsize[pupilsize < 0] = 0.

                    maxinterptime = np.min([maxtimestamp, maxtimelimit]).astype(int)
                    newtime = np.arange(0, maxinterptime, 2)
                    f = interpolate.interp1d(timestamp, pupilsize, fill_value='extrapolate')
                    pupilinterp = f(newtime)

                    if maxinterptime == maxtimelimit:
                        datacount = np.ones(int(maxtimelimit / 2))
                        pupilsize = pupilinterp

                        group_eye_data = pupilsize[:, np.newaxis]
                        avgdatacount = datacount
                        npointslist = [npoints1]
                        ratelist = [rate]

                        # intensity_list = [intensity]
                        # unpleasantness_list = [unpleasantness]
                        # BDI_list = [BDI]
                        # STAIS_list = [STAIS]
                        # STAIT_list = [STAIT]
                        # PCS_list = [PCS]

                        dbnum_list = [dbnum]
                        studygroup_list = [studygroup]
                        patientid_list = [patientid]
                        seriesnumber_list = [seriesnumber]

                        groupcount += 1

                else:
                    npoints, nd = np.shape(group_eye_data)
                    npoints1, nd1 = np.shape(eyedata)
                    npointslist += [npoints1]

                    maxtimestamp = np.max(eyedata[:, 0])
                    maxtimeindex = np.argmax(eyedata[:, 0])
                    timestamp = eyedata[:maxtimeindex, 0]
                    pupilsize = eyedata[:maxtimeindex, 3]
                    pupilsize[pupilsize < 0] = 0.

                    maxinterptime = np.min([maxtimestamp, maxtimelimit]).astype(int)
                    newtime = np.arange(0, maxinterptime, 2)
                    f = interpolate.interp1d(timestamp, pupilsize, fill_value='extrapolate')
                    pupilinterp = f(newtime)

                    if maxinterptime == maxtimelimit:
                        datacount = np.ones(int(maxtimelimit / 2))
                        pupilsize = copy.deepcopy(pupilinterp)
                        avgdatacount += datacount
                        group_eye_data = np.concatenate((group_eye_data, pupilsize[:, np.newaxis]), axis=1)

                        # intensity_list += [intensity]
                        # unpleasantness_list += [unpleasantness]
                        # BDI_list += [BDI]
                        # STAIS_list += [STAIS]
                        # STAIT_list += [STAIT]
                        # PCS_list += [PCS]

                        dbnum_list += [dbnum]
                        studygroup_list += [studygroup]
                        patientid_list += [patientid]
                        seriesnumber_list += [seriesnumber]

                        groupcount += 1

        if np.max(avgdatacount) > 1:
            avgdata = np.sum(group_eye_data, axis=1) / avgdatacount
            avgdata2 = np.repeat(avgdata[:, np.newaxis], groupcount, axis=1)
            sddata = np.sqrt(np.sum((group_eye_data - avgdata2) ** 2, axis=1) / (avgdatacount - 1))
            semdata = np.sqrt(np.sum((group_eye_data - avgdata2) ** 2, axis=1)) / (avgdatacount - 1)
        else:
            avgdata = np.sum(group_eye_data, axis=1) / avgdatacount
            avgdata2 = np.repeat(avgdata[:, np.newaxis], groupcount, axis=1)
            sddata = np.sqrt(np.sum((group_eye_data - avgdata2) ** 2, axis=1) / (avgdatacount + 1.0e-20))
            semdata = np.sqrt(np.sum((group_eye_data - avgdata2) ** 2, axis=1)) / (avgdatacount + 1.0e-20)

        print('found {} data sets'.format(np.max(avgdatacount).astype(int)))
        all_group_data = {'group_eye_data': group_eye_data, 'avgdata': avgdata, 'sddata': sddata, 'semdata': semdata,
                          'intensity': intensity_list, 'unpleasantness': unpleasantness_list, 'BDI': BDI_list,
                          'STAIS': STAIS_list, 'STAIT': STAIT_list, 'PCS': PCS_list, 'dbnum': dbnum_list,
                          'studygroup': studygroup_list, 'patientid': patientid_list, 'seriesnumber': seriesnumber_list}

        if save_groupdata:
            np.save(groupdataname, all_group_data)

        if show_plot:
            ymax = 3000
            newtime = np.arange(0, maxtimelimit, 2)
            plt.plot(newtime, avgdata, linestyle='-', linewidth=2, marker='', color=pltcolor)
            yps = avgdata + semdata
            yms = avgdata - semdata
            xx = np.concatenate((newtime, newtime[::-1]))
            yy = np.concatenate((yps, yms[::-1]))
            # ymax = np.max([ymax, np.max(yy)])
            plt.fill(xx, yy, color=pltcolor, alpha=0.1)
            plt.ylim([0, ymax])

    if show_plot:
        plt.savefig(plotsavename)




def load_group_gazeposition():
    outputname = r'Z:\FMstudy2023\eyetrackingdata.npy'
    DBname = r'Z:\FMstudy2023\PS2023_database_corrected.xlsx'
    covariatename = 'painintensityrating'

    screenh = 1920
    screenv = 1080
    hcenter = 1920/2.0
    vcenter = 1080/2.0
    hrange = 0.5
    vrange = 0.5

    data = np.load(outputname, allow_pickle=True)
    print('finished loading data ...')

    group_horpos = []
    group_vertpos = []
    groupcount = 0

    group = 'HCslow'
    maxtimelimit = 270000.
    show_plot = True
    windownum = 21
    pltcolor = [0,0,1]

    # get covariate
    xls = pd.ExcelFile(DBname, engine='openpyxl')
    xls_sheets = xls.sheet_names

    df1 = pd.read_excel(xls, 'datarecord')
    keylist = df1.keys()
    for kname in keylist:
        if 'Unnamed' in kname: df1.pop(kname)  # remove blank fields from the database

    num = len(df1)  # number of entries
    covdata = []
    loadcount = 0
    for nn in range(num):
        datadir = copy.deepcopy(df1.loc[nn, 'datadir'])
        pname = copy.deepcopy(df1.loc[nn, 'pname'])
        patientgroup = copy.deepcopy(df1.loc[nn, 'patientgroup'])
        patientid = copy.deepcopy(df1.loc[nn, 'patientid'])
        seriesnumber = copy.deepcopy(df1.loc[nn, 'seriesnumber']).astype(int)
        covvals = copy.deepcopy(df1.loc[nn, covariatename])
        covdata.append({'dbnum':nn, 'patientid':patientid, 'pname':pname, 'seriesnumber':seriesnumber, covariatename:covvals, 'studygroup':patientgroup})

    npointslist = []
    nnlist = list(range(len(data)))
    nnlist.remove(268)  # problem entry for FMfast
    painlist = []
    for nn in nnlist:
        sg = data[nn]['studygroup']
        pid = data[nn]['patientid']
        sn = data[nn]['seriesnumber']

        if np.ndim(data[nn]['eyedata']) == 3:
            eyedata = np.array(data[nn]['eyedata'])[0,:,:]
        else:
            eyedata = np.array(data[nn]['eyedata'])
        rate = np.array(data[nn]['rate'])

        if (sg == group) & (np.shape(eyedata)[0] > 0):
            check = [(covdata[xx]['patientid'] == pid) & (covdata[xx]['seriesnumber'] == sn) for xx in
                     range(len(covdata))]
            c = np.where(check)[0]
            covvals = covdata[c[0]][covariatename]
            print('{} series {}   covariate value {}'.format(pid, sn, covvals))

            if groupcount == 0:
                npoints1, nd1 = np.shape(eyedata)

                maxtimestamp = np.max(eyedata[:, 0])
                maxtimeindex = np.argmax(eyedata[:, 0])
                timestamp = eyedata[:maxtimeindex, 0]
                horpos = eyedata[:maxtimeindex, 1]
                vertpos = eyedata[:maxtimeindex, 2]
                horpos[horpos < 0] = 0.
                vertpos[vertpos < 0] = 0.

                maxinterptime = np.min([maxtimestamp, maxtimelimit]).astype(int)
                newtime = np.arange(0, maxinterptime, 2)
                f = interpolate.interp1d(timestamp, horpos, fill_value = 'extrapolate')
                horposinterp = f(newtime)
                f = interpolate.interp1d(timestamp, vertpos, fill_value = 'extrapolate')
                vertposinterp = f(newtime)


                if maxinterptime == maxtimelimit:
                    datacount = np.ones(int(maxtimelimit/2))
                    horpos = horposinterp
                    vertpos = vertposinterp

                    group_horpos = horpos[:,np.newaxis]
                    group_vertpos = vertpos[:,np.newaxis]
                    avgdatacount = datacount
                    npointslist = [npoints1]
                    ratelist = [rate]
                    covlist = [covvals]
                    groupcount += 1

            else:
                npoints,nd = np.shape(group_horpos)
                npoints1, nd1 = np.shape(eyedata)
                npointslist += [npoints1]

                maxtimestamp = np.max(eyedata[:,0])
                maxtimeindex = np.argmax(eyedata[:,0])
                timestamp = eyedata[:maxtimeindex,0]
                horpos = eyedata[:maxtimeindex, 1]
                vertpos = eyedata[:maxtimeindex, 2]
                horpos[horpos < 0] = 0.
                vertpos[vertpos < 0] = 0.

                maxinterptime = np.min([maxtimestamp, maxtimelimit]).astype(int)
                newtime = np.arange(0,maxinterptime,2)
                f = interpolate.interp1d(timestamp,horpos, fill_value = 'extrapolate')
                horposinterp = f(newtime)
                f = interpolate.interp1d(timestamp,vertpos, fill_value = 'extrapolate')
                vertposinterp = f(newtime)

                if maxinterptime == maxtimelimit:
                    datacount = np.ones(int(maxtimelimit/2))
                    horpos = copy.deepcopy(horposinterp)
                    vertpos = copy.deepcopy(vertposinterp)
                    avgdatacount += datacount
                    group_horpos = np.concatenate((group_horpos, horpos[:,np.newaxis]), axis = 1)
                    group_vertpos = np.concatenate((group_vertpos, vertpos[:,np.newaxis]), axis = 1)
                    covlist += [covvals]

                    groupcount += 1

    covlist = np.array(covlist)
    attentionlist = []
    nt, ng = np.shape(group_horpos)
    stimperiod = [120, 150]
    s1 = stimperiod[0]*500
    s2 = stimperiod[1]*500
    for aa in range(ng):
        hmin = hcenter - (screenh*hrange)/2.0
        hmax = hcenter + (screenh*hrange)/2.0
        vmin = vcenter - (screenv*vrange)/2.0
        vmax = vcenter + (screenv*vrange)/2.0

        hpos = group_horpos[s1:s2,aa]
        vpos = group_vertpos[s1:s2,aa]
        attention_flag = (hpos > hmin) & (hpos < hmax) & (vpos > vmin) & (vpos < vmax)
        check = np.where(attention_flag)[0]
        nvals = s2-s1
        nattention = len(check)
        attention_ratio = nattention/nvals
        attentionlist += [attention_ratio]


    if show_plot:
        windownum = 101
        plt.close(windownum)
        fig = plt.figure(windownum)
        plt.plot(attentionlist, covlist, 'ok')



def load_all_eyetracking(DBname, datafields, datanametag, outputname):
    # outputname = r'Y:\FMstudy2023\eyetrackingdata_Feb2025.npy'
    # DBname = r'Y:\FMstudy2023\PS2023_database_corrected_Jan2025B.xlsx'
    # datafields = ['temperature', 'intensitypainrating', 'unpleasantnesspainrating', 'BDI', 'STAIS', 'STAIT', 'PCS']
    # datanametag = 'S23'

    beginloadtime = time.time()
    resume = True
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
        studygroup = copy.deepcopy(df1.loc[nn, 'studygroup'])
        patientid = copy.deepcopy(df1.loc[nn, 'patientid'])
        seriesnumber = copy.deepcopy(df1.loc[nn, 'seriesnumber']).astype(int)

        datavalues = []
        for xx in range(len(datafields)):
            datavalues += [copy.deepcopy(df1.loc[nn, datafields[xx]])]

        try:
            filename = datanametag + patientid[-2:] + 's' + '{:02d}'.format(seriesnumber) + '.asc'
            filename_alt = datanametag + patientid[-2:] + 's' + '{}'.format(seriesnumber) + '.asc'
        except:
            filename = 'notdefined'
            filename_alt = 'notdefined'

        pname1 = copy.deepcopy(pname)
        splitpath = pname.split(os.sep)
        pname2 = copy.deepcopy(splitpath[0])

        etname = 'notdefined'
        etname_alt = 'notdefined'

        tagdir = r'ASCII'
        etdir = 'EDF_eyetrackingfiles'

        check1 = os.path.join(datadir, etdir, filename)
        check1_alt = os.path.join(datadir, etdir, filename_alt)
        if os.path.isfile(check1):
            etname = copy.deepcopy(check1)
            etname_alt = copy.deepcopy(check1_alt)

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
                entry['dbnum'] = copy.deepcopy(nn)
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
                    entry1 = {'eyedata':eyedata, 'starttime':starttime, 'rate':rate, 'dbnum':nn, 'studygroup':studygroup,
                              'patientid':patientid, 'seriesnumber':seriesnumber}

                    entry2 = dict(zip(datafields, datavalues))
                    entry = dict(entry1, **entry2)

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



def show_eyetracking(personnum, DBname, datanametag, outputname, windownum = 10):

    # outputname = r'Z:\FMstudy2023\eyetrackingdata.npy'
    # DBname = r'Y:\FMstudy2023\PS2023_database_corrected_Jan2025B.xlsx'
    # datanametag = 'S23'

    nn = personnum
    beginloadtime = time.time()
    xls = pd.ExcelFile(DBname, engine='openpyxl')
    xls_sheets = xls.sheet_names

    df1 = pd.read_excel(xls, 'datarecord')
    keylist = df1.keys()
    for kname in keylist:
        if 'Unnamed' in kname: df1.pop(kname)  # remove blank fields from the database

    datadir = copy.deepcopy(df1.loc[nn, 'datadir'])
    pname = copy.deepcopy(df1.loc[nn, 'pname'])
    patientgroup = copy.deepcopy(df1.loc[nn, 'patientgroup'])
    patientid = copy.deepcopy(df1.loc[nn, 'patientid'])
    seriesnumber = copy.deepcopy(df1.loc[nn, 'seriesnumber']).astype(int)

    # temperature = copy.deepcopy(df1.loc[nn, 'temperature']).astype(int)
    # intensity = copy.deepcopy(df1.loc[nn, 'intensitypainrating']).astype(int)
    # unpleasantness = copy.deepcopy(df1.loc[nn, 'unpleasantnesspainrating']).astype(int)
    # BDI = copy.deepcopy(df1.loc[nn, 'BDI']).astype(int)
    # STAIS = copy.deepcopy(df1.loc[nn, 'STAIS']).astype(int)
    # STAIT = copy.deepcopy(df1.loc[nn, 'STAIT']).astype(int)
    # PCS = copy.deepcopy(df1.loc[nn, 'PCS']).astype(int)

    etdir = 'EDF_eyetrackingfiles'

    # try:
    #     etdir_subdir = copy.deepcopy(pname)
    #     etdir_subdir = 'ET_' + (etdir_subdir.lower()).capitalize()
    # except:
    #     etdir_subdir = 'notdefined'
    # tagdir = r'ASCII'

    try:
        filename = datanametag + patientid[-2:] + 's' + '{:02d}'.format(seriesnumber) + '.asc'
        filename_alt = datanametag + patientid[-2:] + 's' + '{}'.format(seriesnumber) + '.asc'
    except:
        filename = 'notdefined'
        filename_alt = 'notdefined'

    try:
        etname = os.path.join(datadir, etdir, filename)
        etname_alt = os.path.join(datadir, etdir, filename_alt)
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
            eyedata = entry['eyedata']
        else:
            eyedata, starttime, rate = read_eyedata_ascii(etname)
            if starttime < 0:  # error
                loadsuccess = False
                print('ERROR:  could not load {}'.format(etname))
            else:
                loadsuccess = True
                timestamp = eyedata[:, 0]
                ct = np.where(timestamp >= 0)[0]
                eyedata = eyedata[ct, :]
                print('finished loading data set {}     {}'.format(nn, time.ctime()))
                entry = {'eyedata': eyedata, 'starttime': starttime, 'rate': rate, 'dbnum': nn,
                         'studygroup': patientgroup, 'patientid': patientid, 'seriesnumber': seriesnumber}

                np.save(npname, entry)
                print('saved {}'.format(npname))

        if loadsuccess:
            # data.append(entry)
            # loadcount += 1
            TL = time.time() - beginloadtime
            h = np.floor(TL / 3600.).astype(int)
            m = np.floor((TL % 3600) / 60.).astype(int)
            s = np.floor(TL % 60).astype(int)
            print('time lapsed:  {} hours {} minutes {} seconds'.format(h, m, s))


    tt = eyedata[:, 0]
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
