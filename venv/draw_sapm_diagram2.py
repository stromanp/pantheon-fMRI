# functions to draw and display connectivity diagrams from SAPM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import os
import pandas as pd
import scipy.stats as stats
import pydisplay

#-----------------------------------------------------------------------
# get info about the network to work with-------------------------------
# outputdir = r'D:\threat_safety_python\individual_differences\fixed_C6RD3'
# SEMresultsname = os.path.join(outputdir, 'SEMphysio_model.npy')
# SEMparametersname = os.path.join(outputdir, 'SEMparameters_model5.npy')
# networkfile = r'D:/threat_safety_python/network_model_5cluster_v5_w_3intrinsics.xlsx'
#
# SEMparams = np.load(SEMparametersname, allow_pickle=True).flat[0]
# network = SEMparams['network']
# beta_list = SEMparams['beta_list']
# betanamelist = SEMparams['betanamelist']
# nruns_per_person = SEMparams['nruns_per_person']
# rnamelist = SEMparams['rnamelist']
# fintrinsic_count = SEMparams['fintrinsic_count']
# fintrinsic_region = SEMparams['fintrinsic_region']
# vintrinsic_count = SEMparams['vintrinsic_count']
# nclusterlist = SEMparams['nclusterlist']
# tplist_full = SEMparams['tplist_full']
# tcdata_centered = SEMparams['tcdata_centered']
# ctarget = SEMparams['ctarget']
# csource = SEMparams['csource']
# tsize = SEMparams['tsize']
# timepoint = SEMparams['timepoint']
# epoch = SEMparams['epoch']
# Nintrinsic = fintrinsic_count + vintrinsic_count
#
# # load the SEM results
# SEMresults_load = np.load(SEMresultsname, allow_pickle=True)
#-----------------------------------------------------------------------

def create_sapm_plots(studynum, type, cord_cluster, figurenumber = 0):
    # (cnums, resultsfile, sheetname, regionnames, statname, drawregionsfile, scalefactor, threshold, clusterdataname,
    #  templatename, outputfolder, figurenumber = 5):

    # studynum = 0
    # type = 'fixed'
    # cord_cluster = 0

    outputfolder = r'E:\beta_distribution'

    studynames = ['allthreat','Low','Sens','RS1nostim', 'all_condition']
    studyname = studynames[studynum]
    drawregionsfile = r'E:\beta_distribution\cord_region_drawing_positions.xlsx'
    resultsfile = 'E:\\beta_distribution\\{}_fixed_C6RD{}\\all All Average Mconn values.xlsx'.format(studyname,cord_cluster)
    sheetname = 'average'
    regionnames = 'regions'
    statname = 'beta'
    scalefactor = 10.0
    threshold = 0.0
    if figurenumber == 0:
        figurenumber = 10+studynum

    outputname = os.path.join(outputfolder, 'SAPMplot_{}_{}_{}{}.svg'.format(studyname, sheetname, type, cord_cluster))

    if type == 'fixed':
        if cord_cluster == 0:
            cnums = [0, 3, 3, 1, 4, 1, 3, 3, 4, 1]  # fixed 0
        if cord_cluster == 1:
            cnums = [1, 3, 3, 1, 3, 1, 3, 3, 2, 1]  # fixed 1
        if cord_cluster == 2:
            cnums = [2, 3, 3, 1, 1, 1, 3, 3, 2, 0]  # fixed 2
        if cord_cluster == 3:
            cnums = [3, 3, 2, 1, 0, 1, 2, 3, 4, 1]  # fixed 3
        if cord_cluster == 4:
            cnums = [4, 3, 3, 1, 0, 1, 2, 3, 4, 3]  # fixed 4
    else:
        if cord_cluster == 0:
            cnums = [0, 4, 4, 2, 2, 3, 3, 2, 3, 1]  # random 0
        if cord_cluster == 1:
            cnums = [1, 4, 2, 3, 2, 1, 1, 2, 3, 0]  # random 1
        if cord_cluster == 2:
            cnums = [2, 2, 2, 0, 0, 2, 0, 3, 1, 3]  # random 2
        if cord_cluster == 3:
            cnums = [3, 3, 1, 4, 4, 1, 3, 3, 1, 0]  # random 3
        if cord_cluster == 4:
            cnums = [4, 4, 2, 1, 0, 3, 3, 3, 2, 0]  # random 4

    regions = define_drawing_regions_from_file(drawregionsfile)
    writefigure = True

    draw_general_sapm_plot(regions, resultsfile, sheetname, regionnames, statname, drawregionsfile, figurenumber,
                           scalefactor, cnums, threshold, outputname, writefigure)

    # make figures of regions
    clusterdataname = r'E:\threat_safety_clusterdata.npy'
    templatename = 'ccbs'
    regionlist = [regions[x]['name'] for x in range(len(regions))]
    draw_sapm_regions(regionlist, clusterdataname, cnums, templatename, outputfolder)



def draw_sapm_regions(regionlist, clusterdataname, cnums, templatename, outputfolder):
    # show axial slices?
    if len(clusterdataname) > 0:
        for rr, regionname in enumerate(regionlist):
            if rr < len(cnums):
                clusternum = cnums[rr]
                outputimg = display_anatomical_slices(clusterdataname, regionname, clusternum, templatename)

                # display it somewhere...
                f = 'region_{}{}.svg'.format(regionname,clusternum)
                svgname = os.path.join(outputfolder, f)
                plt.figure(100+rr)
                plt.imshow(outputimg)
                plt.savefig(svgname, format='svg')
                print('saved figure as {}'.format(svgname))


def run_draw_sapm_plot(type, clusternumber):
    # load excel file with results to display
    # type = 'fixed'
    # clusternumber = 0
    if type == 'random':
        offset = 5
    else:
        offset = 0

    regions = define_drawing_regions()

    # temporary -----------------------------------------------------------
    if type == 'fixed':
        if clusternumber == 0:
            cnums = [0, 3, 3, 1, 4, 1, 3, 3, 4, 1]  # fixed 0
        if clusternumber == 1:
            cnums = [1, 3, 3, 1, 3, 1, 3, 3, 2, 1]  # fixed 1
        if clusternumber == 2:
            cnums = [2, 3, 3, 1, 1, 1, 3, 3, 2, 0]  # fixed 2
        if clusternumber == 3:
            cnums = [3, 3, 2, 1, 0, 1, 2, 3, 4, 1]  # fixed 3
        if clusternumber == 4:
            cnums = [4, 3, 3, 1, 0, 1, 2, 3, 4, 3]  # fixed 4
    else:
        if clusternumber == 0:
            cnums = [0, 4, 4, 2, 2, 3, 3, 2, 3, 1]  # random 0
        if clusternumber == 1:
            cnums = [1, 4, 2, 3, 2, 1, 1, 2, 3, 0]  # random 1
        if clusternumber == 2:
            cnums = [2, 2, 2, 0, 0, 2, 0, 3, 1, 3]  # random 2
        if clusternumber == 3:
            cnums = [3, 3, 1, 4, 4, 1, 3, 3, 1, 0]  # random 3
        if clusternumber == 4:
            cnums = [4, 4, 2, 1, 0, 3, 3, 3, 2, 0]  # random 4
    #----------------------------------------------------------------------
    clusterdataname = r'D:/threat_safety_python/threat_safety_clusterdata.npy'

    results_file = r'D:\threat_safety_python\individual_differences\{}_C6RD{}\all All Average Mconn values.xlsx'.format(type,clusternumber)
    sheetname = 'average'  # sheet of excel file to read
    regionnames = 'regions'   # column of excel file to read
    statnames = 'beta'   # column of excel file to read
    scalefactor = 10.0
    threshold = 0.0

    results_file = r'D:\threat_safety_python\individual_differences\{}_C6RD{}\all All Average Mconn values_corr.xlsx'.format(type,clusternumber)
    sheetname = 'correlation'  # sheet of excel file to read
    regionnames = 'regions'   # column of excel file to read
    statnames = 'Z'   # column of excel file to read
    scalefactor = 1.0

    Tthresh = stats.norm.ppf(1 - np.array([1.0, 0.05, 0.01, 0.001]))
    threshold = Zthresh[2]

    figurenumber = clusternumber+1+offset
    draw_sapm_plot(results_file, sheetname, regionnames, regions,statnames,figurenumber, scalefactor, cnums, threshold, True, clusterdataname)


def run_draw_sapm_plot_brain():
    templatename = 'brain'
    NP = 9
    cnums = [7, 6, 2, 4, 3, 2, 5, 5, 1, 2, 2, 4]
    cnums = [2, 2, 2, 4, 3, 2, 5, 5, 1, 2, 2, 4]
    clusterdataname = r'C:\fMRI-EEG_shared_data_X1\fmri-eeg_motor2_clusters.npy'
    results_file = r'C:\fMRI-EEG_shared_data_X1\Average Mconn values.xlsx'

    Tthresh = stats.t.ppf(1 - np.array([1.0, 0.05, 0.01, 0.001]),NP-1)


    sheetname = 'average'  # sheet of excel file to read
    regionnames = 'regions'   # column of excel file to read
    statnames = 'T'   # column of excel file to read
    scalefactor = 0.5
    threshold = Tthresh[2]

    statnames = 'beta'   # column of excel file to read
    scalefactor = 10.0
    threshold = 0.0

    regions = define_drawing_regions_brain()

    figurenumber = 108
    draw_sapm_plot(results_file, sheetname, regionnames, regions,statnames,figurenumber, scalefactor, cnums, threshold, True)



def define_drawing_regions_from_file(regionfilename):
    # setup region labels and positions
    xls = pd.ExcelFile(regionfilename, engine='openpyxl')
    df1 = pd.read_excel(xls, 'regions')
    names = df1['name']
    posx = df1['posx']
    posy = df1['posy']
    offset_x = df1['labeloffset_x']
    offset_y = df1['labeloffset_y']

    regions = []
    for nn in range(len(names)):
        entry = {'name': names[nn], 'pos':[posx[nn],posy[nn]], 'labeloffset':np.array([offset_x[nn],offset_y[nn]])}
        regions.append(entry)

    return regions


def define_drawing_regions():
    # setup region labels and positions
    # rnamelist = ['C6RD',  'DRt', 'Hypothalamus','LC', 'NGC',
    #                'NRM', 'NTS', 'PAG', 'PBN', 'Thalamus']
    regions = []
    entry = {'name': 'C6RD', 'pos':[0.6,0.15], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'DRt', 'pos':[0.2,0.30], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'Hypo', 'pos':[0.3,0.8], 'labeloffset':np.array([0,0.05])}
    regions.append(entry)
    entry = {'name': 'LC', 'pos':[0.1,0.5], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'NGC', 'pos':[0.65,0.45], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'NRM', 'pos':[0.4,0.45], 'labeloffset':np.array([0.05,0.0])}
    regions.append(entry)
    entry = {'name': 'NTS', 'pos':[0.1,0.7], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'PAG', 'pos':[0.5,0.8], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'PBN', 'pos':[0.8,0.6], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'Thal', 'pos':[0.5,0.9], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'int0', 'pos':[0.75,0.15], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'int1', 'pos':[0.65,0.9], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'int2', 'pos':[0.1,0.85], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)

    return regions


def define_drawing_regions_brain():
    # setup region labels and positions
    # rnamelist = ['PreCG', 'PostCG', 'SMA', 'SPL', 'AC', 'PC', 'IC', 'Thalamus',
    #               'Cereb1 ', 'Putamen', 'IFG oper', 'OP']
    regions = []
    entry = {'name': 'PreCG', 'pos':[0.4,0.9], 'labeloffset':np.array([0,0.03])}
    regions.append(entry)
    entry = {'name': 'PostCG', 'pos':[0.6,0.9], 'labeloffset':np.array([0,0.053])}
    regions.append(entry)
    entry = {'name': 'SMA', 'pos':[0.3,0.8], 'labeloffset':np.array([-0.05,0.03])}
    regions.append(entry)
    entry = {'name': 'SPL', 'pos':[0.75,0.8], 'labeloffset':np.array([-0.02,0.03])}
    regions.append(entry)
    entry = {'name': 'AC', 'pos':[0.35,0.5], 'labeloffset':np.array([0,0.03])}
    regions.append(entry)
    entry = {'name': 'PC', 'pos':[0.55,0.5], 'labeloffset':np.array([0.0,-0.05])}
    regions.append(entry)
    entry = {'name': 'IC', 'pos':[0.45,0.4], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'Thalamus', 'pos':[0.65,0.25], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'Cereb1', 'pos':[0.8,0.4], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'Putamen', 'pos':[0.35,0.35], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'IFGoper', 'pos':[0.1,0.5], 'labeloffset':np.array([-0.05,-0.05])}
    regions.append(entry)
    entry = {'name': 'OP', 'pos':[0.9,0.6], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'int0', 'pos':[0.3,0.9], 'labeloffset':np.array([-0.05,0.03])}
    regions.append(entry)
    entry = {'name': 'int1', 'pos':[0.55,0.25], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'int2', 'pos':[0.1,0.6], 'labeloffset':np.array([-0.05,0.03])}
    regions.append(entry)
    entry = {'name': 'int3', 'pos':[0.9,0.5], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)

    return regions


def draw_general_sapm_plot(regions, results_file, sheetname, regionnames, statnames, regiondeffile,figurenumber, scalefactor, cnums, threshold = 0.0, outputname = 'SAPMplot', writefigure = False):
    xls = pd.ExcelFile(results_file, engine='openpyxl')
    df1 = pd.read_excel(xls, sheetname)
    connections = df1[regionnames]
    statvals = df1[statnames]

    plt.close(figurenumber)

    regionlist = [regions[x]['name'] for x in range(len(regions))]
    regionlist_trunc = [regions[x]['name'][:4] for x in range(len(regions))]

    # set some drawing parameters
    ovalsize = (0.1,0.05)
    width = 0.001
    ovalcolor = [0,0,0]

    # start drawing
    fig = plt.figure(figurenumber)
    ax = fig.add_axes([0,0,1,1])

    # add ellipses and labels
    for nn in range(len(regions)):
        ellipse = mpatches.Ellipse(regions[nn]['pos'],ovalsize[0],ovalsize[1], alpha = 0.3)
        ax.add_patch(ellipse)
        if nn < len(cnums):
            ax.annotate('{}{}'.format(regions[nn]['name'],cnums[nn]),regions[nn]['pos']+regions[nn]['labeloffset'])
        else:
            ax.annotate(regions[nn]['name'],regions[nn]['pos']+regions[nn]['labeloffset'])

    an_list = []
    connection_list = []
    acount = 0
    for nn in range(len(connections)):
        # plot lines for connections
        c1 = connections[nn]
        val1 = statvals[nn]
        m,s = parse_statval(val1)
        if np.abs(m) > threshold:
            linethick = np.min([5.0, np.abs(m)*scalefactor])
            if m > 0:
                linecolor = 'k'
            else:
                linecolor = 'r'
            rlist,ilist = parse_connection_name(c1,regionlist_trunc)

            # get positions of ends of lines,arrows, etc... for one connection
            p0 = regions[ilist[0]]['pos']
            p1 = regions[ilist[1]]['pos']
            p2 = regions[ilist[2]]['pos']

            if p0 != p1  and  p1 != p2:
                pe0, pe1a, pe1b, pe2, pe1ab_connectionstyle, specialcase = points_on_ellipses2(p0,p1,p2,ovalsize)
                print('{}  {}'.format(c1,pe1ab_connectionstyle))

                connection_type1 = {'con':'{}-{}'.format(rlist[0],rlist[1]), 'type':'input'}
                connection_type2 = {'con':'{}-{}'.format(rlist[1],rlist[2]), 'type':'output'}
                connection_joiner = {'con':'{}-{}'.format(rlist[1],rlist[1]), 'type':'joiner'}

                if specialcase:
                    print('special case...')
                    an1 = ax.annotate('',xy=pe1a,xytext = pe0, arrowprops=dict(arrowstyle="->", connectionstyle='arc3', linewidth = linethick, color = linecolor, shrinkA = 0.01, shrinkB = 0.01))
                    acount+= 1
                    an_list.append(an1)
                    connection_list.append(connection_type1)
                    an1 = ax.annotate('',xy=pe2,xytext = pe1b, arrowprops=dict(arrowstyle="->", connectionstyle='arc3', linewidth = linethick, color = linecolor, shrinkA = 0.01, shrinkB = 0.01))
                    acount+= 1
                    an_list.append(an1)
                    connection_list.append(connection_type2)
                else:
                    an1 = ax.annotate('',xy=pe1a,xytext = pe0, arrowprops=dict(arrowstyle="->", connectionstyle='arc3', linewidth = linethick, color = linecolor, shrinkA = 0.01, shrinkB = 0.01))
                    acount+= 1
                    an_list.append(an1)
                    connection_list.append(connection_type1)
                    an1 = ax.annotate('',xy=pe2,xytext = pe1b, arrowprops=dict(arrowstyle="->", connectionstyle='arc3', linewidth = linethick, color = linecolor, shrinkA = 0.01, shrinkB = 0.01))
                    acount+= 1
                    an_list.append(an1)
                    connection_list.append(connection_type2)
                    an1 = ax.annotate('',xy=pe1b,xytext = pe1a, arrowprops=dict(arrowstyle="->", connectionstyle=pe1ab_connectionstyle, linewidth = linethick/2.0, color = linecolor, shrinkA = 0.0, shrinkB = 0.0))
                    acount+= 1
                    an_list.append(an1)
                    connection_list.append(connection_joiner)
            else:
                print('ambiguous connection not drawn:  {}'.format(c1))

    # look for inputs and outputs drawn for the same connection.  Only show the input if both exist
    conlist = [connection_list[x]['con'] for x in range(len(connection_list))]
    typelist = [connection_list[x]['type'] for x in range(len(connection_list))]
    for nn in range(len(connection_list)):
        con = conlist[nn]
        c = np.where([conlist[x] == con for x in range(len(conlist))])[0]
        if len(c) > 1:
            t = [typelist[x] for x in c]
            if 'input' in t:   # if some of the connections are inputs, do not draw outputs at the same place
                c2 = np.where([typelist[x] == 'output' for x in c])[0]
                if len(c2) > 0:
                    redundant_c = c[c2]
                    # remove the redundant connections
                    for c3 in redundant_c:
                        a = an_list[c3]
                        a.remove()
                        typelist[c3] = 'removed'
                        connection_list[c3]['type'] = 'removed'


    if writefigure:
        plt.figure(figurenumber)
        plt.savefig(outputname, format='svg')
        print('saved figure as {}'.format(svgname))



def draw_sapm_plot(results_file, sheetname, regionnames, regions,statnames,figurenumber, scalefactor, cnums, threshold = 0.0, writefigure = False):
    # templatename, clusterdataname = []
    #
    xls = pd.ExcelFile(results_file, engine='openpyxl')
    df1 = pd.read_excel(xls, sheetname)
    connections = df1[regionnames]
    statvals = df1[statnames]

    plt.close(figurenumber)

    regionlist = [regions[x]['name'] for x in range(len(regions))]
    regionlist_trunc = [regions[x]['name'][:4] for x in range(len(regions))]

    # set some drawing parameters
    ovalsize = (0.1,0.05)
    width = 0.001
    ovalcolor = [0,0,0]

    # start drawing
    fig = plt.figure(figurenumber)
    ax = fig.add_axes([0,0,1,1])

    # # show axial slices?
    # if len(clusterdataname) > 0:
    #     for rr, regionname in enumerate(regionlist):
    #         clusternum = cnums[rr]
    #         outputimg = display_anatomical_slices(clusterdataname, regionname, clusternum, templatename)
    #         # display it somewhere...

    # add ellipses and labels
    for nn in range(len(regions)):
        ellipse = mpatches.Ellipse(regions[nn]['pos'],ovalsize[0],ovalsize[1], alpha = 0.3)
        ax.add_patch(ellipse)
        if nn < len(cnums):
            ax.annotate('{}{}'.format(regions[nn]['name'],cnums[nn]),regions[nn]['pos']+regions[nn]['labeloffset'])
        else:
            ax.annotate(regions[nn]['name'],regions[nn]['pos']+regions[nn]['labeloffset'])

    an_list = []
    connection_list = []
    acount = 0
    for nn in range(len(connections)):
        # plot lines for connections
        c1 = connections[nn]
        val1 = statvals[nn]
        m,s = parse_statval(val1)
        if np.abs(m) > threshold:
            linethick = np.min([5.0, np.abs(m)*scalefactor])
            if m > 0:
                linecolor = 'k'
            else:
                linecolor = 'r'
            rlist,ilist = parse_connection_name(c1,regionlist_trunc)

            # get positions of ends of lines,arrows, etc... for one connection
            p0 = regions[ilist[0]]['pos']
            p1 = regions[ilist[1]]['pos']
            p2 = regions[ilist[2]]['pos']

            if p0 != p1  and  p1 != p2:
                pe0, pe1a, pe1b, pe2, pe1ab_connectionstyle, specialcase = points_on_ellipses2(p0,p1,p2,ovalsize)
                print('{}  {}'.format(c1,pe1ab_connectionstyle))

                connection_type1 = {'con':'{}-{}'.format(rlist[0],rlist[1]), 'type':'input'}
                connection_type2 = {'con':'{}-{}'.format(rlist[1],rlist[2]), 'type':'output'}
                connection_joiner = {'con':'{}-{}'.format(rlist[1],rlist[1]), 'type':'joiner'}

                if specialcase:
                    print('special case...')
                    an1 = ax.annotate('',xy=pe1a,xytext = pe0, arrowprops=dict(arrowstyle="->", connectionstyle='arc3', linewidth = linethick, color = linecolor, shrinkA = 0.01, shrinkB = 0.01))
                    acount+= 1
                    an_list.append(an1)
                    connection_list.append(connection_type1)
                    an1 = ax.annotate('',xy=pe2,xytext = pe1b, arrowprops=dict(arrowstyle="->", connectionstyle='arc3', linewidth = linethick, color = linecolor, shrinkA = 0.01, shrinkB = 0.01))
                    acount+= 1
                    an_list.append(an1)
                    connection_list.append(connection_type2)
                else:
                    an1 = ax.annotate('',xy=pe1a,xytext = pe0, arrowprops=dict(arrowstyle="->", connectionstyle='arc3', linewidth = linethick, color = linecolor, shrinkA = 0.01, shrinkB = 0.01))
                    acount+= 1
                    an_list.append(an1)
                    connection_list.append(connection_type1)
                    an1 = ax.annotate('',xy=pe2,xytext = pe1b, arrowprops=dict(arrowstyle="->", connectionstyle='arc3', linewidth = linethick, color = linecolor, shrinkA = 0.01, shrinkB = 0.01))
                    acount+= 1
                    an_list.append(an1)
                    connection_list.append(connection_type2)
                    an1 = ax.annotate('',xy=pe1b,xytext = pe1a, arrowprops=dict(arrowstyle="->", connectionstyle=pe1ab_connectionstyle, linewidth = linethick/2.0, color = linecolor, shrinkA = 0.0, shrinkB = 0.0))
                    acount+= 1
                    an_list.append(an1)
                    connection_list.append(connection_joiner)
            else:
                print('ambiguous connection not drawn:  {}'.format(c1))

    # look for inputs and outputs drawn for the same connection.  Only show the input if both exist
    conlist = [connection_list[x]['con'] for x in range(len(connection_list))]
    typelist = [connection_list[x]['type'] for x in range(len(connection_list))]
    for nn in range(len(connection_list)):
        con = conlist[nn]
        c = np.where([conlist[x] == con for x in range(len(conlist))])[0]
        if len(c) > 1:
            t = [typelist[x] for x in c]
            if 'input' in t:   # if some of the connections are inputs, do not draw outputs at the same place
                c2 = np.where([typelist[x] == 'output' for x in c])[0]
                if len(c2) > 0:
                    redundant_c = c[c2]
                    # remove the redundant connections
                    for c3 in redundant_c:
                        a = an_list[c3]
                        a.remove()
                        typelist[c3] = 'removed'
                        connection_list[c3]['type'] = 'removed'


    if writefigure:
        p,f1 = os.path.split(results_file)
        f,e = os.path.splitext(f1)
        svgname = os.path.join(p,f+sheetname+'.svg')
        plt.figure(figurenumber)
        plt.savefig(svgname, format='svg')
        print('saved figure as {}'.format(svgname))


def parse_connection_name(connection, regionlist):

    h1 = connection.index('-')
    h2 = connection[(h1+2):].index('-') + h1 + 2
    r1 = connection[:h1]
    r2 = connection[(h1+1):h2]
    r3 = connection[(h2+1):]

    i1 = regionlist.index(r1)
    i2 = regionlist.index(r2)
    i3 = regionlist.index(r3)

    return (r1,r2,r3),(i1,i2,i3)

def parse_statval(val):
    foundpattern = False
    t = chr(177)   # check for +/- sign
    if t in val:
        x = val.index(t)
        m = float(val[:x])
        s = float(val[(x+1):])
        foundpattern = True

    if '=' in val:
        x = val.index('=')
        m = float(val[(x+1):])
        s = []
        foundpattern = True

    if not foundpattern:
        m = float(val)
        s = 0

    return m,s


    h1 = connection.index('-')
    h2 = connection[(h1+2):].index('-') + h1 + 2
    r1 = connection[:h1]
    r2 = connection[(h1+1):h2]
    r3 = connection[(h2+1):]

    i1 = regionlist.index(r1)
    i2 = regionlist.index(r2)
    i3 = regionlist.index(r3)

    return (r1,r2,r3),(i1,i2,i3)



def points_on_ellipses(pos0, pos1, ovalsize):
    # point on ellipse 0 on line from region 0 to region 1
    ovd = np.array(ovalsize)/2.0

    v01 = np.array(pos1)-np.array(pos0)
    d01 = np.sqrt(1/((v01[0]/ovd[0])**2 + (v01[1]/ovd[1])**2))
    pe0 = pos0 + d01*v01

    # point on ellipse 1 on line from region 1 to region 0
    v10 = np.array(pos0)-np.array(pos1)
    d10 = np.sqrt(1/((v10[0]/ovd[0])**2 + (v10[1]/ovd[1])**2))
    pe1 = pos1 + d10*v10

    return pe0, pe1


def points_on_ellipses2(pos0, pos1, pos2, ovalsize):
    # point on ellipse 0 on line from region 0 to region 1
    ovd = np.array(ovalsize)/2.0

    v01 = np.array(pos1)-np.array(pos0)
    d01 = np.sqrt(1/((v01[0]/ovd[0])**2 + (v01[1]/ovd[1])**2))
    pe0 = pos0 + d01*v01

    # point on ellipse 1 on line from region 1 to region 0
    v10 = np.array(pos0)-np.array(pos1)
    d10 = np.sqrt(1/((v10[0]/ovd[0])**2 + (v10[1]/ovd[1])**2))
    pe1a = pos1 + d10*v10

    v12 = np.array(pos2)-np.array(pos1)
    d12 = np.sqrt(1/((v12[0]/ovd[0])**2 + (v12[1]/ovd[1])**2))
    pe1b = pos1 + d12*v12

    # point on ellipse 1 on line from region 1 to region 0
    v21 = np.array(pos1)-np.array(pos2)
    d21 = np.sqrt(1/((v21[0]/ovd[0])**2 + (v21[1]/ovd[1])**2))
    pe2 = pos2 + d21*v21

    # smooth arc line in region 1, betwen arrows for pos0-->pos1 and pos1-->pos2
    # line starts along vector v01 at point pe1a
    # line ends along vector v12 at point pe1b

    # angle of line along vector v01, wrt x axis
    angleA = (180/np.pi)*np.arctan2(v01[1],v01[0])
    angleA = np.round(angleA).astype(int)

    # angle of line along vector v12, wrt x axis
    angleB = (180/np.pi)*np.arctan2(v12[1],v12[0])
    angleB = np.round(angleB).astype(int)
    anglediff = np.abs(angleB-angleA)

    pe1ab_connectionstyle = "angle3,angleA={},angleB={}".format(angleA,angleB)

    # special case
    specialcase = False
    if np.abs(anglediff-180.0) < 1.0:
        specialcase = True
        pe1ab_connectionstyle = "arc3,rad=0"
        pe1ab_connectionstyle = "bar,fraction=0"

    if np.abs(anglediff) < 1.0:
        specialcase = False
        pe1ab_connectionstyle = "arc3,rad=0"

    # shift lines slightly to allow for reciprocal connections
    offset = 0.007
    dpos1 = np.array([offset*np.sin(angleA*np.pi/180.0), offset*np.cos(angleA*np.pi/180.0)])
    dpos2 = np.array([offset*np.sin(angleB*np.pi/180.0), offset*np.cos(angleB*np.pi/180.0)])

    pe0 += dpos1
    pe1a += dpos1
    pe1b += dpos2
    pe2 += dpos2

    return pe0, pe1a, pe1b, pe2, pe1ab_connectionstyle, specialcase



def display_anatomical_slices(clusterdataname, regionname, clusternum, templatename):
    orientation = 'axial'
    regioncolor = [1,1,0]

    # get the connection and region information
    clusterdata = np.load(clusterdataname, allow_pickle=True).flat[0]
    cluster_properties = clusterdata['cluster_properties']
    template_img = clusterdata['template_img']
    rnamelist = [cluster_properties[x]['rname'] for x in range(len(cluster_properties))]
    targetnum = rnamelist.index(regionname)

    # get the voxel coordinates for the target region
    IDX = clusterdata['cluster_properties'][targetnum]['IDX']
    idxx = np.where(IDX == clusternum)
    cx = clusterdata['cluster_properties'][targetnum]['cx'][idxx]
    cy = clusterdata['cluster_properties'][targetnum]['cy'][idxx]
    cz = clusterdata['cluster_properties'][targetnum]['cz'][idxx]

    #-------------------------------------------------------------------------------------
    # display one slice of an anatomical region in the selected target figure
    outputimg = pydisplay.pydisplayvoxelregionslice(templatename, template_img, cx, cy, cz, orientation, displayslice = [], colorlist = regioncolor)
    return outputimg
