# functions to draw and display connectivity diagrams from SAPM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import os
import pandas as pd

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

def run_draw_sapm_plot(type, clusternumber):
    # load excel file with results to display
    # type = 'fixed'
    # clusternumber = 0
    if type == 'random':
        offset = 5
    else:
        offset = 0

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
    threshold = 2.5

    figurenumber = clusternumber+1+offset
    draw_sapm_plot(results_file, sheetname, regionnames,statnames,figurenumber, scalefactor, threshold, True)


def draw_sapm_plot(results_file, sheetname, regionnames,statnames,figurenumber, scalefactor, threshold = 0.0, writefigure = False):
    xls = pd.ExcelFile(results_file, engine='openpyxl')
    df1 = pd.read_excel(xls, sheetname)
    connections = df1[regionnames]
    statvals = df1[statnames]

    plt.close(figurenumber)

    # setup region labels and positions
    regions = []
    entry = {'name': 'C6RD', 'pos':[0.6,0.15], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'DRt', 'pos':[0.2,0.30], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'NRM', 'pos':[0.4,0.45], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'NGC', 'pos':[0.65,0.45], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'PBN', 'pos':[0.8,0.6], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'NTS', 'pos':[0.1,0.7], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'LC', 'pos':[0.1,0.5], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'Hypo', 'pos':[0.3,0.8], 'labeloffset':np.array([0,0.05])}
    regions.append(entry)
    entry = {'name': 'PAG', 'pos':[0.5,0.8], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'Thal', 'pos':[0.5,0.9], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)

    entry = {'name': 'int0', 'pos':[0.75,0.15], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'int1', 'pos':[0.65,0.9], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)
    entry = {'name': 'int2', 'pos':[0.1,0.85], 'labeloffset':np.array([0,-0.05])}
    regions.append(entry)

    regionlist = [regions[x]['name'] for x in range(len(regions))]

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
        ax.annotate(regions[nn]['name'],regions[nn]['pos']+regions[nn]['labeloffset'])

    for nn in range(len(connections)):
        # plot lines for connections
        c1 = connections[nn]
        val1 = statvals[nn]
        m,s = parse_statval(val1)
        if np.abs(m) > threshold:
            linethick = np.min([5.0, np.abs(m)*scalefactor])
            rlist,ilist = parse_connection_name(c1,regionlist)

            # get positions of ends of lines,arrows, etc... for one connection
            p0 = regions[ilist[0]]['pos']
            p1 = regions[ilist[1]]['pos']
            p2 = regions[ilist[2]]['pos']

            if p0 != p1  and  p1 != p2:
                pe0, pe1a, pe1b, pe2, pe1ab_connectionstyle, specialcase = points_on_ellipses2(p0,p1,p2,ovalsize)
                print('{}  {}'.format(c1,pe1ab_connectionstyle))

                if specialcase:
                    print('special case...')
                    ax.annotate('',xy=pe1a,xytext = pe0, arrowprops=dict(arrowstyle="->", connectionstyle='arc3', linewidth = linethick, shrinkA = 0.01, shrinkB = 0.01))
                    ax.annotate('',xy=pe2,xytext = pe1b, arrowprops=dict(arrowstyle="->", connectionstyle='arc3', linewidth = linethick, shrinkA = 0.01, shrinkB = 0.01))
                else:
                    ax.annotate('',xy=pe1a,xytext = pe0, arrowprops=dict(arrowstyle="->", connectionstyle='arc3', linewidth = linethick, shrinkA = 0.01, shrinkB = 0.01))
                    ax.annotate('',xy=pe2,xytext = pe1b, arrowprops=dict(arrowstyle="->", connectionstyle='arc3', linewidth = linethick, shrinkA = 0.01, shrinkB = 0.01))
                    ax.annotate('',xy=pe1b,xytext = pe1a, arrowprops=dict(arrowstyle="->", connectionstyle=pe1ab_connectionstyle, linewidth = linethick/2.0, shrinkA = 0.0, shrinkB = 0.0))
            else:
                print('ambiguous connection not drawn:  {}'.format(c1))

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
