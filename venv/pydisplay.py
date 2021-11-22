# set of functions for displaying fMRI results in various forms
import numpy as np
import copy
import load_templates
import scipy.ndimage as nd
import pandas as pd
import os

# setup color scales for displays
def colormap(values):
    maxval = np.abs(values.max()) + 1.0e-10

    # red scale
    red = values/maxval
    red[red<0] = 0

    # green scale
    green = 1.0 - np.abs(values)/maxval

    # blue scale
    blue = -values/maxval
    blue[blue<0] = 0

    return red, green, blue


# display a statistical map in color over a gray scale template image
def pydisplaystatmap(Tmap, Tthreshold, template, mask,templatename):
    # function for displaying fMRI results from the brainstem/cord
    if np.ndim(Tmap) > 4:
        print('pydisplaystatmap:  Tmap has too many dimensions: ',np.ndim(Tmap))
        return 0

    if np.ndim(Tmap) > 3:
        print('pydisplaystatmap:  statistical map has 4 dimensions - averaging across 4th dimension for display')
        Tmap = np.mean(Tmap,axis = 3)

    if Tmap.shape != template.shape:
        print('pydisplaystatmap:  statistical map and template are not the same size')
        print('              statistical map is size: ',np.shape(Tmap))
        print('                     template is size: ',np.shape(template))
        return 0

    if Tmap.shape != mask.shape:
        print('pydisplaystatmap:  statistical map and region mask are not the same size')
        print('                 region mask will not be used')
        mask = np.ones(Tmap.shape).astype(float)
    else:
        mask[mask < 0.5] = 0.
        mask[mask >= 0.5] = 1.
        mask = mask.astype(float)

    background = template.astype('double')
    background = background/np.max(background)
    red = copy.deepcopy(background)
    green = copy.deepcopy(background)
    blue = copy.deepcopy(background)

    # sag_slice = np.floor(brain_size2[0]/2).astype('int')
    # tcimg = np.dstack((red[sag_slice,:,:],green[sag_slice,:,:],blue[sag_slice,:,:]))
    # fig = plt.figure(21), plt.imshow(tcimg)

    # find voxels that meet the statistical threshold
    cx, cy, cz = np.where(np.abs(Tmap*mask) > Tthreshold)
    rmap, gmap, bmap = colormap(Tmap[cx,cy,cz])
    red[cx,cy,cz] = rmap
    green[cx,cy,cz] = gmap
    blue[cx,cy,cz] = bmap

    # slice results and put them into a mosaic format image
    xs,ys,zs = np.shape(Tmap)
    ncols = (np.sqrt(zs)).astype(int)
    nrows = np.ceil(zs/ncols).astype(int)

    # this might work for brain regions as well
    if templatename == 'brain':
        # display brain data
        outputimg = np.zeros([nrows*ys,ncols*xs,3])

        for zz in range(zs):
            colnum = zz % ncols
            rownum = np.floor(zz/ncols)
            x1 = rownum*xs
            x2 = (rownum+1)*xs-1
            y1 = colnum*ys
            y2 = (colnum+1)*ys-1
            redrot = red[:,yc1:yc2,zz];  redrot = redrot[:,::-1].T
            greenrot = green[:,yc1:yc2,zz];  greenrot = greenrot[:,::-1].T
            bluerot = blue[:,yc1:yc2,zz];  bluerot = bluerot[:,::-1].T
            outputimg[y1:y2,x1:x2,0:3] = np.dstack((redrot,greenrot,bluerot))

    if templatename == 'ccbs':
        # display brainstem/cord
        yc1 = 20;  yc2 = 76;  ys2 = yc2-yc1
        outputimg = np.zeros([nrows*ys2,ncols*xs,3])

        for zz in range(zs):
            colnum = zz % ncols
            rownum = np.floor(zz/ncols).astype(int)
            x1 = colnum*xs
            x2 = (colnum+1)*xs
            y1 = rownum*ys2
            y2 = (rownum+1)*ys2
            # need to rotate each from by 90 degrees
            redrot = red[:,yc1:yc2,zz];  redrot = redrot[:,::-1].T
            greenrot = green[:,yc1:yc2,zz];  greenrot = greenrot[:,::-1].T
            bluerot = blue[:,yc1:yc2,zz];  bluerot = bluerot[:,::-1].T
            outputimg[y1:y2,x1:x2,0:3] = np.dstack((redrot,greenrot,bluerot))

    if templatename != 'brain' and templatename != 'ccbs':
        # do the other thing
        yc1 = 20;  yc2 = 41;  ys2 = yc2-yc1
        outputimg = np.zeros([nrows*ys2,ncols*xs,3])

        for zz in range(zs):
            colnum = zz % ncols
            rownum = np.floor(zz/ncols).astype(int)
            x1 = colnum*xs
            x2 = (colnum+1)*xs
            y1 = rownum*ys2
            y2 = (rownum+1)*ys2
            # need to rotate each from by 90 degrees
            redrot = red[:,yc1:yc2,zz];  redrot = redrot[:,::-1].T
            greenrot = green[:,yc1:yc2,zz];  greenrot = greenrot[:,::-1].T
            bluerot = blue[:,yc1:yc2,zz];  bluerot = bluerot[:,::-1].T
            outputimg[y1:y2,x1:x2,0:3] = np.dstack((redrot,greenrot,bluerot))

    outputimg[outputimg>1.] = 1.0
    outputimg[outputimg<0.] = 0.0

    return outputimg


# display named regions in different colors
def pydisplayanatregions(templatename, anatnames, colorlist = []):
    resolution = 1
    template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_img = load_templates.load_template_and_masks(templatename, resolution)

    anatnamelist = []
    for name in anatlabels['names']:
        anatnamelist.append(name)

    nregions = len(anatnames)
    colsize = [3,nregions]
    if colsize != np.array(colorlist).shape:
        values = np.mgrid[-1:1:nregions*1j]
        r,g,b = colormap(values)
        colorlist = np.stack((r,g,b))

    regionindices = np.zeros(nregions).astype(int)
    for nn,name in enumerate(anatnames):
        try:
            rnum  = anatnamelist.index(name)
            regionindices[nn] = anatlabels['numbers'][rnum]
        except:
            print('pydisplayanatregions: region ',name,' is not in the anatomical map (ignoring this region)')
            rnum = 0

    background = template_img.astype('double')
    background = background/np.max(background)
    red = copy.deepcopy(background)
    green = copy.deepcopy(background)
    blue = copy.deepcopy(background)

    # color the regions
    for nn,index in enumerate(regionindices):
        if index > 0:
            cx, cy, cz = np.where(regionmap_img == index)
            red[cx,cy,cz] = colorlist[0,nn]
            green[cx,cy,cz] = colorlist[1,nn]
            blue[cx,cy,cz] = colorlist[2,nn]

    # slice results and put them into a mosaic format image
    xs,ys,zs = np.shape(template_img)
    ncols = (np.sqrt(zs)).astype(int)
    nrows = np.ceil(zs/ncols).astype(int)

    # this might work for brain regions as well
    if templatename == 'brain':
        # display brain data
        outputimg = np.zeros([nrows*ys,ncols*xs,3])

        for zz in range(zs):
            colnum = zz % ncols
            rownum = np.floor(zz/ncols)
            x1 = rownum*xs
            x2 = (rownum+1)*xs-1
            y1 = colnum*ys
            y2 = (colnum+1)*ys-1
            redrot = red[:,yc1:yc2,zz];  redrot = redrot[:,::-1].T
            greenrot = green[:,yc1:yc2,zz];  greenrot = greenrot[:,::-1].T
            bluerot = blue[:,yc1:yc2,zz];  bluerot = bluerot[:,::-1].T
            outputimg[y1:y2,x1:x2,0:3] = np.dstack((redrot,greenrot,bluerot))

    if templatename == 'ccbs':
        # display brainstem/cord
        yc1 = 20;  yc2 = 76;  ys2 = yc2-yc1
        outputimg = np.zeros([nrows*ys2,ncols*xs,3])

        for zz in range(zs):
            colnum = zz % ncols
            rownum = np.floor(zz/ncols).astype(int)
            x1 = colnum*xs
            x2 = (colnum+1)*xs
            y1 = rownum*ys2
            y2 = (rownum+1)*ys2
            # need to rotate each from by 90 degrees
            redrot = red[:,yc1:yc2,zz];  redrot = redrot[:,::-1].T
            greenrot = green[:,yc1:yc2,zz];  greenrot = greenrot[:,::-1].T
            bluerot = blue[:,yc1:yc2,zz];  bluerot = bluerot[:,::-1].T
            outputimg[y1:y2,x1:x2,0:3] = np.dstack((redrot,greenrot,bluerot))

    if templatename != 'brain' and templatename != 'ccbs':
        # do the other thing
        yc1 = 20;  yc2 = 41;  ys2 = yc2-yc1
        outputimg = np.zeros([nrows*ys2,ncols*xs,3])

        for zz in range(zs):
            colnum = zz % ncols
            rownum = np.floor(zz/ncols).astype(int)
            x1 = colnum*xs
            x2 = (colnum+1)*xs
            y1 = rownum*ys2
            y2 = (rownum+1)*ys2
            # need to rotate each from by 90 degrees
            redrot = red[:,yc1:yc2,zz];  redrot = redrot[:,::-1].T
            greenrot = green[:,yc1:yc2,zz];  greenrot = greenrot[:,::-1].T
            bluerot = blue[:,yc1:yc2,zz];  bluerot = bluerot[:,::-1].T
            outputimg[y1:y2,x1:x2,0:3] = np.dstack((redrot,greenrot,bluerot))

    outputimg[outputimg>1.] = 1.0
    outputimg[outputimg<0.] = 0.0

    return outputimg



# display named regions in different colors
def pydisplayvoxels(templatename, cx,cy,cz, colorlist = [1,0,0]):
    resolution = 1
    template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_map = load_templates.load_template_and_masks(templatename, resolution)

    background = template_img.astype('double')
    background = background/np.max(background)
    red = copy.deepcopy(background)
    green = copy.deepcopy(background)
    blue = copy.deepcopy(background)

    # color the voxels
    red[cx,cy,cz] = colorlist[0]
    green[cx,cy,cz] = colorlist[1]
    blue[cx,cy,cz] = colorlist[2]

    # slice results and put them into a mosaic format image
    xs,ys,zs = np.shape(template_img)
    ncols = (np.sqrt(zs)).astype(int)
    nrows = np.ceil(zs/ncols).astype(int)

    # this might work for brain regions as well
    if templatename == 'brain':
        # display brain data
        outputimg = np.zeros([nrows*ys,ncols*xs,3])

        for zz in range(zs):
            colnum = zz % ncols
            rownum = np.floor(zz/ncols)
            x1 = rownum*xs
            x2 = (rownum+1)*xs-1
            y1 = colnum*ys
            y2 = (colnum+1)*ys-1
            redrot = red[:,yc1:yc2,zz];  redrot = redrot[:,::-1].T
            greenrot = green[:,yc1:yc2,zz];  greenrot = greenrot[:,::-1].T
            bluerot = blue[:,yc1:yc2,zz];  bluerot = bluerot[:,::-1].T
            outputimg[y1:y2,x1:x2,0:3] = np.dstack((redrot,greenrot,bluerot))

    if templatename == 'ccbs':
        # display brainstem/cord
        yc1 = 20;  yc2 = 76;  ys2 = yc2-yc1
        outputimg = np.zeros([nrows*ys2,ncols*xs,3])

        for zz in range(zs):
            colnum = zz % ncols
            rownum = np.floor(zz/ncols).astype(int)
            x1 = colnum*xs
            x2 = (colnum+1)*xs
            y1 = rownum*ys2
            y2 = (rownum+1)*ys2
            # need to rotate each from by 90 degrees
            redrot = red[:,yc1:yc2,zz];  redrot = redrot[:,::-1].T
            greenrot = green[:,yc1:yc2,zz];  greenrot = greenrot[:,::-1].T
            bluerot = blue[:,yc1:yc2,zz];  bluerot = bluerot[:,::-1].T
            outputimg[y1:y2,x1:x2,0:3] = np.dstack((redrot,greenrot,bluerot))

    if templatename != 'brain' and templatename != 'ccbs':
        # do the other thing
        yc1 = 20;  yc2 = 41;  ys2 = yc2-yc1
        outputimg = np.zeros([nrows*ys2,ncols*xs,3])

        for zz in range(zs):
            colnum = zz % ncols
            rownum = np.floor(zz/ncols).astype(int)
            x1 = colnum*xs
            x2 = (colnum+1)*xs
            y1 = rownum*ys2
            y2 = (rownum+1)*ys2
            # need to rotate each from by 90 degrees
            redrot = red[:,yc1:yc2,zz];  redrot = redrot[:,::-1].T
            greenrot = green[:,yc1:yc2,zz];  greenrot = greenrot[:,::-1].T
            bluerot = blue[:,yc1:yc2,zz];  bluerot = bluerot[:,::-1].T
            outputimg[y1:y2,x1:x2,0:3] = np.dstack((redrot,greenrot,bluerot))

    outputimg[outputimg>1.] = 1.0
    outputimg[outputimg<0.] = 0.0

    return outputimg


def pywriteexcel(data, excelname, excelsheet = 'pydata', write_mode = 'replace', floatformat = '%.2f'):
    # data needs to be an array of dictionaries
    # each dictionary key will be a column in the resulting excel sheet
    # "write_mode" should be "replace" or "append" to indicate if the
    # excel file should be over-written, or added to
    try:
        nrows = len(data)
        keylist = data[0].keys()
    except:
        print('pywriteexcel: unexpected input data format')
        return 0

    dataf = pd.DataFrame(data)

    if (write_mode == 'replace') or not os.path.exists(excelname):
        with pd.ExcelWriter(excelname, engine = 'openpyxl', mode='w') as writer:
            dataf.to_excel(writer, sheet_name=excelsheet, float_format = floatformat)
    else:
        with pd.ExcelWriter(excelname, engine = 'openpyxl', mode='a') as writer:
            dataf.to_excel(writer, sheet_name=excelsheet, float_format = floatformat)




#-------------display group-level analyses---------------------------------------------------
# 2-source SEM results
# results = {'type': '2source', 'CCrecord': CCrecord, 'beta2': beta2, 'beta1': beta1, 'Zgrid2': Zgrid2,
#            'Zgrid1_1': Zgrid1_1, 'Zgrid1_2': Zgrid1_2, 'DBname': self.DBname, 'DBnum': self.DBnum,
#            'cluster_properties': cluster_properties}
#
# for testing 2source data
# filename1 = r'D:/threat_safety_python/SEMresults/SEMresults_2source_record_female.npy'
# filename2 = r'D:/threat_safety_python/SEMresults/SEMresults_2source_record_male.npy'
# excelfilename = r'D:/threat_safety_python/SEMresults/SEMresults_2source_record_female.xlsx'
# excelsheetname = '2S beta1 average 1'
# # get information about the results to be displayed, from excel files (output by py2ndlevelanalysis.py)
# xls = pd.ExcelFile(excelfilename, engine='openpyxl')
# df1 = pd.read_excel(xls, excelsheetname)
# fields = list(df1.keys())
# t = df1['t']
# s1 = df1['s1']
# s2 = df1['s2']
# timepoint = df1['tt']
# nb = df1['nb']
#
#
# # network SEM results
# #  dict_keys(['type', 'resultsnames', 'network', 'regionname', 'clustername', 'DBname', 'DBnum'])
# #   within each entry in 'resultnames':
# #                    'sem_one_target_results'
# #                              array of ['b', 'R2', 'networkcomponent', 'targetcluster'] for each target cluster
#
# for testing network data
# filename1 = r'D:/threat_safety_python/SEMresults/SEMresults_network_record_female.npy'
# filename2 = r'D:/threat_safety_python/SEMresults/SEMresults_network_record_male.npy'
# excelfilename = r'D:/threat_safety_python/SEMresults/SEMresults_network_record_female.xlsx'
# excelsheetname = 'network average 0'
# # get information about the results to be displayed, from excel files (output by py2ndlevelanalysis.py)
# xls = pd.ExcelFile(excelfilename, engine='openpyxl')
# df1 = pd.read_excel(xls, excelsheetname)
# fields = list(df1.keys())
# networkcomponent = df1['networkcomponent']
# tt = df1['tt']
# combo = df1['combo']
# timepoint = df1['timepoint']
# ss = df1['ss']


def display_whisker_plots(filename1, filename2, field_to_plot):
#
    data1 = np.load(filename1, allow_pickle=True).flat[0]
    if len(filename2) > 0:
        data2 = np.load(filename2, allow_pickle=True).flat[0]
        twogroup = True
    else:
        data2 = []
        twogroup = False

    if data1['type'] == '2source':
        # 2-source SEM data
        nclusterlist = [data1['cluster_properties'][nn]['nclusters'] for nn in range(len(data1))]
        namelist = [data1['cluster_properties'][nn]['rname'] for nn in range(len(data1))]

        pdata1 = data1[field_to_plot]
        if twogroup:  pdata2 = data2[field_to_plot]
        nt,ns1,ns2,ntime,NP,nc = np.shape(pdata1)   # number of clusters in target, source1, source2,
                                                    # number of timepoints, number of people, and nc (nc=2) for the number of terms
        # collect the data values to plot
        # t[ii], s1[ii], s2[ii], timepoint[ii], nb[ii]
        plotdata_g1 = []
        plotdata_g2 = []
        plotlabel = []
        for nn in range(len(t)):
            # if nn != 28:
            d = pdata1[t[nn],s1[nn],s2[nn],timepoint[nn],:,nb[nn]]   # one group data for one connection
            plotdata_g1.append(d)
            if nb[nn] == 0:
                s = s1[nn]
            else:
                s = s2[nn]
            regionnamet, clusternumt, regionnumt = py2ndlevelanalysis.get_cluster_info(namelist, nclusterlist, t[nn])
            regionnames, clusternums, regionnums = py2ndlevelanalysis.get_cluster_info(namelist, nclusterlist, s)
            if len(regionnamet) > 4: regionnamet = regionnamet[:4]
            if len(regionnames) > 4: regionnames = regionnames[:4]
            textlabel = '{:4s}{}-{:4s}{}'.format(regionnamet,clusternumt,regionnames,clusternums)
            plotlabel.append(textlabel)
            if twogroup:
                d2 = pdata2[t[nn],s1[nn],s2[nn],timepoint[nn],:,nb[nn]]
                plotdata_g2.append(d2)

        # create the boxplot
        fig = plt.figure(16)
        ax1 = plt.axes()
        ax1.set_title(field_to_plot)
        if twogroup:
            ppos_list = []
            for nn in range(len(plotdata_g1)):
                ppos = (nn-1)*3 + 1
                onecat = [plotdata_g1[nn], plotdata_g2[nn]]
                bp = plt.boxplot(onecat, positions=[ppos, ppos+1], widths=0.6, notch = True, showfliers = False)
                setBoxColors(bp)
                ppos_list.append(ppos+0.5)
            ax1.set_xticks(ppos_list)
            ax1.set_xticklabels(plotlabel, rotation = 90)
            plt.tight_layout()
        else:
            ax1.boxplot(plotdata_g1, notch = True, showfliers = False)
            ax1.set_xticklabels(plotlabel, rotation = 90)
            plt.tight_layout()

        fig.savefig('filename.eps', format='eps')
    else:
        # network data
        resultsnames = data1['resultsnames']
        clustername = data1['clustername']
        clusterdata = np.load(clustername, allow_pickle=True).flat[0]
        nclusterlist = np.array([clusterdata['cluster_properties'][nn]['nclusters'] for nn in range(len(clusterdata['cluster_properties']))])
        namelist = [clusterdata['cluster_properties'][nn]['rname'] for nn in range(len(clusterdata['cluster_properties']))]
        networkmodel = data1['network']
        network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkmodel)

        if twogroup: resultsnames2 = data2['resultsnames']

        # need to sort the input list by network component number ...
        aa = np.argsort(networkcomponent)
        networkcomponent2 = [networkcomponent[x] for x in aa]
        tt2 = [tt[x] for x in aa]
        combo2 = [combo[x] for x in aa]
        timepoint2 = [timepoint[x] for x in aa]
        ss2 = [ss[x] for x in aa]

        networknumberlist = np.unique(networkcomponent2)

        # collect the data values to plot
        plotdata_g1 = []
        plotdata_g2 = []
        plotlabel = []
        for networknumber in networknumberlist:
            fname1 = resultsnames[networknumber]
            ndata = np.load(fname1, allow_pickle=True).flat[0]
            ntclusters = len(ndata['sem_one_target_results'])

            targetname = network[networknumber]['target']
            if len(targetname) > 4: targetname = targetname[:4]
            sources = network[networknumber]['sources']
            targetnum = network[networknumber]['targetnum']
            sourcenums = network[networknumber]['sourcenums']

            if twogroup:
                fname2 = resultsnames2[networknumber]
                ndata2 = np.load(fname2, allow_pickle=True).flat[0]

            for nn in range(len(networkcomponent2)):
                if networkcomponent2[nn] == networknumber:
                    pdata1 = ndata['sem_one_target_results'][tt2[nn]]['b']
                    ncombo, ntime, NP, ns = np.shape(pdata1)

                    d = pdata1[combo2[nn],timepoint2[nn],:,ss2[nn]]   # one group data for one connection
                    plotdata_g1.append(d)

                    # sourcename = cluster_info[sourcenums[ss2[nn]]]['rname']
                    sourcename = namelist[sourcenums[ss2[nn]]]
                    mlist = pysem.ind2sub_ndims(nclusterlist[sourcenums], combo2[nn]).astype(int)  # cluster number for each source
                    sourcecluster = mlist[ss2[nn]]

                    if len(sourcename) > 4: sourcename = sourcename[:4]
                    textlabel = '{:4s}{}-{:4s}{}'.format(targetname,tt2[nn],sourcename,sourcecluster)
                    plotlabel.append(textlabel)
                    if twogroup:
                        pdata2 = ndata2['sem_one_target_results'][tt2[nn]]['b']
                        d = pdata2[combo2[nn],timepoint2[nn],:,ss2[nn]]   # one group data for one connection
                        plotdata_g2.append(d)

        # create the boxplot
        fig = plt.figure(16)
        ax1 = plt.axes()
        ax1.set_title('Network')
        if twogroup:
            ppos_list = []
            for nn in range(len(plotdata_g1)):
                ppos = (nn-1)*3 + 1
                onecat = [plotdata_g1[nn], plotdata_g2[nn]]
                bp = plt.boxplot(onecat, positions=[ppos, ppos+1], widths=0.6, notch = True, showfliers = False)
                setBoxColors(bp)
                ppos_list.append(ppos+0.5)
            ax1.set_xticks(ppos_list)
            ax1.set_xticklabels(plotlabel, rotation = 90)
            plt.tight_layout()
        else:
            ax1.boxplot(plotdata_g1, notch = True, showfliers = False)
            ax1.set_xticklabels(plotlabel, rotation = 90)
            plt.tight_layout()

        fig.savefig('filename.eps', format='eps')


def setBoxColors(bp):
    plt.setp(bp['boxes'][0], color='blue')
    plt.setp(bp['caps'][0], color='blue')
    plt.setp(bp['caps'][1], color='blue')
    plt.setp(bp['whiskers'][0], color='blue')
    plt.setp(bp['whiskers'][1], color='blue')
    plt.setp(bp['medians'][0], color='blue')

    plt.setp(bp['boxes'][1], color='red')
    plt.setp(bp['caps'][2], color='red')
    plt.setp(bp['caps'][3], color='red')
    plt.setp(bp['whiskers'][2], color='red')
    plt.setp(bp['whiskers'][3], color='red')
    plt.setp(bp['medians'][1], color='red')