# set of functions for displaying fMRI results in various forms
import numpy as np
import copy
import load_templates
import scipy.ndimage as nd
import pandas as pd
import os
import py2ndlevelanalysis
import pydatabase
import matplotlib.pyplot as plt

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
            redrot = red[:,:,zz];  redrot = redrot[:,::-1].T
            greenrot = green[:,:,zz];  greenrot = greenrot[:,::-1].T
            bluerot = blue[:,:,zz];  bluerot = bluerot[:,::-1].T
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
            # need to rotate each frame by 90 degrees
            redrot = red[:,yc1:yc2,zz];  redrot = redrot[:,::-1].T
            greenrot = green[:,yc1:yc2,zz];  greenrot = greenrot[:,::-1].T
            bluerot = blue[:,yc1:yc2,zz];  bluerot = bluerot[:,::-1].T
            outputimg[y1:y2,x1:x2,0:3] = np.dstack((redrot,greenrot,bluerot))

    outputimg[outputimg>1.] = 1.0
    outputimg[outputimg<0.] = 0.0

    return outputimg


# display named regions in different colors
def pydisplayanatregionslice(templatename, anatname, orientation, displayslice = [], colorlist = []):
    resolution = 1
    template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_img = load_templates.load_template_and_masks(templatename, resolution)

    anatnamelist = []
    for name in anatlabels['names']:
        anatnamelist.append(name)

    if len(colorlist) == 0:
        colorlist = [0,1,0]  # make it green by default

    try:
        rnum  = anatnamelist.index(anatname)
        regionindex = anatlabels['numbers'][rnum]
    except:
        print('pydisplayanatregionslice: region ',anatname,' is not in the anatomical map (ignoring this region)')
        rnum = 0
        outputimg = []
        return outputimg

    background = template_img.astype('double')
    background = background/np.max(background)
    red = copy.deepcopy(background)
    green = copy.deepcopy(background)
    blue = copy.deepcopy(background)

    # color the region
    cx, cy, cz = np.where(regionmap_img == regionindex)
    red[cx,cy,cz] = colorlist[0]
    green[cx,cy,cz] = colorlist[1]
    blue[cx,cy,cz] = colorlist[2]

    #--------------------------------------------------------------
    # this might work for brain regions as well
    if templatename == 'brain':
        # display brain data
        if orientation == 'axial':
            if len(displayslice) == 0: displayslice = np.round(np.mean(cz)).astype(int)
            redrot = red[:,:,displayslice];  redrot = redrot[:,::-1].T
            greenrot = green[:,:,displayslice];  greenrot = greenrot[:,::-1].T
            bluerot = blue[:,:,displayslice];  bluerot = bluerot[:,::-1].T
            outputimg = np.dstack((redrot,greenrot,bluerot))

        if orientation == 'sagittal':
            if len(displayslice) == 0: displayslice = np.round(np.mean(cx)).astype(int)
            redrot = red[displayslice,:,:];  # redrot = redrot[:,::-1].T
            greenrot = green[displayslice,:,:];  # greenrot = greenrot[:,::-1].T
            bluerot = blue[displayslice,:,:];  # bluerot = bluerot[:,::-1].T
            outputimg = np.dstack((redrot,greenrot,bluerot))

        if orientation == 'coronal':
            if len(displayslice) == 0: displayslice = np.round(np.mean(cy)).astype(int)
            redrot = red[:,displayslice,:];  # redrot = redrot[:,::-1].T
            greenrot = green[:,displayslice,:];  # greenrot = greenrot[:,::-1].T
            bluerot = blue[:,displayslice,:];  # bluerot = bluerot[:,::-1].T
            outputimg = np.dstack((redrot,greenrot,bluerot))
    else:
        if templatename == 'ccbs':
            # display brainstem/cord
            yc1 = 20;  yc2 = 76;  ys2 = yc2-yc1
        else:
            # display any other part of the cord
            yc1 = 20;  yc2 = 41;  ys2 = yc2-yc1

        if orientation == 'axial':
            if len(displayslice) == 0: displayslice = np.round(np.mean(cz)).astype(int)
            redrot = red[:,yc1:yc2,displayslice];  redrot = redrot[:,::-1].T
            greenrot = green[:,yc1:yc2,displayslice];  greenrot = greenrot[:,::-1].T
            bluerot = blue[:,yc1:yc2,displayslice];  bluerot = bluerot[:,::-1].T
            outputimg = np.dstack((redrot,greenrot,bluerot))

        if orientation == 'sagittal':
            if len(displayslice) == 0: displayslice = np.round(np.mean(cx)).astype(int)
            redrot = red[displayslice,yc1:yc2,:];  # redrot = redrot[:,::-1].T
            greenrot = green[displayslice,yc1:yc2,:];  # greenrot = greenrot[:,::-1].T
            bluerot = blue[displayslice,yc1:yc2,:];  # bluerot = bluerot[:,::-1].T
            outputimg = np.dstack((redrot,greenrot,bluerot))

        if orientation == 'coronal':
            if len(displayslice) == 0: displayslice = np.round(np.mean(cy)).astype(int)
            redrot = red[:,displayslice,:];  # redrot = redrot[:,::-1].T
            greenrot = green[:,displayslice,:];  # greenrot = greenrot[:,::-1].T
            bluerot = blue[:,displayslice,:];  # bluerot = bluerot[:,::-1].T
            outputimg = np.dstack((redrot,greenrot,bluerot))

    return outputimg


# display named regions in different colors
def pydisplayvoxelregionslice(templatename, cx, cy, cz, orientation, displayslice = [], colorlist = []):
    resolution = 1
    template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_img = load_templates.load_template_and_masks(templatename, resolution)

    anatnamelist = []
    for name in anatlabels['names']:
        anatnamelist.append(name)

    if len(colorlist) == 0:
        colorlist = [0,1,0]  # make it green by default

    background = template_img.astype('double')
    background = background/np.max(background)
    red = copy.deepcopy(background)
    green = copy.deepcopy(background)
    blue = copy.deepcopy(background)

    # color the region
    red[cx,cy,cz] = colorlist[0]
    green[cx,cy,cz] = colorlist[1]
    blue[cx,cy,cz] = colorlist[2]

    #--------------------------------------------------------------
    # this might work for brain regions as well
    if templatename == 'brain':
        # display brain data
        if orientation == 'axial':
            if len(displayslice) == 0: displayslice = np.round(np.mean(cz)).astype(int)
            redrot = red[:,:,displayslice];  redrot = redrot[:,::-1].T
            greenrot = green[:,:,displayslice];  greenrot = greenrot[:,::-1].T
            bluerot = blue[:,:,displayslice];  bluerot = bluerot[:,::-1].T
            outputimg = np.dstack((redrot,greenrot,bluerot))

        if orientation == 'sagittal':
            if len(displayslice) == 0: displayslice = np.round(np.mean(cx)).astype(int)
            redrot = red[displayslice,:,:];  # redrot = redrot[:,::-1].T
            greenrot = green[displayslice,:,:];  # greenrot = greenrot[:,::-1].T
            bluerot = blue[displayslice,:,:];  # bluerot = bluerot[:,::-1].T
            outputimg = np.dstack((redrot,greenrot,bluerot))

        if orientation == 'coronal':
            if len(displayslice) == 0: displayslice = np.round(np.mean(cy)).astype(int)
            redrot = red[:,displayslice,:];  # redrot = redrot[:,::-1].T
            greenrot = green[:,displayslice,:];  # greenrot = greenrot[:,::-1].T
            bluerot = blue[:,displayslice,:];  # bluerot = bluerot[:,::-1].T
            outputimg = np.dstack((redrot,greenrot,bluerot))
    else:
        if templatename == 'ccbs':
            # display brainstem/cord
            yc1 = 20;  yc2 = 76;  ys2 = yc2-yc1
        else:
            # display any other part of the cord
            yc1 = 20;  yc2 = 41;  ys2 = yc2-yc1

        if orientation == 'axial':
            if len(displayslice) == 0: displayslice = np.round(np.mean(cz)).astype(int)
            redrot = red[:,yc1:yc2,displayslice];  redrot = redrot[:,::-1].T
            greenrot = green[:,yc1:yc2,displayslice];  greenrot = greenrot[:,::-1].T
            bluerot = blue[:,yc1:yc2,displayslice];  bluerot = bluerot[:,::-1].T
            outputimg = np.dstack((redrot,greenrot,bluerot))

        if orientation == 'sagittal':
            if len(displayslice) == 0: displayslice = np.round(np.mean(cx)).astype(int)
            redrot = red[displayslice,yc1:yc2,:];  # redrot = redrot[:,::-1].T
            greenrot = green[displayslice,yc1:yc2,:];  # greenrot = greenrot[:,::-1].T
            bluerot = blue[displayslice,yc1:yc2,:];  # bluerot = bluerot[:,::-1].T
            outputimg = np.dstack((redrot,greenrot,bluerot))

        if orientation == 'coronal':
            if len(displayslice) == 0: displayslice = np.round(np.mean(cy)).astype(int)
            redrot = red[:,displayslice,:];  # redrot = redrot[:,::-1].T
            greenrot = green[:,displayslice,:];  # greenrot = greenrot[:,::-1].T
            bluerot = blue[:,displayslice,:];  # bluerot = bluerot[:,::-1].T
            outputimg = np.dstack((redrot,greenrot,bluerot))

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


#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
#-------------display group-level analyses---------------------------------------------------
def display_whisker_plots(filename1, filename2, connectiondata, field_to_plot, TargetCanvas = [], TargetAxes = []):
#
    titlefont=8
    labelfont=6

    data1 = np.load(filename1, allow_pickle=True).flat[0]
    if len(filename2) > 0:
        data2 = np.load(filename2, allow_pickle=True).flat[0]
        twogroup = True
    else:
        data2 = []
        twogroup = False

    if data1['type'] == '2source':
        # connection data for 2source results
        t = connectiondata['t']
        s1 = connectiondata['s1']
        s2 = connectiondata['s2']
        timepoint = connectiondata['tt']
        nb = connectiondata['nb']

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
        TargetAxes.clear()
        if twogroup:
            ppos_list = []
            for nn in range(len(plotdata_g1)):
                ppos = (nn-1)*3 + 1
                onecat = [plotdata_g1[nn], plotdata_g2[nn]]
                bp = TargetAxes.boxplot(onecat, positions=[ppos, ppos+1], widths=0.6, notch = True, showfliers = False)
                setBoxColors(bp)
                ppos_list.append(ppos+0.5)
            TargetAxes.set_xticks(ppos_list)
            TargetAxes.set_xticklabels(plotlabel, rotation = 90, fontsize=labelfont)
            plt.tight_layout()
        else:
            TargetAxes.boxplot(plotdata_g1, notch = True, showfliers = False)
            TargetAxes.set_xticklabels(plotlabel, rotation = 90, fontsize=labelfont)
            plt.tight_layout()

        plt.yticks(fontsize=labelfont)
        TargetAxes.set_title(field_to_plot, fontsize=titlefont)
        TargetCanvas.draw()
        # TargetCanvas.savefig('filename.eps', format='eps')

    else:
        # connection data for network results
        networkcomponent = connectiondata['networkcomponent']
        tt = connectiondata['tt']
        combo = connectiondata['combo']
        timepoint = connectiondata['timepoint']
        ss = connectiondata['ss']

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
        TargetAxes.clear()
        if twogroup:
            ppos_list = []
            for nn in range(len(plotdata_g1)):
                ppos = (nn-1)*3 + 1
                onecat = [plotdata_g1[nn], plotdata_g2[nn]]
                bp = TargetAxes.boxplot(onecat, positions=[ppos, ppos+1], widths=0.6, notch = True, showfliers = False)
                setBoxColors(bp)
                ppos_list.append(ppos+0.5)
            TargetAxes.set_xticks(ppos_list)
            TargetAxes.set_xticklabels(plotlabel, rotation = 90, fontsize=labelfont)
            plt.tight_layout()
        else:
            TargetAxes.boxplot(plotdata_g1, notch = True, showfliers = False)
            TargetAxes.set_xticklabels(plotlabel, rotation = 90, fontsize=labelfont)
            plt.tight_layout()

        plt.yticks(fontsize=labelfont)
        TargetAxes.set_title('Network', fontsize=titlefont)
        TargetCanvas.draw()
        # fig.savefig('filename.eps', format='eps')


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


#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
def display_correlation_plots(filename1, filename2, connectiondata, field_to_plot, covariates1, covariates2, covariatename = 'none', TargetCanvas = [], TargetAxes = []):
#
# inputs can either provide the covariates for each group, or give the covariate name to be read from the database file
# both types of inputs are not needed. If covariate values are provided, they will be used.
#
    titlefont=8
    labelfont=6

    data1 = np.load(filename1, allow_pickle=True).flat[0]
    if len(filename2) > 0:
        data2 = np.load(filename2, allow_pickle=True).flat[0]
        twogroup = True
    else:
        data2 = []
        twogroup = False

    if (len(covariates1) == 0) and (covariatename != 'none'):
        covariates1 = get_covariate_values(data1['DBname'], data1['DBnum'], covariatename, mode='average_per_person')

    if twogroup and (len(covariates2) == 0) and (covariatename != 'none'):
        covariates2 = get_covariate_values(data2['DBname'], data2['DBnum'], covariatename, mode='average_per_person')

    if data1['type'] == '2source':
        # connection data for 2source results
        t = connectiondata['t'][0]
        s1 = connectiondata['s1'][0]
        s2 = connectiondata['s2'][0]
        timepoint = connectiondata['tt'][0]
        nb = connectiondata['nb'][0]

        # 2-source SEM data
        nclusterlist = [data1['cluster_properties'][nn]['nclusters'] for nn in range(len(data1))]
        namelist = [data1['cluster_properties'][nn]['rname'] for nn in range(len(data1))]

        pdata1 = data1[field_to_plot]
        if twogroup:  pdata2 = data2[field_to_plot]
        nt,ns1,ns2,ntime,NP,nc = np.shape(pdata1)   # number of clusters in target, source1, source2,
                                                    # number of timepoints, number of people, and nc (nc=2) for the number of terms
        # collect the data values to plot
        d = pdata1[t,s1,s2,timepoint,:,nb]   # one group data for one connection
        if twogroup: d2 = pdata2[t, s1, s2, timepoint, :, nb]

        if nb == 0:
            s = s1
        else:
            s = s2

        regionnamet, clusternumt, regionnumt = py2ndlevelanalysis.get_cluster_info(namelist, nclusterlist, t)
        regionnames, clusternums, regionnums = py2ndlevelanalysis.get_cluster_info(namelist, nclusterlist, s)
        if len(regionnamet) > 4: regionnamet = regionnamet[:4]
        if len(regionnames) > 4: regionnames = regionnames[:4]
        textlabel = '{:4s}{}-{:4s}{}'.format(regionnamet,clusternumt,regionnames,clusternums)

        # prep regression lines
        b, fit, R2 = simple_GLMfit(covariates1, d)
        if twogroup: b2, fit2, R22 = simple_GLMfit(covariates2, d2)

        # create the line plot
        TargetAxes.clear()
        TargetAxes.plot(covariates1,d,'bo', markersize=3)
        TargetAxes.plot(covariates1,fit,'b-')
        TargetAxes.set_title(textlabel + ' ' + field_to_plot, fontsize=titlefont)
        if twogroup:
            TargetAxes.plot(covariates2,d2,'ro', markersize=3)
            TargetAxes.plot(covariates2,fit2,'r-')

        # add annotations
        ii = np.argmin(covariates1)
        x = covariates1[ii]
        y = fit[ii]
        if b[0] < 0:  # negative slope
            y += 0.1  # shift text upward
        else:
            y -= 0.1  # shift text downward
        R2text = 'R2 = {:.3f}'.format(R2)
        plt.text(x,y, R2text, color = 'b', fontsize=labelfont)

        if twogroup:
            # add annotations
            ii = np.argmin(covariates2)
            x = covariates2[ii]
            y = fit2[ii]
            if b2[0] < 0:  # negative slope
                y += 0.1  # shift text upward
            else:
                y -= 0.1  # shift text downward
            R22text = 'R2 = {:.3f}'.format(R22)
            TargetAxes.text(x,y,R22text, color = 'r', fontsize=labelfont)

        plt.xticks(fontsize=labelfont)
        plt.yticks(fontsize=labelfont)
        plt.tight_layout()
        TargetCanvas.draw()
        # need input for file name
        # TargetCanvas.savefig('correlation_plot.eps', format='eps')

    else:
        # connection data for network results
        networkcomponent = connectiondata['networkcomponent'][0]
        tt = connectiondata['tt'][0]
        combo = connectiondata['combo'][0]
        timepoint = connectiondata['timepoint'][0]
        ss = connectiondata['ss'][0]

        # network data
        resultsnames = data1['resultsnames']
        clustername = data1['clustername']
        clusterdata = np.load(clustername, allow_pickle=True).flat[0]
        nclusterlist = np.array([clusterdata['cluster_properties'][nn]['nclusters'] for nn in range(len(clusterdata['cluster_properties']))])
        namelist = [clusterdata['cluster_properties'][nn]['rname'] for nn in range(len(clusterdata['cluster_properties']))]
        networkmodel = data1['network']
        network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkmodel)

        if twogroup: resultsnames2 = data2['resultsnames']

        # get the data values
        fname1 = resultsnames[networkcomponent]
        ndata = np.load(fname1, allow_pickle=True).flat[0]
        ntclusters = len(ndata['sem_one_target_results'])

        targetname = network[networkcomponent]['target']
        if len(targetname) > 4: targetname = targetname[:4]
        sources = network[networkcomponent]['sources']
        targetnum = network[networkcomponent]['targetnum']
        sourcenums = network[networkcomponent]['sourcenums']

        pdata1 = ndata['sem_one_target_results'][tt]['b']
        d = pdata1[combo, timepoint, :, ss]  # one group data for one connection

        targetname = network[networkcomponent]['target']
        if len(targetname) > 4: targetname = targetname[:4]
        sourcename = namelist[sourcenums[ss]]
        mlist = pysem.ind2sub_ndims(nclusterlist[sourcenums], combo).astype(int)  # cluster number for each source
        sourcecluster = mlist[ss]

        if len(sourcename) > 4: sourcename = sourcename[:4]
        textlabel = '{:4s}{}-{:4s}{}'.format(targetname, tt, sourcename, sourcecluster)

        if twogroup:
            fname2 = resultsnames2[networkcomponent]
            ndata2 = np.load(fname2, allow_pickle=True).flat[0]
            pdata2 = ndata2['sem_one_target_results'][tt]['b']
            d2 = pdata2[combo, timepoint, :, ss]  # one group data for one connection

        # prep regression lines
        b, fit, R2 = simple_GLMfit(covariates1, d)
        if twogroup: b2, fit2, R22 = simple_GLMfit(covariates2, d2)

        # create the line plot
        TargetAxes.clear()
        TargetAxes.plot(covariates1,d,'bo', markersize=4)
        TargetAxes.plot(covariates1,fit,'b-')
        TargetAxes.set_title(textlabel + ' ' + field_to_plot, fontsize=titlefont)
        if twogroup:
            TargetAxes.plot(covariates2,d2,'ro', markersize=4)
            TargetAxes.plot(covariates2,fit2,'r-')

        # add annotations
        ii = np.argmin(covariates1)
        x = covariates1[ii]
        y = fit[ii]
        if b[0] < 0:  # negative slope
            y += 0.1  # shift text upward
        else:
            y -= 0.1  # shift text downward
        R2text = 'R2 = {:.3f}'.format(R2)
        TargetAxes.text(x,y, R2text, color = 'b', fontsize=labelfont)

        if twogroup:
            # add annotations
            ii = np.argmin(covariates2)
            x = covariates2[ii]
            y = fit2[ii]
            if b2[0] < 0:  # negative slope
                y += 0.1  # shift text upward
            else:
                y -= 0.1  # shift text downward
            R22text = 'R2 = {:.3f}'.format(R22)
            TargetAxes.text(x,y,R22text, color = 'r', fontsize=labelfont)

        plt.xticks(fontsize=labelfont)
        plt.yticks(fontsize=labelfont)
        plt.tight_layout()
        TargetCanvas.draw()
        # need input for file name
        # TargetCanvas.savefig('correlation_plot.eps', format='eps')


def get_covariate_values(DBname, DBnum, covariatename, mode = 'average_per_person'):
# load covariate values from a database, given the database name and entry numbers, and covariate name
    xls = pd.ExcelFile(DBname, engine = 'openpyxl')
    df1 = pd.read_excel(xls, 'datarecord')

    if mode == 'average_per_person':  # average values over entries for the same person
        filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, '', mode='list')
        fieldvalues = []
        for nn in range(NP):
            DBnum_person = dbnum_person_list[nn]
            fv1 = list(df1.loc[DBnum_person,covariatename])
            if type(fv1[0]) == str:
                # print('characteristic is {} value type ... using the first listed value'.format(type(fv1[0])))
                fieldvalues += [fv1[0]]
            else:
                # print('characteristic is {} value type ... using the average value for each participant'.format(type(fv1[0])))
                fieldvalues += [np.mean(fv1)]
    else:
        fieldvalues = list(df1.loc[DBnum,covariatename])

    return fieldvalues


def simple_GLMfit(x, y):
    # function to do GLM regression w.r.t. covariates
    # y = mx + b
    # Y = bG
    x = np.array(x)
    nvals = len(x)
    G = np.concatenate((x[np.newaxis,:], np.ones((1,nvals))),axis=0)   #  2 x nvals
    iGG = np.linalg.inv(G @ G.T)
    b = y @ G.T @ iGG
    fit = b @ G

    ssq = np.sum((y-np.mean(y))**2)
    residual_ssq = np.sum((y-fit)**2)
    tol = 1.0e-10
    R2 = 1.0 - residual_ssq/(ssq + tol)   # determine how much variance is explained, not including the average offset

    return b, fit, R2

#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
def display_anatomical_figure(filename, connectiondata, templatename, regioncolor, orientation, TargetCanvas = [], TargetAxes = []):

    # get the connection and region information
    data1 = np.load(filename, allow_pickle=True).flat[0]

    if data1['type'] == '2source':
        # connection data for 2source results
        t = connectiondata['t'][0]
        s1 = connectiondata['s1'][0]
        s2 = connectiondata['s2'][0]
        timepoint = connectiondata['tt'][0]
        nb = connectiondata['nb'][0]

        # 2-source SEM data
        nclusterlist = [data1['cluster_properties'][nn]['nclusters'] for nn in range(len(data1))]
        namelist = [data1['cluster_properties'][nn]['rname'] for nn in range(len(data1))]

        if nb == 0:
            s = s1
        else:
            s = s2

        regionnamet, clusternumt, regionnumt = py2ndlevelanalysis.get_cluster_info(namelist, nclusterlist, t)
        regionnames, clusternums, regionnums = py2ndlevelanalysis.get_cluster_info(namelist, nclusterlist, s)

        # get the voxel coordinates for the target region
        IDX = data1['cluster_properties'][regionnumt]['IDX']
        idxx = np.where(IDX == clusternumt)
        cx = data1['cluster_properties'][regionnumt]['cx'][idxx]
        cy = data1['cluster_properties'][regionnumt]['cy'][idxx]
        cz = data1['cluster_properties'][regionnumt]['cz'][idxx]

        # get the voxel coordinates for the source region
        IDX = data1['cluster_properties'][regionnums]['IDX']
        idxx = np.where(IDX == clusternums)
        cx2 = data1['cluster_properties'][regionnums]['cx'][idxx]
        cy2 = data1['cluster_properties'][regionnums]['cy'][idxx]
        cz2 = data1['cluster_properties'][regionnums]['cz'][idxx]

    else:
        # connection data for network results
        networkcomponent = connectiondata['networkcomponent'][0]
        tt = connectiondata['tt'][0]
        combo = connectiondata['combo'][0]
        timepoint = connectiondata['timepoint'][0]
        ss = connectiondata['ss'][0]

        # network data
        resultsnames = data1['resultsnames']
        clustername = data1['clustername']
        clusterdata = np.load(clustername, allow_pickle=True).flat[0]
        nclusterlist = np.array([clusterdata['cluster_properties'][nn]['nclusters'] for nn in range(len(clusterdata['cluster_properties']))])
        namelist = [clusterdata['cluster_properties'][nn]['rname'] for nn in range(len(clusterdata['cluster_properties']))]

        networkmodel = data1['network']
        network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkmodel)

        sources = network[networkcomponent]['sources']
        targetnum = network[networkcomponent]['targetnum']
        sourcenums = network[networkcomponent]['sourcenums']
        sourcename = namelist[sourcenums[ss]]
        mlist = pysem.ind2sub_ndims(nclusterlist[sourcenums], combo).astype(int)  # cluster number for each source
        sourcecluster = mlist[ss]

        # get the voxel coordinates for the target region
        IDX = clusterdata['cluster_properties'][targetnum]['IDX']
        idxx = np.where(IDX == tt)
        cx = clusterdata['cluster_properties'][targetnum]['cx'][idxx]
        cy = clusterdata['cluster_properties'][targetnum]['cy'][idxx]
        cz = clusterdata['cluster_properties'][targetnum]['cz'][idxx]

        # get the voxel coordinates for the source region
        IDX = clusterdata['cluster_properties'][sourcenums[ss]]['IDX']
        idxx = np.where(IDX == sourcecluster)
        cx2 = clusterdata['cluster_properties'][sourcenums[ss]]['cx'][idxx]
        cy2 = clusterdata['cluster_properties'][sourcenums[ss]]['cy'][idxx]
        cz2 = clusterdata['cluster_properties'][sourcenums[ss]]['cz'][idxx]

    #-------------------------------------------------------------------------------------
    # display one slice of an anatomical region in the selected target figure
    outputimg = pydisplayvoxelregionslice(templatename, cx, cy, cz, orientation, displayslice = [], colorlist = regioncolor)
    regioncolor2 = 1.0-np.array(regioncolor)
    outputimg2 = pydisplayvoxelregionslice(templatename, cx2, cy2, cz2, orientation, displayslice = [], colorlist = regioncolor2)

    xs,ys,nc = np.shape(outputimg)
    if xs > ys:
        bigimg = np.concatenate((outputimg,outputimg2),axis = 1)
    else:
        bigimg = np.concatenate((outputimg,outputimg2),axis = 0)
    TargetAxes.clear()
    TargetAxes.imshow(bigimg)
    plt.axis('off')
    plt.tight_layout()
    TargetCanvas.draw()