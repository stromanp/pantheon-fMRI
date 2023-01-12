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
import pyclustering
import pysem
import re
import nibabel as nib
import matplotlib.patches as mpatches
import pysapm

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
    templatename = templatename.lower()
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
    if len(cx) > 0:
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
            rownum = np.floor(zz/ncols).astype(int)
            x1 = colnum*xs
            x2 = (colnum+1)*xs
            y1 = rownum*ys
            y2 = (rownum+1)*ys
            # print('zz {}   rownum {} colnum {}  x1,x2 = {},{}  y1,y2 = {},{}'.format(zz,rownum,colnum,x1,x2,y1,y2))
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
        yc1 = 5;  yc2 = 25;  ys2 = yc2-yc1
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
def pydisplayvoxelregionslice(templatename, template_img, cx, cy, cz, orientation, displayslice = [], colorlist = []):

    #
    # if templatename == 'brain':
    #     resolution = 1
    #     template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_img = load_templates.load_template_and_masks(templatename, resolution)
    #     # match affine of data
    #
    #     # for brain data, need to match the template, region map, etc., to the data size/position
    #     dbhome = df1.loc[self.DBnum[0], 'datadir']
    #     fname = df1.loc[self.DBnum[0], 'niftiname']
    #     niiname = os.path.join(dbhome, fname)
    #     fullpath, filename = os.path.split(niiname)
    #     prefix_niiname = os.path.join(fullpath, self.CLprefix + filename)
    #     temp_data = nib.load(prefix_niiname)
    #     img_data_affine = temp_data.affine
    #     hdr = temp_data.header
    #     template_img = i3d.convert_affine_matrices_nearest(template_img, template_affine, img_data_affine, hdr['dim'][1:4])
    #     regionmap_img = i3d.convert_affine_matrices_nearest(regionmap_img, template_affine, img_data_affine, hdr['dim'][1:4])
    #
    # else:
    #     resolution = 1
    #     template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_img = load_templates.load_template_and_masks(templatename, resolution)

    # anatnamelist = []
    # for name in anatlabels['names']:
    #     anatnamelist.append(name)

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
    if templatename.lower()  == 'brain':
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
        if templatename.lower() == 'ccbs':
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


def pydisplay_data_from_database(DBname, DBnumlist, prefix, orientation='sagittal', windownum = 100, refresh_interval = 2.0):

    if isinstance(DBnumlist,str):
        DBnumlist = parsenumlist(DBnumlist)

    xls = pd.ExcelFile(DBname, engine = 'openpyxl')
    df1 = pd.read_excel(xls, 'datarecord')

    plt.close(windownum)
    fig = plt.figure(windownum)

    for dbnum in DBnumlist:
        plt.cla()
        dbhome = df1.loc[dbnum, 'datadir']
        fname = df1.loc[dbnum, 'niftiname']
        niiname = os.path.join(dbhome, fname)

        p,f = os.path.split(niiname)
        prefix_niiname = os.path.join(p,prefix+f)

        temp_data = nib.load(prefix_niiname)
        img_data = temp_data.get_fdata()

        if np.ndim(temp_data) > 3:
            img_data = np.mean(img_data,axis = 3)

        xs,ys,zs = np.shape(img_data)
        dannot = 2

        if orientation not in ['axial','sagittal','coronal']:
            orientation = 'axial'

        if orientation == 'axial':
            z0 = np.floor(zs/2).astype(int)
            plt.imshow(img_data[:,:,z0],'gray')
            xa1 = dannot
            xa2 = ys-dannot
            ya1 = dannot
            ya2 = xs-dannot

        if orientation == 'sagittal':
            x0 = np.floor(xs / 2).astype(int)
            plt.imshow(img_data[x0, :, :], 'gray')
            xa1 = 5
            xa2 = zs-10
            ya1 = 5
            ya2 = ys-10

        if orientation == 'coronal':
            y0 = np.floor(ys / 2).astype(int)
            plt.imshow(img_data[:, y0, :], 'gray')
            xa1 = dannot
            xa2 = zs-dannot
            ya1 = dannot
            ya2 = xs-dannot

        plt.annotate('{}'.format(dbnum),(xa1,ya1), color = [1,1,0],fontsize = 11, horizontalalignment='left')
        plt.annotate('{}'.format(dbnum),(xa1,ya2), color = [1,1,0],fontsize = 11, horizontalalignment='left')
        plt.annotate('{}'.format(dbnum),(xa2,ya1), color = [0,1,0],fontsize = 11, horizontalalignment='right')
        plt.annotate('{}'.format(dbnum),(xa2,ya2), color = [0,1,0],fontsize = 11, horizontalalignment='right')
        plt.show()
        plt.pause(refresh_interval)


def parsenumlist(entered_text):
    # need to make sure we are working with numbers, not text
    # first, replace any double spaces with single spaces, and then replace spaces with commas
    entered_text = re.sub('\ +', ',', entered_text)
    entered_text = re.sub('\,\,+', ',', entered_text)
    # remove any leading or trailing commas
    if entered_text[0] == ',': entered_text = entered_text[1:]
    if entered_text[-1] == ',': entered_text = entered_text[:-1]

    # fix up the text bit, allow the user to specify ranges of numbers with a colon
    # need to change the text to write out the range
    # first see if there are any colons included in the text
    # this part is complicated - need to find any pairs of numbers separated by a colon, to indicate a range
    m = re.search(r'\d*:\d*', entered_text)
    while m:
        # m[0] is the string that matches what we are looking for - two numbers separated by a colon
        # in m[0] we find where there is the colon, with m[0].find(':')
        # so, within m[0] everything in the range of :m[0].find(':') is the first number
        # and everything in the range of (m[0].find(':')+1): is the second number
        num1 = int(m[0][:m[0].find(':')])
        num2 = int(m[0][(m[0].find(':') + 1):])
        # now create the long string that we need to replace the short form indicated with the colon
        numbers = np.arange(num1, num2 + 1)
        numtext = ''
        for n in numbers:
            numtext1 = '{:d},'.format(n)
            numtext += numtext1
        # now insert this where the colon separated pair of number came out
        new_text = entered_text[:m.start()] + numtext + entered_text[(m.end() + 1):]
        entered_text = new_text
        m = re.search(r'\d*:\d*', entered_text)

    entered_values = np.fromstring(entered_text, dtype=int, sep=',')

    return entered_values




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
        print('pywriteexcel: size of input data is {}'.format(np.shape(data)))
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
    if len(filename2) > 0 and os.path.exists(filename2):
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
            textlabel = '{:4s}{}-{:4s}{}'.format(regionnames,clusternums,regionnamet,clusternumt)
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
            TargetAxes.set_xticklabels(plotlabel, rotation = 45, fontsize=labelfont)
            # plt.tight_layout()
        else:
            TargetAxes.boxplot(plotdata_g1, notch = True, showfliers = False)
            TargetAxes.set_xticklabels(plotlabel, rotation = 45, fontsize=labelfont)
            # plt.tight_layout()

        TargetAxes.set_title(field_to_plot, fontsize=titlefont)
        plt.yticks(fontsize=labelfont)
        plt.tight_layout()
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

        # do not bother sorting by network component number ...
        # collect the data values to plot
        plotdata_g1 = []
        plotdata_g2 = []
        plotlabel = []
        for nn,networknumber in enumerate(networkcomponent):
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

            pdata1 = ndata['sem_one_target_results'][tt[nn]]['b']
            ncombo, ntime, NP, ns = np.shape(pdata1)

            d = pdata1[combo[nn],timepoint[nn],:,ss[nn]]   # one group data for one connection
            plotdata_g1.append(d)

            # sourcename = cluster_info[sourcenums[ss2[nn]]]['rname']
            sourcename = namelist[sourcenums[ss[nn]]]
            mlist = pysem.ind2sub_ndims(nclusterlist[sourcenums], combo[nn]).astype(int)  # cluster number for each source
            sourcecluster = mlist[ss[nn]]

            if len(sourcename) > 4: sourcename = sourcename[:4]
            textlabel = '{:4s}{}-{:4s}{}'.format(sourcename,sourcecluster,targetname,tt[nn])
            plotlabel.append(textlabel)
            if twogroup:
                pdata2 = ndata2['sem_one_target_results'][tt[nn]]['b']
                d = pdata2[combo[nn],timepoint[nn],:,ss[nn]]   # one group data for one connection
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
            TargetAxes.set_xticklabels(plotlabel, rotation = 45, fontsize=labelfont)
            # plt.tight_layout()
        else:
            TargetAxes.boxplot(plotdata_g1, notch = True, showfliers = False)
            TargetAxes.set_xticklabels(plotlabel, rotation = 45, fontsize=labelfont)
            # plt.tight_layout()

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
    if len(filename2) > 0 and os.path.exists(filename2):
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
        textlabel = '{:4s}{}-{:4s}{}'.format(regionnames,clusternums,regionnamet,clusternumt)

        # prep regression lines
        b, fit, R2 = simple_GLMfit(covariates1, d)
        if twogroup: b2, fit2, R22 = simple_GLMfit(covariates2, d2)

        # create the line plot
        TargetAxes.clear()
        TargetAxes.plot(covariates1,d, color=(0,1,0), linestyle='None', marker='o', markerfacecolor=(0,1,0), markersize=3)
        TargetAxes.plot(covariates1,fit, color=(0,1,0), linestyle='solid', marker='None')
        TargetAxes.set_title(textlabel + ' ' + field_to_plot, fontsize=titlefont)
        if twogroup:
            TargetAxes.plot(covariates2,d2, color=(1.0,0.5,0.), linestyle='None', marker='o', markerfacecolor=(1.0,0.5,0.), markersize=3)
            TargetAxes.plot(covariates2,fit2, color=(1.0,0.5,0.), linestyle='solid', marker='None')

        # add annotations
        ii = np.argmin(covariates1)
        x = covariates1[ii]
        y = fit[ii]
        if b[0] < 0:  # negative slope
            y += 0.1  # shift text upward
        else:
            y -= 0.1  # shift text downward
        R2text = 'R2 = {:.3f}'.format(R2)
        TargetAxes.text(x,y, R2text, color = (0,1,0), fontsize=labelfont)

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
            TargetAxes.text(x,y,R22text, color=(1.0,0.5,0.), fontsize=labelfont)

        plt.xticks(fontsize=labelfont)
        plt.yticks(fontsize=labelfont)
        # plt.tight_layout()
        TargetCanvas.draw()
        # need input for file name
        # TargetCanvas.savefig('correlation_plot.eps', format='eps')

    else:
        # connection data for network results
        print('connectiondata = {}'.format(connectiondata))

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
        textlabel = '{:4s}{}-{:4s}{}'.format(sourcename, sourcecluster, targetname, tt)

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
        TargetAxes.plot(covariates1,d, color=(0,1,0), linestyle='None', marker='o', markerfacecolor=(0,1,0), markersize=3)
        TargetAxes.plot(covariates1,fit, color=(0,1,0), linestyle='solid', marker='None')
        TargetAxes.set_title(textlabel + ' ' + field_to_plot, fontsize=titlefont)
        if twogroup:
            TargetAxes.plot(covariates2,d2, color=(1.0,0.5,0), linestyle='None', marker='o', markerfacecolor=(1.0,0.5,0), markersize=3)
            TargetAxes.plot(covariates2,fit2, color=(1.0,0.5,0), linestyle='solid', marker='None')

        # add annotations
        ii = np.argmin(covariates1)
        x = covariates1[ii]
        y = fit[ii]
        if b[0] < 0:  # negative slope
            y += 0.1  # shift text upward
        else:
            y -= 0.1  # shift text downward
        R2text = 'R2 = {:.3f}'.format(R2)
        TargetAxes.text(x,y, R2text, color = (0,1,0), fontsize=labelfont)

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
            TargetAxes.text(x,y,R22text, color = (1.0,0.5,0), fontsize=labelfont)

        plt.xticks(fontsize=labelfont)
        plt.yticks(fontsize=labelfont)
        # plt.tight_layout()
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

        clustername = data1['clustername']
        clusterdata = np.load(clustername, allow_pickle=True).flat[0]
        template_img = clusterdata['template_img']

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

        template_img = clusterdata['template_img']

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
    outputimg = pydisplayvoxelregionslice(templatename, template_img, cx, cy, cz, orientation, displayslice = [], colorlist = regioncolor)
    regioncolor2 = 1.0-np.array(regioncolor)
    outputimg2 = pydisplayvoxelregionslice(templatename, template_img, cx2, cy2, cz2, orientation, displayslice = [], colorlist = regioncolor2)

    xs,ys,nc = np.shape(outputimg)
    if xs > ys:
        bigimg = np.concatenate((outputimg,outputimg2),axis = 1)
    else:
        bigimg = np.concatenate((outputimg,outputimg2),axis = 0)
    TargetAxes.clear()
    TargetAxes.imshow(bigimg)
    plt.axis('off')
    # plt.tight_layout()
    TargetCanvas.draw()


def display_sapm_cluster(clusterdataname, regionname, clusternumber):
    cluster_data = np.load(clusterdataname, allow_pickle=True).flat[0]
    cluster_properties = cluster_data['cluster_properties']
    # dict_keys(['cx', 'cy', 'cz', 'IDX', 'nclusters', 'rname', 'regionindex', 'regionnum'])

    nregions = len(cluster_properties)
    nclusterlist = [cluster_properties[i]['nclusters'] for i in range(nregions)]
    rnamelist = [cluster_properties[i]['rname'] for i in range(nregions)]
    nclusterstotal = np.sum(nclusterlist)

    r = rnamelist.index(regionname)
    cx = cluster_properties[r]['cx']
    cy = cluster_properties[r]['cy']
    cz = cluster_properties[r]['cz']



# draw SAPM diagram for single output model----------------------------------------------------
def draw_sem_plot(results_file, sheetname, rownumbers, drawregionsfile, statname, scalefactor, thresholdtext = 'abs>0', writefigure = False):
    # draw a plot of connections between anatomical regions based on SEM results, and a definition of how to draw the network
    regionnames = 'regions'
    figurenumber = 201
    regions = pysapm.define_drawing_regions_from_file(drawregionsfile)

    xls = pd.ExcelFile(results_file, engine='openpyxl')
    df1 = pd.read_excel(xls, sheetname)

    comparisontext, absval, threshold = parse_threshold_text(thresholdtext)
    if statname[0] == 'T':
        statvals = df1['Tvalue']
    else:
        statvals = df1['v1']

    # set rownumbers if not set
    if isinstance(rownumbers,str):
        rownumbers = list(range(len(statvals)))

    # # set scale factor if it is set to 'auto'
    if isinstance(scalefactor,str):
        maxval = 3.0
        maxstat = np.max(np.abs(statvals[rownumbers]))
        scalefactor = maxval/maxstat

    regionlist = [regions[x]['name'] for x in range(len(regions))]
    regionlist_trunc = [regions[x]['name'][:4] for x in range(len(regions))]

    # set some drawing parameters
    ovalsize = (0.1,0.05)
    width = 0.001
    ovalcolor = [0,0,0]

    # start drawing
    plt.close(figurenumber)
    fig = plt.figure(figurenumber)
    ax = fig.add_axes([0,0,1,1])

    # add ellipses and labels
    for nn in range(len(regions)):
        ellipse = mpatches.Ellipse(regions[nn]['pos'],ovalsize[0],ovalsize[1], alpha = 0.3)
        ax.add_patch(ellipse)
        ax.annotate(regions[nn]['name'],regions[nn]['pos']+regions[nn]['labeloffset'])

    an_list = []
    connection_list = []
    acount = 0
    for nn in rownumbers:
        # plot lines for connections
        m = statvals[nn]
        if comparisontext == '>':
            if absval:
                statcondition = np.abs(m) > threshold
                linethick = np.min([5.0, np.abs(m)*scalefactor])
            else:
                statcondition = m > threshold
                linethick = np.min([5.0, np.abs(m)*scalefactor])
                if threshold < 0:
                    linethick = np.max([0.5, linethick])
        else:
            if absval:
                statcondition = np.abs(m) < threshold
                linethick = np.min([5.0, np.abs(m)*scalefactor])
                linethick = np.max([0.5, linethick])
            else:
                statcondition = m < threshold
                linethick = np.min([5.0, np.abs(m)*scalefactor])

        if statcondition:
            if m > 0:
                linecolor = 'k'
            else:
                linecolor = 'r'

            # get coordinates for the two ends of the arrow for one connection
            # need to first identify the regions that are connected
            startpoint, endpoint = connectivity_plot_entry(df1.loc[[nn]],regions, ovalsize)
            connection_type1 = {'con':'{}-{}'.format(df1.iloc[nn]['sname'],df1.iloc[nn]['tname']), 'type':'input'}

            an1 = ax.annotate('',xy=startpoint,xytext = endpoint, arrowprops=dict(arrowstyle="->", connectionstyle='arc3', linewidth = linethick, color = linecolor, shrinkA = 0.01, shrinkB = 0.01))
            acount+= 1
            an_list.append(an1)
            connection_list.append(connection_type1)

    svgname = 'none'
    if writefigure:
        p,f1 = os.path.split(results_file)
        f,e = os.path.splitext(f1)
        svgname = os.path.join(p,f+'_'+statname+'_SEMnetwork.svg')
        plt.figure(figurenumber)
        plt.savefig(svgname, format='svg')

    return svgname


def connectivity_plot_entry(connection,regions, ovalsize):
    # connection input must be in the form of a pandas dataframe with the following entries
    # tname, tcluster, sname, scluster, Tvalue, v1, v1sem, tx, ty, tz, tlimx1, tlimx2, tlimy1, tlimy2, tlimz1, tlimz2,
    # sx, sy, sz, slimx1, slimx2, slimy1, slimy2, slimz1, slimz2, networkcomponent, tt, combo, timepoint, ss
    regionlist = [regions[x]['name'] for x in range(len(regions))]
    target = connection.iloc[0]['tname']
    source = connection.iloc[0]['sname']

    source_offset = np.array([ovalsize[0]*(connection.iloc[0]['sx']-connection.iloc[0]['slimx1'])/(connection.iloc[0]['slimx2']-connection.iloc[0]['slimx1']),
                              ovalsize[1]*(connection.iloc[0]['sy']-connection.iloc[0]['slimy1'])/(connection.iloc[0]['slimy2']-connection.iloc[0]['slimy1'])])
    target_offset = np.array([ovalsize[0]*(connection.iloc[0]['tx']-connection.iloc[0]['tlimx1'])/(connection.iloc[0]['tlimx2']-connection.iloc[0]['tlimx1']),
                              ovalsize[1]*(connection.iloc[0]['ty']-connection.iloc[0]['tlimy1'])/(connection.iloc[0]['tlimy2']-connection.iloc[0]['tlimy1'])])

    ss = regionlist.index(source)
    tt = regionlist.index(target)

    sourcepos = np.array(regions[ss]['pos'])
    targetpos = np.array(regions[tt]['pos'])

    startpoint = sourcepos + source_offset - np.array(ovalsize)/2
    endpoint = targetpos + target_offset - np.array(ovalsize)/2

    return startpoint, endpoint


def parse_threshold_text(thresholdtext):
    # parse thresholdtext
    if '<' in thresholdtext:
        c = thresholdtext.index('<')
        comparisontext = '<'
    else:
        c = thresholdtext.index('>')
        comparisontext = '>'
    threshold = float(thresholdtext[(c+1):])

    if c > 0:
        if 'mag' in thresholdtext[:c]:
            absval = False
        if 'abs' in thresholdtext[:c]:
            absval = True
    else:
        absval = False

    if absval:
        print('threshold is set to absolute value {} {}'.format(comparisontext, threshold))
    else:
        print('threshold is set to {} {}'.format(comparisontext, threshold))
    return comparisontext, absval, threshold


def parse_connection_name(connection, regionlist):
    h1 = connection.index('-')
    if '-' in connection[(h1+2):]:
        h2 = connection[(h1+2):].index('-') + h1 + 2
        r1 = connection[:h1]
        r2 = connection[(h1+1):h2]
        r3 = connection[(h2+1):]

        i1 = regionlist.index(r1)
        i2 = regionlist.index(r2)
        i3 = regionlist.index(r3)
    else:
        r1 = connection[:h1]
        r2 = connection[(h1+1):]
        r3 = 'none'

        i1 = regionlist.index(r1)
        i2 = regionlist.index(r2)
        i3 = -1

    return (r1,r2,r3),(i1,i2,i3)


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
