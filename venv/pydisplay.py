# set of functions for displaying fMRI results in various forms
import numpy as np
import copy
import load_templates
import scipy.ndimage as nd
import pandas as pd

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
    template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map = load_templates.load_template_and_masks(templatename, resolution)

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


def pywriteexcel(data, excelname, excelsheet = 'pydata', write_mode = 'replace'):
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
        with pd.ExcelWriter(excelname, mode='w') as writer:
            dataf.to_excel(writer, sheet_name=excelsheet, float_format = '%.2f')
    else:
        with pd.ExcelWriter(excelname, mode='a') as writer:
            dataf.to_excel(writer, sheet_name=excelsheet, float_format = '%.2f')