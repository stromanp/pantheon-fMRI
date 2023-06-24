# function to map sections of a template to corresponding locations in an input image
# converted from the matlab version

# [coords, angle, angley, section_mapping_coords, original_section, template_section] = calculate_section_positions(
#     template, background2, section_defs(ss), pos_estimate, angle_estimate, position_stiffness, angle_stiffness,
#     fixZ_flag, fix_position_flag, error_correction_flag);

import numpy as np
import nibabel as nib
import image_operations_3D as i3d
import math
import matplotlib.pyplot as plt
from scipy import interpolate
import copy
import os
import py_mirt3D as mirt
from PIL import Image, ImageTk
import tkinter as tk
import load_templates

# -------------rotation_matrix-------------------------------------------------
# ------------------------------------------------------------------------------------
def rotation_matrix(input_angle, axis = 0):
    angle = input_angle*np.pi/180.

    if axis == 0:  # rotate around 1st axis
        M = [[1,0,0],[0,math.cos(angle),-math.sin(angle)],[0,math.sin(angle),math.cos(angle)]]

    if axis == 1:  # rotate around 2nd axis
        M = [[math.cos(angle),0,-math.sin(angle)],[0,1,0],[math.sin(angle),0,math.cos(angle)]]

    if axis == 2:  # rotate around 3rd axis
        M = [[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]]

    return M


# -------------py_calculate_section_positions-------------------------------------------------
# ------------------------------------------------------------------------------------
def py_calculate_section_positions(template, img, section_defs, angle_list, pos_estimate, angle_estimate, position_stiffness,
                                   angle_stiffness, fixZ_flag=0, fix_position_flag=0, error_correction_pass=0, verbose = False):

    print('running py_calculate_section_positions ... ')

    # input is the entire template, and sections are extracted
    xt, yt, zt = np.shape(template)
    xs, ys, zs = np.shape(img)
    img = img/np.max(img)

    # section_defs relate to the template
    zpos = section_defs['center'][2]
    dz = section_defs['dims'][2]
    zr = [zpos - dz, zpos + dz]
    ypos = section_defs['center'][1]
    dy = section_defs['dims'][1]
    yr = [ypos - dy, ypos + dy]
    xpos = section_defs['center'][0]
    dx = section_defs['dims'][0]
    xr = [xpos - dx, xpos + dx]

    dzm = dz
    dym = dy
    zpos2 = zpos
    x1 = 0
    x2 = 2 * dx + 1
    y1 = 0
    y2 = 2*dy+1
    z1 = 0
    z2 = 2*dz+1
    print('coords x ', x1, ' ', x2, '  y ', y1, ' ', y2, '  z ', z1, ' ', z2)
    print('template coords x ', xpos - dx, ' ', xpos + dx, '  y ', ypos - dym, ' ', ypos + dym, '  z ', zpos2 - dzm,
          ' ', zpos2 + dzm)

    if fix_position_flag > 0:
        coords = pos_estimate  # estimates of position and angle are relative to the image, not the template
        angle = angle_estimate
        if np.size(angle) < 1:
            angle = 0
        angley = 0
        # dzm = dz
        # dym = dy
        # zpos2 = zpos
        # x1 = 0
        # x2 = 2*dx + 1
        # y1 = dym - dy
        # y2 = dym + dy
        # z1 = dzm - dz
        # z2 = dzm + dz
    else:
        if np.size(angle_estimate) == 0:
            angle_weight = np.ones((np.size(angle_list)))
        else:
            aw = angle_stiffness
            angle_diff = np.abs(angle_list - angle_estimate)
            angle_weight = 1/(aw*angle_diff + 1)

        # dzm = dz
        # dym = dy
        # zpos2 = zpos
        # x1 = 0
        # x2 = 2 * dx + 1
        # y1 = dym - dy
        # y2 = dym + dy
        # z1 = dzm - dz
        # z2 = dzm + dz
        # print('coords x ',x1,' ',x2,'  y ',y1,' ',y2,'  z ',z1,' ',z2)
        # print('template coords x ',xpos-dx,' ',xpos+dx,'  y ',ypos-dym,' ',ypos+dym,'  z ',zpos2-dzm,' ',zpos2+dzm)

        # rotate the template if desired around xpos, ypos, zpos, then select
        # the rectangular region, and subtract / add the rotation angle from the result
        # added angle needs to be considered when predicting position of next
        # section, and when showing fit sections

        # get the template section - do the reverse rotations
        # this only accounts for the rectangular volume not being aligned with the principal axes
        # if the template is rotated by 'xrot' before the section is extracted, then angle = 0
        # is with the image data rotated by 'xrot' also
        if section_defs['yrot'] == 0:
            rotated_image = template
        else:
            rotated_image = i3d.rotate3D(template, -section_defs['yrot'], section_defs['center'], 1)
        if section_defs['xrot'] != 0:
            rotated_image = i3d.rotate3D(rotated_image, -section_defs['xrot'], section_defs['center'], 0)

        # crop the rotated image back to the correct size, and take out any undefined values
        # temp = rotated_image[xpos-dx:xpos+dx, ypos-dym:ypos+dym, zpos2-dzm:zpos2+dzm]   # take the section from the rotated template
        temp = rotated_image[(xpos - dx):(xpos + dx + 1), (ypos - dym):(ypos + dym + 1), (zpos2 - dzm):(zpos2 + dzm + 1)]
        temp = np.where(np.isfinite(temp), temp, 0.0)  # take out any undefined values
        template_section = temp
        # check = np.where(np.isnan(temp))
        # temp[check] = 0
        xr,yr,zr = np.shape(temp)
        xmid = np.round(xr/2).astype(int)
        window1_display = temp[xmid, :, :]
        fig = plt.figure(1), plt.imshow(window1_display, 'gray')

        # step through the range of rotation angles and find the best fit
        angley_list = np.linspace(-3,3,3)
        cclist = np.zeros((np.size(angle_list), np.size(angley_list), 3))
        cclistR = np.zeros((np.size(angle_list), np.size(angley_list), 3))
        print('searching rotation angles ...')
        mlist = np.zeros((np.size(angle_list), np.size(angley_list)))
        for nn, angle_value in enumerate(angle_list):
            # angle = angle_value + section_defs['xrot'] # account for section rotation - check the direction is correct
            angle = angle_value - section_defs['xrot'] # the range of image angles to search, relative to the template rotation
            # print('angle = ',angle_value)

            # it is the input image that is rotated to see where the template section lines up
            # this way, the unreliable values are out at the edge of the input image
            # positive rotation is clockwise when looking down the axis
            p0 = np.round(np.array([xs,ys,zs])/2).astype('int')
            imgR = i3d.rotate3D(img, angle, p0 , 0)
            window1_display = imgR[9, :, :]
            fig = plt.figure(1), plt.imshow(window1_display, 'gray')

            for mm,angley in enumerate(angley_list):
                p0 = np.round(np.array([xs,ys,zs])/2).astype('int')
                imgRR = i3d.rotate3D(imgR, angley, p0 , 1)
                imgRR = np.where(np.isnan(imgRR),0,imgRR)
                # calculate the cross-correlation between the rotated image and the template section
                cc = i3d.normxcorr3(imgRR/np.max(imgRR), temp, shape='same')

                if np.size(pos_estimate) < 1:
                    pos_weight = np.ones((xs, ys, zs))
                    pos_estimateR = np.nan
                else:
                    # pos estimate is in relation to the original image, not the rotated one
                    # need to rotate pos estimate as well
                    Mx = rotation_matrix(angle, axis=0)
                    My = rotation_matrix(angley, axis=1)
                    Mtotal = np.dot(Mx,My)
                    pos_estimateR = np.round(np.dot((pos_estimate - p0),Mtotal) + p0)
                    pw = position_stiffness
                    ddx, ddy, ddz = np.mgrid[range(xs), range(ys), range(zs)]

                    # find the combination of correlation and proximity to the expected location
                    dist = np.sqrt((ddx - pos_estimateR[0])**2 + (ddy - pos_estimateR[1])**2 + (ddz - pos_estimateR[2])**2)
                    pos_weight = 1/(pw*dist + 1)
                    if fixZ_flag > 0:
                        dist = np.sqrt((ddx - pos_estimateR[0])**2 + (ddy - pos_estimateR[1])**2 + 10*(ddz - pos_estimateR[2])**2)
                        pos_weight = 1/(pw*dist + 1)

                cc_temp = cc*pos_weight
                m = np.max(cc_temp)
                xp,yp,zp = np.where(cc_temp == m)
                bestposR = np.array([xp[0], yp[0], zp[0]])

                # need to find where the rotated template would map, if the image had not been rotated around its center,
                # but instead the template had been rotated.

                # save the coordinates in the rotated image, and map them to the original image later
                Mx = rotation_matrix(-angle, axis=0)
                My = rotation_matrix(-angley, axis=1)
                Mtotal = np.dot(My,Mx)
                bestpos = np.dot((bestposR - p0),Mtotal) + p0

                mlist[nn, mm] = m
                # cclist[nn, mm, 0:3] = np.round(bestpos).astype('int')
                cclistR[nn, mm, 0:3] = bestposR
                cclist[nn, mm, 0:3] = bestpos
                if bestpos[0] < 0:
                    print('pos_estimate: ', pos_estimate, ' pos_estimateR: ', pos_estimateR)
                    print('bestpos: ', bestpos, ' bestposR: ', bestposR)
                    if error_correction_pass > 0:
                        print('something is wrong - attempt to fix it did not work ...')
                    else:
                        print('something is wrong - will try to fix it ...')
        # coordinates need to be rotated back to the original img orientation later on, before they are used

        # finished going through the angle combinations, now find the best result
        angle_weight_extend = np.tile(angle_weight[:,np.newaxis],[1,np.size(angley_list)])
        mtemp = mlist*angle_weight_extend
        maxc = np.max(mtemp)
        nc,mc = np.where(mtemp == maxc)
        coords = np.squeeze(cclist[nc, mc, 0:3])
        coordsR = np.squeeze(cclistR[nc, mc, 0:3])
        # angle_list is relative to the rotated template
        angle = angle_list[nc]- section_defs['xrot']
        angley = angley_list[mc]

        print('best angle = ',angle,' angley = ', angley)
        print('best position:  ',coords)

    # get the coordinates for the image, where the template section maps to
    # first, get the coordinates in the rotated template
    # ===> need the coordinates in the image, not in the template
    # Xt, etc are template coordinates
    Xt, Yt, Zt = np.mgrid[(xpos-dx):(xpos+dx):(2*dx+1)*1j, (ypos-dym):(ypos+dym):(2*dym+1)*1j, (zpos-dzm):(zpos+dzm):(2*dzm+1)*1j]
    # X etc are image coordinates, in the rotate image to start with
    X, Y, Z = np.mgrid[(coordsR[0]-dx):(coordsR[0]+dx):(2*dx+1)*1j, (coordsR[1]-dym):(coordsR[1]+dym):(2*dym+1)*1j, (coordsR[2]-dzm):(coordsR[2]+dzm):(2*dzm+1)*1j]
    # next, need to rotate both sets of coordinates, -xrot for the template, and angle-xrot for the image
    # then rotate these coordinates by section_defs['sectionangle']

    # display the intermediate results
    if verbose:
        tempcopy = copy.deepcopy(rotated_image)
        imgcopy = copy.deepcopy(imgRR)
        tempcopy[Xt.astype('int'), Yt.astype('int'), Zt.astype('int')] = 1
        imgcopy[X.astype('int'), Y.astype('int'), Z.astype('int')] = 1
        fig = plt.figure(3), plt.imshow(imgcopy[9, :, :], 'gray')
        fig = plt.figure(4), plt.imshow(tempcopy[13, :, :], 'gray')

    # this is for rotating the image coordinates
    p0 = np.round(np.array([xs, ys, zs]) / 2).astype('int')
    sa = (np.pi/180)*angle
    Xr = X
    Yr = (Y - p0[1])*math.cos(sa) - (Z - p0[2])*math.sin(sa) + p0[1]
    Zr = (Z - p0[2])*math.cos(sa) + (Y - p0[1])*math.sin(sa) + p0[2]

    sa = (np.pi/180)*angley
    Xrr = (Xr - p0[0])*math.cos(sa) - (Zr - p0[2])*math.sin(sa) + p0[0]
    Yrr = Yr
    Zrr = (Zr - p0[2])*math.cos(sa) + (Xr - p0[0])*math.sin(sa) + p0[2]

    Xr = Xrr
    Yr = Yrr
    Zr = Zrr

    # this is for rotating the template coordinates
    sa = (np.pi/180)*(-section_defs['xrot'])
    Xtr = Xt
    Ytr = (Yt - ypos)*math.cos(sa) - (Zt - zpos) * math.sin(sa) + ypos
    Ztr = (Zt - zpos) * math.cos(sa) + (Yt - ypos) * math.sin(sa) + zpos

    sa = (np.pi/180)*(angley)
    Xtrr = (Xtr - xpos)*math.cos(sa) - (Ztr - zpos)*math.sin(sa) + xpos
    Ytrr = Ytr
    Ztrr = (Ztr - zpos)*math.cos(sa) + (Xtr - xpos)*math.sin(sa) + zpos

    Xtr = Xtrr
    Ytr = Ytrr
    Ztr = Ztrr

    # make sure coordinates are within limits
    Xtr = np.where(Xtr < 0, 0, Xtr)
    Xtr = np.where(Xtr >= xt, xt-1, Xtr)
    Ytr = np.where(Ytr < 0, 0, Ytr)
    Ytr = np.where(Ytr >= yt, yt-1, Ytr)
    Ztr = np.where(Ztr < 0, 0, Ztr)
    Ztr = np.where(Ztr >= zt, zt-1, Ztr)

    # use these limits because values will be rounded when used
    Xr = np.where(Xr <= -0.5, 0, Xr)
    Xr = np.where(Xr >= xs-0.5, xs - 1, Xr)
    Yr = np.where(Yr <= -0.5, 0, Yr)
    Yr = np.where(Yr >= ys-0.5, ys - 1, Yr)
    Zr = np.where(Zr <= -0.5, 0, Zr)
    Zr = np.where(Zr >= zs-0.5, zs - 1, Zr)

    # extract the template and corresponding image sections - for checking on the results
    template_section_check = template[Xtr.astype('int'),Ytr.astype('int'),Ztr.astype('int')]    # check on this, see if it converted properly from matlab
    original_section = img[np.round(Xr).astype('int'),np.round(Yr).astype('int'),np.round(Zr).astype('int')]

    # organize the outputs
    section_mapping_coords = {'X':Xr,'Y':Yr,'Z':Zr,'Xt':Xtr,'Yt':Ytr,'Zt':Ztr}

    return coords, angle, angley, section_mapping_coords, original_section, template_section, window1_display



# -------------py_calculate_chainlink_positions-------------------------------------------------
# ------------------------------------------------------------------------------------
def py_calculate_chainlink_positions(template, img, section_defs, angle_list, fixedpoint, angle_estimate, angle_stiffness, reverse_order = False):
    # fixedpoint is in terms of the image coordinates
    # template is the entire template, not just the section of interest

    print('running py_calculate_chainlink_positions ...  ')

    map_success = True

    xt, yt, zt = np.shape(template)
    xs, ys, zs = np.shape(img)
    img = img/np.max(img)

    zpos = np.round(section_defs['center'][2]).astype('int')
    dz = np.round(section_defs['dims'][2]).astype('int')
    zr = [zpos - dz, zpos + dz]
    ypos = np.round(section_defs['center'][1]).astype('int')
    dy = np.round(section_defs['dims'][1]).astype('int')
    yr = [ypos - dy, ypos + dy]
    xpos = np.round(section_defs['center'][0]).astype('int')
    dx = np.round(section_defs['dims'][0]).astype('int')
    xr = [xpos - dx, xpos + dx]

    # point in the template that links to the previous fixed point
    # if reverse_order:
    #     vcenter_fixedpoint = section_defs['center'] + section_defs['fixedpoint1']
    # else:
    #     vcenter_fixedpoint = section_defs['center'] - section_defs['fixedpoint1'] # vector from fixed_point to section center, in the template
    # correction --------------
    vcenter_fixedpoint = section_defs['center'] - section_defs[
        'fixedpoint1']  # vector from fixed_point to section center, in the template

    aw = angle_stiffness
    angle_diff = np.abs(angle_list - angle_estimate) # angle_estimate must be provided
    angle_weight = 1/(aw*angle_diff + 1)

    if zpos < (2*dz + 1):  # allow room at the edge of the volume for rotation
        zpos2 = 2*dz+1
    else:
        zpos2 = zpos

    dzm = np.floor(np.min([zpos2,(zt-zpos2-1)])).astype('int')
    dym = np.floor(np.min([ypos,(yt-ypos-1)])).astype('int')
    dy = np.min([dym, dy]).astype('int')
    dz = np.min([dzm, dz]).astype('int')

    # get the template section - do the reverse rotations
    # this only accounts for the rectangular volume not being aligned with the principal axes
    # if the template is rotated by 'xrot' before the section is extracted, then angle = 0
    # is with the image data rotated by 'xrot' also
    if section_defs['yrot'] == 0:
        rotated_image = template
    else:
        rotated_image = i3d.rotate3D(template, -section_defs['yrot'], section_defs['center'], 1)
    if section_defs['xrot'] != 0:
        rotated_image = i3d.rotate3D(rotated_image, -section_defs['xrot'], section_defs['center'], 0)

    # make sure there is consistency between the fixedpoint, and the edges of the template section,
    # and the edges of the image section ................

    # crop the rotated image back to the correct size, and take out any undefined values
    temp = rotated_image[(xpos-dx):(xpos+dx+1), (ypos-dy):(ypos+dy+1), (zpos2-dz):(zpos2+dz+1)]
    temp = np.where(np.isfinite(temp), temp, 0.0)   # take out any undefined values
    temp = temp/np.max(temp)
    template_section = temp

    # get the coordinates that define the template section
    xr, yr, zr = np.shape(temp)
    x1 = 0
    x2 = 2*dx+1
    y1 = 0
    y2 = 2*dy+1
    z1 = 0
    z2 = 2*dz+1
    print('coords x ', x1,' ',x2,' y ', y1,' ',y2,' z ',z1,' ',z2)

    angley_list = np.linspace(-2,2,3)   # if angles are any smaller they are indistinguishable
    angley_weight = np.array([0.5, 1.0, 0.5]) # slight weighting toward angley = 0

    mlist = np.zeros((np.size(angle_list), np.size(angley_list)))
    cclist = np.zeros((np.size(angle_list), np.size(angley_list),3))

    dtor = np.pi/180 # convert from degrees to radians

    # in the rotated image, the section position is the same every time
    # if reverse_order:
    #     section_position = np.round(fixedpoint - vcenter_fixedpoint)
    # else:
    #     section_position = np.round(fixedpoint + vcenter_fixedpoint)
    section_position = np.round(fixedpoint + vcenter_fixedpoint)

    print('section_position = ',section_position)

    x = np.linspace(section_position[0] - dx, section_position[0] + dx, 2 * dx + 1).astype('int')
    cx = np.where((x >= 0) & (x < xs))
    y = np.linspace(section_position[1] - dy, section_position[1] + dy, 2 * dy + 1).astype('int')
    cy = np.where((y >= 0) & (y < ys))
    z = np.linspace(section_position[2] - dz, section_position[2] + dz, 2 * dz + 1).astype('int')
    cz = np.where((z >= 0) & (z < zs))

    xi, yi, zi = np.meshgrid(x[cx], y[cy], z[cz], indexing='ij')  # coordinates in the rotated image
    xc, yc, zc = np.meshgrid(cx, cy, cz, indexing='ij')

    # it is the image being normalized that is rotated, to align with the template
    # later, the template section will be rotated and shifted to match the image
    # - need to keep these angles and rotations consistent
    for nn, angle in enumerate(angle_list):
        imgR = i3d.rotate3D(img, angle, fixedpoint, 0)

        for mm, angley in enumerate(angley_list):
            imgRR = i3d.rotate3D(imgR, angley, fixedpoint, 1)
            R = np.corrcoef(temp[xc,yc,zc].flatten(), imgRR[xi,yi,zi].flatten())   # check on the form out output from np.corrcoef
            mlist[nn, mm] = R[0,1]
            cclist[nn, mm, 0:3] = section_position  # this actually doesn't change

    w1 = angle_weight[:,np.newaxis]
    w2 = angley_weight[np.newaxis, :]
    angle_weight_extend = np.dot(w1,w2)

    mtemp = mlist*angle_weight_extend
    maxc = np.max(mtemp)
    nc,mc = np.where(mtemp == maxc)
    if len(nc) > 1:
        nc = nc[0]
        mc = mc[0]
    # bestposR = np.squeeze(cclist[nc, mc, 0:3])
    bestposR = section_position  # the position in the rotated image is the same for every angle, it rotates around in the unrotated image
    angle = angle_list[nc]
    angley = angley_list[mc]

    # print('angle = ', angle)
    # print('angley = ', angley)
    # print('angle_list = ', angle_list)
    # print('angley_list = ', angley_list)
    # print('mtemp = ', mtemp)

    # rotate the positions back to the unrotated image
    # the angles must be negative to get the positions in the unrotated image
    Mx = rotation_matrix(-angle, axis=0)
    My = rotation_matrix(-angley, axis=1)
    Mtotal = np.dot(My, Mx)
    bestpos = np.dot((bestposR - fixedpoint), Mtotal) + fixedpoint   # rotate the coordinates back around fixedpoint
    coords = bestpos   # coordinates of the section center in the non-rotated image
    coordsR = bestposR # coordinates of the section center in the rotated image

    print('best angle = ',angle,' angley = ', angley)
    print('    angle prediction = ',angle_estimate)
    print('best position:  ',coords)

    if (coords[2] < 10) or (coords[2] > zs-10):
        map_success = False

    # get the coordinates for the template section and corresponding image location that it maps to
    # get the coordinates for the image, where the template section maps to
    # first, get the coordinates in the rotated template

    # unlike the brainstem sections, the cord sections are never at an angle, the sections are aligned with the principal axes
    # Xt etc are coordinates in the template
    Xt, Yt, Zt = np.mgrid[(xpos-dx):(xpos+dx):(2*dx+1)*1j, (ypos-dy):(ypos+dy):(2*dy+1)*1j,(zpos-dz):(zpos+dz):(2*dz+1)*1j]
    # X etc are image coordinates in the rotated image, which was matched to the fixed template
    X, Y, Z = np.mgrid[(coordsR[0] - dx):(coordsR[0] + dx):(2*dx+1)*1j,
              (coordsR[1] - dy):(coordsR[1] + dy):(2*dy+1)*1j,
              (coordsR[2] - dz):(coordsR[2] + dz):(2*dz+1)*1j]

    # now the coordinates need to rotated to the original, unrotated, image
    # the image was rotated to match the template, so the coordinates for that section must also
    # be rotated in the same direction to get their values
    #  - important -- these are the coordinates that are being rotated now, not the image itself
    sa = (np.pi / 180) * angle
    Xr = X
    Yr = (Y - fixedpoint[1])*math.cos(sa) - (Z - fixedpoint[2])*math.sin(sa) + fixedpoint[1]
    Zr = (Z - fixedpoint[2])*math.cos(sa) + (Y - fixedpoint[1])*math.sin(sa) + fixedpoint[2]

    sa = (np.pi / 180) * angley
    Xrr = (Xr - fixedpoint[0]) * math.cos(sa) - (Zr - fixedpoint[2]) * math.sin(sa) + fixedpoint[0]
    Yrr = Yr
    Zrr = (Zr - fixedpoint[2]) * math.cos(sa) + (Xr - fixedpoint[0]) * math.sin(sa) + fixedpoint[2]

    Xr = Xrr
    Yr = Yrr
    Zr = Zrr

    # check on rotated coordinates
    coords_check = np.array([Xr[dx,dy,dz], Yr[dx,dy,dz], Zr[dx,dy,dz]])
    print('check on best position rotation:  best position = ', coords_check)

    # make sure coordinates are within limits
    # use these limits because values will be rounded when used as coordinates
    Xr = np.where(Xr <= -0.5, 0, Xr)
    Xr = np.where(Xr >= xs-0.5, xs - 1, Xr)
    Yr = np.where(Yr <= -0.5, 0, Yr)
    Yr = np.where(Yr >= ys-0.5, ys - 1, Yr)
    Zr = np.where(Zr <= -0.5, 0, Zr)
    Zr = np.where(Zr >= zs-0.5, zs - 1, Zr)

    template_section_check = template[np.round(Xt).astype('int'), np.round(Yt).astype('int'), np.round(Zt).astype('int')]  # check on this, see if it converted properly from matlab
    original_section = img[np.round(Xr).astype('int'), np.round(Yr).astype('int'), np.round(Zr).astype('int')]

    # organize the outputs
    section_mapping_coords = {'X': Xr, 'Y': Yr, 'Z': Zr, 'Xt': Xt, 'Yt': Yt, 'Zt': Zt}

    return coords, angle, angley, section_mapping_coords, original_section, template_section, map_success


# -------------py_predict_positions-------------------------------------------------
# ------------------------------------------------------------------------------------
def py_predict_positions(section_defs, result):
    dtor = np.pi/180   # degrees to radians
    ns = np.size(result)
    predicted = np.zeros((ns,ns,3))
    calculated = np.zeros((ns,ns,3))
    for aa in range(ns):
        for bb in np.setdiff1d(range(ns), aa):
            pas = section_defs[aa]['center']  # section used as a reference
            pa = result[aa]['coords']  # assume this result is correct, and predict other section positions
            pbs = section_defs[bb]['center']
            vab = pbs - pas  # vector from a to b in the template
            angle = result[aa]['angle']
            M = rotation_matrix(angle,0)
            vabr = np.dot(vab,M)  #  rotated vector
            pos_predicted = pa + vabr  # predicted position of b, based on mapped section a
            predicted[aa, bb, 0:3] = pos_predicted
            calculated[aa, bb, 0:3] = result[bb]['coords']  # actual calculated position of b

    return predicted, calculated



# -------------py_display_sections_local-------------------------------------------------
# ------------------------------------------------------------------------------------
def py_display_sections_local(template, background, warpdata):
    # create views showing the sections overlying the original image
    results_img = background/np.max(background)
    template = template/np.max(template)
    xs, ys, zs = np.shape(results_img)
    xt, yt, zt = np.shape(template)
    coords_list = np.zeros((np.size(warpdata),3))
    for nn in range(np.size(warpdata)):
        X = np.round(warpdata[nn]['X']).astype('int')
        Y = np.round(warpdata[nn]['Y']).astype('int')
        Z = np.round(warpdata[nn]['Z']).astype('int')
        Xt = np.round(warpdata[nn]['Xt']).astype('int')
        Yt = np.round(warpdata[nn]['Yt']).astype('int')
        Zt = np.round(warpdata[nn]['Zt']).astype('int')

        check = np.where( (Xt>= 0) & (Yt>= 0) & (Zt>= 0) & (X>= 0) & (Y>= 0) & (Z>= 0) & \
                (Xt < xt) & (Yt < yt) & (Zt < zt) & (X< xs) & (Y < ys) & (Z < zs) )

        Tx = Xt[check]
        Ty = Yt[check]
        Tz = Zt[check]

        Bx = X[check]
        By = Y[check]
        Bz = Z[check]

        # display the template section over the image section that it maps to
        coords_list[nn, 0] = np.mean(Bx)
        coords_list[nn, 1] = np.mean(By)
        coords_list[nn, 2] = np.mean(Bz)
        results_img[Bx,By,Bz]= template[Tx,Ty,Tz]

    return results_img


#-------------py_combine_warp_fields-------------------------------------------------
#------------------------------------------------------------------------------------
def py_combine_warp_fields(warpdata, background2, template, fit_order = [3,3,3]):
    # return  T, reverse_map_image, forward_map_image, inv_Rcheck

    # need to come back to this regularization method and investigate other methods
    # it may be possible to get a better rough mapping, and mapping of the template back
    # the original data
    #

    if np.size(fit_order) < 3:
        fit_order = fit_order[0]*np.ones(3)

    # extract complete lists of corresponding coordinates
    for nn in range(np.size(warpdata)):
        X = warpdata[nn]['X']
        Y = warpdata[nn]['Y']
        Z = warpdata[nn]['Z']
        Xt = warpdata[nn]['Xt']
        Yt = warpdata[nn]['Yt']
        Zt = warpdata[nn]['Zt']
        if nn == 0:
            xL = X.flatten()
            yL = Y.flatten()
            zL = Z.flatten()
            xtL = Xt.flatten()
            ytL = Yt.flatten()
            ztL = Zt.flatten()
        else:
            xL = np.hstack((xL, X.flatten()))    # concatenates along axis = 0
            yL = np.hstack((yL, Y.flatten()))
            zL = np.hstack((zL, Z.flatten()))
            xtL = np.hstack((xtL, Xt.flatten()))
            ytL = np.hstack((ytL, Yt.flatten()))
            ztL = np.hstack((ztL, Zt.flatten()))

    D = {'xL':xL, 'yL':yL, 'zL':zL, 'xtL':xtL, 'ytL':ytL, 'ztL':ztL}

    c = np.where(np.isfinite(xL) & np.isfinite(yL) & np.isfinite(zL) & np.isfinite(xtL) & np.isfinite(ytL) & np.isfinite(ztL))
    xL = xL[c];  yL = yL[c];  zL = zL[c];
    xtL = xtL[c];  ytL = ytL[c];  ztL = ztL[c];

    #
    # fill in the missing values of Xt, Yt, Zt
    # extrapolate
    xs,ys,zs = np.shape(background2)
    Xs,Ys,Zs = np.mgrid[range(xs), range(ys), range(zs)]
    xt,yt,zt = np.shape(template)
    Xt,Yt,Zt = np.mgrid[range(xt), range(yt), range(zt)]
    x,y,z = np.mgrid[range(xs), range(ys), range(zs)]

    xt0 = np.mean(Xt)
    yt0 = np.mean(Yt)
    zt0 = np.mean(Zt)
    xs0 = np.mean(Xs)
    ys0 = np.mean(Ys)
    zs0 = np.mean(Zs)
    order = fit_order

    inv_Rcheck = np.zeros(6)  # save the tests for matrix inversion problems
    # new method - ---------------------
    # do a polynomial fit to all of the data coordinates, to make a smooth mapping function
    #
    for nn in range(3):
        if nn == 0:  L = xL
        if nn == 1:  L = yL
        if nn == 2:  L = zL

        G2 = np.ones(np.size(Xt))
        G = np.ones(np.size(xtL))
        for ord in range(1, order[nn] + 1, 1):
            G2 = np.vstack((G2, (Xt.flatten() - xt0) ** ord, (Yt.flatten() - yt0) ** ord, (Zt.flatten() - zt0) ** ord))
            G = np.vstack((G, (xtL.flatten() - xt0) ** ord, (ytL.flatten() - yt0) ** ord, (ztL.flatten() - zt0) ** ord))

        # put the matrices the right way around
        G2 = G2.T  # size is now nvox x (1 + 3*order)
        G = G.T

        # fit = G2 x m  (matrix mult.)
        # original = G x m
        # m = inv(G' x G) x G' x original

        # inv_Rcheck[nn] = np.linalg.cond(np.dot(G.T,G))  # not sure if this is necessary
        # iGG = np.linalg.inv(np.dot(G.T,G))
        # m = np.dot(np.dot(iGG,G.T), L)
        # t_fit = np.dot(G2,m)

        inv_Rcheck[nn] = np.linalg.cond(G.T @ G)  # not sure if this is necessary
        iGG = np.linalg.inv(G.T @ G)
        m = (iGG @ G.T) @ L
        t_fit = G2 @ m
        t_fit = np.reshape(t_fit, [xt, yt, zt])

        if nn == 0: X_fit = t_fit
        if nn == 1: Y_fit = t_fit
        if nn == 2: Z_fit = t_fit

    # finished mapping for each axis

    # go back to the starting values
    xL = D['xL'];  yL = D['yL']; zL = D['zL']; xtL =  D['xtL']; ytL = D['ytL'];  ztL = D['ztL']

    c = np.where(np.isfinite(xL) & np.isfinite(yL) & np.isfinite(zL) & np.isfinite(xtL) & np.isfinite(ytL) & np.isfinite(ztL))
    xL = xL[c];  yL = yL[c];  zL = zL[c];
    xtL = xtL[c];  ytL = ytL[c];  ztL = ztL[c];

    #
    # fill in the missing values of X, Y, Z
    # extrapolate
    x,y,z = np.mgrid[range(xs), range(ys), range(zs)]

    for nn in range(3):
        if nn == 0:  tL = xtL
        if nn == 1:  tL = ytL
        if nn == 2:  tL = ztL

        G2 = np.ones(np.size(Xs))
        G = np.ones(np.size(xL))
        for ord in range(1, order[nn] + 1, 1):
            G2 = np.vstack((G2, (Xs.flatten() - xs0) ** ord, (Ys.flatten() - ys0) ** ord, (Zs.flatten() - zs0) ** ord))
            G = np.vstack((G, (xL.flatten() - xs0) ** ord, (yL.flatten() - ys0) ** ord, (zL.flatten() - zs0) ** ord))

        G = G.T
        G2 = G2.T

        # inv_Rcheck[nn+3] = np.linalg.cond(np.dot(G.T,G))   # dont know if this is necessary
        # iGG = np.linalg.inv(np.dot(G.T,G))
        # m = np.dot(np.dot(iGG,G.T), tL)
        # t_fit = np.dot(G2,m)

        inv_Rcheck[nn + 3] = np.linalg.cond(G.T @ G)  # dont know if this is necessary
        iGG = np.linalg.inv(G.T @ G)
        m = (iGG @ G.T) @ tL
        t_fit = G2 @ m

        t_fit = np.reshape(t_fit, [xs, ys, zs])

        if nn == 0: Xt_fit = t_fit
        if nn == 1: Yt_fit = t_fit
        if nn == 2: Zt_fit = t_fit

    # done the smooth mapping for the template coordinates

    # forward and reverse transform information
    T = {'Xs':X_fit, 'Ys':Y_fit, 'Zs':Z_fit, 'Xt':Xt_fit, 'Yt':Yt_fit, 'Zt':Zt_fit}

    # reverse_map_image is the mapping of the image data into the template space
    # "reverse" mapping because we had the coordinates of where template voxels mapped into
    # the original image data, and we are calculating where the original image data belongs in the
    # template space
    reverse_map_image = i3d.warp_image(background2, X_fit, Y_fit, Z_fit)
    # forward_map_image is the mapping of the template into the original image space
    forward_map_image = i3d.warp_image(template, Xt_fit, Yt_fit, Zt_fit)

    return  T, reverse_map_image, forward_map_image, inv_Rcheck


# -------------py_combine_warp_fields-------------------------------------------------
# ------------------------------------------------------------------------------------
# def py_combine_warp_fields_with_dampening(warpdata, background2, template, fit_order=[3, 3, 3]):
#     # return  T, reverse_map_image, forward_map_image, inv_Rcheck
#
#     # include a first-order fit so the voxels distant from mapping points are not
#     # extremely shifted due to extrapolation problems
#     #
#     xs,ys,zs = np.shape(background2)
#     xt,yt,zt = np.shape(template)
#
#     nwarp = np.size(warpdata)
#     coords_list_s = np.zeros((nwarp,3))
#     coords_list_t = np.zeros((nwarp,3))
#
#     if np.size(fit_order) < 3:
#         fit_order = fit_order[0] * np.ones(3)
#
#     # extract complete lists of corresponding coordinates
#     for nn in range(nwarp):
#         X = warpdata[nn]['X']
#         Y = warpdata[nn]['Y']
#         Z = warpdata[nn]['Z']
#         Xt = warpdata[nn]['Xt']
#         Yt = warpdata[nn]['Yt']
#         Zt = warpdata[nn]['Zt']
#         coords_list_s[nn,:] = np.array([np.mean(X), np.mean(Y), np.mean(Z)])
#         coords_list_t[nn,:] = np.array([np.mean(Xt), np.mean(Yt), np.mean(Zt)])
#         if nn == 0:
#             xL = X.flatten()
#             yL = Y.flatten()
#             zL = Z.flatten()
#             xtL = Xt.flatten()
#             ytL = Yt.flatten()
#             ztL = Zt.flatten()
#         else:
#             xL = np.hstack((xL, X.flatten()))  # concatenates along axis = 0
#             yL = np.hstack((yL, Y.flatten()))
#             zL = np.hstack((zL, Z.flatten()))
#             xtL = np.hstack((xtL, Xt.flatten()))
#             ytL = np.hstack((ytL, Yt.flatten()))
#             ztL = np.hstack((ztL, Zt.flatten()))
#
#     D = {'xL': xL, 'yL': yL, 'zL': zL, 'xtL': xtL, 'ytL': ytL, 'ztL': ztL}
#
#     c = np.where(np.isfinite(xL) & np.isfinite(yL) & np.isfinite(zL) & np.isfinite(xtL)
#                  & np.isfinite(ytL) & np.isfinite(ztL))
#     xL = xL[c];
#     yL = yL[c];
#     zL = zL[c];
#     xtL = xtL[c];
#     ytL = ytL[c];
#     ztL = ztL[c];
#
#     #
#     # fill in the missing values of Xt, Yt, Zt
#     # extrapolate
#     Xs, Ys, Zs = np.mgrid[range(xs), range(ys), range(zs)]
#     Xt, Yt, Zt = np.mgrid[range(xt), range(yt), range(zt)]
#     x, y, z = np.mgrid[range(xs), range(ys), range(zs)]
#
#     xt0 = np.mean(Xt)
#     yt0 = np.mean(Yt)
#     zt0 = np.mean(Zt)
#     xs0 = np.mean(Xs)
#     ys0 = np.mean(Ys)
#     zs0 = np.mean(Zs)
#     order = fit_order
#
#     # new method - ---------------------
#     # do a polynomial fit to all of the data coordinates, to make a smooth mapping function
#     #
#     for nn in range(3):
#         if nn == 0:  L = xL
#         if nn == 1:  L = yL
#         if nn == 2:  L = zL
#
#         G2 = np.ones(np.size(Xt))
#         G = np.ones(np.size(xtL))
#         for ord in range(1, order[nn] + 1, 1):
#             G2 = np.vstack((G2, (Xt.flatten() - xt0) ** ord, (Yt.flatten() - yt0) ** ord, (Zt.flatten() - zt0) ** ord))
#             G = np.vstack((G, (xtL.flatten() - xt0) ** ord, (ytL.flatten() - yt0) ** ord, (ztL.flatten() - zt0) ** ord))
#
#         # put the matrices the right way around
#         G2 = G2.T  # size is now (1 + 3*order) x nvox
#         G = G.T
#
#         # fit = G2 x m  (matrix mult.)
#         # original = G x m
#         # m = inv(G' x G) x G' x original
#
#         # inv_Rcheck[nn] = np.linalg.cond(np.dot(G.T,G))  # not sure if this is necessary
#         # iGG = np.linalg.inv(np.dot(G.T,G))
#         # m = np.dot(np.dot(iGG,G.T), L)
#         # t_fit = np.dot(G2,m)
#
#         iGG = np.linalg.inv(G.T @ G)
#         m = (iGG @ G.T) @ L
#         t_fit = G2 @ m
#         t_fit = np.reshape(t_fit, [xt, yt, zt])
#
#
#         # for limiting the exrapolated values - use only constant and first-order temrms
#         iGGdamp = np.linalg.inv(G[:,:4].T @ G[:,:4])
#         mdamp = (iGGdamp @ G[:,:4].T) @ L
#         t_fitdamp = G2[:,:4] @ mdamp
#         t_fitdamp = np.reshape(t_fitdamp, [xt, yt, zt])
#
#         if nn == 0:
#             X_fit = t_fit
#             X_fitdamp = t_fitdamp
#         if nn == 1:
#             Y_fit = t_fit
#             Y_fitdamp = t_fitdamp
#         if nn == 2:
#             Z_fit = t_fit
#             Z_fitdamp = t_fitdamp
#
#     # combine the dampening fields based on distance from control points
#     dist = np.zeros((xt,yt,zt,nwarp))
#     for nn in range(len(coords_list_t)):
#         p = coords_list_t[nn,:]
#         dist[:,:,:,nn] = (Xt-p[0])**2 + (Yt-p[1])**2 + (Zt-p[2])**2
#     distfactor = np.min(dist,axis=3)
#
#     maxdist = np.max(distfactor)
#     distweight = np.cos(np.tanh((distfactor/maxdist-0.5)*10.))  # weighting is 0 at control points, and 1 at most distance point
#     X_fit = (1-distweight)*X_fit + distweight*X_fitdamp
#     Y_fit = (1-distweight)*Y_fit + distweight*Y_fitdamp
#     Z_fit = (1-distweight)*Z_fit + distweight*Z_fitdamp
#
#     # now do the reverse mapping-----------------------------------------------------
#     # go back to the starting values
#     xL = D['xL'];
#     yL = D['yL'];
#     zL = D['zL'];
#     xtL = D['xtL'];
#     ytL = D['ytL'];
#     ztL = D['ztL']
#
#     c = np.where(
#         np.isfinite(xL) & np.isfinite(yL) & np.isfinite(zL) & np.isfinite(xtL) & np.isfinite(ytL) & np.isfinite(ztL))
#     xL = xL[c];
#     yL = yL[c];
#     zL = zL[c];
#     xtL = xtL[c];
#     ytL = ytL[c];
#     ztL = ztL[c];
#
#     #
#     # fill in the missing values of X, Y, Z
#     # extrapolate
#     x, y, z = np.mgrid[range(xs), range(ys), range(zs)]
#
#     for nn in range(3):
#         if nn == 0:
#             tL = xtL
#         if nn == 1:
#             tL = ytL
#         if nn == 2:
#             tL = ztL
#
#         G2 = np.ones(np.size(Xs))
#         G = np.ones(np.size(xL))
#         for ord in range(1, order[nn] + 1, 1):
#             G2 = np.vstack((G2, (Xs.flatten() - xs0) ** ord, (Ys.flatten() - ys0) ** ord, (Zs.flatten() - zs0) ** ord))
#             G = np.vstack((G, (xL.flatten() - xs0) ** ord, (yL.flatten() - ys0) ** ord, (zL.flatten() - zs0) ** ord))
#
#         G = G.T
#         G2 = G2.T
#
#         # inv_Rcheck[nn+3] = np.linalg.cond(np.dot(G.T,G))   # dont know if this is necessary
#         # iGG = np.linalg.inv(np.dot(G.T,G))
#         # m = np.dot(np.dot(iGG,G.T), tL)
#         # t_fit = np.dot(G2,m)
#
#         iGG = np.linalg.inv(G.T @ G)
#         m = (iGG @ G.T) @ tL
#         t_fit = G2 @ m
#         t_fit = np.reshape(t_fit, [xs, ys, zs])
#
#         # for limiting the exrapolated values - use only constant and first-order temrms
#         iGGdamp = np.linalg.inv(G[:,:4].T @ G[:,:4])
#         mdamp = (iGGdamp @ G[:,:4].T) @ tL
#         t_fitdamp = G2[:,:4] @ mdamp
#         t_fitdamp = np.reshape(t_fitdamp, [xs, ys, zs])
#
#         if nn == 0:
#             Xt_fit = t_fit
#             Xt_fitdamp = t_fitdamp
#         if nn == 1:
#             Yt_fit = t_fit
#             Yt_fitdamp = t_fitdamp
#         if nn == 2:
#             Zt_fit = t_fit
#             Zt_fitdamp = t_fitdamp
#
#     # done the smooth mapping for the template coordinates
#
#     # forward and reverse transform information
#     T = {'Xs': X_fit, 'Ys': Y_fit, 'Zs': Z_fit, 'Xt': Xt_fit, 'Yt': Yt_fit, 'Zt': Zt_fit}
#
#     # reverse_map_image is the mapping of the image data into the template space
#     # "reverse" mapping because we had the coordinates of where template voxels mapped into
#     # the original image data, and we are calculating where the original image data belongs in the
#     # template space
#     reverse_map_image = i3d.warp_image(background2, X_fit, Y_fit, Z_fit)
#     # forward_map_image is the mapping of the template into the original image space
#     forward_map_image = i3d.warp_image(template, Xt_fit, Yt_fit, Zt_fit)
#
#     return T, reverse_map_image, forward_map_image




# -------------py_combine_warp_fields_piecewise_affine-------------------------------------------------
# ------------------------------------------------------------------------------------
# def py_combine_warp_fields_piecewise_affine(template_coords, image_coords, angle_record, background2, template):
#     # return  T, reverse_map_image, forward_map_image, inv_Rcheck
#
#     # include a first-order fit so the voxels distant from mapping points are not
#     # extremely shifted due to extrapolation problems
#     #
#     xs,ys,zs = np.shape(background2)
#     xt,yt,zt = np.shape(template)
#
#     nsections = len(angle_record)
#
#     X, Y, Z = np.mgrid[range(xs), range(ys), range(zs)]  # pixel coordinates in image
#     Xt, Yt, Zt = np.mgrid[range(xt), range(yt), range(zt)]  # pixel coordinates in template
#     # xo, yo, zo = np.mgrid[0:xs:newsize[0] * 1j, 0:ys:newsize[1] * 1j, 0:zs:newsize[2] * 1j]
#
#     ptmap = np.zeros((xt,yt,zt,3))
#     pmap = np.zeros((xt,yt,zt,3))
#
#     # weight the translations and rotations based on the distance from the sections
#     zdist = np.zeros((xt,yt,zt,nsections))
#     zmax = np.max(template_coords[:,2])
#     czmax = np.argmax(template_coords[:,2])
#     axmx, aymx, azmx = np.where(Zt >= zmax)
#     zmin = np.min(template_coords[:,2])
#     czmin = np.argmin(template_coords[:,2])
#     axmn, aymn, azmn = np.where(Zt <= zmin)
#     for ss in range(nsections):
#         zdist[:,:,:,ss] = np.abs(Zt - template_coords[ss,2])
#         if ss == czmax:
#             zdist[axmx,aymx,azmx,ss] = 1.0  # make sure the weighting is strongest for the closest section
#         # else:
#         #     zdist[axmx,aymx,azmx,ss] = 1e6
#         if ss == czmin:
#             zdist[axmn,aymn,azmn,ss] = 1.0  # make sure the weighting is strongest for the closest section
#         # else:
#         #     zdist[axmn,aymn,azmn,ss] = 1e6
#
#     inv_dist = 1.0/(zdist**2+1.0e-6)
#     inv_dist_sum = np.sum(1.0/(zdist**2+1.0e-6), axis = 3)
#     zweight = inv_dist/np.repeat(inv_dist_sum[:,:,:,np.newaxis],nsections,axis=3)
#
#     coords_map = np.zeros((xt,yt,zt,3))
#     tcoords_map = np.zeros((xt,yt,zt,3))
#     angle_map = np.zeros((xt,yt,zt))
#
#     angle_extend = np.tile(angle_record,(xt,yt,zt,1))
#     angle_map = np.sum(zweight*angle_extend,axis=3)  #  [xt,yt,zt,nsections] x [nsections]
#     for nn in range(3):
#         c = template_coords[:,nn]
#         c_extend = np.tile(c,(xt,yt,zt,1))
#         tcoords_map[:,:,:,nn] = np.sum(zweight*c_extend,axis=3)
#
#         c = image_coords[:,nn]
#         c_extend = np.tile(c,(xt,yt,zt,1))
#         coords_map[:,:,:,nn] = np.sum(zweight*c_extend,axis=3)
#
#     # coordinates in image map to Xtm, Ytm, Ztm in template
#     aa = -angle_map*np.pi/180.
#     Xtm = (X-coords_map[:,:,:,0]) + tcoords_map[:,:,:,0]    # x translation
#     Ytm = (Y-coords_map[:,:,:,1])*np.cos(aa) - (Z-coords_map[:,:,:,2])*np.sin(aa) + tcoords_map[:,:,:,1]    # rotation around x, and translation
#     Ztm = (Y-coords_map[:,:,:,1])*np.sin(aa) + (Z-coords_map[:,:,:,2])*np.cos(aa) + tcoords_map[:,:,:,2]    # rotation around x, and translation
#
#     # coordinates in template map to Xm, Ym, Zm in image
#     aa = angle_map*np.pi/180.
#     Xm = (Xt-tcoords_map[:,:,:,0]) + coords_map[:,:,:,0]    # x translation
#     Ym = (Yt-tcoords_map[:,:,:,1])*np.cos(aa) - (Zt-tcoords_map[:,:,:,2])*np.sin(aa) + coords_map[:,:,:,1]    # rotation around x, and translation
#     Zm = (Yt-tcoords_map[:,:,:,1])*np.sin(aa) + (Zt-tcoords_map[:,:,:,2])*np.cos(aa) + coords_map[:,:,:,2]    # rotation around x, and translation
#
#     # done the smooth mapping for the template coordinates
#     # forward and reverse transform information
#     T = {'Xs': Xm, 'Ys': Ym, 'Zs': Zm, 'Xt': Xtm, 'Yt': Ytm, 'Zt': Ztm}
#
#     # reverse_map_image is the mapping of the image data into the template space
#     # "reverse" mapping because we had the coordinates of where template voxels mapped into
#     # the original image data, and we are calculating where the original image data belongs in the
#     # template space
#     reverse_map_image = i3d.warp_image(background2, Xm, Ym, Zm)
#     # forward_map_image is the mapping of the template into the original image space
#     forward_map_image = i3d.warp_image(template, Xtm, Ytm, Ztm)
#
#     return T, reverse_map_image, forward_map_image


#-------------py_apply_normalization----------------------------------------
#---------------------------------------------------------------------------
def py_apply_normalization(input_image, T, Tfine = 'none', map_to_normalized_space = True):
    # T = {'Xs':X_fit, 'Ys':Y_fit, 'Zs':Z_fit, 'Xt':Xt_fit, 'Yt':Yt_fit, 'Zt':Zt_fit}
    # Tfine = {'dXs': dX, 'dYs': dY, 'dZs': dZ, 'dXt': dXt, 'dYt': dYt, 'dZt': dZt}

    if map_to_normalized_space:
        # reverse_map_image is the mapping of the image data into the template space
        # "reverse" mapping because we had the coordinates of where template voxels mapped into
        # the original image data, and we are calculating where the original image data belongs in the
        # template space
        if Tfine == 'none':
            norm_image = i3d.warp_image(input_image, T['Xs'], T['Ys'], T['Zs'])
        else:
            norm_image = i3d.warp_image(input_image, T['Xs']+Tfine['dXs'], T['Ys']+Tfine['dYs'], T['Zs']+Tfine['dZs'])
    else:
        # forward mapping is the mapping of a normalized template, or image, into the original image space
        if Tfine == 'none':
            norm_image = i3d.warp_image(input_image, T['Xt'], T['Yt'], T['Zt'])
        else:
            norm_image = i3d.warp_image(input_image, T['Xt']+Tfine['dXt'], T['Yt']+Tfine['dYt'], T['Zt']+Tfine['dZt'])

    return norm_image



#------------apply normalization to the template (reverse mapping) for a complete 4D nifti file----------
#--------------------------------------------------------------------------------------------------------
def apply_normalization_to_nifti(niiname, T, Tfine = 'None', input_affine = [], normprefix = 'p'):
    input_image, affine = i3d.load_and_scale_nifti(niiname)  # this takes care of resizing the data to 1 mm voxels

    ts = np.shape(input_image)[3]   # get the size of the time dimension
    xs,ys,zs = np.shape(T['Xs'])   # get the dimensions of the resulting normalized images

    output_images = np.zeros((xs,ys,zs,ts))   # initialize the output array

    for tt in range(ts):
        output_images[:,:,:,tt] = py_apply_normalization(input_image[:, :, :, tt], T, Tfine)

    # write result as new NIfTI format image set
    pname, fname = os.path.split(niiname)
    output_niiname = os.path.join(pname, normprefix + fname)
    if len(input_affine) == 0:
        affine = np.array([[-1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    else:
        affine = input_affine
    resulting_img = nib.Nifti1Image(output_images, affine)
    nib.save(resulting_img, output_niiname)

    return output_niiname



#------py_estimate_reverse_transform------------------------------
#----------------------------------------------------------------
def py_estimate_reverse_transform(X,Y,Z, input_size, fit_order=3):
    # return Xt, Yt, Zt

    # Xt, Yt, Zt = py_estimate_reverse_transform(T['Xs'] + dX, T['Ys'] + dY, T['Zs'] + dZ, np.shape(input_image),
    #                                            # fit_order=3)
    # X, Y, Z show where very point in the input maps to in the output
    # they contain input coordinates, at each point in the output space
    xts, yts, zts = np.shape(X)
    xs, ys, zs = input_size[:3]

    x,y,z = np.mgrid[range(xs), range(ys), range(zs)]
    xt,yt,zt = np.mgrid[range(xts), range(yts), range(zts)]

    # xt = mx*B (matrix mult) where B consists of polynomial values of all of the coordinates in X,Y,Z
    # mx = xt*B'*inv(B*B')

    XL = X.flatten()
    YL = Y.flatten()
    ZL = Z.flatten()
    xtL = xt.flatten()
    ytL = yt.flatten()
    ztL = zt.flatten()

    order = fit_order
    B = np.ones(np.size(X))
    for ord in range(order):
        B = np.vstack((B, XL**(ord+1), YL**(ord+1), ZL**(ord+1)))

    iBB = np.linalg.inv(np.dot(B,B.T))
    iBBB = np.dot(B.T,iBB)
    mx = np.dot(xtL,iBBB)   #  [1 x N]*[N x 10]*inv([10 x N]*[N x 10]) == [1 x 10]
    my = np.dot(ytL,iBBB)
    mz = np.dot(ztL,iBBB)

    xL = x.flatten()
    yL = y.flatten()
    zL = z.flatten()

    # now calculate the output location for every point in the input space
    G = np.ones(np.size(x))
    for ord in range(order):
        G = np.vstack((G, xL**(ord+1), yL**(ord+1), zL**(ord+1)))

    xt = np.dot(mx,G)  # [1 x 10]*[10 x N2]
    yt = np.dot(my,G)
    zt = np.dot(mz,G)

    Xt = np.reshape(xt, [xs, ys, zs])
    Yt = np.reshape(yt, [xs, ys, zs])
    Zt = np.reshape(zt, [xs, ys, zs])

    return Xt, Yt, Zt


# -------------py_auto_cord_normalize-------------------------------------------------
# ------------------------------------------------------------------------------------
def py_auto_cord_normalize(background2, template, fit_parameters_input, section_defs, ninitial_fixed_segments, reverse_order, display_output=True):   # , display_window = 'None',display_image1 = 'none', display_window2 = 'None', display_image2 = 'none'
    # return T, warpdata
    print('running py_auto_cord_normalize ...')
    displayrecord = []
    imagerecord = []

    if np.size(fit_parameters_input) < 1:
        fit_parameters_input = [10, 50, 5, 6, -10, 10, -10, 20]   # default values

    fit_parameters = fit_parameters_input*np.array([1e-4, 1e-4, 1e-4, 1e-4, 1, 1, 1, 1])   # actual parameters are scaled from more intuitive values

    # global zpos2_record
    xs2, ys2, zs2 = np.shape(background2)
    xs, ys, zs = np.shape(template)

    # 1) initial sections
    # initialize result and warpdata
    result_entry = {'angle':0, 'angley':0, 'coords':[0,0,0], 'original_section':[], 'template_section':[], 'section_mapping_coords':[]}
    warpdata_entry = {'X': 0, 'Y': 0, 'Z': 0, 'Xt': 0, 'Yt': 0, 'Zt': 0}
    result = []
    warpdata = []

    for ss in range(np.size(section_defs)):
        result.append(result_entry.copy())
        warpdata.append(warpdata_entry.copy())

    # step through the initial segments - the upper portions of the brainstem for cervical cord/BS data, or lumbar/sacral regions
    for ss in range(ninitial_fixed_segments):
        print('starting rough mapping for section ',ss, ' ...')
        position_stiffness = fit_parameters[0]
        angle_stiffness = fit_parameters[2]

        nsearchvalues = np.round((fit_parameters[5] - fit_parameters[4]) / 2 + 1).astype('int')
        if nsearchvalues > 11: nsearchvalues = 11
        angle_list = np.linspace(fit_parameters[4],fit_parameters[5],nsearchvalues)
        # angle_list = np.linspace(fit_parameters[4],fit_parameters[5],np.round((fit_parameters[5]-fit_parameters[4])/2 + 1).astype('int'))

        if (ss == 0) & (np.size(section_defs[0]['start_ref_pos']) > 0):
            pos_estimate = section_defs[0]['start_ref_pos']
            angle_estimate = section_defs[0]['start_angle']
            fix_position_flag = 1
        else:
            pos_estimate = section_defs[ss]['pos_estimate']
            angle_estimate =[]
            fix_position_flag = 0

        pos = section_defs[ss]['center']
        d = section_defs[ss]['dims']
        fixZ_flag = section_defs[ss]['fixdistance']
        error_correction_flag = 0
        coords, angle, angley, section_mapping_coords, original_section, template_section, window1_display = \
                py_calculate_section_positions(template, background2, section_defs[ss], angle_list, pos_estimate, angle_estimate, \
                            position_stiffness, angle_stiffness, fixZ_flag, fix_position_flag, error_correction_flag)

        print('completed rough mapping for section ',ss, ' ...')

        result[ss]['angle'] = angle
        result[ss]['angley'] = angley
        result[ss]['coords'] = coords
        result[ss]['original_section'] = original_section
        result[ss]['template_section'] = template_section
        result[ss]['section_mapping_coords'] = section_mapping_coords

        # save another copy, for convenience later
        warpdata[ss]['X'] = result[ss]['section_mapping_coords']['X']
        warpdata[ss]['Y'] = result[ss]['section_mapping_coords']['Y']
        warpdata[ss]['Z'] = result[ss]['section_mapping_coords']['Z']
        warpdata[ss]['Xt'] = result[ss]['section_mapping_coords']['Xt']
        warpdata[ss]['Yt'] = result[ss]['section_mapping_coords']['Yt']
        warpdata[ss]['Zt'] = result[ss]['section_mapping_coords']['Zt']

        print('finished compiling results for section ',ss, ' ...')

        # display the results, as the mapping progresses ....
        results_img = py_display_sections_local(template, background2, warpdata[:ss+1])
        x = np.round(xs2/2).astype('int')
        display_image = results_img[x, :, :]
        image_tk = ImageTk.PhotoImage(Image.fromarray(display_image))
        displayrecord.append(image_tk)
        imagerecord.append({'img':display_image})

    # 2) check results for initial sections
    print('Checking results for initial sections ...')
    if ninitial_fixed_segments > 2:   # if there are "initial" segments defined, and there are at least 3 of them
        # need a way to check on the result and fix bad mapping if it occurs
        # predict section positions based on other sections
        predicted, calculated = py_predict_positions(section_defs[:ninitial_fixed_segments], result[:ninitial_fixed_segments])
        # print('predicted positions: ', predicted)
        # print('calculated positions: ', calculated)

        err = np.abs(predicted-calculated)
        errdist = np.sqrt(np.sum(err**2, axis = 2))  # error in the position (distance) based on difference between predicted and calculated
        print('error (between predicted and calculated):  ', errdist)

        rn,cn = np.where(errdist > 18)
        number_of_error_positions = np.size(rn)
        if np.size(rn) > 2:
            count = np.zeros(3)
            for aa in range(3):
                count[aa] = np.size(np.where( (rn == aa) | (cn == aa)))
            commonfactor = np.array(np.where(count == np.size(rn))).flatten()   # find the section that is consistently out of position
        else:
            commonfactor = []

        if np.size(commonfactor) > 0:
            print('section that consistently appears to be in error is ',commonfactor[0])
            slist = np.setdiff1d(range(3), commonfactor)
            p1 = (np.squeeze(predicted[slist[0], commonfactor, 0:3]) + np.squeeze(predicted[slist[1], commonfactor, 0:3]) )/2
            print('predicted position of section ',commonfactor,'   is  ', np.round(p1))
            p = np.squeeze(calculated[slist[1], commonfactor, 0:3])
            print('calculated position of section ',commonfactor,'  is ',np.round(p))

            # recalculate error section, with constraints
            position_stiffness = fit_parameters[0]
            position_stiffness = 1
            angle_stiffness = fit_parameters[2]
            angle_estimate =[]
            fixZ_flag = section_defs[commonfactor[0]]['fixdistance']
            error_correction_flag = 1;
            coords, angle, angley, section_mapping_coords, original_section, template_section, window1_display = py_calculate_section_positions(template, background2, section_defs[commonfactor[0]], angle_list, p1, angle_estimate, position_stiffness, angle_stiffness, fixZ_flag, fix_position_flag, error_correction_flag)

            result[commonfactor[0]]['angle'] = angle
            result[commonfactor[0]]['angley'] = angley
            result[commonfactor[0]]['coords'] = coords
            result[commonfactor[0]]['original_section'] = original_section
            result[commonfactor[0]]['template_section'] = template_section
            result[commonfactor[0]]['section_mapping_coords'] = section_mapping_coords

            # save another copy, for convenience later
            warpdata[commonfactor[0]]['X'] = result[commonfactor[0]]['section_mapping_coords']['X']
            warpdata[commonfactor[0]]['Y'] = result[commonfactor[0]]['section_mapping_coords']['Y']
            warpdata[commonfactor[0]]['Z'] = result[commonfactor[0]]['section_mapping_coords']['Z']
            warpdata[commonfactor[0]]['Xt'] = result[commonfactor[0]]['section_mapping_coords']['Xt']
            warpdata[commonfactor[0]]['Yt'] = result[commonfactor[0]]['section_mapping_coords']['Yt']
            warpdata[commonfactor[0]]['Zt'] = result[commonfactor[0]]['section_mapping_coords']['Zt']
        else:
            if number_of_error_positions >= 6:
                print('too many sections were detected as being consistently in error to be able to fix them')
            else:
                print('no sections were detected as being consistently in error')

        # display the results, as the mapping progresses ....
        results_img = py_display_sections_local(template, background2, warpdata[:ninitial_fixed_segments])
        x = np.round(xs2/2).astype('int')
        display_image = results_img[x, :, :]
        image_tk = ImageTk.PhotoImage(Image.fromarray(display_image))

    # 3) map the remaining sections based on the initial sections
    # the later sections are guided by the first section

    # -----now, map the cord segments ---------------------------
    resultsplot = []
    dtor = np.pi/180   # convert degrees to radians
    ncordsegments = np.size(section_defs) - ninitial_fixed_segments
    for ss in range(ncordsegments):
        position_stiffness = fit_parameters[1]
        angle_stiffness = fit_parameters[3]
        angle_list = np.linspace(fit_parameters[6], fit_parameters[7], np.round((fit_parameters[7]-fit_parameters[6])/2 + 1).astype('int'))

        if ss == 0:   # the first defined section has to be the one just above the cord sections that are chained together
            #  --- important revision to test -- does the previous section_def angle need to be added for the prediction?
            angle = result[0]['angle'] + section_defs[0]['xrot']
            angley = result[0]['angley']
            coords = result[0]['coords']
            pos = section_defs[0]['center']
            first_region_connection_point = section_defs[0]['first_region_connection_point']
            # if reverse_order:
            #     vpos_connectionpoint = first_region_connection_point + pos
            # else:
            #     vpos_connectionpoint = first_region_connection_point - pos  # vector from the center of the section to the connection point
            # correction-----------
            vpos_connectionpoint = first_region_connection_point - pos

            # -- important revision --------  make angles negative to get correct rotation
            Mx = rotation_matrix(-angle, 0)
            My = rotation_matrix(-angley, 1)
            Mtotal = np.dot(Mx,My)
            # original-----------------
            # Mx = rotation_matrix(angle,0)
            # My = rotation_matrix(angley,1)
            # Mtotal = np.dot(Mx,My)
            # end of original---------------
            rvpos = np.dot(vpos_connectionpoint, Mtotal)   # rotated vector
            # if reverse_order:
            #     fixedpoint = coords - rvpos  # mapped location of fixedpoint in the image data
            # else:
            #     fixedpoint = coords + rvpos  # mapped location of fixedpoint in the image data
            fixedpoint = coords + rvpos  # mapped location of fixedpoint in the image data
        else:
            angle = result[ss+ninitial_fixed_segments-1]['angle']
            coords = result[ss+ninitial_fixed_segments-1]['coords']
            pos = section_defs[ss+ninitial_fixed_segments-1]['center']
            # if reverse_order:
            #     vpos_connectionpoint = section_defs[ss + ninitial_fixed_segments - 1]['fixedpoint2'] + pos
            # else:
            #     vpos_connectionpoint = section_defs[ss+ninitial_fixed_segments-1]['fixedpoint2'] - pos   # vector from the center of the section to the connection point
            # correction--------------------------
            vpos_connectionpoint = section_defs[ss + ninitial_fixed_segments - 1][
                                       'fixedpoint2'] - pos  # vector from the center of the section to the connection point

            Mx = rotation_matrix(-angle,0)
            My = rotation_matrix(-angley,1)
            Mtotal = np.dot(Mx,My)
            rvpos = np.dot(vpos_connectionpoint, Mtotal)   # rotated vector
            # if reverse_order:
            #     fixedpoint = coords - rvpos  # mapped location of fixedpoint in the image data
            # else:
            #     fixedpoint = coords + rvpos  # mapped location of fixedpoint in the image data
            fixedpoint = coords + rvpos  # mapped location of fixedpoint in the image data

        angle_estimate = angle
        # angle_stiffness = 1e-3

        print('cord section ',ss,'   fixed point: ', fixedpoint)   # fixed point is in the image coordinates
        coords, angle, angley, section_mapping_coords, original_section, template_section, map_success = \
            py_calculate_chainlink_positions(template, background2, section_defs[ss+ninitial_fixed_segments], \
                                             angle_list, fixedpoint, angle_estimate, angle_stiffness, reverse_order)
                                                                                                                        #  (template, img, section_defs, angle_list, fixedpoint, angle_estimate, angle_stiffness):
        # if sections cannot be defined because they run off the end of the volume,
        # then don't use them -------------------------------------
        if map_success:
            result[ss+ninitial_fixed_segments]['angle'] = angle
            result[ss+ninitial_fixed_segments]['angley'] = angley
            result[ss+ninitial_fixed_segments]['coords'] = coords
            result[ss+ninitial_fixed_segments]['original_section'] = original_section
            result[ss+ninitial_fixed_segments]['template_section'] = template_section
            result[ss+ninitial_fixed_segments]['section_mapping_coords'] = section_mapping_coords

            warpdata[ss+ninitial_fixed_segments]['X'] = section_mapping_coords['X']
            warpdata[ss+ninitial_fixed_segments]['Y'] = section_mapping_coords['Y']
            warpdata[ss+ninitial_fixed_segments]['Z'] = section_mapping_coords['Z']
            warpdata[ss+ninitial_fixed_segments]['Xt'] = section_mapping_coords['Xt']
            warpdata[ss+ninitial_fixed_segments]['Yt'] = section_mapping_coords['Yt']
            warpdata[ss+ninitial_fixed_segments]['Zt'] = section_mapping_coords['Zt']

            midline = np.zeros(ss+ninitial_fixed_segments)
            for aa in range(ss+ninitial_fixed_segments):
                midline[aa] = result[aa]['coords'][0]
            midline = np.round(np.mean(midline)).astype('int')

            results_img = py_display_sections_local(template, background2, warpdata[:(ss+ninitial_fixed_segments)])
            # display the result again
            # need to figure out how to control which figure is used for display  - come back to this ....
            display_image = results_img[midline, :, :]
            image_tk = ImageTk.PhotoImage(Image.fromarray(display_image))
            displayrecord.append(image_tk)
            imagerecord.append({'img':display_image})

            # show fixed point
            fixedpoint_previous = fixedpoint
            angle = result[ss + ninitial_fixed_segments]['angle']
            coords = result[ss + ninitial_fixed_segments]['coords']
            pos = section_defs[ss + ninitial_fixed_segments]['center']
            # if reverse_order:
            #     vpos_connectionpoint = section_defs[ss + ninitial_fixed_segments]['fixedpoint2'] + pos
            # else:
            #     vpos_connectionpoint = section_defs[ss + ninitial_fixed_segments]['fixedpoint2'] - pos # vector from the center of the section to the connection point
            # correction-----------------------------
            vpos_connectionpoint = section_defs[ss + ninitial_fixed_segments]['fixedpoint2'] - pos # vector from the center of the section to the connection point

            Mx = rotation_matrix(-angle,0)
            My = rotation_matrix(-angley,1)
            Mtotal = np.dot(Mx,My)
            rvpos = np.dot(vpos_connectionpoint, Mtotal) # rotated vector
            # if reverse_order:
            #     fixedpoint = coords - rvpos  # mapped location of the fixed point in the image data
            # else:
            #     fixedpoint = coords + rvpos  # mapped location of the fixed point in the image data
            fixedpoint = coords + rvpos  # mapped location of the fixed point in the image data

            entry = {'coords':coords, 'fixedpoint':fixedpoint, 'fixedpoint_previous':fixedpoint_previous}
            resultsplot.append(entry)
        else:
            ncordsegments = ss-1
            result = result[:(ninitial_fixed_segments+ncordsegments)]
            warpdata = warpdata[:(ninitial_fixed_segments+ncordsegments)]
            break

    if display_output:
        fig = plt.figure(1), plt.imshow(background2[np.round(xs2/2).astype(int),:,:], 'gray')
        for nn in range(len(resultsplot)):
            coords = resultsplot[nn]['coords']
            fixedpoint = resultsplot[nn]['fixedpoint']
            plt.plot([coords[2],fixedpoint[2]], [coords[1],fixedpoint[1]], 'xr-')
            plt.plot([fixedpoint_previous[2], fixedpoint_previous[2]], [fixedpoint_previous[1], fixedpoint_previous[1]], 'xb-')
        plt.show(block = False)

    # display the results - ----------------------
    coords_list = np.zeros((np.size(result),3))
    for nn in range(np.size(result)):
        coords_list[nn, 0:3]= result[nn]['coords']

    # getting ready to create a warping map for the entire image now
    temp, ee = np.unique(coords_list[:,2], return_index = True);
    # use a spline interpolation
    tck = interpolate.splrep(coords_list[ee, 2], coords_list[ee, 1], s=0)
    yy = interpolate.splev(range(zs2), tck, der=0)

    yy = np.where(yy <= -0.5,0,yy)
    yy = np.where(yy >= ys2-0.5,ys2-1,yy)

    cor_results_img = np.zeros((xs2, zs2))
    for zz in range(zs2):
        cor_results_img[:,zz] = results_img[:, np.round(yy[zz]).astype('int'), zz]

    # 4) combine the warp fields from each section into one map
    fit_order = [2, 4, 2]  # "fit_order" could be an input parameter
    found_stable = False
    while not found_stable:
        T, reverse_map_image, forward_map_image, inv_Rcheck = py_combine_warp_fields(warpdata, background2, template, fit_order)
        # reverse_map_image is the mapping of the image data into the template space
        # "reverse" mapping because we had the coordinates of where template voxels mapped into
        # the original image data, and we are calculating where the original image data belongs in the
        # template space

        if np.any(inv_Rcheck > 1.0e22): print('py_cord_normalize:  matrix inversion may be unstable, y fit order = ',fit_order[1],'  ... will try to correct with a lower fit order')
        Ys_max = np.max(T['Ys']);  Ys_min = np.min(T['Ys']);   Ymap_check = (Ys_max < (2*ys2)) & (Ys_max > (-ys2))
        if Ymap_check | (fit_order[1] <= 2):
            found_stable = True
        else:
            fit_order[1] = fit_order[1]-1

    print('py_auto_cord_normalize:  Found a stable warp field solution')

    save_data = True   # turn this on for error-checking
    if save_data:
        wdir = os.path.dirname(os.path.realpath(__file__))
        savename = os.path.join(wdir, 'test_functions/auto_normalize_data_check2.npy')
        results_record = {'T':T, 'reverse_map_image':reverse_map_image, 'forward_map_image':forward_map_image, 'warpdata':warpdata, 'result':result}
        np.save(savename, results_record)

    return T, warpdata, reverse_map_image, forward_map_image, displayrecord, imagerecord, resultsplot, result


#-------------py_norm_fine_tuning----------------------------------------
#---------------------------------------------------------------------------
def py_norm_fine_tuning(input_image, template, T, input_type = 'normalized'):
    # the input_image is the original, non-normalized, image
    # input_type can be:
    #   'normalized' - input_image is image data that have already been rough normalized
    #   'original' - input_image is image data in the orignal format
    #   'filename' - input_image is actually a file name for an original format (not normalized) nifti image

    if input_type == 'original':
        norm_img = py_apply_normalization(input_image, T)
        norm_img = np.where(np.isfinite(norm_img), norm_img, 0.)

    if input_type == 'normalized':
        norm_img = input_image
        norm_img = np.where(np.isfinite(norm_img), norm_img, 0.)

    if input_type == 'filename':
        niiname = input_image   # input_image is a text string in this case
        input_image, affine = i3d.load_and_scale_nifti(niiname)
        norm_img = py_apply_normalization(input_image[:,:,:,3], T)
        norm_img = np.where(np.isfinite(norm_img), norm_img, 0.)

    # set default main settings for MIRT coregistration
    main_init = {'similarity': 'cc',  # similarity measure, e.g. SSD, CC, SAD, RC, CD2, MS, MI
                 'subdivide': 1,  # use 1 hierarchical level
                 'okno': 4,  # mesh window size
                 'lambda': 0.5,  # transformation regularization weight, 0 for none
                 'single': 1}

    # Optimization settings
    optim_init = {'maxsteps': 100,  # maximum number of iterations at each hierarchical level
                  'fundif': 1e-6,  # tolerance (stopping criterion)
                  'gamma': 0.1,  # initial optimization step size
                  'anneal': 0.7}  # annealing rate on the optimization step


    norm_img = norm_img/np.max(norm_img)
    template = template/np.max(template)

    optim = copy.deepcopy(optim_init)
    main = copy.deepcopy(main_init)

    res, norm_img_fine = mirt.py_mirt3D_register(template, norm_img, main, optim)
    print('completed fine-tune mapping with py_norm_fine_tuning ...')

    F = mirt.py_mirt3D_F(res['okno']); # Precompute the matrix B - spline basis functions
    Xx, Xy, Xz = mirt.py_mirt3D_nodes2grid(res['X'], F, res['okno']); # obtain the position of all image voxels (Xx, Xy, Xz)
                                                            # from the positions of B-spline control points (res['X']

    xs, ys, zs = np.shape(norm_img)
    X, Y, Z = np.mgrid[range(xs), range(ys), range(zs)]

    # fine-tuning deviation from the original positions
    dX = Xx[:xs,:ys,:zs]-X
    dY = Xy[:xs,:ys,:zs]-Y
    dZ = Xz[:xs,:ys,:zs]-Z

    print('estimating reverse mapping...');
    fit_order = 3;
    # output dimensions will be scaled to 1 mm voxels so the size = FOV
    Xt, Yt, Zt = py_estimate_reverse_transform(T['Xs']+dX,T['Ys']+dY,T['Zs']+dZ, np.shape(input_image), fit_order=3)

    dXt = Xt - T['Xs']
    dYt = Yt - T['Ys']
    dZt = Zt - T['Zs']

    Tfine = {'dXs':dX, 'dYs':dY, 'dZs':dZ, 'dXt':dXt, 'dYt':dYt, 'dZt':dZt}
    print('fine-tuning mapping function complete...')

    return Tfine, norm_img_fine


#------------define_sections--------------------------------------------------------
#-----------------------------------------------------------------------------------
def define_sections(template_name, dataname):
    # define sections and reference point coordinates for this choice of
    # template section

    # need to consistent with functions in load_templates.py
    region_name = 'notdefined'
    region_name2 = 'notdefined'
    reverse_order = False
    range_specified = False
    a = template_name.find('to')
    if a > 0:  # a range of cord segments is specified
        region_name2 = template_name[a + 2:]
        region_name = template_name[:a]
        range_specified = True
    else:
        region_name2 = ''

    if template_name.lower() == 'thoracic':
        region_name = 'T1'
        region_name2 = 'T12'

    if range_specified:
        input_img = nib.load(dataname)
        hdr = input_img.header
        FOV = hdr['pixdim'][1:4]*hdr['dim'][1:4]
        pos_estimate = np.array([np.round(FOV[0]/2), np.round(FOV[1]/2), np.round(FOV[2]/2)]).astype(int)   # start at the middle of the image

        resolution = 1
        template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_map = load_templates.load_template_and_masks(
            template_name, resolution)
        namelist = [name[:3] for num,name in enumerate(anatlabels['names'])]
        nd = len(region_name)
        rnums = [anatlabels['numbers'][x] for x, name in enumerate(anatlabels['names']) if name[:nd] == region_name]
        test = np.array([regionmap_img == val for val in rnums]).any(axis=0)
        cx,cy,cz = np.where(test)   # region at superior end of range
        zmax = np.max(cz)
        nd = len(region_name2)
        rnums = [anatlabels['numbers'][x] for x, name in enumerate(anatlabels['names']) if name[:nd] == region_name2]
        test = np.array([regionmap_img == val for val in rnums]).any(axis=0)
        cx,cy,cz = np.where(test)   # region at inferior end of range
        zmin = np.min(cz)

        # check if lower and/or upper lumbar region is included in range
        section1 = [x for x, v in enumerate(['C', 'T', 'L', 'S']) if v == region_name[0]]
        segment1 = int(region_name[1:])
        section2 = [x for x, v in enumerate(['C', 'T', 'L', 'S']) if v == region_name2[0]]
        segment2 = int(region_name2[1:])
        value1 = 20*section1[0] + segment1
        value2 = 20*section2[0] + segment2

        # define sections in reverse order when starting at lumbar regions
        reverse_order = False
        ninitial_fixed_segments = 0

        rnums = [anatlabels['numbers'][x] for x, name in enumerate(anatlabels['names']) if name[:3] == 'T12']
        test = np.array([regionmap_img == val for val in rnums]).any(axis=0)
        cx, cy, cz = np.where(test)
        z_bottom_of_T12 = np.round(np.min(cz)).astype(int)  # the bottom of the T12 segment

        z_connection_point = zmax  # ... top of range

        # check for lumbar regions - specify first sections to match
        if value1 < 60 and value2 > 63:
            z_connection_point = z_bottom_of_T12
            reverse_order = True
            ninitial_fixed_segments += 1
            print('range includes sacral regions ...')
            # rnums = [anatlabels['numbers'][x] for x, name in enumerate(anatlabels['names']) if name[:2] == 'S2']
            # test = np.array([regionmap_img == val for val in rnums]).any(axis=0)
            # cx, cy, cz = np.where(test)
            # zref = np.round(np.mean(cz)).astype(int)   # the middle of the L4 segment

            rnums = [anatlabels['numbers'][x] for x, name in enumerate(anatlabels['names']) if name[:2] == 'S5']
            test = np.array([regionmap_img == val for val in rnums]).any(axis=0)
            cx, cy, cz = np.where(test)
            z_bottom_of_S5 = np.round(np.min(cz)).astype(int)  # the bottom of the S5 segment

            rnums = [anatlabels['numbers'][x] for x, name in enumerate(anatlabels['names']) if name[:2] == 'S1']
            test = np.array([regionmap_img == val for val in rnums]).any(axis=0)
            cx, cy, cz = np.where(test)
            z_top_of_S1 = np.round(np.max(cz)).astype(int)  # the top of the S1 segment

            zref = np.floor((z_bottom_of_S5 + z_top_of_S1) / 2.0).astype(int)
            zdim = z_top_of_S1 - zref  # +/- range

            sacraldefined = True
            sacral = {'name': 'sacral',
                       'center': np.array([12, 15, zref]),
                       'dims': np.array([6, 10, zdim]),
                       'xrot': 0,
                       'yrot': 0,
                       'start_ref_pos': [],
                       'pos_estimate': pos_estimate,
                       'fixdistance': 0}
        else:
            sacraldefined = False

        # check for lumbar regions - specify first sections to match
        if value1 < 44 and value2 > 45:
            z_connection_point = z_bottom_of_T12
            reverse_order = True
            ninitial_fixed_segments += 1
            print('range includes lower lumbar region ...')
            # rnums = [anatlabels['numbers'][x] for x, name in enumerate(anatlabels['names']) if name[:2] == 'L4']
            # test = np.array([regionmap_img == val for val in rnums]).any(axis=0)
            # cx, cy, cz = np.where(test)
            # zref = np.round(np.mean(cz)).astype(int)   # the middle of the L4 segment

            rnums = [anatlabels['numbers'][x] for x, name in enumerate(anatlabels['names']) if name[:2] == 'L5']
            test = np.array([regionmap_img == val for val in rnums]).any(axis=0)
            cx, cy, cz = np.where(test)
            z_bottom_of_L5 = np.round(np.min(cz)).astype(int)  # the bottom of the L5 segment

            rnums = [anatlabels['numbers'][x] for x, name in enumerate(anatlabels['names']) if name[:2] == 'L3']
            test = np.array([regionmap_img == val for val in rnums]).any(axis=0)
            cx, cy, cz = np.where(test)
            z_middle_of_L3 = np.round(np.mean(cz)).astype(int)  # the middle of the L3 segment

            zref = np.floor((z_bottom_of_L5 + z_middle_of_L3) / 2.0).astype(int)
            zdim = z_middle_of_L3 - zref  # +/- range

            lowerlumbardefined = True
            lowerlumbar = {'name': 'lower_lumbar',
                       'center': np.array([12, 15, zref]),
                       'dims': np.array([6, 10, zdim]),
                       'xrot': 0,
                       'yrot': 0,
                       'start_ref_pos': [],
                       'pos_estimate': pos_estimate,
                       'fixdistance': 0}
        else:
            lowerlumbardefined = False

        # check for lumbar regions - specify first sections to match
        if value1 < 40 and value2 > 43:
            z_connection_point = z_bottom_of_T12
            reverse_order = True
            ninitial_fixed_segments += 1
            print('range includes upper lumbar region ...')
            # rnums = [anatlabels['numbers'][x] for x, name in enumerate(anatlabels['names']) if name[:2] == 'L2']
            # test = np.array([regionmap_img == val for val in rnums]).any(axis=0)
            # cx, cy, cz = np.where(test)
            # zref = np.round(np.mean(cz)).astype(int)   # the middle of the L2 segment

            rnums = [anatlabels['numbers'][x] for x, name in enumerate(anatlabels['names']) if name[:2] == 'L3']
            test = np.array([regionmap_img == val for val in rnums]).any(axis=0)
            cx, cy, cz = np.where(test)
            z_middle_of_L3 = np.round(np.mean(cz)).astype(int)  # the middle of the L3 segment

            rnums = [anatlabels['numbers'][x] for x, name in enumerate(anatlabels['names']) if name[:2] == 'L1']
            test = np.array([regionmap_img == val for val in rnums]).any(axis=0)
            cx, cy, cz = np.where(test)
            z_top_of_L1 = np.round(np.max(cz)).astype(int)  # the top of the L1 segment

            zref = np.floor((z_top_of_L1 + z_middle_of_L3) / 2.0).astype(int)  # middle of range
            zdim = z_top_of_L1 - zref  # +/- range

            upperlumbardefined = True
            upperlumbar = {'name': 'upper_lumbar',
                       'center': np.array([12, 15, zref]),
                       'dims': np.array([6, 10, zdim]),
                       'xrot': 0,
                       'yrot': 0,
                       'start_ref_pos': [],
                       'pos_estimate': pos_estimate,
                       'fixdistance': 0,
                       'first_region_connection_point': [12, 15, z_bottom_of_T12]}  # first_region_connection_point is the bottom edge of T12
        else:
            upperlumbardefined = False

        # make consistent with other sections ...............[update this]-------------
        # first_region_connection_point is where bottom of thoracic region connects to lumbar region
        first_region_connection_point = np.array([12, 15, z_connection_point])  # [13 31 7] + [0 0 259] in matlab
        dz = 13
        # if reverse_order:
        #     ncsections = np.floor((zmax-z_connection_point)/ dz - 0.5).astype('int')
        # else:
        #     ncsections = np.floor(z_connection_point / dz - 0.5).astype('int')
        ncsections = np.floor(z_connection_point / dz - 0.5).astype('int')
        # working from top-down, or reverse order?

        # initialize_section_defs
        single_def = {'name': 0, 'center': 0, 'dims': 0, 'xrot': 0, 'yrot': 0, 'pos_estimate': 0, 'fixdistance': 0,
                      'fixedpoint1': 0, 'fixedpoint2': 0, 'first_region_connection_point': 0}
        section_defs = []
        for ss in range(ninitial_fixed_segments + ncsections):
            section_defs.append(single_def.copy())

        # put in initial sections
        segcount = 0
        if upperlumbardefined:
            section_defs[segcount] = upperlumbar   # needs to be first
            segcount += 1
        if lowerlumbardefined:
            section_defs[segcount] = lowerlumbar
            segcount += 1
        if sacraldefined:
            section_defs[segcount] = sacral
            segcount += 1

        for ss in range(ncsections):
            if reverse_order:
                zcenter = (z_connection_point + np.floor(dz * (ss + 0.5))).astype('int')
                zrefpoint = (z_connection_point + np.floor(dz * 0.5)).astype('int')
                fixedpoint1 = first_region_connection_point + np.array([0, 0, dz * ss])
                fixedpoint2 = first_region_connection_point + np.array([0, 0, dz * (ss + 1)])
            else:
                zcenter = (z_connection_point - np.floor(dz * (ss + 0.5))).astype('int')
                zrefpoint = (z_connection_point - np.floor(dz * 0.5)).astype('int')
                fixedpoint1 = first_region_connection_point - np.array([0, 0, dz * ss])
                fixedpoint2 = first_region_connection_point - np.array([0, 0, dz * (ss + 1)])

            section_defs[ss+ninitial_fixed_segments]['center'] = [12, 15, zcenter]
            section_defs[ss+ninitial_fixed_segments]['dims'] = [6, 10, np.floor(dz/2).astype('int')]  # span +/- from center
            section_defs[ss+ninitial_fixed_segments]['name'] = 'cord{}'.format(ss)
            section_defs[ss+ninitial_fixed_segments]['fixdistance'] = 1
            section_defs[ss+ninitial_fixed_segments]['refpoint'] = [12, 30, zrefpoint]
            section_defs[ss+ninitial_fixed_segments]['pos_estimate'] = []
            section_defs[ss+ninitial_fixed_segments]['xrot'] = 0
            section_defs[ss+ninitial_fixed_segments]['yrot'] = 0
            section_defs[ss+ninitial_fixed_segments]['fixedpoint1'] = fixedpoint1  # more superior connection point
            section_defs[ss+ninitial_fixed_segments]['fixedpoint2'] = fixedpoint2  # most inferior connection point

        # section_defs[0]['first_region_connection_point'] = first_region_connection_point  # more superior connection point
        #---------------above here is not done yet------------------------------

    # if template_name.lower() not in ['thoracic', 'sacral']:
    #     template_name = 'ccbs'   # default

    if template_name.lower() == 'thoracic':
        # 18 pixels of overlap with cervical region, in 3rd (z) dimension
        # top of cord in cervical is at 121 + 259 = 380
        # sections are length 13 (dz), top segment in thoracic is at  380 - 8 * 13 = 276
        # need to check the positions in the new template - July 2020 -----------------------------------------
        first_region_connection_point = np.array([12, 15, 265])  # [13 31 7] + [0 0 259] in matlab
        ninitial_fixed_segments = 1
        dz = 13
        ncsections = np.floor(286/dz - 0.5).astype('int')

        # initialize_section_defs
        single_def = {'name': 0, 'center': 0, 'dims': 0, 'xrot':0, 'yrot':0, 'pos_estimate': 0, 'fixdistance': 0, 'fixedpoint1': 0, 'fixedpoint2': 0, 'first_region_connection_point':0}
        section_defs = []
        for ss in range(ninitial_fixed_segments+ncsections):
            section_defs.append(single_def.copy())

        for ss in range(ncsections):
            section_defs[ss]['center'] = [12, 15, (275 - np.floor(dz*(ss+0.5))).astype('int')]
            section_defs[ss]['dims'] = [6, 10, np.floor(dz/2).astype('int')]  #span +/- from center

            section_defs[ss]['name'] = sprintf('tcord%d', ss);
            section_defs[ss]['fixdistance'] = 1;
            section_defs[ss]['refpoint'] = [12, 15, (120 + 259 - np.floor(dz*0.5)).astype('int')];
            section_defs[ss]['pos_estimate'] = []
            section_defs[ss]['xrot'] = 0
            section_defs[ss]['yrot'] = 0
            section_defs[ss]['fixedpoint1'] = first_region_connection_point - np.array([0, 0, dz*ss]) # more superior connection point
            section_defs[ss]['fixedpoint2'] = first_region_connection_point - np.array([0, 0, dz*(ss+1)]) # most inferior connection point

        section_defs[0]['first_region_connection_point'] = first_region_connection_point  # more superior connection point

        # map the cervical ref image to the thoracic image to determine the relationship between
        # the coordinates
        # get the first section position for thoracic from the last seciton position for cervical

        print('Setting up reference information based on thoracic reference ...')
        refdata = np.load(cervicalrefnormdata)   # this cervicalrefnormdata has not been defined yet ........(July 22 2020)--------

        # ---------need to sort this out for the python version ---------------------------
        Mcoords = refdata['warpdata'][-1]['coords'] # the last section for cervical is the same as the first section for thoracic
        new_pixdim = [1,1,1]  # images are resized to 1 mm cubic voxels for normalization

        # Mapped_coords = QU_map_one_image_to_another(filename, cervicalrefimg, Mcoords, new_pixdim)
        print('pynormalization:   ... not yet set up to handle thoracic normalization....')

        thoracic_start_ref_pos = Mapped_coords
        section_defs[0]['start_ref_pos'] = np.round(thoracic_start_ref_pos).astype('int')
        section_defs[0]['start_angle'] = refdata['warpdata'][11]['angle']
        section_defs[0]['pos_estimate'] = thoracic_start_ref_pos
        section_defs[0]['fixdistance'] = 1
        section_defs[0]['xrot'] = 0
        section_defs[0]['yrot'] = 0

        print('Set up thoracic section information based on cervical reference ...')

    if template_name.lower() == 'ccbs':
        ninitial_fixed_segments = 3
        input_img = nib.load(dataname)
        hdr = input_img.header
        FOV = hdr['pixdim'][1:4]*hdr['dim'][1:4]
        pos_estimate = [FOV[0]/2, FOV[1]/2, np.round(FOV[2]*0.75)]
        # put the initial sections anywhere in the right half of the image
        first_region_connection_point = np.array([12, 31, 122]); # point on the section that links to the first cord section

        # define template sections
        # ccbs template sections, 1mm resolution
        medulla = {'name': 'medulla',
                   'center': np.array([12, 36, 136]),
                   'dims': np.array([6, 15, 15]),
                   'xrot': 20,
                   'yrot': 0,
                   'start_ref_pos': [],
                   'pos_estimate': pos_estimate,
                   'fixdistance': 0,
                   'first_region_connection_point': first_region_connection_point}  # first_region_connection_point is the top edge of C1

        midbrain = {'name':'midbrain',
                    'center': np.array([12, 51, 173]),
                    'dims': np.array([6, 15, 15]),
                    'xrot': 20,
                    'yrot': 0,
                    'start_ref_pos': [],
                    'pos_estimate': pos_estimate,
                    'fixdistance': 0}

        thalamus = {'name':'thalamus',
                    'center': np.array([12, 64, 200]),
                    'dims': np.array([6, 15, 15]),
                    'xrot': 20,
                    'yrot': 0,
                    'start_ref_pos': [],
                    'pos_estimate': pos_estimate,
                    'fixdistance': 0}

        # cord sections
        dz = 13
        ncsections = np.floor(121/dz).astype('int')

        # initialize_section_defs
        single_def = {'name': 0, 'center': 0, 'dims': 0, 'xrot':0, 'yrot':0, 'pos_estimate': 0, 'fixdistance': 0, 'fixedpoint1': 0, 'fixedpoint2': 0, 'first_region_connection_point':0}
        section_defs = []
        for ss in range(ninitial_fixed_segments+ncsections):
            section_defs.append(single_def.copy())

        section_defs[0] = medulla   # start with these initial segments
        section_defs[1] = midbrain
        section_defs[2] = thalamus

        for ss in range(ncsections):
            section_defs[ss+ninitial_fixed_segments]['center'] = first_region_connection_point - np.array([0, 0, np.round(dz*(ss + 0.5))])
            section_defs[ss+ninitial_fixed_segments]['dims'] = [6, 10, np.round(dz/2).astype('int')]  # span +/- from center
            section_defs[ss+ninitial_fixed_segments]['fixedpoint1'] = first_region_connection_point - np.array([0, 0, dz*ss]) # more superior connection point
            section_defs[ss+ninitial_fixed_segments]['fixedpoint2'] = first_region_connection_point - np.array([0, 0, dz*(ss+1)]) # more inferior connection point
            section_defs[ss+ninitial_fixed_segments]['name'] = 'cord{}'.format(ss+1)
            section_defs[ss+ninitial_fixed_segments]['pos_estimate'] = []
            section_defs[ss+ninitial_fixed_segments]['fixdistance'] = 1
            section_defs[ss+ninitial_fixed_segments]['xrot'] = 0
            section_defs[ss+ninitial_fixed_segments]['yrot'] = 0

        print('Set up reference information based on cervical reference ...')

    return section_defs, ninitial_fixed_segments, reverse_order



#------------run_rough_normalization_calculations--------------------------------------------------------
#-----------------------------------------------------------------------------------
def run_rough_normalization_calculations(niiname, normtemplatename, template_img, fit_parameters):  # , display_window = 'None', display_image1 = 'None', display_window2 = 'None', display_image2 = 'None'
    # this might go in a separate module or function
    section_defs, ninitial_fixed_segments, reverse_order = define_sections(normtemplatename, niiname)
    # resolution = 1
    # template_img, regionmap_img, template_affine = load_templates.load_template(normtemplatename, resolution)
    # actually run the normalization steps now

    # load the nifti data and rescale to 1 mm voxels
    input_datar, affiner = i3d.load_and_scale_nifti(niiname)

    # displaycount = 0
    # display_record = []
    # print('Updating initial display in Window 1 ...')
    if np.ndim(input_datar) > 3:
        x,y,z,t = np.shape(input_datar)
        if t > 3:
            t0 = 3
        else:
            t0=0
        input_image = input_datar[:,:,:,t0]
    else:
        x,y,z = np.shape(input_datar)
        input_image = input_datar

    # rough normalization
    T, warpdata, reverse_map_image, forward_map_image, displayrecord, imagerecord, resultsplot, result = py_auto_cord_normalize(input_image, template_img,
                    fit_parameters, section_defs, ninitial_fixed_segments, reverse_order, display_output = False)  # , display_window, display_image1, display_window2, display_image2
    # fine-tune the normalization
    # Tfine, norm_image_fine = normcalc.py_norm_fine_tuning(input_datar, template_img, T)
    return T, warpdata, reverse_map_image, displayrecord, imagerecord, resultsplot, result


#------------align_override_sections--------------------------------------------------------
#-----------------------------------------------------------------------------------
def align_override_sections(normalization_results, adjusted_sections, niiname, normtemplatename):
    # after manual over-ride has been used to move template sections,
    # the positions must be made consistent across all sections
    # "adjusted_sections" is the list of sections that have been moved,
    # try to keep these where they are
    # move all the rest of the sections as little as possible
    output_results = copy.deepcopy(normalization_results)
    section_defs, ninitial_fixed_segments, reverse_order = define_sections(normtemplatename, niiname)

    nsections = len(normalization_results)
    # initial cord segments must be handled differently - can be out of order and not connected
    fixedrecord1 = []
    fixedrecord2 = []
    ncordsegments = len(section_defs) - ninitial_fixed_segments

    nsections = len(normalization_results)
    ncordsegments = nsections- ninitial_fixed_segments
    angle = np.zeros(ncordsegments)
    angley = np.zeros(ncordsegments)
    coords = np.zeros((ncordsegments,3))
    for ss in range(ncordsegments):
        angle[ss] = normalization_results[ss + ninitial_fixed_segments]['angle']
        angley[ss] = normalization_results[ss + ninitial_fixed_segments]['angley']
        coords[ss,:] = normalization_results[ss + ninitial_fixed_segments]['coords']

    # use a minimization method - gradient descent
    # cost = sum of distances between points that should be joining
    #       plus a scaled amout of the sum of the amounts that sections need to be moved
    #       with sections weighted according to "importance"
    delta_angle = np.zeros(ncordsegments)
    delta_angley = np.zeros(ncordsegments)
    delta_coords = np.zeros((ncordsegments,3))
    weighting = np.ones(ncordsegments)
    if len(adjusted_sections) > 0:
        weighting[adjusted_sections - ninitial_fixed_segments] = 10.0  # give more weighting to keeping the adjust sections where they were put to

    deltav = 1.0e-1
    alpha = 1.0e-1
    total_cost, dist = align_cost(normalization_results, section_defs, ninitial_fixed_segments, coords, angle, angley, delta_coords, delta_angle, delta_angley, weighting)

    nitermax = 5000
    tol = 1.0e-4
    delta_cost = total_cost
    cost_record = [total_cost]
    descent_record = []
    print('pynormalization:  aligning sections for consistency ...')
    nn=0
    while (nn < nitermax) & (delta_cost > tol):
        total_cost, dist = align_cost(normalization_results, section_defs, ninitial_fixed_segments, coords, angle, angley, delta_coords, delta_angle, delta_angley, weighting)
        # calculate gradients
        coords_cost_gradient, angle_cost_gradient, angley_cost_gradient = align_cost_gradients(deltav, normalization_results, section_defs, ninitial_fixed_segments, coords, angle, angley, delta_coords, delta_angle, delta_angley, weighting)
        delta_coords = delta_coords - alpha*coords_cost_gradient
        delta_angle = delta_angle - alpha * angle_cost_gradient
        delta_angley = delta_angley - alpha * angley_cost_gradient
        new_total_cost, dist = align_cost(normalization_results, section_defs, ninitial_fixed_segments, coords, angle, angley, delta_coords, delta_angle, delta_angley, weighting)
        cost_record.append(new_total_cost)
        delta_cost = total_cost - new_total_cost
        entry = {'delta_coords':delta_coords, 'delta_angle':delta_angle, 'delta_angley':delta_angley, 'iteration':nn, 'delta_cost':delta_cost}
        descent_record.append(entry)
        # print('iteration {}: cost = {} '.format(nn,new_total_cost))
        nn += 1

    for ss in range(ncordsegments):
        output_results[ss + ninitial_fixed_segments]['angle'] = angle[ss] + delta_angle[ss]
        output_results[ss + ninitial_fixed_segments]['angley'] = angley[ss] + delta_angley[ss]
        output_results[ss + ninitial_fixed_segments]['coords'] = coords[ss,:] + delta_coords[ss,:]

    wdir = os.path.dirname(os.path.realpath(__file__))
    debugrecordname = os.path.join(wdir,'debug_mano_data_check.npy')
    np.save(debugrecordname, descent_record)

    return output_results


def align_cost(normalization_results, section_defs, ninitial_fixed_segments, coords, angle, angley, delta_coords, delta_angle, delta_angley, weighting):
    nsections = len(normalization_results)
    ncordsegments = len(angle)

    fixedrecord1 = []
    fixedrecord2 = []

    for ss in range(ncordsegments):
        # angle = normalization_results[ss + ninitial_fixed_segments]['angle']
        # angley = normalization_results[ss + ninitial_fixed_segments]['angley']
        # coords = normalization_results[ss + ninitial_fixed_segments]['coords']
        pos = section_defs[ss + ninitial_fixed_segments]['center']
        vpos_connectionpoint1 = section_defs[ss + ninitial_fixed_segments]['fixedpoint1'] - pos
        vpos_connectionpoint2 = section_defs[ss + ninitial_fixed_segments]['fixedpoint2'] - pos

        Mx = rotation_matrix(-angle[ss]-delta_angle[ss], 0)
        My = rotation_matrix(-angley[ss]-delta_angley[ss], 1)
        Mtotal = np.dot(Mx, My)
        rvpos = np.dot(vpos_connectionpoint1, Mtotal)  # rotated vector
        fixedpoint1 = coords[ss,:]+delta_coords[ss,:] + rvpos  # mapped location of fixedpoint in the image data
        rvpos = np.dot(vpos_connectionpoint2, Mtotal)  # rotated vector
        fixedpoint2 = coords[ss,:]+delta_coords[ss,:] + rvpos  # mapped location of fixedpoint in the image data
        fixedrecord1.append(fixedpoint1)
        fixedrecord2.append(fixedpoint2)

    dist = np.linalg.norm(np.array(fixedrecord1[1:][:]) - np.array(fixedrecord2[:-1][:]),axis=1)
    offset = np.linalg.norm(delta_coords,axis=1)
    total_cost = 1.0e-3*np.sum(weighting*offset) + np.sum(np.abs(dist-0.5))

    return total_cost, dist


def align_cost_gradients(deltav, normalization_results, section_defs, ninitial_fixed_segments, coords, angle, angley, delta_coords, delta_angle, delta_angley, weighting):

    ndc = np.size(delta_coords)
    n1,n2 = np.shape(delta_coords)
    coords_cost_record = np.zeros(ndc)
    angle_cost_record = np.zeros(n1)
    angley_cost_record = np.zeros(n1)

    for nn in range(ndc):
        dc = np.zeros(ndc)
        dc[nn] = deltav
        dc = dc.reshape(n1,n2)
        total_cost, dist = align_cost(normalization_results, section_defs, ninitial_fixed_segments, coords, angle, angley, dc+delta_coords, delta_angle, delta_angley, weighting)
        coords_cost_record[nn] = total_cost
    coords_cost_record = coords_cost_record.reshape(n1,n2)

    for nn in range(n1):
        da = np.zeros(n1)
        da[nn] = deltav
        total_cost, dist = align_cost(normalization_results, section_defs, ninitial_fixed_segments, coords, angle, angley, delta_coords, da+delta_angle, delta_angley, weighting)
        angle_cost_record[nn] = total_cost

    for nn in range(n1):
        day = np.zeros(n1)
        day[nn] = deltav
        total_cost, dist = align_cost(normalization_results, section_defs, ninitial_fixed_segments, coords, angle, angley, delta_coords, delta_angle, day+delta_angley, weighting)
        angley_cost_record[nn] = total_cost

    total_cost, dist = align_cost(normalization_results, section_defs, ninitial_fixed_segments, coords, angle, angley, delta_coords, delta_angle, delta_angley, weighting)
    coords_cost_gradient = (coords_cost_record - total_cost)/deltav
    angle_cost_gradient = (angle_cost_record - total_cost)/deltav
    angley_cost_gradient = (angley_cost_record - total_cost)/deltav

    return coords_cost_gradient, angle_cost_gradient, angley_cost_gradient


# -------------py_load_modified_normalization-------------------------------------------------
# ------------------------------------------------------------------------------------
def py_load_modified_normalization(niiname, normtemplatename, new_result):   # , display_window = 'None',display_image1 = 'none', display_window2 = 'None', display_image2 = 'none'
    # return T, warpdata, reverse_map_image, forward_map_image, new_result
    imagerecord = []
    displayrecord = []

    resolution = 1
    template, regionmap_img, template_affine, anatlabels = load_templates.load_template(normtemplatename, resolution)

    print('running py_load_modified_normalization ...')
    section_defs, ninitial_fixed_segments, reverse_order = define_sections(normtemplatename, niiname)

    # load the nifti data and rescale to 1 mm voxels
    input_datar, affiner = i3d.load_and_scale_nifti(niiname)
    if np.ndim(input_datar) > 3:
        x,y,z,t = np.shape(input_datar)
        if t > 3:
            t0 = 3
        else:
            t0=0
        input_image = input_datar[:,:,:,t0]
    else:
        x,y,z = np.shape(input_datar)
        input_image = input_datar

    background2 = input_image

    xs2, ys2, zs2 = np.shape(background2)
    xs, ys, zs = np.shape(template)

    # 1) initial sections
    # initialize result and warpdata
    # result_entry = {'angle':0, 'angley':0, 'coords':[0,0,0], 'original_section':[], 'template_section':[], 'section_mapping_coords':[]}
    warpdata_entry = {'X': 0, 'Y': 0, 'Z': 0, 'Xt': 0, 'Yt': 0, 'Zt': 0}
    warpdata = []

    nsections = len(new_result)

    midline = np.zeros(nsections)
    for aa in range(nsections):
        midline[aa] = new_result[aa]['coords'][0]
    midline = np.round(np.mean(midline)).astype('int')

    for ss in range(nsections):
        warpdata.append(warpdata_entry.copy())
        coords = new_result[ss]['coords']
        angle = new_result[ss]['angle']
        angley = new_result[ss]['angley']
        # new part---------------------------------
        # calculate values for modified normalization positions - done with manual override
        section_mapping_coords, original_section = py_modify_section_positions(template, background2, section_defs[ss], coords, angle, angley)

        new_result[ss]['original_section'] = original_section
        # new_result[ss]['template_section'] = template_section    # does not need to be updated
        new_result[ss]['section_mapping_coords'] = section_mapping_coords

        # check values
        print('section {}  coords = {}'.format(ss,coords))
        xx = np.mean(new_result[ss]['section_mapping_coords']['X'])
        yy = np.mean(new_result[ss]['section_mapping_coords']['Y'])
        zz = np.mean(new_result[ss]['section_mapping_coords']['Z'])
        print('          X,Y,Z midpoints = {}'.format((xx,yy,zz)))

        # save another copy, for convenience later
        warpdata[ss]['X'] = new_result[ss]['section_mapping_coords']['X']
        warpdata[ss]['Y'] = new_result[ss]['section_mapping_coords']['Y']
        warpdata[ss]['Z'] = new_result[ss]['section_mapping_coords']['Z']
        warpdata[ss]['Xt'] = new_result[ss]['section_mapping_coords']['Xt']
        warpdata[ss]['Yt'] = new_result[ss]['section_mapping_coords']['Yt']
        warpdata[ss]['Zt'] = new_result[ss]['section_mapping_coords']['Zt']

        # print('finished compiling results for section ',ss, ' ...')

        print('py_load_modified_normalization: ninitial_fixed_segments = ',ninitial_fixed_segments)
        print('py_load_modified_normalization: ss = ',ss)
        print('py_load_modified_normalization: size of warpdata = ',np.shape(warpdata))

        # results_img = py_display_sections_local(template, background2, warpdata[:(ss+ninitial_fixed_segments+1)])
        results_img = py_display_sections_local(template, background2, warpdata[:ss])

        # display the result again
        # need to figure out how to control which figure is used for display  - come back to this ....
        display_image = results_img[midline, :, :]
        image_tk = ImageTk.PhotoImage(Image.fromarray(display_image))
        displayrecord.append(image_tk)
        imagerecord.append({'img':display_image})

    # 4) combine the warp fields from each section into one map
    fit_order = [2, 4, 2]  # "fit_order" could be an input parameter
    found_stable = False
    while not found_stable:
        T, reverse_map_image, forward_map_image, inv_Rcheck = py_combine_warp_fields(warpdata, background2, template, fit_order)
        # reverse_map_image is the mapping of the image data into the template space
        # "reverse" mapping because we had the coordinates of where template voxels mapped into
        # the original image data, and we are calculating where the original image data belongs in the
        # template space

        if np.any(inv_Rcheck > 1.0e22): print('py_cord_normalize:  matrix inversion may be unstable, y fit order = ',fit_order[1],'  ... will try to correct with a lower fit order')
        Ys_max = np.max(T['Ys']);  Ys_min = np.min(T['Ys']);   Ymap_check = (Ys_max < (2*ys2)) & (Ys_max > (-ys2))
        if Ymap_check | (fit_order[1] <= 2):
            found_stable = True
        else:
            fit_order[1] = fit_order[1]-1

    print('py_auto_cord_normalize:  Found a stable warp field solution')

    save_data = True   # turn this on for error-checking
    if save_data:
        wdir = os.path.dirname(os.path.realpath(__file__))
        savename = os.path.join(wdir, 'test_functions/auto_normalize_data_check2.npy')
        result = new_result
        results_record = {'T':T, 'reverse_map_image':reverse_map_image, 'forward_map_image':forward_map_image, 'warpdata':warpdata, 'result':result}
        np.save(savename, results_record)

    return T, warpdata, reverse_map_image, forward_map_image, new_result, imagerecord, displayrecord


#----------------------------------------------------------------------------
#-------------calculate normalization values with new positions--------------
def py_modify_section_positions(template, img, section_defs, coords, angle, angley):
    # return section_mapping_coords, original_section, template_section

    # input is the entire template, and sections are extracted
    xt, yt, zt = np.shape(template)
    xs, ys, zs = np.shape(img)
    img = img/np.max(img)

    # section_defs relate to the template
    zpos = section_defs['center'][2]
    dz = section_defs['dims'][2]
    zr = [zpos - dz, zpos + dz]
    ypos = section_defs['center'][1]
    dy = section_defs['dims'][1]
    yr = [ypos - dy, ypos + dy]
    xpos = section_defs['center'][0]
    dx = section_defs['dims'][0]
    xr = [xpos - dx, xpos + dx]

    # get coordinates of image section and template
    dzm = dz
    dym = dy
    zpos2 = zpos
    x1 = 0
    x2 = 2*dx + 1
    y1 = dym - dy
    y2 = dym + dy
    z1 = dzm - dz
    z2 = dzm + dz

    # reverse rotation:
    p0 = np.round(np.array([xs, ys, zs]) / 2).astype('int')
    Mx = rotation_matrix(angle, axis=0)
    My = rotation_matrix(angley, axis=1)
    Mtotal = np.dot(My,Mx)
    coordsR = np.dot((coords - p0),Mtotal) + p0

    # rotate from image coordinates:
    # coordsR = input_coords
    # p0 = np.round(np.array([xs, ys, zs]) / 2).astype('int')
    # Mx = rotation_matrix(-angle, axis=0)
    # My = rotation_matrix(-angley, axis=1)
    # Mtotal = np.dot(My,Mx)
    # coords = np.dot((coordsR - p0),Mtotal) + p0

    # get the coordinates for the image, where the template section maps to
    # first, get the coordinates in the rotated template
    # ===> need the coordinates in the image, not in the template
    # Xt, etc are template coordinates
    Xt, Yt, Zt = np.mgrid[(xpos-dx):(xpos+dx):(2*dx+1)*1j, (ypos-dym):(ypos+dym):(2*dym+1)*1j, (zpos-dzm):(zpos+dzm):(2*dzm+1)*1j]
    # X etc are image coordinates, in the rotate image to start with
    X, Y, Z = np.mgrid[(coordsR[0]-dx):(coordsR[0]+dx):(2*dx+1)*1j, (coordsR[1]-dym):(coordsR[1]+dym):(2*dym+1)*1j, (coordsR[2]-dzm):(coordsR[2]+dzm):(2*dzm+1)*1j]
    # next, need to rotate both sets of coordinates, -xrot for the template, and angle-xrot for the image
    # then rotate these coordinates by section_defs['sectionangle']

    # this is for rotating the image coordinates
    p0 = np.round(np.array([xs, ys, zs]) / 2).astype('int')
    sa = (np.pi/180)*angle
    Xr = X
    Yr = (Y - p0[1])*math.cos(sa) - (Z - p0[2])*math.sin(sa) + p0[1]
    Zr = (Z - p0[2])*math.cos(sa) + (Y - p0[1])*math.sin(sa) + p0[2]

    sa = (np.pi/180)*angley
    Xrr = (Xr - p0[0])*math.cos(sa) - (Zr - p0[2])*math.sin(sa) + p0[0]
    Yrr = Yr
    Zrr = (Zr - p0[2])*math.cos(sa) + (Xr - p0[0])*math.sin(sa) + p0[2]

    Xr = Xrr
    Yr = Yrr
    Zr = Zrr

    # this is for rotating the template coordinates
    sa = (np.pi/180)*(-section_defs['xrot'])
    Xtr = Xt
    Ytr = (Yt - ypos)*math.cos(sa) - (Zt - zpos) * math.sin(sa) + ypos
    Ztr = (Zt - zpos) * math.cos(sa) + (Yt - ypos) * math.sin(sa) + zpos

    # is this necessary?
    # sa = (np.pi/180)*(angley)
    # Xtrr = (Xtr - xpos)*math.cos(sa) - (Ztr - zpos)*math.sin(sa) + xpos
    # Ytrr = Ytr
    # Ztrr = (Ztr - zpos)*math.cos(sa) + (Xtr - xpos)*math.sin(sa) + zpos
    #
    # Xtr = Xtrr
    # Ytr = Ytrr
    # Ztr = Ztrr

    # make sure coordinates are within limits
    Xtr = np.where(Xtr < 0, 0, Xtr)
    Xtr = np.where(Xtr >= xt, xt-1, Xtr)
    Ytr = np.where(Ytr < 0, 0, Ytr)
    Ytr = np.where(Ytr >= yt, yt-1, Ytr)
    Ztr = np.where(Ztr < 0, 0, Ztr)
    Ztr = np.where(Ztr >= zt, zt-1, Ztr)

    Xr = np.where(Xr < 0, 0, Xr)
    Xr = np.where(Xr >= xs, xs - 1, Xr)
    Yr = np.where(Yr < 0, 0, Yr)
    Yr = np.where(Yr >= ys, ys - 1, Yr)
    Zr = np.where(Zr < 0, 0, Zr)
    Zr = np.where(Zr >= zs, zs - 1, Zr)

    # extract the template and corresponding image sections - for checking on the results
    template_section_check = template[Xtr.astype('int'),Ytr.astype('int'),Ztr.astype('int')]    # check on this, see if it converted properly from matlab
    original_section = img[Xr.astype('int'),Yr.astype('int'),Zr.astype('int')]

    # organize the outputs
    section_mapping_coords = {'X':Xr,'Y':Yr,'Z':Zr,'Xt':Xtr,'Yt':Ytr,'Zt':Ztr}

    return section_mapping_coords, original_section    #, template_section
