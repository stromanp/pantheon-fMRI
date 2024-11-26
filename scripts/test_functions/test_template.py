import load_templates
import matplotlib.pyplot as plt
import numpy as np


resolution = 1
templatename = 'ccbs'
template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_img = load_templates.load_template_and_masks(templatename, resolution)
xs,ys,zs = np.shape(template_img)
x0 = np.floor(xs/2).astype(int)

fig = plt.figure(10)
plt.imshow(template_img[x0,:,:],'gray')