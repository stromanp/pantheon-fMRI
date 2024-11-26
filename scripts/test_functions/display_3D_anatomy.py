# display anatomy etc in 3D with animations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import copy


targetnum = 'C6RD'
clusterdataname = r'E:/SAPMresults2_Oct2022\\Pain_cluster_def.npy'
templatename = 'ccbs'

clusterdata = np.load(clusterdataname, allow_pickle=True).flat[0]
cluster_properties = clusterdata['cluster_properties']
nregions = len(cluster_properties)
nclusterlist = [cluster_properties[i]['nclusters'] for i in range(nregions)]
rnamelist = [cluster_properties[i]['rname'] for i in range(nregions)]
nclusterstotal = np.sum(nclusterlist)

if type(targetnum) == int:
    r = targetnum
else:
    # assume "targetnum" input is a region name
    r = rnamelist.index(targetnum)

IDX = clusterdata['cluster_properties'][r]['IDX']
nclusters = nclusterlist[r]

targetcluster = 0
idxx = np.where(IDX == targetcluster)
cx = clusterdata['cluster_properties'][r]['cx'][idxx]
cy = clusterdata['cluster_properties'][r]['cy'][idxx]
cz = clusterdata['cluster_properties'][r]['cz'][idxx]

# load template
if templatename == 'brain':
    resolution = 2
else:
    resolution = 1
template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_img = load_templates.load_template_and_masks(
    templatename, resolution)

window_number = 99
plt.close(window_number)
fig = plt.figure(window_number)
ax = fig.add_subplot(projection='3d')
# ax = Axes3D(fig)
# ax.axis('off')
# plt.rcParams["figure.figsize"] = [7.00, 3.50]
# plt.rcParams["figure.autolayout"] = True


voxelarray = copy.deepcopy(roi_map)
# combine the objects into a single boolean array
# set the colors of each object
colors = np.empty(roi_map.shape, dtype=object)
colors[voxelarray] = 'red'
# colors[cube1] = 'blue'
# colors[cube2] = 'green'

ax.voxels(voxelarray, facecolors=colors, edgecolor='k', alpha = 0.2)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Make data
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 10 * np.outer(np.cos(u), np.sin(v))
y = 10 * np.outer(np.sin(u), np.sin(v))
z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface
ax.plot_surface(x, y, z)

# Set an equal aspect ratio
ax.set_aspect('equal')


plt.show()


def animate(num, data, line):
   colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',   '#7f7f7f', '#bcbd22', '#17becf']
   line.set_color(colors[num % len(colors)])
   line.set_alpha(0.7)
   line.set_data(data[0:2, :num])
   line.set_3d_properties(data[2, :num])
   return line

t = np.arange(0, 20, 0.2)
x = np.cos(t) - 1
y = 1 / 2 * (np.cos(2 * t) - 1)
data = np.array([x, y, t])
N = len(t)
fig = plt.figure()
ax = Axes3D(fig)
ax.axis('off')

line, = plt.plot(data[0], data[1], data[2], lw=7, c='red')
line_ani = animation.FuncAnimation(fig, animate, frames=N, fargs=(data, line), interval=50, blit=False)

plt.show()



# outputimg = pydisplay.pydisplayvoxelregionslice(templatename, template_img, cx, cy, cz, orientation, displayslice=[],
#                                                 colorlist=regioncolor)
# imgname = 'undefined'
# if write_output:
#     p, f = os.path.split(clusterdataname)
#     imgname = os.path.join(p, 'cluster_{}_{}_{}.png'.format(targetnum, targetcluster, orientation[:3]))
#     matplotlib.image.imsave(imgname, outputimg)


def transferFunction(x):
    """Transfer Function returns r,g,b,a values as a function of density x"""

    r = 1.0 * np.exp(-(x - 9.0) ** 2 / 1.0) + 0.1 * np.exp(-(x - 3.0) ** 2 / 0.1) + 0.1 * np.exp(-(x - -3.0) ** 2 / 0.5)
    g = 1.0 * np.exp(-(x - 9.0) ** 2 / 1.0) + 1.0 * np.exp(-(x - 3.0) ** 2 / 0.1) + 0.1 * np.exp(-(x - -3.0) ** 2 / 0.5)
    b = 0.1 * np.exp(-(x - 9.0) ** 2 / 1.0) + 0.1 * np.exp(-(x - 3.0) ** 2 / 0.1) + 1.0 * np.exp(-(x - -3.0) ** 2 / 0.5)
    a = 0.6 * np.exp(-(x - 9.0) ** 2 / 1.0) + 0.1 * np.exp(-(x - 3.0) ** 2 / 0.1) + 0.01 * np.exp(
        -(x - -3.0) ** 2 / 0.5)

    return r, g, b, a



# Do Volume Rendering
image = np.zeros((camera_grid.shape[1],camera_grid.shape[2],3))

for dataslice in camera_grid:
	r,g,b,a = transferFunction(np.log(dataslice))
	image[:,:,0] = a*r + (1-a)*image[:,:,0]
	image[:,:,1] = a*g + (1-a)*image[:,:,1]
	image[:,:,2] = a*b + (1-a)*image[:,:,2]



# Load the Original Datacube
f = h5.File('datacube.hdf5', 'r')
datacube = np.array(f['density'])

# Construct the Corresponding Datacube Grid Coordinates
Nx, Ny, Nz = datacube.shape
x = np.linspace(-Nx/2, Nx/2, Nx)
y = np.linspace(-Ny/2, Ny/2, Ny)
z = np.linspace(-Nz/2, Nz/2, Nz)
points = (x, y, z)

# Construct the Camera Grid / Query Points -- rotate camera view
angle = np.pi/3.0  # my viewing angle
N = 180  # camera grid resolution
c = np.linspace(-N/2, N/2, N)
qx, qy, qz = np.meshgrid(c,c,c)  # query points
# apply rotation
qxR = qx
qyR = qy * np.cos(angle) - qz * np.sin(angle)
qzR = qy * np.sin(angle) + qz * np.cos(angle)
qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T

# Interpolate onto Camera Grid
camera_grid = interpn(points, datacube, qi, method='linear').reshape((N,N,N))




#--------------from GitHub--------------------------------
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy.interpolate import interpn

"""
Create Your Own Volume Rendering (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz
Simulate the Schrodinger-Poisson system with the Spectral method
"""


def transferFunction(x):
    r = 1.0 * np.exp(-(x - 9.0) ** 2 / 1.0) + 0.1 * np.exp(-(x - 3.0) ** 2 / 0.1) + 0.1 * np.exp(-(x - -3.0) ** 2 / 0.5)
    g = 1.0 * np.exp(-(x - 9.0) ** 2 / 1.0) + 1.0 * np.exp(-(x - 3.0) ** 2 / 0.1) + 0.1 * np.exp(-(x - -3.0) ** 2 / 0.5)
    b = 0.1 * np.exp(-(x - 9.0) ** 2 / 1.0) + 0.1 * np.exp(-(x - 3.0) ** 2 / 0.1) + 1.0 * np.exp(-(x - -3.0) ** 2 / 0.5)
    a = 0.6 * np.exp(-(x - 9.0) ** 2 / 1.0) + 0.1 * np.exp(-(x - 3.0) ** 2 / 0.1) + 0.01 * np.exp(
        -(x - -3.0) ** 2 / 0.5)

    return r, g, b, a


def main():
    """ Volume Rendering """

    # Load Datacube
    f = h5.File('datacube.hdf5', 'r')
    datacube = np.array(f['density'])

    # Datacube Grid
    Nx, Ny, Nz = datacube.shape
    x = np.linspace(-Nx / 2, Nx / 2, Nx)
    y = np.linspace(-Ny / 2, Ny / 2, Ny)
    z = np.linspace(-Nz / 2, Nz / 2, Nz)
    points = (x, y, z)

    # Do Volume Rendering at Different Veiwing Angles
    Nangles = 10
    for i in range(Nangles):

        print('Rendering Scene ' + str(i + 1) + ' of ' + str(Nangles) + '.\n')

        # Camera Grid / Query Points -- rotate camera view
        angle = np.pi / 2 * i / Nangles
        N = 180
        c = np.linspace(-N / 2, N / 2, N)
        qx, qy, qz = np.meshgrid(c, c, c)
        qxR = qx
        qyR = qy * np.cos(angle) - qz * np.sin(angle)
        qzR = qy * np.sin(angle) + qz * np.cos(angle)
        qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T

        # Interpolate onto Camera Grid
        camera_grid = interpn(points, datacube, qi, method='linear').reshape((N, N, N))

        # Do Volume Rendering
        image = np.zeros((camera_grid.shape[1], camera_grid.shape[2], 3))

        for dataslice in camera_grid:
            r, g, b, a = transferFunction(np.log(dataslice))
            image[:, :, 0] = a * r + (1 - a) * image[:, :, 0]
            image[:, :, 1] = a * g + (1 - a) * image[:, :, 1]
            image[:, :, 2] = a * b + (1 - a) * image[:, :, 2]

        image = np.clip(image, 0.0, 1.0)

        # Plot Volume Rendering
        plt.figure(figsize=(4, 4), dpi=80)

        plt.imshow(image)
        plt.axis('off')

        # Save figure
        plt.savefig('volumerender' + str(i) + '.png', dpi=240, bbox_inches='tight', pad_inches=0)

    # Plot Simple Projection -- for Comparison
    plt.figure(figsize=(4, 4), dpi=80)

    plt.imshow(np.log(np.mean(datacube, 0)), cmap='viridis')
    plt.clim(-5, 5)
    plt.axis('off')

    # Save figure
    plt.savefig('projection.png', dpi=240, bbox_inches='tight', pad_inches=0)
    plt.show()

    return 0


if __name__ == "__main__":
    main()

