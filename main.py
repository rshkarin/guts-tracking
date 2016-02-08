import numpy as np
import scipy.ndimage.interpolation as itp
import scipy.interpolate as interpolate
from scipy.ndimage import map_coordinates
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def interp3(x, y, z, v, xi, yi, zi, **kwargs):
    """Sample a 3D array "v" with pixel corner locations at "x","y","z" at the
    points in "xi", "yi", "zi" using linear interpolation. Additional kwargs
    are passed on to ``scipy.ndimage.map_coordinates``."""
    def index_coords(corner_locs, interp_locs):
        index = np.arange(len(corner_locs))
        if np.all(np.diff(corner_locs) < 0):
            corner_locs, index = corner_locs[::-1], index[::-1]
        return np.interp(interp_locs, corner_locs, index)

    orig_shape = np.asarray(xi).shape
    xi, yi, zi = np.atleast_1d(xi, yi, zi)
    for arr in [xi, yi, zi]:
        arr.shape = -1

    output = np.empty(xi.shape, dtype=float)
    coords = [index_coords(*item) for item in zip([x, y, z], [xi, yi, zi])]

    map_coordinates(v, coords, order=1, output=output, **kwargs)

    return output.reshape(orig_shape)

def test_data_slice2():
    data = np.memmap("E:\\guts_tracking\\brain_8bit_256x256x129.raw", dtype='uint8', shape=(129,256,256)).copy()
    #plt.imshow(data[60], cmap='gray')
    #plt.show()
    #return

    phi = 0 #z
    theta = 0 #y
    psi = 0 #x

    r = np.radians([phi, theta, psi])
    dims = data.shape

    p0 = np.matrix([round(dims[0]/2.), round(dims[1]/2.), round(dims[2]/2.)]).T #image center
    #p0 = np.matrix([50, round(dims[1]/2.), round(dims[2]/2.)]).T
    #p0 = np.matrix([63, round(dims[1]/2.), 1]).T
    #p0 = np.matrix([140, 100, 60]).T
    #p0 = np.matrix([71, 129, 103]).T
    #p0 = np.matrix([0, 0, 0]).T

    Rz = np.matrix([[1., 0., 0., 0.], \
                   [0., np.cos(r[0]), np.sin(r[0]), 0.], \
                   [0., -np.sin(r[0]), np.cos(r[0]), 0.], \
                   [0., 0., 0., 1.]])

    Ry = np.matrix([[np.cos(r[1]), 0., -np.sin(r[1]), 0.], \
                   [0., 1., 0., 0.], \
                   [np.sin(r[1]), 0., np.cos(r[1]), 0.], \
                   [0., 0., 0., 1.]])
    Rx = np.matrix([[np.cos(r[2]), np.sin(r[2]), 0., 0.], \
                   [-np.sin(r[2]), np.cos(r[2]), 0., 0.], \
                   [0., 0., 1., 0.], \
                   [0., 0., 0., 1.]])
    R = Rz*Ry*Rx

    #make affine matrix to rotate about center of image instead of origin
    T = (np.identity(3) - R[0:3,0:3]) * p0[0:3]
    A = R
    A[0:3,3] = T

    rot_old_to_new = A
    #rot_new_to_old = np.linalg.pinv(rot_old_to_new)
    rot_new_to_old = rot_old_to_new.I

    #this is the transformation
    #I assume you want a volume with the same dimensions as your old volume
    zv, yv, xv = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]))

    print zv
    print yv
    print xv

    #the coordinates you want to find a value for
    coordinates_axes_new = np.array([np.ravel(zv).T, np.ravel(yv).T, np.ravel(xv).T, np.ones(len(np.ravel(zv))).T])

    #the coordinates where you can find those values
    coordinates_axes_old = np.array(rot_new_to_old * coordinates_axes_new)
    z_coordinates = np.reshape(coordinates_axes_old[0,:], dims)
    y_coordinates = np.reshape(coordinates_axes_old[1,:], dims)
    x_coordinates = np.reshape(coordinates_axes_old[2,:], dims)

    #get the values for your new coordinates
    new_data = interp3(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), data, z_coordinates, y_coordinates, x_coordinates)

    #plt3d = plt.figure().gca(projection='3d')
    #plt3d.plot_surface(x_coordinates, y_coordinates, y_coordinates, cmap=cm.hot)

    print new_data.shape

    plt.imshow(new_data[60], cmap='gray')
    plt.show()

#normal order x, y, z
def test_data_slice():
    data = np.memmap("E:\\guts_tracking\\brain_8bit_256x256x129.raw", dtype='uint8', shape=(256,256,129), order='F').copy()
    #data = np.rot90(data[:,::-1,:])
    #plt.imshow(data[:,:,60])
    #plt.show()
    #return

    phi = 10 #x
    theta = 0 #y
    psi = 0 #z

    r = np.radians([phi, theta, psi])
    dims = data.shape

    #p0 = np.matrix([round(dims[0]/2.), round(dims[1]/2.), round(dims[2]/2.)]).T #image center
    #p0 = np.matrix([50, round(dims[1]/2.), round(dims[2]/2.)]).T
    #p0 = np.matrix([63, round(dims[1]/2.), 1]).T
    #p0 = np.matrix([140, 100, 60]).T
    p0 = np.matrix([103, 129, 71]).T

    Rx = np.matrix([[1., 0., 0., 0.], \
                   [0., np.cos(r[0]), -np.sin(r[0]), 0.], \
                   [0., np.sin(r[0]), np.cos(r[0]), 0.], \
                   [0., 0., 0., 1.]])

    Ry = np.matrix([[np.cos(r[1]), 0., np.sin(r[1]), 0.], \
                   [0., 1., 0., 0.], \
                   [-np.sin(r[1]), 0., np.cos(r[1]), 0.], \
                   [0., 0., 0., 1.]])
    Rz = np.matrix([[np.cos(r[2]), -np.sin(r[2]), 0., 0.], \
                   [np.sin(r[2]), np.cos(r[2]), 0., 0.], \
                   [0., 0., 1., 0.], \
                   [0., 0., 0., 1.]])
    R = Rx*Ry*Rz

    #make affine matrix to rotate about center of image instead of origin
    T = (np.identity(3) - R[0:3,0:3]) * p0[0:3]
    A = R
    A[0:3,3] = T

    rot_old_to_new = A
    #rot_new_to_old = np.linalg.pinv(rot_old_to_new)
    rot_new_to_old = rot_old_to_new.I

    #this is the transformation
    #I assume you want a volume with the same dimensions as your old volume
    xv, yv, zv = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]))

    #the coordinates you want to find a value for
    coordinates_axes_new = np.array([np.ravel(xv).T, np.ravel(yv).T, np.ravel(zv).T, np.ones(len(np.ravel(zv))).T])

    #the coordinates where you can find those values
    coordinates_axes_old = np.array(rot_new_to_old * coordinates_axes_new)
    x_coordinates = np.reshape(coordinates_axes_old[0,:], dims)
    y_coordinates = np.reshape(coordinates_axes_old[1,:], dims)
    z_coordinates = np.reshape(coordinates_axes_old[2,:], dims)

    #get the values for your new coordinates
    new_data = interp3(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), data, x_coordinates, y_coordinates, z_coordinates)

    plt.imshow(new_data[:,:,100], cmap='gray')
    plt.show()

if __name__ == "__main__":
    #test_data_slice()
    test_data_slice2()
