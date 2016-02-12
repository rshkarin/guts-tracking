import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import distance
import scipy.ndimage.interpolation as itp
import scipy.interpolate as interpolate
from scipy.ndimage import map_coordinates
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import skimage as ski
from skimage import measure
from skimage import filters
from skimage import feature
from skimage.transform import hough_ellipse, hough_circle
from skimage.draw import circle_perimeter, ellipse_perimeter, circle
from skimage.morphology import disk, watershed

import profile
import pstats
import time
from functools import wraps

def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" %
               (function.func_name, str(t1-t0))
               )
        return result
    return function_timer

@fn_timer
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

    map_coordinates(v, coords, output=output, **kwargs)

    return output.reshape(orig_shape)

def get_oblique_slice(phi_z=0, theta_y=0, psi_x=0, slice_idx=65):
    data = np.memmap("E:\\guts_tracking\\brain_8bit_256x256x129.raw", dtype='uint8', shape=(129,256,256)).copy()
    #plt.imshow(data[60], cmap='gray')
    #plt.show()
    #return

    #phi = 0 #z
    #theta = 15 #y
    #psi = 0 #x

    phi, theta, psi = phi_z, theta_y, psi_x

    r = np.radians([phi, theta, psi])
    dims = data.shape

    p0 = np.matrix([round(dims[0]/2.), round(dims[1]/2.), round(dims[2]/2.)]).T #image center
    #p0 = np.matrix([50, round(dims[1]/2.), round(dims[2]/2.)]).T
    #p0 = np.matrix([63, round(dims[1]/2.), 1]).T
    #p0 = np.matrix([140, 100, 60]).T
    #p0 = np.matrix([65, 115, 140]).T
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
    zv, yv, xv = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing='ij')

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

    #plt.imshow(new_data[65], cmap='gray')
    #plt.show()
    return new_data[slice_idx]

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

def plot_obique_slices():
    phi = np.linspace(-90,90,10)
    theta = np.linspace(-90,90,10)
    psi = np.linspace(-90,90,10)

    fig, axes = plt.subplots(10, 10)

    for i, theta_val in enumerate(theta):
        #for j, psi_val in enumerate(psi):
        for j, phi_val in enumerate(phi):
            #axes[i,j].imshow(get_oblique_slice(theta_y=theta_val, psi_x=psi_val), cmap='gray')
            axes[i,j].imshow(get_oblique_slice(phi_z=phi_val, theta_y=theta_val), cmap='gray')
            axes[i,j].get_xaxis().set_visible(False)
            axes[i,j].get_yaxis().set_visible(False)

    plt.show()

@fn_timer
def get_guts_oblique_slice(phi_z=0, theta_y=0, psi_x=0, slice_idx=125, rot_p0=None, order=1):
    data = np.memmap("E:\\guts_tracking\\data\\fish202_aligned_masked_8bit_150x200x440.raw", dtype='uint8', shape=(440,200,150)).copy()
    #plt.imshow(data[slice_idx], cmap='gray')
    #plt.show()
    #return

    phi, theta, psi = phi_z, theta_y, psi_x

    r = np.radians([phi, theta, psi])
    dims = data.shape

    if not rot_p0:
        rot_p0 = [round(dims[0]/2.), round(dims[1]/2.), round(dims[2]/2.)]

    p0 = np.matrix(rot_p0).T #image center

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
    rot_new_to_old = rot_old_to_new.I

    #this is the transformation
    #I assume you want a volume with the same dimensions as your old volume
    zv, yv, xv = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing='ij')

    #the coordinates you want to find a value for
    coordinates_axes_new = np.array([np.ravel(zv).T, np.ravel(yv).T, np.ravel(xv).T, np.ones(len(np.ravel(zv))).T])

    #the coordinates where you can find those values
    coordinates_axes_old = np.array(rot_new_to_old * coordinates_axes_new)
    z_coordinates = np.reshape(coordinates_axes_old[0,:], dims)
    y_coordinates = np.reshape(coordinates_axes_old[1,:], dims)
    x_coordinates = np.reshape(coordinates_axes_old[2,:], dims)

    #get the values for your new coordinates
    new_data = interp3(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), data, z_coordinates, y_coordinates, x_coordinates, order=order)

    #plot data
    plot_comparative(data, new_data, slice_idx=slice_idx, phi=phi, theta=theta, psi=psi)

def plot_comparative(data, new_data, slice_idx=125, phi=0, theta=0, psi=0):
    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(data[slice_idx], cmap='gray')
    axes[0].set_title('Original')

    axes[1].imshow(new_data[slice_idx], cmap='gray')
    axes[1].set_title('Rz = %d, Ry = %d, Rx = %d' % (phi, theta, psi))

    for ax in axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

def _calc_circularity(area, perimeter):
    return 4.0 * np.pi * area / (perimeter * perimeter)

def preprocess_data(data):
    return data

def segment_data(processed_data):
    th_val = filters.threshold_otsu(processed_data)
    thresholded_parts = processed_data <= th_val
    labeled_data, num_features = ndi.measurements.label(thresholded_parts)
    return labeled_data, num_features

def slice_stats(segmented_data, slice_idx=-1):
    properties = ['label','area','centroid','equivalent_diameter', \
                'major_axis_length','minor_axis_length','orientation','bbox','perimeter']
    extra_props = ['circularity','slice_idx']

    u_labeled_data = np.unique(segmented_data)
    labeled_data = np.searchsorted(u_labeled_data, segmented_data)

    stats = pd.DataFrame(columns=properties)

    for region in measure.regionprops(labeled_data):
        stats = stats.append({_property: region[_property] for _property in properties}, \
                                ignore_index=True)

    for prop in extra_props:
        if prop == 'circularity':
            stats[prop] = stats.apply(lambda row: 0.0 if row['perimeter'] == 0 \
                else _calc_circularity(row['area'], row['perimeter']), axis=1)

        if prop == 'slice_idx':
            stats[prop] = slice_idx

    return stats

def get_ellipses(data):
    edges = feature.canny(data, sigma=3.0, low_threshold=0.55, high_threshold=0.8)
    result = hough_ellipse(edges, threshold=4, accuracy=5, min_size=20, max_size=300)
    result.sort(order='accumulator')
    return result

def get_circles(data):
    edges = feature.canny(data, sigma=3.0, low_threshold=0.55, high_threshold=0.8)
    hough_radii = np.arange(15, 30, 2)
    result = hough_circle(edges, hough_radii)
    result.sort(order='accumulator')
    return result

# def draw_esllipse(data, ellipse):
#     if ellipse:
#         image = ski.color.gray2rgb(data)
#
#         #best = list(ellipse[-1])
#         best = ellipse
#         yc, xc, a, b = [int(round(x)) for x in best[1:5]]
#         orientation = best[5]
#
#         cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
#         image[cy, cx] = (220, 20, 20)
#
#         rr, cc = circle(yc, xc, 2)
#         data[rr, cc] = (220, 20, 20)
#     else:
#         return np.zeros(data.shape)
#
#     return data

# def get_nearest_ellipse(ellipses_props, gathered_ellipses):
#     if not len(gathered_ellipses):
#         return list(ellipses_props[-1])
#
#     prev_ellipse = gathered_ellipses[-1]
#     nearest_ellipse_to_prev, distance = None, 0
#
#     def get_distance(ellipse, prev_ellipse):
#         return np.abs(prev_ellipse[1] - ellipse[1]) + np.abs(prev_ellipse[2] - ellipse[2])
#
#     for ellipse in ellipses_props:
#         if not nearest_ellipse_to_prev:
#             nearest_ellipse_to_prev = ellipse
#             distance = get_distance(ellipse, prev_ellipse)
#             continue
#
#         new_distance = get_distance(ellipse, prev_ellipse)
#
#         if new_distance <= distance:
#             distance = new_distance
#             nearest_ellipse_to_prev = ellipse
#
#     return nearest_ellipse_to_prev

def rotate_volume(data, angles, order=3):
    phi, theta, psi = angles

    r = np.radians([phi, theta, psi])
    dims = data.shape

    rot_p0 = [round(dims[0]/2.), round(dims[1]/2.), round(dims[2]/2.)]

    p0 = np.matrix(rot_p0).T #image center

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
    rot_new_to_old = rot_old_to_new.I

    #this is the transformation
    #I assume you want a volume with the same dimensions as your old volume
    zv, yv, xv = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing='ij')

    #the coordinates you want to find a value for
    coordinates_axes_new = np.array([np.ravel(zv).T, np.ravel(yv).T, np.ravel(xv).T, np.ones(len(np.ravel(zv))).T])

    #the coordinates where you can find those values
    coordinates_axes_old = np.array(rot_new_to_old * coordinates_axes_new)
    z_coordinates = np.reshape(coordinates_axes_old[0,:], dims)
    y_coordinates = np.reshape(coordinates_axes_old[1,:], dims)
    x_coordinates = np.reshape(coordinates_axes_old[2,:], dims)

    #get the values for your new coordinates
    new_data = interp3(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), data, z_coordinates, y_coordinates, x_coordinates, order=order)

    return new_data

def get_init_ellipses(data):
    stats = pd.DataFrame()

    for idx,_slice in enumerate(data):
        if np.count_nonzero(_slice):
            #segemnt frame
            labeled_data, num_labels = segment_data(_slice)

            #remove all big and non-circualr labels
            stats = slice_stats(labeled_data, slice_idx=idx)
            stats = stats[(stats.area > 20) & ((stats.major_axis_length < _slice.shape[0]) | (stats.major_axis_length < _slice.shape[1]))]
            stats = stats[stats.circularity > 0.5]

            if not stats.size:
                continue

            break

    return stats

def get_nearest_ellipse(ellipses_stats, gathered_ellipses, tolerance=50.0):
    if not gathered_ellipses.size:
        raise ValueError('There are no initial ellipses.')

    if not ellipses_stats.size:
        raise ValueError('There are no detected ellipses.')

    def get_distance(ellipse, prev_ellipse):
        return np.hypot(ellipse.centroid[0] - prev_ellipse.centroid[0], \
                        ellipse.centroid[1] - prev_ellipse.centroid[1])

    #get last ellipse
    prev_ellipse = gathered_ellipses.iloc[-1]
    nearest_ellipse_to_prev, dist = pd.Series(), 0

    for index, ellipse in ellipses_stats.iterrows():
        if nearest_ellipse_to_prev.empty:
            nearest_ellipse_to_prev = ellipse
            dist = get_distance(ellipse, prev_ellipse)
            continue

        new_dist = get_distance(ellipse, prev_ellipse)

        if new_dist <= dist:
            dist = new_dist
            nearest_ellipse_to_prev = ellipse

    if get_distance(ellipse, prev_ellipse) > tolerance:
        #raise ValueError('The detected ellipse is too far away.')
        print 'The detected ellipse is too far away.'
        return pd.Series(), 1e10

    return nearest_ellipse_to_prev, dist


def get_arbitrary_slice(data, slice_idx, phi_z=0, theta_y=0, psi_x=0, rot_p0=None, order=1):
    phi, theta, psi = phi_z, theta_y, psi_x

    r = np.radians([phi, theta, psi])
    dims = data.shape

    if not rot_p0.any():
        rot_p0 = [round(dims[0]/2.), round(dims[1]/2.), round(dims[2]/2.)]

    p0 = np.matrix(rot_p0).T #image center

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
    rot_new_to_old = rot_old_to_new.I

    #this is the transformation
    #I assume you want a volume with the same dimensions as your old volume
    zv, yv, xv = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing='ij')

    #the coordinates you want to find a value for
    coordinates_axes_new = np.array([np.ravel(zv).T, np.ravel(yv).T, np.ravel(xv).T, np.ones(len(np.ravel(zv))).T])

    #the coordinates where you can find those values
    coordinates_axes_old = np.array(rot_new_to_old * coordinates_axes_new)
    z_coordinates = np.reshape(coordinates_axes_old[0,:], dims)
    y_coordinates = np.reshape(coordinates_axes_old[1,:], dims)
    x_coordinates = np.reshape(coordinates_axes_old[2,:], dims)

    #get the values for your new coordinates
    new_data = interp3(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), data, z_coordinates, y_coordinates, x_coordinates, order=order)

    return new_data[slice_idx]



def predict_ellipse(data, gathered_ellipses, slice_idx):
    prev_ellipse = gathered_ellipses.iloc[-1]

    #rotation point
    rot_point = np.array([prev_ellipse.slice_idx, prev_ellipse.centroid[1], prev_ellipse.centroid[0]])

    #directions
    x_angles, y_angles, z_angles = np.linspace(-15,15,3), np.linspace(-15,15,3), np.linspace(-15,15,3)

    #storage of candidates
    res_shape = tuple([len(x_angles), len(y_angles), len(z_angles)])
    slice_shape = data[0].shape

    results_ellipses = np.empty(len(x_angles) * len(y_angles) * len(z_angles), dtype=object)
    results_distances = np.empty(len(x_angles) * len(y_angles) * len(z_angles))
    results_slices = np.zeros((len(x_angles) * len(y_angles) * len(z_angles), slice_shape[0], slice_shape[1]))

    results_ellipses.fill(np.nan)
    results_distances.fill(10e9)

    results_ellipses = results_ellipses.reshape(res_shape)
    results_distances = results_distances.reshape(res_shape)

    def ravel_index(x, dims):
        i = 0
        for dim, j in zip(dims, x):
            i *= dim
            i += j
        return i

    for i,x_deg in enumerate(x_angles):
        for j,y_deg in enumerate(y_angles):
            for k,z_deg in enumerate(z_angles):
                oblique_slice = get_arbitrary_slice(data, slice_idx=slice_idx, phi_z=z_deg, \
                                    theta_y=y_deg, psi_x=x_deg, rot_p0=rot_point)

                slice_data = preprocess_data(oblique_slice)

                if np.count_nonzero(slice_data):
                    labeled_data, num_features = segment_data(slice_data)

                    stats = slice_stats(labeled_data, slice_idx=slice_idx)
                    stats = stats[(stats.area > 20) & ((stats.major_axis_length < slice_data.shape[0]) | (stats.major_axis_length < slice_data.shape[1]))]
                    stats = stats[stats.circularity > 0.4]

                    if stats.size:
                        nearest_ellipse, distance = get_nearest_ellipse(stats, gathered_ellipses)

                        print nearest_ellipse

                        results_ellipses[i,j,k] = nearest_ellipse
                        results_distances[i,j,k] = distance
                        results_slices[ravel_index((k, j, i), res_shape)] = oblique_slice

    results_ellipses = results_ellipses.ravel()
    results_distances = results_distances.ravel()

    print 'Distances and ellipses:'
    print results_distances
    print results_ellipses

    min_dist_idx = results_distances.argmin(axis=0)
    nearest_ellipse = results_ellipses[min_dist_idx]
    output_slice = results_slices[min_dist_idx]

    return nearest_ellipse, output_slice

def segment_guts():
    data = np.memmap("E:\\guts_tracking\\data\\fish202_aligned_masked_8bit_150x200x440.raw", dtype='uint8', shape=(440,200,150)).copy()

    #ellipse_data = get_ellipses(data[15])
    #print ellipse_data
    #new_slice = draw_esllipse(data[15], ellipse_data)

    #fig, (ax1, ax2) = plt.subplots(1, 2)

    #ax1.imshow(new_slice, cmap='gray')
    #ax2.imshow(data[15], cmap='gray')
    #plt.show()
    #return

    # slice_data = preprocess_data(data[12])
    # seg_data = segment_data(slice_data)
    # stats = slice_stats(seg_data)
    #
    # plt.imshow(seg_data, cmap='gray')
    # plt.show()
    # return


    fig = plt.figure()
    im = plt.imshow(data[10], animated=True, cmap='gray')

    gathered_ellipses = []

    for i,_slice in enumerate(data):
        plt.title('Frame %d' % i)

        ellipses_props = get_ellipses(_slice)

        new_slice = _slice

        if len(ellipses_props):
            nearest_ellipse = get_nearest_ellipse(ellipses_props, gathered_ellipses)
            print nearest_ellipse

            gathered_ellipses.append(nearest_ellipse)

            new_slice = draw_esllipse(_slice, nearest_ellipse)

        im.set_data(new_slice)

        plt.draw()



    # def init():
    #     im.set_data(np.zeros(data[0].shape))
    #     return im,
    #
    # def animate(i):
    #     plt.title('Frame %d' % i)
    #
    #     ellipses_props = get_ellipses(data[i])
    #
    #     new_slice = data[i]
    #
    #     if len(ellipses_props):
    #         nearest_ellipse = get_nearest_ellipse(ellipses_props, gathered_ellipses)
    #         print nearest_ellipse
    #
    #         gathered_ellipses.append(nearest_ellipse)
    #
    #         new_slice = draw_esllipse(data[i], nearest_ellipse)
    #
    #     im.set_data(new_slice)
    #
    #     return im,

        #slice_data = preprocess_data(data[i])
        #seg_data = segment_data(slice_data)
        #stats = slice_stats(seg_data)

    #anim = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=100)
    #anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()
    #start from the tail
    #for i,_slice in enumerate(data):
        #yc, xc, a, b, orientation = get_ellipses(_slice)
        #new_slice = draw_esllipse(_slice, yc, xc, a, b, orientation)

def test_canny():
    data = np.memmap("E:\\guts_tracking\\data\\fish202_aligned_masked_8bit_150x200x440.raw", dtype='uint8', shape=(440,200,150)).copy()

    data_slice = data[150]

    sigmas = np.linspace(1,5,5)
    low_thresholds = np.linspace(0.1,0.55,5)

    thresholds = np.linspace(5,10,5)
    accuracies = np.linspace(5,25,5)

    edges = feature.canny(data_slice, sigma=3.0, low_threshold=0.4, high_threshold=0.8)
    ellipses = hough_ellipse(edges, threshold=4, accuracy=1, min_size=15, max_size=300)
    print ellipses
    ellipses.sort(order='accumulator')
    new_slice = draw_esllipse(edges, ellipses)
    plt.imshow(new_slice, cmap='gray')


    #fig, axes = plt.subplots(5, 5)

    #for i,sigma in enumerate(sigmas):
        #for j,low_threshold in enumerate(low_thresholds):
    # for i,threshold in enumerate(thresholds):
    #     for j,accuracy in enumerate(accuracies):
    #         edges = feature.canny(data_slice, sigma=3.0, low_threshold=0.4, high_threshold=0.8)
    #         ellipses = hough_ellipse(edges, threshold=threshold, accuracy=accuracy, min_size=15, max_size=300)
    #         ellipses.sort(order='accumulator')
    #         new_slice = draw_esllipse(data_slice, ellipses)
    #
    #         axes[i,j].imshow(new_slice, cmap='gray')
    #         #axes[i,j].set_title('sigma=%f, low_th=%f' % (sigma, low_threshold))
    #         axes[i,j].set_title('threshold=%f, accuracy=%f' % (threshold, accuracy))
    #         axes[i,j].get_xaxis().set_visible(False)
    #         axes[i,j].get_yaxis().set_visible(False)

    plt.show()

def test_circles():
    data = np.memmap("E:\\guts_tracking\\data\\fish202_aligned_masked_8bit_150x200x440.raw", dtype='uint8', shape=(440,200,150)).copy()

    fig = plt.figure()
    im = plt.imshow(data[10], animated=True, cmap='gray')

    def init():
        im.set_data(np.zeros(data[0].shape))
        return im,

    def animate(i):
        print 'Frame %d' % i
        plt.title('Frame %d' % i)

        image = data[i]

        hough_radii = np.arange(10, 100, 10)
        edges = feature.canny(data[i], sigma=3.0, low_threshold=0.4, high_threshold=0.8)
        hough_res = hough_circle(edges, hough_radii)

        centers = []
        accums = []
        radii = []

        for radius, h in zip(hough_radii, hough_res):
            peaks = feature.peak_local_max(h)
            centers.extend(peaks)
            accums.extend(h[peaks[:, 0], peaks[:, 1]])
            radii.extend([radius] * len(peaks))

        image = ski.color.gray2rgb(data[i])

        for idx in np.argsort(accums)[::-1][:5]:
            center_x, center_y = centers[idx]
            radius = radii[idx]
            cx, cy = circle_perimeter(center_y, center_x, radius)

            if max(cx) < 150 and max(cy) < 200:
                image[cy, cx] = (220, 20, 20)

        im.set_data(image)

        return im,

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=100)

    plt.show()

def test_circles2():
    data = np.memmap("E:\\guts_tracking\\data\\fish202_aligned_masked_8bit_150x200x440.raw", dtype='uint8', shape=(440,200,150)).copy()

    i = 157

    hough_radii = np.arange(10, 100, 10)
    edges = feature.canny(data[i], sigma=3.0, low_threshold=0.4, high_threshold=0.8)
    hough_res = hough_circle(edges, hough_radii)

    centers = []
    accums = []
    radii = []

    for radius, h in zip(hough_radii, hough_res):
        peaks = feature.peak_local_max(h)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius] * len(peaks))

    image = ski.color.gray2rgb(data[i])

    for idx in np.argsort(accums)[::-1][:5]:
        center_x, center_y = centers[idx]
        radius = radii[idx]
        cx, cy = circle_perimeter(center_y, center_x, radius)

        if max(cx) < 150 and max(cy) < 200:
            image[cy, cx] = (220, 20, 20)

    plt.imshow(image, cmap='gray')

    plt.show()

def test_detect_by_stats():
    data = np.memmap("E:\\guts_tracking\\data\\fish202_aligned_masked_8bit_150x200x440.raw", dtype='uint8', shape=(440,200,150)).copy()

    data_slice = data[68]

    slice_data = preprocess_data(data_slice)
    labeled_data, num_features = segment_data(slice_data)

    #plt.imshow(labeled_data == 4, cmap='gray')
    #plt.show()

    stats = slice_stats(labeled_data)

    #stats = stats[(stats.area > 20) & (stats.area < np.pi * min(data_slice.shape)**2)]
    #stats = stats[(stats.area > 20) & (stats.area < 20000)]
    stats = stats[(stats.area > 2) & (stats.area < 20000)]
    #print stats
    #stats = stats[stats.circularity > 0.2]

    image = data_slice

    for index, row in stats.iterrows():
        print row
        yc, xc = [int(round(x)) for x in row.centroid]
        orientation = row.orientation
        major_axis = int(round(row.major_axis_length/2.))
        minor_axis = int(round(row.minor_axis_length/2.))

        image = ski.color.gray2rgb(image)

        cy, cx = ellipse_perimeter(yc, xc, minor_axis, major_axis, -orientation)
        image[cy, cx] = (220, 20, 20)

        rr, cc = circle(yc, xc, 2)
        image[rr, cc] = (220, 20, 20)

    plt.imshow(image, cmap='gray')
    plt.show()

    print stats

def test_detect_by_stats2():
    data = np.memmap("E:\\guts_tracking\\data\\fish202_aligned_masked_8bit_150x200x440.raw", dtype='uint8', shape=(440,200,150)).copy()

    fig = plt.figure()
    im = plt.imshow(data[10], animated=True, cmap='gray')

    frame_shape = data[0].shape

    def init():
        im.set_data(np.zeros(data[0].shape))
        return im,

    def animate(i):
        plt.title('Frame %d' % i)

        slice_data = preprocess_data(data[i + 100])

        if np.count_nonzero(slice_data):
            labeled_data, num_features = segment_data(slice_data)

            stats = slice_stats(labeled_data)
            stats = stats[(stats.area > 20) & ((stats.major_axis_length < frame_shape[0]) | (stats.major_axis_length < frame_shape[1]))]
            stats = stats[stats.circularity > 0.5]

            for index, row in stats.iterrows():
                print 'Frame# %d, Circle# %d [circularity = %f]' % (i, row.label, row.circularity)

                yc, xc = [int(round(x)) for x in row.centroid]
                orientation = row.orientation
                major_axis = int(round(row.major_axis_length/2.))
                minor_axis = int(round(row.minor_axis_length/2.))

                slice_data = ski.color.gray2rgb(slice_data)

                cy, cx = ellipse_perimeter(yc, xc, minor_axis, major_axis, orientation)
                slice_data[cy, cx] = (220, 20, 20)

                rr, cc = circle(yc, xc, 2)
                slice_data[rr, cc] = (220, 20, 20)

        im.set_data(slice_data)

        return im,

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=100)
    plt.show()

def draw_ellipses(slice_data, ellipse, color=(220, 20, 20)):
    yc, xc = [int(round(x)) for x in ellipse.centroid]
    orientation = ellipse.orientation
    major_axis = int(round(ellipse.major_axis_length/2.))
    minor_axis = int(round(ellipse.minor_axis_length/2.))

    image = ski.color.gray2rgb(slice_data)

    cy, cx = ellipse_perimeter(yc, xc, minor_axis, major_axis, -orientation)
    image[cy, cx] = color

    rr, cc = circle(yc, xc, 2)
    image[rr, cc] = color

    return image

def track_guts_animation(data, inital_ellipse):
    fig = plt.figure()
    im = plt.imshow(data[inital_ellipse.slice_idx], animated=True, cmap='gray')

    frame_shape = data[inital_ellipse.slice_idx].shape

    global gathered_ellipses

    def init():
        global gathered_ellipses
        gathered_ellipses = pd.DataFrame()
        gathered_ellipses = gathered_ellipses.append(inital_ellipse, ignore_index=True)

        im.set_data(np.zeros(data[inital_ellipse.slice_idx].shape))
        return im,

    def animate(i):
        global gathered_ellipses

        index = i + inital_ellipse.slice_idx + 1

        plt.title('Frame %d' % index)

        slice_data = preprocess_data(data[index])

        print 'FRAME #%d' % index

        if np.count_nonzero(slice_data):
            #segemnt frame
            labeled_data, num_features = segment_data(slice_data)

            #remove all big and non-circualr labels
            stats = slice_stats(labeled_data, slice_idx=index)
            stats = stats[(stats.area > 20) & ((stats.major_axis_length < frame_shape[0]) | (stats.major_axis_length < frame_shape[1]))]
            #stats = stats[stats.area > 20]

            #stats = stats[stats.circularity > 0.5]

            if stats.size:
                #find the nearest ellipse and collect
                nearest_ellipse, distance = get_nearest_ellipse(stats, gathered_ellipses)

                if nearest_ellipse.empty:
                    nearest_ellipse, slice_data = predict_ellipse(data, gathered_ellipses, index)

                gathered_ellipses = gathered_ellipses.append(nearest_ellipse, ignore_index=True)
                print nearest_ellipse.centroid

                #draw ellipse
                slice_data = draw_ellipses(slice_data, nearest_ellipse)

        im.set_data(slice_data)

        return im,

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=300, interval=100, repeat=False)
    plt.show()

    print gathered_ellipses[['centroid','slice_idx']]

def track_guts_noa(data, inital_ellipse):
    fig = plt.figure()
    ax = fig.gca()
    plt.ion()
    plt.show()

    frame_shape = data[inital_ellipse.slice_idx].shape

    gathered_ellipses = pd.DataFrame()
    gathered_ellipses = gathered_ellipses.append(inital_ellipse, ignore_index=True)

    for slice_idx in np.arange(inital_ellipse.slice_idx + 1, 200):
        plt.title('Frame %d' % slice_idx)
        print 'FRAME #%d' % slice_idx

        slice_data = preprocess_data(data[slice_idx])

        if np.count_nonzero(slice_data):
            #segemnt frame
            labeled_data, num_features = segment_data(slice_data)

            #remove all big and non-circualr labels
            stats = slice_stats(labeled_data, slice_idx=slice_idx)
            stats = stats[(stats.area > 20) & ((stats.major_axis_length < frame_shape[0]) | (stats.major_axis_length < frame_shape[1]))]

            if stats.size:
                #find the nearest ellipse and collect
                nearest_ellipse, distance = get_nearest_ellipse(stats, gathered_ellipses)

                if nearest_ellipse.empty:
                    nearest_ellipse, slice_data = predict_ellipse(data, gathered_ellipses, slice_idx)
                    print 'Found ellipse:'
                    print nearest_ellipse

                gathered_ellipses = gathered_ellipses.append(nearest_ellipse, ignore_index=True)

                #draw ellipse
                slice_data = draw_ellipses(slice_data, nearest_ellipse)
        if slice_idx > 150:
            ax.imshow(slice_data, cmap='gray')
            plt.draw()

def segment_guts_v2():
    data = np.memmap("E:\\guts_tracking\\data\\fish202_aligned_masked_8bit_150x200x440.raw", dtype='uint8', shape=(440,200,150)).copy()

    initial_ellipses = get_init_ellipses(data)

    for index, ellipse in initial_ellipses.iterrows():
        print '---Track guts from slice# %d at point %s' % (ellipse.slice_idx, str(ellipse.centroid))
        #track_guts_animation(data, ellipse)
        track_guts_noa(data, ellipse)

def collect_stats(segmented_data, slice_idx=-1):
    properties = ['label','area','centroid','equivalent_diameter', \
                    'major_axis_length','minor_axis_length','orientation','bbox','perimeter']
    extra_props = ['circularity','slice_idx']

    u_labeled_data = np.unique(segmented_data)
    labeled_data = np.searchsorted(u_labeled_data, segmented_data)

    stats = pd.DataFrame(columns=properties)

    for region in measure.regionprops(labeled_data):
        stats = stats.append({_property: region[_property] for _property in properties}, \
                                ignore_index=True)

    for prop in extra_props:
        if prop == 'circularity':
            stats[prop] = stats.apply(lambda row: 0.0 if row['perimeter'] == 0 \
                else _calc_circularity(row['area'], row['perimeter']), axis=1)

        if prop == 'slice_idx':
            stats[prop] = slice_idx

    return stats

# def collect_circles(data):
#     frame_shape = data[0].shape
#
#     stats_z, stats_y, stats_x = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
#
#     for dim in data.shape:
#         slice_data = None
#         for slice_idx in np.arange(dim):
#             if dim == 0:
#                 slice_data = preprocess_data(data[slice_idx,:,:])
#             elif dim == 1:
#                 slice_data = preprocess_data(data[:,slice_idx,:])
#             elif dim == 2:
#                 slice_data = preprocess_data(data[:,:,slice_idx])
#             else:
#                 raise ValueError('Incorrect dimension of slicing.')
#
#             if np.count_nonzero(slice_data):
#                 #segemnt frame
#                 labeled_data, num_features = segment_data(slice_data)
#
#                 #remove all big and non-circualr labels
#                 stats = collect_stats(labeled_data, slice_idx=slice_idx)
#                 stats = stats[(stats.area > 20) & ((stats.major_axis_length < frame_shape[0]) | (stats.major_axis_length < frame_shape[1]))]
#                 stats = stats[stats.circularity > 0.6]
#
#                 if dim == 0:
#                     stats_z = stats_z.append(stats, ignore_index=True)
#                 elif dim == 1:
#                     stats_y = stats_y.append(stats, ignore_index=True)
#                 elif dim == 2:
#                     stats_x = stats_x.append(stats, ignore_index=True)
#                 else:
#                     raise ValueError('Incorrect dimension of stats.')
#
#     return stats_z, stats_y, stats_x

def collect_circles(data, dim=0):
    frame_size = min(data.shape)

    collected_stats = pd.DataFrame()
    slice_data = None

    for slice_idx in np.arange(data.shape[dim]):
        if dim == 0:
            slice_data = preprocess_data(data[slice_idx,:,:])
        elif dim == 1:
            slice_data = preprocess_data(data[:,slice_idx,:])
        elif dim == 2:
            slice_data = preprocess_data(data[:,:,slice_idx])
        else:
            raise ValueError('Incorrect dimension of slicing.')

        if np.count_nonzero(slice_data):
            #segemnt frame
            labeled_data, num_features = segment_data(slice_data)

            #remove all big and non-circualr labels
            stats = collect_stats(labeled_data, slice_idx=slice_idx)
            stats = stats[(stats.area > 50) & (stats.major_axis_length < frame_size)]
            stats = stats[stats.circularity > 0.8]
            collected_stats = collected_stats.append(stats, ignore_index=True)

    return collected_stats

def create_volume(circle_stats, output_shape, dim=0):
    volume = np.zeros(output_shape)

    for index, circle in circle_stats.iterrows():
        if dim == 0:
            volume[circle.slice_idx, circle.centroid[0], circle.centroid[1]] = 1
        elif dim == 1:
            volume[circle.centroid[0], circle.slice_idx, circle.centroid[1]] = 1
        elif dim == 2:
            volume[circle.centroid[0], circle.centroid[1], circle.slice_idx] = 1

    return volume

def produce_volume_points(data, dim=0):
    stats = collect_circles(data, dim=dim)
    transformed_volume = create_volume(stats, data.shape, dim=dim)

    return transformed_volume

def get_points(data):
    points = []

    for z in np.arange(data.shape[0]):
        for y in np.arange(data.shape[1]):
            for x in np.arange(data.shape[2]):
                if data[z,y,x] != 0:
                    points.append([z,y,x])

    return np.array(points)

def segment_guts_multiprojections():
    data = np.memmap("E:\\guts_tracking\\data\\fish202_aligned_masked_8bit_150x200x440.raw", dtype='uint8', shape=(440,200,150)).copy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #(z,y,x)
    angles = np.array([[0,0,0], [45,0,0], [90,0,0], [135,0,0], [0,0,45], [0,0,90], [0,0,135],\
              [0,0,160], [0,45,0], [0,90,0], [0,135,0], [0,160,0]])

    collected_vol_points = np.zeros(data.shape, dtype=np.int)

    for angle_set in angles:
        print 'Angles %s' % str(angle_set)
        rotated = data
        if any(angle_set):
            rotated = rotate_volume(data, angle_set, order=3)

        for dim in range(len(data.shape)):
            print 'Dim %d' % dim
            rotated_vol = produce_volume_points(rotated, dim=dim)

            if any(angle_set):
                rotated_vol = rotate_volume(rotated_vol, -angle_set, order=3)

            collected_vol_points = np.logical_or(collected_vol_points, rotated_vol).astype(np.int)

    print 'Obtaining points...'
    points = get_points(collected_vol_points)

    ax.scatter(points[:,2], points[:,0], points[:,1], c='r', marker='o')
    plt.show()


    #plot_comparative(data, rotated_inv, slice_idx=round(data.shape[0]/2.), phi=0, theta=-45, psi=-22)
    #plt.imshow(rotated[100], cmap='gray')
    #plt.show()

def test3d():
    mu, sigma = 0, 0.1
    x = 10*np.random.normal(mu, sigma, 5000)
    y = 10*np.random.normal(mu, sigma, 5000)
    z = 10*np.random.normal(mu, sigma, 5000)

    xyz = np.vstack([x,y,z])
    kde = stats.gaussian_kde(xyz)
    density = kde(xyz)

    # Plot scatter with mayavi
    figure = mlab.figure('DensityPlot')
    pts = mlab.points3d(x, y, z, density, scale_mode='none', scale_factor=0.07)
    mlab.show()


if __name__ == "__main__":
    #test3d()
    #test_data_slice()
    #get_oblique_slice()
    #plot_obique_slices()
    #profile.run('get_guts_oblique_slice(theta_y=16, psi_x=7, slice_idx=283, rot_p0=[289, 109, 84]); print')
    #get_guts_oblique_slice(theta_y=16, psi_x=7, slice_idx=283, rot_p0=[289, 109, 84])
    #segment_guts()
    #test_canny()
    #test_circles()
    #test_circles2()
    #test_detect_by_stats()
    #test_detect_by_stats2()
    #segment_guts_v2()
    segment_guts_multiprojections()
