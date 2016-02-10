import numpy as np
import pandas as pd
from scipy import ndimage as ndi
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

    map_coordinates(v, coords, order=1, output=output, **kwargs)

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
def get_guts_oblique_slice(phi_z=0, theta_y=0, psi_x=0, slice_idx=125, rot_p0=None):
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
    new_data = interp3(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), data, z_coordinates, y_coordinates, x_coordinates)

    #plot data
    #plot_comparative(data, new_data, slice_idx=slice_idx, phi=phi, theta=theta, psi=psi)

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

def slice_stats(segmented_data):
    properties = ['label','area','centroid','equivalent_diameter', \
                'major_axis_length','minor_axis_length','orientation','bbox','perimeter']
    extra_props = ['circularity']

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

def draw_esllipse(data, ellipse):
    if ellipse:
        image = ski.color.gray2rgb(data)

        #best = list(ellipse[-1])
        best = ellipse
        yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        orientation = best[5]

        cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        image[cy, cx] = (220, 20, 20)

        rr, cc = circle(yc, xc, 2)
        data[rr, cc] = (220, 20, 20)
    else:
        return np.zeros(data.shape)

    return data

def get_nearest_ellipse(ellipses_props, gathered_ellipses):
    if not len(gathered_ellipses):
        return list(ellipses_props[-1])

    prev_ellipse = gathered_ellipses[-1]
    nearest_ellipse_to_prev, distance = None, 0

    def get_distance(ellipse, prev_ellipse):
        return np.abs(prev_ellipse[1] - ellipse[1]) + np.abs(prev_ellipse[2] - ellipse[2])

    for ellipse in ellipses_props:
        if not nearest_ellipse_to_prev:
            nearest_ellipse_to_prev = ellipse
            distance = get_distance(ellipse, prev_ellipse)
            continue

        new_distance = get_distance(ellipse, prev_ellipse)

        if new_distance <= distance:
            distance = new_distance
            nearest_ellipse_to_prev = ellipse

    return nearest_ellipse_to_prev

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

    data_slice = data[40]

    slice_data = preprocess_data(data_slice)
    labeled_data, num_features = segment_data(slice_data)

    plt.imshow(labeled_data, cmap='gray')
    plt.show()

    stats = slice_stats(labeled_data)
    print stats

    stats = stats[stats.area > 20]
    stats = stats[stats.circularity > 0.7]

    image = data_slice

    for index, row in stats.iterrows():
        yc, xc = [int(round(x)) for x in row.centroid]
        orientation = row.orientation
        major_axis = int(round(row.major_axis_length/2.))
        minor_axis = int(round(row.minor_axis_length/2.))

        image = ski.color.gray2rgb(data_slice)

        cy, cx = ellipse_perimeter(yc, xc, minor_axis, major_axis, orientation)
        image[cy, cx] = (220, 20, 20)

        rr, cc = circle(yc, xc, 2)
        image[rr, cc] = (220, 20, 20)

    plt.imshow(labeled_data, cmap='gray')
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

if __name__ == "__main__":
    #test_data_slice()
    #get_oblique_slice()
    #plot_obique_slices()
    #profile.run('get_guts_oblique_slice(theta_y=16, psi_x=7, slice_idx=283, rot_p0=[289, 109, 84]); print')
    get_guts_oblique_slice(theta_y=16, psi_x=7, slice_idx=283, rot_p0=[289, 109, 84])
    #segment_guts()
    #test_canny()
    #test_circles()
    #test_circles2()
    #test_detect_by_stats()
    #test_detect_by_stats2()
