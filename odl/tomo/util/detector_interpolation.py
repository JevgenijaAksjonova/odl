# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Switching between flat and curved detectors by using interpolation."""

from __future__ import print_function, division, absolute_import
from builtins import object
import numpy as np

from odl.discr import uniform_partition
from odl.discr.discr_utils import linear_interpolator
from odl.tomo.geometry.detector import (Detector,
                                        Flat1dDetector,
                                        Flat2dDetector,
                                        CircularDetector,
                                        CylindricalDetector,
                                        SphericalDetector)


__all__ = ('flat_to_curved', 'curved_to_flat',
           'project_data_to_flat', 'project_data_to_curved')

def flat_to_curved(detector, radius):
    """Transforms data from Flat1d detector to Circular.

    Parameters
    ----------
    detector : `Flat1dDetector` or `Flat2dDetector`
        The detector on which the data was sampled.
    radius :  nonnegatice float or 2-tuple of nonnegative floats
        Radius or radii of the detector curvature.
        If ``r`` a circular or cylindrical detector is created.
        If ``(r, None)`` or ``(r, float('inf'))``, a cylindrical
        detector is created.
        If ``(r1, r2)``, a spherical detector is created.

    Returns
    -------
    transformed_detector : `CircularDetector`, `CylindricalDetector`
        or `SphericalDetector`
        A new transformed detector.

    Examples
    --------
    Transforming a flat detector to a circular. In this example
    a flat detector has a range [-1, 1], with uniform discretization
    [-2/3, 0, 2/3]. The corresponding circular detector has a range [-pi/4, pi/4]
    with uniform discretization [-pi/6, 0, pi/6], which corresponds to points
    [-tan(pi/6), 0, tan(pi/6)] on a flat detector.

    >>> part = odl.uniform_partition(-1, 1, 3)
    >>> det = odl.tomo.Flat1dDetector(part, axis=[1, 0])
    >>> new_det = flat_to_curved(det, 1)
    >>> new_det
    array([ 0., 0.,  0.])
    """

    part = detector.partition
    radius = np.array(radius, dtype=np.float, ndmin=1)
    if isinstance(detector, Flat1dDetector):

        if radius <= 0 or radius.size > 1:
            raise ValueError('`radius` must be positive float')

        min_pt = np.arctan2(part.min_pt, radius)
        max_pt = np.arctan2(part.max_pt, radius)
        transformed_part = uniform_partition(min_pt, max_pt, part.shape)
        transformed_det = CircularDetector(transformed_part,
                                       detector.axis, radius,
                                       check_bounds=detector.check_bounds)

    elif isinstance(detector, Flat2dDetector):

        if radius.size == 1 or radius[1] == None or radius[1] == float('inf'):

            if radius[0] <= 0:
                raise ValueError('`radius` must be positive float')

            min_pt = [np.arctan2(part.min_pt[0], radius[0]), part.min_pt[1]]
            max_pt = [np.arctan2(part.max_pt[0], radius[0]), part.max_pt[1]]
            transformed_part = uniform_partition(min_pt, max_pt, part.shape)
            transformed_det = CylindricalDetector(
                transformed_part,
                radius=radius[0],
                axes=detector.axis,
                check_bounds=detector.check_bounds)

        elif radius[0] == radius[1]:

            if radius[0] <= 0:
                raise ValueError('`radius` must be positive float')

            min_pt = np.arctan2(part.min_pt, radius)
            max_pt = np.arctan2(part.max_pt, radius)
            transformed_part = uniform_partition(min_pt, max_pt, part.shape)
            transformed_det = SphericalDetector(
                transformed_part,
                radius=radius[0],
                axes=detector.axis,
                check_bounds=detector.check_bounds)

        else:
            raise NotImplementedError('Curved detector with different '
                                      'curvature radii')
    else:
        raise ValueError('Detector must be flat, got '.format(detector))

    return transformed_det


def curved_to_flat(detector):
    """Transforms curved detector to flat.

    Parameters
    ----------
    detector : `CircularDetector`, `CylindricalDetector`
        or `SphericalDetector`
        The detector on which the data was sampled.

    Returns
    -------
    transformed_detector : `Flat1dDetector` or `Flat2dDetector`
        A new transformed detector.

    Examples
    --------
    Transforming a circular detector to a flat. In this example
    a circular detector has a range [-pi/4, pi/4], with uniform discretization
    [-pi/6, 0, pi/6]. The corresponding flat detector has a range [-1, 1]
    with uniform discretization [-2/3, 0, 2/3], which corresponds to points
    [-arctan(2/3), 0, arctan(2/3)] on a circular detector.

    >>> part = odl.uniform_partition(-np.pi / 4, np.pi / 4, 3)
    >>> det = odl.tomo.CircularDetector(part, axis=[1, 0], radius=1)
    >>> new_det = curved_to_flat(data, det, 1)
    >>> new_det
    array([-0.88,  0.  ,  0.88])
    """

    part = detector.partition
    if isinstance(detector, CircularDetector):

        min_pt = detector.radius * np.tan(part.min_pt)
        max_pt = detector.radius * np.tan(part.max_pt)
        transformed_part = uniform_partition(min_pt, max_pt, part.shape)
        transformed_det = Flat1dDetector(transformed_part,
                                         detector.axis,
                                         check_bounds=detector.check_bounds)

    elif isinstance(detector, CylindricalDetector):

        min_pt = [detector.radius * np.tan(part.min_pt[0]), part.min_pt[1]]
        max_pt = [detector.radius * np.tan(part.max_pt[0]), part.max_pt[1]]
        transformed_part = uniform_partition(min_pt, max_pt, part.shape)
        transformed_det = Flat2dDetector(transformed_part,
                                         detector.axis,
                                         check_bounds=detector.check_bounds)

    elif isinstance(detector, SphericalDetector):

        min_pt = detector.radius * np.tan(part.min_pt)
        max_pt = detector.radius * np.tan(part.max_pt)
        transformed_part = uniform_partition(min_pt, max_pt, part.shape)
        transformed_det = Flat2dDetector(transformed_part,
                                         detector.axis,
                                         check_bounds=detector.check_bounds)

    else:
        raise ValueError('Detector must be flat, got '.format(detector))

    return transformed_det


def project_data_to_flat(data, old_detector, new_detector):
    """Transforms data one detector to another using linear interpolation.

    Parameters
    ----------
    data : `numpy.ndarray`
        Data sampled on a partition of the detector.
    old_detector : Detector
        The detector on which the data was sampled.
    new_detector : Detector
        The detector to which the data is projected.

    Returns
    -------
    resampled_data : `numpy.ndarray`
        Resampled data, which corresponds to a new detector

    Examples
    --------
    Transforming a flat detector to a circular. In this example
    a flat detector has a range [-1, 1], with uniform discretization
    [-2/3, 0, 2/3]. The corresponding circular detector has a range [-pi/4, pi/4]
    with uniform discretization [-pi/6, 0, pi/6], which corresponds to points
    [-tan(pi/6), 0, tan(pi/6)] on a flat detector. In the case of linear
    interpolation, values at these points would be [-0.87, 0, 0.87]
    (can be verified using proportion rule tan(pi/6) : 2/3 = x : 1).

    >>> part = odl.uniform_partition(-1, 1, 3)
    >>> det = odl.tomo.Flat1dDetector(part, axis=[1, 0])
    >>> data = np.arange(-1, 2)
    >>> new_det = flat_to_curved(det, 1)
    >>> new_data = project_data(data, det, new_det)
    >>> np.round(new_data, 2)
    array([-0.87,  0.  ,  0.87])

    Transforming a circular detector to a flat. In this example
    a circular detector has a range [-pi/4, pi/4], with uniform discretization
    [-pi/6, 0, pi/6]. The corresponding flat detector has a range [-1, 1]
    with uniform discretization [-2/3, 0, 2/3], which corresponds to points
    [-arctan(2/3), 0, arctan(2/3)] on a circular detector. In the case of linear
    interpolation, values at these points would be [-0.88, 0, 0.88].
    (can be verified by noting that the value at arctan(2/3) should be equal to
    the value at pi/3 - arctan(2/3) and using proportion rule
    (pi/3 - arctan(2/3)) : pi/6 = x : 1).

    >>> part = odl.uniform_partition(-np.pi / 4, np.pi / 4, 3)
    >>> det = odl.tomo.CircularDetector(part, axis=[1, 0], radius=1)
    >>> data = np.arange(-1, 2)
    >>> new_det = curved_to_flat(det)
    >>> new_data = project_data(data, det, new_det)
    >>> np.round(new_data, 2)
    array([-0.88,  0.  ,  0.88])

    The method is vectorized, i.e., it can be called for multiple observations
    of values on the detector (most often corresponding to different angles):

    >>> part = odl.uniform_partition(-1, 1, 3)
    >>> det = odl.tomo.Flat1dDetector(part, axis=[1, 0])
    >>> data_row = np.arange(-1, 2)
    >>> data = np.stack([data_row] * 2)
    >>> new_det = flat_to_curved(det, 1)
    >>> new_data = project_data(data, det, new_det)
    >>> np.round(new_data, 2)
    array([[-0.87,  0.  ,  0.87],
           [-0.87,  0.  ,  0.87]])
    """

    assert isinstance(old_detector, Detector)
    assert isinstance(new_detector, Detector)
    if old_detector.axis != new_detector.axis:
        NotImplementedError('Detectors are axis not the same, {} and {}'
                            ''.format(old_detector.axis, new_detector.axis))
    part = old_detector.partition
    d = len(part.shape)
    data = np.asarray(data, dtype=float)
    if data.shape[-d:] != part.shape:
        raise ValueError('Last dimensions of `data.shape` must '
                         'correspond to the detector partitioning, '
                         'got {} and {}'.format(data.shape[-d:], part.shape))

    # reshape the data
    original_shape = data.shape
    new_shape = np.append([-1], part.shape)
    data = data.reshape(new_shape)

    # extend detectors partition for multiple samples
    n = data.shape[0]
    data_part = uniform_partition(np.append([0], part.min_pt),
                                  np.append([n], part.max_pt),
                                  np.append([n], part.shape))
    i = data_part.meshgrid[0] # sample indices

    if isinstance(old_detector, Flat1dDetector):
        assert isinstance(new_detector, CircularDetector)

        phi = new_detector.partition.meshgrid
        u = new_detector.radius * np.tan(phi)
        interpolator = linear_interpolator(data, data_part.coord_vecs)
        resampled_data = interpolator((i,u))

    elif isinstance(old_detector, CircularDetector):
        assert isinstance(new_detector, Flat1dDetector)

        u = new_detector.partition.meshgrid
        phi = np.arctan2(u, new_detector.radius)
        interpolator = linear_interpolator(data, data_part.coord_vecs)
        resampled_data = interpolator((i, phi))

    elif isinstance(old_detector, Flat2dDetector):
        assert isinstance(new_detector, [CylindricalDetector,
                                         SphericalDetector])

        r = new_detector.radius
        if isinstance(new_detector, CylindricalDetector):
            phi, h = new_detector.partition.meshgrid
            u = r * np.tan(phi)
            v = h / r * np.sqrt(r * r + u * u)
        else:
            phi, theta = new_detector.partition.meshgrid
            u = r * np.tan(phi)
            v = np.tan(theta) * np.sqrt(r * r + u * u)
        interpolator = linear_interpolator(data, data_part.coord_vecs)
        resampled_data = interpolator((i,u,v))

    elif isinstance(old_detector, CylindricalDetector):
        assert isinstance(new_detector, [Flat2dDetector,
                                         SphericalDetector])

        r = old_detector.radius
        if isinstance(new_detector, Flat2dDetector):
            u, v = new_detector.partition.meshgrid
            phi = np.arctan2(u, r)
            h = v * r / np.sqrt(r * r + u * u)
        else:
            phi, theta = new_detector.partition.meshgrid
            h = r * np.tan(theta)
        interpolator = linear_interpolator(data, data_part.coord_vecs)
        resampled_data = interpolator((i, phi, h))

    elif isinstance(old_detector, SphericalDetector):
        assert isinstance(new_detector, [Flat2dDetector,
                                         CylindricalDetector])

        r = old_detector.radius
        if isinstance(new_detector, Flat2dDetector):
            u, v = new_detector.partition.meshgrid
            phi = np.arctan2(u, r)
            theta = np.arctan2(v / np.sqrt(r * r + u * u))
        else:
            phi, h = new_detector.partition.meshgrid
            theta = np.arctan2(h, r)
        interpolator = linear_interpolator(data, data_part.coord_vecs)
        resampled_data = interpolator((i, phi, theta))

    else:
        NotImplementedError('Data transformation between detectors {} and {}'
                            'is not implemented'.format(old_detector,
                                                        new_detector))

    # reshape the data to its original shape
    resampled_data = resampled_data.reshape(original_shape)
    return resampled_data


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
