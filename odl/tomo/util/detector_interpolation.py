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

from odl.discr import uniform_partition, uniform_discr
from odl.tomo.geometry.detector import (Flat1dDetector, CircularDetector)


__all__ = ('flat1d_to_circular', 'circular_to_flat1d')

def flat1d_to_circular(data, detector, radius, interp='linear'):
    """Transforms data from Flat1d detector to Circular.

    Parameters
    ----------
    data : `numpy.ndarray`
        Data sampled on a partition of the detector.
    detector : Flat1dDetector
        The detector on which the data was sampled.
    radius :
        Curvature radius of the circular detector.

    Returns
    -------
    resampled_data : `numpy.ndarray`
        Resampled data, which corresponds to a new detector
    transformed_detector : `CircularDetector`
        A new transformed detector``.

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
    >>> new_data, new_det = flat1d_to_circular(data, det, 1)
    >>> np.round(new_data, 2)
    array([-0.87,  0.  ,  0.87])

    The method is vectorized, i.e., it can be called for multiple observations
    of values on the detector (most often corresponding to different angles):

    >>> part = odl.uniform_partition(-1, 1, 3)
    >>> det = odl.tomo.Flat1dDetector(part, axis=[1, 0])
    >>> data_row = np.arange(-1, 2)
    >>> data = np.stack([data_row] * 2)
    >>> new_data, new_det = flat1d_to_circular(data, det, 1)
    >>> np.round(new_data, 2)
    array([[-0.87,  0.  ,  0.87],
           [-0.87,  0.  ,  0.87]])
    """
    part = detector.partition
    d = len(part.shape)
    data = np.asarray(data, dtype=float)
    if data.shape[-d:] != part.shape:
        raise ValueError('Last dimensions of `data.shape` must '
                         'correspond to the detector partitioning, '
                         'got {} and {}'.format(data.shape[-d:], part.shape))

    assert isinstance(detector, Flat1dDetector)

    radius = float(radius)
    if radius <= 0:
        raise ValueError('`radius` must be positive')

    # reshape the data
    original_shape = data.shape
    new_shape = np.append([-1], part.shape)
    data = data.reshape(new_shape)

    # define data space
    n = data.shape[0]
    if n > 1:
        min_pt = np.append([0], part.min_pt)
        max_pt = np.append([n], part.max_pt)
        shape = np.append([n], part.shape)
    else:
        min_pt = part.min_pt
        max_pt = part.max_pt
        shape = part.shape
    space = uniform_discr(min_pt, max_pt, shape, interp='linear')

    # create the transformed detector
    min_pt = np.arctan2(part.min_pt, radius)
    max_pt = np.arctan2(part.max_pt, radius)
    transformed_part = uniform_partition(min_pt, max_pt, part.shape)
    transformed_det = CircularDetector(transformed_part,
                                       detector.axis, radius,
                                       check_bounds=detector.check_bounds)

    # re-sample the data using interpolation
    phi = transformed_det.partition.meshgrid[0]
    u = radius * np.tan(phi)
    if n > 1:
        u = u.reshape(1, -1)
        i = space.partition.meshgrid[0]
        resampled_data = space.element(data).interpolation((i, u))
    else:
        resampled_data = space.element(data[0]).interpolation(u)

    # reshape the data to its original shape
    resampled_data = resampled_data.reshape(original_shape)

    return resampled_data, transformed_det


def circular_to_flat1d(data, detector, interp='linear'):
    """Transforms data from Circular detector to Flat1d.

    Parameters
    ----------
    data : `numpy.ndarray`
        Data sampled on a partition of the detector.
    detector : Flat1dDetector
        The detector on which the data was sampled.

    Returns
    -------
    resampled_data : `numpy.ndarray`
        Resampled data, which corresponds to a new detector
    transformed_detector : `CircularDetector`
        A new transformed detector``.

    Examples
    --------
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
    >>> new_data, new_det = circular_to_flat1d(data, det, 1)
    >>> np.round(new_data, 2)
    array([-0.88,  0.  ,  0.88])

    The method is vectorized, i.e., it can be called for multiple observations
    of values on the detector (most often corresponding to different angles):

    >>> part = odl.uniform_partition(-np.pi / 4, np.pi / 4, 3)
    >>> det = odl.tomo.CircularDetector(part, axis=[1, 0], radius=1)
    >>> data_row = np.arange(-1, 2)
    >>> data = np.stack([data_row] * 2)
    >>> new_data, new_det = circular_to_flat1d(data, det, 1)
    >>> np.round(new_data, 2)
    array([[-0.88,  0.  ,  0.88],
           [-0.88,  0.  ,  0.88]])
    """
    part = detector.partition
    d = len(part.shape)
    data = np.asarray(data, dtype=float)
    if data.shape[-d:] != part.shape:
        raise ValueError('Last dimensions of `data.shape` must '
                         'correspond to the detector partitioning, '
                         'got {} and {}'.format(data.shape[-d:], part.shape))

    assert isinstance(detector, CircularDetector)

    # reshape the data
    original_shape = data.shape
    new_shape = np.append([-1], part.shape)
    data = data.reshape(new_shape)

    # define data space
    n = data.shape[0]
    if n > 1:
        min_pt = np.append([0], part.min_pt)
        max_pt = np.append([n], part.max_pt)
        shape = np.append([n], part.shape)
    else:
        min_pt = part.min_pt
        max_pt = part.max_pt
        shape = part.shape
    space = uniform_discr(min_pt, max_pt, shape, interp='linear')

    # create the transformed detector
    min_pt = detector.radius * np.tan(part.min_pt)
    max_pt = detector.radius * np.tan(part.max_pt)
    transformed_part = uniform_partition(min_pt, max_pt, part.shape)
    transformed_det = Flat1dDetector(transformed_part,
                                    detector.axis,
                                    check_bounds=detector.check_bounds)

    # re-sample the data using interpolation
    u = transformed_det.partition.meshgrid[0]
    phi = np.arctan2(u, detector.radius)
    if n > 1:
        phi = phi.reshape(1, -1)
        i = space.partition.meshgrid[0]
        resampled_data = space.element(data).interpolation((i, phi))
    else:
        resampled_data = space.element(data[0]).interpolation(phi)

    # reshape the data to its original shape
    resampled_data = resampled_data.reshape(original_shape)

    return resampled_data, transformed_det


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
