# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Switching between flat and curved detectors by using interpolation."""

from __future__ import print_function, division, absolute_import

import numpy as np

from odl.discr import uniform_partition
from odl.discr.discr_utils import linear_interpolator
from odl.tomo.geometry.detector import (Detector,
                                        Flat1dDetector,
                                        Flat2dDetector,
                                        CircularDetector,
                                        CylindricalDetector,
                                        SphericalDetector)


__all__ = ('flat_to_curved', 'curved_to_flat', 'project_data')


def flat_to_curved(detector, radius):
    """Transforms a flat detector to a curved.

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
    Transforming a flat detector with range [-1, 1] to
    a circular with radius 1 and range [-pi/4, pi/4].

    >>> part = odl.uniform_partition(-1, 1, 3)
    >>> det = odl.tomo.Flat1dDetector(part, axis=[1, 0])
    >>> new_det = flat_to_curved(det, radius=1)
    >>> new_det
    CircularDetector(
        uniform_partition(-0.7854, 0.7854, 3),
        axis='[ 1.,  0.]',
        radius='1.0'
    )

    Transforming a flat detector with height 2 to
    a cylindrical with radius 1 and height 2.
    >>> part = odl.uniform_partition([-1, -1], [1, 1], (3, 2))
    >>> det = odl.tomo.Flat2dDetector(part, axes=[[1, 0, 0], [0, 0, 1]])
    >>> new_det = flat_to_curved(det, radius=1)
    >>> new_det
    CylindricalDetector(
        uniform_partition([-0.7854, -1.    ], [ 0.7854,  1.    ], (3, 2)),
        axes=('[ 1.,  0.,  0.]', '[ 0.,  0.,  1.]'),
        radius='1.0'
    )
    """

    part = detector.partition
    radius = np.array(radius, dtype=np.float, ndmin=1)
    if isinstance(detector, Flat1dDetector):

        if radius <= 0 or radius.size > 1:
            raise ValueError('`radius` must be positive float')

        min_pt = np.arctan2(part.min_pt, radius)
        max_pt = np.arctan2(part.max_pt, radius)
        transformed_part = uniform_partition(min_pt, max_pt,
                                             part.shape,
                                             nodes_on_bdry=part.nodes_on_bdry)
        transformed_det = CircularDetector(transformed_part,
                                           detector.axis, radius,
                                           check_bounds=detector.check_bounds)

    elif isinstance(detector, Flat2dDetector):

        if radius.size == 1 or radius[1] is None or radius[1] == float('inf'):

            if radius[0] <= 0:
                raise ValueError('`radius` must be positive float')

            min_pt = [np.arctan2(part.min_pt[0], radius[0]), part.min_pt[1]]
            max_pt = [np.arctan2(part.max_pt[0], radius[0]), part.max_pt[1]]
            transformed_part = uniform_partition(
                min_pt, max_pt,
                part.shape,
                nodes_on_bdry=part.nodes_on_bdry)
            transformed_det = CylindricalDetector(
                transformed_part,
                radius=radius[0],
                axes=detector.axes,
                check_bounds=detector.check_bounds)

        elif radius[0] == radius[1]:

            if radius[0] <= 0:
                raise ValueError('`radius` must be positive float')

            min_pt = np.arctan2(part.min_pt, radius)
            max_pt = np.arctan2(part.max_pt, radius)
            transformed_part = uniform_partition(
                min_pt, max_pt,
                part.shape,
                nodes_on_bdry=part.nodes_on_bdry)
            transformed_det = SphericalDetector(
                transformed_part,
                radius=radius[0],
                axes=detector.axes,
                check_bounds=detector.check_bounds)

        else:
            raise NotImplementedError('Curved detector with different '
                                      'curvature radii')
    else:
        raise ValueError('Detector must be flat, got '.format(detector))

    return transformed_det


def curved_to_flat(detector):
    """Transforms a curved detector to a flat.

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
    Transforming a circular detector with range [-pi/4, pi/4]
    to a flat with range [-1, 1].

    >>> part = odl.uniform_partition(-np.pi / 4, np.pi / 4, 3,
    ...                              nodes_on_bdry=True)
    >>> det = odl.tomo.CircularDetector(part, axis=[1, 0], radius=1)
    >>> new_det = curved_to_flat(det)
    >>> new_det
    Flat1dDetector(
        uniform_partition(-1.0, 1.0, 3, nodes_on_bdry=True),
        axis='[ 1.,  0.]'
    )

    Transforming a cylindrical detector with radius 1 and height 2 to
    a flat with height 2sqrt(2) (since  edge point are projected higher).
    >>> part = odl.uniform_partition([-np.pi / 4, -1], [np.pi / 4, 1], (3, 2),
    ...                              nodes_on_bdry=True)
    >>> det = odl.tomo.CylindricalDetector(part,
    ...                                    axes=[[1, 0, 0], [0, 0, 1]],
    ...                                    radius=1)
    >>> new_det = curved_to_flat(det)
    >>> new_det
    Flat2dDetector(
        uniform_partition([-1.    , -1.4142], [ 1.    ,  1.4142], (3, 2),
            nodes_on_bdry=True),
        axes=('[ 1.,  0.,  0.]', '[ 0.,  0.,  1.]')
    )
    """

    part = detector.partition
    if isinstance(detector, CircularDetector):

        min_pt = detector.radius * np.tan(part.min_pt)
        max_pt = detector.radius * np.tan(part.max_pt)
        transformed_part = uniform_partition(
            min_pt, max_pt,
            part.shape,
            nodes_on_bdry=part.nodes_on_bdry)
        transformed_det = Flat1dDetector(transformed_part,
                                         detector.axis,
                                         check_bounds=detector.check_bounds)

    elif isinstance(detector, CylindricalDetector):

        r = detector.radius
        R = r / np.minimum(np.cos(part.min_pt[0]), np.cos(part.max_pt[0]))
        min_pt = [r * np.tan(part.min_pt[0]), R / r * part.min_pt[1]]
        max_pt = [r * np.tan(part.max_pt[0]), R / r * part.max_pt[1]]
        transformed_part = uniform_partition(
            min_pt, max_pt,
            part.shape,
            nodes_on_bdry=part.nodes_on_bdry)
        transformed_det = Flat2dDetector(transformed_part,
                                         detector.axes,
                                         check_bounds=detector.check_bounds)

    elif isinstance(detector, SphericalDetector):

        r = detector.radius
        R = r / np.minimum(np.cos(part.min_pt[0]), np.cos(part.max_pt[0]))
        min_pt = [r * np.tan(part.min_pt[0]), R * np.tan(part.min_pt[1])]
        max_pt = [r * np.tan(part.max_pt[0]), R * np.tan(part.max_pt[1])]
        transformed_part = uniform_partition(
            min_pt, max_pt,
            part.shape,
            nodes_on_bdry=part.nodes_on_bdry)
        transformed_det = Flat2dDetector(transformed_part,
                                         detector.axes,
                                         check_bounds=detector.check_bounds)

    else:
        raise ValueError('Detector must be curved, got '.format(detector))

    return transformed_det


def project_data(data, old_detector, new_detector):
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
    [-1, -0.5, 0, 0.5, 1]. The circular detector has a range [-pi/4, pi/4]
    with uniform discretization [-pi/4, -pi/8, 0, pi/8, pi/4], which
    corresponds to [-1, -0.41, 0, 0.41, 1] on the flat detector,
    since tg(pi/8) approx. 0.41. Thus, values at points -pi/8 and pi/8 obtained
    through interpolation are -0.83 and 0.83 (approx. 0.41/0.5).

    >>> part = odl.uniform_partition(-1, 1, 5, nodes_on_bdry=True)
    >>> det = odl.tomo.Flat1dDetector(part, axis=[1, 0])
    >>> det.partition.meshgrid[0]
        array([-1. , -0.5,  0. ,  0.5,  1. ])
    >>> data = np.arange(-2, 3)
    >>> new_det = flat_to_curved(det, radius=1)
    >>> new_data = project_data(data, det, new_det)
    >>> np.round(new_data, 2)
    array([-2.  , -0.83,  0.  ,  0.83,  2.  ])

    Transforming a circular detector to a flat. In this example
    a circular detector has a range [-pi/4, pi/4] with uniform discretization
    [-0.79, -0.39, 0, 0.39, 0.79]. The corresponding flat detector has
    uniform discretization [-1, -0.5, 0, 0.5, 1], which corresponds to points
    [-0.79, -0.46, 0, 0.46, 0.79] on the circular detector,
    since arctg(0.5) = 0.46. Thus, values at points -0.5 and 0.5 obtained
    through interpolation are -1.18 and 1.18
    (approx. (0.79-0.46)/0.39*1 + (0.46-0.39)/0.39*2).

    >>> part = odl.uniform_partition(-np.pi / 4, np.pi / 4, 5,
    ...                              nodes_on_bdry=True)
    >>> det = odl.tomo.CircularDetector(part, axis=[1, 0], radius=1)
    >>> data = np.arange(-2, 3)
    >>> new_det = curved_to_flat(det)
    >>> new_data = project_data(data, det, new_det)
    >>> np.round(new_data, 2)
    array([-2.  ,  -1.18, 0.  , 1.18,  2.  ])

    Previous example extended to 2D cylindrical detector with height 2 and
    uniform partition along height axis [-1, -0.5, 0, 0.5, 1].
    The height partition of corresponding 2D flat detector is
    [-1.41, -0.71, 0., 0.71, 1.41].
    We can see that points that are closer to side edges of the cylinder are
    are projected higher on the flat detector.
    >>> part = odl.uniform_partition([-np.pi / 4, -1], [np.pi / 4, 1], (5, 5),
    ...                              nodes_on_bdry=True)
    >>> det = odl.tomo.CylindricalDetector(part,
    ...                                    axes=[[1, 0, 0], [0, 0, 1]],
    ...                                    radius=1)
    >>> data = np.zeros((5,5))
    >>> data[:, 1:4] = 1
    >>> new_det = curved_to_flat(det)
    >>> np.round(new_det.partition.meshgrid[1], 2)
    array([[-1.41, -0.71,  0.  ,  0.71,  1.41]])
    >>> new_data = project_data(data, det, new_det)
    >>> np.round(new_data.T, 2)
    array([[ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 1.  ,  0.74,  0.59,  0.74,  1.  ],
           [ 1.  ,  1.  ,  1.  ,  1.  ,  1.  ],
           [ 1.  ,  0.74,  0.59,  0.74,  1.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ]])

    Now projecting this back to curved detector.
    >>> new_data = project_data(new_data, new_det, det)
    >>> np.round(new_data.T, 2)
    array([[ 0.  ,  0.33,  0.34,  0.33,  0.  ],
           [ 1.  ,  0.78,  0.71,  0.78,  1.  ],
           [ 1.  ,  1.  ,  1.  ,  1.  ,  1.  ],
           [ 1.  ,  0.78,  0.71,  0.78,  1.  ],
           [ 0.  ,  0.33,  0.34,  0.33,  0.  ]])

    The method is vectorized, i.e., it can be called for multiple observations
    of values on the detector (most often corresponding to different angles):

    >>> part = odl.uniform_partition(-1, 1, 3, nodes_on_bdry=True)
    >>> det = odl.tomo.Flat1dDetector(part, axis=[1, 0])
    >>> data_row = np.arange(-1, 2)
    >>> data = np.stack([data_row] * 2, axis=0)
    >>> new_det = flat_to_curved(det, 1)
    >>> new_data = project_data(data, det, new_det)
    >>> np.round(new_data, 2)
    array([[-1.,  0.,  1.],
           [-1.,  0.,  1.]])
    """

    assert isinstance(old_detector, Detector)
    assert isinstance(new_detector, Detector)
    part = old_detector.partition
    d = len(part.shape)
    if d == 1 and any(old_detector.axis != new_detector.axis):
        NotImplementedError('Detectors are axis not the same, {} and {}'
                            ''.format(old_detector.axis, new_detector.axis))
    elif d > 1 and (any(old_detector.axes[0] != new_detector.axes[0])
                    or any(old_detector.axes[1] != new_detector.axes[1])):
        NotImplementedError('Detectors are axis not the same, {} and {}'
                            ''.format(old_detector.axes, new_detector.axes))
    data = np.asarray(data, dtype=float)
    if data.shape[-d:] != part.shape:
        raise ValueError('Last dimensions of `data.shape` must '
                         'correspond to the detector partitioning, '
                         'got {} and {}'.format(data.shape[-d:], part.shape))

    # find out if there are multiple data points
    if d < len(data.shape):
        n = data.shape[0]
    else:
        n = 1

    if n > 1:
        # extend detectors partition for multiple samples
        data_part = uniform_partition(np.append([0], part.min_pt),
                                      np.append([n], part.max_pt),
                                      np.append([n], part.shape),
                                      nodes_on_bdry=part.nodes_on_bdry)
    else:
        data_part = part

    if isinstance(old_detector, Flat1dDetector):
        assert isinstance(new_detector, CircularDetector)

        phi = new_detector.partition.meshgrid[0]
        u = new_detector.radius * np.tan(phi)
        interpolator = linear_interpolator(data, data_part.coord_vectors)
        if n > 1:
            i = data_part.meshgrid[0]
            u = u.reshape(1, -1)
            resampled_data = interpolator((i, u))
        else:
            resampled_data = interpolator(u)

    elif isinstance(old_detector, CircularDetector):
        assert isinstance(new_detector, Flat1dDetector)

        u = new_detector.partition.meshgrid
        phi = np.arctan2(u, old_detector.radius)
        interpolator = linear_interpolator(data, data_part.coord_vectors)
        if n > 1:
            i = data_part.meshgrid[0]
            phi = phi.reshape(-1, 1)
            resampled_data = interpolator((i, phi))
        else:
            resampled_data = interpolator(phi)

    elif isinstance(old_detector, Flat2dDetector):
        assert isinstance(new_detector, (CylindricalDetector,
                                         SphericalDetector))

        r = new_detector.radius
        if isinstance(new_detector, CylindricalDetector):
            phi, h = new_detector.partition.meshgrid
            u = r * np.tan(phi)
            v = h / r * np.sqrt(r * r + u * u)
        else:
            phi, theta = new_detector.partition.meshgrid
            u = r * np.tan(phi)
            v = np.tan(theta) * np.sqrt(r * r + u * u)
        interpolator = linear_interpolator(data, data_part.coord_vectors)
        if n > 1:
            i = data_part.meshgrid[0]
            u = np.expand_dims(u, axis=0)
            v = np.expand_dims(v, axis=0)
            resampled_data = interpolator((i, u, v))
        else:
            u = np.repeat(u, v.shape[1], axis=-1)
            coord = np.stack([u, v], axis=-1).reshape((-1, 2))
            resampled_data = interpolator(coord.T).reshape(
                new_detector.partition.shape)

    elif isinstance(old_detector, CylindricalDetector):
        assert isinstance(new_detector, (Flat2dDetector,
                                         SphericalDetector))

        r = old_detector.radius
        if isinstance(new_detector, Flat2dDetector):
            u, v = new_detector.partition.meshgrid
            phi = np.arctan2(u, r)
            h = v * r / np.sqrt(r * r + u * u)
        else:
            phi, theta = new_detector.partition.meshgrid
            h = r * np.tan(theta)
        interpolator = linear_interpolator(data, data_part.coord_vectors)
        if n > 1:
            i = data_part.meshgrid[0]
            phi = np.expand_dims(phi, axis=0)
            h = np.expand_dims(h, axis=0)
            resampled_data = interpolator((i, phi, h))
        else:
            phi = np.repeat(phi, h.shape[1], axis=-1)
            if h.shape[0] != phi.shape[0]:
                h = np.repeat(h, phi.shape[0], axis=0)
            coord = np.stack([phi, h], axis=-1).reshape((-1, 2))
            resampled_data = interpolator(coord.T).reshape(
                new_detector.partition.shape)

    elif isinstance(old_detector, SphericalDetector):
        assert isinstance(new_detector, (Flat2dDetector,
                                         CylindricalDetector))

        r = old_detector.radius
        if isinstance(new_detector, Flat2dDetector):
            u, v = new_detector.partition.meshgrid
            phi = np.arctan2(u, r)
            theta = np.arctan2(v, np.sqrt(r * r + u * u))
        else:
            phi, h = new_detector.partition.meshgrid
            theta = np.arctan2(h, r)
        interpolator = linear_interpolator(data, data_part.coord_vectors)
        if n > 1:
            i = data_part.meshgrid[0]
            phi = np.expand_dims(phi, axis=0)
            theta = np.expand_dims(theta, axis=0)
            resampled_data = interpolator((i, phi, theta))
        else:
            phi = np.repeat(phi, theta.shape[1], axis=-1)
            if theta.shape[0] != phi.shape[0]:
                theta = np.repeat(theta, phi.shape[0], axis=0)
            coord = np.stack([phi, theta], axis=-1).reshape((-1, 2))
            resampled_data = interpolator(coord.T).reshape(
                new_detector.partition.shape)

    else:
        NotImplementedError('Data transformation between detectors {} and {}'
                            'is not implemented'.format(old_detector,
                                                        new_detector))

    return resampled_data


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
