import cupy as cp
from cupyx.scipy import ndimage as ndi_gpu


def affine_transform(
    input,
    matrix,
    offset=0.0,
    output_shape=None,
    output=None,
    order=3,
    mode="constant",
    cval=0.0,
    prefilter=True,
    **kwargs,
):
    """CuPy affine_transform wrapper that handles host/device transfers.

    Parameters
    ----------
    input : ndarray
        The input image. It will be transfered to the GPU if it is not already
        a cupy.ndarray.
    matrix : ndarray
        The inverse coordinate transformation matrix, mapping output
        coordinates to input coordinates. It will be transfered to the GPU if
        it is not already a cupy.ndarray. If ``ndim`` is the number of
        dimensions of ``input``, the given matrix must have one of the
        following shapes:

            - ``(ndim, ndim)``: the linear transformation matrix for each
              output coordinate.
            - ``(ndim,)``: assume that the 2-D transformation matrix is
              diagonal, with the diagonal specified by the given value. A more
              efficient algorithm is then used that exploits the separability
              of the problem.
            - ``(ndim + 1, ndim + 1)``: assume that the transformation is
              specified using homogeneous coordinates [1]_. In this case, any
              value passed to ``offset`` is ignored.
            - ``(ndim, ndim + 1)``: as above, but the bottom row of a
              homogeneous transformation matrix is always ``[0, 0, ..., 1]``,
              and may be omitted.

    offset : float or sequence, optional
        The offset into the array where the transform is applied. If a float,
        `offset` is the same for each axis. If a sequence, `offset` should
        contain one value for each axis.
    output_shape : tuple of ints, optional
        Shape tuple.
    output : cupy.ndarray or dtype
        The array in which to place the output, or the dtype of the returned
        array.
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    mode : {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', \
               'mirror', 'grid-wrap', 'wrap'}, optional
        The `mode` parameter determines how the input array is extended
        beyond its boundaries. Default is 'constant'. Behavior for each valid
        value is as follows (see additional plots and details on
        :ref:`boundary modes <ndimage-interpolation-modes>`):

        'reflect' (`d c b a | a b c d | d c b a`)
            The input is extended by reflecting about the edge of the last
            pixel. This mode is also sometimes referred to as half-sample
            symmetric.

        'grid-mirror'
            This is a synonym for 'reflect'.

        'constant' (`k k k k | a b c d | k k k k`)
            The input is extended by filling all values beyond the edge with
            the same constant value, defined by the `cval` parameter. No
            interpolation is performed beyond the edges of the input.

        'grid-constant' (`k k k k | a b c d | k k k k`)
            The input is extended by filling all values beyond the edge with
            the same constant value, defined by the `cval` parameter.
            Interpolation occurs for samples outside the input's extent as
            well.

        'nearest' (`a a a a | a b c d | d d d d`)
            The input is extended by replicating the last pixel.

        'mirror' (`d c b | a b c d | c b a`)
            The input is extended by reflecting about the center of the last
            pixel. This mode is also sometimes referred to as whole-sample
            symmetric.

        'grid-wrap' (`a b c d | a b c d | a b c d`)
            The input is extended by wrapping around to the opposite edge.

        'wrap' (`d b c d | a b c d | b c a b`)
            The input is extended by wrapping around to the opposite edge, but
            in a way such that the last point and initial point exactly
            overlap. In this case it is not well defined which sample will be
            chosen at the point of overlap.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.
    prefilter : bool, optional
        Determines if the input array is prefiltered with `spline_filter`
        before interpolation. The default is True, which will create a
        temporary `float64` array of filtered values if `order > 1`. If
        setting this to False, the output will be slightly blurred if
        `order > 1`, unless the input is prefiltered, i.e. it is the result
        of calling `spline_filter` on the original input.

    Returns
    -------
    out : numpy.ndarray
        The transformed image.

    Other Parameters
    ----------------
    texture_memory : bool, optional
        If True, uses GPU texture memory. Supports only:

            - Only available in CuPy >= v10.0.0b2
            - 2D and 3D float32 arrays as input
            - ``(ndim + 1, ndim + 1)`` homogeneous float32 transformation
                matrix
            - ``mode='constant'`` and ``mode='nearest'``
            - ``order=0`` (nearest neighbor) and ``order=1`` (linear
                interpolation)
            - NVIDIA CUDA GPUs

    Notes
    -----
    The full set of arguments available in SciPy are supported. See
    ``cupyx.scipy.ndimage.affine_transform`` documentation for full
    description of the available *args and **kwargs.
    """
    return cp.asnumpy(
        ndi_gpu.affine_transform(
            cp.asarray(input),
            cp.asarray(matrix),
            offset=offset,
            output_shape=output_shape,
            output=output,
            order=order,
            mode=mode,
            cval=cval,
            prefilter=prefilter,
            **kwargs,
        )
    )
