"""Deskew utilities"""
import logging

from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

cl = None
try:
    import pyopencl as cl
    from pyopencl.array import empty, to_device

    logger.info("Using pyopencl for affine transforms")
except ImportError:
    try:

        from scipy.ndimage.interpolation import affine_transform

        logger.warning(
            "Could not import pyopencl. "
            "Falling back to scipy for CPU affine transforms"
        )
    except ImportError:
        logger.warning(
            "Could not import pyopencl or scipy."
            "Cannot perform deskew.  Please install one of those packages."
        )
        affine_transform = None

affine_source = """
#ifndef SAMPLER_FILTER
#define SAMPLER_FILTER CLK_FILTER_LINEAR
#endif

#ifndef SAMPLER_ADDRESS
#define SAMPLER_ADDRESS CLK_ADDRESS_CLAMP
#endif

#ifndef DTYPE
#define DTYPE float
#endif

__kernel void affine3D(__read_only image3d_t input, __global DTYPE *output,
                       __constant float *mat) {

  const sampler_t sampler = SAMPLER_ADDRESS | SAMPLER_FILTER;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint k = get_global_id(2);

  float x = i;
  float y = j;
  float z = k;

  float x2 = (mat[8] * z + mat[9] * y + mat[10] * x + mat[11]) + 0.5f;
  float y2 = (mat[4] * z + mat[5] * y + mat[6] * x + mat[7]) + 0.5f;
  float z2 = (mat[0] * z + mat[1] * y + mat[2] * x + mat[3]) + 0.5f;

  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);

  float4 coord_norm = (float4)(x2, y2, z2, 0.f);
  output[i + Nx * j + Nx * Ny * k] = read_imagef(input, sampler, coord_norm).x;
}
"""


class holder:
    pass


GPU = holder()


def get_best_device():
    return sorted(
        [
            device
            for platform in cl.get_platforms()
            for device in platform.get_devices()
        ],
        key=lambda x: x.type * 1e12 + x.get_info(cl.device_info.GLOBAL_MEM_SIZE),
        reverse=True,
    )[0]


def get_gpu(reload=False):
    if reload or not hasattr(GPU, "device"):
        GPU.device = get_best_device()
        GPU.ctx = cl.Context(devices=[GPU.device])
        GPU.queue = cl.CommandQueue(GPU.ctx)
    return GPU


@lru_cache(maxsize=128)
def get_affine_program(ctx, order: int = 1, mode="constant"):

    orders = [
        ["-D", "SAMPLER_FILTER=CLK_FILTER_NEAREST"],
        ["-D", "SAMPLER_FILTER=CLK_FILTER_LINEAR"],
    ]

    modes = {
        "constant": ["-D", "SAMPLER_ADDRESS=CLK_ADDRESS_CLAMP"],
        "wrap": ["-D", "SAMPLER_ADDRESS=CLK_ADDRESS_REPEAT"],
        "edge": ["-D", "SAMPLER_ADDRESS=CLK_ADDRESS_CLAMP_TO_EDGE"],
        "nearest": ["-D", "SAMPLER_ADDRESS=CLK_ADDRESS_CLAMP_TO_EDGE"],
        "mirror": ["-D", "SAMPLER_ADDRESS=CLK_ADDRESS_MIRRORED_REPEAT"],
    }

    affine_prg = cl.Program(ctx, affine_source)
    affine_prg.build(options=orders[order] + modes[mode])
    return affine_prg


def _debug_context(ctx):

    print(
        cl.get_supported_image_formats(
            ctx, cl.mem_flags.READ_WRITE, cl.mem_object_type.IMAGE3D
        )
    )

    for device in ctx.devices:
        print("DEVICE: ", device)
        for attr in dir(device):
            if attr.startswith("image"):
                print(f" {attr}", getattr(device, attr))


@lru_cache(maxsize=32)
def _get_image_format(ctx, num_channels, dtype, ndim, mode="rw"):
    """Maximize chance of finding a supported image format."""
    if mode == "rw":
        mode_flag = cl.mem_flags.READ_WRITE
    elif mode == "r":
        mode_flag = cl.mem_flags.READ_ONLY
    elif mode == "w":
        mode_flag = cl.mem_flags.WRITE_ONLY
    else:
        raise ValueError("invalid value '%s' for 'mode'" % mode)

    if ndim == 3:
        _dim = cl.mem_object_type.IMAGE3D
    elif ndim == 2:
        _dim = cl.mem_object_type.IMAGE2D
    elif ndim == 1:
        _dim = cl.mem_object_type.IMAGE1D
    else:
        raise ValueError(f"Unsupported number of image dimensions: {ndim}")

    supported_formats = cl.get_supported_image_formats(ctx, mode_flag, _dim)
    channel_type = cl.DTYPE_TO_CHANNEL_TYPE[dtype]

    if num_channels == 1:
        for order in [
            cl.channel_order.INTENSITY,
            cl.channel_order.R,
            cl.channel_order.Rx,
        ]:
            fmt = cl.ImageFormat(order, channel_type)
            if fmt in supported_formats:
                return fmt, 0
        fmt = cl.ImageFormat(cl.channel_order.RGBA, channel_type)
        if fmt in supported_formats:
            return fmt, 1
        raise ValueError(
            f"No supported ImageFormat found for dtype {dtype} with 1 channel\n",
            f"Supported formats include: {supported_formats}",
        )
    img_format = {
        2: cl.channel_order.RG,
        3: cl.channel_order.RGB,
        4: cl.channel_order.RGBA,
    }[num_channels]

    return cl.ImageFormat(img_format, channel_type), 0


# vendored from pyopencl.image_from_array so that we can change the img_format
# used for a single channel image to channel_order.INTENSITY
def _image_from_array(ctx, ary, num_channels=None, mode="r", norm_int=False):
    if not ary.flags.c_contiguous:
        raise ValueError("array must be C-contiguous")

    dtype = ary.dtype
    if num_channels is None:

        import pyopencl.cltypes

        try:
            dtype, num_channels = pyopencl.cltypes.vec_type_to_scalar_and_count[dtype]
        except KeyError:
            # It must be a scalar type then.
            num_channels = 1

        shape = ary.shape
        strides = ary.strides

    elif num_channels == 1:
        shape = ary.shape
        strides = ary.strides
    else:
        if ary.shape[-1] != num_channels:
            raise RuntimeError("last dimension must be equal to number of channels")

        shape = ary.shape[:-1]
        strides = ary.strides[:-1]

    if mode == "r":
        mode_flags = cl.mem_flags.READ_ONLY
    elif mode == "w":
        mode_flags = cl.mem_flags.WRITE_ONLY
    else:
        raise ValueError("invalid value '%s' for 'mode'" % mode)

    img_format, reshape = _get_image_format(ctx, num_channels, dtype, ary.ndim)
    if reshape:
        import warnings

        warnings.warn("Device support forced reshaping of single channel array to RGBA")
        ary = np.ascontiguousarray(np.repeat(ary[..., np.newaxis], 4, axis=-1))
        shape = ary.shape[:-1]
        strides = ary.strides[:-1]

    assert ary.strides[-1] == ary.dtype.itemsize

    return cl.Image(
        ctx,
        mode_flags | cl.mem_flags.COPY_HOST_PTR,
        img_format,
        shape=shape[::-1],
        pitches=strides[::-1][1:],
        hostbuf=ary,
    )


def image_from_array(arr, ctx, *args, **kwargs):

    if arr.ndim not in {2, 3, 4}:
        raise ValueError(
            "dimension of array wrong, should be 2 - 4 but is %s" % arr.ndim
        )
    if arr.dtype.type == np.complex64:
        num_channels = 2
        res = cl.Image.empty(arr.shape, dtype=np.float32, num_channels=num_channels)
        res.write_array(arr)
        res.dtype = np.float32
    else:
        num_channels = arr.shape[-1] if arr.ndim == 4 else 1
        res = _image_from_array(
            ctx, np.ascontiguousarray(arr), num_channels=num_channels, *args, **kwargs
        )
        res.dtype = arr.dtype

    res.num_channels = num_channels
    res.ndim = arr.ndim
    return res


if cl:

    def affine_transform(
        input,
        matrix,
        offset=0.0,
        output_shape=None,
        output=None,
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=None,
    ):
        """[summary]

        Parameters
        ----------
        input : array_like
            The input array.
        matrix : ndarray
            The inverse coordinate transformation matrix, mapping output
            coordinates to input coordinates. If ``ndim`` is the number of
            dimensions of ``input``, the given matrix must have one of the
            following shapes:
        offset : float or sequence, optional
            The offset into the array where the transform is applied. If a float,
            `offset` is the same for each axis. If a sequence, `offset` should
            contain one value for each axis.
        output_shape : tuple of ints, optional
            Shape tuple.
        output : array or dtype, optional
            The array in which to place the output, or the dtype of the returned array.
            By default an array of the same dtype as input will be created.
        order : int, optional
            The order of the spline interpolation, default is 0.
            The order has to be in the range 0-1.  (bi-cubic not yet supported)
        mode : {constant', 'nearest', 'mirror', 'wrap'}, optional
            The mode parameter determines how the input array is extended beyond its
            boundaries. Default is 'constant'. Behavior for each valid value is as follows:

            - 'constant' (k k k k | a b c d | k k k k)
                The input is extended by filling all values beyond the edge with the same
                constant value, defined by the cval parameter.
            - 'nearest' (a a a a | a b c d | d d d d)
                The input is extended by replicating the last pixel.
            - 'mirror' (d c b | a b c d | c b a)
                The input is extended by reflecting about the center of the last pixel.
            - 'wrap' (a b c d | a b c d | a b c d)
                The input is extended by wrapping around to the opposite edge.

        cval : scalar, optional
            Value to fill past edges of input if mode is ‘constant’. Default is 0.0.
        prefilter : bool, optional
            not supported
        """
        if order < 0 or order > 1:
            raise NotImplementedError(
                "spline orders other than 0 or 1 not yet supported"
            )
        out_shape = input.shape if output_shape is None else output_shape
        if prefilter is not None:
            raise NotImplementedError("prefilter is not yet supported")

        gpu = get_gpu()
        affine3D = get_affine_program(gpu.ctx, order, mode).affine3D
        dev_img = image_from_array(input.astype(np.float32, copy=False), gpu.ctx)
        res_g = empty(gpu.queue, out_shape, np.float32)
        mat_inv_g = to_device(gpu.queue, np.require(matrix, np.float32, "C"))
        affine3D(gpu.queue, out_shape[::-1], None, dev_img, res_g.data, mat_inv_g.data)
        return res_g.get()


def deskew_block(block, mat=None, out_shape=None, padval=0):
    extradims = block.ndim - 3
    last3dims = (0,) * extradims + (slice(None),) * 3
    array = block[last3dims]

    deskewed = affine_transform(array, mat, output_shape=tuple(out_shape[-3:]), order=0)
    return deskewed[(None,) * extradims + (...,)]


deskew_counter = 0


def get_deskew_func(shape, dz=0.5, dx=0.1, angle=31.5, padval=0):
    # calculate affine matrix from globals
    deskewFactor = np.cos(np.deg2rad(angle)) * dz / dx
    mat = np.eye(4)
    mat[2, 0] = -deskewFactor

    # calculate shape of output array
    (nz, ny, nx) = shape[-3:]
    out_shape = [1] * (len(shape) - 3) + list(shape[-3:])
    # new nx
    out_shape[-1] = np.int(np.floor((nz - 1) * -mat[2, 0]) + nx)
    new_dzdx_ratio = np.sin(np.deg2rad(angle)) * dz / dx

    def noisy_deskew(arr):
        # to see, set:  logging.getLogger("napari_ndtiffs").setLevel(logging.DEBUG)
        global deskew_counter
        deskew_counter += 1
        logger.debug(f"deskew #{deskew_counter}")
        return deskew_block(arr, mat=mat, out_shape=out_shape, padval=padval)

    return noisy_deskew, out_shape, new_dzdx_ratio
