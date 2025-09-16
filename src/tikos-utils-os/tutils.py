import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import logging

# Configure basic logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CUDA KERNELS ---

_MULTIPLY_KERNEL = """
__global__ void elementwise_multiply_batching(
    float *result_slice, 
    int slice_offset_i,
    const float *a, 
    const float *b,
    int a_dim0, int a_dim1, int a_dim2,
    int b_dim0, int b_dim1, int b_dim2,
    int res_dim1, int res_dim2,
    int slice_dim0_actual)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i_local = blockIdx.z * blockDim.z + threadIdx.z;
    int i_global = i_local + slice_offset_i;

    if (i_local < slice_dim0_actual && j < res_dim1 && k < res_dim2) {
        float val_a = 0.0f;
        if (i_global < a_dim0 && j < a_dim1 && k < a_dim2) {
            long long idx_a = (long long)i_global * a_dim1 * a_dim2 + (long long)j * a_dim2 + k;
            val_a = a[idx_a];
        }

        float val_b = 0.0f;
        if (i_global < b_dim0 && j < b_dim1 && k < b_dim2) {
            long long idx_b = (long long)i_global * b_dim1 * b_dim2 + (long long)j * b_dim2 + k;
            val_b = b[idx_b];
        }

        long long res_idx_slice = (long long)i_local * res_dim1 * res_dim2 + (long long)j * res_dim2 + k;
        result_slice[res_idx_slice] = val_a * val_b;
    }
}
"""

_DIVIDE_KERNEL = """
__global__ void elementwise_divide_batching(
    float *result_slice, 
    int slice_offset_i,
    const float *a, 
    const float *b,
    int a_dim0, int a_dim1, int a_dim2,
    int b_dim0, int b_dim1, int b_dim2,
    int res_dim1, int res_dim2,
    int slice_dim0_actual)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i_local = blockIdx.z * blockDim.z + threadIdx.z;
    int i_global = i_local + slice_offset_i;

    if (i_local < slice_dim0_actual && j < res_dim1 && k < res_dim2) {
        float val_a = 0.0f;
        if (i_global < a_dim0 && j < a_dim1 && k < a_dim2) {
            long long idx_a = (long long)i_global * a_dim1 * a_dim2 + (long long)j * a_dim2 + k;
            val_a = a[idx_a];
        }

        float val_b = 0.0f;
        if (i_global < b_dim0 && j < b_dim1 && k < b_dim2) {
            long long idx_b = (long long)i_global * b_dim1 * b_dim2 + (long long)j * b_dim2 + k;
            val_b = b[idx_b];
        }

        long long res_idx_slice = (long long)i_local * res_dim1 * res_dim2 + (long long)j * res_dim2 + k;

        if (val_b != 0.0f) {
            result_slice[res_idx_slice] = val_a / val_b;
        } else {
            result_slice[res_idx_slice] = 0.0f;
        }
    }
}
"""


def _execute_batched_op(tensor_a: np.ndarray, tensor_b: np.ndarray, kernel_string: str, kernel_name: str,
                        slice_size_mb: int, verbose: bool) -> np.ndarray:
    """Internal function to handle the common logic for batched GPU operations."""
    log = logging.getLogger()
    if verbose:
        log.setLevel(logging.INFO)
    else:
        log.setLevel(logging.WARNING)

    # --- Validation and Preparation ---
    log.info("Validating inputs and preparing dimensions.")
    if not isinstance(tensor_a, np.ndarray) or not isinstance(tensor_b, np.ndarray):
        raise TypeError("Inputs must be NumPy arrays.")

    h_a = np.asarray(tensor_a, dtype=np.float32)
    h_b = np.asarray(tensor_b, dtype=np.float32)

    while h_a.ndim < h_b.ndim:
        h_a = np.expand_dims(h_a, axis=0)
    while h_b.ndim < h_a.ndim:
        h_b = np.expand_dims(h_b, axis=0)

    if h_a.ndim > 3:
        raise ValueError("This implementation currently supports up to 3D tensors.")

    # Pad dimensions to 3 if they are smaller
    while h_a.ndim < 3: h_a = np.expand_dims(h_a, axis=0)
    while h_b.ndim < 3: h_b = np.expand_dims(h_b, axis=0)

    target_shape = tuple(np.maximum(h_a.shape, h_b.shape))
    result_host = np.empty(target_shape, dtype=np.float32)
    log.info(f"Target shape for operation: {target_shape}")

    # --- GPU Setup ---
    log.info("Compiling CUDA kernel and allocating persistent GPU memory.")
    d_a = cuda.mem_alloc(h_a.nbytes)
    d_b = cuda.mem_alloc(h_b.nbytes)
    cuda.memcpy_htod(d_a, h_a)
    cuda.memcpy_htod(d_b, h_b)

    module = SourceModule(kernel_string)
    kernel = module.get_function(kernel_name)

    # --- Batch Processing ---
    log.info(f"Processing in slices of max {slice_size_mb} MB.")
    bytes_per_float = np.dtype(np.float32).itemsize
    slice_elements = (slice_size_mb * 1024 * 1024) // bytes_per_float
    elements_per_row = target_shape[1] * target_shape[2]
    slice_dim0 = max(1, slice_elements // elements_per_row)

    BLOCK_DIMS = (8, 8, 8)

    total_time = 0

    for i_offset in range(0, target_shape[0], slice_dim0):
        current_dim0 = min(slice_dim0, target_shape[0] - i_offset)
        log.info(f"  Processing slice: start_row={i_offset}, height={current_dim0}")

        slice_bytes = current_dim0 * elements_per_row * bytes_per_float
        d_result_slice = cuda.mem_alloc(int(slice_bytes))

        grid_dims = (
            (target_shape[2] + BLOCK_DIMS[0] - 1) // BLOCK_DIMS[0],
            (target_shape[1] + BLOCK_DIMS[1] - 1) // BLOCK_DIMS[1],
            (current_dim0 + BLOCK_DIMS[2] - 1) // BLOCK_DIMS[2]
        )

        start_event = cuda.Event()
        end_event = cuda.Event()
        start_event.record()

        kernel(
            d_result_slice, np.int32(i_offset),
            d_a, d_b,
            np.int32(h_a.shape[0]), np.int32(h_a.shape[1]), np.int32(h_a.shape[2]),
            np.int32(h_b.shape[0]), np.int32(h_b.shape[1]), np.int32(h_b.shape[2]),
            np.int32(target_shape[1]), np.int32(target_shape[2]),
            np.int32(current_dim0),
            block=BLOCK_DIMS, grid=grid_dims
        )

        end_event.record()
        end_event.synchronize()
        total_time += start_event.time_till(end_event)

        cuda.memcpy_dtoh(result_host[i_offset: i_offset + current_dim0], d_result_slice)
        d_result_slice.free()

    log.info(f"Total GPU kernel time: {total_time / 1000.0:.6f} seconds")

    # --- 4. Cleanup ---
    log.info("Batch processing complete. Freeing persistent GPU memory.")
    d_a.free()
    d_b.free()

    return result_host


def multiply(tensor_a: np.ndarray, tensor_b: np.ndarray, slice_size_mb: int = 64, verbose: bool = False) -> np.ndarray:
    """
    Computes the padded element-wise product of two tensors by processing the
    result in slices to conserve GPU VRAM. Handles broadcasting up to 3D.

    Args:
        tensor_a (np.ndarray): The first tensor.
        tensor_b (np.ndarray): The second tensor.
        slice_size_mb (int): The maximum VRAM (in MB) to use for each result slice.
        verbose (bool): If True, enables INFO-level logging for detailed steps.

    Returns:
        np.ndarray: The result of the element-wise multiplication.
    """
    return _execute_batched_op(tensor_a, tensor_b, _MULTIPLY_KERNEL, "elementwise_multiply_batching", slice_size_mb,
                               verbose)


def divide(tensor_a: np.ndarray, tensor_b: np.ndarray, slice_size_mb: int = 64, verbose: bool = False) -> np.ndarray:
    """
    Computes the padded element-wise division (a / b) of two tensors by
    processing the result in slices. Division by zero results in 0.0.

    Args:
        tensor_a (np.ndarray): The numerator tensor.
        tensor_b (np.ndarray): The denominator tensor.
        slice_size_mb (int): The maximum VRAM (in MB) to use for each result slice.
        verbose (bool): If True, enables INFO-level logging for detailed steps.

    Returns:
        np.ndarray: The result of the element-wise division.
    """
    return _execute_batched_op(tensor_a, tensor_b, _DIVIDE_KERNEL, "elementwise_divide_batching", slice_size_mb,
                               verbose)

