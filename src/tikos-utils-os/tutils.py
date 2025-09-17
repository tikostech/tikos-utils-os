import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import logging
from typing import Tuple
import datetime

# This allows users of the library to control log output
log = logging.getLogger(__name__)

# --- CUDA Kernels ---

CUDA_KERNEL_STREAMING_MULTIPLY = """
__global__ void elementwise_multiply_streaming(
    // Input/Output pointers for the current slice
    float *result_slice, 
    const float *a_slice, 
    const float *b_slice,

    // The global offset of this slice
    int slice_offset_i,

    // Dimensions of the ORIGINAL full tensors (for padding logic)
    int a_full_dim0, int a_full_dim1, int a_full_dim2,
    int b_full_dim0, int b_full_dim1, int b_full_dim2,

    // Dimensions of the final RESULT tensor (for indexing)
    int res_dim1, int res_dim2,

    // The actual height of the current slice being processed
    int slice_dim0_actual)
{
    // Calculate the 3D index for the current thread
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i_local = blockIdx.z * blockDim.z + threadIdx.z;

    // Calculate the conceptual 'global' row index in the full result tensor
    int i_global = i_local + slice_offset_i;

    // Primary bounds check to prevent writing outside the allocated slice
    if (i_local < slice_dim0_actual && j < res_dim1 && k < res_dim2) {
        float val_a = 0.0f;
        float val_b = 0.0f;

        // Check if the global index is within the bounds of original tensor A
        if (i_global < a_full_dim0 && j < a_full_dim1 && k < a_full_dim2) {
            // Index into the slice using the local row index 'i_local'
            long long idx_a_slice = (long long)i_local * a_full_dim1 * a_full_dim2 + (long long)j * a_full_dim2 + k;
            val_a = a_slice[idx_a_slice];
        }

        // Check if the global index is within the bounds of original tensor B
        if (i_global < b_full_dim0 && j < b_full_dim1 && k < b_full_dim2) {
            long long idx_b_slice = (long long)i_local * b_full_dim1 * b_full_dim2 + (long long)j * b_full_dim2 + k;
            val_b = b_slice[idx_b_slice];
        }

        long long res_idx_slice = (long long)i_local * res_dim1 * res_dim2 + (long long)j * res_dim2 + k;
        result_slice[res_idx_slice] = val_a * val_b;
    }
}
"""

CUDA_KERNEL_STREAMING_DIVIDE = """
__global__ void elementwise_divide_streaming(
    // Input/Output pointers for the current slice
    float *result_slice, 
    const float *a_slice, 
    const float *b_slice,

    // The global offset of this slice
    int slice_offset_i,

    // Dimensions of the ORIGINAL full tensors (for padding logic)
    int a_full_dim0, int a_full_dim1, int a_full_dim2,
    int b_full_dim0, int b_full_dim1, int b_full_dim2,

    // Dimensions of the final RESULT tensor (for indexing)
    int res_dim1, int res_dim2,

    // The actual height of the current slice being processed
    int slice_dim0_actual)
{
    // Calculate the 3D index for the current thread
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i_local = blockIdx.z * blockDim.z + threadIdx.z; // Row index within THIS slice

    // Calculate the conceptual 'global' row index in the full result tensor
    int i_global = i_local + slice_offset_i;

    // Primary bounds check to prevent writing outside the allocated slice
    if (i_local < slice_dim0_actual && j < res_dim1 && k < res_dim2) {
        float val_a = 0.0f; // Numerator
        float val_b = 0.0f; // Denominator

        // Implicit Padding Logic for Tensor A (Numerator)
        if (i_global < a_full_dim0 && j < a_full_dim1 && k < a_full_dim2) {
            long long idx_a_slice = (long long)i_local * a_full_dim1 * a_full_dim2 + (long long)j * a_full_dim2 + k;
            val_a = a_slice[idx_a_slice];
        }

        // Implicit Padding Logic for Tensor B (Denominator)
        if (i_global < b_full_dim0 && j < b_full_dim1 && k < b_full_dim2) {
            long long idx_b_slice = (long long)i_local * b_full_dim1 * b_full_dim2 + (long long)j * b_full_dim2 + k;
            val_b = b_slice[idx_b_slice];
        }

        // Calculate the flattened index to write to the output slice
        long long res_idx_slice = (long long)i_local * res_dim1 * res_dim2 + (long long)j * res_dim2 + k;

        // --- KEY CHANGE: Perform division with a safety check ---
        if (val_b != 0.0f) {
            result_slice[res_idx_slice] = val_a / val_b;
        } else {
            // Define a safe result for division by zero (e.g., 0.0)
            result_slice[res_idx_slice] = 0.0f;
        }
    }
}
"""


def _execute_streaming_op(tensor_a: np.ndarray, tensor_b: np.ndarray, kernel_string: str, kernel_name: str,
                          slice_size_mb: int = 128) -> np.ndarray:
    """
    Private worker function that executes a given CUDA kernel using a streaming
    memory strategy to minimize VRAM usage.

    Args:
        tensor_a: The first input NumPy array.
        tensor_b: The second input NumPy array.
        kernel_string: The string containing the CUDA C++ kernel code.
        kernel_name: The name of the kernel function to execute.
        slice_size_mb: The approximate VRAM size in MB for each slice.

    Returns:
        A new NumPy array containing the result of the operation.

    Raises:
        ValueError: If input tensors have more than 3 dimensions or are not
                    NumPy arrays.
        Exception: Propagates exceptions from PyCUDA during execution.
    """
    log.info("--- CPU: Preparing dimensions and host memory ---")

    if not isinstance(tensor_a, np.ndarray) or not isinstance(tensor_b, np.ndarray):
        raise TypeError("Input tensors must be NumPy ndarrays.")

    tensor_a = np.asarray(tensor_a, dtype=np.float32)
    tensor_b = np.asarray(tensor_b, dtype=np.float32)

    if tensor_a.ndim > 3 or tensor_b.ndim > 3:
        raise ValueError(f"Inputs must not have more than 3 dimensions, "
                         f"got {tensor_a.ndim} and {tensor_b.ndim}.")

    # Unify the number of dimensions by adding new axes
    temp_a, temp_b = tensor_a, tensor_b
    while temp_a.ndim < temp_b.ndim:
        temp_a = np.expand_dims(temp_a, axis=0)
    while temp_b.ndim < temp_a.ndim:
        temp_b = np.expand_dims(temp_b, axis=0)

    full_shape_a = temp_a.shape
    full_shape_b = temp_b.shape

    target_shape = tuple(np.maximum(full_shape_a, full_shape_b))
    result_host = np.empty(target_shape, dtype=np.float32)

    log.info(f"Input shape A: {tensor_a.shape} -> Unified: {full_shape_a}")
    log.info(f"Input shape B: {tensor_b.shape} -> Unified: {full_shape_b}")
    log.info(f"Target result shape: {target_shape}")

    log.info("--- GPU: Compiling kernel ---")
    module = SourceModule(kernel_string)
    kernel = module.get_function(kernel_name)

    log.info("--- GPU: Processing result in streaming batches ---")

    bytes_per_float = np.dtype(np.float32).itemsize
    elements_per_row = np.prod(target_shape[1:]) if len(target_shape) > 1 else 1
    if elements_per_row == 0: elements_per_row = 1

    slice_elements = (slice_size_mb * 1024 * 1024) // bytes_per_float
    slice_dim0 = max(1, int(slice_elements // elements_per_row))

    BLOCK_DIMS: Tuple[int, int, int] = (8, 8, 8)

    gpustart = datetime.datetime.now()

    for i_offset in range(0, target_shape[0], slice_dim0):
        current_dim0 = min(slice_dim0, target_shape[0] - i_offset)
        i_end = i_offset + current_dim0
        log.debug(f"  Processing global rows {i_offset} to {i_end}...")

        a_slice_gpu, b_slice_gpu, result_slice_gpu = None, None, None
        try:
            # Prepare HOST slices of the inputs
            a_slice_host = temp_a[i_offset:i_end, ...]
            b_slice_host = temp_b[i_offset:i_end, ...]

            # Allocate GPU memory for THIS BATCH ONLY
            a_slice_gpu = cuda.mem_alloc(a_slice_host.nbytes)
            b_slice_gpu = cuda.mem_alloc(b_slice_host.nbytes)
            result_slice_bytes = current_dim0 * elements_per_row * bytes_per_float
            result_slice_gpu = cuda.mem_alloc(int(result_slice_bytes))

            cuda.memcpy_htod(a_slice_gpu, a_slice_host)
            cuda.memcpy_htod(b_slice_gpu, b_slice_host)

            # Pad shape tuples to ensure they always have 3 elements for the kernel
            padded_full_shape_a = full_shape_a + (1,) * (3 - len(full_shape_a))
            padded_full_shape_b = full_shape_b + (1,) * (3 - len(full_shape_b))
            padded_target_shape = target_shape + (1,) * (3 - len(target_shape))

            grid_dims: Tuple[int, int, int] = (
                int((padded_target_shape[2] + BLOCK_DIMS[0] - 1) // BLOCK_DIMS[0]),
                int((padded_target_shape[1] + BLOCK_DIMS[1] - 1) // BLOCK_DIMS[1]),
                int((current_dim0 + BLOCK_DIMS[2] - 1) // BLOCK_DIMS[2])
            )

            kernel(
                result_slice_gpu, a_slice_gpu, b_slice_gpu,
                np.int32(i_offset),
                np.int32(padded_full_shape_a[0]), np.int32(padded_full_shape_a[1]), np.int32(padded_full_shape_a[2]),
                np.int32(padded_full_shape_b[0]), np.int32(padded_full_shape_b[1]), np.int32(padded_full_shape_b[2]),
                np.int32(padded_target_shape[1]), np.int32(padded_target_shape[2]),
                np.int32(current_dim0),
                block=BLOCK_DIMS, grid=grid_dims
            )

            cuda.memcpy_dtoh(result_host[i_offset:i_end, ...], result_slice_gpu)

        finally:
            # CRITICAL: Ensure memory is freed even if an error occurs
            if a_slice_gpu: a_slice_gpu.free()
            if b_slice_gpu: b_slice_gpu.free()
            if result_slice_gpu: result_slice_gpu.free()

    gpuend = datetime.datetime.now()
    log.info("--- GPU: Streaming complete. ---")
    log.info(f"GPU Process time: {gpuend - gpustart}")

    return result_host


def multiply(tensor_a: np.ndarray, tensor_b: np.ndarray, slice_size_mb: int = 128) -> np.ndarray:
    """Computes the padded element-wise product of two tensors by streaming
    slices to and from the GPU to conserve VRAM.

    This method is ideal for tensors that are too large to fit entirely in
    GPU memory. It implicitly handles broadcasting rules similar to NumPy.

    Args:
        tensor_a: The first input tensor.
        tensor_b: The second input tensor.
        slice_size_mb: The maximum VRAM size in megabytes for
            each processing slice. A smaller size reduces VRAM usage but may
            increase execution time due to overhead. Defaults to 128.

    Returns:
        A new tensor containing the element-wise product.
    """
    return _execute_streaming_op(
        tensor_a,
        tensor_b,
        CUDA_KERNEL_STREAMING_MULTIPLY,
        "elementwise_multiply_streaming",
        slice_size_mb=slice_size_mb
    )


def divide(tensor_a: np.ndarray, tensor_b: np.ndarray, slice_size_mb: int = 128) -> np.ndarray:
    """Computes the padded element-wise division of two tensors by streaming
    slices to and from the GPU. Division by zero results in 0.0.

    This method is ideal for tensors that are too large to fit entirely in
    GPU memory. It implicitly handles broadcasting rules similar to NumPy.

    Args:
        tensor_a: The numerator tensor.
        tensor_b: The denominator tensor.
        slice_size_mb: The maximum VRAM size in megabytes for
            each processing slice. A smaller size reduces VRAM usage but may
            increase execution time due to overhead. Defaults to 128.

    Returns:
        A new tensor containing the element-wise quotient.
    """
    return _execute_streaming_op(
        tensor_a,
        tensor_b,
        CUDA_KERNEL_STREAMING_DIVIDE,
        "elementwise_divide_streaming",
        slice_size_mb=slice_size_mb
    )


