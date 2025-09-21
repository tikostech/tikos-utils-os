# tikos-utils-os
# Open Tikos Utilities: (tikos-utils-os)
# CUDA Batched Tensor Operations
A PyPI package that leverages **PyCUDA** for memory-efficient, element-wise tensor operations with unequal dimensions (multiplication and division) on NVIDIA GPUs.

This package is designed for large tensors that may not fit entirely in GPU VRAM. It processes the element-wise operations in slices (batches), ensuring that memory usage remains predictable and constrained. It implicitly handles broadcasting rules similar to NumPy, padding smaller tensors to match the shape of larger ones during the operation.

---

## Installation

```bash
pip install tikos-utils-os
```

Note: This is a placeholder name. Once published, you would install it via the name you choose on PyPI.

## Prerequisites
You must have the NVIDIA CUDA Toolkit installed and your environment correctly configured for PyCUDA to function.

## Usage
Here are examples of how to use the `multiply` and `divide` functions.

### Example: Element-wise Multiplication
```bash
import numpy as np
from tikos-utils-os import multiply

# Create two tensors of different shapes
tensor_a = np.random.randn(500, 10).astype(np.float32)
tensor_b = np.random.randn(160, 12800, 640).astype(np.float32)

# Perform memory-optimized multiplication on the GPU.
# The 2D tensor 'a' will be broadcast to match the 3D tensor 'b'.
# verbose=True prints logs and timing information.
# slice_size_mb controls the VRAM used for each batch.
product = multiply(tensor_a, tensor_b, slice_size_mb=128, verbose=True)

print("\nMultiplication complete.")
print(f"Result shape: {product.shape}")
```

### Example: Safe Element-wise Division
```bash
import numpy as np
from tikos-utils-os import divide

# Create two tensors
numerator = np.full((100, 200, 300), 10, dtype=np.float32)
denominator = np.random.randn(100, 200, 300).astype(np.float32)

# Introduce some zeros into the denominator to test safety
denominator[10, 20, 30] = 0.0

# Perform safe division. The kernel ensures that division by zero results in 0.0.
result = divide(numerator, denominator, slice_size_mb=64, verbose=True)

print("\nDivision complete.")
# The value at result[10, 20, 30] should be 0 because of the safe division.
print(f"Value at result[10, 20, 30]: {result[10, 20, 30]}")
assert result[10, 20, 30] == 0.0
print("Verification successful!")
```
