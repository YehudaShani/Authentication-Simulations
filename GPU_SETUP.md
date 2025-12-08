# GPU Acceleration Setup Guide

This codebase supports GPU acceleration using CuPy for faster computation of optimal wallet occurrences.

## Installation

### Option 1: Using pip (Linux/Mac)

For CUDA 11.x:
```bash
pip install cupy-cuda11x
```

For CUDA 12.x:
```bash
pip install cupy-cuda12x
```

### Option 2: Using conda (Recommended)

```bash
conda install -c conda-forge cupy
```

### Option 3: Windows

On Windows, you may need to install CuPy manually. Check the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) for Windows-specific instructions.

## Usage

The code automatically detects if CuPy is available and will use GPU acceleration if:
1. CuPy is installed
2. A compatible NVIDIA GPU with CUDA support is available
3. `use_gpu=True` is set in the main function

### Configuration

In `main.py`, you can control GPU usage:

```python
use_gpu = True  # Set to False to force CPU
batch_size = 1000  # Number of scenarios to process per GPU batch
```

### Performance Tips

1. **Batch Size**: Adjust `batch_size` based on your GPU memory:
   - Larger batch sizes = faster but more GPU memory usage
   - Smaller batch sizes = slower but less memory usage
   - Default: 1000 scenarios per batch

2. **Memory Management**: If you run out of GPU memory:
   - Reduce `batch_size`
   - Process fewer scenarios at once
   - Use CPU fallback (`use_gpu=False`)

3. **GPU Selection**: If you have multiple GPUs, CuPy will use GPU 0 by default. To use a different GPU:
   ```python
   import cupy as cp
   cp.cuda.Device(1).use()  # Use GPU 1
   ```

## How It Works

The GPU acceleration works by:
1. **Vectorization**: Converting wallet coverage checks to vectorized array operations
2. **Batch Processing**: Processing multiple probability scenarios in parallel
3. **Memory Efficiency**: Transferring data to/from GPU in batches to manage memory

## Fallback Behavior

If CuPy is not available or GPU is not detected, the code automatically falls back to CPU computation using NumPy. No code changes are required.

## Troubleshooting

### "CuPy not available" message
- Install CuPy following the instructions above
- Verify CUDA is installed: `nvcc --version`
- Check GPU is detected: `nvidia-smi`

### Out of memory errors
- Reduce `batch_size` in `main.py`
- Process fewer scenarios at once
- Use CPU fallback

### Performance not improved
- Ensure you're processing a large number of scenarios (GPU overhead is only worth it for large batches)
- Check GPU utilization: `nvidia-smi` while running
- Verify data is actually being processed on GPU (check console output)

