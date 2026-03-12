# FHT - Fast Hadamard Transform

High-performance Fast Hadamard Transform library with SIMD-optimized implementations for x86 (SSE/AVX) and ARM (NEON), and Python bindings via nanobind.

## Install

```bash
pip install fht_cpu
```

From source:

```bash
git clone https://github.com/grigori-hpnalgs-lab/fht_cpu.git
cd fht
pip install .
```

## Python Usage

```python
import numpy as np
from fht import fht

# 1D transform (in-place)
x = np.random.randn(1024).astype(np.float32)
fht(x)

# Allocating mode (returns new array, original unchanged)
y = fht(x, inplace=False)

# Preallocated output
out = np.empty_like(x)
fht(x, out=out)

# 2D — each row transformed in parallel via OpenMP
X = np.random.randn(1000, 2**16).astype(np.float32)
fht(X, axis=-1)

# Control parallelism: -1 = all cores (default), 0 or 1 = single-threaded, N = exact thread count
fht(X, axis=-1, num_threads=4)

# Complex arrays (decomposes into real/imag, transforms separately)
z = np.random.randn(512).astype(np.complex128)
fht(z)
```

Supported dtypes: `float32`, `float64`, `complex64`, `complex128`.

The transform axis must have a power-of-2 length. For 2D arrays, rows (or columns) are processed in parallel with OpenMP. Thread count is set with the `num_threads` parameter.

## C/C++ Usage

Header-only. Just include and compile:

```cpp
#include <fht/fht.h>

float buf[1024];
fht_float(buf, 10);  // log2(1024) = 10
```

### C API

```c
int fht_float(float *buf, int log_n);
int fht_double(double *buf, int log_n);
int fht_float_oop(float *in, float *out, int log_n);
int fht_double_oop(double *in, double *out, int log_n);
```

Returns 0 on success, 1 on invalid `log_n` (valid range: 0-30).

### CMake Integration

```cmake
# Via CPM (recommended)
CPMAddPackage("gh:grigori-hpnalgs-lab/fht@1.0.0")
target_link_libraries(myapp PRIVATE fht::fht)

# Or as subdirectory
add_subdirectory(fht)
target_link_libraries(myapp PRIVATE fht::fht)
```

Compile with `-mavx` on x86 for best performance.


## Platform Support

| Platform | Float | Double |
|----------|-------|--------|
| x86_64 + AVX | yes | yes | 
| x86_64 + SSE | yes | yes |
| ARM64 (NEON) | yes | yes |


## Re-optimizing the code for your CPU

```bash
cmake -B build -DFHT_OPTIMIZE_FOR_HOST=ON
cmake --build build
```

## Limitations

Known issues:

- [ ] **`inplace=False` / `out=` does copy + in-place** — we will add an out of place version soon, it just requires some changes to the codegen files.
- [ ] **Complex number support is not optimal** — we will provide separate kernels for complex, currently we deinterleave into real and imaginary parts, apply the transform and then recombine. 

## Acknowledgments

The x86 AVX/SSE implementation is based on [FFHT](https://github.com/FALCONN-LIB/FFHT) from the [FALCONN](https://github.com/FALCONN-LIB/FALCONN) project by Alexandr Andoni, Piotr Indyk, Thijs Laarhoven, Ilya Razenshteyn, and Ludwig Schmidt. The original code was copied and integrated with minor modifications.

The ARM NEON implementation was written from scratch with auto-tuned code generation.

## License

See [LICENSE](LICENSE).
