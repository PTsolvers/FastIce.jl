# Library

## Running tests

### CPU tests

To run the FastIce test suite on the CPU, simple run `test` from within the package mode or using `Pkg`:
```julia-repl
using Pkg
Pkg.test("FastIce")
```

### GPU tests

To run the FastIce test suite on CUDA or ROC Backend (Nvidia or AMD GPUs), respectively, run the tests using `Pkg` adding following `test_args`:

#### For CUDA backend (Nvidia GPU):

```julia-repl
using Pkg
Pkg.test("FastIce"; test_args=["--backend=CUDA"])
```

#### For ROC backend (AMD GPU):

```julia-repl
using Pkg
Pkg.test("FastIce"; test_args=["--backend=AMDGPU"])
```
