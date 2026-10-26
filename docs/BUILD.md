# Building the optional 3D solver

Archive analysis requires only the Python dependencies. Running a new 3D fracture simulation additionally requires a C11 compiler with C99 complex arithmetic, an OpenMP development runtime and Intel oneMKL with the LP64 PARDISO interface. Obtain these dependencies separately.

## CMake build

With oneMKL's environment initialized and CMake 3.21 or later:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
```

If oneMKL is not discovered, pass `-DMKL_DIR=<oneMKL CMake configuration directory>` or initialize the oneAPI environment. Intel documents the imported `MKL::MKL` target in its [oneMKL CMake guide](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2023-0/cmake-config-for-onemkl.html).

The provided configuration uses OpenMP for particle loops and defaults to sequential oneMKL calls, avoiding two distinct OpenMP runtimes in one process. To reproduce the archived runtime configuration with Intel-threaded MKL, use a matching Intel OpenMP toolchain and set `-DMKL_THREADING=intel_thread`. Runtime measurements depend on this choice. Ensure there is only one compatible OpenMP runtime in the process.

The source uses C99 complex arithmetic. Linux GCC/Clang or Windows Clang/MinGW are suitable toolchains. The MSVC C compiler does not implement the same C complex-number interface. The CMake path is provided for portability; the local release check used the direct Windows Clang-compatible build described below, not a separately tested CMake installation.

## Windows direct compiler build

For a GNU-target Clang-compatible compiler and suitable MKL/OpenMP headers and import libraries, the equivalent commands are:

```sh
clang -O3 -std=c11 -I "$MKL_INCLUDE" -c src/native/pardiso_wrapper.c -o build/pardiso_wrapper.o
clang -O3 -std=c11 -fopenmp -DPARDISO=pardiso_abi_safe \
  -I "$MKL_INCLUDE" -I "$OPENMP_INCLUDE" \
  src/native/cohesive_solver.c build/pardiso_wrapper.o \
  -L "$MKL_LIBRARY_DIR" -lmkl_rt -liomp5 -o build/cohesive_solver.exe
```

Set these paths to the installed headers/import libraries. Add the corresponding runtime DLL directory to `PATH`. The release check used Zig's `cc` front end with `-target x86_64-windows-gnu` and existing oneMKL/OpenMP headers/import libraries. Those local dependency files are not distributed here.

`pardiso_wrapper.c` includes the archived Windows GNU/Clang register-preservation workaround. The assembly is guarded by platform/compiler checks. Elsewhere the wrapper calls PARDISO directly. `PARDISO=pardiso_abi_safe` redirects the solver call without changing the numerical algorithm.

## Check before solving

```sh
python -m lpm_fracture export --config examples/smoke_ghost.json --output outputs/smoke/model.bin
python -m lpm_fracture solve --config examples/smoke_ghost.json --solver build/cohesive_solver.exe --model outputs/smoke/model.bin --output outputs/smoke/check --threads 2 --check
```

Inspect the log for `RESULT checkE`. This compares the energy gradient and tangent action with central differences. Release QA measured relative discrepancies near 10⁻¹⁰ for its small ghost model. This check is independent of the full loading calculation.

