# NLSE++
C++ library for solving the nonlinear Schrödinger equation

[![Linux](https://github.com/savithru-j/nlse-cpp/actions/workflows/linux.yml/badge.svg?branch=main)](https://github.com/savithru-j/nlse-cpp/actions/workflows/linux.yml)
[![Windows](https://github.com/savithru-j/nlse-cpp/actions/workflows/windows.yml/badge.svg?branch=main)](https://github.com/savithru-j/nlse-cpp/actions/workflows/windows.yml)

### Building with CMake

```
git clone https://github.com/savithru-j/nlse-cpp.git
cd nlse-cpp              #Main project directory
mkdir build              #Create directory for build files: build/release
cd build
mkdir release 
cd release
cmake ../../             #Configure build
cmake --build . -j 4     #Build in parallel with 4 threads
```