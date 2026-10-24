# Strata - the project's version, and a pointer to the engine's build.
#
# This fork runs on one machine type: an Intel Arc Pro B70 (Battlemage, bmg-g31) on Linux. The engine is the SYCL
# build in sycl/ (sycl/CMakeLists.txt, built by sycl/tools/build.sh in the strata-sycl-dev image; docs/B70.md).
# The CUDA / HIP build that used to live here is gone. This file stays because the project's version is read from
# the project() line below: by sycl/CMakeLists.txt, setup.py, strata_cli and the packaging scripts.
cmake_minimum_required(VERSION 3.24)
project(strata VERSION 0.1.49.2 LANGUAGES CXX)
message(FATAL_ERROR "Strata's engine is built from sycl/: cmake -S sycl -B build-sycl-aot -G Ninja "
                    "-DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_AOT=bmg-g31 "
                    "(or sycl/tools/build.sh in the strata-sycl-dev image; docs/B70.md)")
