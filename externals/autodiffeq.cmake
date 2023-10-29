include(ExternalProject)

ExternalProject_Add(
  autodiffeq
  GIT_REPOSITORY https://github.com/savithru-j/autodiffeq.git
  GIT_TAG main
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/externals/autodiffeq/install
  BUILD_COMMAND make -j4
  BUILD_IN_SOURCE 0
  BINARY_DIR "${CMAKE_BINARY_DIR}/externals/autodiffeq/build"
  INSTALL_COMMAND make install
  INSTALL_DIR "${CMAKE_BINARY_DIR}/externals/autodiffeq/install"
  )


ExternalProject_Get_Property(autodiffeq source_dir)
message(${source_dir})
set(AUTODIFFEQ_INCLUDE_DIR ${source_dir} CACHE PATH "autodiffeq include directory")

ExternalProject_Get_Property(autodiffeq binary_dir)
message(${binary_dir})
set(AUTODIFFEQ_BUILD_DIR ${binary_dir} CACHE PATH "autodiffeq build directory")

ExternalProject_Get_Property(autodiffeq install_dir)
message(${install_dir})
set(AUTODIFFEQ_INSTALL_DIR ${install_dir} CACHE PATH "autodiffeq install directory")