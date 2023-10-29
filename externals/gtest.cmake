include(ExternalProject)

SET(GTEST_CXX_FLAGS "-std=c++14 -fPIC")

ExternalProject_Add(
  googletest
  CMAKE_ARGS 
    -DBUILD_SHARED_LIBS=true
    -DCMAKE_CXX_FLAGS=${GTEST_CXX_FLAGS}
    -DBUILD_GMOCK=false
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/externals/googletest/"
  BUILD_IN_SOURCE 0
  BUILD_COMMAND make -j4
  BINARY_DIR "${CMAKE_BINARY_DIR}/externals/googletest/build"
  INSTALL_COMMAND ""
  INSTALL_DIR "${CMAKE_BINARY_DIR}/externals/googletest/install"
  )

ExternalProject_Get_Property(googletest source_dir)
set(GTEST_INCLUDE_DIR ${source_dir}/googletest/include CACHE PATH "GTest include directory")

ExternalProject_Get_Property(googletest binary_dir)

#Build shared libraries
add_library(gtest SHARED IMPORTED GLOBAL)
set_target_properties(gtest PROPERTIES IMPORTED_LOCATION ${binary_dir}/lib/libgtest.so)
add_dependencies(gtest googletest)

add_library(gtest_main SHARED IMPORTED GLOBAL)
set_target_properties(gtest_main PROPERTIES IMPORTED_LOCATION ${binary_dir}/lib/libgtest_main.so)
add_dependencies(gtest_main googletest)
