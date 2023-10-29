INCLUDE(cmake/ForceOutOfSource.cmake) #This must be the first thing included

#Get the name of the binary directory
STRING(TOUPPER ${CMAKE_BINARY_DIR} BIN_DIR_NAME)

CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(nlse++ VERSION 0.5.0 LANGUAGES CXX)

SET(CMAKE_CXX_STANDARD 14) 
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF) #without compiler extensions like gnu++11

INCLUDE(FetchContent)
FetchContent_Declare(autodiffeq
        GIT_REPOSITORY https://github.com/savithru-j/autodiffeq.git
        GIT_TAG main
        )
FetchContent_MakeAvailable(autodiffeq)
MESSAGE("Fetched autodiffeq.")

# 
find_package(OpenMP)
if (AUTODIFFEQ_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
  add_definitions(-DENABLE_OPENMP)
  MESSAGE("Enabled OpenMP.")
else()
  MESSAGE("OpenMP not found. Disabled option.")
endif()

if(AUTODIFFEQ_ENABLE_CUDA)
  enable_language(CUDA)
  add_definitions(-DENABLE_CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_ARCHITECTURES "70;86")
  MESSAGE("Enabled CUDA.")
else()
  MESSAGE("CUDA not found. Disabled option.")
endif()

INCLUDE(CheckCXXSourceCompiles)
INCLUDE(CheckCXXSourceRuns)

# Include the cmake variables with values for installation directories
INCLUDE(GNUInstallDirs)

#Enable testing for the project
enable_testing()

# Configure google-test an external library.
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/externals)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/externals)
INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIR})

# list(APPEND CMAKE_PREFIX_PATH ${AUTODIFFEQ_INSTALL_DIR})
# message("CMAKE_PREFIX_PATH:" ${CMAKE_PREFIX_PATH})

# find_package(autodiffeq REQUIRED)
 
#Bring the headers into the project
INCLUDE_DIRECTORIES(src)

INCLUDE(cmake/CompileFlags.cmake)
#INCLUDE( CMakeInclude/Dependencies.cmake )

SET(LIBS 
    # libNumerics
   )
  
ADD_SUBDIRECTORY(examples)
#ADD_SUBDIRECTORY(test)
