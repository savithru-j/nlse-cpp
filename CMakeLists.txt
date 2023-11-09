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

INCLUDE(cmake/CompileFlags.cmake)

#Bring the headers into the project
INCLUDE_DIRECTORIES(src)

SET(NLSECPP_LIBS 
    libODE
   )

if (OpenMP_CXX_FOUND)
  list (APPEND NLSECPP_LIBS OpenMP::OpenMP_CXX autodiffeq::autodiffeq)
else()
  list (APPEND NLSECPP_LIBS autodiffeq::autodiffeq)
endif()

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(examples)

#Enable testing for the project
enable_testing()

# Configure google-test
INCLUDE(FetchContent)
FetchContent_Declare(
googletest
URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

#Add all unit tests
ADD_SUBDIRECTORY(test)

