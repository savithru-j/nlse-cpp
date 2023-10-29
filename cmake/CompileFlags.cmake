MESSAGE("Compiler ID: " ${CMAKE_CXX_COMPILER_ID}  ${CMAKE_CXX_COMPILER_VERSION})
IF( CMAKE_COMPILER_IS_GNUCXX )
  SET( GCC_MIN_VERSION 5.0)
  IF (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS ${GCC_MIN_VERSION})
    MESSAGE(FATAL_ERROR "This library relies on C++14 standards that only exist in g++ version ${GCC_MIN_VERSION} or higher. Current version is ${CMAKE_CXX_COMPILER_VERSION}.")
  ENDIF()
ELSEIF( ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  SET( CLANG_MIN_VERSION 3.4 )
  IF (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS ${CLANG_MIN_VERSION})
    MESSAGE(FATAL_ERROR "This library relies on C++14 standards that only exist in clang version ${CLANG_MIN_VERSION} or higher. Current version is ${CMAKE_CXX_COMPILER_VERSION}.")
  ENDIF()
ENDIF()

IF( APPLE )
  #Apple defines a macro named 'check' in AssertMacros.h unless this is used
  #to suppress the macro definition.
  ADD_DEFINITIONS( -D__ASSERT_MACROS_DEFINE_VERSIONS_WITHOUT_UNDERSCORES=0 )
ENDIF()

#==================================================
# Default compiler flags, these can be modified under
# the advanced options using ccmake
#==================================================
IF( NOT DEFINED CMAKE_FLAGS_INIT )

  #===============================
  # Set the build type to release by default, but debug if the binary directory contains the name debug
  SET( BUILD_TYPE_STRING "Choose the type of build, options are: Debug Release RelWithDebInfo Memcheck." )

  IF( NOT CMAKE_BUILD_TYPE )
    IF( BIN_DIR_NAME MATCHES "DEBUG" )
      SET(CMAKE_BUILD_TYPE "Debug" CACHE STRING ${BUILD_TYPE_STRING} FORCE)
    ELSEIF( BIN_DIR_NAME MATCHES "RELEASE" OR BIN_DIR_NAME MATCHES "DEPLOY" )
      SET(CMAKE_BUILD_TYPE "Release" CACHE STRING ${BUILD_TYPE_STRING} FORCE)
    ELSEIF( BIN_DIR_NAME MATCHES "RELWITHDEBINFO" )
      SET(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING ${BUILD_TYPE_STRING} FORCE)
    ELSEIF( BIN_DIR_NAME MATCHES "MEMCHECK" )
      SET(CMAKE_BUILD_TYPE "Memcheck" CACHE STRING ${BUILD_TYPE_STRING} FORCE)
    ELSE()
      #SET(CMAKE_BUILD_TYPE "Release" CACHE STRING ${BUILD_TYPE_STRING} FORCE)
      SET(CMAKE_BUILD_TYPE "Debug" CACHE STRING ${BUILD_TYPE_STRING} FORCE) #Default to debug for now
    ENDIF()
  ENDIF()
  
  #=============================

  SET( CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG" CACHE STRING "C++ Release Flags" FORCE )

  #Compiler flags for the C++ compiler
  IF(CMAKE_COMPILER_IS_GNUCXX )

    SET( GNU_WARNING_FLAGS "-Wall -Wextra -Wno-unused-parameter -Wunused-result -Winit-self -Wno-variadic-macros -Wno-vla -Wno-strict-overflow -Wno-int-in-bool-context" )
    IF (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 5 AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 6)
      SET( GNU_WARNING_FLAGS "${GNU_WARNING_FLAGS} -Wno-maybe-uninitialized")
    ENDIF()
    
    SET( CMAKE_CXX_FLAGS "${GNU_WARNING_FLAGS} -std=c++14 -fstrict-aliasing -Wstrict-aliasing -pedantic -Wnon-virtual-dtor" CACHE STRING "C++ Flags" FORCE)
    SET( CMAKE_C_FLAGS "${GNU_WARNING_FLAGS} -fstrict-aliasing -Wstrict-aliasing" CACHE STRING "C Flags" FORCE)
    IF( NOT CYGWIN )
      SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -pthread" CACHE STRING "C++ Flags" FORCE)
      SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -pthread" CACHE STRING "C Flags" FORCE)
    ELSE()
      SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -U__STRICT_ANSI__ -Wa,-mbig-obj -Og" CACHE STRING "C++ Flags" FORCE)
    ENDIF()

    SET( CMAKE_CXX_FLAGS_DEBUG "-g -ftrapv -fbounds-check" CACHE STRING "C++ Debug Flags" FORCE )
    IF( NOT CYGWIN )
      SET( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0" CACHE STRING "C++ Debug Flags" FORCE )
    ELSE()
      SET( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og" CACHE STRING "C++ Debug Flags" FORCE )
    ENDIF()
    
    SET( CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -funroll-loops" CACHE STRING "C++ Release Flags" FORCE )
    SET( CMAKE_CXX_FLAGS_MEMCHECK "-g -Os -fsanitize=address -fno-omit-frame-pointer" CACHE STRING "C++ Compiler Memory Check Flags" FORCE )

    SET( CMAKE_C_FLAGS_DEBUG "-g -O0 -ftrapv -fbounds-check" CACHE STRING "C Debug Flags" FORCE )
    SET( CMAKE_C_FLAGS_RELEASE "-O3 -funroll-loops" CACHE STRING "C Release Flags" FORCE )
    SET( CMAKE_C_FLAGS_MEMCHECK "-g -Os -fsanitize=address -fno-omit-frame-pointer" CACHE STRING "C Compiler Memory Check Flags" FORCE )

    SET( GNU_NO_INLINE_FLAGS "-DALWAYS_INLINE=inline -fno-inline -fno-inline-functions -fno-inline-small-functions -fno-inline-functions-called-once -fno-default-inline -fno-implicit-inline-templates" )

    IF( NOT CYGWIN AND NOT MINGW )
      SET( CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags" CACHE STRING "Executable Link Flags" FORCE )
    ENDIF()
    SET( CMAKE_EXE_LINKER_FLAGS_MEMCHECK "-fsanitize=address -fuse-ld=gold -Wl,--disable-new-dtags -Wl,--allow-shlib-undefined" CACHE STRING "Executable Link Flags For Memcheck" FORCE )

    SET( CMAKE_SHARED_LINKER_FLAGS "-Wl,--disable-new-dtags -Wl,--no-undefined" CACHE STRING "Shared Library Link Flags" FORCE )
    SET( CMAKE_SHARED_LINKER_FLAGS_MEMCHECK "${CMAKE_EXE_LINKER_FLAGS_MEMCHECK}" CACHE STRING "Shared Library Link Flags For Memcheck" FORCE )

  ELSEIF( ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel" )
    # -Wstrict-aliasing creates all kinds of crazy warnings for intel
    SET( INTEL_WARINNGS "-wd3415" )
    SET( INTEL_INLINE "-finline-functions" )
    SET( CMAKE_CXX_FLAGS "-Wall -std=c++14 -fPIC -pthread -fstrict-aliasing -ansi-alias-check ${INTEL_WARINNGS}" CACHE STRING "C++ Flags" FORCE )
    SET( CMAKE_C_FLAGS "-Wall -fPIC -pthread -fstrict-aliasing -ansi-alias-check" CACHE STRING "C Flags" FORCE )
    SET( CMAKE_CXX_FLAGS_DEBUG "-g -O0" CACHE STRING "C++ Debug Flags" FORCE )
    SET( CMAKE_C_FLAGS_DEBUG "-g -O0" CACHE STRING "C Debug Flags" FORCE )
    SET( CMAKE_CXX_FLAGS_RELEASE "-O3 -xHost ${INTEL_INLINE} -qopt-subscript-in-range" CACHE STRING "C++ Release Flags" FORCE )
    SET( CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "C Release Flags" FORCE )
    IF( APPLE )
      SET( CMAKE_SHARED_LINKER_FLAGS "-Wl,-undefined,error" CACHE STRING "Flags used by the linker during the creation of dll's." FORCE )
    ELSE()
      SET( CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags" CACHE STRING "Executable Link Flags" FORCE )
      SET( CMAKE_SHARED_LINKER_FLAGS "-Wl,--disable-new-dtags -Wl,--no-undefined" CACHE STRING "Flags used by the linker during the creation of dll's." FORCE )
    ENDIF()
  ELSEIF( ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    #-Weverything -Wno-unreachable-code -Wno-newline-eof -Wno-c++98-compat -Wno-c++98-compat-pedantic
    SET( CLANG_WARNING_FLAGS "-Wall -Wstrict-aliasing -Wnon-virtual-dtor -pedantic -Wno-variadic-macros -Wno-#pragma-messages")
    IF( UNIX )
      IF( APPLE )
        IF ( ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 8.1.0 )
          SET( CLANG_WARNING_FLAGS "${CLANG_WARNING_FLAGS} -Wno-undefined-var-template" )
        ENDIF()
      ELSEIF( ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 3.8.0 )
        SET( CLANG_WARNING_FLAGS "${CLANG_WARNING_FLAGS} -Wno-undefined-var-template" )
      ENDIF()
    ENDIF()
    SET( CMAKE_CXX_FLAGS "-std=c++14 ${CLANG_WARNING_FLAGS} -fstrict-aliasing" CACHE STRING "C++ Flags" FORCE)
    SET( CMAKE_C_FLAGS "-Wall -fstrict-aliasing -Wstrict-aliasing" CACHE STRING "C Flags" FORCE)
    IF( APPLE )
      SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC" CACHE STRING "C++ Flags" FORCE)
      SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC" CACHE STRING "C Flags" FORCE)
    ELSEIF( CYGWIN )
    ELSE()
      SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -pthread" CACHE STRING "C++ Flags" FORCE)
      SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -pthread" CACHE STRING "C Flags" FORCE)
    ENDIF()
    SET( CMAKE_CXX_FLAGS_DEBUG "-g -O0 -ftrapv" CACHE STRING "C++ Debug Flags" FORCE )
    SET( CMAKE_C_FLAGS_DEBUG   "-g -O0 -ftrapv" CACHE STRING "C Debug Flags" FORCE )

    SET( CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops" CACHE STRING "C++ Release Flags" FORCE )
    SET( CMAKE_C_FLAGS_RELEASE   "-O3 -funroll-loops" CACHE STRING "C Release Flags" FORCE )

    SET( CMAKE_CXX_FLAGS_MEMCHECK "-g -Os -fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls" CACHE STRING "C++ Compiler Memory Check Flags" FORCE )
    SET( CMAKE_C_FLAGS_MEMCHECK   "-g -Os -fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls" CACHE STRING "C Compiler Memory Check Flags" FORCE )

    #IF( NOT CYGWIN )
    #  SET( CMAKE_EXE_LINKER_FLAGS "-lc++" CACHE STRING "Executable Link Flags." FORCE )
    #ENDIF()
    IF( APPLE )
      SET( CMAKE_SHARED_LINKER_FLAGS "-Wl,-undefined,error" CACHE STRING "Flags used by the linker during the creation of dll's." FORCE )
    ELSE()
      SET( CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags" CACHE STRING "Executable Link Flags" FORCE )
      SET( CMAKE_SHARED_LINKER_FLAGS "-Wl,--disable-new-dtags -Wl,--no-undefined" CACHE STRING "Flags used by the linker during the creation of dll's." FORCE )
    ENDIF()
  ENDIF()
  
ENDIF()

MARK_AS_ADVANCED( FORCE
                  CMAKE_CXX_FLAGS_DEBUG
                  CMAKE_CXX_FLAGS_RELEASE
                  CMAKE_CXX_FLAGS_MEMCHECK
                  CMAKE_EXE_LINKER_FLAGS
                  CMAKE_EXE_LINKER_FLAGS_MEMCHECK
                  CMAKE_SHARED_LINKER_FLAGS
                  CMAKE_SHARED_LINKER_FLAGS_MEMCHECK
                  CMAKE_C_FLAGS
                  CMAKE_C_FLAGS_DEBUG
                  CMAKE_C_FLAGS_RELEASE
                  CMAKE_C_FLAGS_MEMCHECK
                )

IF( CMAKE_BUILD_TYPE )
  STRING( TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE )
  MARK_AS_ADVANCED( CLEAR CMAKE_CXX_FLAGS )
  MARK_AS_ADVANCED( CLEAR CMAKE_CXX_FLAGS_${BUILD_TYPE} )
ENDIF()

SET( CMAKE_FLAGS_INIT TRUE CACHE INTERNAL "Indicator that this is the first run of cmake" )

#==============================================================================
# Check that the compiler actually works with C++14 features

SET(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${BUILD_TYPE}} ${CMAKE_EXE_LINKER_FLAGS_${BUILD_TYPE}}")
MESSAGE("CMAKE_REQUIRED_FLAGS:" ${CMAKE_REQUIRED_FLAGS})

# try to compile a simple program to make sure the c++11 auto feature works
CHECK_CXX_SOURCE_COMPILES(
  "
  #include <vector>
  int main()
  {
    std::vector<double> vec0{1, 2};
    auto vec = vec0;
    return 0;
  }
  "
  CPP11_AUTO_COMPILES)


# try to compile a simple program to make sure the c++11 shared pointer feature works
CHECK_CXX_SOURCE_COMPILES(
  "
  #include <memory>
  int main()
  {
    std::shared_ptr<double> vec0;
    vec0 = std::make_shared<double>(2.0);
    return 0; 
  }
  "
  CPP11_SHAREDPTR_COMPILES)

# try to compile a simple program to make sure the c++14 auto return feature works
CHECK_CXX_SOURCE_COMPILES(
  "
  #include <tuple>
  template<typename T, typename U>
  auto f(T a, U b) {
    return std::make_tuple(a, b);
  }

  int main()
  {
    auto x = f(2,3);
    auto y = f(2.0, \"str\");
    return 0;
  }
  "
  CPP14_AUTO_COMPILES)


UNSET(CMAKE_REQUIRED_FLAGS)

IF(NOT CPP11_AUTO_COMPILES OR NOT CPP11_SHAREDPTR_COMPILES)
  MESSAGE( "====================================================================" )
  MESSAGE( "Basic tests of C++11 cannot be compiled.")
  MESSAGE( "Please make sure your compiler supports all C++11 features." )
  MESSAGE( "" )
  MESSAGE( "See CMakeFiles/CMakeError.log for more details.")
  MESSAGE( "====================================================================" )
  MESSAGE( "" )
  MESSAGE( FATAL_ERROR "" )
ENDIF()

IF(NOT CPP14_AUTO_COMPILES)
  MESSAGE( "====================================================================" )
  MESSAGE( "Basic tests of C++14 cannot be compiled.")
  MESSAGE( "Please make sure your compiler supports all C++14 features." )
  MESSAGE( "" )
  MESSAGE( "See CMakeFiles/CMakeError.log for more details.")
  MESSAGE( "====================================================================" )
  MESSAGE( "" )
  MESSAGE( FATAL_ERROR "" )
ENDIF()

