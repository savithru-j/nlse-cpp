INCLUDE( ${CMAKE_SOURCE_DIR}/cmake/ForceOutOfSource.cmake ) #This must be the first thing included

#Get all test files
file(GLOB_RECURSE TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *_test.cpp)

if (AUTODIFFEQ_ENABLE_CUDA)
  #Set the language of all .cpp files to cuda so that they also get compiled with nvcc
  set_source_files_properties(${TEST_SRCS} PROPERTIES LANGUAGE CUDA)

  #Find and append cuda source files
  file(GLOB_RECURSE TEST_CUDA_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *_gputest.cu)
  list (APPEND TEST_SRCS ${TEST_CUDA_SRCS})
endif()

ADD_CUSTOM_TARGET( test_build )

foreach(testSrc ${TEST_SRCS})
  #Extract the filename without an extension (NAME_WE)
  get_filename_component(testName ${testSrc} NAME_WE)

  #Add compile target
  add_executable(${testName}_build ${testSrc})

  #link to libraries and dependencies
  target_link_libraries(${testName}_build PRIVATE autodiffeq::autodiffeq gtest gtest_main)
  
  #Make test_build depend on all unit tests        
  ADD_DEPENDENCIES( test_build ${testName}_build )

  #ABSTRACT a target for executing the test
  ADD_CUSTOM_TARGET( ${testName} 
  COMMAND $<TARGET_FILE:${testName}_build>
  DEPENDS ${testName}_build
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )

  #Add to test list
  ADD_TEST( NAME ${testName} COMMAND ${testName}_build
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )

endforeach()

if (TEST_SRCS)
  #Add compile target
  add_executable(unit_build EXCLUDE_FROM_ALL ${TEST_SRCS})

  #link to libraries AND your targets and dependencies
  target_link_libraries(unit_build PRIVATE autodiffeq::autodiffeq gtest gtest_main)

  #Add a target for executing the test
  ADD_CUSTOM_TARGET( unit
                    COMMAND $<TARGET_FILE:unit_build>
                    DEPENDS unit_build
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
endif()