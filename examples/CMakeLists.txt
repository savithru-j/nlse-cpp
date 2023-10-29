INCLUDE( ${CMAKE_SOURCE_DIR}/cmake/ForceOutOfSource.cmake ) #This must be the first thing included

#Get all test files
file(GLOB_RECURSE EXAMPLE_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

if (AUTODIFFEQ_ENABLE_CUDA)
  #Set the language of all .cpp files to cuda so that they also get compiled with nvcc
  set_source_files_properties(${EXAMPLE_SRCS} PROPERTIES LANGUAGE CUDA)

  #Find and append cuda source files
  file(GLOB_RECURSE EXAMPLE_CUDA_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cu)
  list (APPEND EXAMPLE_SRCS ${EXAMPLE_CUDA_SRCS})
endif()

foreach(exampleSrc ${EXAMPLE_SRCS})
  #Extract the filename without an extension (NAME_WE)
  get_filename_component(exampleName ${exampleSrc} NAME_WE)
  get_filename_component(exampleExt ${exampleSrc} EXT)

  #Add compile target
  add_executable(${exampleName} ${exampleSrc})

  if (AUTODIFFEQ_ENABLE_CUDA)
    if (exampleExt STREQUAL ".cu")
      target_compile_options(${exampleName} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
    endif()
  endif()

  add_dependencies(${exampleName} autodiffeq)

  #link to libraries and dependencies
  if (OpenMP_CXX_FOUND)
    target_link_libraries(${exampleName} PRIVATE OpenMP::OpenMP_CXX autodiffeq::autodiffeq)
  else()
    target_link_libraries(${exampleName} PRIVATE autodiffeq::autodiffeq )
  endif()
  
endforeach()
