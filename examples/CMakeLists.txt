INCLUDE( ${CMAKE_SOURCE_DIR}/cmake/ForceOutOfSource.cmake ) #This must be the first thing included

add_custom_target(copy_example_data)
ADD_CUSTOM_COMMAND(TARGET copy_example_data POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_CURRENT_SOURCE_DIR}/data
                   ${CMAKE_CURRENT_BINARY_DIR}/data
                   COMMENT "Copying data files to '${CMAKE_CURRENT_BINARY_DIR}/data'")

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

    if (OpenMP_CXX_FOUND)
      target_compile_options(${exampleName} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
    endif()
  endif()

  #link to libraries and dependencies
  target_link_libraries(${exampleName} PRIVATE ${NLSECPP_LIBS} )
  add_dependencies(${exampleName} copy_example_data)
  
endforeach()