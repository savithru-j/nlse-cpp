INCLUDE( ${CMAKE_SOURCE_DIR}/cmake/ForceOutOfSource.cmake ) #This must be the first thing included

SET( ODE_SRCS
    MultimodeNLSE.cpp
   )

if (AUTODIFFEQ_ENABLE_CUDA)
   #Set the language of all .cpp files to cuda so that they also get compiled with nvcc
   set_source_files_properties(${ODE_SRCS} PROPERTIES LANGUAGE CUDA)
endif()

ADD_LIBRARY( libODE STATIC ${ODE_SRCS} )
target_link_libraries(libODE PRIVATE autodiffeq::autodiffeq)
