set(INCLUDE_DIRECTORIES
	${ROOT_INCLUDE_DIR} 
	${CUDA_INCLUDE}
	${GLPK_INCLUDE_DIR}
	${GEANT3_INCLUDE_DIR}
	${CMAKE_SOURCE_DIR}/fairtools
	${CMAKE_SOURCE_DIR}/geobase 
	${CMAKE_SOURCE_DIR}/parbase 
	${CMAKE_SOURCE_DIR}/base 
	${CMAKE_SOURCE_DIR}/pnddata 
	${CMAKE_SOURCE_DIR}/pnddata/SttData
	${CMAKE_SOURCE_DIR}/trackbase
	${CMAKE_SOURCE_DIR}/pnddata/TrackData
	${CMAKE_SOURCE_DIR}/field 
	${CMAKE_SOURCE_DIR}/stt
	${CMAKE_SOURCE_DIR}/sttmvdtracking
	${CMAKE_SOURCE_DIR}/mvd/MvdTracking
	${CMAKE_SOURCE_DIR}/pnddata/SdsData
	${CMAKE_SOURCE_DIR}/pnddata/SciTData
	${CMAKE_SOURCE_DIR}/gem 
	${CMAKE_SOURCE_DIR}/pnddata/GemData 
	${CMAKE_SOURCE_DIR}/geane 
	${CMAKE_SOURCE_DIR}/cuda
)
include_directories( ${INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
	${ROOT_LIBRARY_DIR}
	${GLPK_LIB_DIR}
)
link_directories( ${LINK_DIRECTORIES})

set(PNDCUDA_SRC
	PndCudaExampleTask.cxx
)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/example)

if(RULE_CHECKER_FOUND)
  CHECK_RULES("${PNDCUDA_SRC}" "${INCLUDE_DIRECTORIES}" PNDCUDA_RULES)
endif(RULE_CHECKER_FOUND)

# fill list of header files from list of source files
# by exchanging the file extension
CHANGE_FILE_EXTENSION(*.cxx *.h PNDCUDA_HEADERS "${PNDCUDA_SRC}")

set(PNDCUDA_LINKDEF PndCudaLinkDef.h)
set(PNDCUDA_DICTIONARY ${CMAKE_CURRENT_BINARY_DIR}/PndCudaDict.cxx) 

ROOT_GENERATE_DICTIONARY("${PNDCUDA_HEADERS}" "${PNDCUDA_LINKDEF}" "${PNDCUDA_DICTIONARY}" "${INCLUDE_DIRECTORIES}")

SET(PNDCUDA_SRC ${PNDCUDA_SRC} ${PNDCUDA_DICTIONARY})

# message(${PNDCUDAEXAMPLESONAME}) # DEBUG test if variable is set

add_library(PndCuda SHARED ${PNDCUDA_SRC})
add_dependencies(PndCuda ${PNDCUDAEXAMPLESONAME})
target_link_libraries(PndCuda ${ROOT_LIBRARIES}
 ${CMAKE_BINARY_DIR}/lib/lib${PNDCUDAEXAMPLESONAME}.so
  ${GLPK_LIB_DIR}/libglpk.so)
set_target_properties(PndCuda PROPERTIES ${FAIRROOT_LIBRARY_PROPERTIES})
################ install ###################
install(TARGETS PndCuda DESTINATION ${CMAKE_BINARY_DIR}/lib)
