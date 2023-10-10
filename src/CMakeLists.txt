add_compile_definitions(CUDA)
add_compile_definitions(HIGH_PRECISION)

aux_source_directory("${PROJECT_SOURCE_DIR}/src/display" DIR_SOURCE_DISPLAY)
aux_source_directory("${PROJECT_SOURCE_DIR}/src/furion_scri" DIR_SOURCE_FUR_SCRI)
aux_source_directory("${PROJECT_SOURCE_DIR}/src/import_and_display_srw" DIR_SOURCE_IMP_DIS_SRW)
aux_source_directory("${PROJECT_SOURCE_DIR}/src/optical_element" DIR_SOURCE_OPT_ELE)
aux_source_directory("${PROJECT_SOURCE_DIR}/src/propagation" DIR_SOURCE_PROPAGATION)
aux_source_directory("${PROJECT_SOURCE_DIR}/src/surface" DIR_SOURCE_SURFACE)
aux_source_directory("${PROJECT_SOURCE_DIR}/src/xop" DIR_SOURCE_XOP)
aux_source_directory("${PROJECT_SOURCE_DIR}/src" DIR_SOURCE_SRC)

# 指定生成目标
add_executable( Furion 
                ${DIR_SOURCE_DISPLAY}
				${DIR_SOURCE_FUR_SCRI}
				${DIR_SOURCE_IMP_DIS_SRW}
				${DIR_SOURCE_OPT_ELE}
				${DIR_SOURCE_PROPAGATION}
				${DIR_SOURCE_SURFACE}
				${DIR_SOURCE_XOP}
                ${DIR_SOURCE_SRC}
               ) 

#target_link_libraries(Furion MPI::MPI_CXX)

message(STATUS "DISPLAY_H: " ${DISPLAY_H})
message(STATUS "FUR_SCRI_H: " ${FUR_SCRI_H})
message(STATUS "IMP_DIS_SRW_H: " ${IMP_DIS_SRW_H})
message(STATUS "OPT_ELE_H: " ${OPT_ELE_H})
message(STATUS "PROPAGATION_H: " ${PROPAGATION_H})
message(STATUS "SURFACE_H: " ${SURFACE_H})
message(STATUS "XOP_H: " ${XOP_H})
message(STATUS "SRC_H: " ${SRC_H})
message(STATUS "CUDA_H: " ${CUDA_H})

target_include_directories(Furion PRIVATE
                           ${DISPLAY_H}
                           ${FUR_SCRI_H}
                           ${IMP_DIS_SRW_H}
                           ${OPT_ELE_H}
                           ${PROPAGATION_H}
                           ${SURFACE_H}
                           ${XOP_H}
                           ${SRC_H}
                           ${CUDA_H}
                           )

install(TARGETS Furion RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin)

# file(GLOB SOURCES "*.cpp")
# add_executable(Furion ${SOURCES}) 
#add_executable(Furion Furion.cpp)



