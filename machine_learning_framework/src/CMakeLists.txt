


#set( src_directory "${PROJECT_SOURCE_DIR}/src")
#file(GLOB_RECURSE NN_framework_headers "${src_directory}/framework_headers/*"
include_directories("framework_headers/")
file(GLOB_RECURSE NN_framework_headers "framework_headers/*")


# Project sources files
add_executable(NNF_test main.cpp ${NN_framework_headers})
