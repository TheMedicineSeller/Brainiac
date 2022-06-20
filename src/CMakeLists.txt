cmake_minimum_required(VERSION 3.22)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# add_executable(test test.cpp
#                     KNN.cpp
#                     # Kmeans.cpp
#                     Imgdata_handler.cpp
#                     Imgdata.cpp
# )

# add_executable(test test.cpp
#                     Csvdata_handler.cpp
# )

add_executable(test test.cpp
                    Imgdata.cpp
                    Imgdata_handler.cpp
                    NeuralNet.cpp)