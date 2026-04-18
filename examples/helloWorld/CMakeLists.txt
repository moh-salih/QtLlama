cmake_minimum_required(VERSION 3.16)

project(helloWorld LANGUAGES CXX VERSION 1.0.0)


add_executable(helloWorld main.cpp)

target_link_libraries(helloWorld PRIVATE QtLlama::QtLlama)

set_target_properties(helloWorld PROPERTIES 
    CMAKE_AUTOMOC ON 
    CXX_STANDARD 17 
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
