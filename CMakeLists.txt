cmake_minimum_required(VERSION 3.16)

# 1. Project definition with version
project(QtLlama VERSION 1.0.0 LANGUAGES CXX)

option(QTLLAMA_BUILD_EXAMPLES "Build QtLlama examples" OFF)


find_package(Qt6 COMPONENTS Core QUIET)
if(Qt6_FOUND)
    set(QT_VERSION_MAJOR 6)
    message(STATUS "QtLlama: Found Qt 6. Building with Qt6::Core")
else()
    find_package(Qt5 REQUIRED COMPONENTS Core)
    set(QT_VERSION_MAJOR 5)
    message(STATUS "QtLlama: Found Qt 5. Building with Qt5::Core")
endif()

find_package(llama REQUIRED)

# 2. Add the library with all headers and sources
add_library(QtLlama STATIC 
    src/Session.cpp
    src/Engine.cpp
    src/Embedder.cpp
    src/EmbeddingWorker.cpp 

    include/QtLlama/Types.h
    include/QtLlama/IEngine.h
    include/QtLlama/IEmbedder.h
    include/QtLlama/Session.h
    include/QtLlama/Engine.h
    include/QtLlama/Embedder.h
    include/QtLlama/EmbeddingWorker.h
)

# 3. Create the ALIAS target for seamless FetchContent/find_package usage
add_library(QtLlama::QtLlama ALIAS QtLlama)

# 4. Export include directories
target_include_directories(QtLlama PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> 
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(QtLlama PUBLIC Qt${QT_VERSION_MAJOR}::Core llama)

set_target_properties(QtLlama PROPERTIES AUTOMOC ON CXX_STANDARD 23)


if(QTLLAMA_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ==============================================================================
# INSTALLATION & EXPORT LOGIC
# ==============================================================================
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# A. Install the compiled library and export the target
install(TARGETS QtLlama
    EXPORT QtLlamaTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# B. Install the header files
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# C. Generate and install the QtLlamaTargets.cmake file
install(EXPORT QtLlamaTargets
    FILE QtLlamaTargets.cmake
    NAMESPACE QtLlama::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QtLlama
)

# D. Generate Config and Version files
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/QtLlamaConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/QtLlamaConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QtLlama
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/QtLlamaConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/QtLlamaConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/QtLlamaConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QtLlama
)
