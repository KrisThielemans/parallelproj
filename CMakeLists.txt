# structure and content of CMakeLists.txt files adapted from 
# https://stackoverflow.com/questions/55635294/how-to-create-packages-with-cmake

cmake_minimum_required(VERSION 3.9.0)
include(CMakePackageConfigHelpers)
include(CheckLanguage)

project(parallelproj LANGUAGES C)
set (CMAKE_BUILD_TYPE Release)

#-------------------------------------------------------------
# define the version number
set(VERSION_MAJOR   0)
set(VERSION_MINOR   5)
set(VERSION_PATCH   0-prealpha)

set(parallelproj_VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})


#-------------------------------------------------------------
# checks for OpenMP, CUDA and doxygen

# find the required OpeMP
find_package(OpenMP REQUIRED)

# check if CUDA is available
check_language(CUDA)

# check if Doxygen is installed
find_package(Doxygen)



#-------------------------------------------------------------
# build the C/OpenMP lib
add_subdirectory(c)


# only build CUDA lib if CUDA is available
if(CMAKE_CUDA_COMPILER)
  add_subdirectory(cuda)
endif()


#-------------------------------------------------------------
# install the targets
install(
  EXPORT parallelprojTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/parallelproj
  NAMESPACE parallelproj::
  FILE parallelprojTargets.cmake # Not sure if this is still needed
  )



#-------------------------------------------------------------
# generate and install cmake package and version files

configure_package_config_file(
  Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/parallelprojConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/parallelproj
  PATH_VARS
    CMAKE_INSTALL_LIBDIR
  )

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/parallelprojConfigVersion.cmake
  VERSION ${parallelproj_VERSION}
  COMPATIBILITY SameMajorVersion
  )

# Install Config and ConfigVersion files
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/parallelprojConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/parallelprojConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/parallelproj"
  )



#-------------------------------------------------------------
# build the documentation with Doxygen

if (DOXYGEN_FOUND)
  # set input and output files
  set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
  set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  # request to configure the file
  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

  # note the option ALL which allows to build the docs together with the application
  add_custom_target( doc_doxygen ALL
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen"
      VERBATIM )
endif (DOXYGEN_FOUND)
