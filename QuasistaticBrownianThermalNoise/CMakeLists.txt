##
#  CMake script for the QuasistaticBrownianThermalNoise tutorial program:
##

# Set the name of the project and target:
SET(TARGET "QuasistaticBrownianThermalNoise")

# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
#    FILE(GLOB_RECURSE TARGET_SRC  "source/*.cpp")
#    FILE(GLOB_RECURSE TARGET_INC  "include/*.hpp")
#    SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC}) 
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.
SET(TARGET_SRC
  ${TARGET}.cpp
  )

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.3 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

# Find yaml-cpp, used for parsing input files
FIND_PATH(YAMLCPP_INCLUDE_DIRS yaml-cpp/yaml.h
  PATH_SUFFIXES include
  HINTS ${YAMLCPP_ROOT}/include/)
MESSAGE("Found yaml-cpp includes in ${YAMLCPP_INCLUDE_DIRS}")
INCLUDE_DIRECTORIES(${YAMLCPP_INCLUDE_DIRS})

FIND_LIBRARY(YAMLCPP_LIBRARIES
  NAMES libyaml-cpp.a yaml-cpp
  PATH_SUFFIXES lib64 lib
  HINTS ${YAMLCPP_ROOT})
MESSAGE("Found yaml-cpp libraries ${YAMLCPP_LIBRARIES}")
LINK_LIBRARIES(${YAMLCPP_LIBRARIES})

FIND_PATH(CATCH_INCLUDE_DIRS catch2/catch.h
  HINTS ${CATCH_ROOT}/)
MESSAGE("Found catch includes in ${CATCH_INCLUDE_DIRS}")
INCLUDE_DIRECTORIES(${CATCH_INCLUDE_DIRS})

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()
