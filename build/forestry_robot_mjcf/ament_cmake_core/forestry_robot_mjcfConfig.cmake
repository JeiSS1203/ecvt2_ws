# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_forestry_robot_mjcf_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED forestry_robot_mjcf_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(forestry_robot_mjcf_FOUND FALSE)
  elseif(NOT forestry_robot_mjcf_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(forestry_robot_mjcf_FOUND FALSE)
  endif()
  return()
endif()
set(_forestry_robot_mjcf_CONFIG_INCLUDED TRUE)

# output package information
if(NOT forestry_robot_mjcf_FIND_QUIETLY)
  message(STATUS "Found forestry_robot_mjcf: 0.0.1 (${forestry_robot_mjcf_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'forestry_robot_mjcf' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${forestry_robot_mjcf_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(forestry_robot_mjcf_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${forestry_robot_mjcf_DIR}/${_extra}")
endforeach()
