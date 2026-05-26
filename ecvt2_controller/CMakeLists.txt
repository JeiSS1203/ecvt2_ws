cmake_minimum_required(VERSION 3.8)
project(ecvt2_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(osqp REQUIRED)
find_package(OsqpEigen REQUIRED)
find_package(pinocchio REQUIRED)
find_package(ifopt REQUIRED)

# ===================== kdl_utils 라이브러리 =====================
add_library(kdl_utils src/kdl_utils.cpp)
target_include_directories(kdl_utils PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS} 
)
ament_target_dependencies(kdl_utils rclcpp ament_index_cpp pinocchio)
target_link_libraries(kdl_utils Eigen3::Eigen pinocchio::pinocchio)
# ==============================================================

# ===================== wbc_hqp_lib 라이브러리 =====================
add_library(wbc_hqp_lib src/wbc_hqp_lib.cpp)
target_include_directories(wbc_hqp_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(wbc_hqp_lib
  Eigen3::Eigen
  OsqpEigen::OsqpEigen
  osqp::osqp
)
# ==============================================================

# ================base_opt_controller===============
add_executable(base_opt_controller src/base_opt_controller.cpp)
# base_opt_controller는 kdl_utils만 링크 (plain)
target_link_libraries(base_opt_controller
  kdl_utils
  OsqpEigen::OsqpEigen
  osqp::osqp
)
ament_target_dependencies(base_opt_controller rclcpp geometry_msgs sensor_msgs std_msgs visualization_msgs)
# ==============================================

# ================wbc_hqp===============
add_executable(wbc_hqp src/wbc_hqp.cpp)
# wbc_hqp는 kdl_utils만 링크 (plain)
target_link_libraries(wbc_hqp
  kdl_utils
  OsqpEigen::OsqpEigen
  osqp::osqp
)
ament_target_dependencies(wbc_hqp rclcpp geometry_msgs sensor_msgs std_msgs visualization_msgs)
# ==============================================

# ================wbc_hqp_v2===============
add_executable(wbc_hqp_v2 src/wbc_hqp_v2.cpp)
# wbc_hqp_v2는 kdl_utils만 링크 (plain)
target_link_libraries(wbc_hqp_v2
  kdl_utils
  wbc_hqp_lib
  OsqpEigen::OsqpEigen
  osqp::osqp
)
ament_target_dependencies(wbc_hqp_v2 rclcpp geometry_msgs sensor_msgs std_msgs visualization_msgs)
# ==============================================

# ================wbc_hqp_v3===============
add_executable(wbc_hqp_v3 src/wbc_hqp_v3.cpp)
# wbc_hqp_v3는 kdl_utils만 링크 (plain)
target_link_libraries(wbc_hqp_v3
  kdl_utils
  wbc_hqp_lib
  OsqpEigen::OsqpEigen
  osqp::osqp
)
ament_target_dependencies(wbc_hqp_v3 rclcpp geometry_msgs sensor_msgs std_msgs visualization_msgs)
# ==============================================

# ================base_arm_task_controller===============
add_executable(base_arm_task_controller src/base_arm_task_controller.cpp)
# base_arm_task_controller는 kdl_utils만 링크 (plain)
target_link_libraries(base_arm_task_controller
  kdl_utils
  OsqpEigen::OsqpEigen
  osqp::osqp
)
ament_target_dependencies(base_arm_task_controller rclcpp geometry_msgs sensor_msgs std_msgs visualization_msgs)
# ==============================================

# ================base_arm_task_controller_v2===============
add_executable(base_arm_task_controller_v2 src/base_arm_task_controller_v2.cpp)
target_link_libraries(base_arm_task_controller_v2
  kdl_utils
  wbc_hqp_lib
  OsqpEigen::OsqpEigen
  osqp::osqp
)
ament_target_dependencies(base_arm_task_controller_v2 rclcpp geometry_msgs sensor_msgs std_msgs visualization_msgs)
# ==============================================

# ================base_arm_task_controller_full===============
add_executable(base_arm_task_controller_full src/base_arm_task_controller_full.cpp)
target_link_libraries(base_arm_task_controller_full
  kdl_utils
  wbc_hqp_lib
  OsqpEigen::OsqpEigen
  osqp::osqp
)
ament_target_dependencies(base_arm_task_controller_full rclcpp geometry_msgs sensor_msgs std_msgs visualization_msgs ament_index_cpp)
# ==============================================

# ================teleop_goal_publisher===============
add_executable(teleop_goal_publisher src/teleop_goal_publisher.cpp)
ament_target_dependencies(teleop_goal_publisher rclcpp geometry_msgs sensor_msgs std_msgs)

install(TARGETS teleop_goal_publisher
  DESTINATION lib/${PROJECT_NAME})
# ====================================================

# ================com_trajectory_publisher===============
add_executable(com_trajectory_publisher src/com_trajectory_publisher.cpp)
ament_target_dependencies(com_trajectory_publisher rclcpp geometry_msgs)
# ========================================================

# ================pin_test===============
add_executable(pin_test src/pin_test.cpp)
target_link_libraries(pin_test
  kdl_utils
  pinocchio::pinocchio
)
ament_target_dependencies(pin_test
  rclcpp
  geometry_msgs
  sensor_msgs
  ament_index_cpp
)
# ====================================================

# ================live_upper_dynamics_node===============
add_executable(live_upper_dynamics_node
  src/live_upper_dynamics_node.cpp
)
ament_target_dependencies(live_upper_dynamics_node
  rclcpp
  sensor_msgs
  std_msgs
)
target_link_libraries(live_upper_dynamics_node
  Eigen3::Eigen
  pinocchio::pinocchio
)
install(TARGETS
  live_upper_dynamics_node
  DESTINATION lib/${PROJECT_NAME}
)
# ====================================================

# ================vp_sto_global_planner_node===============
add_executable(vp_sto_global_planner_node
  src/vp_sto_global_planner_node.cpp
)
ament_target_dependencies(vp_sto_global_planner_node
  rclcpp
  sensor_msgs
  trajectory_msgs
  std_msgs
)
target_link_libraries(vp_sto_global_planner_node
  Eigen3::Eigen
  pinocchio::pinocchio
)
install(TARGETS
  vp_sto_global_planner_node
  DESTINATION lib/${PROJECT_NAME}
)

# ================vp_sto_global_planner_full_node===============
add_executable(vp_sto_global_planner_full_node
  src/vp_sto_global_planner_full_node.cpp
)
ament_target_dependencies(vp_sto_global_planner_full_node
  rclcpp
  sensor_msgs
  trajectory_msgs
  std_msgs
)
target_link_libraries(vp_sto_global_planner_full_node
  Eigen3::Eigen
  pinocchio::pinocchio
  ifopt::ifopt_core
  ifopt::ifopt_ipopt
)
install(TARGETS
  vp_sto_global_planner_full_node
  DESTINATION lib/${PROJECT_NAME}
)

# ================vp_sto_paper_global_planner_node===============
add_executable(vp_sto_paper_global_planner_node
  src/vp_sto_paper_global_planner_node.cpp
)
ament_target_dependencies(vp_sto_paper_global_planner_node
  rclcpp
  sensor_msgs
  trajectory_msgs
  std_msgs
)
target_link_libraries(vp_sto_paper_global_planner_node
  Eigen3::Eigen
  pinocchio::pinocchio
)
install(TARGETS
  vp_sto_paper_global_planner_node
  DESTINATION lib/${PROJECT_NAME}
)

# ================= install ====================
install(DIRECTORY include/ DESTINATION include)
install(TARGETS
  kdl_utils
  wbc_hqp_lib
  base_opt_controller
  wbc_hqp
  wbc_hqp_v2
  wbc_hqp_v3
  base_arm_task_controller
  base_arm_task_controller_v2
  base_arm_task_controller_full
  teleop_goal_publisher
  com_trajectory_publisher
  pin_test
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
# ==============================================


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
