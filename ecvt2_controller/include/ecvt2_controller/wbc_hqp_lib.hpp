#pragma once

#include "ecvt2_controller/kdl_utils.hpp"

#include <Eigen/Dense>

#include <array>
#include <vector>

namespace ecvt2_controller::wbc_hqp_lib
{

constexpr int kModelDof = 26;
constexpr int kLowerDof = 22;
using Vec26 = Eigen::Matrix<double, kModelDof, 1>;

struct JointBoundData
{
  Vec26 qdot_lower = Vec26::Zero();
  Vec26 qdot_upper = Vec26::Zero();
  Vec26 raw_qdot_lower = Vec26::Zero();
  Vec26 raw_qdot_upper = Vec26::Zero();
};

double orientationErrorAroundZ(const Eigen::Matrix3d &R_current, const Eigen::Matrix3d &R_desired);

Eigen::MatrixXd buildFullJacobian(const FKJacobianResult &fk_result);

JointBoundData computeJointBoundData(
    const Eigen::Ref<const Eigen::VectorXd> &q_hw,
    const std::array<int, kModelDof> &model_to_hw_index,
    const Vec26 &q_min_full_safe,
    const Vec26 &q_max_full_safe,
    const Vec26 &qd_min_full,
    const Vec26 &qd_max_full,
    double loop_dt,
    double pos_limit_rate_scale,
    double base_velocity_limit);

bool solveHQPUpperLevel(
    const Eigen::MatrixXd &task_jacobian,
    const Eigen::Vector3d &task_velocity_command,
    const std::vector<Eigen::MatrixXd> &locked_jacobians,
    const std::vector<Eigen::Vector3d> &locked_task_velocity_commands,
    const Eigen::MatrixXd &contact_jacobian,
    const Eigen::VectorXd &qdot_lower_limits,
    const Eigen::VectorXd &qdot_upper_limits,
    const Eigen::VectorXd &qdot_previous,
    double lambda_qdot_norm,
    double lambda_qdot_smooth,
    double base_weight,
    double arm_weight,
    Eigen::VectorXd &qdot_solution,
    Eigen::Vector3d &task_slack);

bool solveUpperBodyHqp(
    const Eigen::MatrixXd &base_orientation_jacobian,
    const Eigen::MatrixXd &end_effector_jacobian,
    const Eigen::MatrixXd &base_linear_jacobian,
    const Eigen::Vector3d &base_orientation_cmd,
    const Eigen::Vector3d &end_effector_linear_cmd,
    const Eigen::Vector3d &base_linear_cmd,
    const Eigen::MatrixXd &contact_jacobian,
    const Vec26 &qdot_lower_limits,
    const Vec26 &qdot_upper_limits,
    const Eigen::VectorXd &qdot_previous,
    double lambda_qdot_norm,
    double lambda_qdot_smooth,
    double base_weight,
    double arm_weight,
    Eigen::VectorXd &qdot_level3_solution,
    Eigen::VectorXd &qdot_upper_final_solution);

bool solveLowerBodyHqp(
    const Eigen::MatrixXd &wheel_jacobian,
    const Eigen::Vector4d &wheel_velocity_command,
    const Eigen::MatrixXd &base_orientation_jacobian,
    const Eigen::Vector3d &base_orientation_cmd,
    const Eigen::MatrixXd &base_linear_jacobian,
    const Eigen::Vector3d &base_linear_cmd,
    const Eigen::MatrixXd &contact_jacobian,
    const Eigen::VectorXd &qdot_lower_limits,
    const Eigen::VectorXd &qdot_upper_limits,
    const Eigen::VectorXd &qdot_previous,
    double lambda_qdot_norm,
    double lambda_qdot_smooth,
    Eigen::VectorXd &qdot_level3_solution,
    Eigen::VectorXd &qdot_lower_final_solution);

bool solveHQPLowerLevel1WheelTask(
    const Eigen::MatrixXd &wheel_jacobian,
    const Eigen::Vector4d &wheel_velocity_command,
    const Eigen::VectorXd &qdot_lower_limits,
    const Eigen::VectorXd &qdot_upper_limits,
    const Eigen::VectorXd &qdot_previous,
    double lambda_qdot_norm,
    double lambda_qdot_smooth,
    Eigen::VectorXd &qdot_solution);

bool solveHQPLowerLevel2Task(
    const Eigen::MatrixXd &task_jacobian,
    const Eigen::VectorXd &task_velocity_command,
    const Eigen::MatrixXd &wheel_jacobian,
    const Eigen::VectorXd &qdot_wheel_l1,
    const Eigen::VectorXd &qdot_lower_limits,
    const Eigen::VectorXd &qdot_upper_limits,
    double lambda_lower_task,
    double lambda_qdot_norm,
    double lambda_qdot_smooth,
    double weight_base_task,
    double weight_feet_task,
    Eigen::VectorXd &qdot_solution);

bool solveHQPLowerLevel2TaskWithHardConstraints(
    const Eigen::MatrixXd &task_jacobian,
    const Eigen::VectorXd &task_velocity_command,
    const Eigen::MatrixXd &hard_constraint_jacobian,
    const Eigen::VectorXd &hard_constraint_rhs,
    const Eigen::VectorXd &qdot_reference,
    const Eigen::VectorXd &qdot_lower_limits,
    const Eigen::VectorXd &qdot_upper_limits,
    double lambda_lower_task,
    double lambda_qdot_norm,
    double lambda_qdot_smooth,
    double weight_base_task,
    double weight_feet_task,
    Eigen::VectorXd &qdot_solution);

} // namespace ecvt2_controller::wbc_hqp_lib
