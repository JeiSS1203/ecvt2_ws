#include "ecvt2_controller/wbc_hqp_lib.hpp"

#include <OsqpEigen/OsqpEigen.h>

#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <vector>

namespace ecvt2_controller::wbc_hqp_lib
{
namespace
{
Eigen::SparseMatrix<double> denseToSparse(const Eigen::MatrixXd &matrix, double tolerance = 0.0)
{
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(static_cast<size_t>(matrix.rows() * matrix.cols() / 4));
  for (int row = 0; row < matrix.rows(); ++row)
  {
    for (int col = 0; col < matrix.cols(); ++col)
    {
      const double value = matrix(row, col);
      if (std::abs(value) > tolerance)
      {
        triplets.emplace_back(row, col, value);
      }
    }
  }

  Eigen::SparseMatrix<double> sparse_matrix(matrix.rows(), matrix.cols());
  sparse_matrix.setFromTriplets(triplets.begin(), triplets.end());
  return sparse_matrix;
}
} // namespace

double orientationErrorAroundZ(const Eigen::Matrix3d &R_current, const Eigen::Matrix3d &R_desired)
{
  const Eigen::Vector3d rotation_error =
      R_current.col(0).cross(R_desired.col(0)) +
      R_current.col(1).cross(R_desired.col(1)) +
      R_current.col(2).cross(R_desired.col(2));
  return rotation_error(2);
}

Eigen::MatrixXd buildFullJacobian(const FKJacobianResult &fk_result)
{
  Eigen::MatrixXd full_jacobian = Eigen::MatrixXd::Zero(21, 26);
  full_jacobian.block<6, 6>(0, 0) = fk_result.J[0];

  full_jacobian.block<3, 6>(6, 0) = fk_result.J[1].block<3, 6>(0, 0);
  full_jacobian.block<3, 4>(6, 6) = fk_result.J[1].block<3, 4>(0, 6);

  full_jacobian.block<3, 6>(9, 0) = fk_result.J[2].block<3, 6>(0, 0);
  full_jacobian.block<3, 4>(9, 10) = fk_result.J[2].block<3, 4>(0, 6);

  full_jacobian.block<3, 6>(12, 0) = fk_result.J[3].block<3, 6>(0, 0);
  full_jacobian.block<3, 4>(12, 14) = fk_result.J[3].block<3, 4>(0, 6);

  full_jacobian.block<3, 6>(15, 0) = fk_result.J[4].block<3, 6>(0, 0);
  full_jacobian.block<3, 4>(15, 18) = fk_result.J[4].block<3, 4>(0, 6);

  full_jacobian.block<3, 6>(18, 0) = fk_result.J[5].block<3, 6>(0, 0);
  full_jacobian.block<3, 4>(18, 22) = fk_result.J[5].block<3, 4>(0, 6);
  return full_jacobian;
}

JointBoundData computeJointBoundData(
    const Eigen::Ref<const Eigen::VectorXd> &q_hw,
    const std::array<int, kModelDof> &model_to_hw_index,
    const Vec26 &q_min_full_safe,
    const Vec26 &q_max_full_safe,
    const Vec26 &qd_min_full,
    const Vec26 &qd_max_full,
    double loop_dt,
    double pos_limit_rate_scale,
    double base_velocity_limit)
{
  const double pos_to_vel_scale = 1.0 / (loop_dt * pos_limit_rate_scale);
  JointBoundData bounds;

  for (int model_joint_idx = 0; model_joint_idx < kModelDof; ++model_joint_idx)
  {
    if (model_joint_idx < 6)
    {
      bounds.qdot_lower(model_joint_idx) = -base_velocity_limit;
      bounds.qdot_upper(model_joint_idx) = base_velocity_limit;
      continue;
    }

    const int hw_joint_idx = model_to_hw_index[model_joint_idx];
    if (hw_joint_idx < 0 || hw_joint_idx >= q_hw.size())
    {
      bounds.qdot_lower(model_joint_idx) = -base_velocity_limit;
      bounds.qdot_upper(model_joint_idx) = base_velocity_limit;
      continue;
    }

    const double q_current = q_hw(hw_joint_idx);
    const double q_min_allowed = q_min_full_safe(model_joint_idx);
    const double q_max_allowed = q_max_full_safe(model_joint_idx);

    const double raw_qdot_lower = (q_min_allowed - q_current) * pos_to_vel_scale;
    const double raw_qdot_upper = (q_max_allowed - q_current) * pos_to_vel_scale;

    bounds.raw_qdot_lower(model_joint_idx) = raw_qdot_lower;
    bounds.raw_qdot_upper(model_joint_idx) = raw_qdot_upper;
    bounds.qdot_lower(model_joint_idx) = std::clamp(raw_qdot_lower, qd_min_full(model_joint_idx), qd_max_full(model_joint_idx));
    bounds.qdot_upper(model_joint_idx) = std::clamp(raw_qdot_upper, qd_min_full(model_joint_idx), qd_max_full(model_joint_idx));
  }

  return bounds;
}

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
    Eigen::Vector3d &task_slack)
{
  const int n = static_cast<int>(qdot_lower_limits.size());
  const int m_task = 3;
  const int contact_rows = static_cast<int>(contact_jacobian.rows());
  const int n_lock = static_cast<int>(locked_jacobians.size());

  const int nvar = n + m_task;
  const int ncon = m_task + 3 * n_lock + contact_rows + n;

  // QP canonical form (OSQP):
  //   minimize  1/2 z' H z + g' z
  //   subject to l <= A z <= u
  //
  // decision z = [qdot(n); slack(3)].
  // slack allows soft tracking at each hierarchy level while preserving feasibility.
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(ncon, nvar);
  Eigen::VectorXd l = Eigen::VectorXd::Zero(ncon);
  Eigen::VectorXd u = Eigen::VectorXd::Zero(ncon);

  int row = 0;

  // (1) Current level task equality with slack:
  //     J_task qdot + slack = v_task
  A.block(row, 0, 3, n) = task_jacobian;
  A.block(row, n, 3, 3) = Eigen::Matrix3d::Identity();
  l.segment<3>(row) = task_velocity_command;
  u.segment<3>(row) = task_velocity_command;
  row += 3;

  // (2) Lock higher-priority levels as hard equalities.
  //     These rhs already include the previous level slack compensation.
  for (int i = 0; i < n_lock; ++i)
  {
    A.block(row, 0, 3, n) = locked_jacobians[i];
    l.segment<3>(row) = locked_task_velocity_commands[i];
    u.segment<3>(row) = locked_task_velocity_commands[i];
    row += 3;
  }

  if (contact_rows > 0)
  {
    // (3) Contact holonomic velocity constraint:
    //     J_contact qdot = 0
    A.block(row, 0, contact_rows, n) = contact_jacobian;
    l.segment(row, contact_rows).setZero();
    u.segment(row, contact_rows).setZero();
    row += contact_rows;
  }

  // (4) Joint velocity bounds.
  A.block(row, 0, n, n) = Eigen::MatrixXd::Identity(n, n);
  l.segment(row, n) = qdot_lower_limits;
  u.segment(row, n) = qdot_upper_limits;

  // Rdiag is variable-group regularization metric.
  // Interpretation:
  // - larger weight => solver avoids moving that group
  // - smaller weight => solver uses that group more aggressively
  Eigen::VectorXd Rdiag = Eigen::VectorXd::Ones(n);
  if (n >= 6)
  {
    Rdiag.segment<6>(0).setConstant(base_weight);
  }

  if (n >= 26)
  {
    Rdiag.segment<4>(22).setConstant(arm_weight);
  }
  else if (n > 22)
  {
    const int arm_len = std::min(4, n - 22);
    Rdiag.segment(22, arm_len).setConstant(arm_weight);
  }

  // Cost construction:
  //   ||slack||^2
  // + lambda_qdot_norm   * ||qdot||_R^2
  // + lambda_qdot_smooth * ||qdot - qdot_prev||_R^2
  //
  // Expanded in OSQP form:
  //   H_qdot = 2*(lambda_norm + lambda_smooth)*R
  //   g_qdot = -2*lambda_smooth*R*qdot_prev
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(nvar, nvar);
  H.block(0, 0, n, n).diagonal().array() =
      2.0 * (lambda_qdot_norm + lambda_qdot_smooth) * Rdiag.array();
  H.block(n, n, 3, 3).diagonal().array() = 2.0;

  Eigen::VectorXd g = Eigen::VectorXd::Zero(nvar);
  g.head(n).array() =
      -2.0 * lambda_qdot_smooth * Rdiag.array() * qdot_previous.array();

  if (!H.allFinite() || !g.allFinite() || !A.allFinite() || !l.allFinite() || !u.allFinite())
  {
    return false;
  }

  OsqpEigen::Solver solver;
  solver.settings()->setVerbosity(false);
  solver.settings()->setWarmStart(false);

  solver.data()->setNumberOfVariables(nvar);
  solver.data()->setNumberOfConstraints(ncon);

  if (!solver.data()->setHessianMatrix(denseToSparse(H, 1e-12)))
  {
    return false;
  }
  if (!solver.data()->setGradient(g))
  {
    return false;
  }
  if (!solver.data()->setLinearConstraintsMatrix(denseToSparse(A, 1e-12)))
  {
    return false;
  }
  if (!solver.data()->setLowerBound(l))
  {
    return false;
  }
  if (!solver.data()->setUpperBound(u))
  {
    return false;
  }

  if (!solver.initSolver())
  {
    return false;
  }
  if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
  {
    return false;
  }

  const Eigen::VectorXd solution = solver.getSolution();
  if (!solution.allFinite())
  {
    return false;
  }
  qdot_solution = solution.head(n);
  task_slack = solution.segment<3>(n);
  return true;
}

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
    Eigen::VectorXd &qdot_upper_final_solution)
{
  // Hierarchical solve strategy:
  // - L1 solve first and record slack_l1
  // - L2 solve while locking L1 to (target - slack_l1)
  // - L3 solve while locking L1/L2 similarly
  //
  // This keeps strict task priority while allowing each level to remain feasible.
  Eigen::VectorXd qdot_l1(kModelDof), qdot_l2(kModelDof), qdot_l3(kModelDof);
  Eigen::Vector3d slack_l1, slack_l2, slack_l3;
  std::vector<Eigen::MatrixXd> locked_jacobians;
  std::vector<Eigen::Vector3d> locked_targets;
  locked_jacobians.reserve(2);
  locked_targets.reserve(2);

  locked_jacobians.clear();
  locked_targets.clear();
  if (!solveHQPUpperLevel(base_orientation_jacobian, base_orientation_cmd,
                          locked_jacobians, locked_targets,
                          contact_jacobian, qdot_lower_limits, qdot_upper_limits,
                          qdot_previous, lambda_qdot_norm, lambda_qdot_smooth,
                          base_weight, arm_weight,
                          qdot_l1, slack_l1))
  {
    return false;
  }

  locked_jacobians.clear();
  locked_targets.clear();
  locked_jacobians.push_back(base_orientation_jacobian);
  locked_targets.push_back(base_orientation_cmd - slack_l1);
  if (!solveHQPUpperLevel(end_effector_jacobian, end_effector_linear_cmd,
                          locked_jacobians, locked_targets,
                          contact_jacobian, qdot_lower_limits, qdot_upper_limits,
                          qdot_previous, lambda_qdot_norm, lambda_qdot_smooth,
                          base_weight, arm_weight,
                          qdot_l2, slack_l2))
  {
    return false;
  }

  locked_jacobians.clear();
  locked_targets.clear();
  locked_jacobians.push_back(base_orientation_jacobian);
  locked_targets.push_back(base_orientation_cmd - slack_l1);
  locked_jacobians.push_back(end_effector_jacobian);
  locked_targets.push_back(end_effector_linear_cmd - slack_l2);
  if (!solveHQPUpperLevel(base_linear_jacobian, base_linear_cmd,
                          locked_jacobians, locked_targets,
                          contact_jacobian, qdot_lower_limits, qdot_upper_limits,
                          qdot_previous, lambda_qdot_norm, lambda_qdot_smooth,
                          base_weight, arm_weight,
                          qdot_l3, slack_l3))
  {
    return false;
  }

  Eigen::VectorXd qdot_upper_final = qdot_l3;
  if (!qdot_upper_final.allFinite() || qdot_upper_final.cwiseAbs().maxCoeff() > 10.0)
  {
    qdot_upper_final = qdot_previous;
  }
  qdot_upper_final = qdot_upper_final.cwiseMax(qdot_lower_limits).cwiseMin(qdot_upper_limits);

  qdot_level3_solution = qdot_l3;
  qdot_upper_final_solution = qdot_upper_final;
  return true;
}

namespace
{
bool solveHQPLevelWithSlack(
    const Eigen::MatrixXd &task_jacobian,
    const Eigen::VectorXd &task_velocity_command,
    const std::vector<Eigen::MatrixXd> &locked_jacobians,
    const std::vector<Eigen::VectorXd> &locked_task_velocity_commands,
    const Eigen::MatrixXd &contact_jacobian,
    const Eigen::VectorXd &qdot_lower_limits,
    const Eigen::VectorXd &qdot_upper_limits,
    const Eigen::VectorXd &qdot_previous,
    double lambda_qdot_norm,
    double lambda_qdot_smooth,
    Eigen::VectorXd &qdot_solution,
    Eigen::VectorXd &task_slack)
{
  const int n = static_cast<int>(qdot_lower_limits.size());
  const int m_task = static_cast<int>(task_jacobian.rows());
  const int contact_rows = static_cast<int>(contact_jacobian.rows());
  const int n_lock = static_cast<int>(locked_jacobians.size());
  if (n <= 0 || m_task <= 0 || task_jacobian.cols() != n ||
      task_velocity_command.size() != m_task ||
      qdot_upper_limits.size() != n || qdot_previous.size() != n ||
      contact_jacobian.cols() != n || n_lock != static_cast<int>(locked_task_velocity_commands.size()))
  {
    return false;
  }

  int locked_rows_sum = 0;
  for (int i = 0; i < n_lock; ++i)
  {
    if (locked_jacobians[i].cols() != n ||
        locked_jacobians[i].rows() != locked_task_velocity_commands[i].size())
    {
      return false;
    }
    locked_rows_sum += static_cast<int>(locked_jacobians[i].rows());
  }

  const int nvar = n + m_task;
  const int ncon = m_task + locked_rows_sum + contact_rows + n;

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(ncon, nvar);
  Eigen::VectorXd l = Eigen::VectorXd::Zero(ncon);
  Eigen::VectorXd u = Eigen::VectorXd::Zero(ncon);
  int row = 0;

  A.block(row, 0, m_task, n) = task_jacobian;
  A.block(row, n, m_task, m_task) = Eigen::MatrixXd::Identity(m_task, m_task);
  l.segment(row, m_task) = task_velocity_command;
  u.segment(row, m_task) = task_velocity_command;
  row += m_task;

  for (int i = 0; i < n_lock; ++i)
  {
    const int lock_rows = static_cast<int>(locked_jacobians[i].rows());
    A.block(row, 0, lock_rows, n) = locked_jacobians[i];
    l.segment(row, lock_rows) = locked_task_velocity_commands[i];
    u.segment(row, lock_rows) = locked_task_velocity_commands[i];
    row += lock_rows;
  }

  if (contact_rows > 0)
  {
    A.block(row, 0, contact_rows, n) = contact_jacobian;
    l.segment(row, contact_rows).setZero();
    u.segment(row, contact_rows).setZero();
    row += contact_rows;
  }

  A.block(row, 0, n, n) = Eigen::MatrixXd::Identity(n, n);
  l.segment(row, n) = qdot_lower_limits;
  u.segment(row, n) = qdot_upper_limits;

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(nvar, nvar);
  H.block(0, 0, n, n).diagonal().array() =
      2.0 * (lambda_qdot_norm + lambda_qdot_smooth);
  H.block(n, n, m_task, m_task).diagonal().array() = 2.0;

  Eigen::VectorXd g = Eigen::VectorXd::Zero(nvar);
  g.head(n) = -2.0 * lambda_qdot_smooth * qdot_previous;

  if (!H.allFinite() || !g.allFinite() || !A.allFinite() || !l.allFinite() || !u.allFinite())
  {
    return false;
  }

  OsqpEigen::Solver solver;
  solver.settings()->setVerbosity(false);
  solver.settings()->setWarmStart(false);

  solver.data()->setNumberOfVariables(nvar);
  solver.data()->setNumberOfConstraints(ncon);

  if (!solver.data()->setHessianMatrix(denseToSparse(H, 1e-12)))
  {
    return false;
  }
  if (!solver.data()->setGradient(g))
  {
    return false;
  }
  if (!solver.data()->setLinearConstraintsMatrix(denseToSparse(A, 1e-12)))
  {
    return false;
  }
  if (!solver.data()->setLowerBound(l))
  {
    return false;
  }
  if (!solver.data()->setUpperBound(u))
  {
    return false;
  }
  if (!solver.initSolver())
  {
    return false;
  }
  if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
  {
    return false;
  }

  const Eigen::VectorXd solution = solver.getSolution();
  if (!solution.allFinite())
  {
    return false;
  }

  qdot_solution = solution.head(n);
  task_slack = solution.segment(n, m_task);
  return true;
}
} // namespace

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
    Eigen::VectorXd &qdot_lower_final_solution)
{
  const int n = static_cast<int>(qdot_lower_limits.size());
  if (n <= 0 ||
      wheel_jacobian.cols() != n || wheel_jacobian.rows() != 4 ||
      base_orientation_jacobian.cols() != n || base_orientation_jacobian.rows() != 3 ||
      base_linear_jacobian.cols() != n || base_linear_jacobian.rows() != 3 ||
      contact_jacobian.cols() != n ||
      qdot_upper_limits.size() != n || qdot_previous.size() != n)
  {
    return false;
  }

  Eigen::VectorXd qdot_l1(n), qdot_l2(n), qdot_l3(n);
  Eigen::VectorXd slack_l1(4), slack_l2(3), slack_l3(3);
  std::vector<Eigen::MatrixXd> locked_jacobians;
  std::vector<Eigen::VectorXd> locked_targets;

  locked_jacobians.clear();
  locked_targets.clear();
  if (!solveHQPLevelWithSlack(
          wheel_jacobian, wheel_velocity_command,
          locked_jacobians, locked_targets,
          contact_jacobian, qdot_lower_limits, qdot_upper_limits,
          qdot_previous, lambda_qdot_norm, lambda_qdot_smooth,
          qdot_l1, slack_l1))
  {
    return false;
  }

  locked_jacobians.clear();
  locked_targets.clear();
  locked_jacobians.push_back(wheel_jacobian);
  locked_targets.push_back(wheel_velocity_command - slack_l1);
  if (!solveHQPLevelWithSlack(
          base_orientation_jacobian, base_orientation_cmd,
          locked_jacobians, locked_targets,
          contact_jacobian, qdot_lower_limits, qdot_upper_limits,
          qdot_previous, lambda_qdot_norm, lambda_qdot_smooth,
          qdot_l2, slack_l2))
  {
    return false;
  }

  locked_jacobians.clear();
  locked_targets.clear();
  locked_jacobians.push_back(wheel_jacobian);
  locked_targets.push_back(wheel_velocity_command - slack_l1);
  locked_jacobians.push_back(base_orientation_jacobian);
  locked_targets.push_back(base_orientation_cmd - slack_l2);
  if (!solveHQPLevelWithSlack(
          base_linear_jacobian, base_linear_cmd,
          locked_jacobians, locked_targets,
          contact_jacobian, qdot_lower_limits, qdot_upper_limits,
          qdot_previous, lambda_qdot_norm, lambda_qdot_smooth,
          qdot_l3, slack_l3))
  {
    return false;
  }

  Eigen::VectorXd qdot_lower_final = qdot_l3;
  if (!qdot_lower_final.allFinite() || qdot_lower_final.cwiseAbs().maxCoeff() > 10.0)
  {
    qdot_lower_final = qdot_previous;
  }
  qdot_lower_final = qdot_lower_final.cwiseMax(qdot_lower_limits).cwiseMin(qdot_upper_limits);

  qdot_level3_solution = qdot_l3;
  qdot_lower_final_solution = qdot_lower_final;
  return true;
}

bool solveHQPLowerLevel1WheelTask(
    const Eigen::MatrixXd &wheel_jacobian,
    const Eigen::Vector4d &wheel_velocity_command,
    const Eigen::VectorXd &qdot_lower_limits,
    const Eigen::VectorXd &qdot_upper_limits,
    const Eigen::VectorXd &qdot_previous,
    double lambda_qdot_norm,
    double lambda_qdot_smooth,
    Eigen::VectorXd &qdot_solution)
{
  const int nvar = 22;
  const int m_task = 4;
  const int ncon = m_task + nvar;
  if (wheel_jacobian.rows() != m_task || wheel_jacobian.cols() != nvar ||
      qdot_lower_limits.size() != nvar || qdot_upper_limits.size() != nvar || qdot_previous.size() != nvar)
  {
    return false;
  }

  // L1 (wheel) objective:
  // - no tracking residual term in cost because wheel task is hard equality
  // - only regularization and smoothness
  Eigen::MatrixXd H = 2.0 * (lambda_qdot_norm + lambda_qdot_smooth) * Eigen::MatrixXd::Identity(nvar, nvar);
  Eigen::VectorXd g = -2.0 * lambda_qdot_smooth * qdot_previous;

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(ncon, nvar);
  Eigen::VectorXd l = Eigen::VectorXd::Zero(ncon);
  Eigen::VectorXd u = Eigen::VectorXd::Zero(ncon);
  int row = 0;

  A.block(row, 0, m_task, nvar) = wheel_jacobian;
  l.segment(row, m_task) = wheel_velocity_command;
  u.segment(row, m_task) = wheel_velocity_command;
  row += m_task;

  A.block(row, 0, nvar, nvar) = Eigen::MatrixXd::Identity(nvar, nvar);
  l.segment(row, nvar) = qdot_lower_limits;
  u.segment(row, nvar) = qdot_upper_limits;
  for (int i = 0; i < nvar; ++i)
  {
    if (l(m_task + i) > u(m_task + i))
    {
      const double midpoint = 0.5 * (l(m_task + i) + u(m_task + i));
      l(m_task + i) = midpoint;
      u(m_task + i) = midpoint;
    }
  }
  if (!H.allFinite() || !g.allFinite() || !A.allFinite() || !l.allFinite() || !u.allFinite())
  {
    return false;
  }

  OsqpEigen::Solver solver;
  solver.settings()->setVerbosity(false);
  solver.settings()->setWarmStart(true);
  solver.data()->setNumberOfVariables(nvar);
  solver.data()->setNumberOfConstraints(ncon);

  if (!solver.data()->setHessianMatrix(denseToSparse(H, 1e-12)))
  {
    return false;
  }
  if (!solver.data()->setGradient(g))
  {
    return false;
  }
  if (!solver.data()->setLinearConstraintsMatrix(denseToSparse(A, 1e-12)))
  {
    return false;
  }
  if (!solver.data()->setLowerBound(l))
  {
    return false;
  }
  if (!solver.data()->setUpperBound(u))
  {
    return false;
  }
  if (!solver.initSolver())
  {
    return false;
  }
  if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
  {
    return false;
  }

  qdot_solution = solver.getSolution();
  if (!qdot_solution.allFinite())
  {
    return false;
  }

  for (int i = 0; i < nvar; ++i)
  {
    qdot_solution(i) = std::min(std::max(qdot_solution(i), qdot_lower_limits(i)), qdot_upper_limits(i));
  }
  return true;
}

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
    Eigen::VectorXd &qdot_solution)
{
  const Eigen::VectorXd wheel_lock_rhs = wheel_jacobian * qdot_wheel_l1;
  return solveHQPLowerLevel2TaskWithHardConstraints(
      task_jacobian,
      task_velocity_command,
      wheel_jacobian,
      wheel_lock_rhs,
      qdot_wheel_l1,
      qdot_lower_limits,
      qdot_upper_limits,
      lambda_lower_task,
      lambda_qdot_norm,
      lambda_qdot_smooth,
      weight_base_task,
      weight_feet_task,
      qdot_solution);
}

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
    Eigen::VectorXd &qdot_solution)
{
  const int nvar = static_cast<int>(qdot_lower_limits.size());
  const int m_task = static_cast<int>(task_jacobian.rows());
  const int m_hard = static_cast<int>(hard_constraint_jacobian.rows());
  const int ncon = m_hard + nvar;
  if (nvar <= 0 || task_jacobian.cols() != nvar ||
      hard_constraint_jacobian.cols() != nvar ||
      task_velocity_command.size() != m_task ||
      hard_constraint_rhs.size() != m_hard ||
      qdot_reference.size() != nvar ||
      qdot_upper_limits.size() != nvar)
  {
    return false;
  }

  // First 6 task rows are reserved for base tracking.
  // Remaining rows (if any) are treated as "feet/other" group.
  Eigen::VectorXd Wdiag = Eigen::VectorXd::Ones(m_task);
  const int base_rows = std::min(6, m_task);
  if (base_rows > 0)
  {
    Wdiag.head(base_rows).setConstant(weight_base_task);
  }
  if (m_task > base_rows)
  {
    Wdiag.tail(m_task - base_rows).setConstant(weight_feet_task);
  }
  const Eigen::DiagonalMatrix<double, Eigen::Dynamic> W(Wdiag);

  Eigen::MatrixXd H = 2.0 * (lambda_lower_task * (task_jacobian.transpose() * W * task_jacobian) +
                              (lambda_qdot_norm + lambda_qdot_smooth) * Eigen::MatrixXd::Identity(nvar, nvar));
  Eigen::VectorXd g = -2.0 * (lambda_lower_task * (task_jacobian.transpose() * W * task_velocity_command) +
                              lambda_qdot_smooth * qdot_reference);

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(ncon, nvar);
  Eigen::VectorXd l = Eigen::VectorXd::Zero(ncon);
  Eigen::VectorXd u = Eigen::VectorXd::Zero(ncon);
  int row = 0;

  if (m_hard > 0)
  {
    A.block(row, 0, m_hard, nvar) = hard_constraint_jacobian;
    l.segment(row, m_hard) = hard_constraint_rhs;
    u.segment(row, m_hard) = hard_constraint_rhs;
    row += m_hard;
  }

  A.block(row, 0, nvar, nvar) = Eigen::MatrixXd::Identity(nvar, nvar);
  l.segment(row, nvar) = qdot_lower_limits;
  u.segment(row, nvar) = qdot_upper_limits;
  for (int i = 0; i < nvar; ++i)
  {
    if (l(row + i) > u(row + i))
    {
      const double midpoint = 0.5 * (l(row + i) + u(row + i));
      l(row + i) = midpoint;
      u(row + i) = midpoint;
    }
  }

  if (!H.allFinite() || !g.allFinite() || !A.allFinite() || !l.allFinite() || !u.allFinite())
  {
    return false;
  }

  OsqpEigen::Solver solver;
  solver.settings()->setVerbosity(false);
  solver.settings()->setWarmStart(true);
  solver.data()->setNumberOfVariables(nvar);
  solver.data()->setNumberOfConstraints(ncon);

  if (!solver.data()->setHessianMatrix(denseToSparse(H, 1e-12)))
  {
    return false;
  }
  if (!solver.data()->setGradient(g))
  {
    return false;
  }
  if (!solver.data()->setLinearConstraintsMatrix(denseToSparse(A, 1e-12)))
  {
    return false;
  }
  if (!solver.data()->setLowerBound(l))
  {
    return false;
  }
  if (!solver.data()->setUpperBound(u))
  {
    return false;
  }
  if (!solver.initSolver())
  {
    return false;
  }
  if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
  {
    return false;
  }

  qdot_solution = solver.getSolution();
  if (!qdot_solution.allFinite())
  {
    return false;
  }

  for (int i = 0; i < nvar; ++i)
  {
    qdot_solution(i) = std::min(std::max(qdot_solution(i), qdot_lower_limits(i)), qdot_upper_limits(i));
  }
  return true;
}

} // namespace ecvt2_controller::wbc_hqp_lib
