#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/joint/joint-free-flyer.hpp>

#include <ifopt/cost_term.h>
#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>
#include <ifopt/variable_set.h>

#include <casadi/casadi.hpp>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cmath>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vpsto
{

struct PlannerConfig
{
  int n_via{5};
  int n_eval{81};
  int population{40};
  int max_iterations{20};
  int elite_count{10};
  int cma_update_eig_every{5};
  int parallel_workers{1};
  int64_t random_seed{-1};

  int ipopt_print_level{5};
  int ipopt_ma97_print_level{0};

  double t_min{1.0};
  double t_max{40.0};
  double sigma0{0.20};
  double stop_sigma{1e-3};
  double infeasible_cost{1e12};
  double ipopt_tolerance{1e-3};
  double ipopt_fd_step{1e-4};
  double ipopt_via_bound_margin{1.0};
  double ipopt_max_cpu_time{300.0};
  double freeze_static_joint_tolerance{-1.0};
  std::string ipopt_linear_solver{"mumps"};
  std::string ipopt_hsl_library;
  std::string ipopt_ma97_order{"auto"};
  std::string ipopt_ma97_scaling{"dynamic"};

  double w_time{10.0};
  double w_smooth{1.0};
  double w_terminal{1.0};
  double w_passive_track{1.0};
  double w_passive_damping{40.0};
  double w_post_terminal_track{10.0};
  double w_post_terminal_energy{10.0};
  double w_via_regularization{1e-4};
  double w_base_regularization{1.0};

  bool normalize_costs{false};
  double cost_norm_eps{1e-6};
  bool solver_friendly_cost{true};
  bool smooth_time_parameterization{true};
  double time_smooth_beta{20.0};
  double time_smooth_eps{1e-6};

  double foot_contact_tolerance{1e-3};

  bool freeze_contact_legs_on_fixed_base{true};
  double base_fixed_tolerance{1e-4};

  double post_terminal_duration{2.0};
  int post_terminal_steps{40};

  bool use_zero_boundary_slopes{true};
  bool log_iteration{true};
};

struct JointInfo
{
  int idx_q{0};
  int idx_v{0};
  int nq{0};
  int nv{0};
  bool is_continuous{false};
};

struct JointStateView
{
  Eigen::VectorXd q;
  Eigen::VectorXd v;
  Eigen::VectorXd qA;
  Eigen::VectorXd vA;
  Eigen::VectorXd qP;
  Eigen::VectorXd vP;
};

struct ActuatedKinematics
{
  Eigen::VectorXd qA;
  Eigen::VectorXd dqdsA;
  Eigen::VectorXd d2qds2A;
};

struct FullTrajectorySample
{
  std::vector<double> t;
  std::vector<Eigen::VectorXd> qA;
  std::vector<Eigen::VectorXd> qdotA;
  std::vector<Eigen::VectorXd> qddA;
  std::vector<Eigen::VectorXd> qP;
  std::vector<Eigen::VectorXd> qdotP;
  std::vector<Eigen::VectorXd> qddP;
  double T{0.0};
  double smoothness_integral{0.0};
  double passive_integral{0.0};
  bool feasible{false};
};

struct CandidateEvaluation
{
  double cost{std::numeric_limits<double>::infinity()};
  bool feasible{false};
  double smooth_int{0.0};
  double passive_vel_int{0.0};
  double post_terminal_track_int{0.0};
  double post_terminal_energy_int{0.0};
  double via_reg{0.0};
  Eigen::VectorXd via_flat;
  FullTrajectorySample trajectory;
};

class CubicSpline1D
{
public:
  void build(const std::vector<double>& x,
             const std::vector<double>& y,
             bool zero_boundary_slopes)
  {
    if (x.size() < 2 || x.size() != y.size()) {
      throw std::runtime_error("CubicSpline1D requires matching knot/value vectors with at least 2 entries");
    }

    x_ = x;
    a_ = y;
    const int n = static_cast<int>(x.size()) - 1;
    b_.assign(n, 0.0);
    c_.assign(n + 1, 0.0);
    d_.assign(n, 0.0);

    std::vector<double> h(n);
    for (int i = 0; i < n; ++i) {
      h[i] = x_[i + 1] - x_[i];
      if (h[i] <= 0.0) {
        throw std::runtime_error("Spline knots must be strictly increasing");
      }
    }

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n + 1, n + 1);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n + 1);

    if (zero_boundary_slopes) {
      A(0, 0) = 2.0 * h[0];
      A(0, 1) = h[0];
      rhs(0) = 3.0 * ((a_[1] - a_[0]) / h[0]);

      A(n, n - 1) = h[n - 1];
      A(n, n) = 2.0 * h[n - 1];
      rhs(n) = -3.0 * ((a_[n] - a_[n - 1]) / h[n - 1]);
    } else {
      A(0, 0) = 1.0;
      A(n, n) = 1.0;
    }

    for (int i = 1; i < n; ++i) {
      A(i, i - 1) = h[i - 1];
      A(i, i) = 2.0 * (h[i - 1] + h[i]);
      A(i, i + 1) = h[i];
      rhs(i) = 3.0 * ((a_[i + 1] - a_[i]) / h[i] - (a_[i] - a_[i - 1]) / h[i - 1]);
    }

    Eigen::VectorXd cvec = A.fullPivLu().solve(rhs);
    for (int i = 0; i <= n; ++i) c_[i] = cvec(i);

    for (int i = 0; i < n; ++i) {
      b_[i] = (a_[i + 1] - a_[i]) / h[i] - h[i] * (2.0 * c_[i] + c_[i + 1]) / 3.0;
      d_[i] = (c_[i + 1] - c_[i]) / (3.0 * h[i]);
    }
  }

  void eval(double x, double& y, double& dydx, double& d2ydx2) const
  {
    if (x_.empty()) {
      throw std::runtime_error("Spline not built");
    }

    const int seg = findSegment(x);
    const double dx = x - x_[seg];

    y = a_[seg] + b_[seg] * dx + c_[seg] * dx * dx + d_[seg] * dx * dx * dx;
    dydx = b_[seg] + 2.0 * c_[seg] * dx + 3.0 * d_[seg] * dx * dx;
    d2ydx2 = 2.0 * c_[seg] + 6.0 * d_[seg] * dx;
  }

private:
  int findSegment(double x) const
  {
    if (x <= x_.front()) return 0;
    if (x >= x_.back()) return static_cast<int>(x_.size()) - 2;
    const auto it = std::upper_bound(x_.begin(), x_.end(), x);
    return static_cast<int>(std::max<std::ptrdiff_t>(0, (it - x_.begin()) - 1));
  }

  std::vector<double> x_;
  std::vector<double> a_, b_, c_, d_;
};

class MultiJointActuatedSpline
{
public:
  void build(const Eigen::VectorXd& q_start,
             const Eigen::VectorXd& q_goal,
             const Eigen::VectorXd& via_flat,
             int n_via,
             bool zero_boundary_slopes)
  {
    nA_ = static_cast<int>(q_start.size());

    if (q_goal.size() != q_start.size()) {
      throw std::runtime_error("q_start and q_goal size mismatch");
    }
    if (via_flat.size() != nA_ * n_via) {
      throw std::runtime_error("via_flat dimension mismatch");
    }

    const int knots = n_via + 2;
    std::vector<double> s(knots, 0.0);
    for (int i = 0; i < knots; ++i) {
      s[i] = static_cast<double>(i) / static_cast<double>(knots - 1);
    }

    splines_.clear();
    splines_.resize(nA_);

    for (int j = 0; j < nA_; ++j) {
      std::vector<double> y(knots);
      y[0] = q_start[j];
      for (int k = 0; k < n_via; ++k) {
        y[k + 1] = via_flat[k * nA_ + j];
      }
      y[knots - 1] = q_goal[j];
      splines_[j].build(s, y, zero_boundary_slopes);
    }
  }

  ActuatedKinematics eval(double s) const
  {
    ActuatedKinematics out;
    out.qA = Eigen::VectorXd::Zero(nA_);
    out.dqdsA = Eigen::VectorXd::Zero(nA_);
    out.d2qds2A = Eigen::VectorXd::Zero(nA_);

    const double sc = std::min(1.0, std::max(0.0, s));
    for (int j = 0; j < nA_; ++j) {
      double y = 0.0, dy = 0.0, d2y = 0.0;
      splines_[j].eval(sc, y, dy, d2y);
      out.qA[j] = y;
      out.dqdsA[j] = dy;
      out.d2qds2A[j] = d2y;
    }
    return out;
  }

private:
  int nA_{0};
  std::vector<CubicSpline1D> splines_;
};

class CraneDynamicsModel
{
public:
  CraneDynamicsModel(const std::string& urdf_path,
                     const std::vector<std::string>& actuated_joints,
                     const std::vector<std::string>& passive_joints)
  : actuated_joints_(actuated_joints), passive_joints_(passive_joints)
  {
    pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), model_);
    data_ = std::make_unique<pinocchio::Data>(model_);
    registerJoints(actuated_joints_, idxA_q_, idxA_v_);
    registerJoints(passive_joints_, idxP_q_, idxP_v_);
  }

  const std::vector<std::string>& actuatedJointNames() const { return actuated_joints_; }
  const std::vector<std::string>& passiveJointNames() const { return passive_joints_; }
  int nA() const { return static_cast<int>(idxA_v_.size()); }
  int nP() const { return static_cast<int>(idxP_v_.size()); }

  pinocchio::FrameIndex frameId(const std::string& name) const
  {
    const auto id = model_.getFrameId(name);
    if (id == static_cast<pinocchio::FrameIndex>(model_.nframes)) {
      throw std::runtime_error("Pinocchio frame not found: " + name);
    }
    return id;
  }

  bool hasFreeFlyer() const
  {
    return model_.njoints > 1 && model_.joints[1].nq() == 7 && model_.joints[1].nv() == 6;
  }

  double passiveTerminalKineticEnergy(const Eigen::VectorXd& qA,
                                      const Eigen::VectorXd& qP,
                                      const Eigen::VectorXd& qdotP) const
  {
    if (qA.size() != nA() || qP.size() != nP() || qdotP.size() != nP()) {
      return std::numeric_limits<double>::infinity();
    }
    if (nP() == 0) {
      return 0.0;
    }

    const Eigen::VectorXd q_full = composeConfiguration(qA, qP);
    pinocchio::Data data(model_);
    pinocchio::crba(model_, data, q_full);
    Eigen::MatrixXd M = data.M;
    M.triangularView<Eigen::StrictlyLower>() =
      M.transpose().triangularView<Eigen::StrictlyLower>();

    Eigen::MatrixXd Mpp = Eigen::MatrixXd::Zero(nP(), nP());
    for (int r = 0; r < nP(); ++r) {
      for (int c = 0; c < nP(); ++c) {
        Mpp(r, c) = M(idxP_v_[static_cast<size_t>(r)], idxP_v_[static_cast<size_t>(c)]);
      }
    }

    return 0.5 * qdotP.transpose() * Mpp * qdotP;
  }

  Eigen::VectorXd passiveAcceleration(const Eigen::VectorXd& qA,
                                      const Eigen::VectorXd& qdotA,
                                      const Eigen::VectorXd& qddA,
                                      const Eigen::VectorXd& qP,
                                      const Eigen::VectorXd& qdotP) const
  {
    if (qA.size() != nA() || qdotA.size() != nA() || qddA.size() != nA() ||
        qP.size() != nP() || qdotP.size() != nP()) {
      return Eigen::VectorXd::Constant(nP(), std::numeric_limits<double>::quiet_NaN());
    }
    if (nP() == 0) {
      return Eigen::VectorXd::Zero(0);
    }

    const Eigen::VectorXd q_full = composeConfiguration(qA, qP);
    const Eigen::VectorXd v_full = composeVelocity(qdotA, qdotP);

    pinocchio::Data data(model_);
    pinocchio::crba(model_, data, q_full);
    Eigen::MatrixXd M = data.M;
    M.triangularView<Eigen::StrictlyLower>() =
      M.transpose().triangularView<Eigen::StrictlyLower>();

    const Eigen::VectorXd nle = pinocchio::nonLinearEffects(model_, data, q_full, v_full);

    Eigen::MatrixXd Mpp = Eigen::MatrixXd::Zero(nP(), nP());
    Eigen::MatrixXd Mpa = Eigen::MatrixXd::Zero(nP(), nA());
    Eigen::VectorXd nleP = Eigen::VectorXd::Zero(nP());
    for (int r = 0; r < nP(); ++r) {
      const int row = idxP_v_[static_cast<size_t>(r)];
      nleP[r] = nle[row];
      for (int c = 0; c < nP(); ++c) {
        Mpp(r, c) = M(row, idxP_v_[static_cast<size_t>(c)]);
      }
      for (int c = 0; c < nA(); ++c) {
        Mpa(r, c) = M(row, idxA_v_[static_cast<size_t>(c)]);
      }
    }

    return Mpp.ldlt().solve(-(Mpa * qddA + nleP));
  }

  bool solvePassiveEquilibrium(const Eigen::VectorXd& qA,
                               const Eigen::VectorXd& qP_seed,
                               Eigen::VectorXd& qP_eq,
                               int max_iter = 12,
                               double tol = 1e-4,
                               double damping = 1e-4) const
  {
    if (qA.size() != nA() || qP_seed.size() != nP()) {
      return false;
    }
    if (nP() == 0) {
      qP_eq.resize(0);
      return true;
    }

    Eigen::VectorXd qP = qP_seed;
    pinocchio::Data data(model_);
    for (int iter = 0; iter < max_iter; ++iter) {
      const Eigen::VectorXd gP = passiveGravity(qA, qP, data);
      if (!allFinite(gP)) {
        return false;
      }
      if (gP.norm() < tol) {
        qP_eq = qP;
        return true;
      }

      Eigen::MatrixXd J = Eigen::MatrixXd::Zero(nP(), nP());
      const double eps = 1e-5;
      for (int j = 0; j < nP(); ++j) {
        Eigen::VectorXd qP_eps = qP;
        qP_eps[j] += eps;
        const Eigen::VectorXd g_eps = passiveGravity(qA, qP_eps, data);
        if (!allFinite(g_eps)) {
          return false;
        }
        J.col(j) = (g_eps - gP) / eps;
      }

      const Eigen::MatrixXd H = J.transpose() * J + damping * Eigen::MatrixXd::Identity(nP(), nP());
      const Eigen::VectorXd rhs = J.transpose() * gP;
      const Eigen::VectorXd delta = H.ldlt().solve(rhs);
      if (!allFinite(delta)) {
        return false;
      }

      qP -= delta;
      if (delta.norm() < tol) {
        qP_eq = qP;
        return true;
      }
    }

    qP_eq = qP;
    const Eigen::VectorXd gP = passiveGravity(qA, qP_eq, data);
    return allFinite(gP) && (gP.norm() < 1e-3);
  }

  JointStateView fromJointMsg(const sensor_msgs::msg::JointState& msg) const
  {
    if (msg.name.size() != msg.position.size() || msg.name.size() != msg.velocity.size()) {
      throw std::runtime_error("JointState arrays have inconsistent length");
    }

    std::unordered_map<std::string, size_t> ros_idx;
    for (size_t i = 0; i < msg.name.size(); ++i) {
      ros_idx[msg.name[i]] = i;
    }

    JointStateView out;
    out.q = pinocchio::neutral(model_);
    out.v = Eigen::VectorXd::Zero(model_.nv);

    auto fill = [&](const std::string& name) {
      const auto it_ros = ros_idx.find(name);
      if (it_ros == ros_idx.end()) {
        throw std::runtime_error("Joint '" + name + "' missing in JointState");
      }

      const JointInfo& info = joint_info_.at(name);
      const double theta = msg.position[it_ros->second];
      const double dtheta = msg.velocity[it_ros->second];

      if (info.is_continuous) {
        out.q[info.idx_q] = std::cos(theta);
        out.q[info.idx_q + 1] = std::sin(theta);
        out.v[info.idx_v] = dtheta;
      } else {
        out.q[info.idx_q] = theta;
        out.v[info.idx_v] = dtheta;
      }
    };

    for (const auto& name : actuated_joints_) fill(name);
    for (const auto& name : passive_joints_) fill(name);

    out.qA = Eigen::VectorXd::Zero(static_cast<int>(actuated_joints_.size()));
    out.vA = Eigen::VectorXd::Zero(static_cast<int>(actuated_joints_.size()));
    for (size_t i = 0; i < actuated_joints_.size(); ++i) {
      const auto it_ros = ros_idx.find(actuated_joints_[i]);
      out.qA[static_cast<int>(i)] = msg.position[it_ros->second];
      out.vA[static_cast<int>(i)] = msg.velocity[it_ros->second];
    }

    out.qP = Eigen::VectorXd::Zero(static_cast<int>(passive_joints_.size()));
    out.vP = Eigen::VectorXd::Zero(static_cast<int>(passive_joints_.size()));
    for (size_t i = 0; i < passive_joints_.size(); ++i) {
      const auto it_ros = ros_idx.find(passive_joints_[i]);
      out.qP[static_cast<int>(i)] = msg.position[it_ros->second];
      out.vP[static_cast<int>(i)] = msg.velocity[it_ros->second];
    }

    return out;
  }

  Eigen::VectorXd composeConfigurationWithBase(const Eigen::VectorXd& base_xyzrpy,
                                               const Eigen::VectorXd& qA,
                                               const Eigen::VectorXd& qP) const
  {
    Eigen::VectorXd q_full = pinocchio::neutral(model_);
    setBasePose(q_full, base_xyzrpy);
    for (size_t i = 0; i < actuated_joints_.size(); ++i) {
      const auto& info = joint_info_.at(actuated_joints_[i]);
      setJointScalar(q_full, info, qA[static_cast<int>(i)]);
    }
    for (size_t i = 0; i < passive_joints_.size(); ++i) {
      const auto& info = joint_info_.at(passive_joints_[i]);
      setJointScalar(q_full, info, qP[static_cast<int>(i)]);
    }
    return q_full;
  }

  std::vector<Eigen::Vector3d> footPositions(const Eigen::VectorXd& base_xyzrpy,
                                             const Eigen::VectorXd& qA,
                                             const Eigen::VectorXd& qP,
                                             const std::vector<pinocchio::FrameIndex>& frame_ids) const
  {
    std::vector<Eigen::Vector3d> out;
    out.reserve(frame_ids.size());
    const Eigen::VectorXd q_full = composeConfigurationWithBase(base_xyzrpy, qA, qP);
    pinocchio::Data data(model_);
    pinocchio::forwardKinematics(model_, data, q_full);
    pinocchio::updateFramePlacements(model_, data);

    for (const auto frame_id : frame_ids) {
      out.push_back(data.oMf[frame_id].translation());
    }
    return out;
  }

  Eigen::VectorXd footPositionError(const Eigen::VectorXd& base_xyzrpy,
                                    const Eigen::VectorXd& qA,
                                    const Eigen::VectorXd& qP,
                                    const std::vector<pinocchio::FrameIndex>& frame_ids,
                                    const Eigen::VectorXd& foot_pos_ref) const
  {
    const auto positions = footPositions(base_xyzrpy, qA, qP, frame_ids);
    Eigen::VectorXd err(static_cast<int>(positions.size()) * 3);
    for (size_t i = 0; i < positions.size(); ++i) {
      err.segment<3>(static_cast<int>(i) * 3) =
        positions[i] - foot_pos_ref.segment<3>(static_cast<int>(i) * 3);
    }
    return err;
  }

  Eigen::MatrixXd footPositionJacobian(const Eigen::VectorXd& base_xyzrpy,
                                       const Eigen::VectorXd& qA,
                                       const Eigen::VectorXd& qP,
                                       const std::vector<pinocchio::FrameIndex>& frame_ids) const
  {
    const int base_dim = 6;
    if (base_xyzrpy.size() != base_dim || qA.size() != nA() || qP.size() != nP()) {
      throw std::runtime_error("Foot Jacobian input size mismatch");
    }

    const Eigen::VectorXd q_full = composeConfigurationWithBase(base_xyzrpy, qA, qP);
    pinocchio::Data data(model_);

    Eigen::MatrixXd Jout =
      Eigen::MatrixXd::Zero(static_cast<int>(frame_ids.size()) * 3, base_dim + nA());
    const Eigen::Matrix3d rpy_to_world_omega = rpyRateToWorldAngular(base_xyzrpy.tail<3>());

    for (size_t f = 0; f < frame_ids.size(); ++f) {
      Eigen::Matrix<double, 6, Eigen::Dynamic> J6(6, model_.nv);
      J6.setZero();
      pinocchio::computeFrameJacobian(
        model_,
        data,
        q_full,
        frame_ids[f],
        pinocchio::LOCAL_WORLD_ALIGNED,
        J6);

      const int row = static_cast<int>(f) * 3;
      Jout.block(row, 0, 3, 3) = J6.block(0, 0, 3, 3);
      Jout.block(row, 3, 3, 3) = J6.block(0, 3, 3, 3) * rpy_to_world_omega;
      for (int j = 0; j < nA(); ++j) {
        Jout.block(row, base_dim + j, 3, 1) = J6.block(0, idxA_v_[static_cast<size_t>(j)], 3, 1);
      }
    }

    return Jout;
  }

  bool projectBaseAndActuatedToFootContact(Eigen::VectorXd& base_xyzrpy,
                                           Eigen::VectorXd& qA,
                                           const Eigen::VectorXd& qP,
                                           const std::vector<pinocchio::FrameIndex>& frame_ids,
                                           const Eigen::VectorXd& foot_pos_ref,
                                           const Eigen::VectorXd& lower,
                                           const Eigen::VectorXd& upper,
                                           int max_iter = 10,
                                           double tol = 1e-5,
                                           double damping = 1e-5) const
  {
    const int base_dim = 6;
    const int nA_local = nA();
    const int nvar = base_dim + nA_local;
    if (base_xyzrpy.size() != base_dim || qA.size() != nA_local ||
        lower.size() != nvar || upper.size() != nvar) {
      return false;
    }

    for (int iter = 0; iter < max_iter; ++iter) {
      const Eigen::VectorXd err = footPositionError(base_xyzrpy, qA, qP, frame_ids, foot_pos_ref);
      if (!allFinite(err)) {
        return false;
      }
      if (err.norm() < tol) {
        return true;
      }

      Eigen::MatrixXd J(err.size(), nvar);
      const double eps = 1e-5;
      for (int j = 0; j < nvar; ++j) {
        Eigen::VectorXd base_eps = base_xyzrpy;
        Eigen::VectorXd qA_eps = qA;
        const double h = eps * (1.0 + std::abs(j < base_dim ? base_xyzrpy[j] : qA[j - base_dim]));
        if (j < base_dim) {
          base_eps[j] += h;
        } else {
          qA_eps[j - base_dim] += h;
        }
        const Eigen::VectorXd err_eps =
          footPositionError(base_eps, qA_eps, qP, frame_ids, foot_pos_ref);
        if (!allFinite(err_eps)) {
          return false;
        }
        J.col(j) = (err_eps - err) / h;
      }

      const Eigen::MatrixXd H =
        J.transpose() * J + damping * Eigen::MatrixXd::Identity(nvar, nvar);
      const Eigen::VectorXd rhs = J.transpose() * err;
      Eigen::VectorXd delta = H.ldlt().solve(rhs);
      if (!allFinite(delta)) {
        return false;
      }

      double alpha = 1.0;
      bool accepted = false;
      const double err_norm = err.norm();
      for (int ls = 0; ls < 8; ++ls) {
        Eigen::VectorXd base_trial = base_xyzrpy - alpha * delta.head(base_dim);
        Eigen::VectorXd qA_trial = qA - alpha * delta.tail(nA_local);
        for (int j = 0; j < nvar; ++j) {
          if (j < base_dim) {
            base_trial[j] = std::min(upper[j], std::max(lower[j], base_trial[j]));
          } else {
            const int ia = j - base_dim;
            qA_trial[ia] = std::min(upper[j], std::max(lower[j], qA_trial[ia]));
          }
        }

        const Eigen::VectorXd err_trial =
          footPositionError(base_trial, qA_trial, qP, frame_ids, foot_pos_ref);
        if (allFinite(err_trial) && err_trial.norm() < err_norm) {
          base_xyzrpy = base_trial;
          qA = qA_trial;
          accepted = true;
          break;
        }
        alpha *= 0.5;
      }

      if (!accepted) {
        return false;
      }
    }

    const Eigen::VectorXd err = footPositionError(base_xyzrpy, qA, qP, frame_ids, foot_pos_ref);
    return allFinite(err) && err.norm() < std::max(tol, 1e-6);
  }

private:
  static bool allFinite(const Eigen::VectorXd& x)
  {
    for (int i = 0; i < x.size(); ++i) {
      if (!std::isfinite(x[i])) return false;
    }
    return true;
  }

  static Eigen::Matrix3d rpyRateToWorldAngular(const Eigen::Vector3d& rpy)
  {
    const double roll = rpy[0];
    const double pitch = rpy[1];
    const double yaw = rpy[2];

    const Eigen::AngleAxisd ry(pitch, Eigen::Vector3d::UnitY());
    const Eigen::AngleAxisd rz(yaw, Eigen::Vector3d::UnitZ());

    Eigen::Matrix3d E;
    E.col(0) = (rz * ry).toRotationMatrix() * Eigen::Vector3d::UnitX();
    E.col(1) = rz.toRotationMatrix() * Eigen::Vector3d::UnitY();
    E.col(2) = Eigen::Vector3d::UnitZ();
    (void)roll;
    return E;
  }

  void setJointScalar(Eigen::VectorXd& q_full, const JointInfo& info, double q_scalar) const
  {
    if (info.is_continuous) {
      q_full[info.idx_q] = std::cos(q_scalar);
      q_full[info.idx_q + 1] = std::sin(q_scalar);
    } else {
      q_full[info.idx_q] = q_scalar;
    }
  }

  Eigen::VectorXd composeConfiguration(const Eigen::VectorXd& qA,
                                       const Eigen::VectorXd& qP) const
  {
    Eigen::VectorXd q_full = pinocchio::neutral(model_);
    for (size_t i = 0; i < actuated_joints_.size(); ++i) {
      const auto& info = joint_info_.at(actuated_joints_[i]);
      setJointScalar(q_full, info, qA[static_cast<int>(i)]);
    }
    for (size_t i = 0; i < passive_joints_.size(); ++i) {
      const auto& info = joint_info_.at(passive_joints_[i]);
      setJointScalar(q_full, info, qP[static_cast<int>(i)]);
    }
    return q_full;
  }

  Eigen::VectorXd composeVelocity(const Eigen::VectorXd& qdotA,
                                  const Eigen::VectorXd& qdotP) const
  {
    Eigen::VectorXd v_full = Eigen::VectorXd::Zero(model_.nv);
    if (hasFreeFlyer() && v_full.size() >= 6) {
      for (int i = 0; i < 6; ++i) {
        v_full[i] = 0.0;
      }
    }
    for (size_t i = 0; i < actuated_joints_.size(); ++i) {
      const auto& info = joint_info_.at(actuated_joints_[i]);
      v_full[info.idx_v] = qdotA[static_cast<int>(i)];
    }
    for (size_t i = 0; i < passive_joints_.size(); ++i) {
      const auto& info = joint_info_.at(passive_joints_[i]);
      v_full[info.idx_v] = qdotP[static_cast<int>(i)];
    }
    return v_full;
  }

  Eigen::VectorXd passiveGravity(const Eigen::VectorXd& qA,
                                 const Eigen::VectorXd& qP) const
  {
    pinocchio::Data data(model_);
    return passiveGravity(qA, qP, data);
  }

  Eigen::VectorXd passiveGravity(const Eigen::VectorXd& qA,
                                 const Eigen::VectorXd& qP,
                                 pinocchio::Data& data) const
  {
    const Eigen::VectorXd q_full = composeConfiguration(qA, qP);
    const Eigen::VectorXd g = pinocchio::computeGeneralizedGravity(model_, data, q_full);
    Eigen::VectorXd gP(nP());
    for (int i = 0; i < nP(); ++i) {
      gP[i] = g[idxP_v_[static_cast<size_t>(i)]];
    }
    return gP;
  }

  void registerJoints(const std::vector<std::string>& names,
                      std::vector<int>& q_idx,
                      std::vector<int>& v_idx)
  {
    for (const auto& name : names) {
      const auto jid = model_.getJointId(name);
      if (jid == 0) {
        throw std::runtime_error("Pinocchio joint not found: " + name);
      }

      const auto& jm = model_.joints[jid];
      JointInfo info;
      info.idx_q = static_cast<int>(jm.idx_q());
      info.idx_v = static_cast<int>(jm.idx_v());
      info.nq = static_cast<int>(jm.nq());
      info.nv = static_cast<int>(jm.nv());
      info.is_continuous = (info.nq == 2 && info.nv == 1);

      if (!((info.nq == 1 && info.nv == 1) || info.is_continuous)) {
        std::ostringstream oss;
        oss << "Unsupported joint layout for " << name
            << ": nq=" << info.nq << ", nv=" << info.nv;
        throw std::runtime_error(oss.str());
      }

      joint_info_[name] = info;
      q_idx.push_back(info.idx_q);
      v_idx.push_back(info.idx_v);
    }
  }

  pinocchio::Model model_;
  mutable std::unique_ptr<pinocchio::Data> data_;
  std::vector<std::string> actuated_joints_;
  std::vector<std::string> passive_joints_;
  std::unordered_map<std::string, JointInfo> joint_info_;
  std::vector<int> idxA_q_, idxA_v_, idxP_q_, idxP_v_;

  void setBasePose(Eigen::VectorXd& q_full, const Eigen::VectorXd& base_xyzrpy) const
  {
    if (!hasFreeFlyer()) {
      return;
    }
    if (base_xyzrpy.size() != 6 || q_full.size() < 7) {
      throw std::runtime_error("Base pose must be size 6 for free-flyer configuration");
    }

    const double roll = base_xyzrpy[3];
    const double pitch = base_xyzrpy[4];
    const double yaw = base_xyzrpy[5];
    const Eigen::AngleAxisd rx(roll, Eigen::Vector3d::UnitX());
    const Eigen::AngleAxisd ry(pitch, Eigen::Vector3d::UnitY());
    const Eigen::AngleAxisd rz(yaw, Eigen::Vector3d::UnitZ());
    const Eigen::Quaterniond q = rz * ry * rx;

    q_full[0] = base_xyzrpy[0];
    q_full[1] = base_xyzrpy[1];
    q_full[2] = base_xyzrpy[2];
    q_full[3] = q.x();
    q_full[4] = q.y();
    q_full[5] = q.z();
    q_full[6] = q.w();
  }
};

class TimeParameterizer
{
public:
  TimeParameterizer(Eigen::VectorXd vmax,
                    Eigen::VectorXd amax)
  : vmax_(std::move(vmax)), amax_(std::move(amax)) {}

  double computeMinimumTime(const std::vector<ActuatedKinematics>& kin,
                            double t_min,
                            double t_max,
                            bool smooth_time = false,
                            double smooth_beta = 20.0,
                            double smooth_eps = 1e-6) const
  {
    if (smooth_time) {
      std::vector<double> candidates;
      candidates.reserve(
        1 + kin.size() * 2 * static_cast<size_t>(std::max<Eigen::Index>(1, vmax_.size())));
      candidates.push_back(t_min);

      const double eps = std::max(1e-12, smooth_eps);
      for (const auto& k : kin) {
        for (int i = 0; i < k.qA.size(); ++i) {
          const double abs_dqds = std::sqrt(k.dqdsA[i] * k.dqdsA[i] + eps * eps);
          const double abs_d2qds2 = std::sqrt(k.d2qds2A[i] * k.d2qds2A[i] + eps * eps);
          candidates.push_back(abs_dqds / std::max(1e-9, vmax_[i]));
          candidates.push_back(std::sqrt(abs_d2qds2 / std::max(1e-9, amax_[i])));
        }
      }

      const double beta = std::max(1.0, smooth_beta);
      const double m = *std::max_element(candidates.begin(), candidates.end());
      double exp_sum = 0.0;
      for (const double v : candidates) {
        exp_sum += std::exp(beta * (v - m));
      }
      const double smooth_max = m + std::log(exp_sum) / beta;
      return std::min(t_max, std::max(t_min, smooth_max));
    }

    double t_vel = t_min;
    double t_acc = t_min;

    for (const auto& k : kin) {
      for (int i = 0; i < k.qA.size(); ++i) {
        t_vel = std::max(t_vel, std::abs(k.dqdsA[i]) / std::max(1e-9, vmax_[i]));
        t_acc = std::max(t_acc, std::sqrt(std::abs(k.d2qds2A[i]) / std::max(1e-9, amax_[i])));
      }
    }

    return std::min(t_max, std::max(t_min, std::max(t_vel, t_acc)));
  }

private:
  Eigen::VectorXd vmax_;
  Eigen::VectorXd amax_;
};

class TrajectoryRolloutEvaluator
{
public:
  struct CostNormalization
  {
    bool enabled{false};
    double time{1.0};
    double smooth{1.0};
    double passive_vel{1.0};
    double post_track{1.0};
    double post_energy{1.0};
    double via_reg{1.0};
  };

  TrajectoryRolloutEvaluator(const PlannerConfig& cfg,
                             Eigen::VectorXd q_goal_A,
                             Eigen::VectorXd q_goal_P_eq,
                             const CraneDynamicsModel* dynamics)
  : cfg_(cfg),
    q_goal_A_(std::move(q_goal_A)),
    q_goal_P_eq_(std::move(q_goal_P_eq)),
    dynamics_(dynamics) {}

  void setNormalization(const CostNormalization& norm)
  {
    normalization_ = norm;
  }

  CandidateEvaluation evaluate(const Eigen::VectorXd& via_flat,
                               const JointStateView& start,
                               TimeParameterizer* time_param) const
  {
    CandidateEvaluation result;
    result.via_flat = via_flat;

    MultiJointActuatedSpline spline;
    spline.build(start.qA, q_goal_A_, via_flat, cfg_.n_via, cfg_.use_zero_boundary_slopes);

    std::vector<ActuatedKinematics> kin(cfg_.n_eval);
    std::vector<double> s_grid(cfg_.n_eval);
    for (int k = 0; k < cfg_.n_eval; ++k) {
      const double s = static_cast<double>(k) / static_cast<double>(cfg_.n_eval - 1);
      s_grid[k] = s;
      kin[k] = spline.eval(s);
    }

    const double T = time_param->computeMinimumTime(
      kin,
      cfg_.t_min,
      cfg_.t_max,
      cfg_.smooth_time_parameterization,
      cfg_.time_smooth_beta,
      cfg_.time_smooth_eps);
    if (!(T > 0.0) || std::isnan(T) || std::isinf(T)) {
      result.cost = cfg_.infeasible_cost;
      result.feasible = false;
      return result;
    }

    FullTrajectorySample traj;
    traj.T = T;
    traj.feasible = true;

    if (dynamics_ == nullptr) {
      result.cost = cfg_.infeasible_cost;
      result.feasible = false;
      return result;
    }
    if (start.qP.size() != dynamics_->nP() || q_goal_P_eq_.size() != dynamics_->nP()) {
      result.cost = cfg_.infeasible_cost;
      result.feasible = false;
      return result;
    }

    const double dt_eval = T / static_cast<double>(std::max(1, cfg_.n_eval - 1));

    double smooth_int = 0.0;
    double passive_track_int = 0.0;
    double passive_vel_int = 0.0;

    Eigen::VectorXd prev_qP = start.qP;
    Eigen::VectorXd prev_qdotP = Eigen::VectorXd::Zero(start.qP.size());

    for (int k = 0; k < cfg_.n_eval; ++k) {
      const double s = s_grid[k];
      const double t = s * T;
      const auto& kk = kin[k];

      const Eigen::VectorXd qdotA = kk.dqdsA / T;
      const Eigen::VectorXd qddA = kk.d2qds2A / (T * T);

      if (!allFinite(kk.qA) || !allFinite(qdotA) || !allFinite(qddA)) {
        result.cost = cfg_.infeasible_cost;
        result.feasible = false;
        return result;
      }

      traj.t.push_back(t);
      traj.qA.push_back(kk.qA);
      traj.qdotA.push_back(qdotA);
      traj.qddA.push_back(qddA);

      Eigen::VectorXd qP_eq;
      if (!dynamics_->solvePassiveEquilibrium(kk.qA, prev_qP, qP_eq)) {
        result.cost = cfg_.infeasible_cost;
        result.feasible = false;
        return result;
      }

      Eigen::VectorXd qdotP = Eigen::VectorXd::Zero(qP_eq.size());
      Eigen::VectorXd qddP = Eigen::VectorXd::Zero(qP_eq.size());
      if (k > 0 && dt_eval > 1e-9) {
        qdotP = (qP_eq - prev_qP) / dt_eval;
      }
      if (k > 1 && dt_eval > 1e-9) {
        qddP = (qdotP - prev_qdotP) / dt_eval;
      }

      traj.qP.push_back(qP_eq);
      traj.qdotP.push_back(qdotP);
      traj.qddP.push_back(qddP);

      const double weight = (k == 0 || k == cfg_.n_eval - 1) ? 0.5 : 1.0;
      smooth_int += weight * qddA.squaredNorm();

      const Eigen::VectorXd qP_ref = (1.0 - s) * start.qP + s * q_goal_P_eq_;
      const Eigen::VectorXd passive_err = qP_eq - qP_ref;
      passive_track_int += weight * passive_err.squaredNorm();
      passive_vel_int += weight * qdotP.squaredNorm();

      prev_qP = qP_eq;
      prev_qdotP = qdotP;
    }

    smooth_int *= dt_eval;
    passive_track_int *= dt_eval;
    passive_vel_int *= dt_eval;

    double post_terminal_track_int = 0.0;
    double post_terminal_energy_int = 0.0;
    if (dynamics_->nP() > 0) {
      const double tail_start_s =
        static_cast<double>(cfg_.n_via) / static_cast<double>(cfg_.n_via + 1);
      int tail_start_idx = 0;
      while (tail_start_idx < cfg_.n_eval - 1 && s_grid[tail_start_idx] < tail_start_s) {
        ++tail_start_idx;
      }

      Eigen::VectorXd qP_dyn = traj.qP[static_cast<size_t>(tail_start_idx)];
      Eigen::VectorXd qdotP_dyn = traj.qdotP[static_cast<size_t>(tail_start_idx)];
      for (int k = tail_start_idx; k < cfg_.n_eval; ++k) {
        const Eigen::VectorXd qddP_dyn = dynamics_->passiveAcceleration(
          traj.qA[static_cast<size_t>(k)],
          traj.qdotA[static_cast<size_t>(k)],
          traj.qddA[static_cast<size_t>(k)],
          qP_dyn,
          qdotP_dyn);
        if (!allFinite(qddP_dyn)) {
          result.cost = cfg_.infeasible_cost;
          result.feasible = false;
          return result;
        }

        if (k < cfg_.n_eval - 1 && dt_eval > 1e-9) {
          qdotP_dyn += dt_eval * qddP_dyn;
          qP_dyn += dt_eval * qdotP_dyn;
        }
      }

      if (cfg_.post_terminal_duration > 0.0 && cfg_.post_terminal_steps > 0) {
        const double post_dt =
          cfg_.post_terminal_duration / static_cast<double>(cfg_.post_terminal_steps);
        const Eigen::VectorXd qdotA_hold = Eigen::VectorXd::Zero(q_goal_A_.size());
        const Eigen::VectorXd qddA_hold = Eigen::VectorXd::Zero(q_goal_A_.size());
        Eigen::VectorXd qP_post = qP_dyn;
        Eigen::VectorXd qdotP_post = qdotP_dyn;

        for (int k = 0; k <= cfg_.post_terminal_steps; ++k) {
          const double weight = (k == 0 || k == cfg_.post_terminal_steps) ? 0.5 : 1.0;
          post_terminal_track_int += weight * (qP_post - q_goal_P_eq_).squaredNorm();

          const double kinetic = dynamics_->passiveTerminalKineticEnergy(
            q_goal_A_, qP_post, qdotP_post);
          if (!std::isfinite(kinetic)) {
            result.cost = cfg_.infeasible_cost;
            result.feasible = false;
            return result;
          }
          post_terminal_energy_int += weight * kinetic;

          if (k < cfg_.post_terminal_steps) {
            const Eigen::VectorXd qddP_post = dynamics_->passiveAcceleration(
              q_goal_A_, qdotA_hold, qddA_hold, qP_post, qdotP_post);
            if (!allFinite(qddP_post)) {
              result.cost = cfg_.infeasible_cost;
              result.feasible = false;
              return result;
            }

            qdotP_post += post_dt * qddP_post;
            qP_post += post_dt * qdotP_post;
          }
        }

        post_terminal_track_int *= post_dt;
        post_terminal_energy_int *= post_dt;
      }
    }

    traj.smoothness_integral = smooth_int;
    traj.passive_integral = passive_track_int + passive_vel_int;

    const Eigen::VectorXd terminal_error = traj.qA.back() - q_goal_A_;
    const double via_reg = via_flat.squaredNorm();

    double time_term = T;
    double smooth_term = smooth_int;
    double passive_vel_term = passive_vel_int;
    double post_track_term = post_terminal_track_int;
    double post_energy_term = post_terminal_energy_int;
    double via_reg_term = via_reg;

    if (normalization_.enabled) {
      time_term /= normalization_.time;
      smooth_term /= normalization_.smooth;
      passive_vel_term /= normalization_.passive_vel;
      post_track_term /= normalization_.post_track;
      post_energy_term /= normalization_.post_energy;
      via_reg_term /= normalization_.via_reg;
    }

    if (cfg_.solver_friendly_cost) {
      time_term = std::log1p(std::max(0.0, time_term));
      passive_vel_term = std::log1p(std::max(0.0, passive_vel_term));
      post_track_term = std::log1p(std::max(0.0, post_track_term));
      post_energy_term = std::log1p(std::max(0.0, post_energy_term));
    }

    result.cost = cfg_.w_time * time_term
                + cfg_.w_smooth * smooth_term
                + cfg_.w_passive_damping * passive_vel_term
                + cfg_.w_post_terminal_track * post_track_term
                + cfg_.w_post_terminal_energy * post_energy_term
                + cfg_.w_via_regularization * via_reg_term;
                // + cfg_.w_terminal * terminal_error.squaredNorm()
                // + cfg_.w_passive_track * passive_track_int
                // + cfg_.w_passive_damping * passive_vel_int
    result.smooth_int = smooth_int;
    result.passive_vel_int = passive_vel_int;
    result.post_terminal_track_int = post_terminal_track_int;
    result.post_terminal_energy_int = post_terminal_energy_int;
    result.via_reg = via_reg;
    result.feasible = true;
    result.trajectory = std::move(traj);
    return result;
  }

private:
  static bool allFinite(const Eigen::VectorXd& x)
  {
    for (int i = 0; i < x.size(); ++i) {
      if (!std::isfinite(x[i])) return false;
    }
    return true;
  }

  PlannerConfig cfg_;
  Eigen::VectorXd q_goal_A_;
  Eigen::VectorXd q_goal_P_eq_;
  const CraneDynamicsModel* dynamics_{nullptr};
  CostNormalization normalization_;
};

class CmaEsVpStoOptimizer
{
public:
  using EvaluationFunction = std::function<CandidateEvaluation(int, const Eigen::VectorXd&)>;

  explicit CmaEsVpStoOptimizer(PlannerConfig cfg)
  : cfg_(std::move(cfg)),
    seed_(makeSeed(cfg_.random_seed)),
    rng_(seed_),
    normal_(0.0, 1.0) {}

  std::mt19937::result_type seed() const { return seed_; }

  CandidateEvaluation optimize(const Eigen::VectorXd& mean_init,
                               const EvaluationFunction& eval_fn,
                               rclcpp::Logger logger)
  {
    struct Sample {
      Eigen::VectorXd z, y, x;
      CandidateEvaluation eval;
    };

    const int dim = static_cast<int>(mean_init.size());
    const int lambda = cfg_.population;
    const int mu = std::min(cfg_.elite_count, lambda / 2);
    if (mu <= 0) {
      throw std::runtime_error("elite_count must be >= 1");
    }

    Eigen::VectorXd weights(mu);
    for (int i = 0; i < mu; ++i) {
      weights[i] = std::log(static_cast<double>(mu) + 0.5) - std::log(static_cast<double>(i + 1));
    }
    weights /= weights.sum();

    const double mueff = 1.0 / weights.array().square().sum();
    const double cs = (mueff + 2.0) / (dim + mueff + 5.0);
    const double ds = 1.0 + cs + 2.0 * std::max(0.0, std::sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0);
    const double cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim);
    const double c1 = 2.0 / ((dim + 1.3) * (dim + 1.3) + mueff);
    const double cmu = std::min(
      1.0 - c1,
      2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) * (dim + 2.0) + mueff));
    const double chiN = std::sqrt(static_cast<double>(dim)) *
      (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim));

    Eigen::VectorXd mean = mean_init;
    Eigen::MatrixXd C = Eigen::MatrixXd::Identity(dim, dim);
    Eigen::MatrixXd B = Eigen::MatrixXd::Identity(dim, dim);
    Eigen::VectorXd D = Eigen::VectorXd::Ones(dim);
    Eigen::MatrixXd invsqrtC = Eigen::MatrixXd::Identity(dim, dim);
    Eigen::VectorXd ps = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd pc = Eigen::VectorXd::Zero(dim);
    double sigma = cfg_.sigma0;

    CandidateEvaluation best_global;
    best_global.cost = std::numeric_limits<double>::infinity();

    const int worker_count = std::max(1, std::min(cfg_.parallel_workers, lambda));
    std::vector<Sample>* active_samples = nullptr;
    std::atomic<int> next_index{0};
    std::mutex pool_mutex;
    std::condition_variable pool_cv;
    std::condition_variable done_cv;
    std::exception_ptr first_exception = nullptr;
    int generation = 0;
    int completed_workers = 0;
    bool stop_workers = false;

    std::vector<std::thread> workers;
    if (worker_count > 1) {
      workers.reserve(static_cast<size_t>(worker_count));
      for (int worker_id = 0; worker_id < worker_count; ++worker_id) {
        workers.emplace_back([&, worker_id]() {
          int seen_generation = 0;
          while (true) {
            std::vector<Sample>* samples = nullptr;
            {
              std::unique_lock<std::mutex> lock(pool_mutex);
              pool_cv.wait(lock, [&]() {
                return stop_workers || generation != seen_generation;
              });
              if (stop_workers) {
                return;
              }
              seen_generation = generation;
              samples = active_samples;
            }

            try {
              while (true) {
                const int k = next_index.fetch_add(1);
                if (k >= lambda) {
                  break;
                }
                (*samples)[static_cast<size_t>(k)].eval =
                  eval_fn(worker_id, (*samples)[static_cast<size_t>(k)].x);
              }
            } catch (...) {
              std::lock_guard<std::mutex> lock(pool_mutex);
              if (!first_exception) {
                first_exception = std::current_exception();
              }
            }

            {
              std::lock_guard<std::mutex> lock(pool_mutex);
              ++completed_workers;
              if (completed_workers == worker_count) {
                done_cv.notify_one();
              }
            }
          }
        });
      }
    }

    auto stop_pool = [&]() {
      if (worker_count <= 1) {
        return;
      }
      {
        std::lock_guard<std::mutex> lock(pool_mutex);
        stop_workers = true;
      }
      pool_cv.notify_all();
      for (auto& worker : workers) {
        if (worker.joinable()) {
          worker.join();
        }
      }
    };

    auto evaluate_samples = [&](std::vector<Sample>& samples) {
      if (worker_count == 1) {
        for (int k = 0; k < lambda; ++k) {
          samples[static_cast<size_t>(k)].eval =
            eval_fn(0, samples[static_cast<size_t>(k)].x);
        }
        return;
      }

      {
        std::lock_guard<std::mutex> lock(pool_mutex);
        active_samples = &samples;
        next_index.store(0);
        completed_workers = 0;
        first_exception = nullptr;
        ++generation;
      }
      pool_cv.notify_all();

      {
        std::unique_lock<std::mutex> lock(pool_mutex);
        done_cv.wait(lock, [&]() {
          return completed_workers == worker_count;
        });
      }
      if (first_exception) {
        std::rethrow_exception(first_exception);
      }
    };

    for (int iter = 0; iter < cfg_.max_iterations; ++iter) {
      if (iter % cfg_.cma_update_eig_every == 0) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(C);
        Eigen::VectorXd evals = es.eigenvalues().cwiseMax(1e-12);
        B = es.eigenvectors();
        D = evals.cwiseSqrt();
        invsqrtC = B * D.cwiseInverse().asDiagonal() * B.transpose();
      }

      std::vector<Sample> samples;
      samples.resize(lambda);

      for (int k = 0; k < lambda; ++k) {
        Eigen::VectorXd z(dim);
        for (int i = 0; i < dim; ++i) {
          z[i] = normal_(rng_);
        }
        Eigen::VectorXd y = B * D.asDiagonal() * z;
        Eigen::VectorXd x = mean + sigma * y;
        samples[static_cast<size_t>(k)].z = std::move(z);
        samples[static_cast<size_t>(k)].y = std::move(y);
        samples[static_cast<size_t>(k)].x = std::move(x);
      }

      try {
        evaluate_samples(samples);
      } catch (...) {
        stop_pool();
        throw;
      }
      for (const auto& sample : samples) {
        if (sample.eval.cost < best_global.cost) {
          best_global = sample.eval;
        }
      }

      std::sort(samples.begin(), samples.end(), [](const Sample& a, const Sample& b) {
        return a.eval.cost < b.eval.cost;
      });

      Eigen::VectorXd mean_old = mean;
      mean.setZero();
      for (int i = 0; i < mu; ++i) {
        mean += weights[i] * samples[i].x;
      }

      const Eigen::VectorXd y_w = (mean - mean_old) / sigma;
      const Eigen::VectorXd z_w = invsqrtC * y_w;

      ps = (1.0 - cs) * ps + std::sqrt(cs * (2.0 - cs) * mueff) * z_w;
      const double norm_ps = ps.norm();
      const bool hsig =
        norm_ps / std::sqrt(1.0 - std::pow(1.0 - cs, 2.0 * (iter + 1))) <
        (1.4 + 2.0 / (dim + 1.0)) * chiN;

      pc = (1.0 - cc) * pc + (hsig ? 1.0 : 0.0) * std::sqrt(cc * (2.0 - cc) * mueff) * y_w;

      Eigen::MatrixXd rank_mu = Eigen::MatrixXd::Zero(dim, dim);
      for (int i = 0; i < mu; ++i) {
        rank_mu += weights[i] * (samples[i].y * samples[i].y.transpose());
      }

      const double delta_hsig = hsig ? 0.0 : cc * (2.0 - cc);
      C = (1.0 - c1 - cmu) * C
        + c1 * (pc * pc.transpose() + delta_hsig * C)
        + cmu * rank_mu;

      sigma *= std::exp((cs / ds) * (norm_ps / chiN - 1.0));

      if (cfg_.log_iteration) {
        std::ostringstream oss;
	        oss << std::fixed << std::setprecision(4)
	            << "[VP-STO] iter=" << iter
	            << " best=" << samples.front().eval.cost
	            << " T=" << samples.front().eval.trajectory.T
	            << " smooth=" << samples.front().eval.smooth_int
	            << " passive_vel=" << samples.front().eval.passive_vel_int
	            << " post_track=" << samples.front().eval.post_terminal_track_int
	            << " post_energy=" << samples.front().eval.post_terminal_energy_int
	            << " via_reg=" << samples.front().eval.via_reg
	            << " sigma=" << sigma;
        RCLCPP_INFO(logger, "%s", oss.str().c_str());
      }

      if (sigma < cfg_.stop_sigma) {
        break;
      }
    }

    stop_pool();
    return best_global;
  }

private:
  static std::mt19937::result_type makeSeed(int64_t configured_seed)
  {
    if (configured_seed >= 0) {
      return static_cast<std::mt19937::result_type>(configured_seed);
    }
    return std::random_device{}();
  }

  PlannerConfig cfg_;
  std::mt19937::result_type seed_{0};
  std::mt19937 rng_;
  std::normal_distribution<double> normal_;
};

class WholeBodyViaVariables : public ifopt::VariableSet
{
public:
  WholeBodyViaVariables(const Eigen::VectorXd& initial,
                        const Eigen::VectorXd& lower,
                        const Eigen::VectorXd& upper)
  : ifopt::VariableSet(static_cast<int>(initial.size()), "whole_body_via"),
    values_(initial),
    lower_(lower),
    upper_(upper)
  {
    if (lower_.size() != values_.size() || upper_.size() != values_.size()) {
      throw std::runtime_error("IFOPT via-point bounds size mismatch");
    }
  }

  Eigen::VectorXd GetValues() const override
  {
    return values_;
  }

  VecBound GetBounds() const override
  {
    VecBound bounds;
    bounds.reserve(static_cast<size_t>(values_.size()));
    for (int i = 0; i < values_.size(); ++i) {
      bounds.emplace_back(lower_[i], upper_[i]);
    }
    return bounds;
  }

  void SetVariables(const Eigen::VectorXd& x) override
  {
    values_ = x;
  }

private:
  Eigen::VectorXd values_;
  Eigen::VectorXd lower_;
  Eigen::VectorXd upper_;
};

class CasadiActuatedSurrogateGradient
{
public:
  CasadiActuatedSurrogateGradient(const PlannerConfig& cfg,
                                  const Eigen::VectorXd& q_start,
                                  const Eigen::VectorXd& q_goal,
                                  const Eigen::VectorXd& velocity_limits,
                                  const Eigen::VectorXd& acceleration_limits,
                                  const TrajectoryRolloutEvaluator::CostNormalization& normalization)
  : n_via_(cfg.n_via),
    n_actuated_(static_cast<int>(q_start.size()))
  {
    if (q_goal.size() != q_start.size() ||
        velocity_limits.size() != q_start.size() ||
        acceleration_limits.size() != q_start.size()) {
      throw std::runtime_error("CasADi surrogate gradient input size mismatch");
    }

    const int dim = n_via_ * n_actuated_;
    const casadi::SX x = casadi::SX::sym("actuated_via", dim);

    const auto basis = buildSplineBasis(cfg, q_start, q_goal);
    const double dt_weight = 1.0 / static_cast<double>(std::max(1, cfg.n_eval - 1));
    const double beta = std::max(1.0, cfg.time_smooth_beta);
    const double eps = std::max(1e-12, cfg.time_smooth_eps);

    std::vector<casadi::SX> time_candidates;
    time_candidates.reserve(
      1 + static_cast<size_t>(cfg.n_eval) * 2 * static_cast<size_t>(n_actuated_));
    time_candidates.push_back(casadi::SX(cfg.t_min));

    casadi::SX smooth_s_numer = casadi::SX(0.0);
    for (int sample = 0; sample < cfg.n_eval; ++sample) {
      const double trap = (sample == 0 || sample == cfg.n_eval - 1) ? 0.5 : 1.0;
      for (int j = 0; j < n_actuated_; ++j) {
        const int row = sample * n_actuated_ + j;
        casadi::SX dqds = basis.dqds_const[row];
        casadi::SX d2qds2 = basis.d2qds2_const[row];
        for (int i = 0; i < dim; ++i) {
          const double b1 = basis.dqds_basis(row, i);
          const double b2 = basis.d2qds2_basis(row, i);
          if (std::abs(b1) > 1e-14) {
            dqds += b1 * x(i);
          }
          if (std::abs(b2) > 1e-14) {
            d2qds2 += b2 * x(i);
          }
        }

        const casadi::SX abs_dqds = casadi::SX::sqrt(dqds * dqds + eps * eps);
        const casadi::SX abs_d2qds2 = casadi::SX::sqrt(d2qds2 * d2qds2 + eps * eps);
        time_candidates.push_back(abs_dqds / std::max(1e-9, velocity_limits[j]));
        time_candidates.push_back(
          casadi::SX::sqrt(abs_d2qds2 / std::max(1e-9, acceleration_limits[j])));
        smooth_s_numer += trap * d2qds2 * d2qds2;
      }
    }

    casadi::SX max_candidate = time_candidates.front();
    for (size_t i = 1; i < time_candidates.size(); ++i) {
      max_candidate = casadi::SX::fmax(max_candidate, time_candidates[i]);
    }
    casadi::SX exp_sum = casadi::SX(0.0);
    for (const auto& candidate : time_candidates) {
      exp_sum += casadi::SX::exp(beta * (candidate - max_candidate));
    }
    const casadi::SX T = max_candidate + casadi::SX::log(exp_sum) / beta;
    const casadi::SX smooth_int = smooth_s_numer * dt_weight / (T * T * T);
    const casadi::SX via_reg = casadi::SX::dot(x, x);

    casadi::SX time_term = T;
    casadi::SX smooth_term = smooth_int;
    casadi::SX via_reg_term = via_reg;
    if (normalization.enabled) {
      time_term /= std::max(1e-12, normalization.time);
      smooth_term /= std::max(1e-12, normalization.smooth);
      via_reg_term /= std::max(1e-12, normalization.via_reg);
    }
    if (cfg.solver_friendly_cost) {
      time_term = casadi::SX::log(1.0 + time_term);
    }

    const casadi::SX surrogate_cost =
      cfg.w_time * time_term +
      cfg.w_smooth * smooth_term +
      cfg.w_via_regularization * via_reg_term;
    const casadi::SX grad = casadi::SX::gradient(surrogate_cost, x);
    grad_fn_ = casadi::Function("vp_sto_actuated_surrogate_grad", {x}, {grad});
    valid_ = true;
  }

  bool valid() const
  {
    return valid_;
  }

  Eigen::VectorXd evaluate(const Eigen::VectorXd& actuated_via) const
  {
    if (!valid_ || actuated_via.size() != n_via_ * n_actuated_) {
      return Eigen::VectorXd::Zero(actuated_via.size());
    }

    std::vector<double> values(static_cast<size_t>(actuated_via.size()));
    for (int i = 0; i < actuated_via.size(); ++i) {
      values[static_cast<size_t>(i)] = actuated_via[i];
    }
    const auto out = grad_fn_(std::vector<casadi::DM>{casadi::DM(values)});
    const std::vector<double> grad_values = out.at(0).nonzeros();
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(actuated_via.size());
    for (int i = 0; i < grad.size() && i < static_cast<int>(grad_values.size()); ++i) {
      grad[i] = std::isfinite(grad_values[static_cast<size_t>(i)])
        ? grad_values[static_cast<size_t>(i)]
        : 0.0;
    }
    return grad;
  }

private:
  struct SplineBasis
  {
    Eigen::VectorXd dqds_const;
    Eigen::VectorXd d2qds2_const;
    Eigen::MatrixXd dqds_basis;
    Eigen::MatrixXd d2qds2_basis;
  };

  static SplineBasis buildSplineBasis(const PlannerConfig& cfg,
                                      const Eigen::VectorXd& q_start,
                                      const Eigen::VectorXd& q_goal)
  {
    const int nA = static_cast<int>(q_start.size());
    const int dim = cfg.n_via * nA;
    const int rows = cfg.n_eval * nA;

    SplineBasis basis;
    basis.dqds_const = Eigen::VectorXd::Zero(rows);
    basis.d2qds2_const = Eigen::VectorXd::Zero(rows);
    basis.dqds_basis = Eigen::MatrixXd::Zero(rows, dim);
    basis.d2qds2_basis = Eigen::MatrixXd::Zero(rows, dim);

    MultiJointActuatedSpline spline;
    const Eigen::VectorXd zero_via = Eigen::VectorXd::Zero(dim);
    spline.build(q_start, q_goal, zero_via, cfg.n_via, cfg.use_zero_boundary_slopes);
    for (int sample = 0; sample < cfg.n_eval; ++sample) {
      const double s = static_cast<double>(sample) / static_cast<double>(cfg.n_eval - 1);
      const auto kin = spline.eval(s);
      for (int j = 0; j < nA; ++j) {
        const int row = sample * nA + j;
        basis.dqds_const[row] = kin.dqdsA[j];
        basis.d2qds2_const[row] = kin.d2qds2A[j];
      }
    }

    const Eigen::VectorXd zero_start = Eigen::VectorXd::Zero(nA);
    const Eigen::VectorXd zero_goal = Eigen::VectorXd::Zero(nA);
    for (int i = 0; i < dim; ++i) {
      Eigen::VectorXd via = Eigen::VectorXd::Zero(dim);
      via[i] = 1.0;
      spline.build(zero_start, zero_goal, via, cfg.n_via, cfg.use_zero_boundary_slopes);
      for (int sample = 0; sample < cfg.n_eval; ++sample) {
        const double s = static_cast<double>(sample) / static_cast<double>(cfg.n_eval - 1);
        const auto kin = spline.eval(s);
        for (int j = 0; j < nA; ++j) {
          const int row = sample * nA + j;
          basis.dqds_basis(row, i) = kin.dqdsA[j];
          basis.d2qds2_basis(row, i) = kin.d2qds2A[j];
        }
      }
    }

    return basis;
  }

  int n_via_{0};
  int n_actuated_{0};
  bool valid_{false};
  casadi::Function grad_fn_;
};

class IfoptVpStoCost : public ifopt::CostTerm
{
public:
  using EvaluationFunction = std::function<CandidateEvaluation(const Eigen::VectorXd&)>;

  IfoptVpStoCost(EvaluationFunction eval_fn,
                 Eigen::VectorXd lower,
                 Eigen::VectorXd upper,
                 Eigen::VectorXd base_ref,
                 int n_via,
                 int n_actuated,
                 double w_base_regularization,
                 int parallel_workers,
                 double finite_difference_step,
                 double infeasible_cost,
                 std::shared_ptr<CasadiActuatedSurrogateGradient> casadi_gradient,
                 rclcpp::Logger logger)
  : ifopt::CostTerm("vp_sto_rollout_cost"),
    eval_fn_(std::move(eval_fn)),
    lower_(std::move(lower)),
    upper_(std::move(upper)),
    base_ref_(std::move(base_ref)),
    n_via_(n_via),
    n_actuated_(n_actuated),
    w_base_regularization_(w_base_regularization),
    parallel_workers_(std::max(1, parallel_workers)),
    finite_difference_step_(finite_difference_step),
    infeasible_cost_(infeasible_cost),
    casadi_gradient_(std::move(casadi_gradient)),
    logger_(logger)
  {
    if (lower_.size() != upper_.size()) {
      throw std::runtime_error("IFOPT cost bounds size mismatch");
    }
    if (base_ref_.size() != 6) {
      throw std::runtime_error("Base reference must be size 6");
    }
  }

private:
  double GetCost() const override
  {
    const Eigen::VectorXd x = currentViaPoints();
    Eigen::VectorXd base_via;
    Eigen::VectorXd actuated_via;
    splitVariables(x, base_via, actuated_via);

    const CandidateEvaluation eval = eval_fn_(actuated_via);
    double cost = (eval.feasible && std::isfinite(eval.cost)) ? eval.cost : infeasible_cost_;
    cost += baseRegularization(base_via);
    return cost;
  }

  void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override
  {
    if (var_set != "whole_body_via") {
      return;
    }

    const Eigen::VectorXd x = currentViaPoints();
    const int n = static_cast<int>(x.size());
    const int base_dim = 6;
    const int block = base_dim + n_actuated_;
    jac_block.resize(1, n);
    jac_block.reserve(Eigen::VectorXi::Constant(1, n));

    Eigen::VectorXd base_via;
    Eigen::VectorXd actuated_via;
    splitVariables(x, base_via, actuated_via);

    std::vector<double> grad(static_cast<size_t>(n), 0.0);
    if (w_base_regularization_ > 0.0) {
      for (int k = 0; k < n_via_; ++k) {
        for (int j = 0; j < base_dim; ++j) {
          const int col = k * block + j;
          grad[static_cast<size_t>(col)] =
            2.0 * w_base_regularization_ * (base_via[k * base_dim + j] - base_ref_[j]);
        }
      }
      for (int j = 0; j < base_dim; ++j) {
        const int col = n_via_ * block + j;
        grad[static_cast<size_t>(col)] =
          2.0 * w_base_regularization_ * (base_via[n_via_ * base_dim + j] - base_ref_[j]);
      }
    }

    bool used_casadi_gradient = false;
    if (casadi_gradient_ && casadi_gradient_->valid()) {
      const Eigen::VectorXd actuated_grad = casadi_gradient_->evaluate(actuated_via);
      if (actuated_grad.size() == actuated_via.size()) {
        for (int k = 0; k < n_via_; ++k) {
          for (int j = 0; j < n_actuated_; ++j) {
            const int full_col = k * block + base_dim + j;
            const int actuated_col = k * n_actuated_ + j;
            grad[static_cast<size_t>(full_col)] = actuated_grad[actuated_col];
          }
        }
        used_casadi_gradient = true;
      }
      }

    if (!used_casadi_gradient) {
      std::vector<int> active_actuated_indices;
      active_actuated_indices.reserve(static_cast<size_t>(n_via_ * n_actuated_));
      for (int k = 0; k < n_via_; ++k) {
        for (int j = 0; j < n_actuated_; ++j) {
          const int full_col = k * block + base_dim + j;
          if (full_col < lower_.size() && std::abs(upper_[full_col] - lower_[full_col]) <= 1e-12) {
            continue;
          }
          active_actuated_indices.push_back(full_col);
        }
      }

      const double f0 = GetCost();
      const int worker_count = std::min<int>(
        std::max(1, parallel_workers_),
        std::max<int>(1, static_cast<int>(active_actuated_indices.size())));

      auto eval_gradient_component = [&](int i) {
        Eigen::VectorXd xp = x;
        const double h = finite_difference_step_ * (1.0 + std::abs(x[i]));
        xp[i] += h;
        Eigen::VectorXd base_via_p;
        Eigen::VectorXd actuated_via_p;
        splitVariables(xp, base_via_p, actuated_via_p);
        const CandidateEvaluation eval_p = eval_fn_(actuated_via_p);
        double fp = (eval_p.feasible && std::isfinite(eval_p.cost)) ? eval_p.cost : infeasible_cost_;
        fp += baseRegularization(base_via_p);
        double gi = (fp - f0) / h;
        if (!std::isfinite(gi)) {
          gi = 0.0;
        }
        grad[static_cast<size_t>(i)] = gi;
      };

      if (worker_count <= 1 || active_actuated_indices.size() <= 1) {
        for (const int i : active_actuated_indices) {
          eval_gradient_component(i);
        }
      } else {
        std::atomic<int> next{0};
        std::exception_ptr first_exception = nullptr;
        std::mutex exception_mutex;

        auto worker = [&]() {
          while (true) {
            const int cursor = next.fetch_add(1);
            if (cursor >= static_cast<int>(active_actuated_indices.size())) {
              break;
            }

            try {
              eval_gradient_component(active_actuated_indices[static_cast<size_t>(cursor)]);
            } catch (...) {
              std::lock_guard<std::mutex> lock(exception_mutex);
              if (!first_exception) {
                first_exception = std::current_exception();
              }
            }
          }
        };

        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(worker_count));
        for (int w = 0; w < worker_count; ++w) {
          threads.emplace_back(worker);
        }
        for (auto& thread : threads) {
          thread.join();
        }

        if (first_exception) {
          std::rethrow_exception(first_exception);
        }
      }
    }

    for (int i = 0; i < n; ++i) {
      if (i < lower_.size() && std::abs(upper_[i] - lower_[i]) <= 1e-12) {
        jac_block.coeffRef(0, i) = 0.0;
        continue;
      }
      jac_block.coeffRef(0, i) = grad[static_cast<size_t>(i)];
    }
  }

  Eigen::VectorXd currentViaPoints() const
  {
    const auto variables = GetVariables()->GetComponent<WholeBodyViaVariables>("whole_body_via");
    if (!variables) {
      throw std::runtime_error("IFOPT variable set 'whole_body_via' not found");
    }
    return variables->GetValues();
  }

  void splitVariables(const Eigen::VectorXd& x,
                      Eigen::VectorXd& base_via,
                      Eigen::VectorXd& actuated_via) const
  {
    const int base_dim = 6;
    const int block = base_dim + n_actuated_;
    if (x.size() != n_via_ * block + base_dim) {
      throw std::runtime_error("Whole-body decision vector has unexpected size");
    }

    base_via.resize((n_via_ + 1) * base_dim);
    actuated_via.resize(n_via_ * n_actuated_);
    for (int k = 0; k < n_via_; ++k) {
      base_via.segment(k * base_dim, base_dim) = x.segment(k * block, base_dim);
      actuated_via.segment(k * n_actuated_, n_actuated_) = x.segment(k * block + base_dim, n_actuated_);
    }
    base_via.segment(n_via_ * base_dim, base_dim) = x.segment(n_via_ * block, base_dim);
  }

  double baseRegularization(const Eigen::VectorXd& base_via) const
  {
    if (w_base_regularization_ <= 0.0) {
      return 0.0;
    }
    double cost = 0.0;
    for (int k = 0; k <= n_via_; ++k) {
      const Eigen::VectorXd base_k = base_via.segment(k * 6, 6);
      cost += (base_k - base_ref_).squaredNorm();
    }
    return w_base_regularization_ * cost;
  }

  EvaluationFunction eval_fn_;
  Eigen::VectorXd lower_;
  Eigen::VectorXd upper_;
  Eigen::VectorXd base_ref_;
  int n_via_{0};
  int n_actuated_{0};
  double w_base_regularization_{0.0};
  int parallel_workers_{1};
  double finite_difference_step_{1e-4};
  double infeasible_cost_{1e12};
  std::shared_ptr<CasadiActuatedSurrogateGradient> casadi_gradient_;
  rclcpp::Logger logger_;
};

class FootContactConstraint : public ifopt::ConstraintSet
{
public:
  FootContactConstraint(const CraneDynamicsModel* dynamics,
                        std::vector<pinocchio::FrameIndex> foot_frames,
                        Eigen::VectorXd qP_ref,
                        Eigen::VectorXd qA_goal,
                        Eigen::VectorXd foot_pos_ref,
                        int n_via,
                        int n_actuated,
                        int parallel_workers,
                        double finite_difference_step,
                        double tolerance)
  : ifopt::ConstraintSet((n_via + 1) * static_cast<int>(foot_frames.size()) * 3, "foot_contact"),
    dynamics_(dynamics),
    foot_frames_(std::move(foot_frames)),
    qP_ref_(std::move(qP_ref)),
    qA_goal_(std::move(qA_goal)),
    foot_pos_ref_(std::move(foot_pos_ref)),
    n_via_(n_via),
    n_actuated_(n_actuated),
    parallel_workers_(std::max(1, parallel_workers)),
    finite_difference_step_(finite_difference_step),
    tolerance_(std::max(0.0, tolerance))
  {
    if (!dynamics_) {
      throw std::runtime_error("FootContactConstraint requires a dynamics model");
    }
    if (foot_pos_ref_.size() != static_cast<int>(foot_frames_.size()) * 3) {
      throw std::runtime_error("Foot reference size mismatch");
    }
    if (qA_goal_.size() != n_actuated_) {
      throw std::runtime_error("FootContactConstraint qA_goal size mismatch");
    }
  }

  VecBound GetBounds() const override
  {
    VecBound bounds(GetRows());
    for (auto& b : bounds) {
      b = ifopt::Bounds(-tolerance_, tolerance_);
    }
    return bounds;
  }

  Eigen::VectorXd GetValues() const override
  {
    const Eigen::VectorXd x = currentViaPoints();
    return evaluateConstraint(x);
  }

  void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override
  {
    if (var_set != "whole_body_via") {
      return;
    }

    const Eigen::VectorXd x = currentViaPoints();
    const int base_dim = 6;
    const int block = base_dim + n_actuated_;
    const int foot_rows = static_cast<int>(foot_frames_.size()) * 3;
    const int rows = GetRows();
    const int cols = static_cast<int>(x.size());
    jac_block.resize(rows, cols);
    jac_block.reserve(Eigen::VectorXi::Constant(rows, block));

    auto fill_dense_block = [&](int row_offset, int col_offset, const Eigen::MatrixXd& Jdense) {
      for (int r = 0; r < Jdense.rows(); ++r) {
        for (int c = 0; c < Jdense.cols(); ++c) {
          const double value = Jdense(r, c);
          jac_block.coeffRef(row_offset + r, col_offset + c) =
            std::isfinite(value) ? value : 0.0;
        }
      }
    };

    for (int k = 0; k < n_via_; ++k) {
      const Eigen::VectorXd base_k = x.segment(k * block, base_dim);
      const Eigen::VectorXd qA_k = x.segment(k * block + base_dim, n_actuated_);
      const Eigen::MatrixXd Jdense =
        dynamics_->footPositionJacobian(base_k, qA_k, qP_ref_, foot_frames_);
      fill_dense_block(k * foot_rows, k * block, Jdense);
    }

    const Eigen::VectorXd base_terminal = x.segment(n_via_ * block, base_dim);
    const Eigen::MatrixXd Jterminal =
      dynamics_->footPositionJacobian(base_terminal, qA_goal_, qP_ref_, foot_frames_);
    fill_dense_block(n_via_ * foot_rows, n_via_ * block, Jterminal.leftCols(base_dim));
  }

private:
  Eigen::VectorXd currentViaPoints() const
  {
    const auto variables = GetVariables()->GetComponent<WholeBodyViaVariables>("whole_body_via");
    if (!variables) {
      throw std::runtime_error("IFOPT variable set 'whole_body_via' not found");
    }
    return variables->GetValues();
  }

  Eigen::VectorXd evaluateConstraint(const Eigen::VectorXd& x) const
  {
    const int base_dim = 6;
    const int block = base_dim + n_actuated_;
    if (x.size() != n_via_ * block + base_dim) {
      throw std::runtime_error("Whole-body decision vector has unexpected size");
    }

    Eigen::VectorXd values((n_via_ + 1) * static_cast<int>(foot_frames_.size()) * 3);
    int cursor = 0;
    for (int k = 0; k < n_via_; ++k) {
      const Eigen::VectorXd base_k = x.segment(k * block, base_dim);
      const Eigen::VectorXd qA_k = x.segment(k * block + base_dim, n_actuated_);
      const auto foot_positions = dynamics_->footPositions(base_k, qA_k, qP_ref_, foot_frames_);
      for (size_t f = 0; f < foot_positions.size(); ++f) {
        const Eigen::Vector3d err = foot_positions[f] - foot_pos_ref_.segment<3>(static_cast<int>(f) * 3);
        values[cursor++] = err[0];
        values[cursor++] = err[1];
        values[cursor++] = err[2];
      }
    }

    const Eigen::VectorXd base_terminal = x.segment(n_via_ * block, base_dim);
    const auto terminal_foot_positions =
      dynamics_->footPositions(base_terminal, qA_goal_, qP_ref_, foot_frames_);
    for (size_t f = 0; f < terminal_foot_positions.size(); ++f) {
      const Eigen::Vector3d err =
        terminal_foot_positions[f] - foot_pos_ref_.segment<3>(static_cast<int>(f) * 3);
      values[cursor++] = err[0];
      values[cursor++] = err[1];
      values[cursor++] = err[2];
    }
    return values;
  }

  const CraneDynamicsModel* dynamics_{nullptr};
  std::vector<pinocchio::FrameIndex> foot_frames_;
  Eigen::VectorXd qP_ref_;
  Eigen::VectorXd qA_goal_;
  Eigen::VectorXd foot_pos_ref_;
  int n_via_{0};
  int n_actuated_{0};
  int parallel_workers_{1};
  double finite_difference_step_{1e-4};
  double tolerance_{1e-3};
};

class IfoptVpStoOptimizer
{
public:
  using EvaluationFunction = std::function<CandidateEvaluation(const Eigen::VectorXd&)>;

  struct OptimizationResult
  {
    CandidateEvaluation eval;
    Eigen::VectorXd solution;
  };

  explicit IfoptVpStoOptimizer(PlannerConfig cfg)
  : cfg_(std::move(cfg)) {}

  OptimizationResult optimize(const Eigen::VectorXd& initial,
                              const Eigen::VectorXd& lower,
                              const Eigen::VectorXd& upper,
                              const EvaluationFunction& eval_fn,
                              const std::shared_ptr<FootContactConstraint>& foot_constraint,
                              const Eigen::VectorXd& base_ref,
                              int n_via,
                              int n_actuated,
                              double w_base_regularization,
                              std::shared_ptr<CasadiActuatedSurrogateGradient> casadi_gradient,
                              rclcpp::Logger logger)
  {
    ifopt::Problem nlp;
    nlp.AddVariableSet(std::make_shared<WholeBodyViaVariables>(initial, lower, upper));
    nlp.AddCostSet(std::make_shared<IfoptVpStoCost>(
      eval_fn,
      lower,
      upper,
      base_ref,
      n_via,
      n_actuated,
      w_base_regularization,
      cfg_.parallel_workers,
      cfg_.ipopt_fd_step,
      cfg_.infeasible_cost,
      casadi_gradient,
      logger));

    if (foot_constraint) {
      nlp.AddConstraintSet(foot_constraint);
    }

    int fixed_variables = 0;
    for (int i = 0; i < lower.size(); ++i) {
      if (std::abs(upper[i] - lower[i]) <= 1e-12) {
        ++fixed_variables;
      }
    }
    RCLCPP_INFO(
      logger,
      "IFOPT variables: total=%d, fixed=%d, active=%d",
      static_cast<int>(initial.size()),
      fixed_variables,
      static_cast<int>(initial.size()) - fixed_variables);
    RCLCPP_INFO(
      logger,
      "IFOPT objective gradient: qA=%s, fallback finite-difference workers=%d",
      (casadi_gradient && casadi_gradient->valid()) ? "CasADi surrogate" : "finite_difference",
      std::max(1, cfg_.parallel_workers));

    ifopt::IpoptSolver solver;
    solver.SetOption("linear_solver", cfg_.ipopt_linear_solver);
    if (!cfg_.ipopt_hsl_library.empty()) {
      solver.SetOption("hsllib", cfg_.ipopt_hsl_library);
    }
    if (cfg_.ipopt_linear_solver == "ma97" || cfg_.ipopt_linear_solver == "MA97") {
      solver.SetOption("ma97_print_level", cfg_.ipopt_ma97_print_level);
      solver.SetOption("ma97_order", cfg_.ipopt_ma97_order);
      solver.SetOption("ma97_scaling", cfg_.ipopt_ma97_scaling);
    }
    solver.SetOption("hessian_approximation", "limited-memory");
    solver.SetOption("jacobian_approximation", "exact");
    solver.SetOption("max_iter", cfg_.max_iterations);
    solver.SetOption("tol", cfg_.ipopt_tolerance);
    solver.SetOption("print_level", cfg_.ipopt_print_level);
    solver.SetOption("max_cpu_time", cfg_.ipopt_max_cpu_time);
    solver.SetOption("acceptable_tol", 10.0 * cfg_.ipopt_tolerance);
    solver.SetOption("acceptable_iter", 3);
    solver.SetOption("acceptable_obj_change_tol", 1e-3);

    RCLCPP_INFO(
      logger,
      "Starting IFOPT/IPOPT solve: dim=%d, linear_solver=%s, max_iter=%d, tol=%.3e, max_cpu_time=%.1f",
      static_cast<int>(initial.size()),
      cfg_.ipopt_linear_solver.c_str(),
      cfg_.max_iterations,
      cfg_.ipopt_tolerance,
      cfg_.ipopt_max_cpu_time);

    try {
      solver.Solve(nlp);
    } catch (const std::exception& e) {
      RCLCPP_WARN(logger, "IFOPT/IPOPT solve threw: %s", e.what());
    }

    const Eigen::VectorXd solution = nlp.GetOptVariables()->GetValues();
    Eigen::VectorXd best_solution = solution;
    Eigen::VectorXd base_via;
    Eigen::VectorXd actuated_via;
    splitVariables(solution, n_via, n_actuated, base_via, actuated_via);
    CandidateEvaluation best = eval_fn(actuated_via);
    if (!best.feasible || !std::isfinite(best.cost)) {
      RCLCPP_WARN(logger, "IFOPT/IPOPT final solution infeasible. Falling back to initial trajectory.");
      best_solution = initial;
      splitVariables(initial, n_via, n_actuated, base_via, actuated_via);
      best = eval_fn(actuated_via);
    }
    return OptimizationResult{best, best_solution};
  }

private:
  static void splitVariables(const Eigen::VectorXd& x,
                             int n_via,
                             int n_actuated,
                             Eigen::VectorXd& base_via,
                             Eigen::VectorXd& actuated_via)
  {
    const int base_dim = 6;
    const int block = base_dim + n_actuated;
    if (x.size() != n_via * block + base_dim) {
      throw std::runtime_error("Whole-body decision vector has unexpected size");
    }

    base_via.resize((n_via + 1) * base_dim);
    actuated_via.resize(n_via * n_actuated);
    for (int k = 0; k < n_via; ++k) {
      base_via.segment(k * base_dim, base_dim) = x.segment(k * block, base_dim);
      actuated_via.segment(k * n_actuated, n_actuated) = x.segment(k * block + base_dim, n_actuated);
    }
    base_via.segment(n_via * base_dim, base_dim) = x.segment(n_via * block, base_dim);
  }

  PlannerConfig cfg_;
};

class VpStoGlobalPlannerNode : public rclcpp::Node
{
public:
  VpStoGlobalPlannerNode()
  : Node("vp_sto_global_planner_full_node")
  {
    declare_parameter<std::string>("urdf_path", "/home/jin/harco/ecvt2_ws/forestry_robot_mjcf/xml/ecvt_v2.urdf");
    declare_parameter<std::string>("joint_state_topic", "/joint_states");
    declare_parameter<std::vector<std::string>>(
      "actuated_joints",
      std::vector<std::string>{
        "FRJ1", "FRJ2", "FRJ3", "FRJ5", "FRJW",
        "FLJ1", "FLJ2", "FLJ3", "FLJ5", "FLJW",
        "RRJ1", "RRJ2", "RRJ3", "RRJ5", "RRJW",
        "RLJ1", "RLJ2", "RLJ3", "RLJ5", "RLJW",
        "UPJ1", "UPJ2", "UPJ3", "UPJ4", "TOOLJ1"});
    declare_parameter<std::vector<std::string>>(
      "passive_joints",
      std::vector<std::string>{"UPJ5", "UPJ6"});

    declare_parameter<std::vector<double>>(
      "goal_actuated",
      std::vector<double>{
        -0.3490658504, 0.3652101468, -0.3652101468, 0.3490658504, 0.0,
        0.3490658504, 0.8726646260, -0.3620857600, -0.3490658504, 0.0,
        0.3490658504, -0.7956085790, 0.7956085790, -0.3490658504, 0.0,
        -0.3490658504, -0.5235987756, 0.5235987756, 0.3490658504, 0.0,
        0.4, 0.5, -0.5, 0.6, 0.0});
    declare_parameter<std::vector<double>>(
      "velocity_limits",
      std::vector<double>{
        0.35, 0.35, 0.35, 1.0, 2.0,
        0.35, 0.35, 0.35, 1.0, 2.0,
        0.35, 0.35, 0.35, 1.0, 2.0,
        0.35, 0.35, 0.35, 1.0, 2.0,
        1.0, 0.5, 0.5, 0.5, 0.6});
    declare_parameter<std::vector<double>>(
      "acceleration_limits",
      std::vector<double>{
        1.0, 1.0, 1.0, 2.0, 4.0,
        1.0, 1.0, 1.0, 2.0, 4.0,
        1.0, 1.0, 1.0, 2.0, 4.0,
        1.0, 1.0, 1.0, 2.0, 4.0,
        1.5, 1.0, 1.0, 1.0, 1.2});

    declare_parameter<std::vector<double>>(
      "base_initial",
      std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    declare_parameter<std::vector<double>>(
      "base_goal",
      std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    declare_parameter<double>("base_bound_margin", 0.2);
    declare_parameter<std::vector<std::string>>(
      "foot_frames",
      std::vector<std::string>{"FRW", "FLW", "RRW", "RLW"});

    declare_parameter<int>("n_via", 5);
    declare_parameter<int>("n_eval", 81);
    declare_parameter<int>("population", 40);
    declare_parameter<int>("max_iterations", 20);
    declare_parameter<int>("elite_count", 10);
    declare_parameter<int>("parallel_workers", 1);
    declare_parameter<int>("ipopt_print_level", 5);
    declare_parameter<int>("ipopt_ma97_print_level", 0);
    declare_parameter<int64_t>("random_seed", -1);
    declare_parameter<double>("sigma0", 0.20);
    declare_parameter<double>("ipopt_tolerance", 1e-3);
    declare_parameter<double>("ipopt_fd_step", 1e-4);
    declare_parameter<double>("ipopt_via_bound_margin", 1.0);
    declare_parameter<double>("ipopt_max_cpu_time", 300.0);
    declare_parameter<std::string>("ipopt_linear_solver", "mumps");
    declare_parameter<std::string>("ipopt_hsl_library", "");
    declare_parameter<std::string>("ipopt_ma97_order", "auto");
    declare_parameter<std::string>("ipopt_ma97_scaling", "dynamic");
    declare_parameter<double>("freeze_static_joint_tolerance", -1.0);
    declare_parameter<double>("t_min", 1.0);
    declare_parameter<double>("t_max", 40.0);
    declare_parameter<double>("w_time", 1.0);
    declare_parameter<double>("w_smooth", 1e-3);
    declare_parameter<double>("w_terminal", 1e4);
    declare_parameter<double>("w_passive_track", 200.0);
    declare_parameter<double>("w_passive_damping", 10.0);
    declare_parameter<double>("w_post_terminal_track", 10.0);
    declare_parameter<double>("w_post_terminal_energy", 10.0);
    declare_parameter<double>("w_via_regularization", 1e-4);
    declare_parameter<double>("w_base_regularization", 0.01);
    declare_parameter<bool>("normalize_costs", false);
    declare_parameter<double>("cost_norm_eps", 1e-6);
    declare_parameter<bool>("solver_friendly_cost", true);
    declare_parameter<bool>("smooth_time_parameterization", true);
    declare_parameter<double>("time_smooth_beta", 20.0);
    declare_parameter<double>("time_smooth_eps", 1e-6);
    declare_parameter<double>("foot_contact_tolerance", 1e-3);
    declare_parameter<bool>("freeze_contact_legs_on_fixed_base", true);
    declare_parameter<double>("base_fixed_tolerance", 1e-4);
    declare_parameter<double>("post_terminal_duration", 2.0);
    declare_parameter<int>("post_terminal_steps", 40);
    declare_parameter<bool>("auto_plan_on_first_state", true);
    declare_parameter<bool>("replan_periodic", false);
    declare_parameter<double>("replan_period_sec", 5.0);

    const auto urdf_path = get_parameter("urdf_path").as_string();
    const auto joint_state_topic = get_parameter("joint_state_topic").as_string();
    const auto actuated = get_parameter("actuated_joints").as_string_array();
    const auto passive = get_parameter("passive_joints").as_string_array();

    cfg_.n_via = get_parameter("n_via").as_int();
    cfg_.n_eval = get_parameter("n_eval").as_int();
    cfg_.population = get_parameter("population").as_int();
    cfg_.max_iterations = get_parameter("max_iterations").as_int();
    cfg_.elite_count = get_parameter("elite_count").as_int();
    cfg_.parallel_workers = get_parameter("parallel_workers").as_int();
    cfg_.ipopt_print_level = get_parameter("ipopt_print_level").as_int();
    cfg_.ipopt_ma97_print_level = get_parameter("ipopt_ma97_print_level").as_int();
    if (cfg_.parallel_workers <= 0) {
      cfg_.parallel_workers = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    }
    cfg_.random_seed = get_parameter("random_seed").as_int();
    cfg_.sigma0 = get_parameter("sigma0").as_double();
    cfg_.ipopt_tolerance = get_parameter("ipopt_tolerance").as_double();
    cfg_.ipopt_fd_step = get_parameter("ipopt_fd_step").as_double();
    cfg_.ipopt_via_bound_margin = get_parameter("ipopt_via_bound_margin").as_double();
    cfg_.ipopt_max_cpu_time = get_parameter("ipopt_max_cpu_time").as_double();
    cfg_.ipopt_linear_solver = get_parameter("ipopt_linear_solver").as_string();
    cfg_.ipopt_hsl_library = get_parameter("ipopt_hsl_library").as_string();
    cfg_.ipopt_ma97_order = get_parameter("ipopt_ma97_order").as_string();
    cfg_.ipopt_ma97_scaling = get_parameter("ipopt_ma97_scaling").as_string();
    cfg_.freeze_static_joint_tolerance = get_parameter("freeze_static_joint_tolerance").as_double();
    cfg_.t_min = get_parameter("t_min").as_double();
    cfg_.t_max = get_parameter("t_max").as_double();
    cfg_.w_time = get_parameter("w_time").as_double();
    cfg_.w_smooth = get_parameter("w_smooth").as_double();
    cfg_.w_terminal = get_parameter("w_terminal").as_double();
    cfg_.w_passive_track = get_parameter("w_passive_track").as_double();
    cfg_.w_passive_damping = get_parameter("w_passive_damping").as_double();
    cfg_.w_post_terminal_track = get_parameter("w_post_terminal_track").as_double();
    cfg_.w_post_terminal_energy = get_parameter("w_post_terminal_energy").as_double();
    cfg_.w_via_regularization = get_parameter("w_via_regularization").as_double();
    cfg_.w_base_regularization = get_parameter("w_base_regularization").as_double();
    cfg_.normalize_costs = get_parameter("normalize_costs").as_bool();
    cfg_.cost_norm_eps = get_parameter("cost_norm_eps").as_double();
    cfg_.solver_friendly_cost = get_parameter("solver_friendly_cost").as_bool();
    cfg_.smooth_time_parameterization = get_parameter("smooth_time_parameterization").as_bool();
    cfg_.time_smooth_beta = get_parameter("time_smooth_beta").as_double();
    cfg_.time_smooth_eps = get_parameter("time_smooth_eps").as_double();
    cfg_.foot_contact_tolerance = get_parameter("foot_contact_tolerance").as_double();
    cfg_.freeze_contact_legs_on_fixed_base =
      get_parameter("freeze_contact_legs_on_fixed_base").as_bool();
    cfg_.base_fixed_tolerance = get_parameter("base_fixed_tolerance").as_double();
    cfg_.post_terminal_duration = get_parameter("post_terminal_duration").as_double();
    cfg_.post_terminal_steps = get_parameter("post_terminal_steps").as_int();

    q_goal_A_ = toEigen(get_parameter("goal_actuated").as_double_array());
    const Eigen::VectorXd vlim = toEigen(get_parameter("velocity_limits").as_double_array());
    const Eigen::VectorXd alim = toEigen(get_parameter("acceleration_limits").as_double_array());

    base_start_ = toEigen(get_parameter("base_initial").as_double_array());
    base_goal_ = toEigen(get_parameter("base_goal").as_double_array());
    base_bound_margin_ = get_parameter("base_bound_margin").as_double();
    foot_frame_names_ = get_parameter("foot_frames").as_string_array();

    if (q_goal_A_.size() != static_cast<int>(actuated.size())) {
      throw std::runtime_error("goal_actuated size must match actuated_joints size");
    }
    if (vlim.size() != static_cast<int>(actuated.size())) {
      std::ostringstream oss;
      oss << "velocity_limits size (" << vlim.size()
          << ") must match actuated_joints size (" << actuated.size() << ")";
      throw std::runtime_error(oss.str());
    }
    if (alim.size() != static_cast<int>(actuated.size())) {
      std::ostringstream oss;
      oss << "acceleration_limits size (" << alim.size()
          << ") must match actuated_joints size (" << actuated.size() << ")";
      throw std::runtime_error(oss.str());
    }
    if (base_start_.size() != 6 || base_goal_.size() != 6) {
      throw std::runtime_error("base_initial and base_goal must be size 6");
    }

    urdf_path_ = urdf_path;
    actuated_joints_ = actuated;
    passive_joints_ = passive;
    velocity_limits_ = vlim;
    acceleration_limits_ = alim;

    dynamics_ = std::make_unique<CraneDynamicsModel>(urdf_path_, actuated_joints_, passive_joints_);
    time_param_ = std::make_unique<TimeParameterizer>(vlim, alim);

    traj_pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>("~/actuated_reference", 10);
    whole_body_pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>("~/whole_body_reference", 10);
    preview_pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>("~/full_reference_preview", 10);
    cost_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("~/planner_debug", 10);

    auto_plan_on_first_state_ = get_parameter("auto_plan_on_first_state").as_bool();
    replan_periodic_ = get_parameter("replan_periodic").as_bool();
    const double replan_period_sec = get_parameter("replan_period_sec").as_double();

    joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      joint_state_topic,
      rclcpp::SensorDataQoS(),
      std::bind(&VpStoGlobalPlannerNode::jointStateCallback, this, std::placeholders::_1));

    if (replan_periodic_) {
      timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(replan_period_sec)),
        std::bind(&VpStoGlobalPlannerNode::planIfPossible, this));
    }

    RCLCPP_INFO(get_logger(), "VP-STO global planner node ready. Waiting for joint state.");
  }

private:
  static Eigen::VectorXd toEigen(const std::vector<double>& x)
  {
    Eigen::VectorXd out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      out[static_cast<int>(i)] = x[i];
    }
    return out;
  }

  static bool isLegJointName(const std::string& name)
  {
    return name.rfind("FR", 0) == 0 || name.rfind("FL", 0) == 0 ||
           name.rfind("RR", 0) == 0 || name.rfind("RL", 0) == 0;
  }

  static void splitWholeBodyVariables(const Eigen::VectorXd& x,
                                      int n_via,
                                      int n_actuated,
                                      Eigen::VectorXd& base_via,
                                      Eigen::VectorXd& actuated_via)
  {
    const int base_dim = 6;
    const int block = base_dim + n_actuated;
    if (x.size() != n_via * block + base_dim) {
      throw std::runtime_error("Whole-body decision vector has unexpected size");
    }

    base_via.resize((n_via + 1) * base_dim);
    actuated_via.resize(n_via * n_actuated);
    for (int k = 0; k < n_via; ++k) {
      base_via.segment(k * base_dim, base_dim) = x.segment(k * block, base_dim);
      actuated_via.segment(k * n_actuated, n_actuated) = x.segment(k * block + base_dim, n_actuated);
    }
    base_via.segment(n_via * base_dim, base_dim) = x.segment(n_via * block, base_dim);
  }

  void buildBaseTrajectorySamples(const Eigen::VectorXd& base_via,
                                  const std::vector<double>& time_grid,
                                  double total_time,
                                  std::vector<Eigen::VectorXd>& base_traj) const
  {
    base_traj.clear();
    if (time_grid.empty() || !(total_time > 0.0)) {
      return;
    }

    if (base_via.size() != (cfg_.n_via + 1) * 6) {
      throw std::runtime_error("Base trajectory variables must contain via bases plus terminal base");
    }

    const Eigen::VectorXd base_goal_opt = base_via.segment(cfg_.n_via * 6, 6);
    const Eigen::VectorXd base_via_only = base_via.head(cfg_.n_via * 6);

    MultiJointActuatedSpline base_spline;
    base_spline.build(base_start_, base_goal_opt, base_via_only, cfg_.n_via, cfg_.use_zero_boundary_slopes);

    for (const double t : time_grid) {
      const double s = std::min(1.0, std::max(0.0, t / total_time));
      const auto kin = base_spline.eval(s);
      base_traj.push_back(kin.qA);
    }
  }

  TrajectoryRolloutEvaluator::CostNormalization buildCostNormalization(
    TrajectoryRolloutEvaluator& evaluator,
    const Eigen::VectorXd& mean_init,
    const JointStateView& start) const
  {
    TrajectoryRolloutEvaluator::CostNormalization norm;
    norm.enabled = cfg_.normalize_costs;
    if (!norm.enabled) {
      return norm;
    }

    Eigen::VectorXd base_via;
    Eigen::VectorXd actuated_via;
    splitWholeBodyVariables(mean_init, cfg_.n_via, dynamics_->nA(), base_via, actuated_via);
    const CandidateEvaluation baseline = evaluator.evaluate(actuated_via, start, time_param_.get());
    if (!baseline.feasible || !std::isfinite(baseline.cost)) {
      RCLCPP_WARN(get_logger(), "Cost normalization baseline infeasible. Disabling normalization.");
      norm.enabled = false;
      return norm;
    }

    const double eps = std::max(1e-12, cfg_.cost_norm_eps);
    norm.time = std::max(baseline.trajectory.T, eps);
    norm.smooth = std::max(baseline.smooth_int, eps);
    norm.passive_vel = std::max(baseline.passive_vel_int, eps);
    norm.post_track = std::max(baseline.post_terminal_track_int, eps);
    norm.post_energy = std::max(baseline.post_terminal_energy_int, eps);
    norm.via_reg = std::max(baseline.via_reg, eps);

    RCLCPP_INFO(
      get_logger(),
      "Cost normalization enabled: time=%.3e smooth=%.3e passive_vel=%.3e post_track=%.3e post_energy=%.3e via_reg=%.3e",
      norm.time,
      norm.smooth,
      norm.passive_vel,
      norm.post_track,
      norm.post_energy,
      norm.via_reg);

    return norm;
  }

  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    latest_joint_state_ = *msg;
    have_state_ = true;
    if (auto_plan_on_first_state_ && !planned_once_) {
      planIfPossible();
    }
  }

  void planIfPossible()
  {
    if (!have_state_) return;

    JointStateView start;
    try {
      start = dynamics_->fromJointMsg(latest_joint_state_);
    } catch (const std::exception& e) {
      RCLCPP_WARN(get_logger(), "State conversion failed: %s", e.what());
      return;
    }

    Eigen::VectorXd q_goal_A_plan = q_goal_A_;
    const int nA = dynamics_->nA();
    const int base_dim = 6;
    const int block = base_dim + nA;
    const int terminal_base_offset = cfg_.n_via * block;
    Eigen::VectorXd mean_init = Eigen::VectorXd::Zero(cfg_.n_via * block + base_dim);
    for (int k = 0; k < cfg_.n_via; ++k) {
      const double alpha = static_cast<double>(k + 1) / static_cast<double>(cfg_.n_via + 1);
      Eigen::VectorXd base_k = (1.0 - alpha) * base_start_ + alpha * base_goal_;
      Eigen::VectorXd qk = (1.0 - alpha) * start.qA + alpha * q_goal_A_plan;
      mean_init.segment(k * block, base_dim) = base_k;
      mean_init.segment(k * block + base_dim, nA) = qk;
    }
    mean_init.segment(terminal_base_offset, base_dim) = base_goal_;

    Eigen::VectorXd q_goal_P_eq = Eigen::VectorXd::Zero(dynamics_->nP());
    if (start.qP.size() == q_goal_P_eq.size()) {
      q_goal_P_eq = start.qP;
    }
    if (dynamics_->nP() > 0 && !dynamics_->solvePassiveEquilibrium(q_goal_A_plan, q_goal_P_eq, q_goal_P_eq)) {
      RCLCPP_WARN(
        get_logger(),
        "Failed to compute passive equilibrium at q_goal_A. Falling back to current passive state.");
    }

    Eigen::VectorXd lower = Eigen::VectorXd::Zero(mean_init.size());
    Eigen::VectorXd upper = Eigen::VectorXd::Zero(mean_init.size());
    const double margin = std::max(0.0, cfg_.ipopt_via_bound_margin);
    const double base_margin = std::max(0.0, base_bound_margin_);
    int frozen_via_variables = 0;
    for (int k = 0; k < cfg_.n_via; ++k) {
      for (int j = 0; j < base_dim; ++j) {
        const int idx = k * block + j;
        const double lo = std::min(base_start_[j], base_goal_[j]) - base_margin;
        const double hi = std::max(base_start_[j], base_goal_[j]) + base_margin;
        lower[idx] = lo;
        upper[idx] = hi;
      }
      for (int j = 0; j < nA; ++j) {
        const int idx = k * block + base_dim + j;
        if (cfg_.freeze_static_joint_tolerance >= 0.0 &&
            std::abs(start.qA[j] - q_goal_A_plan[j]) <= cfg_.freeze_static_joint_tolerance) {
          lower[idx] = q_goal_A_plan[j];
          upper[idx] = q_goal_A_plan[j];
          mean_init[idx] = q_goal_A_plan[j];
          ++frozen_via_variables;
        } else {
          lower[idx] = std::min(start.qA[j], q_goal_A_plan[j]) - margin;
          upper[idx] = std::max(start.qA[j], q_goal_A_plan[j]) + margin;
        }
      }
    }
    for (int j = 0; j < base_dim; ++j) {
      const int idx = terminal_base_offset + j;
      const double lo = std::min(base_start_[j], base_goal_[j]) - base_margin;
      const double hi = std::max(base_start_[j], base_goal_[j]) + base_margin;
      lower[idx] = lo;
      upper[idx] = hi;
    }
    if (frozen_via_variables > 0) {
      RCLCPP_INFO(
        get_logger(),
        "Frozen %d static via-point variables whose start and goal are already equal.",
        frozen_via_variables);
    }

    std::vector<pinocchio::FrameIndex> foot_frames;
    foot_frames.reserve(foot_frame_names_.size());
    for (const auto& name : foot_frame_names_) {
      foot_frames.push_back(dynamics_->frameId(name));
    }
    const auto foot_positions = dynamics_->footPositions(base_start_, start.qA, start.qP, foot_frames);
    Eigen::VectorXd foot_pos_ref(static_cast<int>(foot_positions.size()) * 3);
    for (size_t i = 0; i < foot_positions.size(); ++i) {
      foot_pos_ref.segment<3>(static_cast<int>(i) * 3) = foot_positions[i];
    }

    double max_initial_foot_error = 0.0;
    int projected_via_points = 0;
    int failed_projection_points = 0;
    for (int k = 0; k < cfg_.n_via; ++k) {
      Eigen::VectorXd base_k = mean_init.segment(k * block, base_dim);
      Eigen::VectorXd qA_k = mean_init.segment(k * block + base_dim, nA);
      const Eigen::VectorXd lower_k = lower.segment(k * block, block);
      const Eigen::VectorXd upper_k = upper.segment(k * block, block);

      const bool projected = dynamics_->projectBaseAndActuatedToFootContact(
        base_k,
        qA_k,
        start.qP,
        foot_frames,
        foot_pos_ref,
        lower_k,
        upper_k,
        15,
        std::max(1e-5, cfg_.foot_contact_tolerance),
        1e-4);

      if (projected) {
        ++projected_via_points;
        mean_init.segment(k * block, base_dim) = base_k;
        mean_init.segment(k * block + base_dim, nA) = qA_k;
      } else {
        ++failed_projection_points;
      }

      const Eigen::VectorXd err =
        dynamics_->footPositionError(base_k, qA_k, start.qP, foot_frames, foot_pos_ref);
      if (err.size() > 0) {
        max_initial_foot_error = std::max(max_initial_foot_error, err.cwiseAbs().maxCoeff());
      }
    }
    {
      Eigen::VectorXd base_terminal = mean_init.segment(terminal_base_offset, base_dim);
      Eigen::VectorXd qA_terminal = q_goal_A_plan;
      Eigen::VectorXd lower_terminal(block);
      Eigen::VectorXd upper_terminal(block);
      lower_terminal.head(base_dim) = lower.segment(terminal_base_offset, base_dim);
      upper_terminal.head(base_dim) = upper.segment(terminal_base_offset, base_dim);
      lower_terminal.tail(nA) = q_goal_A_plan;
      upper_terminal.tail(nA) = q_goal_A_plan;

      const bool projected = dynamics_->projectBaseAndActuatedToFootContact(
        base_terminal,
        qA_terminal,
        start.qP,
        foot_frames,
        foot_pos_ref,
        lower_terminal,
        upper_terminal,
        15,
        std::max(1e-5, cfg_.foot_contact_tolerance),
        1e-4);

      if (projected) {
        ++projected_via_points;
        mean_init.segment(terminal_base_offset, base_dim) = base_terminal;
      } else {
        ++failed_projection_points;
      }

      const Eigen::VectorXd err =
        dynamics_->footPositionError(base_terminal, q_goal_A_plan, start.qP, foot_frames, foot_pos_ref);
      if (err.size() > 0) {
        max_initial_foot_error = std::max(max_initial_foot_error, err.cwiseAbs().maxCoeff());
      }
    }
    RCLCPP_INFO(
      get_logger(),
      "Initial foot-contact projection: projected=%d/%d, failed=%d, max_abs_error=%.3e, tolerance=%.3e",
      projected_via_points,
      cfg_.n_via + 1,
      failed_projection_points,
      max_initial_foot_error,
      cfg_.foot_contact_tolerance);

    auto evaluator = std::make_unique<TrajectoryRolloutEvaluator>(
      cfg_, q_goal_A_plan, q_goal_P_eq, dynamics_.get());

    const auto normalization = buildCostNormalization(*evaluator, mean_init, start);
    evaluator->setNormalization(normalization);

    std::shared_ptr<CasadiActuatedSurrogateGradient> casadi_gradient;
    try {
      casadi_gradient = std::make_shared<CasadiActuatedSurrogateGradient>(
        cfg_,
        start.qA,
        q_goal_A_plan,
        velocity_limits_,
        acceleration_limits_,
        normalization);
      RCLCPP_INFO(
        get_logger(),
        "CasADi surrogate objective gradient enabled for qA via variables.");
    } catch (const std::exception& e) {
      RCLCPP_WARN(
        get_logger(),
        "CasADi surrogate objective gradient unavailable: %s. Falling back to finite difference.",
        e.what());
    }

    auto foot_constraint = std::make_shared<FootContactConstraint>(
      dynamics_.get(),
      foot_frames,
      start.qP,
      q_goal_A_plan,
      foot_pos_ref,
      cfg_.n_via,
      nA,
      cfg_.parallel_workers,
      cfg_.ipopt_fd_step,
      cfg_.foot_contact_tolerance);

    IfoptVpStoOptimizer optimizer(cfg_);
    const auto plan_start_time = std::chrono::steady_clock::now();
    const auto result = optimizer.optimize(
      mean_init,
      lower,
      upper,
      [&](const Eigen::VectorXd& x) {
        return evaluator->evaluate(x, start, time_param_.get());
      },
      foot_constraint,
      base_start_,
      cfg_.n_via,
      nA,
      cfg_.w_base_regularization,
      casadi_gradient,
      get_logger());
    const auto plan_end_time = std::chrono::steady_clock::now();
    const double planning_time_sec =
      std::chrono::duration<double>(plan_end_time - plan_start_time).count();
    RCLCPP_WARN(
      get_logger(),
      "VP-STO trajectory planning time: %.3f s",
      planning_time_sec);
    std::cout << "[vp_sto_global_planner_full_node] VP-STO trajectory planning time="
              << std::fixed << std::setprecision(3) << planning_time_sec
              << " s" << std::endl;

    const auto& best = result.eval;
    if (!best.feasible) {
      RCLCPP_WARN(get_logger(), "Planner did not find a feasible trajectory.");
      return;
    }

    Eigen::VectorXd base_via;
    Eigen::VectorXd actuated_via;
    splitWholeBodyVariables(result.solution, cfg_.n_via, nA, base_via, actuated_via);
    double max_solution_foot_error = 0.0;
    for (int k = 0; k < cfg_.n_via; ++k) {
      const Eigen::VectorXd base_k = base_via.segment(k * base_dim, base_dim);
      const Eigen::VectorXd qA_k = actuated_via.segment(k * nA, nA);
      const Eigen::VectorXd err =
        dynamics_->footPositionError(base_k, qA_k, start.qP, foot_frames, foot_pos_ref);
      if (err.size() > 0) {
        max_solution_foot_error = std::max(max_solution_foot_error, err.cwiseAbs().maxCoeff());
      }
    }
    const Eigen::VectorXd optimized_base_goal = base_via.segment(cfg_.n_via * base_dim, base_dim);
    {
      const Eigen::VectorXd err =
        dynamics_->footPositionError(optimized_base_goal, q_goal_A_plan, start.qP, foot_frames, foot_pos_ref);
      if (err.size() > 0) {
        max_solution_foot_error = std::max(max_solution_foot_error, err.cwiseAbs().maxCoeff());
      }
    }
    RCLCPP_INFO(
      get_logger(),
      "Optimized foot-contact max_abs_error=%.3e, tolerance=%.3e",
      max_solution_foot_error,
      cfg_.foot_contact_tolerance);
    RCLCPP_INFO(
      get_logger(),
      "Optimized base goal: x=%.4f y=%.4f z=%.4f roll=%.4f pitch=%.4f yaw=%.4f",
      optimized_base_goal[0],
      optimized_base_goal[1],
      optimized_base_goal[2],
      optimized_base_goal[3],
      optimized_base_goal[4],
      optimized_base_goal[5]);

    std::vector<Eigen::VectorXd> base_traj;
    buildBaseTrajectorySamples(base_via, best.trajectory.t, best.trajectory.T, base_traj);

    publishActuatedTrajectory(best.trajectory);
    publishWholeBodyTrajectory(base_traj, best.trajectory);
    publishPreviewTrajectory(best.trajectory);
    publishDebug(best);

    planned_once_ = true;

	    std::ostringstream oss;
		    oss << std::fixed << std::setprecision(4)
		        << "VP-STO solved: T=" << best.trajectory.T
		        << ", cost=" << best.cost
		        << ", smooth=" << best.trajectory.smoothness_integral
		        << ", passive_vel=" << best.passive_vel_int
		        << ", post_track=" << best.post_terminal_track_int
		        << ", post_energy=" << best.post_terminal_energy_int
		        << ", via_reg=" << best.via_reg
		        << ", planning_time=" << planning_time_sec << " s";
    RCLCPP_INFO(get_logger(), "%s", oss.str().c_str());
  }

  void publishActuatedTrajectory(const FullTrajectorySample& traj)
  {
    trajectory_msgs::msg::JointTrajectory msg;
    msg.header.stamp = now();
    msg.joint_names = dynamics_->actuatedJointNames();

    for (size_t k = 0; k < traj.t.size(); ++k) {
      trajectory_msgs::msg::JointTrajectoryPoint pt;
      pt.time_from_start = rclcpp::Duration::from_seconds(traj.t[k]);
      pt.positions.resize(traj.qA[k].size());
      pt.velocities.resize(traj.qdotA[k].size());
      pt.accelerations.resize(traj.qddA[k].size());

      for (int i = 0; i < traj.qA[k].size(); ++i) {
        pt.positions[static_cast<size_t>(i)] = traj.qA[k][i];
        pt.velocities[static_cast<size_t>(i)] = traj.qdotA[k][i];
        pt.accelerations[static_cast<size_t>(i)] = traj.qddA[k][i];
      }
      msg.points.push_back(std::move(pt));
    }

    traj_pub_->publish(msg);
  }

  void publishWholeBodyTrajectory(const std::vector<Eigen::VectorXd>& base_traj,
                                  const FullTrajectorySample& traj)
  {
    if (base_traj.size() != traj.t.size()) {
      RCLCPP_WARN(get_logger(), "Base trajectory size mismatch. Skipping whole-body publish.");
      return;
    }

    trajectory_msgs::msg::JointTrajectory msg;
    msg.header.stamp = now();
    const std::vector<std::string> base_joint_names{
      "base_x", "base_y", "base_z", "base_roll", "base_pitch", "base_yaw"};
    msg.joint_names = base_joint_names;
    const auto& anames = dynamics_->actuatedJointNames();
    msg.joint_names.insert(msg.joint_names.end(), anames.begin(), anames.end());

    for (size_t k = 0; k < traj.t.size(); ++k) {
      trajectory_msgs::msg::JointTrajectoryPoint pt;
      pt.time_from_start = rclcpp::Duration::from_seconds(traj.t[k]);

      const int n_all = static_cast<int>(base_joint_names.size()) + traj.qA[k].size();
      pt.positions.resize(n_all);
      pt.velocities.resize(n_all, 0.0);
      pt.accelerations.resize(n_all, 0.0);

      int off = 0;
      for (int i = 0; i < base_traj[k].size(); ++i, ++off) {
        pt.positions[off] = base_traj[k][i];
      }
      for (int i = 0; i < traj.qA[k].size(); ++i, ++off) {
        pt.positions[off] = traj.qA[k][i];
        pt.velocities[off] = traj.qdotA[k][i];
        pt.accelerations[off] = traj.qddA[k][i];
      }

      msg.points.push_back(std::move(pt));
    }

    whole_body_pub_->publish(msg);
  }

  void publishPreviewTrajectory(const FullTrajectorySample& traj)
  {
    trajectory_msgs::msg::JointTrajectory msg;
    msg.header.stamp = now();
    msg.joint_names = dynamics_->actuatedJointNames();
    const auto& pnames = dynamics_->passiveJointNames();
    msg.joint_names.insert(msg.joint_names.end(), pnames.begin(), pnames.end());

    for (size_t k = 0; k < traj.t.size(); ++k) {
      trajectory_msgs::msg::JointTrajectoryPoint pt;
      pt.time_from_start = rclcpp::Duration::from_seconds(traj.t[k]);

      const int n_all = traj.qA[k].size() + traj.qP[k].size();
      pt.positions.resize(n_all);
      pt.velocities.resize(n_all);
      pt.accelerations.resize(n_all);

      int off = 0;
      for (int i = 0; i < traj.qA[k].size(); ++i, ++off) {
        pt.positions[off] = traj.qA[k][i];
        pt.velocities[off] = traj.qdotA[k][i];
        pt.accelerations[off] = traj.qddA[k][i];
      }
      for (int i = 0; i < traj.qP[k].size(); ++i, ++off) {
        pt.positions[off] = traj.qP[k][i];
        pt.velocities[off] = traj.qdotP[k][i];
        pt.accelerations[off] = traj.qddP[k][i];
      }

      msg.points.push_back(std::move(pt));
    }

    preview_pub_->publish(msg);
  }

  void publishDebug(const CandidateEvaluation& best)
  {
    std_msgs::msg::Float64MultiArray msg;
    msg.data = {
	      best.cost,
	      best.trajectory.T,
	      best.smooth_int,
	      best.passive_vel_int,
	      best.post_terminal_track_int,
	      best.post_terminal_energy_int,
	      best.via_reg,
	      static_cast<double>(best.trajectory.feasible ? 1.0 : 0.0)
	    };
    cost_pub_->publish(msg);
  }

  PlannerConfig cfg_;
  bool have_state_{false};
  bool planned_once_{false};
  bool auto_plan_on_first_state_{true};
  bool replan_periodic_{false};

  sensor_msgs::msg::JointState latest_joint_state_;
  Eigen::VectorXd q_goal_A_;
  Eigen::VectorXd velocity_limits_;
  Eigen::VectorXd acceleration_limits_;
  Eigen::VectorXd base_start_;
  Eigen::VectorXd base_goal_;
  double base_bound_margin_{0.0};
  std::string urdf_path_;
  std::vector<std::string> actuated_joints_;
  std::vector<std::string> passive_joints_;
  std::vector<std::string> foot_frame_names_;

  std::unique_ptr<CraneDynamicsModel> dynamics_;
  std::unique_ptr<TimeParameterizer> time_param_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr traj_pub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr whole_body_pub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr preview_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cost_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace vpsto

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<vpsto::VpStoGlobalPlannerNode>();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    std::cerr << "[vp_sto_global_planner_full_node] fatal: " << e.what() << std::endl;
  }
  rclcpp::shutdown();
  return 0;
}
