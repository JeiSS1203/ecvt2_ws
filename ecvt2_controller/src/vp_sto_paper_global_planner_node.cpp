#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>

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
  std::string ipopt_linear_solver{"mumps"};

  double t_min{1.0};
  double t_max{40.0};
  double sigma0{0.20};
  double stop_sigma{1e-3};
  double infeasible_cost{1e12};
  double ipopt_tolerance{1e-3};
  double ipopt_fd_step{1e-4};
  double ipopt_via_bound_margin{1.0};
  double ipopt_max_cpu_time{300.0};
  double freeze_static_joint_tolerance{1e-5};

  double w_time{10.0};
  double w_smooth{1.0};
  double w_terminal{1.0};
  double w_passive_track{1.0};
  double w_passive_damping{10.0};
  double w_post_terminal_track{1.0};
  double w_post_terminal_energy{10.0};
  double w_via_regularization{1e-4};

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
    pinocchio::urdf::buildModel(urdf_path, model_);
    data_ = std::make_unique<pinocchio::Data>(model_);
    registerJoints(actuated_joints_, idxA_q_, idxA_v_);
    registerJoints(passive_joints_, idxP_q_, idxP_v_);
  }

  const std::vector<std::string>& actuatedJointNames() const { return actuated_joints_; }
  const std::vector<std::string>& passiveJointNames() const { return passive_joints_; }
  int nA() const { return static_cast<int>(idxA_v_.size()); }
  int nP() const { return static_cast<int>(idxP_v_.size()); }

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
    pinocchio::crba(model_, *data_, q_full);
    Eigen::MatrixXd M = data_->M;
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

    pinocchio::crba(model_, *data_, q_full);
    Eigen::MatrixXd M = data_->M;
    M.triangularView<Eigen::StrictlyLower>() =
      M.transpose().triangularView<Eigen::StrictlyLower>();

    const Eigen::VectorXd nle = pinocchio::nonLinearEffects(model_, *data_, q_full, v_full);

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
    for (int iter = 0; iter < max_iter; ++iter) {
      const Eigen::VectorXd gP = passiveGravity(qA, qP);
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
        const Eigen::VectorXd g_eps = passiveGravity(qA, qP_eps);
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
    const Eigen::VectorXd gP = passiveGravity(qA, qP_eq);
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

private:
  static bool allFinite(const Eigen::VectorXd& x)
  {
    for (int i = 0; i < x.size(); ++i) {
      if (!std::isfinite(x[i])) return false;
    }
    return true;
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
    const Eigen::VectorXd q_full = composeConfiguration(qA, qP);
    const Eigen::VectorXd g = pinocchio::computeGeneralizedGravity(model_, *data_, q_full);
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
};

class TimeParameterizer
{
public:
  TimeParameterizer(Eigen::VectorXd vmax,
                    Eigen::VectorXd amax)
  : vmax_(std::move(vmax)), amax_(std::move(amax)) {}

  double computeMinimumTime(const std::vector<ActuatedKinematics>& kin,
                            double t_min,
                            double t_max) const
  {
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
  TrajectoryRolloutEvaluator(const PlannerConfig& cfg,
                             Eigen::VectorXd q_goal_A,
                             Eigen::VectorXd q_goal_P_eq,
                             const CraneDynamicsModel* dynamics)
  : cfg_(cfg),
    q_goal_A_(std::move(q_goal_A)),
    q_goal_P_eq_(std::move(q_goal_P_eq)),
    dynamics_(dynamics) {}

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

    const double T = time_param->computeMinimumTime(kin, cfg_.t_min, cfg_.t_max);
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

    result.cost = cfg_.w_time * T
                + cfg_.w_smooth * smooth_int
                + cfg_.w_passive_damping * passive_vel_int
                + cfg_.w_post_terminal_track * post_terminal_track_int
                + cfg_.w_post_terminal_energy * post_terminal_energy_int
                + cfg_.w_via_regularization * via_reg;
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

static std::vector<double> eigenToStdVector(const Eigen::VectorXd& value)
{
  std::vector<double> out(static_cast<size_t>(value.size()));
  for (int i = 0; i < value.size(); ++i) {
    out[static_cast<size_t>(i)] = value[i];
  }
  return out;
}

class CasadiVpStoCostCallback : public casadi::Callback
{
public:
  using EvaluationFunction = std::function<CandidateEvaluation(const Eigen::VectorXd&)>;

  CasadiVpStoCostCallback(casadi_int dimension,
                          EvaluationFunction eval_fn,
                          double infeasible_cost)
  : dimension_(dimension),
    eval_fn_(std::move(eval_fn)),
    infeasible_cost_(infeasible_cost)
  {
    casadi::Dict opts;
    opts["enable_fd"] = true;
    construct("vp_sto_rollout_cost", opts);
  }

  casadi_int get_n_in() override
  {
    return 1;
  }

  casadi_int get_n_out() override
  {
    return 1;
  }

  casadi::Sparsity get_sparsity_in(casadi_int) override
  {
    return casadi::Sparsity::dense(dimension_, 1);
  }

  casadi::Sparsity get_sparsity_out(casadi_int) override
  {
    return casadi::Sparsity::scalar();
  }

  std::vector<casadi::DM> eval(const std::vector<casadi::DM>& arg) const override
  {
    const std::vector<double> x_values = arg.at(0).nonzeros();
    Eigen::VectorXd x(static_cast<int>(x_values.size()));
    for (int i = 0; i < x.size(); ++i) {
      x[i] = x_values[static_cast<size_t>(i)];
    }

    const CandidateEvaluation eval = eval_fn_(x);
    const double cost =
      (eval.feasible && std::isfinite(eval.cost)) ? eval.cost : infeasible_cost_;
    return {casadi::DM(cost)};
  }

private:
  casadi_int dimension_{0};
  EvaluationFunction eval_fn_;
  double infeasible_cost_{1e12};
};

class CasadiVpStoOptimizer
{
public:
  using EvaluationFunction = std::function<CandidateEvaluation(const Eigen::VectorXd&)>;

  explicit CasadiVpStoOptimizer(PlannerConfig cfg)
  : cfg_(std::move(cfg)) {}

  CandidateEvaluation optimize(const Eigen::VectorXd& initial,
                               const Eigen::VectorXd& lower,
                               const Eigen::VectorXd& upper,
                               const EvaluationFunction& eval_fn,
                               rclcpp::Logger logger)
  {
    if (initial.size() == 0 || lower.size() != initial.size() || upper.size() != initial.size()) {
      RCLCPP_WARN(logger, "CasADi/IPOPT input size mismatch. Falling back to initial trajectory.");
      return eval_fn(initial);
    }

    CasadiVpStoCostCallback cost_callback(
      static_cast<casadi_int>(initial.size()), eval_fn, cfg_.infeasible_cost);

    const casadi::MX x = casadi::MX::sym("via_points", initial.size());
    casadi::MXDict nlp;
    nlp["x"] = x;
    nlp["f"] = cost_callback(std::vector<casadi::MX>{x}).at(0);

    casadi::Dict opts;
    opts["expand"] = false;
    opts["ipopt.linear_solver"] = cfg_.ipopt_linear_solver;
    opts["ipopt.hessian_approximation"] = "limited-memory";
    opts["ipopt.max_iter"] = cfg_.max_iterations;
    opts["ipopt.tol"] = cfg_.ipopt_tolerance;
    opts["ipopt.print_level"] = cfg_.ipopt_print_level;
    opts["ipopt.max_cpu_time"] = cfg_.ipopt_max_cpu_time;
    opts["ipopt.acceptable_tol"] = 10.0 * cfg_.ipopt_tolerance;
    opts["ipopt.acceptable_iter"] = 3;
    opts["ipopt.findiff_perturbation"] = cfg_.ipopt_fd_step;

    RCLCPP_INFO(
      logger,
      "Starting CasADi/IPOPT solve: dim=%d, linear_solver=%s, max_iter=%d, tol=%.3e, max_cpu_time=%.1f",
      static_cast<int>(initial.size()),
      cfg_.ipopt_linear_solver.c_str(),
      cfg_.max_iterations,
      cfg_.ipopt_tolerance,
      cfg_.ipopt_max_cpu_time);

    Eigen::VectorXd solution = initial;
    try {
      casadi::Function solver = casadi::nlpsol("vp_sto_solver", "ipopt", nlp, opts);
      casadi::DMDict args;
      args["x0"] = eigenToStdVector(initial);
      args["lbx"] = eigenToStdVector(lower);
      args["ubx"] = eigenToStdVector(upper);

      const casadi::DMDict result = solver(args);
      const casadi::Dict stats = solver.stats();
      const auto success_it = stats.find("success");
      if (success_it != stats.end() && !static_cast<bool>(success_it->second)) {
        std::string return_status = "unknown";
        const auto status_it = stats.find("return_status");
        if (status_it != stats.end()) {
          return_status = static_cast<std::string>(status_it->second);
        }
        RCLCPP_WARN(logger, "CasADi/IPOPT reported failure: %s", return_status.c_str());
      }

      const std::vector<double> x_solution = result.at("x").nonzeros();
      if (static_cast<int>(x_solution.size()) == initial.size()) {
        for (int i = 0; i < solution.size(); ++i) {
          solution[i] = x_solution[static_cast<size_t>(i)];
        }
      }
    } catch (const std::exception& e) {
      RCLCPP_WARN(logger, "CasADi/IPOPT solve threw: %s", e.what());
    }

    CandidateEvaluation best = eval_fn(solution);
    if (!best.feasible || !std::isfinite(best.cost)) {
      RCLCPP_WARN(logger, "CasADi/IPOPT final solution infeasible. Falling back to initial trajectory.");
      best = eval_fn(initial);
    }
    return best;
  }

private:
  PlannerConfig cfg_;
};

class VpStoGlobalPlannerNode : public rclcpp::Node
{
public:
  VpStoGlobalPlannerNode()
  : Node("vp_sto_paper_global_planner_node")
  {
    declare_parameter<std::string>(
      "urdf_path",
      "/home/jin/harco/ecvt2_ws/forestry_robot_mjcf/xml/ecvt_v2_upper.urdf");
    declare_parameter<std::string>("joint_state_topic", "/joint_states");
    declare_parameter<std::vector<std::string>>(
      "actuated_joints",
      std::vector<std::string>{"UPJ1", "UPJ2", "UPJ3", "UPJ4", "TOOLJ1"});
    declare_parameter<std::vector<std::string>>(
      "passive_joints",
      std::vector<std::string>{"UPJ5", "UPJ6"});

    declare_parameter<std::vector<double>>(
      "goal_actuated",
      std::vector<double>{0.4, 0.5, -0.5, 0.6, 0.0});
    declare_parameter<std::vector<double>>(
      "velocity_limits",
      std::vector<double>{5.0, 5.0, 5.0, 5.0, 5.0});
    declare_parameter<std::vector<double>>(
      "acceleration_limits",
      std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0});

    declare_parameter<int>("n_via", 5);
    declare_parameter<int>("n_eval", 81);
    declare_parameter<int>("population", 40);
    declare_parameter<int>("max_iterations", 20);
    declare_parameter<int>("elite_count", 10);
    declare_parameter<int>("parallel_workers", 1);
    declare_parameter<int>("ipopt_print_level", 5);
    declare_parameter<std::string>("ipopt_linear_solver", "mumps");
    declare_parameter<int64_t>("random_seed", -1);
    declare_parameter<double>("sigma0", 0.20);
    declare_parameter<double>("ipopt_tolerance", 1e-3);
    declare_parameter<double>("ipopt_fd_step", 1e-4);
    declare_parameter<double>("ipopt_via_bound_margin", 1.0);
    declare_parameter<double>("ipopt_max_cpu_time", 300.0);
    declare_parameter<double>("freeze_static_joint_tolerance", 1e-5);
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
    cfg_.ipopt_linear_solver = get_parameter("ipopt_linear_solver").as_string();
    if (cfg_.parallel_workers <= 0) {
      cfg_.parallel_workers = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    }
    cfg_.random_seed = get_parameter("random_seed").as_int();
    cfg_.sigma0 = get_parameter("sigma0").as_double();
    cfg_.ipopt_tolerance = get_parameter("ipopt_tolerance").as_double();
    cfg_.ipopt_fd_step = get_parameter("ipopt_fd_step").as_double();
    cfg_.ipopt_via_bound_margin = get_parameter("ipopt_via_bound_margin").as_double();
    cfg_.ipopt_max_cpu_time = get_parameter("ipopt_max_cpu_time").as_double();
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
    cfg_.post_terminal_duration = get_parameter("post_terminal_duration").as_double();
    cfg_.post_terminal_steps = get_parameter("post_terminal_steps").as_int();

    q_goal_A_ = toEigen(get_parameter("goal_actuated").as_double_array());
    const Eigen::VectorXd vlim = toEigen(get_parameter("velocity_limits").as_double_array());
    const Eigen::VectorXd alim = toEigen(get_parameter("acceleration_limits").as_double_array());

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

    urdf_path_ = urdf_path;
    actuated_joints_ = actuated;
    passive_joints_ = passive;

    dynamics_ = std::make_unique<CraneDynamicsModel>(urdf_path_, actuated_joints_, passive_joints_);
    time_param_ = std::make_unique<TimeParameterizer>(vlim, alim);

    traj_pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>("~/actuated_reference", 10);
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

    const int nA = dynamics_->nA();
    Eigen::VectorXd mean_init = Eigen::VectorXd::Zero(cfg_.n_via * nA);
    for (int k = 0; k < cfg_.n_via; ++k) {
      const double alpha = static_cast<double>(k + 1) / static_cast<double>(cfg_.n_via + 1);
      Eigen::VectorXd qk = (1.0 - alpha) * start.qA + alpha * q_goal_A_;
      mean_init.segment(k * nA, nA) = qk;
    }

    Eigen::VectorXd q_goal_P_eq = Eigen::VectorXd::Zero(dynamics_->nP());
    if (start.qP.size() == q_goal_P_eq.size()) {
      q_goal_P_eq = start.qP;
    }
    if (dynamics_->nP() > 0 && !dynamics_->solvePassiveEquilibrium(q_goal_A_, q_goal_P_eq, q_goal_P_eq)) {
      RCLCPP_WARN(
        get_logger(),
        "Failed to compute passive equilibrium at q_goal_A. Falling back to current passive state.");
    }

    Eigen::VectorXd lower = Eigen::VectorXd::Zero(mean_init.size());
    Eigen::VectorXd upper = Eigen::VectorXd::Zero(mean_init.size());
    const double margin = std::max(0.0, cfg_.ipopt_via_bound_margin);
    int frozen_via_variables = 0;
    for (int k = 0; k < cfg_.n_via; ++k) {
      for (int j = 0; j < nA; ++j) {
        const int idx = k * nA + j;
        if (std::abs(start.qA[j] - q_goal_A_[j]) <= cfg_.freeze_static_joint_tolerance) {
          lower[idx] = q_goal_A_[j];
          upper[idx] = q_goal_A_[j];
          mean_init[idx] = q_goal_A_[j];
          ++frozen_via_variables;
        } else {
          lower[idx] = std::min(start.qA[j], q_goal_A_[j]) - margin;
          upper[idx] = std::max(start.qA[j], q_goal_A_[j]) + margin;
        }
      }
    }
    if (frozen_via_variables > 0) {
      RCLCPP_INFO(
        get_logger(),
        "Frozen %d static via-point variables whose start and goal are already equal.",
        frozen_via_variables);
    }

    auto evaluator = std::make_unique<TrajectoryRolloutEvaluator>(
      cfg_, q_goal_A_, q_goal_P_eq, dynamics_.get());

    CasadiVpStoOptimizer optimizer(cfg_);
	    const auto plan_start_time = std::chrono::steady_clock::now();
	    const auto best = optimizer.optimize(
	      mean_init,
	      lower,
	      upper,
	      [&](const Eigen::VectorXd& x) {
	        return evaluator->evaluate(x, start, time_param_.get());
	      },
	      get_logger());
	    const auto plan_end_time = std::chrono::steady_clock::now();
	    const double planning_time_sec =
	      std::chrono::duration<double>(plan_end_time - plan_start_time).count();
	    RCLCPP_WARN(
	      get_logger(),
	      "VP-STO trajectory planning time: %.3f s",
	      planning_time_sec);
	    std::cout << "[vp_sto_paper_global_planner_node] VP-STO trajectory planning time="
	              << std::fixed << std::setprecision(3) << planning_time_sec
	              << " s" << std::endl;
	
	    if (!best.feasible) {
	      RCLCPP_WARN(get_logger(), "Planner did not find a feasible trajectory.");
	      return;
    }

    publishActuatedTrajectory(best.trajectory);
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
  std::string urdf_path_;
  std::vector<std::string> actuated_joints_;
  std::vector<std::string> passive_joints_;

  std::unique_ptr<CraneDynamicsModel> dynamics_;
  std::unique_ptr<TimeParameterizer> time_param_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr traj_pub_;
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
    std::cerr << "[vp_sto_paper_global_planner_node] fatal: " << e.what() << std::endl;
  }
  rclcpp::shutdown();
  return 0;
}
