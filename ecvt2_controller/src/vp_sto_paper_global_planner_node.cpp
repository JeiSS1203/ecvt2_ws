#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vpsto_paper
{

struct PlannerConfig
{
  int n_via{5};
  int n_eval{81};
  int population{40};
  int max_iterations{20};
  int elite_count{10};
  int cma_update_eig_every{5};
  int64_t random_seed{-1};

  double t_min{1.0};
  double t_max{40.0};
  double sigma0{0.20};
  double stop_sigma{1e-3};
  double infeasible_cost{1e12};

  double w_time{1.0};
  double w_smooth{1e-3};
  double w_terminal{1e4};
  double w_via_regularization{1e-4};

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
  Eigen::VectorXd qA;
  Eigen::VectorXd vA;
};

struct ActuatedKinematics
{
  Eigen::VectorXd qA;
  Eigen::VectorXd dqdsA;
  Eigen::VectorXd d2qds2A;
};

struct TrajectorySample
{
  std::vector<double> t;
  std::vector<Eigen::VectorXd> qA;
  std::vector<Eigen::VectorXd> qdotA;
  std::vector<Eigen::VectorXd> qddA;
  double T{0.0};
  double smoothness_integral{0.0};
  bool feasible{false};
};

struct CandidateEvaluation
{
  double cost{std::numeric_limits<double>::infinity()};
  bool feasible{false};
  Eigen::VectorXd via_flat;
  TrajectorySample trajectory;
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

    const Eigen::VectorXd cvec = A.fullPivLu().solve(rhs);
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

class ActuatedJointModel
{
public:
  ActuatedJointModel(const std::string& urdf_path,
                     const std::vector<std::string>& actuated_joints)
  : actuated_joints_(actuated_joints)
  {
    pinocchio::urdf::buildModel(urdf_path, model_);
    data_ = std::make_unique<pinocchio::Data>(model_);
    registerJoints(actuated_joints_);
  }

  const std::vector<std::string>& actuatedJointNames() const { return actuated_joints_; }
  int nA() const { return static_cast<int>(actuated_joints_.size()); }

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
    out.qA = Eigen::VectorXd::Zero(static_cast<int>(actuated_joints_.size()));
    out.vA = Eigen::VectorXd::Zero(static_cast<int>(actuated_joints_.size()));

    for (size_t i = 0; i < actuated_joints_.size(); ++i) {
      const auto it_ros = ros_idx.find(actuated_joints_[i]);
      if (it_ros == ros_idx.end()) {
        throw std::runtime_error("Joint '" + actuated_joints_[i] + "' missing in JointState");
      }
      out.qA[static_cast<int>(i)] = msg.position[it_ros->second];
      out.vA[static_cast<int>(i)] = msg.velocity[it_ros->second];
    }

    return out;
  }

private:
  void registerJoints(const std::vector<std::string>& names)
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
    }
  }

  pinocchio::Model model_;
  std::unique_ptr<pinocchio::Data> data_;
  std::vector<std::string> actuated_joints_;
  std::unordered_map<std::string, JointInfo> joint_info_;
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
                             Eigen::VectorXd q_goal_A)
  : cfg_(cfg),
    q_goal_A_(std::move(q_goal_A)) {}

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

    TrajectorySample traj;
    traj.T = T;
    traj.feasible = true;

    const double dt_eval = T / static_cast<double>(std::max(1, cfg_.n_eval - 1));
    double smooth_int = 0.0;

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

      const double weight = (k == 0 || k == cfg_.n_eval - 1) ? 0.5 : 1.0;
      smooth_int += weight * qddA.squaredNorm();
    }

    smooth_int *= dt_eval;
    traj.smoothness_integral = smooth_int;

    const Eigen::VectorXd terminal_error = traj.qA.back() - q_goal_A_;
    const double via_reg = via_flat.squaredNorm();

    result.cost = cfg_.w_time * T
                + cfg_.w_smooth * smooth_int
                + cfg_.w_terminal * terminal_error.squaredNorm()
                + cfg_.w_via_regularization * via_reg;

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
};

class CmaEsVpStoOptimizer
{
public:
  explicit CmaEsVpStoOptimizer(PlannerConfig cfg)
  : cfg_(std::move(cfg)),
    seed_(makeSeed(cfg_.random_seed)),
    rng_(seed_),
    normal_(0.0, 1.0) {}

  std::mt19937::result_type seed() const { return seed_; }

  CandidateEvaluation optimize(const Eigen::VectorXd& mean_init,
                               const std::function<CandidateEvaluation(const Eigen::VectorXd&)>& eval_fn,
                               rclcpp::Logger logger)
  {
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

    for (int iter = 0; iter < cfg_.max_iterations; ++iter) {
      if (iter % cfg_.cma_update_eig_every == 0) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(C);
        Eigen::VectorXd evals = es.eigenvalues().cwiseMax(1e-12);
        B = es.eigenvectors();
        D = evals.cwiseSqrt();
        invsqrtC = B * D.cwiseInverse().asDiagonal() * B.transpose();
      }

      struct Sample {
        Eigen::VectorXd z, y, x;
        CandidateEvaluation eval;
      };

      std::vector<Sample> samples;
      samples.reserve(lambda);

      for (int k = 0; k < lambda; ++k) {
        Eigen::VectorXd z(dim);
        for (int i = 0; i < dim; ++i) {
          z[i] = normal_(rng_);
        }
        Eigen::VectorXd y = B * D.asDiagonal() * z;
        Eigen::VectorXd x = mean + sigma * y;
        CandidateEvaluation e = eval_fn(x);
        samples.push_back({z, y, x, std::move(e)});
        if (samples.back().eval.cost < best_global.cost) {
          best_global = samples.back().eval;
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
            << "[VP-STO paper] iter=" << iter
            << " best=" << samples.front().eval.cost
            << " T=" << samples.front().eval.trajectory.T
            << " sigma=" << sigma;
        RCLCPP_INFO(logger, "%s", oss.str().c_str());
      }

      if (sigma < cfg_.stop_sigma) {
        break;
      }
    }

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

class VpStoPaperGlobalPlannerNode : public rclcpp::Node
{
public:
  VpStoPaperGlobalPlannerNode()
  : Node("vp_sto_paper_global_planner_node")
  {
    declare_parameter<std::string>("urdf_path", "/home/jin/harco/ecvt2_ws/forestry_robot_mjcf/ecvt_v2_upper.urdf");
    declare_parameter<std::string>("joint_state_topic", "/joint_states");
    declare_parameter<std::vector<std::string>>(
      "actuated_joints",
      std::vector<std::string>{"UPJ1", "UPJ2", "UPJ3", "UPJ4", "TOOLJ1"});
    declare_parameter<std::vector<double>>(
      "goal_actuated",
      std::vector<double>{0.2, 0.8, -0.7, 0.9, 0.0});
    declare_parameter<std::vector<double>>(
      "velocity_limits",
      std::vector<double>{0.8, 0.8, 0.8, 0.4, 0.6});
    declare_parameter<std::vector<double>>(
      "acceleration_limits",
      std::vector<double>{1.5, 1.5, 1.5, 1.0, 1.2});

    declare_parameter<int>("n_via", 5);
    declare_parameter<int>("n_eval", 81);
    declare_parameter<int>("population", 40);
    declare_parameter<int>("max_iterations", 20);
    declare_parameter<int>("elite_count", 10);
    declare_parameter<int64_t>("random_seed", -1);
    declare_parameter<double>("sigma0", 0.20);
    declare_parameter<double>("t_min", 1.0);
    declare_parameter<double>("t_max", 40.0);
    declare_parameter<double>("w_time", 1.0);
    declare_parameter<double>("w_smooth", 1e-3);
    declare_parameter<double>("w_terminal", 1e4);
    declare_parameter<double>("w_via_regularization", 1e-4);
    declare_parameter<bool>("auto_plan_on_first_state", true);
    declare_parameter<bool>("replan_periodic", false);
    declare_parameter<double>("replan_period_sec", 5.0);

    const auto urdf_path = get_parameter("urdf_path").as_string();
    const auto joint_state_topic = get_parameter("joint_state_topic").as_string();
    const auto actuated = get_parameter("actuated_joints").as_string_array();

    cfg_.n_via = get_parameter("n_via").as_int();
    cfg_.n_eval = get_parameter("n_eval").as_int();
    cfg_.population = get_parameter("population").as_int();
    cfg_.max_iterations = get_parameter("max_iterations").as_int();
    cfg_.elite_count = get_parameter("elite_count").as_int();
    cfg_.random_seed = get_parameter("random_seed").as_int();
    cfg_.sigma0 = get_parameter("sigma0").as_double();
    cfg_.t_min = get_parameter("t_min").as_double();
    cfg_.t_max = get_parameter("t_max").as_double();
    cfg_.w_time = get_parameter("w_time").as_double();
    cfg_.w_smooth = get_parameter("w_smooth").as_double();
    cfg_.w_terminal = get_parameter("w_terminal").as_double();
    cfg_.w_via_regularization = get_parameter("w_via_regularization").as_double();

    q_goal_A_ = toEigen(get_parameter("goal_actuated").as_double_array());
    const Eigen::VectorXd vlim = toEigen(get_parameter("velocity_limits").as_double_array());
    const Eigen::VectorXd alim = toEigen(get_parameter("acceleration_limits").as_double_array());

    if (q_goal_A_.size() != static_cast<int>(actuated.size())) {
      throw std::runtime_error("goal_actuated size must match actuated_joints size");
    }
    if (vlim.size() != static_cast<int>(actuated.size())) {
      throw std::runtime_error("velocity_limits size must match actuated_joints size");
    }
    if (alim.size() != static_cast<int>(actuated.size())) {
      throw std::runtime_error("acceleration_limits size must match actuated_joints size");
    }

    joint_model_ = std::make_unique<ActuatedJointModel>(urdf_path, actuated);
    time_param_ = std::make_unique<TimeParameterizer>(vlim, alim);

    traj_pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>("~/actuated_reference", 10);
    cost_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("~/planner_debug", 10);

    auto_plan_on_first_state_ = get_parameter("auto_plan_on_first_state").as_bool();
    replan_periodic_ = get_parameter("replan_periodic").as_bool();
    const double replan_period_sec = get_parameter("replan_period_sec").as_double();

    joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      joint_state_topic,
      rclcpp::SensorDataQoS(),
      std::bind(&VpStoPaperGlobalPlannerNode::jointStateCallback, this, std::placeholders::_1));

    if (replan_periodic_) {
      timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(replan_period_sec)),
        std::bind(&VpStoPaperGlobalPlannerNode::planIfPossible, this));
    }

    RCLCPP_INFO(get_logger(), "VP-STO paper global planner node ready. Waiting for joint state.");
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
      start = joint_model_->fromJointMsg(latest_joint_state_);
    } catch (const std::exception& e) {
      RCLCPP_WARN(get_logger(), "State conversion failed: %s", e.what());
      return;
    }

    const int nA = joint_model_->nA();
    Eigen::VectorXd mean_init = Eigen::VectorXd::Zero(cfg_.n_via * nA);
    for (int k = 0; k < cfg_.n_via; ++k) {
      const double alpha = static_cast<double>(k + 1) / static_cast<double>(cfg_.n_via + 1);
      const Eigen::VectorXd qk = (1.0 - alpha) * start.qA + alpha * q_goal_A_;
      mean_init.segment(k * nA, nA) = qk;
    }

    TrajectoryRolloutEvaluator evaluator(cfg_, q_goal_A_);
    CmaEsVpStoOptimizer optimizer(cfg_);
    const auto optimizer_seed = static_cast<unsigned long>(optimizer.seed());
    RCLCPP_WARN(get_logger(), "VP-STO paper optimizer seed=%lu", optimizer_seed);
    std::cout << "[vp_sto_paper_global_planner_node] VP-STO paper optimizer seed="
              << optimizer_seed << std::endl;

    const auto best = optimizer.optimize(
      mean_init,
      [&](const Eigen::VectorXd& x) {
        return evaluator.evaluate(x, start, time_param_.get());
      },
      get_logger());

    if (!best.feasible) {
      RCLCPP_WARN(get_logger(), "Planner did not find a feasible trajectory.");
      return;
    }

    publishActuatedTrajectory(best.trajectory);
    publishDebug(best);

    planned_once_ = true;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4)
        << "VP-STO paper solved: T=" << best.trajectory.T
        << ", cost=" << best.cost
        << ", smooth=" << best.trajectory.smoothness_integral;
    RCLCPP_INFO(get_logger(), "%s", oss.str().c_str());
  }

  void publishActuatedTrajectory(const TrajectorySample& traj)
  {
    trajectory_msgs::msg::JointTrajectory msg;
    msg.header.stamp = now();
    msg.joint_names = joint_model_->actuatedJointNames();

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

  void publishDebug(const CandidateEvaluation& best)
  {
    std_msgs::msg::Float64MultiArray msg;
    msg.data = {
      best.cost,
      best.trajectory.T,
      best.trajectory.smoothness_integral,
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

  std::unique_ptr<ActuatedJointModel> joint_model_;
  std::unique_ptr<TimeParameterizer> time_param_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr traj_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cost_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace vpsto_paper

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<vpsto_paper::VpStoPaperGlobalPlannerNode>();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    std::cerr << "[vp_sto_paper_global_planner_node] fatal: " << e.what() << std::endl;
  }
  rclcpp::shutdown();
  return 0;
}
