#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <Eigen/Dense>

#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>

class LiveUpperDynamicsNode : public rclcpp::Node
{
public:
  LiveUpperDynamicsNode()
  : Node("live_upper_dynamics_node"),
    first_sample_(true),
    callback_count_(0)
  {
    declare_parameter<std::string>(
      "urdf_path",
      "/home/jin/harco/ecvt2_ws/forestry_robot_mjcf/ecvt_v2_upper.urdf");
    declare_parameter<std::string>("joint_state_topic", "/joint_states");

    declare_parameter<std::vector<std::string>>(
      "state_joints",
      std::vector<std::string>{"UPJ1","UPJ2","UPJ3","UPJ4","UPJ5","UPJ6","TOOLJ1"});

    declare_parameter<std::vector<std::string>>(
      "actuated_joints",
      std::vector<std::string>{"UPJ1","UPJ2","UPJ3","UPJ4"});

    declare_parameter<std::vector<std::string>>(
      "passive_joints",
      std::vector<std::string>{"UPJ5","UPJ6"});

    declare_parameter<bool>("print_full_matrix", false);
    declare_parameter<int>("log_every_n", 20);

    urdf_path_         = get_parameter("urdf_path").as_string();
    joint_state_topic_ = get_parameter("joint_state_topic").as_string();
    state_joints_      = get_parameter("state_joints").as_string_array();
    actuated_joints_   = get_parameter("actuated_joints").as_string_array();
    passive_joints_    = get_parameter("passive_joints").as_string_array();
    print_full_matrix_ = get_parameter("print_full_matrix").as_bool();
    log_every_n_       = get_parameter("log_every_n").as_int();

    pinocchio::urdf::buildModel(urdf_path_, model_);
    data_ = std::make_unique<pinocchio::Data>(model_);

    build_joint_index_maps();

    tau_a_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("~/tau_a_est", 10);
    qdd_p_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("~/qdd_p_est", 10);

    sub_ = create_subscription<sensor_msgs::msg::JointState>(
      joint_state_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&LiveUpperDynamicsNode::jointStateCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "Loaded URDF: %s", urdf_path_.c_str());
    RCLCPP_INFO(get_logger(), "Model name: %s, nq=%d, nv=%d",
                model_.name.c_str(), model_.nq, model_.nv);
  }

private:
  struct JointIdx
  {
    int idx_q;
    int idx_v;
    int nq;
    int nv;
    bool is_continuous;
  };

  pinocchio::Model model_;
  std::unique_ptr<pinocchio::Data> data_;

  std::string urdf_path_;
  std::string joint_state_topic_;
  std::vector<std::string> state_joints_;
  std::vector<std::string> actuated_joints_;
  std::vector<std::string> passive_joints_;

  bool print_full_matrix_;
  int log_every_n_;

  std::unordered_map<std::string, JointIdx> joint_to_pin_;
  std::vector<int> idxA_q_, idxA_v_, idxP_q_, idxP_v_, idxState_q_, idxState_v_;

  bool first_sample_;
  size_t callback_count_;
  rclcpp::Time prev_stamp_;
  Eigen::VectorXd prev_vA_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr tau_a_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr qdd_p_pub_;

  void build_joint_index_maps()
  {
    auto register_joint = [this](const std::string & name)
    {
      auto jid = model_.getJointId(name);
      if (jid == 0) {
        throw std::runtime_error("Pinocchio joint not found: " + name);
      }

      const auto & jmodel = model_.joints[jid];

      JointIdx info;
      info.idx_q = static_cast<int>(jmodel.idx_q());
      info.idx_v = static_cast<int>(jmodel.idx_v());
      info.nq    = static_cast<int>(jmodel.nq());
      info.nv    = static_cast<int>(jmodel.nv());

      // Pinocchio continuous joint: nq=2, nv=1
      info.is_continuous = (info.nq == 2 && info.nv == 1);

      // 지금 노드는 아래 두 경우만 우선 지원
      // 1) 일반 1-DoF joint: nq=1, nv=1
      // 2) continuous joint: nq=2, nv=1
      if (!((info.nq == 1 && info.nv == 1) || info.is_continuous)) {
        std::ostringstream oss;
        oss << "Unsupported joint layout for '" << name
            << "': nq=" << info.nq << ", nv=" << info.nv
            << ". This node currently supports (1,1) or continuous (2,1).";
        throw std::runtime_error(oss.str());
      }

      joint_to_pin_[name] = info;
    };

    for (const auto & j : state_joints_)    register_joint(j);
    for (const auto & j : actuated_joints_) register_joint(j);
    for (const auto & j : passive_joints_)  register_joint(j);

    idxA_q_.clear(); idxA_v_.clear();
    idxP_q_.clear(); idxP_v_.clear();
    idxState_q_.clear(); idxState_v_.clear();

    for (const auto & j : state_joints_) {
      const auto & info = joint_to_pin_.at(j);
      for (int k = 0; k < info.nq; ++k) idxState_q_.push_back(info.idx_q + k);
      for (int k = 0; k < info.nv; ++k) idxState_v_.push_back(info.idx_v + k);
    }

    for (const auto & j : actuated_joints_) {
      const auto & info = joint_to_pin_.at(j);
      for (int k = 0; k < info.nq; ++k) idxA_q_.push_back(info.idx_q + k);
      for (int k = 0; k < info.nv; ++k) idxA_v_.push_back(info.idx_v + k);
    }

    for (const auto & j : passive_joints_) {
      const auto & info = joint_to_pin_.at(j);
      for (int k = 0; k < info.nq; ++k) idxP_q_.push_back(info.idx_q + k);
      for (int k = 0; k < info.nv; ++k) idxP_v_.push_back(info.idx_v + k);
    }

    // acceleration 추정은 velocity-space 기준
    int nvA = 0;
    for (const auto & j : actuated_joints_) {
      nvA += joint_to_pin_.at(j).nv;
    }
    prev_vA_ = Eigen::VectorXd::Zero(nvA);
  }

  static Eigen::MatrixXd selectRowsCols(
    const Eigen::MatrixXd & M,
    const std::vector<int> & rows,
    const std::vector<int> & cols)
  {
    Eigen::MatrixXd out(rows.size(), cols.size());
    for (size_t i = 0; i < rows.size(); ++i)
      for (size_t j = 0; j < cols.size(); ++j)
        out(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = M(rows[i], cols[j]);
    return out;
  }

  static Eigen::VectorXd selectRows(const Eigen::VectorXd & v, const std::vector<int> & rows)
  {
    Eigen::VectorXd out(rows.size());
    for (size_t i = 0; i < rows.size(); ++i)
      out(static_cast<Eigen::Index>(i)) = v(rows[i]);
    return out;
  }

  static std::vector<int> fullIndex(int n)
  {
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    return idx;
  }

  static std_msgs::msg::Float64MultiArray toMsg(const Eigen::VectorXd & x)
  {
    std_msgs::msg::Float64MultiArray msg;
    msg.data.resize(static_cast<size_t>(x.size()));
    for (int i = 0; i < x.size(); ++i) msg.data[static_cast<size_t>(i)] = x[i];
    return msg;
  }

  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    if (msg->name.size() != msg->position.size() || msg->name.size() != msg->velocity.size()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "JointState size mismatch");
      return;
    }

    std::unordered_map<std::string, size_t> ros_idx;
    ros_idx.reserve(msg->name.size());
    for (size_t i = 0; i < msg->name.size(); ++i) {
      ros_idx[msg->name[i]] = i;
    }

    Eigen::VectorXd q = Eigen::VectorXd::Zero(model_.nq);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model_.nv);

    for (const auto & j : state_joints_) {
      auto it = ros_idx.find(j);
      if (it == ros_idx.end()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                            "Joint '%s' not found in JointState", j.c_str());
        return;
      }

      const auto & pin_idx = joint_to_pin_.at(j);
      const size_t k = it->second;

      const double theta  = msg->position[k];
      const double dtheta = msg->velocity[k];

      if (pin_idx.is_continuous) {
        // continuous joint in Pinocchio: q = [cos(theta), sin(theta)], v = [dtheta]
        q(pin_idx.idx_q + 0) = std::cos(theta);
        q(pin_idx.idx_q + 1) = std::sin(theta);
        v(pin_idx.idx_v)     = dtheta;
      } else {
        q(pin_idx.idx_q) = theta;
        v(pin_idx.idx_v) = dtheta;
      }
    }

    pinocchio::computeAllTerms(model_, *data_, q, v);

    pinocchio::crba(model_, *data_, q);
    Eigen::MatrixXd M = data_->M;
    M.triangularView<Eigen::StrictlyLower>() =
      M.transpose().triangularView<Eigen::StrictlyLower>();

    pinocchio::computeCoriolisMatrix(model_, *data_, q, v);
    Eigen::MatrixXd C = data_->C;

    pinocchio::computeGeneralizedGravity(model_, *data_, q);
    Eigen::VectorXd g = data_->g;

    pinocchio::nonLinearEffects(model_, *data_, q, v);
    Eigen::VectorXd nle = data_->nle;   // C*v + g

    const std::vector<int> idxAll = fullIndex(model_.nv);

    Eigen::MatrixXd D_A = selectRowsCols(M, idxA_v_, idxA_v_);
    Eigen::MatrixXd D_M = selectRowsCols(M, idxP_v_, idxA_v_);
    Eigen::MatrixXd D_P = selectRowsCols(M, idxP_v_, idxP_v_);

    Eigen::MatrixXd C_A = selectRowsCols(C, idxA_v_, idxAll);
    Eigen::MatrixXd C_P = selectRowsCols(C, idxP_v_, idxAll);

    Eigen::VectorXd c_A = C_A * v;
    Eigen::VectorXd c_P = C_P * v;
    Eigen::VectorXd g_A = selectRows(g, idxA_v_);
    Eigen::VectorXd g_P = selectRows(g, idxP_v_);
    Eigen::VectorXd v_A = selectRows(v, idxA_v_);

    rclcpp::Time stamp = msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0
      ? now() : rclcpp::Time(msg->header.stamp);

    if (first_sample_) {
      prev_stamp_ = stamp;
      prev_vA_ = v_A;
      first_sample_ = false;
      return;
    }

    double dt = (stamp - prev_stamp_).seconds();
    if (dt <= 1e-6) {
      return;
    }

    Eigen::VectorXd qdd_A_est = (v_A - prev_vA_) / dt;
    Eigen::VectorXd qdd_P_est = -D_P.ldlt().solve(D_M * qdd_A_est + c_P + g_P); //논문식 (3)
    Eigen::VectorXd tau_A_est = D_A * qdd_A_est + D_M.transpose() * qdd_P_est + c_A + g_A;

    tau_a_pub_->publish(toMsg(tau_A_est));
    qdd_p_pub_->publish(toMsg(qdd_P_est));

    callback_count_++;
    if (callback_count_ % static_cast<size_t>(log_every_n_) == 0) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(5);
      oss << "\n========== live upper dynamics ==========\n";
      oss << "qdd_A_est^T = " << qdd_A_est.transpose() << "\n";
      oss << "qdd_P_est^T = " << qdd_P_est.transpose() << "\n";
      oss << "tau_A_est^T = " << tau_A_est.transpose() << "\n";
      oss << "g^T         = " << g.transpose() << "\n";
      oss << "nle^T       = " << nle.transpose() << "\n";

      if (print_full_matrix_) {
        oss << "M =\n" << M << "\n";
        oss << "C =\n" << C << "\n";
      } else {
        oss << "D_A =\n" << D_A << "\n";
        oss << "D_M =\n" << D_M << "\n";
        oss << "D_P =\n" << D_P << "\n";
      }

      RCLCPP_INFO(get_logger(), "%s", oss.str().c_str());
    }

    prev_stamp_ = stamp;
    prev_vA_ = v_A;
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LiveUpperDynamicsNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}