#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "ecvt2_controller/kdl_utils.hpp"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <filesystem>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <stdexcept>

struct JacobianTarget
{
  std::string name;
  pinocchio::FrameIndex frame_id{0};
  int kdl_index{-1};

  // Reduced Pinocchio Jacobian column order after base 6 columns.
  std::vector<std::string> pin_joint_names;
  std::vector<Eigen::Index> pin_joint_v_cols;

  // Mapping from reduced Pinocchio joint columns to KDL 4 joint columns.
  // KDL format is always [base6 + 4 joints] => 6x10.
  std::vector<int> kdl_joint_map;

  // If true, apply mimic projection: J2'=J2-J4, J3'=J3-J4, use J5.
  bool use_leg_mimic_projection{false};
};

//+--------------------------------------------------------------------------------------------------------------------+
//| Main Node Class                                                                                                    |
//+--------------------------------------------------------------------------------------------------------------------+
class ComCompareSimplifyPinocchioNode : public rclcpp::Node
{
public:
  ComCompareSimplifyPinocchioNode()
  : Node("com_compare_simplify_pinocchio")
  {
    std::string urdf_path =
      ament_index_cpp::get_package_share_directory("forestry_robot_mjcf") +
      "/xml/ecvt_v2_simplify.urdf";
    if (!std::filesystem::exists(urdf_path)) {
      const std::string src_fallback =
        "/home/harco/ecvt2_ws/src/forestry_robot_mjcf/xml/ecvt_v2_simplify.urdf";
      if (std::filesystem::exists(src_fallback)) {
        urdf_path = src_fallback;
        RCLCPP_WARN(
          this->get_logger(),
          "Installed URDF not found. Falling back to source path: %s",
          urdf_path.c_str());
      }
    }

    pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), model_, false);
    data_ = std::make_unique<pinocchio::Data>(model_);
    q_ = pinocchio::neutral(model_);
    jac_frame_full_.resize(6, model_.nv);
    jac_frame_full_.setZero();

    RCLCPP_INFO(this->get_logger(), "Loaded URDF: %s", urdf_path.c_str());
    RCLCPP_INFO(this->get_logger(), "Pinocchio model nq=%d nv=%d", model_.nq, model_.nv);

    initJacobianTargets();
    initComJacobianReductionLayout();
    logPinocchioVelocityLayout();

    const auto sensor_qos = rclcpp::SensorDataQoS();

    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/hanyang/joint_states", sensor_qos,
      std::bind(&ComCompareSimplifyPinocchioNode::jointStateCallback, this, std::placeholders::_1));

    base_pos_sub_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
      "/hanyang/base_pos", sensor_qos,
      std::bind(&ComCompareSimplifyPinocchioNode::basePosCallback, this, std::placeholders::_1));

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "/hanyang/imu", sensor_qos,
      std::bind(&ComCompareSimplifyPinocchioNode::imuCallback, this, std::placeholders::_1));

    com_msg_sub_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
      "/hanyang/com_pos", sensor_qos,
      std::bind(&ComCompareSimplifyPinocchioNode::comCallback, this, std::placeholders::_1));

    pin_com_pub_ = this->create_publisher<geometry_msgs::msg::Vector3Stamped>(
      "/hanyang/pin_com_pos", 10);

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(50),
      std::bind(&ComCompareSimplifyPinocchioNode::compareCom, this));
  }

private:
  //+------------------------------------------------------------------------------------------------------------------+
  //| Pinocchio qdot Layout                                                                                           |
  //+------------------------------------------------------------------------------------------------------------------+
  // Jcom (3 x nv) must be multiplied by v in this exact column order.
  void logPinocchioVelocityLayout() const
  {
    std::ostringstream oss;
    oss << "Pinocchio velocity layout (for Jcom * v), nv=" << model_.nv << '\n';
    oss << "  v[0..5] = free-flyer base velocity (Pinocchio tangent space order)\n";
    oss << "  joints:\n";

    for (pinocchio::JointIndex jid = 1; jid < static_cast<pinocchio::JointIndex>(model_.joints.size()); ++jid) {
      const auto & jm = model_.joints[jid];
      oss << "    " << model_.names[jid]
          << " : idx_v=" << jm.idx_v()
          << ", nv=" << jm.nv() << '\n';
    }

    RCLCPP_INFO_STREAM(this->get_logger(), oss.str());
  }

  //+------------------------------------------------------------------------------------------------------------------+
  //| ROS Topic Callbacks                                                                                              |
  //+------------------------------------------------------------------------------------------------------------------+
  // Cache latest measurements from topics.
  // These values are used to build q and run COM/Jacobian comparison.
  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    joint_pos_ = msg->position;
    has_joint_ = true;
  }

  void comCallback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
  {
    com_from_topic_(0) = msg->vector.x;
    com_from_topic_(1) = msg->vector.y;
    com_from_topic_(2) = msg->vector.z;
    has_com_ = true;
  }

  void basePosCallback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
  {
    base_pos_(0) = msg->vector.x;
    base_pos_(1) = msg->vector.y;
    base_pos_(2) = msg->vector.z;
    has_base_pos_ = true;
  }

  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    Eigen::Vector4d q;
    q << msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w;
    const double n = q.norm();
    if (!q.allFinite() || n < 1e-12) {
      return;
    }
    base_quat_xyzw_ = q / n;
    has_imu_ = true;
  }

  //+------------------------------------------------------------------------------------------------------------------+
  //| q Mapping Helpers                                                                                                |
  //+------------------------------------------------------------------------------------------------------------------+
  // Writes one scalar joint value into Pinocchio configuration vector q_.
  // For continuous nq=2 joints, uses q=[cos(theta), sin(theta)].
  void setJointIfExists(const std::string & joint_name, double value)
  {
    const pinocchio::JointIndex jid = model_.getJointId(joint_name);
    if (jid == 0) {
      return;
    }

    const auto & jmodel = model_.joints[jid];
    if (jmodel.nq() == 1) {
      q_(jmodel.idx_q()) = value;
      return;
    }

    if (jmodel.nq() == 2) {
      // Continuous joint convention in Pinocchio: [cos(theta), sin(theta)]
      q_(jmodel.idx_q()) = std::cos(value);
      q_(jmodel.idx_q() + 1) = std::sin(value);
    }
  }

  void mapLeg(
    const std::string & prefix,
    size_t idx_j1, size_t idx_j2, size_t idx_j3, size_t idx_j5, size_t idx_jw)
  {
    const double j1 = joint_pos_.at(idx_j1);
    const double j2 = joint_pos_.at(idx_j2);
    const double j3 = joint_pos_.at(idx_j3);
    // Mimic equation used in this node:
    //   q4 = -(q2 + q3)
    const double j4 = -(j2 + j3);
    const double j5 = joint_pos_.at(idx_j5);
    const double jw = joint_pos_.at(idx_jw);

    setJointIfExists(prefix + "J1", j1);
    setJointIfExists(prefix + "J2", j2);
    setJointIfExists(prefix + "J3", j3);
    setJointIfExists(prefix + "J4", j4);
    setJointIfExists(prefix + "J5", j5);
    setJointIfExists(prefix + "JW", jw);
  }

  bool buildQFromTopics(Eigen::VectorXd & q_kdl)
  {
    if (joint_pos_.size() < 24) {
      return false;
    }

    q_ = pinocchio::neutral(model_);
    // Free-flyer q: [x y z qx qy qz qw]
    q_.segment<3>(0) = base_pos_;
    q_.segment<4>(3) = base_quat_xyzw_;

    mapLeg("FR", 0, 1, 2, 3, 4);
    mapLeg("FL", 5, 6, 7, 8, 9);
    mapLeg("RR", 10, 11, 12, 13, 14);
    mapLeg("RL", 15, 16, 17, 18, 19);

    setJointIfExists("UPJ1", joint_pos_.at(20));
    setJointIfExists("UPJ2", joint_pos_.at(21));
    setJointIfExists("UPJ3", joint_pos_.at(22));
    setJointIfExists("UPJ4", joint_pos_.at(23));
    if (joint_pos_.size() > 24) {
      setJointIfExists("UPJ5", joint_pos_.at(24));
    }
    if (joint_pos_.size() > 25) {
      setJointIfExists("UPJ6", joint_pos_.at(25));
    }

    // kdl_utils input convention:
    //   q_kdl in R^27 (articulation-only vector)
    q_kdl = Eigen::VectorXd::Zero(27);
    const size_t n = std::min(joint_pos_.size(), static_cast<size_t>(27));
    for (size_t i = 0; i < n; ++i) {
      q_kdl(static_cast<Eigen::Index>(i)) = joint_pos_.at(i);
    }
    return true;
  }

  //+------------------------------------------------------------------------------------------------------------------+
  //| Core 1: Pinocchio COM                                                                                           |
  //+------------------------------------------------------------------------------------------------------------------+
  // Computes center of mass:
  //   c = COM(q)
  Eigen::Vector3d computePinocchioCom()
  {
    pinocchio::centerOfMass(model_, *data_, q_);
    return data_->com[0];
  }

  // Computes CoM Jacobian:
  //   v_com = J_com(q) * v,   J_com in R^{3 x nv}
  Eigen::MatrixXd computePinocchioComJacobian()
  {
    pinocchio::jacobianCenterOfMass(model_, *data_, q_, false);
    return data_->Jcom;
  }

  // Builds reduced control-space CoM Jacobian:
  //   Jcom_u in R^{3x30}, u = [base6, legs20, arm4]
  // Leg per-limb order (5 cols): [J1, J2-J4, J3-J4, J5, JW]
  // with mimic velocity relation: J4dot = -(J2dot + J3dot)
  Eigen::Matrix<double, 3, 30> buildReducedComJacobian(
    const Eigen::Ref<const Eigen::MatrixXd> & jcom_full) const
  {
    Eigen::Matrix<double, 3, 30> out = Eigen::Matrix<double, 3, 30>::Zero();
    out.block<3, 6>(0, 0) = jcom_full.block(0, 0, 3, 6);

    int out_col = 6;
    for (size_t i = 0; i < leg_cols_.size(); ++i) {
      const auto & leg = leg_cols_[i];
      out.col(out_col++) = jcom_full.col(leg.j1);
      out.col(out_col++) = jcom_full.col(leg.j2) - jcom_full.col(leg.j4);
      out.col(out_col++) = jcom_full.col(leg.j3) - jcom_full.col(leg.j4);
      out.col(out_col++) = jcom_full.col(leg.j5);
      out.col(out_col++) = jcom_full.col(leg.jw);
    }

    for (size_t i = 0; i < arm_cols_.size(); ++i) {
      out.col(out_col++) = jcom_full.col(arm_cols_[i]);
    }
    return out;
  }

  //+------------------------------------------------------------------------------------------------------------------+
  //| Core 2: KDL Jacobian                                                                                            |
  //+------------------------------------------------------------------------------------------------------------------+
  // Delegates to kdl_utils:
  //   {T_i(q), J_i(q)} = computeFKAllAndJacobian(...)
  ecvt2_controller::FKJacobianResult computeKdlJacobians(const Eigen::Ref<const Eigen::VectorXd> & q_kdl)
  {
    return ecvt2_controller::computeFKAllAndJacobian(base_pos_, base_quat_xyzw_, q_kdl);
  }

  //+------------------------------------------------------------------------------------------------------------------+
  //| Jacobian Format Helper                                                                                          |
  //+------------------------------------------------------------------------------------------------------------------+
  // Extracts one Jacobian column with row order:
  //   [v_x v_y v_z w_x w_y w_z]^T
  // (linear first, angular next)
  static Eigen::Matrix<double, 6, 1> toLinearAngularColumn(
    const Eigen::Ref<const Eigen::MatrixXd> & jac_pin, Eigen::Index col)
  {
    Eigen::Matrix<double, 6, 1> out;
    out.segment<3>(0) = jac_pin.block<3, 1>(0, col);
    out.segment<3>(3) = jac_pin.block<3, 1>(3, col);
    return out;
  }

  //+------------------------------------------------------------------------------------------------------------------+
  //| Core 3: Pinocchio J                                                                                            |
  //+------------------------------------------------------------------------------------------------------------------+
  // Builds KDL-compatible Jacobian per target:
  //   J_pin_kdl_like in R^{6x10} = [J_base(6 cols), J_joint(4 cols)]
  //
  // Leg projection (mimic):
  //   J2_eff = J2 - J4
  //   J3_eff = J3 - J4
  //   keep J1, J5
  Eigen::Matrix<double, 6, 10> computePinocchioJacobianKdlLayout(const JacobianTarget & target)
  {
    // Full Jacobian for this frame: 6 x nv
    pinocchio::getFrameJacobian(
      model_, *data_, target.frame_id,
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
      jac_frame_full_);

    Eigen::MatrixXd full_linear_angular = Eigen::MatrixXd::Zero(6, model_.nv);
    for (int col = 0; col < model_.nv; ++col) {
      full_linear_angular.col(col) = toLinearAngularColumn(jac_frame_full_, static_cast<Eigen::Index>(col));
    }

    // Reduced: [base6 + all target joints]
    Eigen::MatrixXd reduced = Eigen::MatrixXd::Zero(
      6, 6 + static_cast<Eigen::Index>(target.pin_joint_v_cols.size()));
    reduced.block(0, 0, 6, 6) = full_linear_angular.block(0, 0, 6, 6);
    for (size_t i = 0; i < target.pin_joint_v_cols.size(); ++i) {
      reduced.col(6 + static_cast<Eigen::Index>(i)) = full_linear_angular.col(target.pin_joint_v_cols[i]);
    }

    // Project to KDL-compatible 6x10
    Eigen::Matrix<double, 6, 10> out = Eigen::Matrix<double, 6, 10>::Zero();
    out.block<6, 6>(0, 0) = reduced.block(0, 0, 6, 6);

    if (target.use_leg_mimic_projection) {
      // Leg reduced order: [base6, J1, J2, J3, J4, J5, JW]
      // KDL effective columns: [J1, J2-J4, J3-J4, J5]
      const Eigen::Matrix<double, 6, 1> J1 = reduced.col(6);
      const Eigen::Matrix<double, 6, 1> J2 = reduced.col(7);
      const Eigen::Matrix<double, 6, 1> J3 = reduced.col(8);
      const Eigen::Matrix<double, 6, 1> J4 = reduced.col(9);
      const Eigen::Matrix<double, 6, 1> J5 = reduced.col(10);

      out.col(6) = J1;
      out.col(7) = J2 - J4;
      out.col(8) = J3 - J4;
      out.col(9) = J5;
      return out;
    }

    // Arm: direct mapping to 4 KDL columns.
    for (int i = 0; i < 4; ++i) {
      out.col(6 + i) = reduced.col(6 + target.kdl_joint_map[static_cast<size_t>(i)]);
    }
    return out;
  }

  static std::string formatMatrix(const Eigen::Ref<const Eigen::MatrixXd> & mat)
  {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    for (Eigen::Index r = 0; r < mat.rows(); ++r) {
      for (Eigen::Index c = 0; c < mat.cols(); ++c) {
        oss << std::setw(12) << mat(r, c);
      }
      oss << '\n';
    }
    return oss.str();
  }

  //+------------------------------------------------------------------------------------------------------------------+
  //| Core 4: J Comparison                                                                                            |
  //+------------------------------------------------------------------------------------------------------------------+
  // For each target:
  //   J_diff = J_pin_kdl_like - J_kdl
  // and prints KDL / PIN / DIFF matrices.
  double compareKdlAndPinJacobians(const ecvt2_controller::FKJacobianResult & kdl)
  {
    double pin_layout_us_total = 0.0;
    for (const auto & target : targets_) {
      const Eigen::MatrixXd & J_kdl = kdl.J[static_cast<size_t>(target.kdl_index)];
      const auto pin_layout_start = std::chrono::steady_clock::now();
      const Eigen::Matrix<double, 6, 10> J_pin = computePinocchioJacobianKdlLayout(target);
      const auto pin_layout_end = std::chrono::steady_clock::now();
      pin_layout_us_total += std::chrono::duration<double, std::micro>(
        pin_layout_end - pin_layout_start).count();

      // Prevent aggressive optimization from eliminating Jacobian work during timing.
      jacobian_sink_ += J_pin(0, 0) + J_kdl(0, 0);

      // RCLCPP_INFO_STREAM(this->get_logger(),
      //   "[JAC KDL][" << target.name << "]\n" << formatMatrix(J_kdl));
      // RCLCPP_INFO_STREAM(this->get_logger(),
      //   "[JAC PIN][" << target.name << "]\n" << formatMatrix(J_pin));
      // RCLCPP_INFO_STREAM(this->get_logger(),
      //   "[JAC DIFF][" << target.name << "]\n" << formatMatrix(J_diff));
    }
    return pin_layout_us_total;
  }

  //+------------------------------------------------------------------------------------------------------------------+
  //| Target Initialization                                                                                           |
  //+------------------------------------------------------------------------------------------------------------------+
  // Defines FR/FL/RR/RL/UP5 targets, frame IDs,
  // and joint-column mappings used by comparison.
  void initJacobianTargets()
  {
    auto frame_or_zero = [&](const std::string & frame_name) -> pinocchio::FrameIndex {
      const pinocchio::FrameIndex id = model_.getFrameId(frame_name);
      if (id >= model_.frames.size()) {
        RCLCPP_ERROR(this->get_logger(), "Frame '%s' not found.", frame_name.c_str());
        return 0;
      }
      return id;
    };

    auto joint_v_or_neg1 = [&](const std::string & joint_name) -> Eigen::Index {
      const pinocchio::JointIndex jid = model_.getJointId(joint_name);
      if (jid == 0) {
        RCLCPP_ERROR(this->get_logger(), "Joint '%s' not found.", joint_name.c_str());
        return -1;
      }
      const auto & jm = model_.joints[jid];
      if (jm.nv() != 1) {
        RCLCPP_ERROR(this->get_logger(), "Joint '%s' nv=%d is unsupported.", joint_name.c_str(), jm.nv());
        return -1;
      }
      return static_cast<Eigen::Index>(jm.idx_v());
    };

    auto make_target = [&](JacobianTarget target) {
      target.pin_joint_v_cols.clear();
      target.pin_joint_v_cols.reserve(target.pin_joint_names.size());
      for (const auto & name : target.pin_joint_names) {
        target.pin_joint_v_cols.push_back(joint_v_or_neg1(name));
      }
      targets_.push_back(std::move(target));
    };

    targets_.clear();
    targets_.reserve(5);

    make_target({
      "FRW", frame_or_zero("FRW"), 1,
      {"FRJ1", "FRJ2", "FRJ3", "FRJ4", "FRJ5", "FRJW"}, {}, {0, 1, 2, 4}, true});
    make_target({
      "FLW", frame_or_zero("FLW"), 2,
      {"FLJ1", "FLJ2", "FLJ3", "FLJ4", "FLJ5", "FLJW"}, {}, {0, 1, 2, 4}, true});
    make_target({
      "RRW", frame_or_zero("RRW"), 3,
      {"RRJ1", "RRJ2", "RRJ3", "RRJ4", "RRJ5", "RRJW"}, {}, {0, 1, 2, 4}, true});
    make_target({
      "RLW", frame_or_zero("RLW"), 4,
      {"RLJ1", "RLJ2", "RLJ3", "RLJ4", "RLJ5", "RLJW"}, {}, {0, 1, 2, 4}, true});
    make_target({
      "UP5", frame_or_zero("UP5"), 5,
      {"UPJ1", "UPJ2", "UPJ3", "UPJ4"}, {}, {0, 1, 2, 3}, false});
  }

  Eigen::Index requireSingleDoFJointVCol(const std::string & joint_name) const
  {
    const pinocchio::JointIndex jid = model_.getJointId(joint_name);
    if (jid == 0) {
      throw std::runtime_error("Joint not found: " + joint_name);
    }
    const auto & jm = model_.joints[jid];
    if (jm.nv() != 1) {
      throw std::runtime_error("Joint nv != 1: " + joint_name);
    }
    return static_cast<Eigen::Index>(jm.idx_v());
  }

  void initComJacobianReductionLayout()
  {
    auto set_leg = [&](size_t idx, const std::string & prefix) {
      leg_cols_[idx] = {
        requireSingleDoFJointVCol(prefix + "J1"),
        requireSingleDoFJointVCol(prefix + "J2"),
        requireSingleDoFJointVCol(prefix + "J3"),
        requireSingleDoFJointVCol(prefix + "J4"),
        requireSingleDoFJointVCol(prefix + "J5"),
        requireSingleDoFJointVCol(prefix + "JW")
      };
    };

    set_leg(0, "FR");
    set_leg(1, "FL");
    set_leg(2, "RR");
    set_leg(3, "RL");

    arm_cols_[0] = requireSingleDoFJointVCol("UPJ1");
    arm_cols_[1] = requireSingleDoFJointVCol("UPJ2");
    arm_cols_[2] = requireSingleDoFJointVCol("UPJ3");
    arm_cols_[3] = requireSingleDoFJointVCol("UPJ4");

    RCLCPP_INFO(this->get_logger(),
      "CoM Jacobian control layout ready: 3x30 = 3x(6 base + 20 legs + 4 arm)");
  }

  //+------------------------------------------------------------------------------------------------------------------+
  //| Main Periodic Routine                                                                                           |
  //+------------------------------------------------------------------------------------------------------------------+
  // Pipeline:
  // 1) Build q from topic cache
  // 2) Pinocchio Jacobian precompute: computeJointJacobians + updateFramePlacements
  // 3) Compute/publish COM
  // 4) Compute KDL Jacobians
  // 5) Compare/print KDL vs Pinocchio Jacobians
  void compareCom()
  {
    if (!has_joint_ || !has_com_ || !has_base_pos_ || !has_imu_) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Waiting data... joint=%s com=%s base=%s imu=%s",
        has_joint_ ? "ok" : "missing",
        has_com_ ? "ok" : "missing",
        has_base_pos_ ? "ok" : "missing",
        has_imu_ ? "ok" : "missing");
      return;
    }

    Eigen::VectorXd q_kdl;
    if (!buildQFromTopics(q_kdl)) {
      return;
    }

    // Prepare Pinocchio Jacobian pipeline once, then reuse for all targets.
    const auto pin_setup_start = std::chrono::steady_clock::now();
    pinocchio::computeJointJacobians(model_, *data_, q_);
    pinocchio::updateFramePlacements(model_, *data_);
    const auto pin_setup_end = std::chrono::steady_clock::now();

    const Eigen::Vector3d com_pin = computePinocchioCom();
    const Eigen::MatrixXd jcom_pin = computePinocchioComJacobian();
    const Eigen::Matrix<double, 3, 30> jcom_reduced = buildReducedComJacobian(jcom_pin);

    geometry_msgs::msg::Vector3Stamped pin_com_msg;
    pin_com_msg.header.stamp = this->now();
    pin_com_msg.header.frame_id = "world";
    pin_com_msg.vector.x = com_pin(0);
    pin_com_msg.vector.y = com_pin(1);
    pin_com_msg.vector.z = com_pin(2);
    pin_com_pub_->publish(pin_com_msg);

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "[COM JAC PIN] shape=%ldx%ld, fro_norm=%.6f",
      static_cast<long>(jcom_pin.rows()),
      static_cast<long>(jcom_pin.cols()),
      jcom_pin.norm());
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "[COM JAC REDUCED] shape=%ldx%ld, fro_norm=%.6f",
      static_cast<long>(jcom_reduced.rows()),
      static_cast<long>(jcom_reduced.cols()),
      jcom_reduced.norm());
    const Eigen::MatrixXd jcom_reduced_dyn = jcom_reduced;
    RCLCPP_INFO_STREAM_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "[COM JAC REDUCED MATRIX]\n" << formatMatrix(jcom_reduced_dyn));

    const auto kdl_start = std::chrono::steady_clock::now();
    const auto kdl = computeKdlJacobians(q_kdl);
    const auto kdl_end = std::chrono::steady_clock::now();

    const double pin_layout_us = compareKdlAndPinJacobians(kdl);
    const double pin_setup_us = std::chrono::duration<double, std::micro>(
      pin_setup_end - pin_setup_start).count();
    const double kdl_us = std::chrono::duration<double, std::micro>(kdl_end - kdl_start).count();
    const double pin_total_us = pin_setup_us + pin_layout_us;

    if (!timing_warmup_done_) {
      timing_warmup_done_ = true;
      return;
    }

    timing_samples_++;
    pin_jac_calc_us_acc_ += pin_total_us;
    kdl_us_acc_ += kdl_us;

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "[JAC TIME] pin_jac_calc: last=%.3f ms (setup=%.3f + layout=%.3f), avg=%.3f ms | "
      "kdl_jac_calc: last=%.3f ms, avg=%.3f ms | samples=%zu",
      pin_total_us * 1e-3,
      pin_setup_us * 1e-3,
      pin_layout_us * 1e-3,
      (pin_jac_calc_us_acc_ / static_cast<double>(timing_samples_)) * 1e-3,
      kdl_us * 1e-3,
      (kdl_us_acc_ / static_cast<double>(timing_samples_)) * 1e-3,
      timing_samples_);
  }

private:
  //+------------------------------------------------------------------------------------------------------------------+
  //| Runtime State                                                                                                   |
  //+------------------------------------------------------------------------------------------------------------------+
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr base_pos_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr com_msg_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr pin_com_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  pinocchio::Model model_;
  std::unique_ptr<pinocchio::Data> data_;
  Eigen::VectorXd q_;
  Eigen::MatrixXd jac_frame_full_;
  std::vector<JacobianTarget> targets_;

  std::vector<double> joint_pos_;
  Eigen::Vector3d com_from_topic_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d base_pos_ = Eigen::Vector3d::Zero();
  Eigen::Vector4d base_quat_xyzw_ = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
  bool has_joint_ = false;
  bool has_com_ = false;
  bool has_base_pos_ = false;
  bool has_imu_ = false;

  size_t timing_samples_ = 0;
  double pin_jac_calc_us_acc_ = 0.0;
  double kdl_us_acc_ = 0.0;
  bool timing_warmup_done_ = false;
  volatile double jacobian_sink_ = 0.0;

  struct LegJointVCols
  {
    Eigen::Index j1{-1};
    Eigen::Index j2{-1};
    Eigen::Index j3{-1};
    Eigen::Index j4{-1};
    Eigen::Index j5{-1};
    Eigen::Index jw{-1};
  };
  std::array<LegJointVCols, 4> leg_cols_{};
  std::array<Eigen::Index, 4> arm_cols_{{-1, -1, -1, -1}};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ComCompareSimplifyPinocchioNode>());
  rclcpp::shutdown();
  return 0;
}
