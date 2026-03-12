#include "ecvt2_controller/kdl_utils.hpp"
#include "ecvt2_controller/wbc_hqp_lib.hpp"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <OsqpEigen/OsqpEigen.h>
#include <iostream>
#include <sstream>
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <array>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <ctime>
#include <cmath>
#include <vector>

// --- 프로파일링용 ---
uint64_t iter_count_ = 0;
double ema_ms_ = 0.0;
double max_ms_ = 0.0;

static double deg2rad(double deg) { return deg * M_PI / 180.0; }

using namespace ecvt2_controller;
namespace hqp = ecvt2_controller::wbc_hqp_lib;

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
                triplets.emplace_back(row, col, value);
        }
    }

    Eigen::SparseMatrix<double> sparse(matrix.rows(), matrix.cols());
    sparse.setFromTriplets(triplets.begin(), triplets.end());
    return sparse;
}
} // namespace

class EcvtController : public rclcpp::Node
{
public:
    static constexpr int kModelDof = 26;
    static constexpr int kLowerDof = 22;
    static constexpr int kArmDof = 4;
    static constexpr double kInfBound = 1e10;
    static constexpr double kLoopDt = 0.01;
    static constexpr double kJointPosSafetyRatio = 0.9;
    static constexpr double kPosLimitRateScale = 10.0;

    EcvtController() : Node("base_arm_task_controller_v2")
    {
        // ----- Subscribers -----
        base_pos_sub_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
            "/hanyang/base_pos", 10,
            std::bind(&EcvtController::basePosCallback, this, std::placeholders::_1));

        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/hanyang/imu", 10,
            std::bind(&EcvtController::imuCallback, this, std::placeholders::_1));

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/hanyang/joint_states", 10,
            std::bind(&EcvtController::jointStateCallback, this, std::placeholders::_1));

        // ----- Publishers -----
        velocity_cmd_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
            "/hanyang/velocity_cmd", 10);

        com_pos_sub_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
            "/hanyang/com_pos", 10,
            std::bind(&EcvtController::comPosCallback, this, std::placeholders::_1));

        control_enable_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/hanyang/control_enable", 10,
            [this](const std_msgs::msg::Bool::SharedPtr msg)
            {
                prev_control_enabled_ = control_enabled_;
                control_enabled_ = msg->data;
                RCLCPP_INFO(this->get_logger(), "control_enabled: %s", control_enabled_ ? "true" : "false");

                // 상승엣지에서 CSV 오픈
                if (!prev_control_enabled_ && control_enabled_)
                {
                    capture_desired_on_next_loop_ = true;
                    openCsvIfNeeded();
                }
            });

        base_target_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/hanyang/base_target_zyx", 10,
            std::bind(&EcvtController::baseTargetCallback, this, std::placeholders::_1));

        ee_target_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
            "/hanyang/ee_target", 10,
            std::bind(&EcvtController::eeTargetCallback, this, std::placeholders::_1));

        marker_array_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "eigen_markers", rclcpp::QoS(1).transient_local().reliable());

        vcmd_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "v_cmd_marker", 10);

        ee_pos_pub_ = this->create_publisher<geometry_msgs::msg::Vector3Stamped>(
            "/hanyang/ee_pos", 10);

        arm_debug_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/hanyang/arm_joint_debug", 10);

        // ----- Control loop timer -----
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&EcvtController::controlLoop, this));

        RCLCPP_INFO(this->get_logger(), "EcvtController initialized with 100Hz loop");

        // 초기 목표 위치 설정
        FR_desired_ = Eigen::Vector3d(2.85, -0.92, 0.65);
        FL_desired_ = Eigen::Vector3d(2.85, 0.92, 0.65);
        RR_desired_ = Eigen::Vector3d(-3.65, -1.06, 0.65);
        RL_desired_ = Eigen::Vector3d(-3.65, 1.06, 0.65);
        ee_desired_ = Eigen::Vector3d(7.0, 0.0, 5.0);
        base_desired_ = Eigen::Vector3d(-0.25, -0.012, 0.14);
        base_R_desired_ = Eigen::Matrix3d::Identity();

        initializeModelToHwIndex();
        initializeJointLimits();
    }

    void closeCsv()
    {
        if (csv_open_)
        {
            RCLCPP_INFO(this->get_logger(), "Closing CSV file...");
            csv_.flush();
            csv_.close();
            csv_open_ = false;
        }
    }

private:
    using Vec26 = Eigen::Matrix<double, kModelDof, 1>;
    using Vec22 = Eigen::Matrix<double, kLowerDof, 1>;

    // ----- CSV -----
    std::ofstream csv_;
    bool csv_open_ = false;
    std::string csv_path_;
    bool prev_control_enabled_ = false;

    // ----- COM pose/velocity (from /hanyang/com_pos) -----
    Eigen::Vector3d com_pos_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d com_pos_prev_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d com_vel_ = Eigen::Vector3d::Zero();
    rclcpp::Time com_stamp_prev_;
    bool have_com_prev_ = false;

    void openCsvIfNeeded()
    {
        if (csv_open_)
            return;

        const std::string dir = "/home/harco/base_opt_csv/csv";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);

        auto now = this->now();
        std::time_t t = now.seconds();
        std::tm tm{};
        localtime_r(&t, &tm);
        char name[64];
        std::strftime(name, sizeof(name), "ecvt_log_%Y%m%d_%H%M%S.csv", &tm);

        csv_path_ = dir + "/" + name;
        csv_.open(csv_path_, std::ios::out | std::ios::trunc);
        if (!csv_.is_open())
        {
            std::cout << "[CSV] Failed to open: " << csv_path_ << std::endl;
            return;
        }

        csv_open_ = true;
        std::cout << "[CSV] OPEN: " << csv_path_ << std::endl;
        writeCsvHeader();
    }

    void writeCsvHeader()
    {
        if (!csv_open_)
            return;
        csv_
            << "time"
            << ",base_x,base_y,base_z"
            << ",fr_x,fr_y,fr_z"
            << ",fl_x,fl_y,fl_z"
            << ",rr_x,rr_y,rr_z"
            << ",rl_x,rl_y,rl_z"
            << ",ee_x,ee_y,ee_z"
            << ",com_x,com_y,com_z"
            << ",com_vx,com_vy,com_vz"
            << ",ee_target_x,ee_target_y,ee_target_z"
            << ",FR0,FR1,FR2,FR3"
            << ",FL0,FL1,FL2,FL3"
            << ",RR0,RR1,RR2,RR3"
            << ",RL0,RL1,RL2,RL3"
            << ",logdetA"
            << ",ARM0,ARM1,ARM2,ARM3"
            << ",PASS1,PASS2"
            << ",h0,h1,h2,h3"
            << ",joint_limit_active"
            << "\n";
        csv_.flush();
    }

    void writeCsvRow(const Eigen::Vector3d pos[6],
                     const Eigen::Matrix<double, 16, 1> &leg_joint_positions,
                     double logdetA,
                     const Eigen::Matrix<double, 4, 1> &q_arm,
                     const Eigen::Matrix<double, 4, 1> &h,
                     bool joint_limit_active,
                     double q_pass1, double q_pass2)
    {
        if (!csv_open_)
            return;

        const double tsec = this->now().seconds();

        csv_ << std::fixed << std::setprecision(6)
             << tsec
             << "," << pos[0].x() << "," << pos[0].y() << "," << pos[0].z()
             << "," << pos[1].x() << "," << pos[1].y() << "," << pos[1].z()
             << "," << pos[2].x() << "," << pos[2].y() << "," << pos[2].z()
             << "," << pos[3].x() << "," << pos[3].y() << "," << pos[3].z()
             << "," << pos[4].x() << "," << pos[4].y() << "," << pos[4].z()
             << "," << pos[5].x() << "," << pos[5].y() << "," << pos[5].z()
             << "," << com_pos_.x() << "," << com_pos_.y() << "," << com_pos_.z()
             << "," << com_vel_.x() << "," << com_vel_.y() << "," << com_vel_.z()
             << "," << ee_desired_.x() << "," << ee_desired_.y() << "," << ee_desired_.z()
             << "," << leg_joint_positions(0) << "," << leg_joint_positions(1) << "," << leg_joint_positions(2) << "," << leg_joint_positions(3)
             << "," << leg_joint_positions(4) << "," << leg_joint_positions(5) << "," << leg_joint_positions(6) << "," << leg_joint_positions(7)
             << "," << leg_joint_positions(8) << "," << leg_joint_positions(9) << "," << leg_joint_positions(10) << "," << leg_joint_positions(11)
             << "," << leg_joint_positions(12) << "," << leg_joint_positions(13) << "," << leg_joint_positions(14) << "," << leg_joint_positions(15)
             << "," << logdetA
             << "," << q_arm(0) << "," << q_arm(1) << "," << q_arm(2) << "," << q_arm(3)
             << "," << q_pass1 << "," << q_pass2
             << "," << h(0) << "," << h(1) << "," << h(2) << "," << h(3)
             << "," << (joint_limit_active ? 1 : 0)
             << "\n";

        if (iter_count_ % 100 == 0)
            csv_.flush();
    }

    // ============================
    // Callbacks
    // ============================
    void basePosCallback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
    {
        base_pos_(0) = msg->vector.x;
        base_pos_(1) = msg->vector.y;
        base_pos_(2) = msg->vector.z;
    }

    void comPosCallback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
    {
        com_pos_(0) = msg->vector.x;
        com_pos_(1) = msg->vector.y;
        com_pos_(2) = msg->vector.z;

        const rclcpp::Time t_now = msg->header.stamp;
        if (have_com_prev_)
        {
            const double dt = (t_now - com_stamp_prev_).seconds();
            if (dt > 0.0)
                com_vel_ = (com_pos_ - com_pos_prev_) / dt;
            else
                com_vel_.setZero();
        }
        else
        {
            com_vel_.setZero();
            have_com_prev_ = true;
        }
        com_pos_prev_ = com_pos_;
        com_stamp_prev_ = t_now;
    }

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        Eigen::Vector4d q;
        q << msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w;

        if (!q.allFinite())
        {
            if (iter_count_ % 100 == 0)
                RCLCPP_WARN(this->get_logger(), "IMU quaternion has NaN/Inf. Keeping previous quaternion.");
            return;
        }

        const double n = q.norm();
        if (n < 1e-12)
        {
            if (iter_count_ % 100 == 0)
                RCLCPP_WARN(this->get_logger(), "IMU quaternion norm is zero. Keeping previous quaternion.");
            return;
        }

        base_quat_ = q / n;
    }

    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        size_t n = msg->position.size();
        q_ = Eigen::VectorXd::Zero(n);
        qd_ = Eigen::VectorXd::Zero(n);

        for (size_t i = 0; i < n; ++i)
        {
            q_(i) = msg->position[i];
            if (i < msg->velocity.size())
                qd_(i) = msg->velocity[i];
        }
    }

    void baseTargetCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (msg->data.size() < 6)
        {
            RCLCPP_WARN(this->get_logger(),
                        "base_target_zyx expects 6 values, got %zu", msg->data.size());
            return;
        }

        base_desired_(0) = msg->data[0];
        base_desired_(1) = msg->data[1];
        base_desired_(2) = msg->data[2];

        const double yaw = msg->data[3];
        const double pitch = msg->data[4];
        const double roll = msg->data[5];
        base_R_desired_ = eulerZYXToRot(yaw, pitch, roll);
        base_target_received_ = true;
    }

    void eeTargetCallback(const geometry_msgs::msg::Vector3::SharedPtr msg)
    {
        ee_desired_(0) = msg->x;
        ee_desired_(1) = msg->y;
        ee_desired_(2) = msg->z;
        ee_target_received_ = true;
    }

    void initializeModelToHwIndex()
    {
        model_to_hw_index_.fill(-1);

        int hw_idx = 0;
        for (int leg = 0; leg < 4; ++leg)
        {
            for (int j = 0; j < 4; ++j)
                model_to_hw_index_[6 + leg * 4 + j] = hw_idx++;
            ++hw_idx; // wheel joint slot in hw order
        }
        for (int j = 0; j < 4; ++j)
            model_to_hw_index_[22 + j] = hw_idx++;
    }

    void initializeJointLimits()
    {
        q_min_full_safe_.setConstant(-kInfBound);
        q_max_full_safe_.setConstant(kInfBound);
        qd_min_full_.setConstant(-kInfBound);
        qd_max_full_.setConstant(kInfBound);

        q_min_full_safe_.segment<4>(6) << deg2rad(-49), deg2rad(-58.4), deg2rad(-55.9), deg2rad(-180);
        q_max_full_safe_.segment<4>(6) << deg2rad(28.9), deg2rad(56.9), deg2rad(56.3), deg2rad(180);

        q_min_full_safe_.segment<4>(10) << deg2rad(-28.9), deg2rad(-58.4), deg2rad(-55.9), deg2rad(-180);
        q_max_full_safe_.segment<4>(10) << deg2rad(49), deg2rad(56.9), deg2rad(56.3), deg2rad(180);

        q_min_full_safe_.segment<4>(14) << deg2rad(-28.9), deg2rad(-56.9), deg2rad(-56.3), deg2rad(-180);
        q_max_full_safe_.segment<4>(14) << deg2rad(49), deg2rad(58.4), deg2rad(55.9), deg2rad(180);

        q_min_full_safe_.segment<4>(18) << deg2rad(-49), deg2rad(-56.9), deg2rad(-56.3), deg2rad(-180);
        q_max_full_safe_.segment<4>(18) << deg2rad(28.9), deg2rad(58.4), deg2rad(55.9), deg2rad(180);

        q_min_full_safe_.segment<4>(22) << -M_PI / 2, 0.0, -2.0106193, 0.0;
        q_max_full_safe_.segment<4>(22) << M_PI / 2, 1.0472, 0.0, 1.65;

        for (int i = 6; i < kModelDof; ++i)
        {
            const double center = 0.5 * (q_min_full_safe_(i) + q_max_full_safe_(i));
            const double half_range = 0.5 * (q_max_full_safe_(i) - q_min_full_safe_(i));
            const double safe_half = half_range * kJointPosSafetyRatio;
            q_min_full_safe_(i) = center - safe_half;
            q_max_full_safe_(i) = center + safe_half;
        }

        qd_min_full_.segment<4>(6) << -0.35, -0.35, -0.35, -1.0;
        qd_max_full_.segment<4>(6) << 0.35, 0.35, 0.35, 1.0;

        qd_min_full_.segment<4>(10) << -0.35, -0.35, -0.35, -1.0;
        qd_max_full_.segment<4>(10) << 0.35, 0.35, 0.35, 1.0;

        qd_min_full_.segment<4>(14) << -0.35, -0.35, -0.35, -1.0;
        qd_max_full_.segment<4>(14) << 0.35, 0.35, 0.35, 1.0;

        qd_min_full_.segment<4>(18) << -0.35, -0.35, -0.35, -1.0;
        qd_max_full_.segment<4>(18) << 0.35, 0.35, 0.35, 1.0;

        qd_min_full_.segment<4>(22) << -1.0, -0.5, -0.5, -0.5;
        qd_max_full_.segment<4>(22) << 1.0, 0.5, 0.5, 0.5;
    }

    template <typename FKResultT>
    void captureDesiredTargetsFromCurrentState(const FKResultT &fk_result)
    {
        // Keep captured current pose as default desired until each target topic is updated.
        if (!ee_target_received_)
            ee_desired_ = fk_result.T[5].template block<3, 1>(0, 3);

        if (!base_target_received_)
        {
            base_desired_ = fk_result.T[0].template block<3, 1>(0, 3);
            base_R_desired_ = fk_result.T[0].template block<3, 3>(0, 0);
        }

        FR_desired_ = fk_result.T[1].template block<3, 1>(0, 3) - Eigen::Vector3d(0.0, 0.0, 0.02);
        FL_desired_ = fk_result.T[2].template block<3, 1>(0, 3) - Eigen::Vector3d(0.0, 0.0, 0.02);
        RR_desired_ = fk_result.T[3].template block<3, 1>(0, 3) - Eigen::Vector3d(0.0, 0.0, 0.02);
        RL_desired_ = fk_result.T[4].template block<3, 1>(0, 3) - Eigen::Vector3d(0.0, 0.0, 0.02);
    }

    template <typename FKResultT>
    void publishEePositionDebug(const FKResultT &fk_result)
    {
        geometry_msgs::msg::Vector3Stamped ee_position_msg;
        ee_position_msg.header.stamp = this->now();
        ee_position_msg.header.frame_id = "world";
        ee_position_msg.vector.x = fk_result.T[5](0, 3);
        ee_position_msg.vector.y = fk_result.T[5](1, 3);
        ee_position_msg.vector.z = fk_result.T[5](2, 3);
        ee_pos_pub_->publish(ee_position_msg);
    }

    void publishArmJointDebug(const hqp::JointBoundData &bounds)
    {
        std_msgs::msg::Float64MultiArray arm_debug_msg;
        arm_debug_msg.data.resize(28);

        for (int arm_joint = 0; arm_joint < 4; ++arm_joint)
        {
            const int model_joint_idx = 22 + arm_joint;
            const int hw_joint_idx = model_to_hw_index_[model_joint_idx];

            double q_current = 0.0;
            if (hw_joint_idx >= 0 && hw_joint_idx < (int)q_.size())
                q_current = q_(hw_joint_idx);

            arm_debug_msg.data[0 + arm_joint] = q_current;
            arm_debug_msg.data[4 + arm_joint] = q_min_full_safe_(model_joint_idx);
            arm_debug_msg.data[8 + arm_joint] = q_max_full_safe_(model_joint_idx);
            arm_debug_msg.data[12 + arm_joint] = bounds.raw_qdot_lower(model_joint_idx);
            arm_debug_msg.data[16 + arm_joint] = bounds.raw_qdot_upper(model_joint_idx);
            arm_debug_msg.data[20 + arm_joint] = qd_min_full_(model_joint_idx);
            arm_debug_msg.data[24 + arm_joint] = qd_max_full_(model_joint_idx);
        }

        arm_debug_pub_->publish(arm_debug_msg);
    }

    sensor_msgs::msg::JointState buildVelocityCommandMessage(
        const Eigen::VectorXd &qdot_lower_final,
        const Eigen::VectorXd &qdot_arm) const
    {
        sensor_msgs::msg::JointState velocity_cmd_msg;
        velocity_cmd_msg.header.stamp = this->now();
        velocity_cmd_msg.name.resize(kModelDof);
        velocity_cmd_msg.velocity.resize(kModelDof, 0.0);
        for (int i = 0; i < kModelDof; ++i)
            velocity_cmd_msg.name[i] = "joint_" + std::to_string(i);

        velocity_cmd_msg.velocity[0] = qdot_lower_final(6);
        velocity_cmd_msg.velocity[1] = qdot_lower_final(7);
        velocity_cmd_msg.velocity[2] = qdot_lower_final(8);
        velocity_cmd_msg.velocity[3] = qdot_lower_final(9);

        velocity_cmd_msg.velocity[5] = qdot_lower_final(10);
        velocity_cmd_msg.velocity[6] = qdot_lower_final(11);
        velocity_cmd_msg.velocity[7] = qdot_lower_final(12);
        velocity_cmd_msg.velocity[8] = qdot_lower_final(13);

        velocity_cmd_msg.velocity[10] = qdot_lower_final(14);
        velocity_cmd_msg.velocity[11] = qdot_lower_final(15);
        velocity_cmd_msg.velocity[12] = qdot_lower_final(16);
        velocity_cmd_msg.velocity[13] = qdot_lower_final(17);

        velocity_cmd_msg.velocity[15] = qdot_lower_final(18);
        velocity_cmd_msg.velocity[16] = qdot_lower_final(19);
        velocity_cmd_msg.velocity[17] = qdot_lower_final(20);
        velocity_cmd_msg.velocity[18] = qdot_lower_final(21);

        velocity_cmd_msg.velocity[20] = qdot_arm(0);
        velocity_cmd_msg.velocity[21] = qdot_arm(1);
        velocity_cmd_msg.velocity[22] = qdot_arm(2);
        velocity_cmd_msg.velocity[23] = qdot_arm(3);
        return velocity_cmd_msg;
    }

    void updatePreviousCommands(
        const Eigen::VectorXd &qdot_lower_final,
        const Eigen::VectorXd &qdot_arm)
    {
        Eigen::VectorXd qdot_command_full = Eigen::VectorXd::Zero(kModelDof);
        qdot_command_full.head<22>() = qdot_lower_final;
        qdot_command_full.segment<4>(22) = qdot_arm;
        q_dot_prev_ = qdot_command_full;
        q_dot_prev_lower_ = qdot_lower_final;
    }

    void publishLoopProfile(const std::chrono::steady_clock::time_point &loop_start_time)
    {
        const auto loop_end_time = std::chrono::steady_clock::now();
        const double loop_ms = std::chrono::duration<double, std::milli>(loop_end_time - loop_start_time).count();
        iter_count_++;
        ema_ms_ = (iter_count_ == 1) ? loop_ms : (0.05 * loop_ms + 0.95 * ema_ms_);
        if (iter_count_ % 100 == 0)
        {
            const double estimated_hz = (loop_ms > 0.0) ? (1000.0 / loop_ms) : 0.0;
            RCLCPP_INFO(this->get_logger(), "[profile] loop_ms=%.3f | ema_ms=%.3f | est_hz=%.1f",
                        loop_ms, ema_ms_, estimated_hz);
        }
    }

    bool solveLowerTaskWithLocks(
        const Eigen::MatrixXd &task_jacobian,
        const Eigen::VectorXd &task_velocity_command,
        const std::vector<Eigen::MatrixXd> &locked_jacobians,
        const std::vector<Eigen::VectorXd> &locked_targets,
        const Eigen::MatrixXd &wheel_jacobian,
        const Eigen::VectorXd &wheel_lock_rhs,
        const Eigen::MatrixXd &contact_jacobian,
        const Eigen::VectorXd &qdot_lower_limits,
        const Eigen::VectorXd &qdot_upper_limits,
        const Eigen::VectorXd &qdot_previous,
        double lambda_qdot_norm,
        double lambda_qdot_smooth,
        Eigen::VectorXd &qdot_solution,
        Eigen::VectorXd &task_slack)
    {
        const int n = kLowerDof;
        const int m_task = static_cast<int>(task_jacobian.rows());
        const int m_wheel = static_cast<int>(wheel_jacobian.rows());
        const int m_contact = static_cast<int>(contact_jacobian.rows());
        const int n_lock = static_cast<int>(locked_jacobians.size());
        int m_locked = 0;

        if (m_task <= 0 || task_jacobian.cols() != n || task_velocity_command.size() != m_task ||
            qdot_lower_limits.size() != n || qdot_upper_limits.size() != n || qdot_previous.size() != n ||
            wheel_jacobian.cols() != n || wheel_lock_rhs.size() != m_wheel ||
            static_cast<int>(locked_targets.size()) != n_lock)
        {
            return false;
        }
        for (int i = 0; i < n_lock; ++i)
        {
            if (locked_jacobians[i].cols() != n || locked_targets[i].size() != locked_jacobians[i].rows())
                return false;
            m_locked += locked_jacobians[i].rows();
        }

        const int nvar = n + m_task;
        const int ncon = m_task + m_locked + m_wheel + m_contact + n;

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
            const int m_lock_i = locked_jacobians[i].rows();
            A.block(row, 0, m_lock_i, n) = locked_jacobians[i];
            l.segment(row, m_lock_i) = locked_targets[i];
            u.segment(row, m_lock_i) = locked_targets[i];
            row += m_lock_i;
        }

        A.block(row, 0, m_wheel, n) = wheel_jacobian;
        l.segment(row, m_wheel) = wheel_lock_rhs;
        u.segment(row, m_wheel) = wheel_lock_rhs;
        row += m_wheel;

        if (m_contact > 0)
        {
            A.block(row, 0, m_contact, n) = contact_jacobian;
            l.segment(row, m_contact).setZero();
            u.segment(row, m_contact).setZero();
            row += m_contact;
        }

        A.block(row, 0, n, n) = Eigen::MatrixXd::Identity(n, n);
        l.segment(row, n) = qdot_lower_limits;
        u.segment(row, n) = qdot_upper_limits;

        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(nvar, nvar);
        H.block(0, 0, n, n).diagonal().array() = 2.0 * (lambda_qdot_norm + lambda_qdot_smooth);
        H.block(n, n, m_task, m_task).diagonal().array() = 2.0;

        Eigen::VectorXd g = Eigen::VectorXd::Zero(nvar);
        g.head(n).array() = -2.0 * lambda_qdot_smooth * qdot_previous.array();

        if (!H.allFinite() || !g.allFinite() || !A.allFinite() || !l.allFinite() || !u.allFinite())
            return false;

        OsqpEigen::Solver solver;
        solver.settings()->setVerbosity(false);
        solver.settings()->setWarmStart(true);
        solver.data()->setNumberOfVariables(nvar);
        solver.data()->setNumberOfConstraints(ncon);

        if (!solver.data()->setHessianMatrix(denseToSparse(H, 1e-12)))
            return false;
        if (!solver.data()->setGradient(g))
            return false;
        if (!solver.data()->setLinearConstraintsMatrix(denseToSparse(A, 1e-12)))
            return false;
        if (!solver.data()->setLowerBound(l))
            return false;
        if (!solver.data()->setUpperBound(u))
            return false;
        if (!solver.initSolver())
            return false;
        if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
            return false;

        const Eigen::VectorXd solution = solver.getSolution();
        if (!solution.allFinite())
            return false;

        qdot_solution = solution.head(n);
        task_slack = solution.segment(n, m_task);
        return true;
    }

    // ============================
    // controlLoop
    // ============================
    void controlLoop()
    {
        // 1) Joint State Check
        if (q_.size() < 20)
        {
            if (iter_count_ % 100 == 0)
                RCLCPP_WARN(this->get_logger(), "Waiting for Joints... Current Size: %zu", q_.size());
            return;
        }

        const auto loop_start_time = std::chrono::steady_clock::now();

        // 2) FK + Jacobians (guard against invalid IMU quaternion)
        Eigen::Vector4d base_quat_normalized = base_quat_;
        if (!base_quat_normalized.allFinite() || base_quat_normalized.norm() < 1e-12)
        {
            if (iter_count_ % 100 == 0)
                RCLCPP_WARN(this->get_logger(), "Invalid base_quat detected in control loop. Falling back to identity.");
            base_quat_normalized << 0.0, 0.0, 0.0, 1.0;
            base_quat_ = base_quat_normalized;
        }
        else
        {
            base_quat_normalized.normalize();
            base_quat_ = base_quat_normalized;
        }

        // auto fk_result = computeFKAllAndJacobian(base_pos_, base_quat_normalized, q_);
        const auto fk_result = ecvt2_controller::computePinocchioFKAllAndJacobian(base_pos_, base_quat_normalized, q_);

        // 3) Capture desired on enable edge to avoid first-step jump
        if (capture_desired_on_next_loop_)
        {
            captureDesiredTargetsFromCurrentState(fk_result);
            q_dot_prev_.setZero();
            q_dot_prev_lower_.setZero();
            capture_desired_on_next_loop_ = false;
            RCLCPP_INFO(this->get_logger(), "Desired targets recaptured on enable edge.");
            return;
        }

        // 4) Holding Logic
        if (!control_enabled_)
        {
            captureDesiredTargetsFromCurrentState(fk_result);
            RCLCPP_INFO(this->get_logger(), "Desired targets captured from current pose.");

            control_enabled_ = true;
            return;
        }

        // 5) Publish EE Pose (debug)
        publishEePositionDebug(fk_result);

        // ---------------------------------------------------------
        // 6) Build full Jacobian (21 x 26)
        //   row layout:
        //   [0..5]   : base twist task rows (lin 3 + ang 3)
        //   [6..17]  : 4 feet contact rows (each 3D position)
        //   [18..20] : upper body task row (EE linear 3D)
        //
        //   col layout:
        //   [0..5]   : floating-base twist variables
        //   [6..21]  : 4 legs x 4 reduced joints
        //   [22..25] : arm 4 joints
        // ---------------------------------------------------------
        Eigen::MatrixXd full_jacobian = hqp::buildFullJacobian(fk_result);

        // Joint-bound conversion:
        // - qdot bound is computed from BOTH
        //   (a) position limits turned into velocity limits using dt
        //   (b) hard actuator velocity limits
        // - final qdot range is intersection of the two.
        //
        // Tuning knobs:
        // - kLoopDt: control period used for pos->vel conversion
        // - kPosLimitRateScale: larger => softer approach to position bounds
        // - qd_min_full_/qd_max_full_: hard speed caps
        // - q_min_full_safe_/q_max_full_safe_: safety-shrunk position limits
        const hqp::JointBoundData bound_data = hqp::computeJointBoundData(
            q_,
            model_to_hw_index_,
            q_min_full_safe_,
            q_max_full_safe_,
            qd_min_full_,
            qd_max_full_,
            kLoopDt,
            kPosLimitRateScale,
            0.5);
        const Vec26 &qdot_lb = bound_data.qdot_lower;
        const Vec26 &qdot_ub = bound_data.qdot_upper;
        publishArmJointDebug(bound_data);

        constexpr double lambda_qdot_norm = 1e-3;
        constexpr double lambda_qdot_smooth = 1e-4;

        const Eigen::MatrixXd task_jac_base_lin_upper = full_jacobian.block<3, 26>(0, 0);
        const Eigen::MatrixXd task_jac_base_ori_upper = full_jacobian.block<3, 26>(3, 0);
        const Eigen::MatrixXd task_jac_ee_upper = full_jacobian.block<3, 26>(18, 0);
        const Eigen::MatrixXd contact_jacobian_upper = full_jacobian.block<12, 26>(6, 0);

        const Eigen::MatrixXd task_jac_base_lin_lower = full_jacobian.block<3, 22>(0, 0);
        const Eigen::MatrixXd task_jac_base_ori_lower = full_jacobian.block<3, 22>(3, 0);
        const Eigen::MatrixXd contact_jacobian_lower = full_jacobian.block<12, 22>(6, 0);

        const Eigen::Matrix3d base_rotation_current = fk_result.T[0].block<3, 3>(0, 0);
        const Eigen::Vector3d base_pos_current = fk_result.T[0].block<3, 1>(0, 3);
        const Eigen::Vector3d base_pos_error = base_desired_ - base_pos_current;
        const Eigen::Vector3d base_ori_error = 0.5 * (base_rotation_current.col(0).cross(base_R_desired_.col(0)) +
                                                       base_rotation_current.col(1).cross(base_R_desired_.col(1)) +
                                                       base_rotation_current.col(2).cross(base_R_desired_.col(2)));

        const Eigen::Vector3d cmd_base_ori = 6.0 * base_ori_error;
        const Eigen::Vector3d cmd_base_lin = 4.0 * base_pos_error;
        const Eigen::Vector3d foot_error_fr = FR_desired_ - fk_result.T[1].block<3, 1>(0, 3);
        const Eigen::Vector3d foot_error_fl = FL_desired_ - fk_result.T[2].block<3, 1>(0, 3);
        const Eigen::Vector3d foot_error_rr = RR_desired_ - fk_result.T[3].block<3, 1>(0, 3);
        const Eigen::Vector3d foot_error_rl = RL_desired_ - fk_result.T[4].block<3, 1>(0, 3);

        const Eigen::VectorXd qdot_lower_limits = qdot_lb.head(kLowerDof);
        const Eigen::VectorXd qdot_upper_limits = qdot_ub.head(kLowerDof);
        Eigen::VectorXd qdot_wheel_l1 = Eigen::VectorXd::Zero(kLowerDof);
        Eigen::VectorXd qdot_lower_l2 = Eigen::VectorXd::Zero(kLowerDof);
        Eigen::VectorXd qdot_lower_l3 = Eigen::VectorXd::Zero(kLowerDof);
        Eigen::VectorXd qdot_lower_final = Eigen::VectorXd::Zero(kLowerDof);

        Eigen::Matrix3d R_base_wheel_target = Eigen::Matrix3d::Zero();
        R_base_wheel_target(0, 0) = -1.0;
        R_base_wheel_target(1, 2) = 1.0;
        R_base_wheel_target(2, 1) = 1.0;
        Eigen::Matrix3d R_world_wheel_target = fk_result.T[0].block<3, 3>(0, 0) * R_base_wheel_target;

        // 1) Lower-body L1: wheel steering task
        Eigen::MatrixXd wheel_jacobian = Eigen::MatrixXd::Zero(4, 22);
        wheel_jacobian.block<1, 10>(0, 0) = fk_result.J[1].block<1, 10>(5, 0);
        wheel_jacobian.block<1, 6>(1, 0) = fk_result.J[2].block<1, 6>(5, 0);
        wheel_jacobian.block<1, 4>(1, 10) = fk_result.J[2].block<1, 4>(5, 6);
        wheel_jacobian.block<1, 6>(2, 0) = fk_result.J[3].block<1, 6>(5, 0);
        wheel_jacobian.block<1, 4>(2, 14) = fk_result.J[3].block<1, 4>(5, 6);
        wheel_jacobian.block<1, 6>(3, 0) = fk_result.J[4].block<1, 6>(5, 0);
        wheel_jacobian.block<1, 4>(3, 18) = fk_result.J[4].block<1, 4>(5, 6);

        Eigen::Vector4d wheel_vel_cmd = Eigen::Vector4d::Zero();
        for (int i = 0; i < 4; ++i)
            wheel_vel_cmd(i) = hqp::orientationErrorAroundZ(fk_result.T[1 + i].block<3, 3>(0, 0), R_world_wheel_target);
        if (!wheel_vel_cmd.allFinite())
            wheel_vel_cmd.setZero();

        bool is_lower_l1_ok = hqp::solveHQPLowerLevel1WheelTask(
            wheel_jacobian, wheel_vel_cmd, qdot_lower_limits, qdot_upper_limits,
            q_dot_prev_lower_, lambda_qdot_norm, lambda_qdot_smooth, qdot_wheel_l1);
        if (!is_lower_l1_ok)
        {
            RCLCPP_WARN(this->get_logger(), "Lower-body HQP L1(wheel) failed, fallback to previous lower-body command");
            qdot_wheel_l1 = q_dot_prev_lower_;
        }

        // 2) Lower-body L2/L3: base ori -> (base pos + feet pos) with wheel/contact constraints
        const Eigen::VectorXd wheel_lock_rhs = wheel_jacobian * qdot_wheel_l1;
        Eigen::VectorXd lower_slack_l2 = Eigen::VectorXd::Zero(3);
        Eigen::VectorXd lower_slack_l3 = Eigen::VectorXd::Zero(15);
        const Eigen::VectorXd cmd_base_ori_vec = cmd_base_ori;
        Eigen::VectorXd cmd_base_and_feet = Eigen::VectorXd::Zero(15);
        cmd_base_and_feet.segment<3>(0) = cmd_base_lin;
        cmd_base_and_feet.segment<3>(3) = 0.5 * foot_error_fr;
        cmd_base_and_feet.segment<3>(6) = 0.5 * foot_error_fl;
        cmd_base_and_feet.segment<3>(9) = 0.5 * foot_error_rr;
        cmd_base_and_feet.segment<3>(12) = 0.5 * foot_error_rl;
        Eigen::MatrixXd task_jac_base_and_feet = Eigen::MatrixXd::Zero(15, kLowerDof);
        task_jac_base_and_feet.block<3, 22>(0, 0) = task_jac_base_lin_lower;
        task_jac_base_and_feet.block<12, 22>(3, 0) = full_jacobian.block<12, 22>(6, 0);

        const bool is_lower_l2_ok = solveLowerTaskWithLocks(
            task_jac_base_ori_lower, cmd_base_ori_vec,
            {}, {},
            wheel_jacobian, wheel_lock_rhs,
            contact_jacobian_lower,
            qdot_lower_limits, qdot_upper_limits,
            qdot_wheel_l1,
            lambda_qdot_norm, lambda_qdot_smooth,
            qdot_lower_l2, lower_slack_l2);

        bool is_lower_l3_ok = false;
        if (is_lower_l2_ok)
        {
            std::vector<Eigen::MatrixXd> locked_jacobians{task_jac_base_ori_lower};
            std::vector<Eigen::VectorXd> locked_targets{cmd_base_ori_vec - lower_slack_l2};
            is_lower_l3_ok = solveLowerTaskWithLocks(
                task_jac_base_and_feet, cmd_base_and_feet,
                locked_jacobians, locked_targets,
                wheel_jacobian, wheel_lock_rhs,
                contact_jacobian_lower,
                qdot_lower_limits, qdot_upper_limits,
                qdot_lower_l2,
                lambda_qdot_norm, lambda_qdot_smooth,
                qdot_lower_l3, lower_slack_l3);
        }

        if (!is_lower_l2_ok)
        {
            RCLCPP_WARN(this->get_logger(), "Lower-body HQP L2(base-ori) failed, fallback to L1(wheel)");
            qdot_lower_final = qdot_wheel_l1;
        }
        else if (!is_lower_l3_ok)
        {
            RCLCPP_WARN(this->get_logger(), "Lower-body HQP L3(base+feet) failed, fallback to L2(base-ori)");
            qdot_lower_final = qdot_lower_l2;
        }
        else
        {
            qdot_lower_final = qdot_lower_l3;
        }

        if (!qdot_lower_final.allFinite() || qdot_lower_final.cwiseAbs().maxCoeff() > 10.0)
        {
            RCLCPP_WARN(this->get_logger(), "Invalid lower-body HQP output detected. Falling back to L1 output.");
            qdot_lower_final = qdot_wheel_l1;
        }
        qdot_lower_final = qdot_lower_final.cwiseMax(qdot_lower_limits).cwiseMin(qdot_upper_limits);

        // lower에서 얻은 base twist(base frame)를 world frame으로 변환
        const Eigen::Matrix3d R_world_from_base = fk_result.T[0].block<3, 3>(0, 0);
        Eigen::VectorXd base_twist_world = Eigen::VectorXd::Zero(6);
        base_twist_world.segment<3>(0) = R_world_from_base * qdot_lower_final.segment<3>(0);
        base_twist_world.segment<3>(3) = R_world_from_base * qdot_lower_final.segment<3>(3);

        // 3) Upper-body: base twist -> ee tracking
        const double upper_base_weight = 1.0;
        const double upper_arm_weight = 0.1;
        const Eigen::Vector3d cmd_base_lin_from_lower = base_twist_world.segment<3>(0);
        const Eigen::Vector3d cmd_base_ori_from_lower = base_twist_world.segment<3>(3);
        const Eigen::Vector3d cmd_ee = 4.0 * (ee_desired_ - fk_result.T[5].block<3, 1>(0, 3));

        Eigen::VectorXd qdot_upper_l1 = q_dot_prev_;
        Eigen::VectorXd qdot_upper_l2 = q_dot_prev_;
        Eigen::VectorXd qdot_upper_l3 = q_dot_prev_;
        Eigen::VectorXd qdot_upper_final = q_dot_prev_;
        Eigen::Vector3d slack_upper_l1 = Eigen::Vector3d::Zero();
        Eigen::Vector3d slack_upper_l2 = Eigen::Vector3d::Zero();
        Eigen::Vector3d slack_upper_l3 = Eigen::Vector3d::Zero();

        const bool is_upper_l1_ok = hqp::solveHQPUpperLevel(
            task_jac_base_ori_upper, cmd_base_ori_from_lower,
            {}, {},
            contact_jacobian_upper, qdot_lb, qdot_ub,
            q_dot_prev_,
            lambda_qdot_norm, lambda_qdot_smooth,
            upper_base_weight, upper_arm_weight,
            qdot_upper_l1, slack_upper_l1);

        bool is_upper_l2_ok = false;
        bool is_upper_l3_ok = false;
        if (is_upper_l1_ok)
        {
            std::vector<Eigen::MatrixXd> locked_jacobians{task_jac_base_ori_upper};
            std::vector<Eigen::Vector3d> locked_targets{cmd_base_ori_from_lower - slack_upper_l1};
            is_upper_l2_ok = hqp::solveHQPUpperLevel(
                task_jac_base_lin_upper, cmd_base_lin_from_lower,
                locked_jacobians, locked_targets,
                contact_jacobian_upper, qdot_lb, qdot_ub,
                q_dot_prev_,
                lambda_qdot_norm, lambda_qdot_smooth,
                upper_base_weight, upper_arm_weight,
                qdot_upper_l2, slack_upper_l2);
        }

        if (is_upper_l2_ok)
        {
            std::vector<Eigen::MatrixXd> locked_jacobians{task_jac_base_ori_upper, task_jac_base_lin_upper};
            std::vector<Eigen::Vector3d> locked_targets{cmd_base_ori_from_lower - slack_upper_l1, cmd_base_lin_from_lower - slack_upper_l2};
            is_upper_l3_ok = hqp::solveHQPUpperLevel(
                task_jac_ee_upper, cmd_ee,
                locked_jacobians, locked_targets,
                contact_jacobian_upper, qdot_lb, qdot_ub,
                q_dot_prev_,
                lambda_qdot_norm, lambda_qdot_smooth,
                upper_base_weight, upper_arm_weight,
                qdot_upper_l3, slack_upper_l3);
        }

        if (!is_upper_l1_ok)
        {
            RCLCPP_WARN(this->get_logger(), "Upper-body HQP L1(base-ori twist) failed, fallback to previous upper command");
            qdot_upper_final = q_dot_prev_;
        }
        else if (!is_upper_l2_ok)
        {
            RCLCPP_WARN(this->get_logger(), "Upper-body HQP L2(base-lin twist) failed, fallback to L1");
            qdot_upper_final = qdot_upper_l1;
        }
        else if (!is_upper_l3_ok)
        {
            RCLCPP_WARN(this->get_logger(), "Upper-body HQP L3(ee) failed, fallback to L2");
            qdot_upper_final = qdot_upper_l2;
        }
        else
        {
            qdot_upper_final = qdot_upper_l3;
        }

        if (!qdot_upper_final.allFinite() || qdot_upper_final.cwiseAbs().maxCoeff() > 10.0)
        {
            RCLCPP_WARN(this->get_logger(), "Invalid upper-body HQP output detected. Using previous upper command.");
            qdot_upper_final = q_dot_prev_;
        }
        qdot_upper_final = qdot_upper_final.cwiseMax(qdot_lb).cwiseMin(qdot_ub);
        Eigen::VectorXd qdot_arm = qdot_upper_final.segment<4>(22);

        // Event-driven diagnostics:
        {
            const double wheel_residual_level1 = (wheel_jacobian * qdot_wheel_l1 - wheel_vel_cmd).norm();
            const double wheel_lock_residual = (wheel_jacobian * qdot_lower_final - wheel_lock_rhs).norm();
            const double base_ori_residual = (task_jac_base_ori_lower * qdot_lower_final - cmd_base_ori).norm();
            const double base_feet_residual = (task_jac_base_and_feet * qdot_lower_final - cmd_base_and_feet).norm();
            const double max_abs_cmd = qdot_lower_final.cwiseAbs().maxCoeff();
            const double min_margin_to_lb = (qdot_lower_final - qdot_lower_limits).minCoeff();
            const double min_margin_to_ub = (qdot_upper_limits - qdot_lower_final).minCoeff();
            const double min_margin_any = std::min(min_margin_to_lb, min_margin_to_ub);
            const bool is_near_bound = (min_margin_any < 1e-4);
            if (wheel_residual_level1 > 5e-3 || wheel_lock_residual > 5e-4 ||
                base_ori_residual > 0.1 || base_feet_residual > 0.2 ||
                is_near_bound || !is_lower_l1_ok || !is_lower_l2_ok || !is_lower_l3_ok)
            {
                RCLCPP_WARN(this->get_logger(),
                            "[diag-lower-v2] l1_ok=%d l2_ok=%d l3_ok=%d wheel_res_l1=%.4e wheel_lock_res=%.4e base_ori_res=%.4e base_feet_res=%.4e max_abs_cmd=%.4f is_near_bound=%d min_margin_any=%.4e",
                            is_lower_l1_ok ? 1 : 0, is_lower_l2_ok ? 1 : 0, is_lower_l3_ok ? 1 : 0,
                            wheel_residual_level1, wheel_lock_residual, base_ori_residual, base_feet_residual,
                            max_abs_cmd, is_near_bound ? 1 : 0, min_margin_any);
            }
        }

        // ---------------------------------------------------------
        // 12) Publish (keep your existing joint index mapping)
        // ---------------------------------------------------------
        velocity_cmd_pub_->publish(buildVelocityCommandMessage(qdot_lower_final, qdot_arm));
        updatePreviousCommands(qdot_lower_final, qdot_arm);

        // ---------------------------------------------------------
        // 13) CSV logging (unchanged - keep yours; optional add manip_speed etc)
        // ---------------------------------------------------------
        if (control_enabled_ && csv_open_)
        {
            Eigen::Vector3d link_positions[6];
            for (int i = 0; i < 6; ++i)
                link_positions[i] = fk_result.T[i].block<3, 1>(0, 3);

            Eigen::Matrix<double, 16, 1> leg_joint_positions;
            if (q_.size() >= 19)
            {
                leg_joint_positions << q_(0), q_(1), q_(2), q_(3), q_(5), q_(6), q_(7), q_(8),
                    q_(10), q_(11), q_(12), q_(13), q_(15), q_(16), q_(17), q_(18);
            }
            else
                leg_joint_positions.setZero();

            Eigen::Matrix<double, 4, 1> arm_joint_positions;
            if (q_.size() >= 24)
                arm_joint_positions = q_.segment<4>(20);
            else
                arm_joint_positions.setZero();

            Eigen::Matrix<double, 4, 1> h_log_values = Eigen::Matrix<double, 4, 1>::Zero();
            writeCsvRow(link_positions, leg_joint_positions, 0.0, arm_joint_positions, h_log_values, false, 0.0, 0.0);
        }

        // 14) Profiling
        publishLoopProfile(loop_start_time);
    }

    // ----- Subscribers / Publishers -----
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr base_pos_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr velocity_cmd_pub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr base_target_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr ee_target_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_array_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr vcmd_marker_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr ee_pos_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr com_pos_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr control_enable_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr arm_debug_pub_;

    // ----- Timer -----
    rclcpp::TimerBase::SharedPtr control_timer_;

    // ----- State Variables -----
    Eigen::Vector3d base_pos_ = Eigen::Vector3d::Zero();
    Eigen::Vector4d base_quat_ = Eigen::Vector4d(0, 0, 0, 1); // [x y z w]
    Eigen::VectorXd q_;
    Eigen::VectorXd qd_;
    Eigen::Vector3d FR_desired_, FL_desired_, RR_desired_, RL_desired_;
    Eigen::Vector3d ee_desired_, base_desired_;
    Eigen::Matrix3d base_R_desired_;
    std::array<int, kModelDof> model_to_hw_index_{};
    Vec26 q_min_full_safe_ = Vec26::Constant(-kInfBound);
    Vec26 q_max_full_safe_ = Vec26::Constant(kInfBound);
    Vec26 qd_min_full_ = Vec26::Constant(-kInfBound);
    Vec26 qd_max_full_ = Vec26::Constant(kInfBound);

    // prev for smoothing
    Eigen::VectorXd q_dot_prev_ = Eigen::VectorXd::Zero(kModelDof);
    Eigen::VectorXd q_dot_prev_lower_ = Eigen::VectorXd::Zero(kLowerDof);

    bool control_enabled_ = false;
    bool capture_desired_on_next_loop_ = false;
    bool base_target_received_ = false;
    bool ee_target_received_ = false;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<EcvtController>();

    rclcpp::on_shutdown([node]()
                        { node->closeCsv(); });

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
