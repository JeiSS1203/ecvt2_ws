#include "ecvt2_controller/kdl_utils.hpp"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <sstream>
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <chrono>  // steady_clock, duration
#include <cstdint> // uint64_t
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <ctime>
#include <cmath>
#include <OsqpEigen/OsqpEigen.h>

// --- 프로파일링용 ---
uint64_t iter_count_ = 0;
double ema_ms_ = 0.0; // 지수이동평균
double max_ms_ = 0.0; // 관측 최대
double deg2rad(double deg)
{
    return deg * M_PI / 180.0;
}
using namespace ecvt2_controller;

class EcvtController : public rclcpp::Node
{
public:
    EcvtController() : Node("ecvt_controller")
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

        // ----- Control loop timer -----
        // 100Hz → 0.01s
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
    // ----- CSV -----
    std::ofstream csv_;
    bool csv_open_ = false;
    std::string csv_path_;

    // enable 토글 감지용
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
        std::filesystem::create_directories(dir, ec); // 폴더 없으면 자동 생성

        // 파일명에 날짜/시간 붙이기
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
            << ",ARM0,ARM1,ARM2,ARM3" // << 추가
            << ",PASS1,PASS2"
            << ",h0,h1,h2,h3"        // << 추가 (joint-limit 활성도)
            << ",joint_limit_active" // << 추가 (0/1)
            << "\n";
        csv_.flush();
    }

    // pos[0..5] = base,FR,FL,RR,RL,EE
    // q_legs = 16 (FR0..3, FL0..3, RR0..3, RL0..3)
    void writeCsvRow(const Eigen::Vector3d pos[6],
                     const Eigen::Matrix<double, 16, 1> &q_legs,
                     double logdetA,
                     const Eigen::Matrix<double, 4, 1> &q_arm, // << 추가
                     const Eigen::Matrix<double, 4, 1> &h,     // << 추가
                     bool joint_limit_active,                  // << 추가
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
             << "," << q_legs(0) << "," << q_legs(1) << "," << q_legs(2) << "," << q_legs(3)
             << "," << q_legs(4) << "," << q_legs(5) << "," << q_legs(6) << "," << q_legs(7)
             << "," << q_legs(8) << "," << q_legs(9) << "," << q_legs(10) << "," << q_legs(11)
             << "," << q_legs(12) << "," << q_legs(13) << "," << q_legs(14) << "," << q_legs(15)
             << "," << logdetA
             << "," << q_arm(0) << "," << q_arm(1) << "," << q_arm(2) << "," << q_arm(3) // << ARM pos
             << "," << q_pass1 << "," << q_pass2
             << "," << h(0) << "," << h(1) << "," << h(2) << "," << h(3) // << h 활성도
             << "," << (joint_limit_active ? 1 : 0)                      // << 플래그
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
        // COM 현재 위치
        com_pos_(0) = msg->vector.x;
        com_pos_(1) = msg->vector.y;
        com_pos_(2) = msg->vector.z;

        // stamp Δt로 속도
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
        // orientation = [x, y, z, w] 순서 그대로 저장
        base_quat_(0) = msg->orientation.x;
        base_quat_(1) = msg->orientation.y;
        base_quat_(2) = msg->orientation.z;
        base_quat_(3) = msg->orientation.w;
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
            {
                qd_(i) = msg->velocity[i];
            }
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
        // 앞 3개: base_desired_
        base_desired_(0) = msg->data[0];
        base_desired_(1) = msg->data[1];
        base_desired_(2) = msg->data[2];

        // 뒤 3개: ZYX (yaw, pitch, roll)
        const double yaw = msg->data[3];
        const double pitch = msg->data[4];
        const double roll = msg->data[5];
        base_R_desired_ = eulerZYXToRot(yaw, pitch, roll);
    }

    void eeTargetCallback(const geometry_msgs::msg::Vector3::SharedPtr msg)
    {
        ee_desired_(0) = msg->x;
        ee_desired_(1) = msg->y;
        ee_desired_(2) = msg->z;
        // std::cout << "[EE Target] " << ee_desired_.transpose() << std::endl;
    }

    void publishEigenMarkersArray(const Eigen::MatrixXd &eigenvectors,
                                  const Eigen::VectorXd &eigenvalues,
                                  const Eigen::Vector3d &origin)
    {
        visualization_msgs::msg::MarkerArray arr;
        const std::string frame_id = "world";
        const rclcpp::Time stamp = this->now();

        // 보기 좋게 길이 스케일 (원하시는 대로 조정/클램프)
        auto length_of = [&](int i)
        {
            double l = eigenvalues(i);
            // 음수면 0으로(시각화용); 너무 크면 클램프
            l = std::max(0.0, std::min(l, 1.0));
            return 1.5 * l; // 시각화 gain
        };

        for (int i = 0; i < 3; ++i)
        {
            visualization_msgs::msg::Marker arrow;
            arrow.header.frame_id = frame_id;
            arrow.header.stamp = stamp;
            arrow.ns = "eigen";
            arrow.id = i;
            arrow.type = visualization_msgs::msg::Marker::ARROW;
            arrow.action = visualization_msgs::msg::Marker::ADD;
            arrow.frame_locked = false; // 필요시 true

            // 유효한 단위 쿼터니언(일부 RViz/버전에서 pose 무시되더라도 0쿼터니언은 피하기)
            arrow.pose.orientation.w = 1.0;

            // 시작점 = origin
            geometry_msgs::msg::Point p0;
            p0.x = origin.x();
            p0.y = origin.y();
            p0.z = origin.z();

            // 끝점 = origin + scale * eigenvector
            const double L = length_of(i);
            geometry_msgs::msg::Point p1;
            p1.x = origin.x() + L * eigenvectors(0, i);
            p1.y = origin.y() + L * eigenvectors(1, i);
            p1.z = origin.z() + L * eigenvectors(2, i);

            arrow.points.clear();
            arrow.points.push_back(p0);
            arrow.points.push_back(p1);

            // 색상
            if (i == 0)
            {
                arrow.color.r = 1.0;
                arrow.color.g = 0.0;
                arrow.color.b = 0.0;
            }
            else if (i == 1)
            {
                arrow.color.r = 0.0;
                arrow.color.g = 1.0;
                arrow.color.b = 0.0;
            }
            else
            {
                arrow.color.r = 0.0;
                arrow.color.g = 0.0;
                arrow.color.b = 1.0;
            }
            arrow.color.a = 1.0;

            // 스케일: ARROW는 x=몸통 지름, y=화살촉 지름, z=화살촉 길이
            arrow.scale.x = 0.02;
            arrow.scale.y = 0.06;
            arrow.scale.z = 0.10; // 이 값이 0이면 어떤 RViz에선 화살촉이 안 보일 수 있음

            // (선택) lifetime=0 → 항상 유지. 필요시 짧게 주면 프레임별로 리프레시 느낌
            // arrow.lifetime = rclcpp::Duration(0,0);

            arr.markers.push_back(arrow);
        }

        marker_array_pub_->publish(arr);
    }

    void publishVcmdMarker(const Eigen::Vector3d &origin_W,
                           const Eigen::Vector3d &v_W)
    {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = "world";
        m.header.stamp = this->now();
        m.ns = "v_cmd";
        m.id = 0;
        m.type = visualization_msgs::msg::Marker::ARROW;
        m.action = visualization_msgs::msg::Marker::ADD;

        // 포즈 쿼터니언은 단위로 (ARROW는 points로 방향/길이를 표현)
        m.pose.orientation.w = 1.0;

        // 시작점(origin)과 끝점(origin + gain * v)
        const double vis_gain = 1.0; // 화살표 길이 스케일 (원하면 조정)
        const double max_len = 2.0;  // 너무 길면 보기 힘드니 클램프
        Eigen::Vector3d tip = origin_W + vis_gain * v_W;
        if ((tip - origin_W).norm() > max_len)
        {
            tip = origin_W + (max_len * (tip - origin_W).normalized());
        }

        geometry_msgs::msg::Point p0, p1;
        p0.x = origin_W.x();
        p0.y = origin_W.y();
        p0.z = origin_W.z();
        p1.x = tip.x();
        p1.y = tip.y();
        p1.z = tip.z();
        m.points.clear();
        m.points.push_back(p0);
        m.points.push_back(p1);

        // 하얀색
        m.color.r = 1.0;
        m.color.g = 1.0;
        m.color.b = 1.0;
        m.color.a = 1.0;

        // ARROW 스케일: x=몸통 지름, y=화살촉 지름, z=화살촉 길이
        m.scale.x = 0.03; // shaft diameter
        m.scale.y = 0.09; // head diameter
        m.scale.z = 0.12; // head length

        // (선택) 수명 0 → 계속 유지. 빠르게 갱신되니 0으로 두면 됨
        // m.lifetime = rclcpp::Duration(0,0);

        vcmd_marker_pub_->publish(m);
    }

    void controlLoop()
    {
        // 1. Joint State Check (필수 안전장치)
        // Joint State가 아직 안 들어왔거나 개수가 부족하면 계산 중단 (메모리 오염 방지)
        if (q_.size() < 20)
        {
            if (iter_count_ % 100 == 0)
            {
                RCLCPP_WARN(this->get_logger(), "Waiting for Joints... Current Size: %ld", q_.size());
            }
            return;
        }

        const auto t_start = std::chrono::steady_clock::now();

        // 2. FK Calculation
        auto result = computeFKAllAndJacobian(base_pos_, base_quat_, q_);

        // 3. Holding Logic
        if (!control_enabled_)
        {
            ee_desired_ = result.T[5].block<3, 1>(0, 3);
            base_desired_ = result.T[0].block<3, 1>(0, 3);
            FR_desired_ = result.T[1].block<3, 1>(0, 3) - Eigen::Vector3d(0.0, 0.0, 0.05);
            FL_desired_ = result.T[2].block<3, 1>(0, 3) - Eigen::Vector3d(0.0, 0.0, 0.05);
            RR_desired_ = result.T[3].block<3, 1>(0, 3) - Eigen::Vector3d(0.0, 0.0, 0.05);
            RL_desired_ = result.T[4].block<3, 1>(0, 3) - Eigen::Vector3d(0.0, 0.0, 0.05);
            std::cout << "FR_desired_: " << FR_desired_.transpose() << std::endl;
            std::cout << "FL_desired_: " << FL_desired_.transpose() << std::endl;
            std::cout << "RR_desired_: " << RR_desired_.transpose() << std::endl;
            std::cout << "RL_desired_: " << RL_desired_.transpose() << std::endl;
            // openCsvIfNeeded();
            control_enabled_ = true;
            return;
        }

        // 4. Publish EE Pose
        geometry_msgs::msg::Vector3Stamped ee_pos_msg;
        ee_pos_msg.header.stamp = this->now();
        ee_pos_msg.header.frame_id = "world";
        ee_pos_msg.vector.x = result.T[5](0, 3);
        ee_pos_msg.vector.y = result.T[5](1, 3);
        ee_pos_msg.vector.z = result.T[5](2, 3);
        ee_pos_pub_->publish(ee_pos_msg);

        // ---------------------------------------------------------
        // 5. QP Data Setup
        // ---------------------------------------------------------
        // Jacobian Assembly
        // Index: 0~5(Base), 6~8(FR), 9~11(FL), 12~14(RR), 15~17(RL), 18~20(EE)
        Eigen::MatrixXd J_full = Eigen::MatrixXd::Zero(21, 26);
        J_full.block<6, 6>(0, 0) = result.J[0];
        J_full.block<3, 6>(6, 0) = result.J[1].block<3, 6>(0, 0);
        J_full.block<3, 4>(6, 6) = result.J[1].block<3, 4>(0, 6);
        J_full.block<3, 6>(9, 0) = result.J[2].block<3, 6>(0, 0);
        J_full.block<3, 4>(9, 10) = result.J[2].block<3, 4>(0, 6);
        J_full.block<3, 6>(12, 0) = result.J[3].block<3, 6>(0, 0);
        J_full.block<3, 4>(12, 14) = result.J[3].block<3, 4>(0, 6);
        J_full.block<3, 6>(15, 0) = result.J[4].block<3, 6>(0, 0);
        J_full.block<3, 4>(15, 18) = result.J[4].block<3, 4>(0, 6);
        J_full.block<3, 6>(18, 0) = result.J[5].block<3, 6>(0, 0);
        J_full.block<3, 4>(18, 22) = result.J[5].block<3, 4>(0, 6);

        // [튜닝 파라미터]
        double lambda_vel = 0.001; // 속도 최소화 가중치 (기존)
        double lambda_acc = 0.01;  // [추가] 가속도(변화량) 최소화 가중치 (이 값을 키우면 더 부드러워짐)

        // 1) Jacobians from J_full (already assembled 21x26)
        // Rows: 0~5 base (pos+ori), 6~8 FR, 9~11 FL, 12~14 RR, 15~17 RL, 18~20 EE

        Eigen::MatrixXd J_base = J_full.block<6, 26>(0, 0);
        Eigen::MatrixXd J_FR = J_full.block<3, 26>(6, 0);
        Eigen::MatrixXd J_FL = J_full.block<3, 26>(9, 0);
        Eigen::MatrixXd J_RR = J_full.block<3, 26>(12, 0);
        Eigen::MatrixXd J_RL = J_full.block<3, 26>(15, 0);
        Eigen::MatrixXd J_ee = J_full.block<3, 26>(18, 0);

        // 2) Errors -> desired task-space velocity (xdot)
        Eigen::Vector3d base_pos_cur = result.T[0].block<3, 1>(0, 3);
        Eigen::Vector3d e_base_pos = base_desired_ - base_pos_cur;

        // base orientation error (현재 네 방식 유지 가능)
        Eigen::Matrix3d R_cur = result.T[0].block<3, 3>(0, 0);
        Eigen::Vector3d e_base_ori =
            R_cur.col(0).cross(base_R_desired_.col(0)) +
            R_cur.col(1).cross(base_R_desired_.col(1)) +
            R_cur.col(2).cross(base_R_desired_.col(2));

        // feet errors
        Eigen::Vector3d FR_cur = result.T[1].block<3, 1>(0, 3);
        Eigen::Vector3d FL_cur = result.T[2].block<3, 1>(0, 3);
        Eigen::Vector3d RR_cur = result.T[3].block<3, 1>(0, 3);
        Eigen::Vector3d RL_cur = result.T[4].block<3, 1>(0, 3);
        Eigen::Vector3d e_FR = FR_desired_ - FR_cur;
        Eigen::Vector3d e_FL = FL_desired_ - FL_cur;
        Eigen::Vector3d e_RR = RR_desired_ - RR_cur;
        Eigen::Vector3d e_RL = RL_desired_ - RL_cur;

        // ee error
        Eigen::Vector3d ee_cur = result.T[5].block<3, 1>(0, 3);
        Eigen::Vector3d e_ee = ee_desired_ - ee_cur;

        // gains (튜닝)
        double Kp_base_pos = 5.0;
        double Kp_base_ori = 3.0;
        double Kp_foot = 0.1;
        double Kp_ee = 3.0;

        // desired twists/vels
        Eigen::VectorXd xdot_base(6);
        xdot_base.segment<3>(0) = Kp_base_pos * e_base_pos;
        xdot_base.segment<3>(3) = Kp_base_ori * e_base_ori;

        Eigen::Vector3d xdot_FR = Kp_foot * e_FR;
        Eigen::Vector3d xdot_FL = Kp_foot * e_FL;
        Eigen::Vector3d xdot_RR = Kp_foot * e_RR;
        Eigen::Vector3d xdot_RL = Kp_foot * e_RL;

        Eigen::Vector3d xdot_ee = Kp_ee * e_ee;

        // ===============================
        // [REPLACE START] Hierarchical QP (base pos > base ori > ee pos)
        // ===============================

        // ---------------------------------------------------------
        // [ADD HERE] QP bounds (qp_lb / qp_ub) for qdot
        // ---------------------------------------------------------
        double v_lim_base = 0.5;   // base velocity limit
        double safety_ratio = 0.9; // joint range safety (90%)

        Eigen::VectorXd q_min_full = Eigen::VectorXd::Constant(26, -1e10);
        Eigen::VectorXd q_max_full = Eigen::VectorXd::Constant(26, 1e10);
        Eigen::VectorXd qd_min_full = Eigen::VectorXd::Constant(26, -1e10);
        Eigen::VectorXd qd_max_full = Eigen::VectorXd::Constant(26, 1e10);

        // --- Position limits ---
        // FR (6~9)
        q_min_full.segment<4>(6) << deg2rad(-49), deg2rad(-58.4), deg2rad(-55.9), deg2rad(-180);
        q_max_full.segment<4>(6) << deg2rad(28.9), deg2rad(56.9), deg2rad(56.3), deg2rad(180);
        // FL (10~13)
        q_min_full.segment<4>(10) << deg2rad(-28.9), deg2rad(-58.4), deg2rad(-55.9), deg2rad(-180);
        q_max_full.segment<4>(10) << deg2rad(49), deg2rad(56.9), deg2rad(56.3), deg2rad(180);
        // RR (14~17)
        q_min_full.segment<4>(14) << deg2rad(-28.9), deg2rad(-56.9), deg2rad(-56.3), deg2rad(-180);
        q_max_full.segment<4>(14) << deg2rad(49), deg2rad(58.4), deg2rad(55.9), deg2rad(180);
        // RL (18~21)
        q_min_full.segment<4>(18) << deg2rad(-49), deg2rad(-56.9), deg2rad(-56.3), deg2rad(-180);
        q_max_full.segment<4>(18) << deg2rad(28.9), deg2rad(58.4), deg2rad(55.9), deg2rad(180);
        // Arm (22~25)
        q_min_full.segment<4>(22) << -M_PI / 2, 0.0, -2.0106193, 0.0;
        q_max_full.segment<4>(22) << M_PI / 2, 1.0472, 0.0, 1.65;

        // --- Velocity limits ---
        qd_min_full.segment<4>(6) << -0.35, -0.35, -0.35, -1.0;
        qd_max_full.segment<4>(6) << 0.35, 0.35, 0.35, 1.0;
        qd_min_full.segment<4>(10) << -0.35, -0.35, -0.35, -1.0;
        qd_max_full.segment<4>(10) << 0.35, 0.35, 0.35, 1.0;
        qd_min_full.segment<4>(14) << -0.35, -0.35, -0.35, -1.0;
        qd_max_full.segment<4>(14) << 0.35, 0.35, 0.35, 1.0;
        qd_min_full.segment<4>(18) << -0.35, -0.35, -0.35, -1.0;
        qd_max_full.segment<4>(18) << 0.35, 0.35, 0.35, 1.0;
        qd_min_full.segment<4>(22) << -1.0, -0.5, -0.5, -0.5;
        qd_max_full.segment<4>(22) << 1.0, 0.5, 0.5, 0.5;

        // safety shrink (6~25)
        for (int i = 6; i < 26; ++i)
        {
            double center = 0.5 * (q_min_full(i) + q_max_full(i));
            double half_range = 0.5 * (q_max_full(i) - q_min_full(i));
            double safe_half = half_range * safety_ratio;
            q_min_full(i) = center - safe_half;
            q_max_full(i) = center + safe_half;
        }

        // compute qp_lb/qp_ub
        Eigen::VectorXd qp_lb = Eigen::VectorXd::Zero(26);
        Eigen::VectorXd qp_ub = Eigen::VectorXd::Zero(26);

        // build mapping (same as your original)
        std::vector<int> q_map(26, -1);
        int hw_idx = 0;
        for (int leg = 0; leg < 4; ++leg)
        {
            for (int j = 0; j < 4; ++j)
                q_map[6 + leg * 4 + j] = hw_idx++;
            hw_idx++; // passive skip
        }
        for (int j = 0; j < 4; ++j)
            q_map[22 + j] = hw_idx++;

        for (int i = 0; i < 26; ++i)
        {
            if (i < 6)
            {
                qp_lb(i) = -v_lim_base;
                qp_ub(i) = v_lim_base;
            }
            else
            {
                int q_idx = q_map[i];
                if (q_idx < 0 || q_idx >= (int)q_.size())
                {
                    qp_lb(i) = -v_lim_base;
                    qp_ub(i) = v_lim_base;
                    continue;
                }

                double q_curr = q_(q_idx);
                double min_val = q_min_full(i);
                double max_val = q_max_full(i);

                double dt = 0.01; // control loop timestep (100Hz)
                double raw_lb = (min_val - q_curr)/(10.0*dt); // /dt 원하면 여기에 /dt
                double raw_ub = (max_val - q_curr)/(10.0*dt);

                qp_lb(i) = std::min(std::max(raw_lb, qd_min_full(i)), qd_max_full(i));
                qp_ub(i) = std::max(std::min(raw_ub, qd_max_full(i)), qd_min_full(i));
            }
        }

        // base Jacobian 분리 (이미 위에서 J_base, J_ee, xdot_base, xdot_ee 계산된 상태)
        Eigen::MatrixXd J_base_pos = J_base.block<3, 26>(0, 0); // base position rows
        Eigen::MatrixXd J_base_ori = J_base.block<3, 26>(3, 0); // base orientation rows
        Eigen::Vector3d xdot_base_pos = xdot_base.segment<3>(0);
        Eigen::Vector3d xdot_base_ori = xdot_base.segment<3>(3);

        // ---------------------------
        // HQP with slack variables
        // vars = [ qdot(26) ; s_pos(3) ; s_ori(3) ]  => 32 vars
        // constraints:
        //  (1) J_base_pos*qdot + s_pos = xdot_base_pos   (3 eq)
        //  (2) J_base_ori*qdot + s_ori = xdot_base_ori   (3 eq)
        //  (3) qp_lb <= qdot <= qp_ub                    (26 box)
        // objective:
        //  w_ee * ||J_ee*qdot - xdot_ee||^2
        // + rho_pos * ||s_pos||^2   (매우 크게)
        // + rho_ori * ||s_ori||^2   (매우 크게, pos보단 작게)
        // + lambda_vel||qdot||^2 + lambda_acc||qdot - qprev||^2
        // ---------------------------

        // [튜닝] 계층 강제 정도 (슬랙 패널티)
        // pos가 1순위이므로 rho_pos >> rho_ori >> w_ee 권장
        const double w_ee = 1.0;
        const double rho_pos = 1e6; // base pos 거의 hard
        const double rho_ori = 1e5; // base ori는 pos보다 한 단계 낮게

        const int n_qdot = 26;
        const int n_spos = 3;
        const int n_sori = 3;
        const int n_vars = n_qdot + n_spos + n_sori; // 32

        const int m_eq = 6;                     // base pos(3) + base ori(3)
        const int m_box = 26;                   // qdot bounds
        const int n_constraints = m_eq + m_box; // 32

        // ---------------------------
        // Build H, g (dense -> sparse)
        // ---------------------------
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n_vars, n_vars);
        Eigen::VectorXd g = Eigen::VectorXd::Zero(n_vars);

        // ee tracking term on qdot block
        H.block(0, 0, n_qdot, n_qdot).noalias() += w_ee * (J_ee.transpose() * J_ee);
        g.segment(0, n_qdot).noalias() += -w_ee * (J_ee.transpose() * xdot_ee);

        // regularization + smoothing on qdot
        H.block(0, 0, n_qdot, n_qdot).diagonal().array() += (lambda_vel + lambda_acc);
        g.segment(0, n_qdot).noalias() += -lambda_acc * q_dot_prev_;

        // slack penalties (pos > ori)
        H.block(n_qdot, n_qdot, n_spos, n_spos).diagonal().array() += rho_pos;
        H.block(n_qdot + n_spos, n_qdot + n_spos, n_sori, n_sori).diagonal().array() += rho_ori;

        // NaN check
        if (H.hasNaN() || g.hasNaN())
            return;

        Eigen::SparseMatrix<double> H_sparse = H.sparseView();

        // ---------------------------
        // Build A (sparse)
        // ---------------------------
        Eigen::SparseMatrix<double> A_sparse(n_constraints, n_vars);
        std::vector<Eigen::Triplet<double>> trip;
        trip.reserve((m_eq * n_qdot) + (m_eq) + (m_box));

        // (1) base pos eq: rows 0..2
        // [J_base_pos  I  0] * [qdot;spos;sori] = xdot_base_pos
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < n_qdot; ++c)
            {
                double v = J_base_pos(r, c);
                if (std::abs(v) > 1e-12)
                    trip.emplace_back(r, c, v);
            }
            trip.emplace_back(r, n_qdot + r, 1.0); // s_pos identity
        }

        // (2) base ori eq: rows 3..5
        // [J_base_ori  0  I] * [qdot;spos;sori] = xdot_base_ori
        for (int r = 0; r < 3; ++r)
        {
            int row = 3 + r;
            for (int c = 0; c < n_qdot; ++c)
            {
                double v = J_base_ori(r, c);
                if (std::abs(v) > 1e-12)
                    trip.emplace_back(row, c, v);
            }
            trip.emplace_back(row, n_qdot + n_spos + r, 1.0); // s_ori identity
        }

        // (3) box constraints on qdot: rows 6..31
        // [I 0 0] * vars = qdot  in [qp_lb, qp_ub]
        for (int i = 0; i < n_qdot; ++i)
        {
            trip.emplace_back(m_eq + i, i, 1.0);
        }

        A_sparse.setFromTriplets(trip.begin(), trip.end());

        // ---------------------------
        // Bounds lb/ub
        // ---------------------------
        Eigen::VectorXd lb = Eigen::VectorXd::Constant(n_constraints, -1e10);
        Eigen::VectorXd ub = Eigen::VectorXd::Constant(n_constraints, 1e10);

        // equality bounds
        lb.segment<3>(0) = xdot_base_pos;
        ub.segment<3>(0) = xdot_base_pos;

        lb.segment<3>(3) = xdot_base_ori;
        ub.segment<3>(3) = xdot_base_ori;

        // box bounds
        lb.segment(m_eq, m_box) = qp_lb; // 기존에 계산한 qp_lb (26)
        ub.segment(m_eq, m_box) = qp_ub; // 기존에 계산한 qp_ub (26)

        // ---------------------------
        // Solve OSQP
        // ---------------------------
        OsqpEigen::Solver solver_local;
        solver_local.settings()->setVerbosity(false);
        solver_local.settings()->setWarmStart(true);

        solver_local.data()->setNumberOfVariables(n_vars);
        solver_local.data()->setNumberOfConstraints(n_constraints);

        solver_local.data()->clearHessianMatrix();
        solver_local.data()->clearLinearConstraintsMatrix();

        bool set_ok = true;
        if (!solver_local.data()->setHessianMatrix(H_sparse))
            set_ok = false;
        if (!solver_local.data()->setGradient(g))
            set_ok = false;
        if (!solver_local.data()->setLinearConstraintsMatrix(A_sparse))
            set_ok = false;
        if (!solver_local.data()->setLowerBound(lb))
            set_ok = false;
        if (!solver_local.data()->setUpperBound(ub))
            set_ok = false;

        Eigen::VectorXd sol = Eigen::VectorXd::Zero(n_vars);
        Eigen::VectorXd q_dot_sol = Eigen::VectorXd::Zero(n_qdot);

        if (set_ok && solver_local.initSolver())
        {
            if (solver_local.solveProblem() == OsqpEigen::ErrorExitFlag::NoError)
            {
                sol = solver_local.getSolution();
                q_dot_sol = sol.head(n_qdot); // 최종 qdot
                q_dot_prev_ = q_dot_sol;      // smoothing 유지

                // (선택) 슬랙 모니터링: hierarchy가 얼마나 깨지는지 확인
                if (iter_count_ % 100 == 0)
                {
                    Eigen::Vector3d s_pos = sol.segment<3>(n_qdot);
                    Eigen::Vector3d s_ori = sol.segment<3>(n_qdot + n_spos);
                    std::cout << "[HQP slack] |s_pos|=" << s_pos.norm()
                              << " |s_ori|=" << s_ori.norm() << std::endl;
                }
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "HQP(OSQP) Solve Failed");
                return;
            }
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "HQP(OSQP) Setup Failed");
            return;
        }

        // 8. Mapping
        Eigen::VectorXd q_dot_sol_upper = Eigen::VectorXd::Zero(10);
        q_dot_sol_upper.segment<6>(0) = q_dot_sol.segment<6>(0);
        q_dot_sol_upper.segment<4>(6) = q_dot_sol.segment<4>(22);

        Eigen::VectorXd qdot_upper = q_dot_sol_upper;
        Eigen::VectorXd qdot_arm_cmd = qdot_upper.segment<4>(6);

        // BASE 속도를 내기위한 계산
        xdot_base.segment<3>(0) = result.T[0].block<3, 3>(0, 0) * qdot_upper.segment<3>(0); // base x,y,z velocity
        xdot_base.segment<3>(3) = result.T[0].block<3, 3>(0, 0) * qdot_upper.segment<3>(3); // base roll,pitch,yaw velocity

        Eigen::Vector3d FR_error = FR_desired_ - result.T[1].block<3, 1>(0, 3);
        Eigen::Vector3d FL_error = FL_desired_ - result.T[2].block<3, 1>(0, 3);
        Eigen::Vector3d RR_error = RR_desired_ - result.T[3].block<3, 1>(0, 3);
        Eigen::Vector3d RL_error = RL_desired_ - result.T[4].block<3, 1>(0, 3);

        Eigen::VectorXd xdot_lower = Eigen::VectorXd::Zero(18);
        xdot_lower.segment<6>(0) = xdot_base;
        xdot_lower.segment<3>(6) = 0.5 * FR_error;
        xdot_lower.segment<3>(9) = 0.5 * FL_error;
        xdot_lower.segment<3>(12) = 0.5 * RR_error;
        xdot_lower.segment<3>(15) = 0.5 * RL_error;

        ////////////////////////////////////////// 바퀴 조향 고정 용 nullspace projection /////////////////////////////////////////////////
        Eigen::Matrix3d R_base_wheel_desired = Eigen::Matrix3d::Zero();
        R_base_wheel_desired(0, 0) = -1.0;
        R_base_wheel_desired(1, 2) = 1.0;
        R_base_wheel_desired(2, 1) = 1.0;
        Eigen::MatrixXd R_world_wheel_desired = result.T[0].block<3, 3>(0, 0) * R_base_wheel_desired;

        Eigen::MatrixXd J_wheel = Eigen::MatrixXd::Zero(4, 22);
        J_wheel.block<1, 10>(0, 0) = result.J[1].block<1, 10>(5, 0); // FR
        J_wheel.block<1, 6>(1, 0) = result.J[2].block<1, 6>(5, 0);   // FL
        J_wheel.block<1, 4>(1, 10) = result.J[2].block<1, 4>(5, 6);  // FL
        J_wheel.block<1, 6>(2, 0) = result.J[3].block<1, 6>(5, 0);   // RR
        J_wheel.block<1, 4>(2, 14) = result.J[3].block<1, 4>(5, 6);  // RR
        J_wheel.block<1, 6>(3, 0) = result.J[4].block<1, 6>(5, 0);   // RL
        J_wheel.block<1, 4>(3, 18) = result.J[4].block<1, 4>(5, 6);  // RL

        Eigen::Vector4d wheel_vel_desired = Eigen::Vector4d::Zero();
        wheel_vel_desired(0) = (result.T[1].block<3, 1>(0, 0).cross(R_world_wheel_desired.block<3, 1>(0, 0)) +
                                result.T[1].block<3, 1>(0, 1).cross(R_world_wheel_desired.block<3, 1>(0, 1)) +
                                result.T[1].block<3, 1>(0, 2).cross(R_world_wheel_desired.block<3, 1>(0, 2)))(2);
        wheel_vel_desired(1) = (result.T[2].block<3, 1>(0, 0).cross(R_world_wheel_desired.block<3, 1>(0, 0)) +
                                result.T[2].block<3, 1>(0, 1).cross(R_world_wheel_desired.block<3, 1>(0, 1)) +
                                result.T[2].block<3, 1>(0, 2).cross(R_world_wheel_desired.block<3, 1>(0, 2)))(2);
        wheel_vel_desired(2) = (result.T[3].block<3, 1>(0, 0).cross(R_world_wheel_desired.block<3, 1>(0, 0)) +
                                result.T[3].block<3, 1>(0, 1).cross(R_world_wheel_desired.block<3, 1>(0, 1)) +
                                result.T[3].block<3, 1>(0, 2).cross(R_world_wheel_desired.block<3, 1>(0, 2)))(2);
        wheel_vel_desired(3) = (result.T[4].block<3, 1>(0, 0).cross(R_world_wheel_desired.block<3, 1>(0, 0)) +
                                result.T[4].block<3, 1>(0, 1).cross(R_world_wheel_desired.block<3, 1>(0, 1)) +
                                result.T[4].block<3, 1>(0, 2).cross(R_world_wheel_desired.block<3, 1>(0, 2)))(2);
        Eigen::VectorXd null_qd_desired = J_wheel.transpose() * (J_wheel * J_wheel.transpose()).inverse() * wheel_vel_desired;

        Eigen::MatrixXd J_lower = J_full.block<18, 22>(0, 0);
        Eigen::MatrixXd J_lower_pseudoInv = J_lower.transpose() * (J_lower * J_lower.transpose()).inverse();
        Eigen::VectorXd qdot_leg_cmd = J_lower_pseudoInv * xdot_lower + (Eigen::MatrixXd::Identity(22, 22) - J_lower_pseudoInv * J_lower) * null_qd_desired;

        // Publish
        sensor_msgs::msg::JointState vel_cmd_msg;
        vel_cmd_msg.header.stamp = this->now();
        vel_cmd_msg.name.resize(26);
        vel_cmd_msg.velocity.resize(26, 0.0);
        for (size_t i = 0; i < 26; ++i)
            vel_cmd_msg.name[i] = "joint_" + std::to_string(i);

        vel_cmd_msg.velocity[0] = qdot_leg_cmd(6);
        vel_cmd_msg.velocity[1] = qdot_leg_cmd(7);
        vel_cmd_msg.velocity[2] = qdot_leg_cmd(8);
        vel_cmd_msg.velocity[3] = qdot_leg_cmd(9);
        vel_cmd_msg.velocity[5] = qdot_leg_cmd(10);
        vel_cmd_msg.velocity[6] = qdot_leg_cmd(11);
        vel_cmd_msg.velocity[7] = qdot_leg_cmd(12);
        vel_cmd_msg.velocity[8] = qdot_leg_cmd(13);
        vel_cmd_msg.velocity[10] = qdot_leg_cmd(14);
        vel_cmd_msg.velocity[11] = qdot_leg_cmd(15);
        vel_cmd_msg.velocity[12] = qdot_leg_cmd(16);
        vel_cmd_msg.velocity[13] = qdot_leg_cmd(17);
        vel_cmd_msg.velocity[15] = qdot_leg_cmd(18);
        vel_cmd_msg.velocity[16] = qdot_leg_cmd(19);
        vel_cmd_msg.velocity[17] = qdot_leg_cmd(20);
        vel_cmd_msg.velocity[18] = qdot_leg_cmd(21);

        vel_cmd_msg.velocity[20] = qdot_arm_cmd(0);
        vel_cmd_msg.velocity[21] = qdot_arm_cmd(1);
        vel_cmd_msg.velocity[22] = qdot_arm_cmd(2);
        vel_cmd_msg.velocity[23] = qdot_arm_cmd(3);

        velocity_cmd_pub_->publish(vel_cmd_msg);

        // CSV Logging
        if (control_enabled_ && csv_open_)
        {
            Eigen::Vector3d pos_arr[6];
            for (int i = 0; i < 6; ++i)
                pos_arr[i] = result.T[i].block<3, 1>(0, 3);

            Eigen::Matrix<double, 16, 1> q_legs;
            if (q_.size() >= 19)
            {
                q_legs << q_(0), q_(1), q_(2), q_(3), q_(5), q_(6), q_(7), q_(8),
                    q_(10), q_(11), q_(12), q_(13), q_(15), q_(16), q_(17), q_(18);
            }
            else
                q_legs.setZero();

            Eigen::Matrix<double, 4, 1> q_arm_log;
            if (q_.size() >= 24)
                q_arm_log = q_.segment<4>(20);
            else
                q_arm_log.setZero();

            // 추가 로깅용 데이터
            // h (limit 활성도) = q - limit (여기서는 단순 0 처리하거나 필요시 계산)
            Eigen::Matrix<double, 4, 1> h_val = Eigen::Matrix<double, 4, 1>::Zero();

            writeCsvRow(pos_arr, q_legs, 0.0, q_arm_log, h_val, false, 0.0, 0.0);
        }

        // Profiling
        const auto t_end = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        iter_count_++;
        ema_ms_ = (iter_count_ == 1) ? ms : (0.05 * ms + 0.95 * ema_ms_);
        if (ms > max_ms_)
            max_ms_ = ms;
        if (iter_count_ % 100 == 0)
        {
            const double hz_est = (ms > 0.0) ? (1000.0 / ms) : 0.0;
            RCLCPP_INFO(this->get_logger(), "[profile] loop_ms=%.3f | ema_ms=%.3f | est_hz=%.1f", ms, ema_ms_, hz_est);
        }
    }

    // ----- Subscribers -----
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr base_pos_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr velocity_cmd_pub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr base_target_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr ee_target_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_array_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr vcmd_marker_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr ee_pos_pub_;
    // COM 토픽 구독자
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr com_pos_sub_;

    // ----- Timer -----
    rclcpp::TimerBase::SharedPtr control_timer_;

    // ----- State Variables -----.
    Eigen::Vector3d base_pos_ = Eigen::Vector3d::Zero();                // Base position
    Eigen::Vector4d base_quat_ = Eigen::Vector4d(0, 0, 0, 1);           // Base orientation [x,y,z,w]
    Eigen::VectorXd q_;                                                 // Joint positions
    Eigen::VectorXd qd_;                                                // Joint velocities
    Eigen::Vector3d FR_desired_, FL_desired_, RR_desired_, RL_desired_; // Desired foot positions
    Eigen::Vector3d ee_desired_, base_desired_;                         // Desired end-effector position(passive d1 joint)
    Eigen::Matrix3d base_R_desired_;                                    // Desired base orientation
    Eigen::VectorXd q_dot_prev_ = Eigen::VectorXd::Zero(26);

    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr control_enable_sub_;
    bool control_enabled_ = false;
    double prev_grad_gain_ = 0.0; // Slew-Limit 상태 유지
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<EcvtController>();

    // Ctrl+C (SIGINT) 등으로 종료 시, CSV를 안전하게 닫기
    rclcpp::on_shutdown([node]()
                        { node->closeCsv(); });

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}