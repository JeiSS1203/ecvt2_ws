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
#include <chrono>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <ctime>
#include <cmath>
#include <OsqpEigen/OsqpEigen.h>

// --- 프로파일링용 ---
uint64_t iter_count_ = 0;
double ema_ms_ = 0.0;
double max_ms_ = 0.0;

static double deg2rad(double deg) { return deg * M_PI / 180.0; }

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
                     const Eigen::Matrix<double, 16, 1> &q_legs,
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
             << "," << q_legs(0) << "," << q_legs(1) << "," << q_legs(2) << "," << q_legs(3)
             << "," << q_legs(4) << "," << q_legs(5) << "," << q_legs(6) << "," << q_legs(7)
             << "," << q_legs(8) << "," << q_legs(9) << "," << q_legs(10) << "," << q_legs(11)
             << "," << q_legs(12) << "," << q_legs(13) << "," << q_legs(14) << "," << q_legs(15)
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
    }

    void eeTargetCallback(const geometry_msgs::msg::Vector3::SharedPtr msg)
    {
        ee_desired_(0) = msg->x;
        ee_desired_(1) = msg->y;
        ee_desired_(2) = msg->z;
    }

    // ============================
    // Utilities (sparse conversion)
    // ============================
    static Eigen::SparseMatrix<double> denseToSparse(const Eigen::MatrixXd &M, double tol = 0.0)
    {
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve((size_t)(M.rows() * M.cols() / 4));
        for (int r = 0; r < M.rows(); ++r)
            for (int c = 0; c < M.cols(); ++c)
            {
                double v = M(r, c);
                if (std::abs(v) > tol)
                    triplets.emplace_back(r, c, v);
            }
        Eigen::SparseMatrix<double> S(M.rows(), M.cols());
        S.setFromTriplets(triplets.begin(), triplets.end());
        return S;
    }

    // ============================
    // HQP Level solver (weighted-norm + smoothing):
    //   min   ||w||^2
    //       + lambda_vel * qdot^T R qdot
    //       + lambda_acc * (qdot - qdot_prev)^T R (qdot - qdot_prev)
    // s.t.  J_task qdot + w = xdot
    //       (locks) J_fix qdot = rhs_fix
    //       Jc qdot = 0
    //       qd_lb <= qdot <= qd_ub
    //
    // R = blkdiag(r_base * I6, 1*I16, r_arm * I4)
    //   (base: 0..5, legs: 6..21, arm: 22..25)
    // ============================
    bool solveHQPUpperLevel(
        const Eigen::MatrixXd &J_task,
        const Eigen::Vector3d &xdot_task,
        const std::vector<Eigen::MatrixXd> &J_fix,
        const std::vector<Eigen::Vector3d> &rhs_fix,
        const Eigen::MatrixXd &Jc,
        const Eigen::VectorXd &qd_lb,
        const Eigen::VectorXd &qd_ub,
        const Eigen::VectorXd &qdot_prev, // (n)
        double lambda_vel,
        double lambda_acc,
        Eigen::VectorXd &qdot_out,
        Eigen::Vector3d &w_out)
    {
        const int n = (int)qd_lb.size();
        const int m_task = 3;
        const int mc = (int)Jc.rows();
        const int nfix = (int)J_fix.size();

        const int nvar = n + m_task;
        const int ncon = m_task + 3 * nfix + mc + n;

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(ncon, nvar);
        Eigen::VectorXd l = Eigen::VectorXd::Zero(ncon);
        Eigen::VectorXd u = Eigen::VectorXd::Zero(ncon);

        int row = 0;

        // (1) task with slack: J_task qdot + w = xdot_task
        A.block(row, 0, 3, n) = J_task;
        A.block(row, n, 3, 3) = Eigen::Matrix3d::Identity();
        l.segment<3>(row) = xdot_task;
        u.segment<3>(row) = xdot_task;
        row += 3;

        // (2) lock equalities (no slack): J_fix[i] qdot = rhs_fix[i]
        for (int i = 0; i < nfix; ++i)
        {
            A.block(row, 0, 3, n) = J_fix[i];
            l.segment<3>(row) = rhs_fix[i];
            u.segment<3>(row) = rhs_fix[i];
            row += 3;
        }

        // (3) contact equality: Jc qdot = 0
        if (mc > 0)
        {
            A.block(row, 0, mc, n) = Jc;
            l.segment(row, mc).setZero();
            u.segment(row, mc).setZero();
            row += mc;
        }

        // (4) box on qdot
        A.block(row, 0, n, n) = Eigen::MatrixXd::Identity(n, n);
        l.segment(row, n) = qd_lb;
        u.segment(row, n) = qd_ub;
        row += n;

        // ----------------------------
        // Weighted metric R (diagonal)
        // R = blkdiag(r_base*I6, 1*I16, r_arm*I4)
        // 가중치 작을수록 더 많이 움직임
        // ----------------------------
        double r_base = 1.0;
        double r_arm = 0.1;

        Eigen::VectorXd Rdiag = Eigen::VectorXd::Ones(n);
        if (n >= 6)
            Rdiag.segment<6>(0).setConstant(r_base);

        // arm indices are 22..25 for n=26
        if (n >= 26)
            Rdiag.segment<4>(22).setConstant(r_arm);
        else if (n > 22)
        {
            // safe for unexpected n (partial models)
            const int arm_len = std::min(4, n - 22);
            Rdiag.segment(22, arm_len).setConstant(r_arm);
        }

        // ----------------------------
        // Cost:
        //   ||w||^2
        // + lambda_vel * qdot^T R qdot
        // + lambda_acc * (qdot - qdot_prev)^T R (qdot - qdot_prev)
        //
        // Expand qdot part:
        //   (lambda_vel + lambda_acc) qdot^T R qdot - 2 lambda_acc qdot_prev^T R qdot + const
        // ----------------------------
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(nvar, nvar);
        H.block(0, 0, n, n).diagonal().array() =
            2.0 * (lambda_vel + lambda_acc) * Rdiag.array();
        H.block(n, n, 3, 3).diagonal().array() = 2.0; // ||w||^2

        Eigen::VectorXd g = Eigen::VectorXd::Zero(nvar);
        g.head(n).array() =
            -2.0 * lambda_acc * Rdiag.array() * qdot_prev.array();

        // (optional) sanity: avoid NaN/Inf
        if (!H.allFinite() || !g.allFinite() || !A.allFinite() || !l.allFinite() || !u.allFinite())
            return false;

        OsqpEigen::Solver solver;
        solver.settings()->setVerbosity(false);
        solver.settings()->setWarmStart(false);

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

        Eigen::VectorXd z = solver.getSolution();
        if (!z.allFinite())
            return false;
        qdot_out = z.head(n);
        w_out = z.segment<3>(n);
        return true;
    }


    bool solveHQPLowerLevel1Wheel22(
        const Eigen::MatrixXd &J_wheel_4x22,
        const Eigen::Vector4d &wheel_vel_desired,
        const Eigen::VectorXd &lb22,
        const Eigen::VectorXd &ub22,
        const Eigen::VectorXd &qdot_prev22,
        double lambda_vel,
        double lambda_acc,
        Eigen::VectorXd &qdot_out22)
    {
        const int nvar = 22;
        const int m_task = 4;
        const int ncon = m_task + nvar;
        if (J_wheel_4x22.rows() != 4 || J_wheel_4x22.cols() != nvar ||
            lb22.size() != nvar || ub22.size() != nvar || qdot_prev22.size() != nvar)
            return false;

        // L1: hard equality on wheel steering + box constraints.
        Eigen::MatrixXd H = 2.0 * (lambda_vel + lambda_acc) * Eigen::MatrixXd::Identity(nvar, nvar);
        Eigen::VectorXd g = -2.0 * lambda_acc * qdot_prev22;

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(ncon, nvar);
        Eigen::VectorXd l = Eigen::VectorXd::Zero(ncon);
        Eigen::VectorXd u = Eigen::VectorXd::Zero(ncon);
        int row = 0;

        A.block(row, 0, m_task, nvar) = J_wheel_4x22;
        l.segment(row, m_task) = wheel_vel_desired;
        u.segment(row, m_task) = wheel_vel_desired;
        row += m_task;

        A.block(row, 0, nvar, nvar) = Eigen::MatrixXd::Identity(nvar, nvar);
        l.segment(row, nvar) = lb22;
        u.segment(row, nvar) = ub22;
        for (int i = 0; i < nvar; ++i)
        {
            if (l(m_task + i) > u(m_task + i))
            {
                const double m = 0.5 * (l(m_task + i) + u(m_task + i));
                l(m_task + i) = m;
                u(m_task + i) = m;
            }
        }
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

        qdot_out22 = solver.getSolution();
        if (!qdot_out22.allFinite())
            return false;
        // Hard safety: enforce box exactly even under solver tolerance.
        for (int i = 0; i < nvar; ++i)
            qdot_out22(i) = std::min(std::max(qdot_out22(i), lb22(i)), ub22(i));
        return true;
    }

    bool solveHQPLowerLevel2Task22(
        const Eigen::MatrixXd &J_task_18x22,
        const Eigen::VectorXd &xdot_task_18,
        const Eigen::MatrixXd &J_wheel_4x22,
        const Eigen::VectorXd &qdot_wheel22,
        const Eigen::VectorXd &lb22,
        const Eigen::VectorXd &ub22,
        double lambda_task,
        double lambda_qdot_norm,
        double lambda_qdot_smooth,
        double w_base_task,
        double w_feet_task,
        Eigen::VectorXd &qdot_out22)
    {
        const int nvar = 22;
        const int m_task = 18;
        const int m_wheel = 4;
        const int ncon = m_wheel + nvar;
        if (J_task_18x22.rows() != m_task || J_task_18x22.cols() != nvar ||
            J_wheel_4x22.rows() != 4 || J_wheel_4x22.cols() != nvar ||
            xdot_task_18.size() != m_task || qdot_wheel22.size() != nvar ||
            lb22.size() != nvar || ub22.size() != nvar)
            return false;

        // L2: keep L1 wheel steering hard, optimize base+feet task in wheel nullspace.
        const Eigen::VectorXd wheel_lock_rhs = J_wheel_4x22 * qdot_wheel22;

        // Task weighting (same style as weighted cost in upper HQP):
        // rows 0..5   : base tracking weight
        // rows 6..17  : feet tracking weight
        Eigen::VectorXd Wdiag = Eigen::VectorXd::Ones(m_task);
        Wdiag.segment<6>(0).setConstant(w_base_task);
        Wdiag.segment<12>(6).setConstant(w_feet_task);
        const Eigen::DiagonalMatrix<double, Eigen::Dynamic> W(Wdiag);

        Eigen::MatrixXd H = 2.0 * (lambda_task * (J_task_18x22.transpose() * W * J_task_18x22) +
                                    (lambda_qdot_norm + lambda_qdot_smooth) * Eigen::MatrixXd::Identity(nvar, nvar));
        Eigen::VectorXd g = -2.0 * (lambda_task * (J_task_18x22.transpose() * W * xdot_task_18) +
                                     lambda_qdot_smooth * qdot_wheel22);

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(ncon, nvar);
        Eigen::VectorXd l = Eigen::VectorXd::Zero(ncon);
        Eigen::VectorXd u = Eigen::VectorXd::Zero(ncon);
        int row = 0;

        A.block(row, 0, m_wheel, nvar) = J_wheel_4x22;
        l.segment(row, m_wheel) = wheel_lock_rhs;
        u.segment(row, m_wheel) = wheel_lock_rhs;
        row += m_wheel;

        A.block(row, 0, nvar, nvar) = Eigen::MatrixXd::Identity(nvar, nvar);
        l.segment(row, nvar) = lb22;
        u.segment(row, nvar) = ub22;
        for (int i = 0; i < nvar; ++i)
        {
            if (l(m_wheel + i) > u(m_wheel + i))
            {
                const double m = 0.5 * (l(m_wheel + i) + u(m_wheel + i));
                l(m_wheel + i) = m;
                u(m_wheel + i) = m;
            }
        }
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

        qdot_out22 = solver.getSolution();
        if (!qdot_out22.allFinite())
            return false;
        // Hard safety: enforce box exactly even under solver tolerance.
        for (int i = 0; i < nvar; ++i)
            qdot_out22(i) = std::min(std::max(qdot_out22(i), lb22(i)), ub22(i));
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
                RCLCPP_WARN(this->get_logger(), "Waiting for Joints... Current Size: %ld", q_.size());
            return;
        }

        const auto t_start = std::chrono::steady_clock::now();

        // 2) FK + Jacobians (guard against invalid IMU quaternion)
        Eigen::Vector4d base_quat_safe = base_quat_;
        if (!base_quat_safe.allFinite() || base_quat_safe.norm() < 1e-12)
        {
            if (iter_count_ % 100 == 0)
                RCLCPP_WARN(this->get_logger(), "Invalid base_quat detected in control loop. Falling back to identity.");
            base_quat_safe << 0.0, 0.0, 0.0, 1.0;
            base_quat_ = base_quat_safe;
        }
        else
        {
            base_quat_safe.normalize();
            base_quat_ = base_quat_safe;
        }

        // auto result = computeFKAllAndJacobian(base_pos_, base_quat_safe, q_);
        auto result = ecvt2_controller::computePinocchioFKAllAndJacobian(base_pos_, base_quat_safe, q_);
        Eigen::Vector3d com2 = result.com;
        Eigen::Matrix<double, 3, 30> Jcom2 = result.com_jacobian;



        // 3) Holding Logic
        if (!control_enabled_)
        {
            ee_desired_ = result.T[5].block<3, 1>(0, 3);
            base_desired_ = result.T[0].block<3, 1>(0, 3);
            FR_desired_ = result.T[1].block<3, 1>(0, 3) - Eigen::Vector3d(0.0, 0.0, 0.02);
            FL_desired_ = result.T[2].block<3, 1>(0, 3) - Eigen::Vector3d(0.0, 0.0, 0.02);
            RR_desired_ = result.T[3].block<3, 1>(0, 3) - Eigen::Vector3d(0.0, 0.0, 0.02);
            RL_desired_ = result.T[4].block<3, 1>(0, 3) - Eigen::Vector3d(0.0, 0.0, 0.02);

            std::cout << "FR_desired_: " << FR_desired_.transpose() << std::endl;
            std::cout << "FL_desired_: " << FL_desired_.transpose() << std::endl;
            std::cout << "RR_desired_: " << RR_desired_.transpose() << std::endl;
            std::cout << "RL_desired_: " << RL_desired_.transpose() << std::endl;

            control_enabled_ = true;
            return;
        }

        // 4) Publish EE Pose (debug)
        {
            geometry_msgs::msg::Vector3Stamped ee_pos_msg;
            ee_pos_msg.header.stamp = this->now();
            ee_pos_msg.header.frame_id = "world";
            ee_pos_msg.vector.x = result.T[5](0, 3);
            ee_pos_msg.vector.y = result.T[5](1, 3);
            ee_pos_msg.vector.z = result.T[5](2, 3);
            ee_pos_pub_->publish(ee_pos_msg);
        }

        // ---------------------------------------------------------
        // 5) Build J_full (21 x 26)
        // ---------------------------------------------------------
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

        // ---------------------------------------------------------
        // 6) Manipulability direction (Task3 command)
        // ---------------------------------------------------------
        Eigen::Matrix3d A_man = computeManipulabilityMatrix(result, q_);
        Eigen::Vector3d grad = computeVMaxDirection(base_pos_, base_quat_safe, q_, A_man);

        // IMPORTANT: make a bounded command (otherwise it can be tiny/huge)
        double grad_norm = grad.norm();
        Eigen::Vector3d grad_dir = Eigen::Vector3d::Zero();
        if (grad_norm > 1e-9)
            grad_dir = grad / grad_norm;
        double v_man_max = 0.2;                        // 너가 쓰던 clamp
        double v_man = std::min(v_man_max, grad_norm); // magnitude
        Eigen::Vector3d v_cmd_base_lin = v_man * grad_dir;

        // ---------------------------------------------------------
        // 7) Tasks (Task1=BaseOri, Task2=EE, Task3=BaseLin(manip))
        // ---------------------------------------------------------
        Eigen::MatrixXd J1 = J_full.block<3, 26>(3, 0);  // base ori rows
        Eigen::MatrixXd J2 = J_full.block<3, 26>(18, 0); // ee pos rows
        Eigen::MatrixXd J3 = J_full.block<3, 26>(0, 0);  // base lin rows

        // Task2 EE command
        Eigen::Vector3d ee_error = ee_desired_ - result.T[5].block<3, 1>(0, 3);
        Eigen::Vector3d v_cmd_ee = 4.0 * ee_error;

        // Task1 base ori command
        Eigen::Matrix3d R_cur = result.T[0].block<3, 3>(0, 0);
        Eigen::Vector3d base_ori_err =
            R_cur.col(0).cross(base_R_desired_.col(0)) +
            R_cur.col(1).cross(base_R_desired_.col(1)) +
            R_cur.col(2).cross(base_R_desired_.col(2));
        Eigen::Vector3d v_cmd_base_ori = 3.0 * base_ori_err;

        // Contact equality: feet position fixed (12)
        Eigen::MatrixXd Jc = J_full.block<12, 26>(6, 0);

        // ---------------------------------------------------------
        // 8) Joint bounds (your logic)  **FIXED: /dt**
        // ---------------------------------------------------------
        const double dt = 0.01;
        const double v_lim_base = 0.5;
        const double safety_ratio = 0.9;

        Eigen::VectorXd q_min_full = Eigen::VectorXd::Constant(26, -1e10);
        Eigen::VectorXd q_max_full = Eigen::VectorXd::Constant(26, 1e10);
        Eigen::VectorXd qd_min_full = Eigen::VectorXd::Constant(26, -1e10);
        Eigen::VectorXd qd_max_full = Eigen::VectorXd::Constant(26, 1e10);

        // --- actuator position limits ---
        q_min_full.segment<4>(6) << deg2rad(-49), deg2rad(-58.4), deg2rad(-55.9), deg2rad(-180);
        q_max_full.segment<4>(6) << deg2rad(28.9), deg2rad(56.9), deg2rad(56.3), deg2rad(180);

        q_min_full.segment<4>(10) << deg2rad(-28.9), deg2rad(-58.4), deg2rad(-55.9), deg2rad(-180);
        q_max_full.segment<4>(10) << deg2rad(49), deg2rad(56.9), deg2rad(56.3), deg2rad(180);

        q_min_full.segment<4>(14) << deg2rad(-28.9), deg2rad(-56.9), deg2rad(-56.3), deg2rad(-180);
        q_max_full.segment<4>(14) << deg2rad(49), deg2rad(58.4), deg2rad(55.9), deg2rad(180);

        q_min_full.segment<4>(18) << deg2rad(-49), deg2rad(-56.9), deg2rad(-56.3), deg2rad(-180);
        q_max_full.segment<4>(18) << deg2rad(28.9), deg2rad(58.4), deg2rad(55.9), deg2rad(180);

        q_min_full.segment<4>(22) << -M_PI / 2, 0.0, -2.0106193, 0.0;
        q_max_full.segment<4>(22) << M_PI / 2, 1.0472, 0.0, 1.65;

        // --- velocity limits ---
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

        // shrink position range
        for (int i = 6; i < 26; ++i)
        {
            double center = 0.5 * (q_min_full(i) + q_max_full(i));
            double half_range = 0.5 * (q_max_full(i) - q_min_full(i));
            double safe_half = half_range * safety_ratio;
            q_min_full(i) = center - safe_half;
            q_max_full(i) = center + safe_half;
        }

        // model-index(26) -> hw-index(q_) mapping (your existing assumption)
        std::vector<int> q_map(26, -1);
        int hw_idx = 0;
        for (int leg = 0; leg < 4; ++leg)
        {
            for (int j = 0; j < 4; ++j)
                q_map[6 + leg * 4 + j] = hw_idx++;
            hw_idx++; // skip
        }
        for (int j = 0; j < 4; ++j)
            q_map[22 + j] = hw_idx++;

        // qdot bounds (26)
        Eigen::VectorXd qp_lb = Eigen::VectorXd::Zero(26);
        Eigen::VectorXd qp_ub = Eigen::VectorXd::Zero(26);
        Eigen::VectorXd raw_qp_lb = Eigen::VectorXd::Zero(26);
        Eigen::VectorXd raw_qp_ub = Eigen::VectorXd::Zero(26);

        for (int i = 0; i < 26; ++i)
        {
            if (i < 6)
            {
                qp_lb(i) = -v_lim_base;
                qp_ub(i) = v_lim_base;
                continue;
            }

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

            double raw_lb = (min_val - q_curr) / (dt*10.0);
            double raw_ub = (max_val - q_curr) / (dt*10.0);

            raw_qp_lb(i) = raw_lb;
            raw_qp_ub(i) = raw_ub;

            qp_lb(i) = std::min(std::max(raw_lb, qd_min_full(i)), qd_max_full(i));
            qp_ub(i) = std::max(std::min(raw_ub, qd_max_full(i)), qd_min_full(i));
        }

        // ----------------------------
        // Arm bounds publish (indices 22..25)
        //[ 0..3 ]   q_arm
        //[ 4..7 ]   qmin_arm
        //[ 8..11 ]  qmax_arm
        //[12..15 ]  raw_lb_arm      = (qmin - q) / dt
        //[16..19 ]  raw_ub_arm      = (qmax - q) / dt
        //[20..23 ]  qdmin_arm
        //[24..27 ]  qdmax_arm
        // ----------------------------
        {
            std_msgs::msg::Float64MultiArray msg;
            msg.data.resize(28);

            for (int k = 0; k < 4; ++k)
            {
                const int model_idx = 22 + k;        // arm joints in 26-dim model
                const int q_idx = q_map[model_idx]; // hw index into q_

                double q_curr = 0.0;
                if (q_idx >= 0 && q_idx < (int)q_.size())
                    q_curr = q_(q_idx);

                // q
                msg.data[0 + k]  = q_curr;

                // position limits
                msg.data[4 + k]  = q_min_full(model_idx);
                msg.data[8 + k]  = q_max_full(model_idx);

                // raw bounds
                msg.data[12 + k] = raw_qp_lb(model_idx);
                msg.data[16 + k] = raw_qp_ub(model_idx);

                // velocity limits
                msg.data[20 + k] = qd_min_full(model_idx);
                msg.data[24 + k] = qd_max_full(model_idx);
            }

            arm_debug_pub_->publish(msg);
        }

        // ---------------------------------------------------------
        // 9) HQP Solve (3 levels, 26-dim qdot)
        // ---------------------------------------------------------
        Eigen::VectorXd qdot1(26), qdot2(26), qdot3(26);
        Eigen::Vector3d w1, w2, w3;
        const double eps = 1e-4;
        double lambda_vel = 1e-3;
        double lambda_acc = 1e-4;

        // L1: BaseOri
        {
            std::vector<Eigen::MatrixXd> J_fix;
            std::vector<Eigen::Vector3d> rhs_fix;

            if (!solveHQPUpperLevel(J1, v_cmd_base_ori,
                               J_fix, rhs_fix,
                               Jc, qp_lb, qp_ub,
                               q_dot_prev_, lambda_vel, lambda_acc,
                               qdot1, w1))
            {
                RCLCPP_WARN(this->get_logger(), "HQP L1(BaseOri) failed");
                return;
            }
        }

        // L2: EE, lock L1
        {
            std::vector<Eigen::MatrixXd> J_fix;
            std::vector<Eigen::Vector3d> rhs_fix;

            J_fix.push_back(J1);
            rhs_fix.push_back(v_cmd_base_ori - w1);

            if (!solveHQPUpperLevel(J2, v_cmd_ee,
                               J_fix, rhs_fix,
                               Jc, qp_lb, qp_ub,
                               q_dot_prev_, lambda_vel, lambda_acc,
                               qdot2, w2))
            {
                RCLCPP_WARN(this->get_logger(), "HQP L2(EE) failed");
                return;
            }
        }

        // L3: BaseLin(manip), lock L1~L2
        {
            std::vector<Eigen::MatrixXd> J_fix;
            std::vector<Eigen::Vector3d> rhs_fix;

            J_fix.push_back(J1);
            rhs_fix.push_back(v_cmd_base_ori - w1);
            J_fix.push_back(J2);
            rhs_fix.push_back(v_cmd_ee - w2);

            if (!solveHQPUpperLevel(J3, v_cmd_base_lin,
                               J_fix, rhs_fix,
                               Jc, qp_lb, qp_ub,
                               q_dot_prev_, lambda_vel, lambda_acc,
                               qdot3, w3))
            {
                RCLCPP_WARN(this->get_logger(), "HQP L3(BaseLin) failed");
                return;
            }
        }

        Eigen::VectorXd qdot_final = qdot3;
        if (!qdot_final.allFinite() || qdot_final.cwiseAbs().maxCoeff() > 10.0)
        {
            RCLCPP_WARN(this->get_logger(), "Invalid/oversized upper-body HQP output. Falling back to previous command.");
            qdot_final = q_dot_prev_;
        }
        for (int i = 0; i < 26; ++i)
            qdot_final(i) = std::min(std::max(qdot_final(i), qp_lb(i)), qp_ub(i));

        // ---------------------------------------------------------
        // 11) Lower-body 2-stage QP (22 vars)
        //   (1) wheel steering hard task + bounds + norm/smoothing
        //   (2) base(world)+feet tracking in wheel nullspace + bounds
        // ---------------------------------------------------------
        Eigen::VectorXd qdot_arm4 = qdot_final.segment<4>(22);

        const Eigen::Matrix3d R_wb = result.T[0].block<3, 3>(0, 0);
        Eigen::VectorXd xdot_base_world6 = Eigen::VectorXd::Zero(6);
        xdot_base_world6.segment<3>(0) = R_wb * qdot3.segment<3>(0);
        xdot_base_world6.segment<3>(3) = R_wb * qdot3.segment<3>(3);

        Eigen::Vector3d FR_error = FR_desired_ - result.T[1].block<3, 1>(0, 3);
        Eigen::Vector3d FL_error = FL_desired_ - result.T[2].block<3, 1>(0, 3);
        Eigen::Vector3d RR_error = RR_desired_ - result.T[3].block<3, 1>(0, 3);
        Eigen::Vector3d RL_error = RL_desired_ - result.T[4].block<3, 1>(0, 3);

        Eigen::VectorXd xdot_task18 = Eigen::VectorXd::Zero(18);
        xdot_task18.segment<6>(0) = xdot_base_world6;
        xdot_task18.segment<3>(6) = 0.5 * FR_error;
        xdot_task18.segment<3>(9) = 0.5 * FL_error;
        xdot_task18.segment<3>(12) = 0.5 * RR_error;
        xdot_task18.segment<3>(15) = 0.5 * RL_error;

        Eigen::MatrixXd J_task18x22 = J_full.block<18, 22>(0, 0);
        // Task weighting for L2 cost (applied inside solver):
        //   base rows (0..5): nominal weight
        //   feet rows (6..17): stronger weight for safer ground-contact behavior
        const double w_base_task = 1.0;
        const double w_feet_task = 4.0;
        const Eigen::VectorXd lb22 = qp_lb.head(22);
        const Eigen::VectorXd ub22 = qp_ub.head(22);
        Eigen::VectorXd qdot_wheel22 = Eigen::VectorXd::Zero(22);
        Eigen::VectorXd qdot_lower22 = Eigen::VectorXd::Zero(22);

        Eigen::Matrix3d R_base_wheel_desired = Eigen::Matrix3d::Zero();
        R_base_wheel_desired(0, 0) = -1.0;
        R_base_wheel_desired(1, 2) = 1.0;
        R_base_wheel_desired(2, 1) = 1.0;
        Eigen::Matrix3d R_world_wheel_desired = result.T[0].block<3, 3>(0, 0) * R_base_wheel_desired;

        Eigen::MatrixXd J_wheel = Eigen::MatrixXd::Zero(4, 22);
        J_wheel.block<1, 10>(0, 0) = result.J[1].block<1, 10>(5, 0);
        J_wheel.block<1, 6>(1, 0) = result.J[2].block<1, 6>(5, 0);
        J_wheel.block<1, 4>(1, 10) = result.J[2].block<1, 4>(5, 6);
        J_wheel.block<1, 6>(2, 0) = result.J[3].block<1, 6>(5, 0);
        J_wheel.block<1, 4>(2, 14) = result.J[3].block<1, 4>(5, 6);
        J_wheel.block<1, 6>(3, 0) = result.J[4].block<1, 6>(5, 0);
        J_wheel.block<1, 4>(3, 18) = result.J[4].block<1, 4>(5, 6);

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
        if (!wheel_vel_desired.allFinite())
            wheel_vel_desired.setZero();

        bool lower_l1_ok = solveHQPLowerLevel1Wheel22(J_wheel, wheel_vel_desired, lb22, ub22,
                                                           q_dot_prev_lower22_, lambda_vel, lambda_acc, qdot_wheel22);
        if (!lower_l1_ok)
        {
            RCLCPP_WARN(this->get_logger(), "Lower-body HQP L1(wheel) failed, fallback to previous lower-body command");
            qdot_wheel22 = q_dot_prev_lower22_;
        }

        const double lambda_task = 1.0;
        const double lambda_lower_norm = 1e-4;
        const double lambda_lower_smooth = 1e-4;
        bool lower_l2_ok = solveHQPLowerLevel2Task22(J_task18x22, xdot_task18, J_wheel, qdot_wheel22, lb22, ub22,
                                                          lambda_task, lambda_lower_norm, lambda_lower_smooth,
                                                          w_base_task, w_feet_task, qdot_lower22);
        if (!lower_l2_ok)
        {
            RCLCPP_WARN(this->get_logger(), "Lower-body HQP L2(task) failed, fallback to L1(wheel)");
            qdot_lower22 = qdot_wheel22;
        }

        // Runtime safety gate against numeric blow-up (sim reset/glitch).
        if (!qdot_wheel22.allFinite())
        {
            RCLCPP_WARN(this->get_logger(), "Invalid L1 output detected. Using zero lower-body command.");
            qdot_wheel22.setZero();
        }
        if (!qdot_lower22.allFinite() || qdot_lower22.cwiseAbs().maxCoeff() > 10.0)
        {
            RCLCPP_WARN(this->get_logger(), "Invalid/oversized L2 output detected. Falling back to L1 output.");
            qdot_lower22 = qdot_wheel22;
        }
        // Final hard clamp before publish.
        for (int i = 0; i < 22; ++i)
            qdot_lower22(i) = std::min(std::max(qdot_lower22(i), lb22(i)), ub22(i));

        // Event-driven diagnostics to catch blow-up conditions near wheel-control failure.
        {
            // L1 (wheel hard task) quality.
            const double wheel_residual_l1 = (J_wheel * qdot_wheel22 - wheel_vel_desired).norm();
            // L2 final quality under L1 lock.
            const double wheel_residual_l2 = (J_wheel * qdot_lower22 - wheel_vel_desired).norm();
            const double wheel_lock_residual = (J_wheel * qdot_lower22 - J_wheel * qdot_wheel22).norm();
            const double task_residual_l2 = (J_task18x22 * qdot_lower22 - xdot_task18).norm();
            const double max_cmd = qdot_lower22.cwiseAbs().maxCoeff();
            const double min_margin_lb = (qdot_lower22 - lb22).minCoeff();
            const double min_margin_ub = (ub22 - qdot_lower22).minCoeff();
            const double min_margin = std::min(min_margin_lb, min_margin_ub);
            const double feet_err =
                FR_error.norm() + FL_error.norm() + RR_error.norm() + RL_error.norm();

            const bool near_bound = (min_margin < 1e-4);
            if (wheel_residual_l1 > 5e-3 || wheel_lock_residual > 5e-4 || task_residual_l2 > 0.2 ||
                near_bound || !lower_l1_ok || !lower_l2_ok)
            {
                RCLCPP_WARN(this->get_logger(),
                            "[diag-lower-v2] l1_ok=%d l2_ok=%d wheel_res_l1=%.4e wheel_res_l2=%.4e wheel_lock_res=%.4e task_res_l2=%.4e max_cmd=%.4f near_bound=%d min_margin=%.4e feet_err_sum=%.4f",
                            lower_l1_ok ? 1 : 0, lower_l2_ok ? 1 : 0,
                            wheel_residual_l1, wheel_residual_l2, wheel_lock_residual, task_residual_l2,
                            max_cmd, near_bound ? 1 : 0, min_margin, feet_err);
            }
        }

        // ---------------------------------------------------------
        // 12) Publish (keep your existing joint index mapping)
        // ---------------------------------------------------------
        sensor_msgs::msg::JointState vel_cmd_msg;
        vel_cmd_msg.header.stamp = this->now();
        vel_cmd_msg.name.resize(26);
        vel_cmd_msg.velocity.resize(26, 0.0);
        for (size_t i = 0; i < 26; ++i)
            vel_cmd_msg.name[i] = "joint_" + std::to_string(i);

        // legs (from qdot_lower22)
        vel_cmd_msg.velocity[0] = qdot_lower22(6);
        vel_cmd_msg.velocity[1] = qdot_lower22(7);
        vel_cmd_msg.velocity[2] = qdot_lower22(8);
        vel_cmd_msg.velocity[3] = qdot_lower22(9);

        vel_cmd_msg.velocity[5] = qdot_lower22(10);
        vel_cmd_msg.velocity[6] = qdot_lower22(11);
        vel_cmd_msg.velocity[7] = qdot_lower22(12);
        vel_cmd_msg.velocity[8] = qdot_lower22(13);

        vel_cmd_msg.velocity[10] = qdot_lower22(14);
        vel_cmd_msg.velocity[11] = qdot_lower22(15);
        vel_cmd_msg.velocity[12] = qdot_lower22(16);
        vel_cmd_msg.velocity[13] = qdot_lower22(17);

        vel_cmd_msg.velocity[15] = qdot_lower22(18);
        vel_cmd_msg.velocity[16] = qdot_lower22(19);
        vel_cmd_msg.velocity[17] = qdot_lower22(20);
        vel_cmd_msg.velocity[18] = qdot_lower22(21);

        // arm (from HQP final)
        vel_cmd_msg.velocity[20] = qdot_arm4(0);
        vel_cmd_msg.velocity[21] = qdot_arm4(1);
        vel_cmd_msg.velocity[22] = qdot_arm4(2);
        vel_cmd_msg.velocity[23] = qdot_arm4(3);

        velocity_cmd_pub_->publish(vel_cmd_msg);

        Eigen::VectorXd qdot_cmd26 = Eigen::VectorXd::Zero(26);
        qdot_cmd26.head<22>() = qdot_lower22;
        qdot_cmd26.segment<4>(22) = qdot_arm4;
        q_dot_prev_ = qdot_cmd26;
        q_dot_prev_lower22_ = qdot_lower22;

        // ---------------------------------------------------------
        // 13) CSV logging (unchanged - keep yours; optional add v_man etc)
        // ---------------------------------------------------------
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

            Eigen::Matrix<double, 4, 1> h_val = Eigen::Matrix<double, 4, 1>::Zero();
            writeCsvRow(pos_arr, q_legs, 0.0, q_arm_log, h_val, false, 0.0, 0.0);
        }

        // 14) Profiling
        const auto t_end = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        iter_count_++;
        ema_ms_ = (iter_count_ == 1) ? ms : (0.05 * ms + 0.95 * ema_ms_);
        // if (ms > max_ms_)
        //     max_ms_ = ms;
        // RCLCPP_INFO(this->get_logger(), "[loop-time] loop_ms=%.3f", ms);
        if (iter_count_ % 100 == 0)
        {
            const double hz_est = (ms > 0.0) ? (1000.0 / ms) : 0.0;
            RCLCPP_INFO(this->get_logger(), "[profile] loop_ms=%.3f | ema_ms=%.3f | est_hz=%.1f",
                        ms, ema_ms_, hz_est);
        }
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

    // prev for smoothing
    Eigen::VectorXd q_dot_prev_ = Eigen::VectorXd::Zero(26);
    Eigen::VectorXd q_dot_prev_lower22_ = Eigen::VectorXd::Zero(22);

    bool control_enabled_ = false;
    double prev_grad_gain_ = 0.0;
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

// 하부체 base tracking -> wheel control 순서인 경우
// #include "ecvt2_controller/kdl_utils.hpp"
// #include "rclcpp/rclcpp.hpp"
// #include "geometry_msgs/msg/vector3_stamped.hpp"
// #include "sensor_msgs/msg/imu.hpp"
// #include "sensor_msgs/msg/joint_state.hpp"
// #include <Eigen/Dense>
// #include <Eigen/Geometry>
// #include <iostream>
// #include <sstream>
// #include "std_msgs/msg/bool.hpp"
// #include "std_msgs/msg/float64_multi_array.hpp"
// #include "visualization_msgs/msg/marker.hpp"
// #include "visualization_msgs/msg/marker_array.hpp"
// #include <chrono>
// #include <cstdint>
// #include <fstream>
// #include <filesystem>
// #include <iomanip>
// #include <ctime>
// #include <cmath>
// #include <OsqpEigen/OsqpEigen.h>

// // --- 프로파일링용 ---
// uint64_t iter_count_ = 0;
// double ema_ms_ = 0.0;
// double max_ms_ = 0.0;

// static double deg2rad(double deg) { return deg * M_PI / 180.0; }

// using namespace ecvt2_controller;

// class EcvtController : public rclcpp::Node
// {
// public:
//     EcvtController() : Node("ecvt_controller")
//     {
//         // ----- Subscribers -----
//         base_pos_sub_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
//             "/hanyang/base_pos", 10,
//             std::bind(&EcvtController::basePosCallback, this, std::placeholders::_1));

//         imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
//             "/hanyang/imu", 10,
//             std::bind(&EcvtController::imuCallback, this, std::placeholders::_1));

//         joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
//             "/hanyang/joint_states", 10,
//             std::bind(&EcvtController::jointStateCallback, this, std::placeholders::_1));

//         // ----- Publishers -----
//         velocity_cmd_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
//             "/hanyang/velocity_cmd", 10);

//         com_pos_sub_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
//             "/hanyang/com_pos", 10,
//             std::bind(&EcvtController::comPosCallback, this, std::placeholders::_1));

//         control_enable_sub_ = this->create_subscription<std_msgs::msg::Bool>(
//             "/hanyang/control_enable", 10,
//             [this](const std_msgs::msg::Bool::SharedPtr msg)
//             {
//                 prev_control_enabled_ = control_enabled_;
//                 control_enabled_ = msg->data;
//                 RCLCPP_INFO(this->get_logger(), "control_enabled: %s", control_enabled_ ? "true" : "false");

//                 // 상승엣지에서 CSV 오픈
//                 if (!prev_control_enabled_ && control_enabled_)
//                 {
//                     openCsvIfNeeded();
//                 }
//             });

//         base_target_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
//             "/hanyang/base_target_zyx", 10,
//             std::bind(&EcvtController::baseTargetCallback, this, std::placeholders::_1));

//         ee_target_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
//             "/hanyang/ee_target", 10,
//             std::bind(&EcvtController::eeTargetCallback, this, std::placeholders::_1));

//         marker_array_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
//             "eigen_markers", rclcpp::QoS(1).transient_local().reliable());

//         vcmd_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
//             "v_cmd_marker", 10);

//         ee_pos_pub_ = this->create_publisher<geometry_msgs::msg::Vector3Stamped>(
//             "/hanyang/ee_pos", 10);

//         // ----- Control loop timer -----
//         control_timer_ = this->create_wall_timer(
//             std::chrono::milliseconds(10),
//             std::bind(&EcvtController::controlLoop, this));

//         RCLCPP_INFO(this->get_logger(), "EcvtController initialized with 100Hz loop");

//         // 초기 목표 위치 설정
//         FR_desired_ = Eigen::Vector3d(2.85, -0.92, 0.65);
//         FL_desired_ = Eigen::Vector3d(2.85, 0.92, 0.65);
//         RR_desired_ = Eigen::Vector3d(-3.65, -1.06, 0.65);
//         RL_desired_ = Eigen::Vector3d(-3.65, 1.06, 0.65);
//         ee_desired_ = Eigen::Vector3d(7.0, 0.0, 5.0);
//         base_desired_ = Eigen::Vector3d(-0.25, -0.012, 0.14);
//         base_R_desired_ = Eigen::Matrix3d::Identity();
//     }

//     void closeCsv()
//     {
//         if (csv_open_)
//         {
//             RCLCPP_INFO(this->get_logger(), "Closing CSV file...");
//             csv_.flush();
//             csv_.close();
//             csv_open_ = false;
//         }
//     }

// private:
//     // ----- CSV -----
//     std::ofstream csv_;
//     bool csv_open_ = false;
//     std::string csv_path_;
//     bool prev_control_enabled_ = false;

//     // ----- COM pose/velocity (from /hanyang/com_pos) -----
//     Eigen::Vector3d com_pos_ = Eigen::Vector3d::Zero();
//     Eigen::Vector3d com_pos_prev_ = Eigen::Vector3d::Zero();
//     Eigen::Vector3d com_vel_ = Eigen::Vector3d::Zero();
//     rclcpp::Time com_stamp_prev_;
//     bool have_com_prev_ = false;

//     void openCsvIfNeeded()
//     {
//         if (csv_open_)
//             return;

//         const std::string dir = "/home/harco/base_opt_csv/csv";
//         std::error_code ec;
//         std::filesystem::create_directories(dir, ec);

//         auto now = this->now();
//         std::time_t t = now.seconds();
//         std::tm tm{};
//         localtime_r(&t, &tm);
//         char name[64];
//         std::strftime(name, sizeof(name), "ecvt_log_%Y%m%d_%H%M%S.csv", &tm);

//         csv_path_ = dir + "/" + name;
//         csv_.open(csv_path_, std::ios::out | std::ios::trunc);
//         if (!csv_.is_open())
//         {
//             std::cout << "[CSV] Failed to open: " << csv_path_ << std::endl;
//             return;
//         }

//         csv_open_ = true;
//         std::cout << "[CSV] OPEN: " << csv_path_ << std::endl;
//         writeCsvHeader();
//     }

//     void writeCsvHeader()
//     {
//         if (!csv_open_)
//             return;
//         csv_
//             << "time"
//             << ",base_x,base_y,base_z"
//             << ",fr_x,fr_y,fr_z"
//             << ",fl_x,fl_y,fl_z"
//             << ",rr_x,rr_y,rr_z"
//             << ",rl_x,rl_y,rl_z"
//             << ",ee_x,ee_y,ee_z"
//             << ",com_x,com_y,com_z"
//             << ",com_vx,com_vy,com_vz"
//             << ",ee_target_x,ee_target_y,ee_target_z"
//             << ",FR0,FR1,FR2,FR3"
//             << ",FL0,FL1,FL2,FL3"
//             << ",RR0,RR1,RR2,RR3"
//             << ",RL0,RL1,RL2,RL3"
//             << ",logdetA"
//             << ",ARM0,ARM1,ARM2,ARM3"
//             << ",PASS1,PASS2"
//             << ",h0,h1,h2,h3"
//             << ",joint_limit_active"
//             << "\n";
//         csv_.flush();
//     }

//     void writeCsvRow(const Eigen::Vector3d pos[6],
//                      const Eigen::Matrix<double, 16, 1> &q_legs,
//                      double logdetA,
//                      const Eigen::Matrix<double, 4, 1> &q_arm,
//                      const Eigen::Matrix<double, 4, 1> &h,
//                      bool joint_limit_active,
//                      double q_pass1, double q_pass2)
//     {
//         if (!csv_open_)
//             return;

//         const double tsec = this->now().seconds();

//         csv_ << std::fixed << std::setprecision(6)
//              << tsec
//              << "," << pos[0].x() << "," << pos[0].y() << "," << pos[0].z()
//              << "," << pos[1].x() << "," << pos[1].y() << "," << pos[1].z()
//              << "," << pos[2].x() << "," << pos[2].y() << "," << pos[2].z()
//              << "," << pos[3].x() << "," << pos[3].y() << "," << pos[3].z()
//              << "," << pos[4].x() << "," << pos[4].y() << "," << pos[4].z()
//              << "," << pos[5].x() << "," << pos[5].y() << "," << pos[5].z()
//              << "," << com_pos_.x() << "," << com_pos_.y() << "," << com_pos_.z()
//              << "," << com_vel_.x() << "," << com_vel_.y() << "," << com_vel_.z()
//              << "," << ee_desired_.x() << "," << ee_desired_.y() << "," << ee_desired_.z()
//              << "," << q_legs(0) << "," << q_legs(1) << "," << q_legs(2) << "," << q_legs(3)
//              << "," << q_legs(4) << "," << q_legs(5) << "," << q_legs(6) << "," << q_legs(7)
//              << "," << q_legs(8) << "," << q_legs(9) << "," << q_legs(10) << "," << q_legs(11)
//              << "," << q_legs(12) << "," << q_legs(13) << "," << q_legs(14) << "," << q_legs(15)
//              << "," << logdetA
//              << "," << q_arm(0) << "," << q_arm(1) << "," << q_arm(2) << "," << q_arm(3)
//              << "," << q_pass1 << "," << q_pass2
//              << "," << h(0) << "," << h(1) << "," << h(2) << "," << h(3)
//              << "," << (joint_limit_active ? 1 : 0)
//              << "\n";

//         if (iter_count_ % 100 == 0)
//             csv_.flush();
//     }

//     // ============================
//     // Callbacks
//     // ============================
//     void basePosCallback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
//     {
//         base_pos_(0) = msg->vector.x;
//         base_pos_(1) = msg->vector.y;
//         base_pos_(2) = msg->vector.z;
//     }

//     void comPosCallback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
//     {
//         com_pos_(0) = msg->vector.x;
//         com_pos_(1) = msg->vector.y;
//         com_pos_(2) = msg->vector.z;

//         const rclcpp::Time t_now = msg->header.stamp;
//         if (have_com_prev_)
//         {
//             const double dt = (t_now - com_stamp_prev_).seconds();
//             if (dt > 0.0)
//                 com_vel_ = (com_pos_ - com_pos_prev_) / dt;
//             else
//                 com_vel_.setZero();
//         }
//         else
//         {
//             com_vel_.setZero();
//             have_com_prev_ = true;
//         }
//         com_pos_prev_ = com_pos_;
//         com_stamp_prev_ = t_now;
//     }

//     void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
//     {
//         base_quat_(0) = msg->orientation.x;
//         base_quat_(1) = msg->orientation.y;
//         base_quat_(2) = msg->orientation.z;
//         base_quat_(3) = msg->orientation.w;
//     }

//     void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
//     {
//         size_t n = msg->position.size();
//         q_ = Eigen::VectorXd::Zero(n);
//         qd_ = Eigen::VectorXd::Zero(n);

//         for (size_t i = 0; i < n; ++i)
//         {
//             q_(i) = msg->position[i];
//             if (i < msg->velocity.size())
//                 qd_(i) = msg->velocity[i];
//         }
//     }

//     void baseTargetCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
//     {
//         if (msg->data.size() < 6)
//         {
//             RCLCPP_WARN(this->get_logger(),
//                         "base_target_zyx expects 6 values, got %zu", msg->data.size());
//             return;
//         }

//         base_desired_(0) = msg->data[0];
//         base_desired_(1) = msg->data[1];
//         base_desired_(2) = msg->data[2];

//         const double yaw = msg->data[3];
//         const double pitch = msg->data[4];
//         const double roll = msg->data[5];
//         base_R_desired_ = eulerZYXToRot(yaw, pitch, roll);
//     }

//     void eeTargetCallback(const geometry_msgs::msg::Vector3::SharedPtr msg)
//     {
//         ee_desired_(0) = msg->x;
//         ee_desired_(1) = msg->y;
//         ee_desired_(2) = msg->z;
//     }

//     // ============================
//     // Utilities (sparse conversion)
//     // ============================
//     static Eigen::SparseMatrix<double> denseToSparse(const Eigen::MatrixXd &M, double tol = 0.0)
//     {
//         std::vector<Eigen::Triplet<double>> triplets;
//         triplets.reserve((size_t)(M.rows() * M.cols() / 4));
//         for (int r = 0; r < M.rows(); ++r)
//             for (int c = 0; c < M.cols(); ++c)
//             {
//                 double v = M(r, c);
//                 if (std::abs(v) > tol)
//                     triplets.emplace_back(r, c, v);
//             }
//         Eigen::SparseMatrix<double> S(M.rows(), M.cols());
//         S.setFromTriplets(triplets.begin(), triplets.end());
//         return S;
//     }

//     // ============================
//     // HQP Level solver (weighted-norm + smoothing):
//     //   min   ||w||^2
//     //       + lambda_vel * qdot^T R qdot
//     //       + lambda_acc * (qdot - qdot_prev)^T R (qdot - qdot_prev)
//     // s.t.  J_task qdot + w = xdot
//     //       (locks) J_fix qdot = rhs_fix
//     //       Jc qdot = 0
//     //       qd_lb <= qdot <= qd_ub
//     //
//     // R = blkdiag(r_base * I6, 1*I16, r_arm * I4)
//     //   (base: 0..5, legs: 6..21, arm: 22..25)
//     // ============================
//     bool solveHQPLevel(
//         const Eigen::MatrixXd &J_task,
//         const Eigen::Vector3d &xdot_task,
//         const std::vector<Eigen::MatrixXd> &J_fix,
//         const std::vector<Eigen::Vector3d> &rhs_fix,
//         const Eigen::MatrixXd &Jc,
//         const Eigen::VectorXd &qd_lb,
//         const Eigen::VectorXd &qd_ub,
//         const Eigen::VectorXd &qdot_prev, // (n)
//         double lambda_vel,
//         double lambda_acc,
//         Eigen::VectorXd &qdot_out,
//         Eigen::Vector3d &w_out)
//     {
//         const int n = (int)qd_lb.size();
//         const int m_task = 3;
//         const int mc = (int)Jc.rows();
//         const int nfix = (int)J_fix.size();

//         // Runtime guard: malformed/invalid inputs should not enter solver.
//         if (n <= 0 || qd_ub.size() != n || qdot_prev.size() != n)
//             return false;
//         if (J_task.rows() != 3 || J_task.cols() != n)
//             return false;
//         if (Jc.cols() != n)
//             return false;
//         if (rhs_fix.size() != J_fix.size())
//             return false;
//         for (int i = 0; i < nfix; ++i)
//         {
//             if (J_fix[i].rows() != 3 || J_fix[i].cols() != n)
//                 return false;
//             if (!J_fix[i].allFinite() || !rhs_fix[i].allFinite())
//                 return false;
//         }
//         if (!J_task.allFinite() || !xdot_task.allFinite() || !Jc.allFinite() ||
//             !qd_lb.allFinite() || !qd_ub.allFinite() || !qdot_prev.allFinite())
//             return false;
//         if ((qd_lb.array() > qd_ub.array()).any())
//             return false;

//         const int nvar = n + m_task;
//         const int ncon = m_task + 3 * nfix + mc + n;

//         Eigen::MatrixXd A = Eigen::MatrixXd::Zero(ncon, nvar);
//         Eigen::VectorXd l = Eigen::VectorXd::Zero(ncon);
//         Eigen::VectorXd u = Eigen::VectorXd::Zero(ncon);

//         int row = 0;

//         // (1) task with slack: J_task qdot + w = xdot_task
//         A.block(row, 0, 3, n) = J_task;
//         A.block(row, n, 3, 3) = Eigen::Matrix3d::Identity();
//         l.segment<3>(row) = xdot_task;
//         u.segment<3>(row) = xdot_task;
//         row += 3;

//         // (2) lock equalities (no slack): J_fix[i] qdot = rhs_fix[i]
//         for (int i = 0; i < nfix; ++i)
//         {
//             A.block(row, 0, 3, n) = J_fix[i];
//             l.segment<3>(row) = rhs_fix[i];
//             u.segment<3>(row) = rhs_fix[i];
//             row += 3;
//         }

//         // (3) contact equality: Jc qdot = 0
//         if (mc > 0)
//         {
//             A.block(row, 0, mc, n) = Jc;
//             l.segment(row, mc).setZero();
//             u.segment(row, mc).setZero();
//             row += mc;
//         }

//         // (4) box on qdot
//         A.block(row, 0, n, n) = Eigen::MatrixXd::Identity(n, n);
//         l.segment(row, n) = qd_lb;
//         u.segment(row, n) = qd_ub;
//         row += n;

//         // ----------------------------
//         // Weighted metric R (diagonal)
//         // R = blkdiag(r_base*I6, 1*I16, r_arm*I4)
//         // 가중치 작을수록 더 많이 움직임
//         // ----------------------------
//         double r_base = 1.0;
//         double r_arm = 0.1;

//         Eigen::VectorXd Rdiag = Eigen::VectorXd::Ones(n);
//         if (n >= 6)
//             Rdiag.segment<6>(0).setConstant(r_base);

//         // arm indices are 22..25 for n=26
//         if (n >= 26)
//             Rdiag.segment<4>(22).setConstant(r_arm);
//         else if (n > 22)
//         {
//             // safe for unexpected n (partial models)
//             const int arm_len = std::min(4, n - 22);
//             Rdiag.segment(22, arm_len).setConstant(r_arm);
//         }

//         // ----------------------------
//         // Cost:
//         //   ||w||^2
//         // + lambda_vel * qdot^T R qdot
//         // + lambda_acc * (qdot - qdot_prev)^T R (qdot - qdot_prev)
//         //
//         // Expand qdot part:
//         //   (lambda_vel + lambda_acc) qdot^T R qdot - 2 lambda_acc qdot_prev^T R qdot + const
//         // ----------------------------
//         Eigen::MatrixXd H = Eigen::MatrixXd::Zero(nvar, nvar);
//         H.block(0, 0, n, n).diagonal().array() =
//             2.0 * (lambda_vel + lambda_acc) * Rdiag.array();
//         H.block(n, n, 3, 3).diagonal().array() = 2.0; // ||w||^2

//         Eigen::VectorXd g = Eigen::VectorXd::Zero(nvar);
//         g.head(n).array() =
//             -2.0 * lambda_acc * Rdiag.array() * qdot_prev.array();

//         // (optional) sanity: avoid NaN/Inf
//         if (!H.allFinite() || !g.allFinite() || !A.allFinite() || !l.allFinite() || !u.allFinite())
//             return false;

//         OsqpEigen::Solver solver;
//         solver.settings()->setVerbosity(false);
//         solver.settings()->setWarmStart(false);

//         solver.data()->setNumberOfVariables(nvar);
//         solver.data()->setNumberOfConstraints(ncon);

//         if (!solver.data()->setHessianMatrix(denseToSparse(H, 1e-12)))
//             return false;
//         if (!solver.data()->setGradient(g))
//             return false;
//         if (!solver.data()->setLinearConstraintsMatrix(denseToSparse(A, 1e-12)))
//             return false;
//         if (!solver.data()->setLowerBound(l))
//             return false;
//         if (!solver.data()->setUpperBound(u))
//             return false;

//         if (!solver.initSolver())
//             return false;
//         if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
//             return false;

//         Eigen::VectorXd z = solver.getSolution();
//         if (z.size() != nvar || !z.allFinite())
//             return false;
//         qdot_out = z.head(n);
//         if (!qdot_out.allFinite())
//             return false;
//         qdot_out = qdot_out.cwiseMax(qd_lb).cwiseMin(qd_ub);
//         w_out = z.segment<3>(n);
//         if (!w_out.allFinite())
//             return false;
//         return true;
//     }

//     bool solveFinalSmoothingQP(
//         const Eigen::VectorXd &qdot_prev,                       // (26)
//         const Eigen::MatrixXd &J1, const Eigen::Vector3d &rhs1, // J1 qdot = rhs1
//         const Eigen::MatrixXd &J2, const Eigen::Vector3d &rhs2,
//         const Eigen::MatrixXd &J3, const Eigen::Vector3d &rhs3,
//         const Eigen::MatrixXd &Jc,                            // (mc x 26), Jc qdot = 0
//         const Eigen::VectorXd &lb, const Eigen::VectorXd &ub, // (26)
//         double lambda_vel,
//         double lambda_acc,
//         Eigen::VectorXd &qdot_out) // (26)
//     {
//         const int n = 26;
//         const int mc = (int)Jc.rows();

//         // vars: qdot (26)
//         const int nvar = n;

//         // constraints: J1(3) + J2(3) + J3(3) + Jc(mc) + box(n)
//         const int ncon = 3 + 3 + 3 + mc + n;

//         Eigen::MatrixXd A = Eigen::MatrixXd::Zero(ncon, nvar);
//         Eigen::VectorXd l = Eigen::VectorXd::Zero(ncon);
//         Eigen::VectorXd u = Eigen::VectorXd::Zero(ncon);

//         int row = 0;

//         // J1 qdot = rhs1
//         A.block(row, 0, 3, n) = J1;
//         l.segment<3>(row) = rhs1;
//         u.segment<3>(row) = rhs1;
//         row += 3;

//         // J2 qdot = rhs2
//         A.block(row, 0, 3, n) = J2;
//         l.segment<3>(row) = rhs2;
//         u.segment<3>(row) = rhs2;
//         row += 3;

//         // J3 qdot = rhs3
//         A.block(row, 0, 3, n) = J3;
//         l.segment<3>(row) = rhs3;
//         u.segment<3>(row) = rhs3;
//         row += 3;

//         // Jc qdot = 0
//         if (mc > 0)
//         {
//             A.block(row, 0, mc, n) = Jc;
//             l.segment(row, mc).setZero();
//             u.segment(row, mc).setZero();
//             row += mc;
//         }

//         // box
//         A.block(row, 0, n, n) = Eigen::MatrixXd::Identity(n, n);
//         l.segment(row, n) = lb;
//         u.segment(row, n) = ub;
//         row += n;

//         // Cost:
//         // lambda_vel * ||qdot||^2 + lambda_acc * ||qdot - qdot_prev||^2

//         Eigen::MatrixXd H =
//             2.0 * (lambda_vel + lambda_acc) * Eigen::MatrixXd::Identity(nvar, nvar);

//         Eigen::VectorXd g =
//             -2.0 * lambda_acc * qdot_prev;

//         Eigen::SparseMatrix<double> Hs = denseToSparse(H, 1e-12);
//         Eigen::SparseMatrix<double> As = denseToSparse(A, 1e-12);

//         OsqpEigen::Solver solver;
//         solver.settings()->setVerbosity(false);
//         solver.settings()->setWarmStart(true);

//         solver.data()->setNumberOfVariables(nvar);
//         solver.data()->setNumberOfConstraints(ncon);

//         if (!solver.data()->setHessianMatrix(Hs))
//             return false;
//         if (!solver.data()->setGradient(g))
//             return false;
//         if (!solver.data()->setLinearConstraintsMatrix(As))
//             return false;
//         if (!solver.data()->setLowerBound(l))
//             return false;
//         if (!solver.data()->setUpperBound(u))
//             return false;

//         if (!solver.initSolver())
//             return false;
//         if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
//             return false;

//         qdot_out = solver.getSolution();
//         return true;
//     }

//     bool solveLowerBodyQP(
//         const Eigen::VectorXd &qdot_base6,                        // (6) fixed from HQP
//         const Eigen::MatrixXd &J_feet_12x26,                      // feet pos jacobian (12x26)
//         const Eigen::VectorXd &v_feet_12,                         // desired feet vel (12)
//         const Eigen::VectorXd &lb26, const Eigen::VectorXd &ub26, // bounds (26) -> use legs part
//         const Eigen::VectorXd &qdot_legs_prev16,                  // (16) for smoothing
//         double lambda,                                            // smoothing weight
//         Eigen::VectorXd &qdot_legs16_out)                         // (16)
//     {
//         // Variables: qdot_legs (16)  for joints indices 6..21 in your 26 model
//         // We assume feet are affected by base6 and legs16 (arm doesn't affect feet).
//         const int n_leg = 16;
//         const int mc_feet = 12;

//         // Runtime guard: malformed/invalid lower-body QP input.
//         if (qdot_base6.size() != 6 || J_feet_12x26.rows() != mc_feet || J_feet_12x26.cols() < 22 ||
//             v_feet_12.size() != mc_feet || lb26.size() < 22 || ub26.size() < 22 || qdot_legs_prev16.size() != n_leg)
//             return false;
//         if (!qdot_base6.allFinite() || !J_feet_12x26.allFinite() || !v_feet_12.allFinite() ||
//             !lb26.allFinite() || !ub26.allFinite() || !qdot_legs_prev16.allFinite() || !std::isfinite(lambda))
//             return false;

//         const Eigen::VectorXd lb_leg = lb26.segment(6, n_leg);
//         const Eigen::VectorXd ub_leg = ub26.segment(6, n_leg);
//         if ((lb_leg.array() > ub_leg.array()).any())
//             return false;

//         // Split J_feet into base and legs blocks
//         // legs columns in full 26 are [6..21] contiguous 16 columns
//         Eigen::MatrixXd Jb = J_feet_12x26.block(0, 0, mc_feet, 6);     // (12x6)
//         Eigen::MatrixXd Jl = J_feet_12x26.block(0, 6, mc_feet, n_leg); // (12x16)

//         // Equality: Jl*qdot_legs = v_feet - Jb*qdot_base
//         Eigen::VectorXd rhs = v_feet_12 - Jb * qdot_base6; // (12)

//         // Constraints:
//         // 1) feet equality (12)
//         // 2) box on legs (16)
//         const int nvar = n_leg;
//         const int ncon = mc_feet + n_leg;

//         Eigen::MatrixXd A = Eigen::MatrixXd::Zero(ncon, nvar);
//         Eigen::VectorXd l = Eigen::VectorXd::Zero(ncon);
//         Eigen::VectorXd u = Eigen::VectorXd::Zero(ncon);

//         int row = 0;

//         // feet eq
//         A.block(row, 0, mc_feet, n_leg) = Jl;
//         l.segment(row, mc_feet) = rhs;
//         u.segment(row, mc_feet) = rhs;
//         row += mc_feet;

//         // legs box from lb26/ub26 (indices 6..21)
//         A.block(row, 0, n_leg, n_leg) = Eigen::MatrixXd::Identity(n_leg, n_leg);
//         l.segment(row, n_leg) = lb_leg;
//         u.segment(row, n_leg) = ub_leg;
//         row += n_leg;

//         // Cost: ||qdot_legs - qdot_legs_prev||^2 + lambda||qdot_legs||^2
//         Eigen::MatrixXd H = 2.0 * (1.0 + lambda) * Eigen::MatrixXd::Identity(nvar, nvar);
//         Eigen::VectorXd g = -2.0 * qdot_legs_prev16;
//         if (!A.allFinite() || !l.allFinite() || !u.allFinite() || !H.allFinite() || !g.allFinite())
//             return false;

//         Eigen::SparseMatrix<double> Hs = denseToSparse(H, 1e-12);
//         Eigen::SparseMatrix<double> As = denseToSparse(A, 1e-12);

//         OsqpEigen::Solver solver;
//         solver.settings()->setVerbosity(false);
//         solver.settings()->setWarmStart(true);

//         solver.data()->setNumberOfVariables(nvar);
//         solver.data()->setNumberOfConstraints(ncon);

//         if (!solver.data()->setHessianMatrix(Hs))
//             return false;
//         if (!solver.data()->setGradient(g))
//             return false;
//         if (!solver.data()->setLinearConstraintsMatrix(As))
//             return false;
//         if (!solver.data()->setLowerBound(l))
//             return false;
//         if (!solver.data()->setUpperBound(u))
//             return false;

//         if (!solver.initSolver())
//             return false;
//         if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
//             return false;

//         const Eigen::VectorXd z = solver.getSolution();
//         if (z.size() != n_leg || !z.allFinite())
//             return false;
//         qdot_legs16_out = z.cwiseMax(lb_leg).cwiseMin(ub_leg);
//         if (!qdot_legs16_out.allFinite())
//             return false;
//         return true;
//     }

//     // ============================
//     // controlLoop
//     // ============================
//     void controlLoop()
//     {
//         // 1) Joint State Check
//         if (q_.size() < 20)
//         {
//             if (iter_count_ % 100 == 0)
//                 RCLCPP_WARN(this->get_logger(), "Waiting for Joints... Current Size: %ld", q_.size());
//             return;
//         }

//         const auto t_start = std::chrono::steady_clock::now();

//         // 2) FK + Jacobians
//         Eigen::Vector4d base_quat_safe = base_quat_;
//         if (!base_quat_safe.allFinite() || base_quat_safe.norm() < 1e-12)
//         {
//             if (iter_count_ % 100 == 0)
//                 RCLCPP_WARN(this->get_logger(), "Invalid base_quat in control loop. Falling back to identity.");
//             base_quat_safe << 0.0, 0.0, 0.0, 1.0;
//         }
//         else
//         {
//             base_quat_safe.normalize();
//         }
//         base_quat_ = base_quat_safe;

//         auto result = computeFKAllAndJacobian(base_pos_, base_quat_safe, q_);

//         // 3) Holding Logic
//         if (!control_enabled_)
//         {
//             ee_desired_ = result.T[5].block<3, 1>(0, 3);
//             base_desired_ = result.T[0].block<3, 1>(0, 3);
//             FR_desired_ = result.T[1].block<3, 1>(0, 3) - Eigen::Vector3d(0.0, 0.0, 0.05);
//             FL_desired_ = result.T[2].block<3, 1>(0, 3) - Eigen::Vector3d(0.0, 0.0, 0.05);
//             RR_desired_ = result.T[3].block<3, 1>(0, 3) - Eigen::Vector3d(0.0, 0.0, 0.05);
//             RL_desired_ = result.T[4].block<3, 1>(0, 3) - Eigen::Vector3d(0.0, 0.0, 0.05);

//             std::cout << "FR_desired_: " << FR_desired_.transpose() << std::endl;
//             std::cout << "FL_desired_: " << FL_desired_.transpose() << std::endl;
//             std::cout << "RR_desired_: " << RR_desired_.transpose() << std::endl;
//             std::cout << "RL_desired_: " << RL_desired_.transpose() << std::endl;

//             control_enabled_ = true;
//             return;
//         }

//         // 4) Publish EE Pose (debug)
//         {
//             geometry_msgs::msg::Vector3Stamped ee_pos_msg;
//             ee_pos_msg.header.stamp = this->now();
//             ee_pos_msg.header.frame_id = "world";
//             ee_pos_msg.vector.x = result.T[5](0, 3);
//             ee_pos_msg.vector.y = result.T[5](1, 3);
//             ee_pos_msg.vector.z = result.T[5](2, 3);
//             ee_pos_pub_->publish(ee_pos_msg);
//         }

//         // ---------------------------------------------------------
//         // 5) Build J_full (21 x 26)
//         // ---------------------------------------------------------
//         Eigen::MatrixXd J_full = Eigen::MatrixXd::Zero(21, 26);
//         J_full.block<6, 6>(0, 0) = result.J[0];

//         J_full.block<3, 6>(6, 0) = result.J[1].block<3, 6>(0, 0);
//         J_full.block<3, 4>(6, 6) = result.J[1].block<3, 4>(0, 6);

//         J_full.block<3, 6>(9, 0) = result.J[2].block<3, 6>(0, 0);
//         J_full.block<3, 4>(9, 10) = result.J[2].block<3, 4>(0, 6);

//         J_full.block<3, 6>(12, 0) = result.J[3].block<3, 6>(0, 0);
//         J_full.block<3, 4>(12, 14) = result.J[3].block<3, 4>(0, 6);

//         J_full.block<3, 6>(15, 0) = result.J[4].block<3, 6>(0, 0);
//         J_full.block<3, 4>(15, 18) = result.J[4].block<3, 4>(0, 6);

//         J_full.block<3, 6>(18, 0) = result.J[5].block<3, 6>(0, 0);
//         J_full.block<3, 4>(18, 22) = result.J[5].block<3, 4>(0, 6);

//         // ---------------------------------------------------------
//         // 6) Manipulability direction (Task3 command)
//         // ---------------------------------------------------------
//         Eigen::Matrix3d A_man = computeManipulabilityMatrix(result, q_);
//         Eigen::Vector3d grad = computeVMaxDirection(base_pos_, base_quat_, q_, A_man);

//         // IMPORTANT: make a bounded command (otherwise it can be tiny/huge)
//         double grad_norm = grad.norm();
//         Eigen::Vector3d grad_dir = Eigen::Vector3d::Zero();
//         if (grad_norm > 1e-9)
//             grad_dir = grad / grad_norm;
//         double v_man_max = 0.2;                        // 너가 쓰던 clamp
//         double v_man = std::min(v_man_max, grad_norm); // magnitude
//         Eigen::Vector3d v_cmd_base_lin = v_man * grad_dir;

//         // ---------------------------------------------------------
//         // 7) Tasks (Task1=BaseOri, Task2=EE, Task3=BaseLin(manip))
//         // ---------------------------------------------------------
//         Eigen::MatrixXd J1 = J_full.block<3, 26>(3, 0);  // base ori rows
//         Eigen::MatrixXd J2 = J_full.block<3, 26>(18, 0); // ee pos rows
//         Eigen::MatrixXd J3 = J_full.block<3, 26>(0, 0);  // base lin rows

//         // Task2 EE command
//         Eigen::Vector3d ee_error = ee_desired_ - result.T[5].block<3, 1>(0, 3);
//         Eigen::Vector3d v_cmd_ee = 4.0 * ee_error;

//         // Task1 base ori command
//         Eigen::Matrix3d R_cur = result.T[0].block<3, 3>(0, 0);
//         Eigen::Vector3d base_ori_err =
//             R_cur.col(0).cross(base_R_desired_.col(0)) +
//             R_cur.col(1).cross(base_R_desired_.col(1)) +
//             R_cur.col(2).cross(base_R_desired_.col(2));
//         Eigen::Vector3d v_cmd_base_ori = 3.0 * base_ori_err;

//         // Contact equality: feet position fixed (12)
//         Eigen::MatrixXd Jc = J_full.block<12, 26>(6, 0);

//         // ---------------------------------------------------------
//         // 8) Joint bounds (your logic)  **FIXED: /dt**
//         // ---------------------------------------------------------
//         const double dt = 0.01;
//         const double v_lim_base = 0.5;
//         const double safety_ratio = 0.9;

//         Eigen::VectorXd q_min_full = Eigen::VectorXd::Constant(26, -1e10);
//         Eigen::VectorXd q_max_full = Eigen::VectorXd::Constant(26, 1e10);
//         Eigen::VectorXd qd_min_full = Eigen::VectorXd::Constant(26, -1e10);
//         Eigen::VectorXd qd_max_full = Eigen::VectorXd::Constant(26, 1e10);

//         // --- actuator position limits ---
//         q_min_full.segment<4>(6) << deg2rad(-49), deg2rad(-58.4), deg2rad(-55.9), deg2rad(-180);
//         q_max_full.segment<4>(6) << deg2rad(28.9), deg2rad(56.9), deg2rad(56.3), deg2rad(180);

//         q_min_full.segment<4>(10) << deg2rad(-28.9), deg2rad(-58.4), deg2rad(-55.9), deg2rad(-180);
//         q_max_full.segment<4>(10) << deg2rad(49), deg2rad(56.9), deg2rad(56.3), deg2rad(180);

//         q_min_full.segment<4>(14) << deg2rad(-28.9), deg2rad(-56.9), deg2rad(-56.3), deg2rad(-180);
//         q_max_full.segment<4>(14) << deg2rad(49), deg2rad(58.4), deg2rad(55.9), deg2rad(180);

//         q_min_full.segment<4>(18) << deg2rad(-49), deg2rad(-56.9), deg2rad(-56.3), deg2rad(-180);
//         q_max_full.segment<4>(18) << deg2rad(28.9), deg2rad(58.4), deg2rad(55.9), deg2rad(180);

//         q_min_full.segment<4>(22) << -M_PI / 2, 0.0, -2.0106193, 0.0;
//         q_max_full.segment<4>(22) << M_PI / 2, 1.0472, 0.0, 1.65;

//         // --- velocity limits ---
//         qd_min_full.segment<4>(6) << -0.35, -0.35, -0.35, -1.0;
//         qd_max_full.segment<4>(6) << 0.35, 0.35, 0.35, 1.0;

//         qd_min_full.segment<4>(10) << -0.35, -0.35, -0.35, -1.0;
//         qd_max_full.segment<4>(10) << 0.35, 0.35, 0.35, 1.0;

//         qd_min_full.segment<4>(14) << -0.35, -0.35, -0.35, -1.0;
//         qd_max_full.segment<4>(14) << 0.35, 0.35, 0.35, 1.0;

//         qd_min_full.segment<4>(18) << -0.35, -0.35, -0.35, -1.0;
//         qd_max_full.segment<4>(18) << 0.35, 0.35, 0.35, 1.0;

//         qd_min_full.segment<4>(22) << -1.0, -0.5, -0.5, -0.5;
//         qd_max_full.segment<4>(22) << 1.0, 0.5, 0.5, 0.5;

//         // shrink position range
//         for (int i = 6; i < 26; ++i)
//         {
//             double center = 0.5 * (q_min_full(i) + q_max_full(i));
//             double half_range = 0.5 * (q_max_full(i) - q_min_full(i));
//             double safe_half = half_range * safety_ratio;
//             q_min_full(i) = center - safe_half;
//             q_max_full(i) = center + safe_half;
//         }

//         // model-index(26) -> hw-index(q_) mapping (your existing assumption)
//         std::vector<int> q_map(26, -1);
//         int hw_idx = 0;
//         for (int leg = 0; leg < 4; ++leg)
//         {
//             for (int j = 0; j < 4; ++j)
//                 q_map[6 + leg * 4 + j] = hw_idx++;
//             hw_idx++; // skip
//         }
//         for (int j = 0; j < 4; ++j)
//             q_map[22 + j] = hw_idx++;

//         // qdot bounds (26)
//         Eigen::VectorXd qp_lb = Eigen::VectorXd::Zero(26);
//         Eigen::VectorXd qp_ub = Eigen::VectorXd::Zero(26);

//         for (int i = 0; i < 26; ++i)
//         {
//             if (i < 6)
//             {
//                 qp_lb(i) = -v_lim_base;
//                 qp_ub(i) = v_lim_base;
//                 continue;
//             }

//             int q_idx = q_map[i];
//             if (q_idx < 0 || q_idx >= (int)q_.size())
//             {
//                 qp_lb(i) = -v_lim_base;
//                 qp_ub(i) = v_lim_base;
//                 continue;
//             }

//             double q_curr = q_(q_idx);
//             double min_val = q_min_full(i);
//             double max_val = q_max_full(i);

//             double raw_lb = (min_val - q_curr) / (10.0 * dt); // more conservative for velocity limits
//             double raw_ub = (max_val - q_curr) / (10.0 * dt);

//             qp_lb(i) = std::min(std::max(raw_lb, qd_min_full(i)), qd_max_full(i));
//             qp_ub(i) = std::max(std::min(raw_ub, qd_max_full(i)), qd_min_full(i));
//         }

//         // ---------------------------------------------------------
//         // 9) HQP Solve (3 levels, 26-dim qdot)
//         // ---------------------------------------------------------
//         Eigen::VectorXd qdot1(26), qdot2(26), qdot3(26);
//         Eigen::Vector3d w1, w2, w3;
//         double lambda_vel = 1e-3;
//         double lambda_acc = 1e-4;
//         bool upper_hqp_ok = true;
//         Eigen::VectorXd qdot_fallback = q_dot_prev_;
//         if (qdot_fallback.size() != 26 || !qdot_fallback.allFinite())
//             qdot_fallback = Eigen::VectorXd::Zero(26);

//         // L1: BaseOri
//         {
//             std::vector<Eigen::MatrixXd> J_fix;
//             std::vector<Eigen::Vector3d> rhs_fix;

//             if (!solveHQPLevel(J1, v_cmd_base_ori,
//                                J_fix, rhs_fix,
//                                Jc, qp_lb, qp_ub,
//                                q_dot_prev_, lambda_vel, lambda_acc,
//                                qdot1, w1))
//             {
//                 RCLCPP_WARN(this->get_logger(), "HQP L1(BaseOri) failed");
//                 upper_hqp_ok = false;
//             }
//         }

//         // L2: EE, lock L1
//         if (upper_hqp_ok)
//         {
//             std::vector<Eigen::MatrixXd> J_fix;
//             std::vector<Eigen::Vector3d> rhs_fix;

//             J_fix.push_back(J1);
//             rhs_fix.push_back(v_cmd_base_ori - w1);

//             if (!solveHQPLevel(J2, v_cmd_ee,
//                                J_fix, rhs_fix,
//                                Jc, qp_lb, qp_ub,
//                                q_dot_prev_, lambda_vel, lambda_acc,
//                                qdot2, w2))
//             {
//                 RCLCPP_WARN(this->get_logger(), "HQP L2(EE) failed");
//                 upper_hqp_ok = false;
//             }
//         }

//         // L3: BaseLin(manip), lock L1~L2
//         if (upper_hqp_ok)
//         {
//             std::vector<Eigen::MatrixXd> J_fix;
//             std::vector<Eigen::Vector3d> rhs_fix;

//             J_fix.push_back(J1);
//             rhs_fix.push_back(v_cmd_base_ori - w1);
//             J_fix.push_back(J2);
//             rhs_fix.push_back(v_cmd_ee - w2);

//             if (!solveHQPLevel(J3, v_cmd_base_lin,
//                                J_fix, rhs_fix,
//                                Jc, qp_lb, qp_ub,
//                                q_dot_prev_, lambda_vel, lambda_acc,
//                                qdot3, w3))
//             {
//                 RCLCPP_WARN(this->get_logger(), "HQP L3(BaseLin) failed");
//                 upper_hqp_ok = false;
//             }
//         }

//         Eigen::VectorXd qdot_final = upper_hqp_ok ? qdot3 : qdot_fallback;
//         if (!qdot_final.allFinite() || qdot_final.cwiseAbs().maxCoeff() > 10.0)
//         {
//             RCLCPP_WARN(this->get_logger(), "Invalid/oversized upper HQP output. Falling back to previous command.");
//             qdot_final = qdot_fallback;
//         }
//         qdot_final = qdot_final.cwiseMax(qp_lb).cwiseMin(qp_ub);
//         q_dot_prev_ = qdot_final;

//         // auto e1 = (J1*qdot_final - rhs1).norm();
//         // auto e2 = (J2*qdot_final - rhs2).norm();
//         // auto e3 = (J3*qdot_final - rhs3).norm();
//         // std::cout << "lock residuals: " << e1 << " " << e2 << " " << e3 << std::endl;

//         if (upper_hqp_ok)
//         {
//             auto e1 = w1.norm();
//             auto e2 = w2.norm();
//             auto e3 = w3.norm();
//             std::cout << "lock residuals: " << e1 << " " << e2 << " " << e3 << std::endl;
//         }

//         // ---------------------------------------------------------
//         // 11) Lower-body QP: compute legs16 to satisfy feet tracking + joint limits
//         //     base6 fixed from qdot_final
//         // ---------------------------------------------------------
//         Eigen::VectorXd qdot_base6 = qdot_final.head<6>();
//         Eigen::VectorXd qdot_arm4 = qdot_final.segment<4>(22);

//         // desired feet velocity from position error (you used 0.5*error)
//         Eigen::Vector3d FR_error = FR_desired_ - result.T[1].block<3, 1>(0, 3);
//         Eigen::Vector3d FL_error = FL_desired_ - result.T[2].block<3, 1>(0, 3);
//         Eigen::Vector3d RR_error = RR_desired_ - result.T[3].block<3, 1>(0, 3);
//         Eigen::Vector3d RL_error = RL_desired_ - result.T[4].block<3, 1>(0, 3);

//         Eigen::VectorXd v_feet_12(12);
//         v_feet_12.segment<3>(0) = 0.5 * FR_error;
//         v_feet_12.segment<3>(3) = 0.5 * FL_error;
//         v_feet_12.segment<3>(6) = 0.5 * RR_error;
//         v_feet_12.segment<3>(9) = 0.5 * RL_error;
//         if (!v_feet_12.allFinite())
//         {
//             RCLCPP_WARN(this->get_logger(), "Invalid lower-body task command detected. Using zero feet command.");
//             v_feet_12.setZero();
//         }

//         // feet jacobian = rows 6..17 of J_full (12x26)
//         Eigen::MatrixXd J_feet_12x26 = J_full.block<12, 26>(6, 0);

//         // legs smoothing state (store in q_dot_prev_ legs part; if you want separate state, add member)
//         Eigen::VectorXd qdot_legs_prev16 = q_dot_prev_.segment<16>(6);
//         if (!qdot_legs_prev16.allFinite())
//             qdot_legs_prev16.setZero();

//         Eigen::VectorXd qdot_legs16(16);
//         double lambda_leg_smooth = 1e-2;
//         Eigen::VectorXd qdot_legs_fallback = qdot_final.segment<16>(6);
//         if (!qdot_legs_fallback.allFinite())
//             qdot_legs_fallback.setZero();
//         bool is_lower_qp_ok = solveLowerBodyQP(qdot_base6, J_feet_12x26, v_feet_12, qp_lb, qp_ub, qdot_legs_prev16, lambda_leg_smooth, qdot_legs16);
//         if (!is_lower_qp_ok)
//         {
//             RCLCPP_WARN(this->get_logger(), "Lower-body QP failed, fallback to HQP legs");
//             qdot_legs16 = qdot_legs_fallback;
//         }
//         if (!qdot_legs16.allFinite() || qdot_legs16.cwiseAbs().maxCoeff() > 10.0)
//         {
//             RCLCPP_WARN(this->get_logger(), "Invalid/oversized lower-body QP output. Falling back to HQP legs.");
//             qdot_legs16 = qdot_legs_fallback;
//         }
//         qdot_legs16 = qdot_legs16.cwiseMax(qp_lb.segment<16>(6)).cwiseMin(qp_ub.segment<16>(6));

//         // compose 22 vector [base6; legs16] for your publish mapping
//         Eigen::VectorXd qdot_lower22(22);
//         qdot_lower22.head<6>() = qdot_base6;
//         qdot_lower22.segment<16>(6) = qdot_legs16;

//         // ---------------------------------------------------------
//         // 12) Publish (keep your existing joint index mapping)
//         // ---------------------------------------------------------
//         sensor_msgs::msg::JointState vel_cmd_msg;
//         vel_cmd_msg.header.stamp = this->now();
//         vel_cmd_msg.name.resize(26);
//         vel_cmd_msg.velocity.resize(26, 0.0);
//         for (size_t i = 0; i < 26; ++i)
//             vel_cmd_msg.name[i] = "joint_" + std::to_string(i);

//         // legs (from qdot_lower22)
//         vel_cmd_msg.velocity[0] = qdot_lower22(6);
//         vel_cmd_msg.velocity[1] = qdot_lower22(7);
//         vel_cmd_msg.velocity[2] = qdot_lower22(8);
//         vel_cmd_msg.velocity[3] = qdot_lower22(9);

//         vel_cmd_msg.velocity[5] = qdot_lower22(10);
//         vel_cmd_msg.velocity[6] = qdot_lower22(11);
//         vel_cmd_msg.velocity[7] = qdot_lower22(12);
//         vel_cmd_msg.velocity[8] = qdot_lower22(13);

//         vel_cmd_msg.velocity[10] = qdot_lower22(14);
//         vel_cmd_msg.velocity[11] = qdot_lower22(15);
//         vel_cmd_msg.velocity[12] = qdot_lower22(16);
//         vel_cmd_msg.velocity[13] = qdot_lower22(17);

//         vel_cmd_msg.velocity[15] = qdot_lower22(18);
//         vel_cmd_msg.velocity[16] = qdot_lower22(19);
//         vel_cmd_msg.velocity[17] = qdot_lower22(20);
//         vel_cmd_msg.velocity[18] = qdot_lower22(21);

//         // arm (from HQP final)
//         vel_cmd_msg.velocity[20] = qdot_arm4(0);
//         vel_cmd_msg.velocity[21] = qdot_arm4(1);
//         vel_cmd_msg.velocity[22] = qdot_arm4(2);
//         vel_cmd_msg.velocity[23] = qdot_arm4(3);

//         velocity_cmd_pub_->publish(vel_cmd_msg);

//         // ---------------------------------------------------------
//         // 13) CSV logging (unchanged - keep yours; optional add v_man etc)
//         // ---------------------------------------------------------
//         if (control_enabled_ && csv_open_)
//         {
//             Eigen::Vector3d pos_arr[6];
//             for (int i = 0; i < 6; ++i)
//                 pos_arr[i] = result.T[i].block<3, 1>(0, 3);

//             Eigen::Matrix<double, 16, 1> q_legs;
//             if (q_.size() >= 19)
//             {
//                 q_legs << q_(0), q_(1), q_(2), q_(3), q_(5), q_(6), q_(7), q_(8),
//                     q_(10), q_(11), q_(12), q_(13), q_(15), q_(16), q_(17), q_(18);
//             }
//             else
//                 q_legs.setZero();

//             Eigen::Matrix<double, 4, 1> q_arm_log;
//             if (q_.size() >= 24)
//                 q_arm_log = q_.segment<4>(20);
//             else
//                 q_arm_log.setZero();

//             Eigen::Matrix<double, 4, 1> h_val = Eigen::Matrix<double, 4, 1>::Zero();
//             writeCsvRow(pos_arr, q_legs, 0.0, q_arm_log, h_val, false, 0.0, 0.0);
//         }

//         // 14) Profiling
//         const auto t_end = std::chrono::steady_clock::now();
//         const double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
//         iter_count_++;
//         ema_ms_ = (iter_count_ == 1) ? ms : (0.05 * ms + 0.95 * ema_ms_);
//         if (ms > max_ms_)
//             max_ms_ = ms;
//         if (iter_count_ % 100 == 0)
//         {
//             const double hz_est = (ms > 0.0) ? (1000.0 / ms) : 0.0;
//             RCLCPP_INFO(this->get_logger(), "[profile] loop_ms=%.3f | ema_ms=%.3f | est_hz=%.1f",
//                         ms, ema_ms_, hz_est);
//         }
//     }

//     // ----- Subscribers / Publishers -----
//     rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr base_pos_sub_;
//     rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
//     rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
//     rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr velocity_cmd_pub_;
//     rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr base_target_sub_;
//     rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr ee_target_sub_;
//     rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_array_pub_;
//     rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr vcmd_marker_pub_;
//     rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr ee_pos_pub_;
//     rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr com_pos_sub_;
//     rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr control_enable_sub_;

//     // ----- Timer -----
//     rclcpp::TimerBase::SharedPtr control_timer_;

//     // ----- State Variables -----
//     Eigen::Vector3d base_pos_ = Eigen::Vector3d::Zero();
//     Eigen::Vector4d base_quat_ = Eigen::Vector4d(0, 0, 0, 1); // [x y z w]
//     Eigen::VectorXd q_;
//     Eigen::VectorXd qd_;
//     Eigen::Vector3d FR_desired_, FL_desired_, RR_desired_, RL_desired_;
//     Eigen::Vector3d ee_desired_, base_desired_;
//     Eigen::Matrix3d base_R_desired_;

//     // prev for smoothing
//     Eigen::VectorXd q_dot_prev_ = Eigen::VectorXd::Zero(26);

//     bool control_enabled_ = false;
//     double prev_grad_gain_ = 0.0;
// };

// int main(int argc, char *argv[])
// {
//     rclcpp::init(argc, argv);
//     auto node = std::make_shared<EcvtController>();

//     rclcpp::on_shutdown([node]()
//                         { node->closeCsv(); });

//     rclcpp::spin(node);
//     rclcpp::shutdown();
//     return 0;
// }
