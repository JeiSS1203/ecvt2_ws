// teleop_goal_publisher.cpp
// Publishes /hanyang/base_target_zyx (Float64MultiArray: [x y z yaw pitch roll])
// and /hanyang/ee_target (geometry_msgs/Vector3) by applying small increments
// on top of the *current measured values* (base_pos + imu + ee_pos).
//
// Key bindings (hold/repeat):
//   Base position:   i/k = +x/-x,  j/l = +y/-y,  u/o = +z/-z
//   Base orientation: q/e = +yaw/-yaw,  t/g = +pitch/-pitch,  z/x = +roll/-roll
//   EE position:     w/s = +x/-x,  a/d = +y/-y,  r/f = +z/-z
//   Space: publish once (forces publish even if no delta)
//   c: reset targets to current (zero delta)
//   h: help
//   ESC: exit
//
// Notes:
// - yaw/pitch/roll are in radians (ZYX convention: yaw(Z), pitch(Y), roll(X))
// - You can change step sizes via ROS2 parameters.

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <optional>

#include <termios.h>
#include <unistd.h>
#include <sys/select.h>
#include <array>

static inline double wrap_pi(double a)
{
    while (a > M_PI)
        a -= 2.0 * M_PI;
    while (a < -M_PI)
        a += 2.0 * M_PI;
    return a;
}

// Quaternion (x,y,z,w) -> ZYX yaw,pitch,roll
static inline void quat_to_zyx(double qx, double qy, double qz, double qw,
                               double &yaw, double &pitch, double &roll)
{
    // roll (x-axis rotation)
    const double sinr_cosp = 2.0 * (qw * qx + qy * qz);
    const double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
    roll = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    const double sinp = 2.0 * (qw * qy - qz * qx);
    if (std::abs(sinp) >= 1.0)
        pitch = std::copysign(M_PI / 2.0, sinp);
    else
        pitch = std::asin(sinp);

    // yaw (z-axis rotation)
    const double siny_cosp = 2.0 * (qw * qz + qx * qy);
    const double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    yaw = std::atan2(siny_cosp, cosy_cosp);

    yaw = wrap_pi(yaw);
    pitch = wrap_pi(pitch);
    roll = wrap_pi(roll);
}

class Keyboard
{
public:
    Keyboard()
    {
        tcgetattr(STDIN_FILENO, &orig_);
        termios raw = orig_;
        raw.c_lflag &= ~(ICANON | ECHO);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    }

    ~Keyboard()
    {
        tcsetattr(STDIN_FILENO, TCSANOW, &orig_);
    }

    // Non-blocking read; returns optional<char>
    std::optional<char> readChar()
    {
        fd_set set;
        FD_ZERO(&set);
        FD_SET(STDIN_FILENO, &set);

        timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 0;

        const int rv = select(STDIN_FILENO + 1, &set, nullptr, nullptr, &tv);
        if (rv > 0 && FD_ISSET(STDIN_FILENO, &set))
        {
            char c;
            const ssize_t n = ::read(STDIN_FILENO, &c, 1);
            if (n == 1)
                return c;
        }
        return std::nullopt;
    }

private:
    termios orig_{};
};

class TeleopGoalPublisher : public rclcpp::Node
{
public:
    TeleopGoalPublisher()
        : Node("teleop_goal_publisher"), kb_(std::make_unique<Keyboard>())
    {
        // Params (step sizes)
        base_dp_ = this->declare_parameter<double>("base_dp", 0.01);                  // meters
        base_dang_ = this->declare_parameter<double>("base_dang", 0.5 * M_PI / 180.0); // radians (0.5 deg)
        ee_dp_ = this->declare_parameter<double>("ee_dp", 0.05);                      // meters
        base_max_target_distance_m_ = this->declare_parameter<double>("base_max_target_distance_m", 0.2); // meters
        ee_max_target_distance_m_ = this->declare_parameter<double>("ee_max_target_distance_m", 1.5);     // meters
        pub_hz_ = this->declare_parameter<double>("pub_hz", 30.0);

        base_target_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/hanyang/base_target_zyx", 10);
        ee_target_pub_ = this->create_publisher<geometry_msgs::msg::Vector3>(
            "/hanyang/ee_target", 10);

        base_pos_sub_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
            "/hanyang/base_pos", 10,
            [this](const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
            {
                base_pos_meas_[0] = msg->vector.x;
                base_pos_meas_[1] = msg->vector.y;
                base_pos_meas_[2] = msg->vector.z;
                have_base_pos_ = true;
                maybe_init_targets();
            });

        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/hanyang/imu", 10,
            [this](const sensor_msgs::msg::Imu::SharedPtr msg)
            {
                double yaw, pitch, roll;
                quat_to_zyx(msg->orientation.x, msg->orientation.y,
                            msg->orientation.z, msg->orientation.w,
                            yaw, pitch, roll);
                base_rpy_meas_[0] = roll;
                base_rpy_meas_[1] = pitch;
                base_rpy_meas_[2] = yaw;
                have_imu_ = true;
                maybe_init_targets();
            });

        ee_pos_sub_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
            "/hanyang/ee_pos", 10,
            [this](const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
            {
                ee_pos_meas_[0] = msg->vector.x;
                ee_pos_meas_[1] = msg->vector.y;
                ee_pos_meas_[2] = msg->vector.z;
                have_ee_pos_ = true;
                maybe_init_targets();
            });

        const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, pub_hz_));
        timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&TeleopGoalPublisher::loop, this));

        print_help();
        RCLCPP_INFO(this->get_logger(),
                    "TeleopGoalPublisher started. Waiting for /hanyang/base_pos, /hanyang/imu, /hanyang/ee_pos ...");
    }

private:
    void print_help()
    {
        std::cout << "\n=== Teleop Goal Publisher ===\n"
                     "Base pos:   i/k (+x/-x), j/l (+y/-y), u/o (+z/-z)\n"
                     "Base rpy:   q/e (+yaw/-yaw), t/g (+pitch/-pitch), z/x (+roll/-roll)\n"
                     "EE pos:     w/s (+x/-x), a/d (+y/-y), r/f (+z/-z)\n"
                     "Space: publish once | c: reset to current | h: help | ESC: quit\n"
                     "Params: base_dp(m), base_dang(rad), ee_dp(m), pub_hz(Hz)\n"
                     "-----------------------------------------\n";
    }

    void maybe_init_targets()
    {
        if (targets_inited_)
            return;
        if (!have_base_pos_ || !have_imu_ || !have_ee_pos_)
            return;

        // Initialize desired targets to current measurements (zero delta)
        base_target_pos_ = base_pos_meas_;
        base_target_rpy_ = base_rpy_meas_; // [roll,pitch,yaw]
        ee_target_ = ee_pos_meas_;

        targets_inited_ = true;

        RCLCPP_INFO(this->get_logger(),
                    "Targets initialized to current. You can teleop now.");
        publish_targets(); // publish initial once
    }

    void reset_to_current()
    {
        if (!have_base_pos_ || !have_imu_ || !have_ee_pos_)
            return;
        base_target_pos_ = base_pos_meas_;
        base_target_rpy_ = base_rpy_meas_;
        ee_target_ = ee_pos_meas_;
        publish_targets();
        RCLCPP_INFO(this->get_logger(), "Reset targets to current.");
    }

    void publish_targets()
    {
        if (!targets_inited_)
            return;

        // base_target_zyx expects: [x y z yaw pitch roll]
        std_msgs::msg::Float64MultiArray base_msg;
        base_msg.data.resize(6);
        base_msg.data[0] = base_target_pos_[0];
        base_msg.data[1] = base_target_pos_[1];
        base_msg.data[2] = base_target_pos_[2];
        base_msg.data[3] = wrap_pi(base_target_rpy_[2]); // yaw
        base_msg.data[4] = wrap_pi(base_target_rpy_[1]); // pitch
        base_msg.data[5] = wrap_pi(base_target_rpy_[0]); // roll
        base_target_pub_->publish(base_msg);

        geometry_msgs::msg::Vector3 ee_msg;
        ee_msg.x = ee_target_[0];
        ee_msg.y = ee_target_[1];
        ee_msg.z = ee_target_[2];
        ee_target_pub_->publish(ee_msg);
    }

    static double distance3(const std::array<double, 3> &a, const std::array<double, 3> &b)
    {
        const double dx = a[0] - b[0];
        const double dy = a[1] - b[1];
        const double dz = a[2] - b[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    bool within_target_distance_limits(double &base_distance, double &ee_distance) const
    {
        base_distance = distance3(base_target_pos_, base_pos_meas_);
        ee_distance = distance3(ee_target_, ee_pos_meas_);
        return (base_distance <= base_max_target_distance_m_) && (ee_distance <= ee_max_target_distance_m_);
    }

    void apply_key(char c, bool &changed, bool &force_pub, bool &quit)
    {
        const auto base_target_pos_prev = base_target_pos_;
        const auto base_target_rpy_prev = base_target_rpy_;
        const auto ee_target_prev = ee_target_;

        // ESC
        if (static_cast<unsigned char>(c) == 27)
        {
            quit = true;
            return;
        }

        switch (c)
        {
        // EE increments
        case 'w':
            ee_target_[0] += ee_dp_;
            changed = true;
            break;
        case 's':
            ee_target_[0] -= ee_dp_;
            changed = true;
            break;
        case 'a':
            ee_target_[1] += ee_dp_;
            changed = true;
            break;
        case 'd':
            ee_target_[1] -= ee_dp_;
            changed = true;
            break;
        case 'r':
            ee_target_[2] += ee_dp_;
            changed = true;
            break;
        case 'f':
            ee_target_[2] -= ee_dp_;
            changed = true;
            break;

        // Base position increments
        case 'i':
            base_target_pos_[0] += base_dp_;
            changed = true;
            break;
        case 'k':
            base_target_pos_[0] -= base_dp_;
            changed = true;
            break;
        case 'j':
            base_target_pos_[1] += base_dp_;
            changed = true;
            break;
        case 'l':
            base_target_pos_[1] -= base_dp_;
            changed = true;
            break;
        case 'u':
            base_target_pos_[2] += base_dp_;
            changed = true;
            break;
        case 'o':
            base_target_pos_[2] -= base_dp_;
            changed = true;
            break;

        // Base orientation increments (roll,pitch,yaw stored as [r,p,y])
        case 'q':
            base_target_rpy_[2] = wrap_pi(base_target_rpy_[2] + base_dang_);
            changed = true;
            break; // +yaw
        case 'e':
            base_target_rpy_[2] = wrap_pi(base_target_rpy_[2] - base_dang_);
            changed = true;
            break; // -yaw
        case 't':
            base_target_rpy_[1] = wrap_pi(base_target_rpy_[1] + base_dang_);
            changed = true;
            break; // +pitch
        case 'g':
            base_target_rpy_[1] = wrap_pi(base_target_rpy_[1] - base_dang_);
            changed = true;
            break; // -pitch
        case 'z':
            base_target_rpy_[0] = wrap_pi(base_target_rpy_[0] + base_dang_);
            changed = true;
            break; // +roll
        case 'x':
            base_target_rpy_[0] = wrap_pi(base_target_rpy_[0] - base_dang_);
            changed = true;
            break; // -roll

        case ' ':
            force_pub = true;
            break;
        case 'c':
            reset_to_current();
            force_pub = true;
            break;
        case 'h':
            print_help();
            break;
        default:
            break;
        }

        if (changed)
        {
            double base_distance = 0.0;
            double ee_distance = 0.0;
            if (!within_target_distance_limits(base_distance, ee_distance))
            {
                base_target_pos_ = base_target_pos_prev;
                base_target_rpy_ = base_target_rpy_prev;
                ee_target_ = ee_target_prev;
                changed = false;

                // RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                //                      "Target rejected: distance limit exceeded (base %.3f/%.3fm, ee %.3f/%.3fm)",
                //                      base_distance, base_max_target_distance_m_,
                //                      ee_distance, ee_max_target_distance_m_);
            }
        }
    }

    void loop()
    {
        maybe_init_targets();
        if (!targets_inited_)
            return;

        bool changed = false;
        bool force_pub = false;
        bool quit = false;

        // Drain all pending keypresses in this tick
        while (true)
        {
            auto oc = kb_->readChar();
            if (!oc.has_value())
                break;
            apply_key(*oc, changed, force_pub, quit);
            if (quit)
                break;
        }

        if (quit)
        {
            RCLCPP_INFO(this->get_logger(), "ESC pressed. Shutting down teleop node.");
            rclcpp::shutdown();
            return;
        }

        if (changed || force_pub)
        {
            publish_targets();
        }
    }

private:
    // Publishers/Subscribers
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr base_target_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr ee_target_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr base_pos_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr ee_pos_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // Keyboard
    std::unique_ptr<Keyboard> kb_;

    // Params
    double base_dp_{0.1};
    double base_dang_{0.5 * M_PI / 180.0};
    double ee_dp_{0.1};
    double base_max_target_distance_m_{0.2};
    double ee_max_target_distance_m_{0.2};
    double pub_hz_{100.0};

    // Measurements (current)
    std::array<double, 3> base_pos_meas_{0.0, 0.0, 0.0};
    std::array<double, 3> base_rpy_meas_{0.0, 0.0, 0.0}; // [roll,pitch,yaw]
    std::array<double, 3> ee_pos_meas_{0.0, 0.0, 0.0};
    bool have_base_pos_{false};
    bool have_imu_{false};
    bool have_ee_pos_{false};

    // Targets (desired)
    std::array<double, 3> base_target_pos_{0.0, 0.0, 0.0};
    std::array<double, 3> base_target_rpy_{0.0, 0.0, 0.0}; // [roll,pitch,yaw]
    std::array<double, 3> ee_target_{0.0, 0.0, 0.0};

    bool targets_inited_{false};
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TeleopGoalPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
