#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>

namespace
{
struct Point3
{
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

Point3 add(const Point3 &a, const Point3 &b)
{
  return Point3{a.x + b.x, a.y + b.y, a.z + b.z};
}

Point3 lerp(const Point3 &a, const Point3 &b, double t)
{
  return Point3{
      a.x + (b.x - a.x) * t,
      a.y + (b.y - a.y) * t,
      a.z + (b.z - a.z) * t};
}
} // namespace

class ComTrajectoryPublisher : public rclcpp::Node
{
public:
  ComTrajectoryPublisher()
  : Node("com_trajectory_publisher")
  {
    pub_hz_ = this->declare_parameter<double>("pub_hz", 100.0);
    segment_duration_ = this->declare_parameter<double>("segment_duration", 5.0);
    loop_trajectory_ = this->declare_parameter<bool>("loop_trajectory", true);
    hold_origin_duration_ = this->declare_parameter<double>("hold_origin_duration", 0.5);

    // Start from current COM, then draw rectangle on x-z plane:
    // +x 0.3 -> +z 0.3 -> -x 0.3 -> -z 0.3 -> origin.
    rel_waypoints_[0] = Point3{0.0, 0.0, 0.0};
    rel_waypoints_[1] = Point3{0.3, 0.0, 0.0};
    rel_waypoints_[2] = Point3{0.3, 0.0, 0.3};
    rel_waypoints_[3] = Point3{0.0, 0.0, 0.3};
    rel_waypoints_[4] = Point3{0.0, 0.0, 0.0};

    com_desired_pub_ = this->create_publisher<geometry_msgs::msg::Vector3>(
      "/hanyang/com_desired", 10);

    com_pos_sub_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
      "/hanyang/com_pos", 10,
      [this](const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
      {
        if (have_origin_)
          return;
        origin_.x = msg->vector.x;
        origin_.y = msg->vector.y;
        origin_.z = msg->vector.z;
        traj_start_time_ = this->now();
        have_origin_ = true;

        RCLCPP_INFO(
          this->get_logger(),
          "COM trajectory origin fixed at (%.3f, %.3f, %.3f)",
          origin_.x, origin_.y, origin_.z);
      });

    const double hz = std::max(1.0, pub_hz_);
    const auto period = std::chrono::duration<double>(1.0 / hz);
    timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&ComTrajectoryPublisher::onTimer, this));

    RCLCPP_INFO(
      this->get_logger(),
      "Publishing /hanyang/com_desired trajectory after first /hanyang/com_pos. segment_duration=%.2fs",
      segment_duration_);
  }

private:
  void onTimer()
  {
    if (!have_origin_)
      return;

    const double seg_t = std::max(1e-3, segment_duration_);
    const double move_total = 4.0 * seg_t;
    const double hold_t = std::max(0.0, hold_origin_duration_);
    const double cycle_t = move_total + hold_t;

    double t = (this->now() - traj_start_time_).seconds();
    if (loop_trajectory_)
    {
      if (cycle_t > 1e-9)
        t = std::fmod(std::max(0.0, t), cycle_t);
      else
        t = 0.0;
    }
    else
    {
      t = std::clamp(t, 0.0, cycle_t);
    }

    Point3 target_world = origin_;

    if (t < move_total)
    {
      int seg_idx = static_cast<int>(t / seg_t);
      seg_idx = std::clamp(seg_idx, 0, 3);
      const double local_t = t - static_cast<double>(seg_idx) * seg_t;
      const double alpha = std::clamp(local_t / seg_t, 0.0, 1.0);

      const Point3 p0 = add(origin_, rel_waypoints_[seg_idx]);
      const Point3 p1 = add(origin_, rel_waypoints_[seg_idx + 1]);
      target_world = lerp(p0, p1, alpha);
    }

    geometry_msgs::msg::Vector3 msg;
    msg.x = target_world.x;
    msg.y = target_world.y;
    msg.z = target_world.z;
    com_desired_pub_->publish(msg);
  }

  rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr com_desired_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr com_pos_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::array<Point3, 5> rel_waypoints_{};
  Point3 origin_{};
  bool have_origin_{false};
  rclcpp::Time traj_start_time_{0, 0, RCL_ROS_TIME};

  double pub_hz_{50.0};
  double segment_duration_{3.0};
  bool loop_trajectory_{true};
  double hold_origin_duration_{0.5};
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ComTrajectoryPublisher>());
  rclcpp::shutdown();
  return 0;
}
