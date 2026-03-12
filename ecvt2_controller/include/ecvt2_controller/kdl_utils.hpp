#pragma once
#include <Eigen/Dense>
#include <array>
#include <string>

namespace ecvt2_controller
{

  // 결과 묶음: [0]=Base, [1]=FR, [2]=FL, [3]=RR, [4]=RL, [5]=ARM
  using FKArray6 = std::array<Eigen::Matrix4d, 6>;
  // 각 부위의 Jacobian: 6 x N (N = q.size())
  using JArray6 = std::array<Eigen::MatrixXd, 6>;

  struct FKJacobianResult
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    FKArray6 T; // 변환행렬들
    JArray6 J;  // Jacobians
    // Optional extras for controllers that also need COM terms.
    Eigen::Vector3d com = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 3, 30> com_jacobian = Eigen::Matrix<double, 3, 30>::Zero();
  };

  struct PinocchioKinematicsResult
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d com = Eigen::Vector3d::Zero();
    // Frame order: [0]=FRW, [1]=FLW, [2]=RRW, [3]=RLW, [4]=UP5
    std::array<Eigen::Matrix<double, 6, 10>, 5> jacobians{};
    // Reduced CoM Jacobian with mimic:
    // 3 x (6 base + 20 legs + 4 arm) = 3 x 30
    Eigen::Matrix<double, 3, 30> com_jacobian = Eigen::Matrix<double, 3, 30>::Zero();
  };

  // FK + Jacobian 동시 계산
  FKJacobianResult computeFKAllAndJacobian(const Eigen::Ref<const Eigen::Vector3d> &base_pos,
                                           const Eigen::Ref<const Eigen::Vector4d> &base_quat_xyzw,
                                           const Eigen::Ref<const Eigen::VectorXd> &q);
  FKJacobianResult computePinocchioFKAllAndJacobian(
      const Eigen::Ref<const Eigen::Vector3d> &base_pos,
      const Eigen::Ref<const Eigen::Vector4d> &base_quat_xyzw,
      const Eigen::Ref<const Eigen::VectorXd> &q,
      const std::string &urdf_path = "");
  PinocchioKinematicsResult computePinocchioComAndJacobians(
      const Eigen::Ref<const Eigen::Vector3d> &base_pos,
      const Eigen::Ref<const Eigen::Vector4d> &base_quat_xyzw,
      const Eigen::Ref<const Eigen::VectorXd> &q,
      const std::string &urdf_path = "");
  Eigen::Matrix3d computeManipulabilityMatrix(
      const FKJacobianResult &result, const Eigen::Ref<const Eigen::VectorXd> &q);
  Eigen::Vector3d computeVMaxDirection(const Eigen::Ref<const Eigen::Vector3d> &base_pos,
                                       const Eigen::Ref<const Eigen::Vector4d> &base_quat_xyzw,
                                       const Eigen::Ref<const Eigen::VectorXd> &q,
                                       const Eigen::Ref<const Eigen::Matrix3d> &A);

  // (선택) 회전/병진 유틸
  Eigen::Matrix4d RotX(double q);
  Eigen::Matrix4d RotY(double q);
  Eigen::Matrix4d RotZ(double q);
  Eigen::Matrix4d Trans(double x, double y, double z);
  void printMatrixPretty(const Eigen::MatrixXd &M, const std::string &name);
  Eigen::Matrix3d eulerZYXToRot(double yaw, double pitch, double roll);
} // namespace ecvt_controller
